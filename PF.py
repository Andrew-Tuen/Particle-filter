# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from tqdm import tqdm

def make_file_list(dir, type):
    r"""读取文件夹下所有文件 

    Args:
        dir: 文件夹路径  str
        type: 文件类型  str
    Return:
        namelist: 包含符合条件的所有文件的列表  list
    """
    namelist=[]
    for filename in os.listdir(dir):
        if filename.endswith(type):
            namelist.append(dir+'/'+filename)
    return np.sort(namelist).tolist()

class PF():
    def __init__(self, N, alpha, beta): #alpha 控制原始和上一帧的参考， beta控制速度的更新速率
        self.N = N
        self.W = 0
        self.H = 0
        self.alpha = alpha
        self.beta = beta
        self.box0 = [0,0,0,0]
        self.hist0 = 0
        self.ref_box = 0
        self.ref_hist = 0
        self.particles = 0
        self.speed = np.array([0,0,0,0])
        self.weights = np.ones(N)/self.N
        self.scale = 1.0

    
    def initial(self, imgname, ref):
        r"""初始化信息
        Args: 
            imgname: 图片路径  str
            ref: 物体初始位置 list
        """
        img = self.read_image(imgname).astype(np.uint8)
        self.W = img.shape[0]
        self.H = img.shape[1]
        self.ref_box = np.array(ref)
        self.ref_hist = self.three_channel_hist(self.cut_img(img,self.ref_box))
        self.particles = np.tile(self.ref_box, (self.N, 1))
        self.box0 = self.ref_box
        self.hist0 = self.ref_hist
        self.pos_list = []

    def cut_img(self, img, box):
        ws = np.max([0,box[1]])
        we = np.min([self.W, box[1]+box[3]])
        we = np.max([we,ws+1])
        hs = np.max([0,box[0]])
        he = np.min([self.H, box[0]+box[2]])
        he = np.max([he,hs+1])
        return img[ws:we,hs:he,:]

    def cal_scale(self, pos):
        self.scale = (1-self.beta)*self.scale + self.beta*self.ref_box[-1]*self.ref_box[-2]/pos[-1]/pos[-2]


    def cal_weight(self, img):
        N = len(self.particles)
        subimgs = [self.cut_img(img, self.particles[i]) for i in range(N)]
        # self.disp_imgs(*subimgs, nx=7, ny=15, size=(100,100), scale=0.7, name='FIGURE', show_img=True)

        subhist = [self.three_channel_hist(subimgs[i]) for i in range(N)]
        dis2ref = [cv2.compareHist(subhist[i], self.ref_hist, method=cv2.HISTCMP_BHATTACHARYYA) for i in range(N)]
        dis2ori = [cv2.compareHist(subhist[i], self.hist0, method=cv2.HISTCMP_BHATTACHARYYA) for i in range(N)]
        # dis2ref = [cv2.compareHist(subhist[i], self.ref_hist, method=cv2.HISTCMP_CORREL) for i in range(N)]
        # dis2ori = [cv2.compareHist(subhist[i], self.hist0, method=cv2.HISTCMP_CORREL) for i in range(N)]
        dis = (1-self.alpha)*np.array(dis2ori) + (self.alpha)*np.array(dis2ref) 
        # print("{}, {}".format(np.max(dis),np.min(dis)))
        self.weights = (dis-np.max(dis)+1e-50)/(np.max(dis)-np.min(dis)+1e-50)
        self.weights /= np.sum(self.weights)
        index = np.argsort(-self.weights)
        self.weights = self.weights[index]
        self.particles = self.particles[index]
        self.weights = self.weights[:self.N]
        self.particles = self.particles[:self.N]
        return 1.0+2*cv2.compareHist(self.hist0, self.ref_hist, method=cv2.HISTCMP_BHATTACHARYYA)

    def predict(self, var):
        tmpp = self.particles + np.random.randn(*self.particles.shape)*np.array([var,var,0,0])
        tmpp += self.speed*np.random.rand(self.N,4)
        # scale = (np.random.randn(self.N,1)*np.sqrt(self.scale)*0.15)+1.0

        # shift = tmpp[:,2:]*(1.0-scale)*0.5
        # tmpp[:,:2] += shift
        # tmpp[:,2:] *= scale
        self.particles = np.round(tmpp).astype(np.int64)
        self.particles[self.particles<1] = 1

    def change_scale(self):
        
        tmpp = self.particles[:20].copy()
        for scale in np.arange(0.6,1.2,0.01):
            scdp = tmpp[:20].copy()*1.0
            shift = scdp[:,2:]*(1.0-scale)*0.5
            scdp[:,:2] += shift
            scdp[:,2:] *= scale
            tmpp = np.vstack([tmpp,scdp])
        self.particles = np.round(tmpp).astype(np.int64)
        self.particles[self.particles<1] = 1

    def get_pos(self):

        
        tmpw = self.weights[:100]
        tmpp = self.particles[:100]

        # subparticle = subparticle[:int(self.N/10)]
        # subweight = subweight[:int(self.N/10)]
        
        tmpw = tmpw / np.sum(tmpw)
        pos = 0.3*self.ref_box + 0.7*np.sum(tmpp*np.expand_dims(tmpw,-1), axis=0)

        # pos = np.sum(self.particles*np.expand_dims(self.weights,-1), axis=0)
        # pos = self.particles[np.argmax(self.weights)]
        return np.round(pos).astype(np.int64)

    def update_speed(self, pos):
        self.speed = (1-self.beta)*self.speed + self.beta*(pos - self.ref_box)
        self.speed[2:] = 0
    
    def update_ref(self, img, pos):
        self.ref_box = pos.copy()
        self.ref_hist = self.three_channel_hist(self.cut_img(img,self.ref_box))
        # self.particles[:,-2] = pos[-2]
        # self.particles[:,-1] = pos[-1]
        self.particles[:] = pos.copy()

    def resample(self):
        tmp = self.particles.copy()
        int_weight = np.cumsum(self.weights)
        total_weight = int_weight[-1]
        for i in range(self.N):
            T = np.random.rand()*total_weight
            self.particles[i] = tmp[np.argmax(int_weight>T)].copy()
            self.weights[i] = 1.0/self.N

    
    def track(self, imgnamelist):
        t = 1.0
        for imgname in tqdm(imgnamelist):
            img = self.read_image(imgname).astype(np.uint8)
            self.resample()
            self.predict(30*t)
            t = self.cal_weight(img)
            self.change_scale()
            self.cal_weight(img)
            pos = self.get_pos()
            # self.show_particles(img, pos)
            # self.cal_scale(pos)
            self.pos_list.append(pos.tolist())
            self.update_speed(pos)
            self.update_ref(img, pos)
            


    @staticmethod
    def three_channel_hist(img):
        r"""计算三通道直方图
        Args: 
            img: 图片 float (H*W*C)
        Return: 
            hist: 直方图  float  (256C)
        """
        
        hist = [cv2.calcHist([img], [i], None, [256], [0,256]) for i in range(3)]
        hist = np.vstack(hist)
        return hist

    @staticmethod
    def read_image(imgname):
        r"""读取图片，采用opencv方法，返回float类型，防止溢出
        Args: 
            imgname: 图片路径  str
        Return: 
            img: 图片  int  (H*W*C)
        """
        # imgname = '../WavingTrees/b'+str(i).zfill(5)+'.bmp'
        # img = plt.imread(imgname)
        img = cv2.imread(imgname)
        # print(img)
        return img.astype(float)

    @staticmethod
    def disp_imgs(*imgs, nx=1, ny=1, size=(160,120), scale=1, name='FIGURE', show_img=False):
        r"""显示多幅图像的算法，不足位置填充为空白

        Args:
            *imgs: img1, img2, ..., imgn  一组图像
            nx: 纵向显示的图片数 int
            ny: 横向显示图片数 int
            size: 图片大小 tuple
            scale: 尺度变换 float/int/double
            name: 窗口名称 str
        Return:
            imgbox: 打包好的图像集合 float 
        """
        n_img = len(imgs)
        iter_img = 0
        scaled_size = (np.ceil(size[0]*scale).astype(np.int),np.ceil(size[1]*scale).astype(np.int))

        for i in range(nx):
            for j in range(ny):
                if iter_img>=n_img:
                    add_img = cv2.resize((imgs[0]*0+255).astype(np.uint8), scaled_size) 
                else:
                    add_img = cv2.resize(imgs[iter_img].astype(np.uint8), scaled_size)
                if j == 0:
                    yimgs = add_img
                else:
                    yimgs = np.hstack([yimgs, add_img])
                iter_img += 1
            if i == 0:
                imgbox = yimgs
            else:
                imgbox = np.vstack([imgbox, yimgs])
        if show_img:
            cv2.namedWindow(name)
            cv2.imshow(name,imgbox)
            cv2.moveWindow(name,50,50)
            cv2.waitKey(0)  
            cv2.destroyAllWindows()
        return imgbox

    def show_particles(self, img, pos):
        boxed_img = img.copy()
        best_img = img.copy()
        point_color = (0, 255, 0) # BGR
        thickness = 1 
        lineType = 4
        for i in range(self.N):
            box = self.particles[i]
            ptLeftTop = (box[0], box[1])
            ptRightBottom = (box[0]+box[2], box[1]+box[3])
            cv2.rectangle(boxed_img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        box = pos
        ptLeftTop = (box[0], box[1])
        ptRightBottom = (box[0]+box[2], box[1]+box[3])
        cv2.rectangle(best_img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        self.disp_imgs(best_img, boxed_img, ny=2, nx=1, size=(img.shape[1],img.shape[0]), scale=1.0, show_img=True)

    def show_box(self, imglist, boxlist):
        r"""显示原图和加了box的图
        Args: 
            imglist: 图片路径合集  list of str
            boxlist: 位置合集
        """
        point_color = (0, 255, 0) # BGR
        thickness = 1 
        lineType = 4
        # print(imglist)
        # print(boxlist)
        for imgname, box in tqdm(zip(imgnamelist, boxlist)):
            img = self.read_image(imgname).astype(np.uint8)
            inbox = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:]
            hist = self.three_channel_hist(inbox)
            ptLeftTop = (box[0], box[1])
            ptRightBottom = (box[0]+box[2], box[1]+box[3])
            boxed = cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
            img = self.read_image(imgname).astype(np.uint8)
            ibx = self.disp_imgs(img[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:], boxed, ny=2, nx=1, size=(img.shape[1],img.shape[0]), scale=0.5, show_img=True)

    def save_video(self, imglist):

        fps = 8
        size = (self.H*2,self.W)
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        outv = cv2.VideoWriter()
        outv.open('./result.mp4', fourcc, fps, size, isColor=True)
        point_color = (0, 255, 0) # BGR
        thickness = 2 
        lineType = 4
        boxlist = self.pos_list

        for imgname, box in tqdm(zip(imgnamelist, boxlist)):
            img = self.read_image(imgname).astype(np.uint8)
            ptLeftTop = (box[0], box[1])
            ptRightBottom = (box[0]+box[2], box[1]+box[3])
            boxed = cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
            img = self.read_image(imgname).astype(np.uint8)
            ibx = self.disp_imgs(img[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:], boxed, ny=2, nx=1, size=(img.shape[1],img.shape[0]), scale=1.0, show_img=False)
            outv.write(ibx)

        outv.release()

if __name__ == '__main__':
    imgdir = r'./data/Man/img/'
    gtfile = r'./data/Man/groundtruth_rect.txt'
    imgtype = r'jpg'

    imgnamelist = make_file_list(imgdir, imgtype)
    groundtruth = np.loadtxt(gtfile, delimiter=',',dtype=int).tolist()
    # groundtruth = [[97, 79, 100, 100]]
    pf = PF(600, 0.6, 0.8)

    pf.initial(imgnamelist[0], groundtruth[0])
    pf.track(imgnamelist)

    show_img = True
    
    # print(groundtruth)
    # pf.show_box(imgnamelist, groundtruth)
    # pf.show_box(imgnamelist, pf.pos_list)
    pf.save_video(imgnamelist)