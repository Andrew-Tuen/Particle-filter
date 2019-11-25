import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    def __init__(self, N):
        self.N = N
        self.W = 0
        self.H = 0
        self.sigma = np.zeros(2,dtype=int)
        self.s0 = np.zeros(2,dtype=int)
        #self.hist = 
    
    def initial(self, img, ref):
        self.W = img.shape[0]
        self.H = img.shape[1]
        self.sigma = np.array(ref[2:])
        self.s0 = np.array(ref[:2])


    @staticmethod
    def three_channel_hist(img):
        hist = np.zeros((3,256))
        for i in range(3):
            hist[i] = np.squeeze(cv2.calcHist(img, [i], None, [256], [0,256]))
        
        for i in range(256):
            print(hist[:,i])     
        return hist


    @staticmethod
    def read_image(imgname):
        r"""读取图片，采用opencv方法，返回float类型，防止溢出
        Args: 
            imgname: 图片路径  str
        Return: 
            img: 图片  float  (H*W*C)
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
            cv2.moveWindow(name,200,50)
            cv2.waitKey(0)  
            cv2.destroyAllWindows()
        return imgbox

    def show_box(self, imglist, boxlist):
        point_color = (0, 255, 0) # BGR
        thickness = 1 
        lineType = 4
        # print(imglist)
        # print(boxlist)
        for imgname, box in tqdm(zip(imgnamelist, boxlist)):
            img = self.read_image(imgname).astype(np.uint8)
            print(box[0]+box[2])
            inbox = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:]
            hist = self.three_channel_hist(inbox)
            ptLeftTop = (box[0], box[1])
            ptRightBottom = (box[0]+box[2], box[1]+box[3])
            boxed = cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
            img = self.read_image(imgname).astype(np.uint8)
            ibx = self.disp_imgs(img[box[1]:box[1]+box[3],box[0]:box[0]+box[2],:], boxed, ny=2, nx=1, size=(img.shape[1],img.shape[0]), scale=0.8, show_img=True)


if __name__ == '__main__':
    imgdir = r'./Bird1/img/'
    gtfile = r'./Bird1/groundtruth_rect.txt'
    imgtype = r'jpg'
    pf = PF(500)
    show_img = True
    imgnamelist = make_file_list(imgdir, imgtype)
    groundtruth = np.loadtxt(gtfile, delimiter=',',dtype=int).tolist()
    # print(groundtruth)
    pf.show_box(imgnamelist, groundtruth)