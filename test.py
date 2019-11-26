# -*- coding:utf-8 -*-

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


#权重计算函数
def likelihood(x, y, func, image, w=30, h=30):
    x1 = np.int32(max(0, x - w / 2))
    y1 = np.int32(max(0, y - h / 2))
    x2 = np.int32(min(image.shape[1], x + w / 2))
    y2 = np.int32(min(image.shape[0], y + h / 2))
    #截取图片区域
    region = image[y1:y2, x1:x2]
    #统计符合色值的像素点个数
    count = region[func(region)].size
    return (float(count) / image.size) if count > 0 else 0.0001

#particle的初始化函数
# def init_particles(func, image):
#     mask = image.copy()
#     mask[func(mask) == False] = 0
#     #查找检测物体的轮廓
#     _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) <= 0:
#         return None
#     #面积最大轮廓
#     max_contour = max(contours, key=cv2.contourArea)
#     #获取外部矩形边界（返回值：返回四个值，分别是x，y，w，h；x、y是矩阵左上方的坐标，w、h是矩阵的宽和高）
#     max_rect = np.array(cv2.boundingRect(max_contour))
#     max_rect = max_rect[:2] + max_rect[2:] / 2
#     weight = likelihood(max_rect[0], max_rect[1], func, image)
#     particles = np.ndarray((500, 3), dtype=np.float32)
#     particles[:] = [max_rect[0], max_rect[1], weight]
    
#     return particles

def init_particles(func, image, ref_box):
    #获取外部矩形边界（返回值：返回四个值，分别是x，y，w，h；x、y是矩阵左上方的坐标，w、h是矩阵的宽和高）
    max_rect = np.array(ref_box)
    max_rect = max_rect[:2] + max_rect[2:] / 2
    weight = likelihood(max_rect[0], max_rect[1], func, image)
    particles = np.ndarray((500, 3), dtype=np.float32)
    particles[:] = [max_rect[0], max_rect[1], weight]
    
    return particles

#根据粒子的权重的后验概率分布重新采样
def resample(particles):
    tmp_particles = particles.copy()
    #元素累加
    weights = particles[:, 2].cumsum()
    last_weight = weights[weights.shape[0] - 1]
    for i in range(particles.shape[0]):
        weight = np.random.rand() * last_weight
        #只返回第一次出现的关系表达式为真的索引
        particles[i] = tmp_particles[(weights > weight).argmax()]
        particles[i][2] = 1.0

#预测
def predict(particles, variance=13.0):
    #np.random.randn：以给定的形状创建一个数组，数组元素符合标准正态分布N(0,1)
    #13位半径，向外随机扩散
    particles[:, 0] += np.random.randn((particles.shape[0])) * variance
    particles[:, 1] += np.random.randn((particles.shape[0])) * variance

#权重处理
def weight(particles, func, image):
    for i in range(particles.shape[0]):
        particles[i][2] = likelihood(particles[i][0], particles[i][1], func, image)
        #权重加和
        sum_weight = particles[:, 2].sum()
        particles[:, 2] *= (particles.shape[0] / sum_weight)

#测定坐标
def measure(particles):
    x = (particles[:, 0] * particles[:, 2]).sum()
    y = (particles[:, 1] * particles[:, 2]).sum()
    weight = particles[:, 2].sum()
    return x / weight, y / weight

#particle、坐标获取
particle_filter_cur_frame = 0
def particle_filter(particles, func, image, ref_box, max_frame=10):
    global particle_filter_cur_frame

    if image[func(image)].size <= 0:
        if particle_filter_cur_frame >= max_frame:
            return None, -1, -1
        particle_filter_cur_frame = min(particle_filter_cur_frame + 1, max_frame)
    else:
        particle_filter_cur_frame = 0
        if particles is None:
            particles = init_particles(func, image, ref_box)

    if particles is None:
        return None, -1, -1

    resample(particles)
    predict(particles, variance=30)
    weight(particles, func, image)
    x, y = measure(particles)
    return particles, x, y

if __name__ == '__main__':
    def is_color(region):
        #色调范围限制
        return (region >= 11) & (region < 34)

    #打开摄像头
    imgdir = r'./data/Man/img/'
    gtfile = r'./data/Man/groundtruth_rect.txt'
    imgtype = r'jpg'

    imgnamelist = make_file_list(imgdir, imgtype)
    groundtruth = np.loadtxt(gtfile, delimiter=',',dtype=int).tolist()

    for imgname in tqdm(imgnamelist):
        
        #读取图像
        frame = cv2.imread(imgname)

        #图像转换为HSV格式
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_h = frame_hsv[:, :, 0]
        _, frame_s = cv2.threshold(frame_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, frame_v = cv2.threshold(frame_hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #S或V为0的像素点，H赋值为0
        frame_h[(frame_s == 0) | (frame_v == 0)] = 0

        #获取粒子点、中心坐标
        particles = None
        particles, x, y = particle_filter(particles, is_color, frame_h, groundtruth[0])

        if particles is not None:
            #防止出边界
            valid_particles = np.int32(particles[(particles[:, 0] >= 0) & (particles[:, 0] < frame.shape[1]) &
                                        (particles[:, 1] >= 0) & (particles[:, 1] < frame.shape[0])])
            #修改粒子点的颜色
            for i in range(valid_particles.shape[0]):
                frame[valid_particles[i][1], valid_particles[i][0]] = [255, 0, 0]
            p = np.array([x, y], dtype=np.int32)
            #根据中心点画框
            cv2.rectangle(frame, tuple(p - 15), tuple(p + 15), (0, 0, 255), thickness=2)


        cv2.imshow('green', frame)
        cv2.waitKey(0)  

    cv2.destroyAllWindows()