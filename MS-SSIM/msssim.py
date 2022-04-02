
import os
import cv2
import numpy as np
import paddle
from paddle_msssim import ssim, ms_ssim

def file_name(file_dir):
    img_path_list = []
    file_name_list = []
    for root, dirs, files in os.walk(file_dir):
    # for dir in os.listdir(file_dir):
    #     print(root)  # 当前目录路径
    #     print(dirs)  # 当前路径下所有子目录
    #     print(files)  # 当前路径下所有非目录子文件
        for file in files:
             img_path_list.append((os.path.join(root, file),file))
    return img_path_list
def imread(img_path):
    img = cv2.imread(img_path)
    return paddle.to_tensor(img.transpose(2, 0, 1)[None, ...], dtype=paddle.float32)
if __name__ == '__main__':
    file_dir = 'imgs/pix2pixhd_912to512_onehot_epoch400_niter200_upnoise'  # 待取名称文件夹的绝对路径
    target_dir = 'imgs/real_target'

    img_path_list = file_name(file_dir)
    target_path_list = file_name(target_dir)
    d = 0
    for i in range(img_path_list.__len__()):
        (img_path, img_name) = img_path_list[i]
        (target_path, target_name) = target_path_list[i]
        print(img_path)
        print(target_path)
        fake = imread(img_path)
        real = imread(target_path)
        distance = ms_ssim(real, fake).cpu().numpy()
        print(distance)
        d += distance
print('!!!!!!!!!')
print(d/img_path_list.__len__())
