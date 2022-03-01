import torch.nn as nn
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt

def plot_demo(image):
    # numpy的ravel函数功能是将多维数组降为一维数组
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show("直方图")


def image_hist_demo(image):
    color = {"blue", "green", "red"}
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据下标和数据，一般用在 for 循环当中。
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

eps = 1e-5
input1 = cv2.imread('zyy.jpg')
plt.hist(input1.ravel(), 256)
plt.savefig("zyy_hist.jpg")
plt.close()

input1 = torch.from_numpy(np.transpose(input1, (2, 0, 1))).float()
input1 = torch.unsqueeze(input1, 0)


input2 = cv2.imread('zhu.jpg')
plt.hist(input2.ravel(),256)
plt.savefig("zhu_hist.jpg")
plt.close()


input2 = torch.from_numpy(np.transpose(input2, (2, 0, 1))).float()
input2 = torch.unsqueeze(input2, 0)

in1_mean, in1_var = torch.mean(input1, dim=[2, 3], keepdim=True), torch.var(input1, dim=[2, 3], keepdim=True)
out1_in = (input1 - in1_mean) /  in1_var

in2_mean, in2_var = torch.mean(input2, dim=[2, 3], keepdim=True), torch.var(input2, dim=[2, 3], keepdim=True)

out2_in = in2_var * out1_in + in2_mean

output = torch.squeeze(out2_in,0)
img_cv_2 = np.transpose(output.numpy(), (1, 2, 0))
plt.hist(img_cv_2.ravel(),256)
plt.savefig("zyy_adIN_hist.jpg")
plt.close()

cv2.imwrite('zyy_adIN.jpg',img_cv_2)

