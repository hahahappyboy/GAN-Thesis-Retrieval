import torch.nn as nn
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms


eps = 1e-5
input1 = cv2.imread('zyy.jpg')
input1 = torch.from_numpy(np.transpose(input1, (2, 0, 1))).float()
input1 = torch.unsqueeze(input1, 0)

input2 = cv2.imread('zhu.jpg')
input2 = torch.from_numpy(np.transpose(input2, (2, 0, 1))).float()
input2 = torch.unsqueeze(input2, 0)

in1_mean, in1_var = torch.mean(input1, dim=[2, 3], keepdim=True), torch.var(input1, dim=[2, 3], keepdim=True)
out1_in = (input1 - in1_mean) /  in1_var

in2_mean, in2_var = torch.mean(input2, dim=[2, 3], keepdim=True), torch.var(input1, dim=[2, 3], keepdim=True)

out2_in = in2_var * out1_in + in2_mean

output = torch.squeeze(out2_in,0)
img_cv_2 = np.transpose(output.numpy(), (1, 2, 0))
cv2.imwrite('zyy_adIN.jpg',img_cv_2)

