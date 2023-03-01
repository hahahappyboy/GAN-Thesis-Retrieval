import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import cv2

path = '14.png'

label = Image.open(path)

label = TR.functional.resize(label, (512, 512), Image.NEAREST)
# to tensor
label = TR.functional.to_tensor(label)
label = label * 255
label = label.long()
label_map = label.unsqueeze(0)

bs,_, h, w = label_map.size()
nc = 11
input_label = torch.FloatTensor(bs, nc, h, w).zero_()
input_semantics = input_label.scatter_(1, label_map, 1.0)
"""
    将图片变为实例图
clothes   128 128 0 -> 7
decoration 0 0 128  -> 5
face      128 0 128 -> 9
skin       0 128 128 -> 4
ear        128 128 128 ->  1
eye          64 0 0  -> 2
eyebrow     192 0 0  -> 3
mouth      64  128 0 -> 6
nose       192 128 0 -> 8
hair       64  0 128 -> 0
背景        0 0 0    -> 10
"""


for i in  range(0,nc):
    input_i = input_semantics[:,i,:,:]
    input_i = input_i.squeeze(0).squeeze(0)
    input_i_numpy = input_i.numpy()*255
    cv2.imwrite(str(i)+'.jpg',input_i_numpy)


targets = torch.ones((1,12,512,512)) # 全为1
num_of_classes = input_semantics.shape[1] # 11
integers = torch.argmax(input_semantics, dim=1) # 每个像素的类别，数值范围为0~10
targets = targets[:, 0, :, :] * num_of_classes # 全为11
integers += targets.long() # 范围为11~21
integers = torch.clamp(integers, min=num_of_classes-1) - num_of_classes + 1 # 范围为1~11
print(torch.unique(integers))
