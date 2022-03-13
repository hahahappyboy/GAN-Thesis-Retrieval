import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
unloader = transforms.ToPILImage()
def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT) # #水平翻转
    return img


inst_path =  'aachen_000000_000019_gtFine_labelIds.png'
inst = Image.open('./label_images/'+inst_path)

transform_list = []
transform_list.append(transforms.Lambda(lambda img: __scale_width(img, 1024, 0)))
transform_list += [transforms.ToTensor()]
transform_A = transforms.Compose(transform_list)
t = transform_A(inst)* 255.0
label_map = t.unsqueeze(0)
# edge = torch.ByteTensor(t.size()).zero_()
# edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
# edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
# edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
# edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
# edge = edge.float()
size = label_map.size()# [1,1,512,1024]
oneHot_size = (size[0], 35, size[2], size[3])# [1,35,512,1024]
input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
input_labels = input_label.scatter_(1, label_map.data.long(), 1.0)

for i in range(35):
    input_label = input_labels[0,i,:,:]
    image = unloader(input_label)
    image.save('./label_onehot_results/'+str(i)+'_'+inst_path)

    inst = cv2.imread('./label_onehot_results/'+str(i)+'_'+inst_path)
    # (b,g,r) = cv2.split(inst) #通道分解
    # bH = cv2.equalizeHist(b)
    # gH = cv2.equalizeHist(g)
    # rH = cv2.equalizeHist(r)
    # result = cv2.merge((bH,gH,rH),)#通道合成
    # # dst = cv2.equalizeHist(inst)
    #
    # cv2.imwrite('./label_onehot_results/'+str(i)+'_'+inst_path,result)













