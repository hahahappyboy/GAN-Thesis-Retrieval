import torch

def generate_labelmix(label, fake_image, real_image):# label[1,183,512,512] fake_image/real_iamge [1,3,512,512]
    target_map = torch.argmax(label, dim = 1, keepdim = True) # argmax返回指定维度最大值的序号 [1,1,512,512]
    all_classes = torch.unique(target_map) # 这个label有多少张图片 6 [0,62,73,110,130]
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0,2,(1,))# 随机一个0-1的数，生成一张二值图只有[0,1]
    target_map = target_map.float()
    mixed_image = target_map*real_image+(1-target_map)*fake_image # fake_image和real_image融合
    return mixed_image, target_map
