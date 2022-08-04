
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
class losses_computer():
    def __init__(self):

       self.labelmix_function = torch.nn.MSELoss()
    """
        contain_dontcare_label: 有不关心的类别
        for_real: 是否是希望为真
    """
    def loss(self, input, label, for_real,contain_dontcare_label): # input[1,184,512,512](最后一类为假的) label[1,183,512,512]
        # 得到类别平衡权重
        weight_map = get_class_balancing(label,contain_dontcare_label)
        # n+1 类别的loss
        target = get_n1_target(input, label, for_real)# [1,512,512] 到底是哪一类
        loss = F.cross_entropy(input, target, reduction='none')
        if for_real: # 对真的使用平衡权重
            loss = torch.mean(loss * weight_map[:, 0, :, :])
        else: # 假的不使用
            loss = torch.mean(loss)
        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask*output_D_real+(1-mask)*output_D_fake  # 鉴别器输出的混合 [1,184,512,512]
        return self.labelmix_function(mixed_D_output, output_D_mixed) # MSE loss


def get_class_balancing(label, contain_dontcare_label ): # input[1,184,512,512] label[1,183,512,512]
    class_occurence = torch.sum(label, dim=(0, 2, 3)) # [183] 代表每个类别的数量
    if contain_dontcare_label: # ok 把第0类当作不关心的类别
        class_occurence[0] = 0
    num_of_classes = (class_occurence > 0).sum() # label中有多少类别 [5]
    # torch.numel(label) 用来统计tensor中元素的个数 183*512*512=47972352
    # label.shape[1] = 183
    # num_of_classes * label.shape[1] = 5 * 183 = 16836
    # torch.reciprocal(class_occurence) 返回具有 class_occurence 元素倒数的新张量 代表每个类别的数量的倒数
    coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1]) # [183] 183个类别的权重
    integers = torch.argmax(label, dim=1, keepdim=True) #[1,1,512,512] label的每个像素到底是哪个类别
    if contain_dontcare_label:
        coefficients[0] = 0
    weight_map = coefficients[integers] # [1，1，512，512]取出对应权重
    return weight_map


def get_n1_target(input, label, target_is_real): #  input[1,184,512,512](最后一类为假的) label [1,183,512,512]
    targets = get_target_tensor(input, target_is_real) # [1,184,512,512] 全为1 或则全为0
    num_of_classes = label.shape[1] # 183
    integers = torch.argmax(label, dim=1) # [1,512,512] 每个像素的类别
    targets = targets[:, 0, :, :] * num_of_classes # [1,512,512] 全为183或全为0
    integers += targets.long() # [1,512,512]
    integers = torch.clamp(integers, min=num_of_classes-1) - num_of_classes + 1 # [1，512，512]
    return integers


def get_target_tensor(input, target_is_real):
    if target_is_real:
        return torch.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(input)
    else:
        return torch.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(input)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

def HingeLoss(input, target_is_real, for_discriminator=True):
    if for_discriminator:
        if target_is_real:
            minval = torch.min(input - 1, torch.zeros_like(input))
            loss = -torch.mean(minval)
        else:
            minval = torch.min(-input - 1,torch.zeros_like(input))
            loss = -torch.mean(minval)
    else:
        loss = -torch.mean(input)
    return loss

