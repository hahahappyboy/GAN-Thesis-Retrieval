import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SEModule(nn.Module):
    def __init__(self, channels, reduction):# channels=64 reduction=16
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 64->4
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # 4->64
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

# bottleneck_IR_SE,对面部识别进行了预训练,，从而加速了收敛。
class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == out_channel:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, 1), stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            SEModule(out_channel, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        num_pools = int(np.log2(spatial))# spatial=16->num_pools=4 spatial=32->5
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(out_c, out_c)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(11, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64))
        # 骨干网
        modules = [
            # 低层
            bottleneck_IR_SE(in_channel=64, out_channel=64, stride=2),
            bottleneck_IR_SE(in_channel=64, out_channel=64, stride=1),
            bottleneck_IR_SE(in_channel=64, out_channel=64, stride=1),
            bottleneck_IR_SE(in_channel=64, out_channel=128, stride=2),
            bottleneck_IR_SE(in_channel=128, out_channel=128, stride=1),
            bottleneck_IR_SE(in_channel=128, out_channel=128, stride=1),
            bottleneck_IR_SE(in_channel=128, out_channel=128, stride=1),
            # 中层
            bottleneck_IR_SE(in_channel=128, out_channel=256, stride=2),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            bottleneck_IR_SE(in_channel=256, out_channel=256, stride=1),
            # 高层
            bottleneck_IR_SE(in_channel=256, out_channel=512, stride=2),
            bottleneck_IR_SE(in_channel=512, out_channel=512, stride=1),
            bottleneck_IR_SE(in_channel=512, out_channel=512, stride=1),
        ]
        # 骨干网
        self.body = nn.Sequential(*modules)
        # 风格编码器论文中的map2style
        styles = [
            # 低层卷积层个数少，学习到的特征少
            GradualStyleBlock(in_c=512,out_c=512,spatial=16), # 0 spatial只是影响卷积层的个数
            GradualStyleBlock(in_c=512, out_c=512, spatial=16),# 1
            GradualStyleBlock(in_c=512, out_c=512, spatial=16),# 2
            # 中层
            GradualStyleBlock(in_c=512, out_c=512, spatial=32),# 3
            GradualStyleBlock(in_c=512, out_c=512, spatial=32),# 4
            GradualStyleBlock(in_c=512, out_c=512, spatial=32),# 5
            GradualStyleBlock(in_c=512, out_c=512, spatial=32),# 6
            # 高层卷积层个数多，学习到的特征多
            GradualStyleBlock(in_c=512, out_c=512, spatial=64),  # 7
            GradualStyleBlock(in_c=512, out_c=512, spatial=64),  # 8
            GradualStyleBlock(in_c=512, out_c=512, spatial=64),  # 9
            GradualStyleBlock(in_c=512, out_c=512, spatial=64),  # 10
            GradualStyleBlock(in_c=512, out_c=512, spatial=64),  # 11
            GradualStyleBlock(in_c=512, out_c=512, spatial=64),  # 12
            GradualStyleBlock(in_c=512, out_c=512, spatial=64),  # 13
            GradualStyleBlock(in_c=512, out_c=512, spatial=64),  # 14
            GradualStyleBlock(in_c=512, out_c=512, spatial=64),  # 15
        ]
        self.styles = nn.Sequential(*styles)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):# x [1,11,256,256]
        x = self.input_layer(x) # x [1,64,256,256]
        modulelist = list(self.body._modules.values())
        # 从指定 layer （第 6，20 和 23 层）拿中间的特征图，其实回看之前的block可以发现这几个层刚好属于分界点，
        # 可以认为是FPN结构中coarse、mid、fine的分界点，输入x的维度分别为128，256，512。
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x # c1[1,128,64,64]
            elif i == 20:
                c2 = x #c2 [1,256,32,32]
            elif i == 23:
                c3 = x #c3 [1,512,16,16]

        # 向list添加latent code,，最后将其送入w+空间了。
        latents = []
        latents.append(self.styles[0](c3)) # [1,512]
        latents.append(self.styles[1](c3)) # [1,512]
        latents.append(self.styles[2](c3)) # [1,512]

        #用于上采样+合并两个特征图c2和c3
        y1 = self.latlayer1(c2) # [1,512,32,32]
        _, _, H, W = y1.size() # [32,32]
        p2 = F.interpolate(c3, size=(H, W), mode='bilinear', align_corners=True) + y1 # [1,512,32,32]

        latents.append(self.styles[3](p2)) # [1,512]
        latents.append(self.styles[4](p2)) # [1,512]
        latents.append(self.styles[5](p2)) # [1,512]
        latents.append(self.styles[6](p2)) # [1,512]

        # 用于上采样+合并两个特征图c1和p2
        y2 = self.latlayer2(c1)  # [1,512,64,64]
        _, _, H, W = y2.size()  # [64,64]
        p1 = F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=True) + y2  # [1,512,64,64]

        latents.append(self.styles[7](p1)) # [1,512]
        latents.append(self.styles[8](p1)) # [1,512]
        latents.append(self.styles[9](p1)) # [1,512]
        latents.append(self.styles[10](p1)) # [1,512]
        latents.append(self.styles[11](p1)) # [1,512]
        latents.append(self.styles[12](p1)) # [1,512]
        latents.append(self.styles[13](p1)) # [1,512]
        latents.append(self.styles[14](p1)) # [1,512]
        latents.append(self.styles[15](p1)) # [1,512]

        # latents 有16个 [1，512]
        out = torch.stack(latents, dim=1) # [1,16,512,512]
        return out


