import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ConvBlock(nn.Module): # 子模块
    def __init__(self, dim_in, dim_out):
        super(ConvBlock, self).__init__()

        self.dim_out = dim_out
        self.ConvBlock1 =nn.Sequential(
            nn.InstanceNorm2d(dim_in),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(dim_in, dim_out//2, kernel_size=(3, 3), stride=(1, 1), bias=False) # 通道数减半 64->32
        )

        self.ConvBlock2 = nn.Sequential(
            nn.InstanceNorm2d(dim_out // 2),
            nn.ReLU(True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(dim_out // 2, dim_out // 4, kernel_size=(3, 3), stride=(1, 1), bias=False) # 通道数再减半 32->16
        )

        self.ConvBlock3 = nn.Sequential(
            nn.InstanceNorm2d(dim_out // 4),
            nn.ReLU(True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(dim_out // 4, dim_out // 4, kernel_size=(3, 3), stride=(1, 1), bias=False) # 通道数不变 16->16
        )

        self.ConvBlock4 = nn.Sequential(
            nn.InstanceNorm2d(dim_in),
            nn.ReLU(True),
            nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), stride=(1, 1), bias=False)
        )

    def forward(self, x):
        residual = x # [1，64，256，256]

        x1 = self.ConvBlock1(x) # [1，32，256，256]
        x2 = self.ConvBlock2(x1) # [1，16，256，256]
        x3 = self.ConvBlock3(x2) # [1，16，256，256]
        out = torch.cat((x1, x2, x3), 1) # 通道拼接 [1，64，256，256]

        if residual.size(1) != self.dim_out: # 如果输入输出通道数相同则不用，不同就改为相同
            residual = self.ConvBlock4(residual)

        return residual + out


class HourGlassBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(HourGlassBlock, self).__init__()
        self.ConvBlock1_1 = ConvBlock(dim_in, dim_out)
        self.ConvBlock1_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock2_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock3_2 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_1 = ConvBlock(dim_out, dim_out)
        self.ConvBlock4_2 = ConvBlock(dim_out, dim_out)

        self.ConvBlock5 = ConvBlock(dim_out, dim_out)

        self.ConvBlock6 = ConvBlock(dim_out, dim_out)
        self.ConvBlock7 = ConvBlock(dim_out, dim_out)
        self.ConvBlock8 = ConvBlock(dim_out, dim_out)
        self.ConvBlock9 = ConvBlock(dim_out, dim_out)

    def forward(self, x): # x:[1，64，256，256]
        skip1 = self.ConvBlock1_1(x) # skip1:[1，64，256，256]

        down1 = torch.nn.functional.avg_pool2d(x, 2) # 两倍下采样 [1，64，128，128]
        # 注意这两个输入都是down1
        down1 = self.ConvBlock1_2(down1)# [1，64，128，128]

        skip2 = self.ConvBlock2_1(down1) # [1，64，128，128]

        down2 = F.avg_pool2d(down1, 2) # [1，64，64，64]
        down2 = self.ConvBlock2_2(down2)  # [1，64，64，64]
        skip3 = self.ConvBlock3_1(down2) # [1，64，64，64]


        down3 = F.avg_pool2d(down2, 2) # [1，64，32，32]
        down3 = self.ConvBlock3_2(down3) # [1，64，32，32]
        skip4 = self.ConvBlock4_1(down3) # [1，64，32，32]

        down4 = F.avg_pool2d(down3, 2)  # [1，64，16，16]
        down4 = self.ConvBlock4_2(down4) # [1，64，16，16]
        center = self.ConvBlock5(down4)  # [1，64，16，16]

        up4 = self.ConvBlock6(center) # [1，64，16，16]
        up4 = F.upsample(up4, scale_factor=2) # [1，64，32，32]
        up4 = skip4 + up4 # [1，64，32，32]

        up3 = self.ConvBlock7(up4) # [1，64，32，32]
        up3 = F.upsample(up3, scale_factor=2) # [1，64，64，64]
        up3 = skip3 + up3 # [1，64，64，64]

        up2 = self.ConvBlock8(up3) # [1，64，64，64]
        up2 = F.upsample(up2, scale_factor=2) # [1，64，128，128]
        up2 = skip2 + up2 # [1，64，128，128]

        up1 = self.ConvBlock9(up2) # [1，64，128，128]
        up1 = F.upsample(up1, scale_factor=2) # [1，64，256，256]
        up1 = skip1 + up1 # [1，64，256，256]

        return up1

class HourGlass(nn.Module):
    def __init__(self, dim_in, dim_out, use_res=True):# 64 64   True
        super(HourGlass, self).__init__()
        self.use_res = use_res

        self.HG = nn.Sequential(
            HourGlassBlock(dim_in, dim_out), # [1,64,256,256]
            ConvBlock(dim_out, dim_out), # [1,64,256,256]
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1, bias=False), # [1,64,256,256]
            nn.InstanceNorm2d(dim_out),
            nn.ReLU(True)
        )

        self.Conv1 = nn.Conv2d(dim_out, 3, kernel_size=1, stride=1)

        if self.use_res:
            self.Conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1)
            self.Conv3 = nn.Conv2d(3, dim_out, kernel_size=1, stride=1)

    def forward(self, x): # [1,64,256,256]
        ll = self.HG(x)  # [1,64,256,256]
        tmp_out = self.Conv1(ll)  # [1,3,256,256]

        if self.use_res:
            ll = self.Conv2(ll) # [1,64,256,256]
            tmp_out_ = self.Conv3(tmp_out) # [1,64,256,256]
            return x + ll + tmp_out_

        else:
            return tmp_out









