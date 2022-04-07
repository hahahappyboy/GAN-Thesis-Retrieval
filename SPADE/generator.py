import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import torch
from sync_batchnorm import SynchronizedBatchNorm2d
from enocoder import BaseNetwork,ConvEncoder


class SPADEGenerator(BaseNetwork):
    def __init__(self,opt):
        super().__init__()
        self.fc = nn.Linear(in_features=256, out_features=1024*4*8)

        self.head_0 = SPADEResnetBlock(1024,1024,opt.semantic_nc)
        self.G_middle_0 = SPADEResnetBlock(1024,1024,opt.semantic_nc)
        self.G_middle_1 = SPADEResnetBlock(1024,1024,opt.semantic_nc)

        self.up_0 = SPADEResnetBlock(1024, 512, opt.semantic_nc)
        self.up_1 = SPADEResnetBlock(512, 256, opt.semantic_nc)
        self.up_2 = SPADEResnetBlock(256, 128, opt.semantic_nc)
        self.up_3 = SPADEResnetBlock(128, 64, opt.semantic_nc)

        final_nc = 64

        self.conv_img = nn.Sequential(
            nn.LeakyReLU(2e-1),
            nn.Conv2d(final_nc, 3, 3, padding=1)
        )
        self.up = nn.Upsample(scale_factor=2)


    def forward(self, input, z=None):# input=[1,36,256,512] z=[1,256]
        seg = input # [1,36,256,512]
        if z is None:
            z = torch.randn(input.size(0), 256,
                            dtype=torch.float32, device=input.get_device())
        x = self.fc(z)
        x = x.view(-1, 1024, 4, 8) # [1,1024,4,8]

        x = self.head_0(x, seg) # [1,1024,4,8]

        x = self.up(x) # [1,1024,8,16]
        x = self.G_middle_0(x, seg)

        x = self.up(x)  # [1,1024,16,32]
        x = self.G_middle_1(x, seg) # [1,1024,16,32]

        x = self.up(x)  # [1,1024,32,64]
        x = self.up_0(x, seg) # [1,512,32,64]

        x = self.up(x) # [1,512,64,128]
        x = self.up_1(x, seg) # [1,256,64,128]

        x = self.up(x)  # [1,256,128,256]
        x = self.up_2(x, seg)  # [1,128,128,256]

        x = self.up(x)  # [1,128,256,512]
        x = self.up_3(x, seg)  # [1,64,256,512]

        x = self.conv_img(x) # [1,3,256,512]

        x = F.tanh(x)

        return x

class SPADEResnetBlock(nn.Module):
    """
        参数，fin 输入通道数
            fout 输出通道数
    """
    def __init__(self, fin, fout,semantic_nc):
        super().__init__()

        self.learned_shortcut = (fin != fout)
        if self.learned_shortcut:# 跳跃连接
            self.norm_s = SPADE(fin, semantic_nc)
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.conv_s = torch.nn.utils.spectral_norm(self.conv_s)

        fmiddle = min(fin, fout)


        self.norm_0 = SPADE(fin, semantic_nc)
        self.act_0 = nn.LeakyReLU(2e-1)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_0 = torch.nn.utils.spectral_norm(self.conv_0)

        self.norm_1 = SPADE(fmiddle, semantic_nc)
        self.act_1 = nn.LeakyReLU(2e-1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.conv_1 = torch.nn.utils.spectral_norm(self.conv_1)


    def forward(self, x, seg):# [1,1024,4,8] [1,36,256,512]
        if self.learned_shortcut:# 跳跃连接改变通道数
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.act_0(self.norm_0(x, seg)))
        dx = self.conv_1(self.act_1(self.norm_1(dx, seg)))
        out = x_s + dx # 直接相加
        return out

class SPADE(nn.Module):
    """
        参数：norm_nc 为SPADE的输出
            label_nc segmap通道数
    """
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


    def forward(self, x, segmap):## [1,1024,4,8] [1,36,256,512]
        # SyncBatchNorm
        normalized = self.param_free_norm(x)# [1,1024,4,8]

        # Resize 把segmap裁剪为和x一样大小
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')# [1,36,4,8]

        actv = self.mlp_shared(segmap)# [1,128,4,8]
        gamma = self.mlp_gamma(actv)# [1,1024,4,8]
        beta = self.mlp_beta(actv)# [1,1024,4,8]
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

