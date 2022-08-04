import torch.nn as nn
import torch
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d





class OASIS_Discriminator(nn.Module):
    def __init__(self,semantic_nc):
        super().__init__()

        self.down0 = residual_bolck_D(3,128,is_first_layer=True,is_encoder=True)
        self.down1 = residual_bolck_D(128,128,is_first_layer=False,is_encoder=True)
        self.down2 = residual_bolck_D(128,256,is_first_layer=False,is_encoder=True)
        self.down3 = residual_bolck_D(256,256,is_first_layer=False,is_encoder=True)
        self.down4 = residual_bolck_D(256,512,is_first_layer=False,is_encoder=True)
        self.down5 = residual_bolck_D(512,512,is_first_layer=False,is_encoder=True)

        self.up0 = residual_bolck_D(512,512,is_first_layer=False,is_encoder=False)
        self.up1 = residual_bolck_D(1024,256,is_first_layer=False,is_encoder=False)
        self.up2 = residual_bolck_D(512,256,is_first_layer=False,is_encoder=False)
        self.up3 = residual_bolck_D(512,128,is_first_layer=False,is_encoder=False)
        self.up4 = residual_bolck_D(256,128,is_first_layer=False,is_encoder=False)
        self.up5 = residual_bolck_D(256,64,is_first_layer=False,is_encoder=False)
        self.layer_up_last = nn.Conv2d(64,semantic_nc+1,kernel_size=1,stride=1,padding=0)


    def forward(self, input): # [1,3,512,512]
        x = input

        d0 = self.down0(x) # [1,128,256,256]
        d1 = self.down1(d0) # [1,128,128,128]
        d2 = self.down2(d1) # [1,256,64,64]
        d3 = self.down3(d2) # [1,256,32,32]
        d4 = self.down4(d3) # [1,512,16,16]
        d5 = self.down5(d4) # [1,512,8,8]

        u0 = self.up0(d5) # [1,512,16,16]
        u1 = self.up1(torch.cat((d4,u0),dim=1)) # [1,256,32,32]
        u2 = self.up2(torch.cat((d3,u1),dim=1)) # [1,256,64,64]
        u3 = self.up3(torch.cat((d2,u2),dim=1)) # [1,128,128,128]
        u4 = self.up4(torch.cat((d1,u3),dim=1)) # [1,128,256,256]
        u5 = self.up5(torch.cat((d0,u4),dim=1)) # [1,64,512,512]

        ans = self.layer_up_last(u5) # [1,184,512,512]
        return ans

class residual_bolck_D(nn.Module):

    """
        fin:输入通道数
        fout:输出通道数
    """
    def __init__(self, fin, fout,is_encoder, is_first_layer=False):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.is_first_layer = is_first_layer
        self.is_encoder = is_encoder
        fmiddle = fout
        norm_layer = torch.nn.utils.spectral_norm # 谱归一化
        if is_first_layer: # 是第一个residual_bolck_D
            self.conv1 = nn.Sequential(
                norm_layer(nn.Conv2d(fin,fmiddle,kernel_size=3,stride=1,padding=1))
            )
        else:
            if is_encoder:# 下采样
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(0.2,False),
                    norm_layer(nn.Conv2d(fin,fmiddle,kernel_size=3,stride=1,padding=1))
                )
            else:# 上采样
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(0.2, False),
                    nn.Upsample(scale_factor=2),
                    norm_layer(nn.Conv2d(fin, fmiddle, kernel_size=3, stride=1, padding=1))
                )

        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            norm_layer(nn.Conv2d(fmiddle,fout,kernel_size=3,stride=1,padding=1))
        )

        if self.learned_shortcut:# 跳跃链接
            self.conv_s = norm_layer(nn.Conv2d(fin,fout,kernel_size=1,stride=1,padding=0))
        if is_encoder:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Upsample(scale_factor=2)

    def shortcut(self, x):
        if self.is_first_layer: # 是第一层
            if self.is_encoder:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        return x_s

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.is_encoder:
            dx = self.sampling(dx)
        out = x_s + dx
        return out
