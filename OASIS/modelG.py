import torch.nn as nn
import torch
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d


class OASIS_Generator(nn.Module):
    def __init__(self,semantic_nc):
        super().__init__()
        self.fc = nn.Conv2d(183+64,1024,kernel_size=(3,3),stride=(1,1),padding=(1,1))

        self.ResnetBlock_with_SPADE_0 = ResnetBlock_with_SPADE(1024,1024,semantic_nc+64)
        self.ResnetBlock_with_SPADE_1 = ResnetBlock_with_SPADE(1024,1024,semantic_nc+64)
        self.ResnetBlock_with_SPADE_2 = ResnetBlock_with_SPADE(1024,512,semantic_nc+64)
        self.ResnetBlock_with_SPADE_3 = ResnetBlock_with_SPADE(512, 256, semantic_nc + 64)
        self.ResnetBlock_with_SPADE_4 = ResnetBlock_with_SPADE(256, 128, semantic_nc + 64)
        self.ResnetBlock_with_SPADE_5 = ResnetBlock_with_SPADE(128, 64, semantic_nc + 64)

        final_nc = 64


        self.conv_img = nn.Sequential(
            nn.LeakyReLU(2e-1),
            nn.Conv2d(final_nc, 3, 3, padding=1)
        )
        self.up = nn.Upsample(scale_factor=2)

    def forward(self,input):
        seg = input# [1,183,512,512] # 只有one-hot编码，没有边缘
        # label加入随机噪声
        z = torch.randn(seg.size(0), 64) # [1,64]
        z = z.view(z.size(0), 64, 1, 1)# [1,64,1,1]
        z = z.expand(z.size(0), 64 , seg.size(2), seg.size(3)) # [1,64,512,512]
        # 与输入label拼接
        seg = torch.cat((z,seg),dim=1) # [1,64+183=247,512,512]

        # 第一层输入
        x = F.interpolate(seg,size=(16,16))
        x = self.fc(x) #[1,1024,16,16]
        # 第二层
        x = self.ResnetBlock_with_SPADE_0(x,seg) #[1,1024,16,16]
        x = self.up(x) #[1,1024,32,32]
        # 第三层
        x = self.ResnetBlock_with_SPADE_1(x,seg)#[1,1024,32,32]
        x = self.up(x)  #[1,1024,64,64]
        # 第四层
        x = self.ResnetBlock_with_SPADE_2(x,seg)  #[1,512,64,64]
        x = self.up(x)  #[1,512,128,128]
        # 第五层
        x = self.ResnetBlock_with_SPADE_3(x, seg)#[1,256,128,128]
        x = self.up(x) #[1,256,256,256]
        # 第六层
        x = self.ResnetBlock_with_SPADE_4(x, seg) #[1,128,256,256]
        x = self.up(x)  #[1,128,512,512]
        # 第七层
        x = self.ResnetBlock_with_SPADE_5(x, seg) # [1,64,512,512]

        x = self.conv_img(x)  # [1,3,512,512]

        x = F.tanh(x)

        return x








class ResnetBlock_with_SPADE(nn.Module):
    """
           参数，fin 输入通道数
               fout 输出通道数
    """

    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        # 如果输入通道数不等于输出通道数
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        if self.learned_shortcut:  # 跳跃连接
            self.norm_s = SPADE(fin, semantic_nc)
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.conv_s = torch.nn.utils.spectral_norm(self.conv_s)


        self.norm_0 = SPADE(fin, semantic_nc)
        self.act_0 = nn.LeakyReLU(2e-1)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_0 = torch.nn.utils.spectral_norm(self.conv_0)

        self.norm_1 = SPADE(fmiddle, semantic_nc)
        self.act_1 = nn.LeakyReLU(2e-1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.conv_1 = torch.nn.utils.spectral_norm(self.conv_1)

    def forward(self, x, seg):  # [1,1024,4,8] [1,36,256,512]
        if self.learned_shortcut:  # 跳跃连接改变通道数
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.act_0(self.norm_0(x, seg)))
        dx = self.conv_1(self.act_1(self.norm_1(dx, seg)))
        out = x_s + dx  # 直接相加
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



