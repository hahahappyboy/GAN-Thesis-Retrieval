import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import torch
from sync_batchnorm import SynchronizedBatchNorm2d

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class CondConvGenerator(BaseNetwork):
    def __init__(self,opt):
        super().__init__()
        self.fc = nn.Linear(in_features=256, out_features=1024*16*16)

        self.labelenc1 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(5, 64, 3, padding=1)),
            nn.BatchNorm2d(64,affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labelenc2 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1, stride=2)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labelenc3 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1, stride=2)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labelenc4 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1, stride=2)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labelenc5 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1, stride=2)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labelenc6 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1, stride=2)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )

        self.labellat1 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labellat2 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labellat3 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labellat4 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labellat5 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )

        self.labeldec1 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labeldec2 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labeldec3 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labeldec4 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.labeldec5 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 64, 3, padding=1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, True)
        )

        self.head_0 = DepthsepCCBlock(1024,1024,opt.semantic_nc + 64)
        self.G_middle_0 = DepthsepCCBlock(1024,1024,opt.semantic_nc + 64)
        self.G_middle_1 = DepthsepCCBlock(1024,1024,opt.semantic_nc + 64)

        self.up_0 = DepthsepCCBlock(1024, 512,  opt.semantic_nc + 64)
        self.up_1 = DepthsepCCBlock(512, 256,  opt.semantic_nc + 64)
        self.up_2 = DepthsepCCBlock(256, 128,  opt.semantic_nc + 64)
        self.up_3 = DepthsepCCBlock(128, 64,  opt.semantic_nc + 64)
        self.conv_img = nn.Conv2d(64, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, input, z=None): # [1,5,512,512]
        seg = input
        if z is None:
            z = torch.randn(input.size(0), 256,
                            dtype=torch.float32, device=input.get_device())
        x = self.fc(z)
        x = x.view(-1, 1024, 16, 16) # [1,1024,16,16]
        ######## 权重预测 下采样
        seg1 = self.labelenc1(seg) # [1,64,512,512]
        seg2 = self.labelenc2(seg1) # [1,64,256,256]
        seg3 = self.labelenc3(seg2) # [1,64,128,128]
        seg4 = self.labelenc4(seg3) # [1,64,64,64]
        seg5 = self.labelenc5(seg4) # [1,64,32,32]
        seg6 = self.labelenc6(seg5) # [1,64,16,16]

        ######### 权重预测 上采样
        segout1 = seg6 #[1,64,16,16]

        up2 = self.up(segout1) # 上采样[1,64,32,32]
        lat2 = self.labellat1(seg5)  # [1,64,32,32]
        segout2 = up2+lat2 # 跳连 # [1,64,32,32]
        segout2 = self.labeldec1(segout2) # [1,64,32,32]

        up3 = self.up(segout2) # 上采样[1,64,64,64]
        lat3 = self.labellat2(seg4)  # [1,64,64,64]
        segout3 = up3+lat3
        segout3 = self.labeldec2(segout3) # [1,64,64,64]

        up4 = self.up(segout3)  # 上采样[1,64,128,128]
        lat4 = self.labellat3(seg3)  # [1,64,128,128]
        segout4 = up4 + lat4
        segout4 = self.labeldec2(segout4)  # [1,64,128,128]

        up5 = self.up(segout4)  # 上采样[1,64,256,256]
        lat5 = self.labellat4(seg2)  # [1,64,256,256]
        segout5 = up5 + lat5
        segout5 = self.labeldec2(segout5)  # [1,64,256，256]

        up6 = self.up(segout5)  # 上采样[1,64,256,256]
        lat6 = self.labellat5(seg1)  # [1,64,256,256]
        segout6 = up6 + lat6
        segout6 = self.labeldec2(segout6)  # [1,64,256，256]

        # 第一个CC 块
        seg_resize0 = F.interpolate(seg, size=x.size()[2:], mode='nearest') # 把语义图采样从与当前从特征图同样大小 # [1, 5, 16, 16]
        seg_cat0 = torch.cat((seg_resize0, segout1), dim=1) # [1, 64+5, 16, 16]
        x = self.head_0(x,seg_cat0) # [1,1024,16,16]
        # 第二个CC 块
        x = self.up(x) # [1,1024,32,32]
        seg_resize1 = F.interpolate(seg, size=x.size()[2:], mode='nearest') # [1, 5, 32, 32]
        seg_cat1 = torch.cat((seg_resize1, segout2), dim=1) # [1, 64+5, 32, 32]
        x = self.G_middle_0(x,seg_cat1)  # [1,1024,32,32]

        seg_resize2 = F.interpolate(seg, size=x.size()[2:], mode='nearest') # [1, 5, 32, 32]
        seg_cat2 = torch.cat((seg_resize2, segout2), dim=1) # [1, 64+5, 32, 32]
        x = self.G_middle_1(x,seg_cat2)  # [1,1024,32,32]

        x = self.up(x) # [1,1024,64,64]
        seg_resize3 = F.interpolate(seg, size=x.size()[2:], mode='nearest')  # [1, 5, 64, 64]
        seg_cat3 = torch.cat((seg_resize3, segout3), dim=1)  # [1, 64+5, 64, 64]
        x = self.up_0(x, seg_cat3)  # [1,512,64,64]

        x = self.up(x)  # [1,512,128,128]
        seg_resize4 = F.interpolate(seg, size=x.size()[2:], mode='nearest')  # [1, 5, 128, 128]
        seg_cat4 = torch.cat((seg_resize4, segout4), dim=1)  # [1, 64+5, 128, 128]
        x = self.up_1(x, seg_cat4)  # [1,256,128,128]

        x = self.up(x)  # [1,256,256,256]
        seg_resize5 = F.interpolate(seg, size=x.size()[2:], mode='nearest')  # [1, 5, 256, 256]
        seg_cat5 = torch.cat((seg_resize5, segout5), dim=1)  # [1, 64+5, 256, 256]
        x = self.up_2(x, seg_cat5)  # [1,128,256,256]

        x = self.up(x)  # [1,128,512,512]
        seg_resize6 = F.interpolate(seg, size=x.size()[2:], mode='nearest')  # [1, 5, 512, 512]
        seg_cat6 = torch.cat((seg_resize6, segout6), dim=1)  # [1, 64+5, 512, 512]
        x = self.up_3(x, seg_cat6)  # [1,64,512,512]

        x = self.conv_img(F.leaky_relu(x, 2e-1)) # [1, 3, 512, 512]
        x = F.tanh(x)

        return x





class DepthsepCCBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        nhidden = 128
        self.weight_channels = fmiddle * 9
        # 通道卷积权重
        self.gen_weights1 = nn.Sequential(
            nn.Conv2d(semantic_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fin * 9, kernel_size=3, padding=1))

        self.gen_weights2 = nn.Sequential(
            nn.Conv2d(semantic_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fout * 9, kernel_size=3, padding=1))
        # 条件注意力权重预测
        self.gen_se_weights1 = nn.Sequential(
            nn.Conv2d(semantic_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fmiddle, kernel_size=3, padding=1),
            nn.Sigmoid())
        self.gen_se_weights2 = nn.Sequential(
            nn.Conv2d(semantic_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nhidden, fout, kernel_size=3, padding=1),
            nn.Sigmoid())

        self.conv_0 = DepthConv(fin)
        self.norm_0 = nn.BatchNorm2d(fmiddle, affine=True)
        self.conv_1 = nn.Conv2d(fin, fmiddle, kernel_size=1)
        self.norm_1 = nn.BatchNorm2d(fin, affine=True)
        self.conv_2 = DepthConv(fmiddle)
        self.norm_2 = nn.BatchNorm2d(fmiddle, affine=True)
        self.conv_3 = nn.Conv2d(fmiddle, fout, kernel_size=1)

        self.conv_s = torch.nn.utils.spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))
        self.norm_s = SPADE(fin, semantic_nc)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def forward(self, x, seg):# x[1,1024,16,16] seg[1,69,16,16]

        segmap = F.interpolate(seg, size=x.size()[2:], mode='nearest') # [1,69,16,16]
        # 通道卷积权重
        conv_weights1 = self.gen_weights1(segmap) # [1,9216,16,16]
        conv_weights2 = self.gen_weights2(segmap) # [1,9216,16,16]
        # 条件注意力权重预测
        se_weights1 = self.gen_se_weights1(segmap) # [1,1024,16,16]
        se_weights2 = self.gen_se_weights2(segmap) # [1,1024,16,16]

        x_s = self.shortcut(x, segmap) # [1,1024,16,16]

        dx = self.norm_1(x)# norm [1,1024,16,16]
        dx = self.conv_0(dx, conv_weights1)#通道卷积 [1,1024,16,16]
        dx = self.conv_1(dx)# 点卷积 [1,1024,16,16]
        dx = torch.mul(dx, se_weights1)# 条卷注意力 [1,1024,16,16]
        dx = self.actvn(dx) # 激活函数[1,1024,16,16]

        dx = self.norm_2(dx)# norm [1,1024,16,16]
        dx = self.conv_2(dx, conv_weights2) # 通道卷积[1,1024,16,16]
        dx = self.conv_3(dx) # 点卷积 [1,1024,16,16]
        dx = torch.mul(dx, se_weights2) # 条卷注意力 [1,1024,16,16]
        dx = self.actvn(dx)# 激活函数[1,1024,16,16]

        out = x_s + dx

        return out
class DepthConv(nn.Module):
    def __init__(self, fmiddle):
        super().__init__()
        # 提取出滑动的局部区域块，也就是卷积操作中的提取kernel filter对应的滑动窗口,并展平
        self.unfold = nn.Unfold(kernel_size=(3,3), dilation=1, padding=1, stride=1)

        self.norm_layer = nn.BatchNorm2d(fmiddle, affine=True)
    def forward(self, x, conv_weights):# x[1,1024,16,16]  conv_weights[1,9*1024,16,16]

        N, C, H, W = x.size()

        conv_weights = conv_weights.view(N * C, 9 , H , W )# 1024,9,16,16
        x = self.unfold(x).view(N * C, 9 , H , W )# 1024,9,16,16
        x = torch.mul(conv_weights, x).sum(dim=1, keepdim=False).view(N, C, H, W)# x[1,1024,16,16]

        return x
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