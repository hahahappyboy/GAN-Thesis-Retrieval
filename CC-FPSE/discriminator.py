import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import torch
from sync_batchnorm import SynchronizedBatchNorm2d
from generator import BaseNetwork
class FPSEDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()

        label_nc = opt.semantic_nc
        # 下采样
        self.enc1 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(64,affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.enc2 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(128,affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.enc3 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.enc4 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(512, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.enc5 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)),
            nn.InstanceNorm2d(512, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        # 上采样
        self.lat2 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=1)),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.lat3 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=1)),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.lat4 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(512, 256, kernel_size=1)),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.lat5 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(512, 256, kernel_size=1)),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        # 上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        # 最后的层
        self.final2 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.final3 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.final4 = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Conv2d(256, 128, kernel_size=3, padding=1)),
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, True)
        )
        self.tf = nn.Conv2d(128, 1, kernel_size=1)
        self.seg = nn.Conv2d(128, 128, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, 128, kernel_size=1)
    def forward(self, fake_and_real_img, segmap):# [2,3,512,512] ,[2,5,512,512]
        # 下采样
        feat11 = self.enc1(fake_and_real_img)# [2,64,256,256]
        feat12 = self.enc2(feat11)# [2,128,128,128]
        feat13 = self.enc3(feat12)# [2,256,64,64]
        feat14 = self.enc4(feat13)# [2,512,32,32]
        feat15 = self.enc5(feat14)# [2,512,16,16]
        # 特征匹配损失用到的中间特征
        feats = [feat12, feat13, feat14, feat15]

        # 上采样+跳远
        feat25 = self.lat5(feat15) # [2,256,16,16]
        feat24 = self.up(feat25) + self.lat4(feat14)# [2,256,32,32]
        feat23 = self.up(feat24) + self.lat3(feat13)# [2,256,64,64]
        feat22 = self.up(feat23) + self.lat2(feat12)# [2,256,128,128]
        # 上采样+跳远后再经过一层卷积，降维
        feat32 = self.final2(feat22)# [2,128,128,128]
        feat33 = self.final3(feat23)# [2,128,64,64]
        feat34 = self.final4(feat24)# [2,128,32,32]
        # Patch预测真假
        patch_pred2 = self.tf(feat32)# [2,1,128,128]
        patch_pred3 = self.tf(feat33)# [2,1,64,64]
        patch_pred4 = self.tf(feat34)# [2,1,32,32]

        # segmap下采样到与u-net输出相同的分辨率
        segemb = self.embedding(segmap)# [2,128,512,512]
        segemb = F.avg_pool2d(segemb, kernel_size=2, stride=2)# [2,128,256,256]
        segemb2 = F.avg_pool2d(segemb, kernel_size=2, stride=2)# [2,128,128,128]
        segemb3 = F.avg_pool2d(segemb2, kernel_size=2, stride=2)# [2,128,64,64]
        segemb4 = F.avg_pool2d(segemb3, kernel_size=2, stride=2)# [2,128,32,32]

        # 做内积得到语义匹配分数
        seg2 = self.seg(feat32)  # [2,128,128,128]
        seg3 = self.seg(feat33)  # [2,128,64,64]
        seg4 = self.seg(feat34)  # [2,128,32,32]

        segemb_pred2 = torch.mul(segemb2, seg2)# [2, 128, 128, 128]
        segemb_pred2 = segemb_pred2.sum(dim=1, keepdim=True)# [2, 1, 128, 128]
        segemb_pred3 = torch.mul(segemb3, seg3)# [2, 128, 64, 64]
        segemb_pred3 = segemb_pred3.sum(dim=1, keepdim=True)# [2, 1, 64, 64]
        segemb_pred4 = torch.mul(segemb4, seg4)# [2, 128, 32, 32]
        segemb_pred4 = segemb_pred4.sum(dim=1, keepdim=True)# [2, 1, 32, 32]

        pred2 = patch_pred2+segemb_pred2# [2, 1, 128, 128]
        pred3 = patch_pred3+segemb_pred3# [2, 1, 64, 64]
        pred4 = patch_pred4+segemb_pred4# [2, 1, 32, 32]

        results = [pred2, pred3, pred4]

        return [feats, results]

