import numpy as np
import torch
import os
from torch.autograd import Variable
import torch.nn as nn

from model import Vgg19 , GlobalGenerator ,MultiscaleDiscriminator
from util import VGGLoss

netG = GlobalGenerator()
netD = MultiscaleDiscriminator()


criterionGAN = nn.MSELoss() # GAN LOSS
criterionFeat = nn.L1Loss() # 特征 LOSS
criterionVGG = VGGLoss() # VGG LOSS

optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

real_image = torch.randn((1,3,256,256))
input_label = torch.randn((1,3,256,256))

input_concat = input_label
fake_image = netG(input_concat)

"""训练鉴别器"""
# 伪造数据的GANloss
input_concat = torch.cat((input_label, fake_image.detach()), dim=1)
pred_fake_pool = netD(input_concat)

fake_tensor_0 = torch.Tensor(pred_fake_pool[0][-1].size()).fill_(0)
loss_D_fake_0 = criterionGAN(pred_fake_pool[0][-1], fake_tensor_0) # 鉴别器0，的最后一层输出的loss
fake_tensor_1 = torch.Tensor(pred_fake_pool[1][-1].size()).fill_(0)
loss_D_fake_1 = criterionGAN(pred_fake_pool[1][-1], fake_tensor_1) # 鉴别器1，的最后一层输出的loss
loss_D_fake = loss_D_fake_0 + loss_D_fake_1

# 真实数据的GANloss
input_concat = torch.cat((input_label, real_image.detach()), dim=1)
pred_real = netD(input_concat)

real_tensor_0 = torch.Tensor(pred_real[0][-1].size()).fill_(1)
loss_D_real_0 = criterionGAN(pred_real[0][-1], real_tensor_0) # 鉴别器0，的最后一层输出的loss
real_tensor_1 = torch.Tensor(pred_real[1][-1].size()).fill_(1)
loss_D_real_1 = criterionGAN(pred_real[1][-1], real_tensor_1) # 鉴别器1，的最后一层输出的loss
loss_D_real = loss_D_real_0 + loss_D_real_1

"""训练生成器"""
pred_fake = netD(torch.cat((input_label, fake_image), dim=1))
real_tensor_0 = torch.Tensor(pred_fake[0][-1].size()).fill_(1)
loss_G_fake_0 = criterionGAN(pred_fake[0][-1], real_tensor_0)
real_tensor_1 = torch.Tensor(pred_fake[1][-1].size()).fill_(1)
loss_G_fake_1 = criterionGAN(pred_fake[1][-1], real_tensor_1)

loss_G_GAN = loss_G_fake_0 + loss_G_fake_1

# GAN的特征匹配损失
loss_G_GAN_Feat = 0
feat_weights = 1
D_weights = 0.5
lambda_feat = 10
for i in range(2):# 计算两个鉴别器的特征匹配损失
    for j in range(len(pred_fake[i]) - 1):#  计算每层特征                                        注意这里有个detach
        loss_G_GAN_Feat += D_weights * feat_weights * criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * lambda_feat

# VGG特征匹配损失
loss_G_VGG = criterionVGG(fake_image, real_image) * lambda_feat

loss_D = (loss_D_real + loss_D_fake) * 0.5
loss_G = loss_G_GAN + loss_G_GAN_Feat +loss_G_VGG

optimizer_G.zero_grad()
loss_G.backward()
optimizer_G.step()

optimizer_D.zero_grad()
loss_D.backward()
optimizer_D.step()


