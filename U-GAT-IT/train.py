
import itertools
from torch import nn as nn
from model import ResnetGenerator, Discriminator
import torch
from utils import RhoClipper

Rho_clipper = RhoClipper(0, 1)

genA2B = ResnetGenerator()
genB2A = ResnetGenerator()
disGA = Discriminator(layer=7)
disGB = Discriminator(layer=7)
disLA = Discriminator(layer=5)
disLB = Discriminator(layer=5)

L1_loss = nn.L1Loss()
MSE_loss = nn.MSELoss()
BCE_loss = nn.BCEWithLogitsLoss()

G_optim = torch.optim.Adam(itertools.chain(genA2B.parameters(),genB2A.parameters()),lr=0.0001,betas=(0.5,0.999),weight_decay=0.0001)
D_optim = torch.optim.Adam(itertools.chain(disGA.parameters(), disGB.parameters(), disLA.parameters(), disLB.parameters()), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)

real_A = torch.randn((1,3,256,256))
real_B = torch.randn((1,3,256,256))

##########
# 跟新鉴别器
##########
fake_A2B, _, _ = genA2B(real_A) # fake_A2B:伪造的B
fake_B2A, _, _ = genB2A(real_B) # fake_B2A:伪造的A

real_GA_logit, real_GA_cam_logit, _ = disGA(real_A) # real_GA_logit：判断真实A的得分 real_GA_cam_logit：辅助分类器的得分
real_LA_logit, real_LA_cam_logit, _ = disLA(real_A) # 多尺度
real_GB_logit, real_GB_cam_logit, _ = disGB(real_B) # real_GB_logit：判断真实B的得分 real_GB_cam_logit：辅助分类器的得分
real_LB_logit, real_LB_cam_logit, _ = disLB(real_B) # 多尺度

fake_GA_logit, fake_GA_cam_logit, _ = disGA(fake_B2A)
fake_LA_logit, fake_LA_cam_logit, _ = disLA(fake_B2A)
fake_GB_logit, fake_GB_cam_logit, _ = disGB(fake_A2B)
fake_LB_logit, fake_LB_cam_logit, _ = disLB(fake_A2B)

# GANLoss 希望真实A的得分越接近于1越好，希望伪造A的得分越接近于0越好
D_ad_loss_GA = MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit))\
               + MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit))
D_ad_loss_LA = MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit))\
               + MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit))
# GANLoss 希望真实B的得分越接近于1越好，希望伪造B的得分越接近于0越好
D_ad_loss_GB = MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit))\
               + MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit))
D_ad_loss_LB = MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit))\
               + MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit))

# CAMLoss 希望真实A在辅助分类器中的得分越接近于1越好，希望伪造A在辅助分类器的得分越接近于0越好
D_ad_cam_loss_GA = MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit))\
                   + MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit))
D_ad_cam_loss_LA = MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit))\
                   + MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit))
# CAMLoss 希望真实B在辅助分类器中的得分越接近于1越好，希望伪造B在辅助分类器的得分越接近于0越好
D_ad_cam_loss_GB = MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit))\
                   + MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit))
D_ad_cam_loss_LB = MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit))\
                   + MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit))

D_loss_A = 1 * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
D_loss_B = 1 * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

Discriminator_loss = D_loss_A + D_loss_B

D_optim.zero_grad()
Discriminator_loss.backward()
D_optim.step()

##########
# 跟新生成器
##########

fake_A2B, fake_A2B_cam_logit, _ = genA2B(real_A) # fake_A2B:伪造的B fake_A2B_cam_logit：辅助分类器对B的判断
fake_B2A, fake_B2A_cam_logit, _ = genB2A(real_B)
# CycleGAN
fake_A2B2A, _, _ = genB2A(fake_A2B)
fake_B2A2B, _, _ = genA2B(fake_B2A)

# 为了计算一致性损失
fake_A2A, fake_A2A_cam_logit, _ = genB2A(real_A)
fake_B2B, fake_B2B_cam_logit, _ = genA2B(real_B)

fake_GA_logit, fake_GA_cam_logit, _ = disGA(fake_B2A) # 判断伪造的A
fake_LA_logit, fake_LA_cam_logit, _ = disLA(fake_B2A) # 多尺度判断
fake_GB_logit, fake_GB_cam_logit, _ = disGB(fake_A2B) # 判断伪造的B
fake_LB_logit, fake_LB_cam_logit, _ = disLB(fake_A2B) # 多尺度判断

# GANLoss 生成器希望生成的伪造图在判别器的得分越接近1越好
G_ad_loss_GA = MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit))
G_ad_loss_LA = MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit))
G_ad_loss_GB = MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit))
G_ad_loss_LB = MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit))
# CAMLoss 生成器希望生成的伪造图在判别器的辅助分类器得分越接近1越好
G_ad_cam_loss_GA = MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit))
G_ad_cam_loss_LA = MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit))
G_ad_cam_loss_GB = MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit))
G_ad_cam_loss_LB = MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit))

# 循环Loss
G_recon_loss_A = L1_loss(fake_A2B2A, real_A)
G_recon_loss_B = L1_loss(fake_B2A2B, real_B)

# 一致性Loss
G_identity_loss_A = L1_loss(fake_A2A, real_A)
G_identity_loss_B = L1_loss(fake_B2B, real_B)

# CAMLoss 生成器希望源域为0，目标域为1
G_cam_loss_A = BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit)) \
               + BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit))
G_cam_loss_B = BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit)) \
               + BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit))

#  生成器Loss = GANLoss + D的CAMLoss + 循环Loss + 一致性Loss +  G的CAMLoss
G_loss_A = 1 * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + 10 * G_recon_loss_A + 10 * G_identity_loss_A + 1000 * G_cam_loss_A
G_loss_B = 1 * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + 10 * G_recon_loss_B + 10 * G_identity_loss_B + 1000 * G_cam_loss_B


G_optim.zero_grad()
Generator_loss = G_loss_A + G_loss_B
Generator_loss.backward()
G_optim.step()


# 控制AdaILN 和 ILN的rho值在[0,1之间]
genA2B.apply(Rho_clipper)
genB2A.apply(Rho_clipper)











