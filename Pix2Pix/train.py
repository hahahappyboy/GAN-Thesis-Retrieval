from model import UnetGenerator,NLayerDiscriminator
import torch
import torch.nn as nn
# 真实图片
real_A = torch.randn((1,3,256,256))
real_B = torch.randn((1,3,256,256))

real_label = torch.FloatTensor(1, 1, 30, 30).fill_(1.0)
fake_label = torch.FloatTensor(1, 1, 30, 30).fill_(0.0)

netG = UnetGenerator()
netD = NLayerDiscriminator()

criterionGAN = nn.BCEWithLogitsLoss()
criterionL1 = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

fake_B = netG(real_A) # 生成伪造图片

##########
# 更新鉴别器
##########
fake_AB = torch.cat((real_A, fake_B), 1) # 拼接
pred_fake = netD(fake_AB.detach()) # 预测伪造图片

loss_D_fake = criterionGAN(pred_fake,fake_label) # 伪造图片loss

real_AB = torch.cat((real_A, real_B), 1)
pred_real = netD(real_AB)
loss_D_real = criterionGAN(pred_fake,real_label)

loss_D = (loss_D_fake + loss_D_real) * 0.5

optimizer_D.zero_grad()
loss_D.backward()
optimizer_D.step()

##########
# 更新生成器
##########
fake_AB = torch.cat((real_A, fake_B), 1)
pred_fake = netD(fake_AB)

loss_G_GAN = criterionGAN(pred_fake,real_label)
loss_G_L1 = criterionL1(fake_B, real_B) * 100
loss_G = loss_G_GAN + loss_G_L1

optimizer_G.zero_grad()
loss_G.backward()
optimizer_G.step()