from modelD import OASIS_Discriminator
from modelG import OASIS_Generator
from loss import VGGLoss, losses_computer
import torch
from utils import generate_labelmix
semantic_nc = 183
netG = OASIS_Generator(semantic_nc)
netD = OASIS_Discriminator(semantic_nc)

VGG_loss = VGGLoss()
optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0001, betas=(0.0, 0.999))

image = torch.randn((1,3,512,512))
label = torch.randn((1,semantic_nc,512,512))


"""更新生成器"""
loss_G = 0
netG.zero_grad()
fake = netG(label)
output_D = netD(fake)

loss_G_adv = losses_computer().loss(input=output_D,label=label,for_real=True,contain_dontcare_label=True)
loss_G_vgg = None # 不适用VGGLoSS
loss_G += loss_G_adv
loss_G = loss_G.mean()


loss_G.backward()
optimizer_G.step()
"""更新鉴别器"""
loss_D = 0
with torch.no_grad():
    fake = netG(label)
# 假的
output_D_fake = netD(fake)
loss_D_fake = losses_computer().loss(output_D_fake,label,for_real=False,contain_dontcare_label=True) # 接近0越好
loss_D += loss_D_fake
# 真的
output_D_real = netD(image)
loss_D_real = netD(image)
output_D_real = losses_computer().loss(output_D_real,label,for_real=True,contain_dontcare_label=True)
loss_D += output_D_real
# 标签混合
mixed_inp, mask = generate_labelmix(label=label,fake_image=fake,real_image=image)
output_D_mixed = netD(mixed_inp)
# 标签混合loss
loss_D_lm = 10.0 * losses_computer().loss_labelmix(mask,output_D_mixed,output_D_fake,output_D_real)
loss_D += loss_D_lm

loss_D.backward()
optimizer_D.step()
