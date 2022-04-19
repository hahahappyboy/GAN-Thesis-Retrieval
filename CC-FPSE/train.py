
import torch
from enocoder import ConvEncoder
from generator import CondConvGenerator
from discriminator import FPSEDiscriminator
from loss import KLDLoss,VGGLoss,HingeLoss
import argparse

parser = argparse.ArgumentParser(description="CC-FPSE")
#  有4类别 实例映射为1类
parser.add_argument('--semantic_nc', type=int, default=5,
                    help='语义图的通道数')
opt = parser.parse_args()


input_semantics = torch.randn((1,opt.semantic_nc,512,512))
real_image = torch.randn((1,3,512,512))

netE = ConvEncoder()
netG = CondConvGenerator(opt)
netD = FPSEDiscriminator(opt)


FeatLoss = torch.nn.L1Loss()
VGGLoss = VGGLoss()
KLDLoss = KLDLoss()

G_params = list(netG.parameters())
G_params += list(netE.parameters())
D_params = list(netD.parameters())
optimizer_G = torch.optim.Adam(G_params, lr=0.0001, betas=(0.0, 0.9))
optimizer_D = torch.optim.Adam(D_params, lr=0.0001, betas=(0.0, 0.9))
G_losses = {}

z, mu, logvar = netE(real_image)

KLD_loss = KLDLoss(mu, logvar)*0.05
G_losses['KLD'] = KLD_loss

fake_image = netG(input_semantics, z=z) # [1,3,512,512]


"""更新生成器"""""
fake_and_real_img = torch.cat([fake_image, real_image], dim=0)  # [2,3,512,512]
fake_and_real_segmap = torch.cat((input_semantics, input_semantics), dim=0) # [2,5,512,512]
# 鉴别器输出
# 特征匹配 [2,128,128,128] [2,256,64,64] [2,512,32,32] [2,512,16,16]
# 预测patch [2,1,128,128] [2,1,64,64] [2,1,32,32]
pred = netD(fake_and_real_img,fake_and_real_segmap)

feat_fake = []
pred_fake = []
feat_real = []
pred_real = []
# 把正图和假图的预测划分出来
for p in pred[0]:# 特征匹配
    feat_fake.append(p[:p.size(0) // 2])
    feat_real.append(p[p.size(0) // 2:])
for p in pred[1]:# 预测patch
    pred_fake.append(p[:p.size(0) // 2])
    pred_real.append(p[p.size(0) // 2:])

##### 特征匹配损失
GAN_Feat_loss = 0
for i in range(len(feat_fake)):  # 两个生成器
    GAN_Feat_loss += FeatLoss(feat_fake[i],feat_real[i].detach())*10/len(feat_fake)
G_losses['GAN_Feat'] = GAN_Feat_loss

##### GAN损失
GANloss_G = 0
for pred_i in pred_fake:
    loss_tensor = HingeLoss(pred_i, target_is_real=True, for_discriminator=False)
    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
    GANloss_G += new_loss
G_losses['GAN'] = GANloss_G / len(pred_fake)

##### 特征匹配损失
G_losses['VGG'] = VGGLoss(fake_image, real_image) * 10

##### 总损失
g_loss = sum(G_losses.values()).mean()

optimizer_G.zero_grad()
g_loss.backward()
optimizer_G.step()

"""更新鉴别器"""""
D_losses = {}

with torch.no_grad():
    z, mu, logvar = netE(real_image)
    fake_image = netG(input_semantics, z=z)
    fake_image = fake_image.detach()
    fake_image.requires_grad_()

fake_and_real_img = torch.cat([fake_image, real_image], dim=0) # [2,3,512,512]
fake_and_real_segmap = torch.cat([input_semantics, input_semantics], dim=0)

discriminator_out = netD(fake_and_real_img,fake_and_real_segmap)

pred = discriminator_out
feat_fake = []
pred_fake = []
feat_real = []
pred_real = []
# 把正图和假图的预测划分出来
for p in pred[0]:# 特征匹配
    feat_fake.append(p[:p.size(0) // 2])
    feat_real.append(p[p.size(0) // 2:])
for p in pred[1]:# 预测patch
    pred_fake.append(p[:p.size(0) // 2])
    pred_real.append(p[p.size(0) // 2:])

# 假图的GANLoss
GANloss_fake_D = 0
for pred_i in pred_fake:
    loss_tensor = HingeLoss(pred_i, target_is_real=False, for_discriminator=True)
    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
    GANloss_fake_D += new_loss
D_losses['D_Fake'] = GANloss_fake_D / len(pred_fake)

# 真图的GANLoss
GANloss_real_D = 0
for pred_i in pred_real:
    loss_tensor = HingeLoss(pred_i, target_is_real=True, for_discriminator=True)
    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
    GANloss_real_D += new_loss
D_losses['D_real'] = GANloss_real_D / len(pred_real)

d_loss = sum(D_losses.values()).mean()

optimizer_D.zero_grad()
d_loss.backward()
optimizer_D.step()
