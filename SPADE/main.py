
import torch
from enocoder import ConvEncoder
from generator import SPADEGenerator
from dicriminator import MultiscaleDiscriminator
from loss import KLDLoss,VGGLoss,HingeLoss
import argparse

parser = argparse.ArgumentParser(description="SPADE")
#  有35类别 实例映射为1类
parser.add_argument('--semantic_nc', type=int, default=36,
                    help='语义图的通道数')
opt = parser.parse_args()


input_semantics = torch.randn((1,opt.semantic_nc,256,512))
real_image = torch.randn((1,3,256,512))

netE = ConvEncoder()
netG = SPADEGenerator(opt)
netD = MultiscaleDiscriminator(opt)


FeatLoss = torch.nn.L1Loss()
VGGLoss  = VGGLoss()
KLDLoss = KLDLoss()

G_params = list(netG.parameters())
G_params += list(netE.parameters())
D_params = list(netD.parameters())
optimizer_G = torch.optim.Adam(G_params, lr=0.0001, betas=(0.0, 0.9))
optimizer_D = torch.optim.Adam(D_params, lr=0.0001, betas=(0.0, 0.9))


z, mu, logvar = netE(real_image)

KLD_loss = KLDLoss(mu, logvar)*0.05

fake_image = netG(input_semantics, z=z)


"""更新生成器"""""
fake_concat = torch.cat([input_semantics, fake_image], dim=1)  # [1,39,256,512]
real_concat = torch.cat([input_semantics, real_image], dim=1) # [1,39,256,512]
# 正和假图像同时给D
fake_and_real = torch.cat([fake_concat, real_concat], dim=0)  # [2,39,256,512]
# list0 [2,64,129,257] [2,128,65,129] [2,256,33,65] [2,512,33,65] [2,1,35,67]
# list1 [2,64,65,129] [2,256,33,65] [2,512,17,33] [2,512,18,34] [2,1,19,35]
pred = netD(fake_and_real)

pred_fake = []
pred_real = []
# 把正图和假图的预测划分出来
for p in pred:
    pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
    pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])

G_losses = {}
G_losses['KLD'] = KLD_loss

##### GAN损失
GANloss_G = 0
for pred_i in pred_fake:
    if isinstance(pred_i, list):
        pred_i = pred_i[-1] # 取最后一个loss
    loss_tensor = HingeLoss(pred_i, target_is_real=True, for_discriminator=False)
    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
    GANloss_G += new_loss
G_losses['GAN'] = GANloss_G / len(pred_fake)

##### 特征匹配损失

GAN_Feat_loss = 0
for i in range(2):  # 两个生成器
    # 最后一个是GANLoss计算，所以排除掉
    num_intermediate_outputs = len(pred_fake[i]) - 1
    for j in range(num_intermediate_outputs):  # for each layer output
        unweighted_loss = FeatLoss(
            pred_fake[i][j], pred_real[i][j].detach())
        GAN_Feat_loss += unweighted_loss * 10 / 2
G_losses['GAN_Feat'] = GAN_Feat_loss

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

fake_concat = torch.cat([input_semantics, fake_image], dim=1) # [1,39,256,512]
real_concat = torch.cat([input_semantics, real_image], dim=1)

fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
discriminator_out = netD(fake_and_real)
pred = discriminator_out
pred_fake = []
pred_real = []
# 把正图和假图的预测划分出来
for p in pred:
    pred_fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
    pred_real.append([tensor[tensor.size(0) // 2:] for tensor in p])

# 假图的GANLoss
GANloss_fake_D = 0
for pred_i in pred_fake:
    if isinstance(pred_i, list):
        pred_i = pred_i[-1] # 取最后一个loss
    loss_tensor = HingeLoss(pred_i, target_is_real=False, for_discriminator=True)
    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
    GANloss_fake_D += new_loss
D_losses['D_Fake'] = GANloss_fake_D / len(pred_fake)

# 真图的GANLoss
GANloss_real_D = 0
for pred_i in pred_real:
    if isinstance(pred_i, list):
        pred_i = pred_i[-1] # 取最后一个loss
    loss_tensor = HingeLoss(pred_i, target_is_real=True, for_discriminator=True)
    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
    GANloss_real_D += new_loss
D_losses['D_real'] = GANloss_real_D / len(pred_real)

d_loss = sum(D_losses.values()).mean()

optimizer_D.zero_grad()
d_loss.backward()
optimizer_D.step()
