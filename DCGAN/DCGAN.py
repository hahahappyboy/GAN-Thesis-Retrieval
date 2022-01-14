import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
from model import Generator,Discriminator,Generator_UseLeakyReLU,Discriminator_UseReLU,Generator_UseIN,Discriminator_UseIN
import os
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64,help='批次大小')
parser.add_argument('--imageSize', type=int, default=96,help='图像大小')
parser.add_argument('--nz', type=int, default=100, help='输入向量大小')
parser.add_argument('--ngf', type=int, default=64, help='生成器中间通道数')
parser.add_argument('--ndf', type=int, default=64, help='鉴别器中间通道数')
parser.add_argument('--epoch', type=int, default=25, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam b1')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam b2')
parser.add_argument('--data_path', default='data/', help='数据集')
parser.add_argument('--result_path', default='result/', help='结果')
opt = parser.parse_args()



# 创建文件夹
os.makedirs(opt.result_path,exist_ok=True)
# 能不能使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator(opt.nz, opt.ngf).to(device)
netD = Discriminator(opt.ndf).to(device)

#图像读入与预处理
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Scale(opt.imageSize),# 裁剪成96*96
    torchvision.transforms.ToTensor(),# 转为tenosr 归一化到0-1
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]) # 归一化到-1-1

# 加载数据集
dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize, #64
    shuffle=True,
    drop_last=True,
)

# 定义优化器和损失
criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


real_label = torch.FloatTensor(opt.batchSize, 1,1,1).fill_(1.0).to(device)
fake_label = torch.FloatTensor(opt.batchSize, 1,1,1).fill_(0.0).to(device)

for epoch in range(1, opt.epoch + 1):
    for i, (real_img,_) in enumerate(dataloader):# [64,3,96,96]
        real_img = real_img.to(device)
        noise = torch.randn(opt.batchSize, opt.nz, 1, 1).to(device) # 生成100维的随机噪声 [64,100,1,1]
        fake_img = netG(noise)
        #########
        # 更新鉴别器
        #########

        # 真实图片loss
        validity_real = netD(real_img)
        d_real_loss = criterion(validity_real, real_label)
        # 伪造图片loss
        validity_fake = netD(fake_img.detach())
        d_fake_loss = criterion(validity_fake, fake_label)

        d_loss = (d_real_loss+d_fake_loss)*0.5

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        #########
        # 更新生成器
        #########
        validity_fake = netD(fake_img)
        g_loss = criterion(validity_fake, real_label)

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), d_loss.item(), g_loss.item()))

    save_image(fake_img.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.result_path, epoch),
                      normalize=True)

torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.result_path, epoch))








