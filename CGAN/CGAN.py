import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("result", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size) #(1,32,32)

cuda = True if torch.cuda.is_available() else False # True

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.n_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),# 32*32
            nn.Tanh()
        )


    def get_one_hot(self, labels):
        x = torch.zeros((labels.shape[0], opt.n_classes)).type(FloatTensor)
        for i in range(labels.shape[0]):
            x[i,labels[i]] = FloatTensor([1.])
        return x

    def forward(self, noise, labels):
        one_hot = self.get_one_hot(labels)
        gen_input = torch.cat((one_hot, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()


        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def get_one_hot(self, labels):
        x = torch.zeros((labels.shape[0], opt.n_classes)).type(FloatTensor)
        for i in range(labels.shape[0]):
            x[i, labels[i]] = FloatTensor([1.])
        return x

    def forward(self, img, labels):
        input = torch.cat((img.view(img.size(0), -1), self.get_one_hot(labels)), -1)
        validity = self.model(input)
        return validity

# 损失函数
adversarial_loss = torch.nn.MSELoss()

# 初始化模型
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# 数据集
os.makedirs("data/", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((opt.img_size,opt.img_size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,

)

# 优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 数据类型
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, epoch):
    # 生成随机噪声
    z = FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))
    # 生成标签
    labels = np.array([num for _ in range(n_row) for num in range(n_row)]) #10个 0到9
    labels = LongTensor(labels)
    gen_imgs = generator(z, labels) # [100,1,32,32]
    save_image(gen_imgs.data, "result/%d.png" % epoch, nrow=n_row, normalize=True)

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # 真实图像的标签
        valid = FloatTensor(batch_size, 1).fill_(1.0)
        # 伪造图像的标签
        fake = FloatTensor(batch_size, 1).fill_(0.0)

        # 真实图像
        real_imgs = imgs.type(FloatTensor)
        # 真实图像是数字几
        labels = labels.type(LongTensor)

        # 生成随机噪声
        z = FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))
        # 生成随机标签
        gen_labels = LongTensor(np.random.randint(0, opt.n_classes, batch_size))

        # ---------------------
        #  训练鉴别器
        # ---------------------

        # 生成伪造图片
        gen_imgs = generator(z, gen_labels)
        # 真实图片loss
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # 伪造图片loss
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # 鉴别器总loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # 训练生成器
        # ---------------------

        # 计算loss
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

    sample_image(n_row=10, epoch=epoch)


