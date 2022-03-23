import torch
import torch.nn as nn
from DiffAugment_pytorch import DiffAugment


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,3,1,1)
        self.relu = nn.ReLU(True)
    def forward(self, input):
        x = self.conv(input)
        x = self.relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3,32,(2,2),stride=2)
        self.conv_2 = nn.Conv2d(32,64,(2,2),stride=2)
        self.conv_3 = nn.Conv2d(64,1,(2,2),stride=2)

    def forward(self, input):
        x = self.conv_1(input)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x
policy = 'color,translation,cutout'
D = Discriminator()
G = Generator()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterionGAN = nn.MSELoss() # GAN LOSS


real_img = torch.randn((1,3,512,512))
read_label = torch.ones((1, 1, 64, 64))
fake_label = torch.zeros((1,1,64, 64))


# 更新鉴别器
fake_img = G(real_img)
real_img_aug = DiffAugment(real_img,policy=policy)
fake_img_aug = DiffAugment(fake_img,policy=policy)
real_scores = D(real_img_aug)
fake_scores  = D(fake_img_aug.detach())

loss_real_D = criterionGAN(real_scores,read_label)
loss_fake_D = criterionGAN(fake_scores,fake_label)
loss_D = loss_fake_D+loss_real_D
optimizer_D.zero_grad()
loss_D.backward()
optimizer_D.step()

# 更新生成器
fake_img = G(real_img)
fake_img_aug = DiffAugment(fake_img,policy=policy)
fake_scores  = D(fake_img_aug)
loss_G = criterionGAN(fake_scores,read_label)

optimizer_G.zero_grad()
loss_G.backward()
optimizer_G.step()








