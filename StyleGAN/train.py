from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import torch
import os
from model import StyleDiscriminator,StyleGenerator
from utils import R1Penalty


G = StyleGenerator()
D = StyleDiscriminator()

optim_D = optim.Adam(D.parameters(), lr=0.00001, betas=(0.5, 0.999))
optim_G = optim.Adam(G.parameters(), lr=0.00001, betas=(0.5, 0.999))
scheduler_D = optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
scheduler_G = optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)

fix_z = torch.randn([1, 512])
softplus = nn.Softplus()
Loss_D_list = [0.0]
Loss_G_list = [0.0]

loss_D_list = []
loss_G_list = []
i = 0

real_img = torch.randn((1,3,1024,1024))
real_logit = D(real_img)
# print(real_logit.shape)
z = torch.randn((1,512))
fake_img = G(z)
fake_logit = D(fake_img.detach())

d_loss = softplus(fake_logit).mean() # 伪造样本loss
d_loss = d_loss + softplus(-real_logit).mean() # 真实样本loss

r1_penalty = R1Penalty(real_img.detach(), D) # R1 正则化
d_loss = d_loss + r1_penalty * (10.0 * 0.5)

loss_D_list.append(d_loss.item())

D.zero_grad()
d_loss.backward()
optim_D.step()

if i % 5 == 0:
    fake_logit = D(fake_img)
    g_loss = softplus(-fake_logit).mean() # G的WAN-GP loss
    loss_G_list.append(g_loss.item())

    G.zero_grad()
    g_loss.backward()
    optim_G.step()


