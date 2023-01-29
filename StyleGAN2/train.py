# -*- coding: utf-8 -*-

import random
import torch
from model import Generator,Discriminator
from torch.nn import functional as F
from op import conv2d_gradfix
from torch import  autograd, optim
import math

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()
def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty
def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss
def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


batch = 1
path_batch_shrink = 2 # 路径长度正则化的批处理大小减小因子(减少内存消耗)
r1 = 10 # r1正则化比例权重
d_reg_every = 16 # 使用r1正则化的间隔
g_reg_every = 4 # 应用感知路径长度的间隔
path_regularize = 2 #感知路径长度权重
channel_multiplier = 2 # channel multiplier factor for the model.
loss_dict = {}
# size = 512 # 图像大小  style_dim = 512 # 隐向量大小  n_mlp = 8 # mlp层数

generator = Generator()
discriminator = Discriminator()

g_optim = optim.Adam(
    generator.parameters(),
    lr=0.02,
    betas=(0, 0.99),
    )
d_optim = optim.Adam(
    discriminator.parameters(),
    lr=0.02,
    betas=(0, 0.99),
)

r1_loss = torch.tensor(0.0)
path_loss = torch.tensor(0.0)
path_lengths = torch.tensor(0.0)
mean_path_length = 0

for idx in range(1,1000):
    i = idx
    """训练鉴别器"""
    real_img = torch.randn((1,3,512,512))
    requires_grad(generator, False)
    requires_grad(discriminator, True)
    noises = torch.randn(2, 1, 512).unbind(0) # (2,1,512) ->(1,512)  (1,512)
    fake_img, _ = generator(noises)
    fake_pred = discriminator(fake_img)
    real_pred = discriminator(real_img)
    d_loss = d_logistic_loss(real_pred, fake_pred)
    loss_dict["d"] = d_loss
    loss_dict["real_score"] = real_pred.mean()
    loss_dict["fake_score"] = fake_pred.mean()

    discriminator.zero_grad()
    d_loss.backward()
    d_optim.step()
    ## r1正则话 每d_reg_every个step一次
    d_regularize = i % d_reg_every == 0
    if d_regularize:
        real_img.requires_grad = True
        real_pred = discriminator(real_img)
        r1_loss = d_r1_loss(real_pred, real_img)
        discriminator.zero_grad()
        (r1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0]).backward()
        d_optim.step()
    loss_dict["r1"] = r1_loss
    """训练生成器"""
    requires_grad(generator, True)
    requires_grad(discriminator, False)
    noise = torch.randn(2, 1, 512).unbind(0)
    fake_img, _ = generator(noise)
    fake_pred = discriminator(fake_img)
    g_loss = g_nonsaturating_loss(fake_pred)
    loss_dict["g"] = g_loss

    generator.zero_grad()
    g_loss.backward()
    g_optim.step()

    # 感知路径长度 每g_reg_every个step执行一次
    g_regularize = i % g_reg_every == 0
    if g_regularize:
        path_batch_size = max(1, batch // path_batch_shrink)
        noise = torch.randn(2, 1, 512).unbind(0)
        fake_img, latents = generator(noise)

        path_loss, mean_path_length, path_lengths = g_path_regularize(
            fake_img, latents, mean_path_length
        )
        generator.zero_grad()
        weighted_path_loss = path_regularize * g_reg_every * path_loss

        if path_batch_shrink:
            weighted_path_loss += 0 * fake_img[0, 0, 0, 0]
        weighted_path_loss.backward()
        g_optim.step()
        mean_path_length_avg = mean_path_length.item()
    print('epoch:'+str(i))



