
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import torch
from sync_batchnorm import SynchronizedBatchNorm2d
from generator import BaseNetwork


class ConvEncoder(BaseNetwork):
    """编码器"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            nn.InstanceNorm2d(64),

            nn.LeakyReLU(0.2,False),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            nn.InstanceNorm2d(128),

            nn.LeakyReLU(0.2, False),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            nn.InstanceNorm2d(256),

            nn.LeakyReLU(0.2, False),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            nn.InstanceNorm2d(512),

            nn.LeakyReLU(0.2, False),
            nn.utils.spectral_norm(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            nn.InstanceNorm2d(512),

            nn.LeakyReLU(0.2, False),
            nn.utils.spectral_norm(nn. Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            nn.InstanceNorm2d(512),

            nn.LeakyReLU(0.2, False),
        )

        self.fc_mu = nn.Linear(8192, 256)
        self.fc_var = nn.Linear(8192, 256)

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std) + mu
        return z, mu, logvar