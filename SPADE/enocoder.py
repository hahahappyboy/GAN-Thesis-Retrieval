
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import torch
from sync_batchnorm import SynchronizedBatchNorm2d

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

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