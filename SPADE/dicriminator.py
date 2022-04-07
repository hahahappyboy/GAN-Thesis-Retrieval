import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import torch
from sync_batchnorm import SynchronizedBatchNorm2d
from enocoder import BaseNetwork,ConvEncoder


class NLayerDiscriminator(BaseNetwork):

    def __init__(self,opt):
        super().__init__()

        input_nc = opt.semantic_nc + 3


        model0 = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, False),
        )

        model1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2)),
            nn.InstanceNorm2d(128, affine=False),
            nn.LeakyReLU(0.2, False),
        )

        model2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2)),
            nn.InstanceNorm2d(256, affine=False),
            nn.LeakyReLU(0.2, False),
        )

        model3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=2)),
            nn.InstanceNorm2d(512, affine=False),
            nn.LeakyReLU(0.2, False),
        )

        model4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
        )

        self.add_module('model0',model0)
        self.add_module('model1',model1)
        self.add_module('model2',model2)
        self.add_module('model3',model3)
        self.add_module('model4',model4)


    def forward(self, input):# [2,39,256,512]
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        return results[1:] # [2,64,129,257] [2,128,65,129] [2,256,33,65] [2,512,33,65] [2,1,35,67]
class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self,opt):
        super().__init__()
        subnetD = NLayerDiscriminator(opt)
        self.add_module('discriminator_0',subnetD)
        subnetD = NLayerDiscriminator(opt)
        self.add_module('discriminator_1', subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    def forward(self, input):
        result = []
        for name, D in self.named_children():
            out = D(input) # [2,64,129,257] [2,128,65,129] [2,256,33,65] [2,512,33,65] [2,1,35,67]
            result.append(out)
            input = self.downsample(input)
        return result