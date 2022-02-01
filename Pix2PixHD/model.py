import torch
import torch.nn as nn
from torchvision import models
import functools
from torch.autograd import Variable
import numpy as np

class GlobalGenerator(nn.Module):

    def __init__(self):
        super(GlobalGenerator, self).__init__()

        self.down_0 = nn.Sequential(
            nn.ReflectionPad2d(3),# 3 为扩充长度
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),padding= (0,0)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.down_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.down_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.down_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.down_4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.res_0 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024)
        )

        self.res_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024)
        )

        self.res_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024)
        )

        self.res_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024)
        )

        self.res_4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024)
        )

        self.res_5 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024)
        )

        self.res_6 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024)
        )

        self.res_7 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024)
        )

        self.res_8 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1)),
            nn.InstanceNorm2d(1024)
        )


        self.up_4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1)),
            nn.Tanh()
        )


    def forward(self, input):
        d0 = self.down_0(input)
        d1 = self.down_1(d0)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)

        r0 = d4 + self.res_0(d4)
        r1 = r0 + self.res_1(r0)
        r2 = r0 + self.res_2(r1)
        r3 = r0 + self.res_3(r2)
        r4 = r0 + self.res_4(r3)
        r5 = r0 + self.res_5(r4)
        r6 = r0 + self.res_6(r5)
        r7 = r0 + self.res_7(r6)
        r8 = r0 + self.res_8(r7)

        u4 = self.up_4(r8)
        u3 = self.up_3(u4)
        u2 = self.up_2(u3)
        u1 = self.up_1(u2)
        u0 = self.up_0(u1)

        return u0



class MultiscaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        netD = NLayerDiscriminator()

        self.scale0_layer0 = netD.model_0
        self.scale0_layer1 = netD.model_1
        self.scale0_layer2 = netD.model_2
        self.scale0_layer3 = netD.model_3
        self.scale0_layer4 = netD.model_4

        self.scale1_layer0 = netD.model_0
        self.scale1_layer1 = netD.model_1
        self.scale1_layer2 = netD.model_2
        self.scale1_layer3 = netD.model_3
        self.scale1_layer4 = netD.model_4

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        input_downsampled = input
        model = [self.scale0_layer0,self.scale0_layer1,self.scale0_layer2,self.scale0_layer3,self.scale0_layer4]

        result_0 = [input_downsampled]
        for i in range(len(model)):
            result_0.append(model[i](result_0[-1]))
        result.append(result_0[1:])

        input_downsampled = self.downsample(input_downsampled) # 变成 [1，6，128，128]
        model = [self.scale1_layer0,self.scale1_layer1,self.scale1_layer2,self.scale1_layer3,self.scale1_layer4]
        result_1 = [input_downsampled]
        for i in range(len(model)):
            result_1.append(model[i](result_1[-1]))
        result.append(result_1[1:])
        return result


class NLayerDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # sequence = [
        #     [
        #         nn.Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     ],
        #     [
        #         nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
        #         nn.InstanceNorm2d(128),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     ],
        #     [
        #         nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
        #         nn.InstanceNorm2d(256),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     ],
        #     [
        #         nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2)),
        #         nn.InstanceNorm2d(512),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     ],
        #     [
        #         nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
        #     ]
        # ]

        self.model_0 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.model_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.model_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.model_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.model_4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
        )


    def forward(self, input):
        m0 = self.model_0(input)
        m1 = self.model_1(m0)
        m2 = self.model_2(m1)
        m3 = self.model_3(m2)
        m4 = self.model_4(m3)
        res = [m0, m1, m2, m3, m4]
        return res



class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        # 预训练的VGG的模型 一共有0~36层 但是这里只取了0~29层
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            #   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (1): ReLU(inplace=True)
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            #   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (3): ReLU(inplace=True)
            #   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            #   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (6): ReLU(inplace=True)
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            #   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (8): ReLU(inplace=True)
            #   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            #   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (11): ReLU(inplace=True)
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            #   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (13): ReLU(inplace=True)
            #   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (15): ReLU(inplace=True)
            #   (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (17): ReLU(inplace=True)
            #   (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            #   (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (20): ReLU(inplace=True)
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            #   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (22): ReLU(inplace=True)
            #   (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (24): ReLU(inplace=True)
            #   (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (26): ReLU(inplace=True)
            #   (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            #   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            #   (29): ReLU(inplace=True)
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad: # 模型不进行求导
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out








