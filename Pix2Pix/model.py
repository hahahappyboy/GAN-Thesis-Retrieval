import torch
import os
import torch.nn as nn
from torch.nn import init




class UnetGenerator(nn.Module):
    """
        input_nc 3 输入通道数
        output_nc 3 输出通道数
        num_downs 8 下采样次数
    """
    def __init__(self):
        super(UnetGenerator, self).__init__()
        # downconv = nn.Conv2d(512, 512, kernel_size=(4,4),stride=(2,2), padding=(1,1), bias=False)
        # downrelu = nn.LeakyReLU(0.2, True)
        # downnorm = nn.BatchNorm2d(512)
        #
        # uprelu = nn.ReLU(True)
        # upnorm = nn.BatchNorm2d(512)
        # upconv = nn.ConvTranspose2d(512, 512,kernel_size=(4,4), stride=(2,2),padding=(1,1), bias=False)
        #
        # down = [downrelu, downconv, downnorm]
        # up = [uprelu, upconv, upnorm]
        # model_1 = down + up
        # """unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer"""
        # self.model_1 = nn.Sequential(*model_1)
        #
        # downconv = nn.Conv2d(512, 512, kernel_size=(4,4),stride=(2,2), padding=(1,1), bias=False)
        # downrelu = nn.LeakyReLU(0.2, True)
        # downnorm = nn.BatchNorm2d(512)
        #
        # upconv = nn.ConvTranspose2d(1024, 512,kernel_size=(4,4), stride=(2,2),padding=(1,1), bias=False)
        # uprelu = nn.ReLU(True)
        # upnorm = nn.BatchNorm2d(512)
        #
        # down = [downrelu, downconv, downnorm]
        # up = [uprelu, upconv, upnorm]
        # model_2 = down + [self.model_1] + up + [nn.Dropout(0.5)]
        #
        # self.model_2 = nn.Sequential(*model_2)

        self.down_0 = nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)

        self.down_1 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128)
        )

        self.down_2 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

        self.down_3 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512)
        )

        self.down_4 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512)
        )

        self.down_5 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512)
        )

        self.down_6 = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512)
        )

        self.inner = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512)
        )

        self.up_6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5, inplace=False),
        )

        self.up_5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5, inplace=False),
        )

        self.up_4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.5, inplace=False),
        )

        self.up_3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
        )

        self.up_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
        )

        self.up_1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
        )

        self.up_0 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self,x):
        d0 = self.down_0(x)
        d1 = self.down_1(d0)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)
        d5 = self.down_5(d4)
        d6 = self.down_6(d5)

        inner = self.inner(d6)

        up6 = torch.cat([inner,d6],1)
        up6 = self.up_6(up6)

        up5 = torch.cat([d5,up6],1)
        up5 = self.up_5(up5)

        up5 = torch.cat([d4,up5],1)
        up4 = self.up_4(up5)

        up4 = torch.cat([d3,up4],1)
        up3 = self.up_3(up4)

        up3 = torch.cat([d2,up3],1)
        up2 = self.up_2(up3)

        up2 = torch.cat([d1,up2],1)
        up1 = self.up_1(up2)

        up1 = torch.cat([d0,up1],1)
        up0 = self.up_0(up1)

        return up0



class NLayerDiscriminator(nn.Module):

    def __init__(self):
        super(NLayerDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),

            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),

            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),

            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),

            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

