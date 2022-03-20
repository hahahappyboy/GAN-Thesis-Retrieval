import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1
class FC(nn.Module):
    def __init__(self,
                 in_channels,# 512
                 out_channels,# 512
                 gain=2**(0.5),# 1.4142135623730951
                 use_wscale=False,# True
                 lrmul=1.0,# 0.01
                 bias=True):# True
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale: # yes
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul # 0.00625
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std) # 初始胡权重
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):# [1,512]
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)# [1,512]
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True) # (1,512)
        return out

class G_mapping(nn.Module):
    """映射网络"""
    def __init__(self):
        super().__init__()
        self.func = nn.Sequential(
            nn.Linear(512,512,bias=True),
            nn.LeakyReLU(512,True),

            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(512, True),

            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(512, True),

            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(512, True),

            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(512, True),

            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(512, True),

            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(512, True),

            nn.Linear(512, 512, bias=True),
            nn.LeakyReLU(512, True),

        )
        self.pixel_norm = PixelNorm()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pixel_norm(x)
        out = self.func(x)
        return out

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise

class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Sequential(
                        nn.Linear(latent_size,channels * 2,bias=True),
                        nn.LeakyReLU(True)
                                    )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,1)
                m.bias.data.zero_()

    def forward(self, x, latent):
        style = self.linear(latent) # 论文中的A w 转 style [1,1024]  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x

class AddNoise(nn.Module):
    def __init__(self, channels):  # 特征图维度 噪声维度
        super().__init__()
        self.noise = ApplyNoise(channels)  # 可学习的变化 也就是论文中的B和+号
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x,noise):
        x = self.noise(x, noise)
        x = self.act(x)

        return x


class AdaIN(nn.Module):
    def __init__(self, channels, dlatent_size):# 特征图维度 噪声维度
        super().__init__()

        # self.noise = ApplyNoise(channels) # 可学习的变化 也就是论文中的B
        # self.act = nn.LeakyReLU(negative_slope=0.2)
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_mod = ApplyStyle(dlatent_size, channels)

    def forward(self, x,dlatents_in_slice=None):
        # x = self.noise(x, noise)
        # x = self.act(x)
        x = self.instance_norm(x)
        x = self.style_mod(x, dlatents_in_slice)
        return x


class G_synthesis(nn.Module):
    """主干网络"""
    def __init__(self):
        super().__init__()
        # 噪声初始化
        self.noise_inputs = []
        self.noise_inputs.append(torch.randn((1,1,4,4)))
        self.noise_inputs.append(torch.randn((1,1,4,4)))
        self.noise_inputs.append(torch.randn((1,1,8,8)))
        self.noise_inputs.append(torch.randn((1,1,8,8)))
        self.noise_inputs.append(torch.randn((1,1,16,16)))
        self.noise_inputs.append(torch.randn((1,1,16,16)))
        self.noise_inputs.append(torch.randn((1,1,32,32)))
        self.noise_inputs.append(torch.randn((1,1,32,32)))
        self.noise_inputs.append(torch.randn((1,1,64,64)))
        self.noise_inputs.append(torch.randn((1,1,64,64)))
        self.noise_inputs.append(torch.randn((1,1,128,128)))
        self.noise_inputs.append(torch.randn((1,1,128,128)))
        self.noise_inputs.append(torch.randn((1,1,256,256)))
        self.noise_inputs.append(torch.randn((1,1,256,256)))
        self.noise_inputs.append(torch.randn((1,1,512,512)))
        self.noise_inputs.append(torch.randn((1,1,512,512)))
        self.noise_inputs.append(torch.randn((1,1,1024,1024)))
        self.noise_inputs.append(torch.randn((1,1,1024,1024)))

        self.channel_shrinkage = nn.Conv2d(32,8,kernel_size=(3,3),bias=True,padding=1)
        self.torgb =  nn.Conv2d(8,3,kernel_size=(3,3),bias=True,padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()

        # 固定的输入
        self.const_input = nn.Parameter(torch.ones(1, 512, 4, 4))
        self.bias = nn.Parameter(torch.ones(512))

        self.addNoise_01 = AddNoise(512) # 噪声的维度
        self.adaIN_01 = AdaIN(512,512)
        self.conv_0 = nn.Conv2d(512,512,kernel_size=(3,3),bias=True,padding=1)
        self.addNoise_02 = AddNoise(512) # 噪声的维度
        self.adaIN_02 = AdaIN(512,512)

        # 公共块
        # 4 x 4 -> 8 x 8
        self.up_1 =  nn.Upsample(scale_factor=2)
        self.addNoise_11 = AddNoise(512)
        self.adaIN_11 = AdaIN(512,512)
        self.conv_1 = nn.Conv2d(512,512,kernel_size=(3,3),bias=True,padding=1)
        self.addNoise_12 = AddNoise(512)
        self.adaIN_12 = AdaIN(512, 512)

        # 8 x 8 -> 16 x 16
        self.up_2 = nn.Upsample(scale_factor=2)
        self.addNoise_21 = AddNoise(512)
        self.adaIN_21 = AdaIN(512, 512)
        self.conv_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), bias=True, padding=1)
        self.addNoise_22 = AddNoise(512)
        self.adaIN_22 = AdaIN(512, 512)

        # self.GBlock2 = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     AdaIN(512, 512),
        #     nn.Conv2d(512, 512, kernel_size=(3, 3), bias=True, padding=1),
        #     AdaIN(512, 512)
        # )
        # 16 x 16 -> 32 x 32

        self.up_3 = nn.Upsample(scale_factor=2)
        self.addNoise_31 = AddNoise(512)
        self.adaIN_31 = AdaIN(512, 512)
        self.conv_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), bias=True, padding=1)
        self.addNoise_32 = AddNoise(512)
        self.adaIN_32 = AdaIN(512, 512)

        # 32 x 32 -> 64 x 64
        self.up_4 = nn.Upsample(scale_factor=2)
        self.addNoise_41 = AddNoise(512)
        self.adaIN_41 = AdaIN(512, 512)
        self.conv_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), bias=True, padding=1)
        self.addNoise_42 = AddNoise(512)
        self.adaIN_42 = AdaIN(512, 512)

        # 64 x 64 -> 128 x 128
        self.up_5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.addNoise_51 = AddNoise(256)
        self.adaIN_51 = AdaIN(256, 512)# 特征图维度 噪声维度
        self.conv_5 = nn.Conv2d(256, 256, kernel_size=(3, 3), bias=True, padding=1)
        self.addNoise_52 = AddNoise(256)
        self.adaIN_52 = AdaIN(256, 512)


        # 128 x 128 -> 256 x 256
        self.up_6 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.addNoise_61 = AddNoise(128)
        self.adaIN_61 = AdaIN(128, 512)  # 特征图维度 噪声维度
        self.conv_6 = nn.Conv2d(128, 128, kernel_size=(3, 3), bias=True, padding=1)
        self.addNoise_62 = AddNoise(128)
        self.adaIN_62 = AdaIN(128, 512)



        # 256 x 256 -> 512 x 512
        self.up_7 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.addNoise_71 = AddNoise(64)
        self.adaIN_71 = AdaIN(64, 512)  # 特征图维度 噪声维度
        self.conv_7 = nn.Conv2d(64, 64, kernel_size=(3, 3), bias=True, padding=1)
        self.addNoise_72 = AddNoise(64)
        self.adaIN_72 = AdaIN(64, 512)



        # 512 x 512 -> 1024 x 1024
        self.up_8 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.addNoise_81 = AddNoise(32)
        self.adaIN_81 = AdaIN(32, 512)  # 特征图维度 噪声维度
        self.conv_8 = nn.Conv2d(32, 32, kernel_size=(3, 3), bias=True, padding=1)
        self.addNoise_82 = AddNoise(32)
        self.adaIN_82 = AdaIN(32, 512)

        self.channel_shrinkage = nn.Conv2d(32,8 , kernel_size=(3, 3), bias=True, padding=1)
        self.torgb = nn.Conv2d(8,3 , kernel_size=(1, 1), bias=True, padding=0)

    def forward(self, dlatent):# z [1,18,512]
        x = self.const_input # 可学习常量Noise [1,512,4,4]
        x = x + self.bias.view(1, -1, 1, 1) # 加上偏执

        x = self.addNoise_01(x, self.noise_inputs[0])
        x = self.adaIN_01(x, dlatent[:, 0])

        x = self.conv_0(x)

        x = self.addNoise_02(x, self.noise_inputs[1])
        x = self.adaIN_02(x,  dlatent[:, 1])

        x = self.up_1(x)

        x = self.addNoise_11(x, self.noise_inputs[2])
        x = self.adaIN_11(x, dlatent[:, 1])

        x = self.conv_1(x)

        x = self.addNoise_12(x, self.noise_inputs[3])
        x = self.adaIN_12(x, dlatent[:, 3])

        x = self.up_2(x)

        x = self.addNoise_21(x, self.noise_inputs[4])
        x = self.adaIN_21(x, dlatent[:, 4])

        x = self.conv_2(x)

        x = self.addNoise_22(x, self.noise_inputs[5])
        x = self.adaIN_22(x, dlatent[:,5])

        x = self.up_3(x)

        x = self.addNoise_31(x, self.noise_inputs[6])
        x = self.adaIN_31(x, dlatent[:, 6])

        x = self.conv_3(x)

        x = self.addNoise_32(x, self.noise_inputs[7])
        x = self.adaIN_32(x, dlatent[:, 7])

        x = self.up_4(x)

        x = self.addNoise_41(x, self.noise_inputs[8])
        x = self.adaIN_41(x, dlatent[:, 8])

        x = self.conv_4(x)

        x = self.addNoise_42(x, self.noise_inputs[9])
        x = self.adaIN_42(x, dlatent[:, 9])

        x = self.up_5(x)

        x = self.addNoise_51(x, self.noise_inputs[10])
        x = self.adaIN_51(x, dlatent[:, 10])

        x = self.conv_5(x)

        x = self.addNoise_52(x, self.noise_inputs[11])
        x = self.adaIN_52(x, dlatent[:, 11])

        x = self.up_6(x)

        x = self.addNoise_61(x, self.noise_inputs[12])
        x = self.adaIN_61(x, dlatent[:, 12])

        x = self.conv_6(x)

        x = self.addNoise_62(x, self.noise_inputs[13])
        x = self.adaIN_62(x, dlatent[:, 13])

        x = self.up_7(x)

        x = self.addNoise_71(x, self.noise_inputs[14])
        x = self.adaIN_71(x, dlatent[:, 14])

        x = self.conv_7(x)

        x = self.addNoise_72(x, self.noise_inputs[15])
        x = self.adaIN_72(x, dlatent[:, 15])

        x = self.up_8(x)

        x = self.addNoise_81(x, self.noise_inputs[16])
        x = self.adaIN_81(x, dlatent[:, 16])

        x = self.conv_8(x)

        x = self.addNoise_82(x, self.noise_inputs[17])
        x = self.adaIN_82(x, dlatent[:, 17])

        x = self.channel_shrinkage(x)
        images_out = self.torgb(x)

        return images_out

























class StyleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapping = G_mapping()
        self.synthesis = G_synthesis()

    def forward(self, latents1):# z
        dlatents1 = self.mapping(latents1) #
        dlatents1 = dlatents1.unsqueeze(1)# [1,1,512]
        dlatents1 = dlatents1.expand(-1, 18, -1)# 分成18份 [1,18,512]

        coefs = [[[0.7],  [0.7],  [0.7],  [0.7],  [0.7],  [0.7],  [0.7],  [0.7],  [1. ],  [1. ],  [1. ],  [1. ],  [1. ],  [1. ],  [1. ],  [1. ],  [1. ],  [1. ]]]
        coefs = np.array(coefs)
        dlatents1 = dlatents1 * torch.Tensor(coefs)
        img = self.synthesis(dlatents1)

        return img

class StyleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fromrgb = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1),
                                    nn.LeakyReLU(0.2, True))


        self.dense0 = nn.Sequential(nn.Linear(in_features=8192, out_features=512, bias=True), nn.LeakyReLU(0.2, True))
        self.dense1 =  nn.Sequential(nn.Linear(512, 1,bias=True), nn.LeakyReLU(0.2, True))
        self.sigmoid = nn.Sigmoid()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, input):
        x = self.model(input)
        x = x.view(x.size(0), -1)
        x = self.dense0(x)
        x = self.dense1(x)
        x = self.sigmoid(x)
        return x
