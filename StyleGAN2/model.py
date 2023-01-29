import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import upfirdn2d, conv2d_gradfix
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


"""调制卷积"""
class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        demodulate=True,
        upsample=False,
        downsample=False,
    ):
        super().__init__()
        self.demodulate = demodulate
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.modulation = nn.Linear(512, in_channel)


        # 论文公式1中的si
        fan_in = in_channel * kernel_size ** 2  # 4608
        self.scale = 1 / math.sqrt(fan_in)  # 0.014

        # 对卷积核进行调制和解调
        self.weight = nn.Parameter(  # out_channel=512 in_channel=512 kernel_size=3 kernel_size=3
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        if upsample:
            factor = 2
            p = (4 - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur([1, 3, 3, 1], pad=(pad0, pad1), upsample_factor=factor)



    def forward(self, input, style):
        batch, in_channel, height, width = input.shape # 1 512 4 4
        # 对权重调制 style是latentcode
        style = self.modulation(style) # (1,512)
        style = style.view(batch, 1, in_channel, 1, 1)# style (1,512) -> (1,1,512,1,1)
        weight = self.scale * self.weight * style # 对应论文公式1 (1,512,512,3,3) 调制
        # 解调
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(  # (512,512,3,3)
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        # 上采样
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width) # [1,512,4,4]
            weight = weight.view( # [1,512,512,3,3]
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(# [512,512,3,3]
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d( # [1,512,9,9]反卷积
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)# [1,512,9,9]
            out = self.blur(out)# [1,512,8,8]
        # 保持原来的大小
        else:
            input = input.view(1, batch * in_channel, height, width)# [1,512,4,4]
            # 使用调制和解调后的卷积对const_input进行卷积
            out = conv2d_gradfix.conv2d(  # [1,512,4,4]
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)# [1,512,4,4]
        return out
"""加入噪声"""
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None): # image [1，512，4，4]
        if noise is None:
            batch, _, height, width = image.shape # 1 512 4 4
            noise = image.new_empty(batch, 1, height, width).normal_()# 返回一个大小为 size 的张量，其中填充了未初始化的数据。 [1,1,4,4]

        return image + self.weight * noise


"""风格卷积模块"""
class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel, # 512
        out_channel, # 512
        kernel_size, # 3
        upsample=False, # False
        demodulate=True, #True
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,  # 512
            out_channel,  # 512
            kernel_size,  # 3
            upsample=upsample,  # False
            demodulate=demodulate,  # True
        )

        self.noise = NoiseInjection()
        self.activate = nn.LeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style) # 论文中的调制解调和卷积
        out = self.noise(out, noise=noise) # 论文中的加噪音
        # out = out + self.bias
        out = self.activate(out)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, upsample=True):
        super().__init__()

        if upsample:
            self.upsample = nn.Upsample(scale_factor=2)

        self.conv = ModulatedConv2d(in_channel=in_channel, out_channel=3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):#input (1,512,4,4) style (1,512)
        out = self.conv(input, style) # (1,3,4,4)
        out = out + self.bias# self.bias (1,3,1,1) out (1,3,4,4)

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out

class Generator(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        """映射网络"""
        layers = [PixelNorm()]
        for i in range(8):
            layers.append(
                nn.Sequential(
                    nn.Linear(512, 512, bias=True),
                    nn.LeakyReLU(512, True),
                )
            )
        self.style = nn.Sequential(*layers)

        # 输入得可训练常量
        self.input = nn.Parameter(torch.randn(1, 512, 4, 4))


        self.conv1 = StyledConv(  # self.channels[4]=512  3 style_dim=512
            in_channel=512, out_channel=512, kernel_size=3,
        )

        self.to_rgb1 = ToRGB(512, upsample=False)

        self.log_size = int(math.log(521, 2)) # 9
        self.num_layers = (self.log_size - 2) * 2 + 1 # 15

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        """
                     register_buffer(name, tensor, persistent=True)
                     name(string) -缓冲区的名称。可以使用给定名称从此模块访问缓冲区
                     tensor(Tensor或者None) -要注册的缓冲区。
                     这通常用于注册不应被视为模型参数的缓冲区。
        """
        self.noises.register_buffer(f"noise_{0}", torch.randn(*[1,1,4,4]))
        self.noises.register_buffer(f"noise_{1}", torch.randn(*[1,1,8,8]))
        self.noises.register_buffer(f"noise_{2}", torch.randn(*[1,1,8,8]))
        self.noises.register_buffer(f"noise_{3}", torch.randn(*[1,1,16,16]))
        self.noises.register_buffer(f"noise_{4}", torch.randn(*[1,1,16,16]))
        self.noises.register_buffer(f"noise_{5}", torch.randn(*[1,1,32,32]))
        self.noises.register_buffer(f"noise_{6}", torch.randn(*[1,1,32,32]))
        self.noises.register_buffer(f"noise_{7}", torch.randn(*[1,1,64,64]))
        self.noises.register_buffer(f"noise_{8}", torch.randn(*[1,1,64,64]))
        self.noises.register_buffer(f"noise_{9}", torch.randn(*[1,1,128,128]))
        self.noises.register_buffer(f"noise_{10}", torch.randn(*[1,1,128,128]))
        self.noises.register_buffer(f"noise_{11}", torch.randn(*[1,1,256,256]))
        self.noises.register_buffer(f"noise_{12}", torch.randn(*[1,1,256,256]))
        self.noises.register_buffer(f"noise_{13}", torch.randn(*[1,1,512,512]))
        self.noises.register_buffer(f"noise_{14}", torch.randn(*[1,1,512,512]))

        """第二层"""
        # 有上采样得风格模块
        self.convs.append(
            StyledConv( # 有上采样
                in_channel=512,
                out_channel=512,
                kernel_size=3,
                upsample=True,
            )
        )
        # 无上采样得风格模块
        self.convs.append(
            StyledConv( # 无上采样
                512, 512, 3,
            )
        )
        self.to_rgbs.append(ToRGB(512)) # 转成图片
        """第三层"""
        # 有上采样得风格模块
        self.convs.append(
            StyledConv(
                in_channel=512,
                out_channel=512,
                kernel_size=3,
                upsample=True,
            )
        )
        # 无上采样得风格模块
        self.convs.append(
            StyledConv(
                512, 512, 3,
            )
        )
        self.to_rgbs.append(ToRGB(512))
        """第四层"""
        # 有上采样得风格模块
        self.convs.append(
            StyledConv(
                in_channel=512,
                out_channel=512,
                kernel_size=3,
                upsample=True,
            )
        )
        # 无上采样得风格模块
        self.convs.append(
            StyledConv(
                512, 512, 3,
            )
        )
        self.to_rgbs.append(ToRGB(512))

        """第5层"""
        # 有上采样得风格模块
        self.convs.append(
            StyledConv(
                in_channel=512,
                out_channel=512,
                kernel_size=3,
                upsample=True,
            )
        )
        # 无上采样得风格模块
        self.convs.append(
            StyledConv(
                512, 512, 3,
            )
        )
        self.to_rgbs.append(ToRGB(512))
        """第6层"""
        # 有上采样得风格模块
        self.convs.append(
            StyledConv(
                in_channel=512,
                out_channel=256,
                kernel_size=3,
                upsample=True,
            )
        )
        # 无上采样得风格模块
        self.convs.append(
            StyledConv(
                256, 256, 3,
            )
        )
        self.to_rgbs.append(ToRGB(256))
        """第7层"""
        # 有上采样得风格模块
        self.convs.append(
            StyledConv(
                in_channel=256,
                out_channel=128,
                kernel_size=3,
                upsample=True,
            )
        )
        # 无上采样得风格模块
        self.convs.append(
            StyledConv(
                128, 128, 3,
            )
        )
        self.to_rgbs.append(ToRGB(128))
        """第8层"""
        # 有上采样得风格模块
        self.convs.append(
            StyledConv(
                in_channel=128,
                out_channel=64,
                kernel_size=3,
                upsample=True,
            )
        )
        # 无上采样得风格模块
        self.convs.append(
            StyledConv(
                64, 64, 3,
            )
        )
        self.to_rgbs.append(ToRGB(64))

    def forward(self,styles):
        # latentcode 进行18层的线性映射
        style0 = self.style(styles[0])
        style1= self.style(styles[1])
        styles = [style0,style1] # [ (1,512) ,(1,512)]
        # noise
        noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]  #num_layers=15 [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
        # 将线性映射的2个w用一个随机数随机重组为一个[1,16,512] 的latentcode
        inject_index = random.randint(1, 16 - 1) # (1,15)中随机作为inject_index 5
        latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1) # [1,inject_index,512] eq. [1,5,512]
        latent2 = styles[1].unsqueeze(1).repeat(1, 16 - inject_index, 1) # [1,16 - inject_index,512] eq. [1,16-5==11,512]
        latent = torch.cat([latent, latent2], 1) # [1,16,512] 合并

        const_input = self.input.repeat(latent.shape[0], 1, 1, 1) # [1,512,4,4]
        # 第一层
        out0 = self.conv1(const_input, latent[:, 0], noise=noise[0])  # [1,512,4,4]
        skip0 = self.to_rgb1(out0, latent[:, 1])#out [1,512,4,4]  latent[:, 1] [1, 512] -> [1,3,4,4]
        # 第二层
        out1 = self.convs[0](out0, latent[:, 1], noise=noise[1]) # [1,512,8,8] 上采样
        out1 = self.convs[1](out1, latent[:, 2], noise=noise[2]) # [1,512,8,8]
        skip1 = self.to_rgbs[0](out1, latent[:, 3], skip0)# [1,3,8,8]
        # 第三层
        out2 = self.convs[2](out1, latent[:, 3], noise=noise[3])  # [1,512,16,16] 上采样
        out2 = self.convs[3](out2, latent[:, 4], noise=noise[4])  # [1,512,16,16]
        skip2 = self.to_rgbs[1](out2, latent[:, 5], skip1)  # [1,3,16,16]
        # 第四层
        out3 = self.convs[4](out2, latent[:, 5], noise=noise[5])  # [1,512,32,32] 上采样
        out3 = self.convs[5](out3, latent[:, 6], noise=noise[6])  # [1,512,32,32]
        skip3 = self.to_rgbs[2](out3, latent[:, 7], skip2)  # [1,3,32,32]
        # 第5层
        out4 = self.convs[6](out3, latent[:, 7], noise=noise[7])  # [1,512,64,64] 上采样
        out4 = self.convs[7](out4, latent[:, 8], noise=noise[8])  # [1,512,64,64]
        skip4 = self.to_rgbs[3](out4, latent[:, 9], skip3)  # [1,3,64,64]
        # 第6层
        out5 = self.convs[8](out4, latent[:, 9], noise=noise[9])  # [1,256,128,128] 上采样
        out5 = self.convs[9](out5, latent[:, 10], noise=noise[10])  # [1,256,128,128]
        skip5 = self.to_rgbs[4](out5, latent[:, 11], skip4)  # [1,3,128,128]
        # 第7层
        out6 = self.convs[10](out5, latent[:, 11], noise=noise[11])  # [1,128,256,256] 上采样
        out6 = self.convs[11](out6, latent[:, 12], noise=noise[12])  # [1,128,256,256]
        skip6 = self.to_rgbs[5](out6, latent[:, 13], skip5)  # [1,3,256,256]
        # 第8层
        out7 = self.convs[12](out6, latent[:, 13], noise=noise[13])  # [1,64,512,521] 上采样
        out7 = self.convs[13](out7, latent[:, 14], noise=noise[14])  #  [1,64,512,521]
        skip7 = self.to_rgbs[6](out7, latent[:, 15], skip6)  # [1,3,512,512]
        image = skip7

        return image ,latent


























"""
    鉴别器
"""

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        bias=True,
        activate=True,
    ):
        super().__init__()
        layers = []
        if downsample: # 下采样
            factor = 2
            p = (4 - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur([1, 3, 3, 1], pad=(pad0, pad1)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      padding=self.padding,
                      stride=stride,
                      bias=bias and not activate)
        )

        if activate:
            layers.append(nn.LeakyReLU(True))

        self.conv = nn.Sequential(*layers)
    def forward(self, input):

        return self.conv(input)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = ConvLayer(in_channel=in_channel,out_channel= in_channel, kernel_size=3)
        self.conv2 = ConvLayer(in_channel=in_channel, out_channel=out_channel, kernel_size=3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, kernel_size=1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)# [1,64,512,512]
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class Discriminator(nn.Module):
    def __init__(self,):
        super().__init__()
        self.stddev_group = 4
        self.stddev_feat = 1

        convs = [ConvLayer(in_channel=3, out_channel=64, kernel_size=1)]

        convs.append(ResBlock(64, 128))
        convs.append(ResBlock(128, 256))
        convs.append(ResBlock(256, 512))
        convs.append(ResBlock(512, 512))
        convs.append(ResBlock(512, 512))
        convs.append(ResBlock(512, 512))
        convs.append(ResBlock(512, 512))

        self.convs = nn.Sequential(*convs)
        self.final_conv = ConvLayer(512 + 1, 512, 3)

        self.final_linear = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, input):# [1,3,512,512]
        out = self.convs(input) # [1,512,4,4]
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        # MinibatchStdDev模块主要是计算一个sample的group里的标准差，扩展维度后concat到输入上。
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
