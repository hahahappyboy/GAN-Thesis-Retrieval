import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5): # 64
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1)) # 1,64,1,1
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        # IN的均值和方差
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        # IN的标准化结构
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        # LN的均值和方法
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        # LN的标准化结果
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        # rho控制IN和LN的比例
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        # 再标准化
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out

class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1)) # 32
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d((1,1,1,1))
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class ResnetGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.DownBlock = nn.Sequential(
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn. Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.ResnetBlock_0 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
        )

        self.ResnetBlock_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
        )

        self.ResnetBlock_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
        )

        self.ResnetBlock_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
        )


        self.gap_fc = nn.Linear(in_features=64, out_features=1, bias=False)
        self.gmp_fc = nn.Linear(in_features=64, out_features=1, bias=False)
        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU(True)

        self.FC = nn.Sequential(
            nn.Linear(in_features=262144, out_features=64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=64, bias=False),
            nn.ReLU(inplace=True),
        )

        self.gamma = nn.Linear(in_features=64, out_features=64, bias=False)
        self.beta = nn.Linear(in_features=64, out_features=64, bias=False)

        self.UpBlock1_0 = ResnetAdaILNBlock(dim=64, use_bias=False)
        self.UpBlock1_1 = ResnetAdaILNBlock(dim=64, use_bias=False)
        self.UpBlock1_2 = ResnetAdaILNBlock(dim=64, use_bias=False)
        self.UpBlock1_3 = ResnetAdaILNBlock(dim=64, use_bias=False)

        self.UpBlock2_0 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='nearest'), # scale_factor:指定输出为输入的多少倍数； mode：可使用的上采样算法
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), bias=False),
            ILN(32),
            nn.ReLU(True)
        )

        self.UpBlock2_1 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), bias=False),
            ILN(16),
            nn.ReLU(True)
        )

        self.UpBlock2_2 = nn.Sequential(
            nn.ReflectionPad2d((3,3,3,3)),
            nn.Conv2d(16, 3, kernel_size=(7, 7), stride=(1, 1), bias=False),
            nn.Tanh()
        )




    def forward(self, input):
        d0 = self.DownBlock(input)

        r0 = d0 + self.ResnetBlock_0(d0)
        r1 = r0 + self.ResnetBlock_1(r0)
        r2 = r1 + self.ResnetBlock_2(r1)
        r3 = r2 + self.ResnetBlock_3(r2) # 1，64，64，64
        # 全局平局池化
        gap = torch.nn.functional.adaptive_avg_pool2d(r3, 1) # 特征图全局平均池化gap（1，64，1，1）
        gap_view = gap.view(r3.shape[0], -1) # 展平
        gap_logit = self.gap_fc(gap_view) # 全连接
        gap_weight = list(self.gap_fc.parameters())[0] # 得到权重
        gap = r3 * gap_weight.unsqueeze(2).unsqueeze(3) # 权重乘到特征图上 [1，64，64，64]
        # 全局最大池化
        gmp = torch.nn.functional.adaptive_max_pool2d(r3, 1)
        gmp_view = gmp.view(r3.shape[0], -1)
        gmp_logit = self.gmp_fc(gmp_view)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = r3 * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1) # 按维度拼接,辅助分类器的预测值 [1, 2]
        x = torch.cat([gap, gmp], 1) #[1，128，64，64]
        x = self.conv1x1(x) # [1，64，64，64]
        x = self.relu(x) # [1，64，64，64]

        heatmap = torch.sum(x, dim=1, keepdim=True) #得到热力图 [1,1,64,64]

        # 为了得到gamma, beta
        x_view = x.view(x.shape[0], -1) # 展平torch.Size([1, 262144])
        x_ = self.FC(x_view) # （1，64）
        gamma, beta = self.gamma(x_), self.beta(x_) # 都是[1，64] self.gamma和self.beta也是FC

        u0 = self.UpBlock1_0(x,gamma,beta)
        u1 = self.UpBlock1_1(u0,gamma,beta)
        u2 = self.UpBlock1_2(u1,gamma,beta)
        u3 = self.UpBlock1_3(u2,gamma,beta)

        u4 = self.UpBlock2_0(u3)
        u5 = self.UpBlock2_1(u4)
        u6 = self.UpBlock2_2(u5)

        return u6, cam_logit, heatmap


class Discriminator(nn.Module):
    def __init__(self, layer=7):
        super(Discriminator, self).__init__()
        if layer == 7:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

            )

            self.gap_fc = nn.utils.spectral_norm(nn.Linear(512, 1, bias=False))
            self.gmp_fc = nn.utils.spectral_norm(nn.Linear(512, 1, bias=False))
            self.conv1x1 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
            self.leaky_relu = nn.LeakyReLU(0.2, True)
            self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False))
        if layer == 5:
            self.model = nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(1, 1))),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

            self.gap_fc = nn.utils.spectral_norm(nn.Linear(128, 1, bias=False))
            self.gmp_fc = nn.utils.spectral_norm(nn.Linear(128, 1, bias=False))
            self.conv1x1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
            self.leaky_relu = nn.LeakyReLU(0.2, True)
            self.pad = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(128, 1, kernel_size=(4, 4), stride=(1, 1), bias=False))

    def forward(self, input):
        x = self.model(input) # 下采样 (1,512,7,7)
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1) # 全局平均池化 (1,512,1,1)
        gap_view = gap.view(x.shape[0], -1) # 展开
        gap_logit = self.gap_fc(gap_view) # （1，1）
        gap_weight = list(self.gap_fc.parameters())[0] # 得到权重
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3) # 将权重乘到特征图上面去

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_view = gmp.view(x.shape[0], -1) # 展开
        gmp_logit = self.gmp_fc(gmp_view)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1) # 辅助分类器判断的值进行通道合并
        x = torch.cat([gap, gmp], 1) # 乘了注意力的值进行通道合并
        x = self.conv1x1(x) # 改变通道数
        x = self.leaky_relu(x) # 激活 (1,512,7,7)

        heatmap = torch.sum(x, dim=1, keepdim=True) # (1,1,7,7)

        x = self.pad(x) # (1,512,9,9)
        out = self.conv(x) # (1,1,6,6)

        return out, cam_logit, heatmap

















