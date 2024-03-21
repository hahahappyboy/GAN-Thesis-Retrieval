import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from sync_batchnorm import SynchronizedBatchNorm2d

class SPADE(nn.Module):
    """
        参数：norm_nc 为SPADE的输出
            label_nc segmap通道数
    """
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=3,  padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3,  padding=1)


    def forward(self, segmap):## [1,1024,4,8] [1,36,256,512]
        inputmap = segmap

        actv = self.mlp_shared(inputmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return gamma, beta

class ACE(nn.Module):
    def __init__(self, fin, semantic_nc, use_rgb):
        super().__init__()
        self.style_length = 512
        self.Spade = SPADE(fin, semantic_nc)
        self.use_rgb = use_rgb

        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(fin), requires_grad=True)

        self.param_free_norm = SynchronizedBatchNorm2d(fin, affine=False)

        if self.use_rgb: # True
            self.fc_mu0 = nn.Linear(512, 512)
            self.fc_mu1 = nn.Linear(512, 512)
            self.fc_mu2 = nn.Linear(512, 512)
            self.fc_mu3 = nn.Linear(512, 512)
            self.fc_mu4 = nn.Linear(512, 512)
            self.fc_mu5 = nn.Linear(512, 512)
            self.fc_mu6 = nn.Linear(512, 512)
            self.fc_mu7 = nn.Linear(512, 512)
            self.fc_mu8 = nn.Linear(512, 512)
            self.fc_mu9 = nn.Linear(512, 512)
            self.fc_mu10 = nn.Linear(512, 512)
            self.fc_mu11 = nn.Linear(512, 512)
            self.fc_mu12 = nn.Linear(512, 512)
            self.fc_mu13 = nn.Linear(512, 512)
            self.fc_mu14 = nn.Linear(512, 512)
            self.fc_mu15 = nn.Linear(512, 512)
            self.fc_mu16 = nn.Linear(512, 512)
            self.fc_mu17 = nn.Linear(512, 512)
            self.fc_mu18 = nn.Linear(512, 512)

            self.conv_gamma = nn.Conv2d(512, fin, kernel_size=3, padding=1)
            self.conv_beta = nn.Conv2d(512, fin, kernel_size=3, padding=1)

    def forward(self, x, segmap, style_codes=None, obj_dic=None):  #x[1,1024,16,16] segmap[1,12,16,16] style_codes[1,12,512]
        # 给特征图加上StyleGAN的Noise
        added_noise = torch.randn(x.shape[0], x.shape[3], x.shape[2], 1)# [1, 16, 16 ,1]
        added_noise = added_noise * self.noise_var # [1, 16, 16 ,1024]
        added_noise = added_noise.transpose(1, 3) # [1,1024,16,16]
        normalized = self.param_free_norm(x + added_noise) # [1,1024,16,16] 归一化

        # SEAN归一化
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest') # [1,12,16,16]

        if self.use_rgb:
            [b_size, f_size, h_size, w_size] = normalized.shape  # b_size=1 f_size=1024 h_size=16 w_size=16
            middle_avg = torch.zeros((b_size, self.style_length, h_size, w_size),
                                     device=normalized.device)  # [1,512,16,16] 全为0
            for i in range(b_size):  # 1
                for j in range(segmap.shape[1]):  # [12]
                    component_mask_area = torch.sum(segmap.bool()[i, j])  # [104] j通道不为0的值有多少
                    if component_mask_area > 0:
                        # self.__getattr__('fc_mu' + str(j)  就是 fc_muj
                        # style_codes [1,12,512] style_codes[i][j]取出j通道的风格向量
                        # fc_mu j通道的风格向量解纠缠
                        middle_mu = self.__getattr__('fc_mu' + str(j))(style_codes[i][j]) # [512]
                        middle_mu = F.relu(middle_mu) # [512]
                        middle_mu = middle_mu.reshape(self.style_length, 1) # [512, 1]
                        # 将得到接纠缠风格向量复制104份，也就是论文中的图3广播
                        component_mu = middle_mu.expand(self.style_length, component_mask_area) # [512,104]
                        # middle_avg[i] [512,16,16]
                        # middle_avg[i]（全0）中将segmap.bool()[i, j]位置处的值（104个是有值的）,用component_mu从头开始填充。其他值都是0
                        middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu) # [1，512，16，16]
            # 得到SEAN的gamma和beta
            gamma_avg = self.conv_gamma(middle_avg) # [1，512，16，16]
            beta_avg = self.conv_beta(middle_avg) # [1，512，16，16]
            # SPADE的gamma和beta
            gamma_spade, beta_spade = self.Spade(segmap)  # [1，512，16，16] [1，512，16，16]
            # EAN的gamma和SPADE的gamma的融合系数
            gamma_alpha = F.sigmoid(self.blending_gamma)
            beta_alpha = F.sigmoid(self.blending_beta)
            # 融合
            gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
            beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
            # 归一化
            out = normalized * (1 + gamma_final) + beta_final
        else: # 不使用RGB
            gamma_spade, beta_spade = self.Spade(segmap)
            gamma_final = gamma_spade
            beta_final = beta_spade
            out = normalized * (1 + gamma_final) + beta_final
        return out
# Figure4 图b
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, use_rgb=True):
        super().__init__()
        self.use_rgb = use_rgb # True

        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.conv_s = spectral_norm(self.conv_s)

        self.ace_0 = ACE(fin,opt.semantic_nc, use_rgb=use_rgb)
        self.ace_1 = ACE(fmiddle,opt.semantic_nc, use_rgb=use_rgb)
        if self.learned_shortcut:
            self.ace_s = ACE(fin, opt.semantic_nc, use_rgb=use_rgb)

    def shortcut(self, x, seg, style_codes, obj_dic):
        if self.learned_shortcut:
            x_s = self.ace_s(x, seg, style_codes, obj_dic)
            x_s = self.conv_s(x_s)

        else:
            x_s = x
        return x_s

    def forward(self, x, seg, style_codes, obj_dic=None):# x [1,1024,16,16] seg [1,12,512,512] style_codes [1,12,512]
        x_s = self.shortcut(x, seg, style_codes, obj_dic)# [1,1024,16,16]

        dx = self.ace_0(x, seg, style_codes, obj_dic) # [1,1024,16,16]
        dx = self.conv_0(dx)

        dx = self.ace_1(dx, seg, style_codes, obj_dic)
        dx = self.conv_1(dx)

        out = x_s + dx
        return out

class Zencoder(torch.nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Zencoder, self).__init__()
        self.output_nc = output_nc # 512
        ### 卷冲层
        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(input_nc, 32, kernel_size=3, padding=0),
                 nn.InstanceNorm2d(32),
                 nn.LeakyReLU(0.2, False)]
        model += [nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, False)]
        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, False)]
        ### T卷积层
        model += [nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, False)]
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(256, output_nc, kernel_size=3, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)


    def forward(self, input, segmap):# input[1,3,512,512] segmap [1,12,512,512]
        # 移除图像中风格无关信息
        codes = self.model(input) # [1,512,256,256]
        #
        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest') # [1,12,256,256]

        b_size = codes.shape[0] # 1
        f_size = codes.shape[1] # 512
        s_size = segmap.shape[1] # 12
        # [1,12,512] 风格向量编码 全0
        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

        for i in range(b_size): # 1
            for j in range(s_size): # 12 遍历segmap每个语义图
                # segmap.bool()把segmap中0变为false，其他都为true segmap.bool()[i, j]=[256,256]
                # torch.sum(segmap.bool()[i, j]) 计算segmap中第i个batch图中j通道的语义值中非0的个数
                component_mask_area = torch.sum(segmap.bool()[i, j])  # 30100
                if component_mask_area > 0:
                    # codes[i] [512,256,256]
                    code = codes[i]
                    # segmap.bool()[i, j]=[256,256]
                    segmap_i_j = segmap.bool()[i, j]
                    # code.masked_select(segmap_i_j) 从风格无关特征图code中选出对应于segmap.bool()[i, j]中值为True的值
                    codes_component_feature = code.masked_select(segmap_i_j) # [18320896]
                    # 再reshape为[512,component_mask_area] [512, 35783] 512*35783 = 18320896
                    codes_component_feature = codes_component_feature.reshape(f_size,  component_mask_area)
                    # .mean(1) 按照维度1平均值 从而得到 [512]
                    codes_component_feature = codes_component_feature.mean(1)
                    codes_vector[i][j] = codes_component_feature
        return codes_vector

class SPADEGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt.semantic_nc = opt.semantic_nc
        self.sw, self.sh = 16, 16
        self.fc = nn.Conv2d(self.opt.semantic_nc, 1024, 3, padding=1)

        self.Zencoder = Zencoder(3, 512)

        self.fc = nn.Conv2d(self.opt.semantic_nc, 1024, 3, padding=1)

        self.head_0 = SPADEResnetBlock(1024, 1024, opt, use_rgb=True)
        self.G_middle_0 = SPADEResnetBlock(1024, 1024, opt, use_rgb=True)
        self.G_middle_1 = SPADEResnetBlock(1024, 1024, opt, use_rgb=True)

        self.up_0 = SPADEResnetBlock(1024, 512, opt, use_rgb=True)
        self.up_1 = SPADEResnetBlock(512, 256, opt, use_rgb=True)
        self.up_2 = SPADEResnetBlock(256, 128, opt, use_rgb=True)
        self.up_3 = SPADEResnetBlock(128, 64, opt, use_rgb=False)

        self.conv_img = nn.Conv2d(64, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, input, rgb_img, obj_dic=None):
        seg = input # [1,12,512,512]
        x = F.interpolate(seg, size=(self.sh, self.sw)) # [1,12,16,16]
        x = self.fc(x) # [1,1024,16,16]
        style_codes = self.Zencoder(input=rgb_img, segmap=seg) # 风格矩阵[1，12，512]

        x = self.head_0(x, seg, style_codes, obj_dic=obj_dic) # [1,1024,16,16]

        x = self.up(x) # [1,1024,16,16]

        x = self.G_middle_0(x, seg, style_codes, obj_dic=obj_dic)  # [1,1024,32,32]

        x = self.G_middle_1(x, seg, style_codes,  obj_dic=obj_dic) # [1,1024,32,32]

        x = self.up(x)# [1,1024,64,64]
        x = self.up_0(x, seg, style_codes, obj_dic=obj_dic)# [1,512,64,64]

        x = self.up(x)# [1,512,128,128]
        x = self.up_1(x, seg, style_codes, obj_dic=obj_dic) # [1,256,128,128]

        x = self.up(x)# [1,256,256,256]
        x = self.up_2(x, seg, style_codes, obj_dic=obj_dic)# [1,128,256,256]

        x = self.up(x)# [1,128,512,512]
        x = self.up_3(x, seg, style_codes,  obj_dic=obj_dic)# [1,64,512,512]

        x = self.conv_img(F.leaky_relu(x, 2e-1)) # [1,3,512,512]
        x = F.tanh(x)
        return x















