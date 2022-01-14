import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,nz,ngf):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf*4, kernel_size=(4, 4),stride=(2, 2) ,padding=(1, 1),bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(4, 4), stride=(2, 2) , padding=(1, 1), bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, 3, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1), bias=True),
            nn.Tanh()
        )

    def forward(self,input):
        output = self.model(input)
        return output

class Generator_UseIN(nn.Module):
    def __init__(self,nz,ngf):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf*4, kernel_size=(4, 4),stride=(2, 2) ,padding=(1, 1),bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(4, 4), stride=(2, 2) , padding=(1, 1), bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, 3, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1), bias=True),
            nn.Tanh()
        )

    def forward(self,input):
        output = self.model(input)
        return output

class Generator_UseLeakyReLU(nn.Module):
    def __init__(self,nz,ngf):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf*4, kernel_size=(4, 4),stride=(2, 2) ,padding=(1, 1),bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=(4, 4), stride=(2, 2) , padding=(1, 1), bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf, 3, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1), bias=True),
            nn.Tanh()
        )

    def forward(self,input):
        output = self.model(input)
        return output

class Discriminator(nn.Module):
    def __init__(self,ndf):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=(5, 5), stride=(3,3), padding=(1,1), bias=True),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid(),
        )

    def forward(self,input):
        output = self.model(input)
        return output

class Discriminator_UseReLU(nn.Module):
    def __init__(self,ndf):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=(5, 5), stride=(3,3), padding=(1,1), bias=True),
            nn.BatchNorm2d(ndf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid(),
        )

    def forward(self,input):
        output = self.model(input)
        return output

class Discriminator_UseIN(nn.Module):
    def __init__(self,ndf):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=(5, 5), stride=(3,3), padding=(1,1), bias=True),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.model(input)
        return output