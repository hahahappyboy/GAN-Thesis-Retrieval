import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
 
if not os.path.exists('./img_mlp'): # 报错中间结果
    os.mkdir('./img_mlp')
 
 
def to_img(x):# 将结果的-0.5~0.5变为0~1保存图片
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out
 
 
batch_size = 128
num_epoch = 100
z_dimension = 100
 
# 数据预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),# 图像数据转换成了张量，并且归一化到了[0,1]。
    transforms.Normalize([0.5],[0.5])#这一句的实际结果是将[0，1]的张量归一化到[-1, 1]上。前面的（0.5）均值， 后面(0.5)标准差，
])
# MNIST数据集
mnist = datasets.MNIST(
    root='./data/', train=True, transform=img_transform, download=True)
# 数据集加载器
dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True)
 
 
# 判别器 判别图片是不是来自MNIST数据集

 
 
# 生成器 生成伪造的MNIST数据集
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256), # 输入为100维的随机噪声
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
 
    def forward(self, x):
        x = self.gen(x)
        return x


mlp = Model() # 创建判别器
if torch.cuda.is_available():# 放入GPU
    mlp = mlp.cuda()

# Binary cross entropy loss and optimizer
criterion = nn.MSELoss() # BCELoss 因为可以当成是一个分类任务，如果后面不加Sigmod就用BCEWithLogitsLoss
mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0003) # 优化器

# 开始训练
for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):# img[128,1,28,28]
        num_img = img.size(0) # num_img=batchsize
        # =================train discriminator
        img = img.view(num_img, -1) # 把图片拉平
        real_img = img.cuda() # 装进cuda
        # 训练生成器
        # 计算生成图片的loss
        z = torch.randn(num_img, z_dimension).cuda()  #生成随机噪声 [128,100]
        fake_img = mlp(z) # 生成器伪造图像 [128,784]
        mlp_loss = criterion(fake_img, real_img) # 生成器
        # 更新生成器
        mlp_optimizer.zero_grad()
        mlp_loss.backward()
        mlp_optimizer.step()
 
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], loss: {:.6f}'.format(
                      epoch, num_epoch, mlp_loss.cpu().detach().mean()))
    if epoch == 0: # 保存图片
        real_images = to_img(real_img.cpu().detach())
        save_image(real_images, './img_mlp/real_images.png')
 
    fake_images = to_img(fake_img.cpu().detach())
    save_image(fake_images, './img_mlp/fake_images-{}.png'.format(epoch + 1))
# 保存模型
torch.save(mlp.state_dict(), './mlp.pth')
