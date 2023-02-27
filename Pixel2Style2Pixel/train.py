from encoder import GradualStyleEncoder
from generator import Generator
import torch
from torch import nn
from criteria.lpips.lpips import LPIPS
from criteria import id_loss,w_norm
from ranger import Ranger
import torch.nn.functional as F

encoder = GradualStyleEncoder()
decoder = Generator()


print('Loading encoders weights from irse50!')
encoder_ckpt = torch.load('pretrained_models/model_ir_se50.pth')
encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
encoder.load_state_dict(encoder_ckpt, strict=False)
ckpt = torch.load('pretrained_models/model_ir_se50.pth')
decoder.load_state_dict(ckpt, strict=False)
#如果latent_avg不可用，则通过密集抽样来估计latent_avg
latent_in = torch.randn(100000, 512)
latent = decoder.style(latent_in).mean(0, keepdim=True) # [1,512]
latent_avg = torch.randn(100000, 512)[0].detach()

id_loss = id_loss.IDLoss().eval()
mse_loss = nn.MSELoss().eval()
lpips_loss = LPIPS(net_type='alex').eval()
w_norm_loss = w_norm.WNormLoss(start_from_latent_avg='')

params = list(encoder.parameters())
optimizer = Ranger(params,0.0001)

x = torch.randn(1,11,256,256)
y = torch.randn(1,3,256,256)

codes = encoder(x) # codes[1,16,512]
codes = codes + latent_avg.repeat(codes.shape[0], 1, 1)

y_hat, latent = decoder([codes]) # y_hat [1,3,512,512] latent [1,16,512]
face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
y_hat = face_pool(y_hat) # y_hat [1,3,256,256]

loss_dict = {}
loss = 0.0

# id Loss 只有人脸转正时候用
# loss_id, sim_improvement, id_logs = id_loss(y_hat, y, x)
# loss_dict['loss_id'] = float(loss_id)
# loss_dict['id_improve'] = float(sim_improvement)
# loss += loss_id * 1
# MSELoss
loss_l2 = F.mse_loss(y_hat, y)
loss_dict['loss_l2'] = float(loss_l2)
loss += loss_l2 * 0.001
# MSELoss 裁剪出人脸，再算一次
loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
loss_dict['loss_l2_crop'] = float(loss_l2_crop)
loss += loss_l2_crop * 0.01
# LPIPS Loss
loss_lpips = lpips_loss(y_hat, y)
loss_dict['loss_lpips'] = float(loss_lpips)
loss += loss_lpips * 0.08
# LPIPS Loss 裁剪出人脸，再算一次
loss_lpips_crop = lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
loss += loss_lpips_crop * 0.8

# w_norm_loss
loss_w_norm = w_norm_loss(latent, latent_avg)
loss_dict['loss_w_norm'] = float(loss_w_norm)
loss += loss_w_norm * loss_w_norm

loss_dict['loss'] = float(loss)
loss.backward()
optimizer.step()






