import torch
from model import ViT

v = ViT(
    image_size = 256,       # 图片大小3*256*256
    patch_size = 32,        # patch的大小32*32*3 注意image_size要被patch_size整除
    num_classes = 1000,     # 类别数
    dim = 1024,             # 过了Linear Projection of Flattened Patches之后的维度是多少
    depth = 6,              # 有多少个transformer blocks
    heads = 8,              # 多头注意力机制中的heads个数
    mlp_dim = 2048,         # Transformer Encoder中间FeedForward中MLP隐藏层
    dropout = 0.1,          # dropout概率
    emb_dropout = 0.1       #
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
print(preds.shape)