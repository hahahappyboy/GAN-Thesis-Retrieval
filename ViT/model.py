import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) # 先norm再执行传进来的function

class FeedForward(nn.Module):
    def  __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) #1024->1536 将输入变为q，k，v

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):# x [1,65,1024]
        b, n, _, h = *x.shape, self.heads # b=1 n=65 _=1024 h=8
        # 得到q，k，v
        qkv = self.to_qkv(x) # 将x转为q，k，v [1，65，1536] 65个patch每个patch有一个qkv
        qkv = qkv.chunk(3, dim = -1) # 拆分成三个[1，65，512]的tuple分别是q，k，v，即65个patch每个patch有一个q，k，v
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) # [1，65，512]->[1,8,65，64]将q，k，v Reshape为矩阵用于后面的点积，并且体现出16个heads

        # Q,K点积 b h i d代表q ，b h j d代表k ， * self.scale是除以根号下dk
        q_dots_v = einsum('b h i d, b h j d -> b h i j', q, k) # [1，8，65，65]
        dots = q_dots_v * self.scale
        attn = self.attend(dots) # nn.Softmax [1，8，65，65]

        out = einsum('b h i j, b h j d -> b h i d', attn, v) # 乘以V 得到8个heads的结构[1，8，65，64]
        out = rearrange(out, 'b h n d -> b n (h d)') # 把16个heads的结果concat到一起[1，65，512]
        return self.to_out(out) # W0操作 把16个heads的信息进行交互[1，65，1024]->[1，65，1024]

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x # 跳连
            x = ff(x) + x # 跳连
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # 这里进行的是维度的拆分
            # [1 3 (8 32) (8 32)] -> [1 (8 8) (32 32 3)]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # 图片切分重排 x.transpose(0,2,3,1)->rearrange(x,'b c h w -> b h w c')
            nn.Linear(patch_dim, dim), # Liner Projection of Flattened Patches 32*32*3->1024
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # [1,65,1024]
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # [1,1,1024]的patch0
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape # [1,64,1024] b为batchsize

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) # 把patch0[1,1024]给repeat了b次得到[1,1,1024]
        x = torch.cat((cls_tokens, x), dim=1) # [1,65,1024] 加入patch0

        x += self.pos_embedding # 加入位置信息
        x = self.dropout(x) # 因为使用的是MLP所以用来dropout

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
