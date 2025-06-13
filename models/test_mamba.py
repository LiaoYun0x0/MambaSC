import torch
import torch.nn.functional as F
from mamba2_cross import Mamba2, Mamba2Config, Mamba2_cross
from mamba2 import Mamba2
from einops import rearrange


def _pad64(x):
    l, d = x.shape[-2:]
    pad_len = (64 - l % 64) % 64
    pad_dim = (64 - d % 64) % 64
    return pad_len, pad_dim


def _pad8(x):
    l, d = x.shape[-2:]
    pad_len = (8 - l % 8) % 8
    pad_dim = (8 - d % 8) % 8
    return pad_len, pad_dim


def pad(x, pad_len, pad_dim):
    return F.pad(x, (0, pad_dim, 0, pad_len), mode='constant', value=0)


def unpad(x, pad_len, pad_dim):
    return x[..., :-pad_len, :-pad_dim]


# 假设原始的CV2d数据的宽高为 H,W,通道数为C 均不可被64整除
C = 256
H = 40
W = 40
batch = 1

_C = (C + 63) // 64 * 64
_H = (H + 7) // 8 * 8  # 8 的 倍数
_W = (W + 7) // 8 * 8  # 8 的 倍数

conv_in = torch.nn.Conv2d(C, _C, 1, 1, 0).cuda()
mamba2_block = Mamba2(Mamba2Config(d_model=_C), device='cuda:0')
mamba2_cross_1 = Mamba2_cross(Mamba2Config(d_model=_C), device='cuda:0')
conv_out = torch.nn.Conv2d(_C, C, 1, 1, 0).cuda()
# x = torch.randn(batch, C, H, W).cuda()


x1 = torch.randn(batch, C, H, W).cuda()
x2 = torch.randn(batch, C, H, W).cuda()
x1 = rearrange(x1, 'b c h w -> b (h w) c')
x2 = rearrange(x2, 'b c h w -> b (h w) c')

# print("x.shape:", x.shape)
# x = conv_in(x)
# print("x.shape:", x.shape)
# pad_info = _pad8(x)
# _x = pad(x, *pad_info)
# print("_x.shape:", _x.shape, )
# _x = rearrange(_x, 'b c h w -> b (h w) c')
# print("_x.shape:", _x.shape, )
# y1, y2 = mamba2_cross_1(x1,x2)

_y, h = mamba2_block(x1)
_y, h1 = mamba2_block(_y, h)
# print("_y.shape:", _y.shape, )
# _y = rearrange(_y, 'b (h w) c -> b c h w', h=_H, w=_W)
# print("_y.shape:", _y.shape, )
# y = unpad(_y, *pad_info)
# print("y.shape:", y.shape)
# y = conv_out(y)
# print("y.shape:", y.shape)