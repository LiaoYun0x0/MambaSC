import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import copy
from models.position_encoding import PositionEncodingSine
from models.mamba2 import Mamba2, Mamba2Config
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _pad64(x):
    l, d = x.shape[-2:]
    pad_len = (64 - l % 64) % 64
    pad_dim = (64 - d % 64) % 64
    return pad_len, pad_dim


# 计算填充长度
def _pad8(x):
    l, d = x.shape[-2:]  # 获取最后两个维度
    pad_len = (8 - l % 8) % 8  # 计算在高度方向的填充长度
    pad_dim = (8 - d % 8) % 8  # 计算在宽度方向的填充长度
    return pad_len, pad_dim

# 填充函数
def pad(x, pad_len, pad_dim):
    # 如果计算出的填充长度为0，确保填充值为0
    if pad_len == 0 and pad_dim == 0:
        return x
    return F.pad(x, (0, pad_dim, 0, pad_len), mode='constant', value=0)

# 去填充函数
def unpad(x, pad_len, pad_dim):
    # 只有在填充存在的情况下才去除填充
    if pad_len > 0 or pad_dim > 0:
        # 处理高度方向填充
        if pad_len > 0:
            x = x[..., :-pad_len, :]
        # 处理宽度方向填充
        if pad_dim > 0:
            x = x[..., :, :-pad_dim]
    return x


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes, stride=1)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class FPN(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pos_encoding_2 = PositionEncodingSine(
            config['d_model_2'],
            temp_bug_fix=config['temp_bug_fix'],max_shape=(320, 320))
        self.pos_encoding_4 = PositionEncodingSine(
            config['d_model_4'],
            temp_bug_fix=config['temp_bug_fix'])
        self.pos_encoding_8 = PositionEncodingSine(
            config['d_model_8'],
            temp_bug_fix=config['temp_bug_fix'])
        self.mamba2_block_4_1 = Mamba2(Mamba2Config(d_model=192, n_layer=4), device='cuda')
        self.mamba2_block_8_1 = Mamba2(Mamba2Config(d_model=256, n_layer=4), device='cuda')
        self.mamba2_block_8_cut = Mamba2(Mamba2Config(d_model=64, n_layer=4, headdim = 16), device='cuda')
        self.mamba2_block_4_2 = Mamba2(Mamba2Config(d_model=192, n_layer=4), device='cuda')
        self.norm = nn.BatchNorm2d(config['d_model_4'])
        self.norm2 = nn.BatchNorm2d(config['d_model_8'], momentum=0.1, eps=1e-5)
        self.alpha = nn.Parameter(torch.tensor(0.2))  # 可学习的权重
        self.beta = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.gamma = nn.Parameter(torch.tensor(0.2))
        # 每种卷积的输出通道分配1
        in_channels = config['d_model_8']
        out_channels = config['d_model_8']
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 不同感受野的卷积层
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 融合模块：轻量化的 1x1 卷积
        self.light_fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_blocks(block, initial_dim, block_dims[0], stride=1)  # 1/2
        self.stage2 = self._make_blocks(block, block_dims[0], block_dims[1], stride=2)  # 1/4
        self.stage3 = self._make_blocks(block, block_dims[1], block_dims[2], stride=2)  # 1/8
        self.stage4 = self._make_blocks(block, block_dims[2], block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_blocks(self, block, dim1, dim, stride=1):
            block1 = block(dim1, dim, stride=stride)
            block2 = block(dim, dim, stride=1)
            blocks = (block1, block2)
            return nn.Sequential(*blocks)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.stage1(x0)  # 1/2
        x1_res = x1
        data = {}
        data['x1_h'] = x1.size(2)
        data['x1_w'] = x1.size(3)
        x1 = rearrange(self.pos_encoding_2(x1), 'n c h w -> n (h w) c')
        x1 = rearrange(x1, 'n (h w) c -> n c h w', h=data['x1_h'], w=data['x1_w'])
        x2 = self.stage2(x1)  # 1/4
        x2_res = x2
        data['x2_h'] = x2.size(2)
        data['x2_w'] = x2.size(3)
        x2 = rearrange(self.pos_encoding_4(x2), 'n c h w -> n (h w) c')
        x2_a =x2
        # 使用可学习的权重
        x2, h = self.mamba2_block_4_1(x2)
        x2 = rearrange(x2, 'n (h w) c -> n c h w', h=data['x2_h'], w=data['x2_w'])
        x2 = self.norm(x2)  # 添加归一化层
        x2 = rearrange(x2 ,'n c h w -> n (h w) c')
        x2 = self.alpha * x2 + (1 - self.alpha) * x2_a
        x2 = rearrange(x2, 'n (h w) c -> n c h w', h=data['x2_h'], w=data['x2_w'])
        x3 = self.stage3(x2)  # 1/8
        x3_res = x3
        data['x3_h'] = x3.size(2)
        data['x3_w'] = x3.size(3)
        x3 = rearrange(self.pos_encoding_8(x3), 'n c h w -> n (h w) c')
        # 前向传播
        x3_m, h = self.mamba2_block_8_1(x3)
        x3_m = rearrange(x3_m, 'n (h w) c -> n c h w', h=data['x3_h'], w=data['x3_w'])
        x3_m = self.norm2(x3_m)  # 添加归一化层
        x3_m = rearrange(x3_m, 'n c h w -> n (h w) c')
        x3 = torch.sigmoid(self.beta) * x3 + (1 - torch.sigmoid(self.beta)) * x3_m
        # x3 = (x3 + x3_m)/2
        x3 = rearrange(x3, 'n (h w) c -> n c h w', h=data['x3_h'], w=data['x3_w'])
        # x3_f = rearrange(x3_m2, 'n (h w) c -> n c h w', h=data['x3_h'], w=data['x3_w'])
        x4 = self.stage4(x3)  # 1/16
        x4_out = self.layer4_outconv(x4)  # 1/16
        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)  # 1/8
        x3_out = self.layer3_outconv(x3)  # 1/8
        x3_out = self.layer3_outconv2(x3_out + x4_out_2x)  # 1/8
        x3_out_1 = x3_out
        x3_out_2 = rearrange(x3_out, 'n c h w -> n (h w) c')
        # 不同感受野的卷积特征
        x3_1 = self.conv1x1(x3_out_1)  # 1x1 卷积
        x3_3 = self.conv3x3(x3_out_1)  # 3x3 卷积
        x3_5 = self.conv5x5(x3_out_1)  # 5x5 卷积
        x3_7 = self.conv7x7(x3_out_1)  # 7x7 卷积

        x3 = x3_1 * 0.25 + x3_3 * 0.25 + x3_5 * 0.25 + x3_7 * 0.25
        x3_b = x3 * 0.5 + x3_out * 0.5
        x3_b1 = rearrange(self.pos_encoding_8(x3_b), 'n c h w -> n (h w) c')
        # 首先将通道维度拆分成4个部分，每个部分有64个通道
        b, _, c = x3_b1.size()
        assert c == 256, "_x 的通道数必须为256"
        split_size = c // 4  # 64

        # 使用 torch.split 来拆分通道维度
        _x_splits = torch.split(x3_b1, split_size, dim=2)  # 在通道维度上拆分

        # 创建一个空列表来存储通过 self.mamba2_block_16_1 处理后的输出
        outputs = []

        # 分别将每个拆分后的张量通过 self.mamba2_block_16_1 处理
        for _x_split in _x_splits:
            output, h = self.mamba2_block_8_cut(_x_split)
            # 注意这里返回了两个值，我们只需要output进行拼接
            outputs.append(output)

        x3 = torch.cat(outputs, dim=2)  # 在通道维度上拼接
        x3 = rearrange(x3, 'b (h w) c -> b c h w', h=data['x3_h'], w=data['x3_w'])
        x3 = self.norm2(x3)
        x3 = self.conv1x1_2(x3)
        x3 = rearrange(x3, 'b c h w -> b (h w) c')

        x3 = x3_b1 * 0.5 + x3 * 0.5
        x3_c = rearrange(x3, 'b (h w) c -> b c h w', h=data['x3_h'], w=data['x3_w'])
        x3_out = x3_out * 0.5 + x3_b * 0.25 + x3_c * 0.25
        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)  # 1/4
        x2_out = self.layer2_outconv(x2_res)  # 1/4
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)  # 1/4
        x2_out = rearrange(self.pos_encoding_4(x2_out), 'n c h w -> n (h w) c')
        x2_out_a = x2_out
        x2_out, h = self.mamba2_block_4_2(x2_out)
        x2_out = rearrange(x2_out, 'n (h w) c -> n c h w', h=data['x2_h'], w=data['x2_w'])
        x2_out = self.norm(x2_out)  # 添加归一化层
        x2_out = rearrange(x2_out, 'n c h w -> n (h w) c')

        x2_out = self.gamma * x2_out + (1 - self.gamma) * x2_out_a

        x2_out = rearrange(x2_out, 'n (h w) c -> n c h w', h=data['x2_h'], w=data['x2_w'])
        x1_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)  # 1/2
        x1_out = self.layer1_outconv(x1_res)  # 1/2
        x1_out = self.layer1_outconv2(x1_out + x1_out_2x)  # 1/2
        return [x3_out, x1_out]