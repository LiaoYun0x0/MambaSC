import torch.nn as nn
import torch.nn.functional as F
import torch
from models.position_encoding import PositionEncodingSine
from models.loftr_module.transformer import LocalFeatureTransformer
from einops.einops import rearrange, repeat
from models.mamba2 import Mamba2, Mamba2Config
from torch.nn.utils import clip_grad_norm_  # For gradient clipping

class MSC_module(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        self.pos_encoding = PositionEncodingSine(
            config['msc_1']['d_model'],
            temp_bug_fix=config['msc_1']['temp_bug_fix'])

        # Only two Mamba2 blocks instead of four to reduce complexity
        self.mamba2_block_8_21 = Mamba2(Mamba2Config(d_model=256, n_layer=4), device='cuda')
        self.mamba2_block_8_22 = Mamba2(Mamba2Config(d_model=256, n_layer=4), device='cuda')

        self.SATT = LocalFeatureTransformer(config['msc_1'])
        self.SATT_CATT = LocalFeatureTransformer(config['msc_2'])

        # Class Variable

    def forward(self, x1, x2):
        data = {}
        data['h0_c_8'] = x1.size(2)
        data['h1_c_8'] = x2.size(2)
        data['w0_c_8'] = x1.size(3)
        data['w1_c_8'] = x2.size(3)

        # Positional Encoding
        feat_c0_8_1 = rearrange(self.pos_encoding(x1), 'n c h w -> n (h w) c')
        feat_c1_8_1 = rearrange(self.pos_encoding(x2), 'n c h w -> n (h w) c')
        # First Mamba2 block processing
        feat_c0_8_m, _ = self.mamba2_block_8_21(feat_c0_8_1)
        feat_c1_8_m, _ = self.mamba2_block_8_22(feat_c1_8_1)
        # Fusion of features (use residual connections)
        feat_c0_8_2 = (feat_c0_8_1 + feat_c0_8_m) / 2
        feat_c1_8_2 = (feat_c1_8_1 + feat_c1_8_m) / 2
        # Attention-based transformation (SATT and SATT_CATT)
        feat_c0_8_3, feat_c1_8_3 = self.SATT(feat_c0_8_2, feat_c1_8_2, None, None)
        # Second Mamba2 block processing
        feat_c0_8_m2, _ = self.mamba2_block_8_21(feat_c0_8_3)
        feat_c1_8_m2, _ = self.mamba2_block_8_22(feat_c1_8_3)
        # Fusion of second Mamba2 outputs
        feat_c0_8_4 = (feat_c0_8_3 + feat_c0_8_m2) / 2
        feat_c1_8_4 = (feat_c1_8_3 + feat_c1_8_m2) / 2
        feat_c0_8_5, feat_c1_8_5 = self.SATT_CATT(feat_c0_8_4, feat_c1_8_4, None, None)

        return [feat_c0_8_5, feat_c1_8_5]

    def apply_gradient_clipping(self, max_norm=1.0):
        """ Apply gradient clipping to avoid exploding gradients. """
        for param in self.parameters():
            if param.grad is not None:
                clip_grad_norm_(param, max_norm)

