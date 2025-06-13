import torch.nn as nn
import torch.nn.functional as F
from models.position_encoding import PositionEncodingSine
from models.loftr_module.transformer import LocalFeatureTransformer
from models.class_attention import ClassTransformer
from einops.einops import rearrange, repeat
from models.mamba2 import Mamba2, Mamba2Config
class SIE_module(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.Group_transformer = ClassTransformer(config['class'])
        self.mamba2_block_8_41 = Mamba2(Mamba2Config(d_model=256, n_layer=4), device='cuda')
        self.mamba2_block_8_42 = Mamba2(Mamba2Config(d_model=256, n_layer=4), device='cuda')
        self.class_inter = LocalFeatureTransformer(config['class_trans'])
        self.interactive = LocalFeatureTransformer(config['interactive'])
        # Class Variable

    def forward(self, feat_c0_8, feat_c1_8 ,Sem_K_q, classnum_q, Sem_K_r, classnum_r):
        data = {}
        data['h0_c_8'] = feat_c0_8.size(2)
        data['h1_c_8'] = feat_c1_8.size(2)
        data['w0_c_8'] = feat_c0_8.size(3)
        data['w1_c_8'] = feat_c1_8.size(3)
        feat_c0_8_4 = self.pos_encoding(feat_c0_8)
        feat_c1_8_4 = self.pos_encoding(feat_c1_8)
        feat_c0_8_5 = rearrange(feat_c0_8_4, 'n c h w -> n (h w) c')
        feat_c1_8_5 = rearrange(feat_c1_8_4, 'n c h w -> n (h w) c')
        mask_c0 = mask_c1 = None
        feat_c0_8_b = self.Group_transformer(classnum_q, feat_c0_8_5, Sem_K_q, mask_c0)
        feat_c1_8_b = self.Group_transformer(classnum_r, feat_c1_8_5, Sem_K_r, mask_c1)
        feat_c0 = (feat_c0_8_b + feat_c0_8_5) / 2
        feat_c1 = (feat_c1_8_b + feat_c1_8_5) / 2
        feat_c_8_token_1, feat_c0_8_a = self.class_inter(Sem_K_q, feat_c0)
        feat_c_8_token_2, feat_c1_8_a = self.class_inter(Sem_K_r, feat_c1)
        # feat_c0_8_a1 = feat_c0_8_a
        # feat_c1_8_a2 = feat_c1_8_a
        # feat_c0_8_m4, h = self.mamba2_block_8_41(feat_c0_8_a1)
        # feat_c1_8_m4, h = self.mamba2_block_8_42(feat_c1_8_a2)
        # feat_c0_8_a = feat_c0_8_a * 0.8 + feat_c0_8_m4 * 0.2
        # feat_c1_8_a = feat_c1_8_a * 0.8 + feat_c1_8_m4 * 0.2
        # feat_c0_, feat_c1_ = self.interactive(feat_c0_8_a, feat_c1_8_a)
        feat_c0_8_final = feat_c0_8_a
        feat_c1_8_final = feat_c1_8_a

        return [feat_c0_8_final, feat_c1_8_final]
