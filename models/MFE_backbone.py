import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from models.backbone import build_backbone

class MFE_backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        self.backbone = build_backbone(config)
        # Class Variable

    def forward(self, x1, x2):
        (feat_c0_8, feat_f0_2), (feat_c1_8, feat_f1_2) = self.backbone(
            x1), self.backbone(x2)
        return [feat_c0_8, feat_c1_8, feat_f0_2, feat_f1_2]
