import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.einops import rearrange, repeat

from common.functions import *
from models.position import PositionEmbedding2D, PositionEmbedding1D
from configs.default import get_cfg_defaults, lower_config
from models.loftr_module.transformer import LocalFeatureTransformer
from models.MFE_backbone import MFE_backbone
from models.MSC_module import MSC_module
from common.functions import batch_get_mkpts

# local transformer parameters
cfg = {}
cfg["lo_cfg"] = {}
lo_cfg = cfg["lo_cfg"]
lo_cfg["d_model"] = 128
lo_cfg["layer_names"] = ["self", "cross"] * 1
lo_cfg["nhead"] = 8
lo_cfg["attention"] = "linear"

config = lower_config(get_cfg_defaults())

def _transform_inv(img, mean, std):
    img = img * std + mean
    img = np.uint8(img * 255.0)
    img = img.transpose(1, 2, 0)
    return img

class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.mish = nn.Mish(inplace=True)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.mish(x)
        return x

class MatchingNet(nn.Module):
    def __init__(
            self,
            d_coarse_model: int = 256,
            d_fine_model: int = 128,
            matching_name: str = 'sinkhorn',
            match_threshold: float = 0.2,
            window: int = 5,
            border: int = 1,
            sinkhorn_iterations: int = 50,
    ):
        super().__init__()
        self.backbone = MFE_backbone(config['loftr'])
        self.MSC_module = MSC_module(config['loftr'])
        self.position1d = PositionEmbedding1D(d_fine_model, max_len=window ** 2)
        self.local_transformer = LocalFeatureTransformer(cfg["lo_cfg"])
        self.proj = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.merge = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        # self.conv2d = ConvBN(128, d_fine_model, 1, 1)
        self.regression1 = nn.Linear(d_coarse_model, d_fine_model, bias=True)
        self.regression2 = nn.Linear(3200, d_fine_model, bias=True)
        self.regression = nn.Linear(d_fine_model, 2, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.border = border
        self.window = window
        self.num_iter = sinkhorn_iterations
        self.match_threshold = match_threshold
        self.matching_name = matching_name
        self.step_coarse = 8
        self.step_fine = 2

        if matching_name == 'sinkhorn':
            bin_score = nn.Parameter(torch.tensor(1.))
            self.register_parameter("bin_score", bin_score)
        self.th = 0.1

    def fine_matching(self, x0, x1):
        x0, x1 = self.local_transformer(x0, x1)
        # x0, x1 = self.L2Normalize(x0, dim=0), self.L2Normalize(x1, dim=0)
        return x0, x1

    def _regression(self, feat):
        feat = self.regression1(feat)
        feat = feat.view(feat.shape[0], -1)
        feat = self.dropout(feat)
        feat = self.regression2(feat)
        feat = self.regression(feat)
        return feat

    def compute_confidence_matrix(self, query_lf, refer_lf, gt_matrix=None):
        _d = query_lf.shape[-1]
        query_lf = query_lf / _d
        refer_lf = refer_lf / _d
        similarity_matrix = torch.matmul(query_lf, refer_lf.transpose(1, 2)) / 0.1
        # sim_matrix = torch.einsum("nlc,nsc->nls", query_lf, refer_lf) / 0.1
        confidence_matrix = torch.softmax(similarity_matrix, 1) * torch.softmax(similarity_matrix, 2)
        return confidence_matrix

    def unfold_within_window(self, featmap):
        scale = self.step_coarse - self.step_fine
        # stride = int(math.pow(2, scale))
        stride = 4

        featmap_unfold = F.unfold(
            featmap,
            kernel_size=(self.window, self.window),
            stride=stride,
            padding=self.window // 2
        )

        featmap_unfold = rearrange(
            featmap_unfold,
            "B (C MM) L -> B L MM C",
            MM=self.window ** 2
        )
        return featmap_unfold

    def forward(self, samples0, samples1, gt_matrix):

        device = samples0.device

        # 1x1600x256, 1x256x160x160
        data = {}
        data['hw0_i'] = samples0.shape[2:]
        data['hw1_i'] = samples1.shape[2:]
        feat_c0_8, feat_c1_8, feat_f0_2, feat_f1_2 = self.backbone(samples0, samples1)
        feat_c0_8_final, feat_c1_8_final= self.MSC_module(feat_c0_8, feat_c1_8)
        mdesc0 = feat_c0_8_final
        mdesc1 = feat_c1_8_final
        cm_matrix = self.compute_confidence_matrix(mdesc0, mdesc1)
        fine_featmap0 = feat_f0_2
        fine_featmap1 = feat_f1_2
        cf_matrix = cm_matrix * (cm_matrix == cm_matrix.max(dim=2, keepdim=True)[0]) * (
                    cm_matrix == cm_matrix.max(dim=1, keepdim=True)[0])
        mask_v, all_j_ids = cf_matrix.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        matches = torch.stack([b_ids, i_ids, j_ids]).T

        if matches.shape[0] == 0:
            return {
                "cm_matrix": cm_matrix,
                "mdesc0": mdesc0,
                "mdesc1": mdesc1,
                "mkpts1": torch.Tensor(0, 2),
                'mkpts0': torch.Tensor(0, 2),
                'samples0': samples0,
                'samples1': samples1
            }

        mkpts0, mkpts1 = batch_get_mkpts(matches, samples0, samples1)

        fine_featmap0_unfold = self.unfold_within_window(fine_featmap0)  # 1x1600x25x256
        fine_featmap1_unfold = self.unfold_within_window(fine_featmap1)

        local_desc = torch.cat([
            fine_featmap0_unfold[matches[:, 0], matches[:, 1]],
            fine_featmap1_unfold[matches[:, 0], matches[:, 2]]
        ], dim=0)

        center_desc = repeat(torch.cat([
            mdesc0[matches[:, 0], matches[:, 1]],
            mdesc1[matches[:, 0], matches[:, 2]]
        ], dim=0),
            'N C -> N WW C',
            WW=self.window ** 2)

        center_desc = self.proj(center_desc)
        local_desc = torch.cat([local_desc, center_desc], dim=-1)
        local_desc = self.merge(local_desc)
        local_position = self.position1d(local_desc)
        local_desc = local_desc + local_position

        desc0, desc1 = torch.chunk(local_desc, 2, dim=0)  # [96,25,128]
        fdesc0, fdesc1 = self.fine_matching(desc0, desc1)

        c = self.window ** 2 // 2

        center_desc = repeat(fdesc0[:, c, :], 'N C->N WW C', WW=self.window ** 2)
        center_desc = torch.cat([center_desc, fdesc1], dim=-1)

        expected_coords = self._regression(center_desc)

        mkpts1 = mkpts1[:, 1:] + expected_coords

        return {
            'cm_matrix': cm_matrix,
            'matches': matches,
            'samples0': samples0,
            'samples1': samples1,
            'mkpts1': mkpts1,
            'mkpts0': mkpts0,
            'mdesc0': mdesc0,
            'mdesc1': mdesc1,
        }
