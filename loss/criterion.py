import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import list_spaces
from sklearn.metrics import hinge_loss

from common.functions import *
# import gemo.homographies as homo
from common import Image, NpArray
from common import NoGradientError


class MatchingCriterion(nn.Module):
    def __init__(
            self,
            data_name: str,
            match_type: str = 'dual_doftmax',
            dist_thresh: float = 5,
            weights: list = [1., 1000.],
            eps=1e-10
    ):
        super().__init__()

        self.data_name = data_name
        self.match_type = match_type
        self.dist_thresh = dist_thresh
        self.ws = weights
        self.eps = eps

    def set_weight(self, std, mask=None, regularizer=0.):
        inverse_std = 1. / torch.clamp(std + regularizer, min=self.eps)
        weight = inverse_std / torch.mean(inverse_std)
        weight = weight.detach()

        if mask is not None:
            weight = weight.masked_select(mask.bool())
            weight /= (torch.mean(weight) + self.eps)

        return weight

    def coarse_h(self, preds, targets):
        mkpts0, mkpts1 = preds['mkpts0'][:, 1:], preds['mkpts1']
        query_pts = mkpts0.int().detach().cpu().numpy()
        refer_pts = mkpts1.int().detach().cpu().numpy()
        h = None
        if query_pts.shape[0] < 4 or refer_pts.shape[0] < 4:
            h = torch.eye(3).cuda()
        else:
            h, mask = cv2.findHomography(query_pts, refer_pts, cv2.RANSAC, ransacReprojThreshold=3)
        if h is None:
            h = np.eye(3)
        h = torch.Tensor(h).cuda()

        h_gt = targets['h_gt']
        h_gt = h_gt.to(h.device)  # 将 h_gt 移动到与 h 相同的设备
        loss = torch.pairwise_distance(h.unsqueeze(-1), h_gt.unsqueeze(-1), 2).sum()

        return loss

    def coarse_loss(self, preds, targets):
        confidence_matrix = preds['cm_matrix']
        gt_matrix = targets['gt_matrix']
        loss = (-gt_matrix * torch.log(confidence_matrix + 1e-6)).sum() / preds['cm_matrix'].shape[0]
        return loss

    def compute_dist_within_images(
            self, mkpts0, mkpts1, image0: Image, image1: Image
    ):
        mkpts0_r = image1.project(image0.unproject(mkpts0.T)).T
        dist = torch.norm(mkpts1, mkpts0, dim=-1)

        return dist

    def fine_loss(self, preds, targets):

        samples0, samples1 = preds['samples0'], preds['samples1']
        mkpts0, mkpts1 = preds['mkpts0'], preds['mkpts1']

        gt_mask = targets['gt_matrix'] > 0
        gt_mask_v, gt_all_j_ids = gt_mask.max(dim=2)
        b_ids, i_ids = torch.where(gt_mask_v)
        j_ids = gt_all_j_ids[b_ids, i_ids]
        gt_matches = torch.stack([b_ids, i_ids, j_ids]).T

        gt_mkpts0, gt_mkpts1 = batch_get_mkpts(gt_matches, samples0, samples1)

        if gt_mkpts0.shape[0] == 0:
            return torch.tensor(1e-6, requires_grad=True).cuda()

        gt_mkpts11 = []
        mkpts11 = []
        # mkpts10 = []
        # gt_mkpts10 = []
        # exp = []

        for idx, mkp in enumerate(mkpts0):
            m_gt_mkpts1 = gt_mkpts1[torch.where((gt_mkpts0 == mkp).all(1))]
            if m_gt_mkpts1.shape[0] != 0 and m_gt_mkpts1[0][0] == mkp[0]:
                gt_mkpts11.append(m_gt_mkpts1.squeeze()[1:])
                mkpts11.append(mkpts1[idx])

        if len(gt_mkpts11) != 0:
            gt_mkpts11 = torch.stack(gt_mkpts11).cuda()
            mkpts11 = torch.stack(mkpts11).cuda()
        else:
            #     gt_mkpts11 = torch.Tensor(0, 2).cuda()
            #     mkpts11 = torch.Tensor(0, 2).cuda()
            #
            # if len(gt_mkpts11) == 0:
            # print("++++++++++++++++++++++++++")
            return torch.tensor(2., requires_grad=True).cuda()
        thr = targets['gt_matrix'].sum() if targets['gt_matrix'].sum() else 1600.
        # loss = torch.mean(torch.norm(gt_mkpts11 - mkpts11, p=2, dim=1)) / thr
        loss = torch.mean(torch.norm(gt_mkpts11 - mkpts11, p=2, dim=1)) / preds['cm_matrix'].shape[0]
        # print(loss)
        return loss

    def fine_loss_2(self, preds, targets):

        samples0, samples1 = preds['samples0'], preds['samples1']
        pred_mask = targets['gt_matrix'] * preds['cm_matrix']
        mask_v, all_j_ids = pred_mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        pred_matches = torch.stack([b_ids, i_ids, j_ids]).T

        mask_v, all_j_ids = targets['gt_matrix'].max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        gt_matches = torch.stack([b_ids, i_ids, j_ids]).T

        pred_mkpts0, pred_mkpts1 = batch_get_mkpts(pred_matches, samples0, samples1)
        gt_mkpts0, gt_mkpts1 = batch_get_mkpts(targets['gt_matrix'], samples0, samples1)
        loss = torch.mean(torch.norm(gt_mkpts1 - pred_mkpts1, p=2, dim=1))
        return loss

    def forward(self, preds, targets):
        h_loss = self.coarse_h(preds, targets)
        coarse_loss = self.coarse_loss(
            preds, targets)
        fine_loss = self.fine_loss(preds, targets)
        losses = h_loss + self.ws[0] * coarse_loss + self.ws[1] * fine_loss
        # losses =  self.ws[0] * coarse_loss + self.ws[1] * fine_loss
        loss_dict = {
            'losses': losses,
            'coarse_loss': coarse_loss,
            'h_loss': h_loss,
            'fine_loss': fine_loss
        }
        # print(h_loss,coarse_loss,fine_loss)
        # print(loss_dict)
        return loss_dict
