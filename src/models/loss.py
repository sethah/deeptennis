from typing import Dict

import torch
import torch.nn as nn

from allennlp.data import Vocabulary
from allennlp.models import Model

class SSDLoss(nn.Module):

    def __init__(self, class_criterion, reg_criterion):
        super(SSDLoss, self).__init__()
        self.class_criterion = class_criterion
        self.reg_criterion = reg_criterion

    def forward(self, preds, targ):
        box_idxs, reg_targs = targ
        b = reg_targs.shape[0]
        c = reg_targs.shape[1]

        class_preds, reg_preds = preds
        class_preds = class_preds.view(b, -1)
        reg_preds = reg_preds.view(b, c, -1)
        nbox = reg_preds.shape[2]

        class_targs = torch.zeros(b, nbox, dtype=torch.float32,
                                  device=reg_preds.device)
        class_targs[torch.arange(b), box_idxs.squeeze()] = 1.

        # mining for negatives!
        keep = class_targs > 0
        conf_preds = torch.sigmoid(class_preds)
        conf_preds[keep] = 0.  # only find negatives
        _, topthree = conf_preds.topk(3, 1)  # TODO: make ratio configurable
        keep.scatter_(1, topthree, 1.)
        class_loss = self.class_criterion(class_preds[keep], class_targs[keep])

        reg_preds = reg_preds[torch.arange(b), :, box_idxs.squeeze()]  # shape (b, 5)
        reg_loss = self.reg_criterion(reg_preds, reg_targs)
        return class_loss + reg_loss


class CourtScoreLoss(Model):

    def __init__(self, court_criterion, score_criterion, court_weight=1., score_weight=1.):
        super(CourtScoreLoss, self).__init__(Vocabulary())
        self.court_criterion = court_criterion
        self.score_criterion = score_criterion
        self.court_weight = court_weight
        self.score_weight = score_weight

    def forward(self,
                preds: Dict[str, torch.Tensor],
                targ: Dict[str, torch.Tensor]):
        court_preds = preds['court']
        # court_preds = preds[0] # (b, 4, 56, 56)
        court_targs = targ['court'].to(court_preds.device)
        score_coords = targ['score_offset']
        score_idx = targ['score_class'].long()
        b = court_preds.shape[0]
        device = court_preds.device

        # hard negative mining
        keep_pos = (court_targs > 0.1).view(b, -1)
        _, loss_idx = court_preds.view(b, -1).sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = keep_pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3 * num_pos, max=keep_pos.size(1) - 1)
        keep_neg = idx_rank < num_neg.expand_as(idx_rank)

        targets_weighted = court_targs.view(b, -1)[(keep_pos + keep_neg).gt(0)]
        preds_weighted = court_preds.view(b, -1)[(keep_pos + keep_neg).gt(0)]

        # score_loss = self.score_criterion(preds[1], targ[1].to(device))
        score_loss = self.score_criterion((preds['score_class'], preds['score_offset']),
                                           (score_idx.to(device), score_coords.to(device)))
        court_loss = self.court_criterion(preds_weighted, targets_weighted)
        return score_loss * self.score_weight + court_loss * self.court_weight

class CourtScoreLoss2(Model):

    def __init__(self, court_criterion, score_criterion, court_weight=1., score_weight=1.):
        super(CourtScoreLoss, self).__init__()
        self.court_criterion = court_criterion
        self.score_criterion = score_criterion
        self.court_weight = court_weight
        self.score_weight = score_weight

    def forward(self, preds, targ):
        court_preds = preds[0] # (b, 4, 56, 56)
        court_targs = targ[0].to(court_preds.device)
        score_coords, score_idx = targ[1]
        b = court_preds.shape[0]
        device = preds[0].device

        # hard negative mining
        keep_pos = (court_targs > 0.1).view(b, -1)
        _, loss_idx = court_preds.view(b, -1).sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = keep_pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3 * num_pos, max=keep_pos.size(1) - 1)
        keep_neg = idx_rank < num_neg.expand_as(idx_rank)

        targets_weighted = court_targs.view(b, -1)[(keep_pos + keep_neg).gt(0)]
        preds_weighted = court_preds.view(b, -1)[(keep_pos + keep_neg).gt(0)]

        # score_loss = self.score_criterion(preds[1], targ[1].to(device))
        score_loss = self.score_criterion(preds[1], (score_idx.to(device), score_coords.to(device)))
        court_loss = self.court_criterion(preds_weighted, targets_weighted)
        return score_loss * self.score_weight + court_loss * self.court_weight