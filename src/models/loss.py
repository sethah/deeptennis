from typing import Dict

import torch
import torch.nn as nn

from allennlp.data import Vocabulary
from allennlp.models import Model


class AnchorBoxLoss(nn.Module):

    def forward(self,
                class_pred: torch.Tensor,
                offset_pred: torch.Tensor,
                class_label: torch.Tensor,
                offset_label: torch.Tensor):
        raise NotImplementedError


class SSDLoss(AnchorBoxLoss):

    def __init__(self, class_criterion, reg_criterion):
        super(SSDLoss, self).__init__()
        self.class_criterion = class_criterion
        self.reg_criterion = reg_criterion

    def forward(self,
                class_pred: torch.Tensor,
                offset_pred: torch.Tensor,
                class_label: torch.Tensor,
                offset_label: torch.Tensor):
        """
        :param class_pred: (b x h x w) classification score for every object
        class in each grid cell
        :param offset_pred: (b x 5)
        :param class_label: (b x h x w) One hot tensor.
        :param offset_label: (b x 5) The offset parameters for the best box for each sample.
        :return:
        """
        class_targs, reg_targs = class_label, offset_label

        class_preds, reg_preds = class_pred, offset_pred

        # mining for negatives!
        keep = class_targs > 0
        conf_preds = class_preds
        conf_preds[keep] = 0.  # only find negatives
        _, topthree = conf_preds.topk(3, 1)  # TODO: make ratio configurable
        keep.scatter_(1, topthree, 1.)
        class_loss = self.class_criterion(class_preds[keep], class_targs[keep])

        reg_loss = self.reg_criterion(reg_preds, reg_targs)
        return {'loss': class_loss + reg_loss, 'class_loss': class_loss, 'reg_loss': reg_loss}


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
        court_preds = preds['court']  # (b, 4, 56, 56)
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

        score_loss = self.score_criterion((preds['score_class'], preds['score_offset']),
                                           (score_idx.to(device), score_coords.to(device)))
        court_loss = self.court_criterion(preds_weighted, targets_weighted)
        return {'score_loss': score_loss, 'court_loss': court_loss,
                'loss': score_loss * self.score_weight + court_loss * self.court_weight}
