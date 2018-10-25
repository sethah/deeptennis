import torch
import torch.nn as nn


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

        class_targs = torch.zeros(b, nbox, dtype=torch.float32, device=reg_preds.device)
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


class AnchorBoxes(object):

    def __init__(self, grid_sizes, box_sizes, im_size, angle_scale):
        self.im_size = im_size
        self.boxes, self.offsets = AnchorBoxes.get_anchors(grid_sizes, box_sizes, im_size, angle_scale)

    @staticmethod
    def get_anchors(grid_sizes, box_sizes, im_size, angle_scale):
        boxes = []
        offsets = []
        for gw, gh in grid_sizes:
            ix = torch.arange(gw).unsqueeze(1).repeat(1, gh).view(-1)
            iy = torch.arange(gh).unsqueeze(1).repeat(gw, 1).view(-1)
            cw, ch = im_size[0] / gw, im_size[1] / gh
            for bw, bh in box_sizes:
                scale = torch.tensor([ch, cw, 1, 1, 1], dtype=torch.float32)
                offset = torch.tensor([ch / 2, cw / 2, 0, 0, 0], dtype=torch.float32)
                _boxes = torch.stack((iy, ix, torch.ones_like(ix) * bw, torch.ones_like(ix) * bh,
                                      torch.zeros_like(ix)), dim=1).type(torch.float32)
                _offsets = torch.ones((ix.shape[0], 5)) * \
                           torch.tensor([ch / 2, cw / 2, bw / 2, bh / 2, angle_scale])
                boxes.append(_boxes * scale + offset)
                offsets.append(_offsets)
        return torch.cat(boxes).type(torch.float32), torch.cat(offsets).type(torch.float32)

    def get_best(self, label_boxes):
        self.boxes = self.boxes.to(label_boxes.device)
        diff = label_boxes[:, :2].unsqueeze(1) - self.boxes[:, :2]
        return torch.pow(diff, 2).sum(dim=2).argmin(dim=1)


class CourtScoreLoss(nn.Module):

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
