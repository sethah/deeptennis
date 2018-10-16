import torch
import torch.nn as nn

import logging


def get_best(label_boxes, anchor_boxes):
    diff = label_boxes[:, :2].unsqueeze(1) - anchor_boxes[:, :2]
    return torch.pow(diff, 2).sum(dim=2).argmin(dim=1)


class SSDLoss(nn.modules.loss._Loss):
    def __init__(self, anchor_boxes, class_criterion, reg_criterion, scale_box,
                 size_average=None, reduce=None, reduction='elementwise_mean'):
        super(SSDLoss, self).__init__(size_average, reduce, reduction)
        self.anchor_boxes = anchor_boxes
        self.class_criterion = class_criterion
        self.reg_criterion = reg_criterion
        self.scale_box = scale_box
        self.cnt = 0

    def forward(self, preds, targ):
        self.cnt += 1
        self.anchor_boxes = self.anchor_boxes.to(preds.device)
        self.scale_box = self.scale_box.to(preds.device)

        best_idxs = get_best(targ, self.anchor_boxes)  # shape (b,)
        maps = torch.zeros(targ.shape[0], self.anchor_boxes.shape[0],
                           dtype=torch.float32, device=preds.device)  # shape (b, 26 * 26)
        maps[torch.arange(targ.shape[0]), best_idxs] = 1.
        class_targ = maps
        class_preds = preds[:, 0, :, :].view(targ.shape[0], -1)  # shape (b, 26 * 26)

        # mining for negatives!
        keep = class_targ > 0
        conf_preds = torch.sigmoid(class_preds)
        conf_preds[keep] = 0.  # only find negatives
        _, topthree = conf_preds.topk(3, 1)
        keep.scatter_(1, topthree, 1.)
        class_loss = self.class_criterion(class_preds[keep], class_targ[keep])

        reg_preds = preds[:, 1:].view(targ.shape[0], 5, -1)
        reg_preds = reg_preds[torch.arange(targ.shape[0]), :, best_idxs]  # shape (b, 5)
        reg_preds = torch.tanh(reg_preds)  # shape (b, 5)
        anchor_points = self.anchor_boxes[best_idxs]  # shape (b, 5)

        # the predictions are offsets from the anchor points
        # centers are allowed to move cell_width / 2, box width can change by default box_w / 2, angle can go +/- 10
        #         reg_preds = anchor_points + reg_preds * self.scale_box
        reg_targ = (targ - anchor_points) / self.scale_box
        reg_loss = self.reg_criterion(reg_preds, reg_targ)
        # if self.cnt % 5 == 0:
        #     logging.debug(f"SSD Losses: {reg_loss}, {class_loss}")
        return class_loss + reg_loss