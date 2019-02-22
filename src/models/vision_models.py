import numpy as np
from overrides import overrides
from typing import List, Tuple, Dict, Iterable

from allennlp.models import Model
from allennlp.data import Vocabulary

from src.models.loss import AnchorBoxLoss, SSDLoss, CourtScoreLoss
from src.models import metrics
from src.models.models import FPN, BackboneModel, StdConv, DoubleConv, InConv, Down, Up, OutConv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models


class Im2ImEncoder(Model):

    def __init__(self, input_size: Tuple[int, int]):
        super(Im2ImEncoder, self).__init__(None)
        self.input_size = input_size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_output_shape(self) -> Tuple[int, int]:
        fake_img = torch.randn(1, 3, self.input_size[0], self.input_size[1])
        out = self.forward(fake_img)
        return tuple(out.shape[-2:])

@Im2ImEncoder.register("unet")
class UNet(Im2ImEncoder):

    def __init__(self, keypoints: int, channels: int, filters: int, input_size: Tuple[int, int]):
        super(UNet, self).__init__(input_size)
        self.inc = InConv(channels, filters)
        self.down1 = Down(filters, filters * 2)
        self.down2 = Down(filters * 2, filters * 4)
        self.down3 = Down(filters * 4, filters * 8)
        self.down4 = Down(filters * 8, filters * 8)
        self.up1 = Up(filters * 16, filters * 4)
        self.up2 = Up(filters * 8, filters * 2)
        self.up3 = Up(filters * 4, filters)
        self.up4 = Up(filters * 2, filters)
        self.outc = OutConv(filters, keypoints)

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return {'logits': x}


@Model.register("segmentation")
class SegmentationModel(Model):

    def __init__(self, encoder: Im2ImEncoder):
        """

        :param encoder: ``Im2ImEncoder``, required
            An image encoder that outputs a segmentation mask of size (num_classes x h x w)
        """
        super(SegmentationModel, self).__init__(None)
        self.encoder = encoder
        self.criterion = nn.CrossEntropyLoss()
        self.input_size = self.encoder.input_size
        self._iou = metrics.SegmentationIOU(2, self.input_size, skip=set([0]))
        self._metrics = {}

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """

        :param img: (b x c x h x w)
        :param mask: (b x h x w)
        :return:
        """
        pred = self.encoder.forward(img)
        out = {'pred': pred['logits']}
        if mask is not None:
            loss = self.criterion(pred['logits'], mask.long())
            out['loss'] = loss
            self._iou(self.decode(out)['pred'].cpu(), mask.cpu())
        return out

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert the scoreboard predictions to coordinates.
        """
        logits = output_dict['pred']
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        out = {'pred': preds}
        return out

    @overrides
    def get_metrics(self, reset: bool = False):
        self._metrics.update({"IOU": self._iou.get_metric(reset)})
        return {metric_name: metric for metric_name, metric in self._metrics.items()}

@Im2ImEncoder.register("fpn_encoder")
class FPNEncoder(Im2ImEncoder):

    def __init__(self,
                 input_size: Tuple[int, int],
                 feature_channels: int = 64):
        super(FPNEncoder, self).__init__(input_size)
        backbone = BackboneModel(False)
        self.fpn = FPN(backbone)
        fake_img = torch.randn(1, 3, input_size[0], input_size[1])
        fpn_out = self.fpn(fake_img)
        nmaps = len(fpn_out)
        in_channels = fpn_out[0].shape[1]
        self.court_convs = nn.ModuleList([DoubleConv(in_channels, feature_channels)
                                          for _ in range(nmaps)])
        self.conv1 = StdConv(feature_channels * nmaps, feature_channels, drop=0.4)
        self.out_conv_score = nn.Conv2d(feature_channels,
                                        6,
                                        kernel_size=3, padding=1)

    def forward(self, img: torch.Tensor):
        feature_maps = self.fpn.forward(img)
        out = [layer.forward(feature_map) for layer, feature_map in
               zip(self.court_convs, feature_maps)]
        out = torch.cat([F.interpolate(c, scale_factor=2**j) for j, c
                         in enumerate(out)], dim=1)
        out = self.conv1(out)
        return self.out_conv_score(out)


@Model.register("anchorbox")
class AnchorBoxModel(Model):

    def __init__(self,
                 encoder: Im2ImEncoder,
                 box_sizes: List[Tuple[int, int]],
                 angle_scale: float = 10.,
                 criterion: AnchorBoxLoss = SSDLoss(nn.BCEWithLogitsLoss(), nn.L1Loss())):
        super(AnchorBoxModel, self).__init__(None)
        self.encoder = encoder
        self.input_size = self.encoder.input_size
        self.grid_size = self.encoder.get_output_shape()
        self.boxes, self.offsets = self.get_anchors([self.grid_size], box_sizes, self.input_size,
                                                    angle_scale)
        self.criterion = criterion
        self._iou = metrics.IOU(self.input_size)
        self._metrics = {}

    def forward(self, img: torch.Tensor, labels: torch.Tensor = None):
        featurized = self.encoder(img)
        assert featurized.shape[1] == 6
        classification_image = featurized[:, 0, :, :]
        offset_image = featurized[:, 1:, :, :]
        out = {'class_pred': classification_image, 'offset_pred': offset_image}
        if labels is not None:
            self.boxes = self.boxes.to(labels.device)
            self.offsets = self.offsets.to(labels.device)
            box_index = self._box2index(labels)
            class_label = self._index2class(box_index)
            offset_label = self._box2offset(labels, box_index)
            offset_pred = self._index2pred(offset_image, box_index)
            loss_dict = self.criterion(classification_image,
                                       offset_pred,
                                       class_label,
                                       offset_label)
            self._iou(self.decode(out)['pred'].cpu(), labels.cpu())
            self._metrics = {f'model_{k}': v.item() for k, v in loss_dict.items()}
            out['loss'] = loss_dict['loss']
        return out

    @overrides
    def get_metrics(self, reset: bool = False):
        self._metrics.update({"IOU": self._iou.get_metric(reset)})
        return self._metrics

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert the scoreboard predictions to coordinates.
        """
        class_pred = output_dict['class_pred']  # (b x h x w)
        offset_pred = output_dict['offset_pred']
        boxes = self._pred2box(offset_pred, class_pred)
        out = {'pred': boxes}
        return out

    def _pred2box(self, offset_pred: torch.Tensor, class_pred: torch.Tensor) -> torch.Tensor:
        """

        :param offset_pred: (b x 5 x h' x w')
        :param class_pred: (b x h' x w')
        :return:
        """
        b = class_pred.shape[0]
        box_idxs = torch.argmax(class_pred.view(b, -1), dim=1)
        offset_pred = offset_pred.view(b, 5, -1)[torch.arange(b), :, box_idxs]
        return offset_pred * self.offsets[box_idxs] + self.boxes[box_idxs]

    def _box2index(self, label_boxes: torch.Tensor) -> torch.Tensor:
        """
        Given anchor boxes and a set of label boxes, match each label box
        to the best anchor box. It does this by matching each label box to the
        anchor box with the closest center.

        TODO: this should use Jaccard similarity
        """
        diff = label_boxes[:, :2].unsqueeze(1) - self.boxes[:, :2]
        return torch.pow(diff, 2).sum(dim=2).argmin(dim=1)

    def _index2pred(self, offset_predictions: torch.Tensor, indices: torch.Tensor):
        """
        Choose one offset prediction per instance from the offset predictions for each
        anchor box.
        :param offset_predictions: (b x 5 x h' x w')
        :param indices: (b x (h' * w'))
        :return:
        """
        b, c = offset_predictions.shape[:2]
        return offset_predictions.view(b, c, -1)[torch.arange(b), :, indices.squeeze()]

    def _index2class(self, indices: torch.Tensor):
        """
        :param indices: (b x 1)
        :return: (b x h' x w')
        """
        b = indices.shape[0]
        nbox = np.prod(self.grid_size)
        class_targs = torch.zeros(b, nbox, dtype=torch.float32,
                                  device=indices.device)
        class_targs[torch.arange(b), indices.squeeze()] = 1.
        return class_targs.view(b, self.grid_size[0], self.grid_size[1])

    def _box2offset(self, boxes: torch.Tensor, anchor_index: torch.Tensor) -> torch.Tensor:
        """
        Convert raw label boxes to offsets from their closest anchor.
        :param boxes: (b x k x 5)
        :return:
        """
        return (boxes - self.boxes[anchor_index]) / self.offsets[anchor_index]

    @staticmethod
    def get_anchors(grid_sizes: List[Tuple[int, int]],
                    box_sizes: List[Tuple[int, int]],
                    im_size: Tuple[int, int],
                    angle_scale: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

