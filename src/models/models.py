import numpy as np
import math
import cv2
from overrides import overrides
from typing import List, Tuple, Dict, Iterable

from allennlp.models import Model
from allennlp.data import Vocabulary

from src.models.loss import SSDLoss, CourtScoreLoss
from src.vision.transforms import BoxToCoords, box_to_coords, BoundingBox
from src.models import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models


class KeypointModel(nn.Module):

    def __init__(self, keypoints, channels):
        super().__init__()
        self.keypoints = keypoints
        self.channels = channels

    def predict(self, x):
        return self.forward(x)


class StdConv(nn.Module):

    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.bn(F.relu(self.conv(x))))


class ScoreModel(Model):

    pass

@Model.register("score_head")
class ScoreHead(Model):
    """
    A prediction head that can be stacked on a feature pyramid. This head
    outputs heatmaps for each court vertex as well as SSD predictions for the
    scoreboard location and size.
    """

    def __init__(self, in_channels: int, feature_channels: int = 64, nmaps: int = 4):
        """
        :param in_channels: input channels for each feature map.
        :param feature_channels: number of intermediate channels before output predictions
        :param nmaps: the number of different sized feature maps. These maps should be ordered
                      largest to smallest and should be in increasing powers of two. For example,
                      `nmaps = 4` could have 4 feature maps of sizes [56x56, 28x28, 14x14, 7x7].
        """
        super(ScoreHead, self).__init__(None)

        score_regression_channels = 5
        score_location_channels = 1
        self.court_convs = nn.ModuleList([DoubleConv(in_channels, feature_channels)
                                          for _ in range(nmaps)])
        self.conv1 = StdConv(feature_channels * nmaps, feature_channels, drop=0.4)
        self.out_conv_score = nn.Conv2d(feature_channels,
                                        score_location_channels + score_regression_channels,
                                        kernel_size=3, padding=1)

    def forward(self, feature_maps):
        out = [layer.forward(feature_map) for layer, feature_map in
                         zip(self.court_convs, feature_maps)]
        out = torch.cat([F.interpolate(c, scale_factor=2**j) for j, c
                           in enumerate(out)], dim=1)
        out = self.conv1(out)
        out_score = self.out_conv_score(out)
        out_score_class = out_score[:, 0, :, :]
        out_score_reg = torch.tanh(out_score[:, 1:, :, :])
        return {
            "score_class": out_score_class,
            "score_offset": out_score_reg
        }

@Model.register("anchor_score")
class AnchorScoreModel(Model):
    """
    A model that wraps an SSD model, but holds anchor box data as parameters. These
    parameters are saved along with the model and can be used at inference time.
    """

    def __init__(self, stages: List[Model],
                        grid_sizes: List[List[int]],
                        box_sizes: List[List[int]],
                        im_size: List[int],
                        angle_scale: int):
        super(AnchorScoreModel, self).__init__(Vocabulary())
        self.model = nn.Sequential(*stages)
        self.boxes, self.offsets = self.get_anchors(grid_sizes, box_sizes, im_size, angle_scale)
        self.boxes = torch.nn.Parameter(self.boxes, requires_grad=False)
        self.offsets = torch.nn.Parameter(self.offsets, requires_grad=False)
        self.grid_sizes = grid_sizes
        self.im_size = im_size
        class_crit = nn.BCEWithLogitsLoss()
        reg_crit = nn.L1Loss()
        self.criterion = SSDLoss(class_crit, reg_crit)
        self._metrics = {}
        self._metric = metrics.IOU(im_size)

    def forward(self,
                img: torch.Tensor,
                court: torch.Tensor = None,
                score: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        out = self.model.forward(img)
        if score is not None:
            score_class = AnchorBoxModel.get_best(self.boxes, score)
            score_offset = (score - self.boxes[score_class]) / self.offsets[score_class]
            labels = {'score_offset': score_offset, 'score_class': score_class}
            loss_dict = self.criterion((out['score_class'], out['score_offset']),
                                       (score_class.long().to(score.device), score_offset.to(score.device)))
            self._metric(self.decode(out)['score'].cpu(),
                         score.cpu())
            self._metrics = {f'model_{k}': v.item() for k, v in loss_dict.items()}
            out['loss'] = loss_dict['loss']
        return out

    @overrides
    def get_metrics(self, reset: bool = False):
        self._metrics.update({"IOU": self._metric.get_metric(reset)})
        return self._metrics

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert the scoreboard predictions to coordinates.
        """
        class_score = output_dict['score_class']
        reg_score = output_dict['score_offset']
        b = class_score.shape[0]
        box_idxs = torch.argmax(class_score.view(b, -1), dim=1)
        reg_score = reg_score.view(b, 5, -1)[torch.arange(b), :, box_idxs]
        reg_score = reg_score * self.offsets[box_idxs] + self.boxes[box_idxs]
        out = {'score': reg_score}
        return out

    def heatmaps_to_vertices(self, heatmaps: torch.Tensor, width: int, height: int) -> torch.Tensor:
        hmaps = heatmaps.detach().numpy()
        grid_size = max(self.grid_sizes)
        x, y = np.unravel_index(
            np.argmax(hmaps.reshape(hmaps.shape[0], hmaps.shape[1], -1), axis=2), hmaps.shape[-2:])
        resize_scale = np.array([height / grid_size[1], width / grid_size[0]])
        court_vertices = np.stack([y, x], axis=2) * resize_scale
        return torch.from_numpy(court_vertices)

    def box_to_vertices(self, boxes: torch.Tensor, width: int, height: int):
        coords = box_to_coords(boxes)
        resize = torch.tensor([height / self.im_size[0], width / self.im_size[1]],
                              dtype=torch.float32)
        return coords * resize

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

    @staticmethod
    def get_best(boxes: torch.Tensor, label_boxes: torch.Tensor) -> torch.Tensor:
        """
        Given anchor boxes and a set of label boxes, match each label box
        to the best anchor box. It does this by matching each label box to the
        anchor box with the closest center.

        TODO: this should use Jaccard similarity
        """
        diff = label_boxes[:, :2].unsqueeze(1) - boxes[:, :2]
        return torch.pow(diff, 2).sum(dim=2).argmin(dim=1)


@Model.register("resnet")
class ResnetScoreModel(ScoreModel):

    def __init__(self,
                 keypoints: int,
                 im_size: Tuple[int, int],
                 angle_scale: float = 360.,
                 width_scale: int = 100,
                 height_scale: int = 100):
        super(ResnetScoreModel, self).__init__(None)
        res = torch_models.resnet34(pretrained=True)
        for p in res.parameters():
            p.requires_grad = False
        self.backbone = nn.Sequential(*list(res.children())[:-4])
        self.out_conv = StdConv(128, 32)
        nfeatures = self._get_last_conv_size(im_size, nn.Sequential(self.backbone, self.out_conv))
        self.out = nn.Sequential(nn.Linear(nfeatures, keypoints))
        self.im_size = im_size
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.angle_scale = angle_scale
        self._scale_tensor = torch.tensor([self.im_size[0], self.im_size[1], width_scale,
                                           height_scale, angle_scale])
        self._forward_count = 0
        self._metric = metrics.IOU(im_size)

    @staticmethod
    def _get_last_conv_size(im_size: Tuple[int, int], layers: nn.Module):
        rand = torch.rand(1, 3, im_size[0], im_size[1])
        out = layers.forward(rand)
        return out.numel()

    def forward(self,
                img: torch.Tensor,
                court: torch.Tensor = None,
                score: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        score = score / self._scale_tensor.to(score.device)
        x = self.backbone(img)
        x = self.out_conv(x)
        x = self.out(x.view(x.shape[0], -1))
        pred = torch.sigmoid(x)
        out = {'score': pred}
        if score is not None:
            out['loss'] = F.mse_loss(pred, score)
            self._forward_count += pred.shape[0]
            self._metric(pred.cpu() * self._scale_tensor, score.cpu() * self._scale_tensor)
        return out

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"IOU": self._metric.get_metric(reset)}

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert the scoreboard predictions to coordinates.
        """
        return {'score': output_dict['score'] * self._scale_tensor}


@Model.register("simple")
class SimpleConvNet(ScoreModel):

    def __init__(self,
                 keypoints: int,
                 im_size: Tuple[int, int],
                 angle_scale: float = 360.,
                 width_scale: int = 100,
                 height_scale: int = 100):
        super(SimpleConvNet, self).__init__(None)
        self.layers = nn.Sequential(*[
            StdConv(3, 32, stride=1),
            StdConv(32, 32, stride=2),
            StdConv(32, 64, stride=1),
            StdConv(64, 64, stride=2),
            StdConv(64, 128, stride=1),
            StdConv(128, 128, stride=2),
            StdConv(128, 256, stride=1),
            StdConv(256, 256, stride=2)
        ])
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.angle_scale = angle_scale
        self.im_size = im_size
        self._scale_tensor = torch.tensor([self.im_size[0], self.im_size[1], width_scale,
                                           height_scale, angle_scale])
        fc_params = self._get_last_conv_size(im_size, self.layers)
        self.out = nn.Linear(fc_params, keypoints)
        self._forward_count = 0
        self._metric = metrics.IOU(im_size)

    @staticmethod
    def _get_last_conv_size(im_size: Tuple[int, int], layers: nn.Module):
        rand = torch.rand(1, 3, im_size[0], im_size[1])
        out = layers.forward(rand)
        return out.numel()

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert the scoreboard predictions to coordinates.
        """
        return {'score': output_dict['score'] * self._scale_tensor}

    def forward(self,
                img: torch.Tensor,
                court: torch.Tensor = None,
                score: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        score = score / self._scale_tensor.to(score.device)
        x = self.layers(img)
        x = self.out(x.view(x.shape[0], -1))
        pred = torch.sigmoid(x)
        out = {'score': pred}
        if score is not None:
            out['loss'] = F.mse_loss(pred, score)
            self._forward_count += pred.shape[0]
            self._metric(pred.cpu() * self._scale_tensor, score.cpu() * self._scale_tensor)
        return out

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"IOU": self._metric.get_metric(reset)}


class PoseBlock(nn.Module):

    def __init__(self, nin, nout, kernel_size=7):
        super(PoseBlock, self).__init__()
        n_middle = nout // 2
        self.conv1 = StdConv(nin, n_middle, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = StdConv(n_middle, n_middle, kernel_size=3, stride=2)
        self.conv3 = StdConv(n_middle, n_middle, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv4 = StdConv(n_middle, n_middle, kernel_size=3, stride=2)
        self.conv5 = StdConv(n_middle, nout, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv6 = StdConv(nout, nout, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class PoseFeatureBlock(nn.Module):

    def __init__(self, nin, k, kernel_size=9):
        super(PoseFeatureBlock, self).__init__()
        self.conv1 = StdConv(nin + k, nin, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = StdConv(nin, nin, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(nin, k, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SimplePose(KeypointModel):

    def __init__(self, keypoints, channels, nout, layers=2, pose_kernel=9, feature_kernel=9):
        super(SimplePose, self).__init__(keypoints, channels)
        self.pose_kernel = pose_kernel
        self.feature_kernel = feature_kernel
        self.pose1 = PoseBlock(channels, nout, self.pose_kernel)
        self.poses = nn.ModuleList()
        self.feats = nn.ModuleList()
        for i in range(1, layers):
            self.feats.add_module("feat%d" % i, PoseFeatureBlock(nout, keypoints, self.feature_kernel))
            self.poses.add_module("pose%d" % i, PoseBlock(3, nout, self.pose_kernel))

        self.feat1 = nn.Sequential(StdConv(nout, nout, self.feature_kernel, padding=self.feature_kernel // 2),
                                   StdConv(nout, nout, self.feature_kernel, padding=self.feature_kernel // 2),
                                   # StdConv(nout, nout, 9, padding=4),
                                   nn.Conv2d(nout, keypoints, 1, padding=0))

    def forward(self, x):
        x1 = self.pose1(x)
        out1 = self.feat1(x1)
        outs = [out1]
        for feat, pose in zip(self.feats, self.poses):
            x2 = pose(x)
            x2 = torch.cat([outs[-1], x2], dim=1)
            out2 = feat(x2)
            outs.append(out2)
        return outs

    def predict(self, x):
        output = self.forward(x)
        return output[-1]


@Model.register("pose_unet")
class PoseUNet(Model):

    def __init__(self, keypoints, channels, filters):
        super(PoseUNet, self).__init__(keypoints, channels)
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

    def forward(self, img: torch.Tensor):
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
        return x


class DoubleConv(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(p=0.4)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SamePad2d(nn.Module):
    """
    Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


@Model.register("backbone")
class BackboneModel(Model):

    def __init__(self, freeze: bool = True):
        super(BackboneModel, self).__init__(None)
        base = torch_models.resnet34(pretrained=True)
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1))
        self.stages.append(base.layer2)
        self.stages.append(base.layer3)
        self.stages.append(base.layer4)
        for p in self.parameters():
            p.requires_grad = not freeze
        for p in self.stages[0].parameters():
            p.requires_grad = False

    def get_channels(self) -> List[int]:
        im = torch.randn(1, 3, 224, 224)
        channels = []
        out = im
        for stage in self.stages:
            out = stage.forward(out)
            channels.append(out.shape[1])
        return channels

    def param_groups(self) -> List[Tuple[str, nn.Parameter]]:
        _params = []
        for j, stage in enumerate(self.stages):
            for i, p in enumerate(stage.parameters()):
                _params.append((f'backbone_{j}_{i}', p))
        return _params


@Model.register("fpn")
class FPN(Model):
    """
    A feature pyramid network.

    Similar in concept to U-net, it downsamples the image in stages, extracting coarse,
    high-level features. It then upsamples in stages, combining coarse, high-level
    features with fine-grained, low-level features.

    Reference: https://arxiv.org/abs/1612.03144
    """
    def __init__(self,
                 backbone: BackboneModel,
                 out_channels: int = 128):
        super(FPN, self).__init__(None)
        self.backbone: Model = backbone
        self.out_channels = out_channels
        self.upsample = nn.ModuleDict()

        for i, n_channels in enumerate(self.backbone.get_channels()):
            self.upsample[f'P{i}_conv1'] = StdConv(n_channels, self.out_channels,
                                                       kernel_size=1, stride=1, drop=0.4,
                                                   padding=0)
            self.upsample[f'P{i}_conv2'] = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1))

    def get_grid_sizes(self, im_h, im_w):
        x = torch.randn(1, 3, im_h, im_w)
        out = self.forward(x)
        return [tuple(o.shape[-2:]) for o in out]

    def forward(self, x):
        outputs = []
        out = x
        for stage in self.backbone.stages:
            out = stage(out)
            outputs.append(out)

        n = len(self.backbone.stages)
        pouts = [self.upsample[f'P{n - 1}_conv1'](outputs[-1])]
        out = pouts[-1]
        for i in range(n - 2, -1, -1):
            out = self.upsample[f'P{i}_conv1'](outputs[i]) + F.interpolate(out, scale_factor=2)
            pouts.append(out)
        pouts = [self.upsample[f'P{n - 1 - i}_conv2'](p) for i, p in enumerate(pouts)]

        return list(reversed(pouts))


@Model.register("court_head")
class CourtScoreHead(Model):
    """
    A prediction head that can be stacked on a feature pyramid. This head
    outputs heatmaps for each court vertex as well as SSD predictions for the
    scoreboard location and size.
    """

    def __init__(self, in_channels: int, feature_channels: int = 64, nmaps: int = 4):
        """
        :param in_channels: input channels for each feature map.
        :param feature_channels: number of intermediate channels before output predictions
        :param nmaps: the number of different sized feature maps. These maps should be ordered
                      largest to smallest and should be in increasing powers of two. For example,
                      `nmaps = 4` could have 4 feature maps of sizes [56x56, 28x28, 14x14, 7x7].
        """
        super(CourtScoreHead, self).__init__(None)

        court_keypoints = 4
        score_regression_channels = 5
        score_location_channels = 1
        self.court_convs = nn.ModuleList([DoubleConv(in_channels, feature_channels)
                                          for _ in range(nmaps)])
        self.conv1 = StdConv(feature_channels * nmaps, feature_channels, drop=0.4)
        self.out_conv_court = nn.Conv2d(feature_channels, court_keypoints, kernel_size=3, padding=1)
        self.out_conv_score = nn.Conv2d(feature_channels,
                                        score_location_channels + score_regression_channels,
                                        kernel_size=3, padding=1)

    def forward(self, feature_maps):
        out = [layer.forward(feature_map) for layer, feature_map in
                         zip(self.court_convs, feature_maps)]
        out = torch.cat([F.interpolate(c, scale_factor=2**j) for j, c
                           in enumerate(out)], dim=1)
        out = self.conv1(out)
        out_score = self.out_conv_score(out)
        out_score_class = out_score[:, 0, :, :]
        out_score_reg = torch.tanh(out_score[:, 1:, :, :])
        return {
            "court": self.out_conv_court(out),
            "score_class": out_score_class,
            "score_offset": out_score_reg
        }


@Model.register("anchor")
class AnchorBoxModel(Model):
    """
    A model that wraps an SSD model, but holds anchor box data as parameters. These
    parameters are saved along with the model and can be used at inference time.
    """

    def __init__(self, stages: List[Model],
                        grid_sizes: List[List[int]],
                        box_sizes: List[List[int]],
                        im_size: List[int],
                        angle_scale: int):
        super(AnchorBoxModel, self).__init__(Vocabulary())
        self.model = nn.Sequential(*stages)
        self.boxes, self.offsets = self.get_anchors(grid_sizes, box_sizes, im_size, angle_scale)
        self.boxes = torch.nn.Parameter(self.boxes, requires_grad=False)
        self.offsets = torch.nn.Parameter(self.offsets, requires_grad=False)
        self.grid_sizes = grid_sizes
        self.im_size = im_size
        court_crit = nn.MSELoss()
        class_crit = nn.BCEWithLogitsLoss()
        reg_crit = nn.L1Loss()
        ssd_crit = SSDLoss(class_crit, reg_crit)
        self.criterion = CourtScoreLoss(court_crit, ssd_crit, court_weight=25., score_weight=1.)
        self._metrics = {}

    def forward(self,
                img: torch.Tensor,
                court: torch.Tensor = None,
                score: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        out = self.model.forward(img)
        if court is not None:
            score_class = AnchorBoxModel.get_best(self.boxes, score)
            score_offset = (score - self.boxes[score_class]) / self.offsets[score_class]
            labels = {'court': court, 'score_offset': score_offset, 'score_class': score_class}
            loss_dict = self.criterion(out, labels)
            self._metrics = {f'model_{k}': v.item() for k, v in loss_dict.items()}
            out['loss'] = loss_dict['loss']
        return out

    @overrides
    def get_metrics(self, reset: bool = False):
        return self._metrics

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert the scoreboard predictions to coordinates.
        """
        court = output_dict['court']
        class_score = output_dict['score_class']
        reg_score = output_dict['score_offset']
        b = court.shape[0]
        box_idxs = torch.argmax(class_score.view(b, -1), dim=1)
        reg_score = reg_score.view(b, 5, -1)[torch.arange(b), :, box_idxs]
        reg_score = reg_score * self.offsets[box_idxs] + self.boxes[box_idxs]
        out = {'court': court, 'score': reg_score}
        return out

    def heatmaps_to_vertices(self, heatmaps: torch.Tensor, width: int, height: int) -> torch.Tensor:
        hmaps = heatmaps.detach().numpy()
        grid_size = max(self.grid_sizes)
        x, y = np.unravel_index(
            np.argmax(hmaps.reshape(hmaps.shape[0], hmaps.shape[1], -1), axis=2), hmaps.shape[-2:])
        resize_scale = np.array([height / grid_size[1], width / grid_size[0]])
        court_vertices = np.stack([y, x], axis=2) * resize_scale
        return torch.from_numpy(court_vertices)

    def box_to_vertices(self, boxes: torch.Tensor, width: int, height: int):
        coords = box_to_coords(boxes)
        resize = torch.tensor([height / self.im_size[0], width / self.im_size[1]],
                              dtype=torch.float32)
        return coords * resize

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

    @staticmethod
    def get_best(boxes: torch.Tensor, label_boxes: torch.Tensor) -> torch.Tensor:
        """
        Given anchor boxes and a set of label boxes, match each label box
        to the best anchor box. It does this by matching each label box to the
        anchor box with the closest center.

        TODO: this should use Jaccard similarity
        """
        diff = label_boxes[:, :2].unsqueeze(1) - boxes[:, :2]
        return torch.pow(diff, 2).sum(dim=2).argmin(dim=1)

