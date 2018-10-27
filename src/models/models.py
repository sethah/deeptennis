import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SimpleConvNet(KeypointModel):
    def __init__(self, keypoints, channels):
        super(SimpleConvNet, self).__init__(keypoints, channels)
        self.conv1 = StdConv(channels, 32, stride=1)
        self.conv2 = StdConv(32, 32, stride=2)
        self.conv3 = StdConv(32, 64, stride=1)
        self.conv4 = StdConv(64, 64, stride=2)
        self.out = nn.Conv2d(64, keypoints, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.out(x)


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


class PoseUNet(KeypointModel):

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

    def forward(self, x):
        x1 = self.inc(x)
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


class FPN(nn.Module):
    """
    A feature pyramid network.

    Similar in concept to U-net, it downsample the image in stages, extracting coarse,
    high-level features. It then upsamples in stages, combining coarse, high-level
    features with fine-grained, low-level features.

    Reference: https://arxiv.org/abs/1612.03144
    """
    def __init__(self, C1, C2, C3, C4, C5, out_channels=128):
        super(FPN, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.out_channels = out_channels
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(128, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(64, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)

        p5_out = self.P5_conv1(x)
        p4_out = self.P4_conv1(c4_out) + F.interpolate(p5_out, scale_factor=2)
        p3_out = self.P3_conv1(c3_out) + F.interpolate(p4_out, scale_factor=2)
        p2_out = self.P2_conv1(c2_out) + F.interpolate(p3_out, scale_factor=2)

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)
        return [p2_out, p3_out, p4_out, p5_out, p6_out]


class CourtScoreHead(nn.Module):
    """
    A prediction head that can be stacked on a feature pyramid. This head
    outputs heatmaps for each court vertex as well as SSD predictions for the
    scoreboard location and size.
    """

    def __init__(self, in_channels, out_channels):
        super(CourtScoreHead, self).__init__()

        self.out_conv_score = nn.Conv2d(64, out_channels, 3)
        self.conv1_court = DoubleConv(in_channels, 64)
        self.conv2_court = DoubleConv(in_channels, 64)
        self.conv3_court = DoubleConv(in_channels, 64)
        self.conv4_court = DoubleConv(in_channels, 64)
        self.conv5_court = StdConv(64 * 4, 64, drop=0.4)
        self.out_conv_court = nn.Conv2d(64, 4, 3)

    def forward(self, x):
        court1 = self.conv1_court(x[-2])
        court2 = self.conv2_court(x[-3])
        court3 = self.conv3_court(x[-4])
        court4 = self.conv4_court(x[-5])
        court = torch.cat([F.interpolate(court1, scale_factor=8),
                           F.interpolate(court2, scale_factor=4),
                           F.interpolate(court3, scale_factor=2),
                           court4], dim=1)
        court = self.conv5_court(court)
        out_score = self.out_conv_score(court)
        out_score_class = out_score[:, 0, :, :]
        out_score_reg = out_score[:, 1:, :, :]
        return [self.out_conv_court(court), (out_score_class, torch.tanh(out_score_reg))]


class AnchorBoxModel(nn.Module):
    """
    A model that wraps an SSD model, but holds anchor box data as parameters. These
    parameters are saved along with the model and can be used at inference time.
    """

    def __init__(self, stages: List[nn.Module],
                        grid_sizes: List[Tuple[int, int]],
                        box_sizes: List[Tuple[int, int]],
                        im_size: Tuple[int, int],
                        angle_scale: int):
        super(AnchorBoxModel, self).__init__()
        self.model = nn.Sequential(*stages)
        self.boxes, self.offsets = self.get_anchors(grid_sizes, box_sizes, im_size, angle_scale)
        self.boxes = torch.nn.Parameter(self.boxes, requires_grad=False)
        self.offsets = torch.nn.Parameter(self.offsets, requires_grad=False)

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, loader: torch.utils.data.DataLoader, device: torch.device):
        self.eval()
        inps = []
        score_preds = []
        court_preds = []
        for inp, *_ in loader:
            b = inp.shape[0]
            inp = inp.to(device)
            inps.append(inp.detach().cpu())
            court, (class_score, reg_score) = self.forward(inp)
            box_idxs = torch.argmax(class_score.view(b, -1), dim=1)
            reg_score = reg_score.view(b, 5, -1)[torch.arange(b), :, box_idxs]
            reg_score = reg_score * self.offsets[box_idxs] + self.boxes[box_idxs]

            score_preds.append(reg_score.detach().cpu())
            court_preds.append(court.detach().cpu())
        court_preds = torch.cat(court_preds, dim=0)
        score_preds = torch.cat(score_preds, dim=0)
        inps = torch.cat(inps, dim=0).permute(0, 2, 3, 1)
        return inps, court_preds, score_preds

    @staticmethod
    def get_anchors(grid_sizes: List[Tuple[int, int]],
                    box_sizes: List[Tuple[int, int]],
                    im_size: Tuple[int, int],
                    angle_scale: int):
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
