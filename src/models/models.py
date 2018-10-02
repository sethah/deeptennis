import torch
import torch.nn as nn
import torch.nn.functional as F


class StdConv(nn.Module):

    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.bn(F.relu(self.conv(x))))


class CornerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(0.25), StdConv(128, 128),
                                        StdConv(128, 128), nn.Conv2d(128, 4, 3, padding=1))

    def forward(self, x):
        return self.classifier(x)


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = StdConv(3, 32, stride=1)
        self.conv2 = StdConv(32, 32, stride=2)
        self.conv3 = StdConv(32, 64, stride=1)
        self.conv4 = StdConv(64, 64, stride=2)
        self.out = nn.Conv2d(64, 4, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.out(x)


class PoseBlock(nn.Module):

    def __init__(self, nin, nout):
        super(PoseBlock, self).__init__()
        n_middle = nout // 2
        self.conv1 = StdConv(nin, n_middle, kernel_size=7, stride=1, padding=3)
        self.conv2 = StdConv(n_middle, n_middle, kernel_size=3, stride=2)
        self.conv3 = StdConv(n_middle, nout, kernel_size=7, stride=1, padding=3)
        self.conv4 = StdConv(nout, nout, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class PoseFeatureBlock(nn.Module):

    def __init__(self, nin, k):
        super(PoseFeatureBlock, self).__init__()
        self.conv1 = StdConv(nin + k, nin, 9, padding=4)
        self.conv2 = StdConv(nin, nin, 9, padding=4)
        self.conv3 = StdConv(nin, nin, 9, padding=4)
        self.conv4 = nn.Conv2d(nin, k, 1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class SimplePose(nn.Module):
    def __init__(self, k, nout, layers=2):
        super(SimplePose, self).__init__()
        self.pose1 = PoseBlock(3, nout)
        self.poses = nn.ModuleList()
        self.feats = nn.ModuleList()
        for i in range(1, layers):
            self.feats.add_module("feat%d" % i, PoseFeatureBlock(nout, k))
            self.poses.add_module("pose%d" % i, PoseBlock(3, nout))

        self.feat1 = nn.Sequential(StdConv(nout, nout, 9, padding=4),
                                   StdConv(nout, nout, 9, padding=4),
                                   StdConv(nout, nout, 9, padding=4),
                                   nn.Conv2d(nout, k, 1, padding=0))

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




class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x