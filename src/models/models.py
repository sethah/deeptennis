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
            self.feats.add_module("feat%d" % i, PoseFeatureBlock(nout, k, self.feature_kernel))
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

    def __init__(self, keypoints, channels, n_down=4, n_up=3):
        super(PoseUNet, self).__init__(keypoints, channels)
        self.inc = InConv(channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, keypoints)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
#         x = self.up4(x, x1)
        x = self.outc(x)
        return x


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


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
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

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.Up(x1)
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