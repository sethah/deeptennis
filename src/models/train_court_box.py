import numpy as np
import argparse
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms

from tqdm import tqdm

from data.dataset import ImageDataset
import utils
from src.data.clip import Clip, Video
from src.data.dataset import ImageDatasetBox
from src.vision.transforms import *


class WrapperDataset(torch.utils.data.Dataset):
    def __init__(self, bbox_ds, grid_size):
        self.ds = bbox_ds
        self.grid_size = grid_size

    def __getitem__(self, idx):
        data, target = self.ds[idx]
        gsize = torch.IntTensor(self.grid_size).long().to(data.device)
        im_size = torch.tensor(data.size()[1:]).long().to(data.device)
        grid_coords = (target / (im_size / gsize).double()).long()
        n_coords = 4
        z = torch.zeros([n_coords] + gsize.tolist())
        for i, coord in enumerate(grid_coords):
            c = coord.numpy()
            if np.all(c >= 0) and np.all(c < np.array(self.grid_size)):
                z[i, c[1], c[0]] = 1.
        return data, z

    def __len__(self):
        return len(self.ds)


class StdConv(nn.Module):

    def __init__(self, nin, nout, stride=2, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.bn(F.relu(self.conv(x))))


class CornerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(0.25)
        self.sconv0 = StdConv(128, 128, stride=1)
        self.sconv2 = StdConv(128, 128, stride=1)
        self.out = nn.Conv2d(128, 4, 3, padding=1)

    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv2(x)
        return self.out(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs-path", type=str)
    parser.add_argument("--model-save-path", type=str, default=None)
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--gpu", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=394)
    parser.add_argument("--img-mean", type=str, default=None)
    parser.add_argument("--img-std", type=str, default=None)
    parser.add_argument("--embed-size", type=int, default=32)
    parser.add_argument("--initial-lr", type=float, default=0.001)
    parser.add_argument("--lr-gamma", type=float, default=0.95)
    args = parser.parse_args()

    im_size = (int(x) for x in args.im_size.split(","))

    video_frames = list(Path(args.frame_path).iterdir())
    holdout_match = video_frames[np.random.randint(len(video_frames))]
    logging.debug(f"Holding out match {holdout_match.stem}")

    train_bboxes = []
    train_frames = []

    for v in [f for f in video_frames if f != holdout_match]:
        train_video = Video(f"../data/processed/frames/{v}/")
        train_clips = Clip.from_csv(f"../data/interim/clips/{v}.csv", train_video)
        train_frames += [f for c in train_clips for f in c.frames][:300]
        train_bboxes += [b.reshape(4, 2) for c in train_clips for b in c.bboxes][:300]
    simple_transforms = Compose([Resize(im_size), WrapTransform(transforms.ToTensor())])
    train_ds = ImageDatasetBox(train_frames, train_bboxes, transform=simple_transforms)
