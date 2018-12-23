import numpy as np
import sys
import argparse
from pathlib import Path
import ipdb
import pickle
import logging
from logging.config import fileConfig
from typing import List, Iterable
from itertools import islice

from allennlp.training import Trainer as ATrainer
from allennlp.training.learning_rate_schedulers import LearningRateWithoutMetricsWrapper, SlantedTriangular
from allennlp.data.fields import ArrayField
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator, BasicIterator

import torch.nn as nn
import torch.utils
import torchvision.models as torch_models
from torch.utils import data as torchdata

from src.data.clip import Video
from src.data.dataset import ImageFilesDatasetKeypoints
from src.vision.transforms import *
import src.models.models as models
import src.utils as utils


def get_dataset(videos: List[Video],
                score_path: Path,
                court_path: Path,
                action_path: Path,
                frame_path: Path,
                max_frames: int=None):
    frames = []
    score_labels = []
    corner_labels = []
    for video in videos:
        score_name = score_path / (video.name + ".pkl")
        court_name = court_path / (video.name + ".pkl")
        if not score_name.exists() or not court_name.exists():
            continue
        with open(score_name, 'rb') as f:
            scores = pickle.load(f)
        with open(court_name, 'rb') as f:
            corners = pickle.load(f)
        action_mask = np.load(action_path / (video.name + ".npy"))
        cnt = 0
        for (fname, box), (fname, coords), is_action in zip(scores, corners, action_mask):
            if is_action and box != [0, 0, 0, 0] and np.any(coords):
                # only use frames that are action and have valid labels
                frames.append(frame_path / video.name / fname)
                corner_labels.append([float(x) for x in coords])
                score_labels.append(box)
                cnt += 1
            if max_frames is not None and cnt > max_frames:
                break
    score_labels = np.array(score_labels)
    corner_labels = np.array(corner_labels).reshape(-1, 4, 2)
    return frames, corner_labels, score_labels


def tennis_data_to_allen(ds: torchdata.Dataset) -> Iterable[Instance]:
    for (img, court, score_offset, score_class) in islice(ds, 100):
        fields = {
            'img': ArrayField(img),
            'court': ArrayField(court),
            'score_offset': ArrayField(score_offset),
            'score_class': ArrayField(score_class)
        }
        yield Instance(fields)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n-valid", type=int, default=1)
    parser.add_argument("--freeze-backbone", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument("--img-mean", type=str, default=None)
    parser.add_argument("--img-std", type=str, default=None)
    parser.add_argument("--initial-lr", type=float, default=0.001)
    parser.add_argument("--lr-gamma", type=float, default=0.95)
    parser.add_argument("--lr-milestones", type=str, default="2,4,6,8")
    parser.add_argument('--restore', type=str, default="", help="{'best', 'latest'}")
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--frame-path', type=str, default="")
    parser.add_argument('--court-path', type=str, default="")
    parser.add_argument('--score-path', type=str, default="")
    parser.add_argument('--action-path', type=str, default="")
    parser.add_argument('--validate-every', type=int, default=1)
    args = parser.parse_args()

    fileConfig('logging_config.ini')
    np.random.seed(args.seed)

    im_size = (args.img_height, args.img_width)
    use_gpu = args.gpu or torch.cuda.is_available()
    train_device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    logging.debug(f"Training on device {train_device}.")

    ds_mean = [float(x) for x in args.img_mean.split(",")]
    ds_std = [float(x) for x in args.img_std.split(",")]

    frame_path = Path(args.frame_path)
    score_path = Path(args.score_path)
    court_path = Path(args.court_path)
    action_path = Path(args.action_path)
    video_paths = []
    for v in list(frame_path.iterdir()):
        score_name = score_path / (v.name + ".pkl")
        court_name = court_path / (v.name + ".pkl")
        if score_name.exists() or court_name.exists():
            video_paths.append(v)

    np.random.shuffle(video_paths)
    train_videos = [Video.from_dir(v) for v in video_paths[args.n_valid:]]
    valid_videos = [Video.from_dir(v) for v in video_paths[:args.n_valid]]
    logging.debug(f"Holding out match {[v.name for v in valid_videos]}")

    im_size = (args.img_height, args.img_width)
    batch_size = args.batch_size
    max_frames = None

    res = torch_models.resnet34(pretrained=True)
    utils.freeze(res.parameters())

    C1 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
    C2 = res.layer1
    C3 = res.layer2
    C4 = res.layer3
    C5 = res.layer4
    head = models.CourtScoreHead(128, out_channels=6)
    fpn = models.FPN(C1, C2, C3, C4, C5)
    model = nn.Sequential(fpn, head)
    sample_img = torch.randn(4, 3, im_size[0], im_size[1])
    sample_out = model.forward(sample_img)
    score_grid_size = tuple(sample_out['score_offset'].shape[-2:])
    court_grid_size = tuple(sample_out['court'].shape[-2:])
    model = models.AnchorBoxModel([fpn, head], [score_grid_size], [(50, 20)], im_size, angle_scale=10)
    boxes = model.boxes.data.clone()
    offsets = model.offsets.data.clone()

    def anchor_transform(coord: np.ndarray):
        idx = model.get_best(boxes, coord.unsqueeze(0))
        coord_new = (coord - boxes[idx.item()]) / offsets[idx.item()]
        return idx, coord_new

    train_frames, train_corner_labels, train_score_labels = get_dataset(train_videos, score_path, court_path, action_path, frame_path, max_frames=max_frames)
    valid_frames, valid_corner_labels, valid_score_labels = get_dataset(valid_videos, score_path, court_path, action_path, frame_path, max_frames=max_frames)

    train_ds = ImageFilesDatasetKeypoints(train_frames, corners=train_corner_labels,
                                          scoreboard=train_score_labels,
                                          size=im_size, corners_grid_size=court_grid_size,
                                          mean=ds_mean, std=ds_std, anchor_transform=anchor_transform)
    valid_ds = ImageFilesDatasetKeypoints(valid_frames, corners=valid_corner_labels,
                                          scoreboard=valid_score_labels,
                                          size=im_size, corners_grid_size=court_grid_size,
                                          mean=ds_mean, std=ds_std, anchor_transform=anchor_transform)
    batches_per_epoch = len(train_ds) / args.batch_size
    train_ds = tennis_data_to_allen(train_ds)
    valid_ds = tennis_data_to_allen(valid_ds)
    iterator = BasicIterator(args.batch_size)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.initial_lr)

    lr_sched = SlantedTriangular(optimizer, 5, batches_per_epoch)
    trainer = ATrainer(model, optimizer, iterator, train_ds, valid_ds,
                       learning_rate_scheduler=LearningRateWithoutMetricsWrapper(lr_sched),
                       serialization_dir=args.checkpoint_path,
                       num_epochs=args.epochs)
    trainer.train()
