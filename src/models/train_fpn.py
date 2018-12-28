import argparse
from pathlib import Path
import pickle
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
from typing import Iterable, List, Tuple

from allennlp.training import Trainer as ATrainer
from allennlp.training.learning_rate_schedulers import LearningRateWithoutMetricsWrapper, SlantedTriangular
from allennlp.data.fields import ArrayField
from allennlp.data import Instance
from allennlp.data.iterators import BasicIterator

import torch.nn as nn
from torch.utils import data as torchdata
import torchvision.models as torch_models

from src.data.clip import Video
from src.data.dataset import CourtAndScoreTransform, ImagePathsDataset
from src.vision.transforms import *
import src.models.models as models
import src.utils as utils


def get_dataset(videos: List[Video],
                score_path: Path,
                court_path: Path,
                action_path: Path,
                frame_path: Path,
                max_frames: int=None) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    """
    Given a list of videos, get the court and corner labels for each.
    """
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
    instances = []
    for (img, court, score) in ds:
        fields = {
            'img': ArrayField(img),
            'court': ArrayField(court),
            'score': ArrayField(score),
        }
        instances.append(Instance(fields))
    return instances

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
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--frame-path', type=str, default="")
    parser.add_argument('--court-path', type=str, default="")
    parser.add_argument('--score-path', type=str, default="")
    parser.add_argument('--action-path', type=str, default="")
    args = parser.parse_args()

    np.random.seed(args.seed)

    im_size = (args.img_height, args.img_width)
    use_gpu = args.gpu or torch.cuda.is_available()
    train_device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    logging.debug(f"Training on device {train_device}.")

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
    fpn = models.FPN(C1, res.layer1, res.layer2, res.layer3, res.layer4)
    grid_size = max(fpn.get_grid_sizes(im_size[0], im_size[1]))
    head = models.CourtScoreHead(in_channels=128, out_channels=6)
    model = models.AnchorBoxModel([fpn, head], [grid_size], [(50, 20)], im_size, angle_scale=10)
    boxes = model.boxes.data.clone()
    offsets = model.offsets.data.clone()

    train_frames, train_corner_labels, train_score_labels = get_dataset(train_videos, score_path, court_path, action_path, frame_path, max_frames=max_frames)
    valid_frames, valid_corner_labels, valid_score_labels = get_dataset(valid_videos, score_path, court_path, action_path, frame_path, max_frames=max_frames)

    if args.img_mean is None:
        logging.debug("Begin mean/std computation")
        images = utils.read_images(train_frames)
        ds_mean, ds_std = utils.compute_mean_std(images, nsample=1000)
        ds_mean = [float(x) for x in list(ds_mean)]
        ds_std = [float(x) for x in list(ds_std)]
        logging.debug("End mean/std computation.")
    else:
        ds_mean = [float(x) for x in args.img_mean.split(",")]
        ds_std = [float(x) for x in args.img_std.split(",")]

    train_transform = CourtAndScoreTransform(mean=ds_mean, std=ds_std, size=im_size,
                                             corners_grid_size=grid_size)
    valid_transform = CourtAndScoreTransform(mean=ds_mean, std=ds_std, size=im_size,
                                             corners_grid_size=grid_size,
                                             train=False)
    train_ds = ImagePathsDataset(train_frames, train_corner_labels, train_score_labels,
                                 transform=train_transform)
    valid_ds = ImagePathsDataset(valid_frames, valid_corner_labels, valid_score_labels,
                                 transform=valid_transform)
    batches_per_epoch = len(train_ds) / args.batch_size
    logging.debug(f"Batches per epoch: {batches_per_epoch}")
    train_instances = tennis_data_to_allen(train_ds)
    valid_instances = tennis_data_to_allen(valid_ds)
    iterator = BasicIterator(args.batch_size)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.initial_lr)

    lr_sched = SlantedTriangular(optimizer, args.epochs, int(batches_per_epoch), cut_frac=0.03)
    trainer = ATrainer(model, optimizer, iterator, train_instances, valid_instances,
                       learning_rate_scheduler=LearningRateWithoutMetricsWrapper(lr_sched),
                       serialization_dir=args.checkpoint_path,
                       num_epochs=args.epochs,
                       summary_interval=args.log_interval,
                       should_log_learning_rate=True,
                       cuda_device=0 if args.gpu else -1)
    trainer.train()

