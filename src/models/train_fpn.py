import argparse
from pathlib import Path
import pickle
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
from typing import Iterable, List, Tuple

from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import LearningRateWithoutMetricsWrapper, SlantedTriangular, CosineWithRestarts
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data import Instance
from allennlp.models import Model
from allennlp.data.iterators import BasicIterator
from allennlp.training.optimizers import Optimizer
from allennlp.common import Params

import torch.nn as nn
from torch.utils import data as torchdata
import torchvision.models as torch_models

from src.data.clip import Video
from src.data.dataset import CourtAndScoreTransform, ImagePathsDataset, TennisDatasetReader
from src.vision.transforms import *
import src.models.models as models
import src.utils as utils
from src.learning_rate_schedule import CyclicSlantedTriangular

logger = logging.getLogger(__name__)


# def get_dataset(videos: List[Video],
#                 score_path: Path,
#                 court_path: Path,
#                 action_path: Path,
#                 frame_path: Path,
#                 max_frames: int=None) -> Tuple[List[Path], np.ndarray, np.ndarray]:
#     """
#     Given a list of videos, get the court and corner labels for each.
#     """
#     frames = []
#     score_labels = []
#     corner_labels = []
#     for video in videos:
#         score_name = score_path / (video.name + ".pkl")
#         court_name = court_path / (video.name + ".pkl")
#         if not score_name.exists() or not court_name.exists():
#             continue
#         with open(score_name, 'rb') as f:
#             scores = pickle.load(f)
#         with open(court_name, 'rb') as f:
#             corners = pickle.load(f)
#         action_mask = np.load(action_path / (video.name + ".npy"))
#         cnt = 0
#         for (fname, box), (fname, coords), is_action in zip(scores, corners, action_mask):
#             if is_action and box != [0, 0, 0, 0] and np.any(coords):
#                 # only use frames that are action and have valid labels
#                 frames.append(frame_path / video.name / fname)
#                 corner_labels.append([float(x) for x in coords])
#                 score_labels.append(box)
#                 cnt += 1
#             if max_frames is not None and cnt > max_frames:
#                 break
#     score_labels = np.array(score_labels)
#     corner_labels = np.array(corner_labels).reshape(-1, 4, 2)
#     return frames, corner_labels, score_labels
#
#
# def tennis_data_to_allen(ds: torchdata.Dataset) -> Iterable[Instance]:
#     instances = []
#     # for (img, court, score) in itertools.islice(ds, 100):
#     for (img, court, score) in ds:
#         fields = {
#             'img': ArrayField(img),
#             'court': ArrayField(court),
#             'score': ArrayField(score),
#         }
#         instances.append(Instance(fields))
#     return instances

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
    parser.add_argument('--dataset-path', type=str, default="")
    parser.add_argument('--param-file', type=str, default="")
    parser.add_argument('--action-path', type=str, default="")
    args = parser.parse_args()

    np.random.seed(args.seed)

    im_size = (args.img_height, args.img_width)
    use_gpu = args.gpu or torch.cuda.is_available()
    train_device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    logging.debug(f"Training on device {train_device}.")

    params = Params.from_file(args.param_file)
    frame_path = Path(args.frame_path)
    score_path = Path(args.score_path)
    court_path = Path(args.court_path)
    action_path = Path(args.action_path)
    dataset_path = Path(args.dataset_path)
    # video_paths = []
    # for v in list(frame_path.iterdir()):
    #     score_name = score_path / (v.name + ".pkl")
    #     court_name = court_path / (v.name + ".pkl")
    #     if score_name.exists() or court_name.exists():
    #         video_paths.append(v)

    # np.random.shuffle(video_paths)
    # train_videos = [Video.from_dir(v) for v in video_paths[args.n_valid:]]
    # valid_videos = [Video.from_dir(v) for v in video_paths[:args.n_valid]]
    # logger.info(f"Train videos: {[v.name for v in train_videos]}")
    # logger.info(f"Validation videos: {[v.name for v in valid_videos]}")

    im_size = (args.img_height, args.img_width)
    batch_size = args.batch_size
    # max_frames = 300

    # res = torch_models.resnet34(pretrained=True)
    # utils.freeze(res.parameters())
    #
    # C1 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
    # fpn = models.FPN(C1, res.layer1, res.layer2, res.layer3, res.layer4)
    # fpn = models.FPN()
    # fpn = Model.from_params(params.pop("model"))
    # grid_size = max(fpn.get_grid_sizes(im_size[0], im_size[1]))
    # head = models.CourtScoreHead(in_channels=128, out_channels=6)
    # model = models.AnchorBoxModel([fpn, head], [grid_size], [(50, 20)], im_size, angle_scale=10)
    model = Model.from_params(params.pop("model"))

    # train_frames, train_corner_labels, train_score_labels = get_dataset(train_videos, score_path, court_path, action_path, frame_path, max_frames=max_frames)
    # valid_frames, valid_corner_labels, valid_score_labels = get_dataset(valid_videos, score_path, court_path, action_path, frame_path, max_frames=max_frames)


    # if args.img_mean is None:
    #     logger.info("Begin mean/std computation")
    #     images = utils.read_images(train_frames)
    #     ds_mean, ds_std = utils.compute_mean_std(images, nsample=1000)
    #     ds_mean = [float(x) for x in list(ds_mean)]
    #     ds_std = [float(x) for x in list(ds_std)]
    #     logger.info(f"Mean: {ds_mean}")
    #     logger.info(f"Std: {ds_std}")
    #     logger.info("End mean/std computation.")
    # else:
    #     ds_mean = [float(x) for x in args.img_mean.split(",")]
    #     ds_std = [float(x) for x in args.img_std.split(",")]

    train_reader = DatasetReader.from_params(params.pop("train_reader"))
    valid_reader = DatasetReader.from_params(params.pop("valid_reader"))
    # train_transform = CourtAndScoreTransform(mean=ds_mean, std=ds_std, size=im_size,
    #                                          corners_grid_size=grid_size)
    # valid_transform = CourtAndScoreTransform(mean=ds_mean, std=ds_std, size=im_size,
    #                                          corners_grid_size=grid_size,
    #                                          train=False)
    # train_reader = TennisDatasetReader(train_transform)
    # valid_reader = TennisDatasetReader(valid_transform)

    # train_ds = ImagePathsDataset(train_frames, train_corner_labels, train_score_labels,
    #                              transform=train_transform)
    # valid_ds = ImagePathsDataset(valid_frames, valid_corner_labels, valid_score_labels,
    #                              transform=valid_transform)

    batches_per_epoch = int(100 / args.batch_size)  # TODO
    logger.info(f"Batches per epoch: {batches_per_epoch}")
    # train_instances = tennis_data_to_allen(train_ds)
    # valid_instances = tennis_data_to_allen(valid_ds)
    train_instances = train_reader.read(dataset_path / Path("train").with_suffix(".json"))
    valid_instances = train_reader.read(dataset_path / Path("test").with_suffix(".json"))
    iterator = BasicIterator(args.batch_size)

    fpn = model.model[0]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    backbone_params = []
    for j, backbone_section in enumerate([fpn.C1, fpn.C2, fpn.C3, fpn.C4, fpn.C5]):
        for i, param in enumerate(backbone_section.parameters()):
            backbone_params.append((f"backbone_{j}_{i}", param))
    model_group = {'name': 'fpn', 'params': trainable_params, 'initial_lr': args.initial_lr}
    optimizer = Optimizer.from_params(backbone_params + [(f"fpn{i}", trainable_params[i])
                                                         for i in range(len(trainable_params))],
                                      params.pop("optim"))

    lr_sched = SlantedTriangular(optimizer, args.epochs, batches_per_epoch,
                                 ratio=100, cut_frac=0.5, gradual_unfreezing=True)

    # logger.warning(fpn.get_grid_sizes(224, 224))
    # for p in model.model[0].parameters():
    #     logger.warning(p.requires_grad)
    trainer = Trainer(model, optimizer, iterator, train_instances, valid_instances,
                      learning_rate_scheduler=LearningRateWithoutMetricsWrapper(lr_sched),
                       serialization_dir=args.checkpoint_path,
                       num_epochs=args.epochs,
                       summary_interval=args.log_interval,
                       should_log_learning_rate=True,
                       cuda_device=0 if args.gpu else -1,
                      histogram_interval=args.log_interval * 10,
                      num_serialized_models_to_keep=2)
    # def myhook(module_, inp, outp):
    #     trainer._tensorboard.add_train_scalar("test", outp['loss'], lr_sched.get_lr()[0] * 1000000)
    # model.register_forward_hook(myhook)
    trainer.train()

