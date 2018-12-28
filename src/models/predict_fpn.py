import argparse
from pathlib import Path
import logging
from logging.config import fileConfig
from typing import Dict, List, Iterable, Tuple

import torch.nn as nn
import torch.utils
import torchvision.models as torch_models
import torchvision.transforms as tvt
from torch.utils.data.dataloader import default_collate

from allennlp.data.fields import ArrayField
from allennlp.data import Instance

from src.data.clip import Video
from src.data.dataset import ImageFilesDataset
from src.vision.transforms import *
import src.models.models as models
import src.utils as utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument("--img-mean", type=str, default=None)
    parser.add_argument("--img-std", type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--frame-path', type=str, default="")
    parser.add_argument('--action-path', type=str, default="")
    parser.add_argument('--save-path', type=str, default="")
    args = parser.parse_args()

    fileConfig('logging_config.ini')
    np.random.seed(args.seed)

    im_size = (args.img_height, args.img_width)
    use_gpu = args.gpu or torch.cuda.is_available()
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    logging.debug(f"Predicting on device {device}.")

    ds_mean = [float(x) for x in args.img_mean.split(",")]
    ds_std = [float(x) for x in args.img_std.split(",")]

    frame_path = Path(args.frame_path)
    action_path = Path(args.action_path)

    batch_size = args.batch_size

    res = torch_models.resnet34(pretrained=True)
    C1 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
    fpn = models.FPN(C1, res.layer1, res.layer2, res.layer3, res.layer4)
    grid_size = max(fpn.get_grid_sizes(im_size[0], im_size[1]))
    head = models.CourtScoreHead(in_channels=128, out_channels=6)
    model = models.AnchorBoxModel([fpn, head], [grid_size], [(1, 1)], im_size, angle_scale=10)

    loaded = torch.load(args.checkpoint_path)
    model.load_state_dict(loaded)
    model.eval()
    model = model.to(device)

    action_mask = np.load(action_path)
    video = Video.from_dir(frame_path)
    frames = [f for i, f in enumerate(video.frames) if action_mask[i]]

    tfms = tvt.Compose([tvt.Resize(im_size), tvt.ToTensor(), tvt.Normalize(ds_mean, ds_std)])
    ds = ImageFilesDataset(frames, transform=tfms)
    instances = [Instance({'img': ArrayField(im.numpy())}) for im in ds]

    im_size_full = cv2.imread(str(video.frames[0])).shape[:2]
    out: List[Dict[str, np.ndarray]] = model.forward_on_instances(instances)
    out_score = torch.from_numpy(np.stack([o['score'] for o in out]))
    out_court = torch.from_numpy(np.stack([o['court'] for o in out]))
    score_coords = model.box_to_vertices(out_score, im_size_full[1], im_size_full[0])
    court_coords = model.heatmaps_to_vertices(out_court, im_size_full[1], im_size_full[0])

    out_frames = []
    j = 0
    for i, frame in enumerate(video.frames):
        im = cv2.imread(str(frame))
        if not action_mask[i]:
            out_frames.append(im)
        else:
            im = cv2.polylines(im, [score_coords[j].numpy().astype(np.int32)], True, (0, 255, 255), 4)
            im = cv2.polylines(im, [court_coords[j].numpy().astype(np.int32)], True, (0, 255, 255), 4)
            out_frames.append(im)
            j += 1

    out_path = Path(args.save_path)
    out_path.mkdir(exist_ok=True, parents=True)
    for i, im in enumerate(out_frames):
        full_path = str(out_path / ("%05d.jpg" % i))
        cv2.imwrite(full_path, im)



