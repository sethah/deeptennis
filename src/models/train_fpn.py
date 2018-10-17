import numpy as np
import argparse
from pathlib import Path
import pickle
import logging
from logging.config import fileConfig

import torch.nn as nn
import torch.utils
import torchvision.models as torch_models
from torch.utils.data.dataloader import default_collate

from src.data.clip import Video
from src.data.dataset import ImageFilesDatasetKeypoints
from src.vision.transforms import *
from src.models.loss import SSDLoss, CourtScoreLoss
import src.models.models as models
import src.utils as utils


def valid(epoch, model, loader, optimizer, criterion, device):
    total_loss = 0.
    n = len(loader.sampler)
    n = 0
    model.eval()
    for inp, targ in loader:
        inp = inp.to(device)
        optimizer.zero_grad()
        preds = model.forward(inp)
        loss = criterion(preds, targ)
        total_loss += loss.item() * inp.shape[0]
        n += inp.shape[0]
    total_loss /= n
    logging.debug('Train Epoch: {} Validation Loss: {:.6f}'.format(epoch, total_loss))
    return total_loss


def train(epoch, model, loader, optimizer, criterion, device=torch.device("cpu"), log_interval=100):
    model.train()
    total_loss = 0.
    n = 0.
    logging.debug(f"Begin training epoch: {epoch}")
    for batch_idx, (inp, targ) in enumerate(loader):
        inp = inp.to(device)
        optimizer.zero_grad()
        preds = model.forward(inp)
        loss = criterion(preds, targ)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item() * inp.shape[0]
        n += inp.shape[0]
        if (batch_idx + 1) % log_interval == 0:
            samples_processed = batch_idx * inp.shape[0]
            total_samples = len(loader.sampler)
            logging.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, samples_processed, total_samples,
                100. * batch_idx / len(loader), total_loss / n))
            total_loss = 0.
            n = 0.


def get_anchors(grid_size, im_size, box_size, angle_offset=0):
    gw, gh = grid_size
    iw, ih = im_size
    bw, bh = box_size
    cell_width = iw / gw
    cell_height = ih / gh
    cxs = np.arange(gh) * cell_width + cell_width / 2
    cys = np.arange(gw) * cell_height + cell_height / 2
    coords = itertools.product(cxs, cys)
    box_centers = np.array(list([(y, x) for x, y in coords]))
    boxes = np.concatenate([box_centers,
                            np.ones((box_centers.shape[0], 1)) * bw,
                            np.ones((box_centers.shape[0], 1)) * bh,
                            np.ones((box_centers.shape[0], 1)) * angle_offset],
                           axis=1).astype(np.float32)
    return torch.from_numpy(boxes)


def get_dataset(videos, score_path, court_path, action_path, frame_path, max_frames=None):
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
    parser.add_argument('--checkpoint-file', type=str, default="")
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
    if args.freeze_backbone:
        for p in res.parameters():
            p.requires_grad = False

    C1 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
    C2 = res.layer1
    C3 = res.layer2
    C4 = res.layer3
    C5 = res.layer4
    model = nn.Sequential(models.FPN(C1, C2, C3, C4, C5), models.CourtScoreHead(128))

    sample_img = torch.randn(4, 3, im_size[0], im_size[1])
    sample_out = model.forward(sample_img)
    score_grid_size = tuple(sample_out[1].shape[-2:])
    court_grid_size = tuple(sample_out[0].shape[-2:])

    box_w, box_h = 50, 20
    angle_scale = 10
    boxes = get_anchors(score_grid_size, im_size, (box_w, box_h), angle_offset=0)

    scale_box = torch.tensor([im_size[0] / score_grid_size[0] / 2,
                              im_size[1] / score_grid_size[1] / 2,
                              box_w / 2, box_h / 2, angle_scale], device=train_device)


    train_frames, train_corner_labels, train_score_labels = get_dataset(train_videos, score_path, court_path, action_path, frame_path, max_frames=max_frames)
    valid_frames, valid_corner_labels, valid_score_labels = get_dataset(valid_videos, score_path, court_path, action_path, frame_path, max_frames=max_frames)

    def collate_fn(batch):
        tensors = default_collate(batch)
        return [tensors[0], (tensors[1], tensors[2])]

    train_ds = ImageFilesDatasetKeypoints(train_frames, corners=train_corner_labels,
                                          scoreboard=train_score_labels,
                                          size=im_size, corners_grid_size=court_grid_size,
                                          mean=ds_mean, std=ds_std)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=args.num_workers, collate_fn=collate_fn)
    valid_ds = ImageFilesDatasetKeypoints(valid_frames, corners=valid_corner_labels,
                                          scoreboard=valid_score_labels,
                                          size=im_size, corners_grid_size=court_grid_size,
                                          mean=ds_mean, std=ds_std)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                                               num_workers=args.num_workers, collate_fn=collate_fn)

    court_crit = nn.MSELoss().to(train_device)
    class_crit = nn.BCEWithLogitsLoss().to(train_device)
    reg_crit = nn.L1Loss().to(train_device)
    ssd_crit = SSDLoss(boxes, class_crit, reg_crit, scale_box).to(train_device)
    criterion = CourtScoreLoss(court_crit, ssd_crit, court_weight=50., score_weight=1.)
    model = model.to(train_device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.initial_lr)

    lr_milestones = [int(x) for x in args.lr_milestones.split(",")]
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones,
                                                    gamma=args.lr_gamma)
    logging.debug(f"Training {len(trainable_params)} parameters")

    best_loss = 1000000.
    if args.restore:
        loaded = utils.load_checkpoint(args.checkpoint_path, best=args.restore == 'best')
        model.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['optimizer'])
        lr_sched.load_state_dict(loaded['scheduler'])
        best_loss = loaded.get('best_loss', best_loss)

    for epoch in range(1, args.epochs + 1):
        lr_sched.step()
        logging.debug(f"Learning rate update: {lr_sched.get_lr()}")
        train(epoch, model, train_loader, optimizer, criterion, device=train_device,
              log_interval=args.log_interval)
        if epoch % args.validate_every == 0:
            loss = valid(epoch, model, valid_loader, optimizer, criterion, device=train_device)
            if args.checkpoint_path:
                save_dict = {'model': model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'scheduler': lr_sched.state_dict(),
                             'best_loss': best_loss}
                utils.save_checkpoint(args.checkpoint_path, save_dict)
                if loss < best_loss:
                    best_loss = loss
                    utils.save_checkpoint(args.checkpoint_path, save_dict, best=True)
