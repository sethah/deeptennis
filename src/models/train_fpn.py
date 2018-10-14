import argparse
from pathlib import Path
import pickle
import logging
from logging.config import fileConfig

import torch.nn as nn
import torch.utils
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.models as torch_models
import torchvision.transforms as transforms

from src.data.clip import Video
from src.data.dataset import get_bounding_box_dataset, ImageFilesDatasetKeypoints
from src.vision.transforms import *
from src.models.loss import SSDLoss
import src.models.models as models
import src.utils as utils


def valid(epoch, model, loader, optimizer, criterion, device):
    total_loss = 0.
    n = len(loader.sampler)
    model.eval()
    # TODO: fix the device movement
    for inp, targ in loader:
        inp = inp.to(device)
        optimizer.zero_grad()
        preds = model.forward(inp)
        loss = criterion(preds, targ)
        total_loss += loss.item()
    total_loss /= n
    logging.debug('Train Epoch: {} Validation Loss: {:.6f}'.format(epoch, loss))
    return loss


def train(epoch, model, loader, optimizer, criterion, device=torch.device("cpu"), log_interval=100):
    model.train()
    total_loss = 0.
    n = 0.
    for batch_idx, (inp, targ) in enumerate(loader):
        inp = inp.to(device)
        optimizer.zero_grad()
        preds = model.forward(inp)
        loss = criterion(preds, targ)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
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


def get_dataset(videos, score_path, court_path, max_frames=None):
    frames = []
    score_labels = []
    corner_labels = []
    for video in videos:
        with open(score_path / (video.name + ".pkl"), 'rb') as f:
            scores = pickle.load(f)
        with open(court_path / (video.name + ".pkl"), 'rb') as f:
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
    parser.add_argument("--model-save-path", type=str, default=None)
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--gpu", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n-valid", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=30)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument("--img-mean", type=str, default=None)
    parser.add_argument("--img-std", type=str, default=None)
    parser.add_argument("--embed-size", type=int, default=32)
    parser.add_argument("--initial-lr", type=float, default=0.001)
    parser.add_argument("--lr-gamma", type=float, default=0.95)
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--restore', type=str, default="", help="{'best', 'latest'}")
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--frame-path', type=str, default="")
    parser.add_argument('--clip-path', type=str, default="")
    parser.add_argument('--checkpoint-file', type=str, default="")
    args = parser.parse_args()

    fileConfig('logging_config.ini')
    np.random.seed(args.seed)

    im_size = (args.img_height, args.img_width)
    use_gpu = args.gpu or torch.cuda.is_available()
    train_device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

    frame_path = Path(args.frame_path)
    video_paths = list(frame_path.iterdir())
    score_path = Path(args.score_path)
    court_path = Path(args.court_path)
    action_path = Path(args.action_path)

    np.random.shuffle(video_paths)
    train_videos = [Video.from_dir(v) for v in video_paths[args.n_valid:]]
    valid_videos = [Video.from_dir(v) for v in video_paths[:args.n_valid]]
    logging.debug(f"Holding out match {[v.name for v in valid_videos]}")

    im_size = (224, 224)
    batch_size = 8
    corners_grid_size = (54, 54)
    max_frames = 300

    train_frames, train_corner_labels, train_score_labels = get_dataset(train_videos, score_path, court_path, max_frames=max_frames)
    valid_frames, valid_corner_labels, valid_score_labels = get_dataset(valid_videos, score_path, court_path, max_frames=None)
    train_ds = ImageFilesDatasetKeypoints(train_frames, corners=train_corner_labels,
                                          scoreboard=train_score_labels,
                                          size=im_size, corners_grid_size=corners_grid_size)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                               num_workers=4)
    valid_ds = ImageFilesDatasetKeypoints(valid_frames, corners=valid_corner_labels,
                                          scoreboard=valid_score_labels,
                                          size=im_size, corners_grid_size=corners_grid_size)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                                               num_workers=4)


    class ModelHead(nn.Module):

        def __init__(self, in_channels, out_channels=6):
            super(ModelHead, self).__init__()
            self.conv1_score = models.StdConv(in_channels, in_channels)
            self.out_conv_score = nn.Conv2d(in_channels, out_channels, 3)
            self.conv1_court = models.double_conv(in_channels, 64)
            self.conv2_court = models.double_conv(in_channels, 64)
            self.conv3_court = models.double_conv(in_channels, 64)
            self.conv4_court = models.StdConv(64 * 3, 64)
            self.out_conv_court = nn.Conv2d(64, 4, 3)

        def forward(self, x):
            court1 = self.conv1_court(x[0])
            court2 = self.conv2_court(x[1])
            court3 = self.conv3_court(x[2])
            court = torch.cat([F.interpolate(court1, scale_factor=4),
                               F.interpolate(court2, scale_factor=2),
                               court3], dim=1)
            return [self.out_conv_court(self.conv4_court(court)),
                    self.out_conv_score(self.conv1_score(x[1]))]

    res = torch_models.resnet34(pretrained=True)
    for p in res.parameters():
        p.requires_grad = False

    C1 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
    C2 = res.layer1
    C3 = res.layer2
    C4 = res.layer3
    model = nn.Sequential(models.FPN(C1, C2, C3, C4), ModelHead(128))

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

    class CombinedLoss(nn.modules.loss._Loss):

        def __init__(self, court_criterion, score_criterion, court_weight=1., score_weight=1.,
                     size_average=None, reduce=None, reduction='elementwise_mean'):
            super(CombinedLoss, self).__init__(size_average, reduce, reduction)
            self.court_criterion = court_criterion
            self.score_criterion = score_criterion
            self.court_weight = court_weight
            self.score_weight = score_weight

        def forward(self, preds, targ):
            score_loss = self.score__criterion(preds[1], targ[1].to(preds[1].device))
            court_loss = self.court_criterion(preds[0], targ[0].to(preds[0].device))
            return score_loss * self.score_weight + court_loss * self.court_weight

    court_crit = nn.MSELoss(reduction='sum').to(train_device)
    ssd_crit = SSDLoss(reduction='sum').to(train_device)
    criterion = CombinedLoss(court_crit, ssd_crit, court_weight=1., score_weight=1.)
    model = model.to(train_device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.initial_lr)

    for epoch in range(args.n_epochs):
        train(epoch, model, train_loader, optimizer, criterion,
              device=train_device, log_interval=20)
        valid(epoch, model, criterion, valid_loader, train_device)
