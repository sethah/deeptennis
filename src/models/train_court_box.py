import argparse
from pathlib import Path
import logging
from logging.config import fileConfig

import torch.nn as nn
import torch.utils
import torch.utils.data as data
import torchvision.transforms as transforms

from src.data.clip import Clip, Video
from src.data.dataset import GridDataset, get_bounding_box_dataset
from src.vision.transforms import *
import src.models.models as models
import src.utils as utils


def valid(epoch, model, criterion, loader, device):
    total_loss = 0.
    n = 0
    model.eval()
    for inp, targ in loader:
        inp, targ = inp.to(device), targ.to(device)
        outs = model.forward(inp)
        if not isinstance(outs, list):
            outs = [outs]
        loss = 0.
        for out in outs:
            loss += criterion(out, targ)
        total_loss += loss.item()
        n += inp.shape[0]
    total_loss /= n
    logging.debug('Train Epoch: {} Validation Loss: {:.6f}'.format(
        epoch, loss))
    return loss


def train(epoch, model, loader, optimizer, criterion, device=torch.device("cpu"), log_interval=100):
    model.train()
    total_loss = 0.
    n = 0.
    for batch_idx, (inp, targ) in enumerate(loader):
        inp, targ = inp.to(device), targ.to(device)
        optimizer.zero_grad()
        outs = model.forward(inp)
        if not isinstance(outs, list):
            outs = [outs]
        loss = 0.
        for out in outs:
            loss += criterion(out, targ)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs-path", type=str)
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
    clip_path = Path(args.clip_path)

    np.random.shuffle(video_paths)
    train_videos = [Video(v) for v in video_paths[args.n_valid:]]
    valid_videos = [Video(v) for v in video_paths[:args.n_valid]]
    logging.debug(f"Holding out match {[v.name for v in valid_videos]}")

    train_ds = get_bounding_box_dataset(train_videos, clip_path, filter_valid=True, max_frames=300)
    valid_ds = get_bounding_box_dataset(valid_videos, clip_path, max_frames=300)
    if args.img_mean is None:
        ds_mean, ds_std = train_ds.statistics()
        ds_mean, ds_std = ds_mean.numpy().tolist(), ds_std.numpy().tolist()
        logging.debug(f"Mean: {ds_mean}, std: {ds_std}")
    else:
        ds_mean = [float(x) for x in args.img_mean.split(",")]
        ds_std = [float(x) for x in args.img_std.split(",")]

    if args.model == 'simple':
        model = models.SimpleConvNet(keypoints=4, channels=3)
    elif args.model == 'pose':
        model = models.SimplePose(keypoints=4, channels=3, nout=64, layers=2)
    elif args.model == 'pose-unet':
        model = models.PoseUNet(keypoints=4, channels=3)
    else:
        raise ValueError(f"Model {args.model} not yet supported.")

    fake_img = torch.randn(4, 3, im_size[0], im_size[1])
    outs = model.predict(fake_img)
    channels = outs.shape[1]
    grid_size = tuple(outs.shape[2:])
    logging.debug(f"Grid size: {grid_size}, channels: {channels}")
    model = model.to(train_device)

    img_transforms = Compose([
        RandomRotation((-10, 10), expand=True),
        Resize((int(im_size[0] * 1.), int(im_size[1] * 1.))),
        RandomHorizontalFlip(0.5),
        WrapTransform(
            transforms.ColorJitter(brightness=0.1, hue=0.1, contrast=0.5, saturation=0.5)),
        WrapTransform(transforms.ToTensor()),
        WrapTransform(transforms.Normalize(ds_mean, ds_std)),
        BoxToGrid(grid_size=grid_size)])

    train_ds = train_ds.with_transforms(img_transforms)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=4)
    valid_ds = valid_ds.with_transforms(img_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                                               num_workers=4)
    logging.debug(f"Training on {len(train_ds)} images")
    logging.debug(f"Validating on {len(valid_ds)} images")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.BCEWithLogitsLoss(reduction='sum').to(train_device)
    optimizer = torch.optim.Adam(trainable_params, lr=args.initial_lr)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
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
        loss = valid(epoch, model, criterion, valid_loader, device=train_device)
        if args.checkpoint_path:
            save_dict = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': lr_sched.state_dict(),
                         'best_loss': best_loss}
            utils.save_checkpoint(args.checkpoint_path, save_dict)
            if loss < best_loss:
                best_loss = loss
                utils.save_checkpoint(args.checkpoint_path, save_dict, best=True)
