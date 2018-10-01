import numpy as np
import argparse
from pathlib import Path
import logging
from logging.config import fileConfig

import torch.nn as nn
import torch.utils
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

from src.data.clip import Clip, Video
from src.data.dataset import ImageDatasetBox, GridDataset
from src.vision.transforms import *
import src.utils as utils

def valid(epoch, model, criterion, loader, device):
    total_loss = 0.
    correct = 0.
    n = len(loader.sampler)
    model.eval()
    for inp, targ in loader:
        inp, targ = inp.to(device), targ.to(device)
        outs = model.forward(inp)
        loss = 0.
        for out in outs:
            loss += criterion(out, targ)
        total_loss += loss.item()
    total_loss /= n
    print('Train Epoch: {} Validation Loss: {:.6f}'.format(
        epoch, loss))
    return loss

def train(epoch, model, loader, optimizer, criterion, device=torch.device("cpu"), log_interval=100, limit_batches=None):
    model.train()
    total_loss = 0.
    n = 0.
    for batch_idx, (inp, targ) in enumerate(loader):
        inp, targ = inp.to(device), targ.to(device)
        optimizer.zero_grad()
        outs = model.forward(inp)
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, samples_processed, total_samples,
                100. * batch_idx / len(loader), total_loss / n))
            total_loss = 0.
            n = 0.
        if limit_batches is not None and (batch_idx + 1) >= limit_batches:
            break

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
    parser.add_argument('--model-name', type=str, default="")
    parser.add_argument('--restore', type=str, default="", help="{'best', 'latest'}")
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--checkpoint-file', type=str, default="")
    args = parser.parse_args()

    fileConfig('logging_config.ini')

    im_size = (int(x) for x in args.im_size.split(","))
    train_device = torch.device("cuda:0") if args.gpu else torch.device("cpu")

    frame_path = Path(args.frame_path)
    video_frames = list(frame_path.iterdir())
    holdout_match = video_frames[np.random.randint(len(video_frames))]
    logging.debug(f"Holding out match {holdout_match.stem}")
    clip_path = Path(args.clip_path)


    def get_dataset(videos, clip_path):
        bboxes = []
        frames = []
        for video in videos:
            clips = Clip.from_csv(clip_path / (video.name + ".csv"), video)
            frames += [f for c in clips for f in c.frames]
            bboxes += [b.reshape(4, 2) for c in clips for b in c.bboxes]
        return ImageDatasetBox(frames, bboxes)
    train_videos = [Video(v) for v in video_frames if v != holdout_match]
    valid_videos = [Video(v) for v in video_frames if v == holdout_match]

    simple_transforms = Compose([Resize(im_size), WrapTransform(transforms.ToTensor())])
    train_ds = get_dataset(train_videos, clip_path).with_transfrorms(simple_transforms)
    valid_ds = get_dataset(train_videos, clip_path).with_transfrorms(simple_transforms)
    loader = torch.utils.data.DataLoader(data.ConcatDataset([train_ds, valid_ds]),
                                         batch_size=args.batch_size, shuffle=False)
    ds_mean, ds_std = src.utils.compute_mean_std(data.ConcatDataset([train_ds, valid_ds]))

    img_transforms = img_transforms = Compose([
        RandomRotation((-10, 10), expand=True),
        Resize((int(im_size[0] * 1.), int(im_size[1] * 1.))),
        RandomHorizontalFlip(0.5),
        WrapTransform(
            transforms.ColorJitter(brightness=0.1, hue=0.1, contrast=0.5, saturation=0.5)),
        WrapTransform(transforms.ToTensor()),
        WrapTransform(transforms.Normalize(ds_mean.tolist(), ds_std.tolist()))])

    if args.model == 'resnet34':
        resnet = models.resnet34(pretrained=True)
        layers = list(resnet.children())
        head = CornerHead()
        resnet_chopped = nn.Sequential(*layers[:6])
        model = nn.Sequential('extractor', resnet_chopped)
    elif args.model == 'simple':
        pass
    else:
        raise ValueError(f"Model {args.model} not yet supported.")

    model.add_module('classifier', head)
    utils.freeze(model.extractor.parameters())
    model = model.to(train_device)

    fake_img = torch.randn(4, 3, im_size[0], im_size[1]).to(train_device)
    out = model.forward(fake_img)
    grid_size = tuple(out.shape[-2:])

    train_ds = GridDataset(train_ds.with_transforms(img_transforms), grid_size=grid_size)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=4)
    valid_ds = GridDataset(valid_ds.with_transforms(img_transforms), grid_size=grid_size)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                                               num_workers=4)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.BCEWithLogitsLoss().to(train_device)
    optimizer = torch.optim.Adam(trainable_params, lr=args.initital_lr)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    logging.debug(f"Training {len(trainable_params)} parameters")

    best_loss = 1000000.
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, device=train_device,
              log_interval=args.log_interval)
        loss, acc = valid(epoch, model, valid_loader, device=train_device)
        if args.checkpoint_path:
            save_dict = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
            utils.save_checkpoint(args.checkpoint_path, save_dict, model_name=args.model_name)
            if loss < best_loss:
                best_loss = loss
                utils.save_checkpoint(args.checkpoint_path, save_dict,
                                      model_name=args.model_name, best=True)
