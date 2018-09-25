import numpy as np
import sys
import argparse
from tqdm import tqdm
from pathlib import Path
import logging

import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.decomposition import PCA

from src.data.dataset import ImageDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs-path", type=str)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=394)
    parser.add_argument("--grayscale", type=bool, default=False)
    parser.add_argument("--pca", type=int, default=-1)
    parser.add_argument("--overwrite", type=int, default=0)

    args = parser.parse_args()

    imgs_path = Path(args.imgs_path)
    save_path = Path(args.save_path)

    matches = []
    for match in imgs_path.iterdir():
        match_path = save_path / (match.stem + ".npy")
        if match_path.exists():
            logging.info(f"Skipping match {match.stem}")
            continue
        matches.append(match)
    if len(matches) == 0:
        sys.exit()

    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=False)

    out_size = (args.img_height, args.img_width)
    if args.grayscale:
        img_transforms = transforms.Compose([transforms.Resize(args.img_height),
                                            transforms.CenterCrop(out_size),
                                            transforms.Grayscale(),
                                            transforms.ToTensor()])
    else:
        img_transforms = transforms.Compose([transforms.Resize(args.img_height),
                                             transforms.CenterCrop(out_size),
                                             transforms.ToTensor()])

    resnet = models.resnet34(pretrained=True)
    res_layers = list(resnet.children())[:-1]
    model = nn.Sequential(*res_layers)
    for param in model.parameters():
        param.requires_grad = False
    if args.gpu:
        model = model.to("cuda:0")

    imgs_path = Path(args.imgs_path)
    for match in matches:
        match_path = save_path / (match.stem + ".npy")
        if match_path.exists():
            logging.info(f"Skipping match {match.stem}")
            continue
        ds = ImageDataset(list(sorted(match.iterdir())), transform=img_transforms)
        data_loader = torch.utils.data.DataLoader(ds,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=4)
        features = []
        labels = []
        for im, label in tqdm(data_loader):
            if args.grayscale:
                im = im.repeat(1, 3, 1, 1)
            if args.gpu:
                im = im.to("cuda:0")
            featurized = model.forward(im)
            features.append(featurized)
            labels.append(label.cpu().numpy().ravel())
        features = torch.cat(features)
        labels = np.concatenate(labels)
        feats = features.view(features.shape[0], -1).cpu().numpy()

        if args.pca != -1:
            pca = PCA(n_components=args.pca)
            feats = pca.fit_transform(feats)
        if args.save_path is not None:
            np.save(str(match_path), feats)
