import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from logging.config import fileConfig

import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.decomposition import PCA

from deeptennis.data.dataset import ImageFilesDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", type=str)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=False)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=394)
    parser.add_argument("--grayscale", type=bool, default=False)
    parser.add_argument("--pca", type=int, default=-1)
    parser.add_argument("--overwrite", type=int, default=0)

    fileConfig('logging_config.ini')

    args = parser.parse_args()

    img_path = Path(args.img_path)
    save_path = Path(args.save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir()

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

    ds = ImageFilesDataset(list(sorted(img_path.iterdir())), transform=img_transforms)
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
        np.save(str(save_path), feats)
