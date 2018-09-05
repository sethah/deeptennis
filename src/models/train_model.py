import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm



class MyModel(nn.Module):

    def __init__(self, num_classes=20):
        super(MyModel, self).__init__()
        resnet = models.resnet34(pretrained=True)
        res_layers = list(resnet.children())[:-1]
        self.model = nn.Sequential(*res_layers)
        for param in self.model.parameters():
            param.requires_grad = False
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model.forward(x)
        out = self.head(x.view(x.shape[0], -1))
        return out

"""
python src/models/train_model.py \
--imgs-path ./data/processed/
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs-path", type=str)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--gpu", type=bool, default=False)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    img_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    tennis_data = torchvision.datasets.ImageFolder(args.imgs_path, transform=img_transforms)
    data_loader = torch.utils.data.DataLoader(tennis_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=4)

    model = MyModel()
    if args.gpu:
        model = model.to("cuda:0")

    features = []
    for im, label in tqdm(data_loader):
        if args.gpu:
            im = im.to("cuda:0")
        featurized = model.model.forward(im)
        features.append(featurized)
    features = torch.cat(features)

    if args.save_path is not None:
        features_np = features.view(features.shape[0], -1).cpu().numpy()
        np.save(args.save_path, features_np)
