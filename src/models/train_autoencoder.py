import numpy as np
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms

from tqdm import tqdm

from data.dataset import ImageDataset
import utils


class CAE(nn.Module):
    def __init__(self, height, width, embed_size=32):
        super(CAE, self).__init__()

        self.height = height
        self.width = width
        self.embed_size = embed_size

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(self.height // 8 * self.width // 8 * 128, self.embed_size)
        self.fc2 = nn.Linear(self.embed_size, self.height // 8 * self.width // 8 * 128)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.fc1(out.view(out.shape[0], -1))
        return out

    def decode(self, encoded):
        out = self.fc2(encoded).view(encoded.shape[0], 128, self.height // 8, self.width // 8)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.sigmoid(self.conv5(out))
        return out

    def forward(self, x):
        encoded = self.encode(x)
        return self.decode(encoded)

"""
PYTHONPATH=./src/ python src/models/train_autoencoder.py \
--imgs-path ./data/processed/djo_fed_aus/sample/0/ \
--img-height 256 \
--img-width 256 \
--n_epochs 1 \
--model-save-path ./models/autoencoder/model1
"""
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

    args = parser.parse_args()

    out_size = (args.img_height, args.img_width)
    if args.img_mean is None or args.img_std is None:
        img_transforms = transforms.Compose([transforms.Grayscale(),
                                             transforms.ToTensor()])
        dataset = ImageDataset(list(Path(args.imgs_path).iterdir()), transform=img_transforms)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=4)
        mean_img, std_img = utils.image_norm(map(lambda inp: inp[0], data_loader))
    else:
        mean_img = list(map(lambda x: float(x), args.img_mean.split(",")))
        std_img = list(map(lambda x: float(x), args.img_std.split(",")))
    print(mean_img, std_img)

    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
    img_transforms = transforms.Compose([transforms.Resize(300),
                                         transforms.Grayscale(),
                                         transforms.RandomAffine(5, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                                         transforms.CenterCrop(out_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean_img, std=std_img)])
    dataset = ImageDataset(list(Path(args.imgs_path).iterdir()), transform=img_transforms,
                           device=device)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=4)
    model = CAE(args.img_height, args.img_width).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if args.load_path is not None:
        state = torch.load(args.load_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        lr_sched.load_state_dict(state['sched'])

    for epoch in range(args.n_epochs):
        epoch_loss = 0.
        epoch_samples = 0
        for inp, targ in tqdm(data_loader):
            optimizer.zero_grad()
            out = model.forward(inp)
            loss = criterion(out, inp)
            epoch_loss += loss.item()
            epoch_samples += inp.shape[0]
            loss.backward()
            optimizer.step()
        print(epoch_loss / epoch_samples)
        lr_sched.step(epoch)

    if args.model_save_path is not None:
        Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'sched': lr_sched.state_dict()}, args.model_save_path)
