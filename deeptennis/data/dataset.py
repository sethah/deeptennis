import numpy as np
from pathlib import Path
from PIL import Image
from typing import Callable, List

import torch


def compute_mean_std(ds: torch.utils.data.Dataset):
    """
    Compute the mean and standard deviation for each image channel.
    """
    tsum = 0.
    tcount = 0.
    tsum2 = 0.
    for i in range(len(ds)):
        im, *_ = ds[i]
        im = im.view(im.shape[0], -1)
        tsum = tsum + im.sum(dim=1)
        tcount = tcount + im.shape[1]
        tsum2 = tsum2 + (im * im).sum(dim=1)
    mean = tsum / tcount
    std = torch.sqrt(tsum2 / tcount - mean ** 2)
    return mean, std


class ImageFilesDataset(torch.utils.data.Dataset):

    def __init__(self, files: List[Path], labels: np.ndarray=None, transform: Callable=None):
        self.transform = transform
        self.files = files
        self.labels = np.zeros(len(files)) if labels is None else labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]
        with open(file, 'rb') as f:
            img = Image.open(f)
            sample = img.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, torch.tensor(label, dtype=torch.int64)
