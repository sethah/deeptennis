import numpy as np
from PIL import Image
import itertools
import cv2

from sklearn.metrics.pairwise import rbf_kernel

import torch


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, files, labels=None, transform=None):
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

    @staticmethod
    def get_file_id(file):
        return int(file.name.split(".")[0])


class ImageDatasetBox(torch.utils.data.Dataset):

    def __init__(self, files, labels=None, transform=None):
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
            sample, label = self.transform(sample, label)
        return sample, torch.from_numpy(label)

    def with_transforms(self, transforms):
        self.transform = transforms
        return self

    @staticmethod
    def get_file_id(file):
        return int(file.name.split(".")[0])


class GridDataset(torch.utils.data.Dataset):
    def __init__(self, bbox_ds, grid_size):
        self.ds = bbox_ds
        self.grid_size = grid_size

    def __getitem__(self, idx):
        data, target = self.ds[idx]
        gsize = torch.IntTensor(self.grid_size).double().to(data.device)
        im_size = torch.tensor(data.size()[1:]).double().to(data.device)
        grid_coords = (target / (im_size / gsize)).long()
        n_coords = 4
        z = torch.zeros([n_coords] + gsize.long().tolist())
        for i, coord in enumerate(grid_coords):
            c = coord.numpy()
            if np.all(c >= 0) and np.all(c < np.array(self.grid_size)):
                z[i, c[1], c[0]] = 1.
        return data, z

    def __len__(self):
        return len(self.ds)


class HeatmapDataset(torch.utils.data.Dataset):

    def __init__(self, bbox_ds, im_size, grid_size, gamma=0.01):
        self.ds = bbox_ds
        self.im_size = im_size
        self.ind = np.array(list(itertools.product(range(self.im_size[0]), range(self.im_size[1]))))
        self.grid_size = grid_size
        self.gamma = gamma

    def __getitem__(self, idx):
        data, target = self.ds[idx]
        class_maps = []
        for i, coord in enumerate(target):
            c = coord.numpy()
            if np.all(c >= 0) and np.all(c < np.array(self.im_size)):
                hmap = HeatmapDataset._place_gaussian(c[::-1], self.gamma, self.ind, self.im_size[0], self.im_size[1])
                hmap[hmap > 1] = 1.
                hmap[hmap < 0.0099] = 0.
                class_maps.append(cv2.resize(hmap, self.grid_size))
            else:
                class_maps.append(np.zeros(self.grid_size))
        return data, torch.tensor(class_maps, dtype=torch.float32)

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _place_gaussian(mean, gamma, ind, height, width):
        return rbf_kernel(ind, np.array(mean).reshape(1, 2), gamma=gamma).reshape(height, width)