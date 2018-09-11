import numpy as np
from PIL import Image

import torch


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, files, labels=None, transform=None, device=torch.device("cpu")):
        self.transform = transform
        self.files = files
        self.labels = np.zeros(len(files)) if labels is None else labels
        self.device = device

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
        return sample.to(self.device), torch.tensor(label, dtype=torch.int64).to(self.device)

    @staticmethod
    def get_file_id(file):
        return int(file.name.split(".")[0])