import numpy as np
from PIL import Image
import itertools
import cv2

import logging

from sklearn.metrics.pairwise import rbf_kernel

import torch
import torchvision.transforms as tvt

from src.data.clip import ActionVideo
import src.utils as utils
import src.vision.transforms as transforms


class ImageFilesDataset(torch.utils.data.Dataset):

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

    def with_transforms(self, tfms):
        return ImageFilesDataset(self.files, self.labels, transform=tfms)

    def statistics(self):
        std_transforms = tvt.Compose([tvt.ToTensor()])
        _ds = self.with_transforms(std_transforms)
        return ImageFilesDataset._compute_mean_std(_ds)

    @staticmethod
    def _compute_mean_std(ds):
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


class ImageFilesDatasetBox(ImageFilesDataset):
    """
    A dataset that also applies transforms to a bounding polygon.
    """

    def __getitem__(self, idx):
        # TODO: store labels as torch tensors
        file = self.files[idx]
        label = self.labels[idx]
        with open(file, 'rb') as f:
            img = Image.open(f)
            sample = img.convert('RGB')
        if self.transform is not None:
            sample, label = self.transform(sample, label)
        return sample, label

    def with_transforms(self, tfms):
        return ImageFilesDatasetBox(self.files, self.labels, transform=tfms)

    def statistics(self):
        std_transforms = transforms.Compose([transforms.WrapTransform(tvt.ToTensor())])
        _ds = self.with_transforms(std_transforms)
        return ImageFilesDataset._compute_mean_std(_ds)


def get_bounding_box_dataset(videos, clip_path, filter_valid=False, max_frames=None):
    bboxes = []
    frames = []
    for video in videos:
        clips = ActionVideo.load(clip_path / (video.name + ".pkl"))
        invalid = 0
        cnt = 0
        for clip in clips:
            img = cv2.imread(str(clip.frames[0]))
            im_h, im_w, _ = img.shape
            for frame, bbox in clip:
                # TODO: bandaid code. Check for valid box elsewhere
                if filter_valid and not utils.validate_court_box(*bbox.reshape(-1, 2)[:4], im_w, im_h):
                    invalid += 1
                    continue
                else:
                    cnt += 1
                    frames.append(frame)
                    bboxes.append(bbox.reshape(-1, 2))
            if max_frames and cnt > max_frames:
                break
        logging.debug(f"Filtered {invalid} invalid frames for {video.name}")
    return ImageFilesDatasetBox(frames, bboxes)
