import numpy as np
from PIL import Image
import itertools
import random
import cv2

import logging

from sklearn.metrics.pairwise import rbf_kernel

import torch
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf

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
        return ImageFilesDataset.compute_mean_std(_ds)

    @staticmethod
    def compute_mean_std(ds):
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

    def transform(self, image, keypoints):
        rows, cols = image.size

        if random.random() > 0.5:
            image = tvf.hflip(image)
            keypoints[:, 0] = rows - keypoints[:, 0]
            # TODO this doesn't generalize
            keypoints = keypoints[np.array([1, 0, 3, 2, 5, 4, 7, 6])]

        image = tvf.to_tensor(image)

        return image, keypoints

    def with_transforms(self, tfms):
        return ImageFilesDatasetBox(self.files, self.labels, transform=tfms)

    def statistics(self):
        std_transforms = transforms.Compose([transforms.WrapTransform(tvt.ToTensor())])
        _ds = self.with_transforms(std_transforms)
        return ImageFilesDataset.compute_mean_std(_ds)


class ImageFilesDatasetKeypoints(torch.utils.data.Dataset):
    def __init__(self, files, corners, scoreboard):
        self.files = files
        self.corners = corners
        self.scoreboard = scoreboard
        self.mean, self.std = self.statistics()
        self._train = True

    def __getitem__(self, idx):
        # TODO: store labels as torch tensors
        file = self.files[idx]
        with open(file, 'rb') as f:
            img = Image.open(f)
            sample = img.convert('RGB')
        sample, corners, scoreboard = self.transform(sample, self.corners[idx],
                                                     self.scoreboard[idx])
        return sample, corners, scoreboard

    def __len__(self):
        return len(self.files)

    def eval(self):
        self._train = False

    def train(self):
        self._train = True

    def transform(self, image, corners, scoreboard):
        corners, scoreboard = corners.copy(), scoreboard.copy()
        cols0, rows0 = image.size

        rows, cols = rows0, cols0
        if self._train:
            if random.random() > 0.5:
                image = tvf.hflip(image)
                corners[:, 0] = cols - corners[:, 0]
                scoreboard[:, 0] = cols - scoreboard[:, 0]
                corners = corners[np.array([1, 0, 3, 2])]

        if self._train:
            resample, expand, center = False, True, None
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
            corners = np.dot(M, np.concatenate([corners.T, np.ones((1, corners.shape[0]))])).astype(int).T
            scoreboard = np.dot(M, np.concatenate([scoreboard.T, np.ones((1, scoreboard.shape[0]))])).astype(int).T
            image = tvf.rotate(image, angle, resample, expand, center)
            cols, rows = image.size
            corners = corners + np.array([(cols - cols0) // 2, (rows - rows0) // 2])
            scoreboard = scoreboard + np.array([(cols - cols0) // 2, (rows - rows0) // 2])

        cols_in, rows_in = cols, rows
        final_size = (256, 256)
        image = tvf.resize(image, final_size, Image.BILINEAR)
        cols, rows = image.size
        corners = corners * np.array([cols / cols_in, rows / rows_in])
        scoreboard = scoreboard * np.array([cols / cols_in, rows / rows_in])

        if self._train:
            jitter = tvt.ColorJitter(brightness=0.1, hue=0.1, contrast=0.5, saturation=0.5)
            image = jitter(image)

        image = tvf.to_tensor(image)
        image = tvf.normalize(image, self.mean, self.std)

        scoreboard_im = np.zeros((rows, cols), dtype=np.float32)
        scoreboard = cv2.fillConvexPoly(scoreboard_im, scoreboard.astype(np.int32), 1)
        corners = transforms.place_gaussian(corners, 0.05, rows, cols, (rows, cols))
        # scoreboard = cv2.resize(scoreboard_im, (rows, cols), interpolation=cv2.INTER_NEAREST)

        return image, corners, scoreboard

    def statistics(self):
        _ds = ImageFilesDataset(self.files, transform=tvt.ToTensor())
        return ImageFilesDataset.compute_mean_std(_ds)


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
                bbox = bbox.reshape(-1, 2)
                score_width = abs(bbox[5, 0] - bbox[4, 0])
                if filter_valid and not utils.validate_court_box(*bbox[:4], im_w, im_h):
                    invalid += 1
                    continue
                elif filter_valid and score_width < 10:
                    invalid += 1
                    continue
                else:
                    cnt += 1
                    frames.append(frame)
                    bboxes.append(bbox)
            if max_frames and cnt > max_frames:
                break
        logging.debug(f"Filtered {invalid} invalid frames for {video.name}")
    return ImageFilesDatasetBox(frames, bboxes)
