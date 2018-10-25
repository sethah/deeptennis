import numpy as np
import time
from PIL import Image, ImageDraw
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

    def __init__(self, files, corners, scoreboard, mean=None, std=None,
                 size=(224, 224), corners_grid_size=(56, 56), repeat=1,
                 anchor_boxes=None, anchor_transform=None):
        # TODO: handle mean and std better?
        self.files = files * repeat
        self.corners = np.repeat(corners, repeat, axis=0)
        self.scoreboard = np.repeat(scoreboard, repeat, axis=0)
        self.mean = mean
        self.std = std
        self._train = True
        self.size = size
        self.corners_grid_size = corners_grid_size
        self.anchor_boxes = anchor_boxes
        self.anchor_transform = anchor_transform

    def __getitem__(self, idx):
        # TODO: store labels as torch tensors
        file = self.files[idx]
        with open(file, 'rb') as f:
            img = Image.open(f)
            sample = img.convert('RGB')
        sample, corners, scoreboard, scoreboard_idx = self.transform(sample, self.corners[idx],
                                                     self.scoreboard[idx])
        return sample, corners, scoreboard, scoreboard_idx

    def __len__(self):
        return len(self.files)

    def eval(self):
        self._train = False

    def train(self):
        self._train = True

    def transform(self, image, corners, scoreboard):
        corners, scoreboard = corners.copy(), scoreboard.copy()
        cols0, rows0 = image.size

        # randomly add some pixels to the width
        scoreboard[2] += int(random.random() * 5 + 2)
        scoreboard = transforms.BoxToCoords()(scoreboard)

        rows, cols = rows0, cols0
        if self._train:
            if random.random() > 0.5:
                image = tvf.hflip(image)
                corners[:, 0] = cols - corners[:, 0]
                scoreboard[:, 0] = cols - scoreboard[:, 0]
                corners = corners[np.array([1, 0, 3, 2])]
                scoreboard = scoreboard[np.array([1, 0, 3, 2])]

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
        final_size = self.size
        image = tvf.resize(image, final_size, Image.BILINEAR)
        cols, rows = image.size
        corners = corners * np.array([cols / cols_in, rows / rows_in])
        scoreboard = scoreboard * np.array([cols / cols_in, rows / rows_in])

        # add random pixel patches to the court corners
        if self._train:
            for (x, y) in corners:
                p = random.random()
                if p < 0.4:
                    w = random.randint(20, 50)
                    h = random.randint(20, 50)
                    imarr = np.array(image)
                    start_x, start_y = random.randint(0, cols - w), random.randint(0, rows - h)
                    patch = imarr[start_y:start_y + h, start_x:start_x + w]
                    x1, y1 = max(0, int(x - w / 2)), max(0, int(y - h / 2))
                    patch = patch[:rows - y1, :cols - x1]
                    patch = np.random.randint(0, 255, np.prod(patch.shape)).reshape(patch.shape).astype(np.uint8)
                    image.paste(Image.fromarray(patch), (x1, y1))

        if self._train:
            jitter = tvt.ColorJitter(brightness=0.1, hue=0.1, contrast=0.5, saturation=0.5)
            image = jitter(image)

        image = tvf.to_tensor(image)
        image = tvf.normalize(image, self.mean, self.std)

        scoreboard = transforms.CoordsToBox()(scoreboard)
        scoreboard = torch.from_numpy(scoreboard)
        corners = transforms.place_gaussian(corners, 0.5, rows, cols, self.corners_grid_size)
        scoreboard_idx, scoreboard = self.anchor_transform(scoreboard)
        # scoreboard_idx = transforms.AnchorIndexes(self.anchor_boxes)(scoreboard)
        # scoreboard = (scoreboard - self.anchor_boxes.boxes[scoreboard_idx.item()]) / self.anchor_boxes.offsets[scoreboard_idx.item()]

        return image, corners, scoreboard, scoreboard_idx

    def compute_statistics(self):
        _ds = ImageFilesDataset(self.files, transform=tvt.ToTensor())
        self.mean, self.std = ImageFilesDataset.compute_mean_std(_ds)


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
