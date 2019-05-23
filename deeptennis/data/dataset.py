import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
import random
from typing import Callable, List, Tuple, Iterable

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField
from allennlp.data import Instance
from allennlp.common import Registrable

import torch
import torchvision.transforms as tvt
import torchvision.transforms.functional as tvf

import albumentations as aug
from albumentations.core.transforms_interface import ImageOnlyTransform

import deeptennis.vision.transforms as transforms


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

    @staticmethod
    def get_file_id(file):
        return int(file.name.split(".")[0])

    def with_transforms(self, tfms):
        return ImageFilesDataset(self.files, self.labels, transform=tfms)

    def statistics(self):
        std_transforms = tvt.Compose([tvt.ToTensor()])
        _ds = self.with_transforms(std_transforms)
        return compute_mean_std(_ds)

class AugmentationWrapper(Registrable):

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        raise NotImplementedError


class SwapChannels(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=1.0):
        super(SwapChannels, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return img.transpose(2, 0, 1)


@AugmentationWrapper.register("score_augmentor")
class TennisAugmentor(AugmentationWrapper):

    def __init__(self,
                 mean: List[float],
                 std: List[float],
                 size: Tuple[int, int] = (224, 224),
                 channel_prob: float = 0.5,
                 flip_prob: float = 0.5,
                 train: bool = True):
        self.mean = mean
        self.std = std
        self.size = size
        self._train = train
        if self._train:
            self.aug = aug.Compose([
                aug.Resize(self.size[0], self.size[1], always_apply=True),
                aug.Normalize(mean=self.mean, std=self.std),
                aug.HorizontalFlip(p=flip_prob),
                aug.Rotate(always_apply=True, limit=30),
                aug.ChannelShuffle(p=channel_prob),
                SwapChannels(always_apply=True)
            ])
        else:
            self.aug = aug.Compose([
                aug.Resize(self.size[0], self.size[1], always_apply=True),
                aug.Normalize(mean=self.mean, std=self.std),
                SwapChannels(always_apply=True)
            ])


    def __call__(self, image: np.ndarray, mask: np.ndarray):
        return self.aug(image=image, mask=mask)


class TennisTransform(Registrable):

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@TennisTransform.register("court_score")
class CourtAndScoreTransform(TennisTransform):

    def __init__(self,
                 mean: List[float],
                 std: List[float],
                 size: Tuple[int, int] = (224, 224),
                 corners_grid_size: Tuple[int] = (56, 56),
                 mask_prob: float = 0.5,
                 train: bool = True):
        self.mean = mean
        self.std = std
        self._train = train
        self.size = size
        self.corners_grid_size = corners_grid_size
        self.mask_prob = mask_prob

    def __call__(self,
                 image: Image.Image,
                 corners: np.ndarray,
                 score: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        corners, scoreboard = corners.copy(), score.copy()
        cols0, rows0 = image.size

        # randomly add some pixels to the width of the scoreboard
        # TODO: revisit?
        # scoreboard[2] += int(random.random() * 5 + 2)
        # scoreboard = transforms.BoxToCoords()(scoreboard)

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
            corners = np.dot(M, np.concatenate([corners.T, np.ones((1, corners.shape[0]))])).astype(
                int).T
            scoreboard = np.dot(M, np.concatenate(
                [scoreboard.T, np.ones((1, scoreboard.shape[0]))])).astype(int).T
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
            p = random.random()
            if p < self.mask_prob:
                w = random.randint(20, 80)
                h = random.randint(20, 80)
                imarr = np.array(image)
                # propose a starting point until the patch will fit within the image
                while True:
                    x1, y1 = random.randint(0, imarr.shape[1]), random.randint(0,
                                                                               imarr.shape[0])
                    if x1 + w < imarr.shape[1] and y1 + h < imarr.shape[0]:
                        break
                patch = np.random.randint(0, 255, np.prod((w, h))).reshape(
                    (w, h)).astype(np.uint8)
                image.paste(Image.fromarray(patch), (x1, y1))

        if self._train:
            jitter = tvt.ColorJitter(brightness=0.1, hue=0.1, contrast=0.5, saturation=0.5)
            image = jitter(image)

        image = tvf.to_tensor(image)
        image = tvf.normalize(image, self.mean, self.std)

        scoreboard = transforms.CoordsToBox()(scoreboard)
        scoreboard = torch.from_numpy(scoreboard)
        corners = transforms.place_gaussian(corners, 0.5, rows, cols, self.corners_grid_size)
        return image, torch.from_numpy(corners), scoreboard


class ImagePathsDataset(torch.utils.data.Dataset):
    """
    Implement a joint image and bounding box augmentation.

    TODO: this class is way too hacky
    """

    def __init__(self, files, corners, scoreboard, transform=None):
        self.files = files
        self.corners = corners
        self.scoreboard = scoreboard
        self.transform = transform

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


@DatasetReader.register("tennis")
class TennisDatasetReader(DatasetReader):

    def __init__(self, transform: TennisTransform = None, lazy: bool = True):
        super(TennisDatasetReader, self).__init__(lazy=lazy)
        self.transform = transform

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        You are strongly encouraged to use a generator, so that users can
        read a dataset in a lazy way, if they so choose.
        """
        with open(file_path, 'rb') as f:
            images = json.load(f)
        for image_file in images['annotations']:
            img = Image.open(Path(image_file['path']) / image_file['name'])
            sample = img.convert('RGB')
            annos = image_file['annotations']
            court_bbox = annos[0]['bbox'] if annos[0]['category'] == 'court' else annos[1]['bbox']
            score_bbox = annos[0]['bbox'] if annos[0]['category'] == 'score' else annos[1]['bbox']
            if np.any(court_bbox) and np.any(score_bbox):
                court = court_bbox
                score = score_bbox
                if self.transform is not None:
                    sample, court, score = self.transform(sample,
                                                          np.array(court).reshape(4, 2),
                                                          np.array(score).reshape(4, 2))
                yield self.text_to_instance(sample, court, score)

    def text_to_instance(self, sample, court, score) -> Instance:
        fields = {
            'img': ArrayField(sample),
            # 'court': ArrayField(court),
            'labels': ArrayField(score),
        }
        return Instance(fields)

@DatasetReader.register("score")
class TennisScoreDatasetReader(DatasetReader):

    def __init__(self, transform: AugmentationWrapper = None, lazy: bool = True):
        super(TennisScoreDatasetReader, self).__init__(lazy=lazy)
        self.transform = transform

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        You are strongly encouraged to use a generator, so that users can
        read a dataset in a lazy way, if they so choose.
        """
        with open(file_path, 'rb') as f:
            images = json.load(f)
        for image_file in images['annotations']:
            img = Image.open(Path(image_file['path']) / image_file['name'])
            mask = Image.open(Path(image_file['mask_path']) / image_file['mask_name'])
            mask = np.array(mask)
            sample = img.convert('RGB')
            if self.transform is not None:
                ret = self.transform(image=np.array(sample), mask=mask)

                sample = ret['image']
                mask = ret['mask']
            else:
                sample = np.array(sample)
            yield self.text_to_instance(sample, mask)
            # annos = image_file['annotations']
            # court_bbox = annos[0]['bbox'] if annos[0]['category'] == 'court' else annos[1]['bbox']
            # score_bbox = annos[0]['bbox'] if annos[0]['category'] == 'score' else annos[1]['bbox']
            # if np.any(court_bbox) and np.any(score_bbox):
            #     court = court_bbox
            #     score = score_bbox

    def text_to_instance(self, sample: np.ndarray, mask: np.ndarray) -> Instance:
        fields = {
            'img': ArrayField(sample),
            'mask': ArrayField(mask)
        }
        return Instance(fields)
