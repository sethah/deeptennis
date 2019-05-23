import logging
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
from typing import Dict, Optional, Tuple, Union
from overrides import overrides

import torch
import torch.utils.data as data
import torchvision.transforms as tvt

from allennlp.training.metrics import Metric

from deeptennis.vision.transforms import BoundingBox


class SegmentationIOU(Metric):

    def __init__(self, num_classes: int, im_size: Tuple[int, int], skip=set()):
        self.im_size = im_size
        self._count = 0
        self._sum = 0.0
        self.skip = skip
        self.num_classes = num_classes

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        :param predictions: (b x h x w)
        :param gold_labels: (b x h x w)
        :param mask:
        :return:
        """
        predictions = predictions.detach().cpu()
        gold_labels = gold_labels.detach().cpu()
        class_intersections = []
        class_unions = []
        for k in [x for x in range(self.num_classes) if x not in self.skip]:
            pred_inds = predictions == k
            target_inds = gold_labels == k
            intersection = (pred_inds & target_inds).long().view(pred_inds.shape[0], -1).sum(dim=1)
            union = (pred_inds | target_inds).long().view(pred_inds.shape[0], -1).sum(dim=1)
            class_intersections.append(intersection.float())
            class_unions.append(union.float())
        intersections = torch.stack(class_intersections)
        unions = torch.stack(class_unions)
        ious = intersections.sum(dim=0) / (unions.sum(dim=0) + 1e-8)
        self._count += pred_inds.shape[0]
        self._sum += ious.sum().item()

    @overrides
    def reset(self):
        self._count = 0
        self._sum = 0.0

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        if self._count == 0:
            raise ValueError("AHHHHHH")
        result = float('nan') if self._count == 0 else self._sum / self._count
        if reset:
            self.reset()
        return result


class IOU(Metric):

    def __init__(self, im_size: Tuple[int, int]):
        self.im_size = im_size
        self._count = 0
        self._sum = 0.0

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):

        boxes1 = predictions.cpu().detach().numpy()
        boxes2 = gold_labels.cpu().detach().numpy()
        scores = []
        for i, (b1, b2) in enumerate(zip(boxes1, boxes2)):
            img1 = np.zeros(self.im_size, np.uint8)
            img2 = np.zeros(self.im_size, np.uint8)
            bbox1 = BoundingBox.from_box(b1.tolist(), centered=True)
            bbox2 = BoundingBox.from_box(b2.tolist(), centered=True)
            points1 = bbox1.as_list()
            points2 = bbox2.as_list()
            img1 = cv2.fillConvexPoly(img1, np.array(points1).reshape(4, 2).astype(np.int64), 255)
            img2 = cv2.fillConvexPoly(img2, np.array(points2).reshape(4, 2).astype(np.int64), 255)
            intersection = np.logical_and(img1, img2)
            union = np.logical_or(img1, img2)
            iou_score = np.sum(intersection) / np.sum(union)
            scores.append(iou_score)
        self._count += boxes1.shape[0]
        self._sum += np.sum(scores)

    @overrides
    def reset(self):
        self._count = 0
        self._sum = 0.0

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        result = self._sum / self._count
        if reset:
            self.reset()
        return result

