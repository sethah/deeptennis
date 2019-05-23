import logging
import os
from pathlib import Path
import shutil
import math
import tempfile
from unittest import TestCase

import torch

from deeptennis.models.metrics import *

TEST_DIR = tempfile.mkdtemp(prefix="allennlp_tests")


class MetricTestCase(TestCase):  # pylint: disable=too-many-public-methods

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_iou(self):
    #     im_size = (224, 224)
    #     metric = IOU(im_size)
    #     boxes1 = torch.tensor([0, 0, 10, 10, 0.]).view(1, 5)
    #     boxes2 = torch.tensor([0, 0, 10, 10, 0.]).view(1, 5)
    #     assert metric(boxes1, boxes2) == 1.0
    #
    #     boxes1 = torch.tensor([0, 0, 10, 10, 0.]).view(1, 5)
    #     boxes2 = torch.tensor([0, 0, 10, 20, 0.]).view(1, 5)
    #     assert metric(boxes1, boxes2) == 0.5
    #
    #     boxes1 = torch.tensor([0, 10, 10, 10, 0.]).view(1, 5)
    #     boxes2 = torch.tensor([0, 0, 10, 10, 0.]).view(1, 5)
    #     assert metric(boxes1, boxes2) == 0.0
    #
    #     boxes1 = torch.tensor([0, 0, 10, 10, 0.]).view(1, 5)
    #     boxes2 = torch.tensor([0, 0, 10, 10, 10.]).view(1, 5)
    #     assert metric(boxes1, boxes2) < 1.0

    def test_segmentation_iou(self):
        im_size = (224, 224)
        metric = SegmentationIOU(num_classes=2, im_size=im_size, skip=set([0]))
        target = torch.ones(1, 4, 4)
        predictions = torch.ones(1, 4, 4)
        metric(predictions, target)
        self.assertAlmostEqual(metric.get_metric(True), 1.)

        target = torch.zeros(1, 4, 4)
        metric(predictions, target)
        self.assertAlmostEqual(metric.get_metric(True), 0.)

        target[:, 0, :] = 1.
        metric(predictions, target)
        self.assertAlmostEqual(metric.get_metric(True), 4 / 16)
        # self.assertTrue(math.isnan(metric.get_metric(False)))

        target *= 0
        predictions *= 0
        metric.reset()
        metric(predictions, target)
        self.assertAlmostEqual(metric.get_metric(True), 0.)




