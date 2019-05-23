import logging
import os
from pathlib import Path
import math
from unittest import TestCase

import torch

from deeptennis.models.vision_models import *


class MetricTestCase(TestCase):  # pylint: disable=too-many-public-methods

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_segmentation_model(self):
        im_size = (224, 224)
        batch_size = 4
        keypoints = 3
        encoder = UNet(keypoints=keypoints, channels=3, filters=8, input_size=im_size)
        segmenter = SegmentationModel(encoder)
        im = torch.randn(batch_size, 3, im_size[0], im_size[1])
        output = segmenter.forward(im, mask=None)
        pred = output['pred']
        self.assertTupleEqual(pred.shape, (batch_size, keypoints, im_size[0], im_size[1]))
        # for random input, the logits should be both positive and negative
        self.assertGreater(pred.max(), 0)
        self.assertLess(pred.min(), 0)

        output = segmenter.decode(output)
        pred = output['pred']
        self.assertTupleEqual(pred.shape, (batch_size, im_size[0], im_size[1]))
        # each class should be represented in the output
        self.assertSetEqual(set(torch.unique(pred).numpy().tolist()), set(list(range(keypoints))))

