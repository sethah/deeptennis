import numpy as np

import albumentations.augmentations.functional as af
from albumentations.core.transforms_interface import DualTransform

from allencv.data.transforms import _ImageTransformWrapper, ImageTransform


class CourtKeypointFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return af.hflip_cv2(img)
        else:
            return af.hflip(img)

    def apply_to_bbox(self, bbox, **params):
        return af.bbox_hflip(bbox, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return af.keypoint_hflip(keypoint, **params)

    def apply_to_keypoints(self, keypoints, **params):
        keypoints = [list(keypoint) for keypoint in keypoints]
        kps = [self.apply_to_keypoint(keypoint[:4], **params) + keypoint[4:] for keypoint in keypoints]
        # print(kps)
        # sorted_x = sorted(kps, key=lambda x: x[0])
        # bottom_left = max(sorted_x[:2], key=lambda x: x[1])
        # top_left = min(sorted_x[:2], key=lambda x: x[1])
        # bottom_right = max(sorted_x[2:], key=lambda x: x[1])
        # top_right = min(sorted_x[2:], key=lambda x: x[1])
        # tmp = [bottom_left, bottom_right, top_right, top_left]
        # print(tmp)
        return kps


ImageTransform.register("keypoint_hflip")(_ImageTransformWrapper(CourtKeypointFlip))