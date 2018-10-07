import numpy as np
import itertools
import cv2
import random
import math
import numbers
from collections import Sequence, Iterable
from PIL import Image

from sklearn.metrics.pairwise import rbf_kernel

import torch
import torchvision.transforms.functional as Fv

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class WrapTransform(object):

    def __init__(self, torch_transform):
        self.torch_transform = torch_transform

    def __call__(self, img, bbox):
        return self.torch_transform(img), bbox


class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, bbox):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """
        angle = self.get_params(self.degrees)
        cols, rows = img.size
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
        bbox = np.dot(M, np.concatenate([bbox.T, np.ones((1, bbox.shape[0]))])).astype(int).T
        img = Fv.rotate(img, angle, self.resample, self.expand, self.center)
        cols2, rows2 = img.size
        bbox = bbox + np.array([(cols2 - cols) // 2, (rows2 - rows) // 2])
        return img, bbox

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bbox):
        for t in self.transforms:
            img, bbox = t(img, bbox)
        return img, bbox

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        resized = Fv.resize(img, self.size, self.interpolation)
        rows1, cols1 = img.size
        rows2, cols2 = resized.size
        return resized, bbox * np.array([rows2 / rows1, cols2 / cols1])

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size,
                                                                                interpolate_str)


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False,
                 fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img, bbox):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        angle, translations, scale, shear = self.get_params(self.degrees, self.translate,
                                                            self.scale, self.shear, img.size)

        new_img = Fv.affine(img, angle, translations, scale, shear, resample=self.resample,
                           fillcolor=self.fillcolor)
        return new_img, bbox + np.array(translations)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        rows1, cols1 = img.size
        hdiff = self.size[1] - rows1
        wdiff = self.size[0] - cols1
        return Fv.center_crop(img, self.size), bbox + np.array([hdiff // 2, wdiff // 2])

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = Fv.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = Fv.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            bbox = bbox - np.array([(self.size[1] - img.size[0]), 0.])
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = Fv.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            bbox = bbox - np.array([0, (self.size[0] - img.size[1])])

        i, j, h, w = self.get_params(img, self.size)

        return Fv.crop(img, i, j, h, w), bbox - np.array([j, i])

    #         return img, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        bbox = bbox - np.array([j, i])

        img = Fv.crop(img, i, j, h, w)
        rows1, cols1 = img.size
        cols2, rows2 = self.size
        bbox = bbox * np.array([rows2 / rows1, cols2 / cols1])
        img = Fv.resize(img, self.size, self.interpolation)
        return img, bbox

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class Pad(object):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be padded.
        Returns:
            PIL Image: Padded image.
        """
        bbox = bbox + np.array(self.padding)
        return Fv.pad(img, self.padding, self.fill, self.padding_mode), bbox

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.padding, self.fill, self.padding_mode)


class BoxCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        maxs = np.max(bbox, axis=0)
        mins = np.min(bbox, axis=0)
        i, j = mins
        w, h = maxs[0] - i, maxs[1] - j
        w2 = int(w * np.random.uniform(1.0, 2.))
        h2 = int(h * np.random.uniform(1.0, 2.))
        i -= (w2 - w) // 2
        j -= (h2 - h) // 2
        new_img = Fv.crop(img, j, i, h2, w2)
        if new_img.size[0] == 0 or new_img.size[1] == 0:
            print(i, j, img.size, h2, w2)
            return img, bbox
        return new_img, bbox - np.array([i, j])

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class JitterBox(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, bbox):
        jitter = np.random.normal(self.mean, self.std, bbox.reshape(-1).shape[0]).reshape(bbox.shape[0], 2)
        return img, bbox + jitter

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bbox):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        rows, cols = img.size
        if random.random() < self.p:
            bbox[:, 0] = rows - bbox[:, 0]
            # TODO this doesn't generalize
            return Fv.hflip(img), bbox[np.array([1, 0, 3, 2, 5, 4, 7, 6])]
        return img, bbox

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class BoxToGrid(object):

    def __init__(self, grid_size):
        self.grid_size = grid_size

    def __call__(self, img_t, bbox_t):
        gsize = torch.IntTensor(self.grid_size).double()
        im_size = torch.tensor(img_t.size()[1:]).double()
        grid_coords = (bbox_t / (im_size / gsize)).long()
        # TODO
        n_coords = 4
        grid = torch.zeros([n_coords] + gsize.long().tolist())
        for i, coord in enumerate(grid_coords):
            c = coord.numpy()
            if np.all(c >= 0) and np.all(c < np.array(self.grid_size)):
                grid[i, c[1], c[0]] = 1.
        return img_t, grid


class BoxToHeatmap(object):

    def __init__(self, grid_size, gamma):
        self.grid_size = grid_size
        self.gamma = gamma

    def __call__(self, img_t, bbox_t):
        im_size = tuple(img_t.shape[-2:])
        ind = np.array(list(itertools.product(range(im_size[0]), range(im_size[1]))))
        class_maps = []
        for coord in bbox_t:
            c = coord
            if np.all(c >= 0) and np.all(c < np.array(im_size)):
                hmap = BoxToHeatmap._place_gaussian(c[::-1], self.gamma, ind,
                                                    im_size[0], im_size[1])
                hmap[hmap > 1] = 1.
                hmap[hmap < 0.0099] = 0.
                class_maps.append(cv2.resize(hmap, self.grid_size))
            else:
                class_maps.append(np.zeros(self.grid_size))
        return img_t, torch.tensor(class_maps, dtype=torch.float32)

    @staticmethod
    def _place_gaussian(mean, gamma, ind, height, width):
        return rbf_kernel(ind, np.array(mean).reshape(1, 2), gamma=gamma).reshape(height, width)


class TorchImageToNumpy(object):

    def __init__(self):
        pass

    def __call__(self, img_t):
        return img_t.cpu().numpy().transpose(1, 2, 0)
