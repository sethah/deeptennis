import numpy as np
import itertools
import cv2
import numbers

from sklearn.metrics.pairwise import rbf_kernel

import torch
import torchvision.transforms.functional as tvf


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
        new_img = tvf.crop(img, j, i, h2, w2)
        if new_img.size[0] == 0 or new_img.size[1] == 0:
            return img, bbox
        return new_img, bbox - np.array([i, j])

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CoordsToGrid(object):

    def __init__(self, grid_size, im_size):
        self.grid_size = grid_size
        self.im_size = im_size

    def __call__(self, coords):
        gsize = torch.IntTensor(self.grid_size).double()
        im_size = torch.tensor(self.im_size).double()
        grid_coords = (coords / (im_size / gsize)).long()
        n_coords = coords.shape[0]
        grid = torch.zeros([n_coords] + gsize.long().tolist())
        for i, coord in enumerate(grid_coords):
            c = coord.numpy()
            if np.all(c >= 0) and np.all(c < np.array(self.grid_size)):
                grid[i, c[1], c[0]] = 1.
        return grid


def box_to_coords(boxes: torch.Tensor):
    assert boxes.shape[1] == 5
    x, y, w, h, theta = boxes.transpose(0, 1)
    theta = theta * np.pi / 180.
    phi = torch.atan(h / w)
    d = w / 2 / torch.cos(phi)
    x4 = x - d * torch.cos(theta - phi)
    y4 = y + d * torch.sin(theta - phi)
    x1 = x - d * torch.cos(theta + phi)
    y1 = y + d * torch.sin(theta + phi)
    x3 = x + d * torch.cos(theta + phi)
    y3 = y - d * torch.sin(theta + phi)
    x2 = x + d * torch.cos(theta - phi)
    y2 = y - d * torch.sin(theta - phi)
    return torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=1).view(-1, 4, 2)

class BoxToCoords(object):

    def __call__(self, box):
        if len(box) == 4:
            x, y, w, h = box
            return np.array([x, y + h, x + w, y + h, x + w, y, x, y], dtype=np.float32).reshape(4, 2)
        elif len(box) == 5:
            x, y, w, h, theta = box
            theta = theta * np.pi / 180.
            phi = np.arctan(h / w)
            d = w / 2 / np.cos(phi)
            x4 = x - d * np.cos(theta - phi)
            y4 = y + d * np.sin(theta - phi)
            x1 = x - d * np.cos(theta + phi)
            y1 = y + d * np.sin(theta + phi)
            x3 = x + d * np.cos(theta + phi)
            y3 = y - d * np.sin(theta + phi)
            x2 = x + d * np.cos(theta - phi)
            y2 = y - d * np.sin(theta - phi)
            return np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.float32).reshape(4, 2)
        else:
            raise ValueError(f"Box length must be 4 or 5 but was {len(box)}")


class CoordsToBox(object):

    def __call__(self, coords):
        x1, y1, x2, y2, x3, y3, x4, y4 = coords.ravel()
        theta = np.arctan((y1 - y2) / (x2 - x1))
        w = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        h = np.sqrt((x1 - x4)**2 + (y1 - y4)**2)
        phi = np.arctan(h / w)
        d = w / 2 / np.cos(phi)
        cx = x4 + d * np.cos(theta - phi)
        cy = y4 - d * np.sin(theta - phi)
        return np.array([cx, cy, w, h, theta * 180 / np.pi], dtype=np.float32)


class AnchorIndexes(object):

    def __init__(self, boxes, offsets, ):
        self.boxes = boxes
        self.offsets = offsets

    def __call__(self, coords):
        return self.anchor_boxes.get_best(coords.unsqueeze(0))


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


def place_gaussian(keypoints, gamma, height, width, grid_size):
    # TODO: the `rbf_kernel` call is really inneficient for large
    # TODO: grid sizes and will slow down training
    class_maps = []
    ind = np.array(list(itertools.product(range(grid_size[0]), range(grid_size[1]))))
    for coord in keypoints:
        if np.all(coord >= 0) and np.all(coord < np.array([height, width])):
            query = np.array([coord[1] * grid_size[0] / height, coord[0] * grid_size[1] / width])
            hmap = rbf_kernel(ind, query.reshape(1, 2),
                              gamma=gamma).reshape(grid_size[0], grid_size[1])
            hmap[hmap > 1] = 1.
            hmap[hmap < 0.0099] = 0.
            class_maps.append(hmap)
        else:
            class_maps.append(np.zeros(grid_size))
    return np.array(class_maps, dtype=np.float32)


class TorchImageToNumpy(object):

    def __call__(self, img_t):
        img_np = img_t.cpu().numpy()
        if img_t.dim() == 4:
            img_np = img_np.transpose(0, 3, 1, 2)
        elif img_t.dim() == 3:
            img_np = img_np.transpose(1, 2, 0)
        elif img_t.dim() == 2:
            img_np = img_np
        else:
            raise ValueError(f"image dim must be 2, 3, or 4 but was {img_t.dim()}")
        return img_np
