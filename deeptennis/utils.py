import logging
import itertools
import json
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from PIL import Image
from typing import Any, Dict, Iterable, List, Tuple, Union

import torch
import torchvision.transforms as tvt


def image_norm(image_iter):
    imgs = []
    for img_batch in image_iter:
        imgs.append(img_batch.numpy())
    imgs = np.concatenate(imgs, axis=0)
    return [float(x) for x in np.array(np.mean(imgs)).ravel()], \
           [float(x) for x in np.array(np.std(imgs)).ravel()]


def to_img_np(torch_img):
    np_img = torch_img.cpu().detach().numpy()
    if len(np_img.shape) == 3:
        return np_img.transpose(1, 2, 0)
    else:
        return np_img


def write_json_lines(lines: List[Dict[str, Any]], path: Union[str, Path]):
    lines = [json.dumps(l) + "\n" for l in lines]
    with open(path, 'w') as f:
        f.writelines(lines)


def read_json_lines(file_path: str) -> List[Dict[str, Any]]:
    js_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            js_lines.append(json.loads(line))
    return js_lines


def _IOU(boxes1: torch.Tensor, boxes2: torch.Tensor, im_size: Tuple[int, int]) -> torch.Tensor:
    """
    :param boxes1: (batch x 4)
    :param boxes2: (batch x 4)
    :return:
    (batch x 4) -> (batch x 224 x 224
    """
    boxes1 = boxes1.cpu().detach().numpy()
    boxes2 = boxes2.cpu().detach().numpy()
    scores = []
    for i, (b1, b2) in enumerate(zip(boxes1, boxes2)):
        img1 = np.zeros(im_size, np.uint8)
        img2 = np.zeros(im_size, np.uint8)
        bbox1 = BoundingBox.from_box(b1.tolist())
        bbox2 = BoundingBox.from_box(b2.tolist())
        points1 = bbox1.as_list()
        points2 = bbox2.as_list()
        # print(points1, points2)
        a = np.array(points1).reshape(4, 2)
        # print(a)
        img1 = cv2.fillConvexPoly(img1, np.array(points1).reshape(4, 2).astype(np.int64), 255)
        img2 = cv2.fillConvexPoly(img2, np.array(points2).reshape(4, 2).astype(np.int64), 255)
        intersection = np.logical_and(img1, img2)
        union = np.logical_or(img1, img2)
        iou_score = np.sum(intersection) / np.sum(union)
        scores.append(iou_score)
    return torch.tensor(scores)


def get_trapezoid(x2, y2, x3, y3, x4, y4):
    a = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
    c = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
    b = np.sqrt((x4 - x2)**2 + (y4 - y2)**2)
    theta = np.arccos((a**2 + b**2 - c**2) / (2*a*b))
    m = (y3 - y4) / (x3 - x4)
    phi = np.arctan(m)
    e = np.sqrt(c**2 - (b * np.sin(theta))**2)
    x1 = x2 - (2*e + a) * np.cos(phi)
    y1 = y2 - (2*e + a) * np.sin(phi)
    return np.array([x1, y1, x2, y2, x3, y3, x4, y4])


def read_images(files: Iterable[Path]) -> Iterable[torch.Tensor]:
    for file in files:
        with open(file, 'rb') as f:
            img = Image.open(f)
            sample = img.convert('RGB')
            yield tvt.ToTensor()(sample)


def compute_mean_std(images: Iterable[torch.Tensor], nsample: int = -1):
    """
    Compute the mean and standard deviation for a data loader of image tensors.
    """
    if nsample == -1:
        images = images
    else:
        images = itertools.islice(images, nsample)
    X = torch.stack(list(images))
    X = torch.transpose(X, 0, 1).contiguous().view(3, -1)
    mean = torch.mean(X, dim=1)
    std = torch.std(X, dim=1)
    return mean.numpy(), std.numpy()


def get_match_metadata(path):
    df = pd.read_csv(path)
    d = {}
    for i, row in df.iterrows():
        name = row['match_name']
        d[name] = {k: v for k, v in row.items() if k != 'match_name'}
    return d


def freeze(params: Iterable[torch.nn.Parameter]) -> None:
    for p in params:
        p.requires_grad = False


def unfreeze(params: Iterable[torch.nn.Parameter]) -> None:
    for p in params:
        p.requires_grad = True


def save_checkpoint(checkpoint_dir, save_dict, file_name=None, best=False):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    if file_name is not None:
        save_path = checkpoint_dir / file_name
    else:
        suffix = "latest.pkl" if not best else "best.pkl"
        save_path = checkpoint_dir / suffix
    torch.save(save_dict, str(save_path))
    logging.debug(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_dir, best=False):
    checkpoint_dir = Path(checkpoint_dir)
    suffix = "latest.pkl" if not best else "best.pkl"
    load_path = checkpoint_dir / suffix
    loaded = torch.load(load_path, map_location=lambda storage, loc: storage)
    logging.debug(f"Loaded checkpoint from {load_path.resolve().as_uri()}")
    return loaded


def get_court_area(p1, p2, p3, p4):
    a = abs(p3[0] - p4[0])
    b = abs(p1[0] - p2[0])
    h = abs(p1[1] - p4[1])
    return 0.5 * (a + b) * h


def validate_court_box(p1, p2, p3, p4, im_w, im_h, bot_width_tol=(0.4, 0.99),
                       top_width_tol=(0.1, 0.8), height_tol=(0.1,0.8), area_tol=(0.2,0.7)):

    valid = True
    bot_width = abs(p1[0] - p2[0])
    top_width = abs(p3[0] - p4[0])
    valid &= bot_width > im_w * bot_width_tol[0] and bot_width < im_w * bot_width_tol[1]
    valid &= top_width < bot_width and top_width > top_width_tol[0] * im_w and top_width < top_width_tol[1] * im_w

    # baselines are mostly horizontal
    valid &= abs(p1[1] - p2[1]) < im_h * 0.05
    valid &= abs(p3[1] - p4[1]) < im_h * 0.05

    valid &= p3[0] > p4[0]
    valid &= p2[0] > p1[0]

    valid &= (p1[1] - p4[1] > im_h * height_tol[0]) and (p1[1] - p4[1] < im_h * height_tol[1])
    valid &= (p2[1] - p3[1] > im_h * height_tol[0]) and (p2[1] - p3[1] < im_h * height_tol[1])

    area = get_court_area(p1, p2, p3, p4)
    valid &= area > area_tol[0] * (im_w * im_h) and area < area_tol[1] * (im_h * im_w)

    return valid

