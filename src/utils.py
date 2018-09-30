import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.utils.data as data


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


def compute_mean_std(dataset, batch_size=32):
    """
    Compute the mean and standard deviation for a data loader of image tensors.
    """
    loader = data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    tsum = 0.
    tcount = 0.
    tsum2 = 0.
    for inp, *_ in loader:
        inp = inp.transpose(1, 0).contiguous().view(3, -1)
        tsum = tsum + inp.sum(dim=1)
        tcount = tcount + inp.shape[1]
        tsum2 = tsum2 + (inp * inp).sum(dim=1)
    mean = tsum / tcount
    std = torch.sqrt(tsum2 / tcount - mean**2)
    return mean.numpy(), std.numpy()


def get_match_metadata(path):
    df = pd.read_csv(path)
    d = {}
    for i, row in df.iterrows():
        name = "_".join([str(x) for x in [row['player1'], row['player2'], row['venue'], row['year']]])
        d[name] = {'sensitivity': row['sensitivity'],
                   'threshold': row['threshold'],
                   'peak_distance': row['peak_distance'],
                    'percentile': row['percentile']}
    return d


def freeze(params):
    for p in params:
        p.requires_grad = False


def unfreeze(params):
    for p in params:
        p.requires_grad = True
