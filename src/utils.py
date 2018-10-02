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
                   'peak_distance': row['peak_distance'],
                   'crop_left': row['crop_left'],
                   'crop_right': row['crop_right'],
                   'crop_top': row['crop_top'],
                    'crop_bottom': row['crop_bottom']}
    return d


def freeze(params):
    for p in params:
        p.requires_grad = False


def unfreeze(params):
    for p in params:
        p.requires_grad = True


def save_checkpoint(checkpoint_dir, save_dict, model_name=None, file_name=None, best=False):
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        logging.log(logging.INFO, f"Checkpoint path {checkpoint_dir} does not exist. Creating it.")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if file_name is not None:
        save_path = checkpoint_dir / file_name
    elif model_name is not None:
        suffix = "latest.pkl" if not best else "best.pkl"
        save_path = checkpoint_dir / f"{model_name}.{suffix}"
    else:
        save_path = checkpoint_dir / "model_chk.latest.pkl"
    torch.save(save_dict, str(save_path))
    logging.log(logging.INFO, f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_dir, model_name=None, checkpoint_file=None, best=False):
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_file is not None:
        load_path = checkpoint_dir / checkpoint_file
    elif model_name is not None:
        suffix = "latest.pkl" if not best else "best.pkl"
        load_path = checkpoint_dir / f"{model_name}.{suffix}"
    else:
        raise ValueError("load checkpoint requires model name or file name")
    loaded = torch.load(load_path)
    logging.info(f"Loaded checkpoint from {load_path.resolve().as_uri()}")
    return loaded


def get_court_area(p1, p2, p3, p4):
    a = abs(p3[0] - p4[0])
    b = abs(p1[0] - p2[0])
    h = abs(p1[1] - p4[1])
    return 0.5 * (a + b) * h


def validate_court_box(p1, p2, p3, p4, im_w, im_h, bot_width_tol=(0.4, 0.9),
                       top_width_tol=(0.1, 0.8), height_tol=(0.1,0.8), area_tol=(0.2,0.7)):

    valid = True
    bot_width = abs(p1[0] - p2[0])
    top_width = abs(p3[0] - p4[0])
    valid &= bot_width > im_w * bot_width_tol[0] and bot_width < im_w * bot_width_tol[1]
    valid &= top_width < bot_width and top_width > top_width_tol[0] * im_w and top_width < top_width_tol[1] * im_w

    # baselines are mostly horizontal
    valid &= abs(p1[1] - p2[1]) < im_h * 0.02
    valid &= abs(p3[1] - p4[1]) < im_h * 0.02

    valid &= p3[0] > p4[0]
    valid &= p2[0] > p1[0]

    valid &= (p1[1] - p4[1] > im_h * height_tol[0]) and (p1[1] - p4[1] < im_h * height_tol[1])
    valid &= (p2[1] - p3[1] > im_h * height_tol[0]) and (p2[1] - p3[1] < im_h * height_tol[1])

    area = get_court_area(p1, p2, p3, p4)
    valid &= area > area_tol[0] * (im_w * im_h) and area < area_tol[1] * (im_h * im_w)

    return valid

