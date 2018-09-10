import numpy as np


def image_norm(image_iter):
    imgs = []
    for img_batch in image_iter:
        imgs.append(img_batch.numpy())
    imgs = np.concatenate(imgs, axis=0)
    return np.mean(imgs).tolist(), np.std(imgs).tolist()