import numpy as np


def image_norm(image_iter):
    imgs = []
    for img_batch in image_iter:
        imgs.append(img_batch.numpy())
    imgs = np.concatenate(imgs, axis=0)
    return [float(x) for x in np.array(np.mean(imgs)).ravel()], \
           [float(x) for x in np.array(np.std(imgs)).ravel()]