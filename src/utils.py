import numpy as np


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