import argparse
from pathlib import Path
from logging.config import fileConfig

import torch.nn as nn
import torch.utils
import torchvision.models as torch_models
import torchvision.transforms as tvt
from torch.utils.data.dataloader import default_collate

from src.data.clip import Video
from src.data.dataset import ImageFilesDataset
from src.vision.transforms import *
import src.models.models as models
import src.utils as utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-height", type=int, default=224)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument("--img-mean", type=str, default=None)
    parser.add_argument("--img-std", type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default="")
    parser.add_argument('--frame-path', type=str, default="")
    parser.add_argument('--action-path', type=str, default="")
    parser.add_argument('--save-path', type=str, default="")
    args = parser.parse_args()

    fileConfig('logging_config.ini')
    np.random.seed(args.seed)

    im_size = (args.img_height, args.img_width)
    use_gpu = args.gpu or torch.cuda.is_available()
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    logging.debug(f"Predicting on device {device}.")

    ds_mean = [float(x) for x in args.img_mean.split(",")]
    ds_std = [float(x) for x in args.img_std.split(",")]

    frame_path = Path(args.frame_path)
    action_path = Path(args.action_path)

    batch_size = args.batch_size

    res = torch_models.resnet34(pretrained=True)
    C1 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)
    C2 = res.layer1
    C3 = res.layer2
    C4 = res.layer3
    C5 = res.layer4
    head = models.CourtScoreHead(128, out_channels=6)
    fpn = models.FPN(C1, C2, C3, C4, C5)
    model = nn.Sequential(fpn, head)
    sample_img = torch.randn(4, 3, im_size[0], im_size[1])
    sample_out = model.forward(sample_img)
    score_grid_size = tuple(sample_out[1][1].shape[-2:])
    court_grid_size = tuple(sample_out[0].shape[-2:])
    model = models.AnchorBoxModel([fpn, head], [score_grid_size], [(1, 1)], im_size, angle_scale=10)

    loaded = utils.load_checkpoint(args.checkpoint_path, best=True)
    model.load_state_dict(loaded['model'])
    model = model.to(device)

    action_mask = np.load(action_path)
    video = Video.from_dir(frame_path)
    frames = [f for i, f in enumerate(video.frames) if action_mask[i]]

    tfms = tvt.Compose([tvt.Resize(im_size), tvt.ToTensor(), tvt.Normalize(ds_mean, ds_std)])
    ds = ImageFilesDataset(frames, transform=tfms)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    inps, court_preds, score_preds = model.predict(loader, device)

    score_coords = np.stack([BoxToCoords()(box) for box in score_preds])
    hmaps = court_preds.numpy()
    im_size_full = cv2.imread(str(video.frames[0])).shape[:2]
    x, y = np.unravel_index(np.argmax(hmaps.reshape(hmaps.shape[0], hmaps.shape[1], -1), axis=2), hmaps.shape[-2:])
    resize_scale = np.array([im_size_full[1] / im_size[1], im_size_full[0] / im_size[0]])
    court_rescale = np.array([im_size[1] / court_grid_size[1], im_size[0] / court_grid_size[0]])
    court_vertices = np.stack([y, x], axis=2) * court_rescale * resize_scale

    out_frames = []
    j = 0
    for i, frame in enumerate(video.frames):
        im = cv2.imread(str(frame))
        if not action_mask[i]:
            out_frames.append(im)
        else:
            vertices = score_coords[j] * resize_scale
            im = cv2.polylines(im, [vertices.astype(np.int32)], True, (0, 255, 255), 4)
            im = cv2.polylines(im, [court_vertices[j].astype(np.int32)], True, (0, 255, 255), 4)
            out_frames.append(im)
            j += 1

    out_path = Path(args.save_path)
    out_path.mkdir(exist_ok=True, parents=True)
    for i, im in enumerate(out_frames):
        full_path = str(out_path / ("%05d.jpg" % i))
        cv2.imwrite(full_path, im)



