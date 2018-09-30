import os
import logging
import argparse
import subprocess
from pathlib import Path
import shutil


"""
python src/data/vid2img.py \
--vid-path ./data/raw/ \
--vid-name djokovic_federer_aus_16.mp4 \
--img-path ./data/processed/frames \
--fps 1

python src/data/vid2img.py \
--vid-path ./data/raw/ \
--img-path ./data/processed/frames \
--fps 1
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--vid-path", type=str)
    parser.add_argument("--vid-name", type=str, default=None)
    parser.add_argument("--img-path", type=str)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--vframes", type=int, default=1000)
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()

    vid_path = Path(args.vid_path)
    if args.vid_name is None:
        vids = [v for v in vid_path.iterdir() if v.name[-4:] == ".mp4"]
    else:
        vids = [vid_path / args.vid_name]

    for vid in vids:
        label = vid.name.split(".")[0]
        img_path = Path(args.img_path) / label
        if img_path.exists() and args.overwrite:
            shutil.rmtree(img_path)
        elif img_path.exists():
            logging.info(f"{img_path} exists. Skipping.")
            continue

        if not img_path.exists():
            logging.info(f"Creating image directory {img_path}.")
            img_path.mkdir(parents=True, exist_ok=True)

        subprocess.call(["ffmpeg", "-i", str(vid), "-vframes", str(args.vframes),
                         "-vf", "scale=640:360",
                         "-r", str(args.fps), str(img_path / "%05d.jpg")])


