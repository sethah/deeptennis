import os
import argparse
import subprocess
from pathlib import Path


"""
python src/data/vid2img.py \
--vid-path ./data/raw/djo_fed_aus.mp4 \
--img-path ./data/processed/djo_fed_aus/sample/ \
--fps 0.1
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--vid-path", type=str)
    parser.add_argument("--img-path", type=str)
    args = parser.parse_args()

    vid_path = Path(args.vid_path)
    img_path = Path(args.img_path)

    if not os.path.exists(img_path):
        os.mkdir(img_path)
    subprocess.call(["ffmpeg", "-i", str(vid_path), "-r", str(args.fps), str(img_path / "%05d.jpg")])


