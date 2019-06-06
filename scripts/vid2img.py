import logging
import argparse
import subprocess
from pathlib import Path
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--vid-path", type=str)
    parser.add_argument("--img-path", type=str)
    parser.add_argument("--vframes", type=int, default=1000)
    args = parser.parse_args()

    vid_path = Path(args.vid_path)
    img_path = Path(args.img_path)
    if img_path.exists():
        shutil.rmtree(img_path)

    if not img_path.exists():
        logging.info(f"Creating image directory {img_path}.")
        img_path.mkdir(parents=True, exist_ok=True)

    subprocess.call(["ffmpeg", "-i", str(vid_path), "-vframes", str(args.vframes),
                     "-vf", "scale=640:360", "-q:v", "2",
                     "-r", str(args.fps), str(img_path / "%05d.jpg")])


