import argparse
from logging.config import fileConfig
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--frame-path", type=str)

    args = parser.parse_args()

    save_path = Path(args.save_path)
    frame_path = Path(args.frame_path)

    os.system(
        f"ffmpeg -r 1 -i {str(frame_path)}/%05d.jpg -vcodec mpeg4 -y {save_path / frame_path.stem}.mp4")



