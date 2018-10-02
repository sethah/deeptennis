import argparse
import logging
from logging.config import fileConfig
import os
from pathlib import Path
import shutil

from src.data.clip import Clip, Video
from src.vision.transforms import *
import src.utils as utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--clip-path", type=str)
    parser.add_argument("--frame-path", type=str)

    fileConfig('logging_config.ini')

    args = parser.parse_args()

    save_path = Path(args.save_path)
    frame_path = Path(args.frame_path)
    clip_path = Path(args.clip_path)

    clip_video_path = save_path.parent / save_path.stem
    clip_video_path.mkdir(parents=True)
    logging.debug(f"Processing {clip_video_path.stem}")

    video = Video(frame_path)
    clips = Clip.from_csv(clip_path, video)
    j = 0
    for clip_idx, clip in enumerate(clips):
        for bbox, frame in zip(clip.bboxes, clip.frames):
            img = cv2.imread(str(frame))
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (100, 100)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2

            cv2.putText(img, str(clip_idx),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            box_valid = utils.validate_court_box(*bbox.reshape(4, 2), img.shape[1], img.shape[0])
            lines_color = (0, 255, 255) if box_valid else (0, 255, 0)
            img = cv2.polylines(img, np.int32([bbox.reshape(4, 2)]), True, lines_color, 3)
            cv2.imwrite(str(clip_video_path / ("%05d.jpg" % j)), img)
            j += 1
    os.system(
        f"ffmpeg -r 1 -i {str(clip_video_path)}/%05d.jpg -vcodec mpeg4 -y {clip_video_path.parent / clip_video_path.stem}.mp4")
    shutil.rmtree(str(clip_video_path))



