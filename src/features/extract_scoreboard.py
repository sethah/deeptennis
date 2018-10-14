import sys
import json
import numpy as np
import argparse
from pathlib import Path
import logging
from logging.config import fileConfig
import scipy.signal as sig
import cv2
from tqdm import tqdm
import itertools
import pickle

import csv

from src import utils


def dilate_image(image, thresh_low=180):
    resized = image
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
    gray = gray - opening
    ret, mask = cv2.threshold(gray, thresh_low, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(gray, gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, thresh_low, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(new_img, kernel, iterations=2)
    return dilated, gray


def find_text(dilated, min_w=5, min_h=5):
    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mx_right = 0
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if w < min_w or h < min_h:
            continue
        mx_right = max(mx_right, x + w)
    return mx_right

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", type=str)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--meta-file", type=str, default=None)
    parser.add_argument("--outline-threshold", type=int, default=150)

    args = parser.parse_args()
    fileConfig('logging_config.ini')

    with open(args.meta_file, 'r') as f:
        match_metas = json.load(f)

    frames_path = Path(args.frames_path)
    save_path = Path(args.save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir()

    match_name = frames_path.stem
    match_meta = match_metas.get(match_name, None)
    if match_meta is None:
        sys.exit(0)
    frame_list = np.array(list(sorted(frames_path.iterdir())))

    x, y, w, h = match_meta['box']
    invert = match_meta['invert']
    min_w, min_h = match_meta['min_score_text_width'], match_meta['min_score_text_height']
    thresh_low = match_meta['score_thresh_low']
    min_score_width = match_meta['min_score_width']
    logging.debug(f"Begin bounding box detection for {match_name}")
    score_boxes = []
    for frame in frame_list:
        img = cv2.imread(str(frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[y:y + h, x:x + w].astype(np.int32)
        img = 255 - img if invert else img
        img = img.astype(np.uint8)
        dilated, g = dilate_image(img, thresh_low=thresh_low)
        score_width = find_text(dilated, min_w=min_w, min_h=min_h)
        if score_width < min_score_width:
            score_boxes.append([0, 0, 0, 0])
        else:
            score_boxes.append([x, y, score_width, h])

    save_list = list(zip([f.name for f in frame_list], score_boxes))
    with open(save_path, 'wb') as save_file:
        pickle.dump(save_list, save_file)
