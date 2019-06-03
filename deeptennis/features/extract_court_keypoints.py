import json
import numpy as np
import argparse
from pathlib import Path
import logging
from logging.config import fileConfig
import cv2
import pickle
from typing import List

import deeptennis.utils as utils


def mask_image(img: np.ndarray, pts: np.ndarray, dilate=False):
    """
    Zero out the irrelevant part of the court and do edge detection.

    TODO: separate the crop and the edge detection
    :param img: original RGB image
    :param pts: vertices of the cropping area. Pixels outside this polygon will be zeroed.
    :param dilate: dilate image before
    :return: cropped/edged image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray, 50, 200)
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        cannyed_image = cv2.dilate(cannyed_image, kernel=kernel, iterations=2)
    mask = np.zeros(img.shape[:2], dtype=np.int32)
    match_mask_color = 255
    cv2.fillPoly(mask, [pts], match_mask_color)
    return cv2.bitwise_and(cannyed_image.astype(np.int32), mask)


def get_lines(lines, min_vert_len=100, min_horiz_len=100, min_vert_slope=1.5, max_horiz_slope=0.01):
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        distance = np.sqrt((line[0] - line[2])**2 + (line[1] - line[3])**2)
        slope = (line[1] - line[3]) / ((line[0] - line[2]) + 1e-6)
        if distance > min_horiz_len and abs(slope) < max_horiz_slope:
            horizontal_lines.append(line)
        elif distance > min_vert_len and abs(slope) > min_vert_slope:
            vertical_lines.append(line)
    horizontal_lines = np.array(horizontal_lines)
    vertical_lines = np.array(vertical_lines)
    return horizontal_lines, vertical_lines


def slope(x1: float, y1: float, x2: float, y2: float):
    if x2 == x1:
        return 1000000.
    else:
        return (y2 - y1) / (x2 - x1)


def get_baseline_vertical(horizontal_lines, min_separation=30, max_separation=100):
    if len(horizontal_lines) == 0:
        return 0, 0
    base = max([l[1] for l in horizontal_lines])
    no_base = [l[1] for l in horizontal_lines if abs(l[1] - base) > min_separation \
               and abs(l[1] - base) <= max_separation]
    if len(no_base) == 0:
        return base, 0
    serve = max(no_base)
    return base, serve


def get_sidelines(vertical_lines):
    if len(vertical_lines) == 0:
        return np.zeros(4), np.zeros(4)
    intercepts = [l[0] + (360 - l[1]) / slope(*l) for l in vertical_lines]
    sorted_intercepts = np.argsort(intercepts)[::-1]

    # right sideline should have positive slope, left negative
    sorted_intercepts_pos = [i for i in sorted_intercepts if slope(*vertical_lines[i]) > 0]
    sorted_intercepts_neg = [i for i in sorted_intercepts if slope(*vertical_lines[i]) <= 0]
    if len(sorted_intercepts_neg) == 0 or len(sorted_intercepts_pos) == 0:
        return np.zeros(4), np.zeros(4)
    max_intercept = intercepts[sorted_intercepts_pos[0]]
    min_intercept = intercepts[sorted_intercepts_neg[-1]]
    # group similar lines
    # TODO: their slopes should be similar too
    right_candidates = [i for i in sorted_intercepts_pos if max_intercept - intercepts[i] < 20]
    left_candidates = [i for i in sorted_intercepts_neg[::-1] if intercepts[i] - min_intercept < 20]

    # take median of candidates
    right = vertical_lines[right_candidates[len(right_candidates) // 2]]
    left = vertical_lines[left_candidates[len(left_candidates) // 2]]
    return right, left


def get_keypoints_horizontal(y_base, y_serve, left_sideline, right_sideline):
    x_base_left = left_sideline[0] + (y_base - left_sideline[1]) / slope(*left_sideline)
    x_base_right = right_sideline[0] + (y_base - right_sideline[1]) / slope(*right_sideline)
    x_serve_left = left_sideline[0] + (y_serve - left_sideline[1]) / slope(*left_sideline)
    x_serve_right = right_sideline[0] + (y_serve - right_sideline[1]) / slope(*right_sideline)
    return x_base_left, x_base_right, x_serve_right, x_serve_left


def get_top_corner(x_base, y_base, x_serve, y_serve, x_base_opp):
    w = abs(x_base - x_base_opp)
    m = 1 * slope(x_base, y_base, x_serve, y_serve)
    x4 = x_base + (y_serve - y_base) * 1 / m
    x3 = x_base_opp - (y_serve - y_base) * 1 / m
    Vy = y_base - w / 2 * -m
    Vx = x_base + w / 2
    AV = np.sqrt((x_base - Vx)**2 + (y_base - Vy)**2)
    AB = np.sqrt((x_base - x_serve)**2 + (y_base - y_serve)**2)
    C = 78. / 60  # dimensions of a real tennis court
    l = (AV - AB) * AB / (AV * C - (AV - AB))
    theta = np.arctan(-1 / m)
    x_top = x_serve + l * np.sin(theta)
    y_top = y_serve - l * np.cos(theta)
    return x_top, y_top

def get_court_for_frame(frame: np.ndarray,
                        court_crop_x: List[float],
                        court_crop_y: List[float],
                        min_horiz_line_dist: int,
                        min_vert_line_dist: int,
                        min_vert_slope: float,
                        max_horiz_slope: float,
                        max_baseline_offset: float,
                        dilate_edges: bool = False) -> List[float]:
    im_h, im_w = frame.shape[:2]
    crop_points = np.array([court_crop_x, court_crop_y], dtype=np.int32).T
    masked_image = mask_image(frame.astype(np.uint8), crop_points, dilate=dilate_edges)
    lines = cv2.HoughLinesP(
        masked_image.astype(np.uint8),
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
    court = [0] * 8
    if lines is None:
        return court
    lines = lines[:, 0, :]
    horizontal_lines, vertical_lines = get_lines(lines,
                                                 min_horiz_len=min_horiz_line_dist,
                                                 min_vert_len=min_vert_line_dist,
                                                 min_vert_slope=min_vert_slope,
                                                 max_horiz_slope=max_horiz_slope)
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return court
    y_base, y_serve = get_baseline_vertical(horizontal_lines, min_separation=30)
    if im_h - y_base > max_baseline_offset or y_base == 0 or y_serve == 0:
        return court
    right_sideline, left_sideline = get_sidelines(vertical_lines)
    if not right_sideline.any() or not left_sideline.any():
        return court
    x1, x2, x3, x4 = get_keypoints_horizontal(y_base, y_serve, left_sideline, right_sideline)
    y1, y2, y3, y4 = y_base, y_base, y_serve, y_serve
    x6, y6 = get_top_corner(x1, y1, x4, y4, x2)
    x5, y5 = get_top_corner(x2, y2, x3, y3, x1)
    return [float(x) for x in [x1, y1, x2, y2, x5, y5, x6, y6]]

def get_court_keypoints(frames: List[Path],
                        mask: np.ndarray,
                        court_crop_x: List[float],
                        court_crop_y: List[float],
                        min_horiz_line_dist: int,
                        min_vert_line_dist: int,
                        min_vert_slope: float,
                        max_horiz_slope: float,
                        max_baseline_offset: float,
                        dilate_edges: bool = False
                        ) -> List[List[float]]:
    court_boxes = []
    num_invalid = 0
    for i, frame in enumerate(frames):
        img = cv2.imread(str(frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        court = get_court_for_frame(img, court_crop_x, court_crop_y, min_horiz_line_dist,
                                    min_vert_line_dist, min_vert_slope, max_horiz_slope,
                                    max_baseline_offset, dilate_edges)
        court_boxes.append(court)
    logging.debug(f"{num_invalid}/{np.sum(mask)} were invalid.")
    return court_boxes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-path", type=str)
    parser.add_argument("--frames-path", type=str)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--meta-file", type=str, default=None)

    args = parser.parse_args()
    fileConfig('logging_config.ini')

    with open(args.meta_file, 'r') as f:
        match_metas = json.load(f)

    mask_path = Path(args.mask_path)
    frames_path = Path(args.frames_path)
    save_path = Path(args.save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir()

    mask = utils.read_json_lines(mask_path)
    mask = np.array([x['action'] for x in mask])

    match_name = frames_path.stem
    match_meta = match_metas[match_name]
    frame_list = list(sorted(frames_path.iterdir()))
    court_boxes: List[List[float]] = get_court_keypoints(frame_list, mask, match_meta['court_crop']['x'],
                                      match_meta['court_crop']['y'],
                                      match_meta['min_horiz_line_dist'],
                                      match_meta['min_vert_line_dist'],
                                      match_meta['min_vert_slope'],
                                      match_meta['max_horiz_slope'],
                                      match_meta['max_baseline_offset'],
                                      match_meta['dilate_edges'])

    json_lines = []
    for f, coords in zip(frame_list, court_boxes):
        json_lines.append({'filename': str(f), 'court': coords})
    utils.write_json_lines(json_lines, save_path)
