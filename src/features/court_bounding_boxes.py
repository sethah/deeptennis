import sys
import numpy as np
import argparse
from pathlib import Path
import logging
from logging.config import fileConfig
from collections import Counter
import scipy.signal as sig
import cv2
from tqdm import tqdm
import itertools

import csv

from src import utils


def get_court_outline(im, threshold=10, sensitivity=[200, 200, 200]):
    lower_white = np.array(sensitivity, dtype=np.uint8)
    upper_white = np.array([255,255,255], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(im, lower_white, upper_white)
    # Bitwise-AND mask and original image
    white = cv2.bitwise_and(im, im, mask= mask)
    gray = cv2.cvtColor(white, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresh


def get_baseline(outline, peak_distance=10):
    """
    Detect the bottom baseline and bottom service line from a thresholded image outline.
    """
    z = np.sum(outline, axis=1)
    halfway = z.shape[0] // 2
    z = z[halfway:]
    pks = sig.find_peaks(z, distance=peak_distance)[0]

    if len(pks) < 2:
        return 0, 0, 0, 0
    # take the first two peaks to be the service and baseline
    bottom_baseline_idx = pks[1]
    bottom_service_idx = pks[0]
    bottom_baseline_idx += halfway
    bottom_service_idx += halfway
    pixel_window = 5
    window_top = min(bottom_baseline_idx + pixel_window, outline.shape[0])
    window_bottom = max(bottom_baseline_idx - pixel_window, 0)
    cropped = outline[window_bottom:window_top, :]
    cropped_sum = cropped.sum(axis=0) >= 2 * 255
    x1 = np.argmax(cropped_sum)
    x2 = outline.shape[1] - np.argmax(cropped_sum[::-1]) - 1
    bottom_y1 = np.argmax(cropped[:, x1]) + window_bottom
    bottom_y2 = np.argmax(cropped[:, x2]) + window_bottom
    return x1, x2, bottom_y1, bottom_y2, bottom_service_idx


def compute_slope(zoom, p0):
    j = p0[0] - 1
    slopes = []
    while j >= 0 and j < zoom.shape[0] and np.sum(zoom[j]) > 0:
        i = np.argmax(zoom[j])
        slope = (j - p0[0]) / (i - p0[1])
        slopes.append(slope)
        j -= 1
    slopes = np.array(slopes)
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) == 0:
        return 0
    return np.median(slopes)


def get_slope(court_outline, y_bot, x_bot, crops=(70, 10, 20, 70), flip=False):
    """
    Zoom in on the lower corners of the court, and compute the slope of the sidelines
    :param court_outline:
    :param y_bot:
    :param x_bot:
    :param crops:
    :param flip:
    :return:
    """
    if flip:
        zoom = court_outline[y_bot-crops[0]:y_bot+crops[1], x_bot-crops[3]:x_bot+crops[2]]
        zoom = np.fliplr(zoom)
        y0 = crops[0]
        x0 = crops[2]
    else:
        zoom = court_outline[y_bot-crops[0]:y_bot+crops[1], x_bot-crops[2]:x_bot+crops[3]]
        y0 = crops[0]
        x0 = crops[2]
    slope = compute_slope(zoom, (y0, x0))
    return slope


def get_top_corner(outline, bottom_x1, bottom_x2, bottom_y, serve_y, left=True):
    """
    Use vanishing point geometry to compute the top of the court from the bottom
    and the service line.
    :param outline:
    :param bottom_x1:
    :param bottom_x2:
    :param bottom_y:
    :param serve_y:
    :param left:
    :return:
    """
    bottom_x = bottom_x1 if left else bottom_x2
    slope = get_slope(outline, bottom_y, bottom_x, flip=not left, crops=[100, 10, 20, 50])
    if slope == 0:
        return 0, 0
    w = abs(bottom_x2 - bottom_x1)
    x4 = bottom_x1 + (serve_y - bottom_y) * 1 / slope
    x3 = bottom_x2 - (serve_y - bottom_y) * 1 / slope
    Vy = bottom_y - w / 2 * -slope
    Vx = bottom_x1 + w / 2
    serve_x = x4 if left else x3
    AV = np.sqrt((bottom_x - Vx)**2 + (bottom_y - Vy)**2)
    AB = np.sqrt((bottom_x - serve_x)**2 + (bottom_y - serve_y)**2)
    C = 78. / 60  # dimensions of a real tennis court
    l = (AV - AB) * AB / (AV * C - (AV - AB))
    theta = np.arctan(-1 / slope)
    x_top = serve_x + l * np.sin(theta) * (1 if left else -1)
    y_top = serve_y - l * np.cos(theta)
    return x_top, y_top


def leave_one_out(p1, p2, p3):
    """
    Predict the fourth court corner from three others.

    :param p1: With p2, makes up one baseline.
    :param p2: With p1, makes up one baseline.
    :param p3: One point from the other baseline. Its partner will be predicted.
    :return: tuple(int, int) -> (x4, y4)
    """
    pp1 = p1 if p1[0] < p2[0] else p2
    pp2 = p2 if p1[0] < p2[0] else p1
    midpoint = pp1[0] + (pp2[0] - pp1[0]) / 2
    if p3[0] < midpoint:
        x_interp = midpoint + (midpoint - p3[0])
    else:
        x_interp = midpoint - (p3[0] - midpoint)
    y_interp = p3[1]
    return int(x_interp), int(y_interp)


def get_clip_indices(mask):
    starts = np.where(np.diff(mask) > 0)[0]
    ends = np.where(np.diff(mask) < 0)[0]
    if action_mask[0] == 1:
        starts = np.concatenate([[0], starts])
    if action_mask[-1] == 1:
        ends = np.concatenate([ends, [action_mask.shape[0]]])

    return list(zip(starts + 1, ends + 1))


def propose_bounding_boxes(p1, p2, p3, p4):
    interp_points = [[p1, p2, p3], [p1, p2, p4], [p3, p4, p1], [p3, p4, p2]]
    interp_points = [leave_one_out(*pts) for pts in interp_points]
    verts = [p1, p2, p3, p4]
    proposals = [list(itertools.chain(*verts))]
    vertices = verts.copy()
    for i, p in enumerate(interp_points):
        vertices[-i - 1] = p
        proposals.append(list(itertools.chain(*vertices)))
        vertices = verts.copy()
    return proposals


def get_area(crd):
    a = abs(crd[4] - crd[6])
    b = abs(crd[0] - crd[2])
    h = abs(crd[7] - crd[1])
    return 0.5 * (a + b) * h

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-path", type=str)
    parser.add_argument("--frames-path", type=str)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--meta-file", type=str, default=None)
    parser.add_argument("--outline-threshold", type=int, default=150)

    args = parser.parse_args()
    fileConfig('logging_config.ini')

    match_meta = utils.get_match_metadata(Path(args.meta_file))

    mask_path = Path(args.mask_path)
    frames_path = Path(args.frames_path)
    save_path = Path(args.save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir()

    match_name = mask_path.stem
    action_mask = np.load(mask_path).astype(int)
    frame_list = np.array(list(sorted(frames_path.iterdir())))

    clips = []
    for start_idx, end_idx in get_clip_indices(action_mask):
        clips.append(frame_list[start_idx:end_idx])
    all_clip_frames = [frame for clip in clips for frame in clip]

    sensitivity = match_meta[match_name]['sensitivity']
    sensitivity = np.array([sensitivity, sensitivity, sensitivity])
    peak_distance = match_meta[match_name]['peak_distance']
    crop_top = match_meta[match_name]['crop_top']
    crop_left = match_meta[match_name]['crop_left']
    crop_right = match_meta[match_name]['crop_right']
    crop_bottom = match_meta[match_name]['crop_bottom']

    logging.debug(f"Begin bounding box detection for {match_name}")
    coords = []
    invalids = 0
    for frame in tqdm(all_clip_frames):
        img = cv2.imread(str(frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        court_outline = get_court_outline(img, 10, sensitivity=sensitivity)
        im_h, im_w = court_outline.shape
        cropped_outline = court_outline[crop_top:im_h - crop_bottom, crop_left:im_w - crop_right]
        x1, x2, y1_bot, y2_bot, y_serve = get_baseline(cropped_outline, peak_distance=peak_distance)
        x1 += crop_left
        x2 += crop_left
        y1_bot += crop_top
        y2_bot += crop_top
        y_serve += crop_top
        top_left = get_top_corner(court_outline, x1, x2, y1_bot, y_serve)
        top_right = get_top_corner(court_outline, x1, x2, y2_bot, y_serve, left=False)
        p1 = x1, y1_bot
        p2 = x2, y2_bot
        p3 = top_right
        p4 = top_left
        if not utils.validate_court_box(p1, p2, p3, p4, im_w, im_h):
            invalids += 1
        proposals = propose_bounding_boxes(p1, p2, p3, p4)
        coords.append(proposals)
    logging.debug(f"{invalids}/{len(all_clip_frames)} were invalid")
    coords = np.array(coords)
    median_area = np.median([get_area(c) for c in coords.reshape(-1, 8)])

    best_boxes = []
    for proposals in coords:
        # ensure that it always chooses a valid proposal if it exists
        filtered = [p for p in proposals if
                    utils.validate_court_box(*np.array(p).reshape(4, 2), im_w, im_h)]
        if len(filtered) == 0:
            # no valid proposals
            best_boxes.append(list(proposals[0]))
        else:
            best_idx = np.argmin([abs(get_area(c) - median_area) for c in filtered])
            best_boxes.append(list(filtered[best_idx]))
    logging.debug(f"End bounding box detection for {match_name}")

    with open(save_path, 'w') as csvfile:
        boxwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        frame_idx = 0
        for clip_idx, clip in enumerate(clips):
            for frame in clip:
                boxwriter.writerow([str(frame.stem), clip_idx] + best_boxes[frame_idx])
                frame_idx += 1

