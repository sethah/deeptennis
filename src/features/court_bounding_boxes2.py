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


def get_court_outline_old(im, threshold=140, connectivity=4, sensitivity=100):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(im, lower_white, upper_white)
    # Bitwise-AND mask and original image
    white = cv2.bitwise_and(im, im, mask= mask)
    gray = cv2.cvtColor(cv2.cvtColor(white, cv2.COLOR_HSV2RGB), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    c = Counter(output[1].ravel())

    # the court component should always contain the second most pixels for
    # action shots
    court_idx = c.most_common()[1][0]
    connected = (output[1] == court_idx)
    return connected

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

def get_baseline(outline, peak_distance=10, percentile=95):
    """
    Starting from the top, look for rows of pixels that contain
    lots of non-zero values. This should peak for each horizontal
    line. The first/last of these should be the top/bottom baseline.

    The x coordinate of the top baseline is just the index of the first
    non-zero pixel in that row.
    """
    z = np.sum(outline, axis=1)
    halfway = z.shape[0] // 2
    z = z[halfway:]
    pks = sig.find_peaks(z, distance=peak_distance)[0]

    if len(pks) < 2:
        return 0, 0, 0, 0
    srtd = np.argsort(z[pks] * -1)
    bottom_baseline_idx = pks[srtd[0]]
    bottom_service_idx = pks[srtd[1]]
    bottom_baseline_idx += halfway
    bottom_service_idx += halfway
    pixel_window = 5
    best_x1 = np.argmax(outline[bottom_baseline_idx])
    window_top = min(bottom_baseline_idx + pixel_window, outline.shape[0])
    window_bottom = max(bottom_baseline_idx - pixel_window, 0)
    for i in range(window_bottom, window_top):
        tmp = np.argmax(outline[i])
        if tmp > 0:
            best_x1 = min(best_x1, tmp)
    outline_flipped = np.fliplr(outline)
    best_x2 = np.argmax(outline_flipped[bottom_baseline_idx])
    for i in range(window_bottom, window_top):
        tmp = np.argmax(outline_flipped[i])
        if tmp > 0:
            best_x2 = min(best_x2, tmp)
    best_x2 = outline.shape[1] - best_x2
    return best_x1, best_x2, bottom_baseline_idx, bottom_service_idx


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
    # pcts = np.percentile(slopes, [25, 75])
    # return np.mean(slopes[(slopes >= pcts[0]) & (slopes <= pcts[1])])


def get_slope(court_outline, y_bot, x_bot, crops=[50, 10, 20, 50], flip=False):
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
    C = 78. / 60
    l = (AV - AB) * AB / (AV * C - (AV - AB))
    theta = np.arctan(-1 / slope)
    x_top = serve_x + l * np.sin(theta) * (1 if left else -1)
    y_top = serve_y - l * np.cos(theta)
    return x_top, y_top


def get_baseline_old(outline, top, peak_distance=10, percentile=95):
    """
    Starting from the top, look for rows of pixels that contain
    lots of non-zero values. This should peak for each horizontal
    line. The first/last of these should be the top/bottom baseline.

    The x coordinate of the top baseline is just the index of the first
    non-zero pixel in that row.
    """
    z = np.sum(outline, axis=1)
    halfway = z.shape[0] // 2
    if top:
        z = z[:halfway]
    else:
        z = z[halfway:]
    pks = sig.find_peaks(z, distance=peak_distance)[0]
    pct = np.percentile(z, [percentile])[0]
    pk_idx = 0 if top else -1
    tall_peaks = list(filter(lambda x: z[x] > pct, pks))
    if len(tall_peaks) == 0:
        return (0, 0), (0, 0)
    pk_y = tall_peaks[pk_idx]
    if not top:
        pk_y += halfway
    x1 = np.argmax(outline[pk_y])
    x2 = outline.shape[1] - np.argmax(np.fliplr(outline)[pk_y])
    y1 = y2 = pk_y
    return (x1, y1), (x2, y2)


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

    # assert starts.shape[0] == ends.shape[0]
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
    a = abs(crd[5] - crd[7])
    b = abs(crd[0] - crd[2])
    h = abs(crd[6] - crd[1])
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

    match_name = mask_path.stem
    action_mask = np.load(mask_path).astype(int)
    frame_list = np.array(list(sorted(frames_path.iterdir())))

    clips = []
    for start_idx, end_idx in get_clip_indices(action_mask):
        clips.append(frame_list[start_idx:end_idx])
    all_clip_frames = [frame for clip in clips for frame in clip]

    sensitivity = match_meta[match_name]['sensitivity']
    sensitivity = np.array([sensitivity, sensitivity, sensitivity])
    threshold = match_meta[match_name]['threshold']
    peak_distance = match_meta[match_name]['peak_distance']
    percentile = match_meta[match_name]['percentile']

    logging.debug(f"Begin bounding box detection for {match_name}")
    coords = []
    for frame in tqdm(all_clip_frames):
        img = cv2.imread(str(frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        court_outline = get_court_outline(img, 0, sensitivity=sensitivity)
        x1, x2, y_bot, y_serve = get_baseline(court_outline, peak_distance=peak_distance,
                                              percentile=percentile)
        top_left = get_top_corner(court_outline, x1, x2, y_bot, y_serve)
        top_right = get_top_corner(court_outline, x1, x2, y_bot, y_serve, left=False)
        # p1, p2 = get_baseline(court_outline, top=False, peak_distance=peak_distance, percentile=percentile)
        # p4, p3 = get_baseline(court_outline, top=True, peak_distance=peak_distance, percentile=percentile)
        p1 = x1, y_bot
        p2 = x2, y_bot
        p3 = top_right
        p4 = top_left
        proposals = propose_bounding_boxes(p1, p2, p3, p4)
        coords.append(proposals)
    coords = np.array(coords)
    median_area = np.median([get_area(c) for c in coords.reshape(-1, 8)])

    best_boxes = []
    for proposals in coords:
        best_idx = np.argmin([abs(get_area(c) - median_area) for c in proposals])
        best_boxes.append(list(proposals[best_idx]))
    logging.debug(f"End bounding box detection for {match_name}")

    with open(save_path, 'w') as csvfile:
        boxwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        frame_idx = 0
        for clip_idx, clip in enumerate(clips):
            for frame in clip:
                boxwriter.writerow([str(frame.stem), clip_idx] + best_boxes[frame_idx])
                frame_idx += 1

