import numpy as np
import argparse
from pathlib import Path
import logging
from collections import Counter
import scipy.signal as sig
import cv2
import tqdm
import itertools

import csv


def get_court_outline(im, threshold=140, connectivity=4):
    _, thresh = cv2.threshold(im, threshold, 255, cv2.THRESH_BINARY)
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    c = Counter(output[1].ravel())

    # the court component should always contain the second most pixels for
    # action shots
    court_idx = c.most_common()[1][0]
    connected = (output[1] == court_idx)
    return connected


def get_baseline(outline, top, peak_distance=10):
    """
    Starting from the top, look for rows of pixels that contain
    lots of non-zero values. This should peak for each horizontal
    line. The first/last of these should be the top/bottom baseline.

    The x coordinate of the top baseline is just the index of the first
    non-zero pixel in that row.
    """
    z = np.sum(outline, axis=1)
    pks = sig.find_peaks(z, distance=peak_distance)[0]
    pct = np.percentile(z, [95])[0]
    pk_idx = 0 if top else -1
    tall_peaks = list(filter(lambda x: z[x] > pct, pks))
    if len(tall_peaks) == 0:
        return (0, 0), (0, 0)
    pk_y = tall_peaks[pk_idx]
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
        ends = np.concatenate([ends, action_mask.shape[0]])

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

"""
python src/models/court_bounding_boxes.py
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-path", type=str)
    parser.add_argument("--frames-path", type=str)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--outline-threshold", type=int, default=150)

    args = parser.parse_args()

    mask_path = Path(args.mask_path)
    frames_path = Path(args.frames_path)
    save_path = Path(args.save_path) / "clips"
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=False)
    for mask in mask_path.iterdir():
        match_name = mask.stem
        save_name = save_path / (match_name + ".csv")
        if save_name.exists():
            print(f"skipping {save_name}")
            continue
        action_mask = np.load(mask).astype(int)
        frame_list = np.array(list((frames_path / match_name).iterdir()))

        clips = []
        for start_idx, end_idx in get_clip_indices(action_mask):
            clips.append(frame_list[start_idx:end_idx])
        all_clip_frames = [frame for clip in clips for frame in clip]

        coords = []
        for frame in all_clip_frames:
            img = cv2.imread(str(frame))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            court_outline = get_court_outline(img, threshold=args.outline_threshold)
            p1, p2 = get_baseline(court_outline, top=False, peak_distance=50)
            p4, p3 = get_baseline(court_outline, top=True, peak_distance=50)
            proposals = propose_bounding_boxes(p1, p2, p3, p4)
            coords.append(proposals)
        coords = np.array(coords)
        median_area = np.median([get_area(c) for c in coords.reshape(-1, 8)])

        best_boxes = []
        for proposals in coords:
            best_idx = np.argmin([abs(get_area(c) - median_area) for c in proposals])
            best_boxes.append(list(proposals[best_idx]))

        with open(save_name, 'w') as csvfile:
            boxwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
            clip_idx = 0
            for clip in clips:
                for frame in clip:
                    boxwriter.writerow([str(frame.stem), clip_idx] + best_boxes[clip_idx])
                clip_idx += 1

