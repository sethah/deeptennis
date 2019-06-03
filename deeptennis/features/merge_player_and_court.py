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
from deeptennis.features.extract_court_keypoints import get_court_for_frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", type=str)
    parser.add_argument("--anno-path", type=str)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--meta-file", type=str, default=None)

    args = parser.parse_args()
    # fileConfig('logging_config.ini')

    with open(args.meta_file, 'r') as f:
        match_metas = json.load(f)

    # mask_path = Path(args.mask_path)
    frames_path = Path(args.frames_path)
    anno_path = Path(args.anno_path)
    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()

    frames = list(frames_path.iterdir())
    court_boxes = []
    for frame in frames:
        img = cv2.imread(str(frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        match_name = "_".join(frame.stem.split("_")[:4])
        match_meta = match_metas[match_name]
        with open(anno_path / (frame.stem + ".json"), "r") as f:
            anno = json.load(f)
        court = get_court_for_frame(img, match_meta['court_crop']['x'],
                                      match_meta['court_crop']['y'],
                                      match_meta['min_horiz_line_dist'],
                                      match_meta['min_vert_line_dist'],
                                      match_meta['min_vert_slope'],
                                      match_meta['max_horiz_slope'],
                                      match_meta['max_baseline_offset'],
                                      match_meta['dilate_edges'])
        new_anno = {'filename': anno['filename']}
        img_annos = []
        court_anno = {}
        court_anno['num_keypoints'] = 4
        kp = [[int(x), int(y), 2] for x, y in [court[i:i+2] for i in range(0, len(court), 2)]]
        kp = [item for sublist in kp for item in sublist]
        court_anno['keypoints'] = kp
        court_anno['bbox'] = [int(court[0]), int(court[5]), int(court[2] - court[0]), int(court[1] - court[5])]
        court_anno['class'] = 'court'
        img_annos.append(court_anno)

        for player_anno in anno['regions']:
            a = player_anno['shape_attributes']
            img_annos.append({'num_keypoints': 0,
                              'bbox': [a['x'], a['y'], a['width'], a['height']],
                              'class': 'player'
                              })
        new_anno['objects'] = img_annos
        with open(save_path / (frame.stem + ".json"), 'w') as f:
            json.dump(new_anno, f)

    # json_lines = []
    # for f, coords in zip(frame_list, court_boxes):
    #     json_lines.append({'filename': str(f), 'court': coords})
    # utils.write_json_lines(json_lines, save_path)
