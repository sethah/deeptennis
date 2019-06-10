import argparse
from typing import List

import cv2
import numpy as np
from typing import NamedTuple, Tuple
import logging
import json
import os
from pathlib import Path
import tqdm
import shutil

import deeptennis.utils as utils


class Box(NamedTuple):
    score: float
    coords: np.ndarray
    name: str


def choose_player_boxes(boxes: List[Box],
                        halfway_point: float,
                      min_score: float = 0.0) -> Tuple[Box, Box]:
    top_boxes = [b for b in boxes if (b.coords[3] <= halfway_point) and (b.score > min_score)]
    bottom_boxes = [b for b in boxes if (b.coords[3] > halfway_point) and (b.score > min_score)]
    srtd_top = sorted(top_boxes, key=lambda x: -x.score)
    srtd_bottom = sorted(bottom_boxes, key=lambda x: -x.score)
    top = None if len(srtd_top) == 0 else srtd_top[0]
    bottom = None if len(srtd_bottom) == 0 else srtd_bottom[0]
    return top, bottom


def tennis_rectangles():
    outer_court = [0, 0, 36, 78]
    inner_court = [4.5, 0, 27, 78]
    service1 = [4.5, 18, 13.5, 21]
    service2 = [18, 18, 13.5, 21]
    service3 = [4.5, 39, 13.5, 21]
    service4 = [18, 39, 13.5, 21]
    rects = [outer_court, inner_court, service1, service2, service3, service4]
    return rects


def player_boxes_to_court_points(box: np.ndarray, M: np.ndarray) -> np.ndarray:
    points = np.stack([(box[:, 2] - box[:, 0]) / 2 + box[:, 0], box[:, 3], np.ones(box.shape[0])], axis=0)
    converted_points = image_point_to_court_point(points, M)
    return converted_points


def image_point_to_court_point(points, M, inverse=True):
    if inverse:
        M = np.linalg.inv(M)
    transf_homg_point = M.dot(points)
    transf_homg_point /= transf_homg_point[2]
    return transf_homg_point[:2, :]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--tracking-path", type=str)
    parser.add_argument("--frame-path", type=str)

    args = parser.parse_args()

    save_path = Path(args.save_path)
    frame_path = Path(args.frame_path)
    tracking_path = Path(args.tracking_path)

    tracking_js = utils.read_json_lines(tracking_path)
    frames = sorted(frame_path.iterdir())

    clip_video_path = save_path / "images"
    clip_video_path.mkdir(parents=True, exist_ok=True)

    json_dict = {
        'image_paths': [],
        'is_action': [],
        'top_player': {
            'position': [],
            'box': [],
            'position_unwarped': [],
            'confidence': [],
        },
        'bottom_player': {
            'position': [],
            'box': [],
            'position_unwarped': [],
            'confidence': [],
        },
        'court': {
            'lower_left': [],
            'lower_right': [],
            'upper_right': [],
            'upper_left': [],
            'confidence': []
        }
    }
    for j, tracking in tqdm.tqdm(enumerate(tracking_js)):
        img_scale = np.array([640 / 512, 360 / 512])
        json_dict['image_paths'].append(str(frames[j].name))
        boxes: List[Box] = [Box(score, (np.array(coords).reshape(2, 2) * img_scale).ravel(), name) for score, coords, name in zip(tracking['box_scores'],
                                                                                    tracking['box_proposals'],
                                                                                    tracking['box_class'])]
        player_boxes = [box for box in boxes if box.name == 'player']
        court_boxes = [box for box in boxes if box.name == 'court']
        court_keypoints = np.array(tracking['keypoint_proposals'])
        keypoint_scores = np.array(tracking['keypoint_scores'])

        if len(court_boxes) > 0:
            best_court_box = max(court_boxes, key=lambda b: b.score)
            if best_court_box.score > 0.8:
                json_dict['is_action'].append(True)
            else:
                json_dict['is_action'].append(False)
        else:
            json_dict['is_action'].append(False)
        is_action = json_dict['is_action'][-1]
        if keypoint_scores.shape[0] > 0 and is_action:
            best_kp_index = keypoint_scores.sum(axis=1).argmax()
            _court = court_keypoints[best_kp_index]
            _court = _court.reshape(4, 3)[:, :2][[0, 1, 3, 2]]
            _court = _court * np.array([640 / 512, 360 / 512])
            json_dict['court']['lower_left'].append(tuple([int(x) for x in _court[0]]))
            json_dict['court']['lower_right'].append(tuple([int(x) for x in _court[1]]))
            json_dict['court']['upper_left'].append(tuple([int(x) for x in _court[2]]))
            json_dict['court']['upper_right'].append(tuple([int(x) for x in _court[3]]))
            json_dict['court']['confidence'].append(float(keypoint_scores.sum(axis=1)[best_kp_index]))
        else:
            _court = None
            json_dict['court']['lower_left'].append((-1, -1))
            json_dict['court']['lower_right'].append((-1, -1))
            json_dict['court']['upper_left'].append((-1, -1))
            json_dict['court']['upper_right'].append((-1, -1))
            json_dict['court']['confidence'].append(-1)

        if _court is not None:
            true_court = np.array([
                [0, 0],
                [36, 0],
                [36, 78],
                [0, 78]
            ])
            sorted_x = sorted(_court, key=lambda x: x[0])
            bottom_left = max(sorted_x[:2], key=lambda x: x[1])
            top_left = min(sorted_x[:2], key=lambda x: x[1])
            bottom_right = max(sorted_x[2:], key=lambda x: x[1])
            top_right = min(sorted_x[2:], key=lambda x: x[1])
            _court = [bottom_left, bottom_right, top_right, top_left]
            M = cv2.getPerspectiveTransform(true_court.astype(np.float32),
                                            np.array(_court).reshape(4, 2).astype(np.float32))
            halfway_point = image_point_to_court_point(np.array([0, 78 // 2, 1]).reshape(1, -1).T,
                                                   M, inverse=False)
            halfway_point = halfway_point.ravel()[1]
        else:
            halfway_point = None
        top_player_box, bottom_player_box = choose_player_boxes(player_boxes, min_score=0.01,
                                                                halfway_point=halfway_point or 360 // 2)
        if top_player_box is not None and is_action:
            x1, y1, x2, y2 = top_player_box.coords
            try:
                player_marker = player_boxes_to_court_points(top_player_box.coords.reshape(1, 4), M)
                xw1, yw1 = player_marker.ravel()
            except:
                xw1, yw1 = 0, 0
            json_dict['top_player']['box'].append([int(z) for z in [x1, y1, x2, y2]])
            json_dict['top_player']['position'].append((int((x1 + x2) / 2), int(y2)))
            json_dict['top_player']['position_unwarped'].append((int(xw1), int(yw1)))
            json_dict['top_player']['confidence'].append(top_player_box.score)
        else:
            json_dict['top_player']['box'].append([-1, -1, -1, -1])
            json_dict['top_player']['position'].append((-1, -1))
            json_dict['top_player']['position_unwarped'].append((-1, -1))
            json_dict['top_player']['confidence'].append(-1)
        if bottom_player_box is not None and is_action:
            x1, y1, x2, y2 = bottom_player_box.coords
            try:
                player_marker = player_boxes_to_court_points(bottom_player_box.coords.reshape(1, 4), M)
                xw1, yw1 = player_marker.ravel()
            except:
                xw1, yw1 = 0, 0
            json_dict['bottom_player']['box'].append([int(z) for z in [x1, y1, x2, y2]])
            json_dict['bottom_player']['position'].append((int((x1 + x2) / 2), int(y2)))
            json_dict['bottom_player']['position_unwarped'].append((int(xw1), int(yw1)))
            json_dict['bottom_player']['confidence'].append(bottom_player_box.score)
        else:
            json_dict['bottom_player']['box'].append([-1, -1, -1, -1])
            json_dict['bottom_player']['position'].append((-1, -1))
            json_dict['bottom_player']['position_unwarped'].append((-1, -1))
            json_dict['bottom_player']['confidence'].append(-1)
    with open(str(save_path / save_path.stem) + ".json", "w") as f:
        json.dump(json_dict, f)

    for i in tqdm.tqdm(range(len(frames))):

        img = cv2.imread(str(frames[i]))
        if json_dict['is_action'][i]:
            # draw court
            padding = 50
            scale = 2
            img = cv2.rectangle(img, (0, 0), (36 * scale + padding, 78 * scale + padding), (255, 255, 255), cv2.FILLED)
            for rect in tennis_rectangles():
                x, y, w, h = np.int32(rect) * scale
                x += padding // 2
                y += padding // 2
                img = cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 0), thickness=2)

            x, y = json_dict['top_player']['position_unwarped'][i]
            top_valid = (x, y) != (-1, -1)
            if top_valid:
                try:
                    img = cv2.circle(img, (x * scale + padding // 2, int(78 - y) * scale + padding // 2), 5, (0, 255, 0), -1)
                except:
                    pass
            x, y = json_dict['bottom_player']['position_unwarped'][i]
            bottom_valid = (x, y) != (-1, -1)
            if bottom_valid:
                try:
                    img = cv2.circle(img, (x * scale + padding // 2, int(78 - y) * scale + padding // 2), 5, (0, 255, 0), -1)
                except:
                    pass

            # draw top player box
            if top_valid:
                x1, y1, x2, y2 = json_dict['top_player']['box'][i]
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
                text = "%0.3f" % json_dict['top_player']['confidence'][i]
                position = (int(x1), int(y2) + 10)
                cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

            if bottom_valid:
                # bottom player
                x1, y1, x2, y2 = json_dict['bottom_player']['box'][i]
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
                text = "%0.3f" % json_dict['bottom_player']['confidence'][i]
                position = (int(x1), int(y2) + 10)
                cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        else:
            pass
        cv2.imwrite(str(clip_video_path / ("%05d.png" % i)), img)
    command = f"ffmpeg -y -r 8 -i {str(clip_video_path)}/%05d.png -c:v libx264 -q:v 2 -vf fps=8 -pix_fmt yuv420p {save_path / save_path.stem}.mp4"
    print(command)
    os.system(command)
    # shutil.rmtree(str(clip_video_path))
