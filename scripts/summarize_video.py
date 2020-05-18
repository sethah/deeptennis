import argparse
from typing import List

import cv2
import numpy as np
from typing import *
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

class Keypoint(NamedTuple):
    x: float
    y: float
    visible: float
    score: float

class KeypointSet(NamedTuple):
    keypoints: List[Keypoint]

class FrameSummary(object):
    @staticmethod
    def is_action_frame(court_box_proposals: List[Box]) -> bool:
        if len(court_box_proposals) > 0:
            best_court_box = max(court_box_proposals, key=lambda b: b.score)
            return best_court_box.score > 0.8
        else:
            return False
    @staticmethod
    def from_json(data: Dict[str, Any]):
        img_scale = np.array([640 / 512, 360 / 512])
        boxes = [Box(b, (np.array(a).reshape(2, 2) * img_scale).ravel(), c) for a, b, c in zip(data['box_proposals'], data['box_scores'], data['box_class'])]
        player_boxes = [box for box in boxes if box.name == 'player']
        court_boxes = [box for box in boxes if box.name == 'court']
        kp_sets = []
        for kps, scores in zip(data['keypoint_proposals'], data['keypoint_scores']):
            kp = [Keypoint(p[0], p[1], p[2], score) for p, score in zip(kps, scores)]
            kp_sets.append(KeypointSet(kp))
        if FrameSummary.is_action_frame(court_boxes):
            return ActionFrameSummary(data['image_sizes'],
                                      player_boxes,
                                      kp_sets)
        else:
            return NonActionFrameSummary()


class NonActionFrameSummary(FrameSummary):
    def __init__(self):
        self.is_action = False

    def to_json(self):
        return {
            'is_action': self.is_action,
            'top_player_box': [],
            'top_player_position': (),
            'bottom_player_box': [],
            'bottom_player_position': (),
        }


class ActionFrameSummary(FrameSummary):

    def __init__(self,
                 image_size: Tuple[float, float],
                 box_proposals: List[Box],
                 keypoint_proposals: List[KeypointSet],
                 ):
        self._image_size = image_size
        self._box_proposals = box_proposals
        self._keypoint_proposals = keypoint_proposals
        self.is_action = True
        self.court_confidence, self.court = ActionFrameSummary.court_location(keypoint_proposals)
        self._M = None if self.court is None else ActionFrameSummary.perspective_matrix(self.court)
        self.halfway_court = None if self._M is None else ActionFrameSummary.get_halfway_court(self._M)
        top_box, bottom_box = ActionFrameSummary.get_player_boxes(box_proposals, self.halfway_court)
        self.top_player_box = top_box
        self.bottom_player_box = bottom_box

    @staticmethod
    def perspective_matrix(court_location: np.ndarray) -> np.ndarray:
        true_court = np.array([
            [0, 0],
            [36, 0],
            [36, 78],
            [0, 78]
        ])
        sorted_x = sorted(court_location, key=lambda x: x[0])
        bottom_left = max(sorted_x[:2], key=lambda x: x[1])
        top_left = min(sorted_x[:2], key=lambda x: x[1])
        bottom_right = max(sorted_x[2:], key=lambda x: x[1])
        top_right = min(sorted_x[2:], key=lambda x: x[1])
        _court = [bottom_left, bottom_right, top_right, top_left]
        return cv2.getPerspectiveTransform(true_court.astype(np.float32),
                                        np.array(_court).reshape(4, 2).astype(np.float32))

    @staticmethod
    def get_halfway_court(perspective_matrix: np.ndarray) -> Optional[float]:
        halfway_point = ActionFrameSummary.image_point_to_court_point(
            np.array([0, 78 // 2, 1]).reshape(1, -1).T, perspective_matrix,
            inverse=False)
        return halfway_point.ravel()[1]

    @staticmethod
    def court_location(keypoint_proposals: List[KeypointSet]) -> Optional[Tuple[float, np.ndarray]]:
        if len(keypoint_proposals) > 0:
            best_kp_index = np.array([sum(x.score for x in kp.keypoints) for kp in keypoint_proposals]).argmax()
            best_proposal = keypoint_proposals[best_kp_index]
            _court = np.array([[kp.x, kp.y, kp.visible] for kp in best_proposal.keypoints])
            _court = _court.reshape(4, 3)[:, :2][[0, 1, 3, 2]]
            _court = _court * np.array([640 / 512, 360 / 512])
            return (sum([x.score for x in best_proposal.keypoints]), _court)
        else:
            return None, None

    @staticmethod
    def get_player_boxes(box_proposals: List[Box], halfway_point: Optional[float] = None) -> Optional[Box]:
        player_boxes = [b for b in box_proposals if b.name == "player"]
        a, b = ActionFrameSummary.choose_player_boxes(
            player_boxes, min_score=0.01, halfway_point=halfway_point or 360 // 2)
        return a, b

    @property
    def top_player_position(self) -> Optional[Tuple[float, float]]:
        if self.top_player_box:
            return self._box_to_position(self.top_player_box)

    @property
    def bottom_player_position(self) -> Optional[Tuple[float, float]]:
        if self.bottom_player_box:
            return self._box_to_position(self.bottom_player_box)

    @property
    def bottom_player_position_warped(self):
        if self.bottom_player_box is None:
            return None
        try:
            player_marker = self.player_boxes_to_court_points(self.bottom_player_box.coords.reshape(1, 4), self._M)
            xw1, yw1 = player_marker.ravel()
            return xw1, yw1
        except np.linalg.LinAlgError:
            return None

    @property
    def top_player_position_warped(self):
        if self.top_player_box is None:
            return None
        try:
            player_marker = self.player_boxes_to_court_points(self.top_player_box.coords.reshape(1, 4), self._M)
            xw1, yw1 = player_marker.ravel()
            return xw1, yw1
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _box_to_position(box: Box) -> Tuple[float, float]:
        return float((box.coords[0] + box.coords[2]) / 2), float(box.coords[3])

    @staticmethod
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

    @staticmethod
    def player_boxes_to_court_points(box: np.ndarray, M: np.ndarray) -> np.ndarray:
        # print(box)
        points = np.stack([(box[:, 2] - box[:, 0]) / 2 + box[:, 0], box[:, 3], np.ones(box.shape[0])], axis=0)
        # print(points)
        converted_points = ActionFrameSummary.image_point_to_court_point(points, M)
        # print(converted_points)
        return converted_points

    @staticmethod
    def image_point_to_court_point(points, M, inverse=True):
        if inverse:
            M = np.linalg.inv(M)
        transf_homg_point = M.dot(points)
        transf_homg_point /= transf_homg_point[2]
        return transf_homg_point[:2, :]

    def to_json(self):
        return {
            'is_action': self.is_action,
            'top_player_box': self.top_player_box.coords.tolist(),
            'top_player_position': self.top_player_position,
            'bottom_player_box': self.bottom_player_box.coords.tolist(),
            'bottom_player_position': self.bottom_player_position,
        }

class TennisPoint(object):

    def __init__(self, frames: List[ActionFrameSummary]):
        self._frames = frames

    def to_json(self):
        return {
            'frames': [f.to_json() for f in self._frames]
        }

    @property
    def top_player_path(self):
        return np.array([p.top_player_position for p in self._frames])

    @property
    def bottom_player_path(self):
        return np.array([p.bottom_player_position for p in self._frames])

    @property
    def bottom_player_path_warped(self):
        return np.array(list(filter(lambda x: x is not None, [p.bottom_player_position_warped for p in self._frames])))

    @property
    def top_player_path_warped(self):
        return np.array(list(filter(lambda x: x is not None, [p.top_player_position_warped for p in self._frames])))

    @property
    def top_player_distance_moved(self):
        return TennisPoint.distance_from_coords(self.top_player_path_warped)

    @property
    def bottom_player_distance_moved(self):
        return TennisPoint.distance_from_coords(self.bottom_player_path_warped)

    @staticmethod
    def distance_from_coords(points: np.ndarray) -> float:
        if points.shape[0] < 2:
            return 0
        return float(np.sum(np.sqrt(np.sum((points[1:] - points[:-1]) ** 2, axis=1))))


class VideoSummary(object):

    def __init__(self, frame_summaries: List[FrameSummary]):
        self._frame_summaries = frame_summaries

    def _action_sequence(self):
        return np.array([f.is_action for f in self._frame_summaries]).astype(int)

    def num_points(self):
        return np.sum(np.diff(self._action_sequence()) == 1)

    def points(self) -> List[TennisPoint]:
        a = np.nonzero(np.diff(self._action_sequence()) == 1)[0]
        b = np.nonzero(np.diff(self._action_sequence()) == -1)[0]
        points = []
        for i, j in zip(a, b):
            points.append(TennisPoint(self._frame_summaries[i + 1:j + 1]))
        return points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--tracking-path", type=str)
    parser.add_argument("--frame-path", type=str)

    args = parser.parse_args()

    tracking_path = Path(args.tracking_path)

    tracking_js = utils.read_json_lines(tracking_path)
    video_summary = VideoSummary([FrameSummary.from_json(d) for d in tracking_js])
    print(video_summary.num_points())



