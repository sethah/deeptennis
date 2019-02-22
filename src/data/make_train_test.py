import numpy as np
import argparse
import os
import json
from pathlib import Path
import pickle
from typing import List, Iterable, Tuple, Dict, NamedTuple, Any

from src.data.clip import Video
from src.vision.transforms import box_to_coords, BoundingBox


class Annotation(NamedTuple):
    category: str
    bbox: List[float]


def get_video_boxes(video: Video,
                    court_boxes: Dict[str, BoundingBox],
                    score_boxes: Dict[str, BoundingBox]) -> Iterable[Dict[str, Any]]:
    d = {}
    for fname, bbox in score_boxes.items():
        tmp = bbox.as_box()
        if np.any(np.isnan(tmp)):
            print(bbox.as_list(), tmp)
        anno = Annotation('score', bbox.as_list())
        d[fname] = {'name': fname, 'path': str(video.uri), 'annotations': [anno._asdict()]}
    for fname, bbox in court_boxes.items():
        anno = Annotation('court', bbox.as_list())
        if fname in d:
            d[fname]['annotations'].append(anno._asdict())
        else:
            d[fname] = {'name': fname, 'path': str(video.uri), 'annotations': [anno._asdict()]}
    return d.values()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--frame-path", type=str)
    parser.add_argument("--score-path", type=str)
    parser.add_argument("--court-path", type=str)
    parser.add_argument("--action-path", type=str)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segmentation", type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)

    save_path = Path(args.save_path)
    action_path = Path(args.action_path)
    frame_path = Path(args.frame_path)
    court_path = Path(args.court_path)
    score_path = Path(args.score_path)

    all_vids = [Video.from_dir(d) for d in frame_path.iterdir()]
    nvalid = int(len(all_vids) * args.test_frac)

    np.random.shuffle(all_vids)
    train_videos = all_vids[nvalid:]
    valid_videos = all_vids[:nvalid]

    images = {'train': [], 'test': []}
    if args.segmentation:
        for vid in all_vids:
            phase = 'train' if vid in train_videos else 'test'
            action_mask = np.load(action_path / (vid.name + ".npy"))
            annos = []
            for i, mask_path in enumerate((score_path / vid.name).iterdir()):
                if not action_mask[i]:
                    continue
                anno = {'name': mask_path.with_suffix(".jpg").name,
                              'path': vid.uri,
                              'mask_name': mask_path.name,
                              'mask_path': mask_path.parent}
                images[phase].append({k: str(v) for k, v in anno.items()})

    else:
        for vid in all_vids:
            phase = 'train' if vid in train_videos else 'test'
            action_mask = np.load(action_path / (vid.name + ".npy"))
            with open(score_path / (vid.name + ".pkl"), 'rb') as f:
                scores = [(fname, BoundingBox.from_box(box)) for i, (fname, box) in enumerate(pickle.load(f)) if action_mask[i]]
            with open(court_path / (vid.name + ".pkl"), 'rb') as f:
                courts = [(fname, BoundingBox.from_coords(coords + [0.0] * (8 - len(coords))))
                          for i, (fname, coords) in enumerate(pickle.load(f)) if action_mask[i]]
            for anno in get_video_boxes(vid, dict(courts), dict(scores)):
                images[phase].append(anno)
    save_path.mkdir(exist_ok=True)
    for phase in {'train', 'test'}:
        with open(save_path / Path(phase).with_suffix(".json"), 'w') as f:
            json.dump({'annotations': images[phase]}, f)

