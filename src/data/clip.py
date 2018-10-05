import numpy as np
import pickle
from pathlib import Path


class Video(object):

    def __init__(self, frames, ext='.jpg'):
        self.frames = frames
        self.uri = Path(frames[0].parent)
        self.ext = ext
        # self.frames = sorted([f for f in self.uri.iterdir() if f.suffix == self.ext])

    def _parse_uri(self):
        p1, p2, loc, yr = self.uri.stem.split("_")
        return p1, p2, loc, yr

    @property
    def players(self):
        p1, p2, _, _ = self._parse_uri()
        return p1, p2

    @property
    def venue(self):
        _, _, _venue, _ = self._parse_uri()
        return _venue

    @property
    def year(self):
        _, _, _, _year = self._parse_uri()
        return _year

    @property
    def name(self):
        return self.uri.stem

    def __getitem__(self, item):
        return self.frames[int(item)]

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.frames)

    @classmethod
    def from_dir(cls, folder, ext='.jpg'):
        frames = sorted([f for f in folder.iterdir() if f.suffix == ext])
        return cls(frames, ext=ext)


class ActionVideo(Video):

    def __init__(self, frames, boxes, ext='.jpg'):
        super(ActionVideo, self).__init__(frames, ext=ext)
        self.boxes = boxes

    def __getitem__(self, item):
        item = int(item)
        return self.frames[item], self.boxes[item]

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            clip_dict = pickle.load(f)
        vids = []
        for i, clip_data in enumerate(clip_dict['data']):
            frames = [Path(clip_dict['path']) / fname for fname, _ in clip_data]
            boxes = [np.array(box) for _, box in clip_data]
            vids.append(ActionVideo(frames, boxes))
        return vids

