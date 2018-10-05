import pandas as pd
from pathlib import Path

class Video(object):

    def __init__(self, uri, ext='.jpg'):
        self.uri = Path(uri)
        self.ext = ext
        self.frames = sorted([f for f in self.uri.iterdir() if f.suffix == self.ext])

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
    def name(self):
        return self.uri.stem

    def __getitem__(self, item):
        return self.frames[int(item)]

    def __repr__(self):
        return self.name

class Frame(object):

    def __init__(self, uri):
        self.uri = uri

class ActionFrame(Frame):

    def __init__(self, uri, bbox):
        super().__init__(uri)
        self.bbox = bbox

class BoundingBox(object):

    def __init__(self):
        pass

class Clip(object):
    """
    A contiguous of frames from a video.
    """

    def __init__(self, video):
        self.bboxes = []
        self.video = video
        self.frames = []

    def frame_range(self):
        pass

    def _add_frame(self, frame_id, bbox):
        frame_uri = self.video[frame_id]
        self.frames.append(frame_uri)
        self.bboxes.append(bbox)
        return self

    @staticmethod
    def from_csv(csv_path, video):
        clips_path = Path(csv_path)
        df = pd.read_csv(clips_path, header=None,
                         names=['frame_id', 'clip_id'] + ['bb%d' % i for i in range(8)])
        clips = {}
        for i, row in df.iterrows():
            bbox = row.iloc[2:].values
            clip_id = row['clip_id']
            if clip_id not in clips:
                # TODO: fix this path
                clips[clip_id] = Clip(video)
            clips[clip_id]._add_frame(row['frame_id'] - 1, bbox)
        return [v for k, v in sorted(clips.items(), key=lambda k: k[0])]

