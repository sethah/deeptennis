import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    frame_paths = [p for p in Path(args.frames_path).iterdir()]
    with open(args.output_path, "w") as f:
        for p in sorted(Path(args.frames_path).iterdir()):
            f.write('{"image_path": "%s"}\n' % str(p))
