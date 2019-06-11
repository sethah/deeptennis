# Deeptennis

A project which applies various machine learning, deep learning, and computer vision techniques
to videos of professional tennis matches.

Built with [AllenCV](https://github.com/sethah/allencv).

|   |  |
| ----------------------- | ----------------------- |
| ![alt text](reports/figures/osaka_halep_rome_18.gif) | ![alt text](reports/figures/djokovic_anderson_wim_18.gif) | 
| ![alt text](reports/figures/coric_federer_halle_18.gif) | ![alt text](reports/figures/zverev_isner_miami_18.gif) |


## Installation

In a virtual environment:

```bash
pip install -r requirements.txt
```

## Quick Demo

Start a simple demo server using a pre-trained model.

```
python -m allencv.service.server_simple \
--archive-path "https://deeptennis.s3-us-west-1.amazonaws.com/player_kprcnn_res50_fpn.tar.gz" \
--predictor default_image \
--include-package allencv.data.dataset_readers \
--include-package allencv.modules.im2vec_encoders \
--include-package allencv.modules.im2im_encoders \
--include-package allencv.models \
--include-package allencv.predictors \
--title "Player detector" \
--detection \
--overrides '{"dataset_reader": {"type": "image_annotation", "augmentation": [{"type": "resize", "height": 512, "width": 512}, {"type": "normalize"}], "lazy": true}, "model": {"roi_box_head": {"decoder_detections_per_image": 50}, "rpn": {"archive_file": "https://deeptennis.s3-us-west-1.amazonaws.com/player_rpn_res50_fpn.tar.gz"}}}'
```

Navigate to `localhost:8000` and select an image of a tennis point to view the model's detections. 

## Generate an annotated video with predictions

```
make data
# download a short highlight video for making predictions
curl -o data/raw/federer_cilic_aus_18.mp4 https://deeptennis.s3-us-west-1.amazonaws.com/federer_cilic_aus_18.mp4
# VFRAMES=100 limits the output video to 100 frames
make data/interim/tracking_videos/federer_cilic_aus_18 FPS=1 VFRAMES=100 MODEL_PATH=https://deeptennis.s3-us-west-1.amazonaws.com/player_kprcnn_res50_fpn.tar.gz
# output is in data/interim/tracking_videos/federer_cilic_aus_18/federer_cilic_aus_18.mp4
```

