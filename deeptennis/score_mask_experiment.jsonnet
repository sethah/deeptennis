local mean = [0.4306,0.4381,0.3884];
local std = [0.2100,0.1791,0.1697];
local im_size = [224, 224];
local learning_rate = 1e-3;
{
  "seed": 42,
  "valid_metric": "+IOU",
  "batch_size": 64,
  "gradual_unfreezing": false,
  "epochs": 30,
  "patience": 30,
  "train_reader": {
    "type": "score",
    "transform": {
      "type": "score_augmentor",
      "mean": mean,
      "std": std,
      "size": im_size,
      "channel_prob": 0.5,
      "flip_prob": 0.5,
      "train": true
    },
    "lazy": true
  },
  "valid_reader": {
    "type": "score",
    "transform": {
      "type": "score_augmentor",
      "mean": mean,
      "std": std,
      "size": im_size,
      "train": false
    },
    "lazy": true
  },
  "model": {
    "type": "segmentation",
    "encoder": {
      "type": "unet",
      "keypoints": 2,
      "channels": 3,
      "filters": 8,
      "input_size": im_size
    }
  },
  /*"model": {
    "type": "anchor_score",
    "stages": [
      {
        "type": "fpn",
        "backbone": {
          "freeze": false
        }
      },
      {
        "type": "score_head",
        "in_channels": 128,
      }
    ],
    "grid_sizes": [[56, 56]],
    "box_sizes": [[50, 20]],
    "im_size": im_size,
    "angle_scale": 10
  },*/
  /*"optim": {
    "type": "adam",
    "lr": 1e-4,
    "parameter_groups": [
      [["backbone\\.(0|1).*"], {"initial_lr": 0.0}],
      [["backbone\\.4.*"], {"initial_lr": 0.00000125}],
      [["backbone\\.5.*"], {"initial_lr": 0.0000025}],
      [["out*"], {'initial_lr': 1e-3}],
    ]
  }*/
    "optim": {
    "type": "adam",
    "lr": learning_rate,
    "parameter_groups": [
      [["encoder*"], {'initial_lr': learning_rate}]
    ]
  }
}

