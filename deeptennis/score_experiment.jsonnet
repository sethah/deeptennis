local mean = [0.4306,0.4381,0.3884];
local std = [0.2100,0.1791,0.1697];
local im_size = [224, 224];
{
  "seed": 42,
  // "valid_metric": "+IOU",
  "batch_size": 64,
  "gradual_unfreezing": true,
  "epochs": 16,
  "patience": 5,
  "train_reader": {
    "type": "tennis",
    "transform": {
      "type": "court_score",
      "mean": mean,
      "std": std,
      "size": im_size,
      "mask_prob": 0.0,
      "train": true
    },
    "lazy": true
  },
  "valid_reader": {
    "type": "tennis",
    "transform": {
      "type": "court_score",
      "mean": mean,
      "std": std,
      "size": im_size,
      "train": false
    },
    "lazy": true
  },
  "model": {
    "type": "anchorbox",
    "encoder": {
      "type": "fpn_encoder",
      "input_size": im_size,
    },
    "box_sizes": [[50, 20]],
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
    "lr": 1e-3,
    "parameter_groups": [
      [[".*backbone\\.stages\\.1.*"], {"initial_lr": 0.0000125}],
      [[".*backbone\\.stages\\.2.*"], {"initial_lr": 0.000025}],
      [[".*backbone\\.stages\\.3.*"], {"initial_lr": 0.00005}],
      [["(encoder\\.court_convs.*)|(encoder\\.out_conv_score.*)|(encoder\\.conv1.*)|(encoder\\.fpn\\.upsample.*)"], {"initial_lr": 1e-3}]
    ]
  }
}

