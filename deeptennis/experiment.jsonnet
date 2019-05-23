local mean = [0.4306,0.4381,0.3884];
local std = [0.2100,0.1791,0.1697];
local im_size = [224, 224];
{
  "seed": 42,
  "batch_size": 16,
  "epochs": 12,
  "train_reader": {
    "type": "tennis",
    "transform": {
      "type": "court_score",
      "mean": mean,
      "std": std,
      "size": im_size,
      "mask_prob": 0.7,
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
    "type": "anchor",
    "stages": [
      {
        "type": "fpn",
        "backbone": {
          "freeze": true
        }
      },
      {
        "type": "court_head",
        "in_channels": 128,
      }
    ],
    "grid_sizes": [[56, 56]],
    "box_sizes": [[50, 20]],
    "im_size": im_size,
    "angle_scale": 10
  },
  "optim": {
    "type": "adam",
    "lr": 1e-3,
    "parameter_groups": [
      [[".*backbone\\.stages\\.0.*"], {"initial_lr": 0.000006}],
      [[".*backbone\\.stages\\.1.*"], {"initial_lr": 0.0000125}],
      [[".*backbone\\.stages\\.2.*"], {"initial_lr": 0.000025}],
      [[".*backbone\\.stages\\.3.*"], {"initial_lr": 0.00005}],
      [["(model\\.1.*)|(model\\.0\\.upsample.*)"], {"initial_lr": 1e-3}]
    ]
  }
}

