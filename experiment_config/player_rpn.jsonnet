local NUM_GPUS = 1;
local NUM_THREADS = 1;
local NUM_EPOCHS = 100;

local TRAIN_AUGMENTATION = [
            {
                "type": "resize",
                "height": 512,
                "width": 512
            },
            {
                "type": "bgr_normalize"
            },
            {
                "type": "horizontal_flip",
                "p": 0.5
            },
            {
                "type": "random_brightness_contrast",
                "p": 0.5
            },
            {
                "type": "gaussian_blur",
                "p": 0.5
            },
            {
                "type": "rotate",
                "limit": 20,
                "p": 0.5
            }
        ];
local VALID_AUGMENTATION = [
            {
                "type": "resize",
                "height": 512,
                "width": 512
            },
            {
                "type": "bgr_normalize"
            }
        ];
local TRAIN_READER = {
        "type": "image_annotation",
        "augmentation": TRAIN_AUGMENTATION,
        "lazy": true
};
local VALID_READER = {
        "type": "image_annotation",
        "augmentation": VALID_AUGMENTATION,
        "lazy": true
};

local BASE_ITERATOR = {
  "type": "basic",
  "batch_size": 2 * NUM_GPUS,
};

local MODEL = {
    "type": "detectron_rpn",
    "anchor_sizes": [32, 64, 128, 256, 512],
    "anchor_strides": [4, 8, 16, 32, 64],
    "batch_size_per_image": 256
};

local start_momentum = 0.9;
local initial_lr = 1e-3;
{
  "dataset_reader": TRAIN_READER,
  "validation_dataset_reader": VALID_READER,
  "train_data_path": std.extVar("TRAIN_PATH"),
  "validation_data_path": std.extVar("VALIDATION_PATH"),
  "model": MODEL,
  "iterator": BASE_ITERATOR,
  "trainer": {
    "num_epochs": NUM_EPOCHS,
    "should_log_learning_rate": true,
    "cuda_device" : if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      "type": "adam",
      "lr": initial_lr,
      //"momentum": start_momentum,
      "parameter_groups": [
      [["(^backbone\\._backbone\\.layer1\\.)|(^backbone\\._backbone\\.stem)"], {"initial_lr": initial_lr}],
      [["^backbone\\._backbone\\.layer2\\."], {"initial_lr": initial_lr}],
      [["^backbone\\._backbone\\.layer3\\."], {"initial_lr": initial_lr}],
      [["^backbone\\._backbone\\.layer4\\."], {"initial_lr": initial_lr}],
      [["(^backbone\\._convert)|(^backbone\\._combine)|(^backbone\\.model\\.fpn)"], {"initial_lr": initial_lr}],
      [["(^conv)|(^cls_logits)|(^bbox_pred)"], {"initial_lr": initial_lr}],
     ]
    },
    "learning_rate_scheduler": {
        "type": "slanted_triangular",
        "num_epochs": NUM_EPOCHS,
        "num_steps_per_epoch": 73,
        "discriminative_fine_tuning": true,
        "gradual_unfreezing": true,
        "cut_frac": 0.3,
        "decay_factor": 0.3
    },
    //"momentum_scheduler": {
    //    "type": "inverted_triangular",
    //    "cool_down": 30,
    //    "warm_up": 70,
    //    "ratio": 2,
    //},
    "summary_interval": 10
  }
}