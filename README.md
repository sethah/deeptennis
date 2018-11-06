# Deeptennis

A project which applies various machine learning, deep learning, and computer vision techniques
to videos of professional tennis matches.

## Getting started

```
conda env create -f environment.yml
```

#### Generate frames from training videos

```
make frames FPS=1 VFRAMES=2000
```

#### Featurize images using pre-trained model

```
make featurized USE_GPU=1 BATCH_SIZE=64
```

#### Segment action frames, detect court and scoreboard bounding boxes

```
make court_extract
make score_extract
```

## Results

### Segment action frames

#### Action

![Action Examples](https://github.com/sethah/deeptennis/blob/master/docs/static/img/action_examples.png)

#### Not action

![Not Action Examples](https://github.com/sethah/deeptennis/blob/master/docs/static/img/not_action_examples.png)

### Multi-task model for locating court and score bounding boxes

There are a few key components here. Detecting the court corners is very easy
as long as the corners are not occluded. When they are, we need a model that has a
large receptive field so that it can use information from the non-occluded corners
to detect the occluded one(s). Even with an appropriate model, we need enough examples
of occlusion in the train set. 

For the model, we can use a feature pyramid network to extract low and high level 
features with a large receptive field. For training data, we use image augmentation
to introduce occlusion.

#### Image augmentation

![](https://github.com/sethah/deeptennis/blob/master/docs/static/img/image_augmentation.png)

#### Occlusion predictions

In the occluded examples below, the model still detects the correct corner location,
but is less certain about the prediction. The probability mass is more spread out.

![](https://github.com/sethah/deeptennis/blob/master/docs/static/img/french_occluded.png)

![](https://github.com/sethah/deeptennis/blob/master/docs/static/img/djok_murr_french_occluded_hmap.png)

![](https://github.com/sethah/deeptennis/blob/master/docs/static/img/nadal_marterer_french_occluded.png)

![](https://github.com/sethah/deeptennis/blob/master/docs/static/img/nadal_marterer_french_occluded_hmap.png)

#### Joint prediction

The feature pyramid network is used as a feature extractor with a pre-trained Resnet 
backbone. These features are used to jointly predict the court corners and the location
of the scoreboard.

![](https://github.com/sethah/deeptennis/blob/master/docs/static/img/joint_prediction_outlines.png)
