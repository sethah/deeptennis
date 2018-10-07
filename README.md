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

#### Segment tennis videos into match clips + add bounding box

```
make clips
```

##### Action

![Action Examples](https://github.com/sethah/deeptennis/blob/master/docs/static/img/action_examples.png)

##### Not action

![Not Action Examples](https://github.com/sethah/deeptennis/blob/master/docs/static/img/not_action_examples.png)

#### Train CNN to find tennis court corners

##### Image augmentation

##### Heatmap predictions

![Action Examples](https://github.com/sethah/deeptennis/blob/master/docs/static/img/prediction_heatmap.png)

##### Bounding boxes

![Not Action Examples](https://github.com/sethah/deeptennis/blob/master/docs/static/img/prediction_bbox.png)
