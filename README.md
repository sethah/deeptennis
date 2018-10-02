# Deeptennis

A project which applies various machine learning, deep learning, and computer vision techniques
to videos of professional tennis matches.

## Train a bounding box prediction model

### Generate frames from training videos

```
make frames FPS=1 VFRAMES=2000
```

### Featurize images using pre-trained model

```
make featurized
```

### Segment tennis videos into match clips + add bounding box

* Project featurized images to low dimensional space with PCA
* Cluster images using a Gaussian emission Hidden Markov Model
* Choose cluster which represents action frames
* Use OpenCV to pick out white lines in the images, apply rules to find the four corners
of the tennis court

```
make clips
```
