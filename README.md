# Deeptennis

A project which applies various machine learning, deep learning, and computer vision techniques
to videos of professional tennis matches.

|   |  |
| ----------------------- | ----------------------- |
| ![alt text](reports/figures/osaka_halep_rome_18.gif) | ![alt text](reports/figures/djokovic_anderson_wim_18.gif) | 
| ![alt text](reports/figures/coric_federer_halle_18.gif) | ![alt text](reports/figures/zverev_isner_miami_18.gif) |


## Installation

In a virtual environment:

```bash
pip install -r requirements.txt
```


## Generate frames from training videos

```
make frames FPS=1 VFRAMES=2000
```

