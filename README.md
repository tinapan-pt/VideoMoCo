# VideoMoCo: Contrastive Video Representation Learning with Temporally Adversarial Examples
# Overview
Framework of the proposed approach.

Given a video sequence as an input sample, we improve the temporal feature representations of MoCo from two perspectives. We introduce generative adversarial learning to improve the temporal robustness of the encoder. We use a generator to temporally drop out several frames from this sample. The discriminator is then learned to encode similar feature representations regardless of frame removals. By adaptively dropping out different frames during training iterations of adversarial learning, we augment this input sample to train a temporally robust encoder. Second, we propose a temporally adversarial decay to model key attenuation in the memory queue when computing the contrastive loss. It largely improves MoCo with temporally robust feature representation for the video-related classification/recognition scenario, which is novel in terms of both temporal representation learning methodology and video scenario.
![pipeline.png](https://i.loli.net/2021/05/08/ZVGAwKk2mIuY1aP.png)

# Requirements
- pytroch >= 1.3.0
- tensorboard
- cv2
- kornia

# Usage

## Data preparation

- Download the Kinetics400 dataset from the [official website](https://deepmind.com/research/open-source/kinetics).
- Download the original UCF101 dataset from the [official website](https://www.crcv.ucf.edu/data/UCF101.php).

## Model preparation
take 100 epochs to train D for initialization, and then train G and D via adversarial learning for the remaining 100 epochs.

## Train
```python
python train.py
```
## Evaluation
```python
python eval.py  
```
## TODO
- [ ] Visualizations(attention maps)


