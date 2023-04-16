# Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Models

This is a PyTorch implementation of [Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Models](https://arxiv.org/pdf/2111.10759.pdf) by Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai.
Mask projection code is partially inspired from [FaceXZoo](https://github.com/JDAI-CV/FaceX-Zoo).

<p align="center">
<img src="https://github.com/AlonZolfi/AdversarialMask/blob/master/data/intro.png" />
</p>

![projection pipeline](https://github.com/AlonZolfi/AdversarialMask/blob/master/data/projection_pipeline.png?raw=true)

## Face Recognition Models

Please put the downloaded weights in a local directory called "weights" under each model directory (or change location in the [config](https://github.com/AlonZolfi/AdversarialMask/blob/master/patch/config.py) file).
### ArcFace and CosFace

Code is taken from [here](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch).
Download weights from [here](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d).

### MagFace

Code is taken from [here](https://github.com/IrvingMeng/MagFace).
Download weights from [here](https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/view).

## Landmark Detection Models

### MobileFaceNet

Code is taken from [here](https://github.com/cunjian/pytorch_face_landmark).
Download weights from [here](https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing).
(Weights file is already included in this repository under [landmark_detection/pytorch_face_landmark/weights](https://github.com/AlonZolfi/AdversarialMask/tree/master/landmark_detection/pytorch_face_landmark/weights)).

### Face Alignment

Code is taken from [here](https://github.com/1adrianb/face-alignment).
Weights are downloaded automatically on the first run.

Note: this model is more accurate, however, it is a lot larger than MobileFaceNet and requires a large memory GPU to be able to backpropagate when training the adversarial mask.

## 3D Face Reconstruction Model

Code is taken from [here](https://github.com/YadiraF/PRNet).
Download weights from [here](https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view). (Weights file is already included in this repository under [prnet](https://github.com/AlonZolfi/AdversarialMask/tree/master/prnet)).

## Datasets

### CASIA-WebFace

The cleaned up version of the dataset can be found [here](https://onedrive.live.com/?authkey=%21AJjQxHY%2DaKK%2DzPw&cid=1FD95D6F0AF30F33&id=1FD95D6F0AF30F33%2174855&parId=1FD95D6F0AF30F33%2174853&action=locate), suggested by [this github issue](https://github.com/cmusatyalab/openface/issues/119#issuecomment-455986064).

## Installation

Install the required packages in [req.txt](https://github.com/AlonZolfi/AdversarialMask/tree/master/req.txt).

## Usage

### Configuration

Configurations can be changed in the [config](https://github.com/AlonZolfi/AdversarialMask/blob/master/patch/config.py) file.

### Train

Run the [patch/train.py](https://github.com/AlonZolfi/AdversarialMask/blob/master/patch/train.py) file.

### Test

Run the [patch/test.py](https://github.com/AlonZolfi/AdversarialMask/blob/master/patch/test.py) file. Specify the location of the adversarial mask image in main function.

## Citation
```
@inproceedings{zolfi2023adversarial,
  title={Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Models},
  author={Zolfi, Alon and Avidan, Shai and Elovici, Yuval and Shabtai, Asaf},
  booktitle={Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2022, Grenoble, France, September 19--23, 2022, Proceedings, Part III},
  pages={304--320},
  year={2023},
  organization={Springer}
}
```
