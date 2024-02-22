# Document Image Classification | 문서 이미지 분류
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FBae-ChangHyun%2FDocument-Image-Classification&count_bg=%233D51C8&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
![GitHub forks](https://img.shields.io/github/forks/Bae-ChangHyun/Document-Image-Classification)<br>
프로젝트 기간: `Feb 05, 2024 ~ Feb 19, 2024`

## 목차
 - [Competetion Info](#1-competetion-info)
 - [Directory](#2-directory)
 - [Data Augmentation](#3-data-augentation)
 - [Modeling](#4-modeling)
 - [Ensemble](#5-ensemble)
 - [Result](#6-result)

## 0. Environment
- CUDA Version 12.2 
- NVIDIA GeForce RTX 3090

## 1. Competetion Info

주최: Upstage + Fastcampus on [Aistages](https://stages.ai/)

### 1-1. Team

|![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9233ab6e-25d5-4c16-8dd4-97a7b8535baf) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/e7394268-0f94-4468-8cf5-3cf67e4edd07) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9c75cbd9-f409-4fdd-a5c3-dec082ade3bf) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/388eac05-7cd9-4688-8a87-5b6b742715cf) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/48dd674c-ab93-48d1-9e05-e7e8e402597c) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/0a524747-a854-4eee-95b6-108c84514df8) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [최장원](https://github.com/Jangonechoi)             |            [김영천](https://github.com/dudcjs2779)             |            [배창현](https://github.com/Bae-ChangHyun)             |            [박성우](https://github.com/UpstageAILab)             |            [조예람](https://github.com/huB-ram)             |            [이소영B](https://github.com/UpstageAILab)             |

### 1-2. Overview

Document Image Classification 경진대회는 주어진 데이터를 활용하여 다양한 종류의 문서 이미지의 클래스를 예측.

문서 데이터는 금융,의료,보험,물류 등 산업 전반에 가장 많은 데이터이며 많은 대기업에서 디지털 혁신을 위한 문서 유형 분류를 도입.

의료, 금융 등 여러 비즈니스 분야의 대량의 문서 이미지를 식별하고 자동화하는 것이 중요.

--> 17개 class의 문서 이미지를 분류

### 1-3. Evaluation metric

$$ F1_{\text{macro}} = \frac{1}{N} \sum_{i=1}^{N} F1_i $$

## 2. Directory

```bash
// 저작권으로 인해 데이터 셋은 업로드하지 않습니다.
├── data                    
│   ├── meta.csv
│   ├── train
│   ├── test
│   ├── train.csv
│   └── test.csv
├── code
│──  └── Doc_classification.ipynb
│──  └── Doc_classification(wandb).ipynb
└──  └── sub.ipynb
```

### 2-1. Data descrption

`train.csv`
: (1570,2) / train 데이터 이미지 경로와 해당 이미지의 라벨<br>
`test.csv`
: (3140,2) / test 데이터 이미지 경로와 해당 이미지의 라벨 <br>
`meta.csv`
: (17,2) / 이미지의 실제 클래스와 인코딩된 라벨 <br>
`Doc_classification.ipynb`
: 모델링 및 전체 코드 <br>
`Doc_classification(wandb).ipynb`
:wandb에 자동으로 기록하는 모델 실험용 전체 코드 <br>
`sub.ipynb`
:데이터 증강 및 분할 등 sub 코드 <br>

## 3. Data Augentation

Augraphy + Albumentations + Mixup
 
## 4. Modeling

### 4-1. Model
[resnet50](https://huggingface.co/docs/timm/models/resnet)

### 4-2. Parameter
image size = `224`,`256`,`384` <br>
batch size = `64`,`128`,`256`  <br>
scheduler = `CosineAnnealing`, `ReduceLROnPateau`, `1e-4` <br>
Optimizer = `Adam` <br>
Loss function = `CrossEntropy Loss`, `Focal Loss` <br>

--> Experiment Record: [![WANDB](https://img.shields.io/badge/WANDB-#FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)](https://wandb.ai/bae951753/Docs%20Image%20Classifications?workspace=user-bae951753)

## 5. Ensemble

### 5-1. TTA
Test Time Augmentation with ttach library

### 5-2. Voting
Hard voting + weighted soft voting

## 6. Result

### 6-1. Leader Board
- Rank:2
- Public Score:0.9610 (5th)
- Private Score:0.9594 (2nd)

### 6-2. Reference
- [fastdup](https://github.com/visual-layer/fastdup)
- [augraphy](https://github.com/sparkfish/augraphy)
- [albumentations](https://github.com/albumentations-team/albumentations)
- [focal loss](https://github.com/mathiaszinnen/focal_loss_torch)

## 🛠 Tech Stack 🛠
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white")]()
[![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)]()
[![Linux](https://img.shields.io/badge/linux-FCC624?style=for-the-badge&logo=linux&logoColor=black")]()
[![OpenCV](https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=black)]()
[![WANDB](https://img.shields.io/badge/WANDB-#FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)]()