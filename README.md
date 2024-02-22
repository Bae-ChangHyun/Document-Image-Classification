# Document Image Classification | 문서 이미지 분류
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FBae-ChangHyun%2FDocument-Image-Classification&count_bg=%233D51C8&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
![GitHub forks](https://img.shields.io/github/forks/Bae-ChangHyun/Document-Image-Classification) <br>
프로젝트 기간: `Feb 05, 2024 ~ Feb 19, 2024`

## 목차
 - [Competetion Info](#1-competetion-info)
 - [Directory](#2-directory)
 - [Data description](#3-data-descrption)
 - [Modeling](#4-modeling)
 - [Result](#5-result)


### 0. Environment
- CUDA Version 12.2 
- NVIDIA GeForce RTX 3090

## 1. Competetion Info

주최: Upstage + Fastcampus on [Aistages](https://stages.ai/)

## 1-1. Team

|![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9233ab6e-25d5-4c16-8dd4-97a7b8535baf) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/e7394268-0f94-4468-8cf5-3cf67e4edd07) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/9c75cbd9-f409-4fdd-a5c3-dec082ade3bf) | ![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/388eac05-7cd9-4688-8a87-5b6b742715cf) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/48dd674c-ab93-48d1-9e05-e7e8e402597c) |![image](https://github.com/UpstageAILab/upstage-cv-classification-cv5/assets/96022213/0a524747-a854-4eee-95b6-108c84514df8) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [최장원](https://github.com/UpstageAILab)             |            [김영천](https://github.com/UpstageAILab)             |            [배창현](https://github.com/UpstageAILab)             |            [박성우](https://github.com/UpstageAILab)             |            [조예람](https://github.com/huB-ram)             |            [이소영B](https://github.com/UpstageAILab)             |
|                            팀장                            |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |

### 1-2. Overview

Document Image Classification 경진대회는 주어진 데이터를 활용하여 다양한 종류의 문서 이미지의 클래스를 예측.

문서 데이터는 금융,의료,보험,물류 등 산업 전반에 가장 많은 데이터이며 많은 대기업에서 디지털 혁신을 위한 문서 유형 분류를 도입.

의료, 금융 등 여러 비즈니스 분야의 대량의 문서 이미지를 식별하고 자동화하는 것이 중요.

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
└──  └── Doc_classification(wandb).ipynb
```

## 3. Data descrption

`train.csv`
: (1570,2) / train 데이터 이미지 경로와 해당 이미지의 라벨<br>
`test.csv`
: (3140,2) / test 데이터 이미지 경로와 해당 이미지의 라벨 <br>
`Doc_classification.ipynb`
: 모델링 및 전체 코드 <br>
`Doc_classification(wandb).ipynb`
:wandb에 자동으로 기록하는 모델 실험용 전체 코드 <br>
`sub.ipynb`
:데이터 증강 및 분할 등 sub 코드 <br>
 
## 4. Modeling

### 4-1. Model
`resnet50`

### 4-2. Validation
- 임의의 기간 -> train과 유사한 가장 최근 2023.01~2023.06을 validation set 구성
- k-fold -> k fold를 이용하여 5개의 fold를 나눠 validation set 구성
- Stratified k-fold: target을 구간화하여 train과 valid의 target분포가 유사하도록 fold를 나눠 validation set 구성
- 
### 4-3. Train
- 전체 데이터를 이용한 학습
- 구별로 나눠 학습
- 전용면적 범주별로 나눠 학습
- 아파트별로 나눠 학습

### 4-4. Final
- 여러 실험결과 LGBM+Optuna+특정기간 validationset+ 전체 학습이 가장 좋은 성능을 보였음.
- 
## 5. Result

### 5-1 Leader Board
- Rank:2
- Public Score:14760.6767(2nd)
- Private Score: 10764.6959(2nd)

### 5-2 Presentation
- _Insert your presentaion file(pdf) link_

### Reference
- [실거래가: 국토교통부](https://www.kiep.go.kr/menu.es?mid=a10602010000)
- [서울시 공공주택 아파트정보: 서울열린데이터광장](https://data.seoul.go.kr/dataList/OA-15818/S/1/datasetView.do)
- [서울시 가구총소득: 서울열린데이터광장]: 서울열린데이터광장(https://data.seoul.go.kr/dataList/DT201013B022/S/2/datasetView.do)
- [실거래지수: KOSIS국가통계포털](https://kosis.kr/statHtml/statHtml.do?orgId=408&tblId=DT_KAB_11672_S1)
- [서울시 학교정보: 서울열린데이터광장](https://data.seoul.go.kr/dataList/OA-20502/S/1/datasetView.do)
- [아파트 정보: K-apt 공동주택관리정보시스템](https://www.k-apt.go.kr/board/boardList.do?board_type=03)
- [금리: 한국은행경제통계시스템](https://ecos.bok.or.kr/#/)
- 지하철
- 버스
