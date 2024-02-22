# Document Image Classification | 문서 이미지 분류
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FBae-ChangHyun%2Fapart_price_predict&count_bg=%23003BE7&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
![GitHub forks](https://img.shields.io/github/forks/Bae-ChangHyun/apart_price_predict) <br>
프로젝트 기간: `Feb 05, 2024 ~ Feb 19, 2024`


### 0. Environment
- CUDA Version 12.2 
- NVIDIA GeForce RTX 3090

## 1. Competetion Info

### 1-1 Overview

Document Image Classification 경진대회는 주어진 데이터를 활용하여 다양한 종류의 문서 이미지의 클래스를 예측.

문서 데이터는 금융,의료,보험,물류 등 산업 전반에 가장 많은 데이터이며 많은 대기업에서 디지털 혁신을 위한 문서 유형 분류를 도입.

의료, 금융 등 여러 비즈니스 분야의 대량의 문서 이미지를 식별하고 자동화하는 것이 중요.

### 1-2 Evaluation metric

$$ \text{F1-Macro} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

## 2. Components

### Directory
```bash
├── data                    
│   ├── meta.csv : 문서의 실제 클래스와 인코딩된 라벨 매핑
│   ├── train: 문서 이미지 1570장
│   ├── test: 문서 이미지 3140장 
│   ├── train.csv: train 데이터의 파일명과 라벨
│   └── sample_submission.csv: test 데이터의 파일명
├── code
│──  └── Doc_classification.ipynb
└──  └── Doc_classification(wandb).ipynb
// 저작권으로 인해 데이터 셋은 업로드하지 않습니다.
```

## 3. Data descrption

`Train data`
: (1118822,52) / 2007.01.01~2023.06.30 기간의 아파트 정보 및 실거래가 <br>
`Test data`
: (9272,51) / 2023.07.01~2023.09.26 기간의 아파트 정보 <br>
`seoul_bus`
: (       ) / 서울의 버스 정류소번호, 정류소명, 경위도, 정류소 타입 <br>
`seoul_subway`
:(        ) / 서울 지하철 역사ID, 역사명, 호선, 경위도 <br>
`price_index`
:(        ) / 2007.01~2023.06의 서울 아파트 실거래가격지수 <br>
`interest_rate`
:(        ) / 2007.01~2023.06의 대출금리 및 (   ) <br>
`family_income`
:(        )  / ( )  
 
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

## Team
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [최장원](https://github.com/UpstageAILab)             |            [김영천](https://github.com/UpstageAILab)             |            [배창현](https://github.com/UpstageAILab)             |            [박성우](https://github.com/UpstageAILab)             |            [조예람](https://github.com/huB-ram)             |            [이소영B](https://github.com/UpstageAILab)             |
|                            팀장                            |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |                            팀원                             |

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
