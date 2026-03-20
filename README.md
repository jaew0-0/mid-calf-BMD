# Estimation of Bone Mineral Density through 2D Radiographic Mid-Calf Muscle Thickness

이 저장소는 **2D 방사선 영상에서 측정한 mid-calf muscle thickness**를 이용해  
골밀도(BMD) 및 골다공증과의 연관성을 분석하고, 논문용 figure / table을 생성하는 Python 코드 모음입니다.  
업로드된 코드들은 대상자 선정, 기술통계, 상관분석, tertile 분석, 로지스틱 회귀, ROC 분석, 보조분석 figure 생성에 사용됩니다.
---

## Project overview

이 프로젝트의 핵심 목적은 다음과 같습니다.

- 단순 X-ray 기반 **mid-calf muscle thickness**가 BMD와 관련이 있는지 평가
- muscle-only thickness가 total thickness보다 더 좋은 지표인지 비교
- lateral / AP / combined measurement의 성능 비교
- osteoporosis 예측을 위한 practical cutoff 제시
---

## File structure

### Main analysis scripts

- `table_1.py`  
  연구 대상자의 baseline characteristics를 정리한 **Table 1**을 생성합니다.  
  성별에 따른 비교와 p-value 계산이 포함되어 있으며, 결과는 Excel과 PNG로 저장됩니다. :contentReference[oaicite:4]{index=4}

- `figure_1.py`  
  전체 대상자에서 연구 제외 기준을 적용해 **flowchart summary**를 만들고,  
  exclusion reason 및 study population 정보를 Excel로 저장합니다. :contentReference[oaicite:5]{index=5}

- `figure_3.py`  
  AP / lateral / AP+lateral 측정값과 minimum T-score 간의 관계를  
  남성, 여성, 전체 환자군으로 나누어 2x3 scatter plot 형태로 시각화합니다. :contentReference[oaicite:6]{index=6}

- `figure_4.py`  
  성별 및 촬영 view별 Pearson correlation coefficient를 정리하고,  
  이를 bar plot으로 시각화합니다. :contentReference[oaicite:7]{index=7}

- `figure_5.py`  
  lateral muscle thickness를 tertile로 나누어  
  minimum T-score 분포와 osteoporosis prevalence를 비교합니다. :contentReference[oaicite:8]{index=8}

- `figure_6.py`  
  lateral muscle thickness와 lumbar / hip T-score 간의 상관관계를 비교합니다.  
  scatter plot과 correlation comparison bar chart를 생성합니다. :contentReference[oaicite:9]{index=9}

- `figure_7.py`  
  lateral muscle thickness, sex, age, BMI를 포함한  
  **multivariable logistic regression**을 수행하고 forest plot을 생성합니다. :contentReference[oaicite:10]{index=10}

- `figure_8s7.py`  
  lateral muscle thickness의 골다공증 예측 성능을 평가하는  
  ROC curve 및 cutoff, sensitivity, specificity를 계산합니다.  
  supplementary ROC comparison도 함께 생성합니다. :contentReference[oaicite:11]{index=11}

- `figure_9.py`  
  성별에 따라 lateral muscle thickness의 ROC curve를 분리하여  
  gender-specific cutoff를 제시합니다. :contentReference[oaicite:12]{index=12}

---

## Supplementary analysis scripts

- `figure_s4.py`  
  lumbar 및 hip 영역에서 lowest T-score가 기록된 해부학적 부위 분포를 시각화합니다. :contentReference[oaicite:13]{index=13}

- `figure_s56.py`  
  연령대 및 BMI subgroup별로 lateral muscle thickness와 minimum T-score의 상관성을 분석합니다. :contentReference[oaicite:14]{index=14}

- `figure_s123.py`  
  lateral total, AP muscle, AP total 기준 tertile 분석을 추가로 수행하여  
  supplementary figure를 생성합니다. :contentReference[oaicite:15]{index=15}

---

## Data files

코드는 동일 폴더 내 CSV 파일을 기준으로 실행되도록 작성되어 있습니다.

- `dataset_1.csv`  
  cohort selection, exclusion reason 정리, baseline table 생성에 사용됩니다. :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17}

- `dataset_2.csv`  
  main figure 및 supplementary figure 생성에 사용됩니다.  
  상관분석, tertile 분석, logistic regression, ROC 분석이 포함됩니다. :contentReference[oaicite:18]{index=18} :contentReference[oaicite:19]{index=19} :contentReference[oaicite:20]{index=20}

---

## Requirements

아래 패키지가 필요합니다.

```bash
pip install pandas numpy matplotlib scipy seaborn scikit-learn statsmodels openpyxl

---

## How to run

각 스크립트는 독립적으로 실행할 수 있습니다.

```bash
python table_1.py
python figure_1.py
python figure_3.py
python figure_4.py
python figure_5.py
python figure_6.py
python figure_7.py
python figure_8s7.py
python figure_9.py
python figure_s4.py
python figure_s56.py
python figure_s123.py
```

---

## Output

실행 결과로 다음과 같은 파일들이 생성됩니다.

* `.png` : 논문용 / 발표용 figure
* `.xlsx` : 통계 요약표 및 intermediate summary
* `.csv` : ROC 결과 등 일부 요약 데이터

일부 스크립트는 별도 output 폴더를 자동 생성하여 결과를 저장합니다.
예: `figure4_outputs/`, `figure5_outputs/`, `figure7_outputs/`, `supplementary_subgroup_correlation/` 등.   

---

## Notes

* CSV 인코딩은 `utf-8`, `cp949`, `euc-kr` 순서로 자동 시도하도록 작성된 코드가 포함되어 있습니다.  
* 컬럼명은 원본 데이터셋 구조를 기준으로 자동 탐색하도록 일부 스크립트가 구성되어 있습니다.  
* 실행 전, 각 스크립트와 데이터 파일이 같은 디렉토리에 위치하는지 확인해주세요.  

---

## Study summary

이 분석은
**“Mid-Calf Muscle Thickness on Plain Radiographs as a Predictor of Bone Mineral Density”**
주제를 바탕으로, routine radiograph에서 얻을 수 있는 muscle thickness가
BMD 및 osteoporosis screening에 활용될 수 있는지 평가하기 위해 수행되었습니다.
발표자료에서는 lateral muscle thickness가 practical single-view marker로 제시되며,
ROC 기반 cutoff와 성별별 cutoff도 함께 제안됩니다. 

---

## Author

연구학생: 이재우, 김승한
지도교수: 이순철 
