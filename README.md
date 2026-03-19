# 방사선학적 근육 두께와 골다공증 위험도 분석 (Radiographic Muscle Thickness and Osteoporosis)

본 저장소(Repository)는 2D 경골 방사선 사진(Tibia radiographs)을 통해 측정한 종아리 근육 두께와 골다공증 발병 위험 간의 연관성을 분석하고, 논문에 사용된 피규어(Figure) 및 표(Table)를 생성하기 위한 파이썬(Python) 스크립트를 포함하고 있습니다.

## 📌 요구 사항 (Prerequisites)
이 코드는 Python 3 환경에서 작성되었습니다. 코드를 원활하게 실행하려면 다음 라이브러리가 필요합니다:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `openpyxl`

아래 명령어를 통해 필요한 패키지를 한 번에 설치할 수 있습니다:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn openpyxl

📂 데이터셋 (Dataset)
dataset_1.csv & dataset_2.csv: 환자의 기본 인적 사항, 방사선학적 근육/피하지방 두께 측정값, 그리고 골밀도(T-score) 데이터가 포함된 원본 데이터 파일입니다.
(주의: 환자의 민감한 의료 정보 및 개인정보 보호 규정(IRB)에 따라 원본 데이터셋은 본 공개 저장소에 포함되지 않습니다. 코드를 실행하려면 동일한 컬럼명을 가진 자체 데이터셋을 스크립트와 동일한 디렉토리에 위치시켜야 합니다.)

🛠️ 스크립트 구성 및 설명 (Scripts Overview)
각 스크립트는 논문에 삽입된 특정 Figure 또는 Table을 생성하도록 독립적으로 작성되었습니다.

figure_1.py: 환자 선별 과정(Flowchart) 데이터를 분석하고 제외 사유별 빈도를 계산합니다.

table_1.py: 연구 대상자의 기본 임상 특성(Baseline characteristics; 연령, BMI, T-score, 성별 근육 두께 등)을 요약하여 표(Table) 포맷으로 출력합니다.

figure_3.py: 근육 두께 및 전체 두께와 최저 T-score 간의 선형 상관관계를 보여주는 2x3 산점도 그리드를 생성합니다. (전체, 남성, 여성 그룹 분리)

figure_4.py: 다양한 해부학적 측정 부위(측면, 전후방 등)에 따른 피어슨 상관계수(Pearson's r)를 비교하는 막대그래프를 생성합니다.

figure_5.py: 측면 근육 두께를 3분위(Tertile)로 나누어 각 그룹의 골다공증 유병률을 시각화하고 하위 그룹 간 통계적 유의성을 검증합니다.

figure_6.py: 종아리 측면 근육 두께가 요추(Lumbar)와 고관절(Hip) 골밀도 중 어느 부위와 더 강한 상관관계를 가지는지 비교 분석합니다.

figure_7.py: 골다공증 예측에 대한 다변량 로지스틱 회귀분석을 수행하고, 주요 교란 변수(나이, 성별, BMI)를 통제한 독립적 승산비(Adjusted Odds Ratios)를 포레스트 플롯(Forest Plot)으로 시각화합니다.

figure_8s7.py: 수신자 조작 특성(ROC) 곡선을 그리고, 유든 지수(Youden Index)를 최대화하는 최적의 임상 진단 컷오프(Cut-off) 값을 도출하여 혼동 행렬(Confusion Matrix)과 함께 계산합니다.

figure_9.py: 남녀의 기초 근육량 차이를 반영하여 성별(Male/Female) 분리 맞춤형 ROC 곡선을 그리고 개별 컷오프 값을 시각화합니다.

🚀 실행 방법 (How to Run)
데이터 파일이 스크립트와 같은 폴더에 위치해 있는지 확인한 후, 터미널에서 원하는 스크립트를 개별적으로 실행합니다.

Bash
python figure_7.py
실행이 완료되면 논문 게재 품질(Publication-ready, 300 DPI)의 결과 그래프(.png) 및 요약 통계 표(.csv 또는 .xlsx)가 동일한 디렉토리 또는 지정된 아웃풋 폴더에 자동으로 저장됩니다.
