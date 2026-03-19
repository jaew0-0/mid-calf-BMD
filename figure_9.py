import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 1. 파일 경로 및 데이터 로드 (인코딩 에러 방지)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

file_path = os.path.join(current_dir, "dataset_2.csv")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949') 

# 포함된 환자 필터링 및 숫자형 변환
df = df[df['데이터'] == 'O'].copy()

for col in ['Lat muscle', 'min t score']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Lat muscle', 'min t score', '성별'])

# 2. 골다공증 기준 설정 및 음수 변환
df['Osteoporosis'] = (df['min t score'] <= -2.5).astype(int)
df['Neg_Lat muscle'] = -df['Lat muscle']

# 유든 지수(Youden Index) 기반 최적 컷오프 탐색 함수
def find_optimal_cutoff(target, predicted):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]
    return optimal_threshold, sensitivity, specificity, fpr, tpr, auc(fpr, tpr)

# ==========================================
# 3. 남여 데이터 분리 및 컷오프 개별 계산
# ==========================================
df_m = df[df['성별'] == 'M']
df_f = df[df['성별'] == 'F']

# 남성(Male) 계산
opt_cut_neg_m, sens_m, spec_m, fpr_m, tpr_m, roc_auc_m = find_optimal_cutoff(df_m['Osteoporosis'], df_m['Neg_Lat muscle'])
opt_cut_m = -opt_cut_neg_m

# 여성(Female) 계산
opt_cut_neg_f, sens_f, spec_f, fpr_f, tpr_f, roc_auc_f = find_optimal_cutoff(df_f['Osteoporosis'], df_f['Neg_Lat muscle'])
opt_cut_f = -opt_cut_neg_f

# ==========================================
# [Figure 9] 남여 분리형 ROC Curve 시각화
# ==========================================
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(8, 8))

# 선생님의 시그니처 색감 적용
COLOR_MALE = "#4C78A8"     # Muted Blue (남성)
COLOR_FEMALE = "#F2A65A"   # Muted Orange (여성)

# 1) 여성(Female) ROC 커브 그리기
ax.plot(fpr_f, tpr_f, color=COLOR_FEMALE, lw=3, label=f'Female (AUC = {roc_auc_f:.3f})')
optimal_idx_f = np.argmax(tpr_f - fpr_f)
ax.plot(fpr_f[optimal_idx_f], tpr_f[optimal_idx_f], marker='o', color='red', markersize=10, markeredgecolor='black',
        label=f'Female Cutoff: <= {opt_cut_f:.1f} mm\n(Sens: {sens_f:.2f}, Spec: {spec_f:.2f})')

# 2) 남성(Male) ROC 커브 그리기
ax.plot(fpr_m, tpr_m, color=COLOR_MALE, lw=3, label=f'Male (AUC = {roc_auc_m:.3f})')
optimal_idx_m = np.argmax(tpr_m - fpr_m)
ax.plot(fpr_m[optimal_idx_m], tpr_m[optimal_idx_m], marker='s', color='blue', markersize=10, markeredgecolor='black',
        label=f'Male Cutoff: <= {opt_cut_m:.1f} mm\n(Sens: {sens_m:.2f}, Spec: {spec_m:.2f})')

# 대각선 (기준선)
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# 축 포맷팅
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
ax.set_title('Figure 9. Gender-Specific ROC Curves for Lateral Muscle', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc="lower right", fontsize=12, frameon=True, edgecolor='black')

ax.tick_params(labelsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
fig_path = os.path.join(current_dir, 'Figure9_ROC_Gender.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

# 터미널 출력용 요약 리포트
print("\n" + "="*65)
print(" 👨‍🍳 셰프의 남녀 맞춤형 컷오프(Cut-off) 테이스팅 노트")
print("="*65)
print(f" 🟦 남성 (Male)   - AUC: {roc_auc_m:.3f} | 최적 컷오프: <= {opt_cut_m:.1f} mm")
print(f" 🟧 여성 (Female) - AUC: {roc_auc_f:.3f} | 최적 컷오프: <= {opt_cut_f:.1f} mm")
print("="*65 + "\n")
print(f"✅ 성별 분리 ROC 커브 저장 완료: {fig_path}")