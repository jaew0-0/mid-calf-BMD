import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 1. 파일 경로 및 데이터 로드
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

file_path = os.path.join(current_dir, "dataset_2.csv")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949') 

# 데이터 필터링 및 숫자형 변환
df = df[df['데이터'] == 'O'].copy()

for col in ['Lat muscle', 'AP muscle', 'Lat total', 'AP total', 'min t score']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Lat muscle', 'AP muscle', 'Lat total', 'AP total', 'min t score'])

# 2. 골다공증 기준 및 파생 변수
df['Osteoporosis'] = (df['min t score'] <= -2.5).astype(int)
df['Muscle_Sum'] = df['Lat muscle'] + df['AP muscle']
df['Total_Sum'] = df['Lat total'] + df['AP total']

# ROC 계산을 위한 방향 뒤집기 (음수 처리)
predict_cols = ['Lat muscle', 'AP muscle', 'Muscle_Sum', 'Lat total', 'AP total', 'Total_Sum']
for col in predict_cols:
    df[f'Neg_{col}'] = -df[col]

def find_optimal_cutoff(target, predicted):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]
    return optimal_threshold, sensitivity, specificity, fpr, tpr, auc(fpr, tpr)

# ==========================================
# [Figure 8] 메인 디시: Lateral Muscle ROC Curve
# ==========================================
opt_cut_neg, sens_lat, spec_lat, fpr_lat, tpr_lat, roc_auc_lat = find_optimal_cutoff(df['Osteoporosis'], df['Neg_Lat muscle'])
opt_cut_lat = -opt_cut_neg 

plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr_lat, tpr_lat, color='black', lw=2.5, label=f'Lateral Muscle (AUC = {roc_auc_lat:.3f})')
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

optimal_idx = np.argmax(tpr_lat - fpr_lat)
ax.plot(fpr_lat[optimal_idx], tpr_lat[optimal_idx], marker='o', color='red', markersize=10, 
        label=f'Optimal Cutoff: <= {opt_cut_lat:.1f} mm\n(Sens: {sens_lat:.2f}, Spec: {spec_lat:.2f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
ax.set_title('Figure 8. ROC Curve for Predicting Osteoporosis', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc="lower right", fontsize=13, frameon=True, edgecolor='black')

ax.tick_params(labelsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'Figure8_ROC_Lateral.png'), dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 🚨 [새로 추가된 기능] 최적점에서의 혼동 행렬 (Confusion Matrix) 표 출력
# ==========================================
# 최적 컷오프를 기준으로 환자를 예측합니다 (컷오프보다 얇거나 같으면 1:위험군, 두꺼우면 0:안전군)
df['Pred_Osteo'] = (df['Lat muscle'] <= opt_cut_lat).astype(int)

# 실제 정답과 셰프의 예측을 비교하여 표(Matrix) 생성
cm = confusion_matrix(df['Osteoporosis'], df['Pred_Osteo'])
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*75)
print(f" 👨‍🍳 셰프의 도마 위: Lateral Muscle 최적 컷오프 ({opt_cut_lat:.1f} mm) 환자 분류 결과")
print("="*75)
print(f"                                [실제 결과: 골다공증 O]    [실제 결과: 정상 X]")
print(f" [예측: 위험군 (<= {opt_cut_lat:.1f} mm)]       {tp} 명 (True Positive)      {fp} 명 (False Positive)")
print(f" [예측: 안전군 (> {opt_cut_lat:.1f} mm)]        {fn} 명 (False Negative)     {tn} 명 (True Negative)")
print("="*75)
print(f" * True Positive  (TP): 진짜 환자를 환자라고 족집게처럼 찾아낸 인원 (민감도 담당)")
print(f" * False Positive (FP): 정상인인데 뼈 약하다고 오진한 인원 (헛스윙)")
print(f" * False Negative (FN): 진짜 환자인데 뼈 튼튼하다고 놓쳐버린 인원")
print(f" * True Negative  (TN): 정상인을 정상이라고 올바르게 돌려보낸 인원 (특이도 담당)\n")

# ==========================================
# [Supplementary] 나머지 부위들의 ROC 곡선
# ==========================================
params = [
    ('AP Muscle', 'Neg_AP muscle'),
    ('Muscle Sum (Lat+AP)', 'Neg_Muscle_Sum'),
    ('Lateral Total', 'Neg_Lat total'),
    ('AP Total', 'Neg_AP total'),
    ('Total Sum (Lat+AP)', 'Neg_Total_Sum')
]

results = [{'Measurement': 'Lateral Muscle', 'AUC': roc_auc_lat, 'Optimal Cutoff (mm)': opt_cut_lat, 
            'Sensitivity': sens_lat, 'Specificity': spec_lat}]

fig_supp, ax_supp = plt.subplots(figsize=(10, 8))
colors = ['#4C78A8', '#F2A65A', '#9E9E9E', '#72B7B2', '#E15759']

ax_supp.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

for i, (name, col_name) in enumerate(params):
    opt_cut_neg, sens, spec, fpr, tpr, roc_auc = find_optimal_cutoff(df['Osteoporosis'], df[col_name])
    
    results.append({
        'Measurement': name,
        'AUC': roc_auc,
        'Optimal Cutoff (mm)': -opt_cut_neg,
        'Sensitivity': sens,
        'Specificity': spec
    })
    ax_supp.plot(fpr, tpr, color=colors[i], lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')

ax_supp.set_xlim([0.0, 1.0])
ax_supp.set_ylim([0.0, 1.05])
ax_supp.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
ax_supp.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
ax_supp.set_title('Supplementary Figure. ROC Curves Comparison', fontsize=16, fontweight='bold', pad=20)
ax_supp.legend(loc="lower right", fontsize=12, frameon=True, edgecolor='black')

ax_supp.tick_params(labelsize=13)
ax_supp.spines['top'].set_visible(False)
ax_supp.spines['right'].set_visible(False)
ax_supp.spines['left'].set_linewidth(1.5)
ax_supp.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'SuppFigure_ROC_All.png'), dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# [Supplementary Table] 영문 포맷 데이터 프레임 출력 및 CSV 저장
# ==========================================
df_results = pd.DataFrame(results)

df_results['AUC'] = df_results['AUC'].map('{:.3f}'.format)
df_results['Optimal Cutoff (mm)'] = df_results['Optimal Cutoff (mm)'].map('{:.1f}'.format)
df_results['Sensitivity'] = df_results['Sensitivity'].map('{:.3f}'.format)
df_results['Specificity'] = df_results['Specificity'].map('{:.3f}'.format)

print("="*75)
print(" 📋 Supplementary Table S2: Predictive Performance for Osteoporosis")
print("="*75)
print(df_results.to_markdown(index=False))
print("="*75 + "\n")

table_path = os.path.join(current_dir, 'Supplementary_Table_ROC.csv')
df_results.to_csv(table_path, index=False)
print(f"✅ CSV 파일이 완벽하게 저장되었습니다: {table_path}")