import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

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

# 포함된 환자만 필터링 및 숫자형 변환
df = df[df['데이터'] == 'O'].copy()
for col in ['Lat muscle', 'Lumbar T score', 'Hip T score']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_fig6 = df.dropna(subset=['Lat muscle', 'Lumbar T score', 'Hip T score']).copy()

# 2. 상관계수 계산
r_lumbar, p_lumbar = pearsonr(df_fig6['Lat muscle'], df_fig6['Lumbar T score'])
r_hip, p_hip = pearsonr(df_fig6['Lat muscle'], df_fig6['Hip T score'])

# p-value 텍스트 포맷팅 함수 (*** 제거)
def format_p_value(p):
    if p < 0.001:
        return 'p < 0.001'
    else:
        return f'p = {p:.3f}'

p_lumbar_text = format_p_value(p_lumbar)
p_hip_text = format_p_value(p_hip)

# 3. 논문용 스타일 및 지정해주신 모던(Muted) 색상 설정
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

COLOR_LOW = "#4C78A8"     # muted blue (Lumbar에 사용)
COLOR_MID = "#9E9E9E"     # gray (이번 그래프에서는 2개 항목 비교라 생략)
COLOR_HIGH = "#F2A65A"    # muted orange (Hip에 사용)

color_lumbar = COLOR_LOW
color_hip = COLOR_HIGH

# (A) Lumbar 산포도
sns.regplot(x='Lat muscle', y='Lumbar T score', data=df_fig6, ax=axes[0],
            scatter_kws={'alpha': 0.5, 'color': color_lumbar}, 
            line_kws={'color': '#2C4A6B', 'linewidth': 2.5}) # 회귀선은 가독성을 위해 살짝 더 진하게!
axes[0].set_title('(A) Correlation with Lumbar T-score', fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel('Lateral Muscle Thickness (mm)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Lumbar T-score', fontsize=14, fontweight='bold')

# 텍스트 박스 삽입 (p-value 적용)
axes[0].text(0.05, 0.95, f'r = {r_lumbar:.3f}\n{p_lumbar_text}', transform=axes[0].transAxes,
             fontsize=14, fontweight='bold', va='top', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3'))

# (B) Hip 산포도
sns.regplot(x='Lat muscle', y='Hip T score', data=df_fig6, ax=axes[1],
            scatter_kws={'alpha': 0.5, 'color': color_hip}, 
            line_kws={'color': '#B36B22', 'linewidth': 2.5}) 
axes[1].set_title('(B) Correlation with Hip T-score', fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel('Lateral Muscle Thickness (mm)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Hip T-score', fontsize=14, fontweight='bold')

# 텍스트 박스 삽입 (p-value 적용)
axes[1].text(0.05, 0.95, f'r = {r_hip:.3f}\n{p_hip_text}', transform=axes[1].transAxes,
             fontsize=14, fontweight='bold', va='top', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3'))

# (C) 상관계수 비교 바 차트
df_female = df_fig6[df_fig6['성별'] == 'F']
r_lumbar_f, _ = pearsonr(df_female['Lat muscle'], df_female['Lumbar T score'])
r_hip_f, _ = pearsonr(df_female['Lat muscle'], df_female['Hip T score'])

bar_data = pd.DataFrame({'Group': ['All Patients', 'All Patients', 'Female Only', 'Female Only'], 
                         'Site': ['Lumbar', 'Hip', 'Lumbar', 'Hip'], 
                         'r': [r_lumbar, r_hip, r_lumbar_f, r_hip_f]})

sns.barplot(x='Group', y='r', hue='Site', data=bar_data, ax=axes[2],
            palette=[color_lumbar, color_hip], edgecolor='black', linewidth=1.5)
axes[2].set_title('(C) Pearson\'s r Comparison', fontsize=16, fontweight='bold', pad=15)
axes[2].set_xlabel('')
axes[2].set_ylabel('Pearson\'s r', fontsize=14, fontweight='bold')
axes[2].set_ylim(0, max(bar_data['r']) * 1.3)

# 바 차트 위에 수치 올리기
for p in axes[2].patches:
    height = p.get_height()
    if height > 0:
        axes[2].annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height), 
                         ha='center', va='bottom', fontsize=13, xytext=(0, 4), 
                         textcoords='offset points', fontweight='bold')

# 테두리 및 축 정리 (깔끔한 논문 스타일)
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(labelsize=13)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#e0e0e0', linestyle='--', linewidth=1)

# 이미지 저장
plt.tight_layout()
save_path = os.path.join(current_dir, 'Figure6_Muted_Palette.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Figure 6 (Muted Palette & p-value 반영) 저장 완료: {save_path}")