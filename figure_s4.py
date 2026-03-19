import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 파일 경로 및 데이터 로드 (인코딩 에러 방지 포함)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

file_path = os.path.join(current_dir, "dataset_2.csv")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949') 

# 포함된 환자만 필터링
df = df[df['데이터'] == 'O'].copy()

# 2. 텍스트 라벨 공백 제거 및 결측치 손질
df['Lumbar_clean'] = df['Lumbar'].astype(str).str.strip().str.capitalize().replace('Nan', np.nan)
df['Hip_clean'] = df['Hip'].astype(str).str.strip().str.capitalize().replace('Nan', np.nan)

# ==========================================
# 🚨 [추가된 기능] Other에 들어가는 항목 요약 출력
# ==========================================
lumbar_other_mask = df['Lumbar_clean'] == 'L total'
hip_other_mask = ~df['Hip_clean'].isin(['Hip neck', 'Hip total']) & df['Hip_clean'].notna()

lumbar_others = df.loc[lumbar_other_mask, 'Lumbar'].dropna().unique()
hip_others = df.loc[hip_other_mask, 'Hip'].dropna().unique()

print("\n" + "="*55)
print(" 👨‍🍳 셰프의 재료 손질 리포트 ('Other' 범주 요약)")
print("="*55)
print(f"📍 Lumbar(요추) 'Other' 병합 항목  : {list(lumbar_others)}")
print(f"📍 Hip(고관절) 'Other' 병합 항목    : {list(hip_others)}")
print("="*55 + "\n")
# ==========================================

# 주요 부위만 묶어주기 (기타 항목은 Other로)
df['Lumbar_clean'] = df['Lumbar_clean'].replace({'L total': 'Other', 'L3 ': 'L3'})
df.loc[~df['Hip_clean'].isin(['Hip neck', 'Hip total']), 'Hip_clean'] = 'Other'

# 비율(%) 계산
lumbar_counts = df['Lumbar_clean'].value_counts(normalize=True) * 100
hip_counts = df['Hip_clean'].value_counts(normalize=True) * 100

# 가로형 바 차트를 위해 오름차순 정렬 (가장 큰 값이 맨 위로 가도록)
lumbar_counts = lumbar_counts.sort_values(ascending=True)
hip_counts = hip_counts.sort_values(ascending=True)

# ==========================================
# 3. Publication Ready Plot 생성
# ==========================================
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

fig_supp, axes_supp = plt.subplots(1, 2, figsize=(14, 6))

# 지정해주신 모던(Muted) 색상 설정
COLOR_LOW = "#4C78A8"     # muted blue (Lumbar에 사용)
COLOR_HIGH = "#F2A65A"    # muted orange (Hip에 사용)

# --- (A) Lumbar 부위 가로형 바 차트 ---
bars1 = axes_supp[0].barh(lumbar_counts.index, lumbar_counts.values, 
                          color=COLOR_LOW, edgecolor='black', height=0.6, linewidth=1.5)
axes_supp[0].set_title('(A) Lowest T-score Site (Lumbar)', fontsize=16, fontweight='bold', pad=15)
axes_supp[0].set_xlabel('Percentage (%)', fontsize=14, fontweight='bold')

# 막대 끝에 퍼센트 수치 달아주기
for bar in bars1:
    width = bar.get_width()
    axes_supp[0].annotate(f'{width:.1f}%',
                          xy=(width, bar.get_y() + bar.get_height() / 2),
                          xytext=(5, 0), # 막대 끝에서 5pt 띄움
                          textcoords="offset points",
                          ha='left', va='center', fontsize=13, fontweight='bold')
# 글씨가 잘리지 않도록 X축 한계치 넉넉하게 설정
axes_supp[0].set_xlim(0, max(lumbar_counts.values) * 1.25)

# --- (B) Hip 부위 가로형 바 차트 ---
bars2 = axes_supp[1].barh(hip_counts.index, hip_counts.values, 
                          color=COLOR_HIGH, edgecolor='black', height=0.6, linewidth=1.5)
axes_supp[1].set_title('(B) Lowest T-score Site (Hip)', fontsize=16, fontweight='bold', pad=15)
axes_supp[1].set_xlabel('Percentage (%)', fontsize=14, fontweight='bold')

for bar in bars2:
    width = bar.get_width()
    axes_supp[1].annotate(f'{width:.1f}%',
                          xy=(width, bar.get_y() + bar.get_height() / 2),
                          xytext=(5, 0),
                          textcoords="offset points",
                          ha='left', va='center', fontsize=13, fontweight='bold')
axes_supp[1].set_xlim(0, max(hip_counts.values) * 1.25)

# 깔끔한 축 및 그리드 포맷팅
for ax in axes_supp:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#e0e0e0', linestyle='--', linewidth=1) # 세로 점선 추가
    
    # Y축 항목 이름(L1, Hip neck 등)을 굵게 처리하여 가독성 업!
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

plt.tight_layout()
save_path = os.path.join(current_dir, 'SuppFigure_Muted_Palette.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Supplementary Figure (Muted Palette 반영) 저장 완료: {save_path}")