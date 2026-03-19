import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 1. 파일 경로 자동 설정
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

file_path = os.path.join(current_dir, "dataset_2.csv")

# 2. 데이터 로드 및 전처리
print(f"데이터를 불러오는 중입니다: {file_path}")
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='cp949') 

# 분석에 포함할 환자('O')만 필터링
df = df[df['데이터'] == 'O'].copy()

# 계산을 위해 숫자형으로 변환
num_cols = ['Lat muscle', 'AP muscle', 'Lat total', 'AP total', 'min t score']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. 결합 지표 생성 (덧셈)
df['Muscle_Sum'] = df['Lat muscle'] + df['AP muscle']
df['Total_Sum'] = df['Lat total'] + df['AP total']

# 4. 2x3 그리드 그래프 그리기 함수 (통일된 스타일 적용!)
def plot_6_grid_unified(data, title, filename):
    # 🚨 기존의 불안정한 plt.style.use(...) 제거! 
    # 대신 밑에서 모든 테두리와 그리드를 수동으로 똑같이 세팅합니다.
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
    
    # 폰트 통일
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    plot_defs = [
        # 윗줄: 순수 근육 - Muted Blue
        {'x': 'Lat muscle', 'r': 0, 'c': 0, 'label': 'Lateral Muscle Thickness (mm)', 'color': '#4C78A8'}, 
        {'x': 'AP muscle', 'r': 0, 'c': 1, 'label': 'AP Muscle Thickness (mm)', 'color': '#4C78A8'},
        {'x': 'Muscle_Sum', 'r': 0, 'c': 2, 'label': 'AP + Lateral Muscle Sum (mm)', 'color': '#4C78A8'}, 
        # 아랫줄: 둘레 전체 - Muted Orange
        {'x': 'Lat total', 'r': 1, 'c': 0, 'label': 'Lateral Total Thickness (mm)', 'color': '#F2A65A'},   
        {'x': 'AP total', 'r': 1, 'c': 1, 'label': 'AP Total Thickness (mm)', 'color': '#F2A65A'},
        {'x': 'Total_Sum', 'r': 1, 'c': 2, 'label': 'AP + Lateral Total Sum (mm)', 'color': '#F2A65A'} 
    ]
    
    for p in plot_defs:
        ax = axes[p['r'], p['c']]
        temp_df = data[[p['x'], 'min t score']].dropna()
        
        # 산점도 및 회귀선 (선은 강조를 위해 두껍고 빨간색으로)
        sns.regplot(data=temp_df, x=p['x'], y='min t score', ax=ax,
                    scatter_kws={'alpha': 0.5, 's': 50, 'color': p['color']},
                    line_kws={'color': '#d62728', 'linewidth': 3})
        
        ax.set_xlabel(p['label'], fontsize=14, fontweight='bold')
        ax.set_ylabel('Minimum T-score' if p['c'] == 0 else '', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # ========================================================
        # 🚨 [셰프의 킥] 3-1, 3-2, 3-3 완벽 통일 세팅
        # ========================================================
        # 1) 연한 점선 그리드 추가 (모든 그래프 동일)
        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7, color='gray') 
        ax.set_axisbelow(True) # 점들 뒤로 그리드를 배치
        
        # 2) 바깥 테두리(Spines) 굵기와 색상을 명확하게 수동 지정
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # 적당히 진한 굵기
            spine.set_color('black')
        # ========================================================

        # 상관계수(r) 및 유의확률(p-value) 박스
        if len(temp_df) > 2:
            r, p_val = pearsonr(temp_df[p['x']], temp_df['min t score'])
            p_text = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            ax.text(0.05, 0.90, f"r = {r:.3f}\n{p_text}", transform=ax.transAxes, 
                    fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.3'))
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(current_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 저장 완료: {save_path}")

# 5. 그래프 3종 세트 다시 굽기
print("\n🔥 통일된 스타일로 그래프 생성을 시작합니다...")
plot_6_grid_unified(df[df['성별'] == 'M'], 'Figure 3-1: Muscle vs Total Measurements vs BMD (Male Only)', 'Figure3-1_Male.png')
plot_6_grid_unified(df[df['성별'] == 'F'], 'Figure 3-2: Muscle vs Total Measurements vs BMD (Female Only)', 'Figure3-2_Female.png')
plot_6_grid_unified(df, 'Figure 3-3: Muscle vs Total Measurements vs BMD (All Patients)', 'Figure3-3_All.png')
print("🎉 3종 세트 통일 완료!")