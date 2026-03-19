import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


# ============================================
# 1) 경로 설정
# ============================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_2.csv"
OUT_DIR = BASE_DIR / "figure4_outputs"
OUT_DIR.mkdir(exist_ok=True)

COMBINATION_MODE = "sum"   # "sum" or "product"
DPI = 300


# ============================================
# 2) 데이터 로드
# ============================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"파일이 없습니다: {DATA_PATH}")

last_error = None
for enc in ["utf-8", "cp949", "euc-kr"]:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        used_encoding = enc
        print(f"[OK] 데이터 로드 완료 (encoding={enc})")
        break
    except Exception as e:
        last_error = e
else:
    raise last_error


# ============================================
# 3) 유틸 함수
# ============================================

def find_column(df, candidates, required=False):
    cols = list(df.columns)
    norm_map = {str(c).strip().lower(): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]

    if required:
        raise KeyError(f"컬럼을 찾을 수 없습니다: {candidates}\n현재 컬럼: {cols}")
    return None


def normalize_sex(x):
    s = str(x).strip().lower()
    if s in ["m", "male", "man", "남", "남자"]:
        return "Male"
    if s in ["f", "female", "woman", "여", "여자"]:
        return "Female"
    return np.nan


def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def combine_series(a, b, mode="sum"):
    if mode == "sum":
        return a + b
    elif mode == "product":
        return a * b
    else:
        raise ValueError("mode must be 'sum' or 'product'")


def calc_pearson(df_sub, x_col, y_col):
    d = df_sub[[x_col, y_col]].dropna().copy()
    n = len(d)
    if n < 3:
        return {"n": n, "r": np.nan, "p": np.nan}

    r, p = stats.pearsonr(d[x_col], d[y_col])
    return {"n": n, "r": r, "p": p}


# ============================================
# 4) 컬럼 찾기
# ============================================

COL_SEX = find_column(df, ["성별", "sex", "gender"], required=True)
COL_MIN_T = find_column(df, ["min t score", "Minimum T-score", "minimum t score"], required=True)

COL_LAT_MUSCLE = find_column(df, ["Lat muscle", "lateral muscle", "Lateral muscle"], required=True)
COL_AP_MUSCLE = find_column(df, ["AP muscle", "ap muscle"], required=True)

COL_LAT_TOTAL = find_column(df, ["Lat total", "lateral total", "Lateral total"], required=True)
COL_AP_TOTAL = find_column(df, ["AP total", "ap total"], required=True)

for col in [COL_MIN_T, COL_LAT_MUSCLE, COL_AP_MUSCLE, COL_LAT_TOTAL, COL_AP_TOTAL]:
    df[col] = to_numeric(df[col])

df[COL_SEX] = df[COL_SEX].apply(normalize_sex)

df["_combined_muscle_"] = combine_series(df[COL_AP_MUSCLE], df[COL_LAT_MUSCLE], mode=COMBINATION_MODE)
df["_combined_total_"] = combine_series(df[COL_AP_TOTAL], df[COL_LAT_TOTAL], mode=COMBINATION_MODE)


# ============================================
# 5) 그룹/변수 정의
# ============================================

group_dict = {
    "Male": df[df[COL_SEX] == "Male"].copy(),
    "Female": df[df[COL_SEX] == "Female"].copy(),
    "Total": df.copy(),
}

view_dict = {
    "Lateral": {
        "muscle": COL_LAT_MUSCLE,
        "total": COL_LAT_TOTAL,
    },
    "AP": {
        "muscle": COL_AP_MUSCLE,
        "total": COL_AP_TOTAL,
    },
    "AP+Lateral": {
        "muscle": "_combined_muscle_",
        "total": "_combined_total_",
    },
}


# ============================================
# 6) 상관계수 계산
# ============================================

rows = []

for group_name, gdf in group_dict.items():
    for view_name, cols in view_dict.items():
        muscle_res = calc_pearson(gdf, cols["muscle"], COL_MIN_T)
        total_res = calc_pearson(gdf, cols["total"], COL_MIN_T)

        rows.append({
            "Group": group_name,
            "View": view_name,
            "Measure": "Muscle only",
            "N": muscle_res["n"],
            "Pearson r": muscle_res["r"],
            "p-value": muscle_res["p"],
        })
        rows.append({
            "Group": group_name,
            "View": view_name,
            "Measure": "Total thickness",
            "N": total_res["n"],
            "Pearson r": total_res["r"],
            "p-value": total_res["p"],
        })

summary_df = pd.DataFrame(rows)

csv_path = OUT_DIR / "figure4_barplot_summary.csv"
xlsx_path = OUT_DIR / "figure4_barplot_summary.xlsx"

summary_df.to_csv(csv_path, index=False, encoding=used_encoding)
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)

print(f"[OK] 요약 CSV 저장: {csv_path}")
print(f"[OK] 요약 XLSX 저장: {xlsx_path}")


# ============================================
# 7) Figure 4: bar plot
# ============================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)

group_order = ["Male", "Female", "Total"]
view_order = ["Lateral", "AP", "AP+Lateral"]

bar_width = 0.34
x = np.arange(len(view_order))

color_muscle = "#4C78A8"
color_total = "#F58518"

for ax, group_name in zip(axes, group_order):
    sub = summary_df[summary_df["Group"] == group_name].copy()

    muscle_vals = []
    total_vals = []

    for view_name in view_order:
        muscle_r = sub[(sub["View"] == view_name) & (sub["Measure"] == "Muscle only")]["Pearson r"].values[0]
        total_r = sub[(sub["View"] == view_name) & (sub["Measure"] == "Total thickness")]["Pearson r"].values[0]

        muscle_vals.append(muscle_r)
        total_vals.append(total_r)

    bars1 = ax.bar(
        x - bar_width / 2,
        muscle_vals,
        width=bar_width,
        label="Muscle only",
        color=color_muscle
    )
    bars2 = ax.bar(
        x + bar_width / 2,
        total_vals,
        width=bar_width,
        label="Total thickness",
        color=color_total
    )

    ax.set_title(group_name, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(view_order, fontsize=10)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.25)
    ax.axhline(0, color="black", linewidth=0.8)

    # 숫자 표시
    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            if pd.notna(h):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 0.012,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

axes[0].set_ylabel("Pearson r", fontsize=11)

# y축 범위 자동 설정
all_r = summary_df["Pearson r"].dropna()
ymin = min(0, all_r.min() - 0.08)
ymax = all_r.max() + 0.10
for ax in axes:
    ax.set_ylim(ymin, ymax)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
    fontsize=10
)

fig.suptitle(
    "Figure 4. Pearson correlations with minimum T-score by sex and radiographic view",
    fontsize=13,
    fontweight="bold",
    y=1.08
)

plt.tight_layout()

png_path = OUT_DIR / "Figure4_barplot_correlation_comparison.png"
plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"[OK] Figure 4 저장 완료: {png_path}")
print("[DONE] Figure 4 bar plot 생성 완료")