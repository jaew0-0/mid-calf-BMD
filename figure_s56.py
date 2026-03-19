import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_2.csv"
OUT_DIR = BASE_DIR / "supplementary_subgroup_correlation"
OUT_DIR.mkdir(exist_ok=True)

BLUE = "#4C78A8"
ORANGE = "#F2A65A"
GRID = "#D9D9D9"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# load
last_error = None
for enc in ["utf-8", "cp949", "euc-kr"]:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        break
    except Exception as e:
        last_error = e
else:
    raise last_error

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
        raise KeyError(f"컬럼을 찾을 수 없습니다: {candidates}")
    return None

def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def r_ci(r, n):
    if n <= 3:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zc = stats.norm.ppf(0.975)
    lo = z - zc * se
    hi = z + zc * se
    return np.tanh(lo), np.tanh(hi)

def subgroup_corr(df, group_col, x_col, y_col):
    rows = []
    for g, d in df.groupby(group_col):
        dd = d[[x_col, y_col]].dropna()
        if len(dd) < 4:
            continue
        r, p = stats.pearsonr(dd[x_col], dd[y_col])
        lo, hi = r_ci(r, len(dd))
        rows.append({
            "Group": g,
            "N": len(dd),
            "r": r,
            "CI_low": lo,
            "CI_high": hi,
            "p-value": p
        })
    return pd.DataFrame(rows)

COL_LAT_MUSCLE = find_column(df, ["Lat muscle", "lateral muscle", "Lateral muscle"], required=True)
COL_AGE = find_column(df, ["시행시 나이 (Knee  사진 기준)", "시행시 나이", "Age", "age"], required=True)
COL_BMI = find_column(df, ["BMI", "bmi"], required=True)
COL_MIN_T = find_column(df, ["min t score", "Minimum T-score", "minimum t score"], required=True)

for col in [COL_LAT_MUSCLE, COL_AGE, COL_BMI, COL_MIN_T]:
    df[col] = to_numeric(df[col])

# age group
df["_age_group_"] = pd.cut(
    df[COL_AGE],
    bins=[0, 60, 70, 80, 200],
    labels=["<60", "60-69", "70-79", "≥80"],
    right=False
)

# bmi group
df["_bmi_group_"] = pd.cut(
    df[COL_BMI],
    bins=[0, 23, 25, 30, 100],
    labels=["<23", "23-24.9", "25-29.9", "≥30"],
    right=False
)

age_df = subgroup_corr(df, "_age_group_", COL_LAT_MUSCLE, COL_MIN_T)
bmi_df = subgroup_corr(df, "_bmi_group_", COL_LAT_MUSCLE, COL_MIN_T)

# save
xlsx_path = OUT_DIR / "subgroup_correlation_summary.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    age_df.to_excel(writer, sheet_name="age_group", index=False)
    bmi_df.to_excel(writer, sheet_name="bmi_group", index=False)

# plot
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)

# age
ax = axes[0]
for i, row in age_df.iloc[::-1].reset_index(drop=True).iterrows():
    ax.plot([row["CI_low"], row["CI_high"]], [i, i], color=BLUE, linewidth=1.6)
    ax.scatter(row["r"], i, color=ORANGE, s=50, zorder=3)
ax.axvline(0, color="#888888", linestyle="--", linewidth=1.0)
ax.set_yticks(np.arange(len(age_df)))
ax.set_yticklabels(age_df.iloc[::-1]["Group"])
ax.set_title("Supplementary Figure S5. Age-stratified correlation", fontsize=11, fontweight="bold")
ax.set_xlabel("Pearson r (95% CI)", fontsize=10, fontweight="bold")
ax.grid(axis="x", color=GRID, linewidth=0.5)

# bmi
ax = axes[1]
for i, row in bmi_df.iloc[::-1].reset_index(drop=True).iterrows():
    ax.plot([row["CI_low"], row["CI_high"]], [i, i], color=BLUE, linewidth=1.6)
    ax.scatter(row["r"], i, color=ORANGE, s=50, zorder=3)
ax.axvline(0, color="#888888", linestyle="--", linewidth=1.0)
ax.set_yticks(np.arange(len(bmi_df)))
ax.set_yticklabels(bmi_df.iloc[::-1]["Group"])
ax.set_title("Supplementary Figure S6. BMI-stratified correlation", fontsize=11, fontweight="bold")
ax.set_xlabel("Pearson r (95% CI)", fontsize=10, fontweight="bold")
ax.grid(axis="x", color=GRID, linewidth=0.5)

plt.tight_layout()
png_path = OUT_DIR / "Supplementary_S5_S6_subgroup_correlation.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] 저장 완료: {png_path}")
print(f"[OK] 저장 완료: {xlsx_path}")