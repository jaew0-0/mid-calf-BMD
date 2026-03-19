import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm

# ============================================
# 1) 경로 설정
# ============================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_2.csv"
OUT_DIR = BASE_DIR / "figure7_outputs"
OUT_DIR.mkdir(exist_ok=True)

DPI = 300

# 요청 색감
COLOR_MID = "#9E9E9E"     # gray
COLOR_HIGH = "#F2A65A"    # muted orange
EDGE = "#333333"
GRID = "#D9D9D9"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ============================================
# 2) 데이터 로드
# ============================================

last_error = None
for enc in ["utf-8", "cp949", "euc-kr"]:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        print(f"[OK] 데이터 로드 완료 (encoding={enc})")
        break
    except Exception as e:
        last_error = e
else:
    raise last_error

# ============================================
# 3) 유틸
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

def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def normalize_sex_female_vs_male(x):
    s = str(x).strip().lower()
    if s in ["f", "female", "woman", "여", "여자"]:
        return 1   # female
    if s in ["m", "male", "man", "남", "남자"]:
        return 0   # male
    return np.nan

def normalize_binary_yes(series):
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.isin({"yes", "y", "1", "true"}).astype(int)

# ============================================
# 4) 컬럼 찾기
# ============================================

COL_LAT_MUSCLE = find_column(df, ["Lat muscle", "lateral muscle", "Lateral muscle"], required=True)
COL_AGE = find_column(df, ["시행시 나이 (Knee  사진 기준)", "시행시 나이", "Age", "age"], required=True)
COL_BMI = find_column(df, ["BMI", "bmi"], required=True)
COL_SEX = find_column(df, ["성별", "sex", "gender"], required=True)
COL_MIN_T = find_column(df, ["min t score", "Minimum T-score", "minimum t score"], required=True)
COL_OP = find_column(df, ["osteoporosis", "Osteoporosis"], required=False)

# numeric
df[COL_LAT_MUSCLE] = to_numeric(df[COL_LAT_MUSCLE])
df[COL_AGE] = to_numeric(df[COL_AGE])
df[COL_BMI] = to_numeric(df[COL_BMI])
df[COL_MIN_T] = to_numeric(df[COL_MIN_T])

# 방향 통일 변수
# female vs male
df["_female_"] = df[COL_SEX].apply(normalize_sex_female_vs_male)

# outcome
if COL_OP is not None:
    df["_osteoporosis_"] = normalize_binary_yes(df[COL_OP])
else:
    df["_osteoporosis_"] = (df[COL_MIN_T] <= -2.5).astype(int)

# 위험 증가 방향으로 코딩
# muscle decrease: 값이 클수록 위험 감소라면 decrease 방향으로 뒤집기
df["_lat_muscle_decrease_per10_"] = -df[COL_LAT_MUSCLE] / 10.0

# age increase
df["_age_increase_per10_"] = df[COL_AGE] / 10.0

# bmi decrease
df["_bmi_decrease_per1_"] = -df[COL_BMI]

model_df = df[[
    "_osteoporosis_",
    "_lat_muscle_decrease_per10_",
    "_female_",
    "_age_increase_per10_",
    "_bmi_decrease_per1_"
]].dropna().copy()

print(f"[INFO] Logistic regression N = {len(model_df)}")

# ============================================
# 5) logistic regression
# ============================================

y = model_df["_osteoporosis_"]
X = model_df[[
    "_lat_muscle_decrease_per10_",
    "_female_",
    "_age_increase_per10_",
    "_bmi_decrease_per1_"
]]
X = sm.add_constant(X)

model = sm.Logit(y, X).fit(disp=False)

params = model.params
conf = model.conf_int()
pvals = model.pvalues

result_rows = []
label_map = {
    "_lat_muscle_decrease_per10_": "Lower lateral muscle thickness\n(per 10 mm decrease)",
    "_female_": "Female sex\n(vs Male)",
    "_age_increase_per10_": "Older age\n(per 10-year increase)",
    "_bmi_decrease_per1_": "Lower BMI\n(per 1 kg/m² decrease)",
}

for var in [
    "_lat_muscle_decrease_per10_",
    "_female_",
    "_age_increase_per10_",
    "_bmi_decrease_per1_"
]:
    beta = params[var]
    ci_low = conf.loc[var, 0]
    ci_high = conf.loc[var, 1]

    result_rows.append({
        "Variable": label_map[var],
        "OR": np.exp(beta),
        "CI_low": np.exp(ci_low),
        "CI_high": np.exp(ci_high),
        "p-value": pvals[var],
    })

result_df = pd.DataFrame(result_rows)

# OR 큰 순으로 정렬
result_df = result_df.sort_values("OR", ascending=False).reset_index(drop=True)

result_df["OR_CI_text"] = result_df.apply(
    lambda r: f"{r['OR']:.2f} ({r['CI_low']:.2f}-{r['CI_high']:.2f})", axis=1
)

# 저장
xlsx_path = OUT_DIR / "Figure7_oriented_logistic_regression_results.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    result_df.to_excel(writer, sheet_name="multivariable_OR", index=False)
    pd.DataFrame({
        "Coefficient": params,
        "p-value": pvals
    }).to_excel(writer, sheet_name="model_coefficients")

print(f"[OK] 결과 엑셀 저장 완료: {xlsx_path}")

# ============================================
# 6) Forest plot
# ============================================

fig, ax = plt.subplots(figsize=(7.8, 4.9))

plot_df = result_df.iloc[::-1].reset_index(drop=True)
ypos = np.arange(len(plot_df))

for i, row in plot_df.iterrows():
    ax.plot(
        [row["CI_low"], row["CI_high"]],
        [i, i],
        color=COLOR_MID,
        linewidth=2.0
    )
    ax.scatter(
        row["OR"], i,
        color=COLOR_HIGH,
        s=62,
        zorder=3,
        edgecolor=EDGE,
        linewidth=0.6
    )

ax.axvline(1.0, color="#8A8A8A", linestyle="--", linewidth=1.0)

ax.set_yticks(ypos)
ax.set_yticklabels(plot_df["Variable"], fontsize=10)
ax.set_xlabel("Adjusted odds ratio (95% CI)", fontsize=11, fontweight="bold")
ax.set_title("Figure 7. Multivariable predictors of osteoporosis", fontsize=13, fontweight="bold")
ax.grid(axis="x", color=GRID, linewidth=0.5, alpha=0.8)

xmin = min(0.95, plot_df["CI_low"].min() * 0.95)
xmax = plot_df["CI_high"].max() * 1.45
ax.set_xlim(left=xmin, right=xmax)

for i, row in plot_df.iterrows():
    ax.text(
        xmax * 0.995,
        i,
        row["OR_CI_text"],
        ha="right",
        va="center",
        fontsize=9
    )

plt.tight_layout()
png_path = OUT_DIR / "Figure7_adjusted_OR_forest_plot_oriented.png"
plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"[OK] Figure 7 저장 완료: {png_path}")