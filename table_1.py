import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# -------------------------------------------------
# 1) Load data
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_1.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"파일이 없습니다: {DATA_PATH}")

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

# -------------------------------------------------
# 2) Helper functions
# -------------------------------------------------

def find_column(df, candidates, required=False):
    cols = list(df.columns)
    cols_norm = {c.strip().lower(): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols_norm:
            return cols_norm[key]

    if required:
        raise KeyError(f"다음 후보 컬럼을 찾을 수 없습니다: {candidates}\n현재 컬럼: {cols}")
    return None

def to_numeric_safe(series):
    return pd.to_numeric(series, errors="coerce")

def format_mean_sd_minmax(series, digits=1):
    s = to_numeric_safe(series).dropna()
    if len(s) == 0:
        return ""
    mean = s.mean()
    sd = s.std(ddof=1)
    min_v = s.min()
    max_v = s.max()
    return f"{mean:.{digits}f} ± {sd:.{digits}f} ({min_v:.{digits}f}-{max_v:.{digits}f})"

def format_count_percent(series, positive_values=None):
    if positive_values is None:
        positive_values = {"yes", "y", "1", "true"}

    s = series.fillna("").astype(str).str.strip().str.lower()
    valid = s[s != ""]
    if len(valid) == 0:
        return ""

    n_pos = valid.isin(positive_values).sum()
    pct = (n_pos / len(valid)) * 100
    return f"{n_pos} ({pct:.1f}%)"

def normalize_sex_value(x):
    s = str(x).strip().lower()
    if s in ["m", "male", "man", "남", "남자"]:
        return "Male"
    if s in ["f", "female", "woman", "여", "여자"]:
        return "Female"
    return np.nan

def format_pvalue(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def continuous_pvalue(male_series, female_series):
    m = to_numeric_safe(male_series).dropna()
    f = to_numeric_safe(female_series).dropna()

    if len(m) < 2 or len(f) < 2:
        return np.nan

    # Welch's t-test
    _, p = stats.ttest_ind(m, f, equal_var=False, nan_policy="omit")
    return p

def binary_pvalue(male_series, female_series, positive_values=None):
    if positive_values is None:
        positive_values = {"yes", "y", "1", "true"}

    m = male_series.fillna("").astype(str).str.strip().str.lower()
    f = female_series.fillna("").astype(str).str.strip().str.lower()

    m = m[m != ""]
    f = f[f != ""]

    if len(m) == 0 or len(f) == 0:
        return np.nan

    m_yes = m.isin(positive_values).sum()
    m_no = len(m) - m_yes
    f_yes = f.isin(positive_values).sum()
    f_no = len(f) - f_yes

    table = np.array([[m_yes, m_no], [f_yes, f_no]])

    # expected count 작은 경우 Fisher, 아니면 chi-square
    try:
        chi2, p_chi, dof, expected = stats.chi2_contingency(table)
        if (expected < 5).any():
            _, p = stats.fisher_exact(table)
        else:
            p = p_chi
    except Exception:
        return np.nan

    return p

# -------------------------------------------------
# 3) Detect columns
# -------------------------------------------------

COL_DATA = find_column(df, ["데이터"], required=True)
COL_SEX = find_column(df, ["성별", "sex", "gender"], required=True)
COL_OP = find_column(df, ["osteoporosis", "Osteoporosis"], required=True)

COL_AGE = find_column(df, ["시행시 나이 (Knee  사진 기준)", "시행시 나이", "Age", "age"])
COL_HEIGHT = find_column(df, ["Height", "height", "키", "신장"])
COL_WEIGHT = find_column(df, ["Weight", "weight", "몸무게", "체중"])
COL_BMI = find_column(df, ["BMI", "bmi"])
COL_LUMBAR_T = find_column(df, ["Lumbar T score", "lumbar t score"])
COL_HIP_T = find_column(df, ["Hip T score", "hip t score"])
COL_MIN_T = find_column(df, ["min t score", "Minimum T-score", "minimum t score"])
COL_LAT_MUSCLE = find_column(df, ["Lat muscle", "lateral muscle", "Lateral muscle"])
COL_AP_MUSCLE = find_column(df, ["AP muscle", "ap muscle"])

COL_AP_LAT_SUM = None
if COL_LAT_MUSCLE is not None and COL_AP_MUSCLE is not None:
    df["_AP_LAT_MUSCLE_SUM_"] = to_numeric_safe(df[COL_LAT_MUSCLE]) + to_numeric_safe(df[COL_AP_MUSCLE])
    COL_AP_LAT_SUM = "_AP_LAT_MUSCLE_SUM_"

# -------------------------------------------------
# 4) Define study population
# -------------------------------------------------

df[COL_DATA] = df[COL_DATA].fillna("").astype(str).str.strip()
df[COL_SEX] = df[COL_SEX].apply(normalize_sex_value)

study = df[df[COL_DATA].str.upper() == "O"].copy()

total_n = len(study)
male_df = study[study[COL_SEX] == "Male"].copy()
female_df = study[study[COL_SEX] == "Female"].copy()

male_n = len(male_df)
female_n = len(female_df)

print(f"[INFO] Study population N = {total_n}")
print(f"[INFO] Male N = {male_n}")
print(f"[INFO] Female N = {female_n}")

# -------------------------------------------------
# 5) Build Table 1 content
# -------------------------------------------------

rows = []

def add_continuous_row(label, colname, digits=1):
    if colname is None:
        return
    p = continuous_pvalue(male_df[colname], female_df[colname])
    rows.append({
        "Variable": label,
        f"Total (n={total_n})": format_mean_sd_minmax(study[colname], digits=digits),
        f"Male (n={male_n})": format_mean_sd_minmax(male_df[colname], digits=digits),
        f"Female (n={female_n})": format_mean_sd_minmax(female_df[colname], digits=digits),
        "p-value": format_pvalue(p),
    })

def add_binary_row(label, colname):
    if colname is None:
        return
    p = binary_pvalue(male_df[colname], female_df[colname])
    rows.append({
        "Variable": label,
        f"Total (n={total_n})": format_count_percent(study[colname]),
        f"Male (n={male_n})": format_count_percent(male_df[colname]),
        f"Female (n={female_n})": format_count_percent(female_df[colname]),
        "p-value": format_pvalue(p),
    })

add_binary_row("Osteoporosis, n (%)", COL_OP)
add_continuous_row("Age (years)", COL_AGE, digits=1)
add_continuous_row("Height (cm)", COL_HEIGHT, digits=1)
add_continuous_row("Weight (kg)", COL_WEIGHT, digits=1)
add_continuous_row("BMI (kg/m²)", COL_BMI, digits=1)
add_continuous_row("Lumbar T-score", COL_LUMBAR_T, digits=1)
add_continuous_row("Hip T-score", COL_HIP_T, digits=1)
add_continuous_row("Minimum T-score", COL_MIN_T, digits=1)
add_continuous_row("Lateral muscle thickness (mm)", COL_LAT_MUSCLE, digits=1)
add_continuous_row("AP muscle thickness (mm)", COL_AP_MUSCLE, digits=1)
add_continuous_row("AP + Lateral muscle sum (mm)", COL_AP_LAT_SUM, digits=1)

table1_df = pd.DataFrame(rows)

if table1_df.empty:
    raise ValueError("Table 1에 넣을 수 있는 변수 컬럼을 찾지 못했습니다.")

print("\n[INFO] Table 1 preview:")
print(table1_df.to_string(index=False))

# -------------------------------------------------
# 6) Save Excel
# -------------------------------------------------

excel_path = BASE_DIR / "table1_baseline_characteristics.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    table1_df.to_excel(writer, sheet_name="Table1", index=False)

print(f"[OK] Excel 저장 완료: {excel_path}")

# -------------------------------------------------
# 7) Draw PNG
# -------------------------------------------------

title = "Table 1. Baseline characteristics of the study population."

n_rows = len(table1_df)
fig_h = 0.55 * (n_rows + 2) + 1.2
fig_w = 18

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

fig.text(
    0.02, 0.97, title,
    fontsize=16, fontweight="bold", ha="left", va="top"
)

cell_text = table1_df.values.tolist()
col_labels = table1_df.columns.tolist()

tbl = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="left",
    colLoc="left",
    bbox=[0.0, 0.02, 1.0, 0.90],
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(11)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.8)

    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#EDEDED")
        cell.set_height(0.065)
    else:
        cell.set_height(0.06)

    if c == 0:
        cell.set_text_props(ha="left")
        cell.PAD = 0.02
    else:
        cell.set_text_props(ha="center")
        cell.PAD = 0.01

col_widths = [0.28, 0.21, 0.21, 0.21, 0.09]
for c in range(len(col_labels)):
    for r in range(n_rows + 1):
        tbl[(r, c)].set_width(col_widths[c])

fig.text(
    0.02, 0.01,
    "Continuous variables are presented as mean ± SD (min-max) and compared using Welch's t-test. "
    "Binary variables are presented as n (%) and compared using chi-square or Fisher's exact test.",
    fontsize=10, ha="left", va="bottom"
)

png_path = BASE_DIR / "table1_baseline_characteristics.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] PNG 저장 완료: {png_path}")