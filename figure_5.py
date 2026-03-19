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
OUT_DIR = BASE_DIR / "figure5_outputs"
OUT_DIR.mkdir(exist_ok=True)

DPI = 300

# 이전 figure와 유사한 차분한 색감
COLOR_LOW = "#4C78A8"     # muted blue
COLOR_MID = "#9E9E9E"     # gray
COLOR_HIGH = "#F2A65A"    # muted orange
BOX_EDGE = "#444444"
POINT_COLOR = "#666666"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


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


def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_binary_yes(series):
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.isin({"yes", "y", "1", "true"})


def p_to_stars(p):
    if pd.isna(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_sig_bar(ax, x1, x2, y, h, text, fontsize=11):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c="black")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# ============================================
# 4) 컬럼 찾기
# ============================================

COL_LAT_MUSCLE = find_column(df, ["Lat muscle", "lateral muscle", "Lateral muscle"], required=True)
COL_MIN_T = find_column(df, ["min t score", "Minimum T-score", "minimum t score"], required=True)
COL_OP = find_column(df, ["osteoporosis", "Osteoporosis"], required=False)

df[COL_LAT_MUSCLE] = to_numeric(df[COL_LAT_MUSCLE])
df[COL_MIN_T] = to_numeric(df[COL_MIN_T])

plot_df = df[[COL_LAT_MUSCLE, COL_MIN_T] + ([COL_OP] if COL_OP is not None else [])].copy()
plot_df = plot_df.dropna(subset=[COL_LAT_MUSCLE, COL_MIN_T]).copy()

if len(plot_df) == 0:
    raise ValueError("분석 가능한 데이터가 없습니다.")


# ============================================
# 5) tertile 생성
# ============================================

plot_df["Lat_muscle_tertile"] = pd.qcut(
    plot_df[COL_LAT_MUSCLE],
    q=3,
    labels=["Low", "Mid", "High"],
    duplicates="drop"
)

plot_df = plot_df.dropna(subset=["Lat_muscle_tertile"]).copy()

if plot_df["Lat_muscle_tertile"].nunique() != 3:
    raise ValueError("Tertile 생성에 실패했습니다. Lat muscle 값 분포를 확인해주세요.")

plot_df["Lat_muscle_tertile"] = pd.Categorical(
    plot_df["Lat_muscle_tertile"],
    categories=["Low", "Mid", "High"],
    ordered=True
)

# osteoporosis 변수 준비
if COL_OP is not None:
    plot_df["osteoporosis_yes"] = normalize_binary_yes(plot_df[COL_OP]).astype(int)
else:
    plot_df["osteoporosis_yes"] = (plot_df[COL_MIN_T] <= -2.5).astype(int)


# ============================================
# 6) 통계 계산
# ============================================

low_t = plot_df.loc[plot_df["Lat_muscle_tertile"] == "Low", COL_MIN_T].dropna()
mid_t = plot_df.loc[plot_df["Lat_muscle_tertile"] == "Mid", COL_MIN_T].dropna()
high_t = plot_df.loc[plot_df["Lat_muscle_tertile"] == "High", COL_MIN_T].dropna()

# 전체 비교는 계산만 하고 그래프엔 표시하지 않음
kw_stat, kw_p = stats.kruskal(low_t, mid_t, high_t)

# 쌍별 비교
pair_tests = {
    ("Low", "Mid"): stats.mannwhitneyu(low_t, mid_t, alternative="two-sided").pvalue,
    ("Mid", "High"): stats.mannwhitneyu(mid_t, high_t, alternative="two-sided").pvalue,
    ("Low", "High"): stats.mannwhitneyu(low_t, high_t, alternative="two-sided").pvalue,
}
pair_tests_bonf = {k: min(v * 3, 1.0) for k, v in pair_tests.items()}

# prevalence
prev_table = (
    plot_df.groupby("Lat_muscle_tertile", observed=False)["osteoporosis_yes"]
    .agg(["sum", "count"])
    .reset_index()
)
prev_table["prevalence_pct"] = prev_table["sum"] / prev_table["count"] * 100

contingency = np.array([
    [
        prev_table.loc[prev_table["Lat_muscle_tertile"] == "Low", "sum"].values[0],
        prev_table.loc[prev_table["Lat_muscle_tertile"] == "Low", "count"].values[0]
        - prev_table.loc[prev_table["Lat_muscle_tertile"] == "Low", "sum"].values[0],
    ],
    [
        prev_table.loc[prev_table["Lat_muscle_tertile"] == "Mid", "sum"].values[0],
        prev_table.loc[prev_table["Lat_muscle_tertile"] == "Mid", "count"].values[0]
        - prev_table.loc[prev_table["Lat_muscle_tertile"] == "Mid", "sum"].values[0],
    ],
    [
        prev_table.loc[prev_table["Lat_muscle_tertile"] == "High", "sum"].values[0],
        prev_table.loc[prev_table["Lat_muscle_tertile"] == "High", "count"].values[0]
        - prev_table.loc[prev_table["Lat_muscle_tertile"] == "High", "sum"].values[0],
    ],
])

chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)


# ============================================
# 7) 요약표 저장
# ============================================

summary_rows = []
for grp in ["Low", "Mid", "High"]:
    sub = plot_df.loc[plot_df["Lat_muscle_tertile"] == grp].copy()
    summary_rows.append({
        "Group": grp,
        "N": len(sub),
        "Lateral muscle mean": sub[COL_LAT_MUSCLE].mean(),
        "Lateral muscle SD": sub[COL_LAT_MUSCLE].std(ddof=1),
        "Minimum T-score mean": sub[COL_MIN_T].mean(),
        "Minimum T-score SD": sub[COL_MIN_T].std(ddof=1),
        "Osteoporosis n": sub["osteoporosis_yes"].sum(),
        "Osteoporosis prevalence (%)": sub["osteoporosis_yes"].mean() * 100,
    })

summary_df = pd.DataFrame(summary_rows)

pairwise_df = pd.DataFrame([
    {"Comparison": "Low vs Mid", "raw p": pair_tests[("Low", "Mid")], "Bonferroni p": pair_tests_bonf[("Low", "Mid")], "Stars": p_to_stars(pair_tests_bonf[("Low", "Mid")])},
    {"Comparison": "Mid vs High", "raw p": pair_tests[("Mid", "High")], "Bonferroni p": pair_tests_bonf[("Mid", "High")], "Stars": p_to_stars(pair_tests_bonf[("Mid", "High")])},
    {"Comparison": "Low vs High", "raw p": pair_tests[("Low", "High")], "Bonferroni p": pair_tests_bonf[("Low", "High")], "Stars": p_to_stars(pair_tests_bonf[("Low", "High")])},
])

overall_df = pd.DataFrame([
    {"Test": "Kruskal-Wallis for Minimum T-score across tertiles", "Statistic": kw_stat, "p-value": kw_p},
    {"Test": "Chi-square for osteoporosis prevalence across tertiles", "Statistic": chi2, "p-value": chi2_p},
])

csv_path = OUT_DIR / "figure5_summary.csv"
xlsx_path = OUT_DIR / "figure5_summary.xlsx"

summary_df.to_csv(csv_path, index=False, encoding=used_encoding)

with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="tertile_summary", index=False)
    prev_table.to_excel(writer, sheet_name="prevalence_table", index=False)
    pairwise_df.to_excel(writer, sheet_name="pairwise_tests", index=False)
    overall_df.to_excel(writer, sheet_name="overall_tests", index=False)

print(f"[OK] 요약 CSV 저장: {csv_path}")
print(f"[OK] 요약 XLSX 저장: {xlsx_path}")


# ============================================
# 8) Figure 5A + 5B
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))

# ---------- Figure 5A ----------
ax = axes[0]

groups = ["Low", "Mid", "High"]
colors = [COLOR_LOW, COLOR_MID, COLOR_HIGH]
data_list = [low_t.values, mid_t.values, high_t.values]

bp = ax.boxplot(
    data_list,
    positions=[1, 2, 3],
    widths=0.52,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.4),
    whiskerprops=dict(color=BOX_EDGE, linewidth=1.0),
    capprops=dict(color=BOX_EDGE, linewidth=1.0),
    boxprops=dict(color=BOX_EDGE, linewidth=1.0),
)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.82)

rng = np.random.default_rng(42)
for i, grp in enumerate(groups, start=1):
    vals = plot_df.loc[plot_df["Lat_muscle_tertile"] == grp, COL_MIN_T].dropna().values
    x_jitter = rng.normal(i, 0.05, size=len(vals))
    ax.scatter(
        x_jitter, vals,
        s=11, alpha=0.28, color=POINT_COLOR, edgecolors="none"
    )

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Low", "Mid", "High"], fontsize=11)
ax.set_ylabel("Minimum T-score", fontsize=11)
ax.set_title("Figure 5A. Minimum T-score by lateral muscle tertile", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.22)

y_max = plot_df[COL_MIN_T].max()
y_min = plot_df[COL_MIN_T].min()
yr = y_max - y_min
base_y = y_max + 0.10 * yr
step = 0.08 * yr

add_sig_bar(ax, 1, 2, base_y, 0.018 * yr, p_to_stars(pair_tests_bonf[("Low", "Mid")]))
add_sig_bar(ax, 2, 3, base_y + step, 0.018 * yr, p_to_stars(pair_tests_bonf[("Mid", "High")]))
add_sig_bar(ax, 1, 3, base_y + 2 * step, 0.018 * yr, p_to_stars(pair_tests_bonf[("Low", "High")]))

ax.set_ylim(y_min - 0.05 * yr, base_y + 3.0 * step)

# ---------- Figure 5B ----------
ax = axes[1]

x = np.arange(3)
y = prev_table["prevalence_pct"].values
n_yes = prev_table["sum"].values
n_total = prev_table["count"].values

bars = ax.bar(
    x,
    y,
    color=colors,
    width=0.58,
    edgecolor=BOX_EDGE,
    linewidth=1.0,
    alpha=0.90
)

ax.set_xticks(x)
ax.set_xticklabels(["Low", "Mid", "High"], fontsize=11)
ax.set_ylabel("Osteoporosis prevalence (%)", fontsize=11)
ax.set_title("Figure 5B. Osteoporosis prevalence by lateral muscle tertile", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.22)

for i, b in enumerate(bars):
    h = b.get_height()
    ax.text(
        b.get_x() + b.get_width() / 2,
        h + 1.1,
        f"{h:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax.set_ylim(0, max(y) + 15)

plt.tight_layout()

fig.text(
    0.5, -0.02,
    "* p < 0.05, ** p < 0.01, *** p < 0.001, ns not significant. "
    "Pairwise comparisons in Figure 5A were corrected using Bonferroni adjustment.",
    ha="center",
    fontsize=9
)

png_path = OUT_DIR / "Figure5_lateral_muscle_tertile_analysis.png"
plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"[OK] Figure 5 저장 완료: {png_path}")
print("[DONE] Figure 5 생성 완료")