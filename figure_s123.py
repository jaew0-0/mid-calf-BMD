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
OUT_DIR = BASE_DIR / "supplementary_figure5_style"
OUT_DIR.mkdir(exist_ok=True)

DPI = 300

# Figure 5와 동일 톤
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
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold")


# ============================================
# 4) 컬럼 찾기
# ============================================

COL_MIN_T = find_column(df, ["min t score", "Minimum T-score", "minimum t score"], required=True)
COL_OP = find_column(df, ["osteoporosis", "Osteoporosis"], required=False)

COL_LAT_TOTAL = find_column(df, ["Lat total", "lateral total", "Lateral total"], required=True)
COL_AP_MUSCLE = find_column(df, ["AP muscle", "ap muscle"], required=True)
COL_AP_TOTAL = find_column(df, ["AP total", "ap total"], required=True)

for col in [COL_MIN_T, COL_LAT_TOTAL, COL_AP_MUSCLE, COL_AP_TOTAL]:
    df[col] = to_numeric(df[col])


# ============================================
# 5) 분석 대상 변수 정의
# ============================================

analysis_specs = [
    {
        "fig_id": "S1",
        "var_col": COL_LAT_TOTAL,
        "var_name": "lateral total thickness",
        "short_name": "lat_total",
        "panelA_title": "Supplementary Figure S1A. Minimum T-score by lateral total tertile",
        "panelB_title": "Supplementary Figure S1B. Osteoporosis prevalence by lateral total tertile",
    },
    {
        "fig_id": "S2",
        "var_col": COL_AP_MUSCLE,
        "var_name": "AP muscle thickness",
        "short_name": "ap_muscle",
        "panelA_title": "Supplementary Figure S2A. Minimum T-score by AP muscle tertile",
        "panelB_title": "Supplementary Figure S2B. Osteoporosis prevalence by AP muscle tertile",
    },
    {
        "fig_id": "S3",
        "var_col": COL_AP_TOTAL,
        "var_name": "AP total thickness",
        "short_name": "ap_total",
        "panelA_title": "Supplementary Figure S3A. Minimum T-score by AP total tertile",
        "panelB_title": "Supplementary Figure S3B. Osteoporosis prevalence by AP total tertile",
    },
]


# ============================================
# 6) 개별 figure 생성 함수
# ============================================

def make_tertile_figure(df, target_col, min_t_col, op_col, spec):
    work_cols = [target_col, min_t_col] + ([op_col] if op_col is not None else [])
    plot_df = df[work_cols].copy()
    plot_df = plot_df.dropna(subset=[target_col, min_t_col]).copy()

    if len(plot_df) == 0:
        raise ValueError(f"{spec['var_name']}: 분석 가능한 데이터가 없습니다.")

    # tertile
    plot_df["tertile"] = pd.qcut(
        plot_df[target_col],
        q=3,
        labels=["Low", "Mid", "High"],
        duplicates="drop"
    )
    plot_df = plot_df.dropna(subset=["tertile"]).copy()

    if plot_df["tertile"].nunique() != 3:
        raise ValueError(f"{spec['var_name']}: tertile 생성 실패")

    plot_df["tertile"] = pd.Categorical(
        plot_df["tertile"],
        categories=["Low", "Mid", "High"],
        ordered=True
    )

    # osteoporosis
    if op_col is not None:
        plot_df["osteoporosis_yes"] = normalize_binary_yes(plot_df[op_col]).astype(int)
    else:
        plot_df["osteoporosis_yes"] = (plot_df[min_t_col] <= -2.5).astype(int)

    # 그룹 데이터
    low_t = plot_df.loc[plot_df["tertile"] == "Low", min_t_col].dropna()
    mid_t = plot_df.loc[plot_df["tertile"] == "Mid", min_t_col].dropna()
    high_t = plot_df.loc[plot_df["tertile"] == "High", min_t_col].dropna()

    # pairwise
    pair_tests = {
        ("Low", "Mid"): stats.mannwhitneyu(low_t, mid_t, alternative="two-sided").pvalue,
        ("Mid", "High"): stats.mannwhitneyu(mid_t, high_t, alternative="two-sided").pvalue,
        ("Low", "High"): stats.mannwhitneyu(low_t, high_t, alternative="two-sided").pvalue,
    }
    pair_tests_bonf = {k: min(v * 3, 1.0) for k, v in pair_tests.items()}

    # prevalence
    prev_table = (
        plot_df.groupby("tertile", observed=False)["osteoporosis_yes"]
        .agg(["sum", "count"])
        .reset_index()
    )
    prev_table["prevalence_pct"] = prev_table["sum"] / prev_table["count"] * 100

    # summary tables
    summary_rows = []
    for grp in ["Low", "Mid", "High"]:
        sub = plot_df.loc[plot_df["tertile"] == grp].copy()
        summary_rows.append({
            "Group": grp,
            "N": len(sub),
            f"{spec['var_name']} mean": sub[target_col].mean(),
            f"{spec['var_name']} SD": sub[target_col].std(ddof=1),
            "Minimum T-score mean": sub[min_t_col].mean(),
            "Minimum T-score SD": sub[min_t_col].std(ddof=1),
            "Osteoporosis n": sub["osteoporosis_yes"].sum(),
            "Osteoporosis prevalence (%)": sub["osteoporosis_yes"].mean() * 100,
        })

    summary_df = pd.DataFrame(summary_rows)
    pairwise_df = pd.DataFrame([
        {"Comparison": "Low vs Mid", "raw p": pair_tests[("Low", "Mid")], "Bonferroni p": pair_tests_bonf[("Low", "Mid")], "Stars": p_to_stars(pair_tests_bonf[("Low", "Mid")])},
        {"Comparison": "Mid vs High", "raw p": pair_tests[("Mid", "High")], "Bonferroni p": pair_tests_bonf[("Mid", "High")], "Stars": p_to_stars(pair_tests_bonf[("Mid", "High")])},
        {"Comparison": "Low vs High", "raw p": pair_tests[("Low", "High")], "Bonferroni p": pair_tests_bonf[("Low", "High")], "Stars": p_to_stars(pair_tests_bonf[("Low", "High")])},
    ])

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    colors = [COLOR_LOW, COLOR_MID, COLOR_HIGH]

    # A: boxplot
    ax = axes[0]
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
    for i, grp in enumerate(["Low", "Mid", "High"], start=1):
        vals = plot_df.loc[plot_df["tertile"] == grp, min_t_col].dropna().values
        x_jitter = rng.normal(i, 0.05, size=len(vals))
        ax.scatter(
            x_jitter, vals,
            s=11, alpha=0.28, color=POINT_COLOR, edgecolors="none"
        )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Low", "Mid", "High"], fontsize=11)
    ax.set_ylabel("Minimum T-score", fontsize=11)
    ax.set_title(spec["panelA_title"], fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.22)

    y_max = plot_df[min_t_col].max()
    y_min = plot_df[min_t_col].min()
    yr = y_max - y_min
    base_y = y_max + 0.10 * yr
    step = 0.08 * yr

    add_sig_bar(ax, 1, 2, base_y, 0.018 * yr, p_to_stars(pair_tests_bonf[("Low", "Mid")]))
    add_sig_bar(ax, 2, 3, base_y + step, 0.018 * yr, p_to_stars(pair_tests_bonf[("Mid", "High")]))
    add_sig_bar(ax, 1, 3, base_y + 2 * step, 0.018 * yr, p_to_stars(pair_tests_bonf[("Low", "High")]))

    ax.set_ylim(y_min - 0.05 * yr, base_y + 3.0 * step)

    # B: prevalence barplot
    ax = axes[1]
    x = np.arange(3)
    y = prev_table["prevalence_pct"].values

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
    ax.set_title(spec["panelB_title"], fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.22)

    for b in bars:
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
        "Pairwise comparisons in panel A were corrected using Bonferroni adjustment.",
        ha="center",
        fontsize=9
    )

    png_path = OUT_DIR / f"{spec['fig_id']}_{spec['short_name']}_tertile_analysis.png"
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    # excel
    xlsx_path = OUT_DIR / f"{spec['fig_id']}_{spec['short_name']}_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="tertile_summary", index=False)
        prev_table.to_excel(writer, sheet_name="prevalence_table", index=False)
        pairwise_df.to_excel(writer, sheet_name="pairwise_tests", index=False)

    print(f"[OK] 저장 완료: {png_path.name}")
    print(f"[OK] 저장 완료: {xlsx_path.name}")


# ============================================
# 7) 실행
# ============================================

for spec in analysis_specs:
    make_tertile_figure(
        df=df,
        target_col=spec["var_col"],
        min_t_col=COL_MIN_T,
        op_col=COL_OP,
        spec=spec,
    )

print("[DONE] Supplementary figures 생성 완료")