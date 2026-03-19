import pandas as pd
from pathlib import Path

# -------------------------------------------------
# 1. 데이터 불러오기 (인코딩 자동 처리)
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_1.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"파일이 없습니다: {DATA_PATH}")

last_error = None
for enc in ["utf-8", "cp949", "euc-kr"]:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        print(f"데이터 로드 완료 (encoding={enc})")
        break
    except Exception as e:
        last_error = e
else:
    raise last_error

# -------------------------------------------------
# 2. 컬럼 자동 인식
# -------------------------------------------------

COL_DATA = "데이터"
COL_REMARK = "비고"

if COL_DATA not in df.columns:
    raise KeyError(f"'{COL_DATA}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

if COL_REMARK not in df.columns:
    raise KeyError(f"'{COL_REMARK}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

op_candidates = [c for c in df.columns if c.strip().lower() == "osteoporosis"]
if not op_candidates:
    raise KeyError(f"'osteoporosis' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
COL_OP = op_candidates[0]

# 문자열 정리
df[COL_DATA] = df[COL_DATA].fillna("").astype(str).str.strip()
df[COL_REMARK] = df[COL_REMARK].fillna("").astype(str).str.strip()
df[COL_OP] = df[COL_OP].fillna("").astype(str).str.strip()

# -------------------------------------------------
# 3. 분류 기준 정의
# -------------------------------------------------
# 사용자가 요청한 재분류 반영:
# - "6개월 이내 골밀도 없음", "6개월 이내 골밀도검사 없음" -> 2) Interval > 6 months
# - "수술 후라 근육량이 급감했음", "너무 어림" -> 4) Poor image quality / Measurement difficulty

INTERVAL_PHRASES = [
    "6개월 이내 골밀도 없음",
    "6개월 이내 골밀도검사 없음",
    "6개월 이내 골밀도 검사 없음",
    "body composition이랑 무릎영상 6개월 이상 차이",
    "body composition이랑 무릎 영상 6개월 이상 차이",
    "6개월 이상 차이",
    "1년 넘게 차이",
    "시기 안 맞음",
]

NO_BMD_PHRASES = [
    "골밀도 없음",
    "골밀도검사 없음",
    "골밀도 검사 없음",
    "dexa 없음",
    "bmd 없음",
    "bmd 미촬영",
    "bmd 결측",
    "body composition 없음",
    "tibia 영상 없음, 골밀도 없음",
    "tibia 영상없음, 골밀도 없음",
]

SURGERY_KEYWORDS = [
    "tkr",
    "hto",
    "uka",
    "인공관절",
    "수술",
    "amputation",
    "절단",
]

POOR_IMAGE_PHRASES = [
    "fat과 mid calf 구분 안됨",
    "fat과 mid calf 구분이 잘 안감",
    "영상에서 fat과 mid calf 구분이 잘 안감",
    "mid calf랑 fat 구분 안감",
    "mid calf와 fat 구분 안감",
    "영상에서 fat 안보임",
    "fat 안보임",
    "tibia 명확하지 않음",
    "tibia 명확하지않음",
    "영상 잘림",
    "영상이 잘림",
    "측정 어려움",
    "측정어려움",
    "기구로 측정어려움",
    "기구로 측정 어려움",
    "영상 화질 불량",
    "무릎사진 없음",
    "무릎 사진 없음",
    "tibia 영상 없음",
    "tibia 영상없음",
    "영상이 열리지 않음",
    "수술 후라 근육량이 급감했음",
    "너무 어림",
]

# -------------------------------------------------
# 4. 유틸 함수
# -------------------------------------------------

def normalize_text(text):
    text = str(text).strip().lower()
    text = " ".join(text.split())
    return text

def contains_any_phrase(text, phrase_list):
    t = normalize_text(text)
    return any(normalize_text(p) in t for p in phrase_list)

def contains_any_keyword(text, keyword_list):
    t = normalize_text(text)
    return any(normalize_text(k) in t for k in keyword_list)

# -------------------------------------------------
# 5. 분류 함수
# -------------------------------------------------
# 우선순위:
# 2) Interval -> 3) Surgery -> 4) Poor image -> 1) No BMD -> Unclassified

def classify_reason(text):
    t = normalize_text(text)

    if t == "":
        return "Unclassified"

    if contains_any_phrase(t, INTERVAL_PHRASES):
        return "2) Interval > 6 months"

    if contains_any_phrase(t, SURGERY_KEYWORDS) or contains_any_keyword(t, SURGERY_KEYWORDS):
        return "3) History of leg surgery"

    if contains_any_phrase(t, POOR_IMAGE_PHRASES):
        return "4) Poor image quality / Measurement difficulty"

    if contains_any_phrase(t, NO_BMD_PHRASES):
        return "1) Absence of BMD"

    return "Unclassified"

# -------------------------------------------------
# 6. 데이터 분리
# -------------------------------------------------

df_x = df[df[COL_DATA].str.upper() == "X"].copy()
df_o = df[df[COL_DATA].str.upper() == "O"].copy()

# exclusion reason 분류
df_x["reason"] = df_x[COL_REMARK].apply(classify_reason)

# -------------------------------------------------
# 7. Osteoporosis 카운트
# -------------------------------------------------

def op_group(x):
    x = str(x).strip().lower()

    if x == "yes":
        return "With osteoporosis"
    if x == "no":
        return "Without osteoporosis"
    return "Blank"

df_o["op_group"] = df_o[COL_OP].apply(op_group)

# -------------------------------------------------
# 8. 결과 계산
# -------------------------------------------------

total_n = len(df)
excluded_n = len(df_x)
study_n = len(df_o)

reason_order = [
    "1) Absence of BMD",
    "2) Interval > 6 months",
    "3) History of leg surgery",
    "4) Poor image quality / Measurement difficulty",
    "Unclassified",
]

reason_counts = df_x["reason"].value_counts()
reason_counts = reason_counts.reindex(reason_order, fill_value=0)

op_order = ["With osteoporosis", "Without osteoporosis", "Blank"]
op_counts = df_o["op_group"].value_counts().reindex(op_order, fill_value=0)

# Unclassified 상세
df_unclassified = df_x[df_x["reason"] == "Unclassified"].copy()
unclassified_counts = (
    df_unclassified[COL_REMARK]
    .fillna("")
    .astype(str)
    .str.strip()
    .value_counts()
)

# remark 전체 빈도표
remark_freq_x = (
    df_x[COL_REMARK]
    .fillna("")
    .astype(str)
    .str.strip()
    .value_counts()
    .reset_index()
)
remark_freq_x.columns = [COL_REMARK, "count"]

# reason + remark 빈도표
reason_remark_freq = (
    df_x.groupby(["reason", COL_REMARK], dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values(["reason", "count"], ascending=[True, False])
)

# rule 목록 표
rules_table = []
for p in INTERVAL_PHRASES:
    rules_table.append({"reason": "2) Interval > 6 months", "rule_type": "phrase", "pattern": p})
for p in NO_BMD_PHRASES:
    rules_table.append({"reason": "1) Absence of BMD", "rule_type": "phrase", "pattern": p})
for p in SURGERY_KEYWORDS:
    rules_table.append({"reason": "3) History of leg surgery", "rule_type": "keyword", "pattern": p})
for p in POOR_IMAGE_PHRASES:
    rules_table.append({"reason": "4) Poor image quality / Measurement difficulty", "rule_type": "phrase", "pattern": p})
rules_df = pd.DataFrame(rules_table)

# summary 표
summary_rows = [
    {"metric": "Total patients", "N": total_n},
    {"metric": "Excluded (데이터 = X)", "N": excluded_n},
    {"metric": "Study population (데이터 = O)", "N": study_n},
    {"metric": "With osteoporosis", "N": int(op_counts["With osteoporosis"])},
    {"metric": "Without osteoporosis", "N": int(op_counts["Without osteoporosis"])},
    {"metric": "Osteoporosis blank", "N": int(op_counts["Blank"])},
    {"metric": "1) Absence of BMD", "N": int(reason_counts["1) Absence of BMD"])},
    {"metric": "2) Interval > 6 months", "N": int(reason_counts["2) Interval > 6 months"])},
    {"metric": "3) History of leg surgery", "N": int(reason_counts["3) History of leg surgery"])},
    {"metric": "4) Poor image quality / Measurement difficulty", "N": int(reason_counts["4) Poor image quality / Measurement difficulty"])},
    {"metric": "Unclassified", "N": int(reason_counts["Unclassified"])},
]
summary_df = pd.DataFrame(summary_rows)

# -------------------------------------------------
# 9. 결과 출력
# -------------------------------------------------

print("\n===== FLOWCHART SUMMARY =====")
print(f"Total patients: {total_n}")
print(f"Excluded (데이터 = X): {excluded_n}")
print(f"Study population (데이터 = O): {study_n}")

print("\n--- Exclusion reason ---")
print(reason_counts.to_string())

print("\n--- Osteoporosis ---")
print(op_counts.to_string())

print("\n--- Unclassified cases ---")
print(f"Unclassified count: {len(df_unclassified)}")

if len(df_unclassified) > 0:
    print("\nUnclassified remark frequency:")
    print(unclassified_counts.to_string())

    print("\nUnclassified unique remarks:")
    for i, remark in enumerate(df_unclassified[COL_REMARK].drop_duplicates().tolist(), start=1):
        print(f"{i}. {remark}")
else:
    print("No unclassified rows.")

# -------------------------------------------------
# 10. 엑셀 저장
# -------------------------------------------------

output = BASE_DIR / "flowchart_result.xlsx"

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)

    reason_counts.rename_axis("reason").reset_index(name="N").to_excel(
        writer, sheet_name="exclusion_reason", index=False
    )

    op_counts.rename_axis("osteoporosis_group").reset_index(name="N").to_excel(
        writer, sheet_name="osteoporosis", index=False
    )

    df_x.to_excel(writer, sheet_name="excluded_detail", index=False)
    df_o.to_excel(writer, sheet_name="study_population", index=False)

    df_unclassified.to_excel(writer, sheet_name="unclassified_rows", index=False)

    if len(df_unclassified) > 0:
        unclassified_counts.rename_axis(COL_REMARK).reset_index(name="count").to_excel(
            writer, sheet_name="unclassified_frequency", index=False
        )
    else:
        pd.DataFrame(columns=[COL_REMARK, "count"]).to_excel(
            writer, sheet_name="unclassified_frequency", index=False
        )

    remark_freq_x.to_excel(writer, sheet_name="X_remark_frequency", index=False)
    reason_remark_freq.to_excel(writer, sheet_name="reason_remark_frequency", index=False)
    rules_df.to_excel(writer, sheet_name="rules_used", index=False)

print(f"\n엑셀 저장 완료: {output}")