from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

MASTER_PATH = OUTPUT_DIR / "dataset_master_enriched.xlsx"
CHECKPOINT_PATH = OUTPUT_DIR / "web_supplement_checkpoint.csv"
RESIDUAL_PATH = OUTPUT_DIR / "round3_residual_missing_or_unavailable.csv"

CLEAN_CSV_PATH = OUTPUT_DIR / "analysis_clean_master.csv"
CLEAN_XLSX_PATH = OUTPUT_DIR / "analysis_clean_master.xlsx"
MAIN_CSV_PATH = OUTPUT_DIR / "analysis_main_sample.csv"
MAIN_XLSX_PATH = OUTPUT_DIR / "analysis_main_sample.xlsx"
REPORT_PATH = OUTPUT_DIR / "analysis_quality_report.md"
FIELD_DICT_PATH = OUTPUT_DIR / "analysis_clean_field_dictionary.csv"

ID_COL = "数据集ID"
CORE6 = [
    "download_formats",
    "record_count",
    "data_size",
    "field_names",
    "detail_spatial_scope",
    "detail_time_scope",
]
WEB_OVERLAY_COLS = [
    "download_formats",
    "api_need_apply",
    "format_count",
    "record_count",
    "data_size",
    "field_names",
    "field_count",
    "has_time_field",
    "has_geo_field",
    "field_description_count",
    "has_standard_field_description",
    "has_data_sample",
    "sample_field_headers",
    "recommended_dataset_count",
    "recommended_dataset_names",
    "rating_score",
    "comment_count",
    "detail_spatial_scope",
    "detail_time_scope",
    "scrape_status",
    "scraped_at",
    "has_rdf",
    "has_xml",
    "has_csv",
    "has_json",
    "has_xlsx",
    "备注",
    "detail_url",
    "scrape_error",
]


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalized_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>", "nat"}:
        return ""
    return text


def blank_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip()
    return text.eq("") | text.str.lower().isin(["nan", "none", "<na>", "nat"])


def placeholder_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip()
    return blank_mask(series) | text.isin(["-", "--", "—", "/", "无", "暂无", "不详", "未注明"])


def parse_number(value: object) -> float:
    text = normalized_text(value)
    if not text or text in {"-", "--", "—", "/"}:
        return np.nan
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def parse_data_size_bytes(value: object) -> float:
    text = normalized_text(value)
    if not text or text in {"-", "--", "—", "/"}:
        return np.nan
    compact = text.replace(",", "").replace(" ", "").upper()
    match = re.search(r"(\d+(?:\.\d+)?)(TB|GB|MB|KB|B|T|G|M|K|字节)?", compact)
    if not match:
        return np.nan
    number = float(match.group(1))
    unit = match.group(2) or "B"
    unit_map = {
        "B": 1,
        "字节": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }
    return number * unit_map.get(unit, 1)


def split_terms(value: object) -> list[str]:
    text = normalized_text(value)
    if not text:
        return []
    parts = re.split(r"[;；|,\n\r\t]+", text)
    return [p.strip() for p in parts if p.strip()]


TIME_FIELD_RE = re.compile(
    r"(日期|时间|年月|年度|年份|月份|季度|开始时间|结束时间|有效期|"
    r"\bdate\b|\btime\b|\byear\b|\bmonth\b|\bquarter\b)",
    flags=re.IGNORECASE,
)
GEO_FIELD_RE = re.compile(
    r"(地址|经度|纬度|坐标|行政区|区县|所属区|所在区|所在镇|所属镇|"
    r"街道|乡镇|社区|居委|location|longitude|latitude|\blng\b|\blat\b|"
    r"address|district|town|village)",
    flags=re.IGNORECASE,
)
RECOMMENDED_POLLUTION_RE = re.compile(
    r"(DATA SAMPLE|DATA ITEM|数据样例|数据可视化|字段名称|字段类型|字段长度|字段描述)",
    flags=re.IGNORECASE,
)


def has_pattern_in_terms(value: object, pattern: re.Pattern[str]) -> int:
    return int(any(pattern.search(term) for term in split_terms(value)))


def safe_percentile_clip(series: pd.Series, q: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    cutoff = numeric.quantile(q)
    if pd.isna(cutoff):
        return numeric
    return numeric.clip(upper=cutoff)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = pd.read_excel(MASTER_PATH, dtype=object)
    checkpoint = pd.read_csv(CHECKPOINT_PATH, dtype=str, encoding="utf-8-sig")
    residual = pd.read_csv(RESIDUAL_PATH, dtype=str, encoding="utf-8-sig")
    master["_dataset_id_norm"] = master[ID_COL].map(normalize_id)
    checkpoint["_dataset_id_norm"] = checkpoint[ID_COL].map(normalize_id)
    residual["_dataset_id_norm"] = residual[ID_COL].map(normalize_id)
    return master, checkpoint, residual


def overlay_checkpoint(master: pd.DataFrame, checkpoint: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    checkpoint_latest = checkpoint.drop_duplicates("_dataset_id_norm", keep="last").set_index("_dataset_id_norm")
    diagnostics: dict[str, int] = {}
    for col in WEB_OVERLAY_COLS:
        if col not in checkpoint_latest.columns:
            continue
        before = master[col].copy() if col in master.columns else pd.Series([pd.NA] * len(master), index=master.index)
        after = master["_dataset_id_norm"].map(checkpoint_latest[col])
        before_text = before.map(normalized_text)
        after_text = after.map(normalized_text)
        diagnostics[f"{col}_changed_by_checkpoint_overlay"] = int((before_text != after_text).sum())
        master[col] = after
    master["web_overlay_source"] = "checkpoint_after_round3"
    return master, diagnostics


def add_residual_flags(df: pd.DataFrame, residual: pd.DataFrame) -> pd.DataFrame:
    residual_latest = residual.drop_duplicates("_dataset_id_norm", keep="last").set_index("_dataset_id_norm")
    issues = df["_dataset_id_norm"].map(residual_latest.get("round3_residual_issues", pd.Series(dtype=str)))
    row_no = df["_dataset_id_norm"].map(residual_latest.get("__master_row_number", pd.Series(dtype=str)))
    df["round3_residual_issues"] = issues.fillna("")
    df["round3_master_row_number"] = row_no
    df["round3_residual_flag"] = df["round3_residual_issues"].astype(str).str.strip().ne("").astype(int)
    for issue in [
        "suspicious_field_names",
        "field_terms_too_few",
        "missing_download_formats",
        "missing_record_count",
        "missing_data_size",
        "missing_field_names",
        "missing_detail_spatial_scope",
        "missing_detail_time_scope",
        "unavailable",
    ]:
        df[f"residual_{issue}_flag"] = df["round3_residual_issues"].str.contains(issue, regex=False, na=False).astype(int)
    return df


def add_clean_fields(df: pd.DataFrame) -> pd.DataFrame:
    status = df["scrape_status"].map(normalized_text).str.lower()
    df["is_unavailable"] = status.eq("unavailable").astype(int)
    df["is_scrape_success"] = status.eq("success").astype(int)

    missing_core_fields: list[str] = []
    for idx, row in df.iterrows():
        missing = [col for col in CORE6 if col in df.columns and blank_mask(pd.Series([row[col]])).iloc[0]]
        missing_core_fields.append(";".join(missing))
    df["core_missing_fields"] = missing_core_fields
    df["core_missing_count"] = df["core_missing_fields"].map(lambda x: 0 if not x else len(x.split(";")))
    df["is_core_complete"] = ((df["is_scrape_success"] == 1) & (df["core_missing_count"] == 0)).astype(int)
    df["is_main_analysis_sample"] = df["is_core_complete"]
    df["is_modeling_candidate"] = df["is_main_analysis_sample"]

    exclusion_reason = []
    for _, row in df.iterrows():
        reasons = []
        if row["is_unavailable"] == 1:
            reasons.append("unavailable")
        if row["core_missing_count"] > 0:
            reasons.append("core_missing")
        exclusion_reason.append(";".join(reasons))
    df["analysis_exclusion_reason"] = exclusion_reason

    df["detail_time_scope_raw"] = df["detail_time_scope"]
    df["has_meaningful_time_scope"] = (~placeholder_mask(df["detail_time_scope"])).astype(int)

    maintenance = df["maintenance_span_days"].map(parse_number) if "maintenance_span_days" in df.columns else pd.Series(np.nan, index=df.index)
    df["maintenance_span_days_clean"] = maintenance.where(maintenance >= 0)
    date_anomaly = maintenance.lt(0).fillna(False)
    if {"first_publish_date_parsed", "last_update_date_parsed"}.issubset(df.columns):
        first_dt = pd.to_datetime(df["first_publish_date_parsed"], errors="coerce")
        last_dt = pd.to_datetime(df["last_update_date_parsed"], errors="coerce")
        date_anomaly = date_anomaly | (first_dt.notna() & last_dt.notna() & (first_dt > last_dt))
    df["date_order_anomaly"] = date_anomaly.astype(int)

    df["publication_age_days_clean"] = df["publication_age_days"].map(parse_number) if "publication_age_days" in df.columns else np.nan
    df["update_recency_days_clean"] = df["update_recency_days"].map(parse_number) if "update_recency_days" in df.columns else np.nan

    df["record_count_clean"] = df["record_count"].map(parse_number)
    df["record_count_log"] = np.log1p(df["record_count_clean"])
    df["record_count_winsor_p99"] = safe_percentile_clip(df["record_count_clean"], 0.99)
    df["record_count_log_winsor_p99"] = np.log1p(df["record_count_winsor_p99"])

    df["data_size_raw"] = df["data_size"]
    df["data_size_bytes"] = df["data_size"].map(parse_data_size_bytes)
    df["data_size_log"] = np.log1p(df["data_size_bytes"])
    df["data_size_bytes_winsor_p99"] = safe_percentile_clip(df["data_size_bytes"], 0.99)
    df["data_size_log_winsor_p99"] = np.log1p(df["data_size_bytes_winsor_p99"])

    field_count_numeric = df["field_count"].map(parse_number)
    field_count_from_names = df["field_names"].map(lambda value: len(split_terms(value)))
    df["field_count_clean"] = field_count_numeric.where(field_count_numeric.notna(), field_count_from_names)
    df["field_description_count_clean"] = df["field_description_count"].map(parse_number)
    df["low_field_count_flag"] = (
        (df.get("residual_field_terms_too_few_flag", 0).astype(int) == 1)
        | ((df["is_scrape_success"] == 1) & (df["field_count_clean"].fillna(0) <= 2))
    ).astype(int)
    df["suspicious_field_names_flag"] = df.get("residual_suspicious_field_names_flag", 0).astype(int)
    df["has_time_field_strict"] = df["field_names"].map(lambda value: has_pattern_in_terms(value, TIME_FIELD_RE))
    df["has_geo_field_strict"] = df["field_names"].map(lambda value: has_pattern_in_terms(value, GEO_FIELD_RE))

    for source, target in [
        ("浏览量", "view_count_clean"),
        ("下载量", "download_count_clean"),
        ("接口调用量", "api_call_count_clean"),
    ]:
        df[target] = df[source].map(parse_number)
    df["log_view_count"] = np.log1p(df["view_count_clean"])
    df["log_download_count"] = np.log1p(df["download_count_clean"])
    df["log_api_call_count"] = np.log1p(df["api_call_count_clean"])
    df["usage_all_zero_flag"] = (
        df[["view_count_clean", "download_count_clean", "api_call_count_clean"]].fillna(0).sum(axis=1).eq(0)
    ).astype(int)

    df["format_count_clean"] = df["format_count"].map(parse_number)
    for col in ["has_rdf", "has_xml", "has_csv", "has_json", "has_xlsx", "api_need_apply"]:
        if col in df.columns:
            df[f"{col}_clean"] = df[col].map(parse_number).fillna(0).astype(int)

    df["recommended_dataset_count_clean"] = df["recommended_dataset_count"].map(parse_number)
    df["recommended_names_suspicious_flag"] = df["recommended_dataset_names"].map(
        lambda value: int(bool(RECOMMENDED_POLLUTION_RE.search(normalized_text(value))))
    )
    df["recommended_names_disabled_for_graph"] = df["recommended_names_suspicious_flag"]

    rating_numeric = df["rating_score"].map(parse_number)
    comment_numeric = df["comment_count"].map(parse_number)
    df["rating_score_invalid_flag"] = ((rating_numeric.notna()) & ((rating_numeric < 0) | (rating_numeric > 5))).astype(int)
    df["comment_count_invalid_flag"] = ((comment_numeric.notna()) & (comment_numeric > 100000)).astype(int)
    df["rating_comment_disabled_for_model"] = 1
    df["rating_score_clean"] = np.nan
    df["comment_count_clean"] = np.nan

    resource_type = df["数据资源类型"].map(normalized_text)
    open_attr = df["开放属性"].map(normalized_text)
    df["is_data_interface"] = resource_type.eq("数据接口").astype(int)
    df["is_data_product"] = resource_type.eq("数据产品").astype(int)
    df["is_conditional_open"] = open_attr.eq("有条件开放").astype(int)
    df["is_unconditional_open"] = open_attr.eq("无条件开放").astype(int)

    warning_cols = [
        "low_field_count_flag",
        "suspicious_field_names_flag",
        "recommended_names_suspicious_flag",
        "date_order_anomaly",
    ]
    warning_flags = []
    for _, row in df.iterrows():
        flags = [col for col in warning_cols if row.get(col, 0) == 1]
        if row.get("has_meaningful_time_scope", 0) == 0:
            flags.append("no_meaningful_time_scope")
        warning_flags.append(";".join(flags))
    df["quality_warning_flags"] = warning_flags

    return df


def build_field_dictionary() -> pd.DataFrame:
    rows = [
        ("is_unavailable", "详情页或核心网页信息不可用标记"),
        ("is_core_complete", "核心六字段非空且 scrape_status 为 success"),
        ("is_main_analysis_sample", "进入 9540 条主分析样本的标记"),
        ("is_modeling_candidate", "具备进入基础模型训练的候选标记，后续按模型字段再筛选"),
        ("analysis_exclusion_reason", "不进入主分析样本的原因"),
        ("core_missing_fields", "缺失的核心字段列表"),
        ("has_meaningful_time_scope", "detail_time_scope 是否为有效内容时间范围"),
        ("date_order_anomaly", "首次发布日期晚于最近更新日期或维护跨度为负"),
        ("maintenance_span_days_clean", "负值设为缺失后的维护跨度"),
        ("record_count_clean", "解析后的数据量数值"),
        ("record_count_log", "log1p(record_count_clean)"),
        ("data_size_bytes", "统一为 bytes 的数据大小"),
        ("data_size_log", "log1p(data_size_bytes)"),
        ("field_count_clean", "清洗后的字段数量"),
        ("low_field_count_flag", "字段词条过少预警，不直接剔除"),
        ("suspicious_field_names_flag", "字段名疑似污染或异常预警，不直接剔除"),
        ("has_time_field_strict", "严格口径下字段名中是否存在时间字段"),
        ("has_geo_field_strict", "严格口径下字段名中是否存在空间或地理字段"),
        ("view_count_clean", "解析后的浏览量"),
        ("download_count_clean", "解析后的下载量"),
        ("api_call_count_clean", "解析后的接口调用量"),
        ("log_view_count", "log1p(view_count_clean)"),
        ("log_download_count", "log1p(download_count_clean)"),
        ("log_api_call_count", "log1p(api_call_count_clean)"),
        ("recommended_names_suspicious_flag", "推荐数据集名称疑似混入页面样例或字段文本"),
        ("recommended_names_disabled_for_graph", "推荐名称是否不进入推荐图网络"),
        ("rating_comment_disabled_for_model", "评分和评论字段是否禁用于主模型"),
        ("is_data_interface", "数据资源类型是否为数据接口"),
        ("is_data_product", "数据资源类型是否为数据产品"),
        ("is_conditional_open", "开放属性是否为有条件开放"),
        ("is_unconditional_open", "开放属性是否为无条件开放"),
    ]
    return pd.DataFrame(rows, columns=["field_name", "definition"])


def make_report(df: pd.DataFrame, overlay_diagnostics: dict[str, int]) -> str:
    status_counts = df["scrape_status"].map(normalized_text).value_counts(dropna=False).to_dict()
    residual_issue_counts = {}
    for value in df["round3_residual_issues"].dropna().astype(str):
        for issue in [v.strip() for v in re.split(r"[;,|]+", value) if v.strip()]:
            residual_issue_counts[issue] = residual_issue_counts.get(issue, 0) + 1

    core_rows = []
    for col in CORE6:
        missing = blank_mask(df[col])
        success_missing = (df["is_scrape_success"].eq(1) & missing).sum()
        core_rows.append((col, int(missing.sum()), int(success_missing)))

    report = []
    report.append("# analysis_clean_master 清洗报告")
    report.append("")
    report.append("## 1. 数据源")
    report.append("")
    report.append(f"- 原始增强主表：`{MASTER_PATH}`")
    report.append(f"- 第三轮 checkpoint：`{CHECKPOINT_PATH}`")
    report.append(f"- 第三轮残留质量清单：`{RESIDUAL_PATH}`")
    report.append("- 清洗策略：以 `dataset_master_enriched.xlsx` 为底表，按 `数据集ID` 用第三轮 checkpoint 覆盖网页增强字段；原始 master 不修改。")
    report.append("")
    report.append("## 2. 样本口径")
    report.append("")
    report.append(f"- 全量资源画像样本：{len(df)} 条")
    report.append(f"- scrape_status 分布：{status_counts}")
    report.append(f"- 不可用样本：{int(df['is_unavailable'].sum())} 条")
    report.append(f"- success 但核心六字段任一缺失：{int(((df['is_scrape_success'] == 1) & (df['core_missing_count'] > 0)).sum())} 条")
    report.append(f"- 主分析样本：{int(df['is_main_analysis_sample'].sum())} 条")
    report.append(f"- 建模候选样本：{int(df['is_modeling_candidate'].sum())} 条")
    report.append("")
    report.append("## 3. 核心字段缺失")
    report.append("")
    report.append("| 字段 | 全量缺失 | success 中缺失 |")
    report.append("|---|---:|---:|")
    for col, all_missing, success_missing in core_rows:
        report.append(f"| `{col}` | {all_missing} | {success_missing} |")
    report.append("")
    report.append("## 4. 质量风险标记")
    report.append("")
    report.append(f"- low_field_count_flag：{int(df['low_field_count_flag'].sum())} 条")
    report.append(f"- suspicious_field_names_flag：{int(df['suspicious_field_names_flag'].sum())} 条")
    report.append(f"- has_meaningful_time_scope = 0：{int((df['has_meaningful_time_scope'] == 0).sum())} 条")
    report.append(f"- date_order_anomaly：{int(df['date_order_anomaly'].sum())} 条")
    report.append(f"- recommended_names_suspicious_flag：{int(df['recommended_names_suspicious_flag'].sum())} 条")
    report.append(f"- rating_comment_disabled_for_model：{int(df['rating_comment_disabled_for_model'].sum())} 条")
    report.append("")
    report.append("## 5. 网页字段覆盖同步诊断")
    report.append("")
    report.append("以下数字表示清洗时 checkpoint 覆盖 master 后发生变化的行数，用于说明第三轮最终 checkpoint 是网页字段权威来源。")
    report.append("")
    report.append("| 字段 | 覆盖后变化行数 |")
    report.append("|---|---:|")
    for col in WEB_OVERLAY_COLS:
        key = f"{col}_changed_by_checkpoint_overlay"
        if key in overlay_diagnostics:
            report.append(f"| `{col}` | {overlay_diagnostics[key]} |")
    report.append("")
    report.append("## 6. 残留质量清单 issue 计数")
    report.append("")
    report.append("| issue | 条数 |")
    report.append("|---|---:|")
    for issue, count in sorted(residual_issue_counts.items()):
        report.append(f"| `{issue}` | {count} |")
    report.append("")
    report.append("## 7. 输出文件")
    report.append("")
    report.append(f"- `analysis_clean_master.csv` / `analysis_clean_master.xlsx`：保留 9793 条全量行并增加清洗字段")
    report.append(f"- `analysis_main_sample.csv` / `analysis_main_sample.xlsx`：仅保留主分析样本")
    report.append(f"- `analysis_clean_field_dictionary.csv`：新增清洗字段说明")
    report.append("")
    report.append("## 8. 建模口径提醒")
    report.append("")
    report.append("- `field_terms_too_few`、`suspicious_field_names`、`has_meaningful_time_scope = 0` 是风险标记，不默认剔除。")
    report.append("- `rating_score` 与 `comment_count` 原始字段保留，但主模型禁用。")
    report.append("- `recommended_dataset_names` 原始字段保留，但疑似污染，推荐图网络暂不直接使用。")
    report.append("- `scrape_status` 只用于样本筛选与质量说明，不进入 ActualUse、PotentialUse 或 ExpectedUse 主模型。")
    return "\n".join(report) + "\n"


def export_outputs(df: pd.DataFrame, field_dict: pd.DataFrame, report: str) -> None:
    output = df.drop(columns=["_dataset_id_norm"], errors="ignore").copy()
    main = output[output["is_main_analysis_sample"] == 1].copy()
    output.to_csv(CLEAN_CSV_PATH, index=False, encoding="utf-8-sig")
    main.to_csv(MAIN_CSV_PATH, index=False, encoding="utf-8-sig")
    field_dict.to_csv(FIELD_DICT_PATH, index=False, encoding="utf-8-sig")
    REPORT_PATH.write_text(report, encoding="utf-8")

    # These xlsx files are data artifacts, not the calculation workbook for modeling.
    with pd.ExcelWriter(CLEAN_XLSX_PATH, engine="openpyxl") as writer:
        output.to_excel(writer, index=False, sheet_name="analysis_clean_master")
        field_dict.to_excel(writer, index=False, sheet_name="field_dictionary")
    with pd.ExcelWriter(MAIN_XLSX_PATH, engine="openpyxl") as writer:
        main.to_excel(writer, index=False, sheet_name="analysis_main_sample")
        field_dict.to_excel(writer, index=False, sheet_name="field_dictionary")


def main() -> None:
    master, checkpoint, residual = load_inputs()
    master, overlay_diagnostics = overlay_checkpoint(master, checkpoint)
    master = add_residual_flags(master, residual)
    clean = add_clean_fields(master)
    field_dict = build_field_dictionary()
    report = make_report(clean, overlay_diagnostics)
    export_outputs(clean, field_dict, report)

    print("rows", len(clean))
    print("main_analysis_sample", int(clean["is_main_analysis_sample"].sum()))
    print("unavailable", int(clean["is_unavailable"].sum()))
    print("success_core6_missing", int(((clean["is_scrape_success"] == 1) & (clean["core_missing_count"] > 0)).sum()))
    print("outputs")
    for path in [CLEAN_CSV_PATH, CLEAN_XLSX_PATH, MAIN_CSV_PATH, MAIN_XLSX_PATH, REPORT_PATH, FIELD_DICT_PATH]:
        print(path)


if __name__ == "__main__":
    main()
