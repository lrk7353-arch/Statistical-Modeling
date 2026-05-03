from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

MAIN_SAMPLE_PATH = OUTPUT_DIR / "analysis_main_sample.csv"
FEATURE_CSV_PATH = OUTPUT_DIR / "analysis_v11_features.csv"
FEATURE_XLSX_PATH = OUTPUT_DIR / "analysis_v11_features.xlsx"
RULE_CANDIDATE_CSV_PATH = OUTPUT_DIR / "rule_dormant_candidates_v11.csv"
CRITIC_WEIGHTS_PATH = OUTPUT_DIR / "potentialuse_critic_weights_v11.csv"
PCA_DIAGNOSTICS_PATH = OUTPUT_DIR / "actualuse_pca_diagnostics_v11.csv"
FEATURE_REPORT_PATH = OUTPUT_DIR / "analysis_v11_feature_report.md"
FEATURE_DICTIONARY_PATH = OUTPUT_DIR / "analysis_v11_feature_dictionary.csv"

ID_COL = "数据集ID"
DIMENSION_COLS = [
    "topic_public_value_score",
    "semantic_clarity_score",
    "data_richness_score",
    "machine_readability_score",
    "timeliness_score",
    "spatiotemporal_capability_score",
    "combinability_score",
]


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "<na>", "nat"}:
        return ""
    return s


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def percentile_01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    n = int(values.notna().sum())
    if n <= 1:
        return pd.Series(np.nan, index=series.index)
    ranks = values.rank(method="average", na_option="keep")
    return (ranks - 1) / (n - 1)


def normalize_01(series: pd.Series, q_low: float = 0.0, q_high: float = 0.99, fill: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() == 0:
        return pd.Series(fill, index=series.index, dtype=float)
    low = values.quantile(q_low)
    high = values.quantile(q_high)
    if pd.isna(low) or pd.isna(high) or math.isclose(float(high), float(low)):
        out = pd.Series(0.5, index=series.index, dtype=float)
    else:
        out = ((values.clip(lower=low, upper=high) - low) / (high - low)).astype(float)
    return out.fillna(fill).clip(0, 1)


def log_norm(series: pd.Series, q_high: float = 0.99, fill: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = np.log1p(values.clip(lower=0))
    return normalize_01(values, q_low=0.0, q_high=q_high, fill=fill)


def bool_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1).astype(float)


def clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1)


def pca_one_component(df: pd.DataFrame, mask: pd.Series, input_cols: list[str], label: str) -> tuple[pd.Series, dict[str, object]]:
    output = pd.Series(np.nan, index=df.index, dtype=float)
    sub = df.loc[mask, input_cols].astype(float)
    if len(sub) < 3:
        return output, {
            "resource_type": label,
            "n": len(sub),
            "input_cols": ";".join(input_cols),
            "explained_variance_ratio": np.nan,
            "direction_corr_after": np.nan,
            "loadings": "",
        }

    x = StandardScaler().fit_transform(sub.to_numpy())
    pca = PCA(n_components=1)
    score = pca.fit_transform(x).ravel()
    reference = sub.mean(axis=1).to_numpy()
    corr = np.corrcoef(score, reference)[0, 1]
    sign = -1 if pd.notna(corr) and corr < 0 else 1
    score = score * sign
    loadings = pca.components_[0] * sign
    corr_after = np.corrcoef(score, reference)[0, 1]
    output.loc[mask] = score

    diag = {
        "resource_type": label,
        "n": len(sub),
        "input_cols": ";".join(input_cols),
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "direction_corr_after": float(corr_after),
        "loadings": ";".join(f"{col}={loading:.6f}" for col, loading in zip(input_cols, loadings)),
    }
    return output, diag


def entropy_weights(matrix: pd.DataFrame) -> pd.Series:
    x = matrix.astype(float).clip(lower=0)
    # Avoid all-zero columns by adding a tiny constant only for probability calculation.
    x = x + 1e-12
    col_sum = x.sum(axis=0)
    p = x.div(col_sum, axis=1)
    n = len(x)
    entropy = -(p * np.log(p)).sum(axis=0) / np.log(n)
    divergence = 1 - entropy
    if math.isclose(float(divergence.sum()), 0.0):
        return pd.Series(1 / len(matrix.columns), index=matrix.columns)
    return divergence / divergence.sum()


def critic_weights(matrix: pd.DataFrame) -> pd.DataFrame:
    x = matrix.astype(float).clip(0, 1)
    std = x.std(axis=0, ddof=0)
    corr = x.corr(method="pearson").fillna(0)
    conflict = (1 - corr).sum(axis=1)
    information = std * conflict
    if math.isclose(float(information.sum()), 0.0):
        weights = pd.Series(1 / len(x.columns), index=x.columns)
    else:
        weights = information / information.sum()
    return pd.DataFrame(
        {
            "dimension": x.columns,
            "std": std.reindex(x.columns).to_numpy(),
            "conflict": conflict.reindex(x.columns).to_numpy(),
            "information": information.reindex(x.columns).to_numpy(),
            "critic_weight": weights.reindex(x.columns).to_numpy(),
        }
    )


def domain_public_value(series: pd.Series) -> pd.Series:
    domain_score = {
        "民生服务": 0.95,
        "卫生健康": 0.95,
        "公共安全": 0.90,
        "社保就业": 0.90,
        "城市建设": 0.85,
        "道路交通": 0.85,
        "资源环境": 0.85,
        "绿色低碳": 0.85,
        "经济建设": 0.80,
        "教育科技": 0.80,
        "农业农村": 0.75,
        "社会发展": 0.75,
        "信用服务": 0.70,
        "工业制造": 0.65,
        "气象服务": 0.65,
        "文化休闲": 0.60,
        "机构团体": 0.55,
    }
    return series.map(lambda value: domain_score.get(text(value), 0.60)).astype(float)


def update_frequency_score(series: pd.Series) -> pd.Series:
    score = {
        "即时": 1.00,
        "每天": 0.95,
        "每周": 0.85,
        "每月": 0.75,
        "每季度": 0.65,
        "每半年": 0.55,
        "每年": 0.45,
        "不定期更新": 0.35,
        "静态数据": 0.25,
    }
    return series.map(lambda value: score.get(text(value), 0.40)).astype(float)


def spatial_level_score(series: pd.Series) -> pd.Series:
    score = {
        "跨区域/国家级": 1.00,
        "市级": 0.85,
        "区级": 0.75,
        "其他/不确定": 0.30,
    }
    return series.map(lambda value: score.get(text(value), 0.50)).astype(float)


def build_actual_use(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["view_score"] = percentile_01(to_num(df["log_view_count"]))
    df["download_score"] = percentile_01(to_num(df["log_download_count"]))
    df["api_call_score"] = percentile_01(to_num(df["log_api_call_count"]))

    product_mask = df["数据资源类型"].map(text).eq("数据产品")
    interface_mask = df["数据资源类型"].map(text).eq("数据接口")

    product_score, product_diag = pca_one_component(df, product_mask, ["view_score", "download_score"], "数据产品")
    interface_score, interface_diag = pca_one_component(df, interface_mask, ["view_score", "api_call_score"], "数据接口")

    df["ActualUse_product_pca_raw"] = product_score
    df["ActualUse_interface_pca_raw"] = interface_score
    df["ActualUse_pca_raw"] = product_score.combine_first(interface_score)
    df["ActualUse_type_percentile"] = np.nan
    for mask in [product_mask, interface_mask]:
        df.loc[mask, "ActualUse_type_percentile"] = percentile_01(df.loc[mask, "ActualUse_pca_raw"])
    df["ActualUse_global_percentile"] = percentile_01(df["ActualUse_pca_raw"])
    df["ActualUse_main"] = df["ActualUse_type_percentile"]

    df["ActualUse_equal_score"] = np.where(
        product_mask,
        df[["view_score", "download_score"]].mean(axis=1),
        df[["view_score", "api_call_score"]].mean(axis=1),
    )
    df["ActualUse_equal_type_percentile"] = np.nan
    for mask in [product_mask, interface_mask]:
        df.loc[mask, "ActualUse_equal_type_percentile"] = percentile_01(df.loc[mask, "ActualUse_equal_score"])

    df["ActualUse_depth_score"] = np.where(
        product_mask,
        0.4 * df["view_score"] + 0.6 * df["download_score"],
        0.3 * df["view_score"] + 0.7 * df["api_call_score"],
    )
    df["ActualUse_depth_type_percentile"] = np.nan
    for mask in [product_mask, interface_mask]:
        df.loc[mask, "ActualUse_depth_type_percentile"] = percentile_01(df.loc[mask, "ActualUse_depth_score"])

    return df, pd.DataFrame([product_diag, interface_diag])


def build_potential_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    scene_score = normalize_01(df["scene_keyword_count"], q_high=0.99, fill=0)
    df["topic_public_value_score"] = (
        0.75 * domain_public_value(df["数据领域"]) + 0.25 * scene_score
    ).clip(0, 1)

    title_score = normalize_01(df["title_len"], q_high=0.95, fill=0)
    description_score = normalize_01(df["description_len"], q_high=0.95, fill=0)
    field_hint = bool_num(df["description_has_field_hint"])
    df["semantic_clarity_score"] = (
        0.25 * title_score
        + 0.35 * description_score
        + 0.25 * field_hint
        + 0.15 * scene_score
    ).clip(0, 1)

    field_count_score = log_norm(df["field_count_clean"], q_high=0.99, fill=0)
    field_desc_score = log_norm(df["field_description_count_clean"], q_high=0.99, fill=0)
    record_score = normalize_01(df["record_count_log_winsor_p99"], q_high=0.99, fill=0)
    size_score = normalize_01(df["data_size_log_winsor_p99"], q_high=0.99, fill=0)
    df["data_richness_score"] = (
        0.35 * field_count_score
        + 0.25 * field_desc_score
        + 0.25 * record_score
        + 0.15 * size_score
    ).clip(0, 1)

    format_score = (to_num(df["format_count_clean"]).fillna(0).clip(0, 5) / 5).astype(float)
    df["machine_readability_score"] = (
        0.35 * format_score
        + 0.25 * bool_num(df["has_csv_clean"])
        + 0.25 * bool_num(df["has_json_clean"])
        + 0.10 * bool_num(df["has_xlsx_clean"])
        + 0.03 * bool_num(df["has_xml_clean"])
        + 0.02 * bool_num(df["has_rdf_clean"])
    ).clip(0, 1)

    freq_score = update_frequency_score(df["更新频率"])
    recency_days = to_num(df["update_recency_days_clean"])
    recency_cap = recency_days.quantile(0.95)
    if pd.isna(recency_cap) or recency_cap <= 0:
        recency_score = pd.Series(0.5, index=df.index)
    else:
        recency_score = (1 - recency_days.clip(lower=0, upper=recency_cap) / recency_cap).fillna(0.5)
    df["timeliness_score"] = (0.65 * freq_score + 0.35 * recency_score).clip(0, 1)
    df.loc[to_num(df["date_order_anomaly"]).fillna(0).eq(1), "timeliness_score"] *= 0.90

    spatial_meaningful = (~df["detail_spatial_scope"].astype(str).str.strip().isin(["", "-", "--", "—", "无", "暂无"])).astype(float)
    df["spatiotemporal_capability_score"] = (
        0.30 * bool_num(df["has_meaningful_time_scope"])
        + 0.20 * spatial_meaningful
        + 0.20 * bool_num(df["has_time_field_strict"])
        + 0.20 * bool_num(df["has_geo_field_strict"])
        + 0.10 * spatial_level_score(df["spatial_admin_level"])
    ).clip(0, 1)

    rec_count = to_num(df["recommended_dataset_count_clean"]).fillna(0).clip(0, 10)
    recommended_score = np.log1p(rec_count) / np.log1p(10)
    recommended_score = recommended_score * np.where(bool_num(df["recommended_names_suspicious_flag"]).eq(1), 0.35, 1.0)
    df["combinability_score"] = (
        0.25 * format_score
        + 0.20 * bool_num(df["has_standard_field_description"])
        + 0.20 * field_desc_score
        + 0.15 * recommended_score
        + 0.10 * bool_num(df["has_time_field_strict"])
        + 0.10 * bool_num(df["has_geo_field_strict"])
    ).clip(0, 1)

    return df


def build_potential_use(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dimension_matrix = df[DIMENSION_COLS].astype(float).clip(0, 1)
    weights_df = critic_weights(dimension_matrix)
    critic_weight = weights_df.set_index("dimension")["critic_weight"]
    df["PotentialUse_CRITIC"] = dimension_matrix.mul(critic_weight, axis=1).sum(axis=1)
    df["PotentialUse_CRITIC_percentile"] = percentile_01(df["PotentialUse_CRITIC"])
    df["PotentialUse_equal"] = dimension_matrix.mean(axis=1)
    df["PotentialUse_equal_percentile"] = percentile_01(df["PotentialUse_equal"])
    ent_weight = entropy_weights(dimension_matrix)
    df["PotentialUse_entropy"] = dimension_matrix.mul(ent_weight, axis=1).sum(axis=1)
    df["PotentialUse_entropy_percentile"] = percentile_01(df["PotentialUse_entropy"])
    weights_df["entropy_weight"] = weights_df["dimension"].map(ent_weight)
    return df, weights_df


def build_rule_dormancy(df: pd.DataFrame) -> pd.DataFrame:
    df["DormantScore_rule"] = df["PotentialUse_CRITIC_percentile"] - df["ActualUse_type_percentile"]
    df["DormantScore_rule_rank"] = df["DormantScore_rule"].rank(ascending=False, method="first").astype(int)
    df["rule_dormant_candidate"] = (
        (df["PotentialUse_CRITIC_percentile"] >= 0.70)
        & (df["ActualUse_type_percentile"] <= 0.50)
        & (df["DormantScore_rule"] >= 0.30)
    ).astype(int)

    conditions = [
        (df["PotentialUse_CRITIC_percentile"] >= 0.50) & (df["ActualUse_type_percentile"] >= 0.50),
        (df["PotentialUse_CRITIC_percentile"] >= 0.50) & (df["ActualUse_type_percentile"] < 0.50),
        (df["PotentialUse_CRITIC_percentile"] < 0.50) & (df["ActualUse_type_percentile"] < 0.50),
        (df["PotentialUse_CRITIC_percentile"] < 0.50) & (df["ActualUse_type_percentile"] >= 0.50),
    ]
    labels = ["高潜高用", "高潜低用", "低潜低用", "低潜高用"]
    df["dormant_type_rule"] = np.select(conditions, labels, default="未分类")
    return df


def feature_dictionary() -> pd.DataFrame:
    rows = [
        ("view_score", "log 浏览量的 0-1 百分位分数"),
        ("download_score", "log 下载量的 0-1 百分位分数"),
        ("api_call_score", "log 接口调用量的 0-1 百分位分数"),
        ("ActualUse_type_percentile", "类型适配 PCA 后，在数据产品/数据接口内部转为百分位的主 ActualUse 口径"),
        ("ActualUse_global_percentile", "类型适配 PCA 原始分数合并后的全平台百分位对照口径"),
        ("ActualUse_equal_type_percentile", "等权稳健性 ActualUse 口径"),
        ("ActualUse_depth_type_percentile", "使用深度赋权稳健性 ActualUse 口径"),
        ("topic_public_value_score", "主题公共价值维度分数"),
        ("semantic_clarity_score", "语义清晰度维度分数"),
        ("data_richness_score", "数据丰富度维度分数"),
        ("machine_readability_score", "机器可读性维度分数"),
        ("timeliness_score", "时效性维度分数"),
        ("spatiotemporal_capability_score", "时空能力维度分数"),
        ("combinability_score", "可组合性维度分数"),
        ("PotentialUse_CRITIC", "七维潜力分数经 CRITIC 客观赋权后的潜在价值"),
        ("PotentialUse_CRITIC_percentile", "PotentialUse_CRITIC 的 0-1 百分位"),
        ("PotentialUse_equal", "七维潜力分数等权平均"),
        ("PotentialUse_entropy", "七维潜力分数熵权法综合得分"),
        ("DormantScore_rule", "规则沉睡度：PotentialUse_CRITIC_percentile - ActualUse_type_percentile"),
        ("rule_dormant_candidate", "规则口径沉睡资产候选，阈值为潜力>=0.70、实际<=0.50、差值>=0.30"),
        ("dormant_type_rule", "基于 PotentialUse 和 ActualUse 中位数的四象限分类"),
    ]
    return pd.DataFrame(rows, columns=["field_name", "definition"])


def make_report(df: pd.DataFrame, pca_diag: pd.DataFrame, weights: pd.DataFrame) -> str:
    candidate_count = int(df["rule_dormant_candidate"].sum())
    report = []
    report.append("# analysis_v11_features 特征构造报告")
    report.append("")
    report.append("## 1. 输入与输出")
    report.append("")
    report.append(f"- 输入：`{MAIN_SAMPLE_PATH}`")
    report.append(f"- 输出：`{FEATURE_CSV_PATH}` / `analysis_v11_features.xlsx`")
    report.append(f"- 行数：{len(df)}")
    report.append(f"- 字段数：{len(df.columns)}")
    report.append("")
    report.append("## 2. ActualUse PCA 诊断")
    report.append("")
    report.append("| 资源类型 | 样本数 | 输入变量 | 解释方差占比 | 方向校正后相关 | 载荷 |")
    report.append("|---|---:|---|---:|---:|---|")
    for _, row in pca_diag.iterrows():
        report.append(
            f"| {row['resource_type']} | {int(row['n'])} | `{row['input_cols']}` | "
            f"{row['explained_variance_ratio']:.4f} | {row['direction_corr_after']:.4f} | `{row['loadings']}` |"
        )
    report.append("")
    report.append("## 3. CRITIC 权重")
    report.append("")
    report.append("| 维度 | CRITIC 权重 | 熵权法权重 |")
    report.append("|---|---:|---:|")
    for _, row in weights.iterrows():
        report.append(f"| `{row['dimension']}` | {row['critic_weight']:.4f} | {row['entropy_weight']:.4f} |")
    report.append("")
    report.append("## 4. 规则沉睡识别")
    report.append("")
    report.append(f"- rule_dormant_candidate：{candidate_count} 条")
    report.append(f"- 四象限分布：{df['dormant_type_rule'].value_counts().to_dict()}")
    report.append("")
    report.append("规则阈值：")
    report.append("")
    report.append("```text")
    report.append("PotentialUse_CRITIC_percentile >= 0.70")
    report.append("ActualUse_type_percentile <= 0.50")
    report.append("DormantScore_rule >= 0.30")
    report.append("```")
    report.append("")
    report.append("## 5. 建模口径提醒")
    report.append("")
    report.append("- 本文件已生成 V1.1 规则潜力与规则沉睡度，但尚未生成 `ExpectedUse_model` 和 `DormantScore_model`。")
    report.append("- 下一步 ExpectedUse 必须使用 K-fold out-of-fold 预测。")
    report.append("- 禁止把浏览量、下载量、接口调用量、ActualUse、DormantScore、scrape_status、ID、URL、评分评论等字段放入 ExpectedUse 特征矩阵。")
    return "\n".join(report) + "\n"


def export_outputs(df: pd.DataFrame, weights: pd.DataFrame, pca_diag: pd.DataFrame, report: str) -> None:
    feature_dict = feature_dictionary()
    candidates = df[df["rule_dormant_candidate"] == 1].sort_values("DormantScore_rule", ascending=False).copy()

    df.to_csv(FEATURE_CSV_PATH, index=False, encoding="utf-8-sig")
    candidates.to_csv(RULE_CANDIDATE_CSV_PATH, index=False, encoding="utf-8-sig")
    weights.to_csv(CRITIC_WEIGHTS_PATH, index=False, encoding="utf-8-sig")
    pca_diag.to_csv(PCA_DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig")
    feature_dict.to_csv(FEATURE_DICTIONARY_PATH, index=False, encoding="utf-8-sig")
    FEATURE_REPORT_PATH.write_text(report, encoding="utf-8")

    with pd.ExcelWriter(FEATURE_XLSX_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="analysis_v11_features")
        candidates.to_excel(writer, index=False, sheet_name="rule_candidates")
        weights.to_excel(writer, index=False, sheet_name="critic_weights")
        pca_diag.to_excel(writer, index=False, sheet_name="actualuse_pca")
        feature_dict.to_excel(writer, index=False, sheet_name="feature_dictionary")


def main() -> None:
    df = pd.read_csv(MAIN_SAMPLE_PATH, dtype=str, encoding="utf-8-sig")
    df["feature_version"] = "V1.1"
    df, pca_diag = build_actual_use(df)
    df = build_potential_dimensions(df)
    df, weights = build_potential_use(df)
    df = build_rule_dormancy(df)
    report = make_report(df, pca_diag, weights)
    export_outputs(df, weights, pca_diag, report)

    print("rows", len(df))
    print("cols", len(df.columns))
    print("rule_dormant_candidate", int(df["rule_dormant_candidate"].sum()))
    print("quadrants", df["dormant_type_rule"].value_counts().to_dict())
    print("outputs")
    for path in [
        FEATURE_CSV_PATH,
        FEATURE_XLSX_PATH,
        RULE_CANDIDATE_CSV_PATH,
        CRITIC_WEIGHTS_PATH,
        PCA_DIAGNOSTICS_PATH,
        FEATURE_REPORT_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()
