from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda_v11"
FIG_DIR = EDA_DIR / "figures"
TABLE_DIR = EDA_DIR / "tables"

CLEAN_PATH = OUTPUT_DIR / "analysis_clean_master.csv"
MAIN_PATH = OUTPUT_DIR / "analysis_main_sample.csv"
FEATURE_PATH = OUTPUT_DIR / "analysis_v11_features.csv"
REPORT_PATH = EDA_DIR / "eda_v11_report.md"


def setup_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "<na>", "nat"}:
        return ""
    return s


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def gini(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().clip(lower=0).to_numpy(dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1) / n)


def top_share(values: pd.Series, pct: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().clip(lower=0).sort_values(ascending=False)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    k = max(1, int(math.ceil(len(arr) * pct)))
    return float(arr.head(k).sum() / arr.sum())


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def value_count_table(df: pd.DataFrame, col: str, name: str, top_n: int | None = None) -> pd.DataFrame:
    counts = df[col].fillna("缺失").astype(str).value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)
    out = counts.rename_axis(col).reset_index(name="count")
    out["share"] = out["count"] / len(df)
    save_table(out, name)
    return out


def plot_bar(table: pd.DataFrame, label_col: str, value_col: str, title: str, path: Path, max_rows: int = 20) -> None:
    data = table.head(max_rows).copy()
    data = data.iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.38 * len(data))))
    sns.barplot(data=data, x=value_col, y=label_col, color="#3478A8")
    plt.title(title)
    plt.xlabel("数量")
    plt.ylabel("")
    for i, v in enumerate(data[value_col]):
        plt.text(v, i, f" {int(v)}", va="center", fontsize=8)
    save_fig(path)


def plot_log_hist(df: pd.DataFrame, cols: list[str], labels: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col, label in zip(axes, cols, labels):
        series = np.log1p(num(df[col]).fillna(0))
        sns.histplot(series, bins=45, ax=ax, color="#3A7CA5")
        ax.set_title(f"{label} log1p 分布")
        ax.set_xlabel(f"log1p({label})")
        ax.set_ylabel("频数")
    save_fig(path)


def lorenz_points(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    arr = pd.to_numeric(values, errors="coerce").dropna().clip(lower=0).to_numpy(dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return np.array([0, 1]), np.array([0, 1])
    arr = np.sort(arr)
    cum = np.cumsum(arr)
    y = np.insert(cum / cum[-1], 0, 0)
    x = np.linspace(0, 1, len(y))
    return x, y


def plot_lorenz(df: pd.DataFrame, metric_cols: list[str], labels: list[str], path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for col, label in zip(metric_cols, labels):
        x, y = lorenz_points(df[col])
        plt.plot(x, y, label=f"{label} Gini={gini(df[col]):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="均等线")
    plt.title("使用表现 Lorenz 曲线")
    plt.xlabel("资源累计占比")
    plt.ylabel("使用量累计占比")
    plt.legend()
    save_fig(path)


def cliff_delta_from_u(u: float, n1: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return np.nan
    return float(2 * u / (n1 * n2) - 1)


def mann_whitney_test(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, object]:
    groups = []
    for name, sub in df.groupby(group_col, dropna=False):
        vals = num(sub[value_col]).dropna()
        if len(vals) > 0:
            groups.append((str(name), vals))
    if len(groups) != 2:
        return {
            "test": "Mann-Whitney U",
            "group_col": group_col,
            "value_col": value_col,
            "groups": ";".join(g[0] for g in groups),
            "statistic": np.nan,
            "p_value": np.nan,
            "effect": np.nan,
            "effect_name": "Cliff_delta",
        }
    (g1, v1), (g2, v2) = groups
    result = stats.mannwhitneyu(v1, v2, alternative="two-sided")
    return {
        "test": "Mann-Whitney U",
        "group_col": group_col,
        "value_col": value_col,
        "groups": f"{g1} vs {g2}",
        "n1": len(v1),
        "n2": len(v2),
        "median1": float(v1.median()),
        "median2": float(v2.median()),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect": cliff_delta_from_u(float(result.statistic), len(v1), len(v2)),
        "effect_name": "Cliff_delta",
    }


def kruskal_test(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, object]:
    groups = [num(sub[value_col]).dropna() for _, sub in df.groupby(group_col, dropna=False)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {
            "test": "Kruskal-Wallis",
            "group_col": group_col,
            "value_col": value_col,
            "k": len(groups),
            "statistic": np.nan,
            "p_value": np.nan,
            "effect": np.nan,
            "effect_name": "epsilon_squared",
        }
    result = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    epsilon_sq = max(0.0, (float(result.statistic) - k + 1) / (n - k)) if n > k else np.nan
    return {
        "test": "Kruskal-Wallis",
        "group_col": group_col,
        "value_col": value_col,
        "k": k,
        "n": n,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect": float(epsilon_sq),
        "effect_name": "epsilon_squared",
    }


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    return float(math.sqrt((chi2 / n) / max(1, min(k - 1, r - 1))))


def build_sample_overview(full: pd.DataFrame, main: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"sample": "catalog_full_sample", "rows": len(full), "source": "analysis_clean_master"},
        {"sample": "main_analysis_sample", "rows": len(main), "source": "analysis_main_sample"},
        {"sample": "v11_feature_sample", "rows": len(features), "source": "analysis_v11_features"},
    ]
    out = pd.DataFrame(rows)
    save_table(out, "sample_overview")
    return out


def build_profile_tables(full: pd.DataFrame, main: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["full_status"] = value_count_table(full, "scrape_status", "full_status")
    tables["full_domain"] = value_count_table(full, "数据领域", "full_domain")
    tables["main_domain"] = value_count_table(main, "数据领域", "main_domain")
    tables["main_type"] = value_count_table(main, "数据资源类型", "main_resource_type")
    tables["main_open"] = value_count_table(main, "开放属性", "main_open_attribute")
    tables["main_update"] = value_count_table(main, "更新频率", "main_update_frequency")
    tables["main_spatial"] = value_count_table(main, "spatial_admin_level", "main_spatial_admin_level")
    tables["top_departments"] = value_count_table(full, "数据资源提供部门", "top_departments", top_n=30)

    formats = []
    for fmt in ["rdf", "xml", "csv", "json", "xlsx"]:
        col = f"has_{fmt}_clean" if f"has_{fmt}_clean" in main.columns else f"has_{fmt}"
        if col in main.columns:
            count = int(num(main[col]).fillna(0).gt(0).sum())
            formats.append({"format": fmt.upper(), "count": count, "share": count / len(main)})
    tables["format_prevalence"] = pd.DataFrame(formats)
    save_table(tables["format_prevalence"], "format_prevalence")
    return tables


def build_usage_tables(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    usage_cols = [
        ("view_count_clean", "浏览量"),
        ("download_count_clean", "下载量"),
        ("api_call_count_clean", "接口调用量"),
        ("ActualUse_type_percentile", "ActualUse_type_percentile"),
    ]
    rows = []
    for col, label in usage_cols:
        s = num(features[col]).fillna(0)
        rows.append(
            {
                "metric": label,
                "count": int(s.notna().sum()),
                "zero_count": int(s.eq(0).sum()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p90": float(s.quantile(0.90)),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "gini": gini(s),
                "top1_share": top_share(s, 0.01),
                "top5_share": top_share(s, 0.05),
            }
        )
    summary = pd.DataFrame(rows)
    save_table(summary, "usage_long_tail_summary")

    conv = features.copy()
    conv["download_conversion"] = num(conv["download_count_clean"]).fillna(0) / (num(conv["view_count_clean"]).fillna(0) + 1)
    conv["api_conversion"] = num(conv["api_call_count_clean"]).fillna(0) / (num(conv["view_count_clean"]).fillna(0) + 1)
    conv["overall_conversion"] = (
        num(conv["download_count_clean"]).fillna(0) + num(conv["api_call_count_clean"]).fillna(0)
    ) / (num(conv["view_count_clean"]).fillna(0) + 1)
    conversion_cols = ["download_conversion", "api_conversion", "overall_conversion"]
    conv_summary = conv.groupby("数据资源类型")[conversion_cols].agg(["count", "median", "mean", "max"])
    conv_summary.columns = ["_".join(col) for col in conv_summary.columns]
    conv_summary = conv_summary.reset_index()
    save_table(conv_summary, "conversion_summary_by_type")
    conv[["数据集ID", "数据资源名称", "数据资源类型", "数据领域", *conversion_cols]].to_csv(
        TABLE_DIR / "conversion_record_level.csv", index=False, encoding="utf-8-sig"
    )
    return {"usage_summary": summary, "conversion_summary": conv_summary, "conversion_records": conv}


def build_group_tests(features: pd.DataFrame) -> pd.DataFrame:
    tests = []
    for value_col in [
        "ActualUse_type_percentile",
        "view_score",
        "download_score",
        "api_call_score",
        "PotentialUse_CRITIC_percentile",
        "DormantScore_rule",
    ]:
        tests.append(mann_whitney_test(features, "开放属性", value_col))
        tests.append(mann_whitney_test(features, "数据资源类型", value_col))
        tests.append(kruskal_test(features, "数据领域", value_col))
        tests.append(kruskal_test(features, "更新频率", value_col))
    out = pd.DataFrame(tests)
    save_table(out, "group_difference_tests")

    type_open = pd.crosstab(features["数据资源类型"], features["开放属性"])
    cramers = pd.DataFrame(
        [
            {
                "relationship": "数据资源类型 x 开放属性",
                "cramers_v": cramers_v(type_open),
                "n": int(type_open.to_numpy().sum()),
            }
        ]
    )
    save_table(cramers, "categorical_association_tests")
    type_open.to_csv(TABLE_DIR / "type_open_crosstab.csv", encoding="utf-8-sig")
    return out


def build_feature_relationships(features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ActualUse_type_percentile",
        "PotentialUse_CRITIC_percentile",
        "DormantScore_rule",
        "topic_public_value_score",
        "semantic_clarity_score",
        "data_richness_score",
        "machine_readability_score",
        "timeliness_score",
        "spatiotemporal_capability_score",
        "combinability_score",
        "format_count_clean",
        "field_count_clean",
        "record_count_log_winsor_p99",
        "data_size_log_winsor_p99",
        "update_recency_days_clean",
    ]
    matrix = features[cols].apply(pd.to_numeric, errors="coerce")
    corr = matrix.corr(method="spearman")
    corr.to_csv(TABLE_DIR / "spearman_correlation_matrix.csv", encoding="utf-8-sig")
    pairs = []
    for col in cols:
        if col == "ActualUse_type_percentile":
            continue
        pairs.append(
            {
                "feature": col,
                "spearman_with_actualuse": corr.loc[col, "ActualUse_type_percentile"],
                "spearman_with_potentialuse": corr.loc[col, "PotentialUse_CRITIC_percentile"],
            }
        )
    out = pd.DataFrame(pairs).sort_values("spearman_with_actualuse", key=lambda s: s.abs(), ascending=False)
    save_table(out, "feature_spearman_with_actualuse")

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="RdBu_r", center=0, square=False, cbar_kws={"shrink": 0.7})
    plt.title("主要特征 Spearman 相关矩阵")
    save_fig(FIG_DIR / "spearman_correlation_heatmap.png")
    return out


def build_figures(profile: dict[str, pd.DataFrame], usage: dict[str, pd.DataFrame], features: pd.DataFrame) -> None:
    plot_bar(profile["main_domain"], "数据领域", "count", "主分析样本：数据领域分布", FIG_DIR / "main_domain_distribution.png")
    plot_bar(profile["top_departments"], "数据资源提供部门", "count", "全量样本：提供部门 Top 30", FIG_DIR / "top_departments.png")
    plot_bar(profile["main_update"], "更新频率", "count", "主分析样本：更新频率分布", FIG_DIR / "update_frequency.png")
    plot_bar(profile["format_prevalence"], "format", "count", "主分析样本：下载格式覆盖", FIG_DIR / "format_prevalence.png")

    type_open = pd.crosstab(features["数据资源类型"], features["开放属性"])
    type_open.plot(kind="bar", stacked=True, figsize=(7, 5), color=["#4C78A8", "#F58518"])
    plt.title("资源类型与开放属性交叉分布")
    plt.xlabel("")
    plt.ylabel("数量")
    plt.legend(title="开放属性")
    save_fig(FIG_DIR / "type_open_stacked.png")

    plot_log_hist(
        features,
        ["view_count_clean", "download_count_clean", "api_call_count_clean"],
        ["浏览量", "下载量", "接口调用量"],
        FIG_DIR / "usage_log_histograms.png",
    )
    plot_lorenz(
        features,
        ["view_count_clean", "download_count_clean", "api_call_count_clean"],
        ["浏览量", "下载量", "接口调用量"],
        FIG_DIR / "usage_lorenz_curves.png",
    )

    conv = usage["conversion_records"].copy()
    conv_melt = conv.melt(
        id_vars=["数据资源类型"],
        value_vars=["download_conversion", "api_conversion", "overall_conversion"],
        var_name="conversion_metric",
        value_name="conversion",
    )
    conv_melt["conversion_log1p"] = np.log1p(pd.to_numeric(conv_melt["conversion"], errors="coerce").fillna(0))
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=conv_melt, x="conversion_metric", y="conversion_log1p", hue="数据资源类型")
    plt.title("转化率 log1p 分布：数据产品 vs 数据接口")
    plt.xlabel("")
    plt.ylabel("log1p(转化率)")
    save_fig(FIG_DIR / "conversion_boxplot_by_type.png")

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=features,
        x="ActualUse_type_percentile",
        y="PotentialUse_CRITIC_percentile",
        hue="dormant_type_rule",
        s=12,
        alpha=0.65,
        linewidth=0,
    )
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title("ActualUse 与 PotentialUse 四象限")
    plt.xlabel("ActualUse_type_percentile")
    plt.ylabel("PotentialUse_CRITIC_percentile")
    plt.legend(title="四象限", markerscale=2, fontsize=8)
    save_fig(FIG_DIR / "actualuse_potentialuse_quadrants.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(pd.to_numeric(features["DormantScore_rule"], errors="coerce"), bins=50, color="#7A5195")
    plt.axvline(0.30, color="red", linestyle="--", linewidth=1, label="规则候选阈值 0.30")
    plt.title("规则沉睡度 DormantScore_rule 分布")
    plt.xlabel("DormantScore_rule")
    plt.legend()
    save_fig(FIG_DIR / "dormant_score_distribution.png")


def markdown_table(df: pd.DataFrame, max_rows: int = 12, float_cols: int = 4) -> str:
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.{float_cols}f}" if pd.notna(x) else "")
    return view.to_markdown(index=False)


def make_report(
    full: pd.DataFrame,
    main: pd.DataFrame,
    features: pd.DataFrame,
    profile: dict[str, pd.DataFrame],
    usage: dict[str, pd.DataFrame],
    group_tests: pd.DataFrame,
    feature_rel: pd.DataFrame,
) -> str:
    candidate_count = int(pd.to_numeric(features["rule_dormant_candidate"], errors="coerce").sum())
    type_counts = profile["main_type"].set_index("数据资源类型")["count"].to_dict()
    open_counts = profile["main_open"].set_index("开放属性")["count"].to_dict()
    top_domains = profile["main_domain"].head(6)
    usage_summary = usage["usage_summary"]
    rule_by_type = (
        features[features["rule_dormant_candidate"].astype(str).eq("1")]["数据资源类型"].value_counts().reset_index()
    )
    rule_by_type.columns = ["数据资源类型", "count"]
    rule_by_domain = (
        features[features["rule_dormant_candidate"].astype(str).eq("1")]["数据领域"].value_counts().head(10).reset_index()
    )
    rule_by_domain.columns = ["数据领域", "count"]

    actual_tests = group_tests[group_tests["value_col"].eq("ActualUse_type_percentile")].copy()

    report = []
    report.append("# 上海公共数据资产 EDA V1.1 报告")
    report.append("")
    report.append("## 1. 样本口径")
    report.append("")
    report.append(f"- 全量资源画像样本：{len(full)} 条")
    report.append(f"- 主分析样本：{len(main)} 条")
    report.append(f"- V1.1 特征样本：{len(features)} 条")
    report.append(f"- 规则沉睡候选：{candidate_count} 条")
    report.append("")
    report.append("## 2. 平台资源画像")
    report.append("")
    report.append(f"- 主样本资源类型：{type_counts}")
    report.append(f"- 主样本开放属性：{open_counts}")
    report.append("")
    report.append("主要数据领域 Top 6：")
    report.append("")
    report.append(markdown_table(top_domains, max_rows=6))
    report.append("")
    report.append("![主分析样本：数据领域分布](figures/main_domain_distribution.png)")
    report.append("")
    report.append("![资源类型与开放属性交叉分布](figures/type_open_stacked.png)")
    report.append("")
    report.append("![主分析样本：更新频率分布](figures/update_frequency.png)")
    report.append("")
    report.append("![主分析样本：下载格式覆盖](figures/format_prevalence.png)")
    report.append("")
    report.append("## 3. 使用表现长尾")
    report.append("")
    report.append("核心使用指标摘要：")
    report.append("")
    report.append(markdown_table(usage_summary, max_rows=4))
    report.append("")
    report.append("![使用指标 log1p 分布](figures/usage_log_histograms.png)")
    report.append("")
    report.append("![使用表现 Lorenz 曲线](figures/usage_lorenz_curves.png)")
    report.append("")
    report.append("结论：浏览、下载、接口调用均明显长尾，下载和接口调用的零值比例尤其高；这支持 V1.1 中 `log1p + percentile + 类型适配 ActualUse` 的处理。")
    report.append("")
    report.append("## 4. 转化率分析")
    report.append("")
    report.append("按资源类型统计的转化率：")
    report.append("")
    report.append(markdown_table(usage["conversion_summary"], max_rows=10))
    report.append("")
    report.append("![转化率 log1p 箱线图](figures/conversion_boxplot_by_type.png)")
    report.append("")
    report.append("## 5. 分组差异检验")
    report.append("")
    report.append("ActualUse 主口径的分组检验：")
    report.append("")
    report.append(markdown_table(actual_tests[["test", "group_col", "groups", "statistic", "p_value", "effect", "effect_name"]], max_rows=8))
    report.append("")
    report.append("说明：使用 Mann-Whitney U 和 Kruskal-Wallis，适合长尾和零膨胀使用数据；效应强度比单纯 p 值更适合解释。")
    report.append("")
    report.append("## 6. ActualUse 与 PotentialUse")
    report.append("")
    report.append(f"- 四象限分布：{features['dormant_type_rule'].value_counts().to_dict()}")
    report.append(f"- 规则沉睡候选数：{candidate_count} 条")
    report.append("")
    report.append("规则候选按资源类型：")
    report.append("")
    report.append(markdown_table(rule_by_type))
    report.append("")
    report.append("规则候选按领域 Top 10：")
    report.append("")
    report.append(markdown_table(rule_by_domain, max_rows=10))
    report.append("")
    report.append("![ActualUse 与 PotentialUse 四象限](figures/actualuse_potentialuse_quadrants.png)")
    report.append("")
    report.append("![规则沉睡度分布](figures/dormant_score_distribution.png)")
    report.append("")
    report.append("## 7. 特征关系探索")
    report.append("")
    report.append("与 ActualUse 主口径 Spearman 相关的主要特征：")
    report.append("")
    report.append(markdown_table(feature_rel[["feature", "spearman_with_actualuse", "spearman_with_potentialuse"]], max_rows=12))
    report.append("")
    report.append("![主要特征 Spearman 相关矩阵](figures/spearman_correlation_heatmap.png)")
    report.append("")
    report.append("## 8. 输出文件")
    report.append("")
    report.append("- `tables/`：EDA 统计表 CSV")
    report.append("- `figures/`：EDA 图表 PNG")
    report.append("- `eda_v11_report.md`：本报告")
    report.append("")
    report.append("## 9. 下一步")
    report.append("")
    report.append("EDA 已经证明使用表现长尾明显，并完成了资源画像、转化率、分组差异和特征关系探索。下一步建议进入 `ExpectedUse` 的 K-fold out-of-fold 模型训练，输出模型残差沉睡名单并与规则候选取交集形成高置信沉睡资产。")
    return "\n".join(report) + "\n"


def main() -> None:
    setup_dirs()
    setup_plot_style()

    full = pd.read_csv(CLEAN_PATH, dtype=str, encoding="utf-8-sig")
    main_sample = pd.read_csv(MAIN_PATH, dtype=str, encoding="utf-8-sig")
    features = pd.read_csv(FEATURE_PATH, dtype=str, encoding="utf-8-sig")

    sample_overview = build_sample_overview(full, main_sample, features)
    profile = build_profile_tables(full, main_sample)
    usage = build_usage_tables(features)
    group_tests = build_group_tests(features)
    feature_rel = build_feature_relationships(features)
    build_figures(profile, usage, features)
    report = make_report(full, main_sample, features, profile, usage, group_tests, feature_rel)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("EDA done")
    print("full_rows", len(full))
    print("main_rows", len(main_sample))
    print("feature_rows", len(features))
    print("rule_dormant_candidate", int(pd.to_numeric(features["rule_dormant_candidate"], errors="coerce").sum()))
    print(REPORT_PATH)
    print(FIG_DIR)
    print(TABLE_DIR)


if __name__ == "__main__":
    main()
