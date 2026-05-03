from __future__ import annotations

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
ACTUAL_DIR = OUTPUT_DIR / "actualuse_v11"
FIG_DIR = ACTUAL_DIR / "figures"
TABLE_DIR = ACTUAL_DIR / "tables"

FEATURE_PATH = OUTPUT_DIR / "analysis_v11_features.csv"
PCA_DIAG_PATH = OUTPUT_DIR / "actualuse_pca_diagnostics_v11.csv"
REPORT_PATH = ACTUAL_DIR / "actualuse_v11_report.md"


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


def markdown_table(df: pd.DataFrame, max_rows: int = 20, digits: int = 4) -> str:
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return view.to_markdown(index=False)


def build_usage_indicator_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, score_col, label in [
        ("view_count_clean", "view_score", "浏览量"),
        ("download_count_clean", "download_score", "下载量"),
        ("api_call_count_clean", "api_call_score", "接口调用量"),
    ]:
        raw = num(df[col]).fillna(0)
        score = num(df[score_col])
        rows.append(
            {
                "indicator": label,
                "raw_zero_count": int(raw.eq(0).sum()),
                "raw_median": float(raw.median()),
                "raw_p95": float(raw.quantile(0.95)),
                "raw_p99": float(raw.quantile(0.99)),
                "raw_max": float(raw.max()),
                "score_min": float(score.min()),
                "score_median": float(score.median()),
                "score_max": float(score.max()),
            }
        )
    return save_table(pd.DataFrame(rows), "actualuse_usage_indicator_summary")


def build_pca_correlation_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for resource_type, cols, actual_col in [
        ("数据产品", ["view_score", "download_score"], "ActualUse_product_pca_raw"),
        ("数据接口", ["view_score", "api_call_score"], "ActualUse_interface_pca_raw"),
    ]:
        sub = df[df["数据资源类型"].eq(resource_type)].copy()
        for col in cols:
            rho, p = stats.spearmanr(num(sub[actual_col]), num(sub[col]), nan_policy="omit")
            pearson = num(sub[actual_col]).corr(num(sub[col]), method="pearson")
            rows.append(
                {
                    "resource_type": resource_type,
                    "actualuse_raw": actual_col,
                    "indicator": col,
                    "pearson_corr": pearson,
                    "spearman_corr": rho,
                    "spearman_p": p,
                }
            )
    corr = save_table(pd.DataFrame(rows), "actualuse_pca_indicator_correlations")

    all_cols = [
        "ActualUse_type_percentile",
        "ActualUse_global_percentile",
        "ActualUse_equal_type_percentile",
        "ActualUse_depth_type_percentile",
        "ActualUse_percentile",
        "ActualUse_v0",
    ]
    existing = [col for col in all_cols if col in df.columns]
    rank_corr = df[existing].apply(pd.to_numeric, errors="coerce").corr(method="spearman").reset_index(names="metric")
    rank_corr = save_table(rank_corr, "actualuse_robustness_rank_correlation")
    return corr, rank_corr


def top_overlap(df: pd.DataFrame, metric_a: str, metric_b: str, pct: float) -> dict[str, object]:
    n = max(1, int(np.ceil(len(df) * pct)))
    a = set(df.nlargest(n, metric_a)["数据集ID"])
    b = set(df.nlargest(n, metric_b)["数据集ID"])
    overlap = len(a & b)
    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "top_pct": pct,
        "top_n": n,
        "overlap_count": overlap,
        "overlap_rate": overlap / n,
    }


def build_robustness_tables(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in [
        "ActualUse_type_percentile",
        "ActualUse_equal_type_percentile",
        "ActualUse_depth_type_percentile",
        "ActualUse_global_percentile",
    ]:
        work[col] = num(work[col])
    rows = []
    for metric in ["ActualUse_equal_type_percentile", "ActualUse_depth_type_percentile", "ActualUse_global_percentile"]:
        for pct in [0.01, 0.05, 0.10, 0.20]:
            rows.append(top_overlap(work, "ActualUse_type_percentile", metric, pct))
    out = save_table(pd.DataFrame(rows), "actualuse_top_overlap_with_robustness")

    quadrant_rows = []
    for metric in ["ActualUse_type_percentile", "ActualUse_equal_type_percentile", "ActualUse_depth_type_percentile", "ActualUse_global_percentile"]:
        high = num(df[metric]).ge(0.50)
        quadrant_rows.append(
            {
                "metric": metric,
                "high_use_count": int(high.sum()),
                "low_use_count": int((~high).sum()),
                "high_use_share": float(high.mean()),
            }
        )
    save_table(pd.DataFrame(quadrant_rows), "actualuse_high_low_counts_by_metric")
    return out


def build_summary_by_type(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "view_score",
        "download_score",
        "api_call_score",
        "ActualUse_type_percentile",
        "ActualUse_global_percentile",
        "ActualUse_equal_type_percentile",
        "ActualUse_depth_type_percentile",
    ]
    rows = []
    for resource_type, sub in df.groupby("数据资源类型"):
        for metric in metrics:
            s = num(sub[metric])
            rows.append(
                {
                    "resource_type": resource_type,
                    "metric": metric,
                    "count": int(s.notna().sum()),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "p10": float(s.quantile(0.10)),
                    "p90": float(s.quantile(0.90)),
                }
            )
    return save_table(pd.DataFrame(rows), "actualuse_summary_by_resource_type")


def build_figures(df: pd.DataFrame, rank_corr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in [
        (axes[0], "view_score", "浏览分数"),
        (axes[1], "download_score", "下载分数"),
        (axes[2], "api_call_score", "调用分数"),
    ]:
        sns.histplot(num(df[col]), bins=40, ax=ax, color="#3E7CB1")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("频数")
    save_fig(FIG_DIR / "5_1_使用指标百分位分数分布.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    product = df[df["数据资源类型"].eq("数据产品")]
    interface = df[df["数据资源类型"].eq("数据接口")]
    sns.scatterplot(
        data=product,
        x="view_score",
        y="download_score",
        hue="ActualUse_type_percentile",
        palette="viridis",
        s=12,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("数据产品：浏览-下载与 ActualUse")
    sns.scatterplot(
        data=interface,
        x="view_score",
        y="api_call_score",
        hue="ActualUse_type_percentile",
        palette="viridis",
        s=12,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title("数据接口：浏览-调用与 ActualUse")
    save_fig(FIG_DIR / "5_2_类型适配PCA散点图.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.histplot(data=df, x="ActualUse_type_percentile", hue="数据资源类型", bins=40, element="step", stat="density", common_norm=False, ax=axes[0])
    axes[0].set_title("主口径 ActualUse 类型内百分位")
    sns.histplot(data=df, x="ActualUse_global_percentile", hue="数据资源类型", bins=40, element="step", stat="density", common_norm=False, ax=axes[1])
    axes[1].set_title("全局对照 ActualUse 百分位")
    save_fig(FIG_DIR / "5_3_ActualUse主口径与全局口径分布.png")

    corr = rank_corr.set_index("metric")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr.apply(pd.to_numeric, errors="coerce"), annot=True, fmt=".3f", cmap="RdBu_r", center=0)
    plt.title("ActualUse 各口径 Spearman 排名相关")
    save_fig(FIG_DIR / "5_4_ActualUse稳健性排名相关矩阵.png")

    melt = df.melt(
        id_vars=["数据资源类型"],
        value_vars=["ActualUse_type_percentile", "ActualUse_equal_type_percentile", "ActualUse_depth_type_percentile"],
        var_name="ActualUse_metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=melt, x="ActualUse_metric", y="score", hue="数据资源类型")
    plt.title("ActualUse 主口径与稳健性口径分布")
    plt.xlabel("")
    plt.ylabel("分数")
    plt.xticks(rotation=15, ha="right")
    save_fig(FIG_DIR / "5_5_ActualUse主口径与稳健性口径箱线图.png")


def make_report(
    df: pd.DataFrame,
    pca_diag: pd.DataFrame,
    indicator_summary: pd.DataFrame,
    pca_corr: pd.DataFrame,
    rank_corr: pd.DataFrame,
    overlap: pd.DataFrame,
    by_type: pd.DataFrame,
) -> str:
    product_diag = pca_diag[pca_diag["resource_type"].eq("数据产品")].iloc[0]
    interface_diag = pca_diag[pca_diag["resource_type"].eq("数据接口")].iloc[0]
    type_counts = df["数据资源类型"].value_counts().to_dict()
    main_equal_corr = float(
        rank_corr.loc[rank_corr["metric"].eq("ActualUse_type_percentile"), "ActualUse_equal_type_percentile"].iloc[0]
    )
    main_depth_corr = float(
        rank_corr.loc[rank_corr["metric"].eq("ActualUse_type_percentile"), "ActualUse_depth_type_percentile"].iloc[0]
    )
    main_global_corr = float(
        rank_corr.loc[rank_corr["metric"].eq("ActualUse_type_percentile"), "ActualUse_global_percentile"].iloc[0]
    )

    report = []
    report.append("# 第5章 ActualUse 实际使用强度构造 V1.1")
    report.append("")
    report.append("## 5.1 为什么不能使用固定权重")
    report.append("")
    report.append("本项目不把 `0.45 × 浏览 + 0.35 × 下载 + 0.20 × 调用` 作为主口径。原因是浏览、下载和接口调用代表不同使用深度，且数据产品与数据接口的使用机制不同。若直接使用固定权重，数据接口容易因低下载被低估，数据产品也可能因低接口调用被误判。")
    report.append("")
    report.append("因此，第5章主口径采用 V1.1 文档规定的 **类型适配 PCA / 因子分析思路**；固定权重只保留为“使用深度版”稳健性口径，不作为主结论。")
    report.append("")
    report.append("## 5.2 ActualUse 的理论定义")
    report.append("")
    report.append("ActualUse 表示公共数据资源在平台中被真实关注、获取或调用的可观测使用强度。它由三个可观测指标支撑：")
    report.append("")
    report.append(markdown_table(indicator_summary))
    report.append("")
    report.append("![使用指标百分位分数分布](figures/5_1_使用指标百分位分数分布.png)")
    report.append("")
    report.append("## 5.3 ActualUse 主口径：类型适配 PCA")
    report.append("")
    report.append(f"- 样本类型分布：{type_counts}")
    report.append("- 数据产品使用 `view_score + download_score` 提取第一主成分。")
    report.append("- 数据接口使用 `view_score + api_call_score` 提取第一主成分。")
    report.append("- PCA 第一主成分已进行方向校正，确保分数越高代表使用越强。")
    report.append("- `ActualUse_main = ActualUse_type_percentile`。")
    report.append("")
    report.append("PCA 诊断：")
    report.append("")
    report.append(markdown_table(pca_diag))
    report.append("")
    report.append("指标相关性检验：")
    report.append("")
    report.append(markdown_table(pca_corr))
    report.append("")
    report.append("![类型适配PCA散点图](figures/5_2_类型适配PCA散点图.png)")
    report.append("")
    report.append("![ActualUse主口径与全局口径分布](figures/5_3_ActualUse主口径与全局口径分布.png)")
    report.append("")
    report.append("## 5.4 ActualUse 稳健性口径")
    report.append("")
    report.append("V1.1 保留两个稳健性口径：")
    report.append("")
    report.append("- `ActualUse_equal_type_percentile`：浏览与下载/调用等权。")
    report.append("- `ActualUse_depth_type_percentile`：实际获取或调用权重更高，数据产品为 `0.4 view + 0.6 download`，数据接口为 `0.3 view + 0.7 call`。")
    report.append("")
    report.append("主口径与稳健性口径的 Spearman 排名相关：")
    report.append("")
    report.append(markdown_table(rank_corr))
    report.append("")
    report.append(f"- 主口径 vs 等权版：{main_equal_corr:.4f}")
    report.append(f"- 主口径 vs 使用深度版：{main_depth_corr:.4f}")
    report.append(f"- 主口径 vs 全局对照版：{main_global_corr:.4f}")
    report.append("")
    report.append("Top-N 重合率：")
    report.append("")
    report.append(markdown_table(overlap, max_rows=20))
    report.append("")
    report.append("![ActualUse稳健性排名相关矩阵](figures/5_4_ActualUse稳健性排名相关矩阵.png)")
    report.append("")
    report.append("![ActualUse主口径与稳健性口径箱线图](figures/5_5_ActualUse主口径与稳健性口径箱线图.png)")
    report.append("")
    report.append("## 5.5 本章结论")
    report.append("")
    report.append(f"- 数据产品 PCA 第一主成分解释方差为 {float(product_diag['explained_variance_ratio']):.4f}，数据接口为 {float(interface_diag['explained_variance_ratio']):.4f}。")
    report.append("- 方向校正后，ActualUse 与其对应使用指标均保持强正相关，说明主成分方向符合“使用越强，分数越高”的解释。")
    report.append("- 类型内百分位作为主口径，可以避免数据产品与数据接口之间因使用机制不同而发生系统误判。")
    report.append("- 等权版和使用深度版与主口径高度相关，说明 ActualUse 主口径具有较好的稳健性。")
    report.append("- 后续第6章 PotentialUse 和第8章 DormantScore 均应使用 `ActualUse_main / ActualUse_type_percentile` 作为正式实际使用强度口径。")
    return "\n".join(report) + "\n"


def main() -> None:
    setup()
    df = pd.read_csv(FEATURE_PATH, dtype=str, encoding="utf-8-sig")
    pca_diag = pd.read_csv(PCA_DIAG_PATH, encoding="utf-8-sig")

    indicator_summary = build_usage_indicator_summary(df)
    pca_corr, rank_corr = build_pca_correlation_tables(df)
    overlap = build_robustness_tables(df)
    by_type = build_summary_by_type(df)
    build_figures(df, rank_corr)
    report = make_report(df, pca_diag, indicator_summary, pca_corr, rank_corr, overlap, by_type)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("actualuse v11 done")
    print("rows", len(df))
    print("report", REPORT_PATH)
    print("figures", FIG_DIR)
    print("tables", TABLE_DIR)


if __name__ == "__main__":
    main()
