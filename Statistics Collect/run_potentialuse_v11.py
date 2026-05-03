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
POTENTIAL_DIR = OUTPUT_DIR / "potentialuse_v11"
FIG_DIR = POTENTIAL_DIR / "figures"
TABLE_DIR = POTENTIAL_DIR / "tables"

FEATURE_PATH = OUTPUT_DIR / "analysis_v11_features.csv"
WEIGHT_PATH = OUTPUT_DIR / "potentialuse_critic_weights_v11.csv"
REPORT_PATH = POTENTIAL_DIR / "potentialuse_v11_report.md"

DIMENSION_COLS = [
    "topic_public_value_score",
    "semantic_clarity_score",
    "data_richness_score",
    "machine_readability_score",
    "timeliness_score",
    "spatiotemporal_capability_score",
    "combinability_score",
]

DIMENSION_LABELS = {
    "topic_public_value_score": "主题公共价值",
    "semantic_clarity_score": "语义清晰度",
    "data_richness_score": "数据丰富度",
    "machine_readability_score": "机器可读性",
    "timeliness_score": "时效性",
    "spatiotemporal_capability_score": "时空能力",
    "combinability_score": "可组合性",
}


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


def top_overlap(df: pd.DataFrame, metric_a: str, metric_b: str, pct: float, high: bool = True) -> dict[str, object]:
    n = max(1, int(np.ceil(len(df) * pct)))
    if high:
        a = set(df.nlargest(n, metric_a)["数据集ID"])
        b = set(df.nlargest(n, metric_b)["数据集ID"])
    else:
        a = set(df.nsmallest(n, metric_a)["数据集ID"])
        b = set(df.nsmallest(n, metric_b)["数据集ID"])
    overlap = len(a & b)
    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "top_pct": pct,
        "top_n": n,
        "overlap_count": overlap,
        "overlap_rate": overlap / n,
    }


def build_dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in DIMENSION_COLS:
        s = num(df[col])
        rows.append(
            {
                "dimension": col,
                "dimension_cn": DIMENSION_LABELS[col],
                "count": int(s.notna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "p10": float(s.quantile(0.10)),
                "median": float(s.median()),
                "p90": float(s.quantile(0.90)),
                "max": float(s.max()),
            }
        )
    return save_table(pd.DataFrame(rows), "potentialuse_dimension_summary")


def build_score_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "PotentialUse_CRITIC",
        "PotentialUse_CRITIC_percentile",
        "PotentialUse_equal",
        "PotentialUse_equal_percentile",
        "PotentialUse_entropy",
        "PotentialUse_entropy_percentile",
    ]:
        s = num(df[col])
        rows.append(
            {
                "score": col,
                "count": int(s.notna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "median": float(s.median()),
                "max": float(s.max()),
            }
        )
    return save_table(pd.DataFrame(rows), "potentialuse_score_summary")


def build_correlation_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_cols = [
        "PotentialUse_CRITIC_percentile",
        "PotentialUse_equal_percentile",
        "PotentialUse_entropy_percentile",
        "ActualUse_type_percentile",
        "DormantScore_rule",
    ]
    rank_corr = df[score_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman").reset_index(names="metric")
    save_table(rank_corr, "potentialuse_rank_correlation")

    dim_corr = df[DIMENSION_COLS + ["PotentialUse_CRITIC_percentile"]].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    dim_corr.to_csv(TABLE_DIR / "potentialuse_dimension_spearman_matrix.csv", encoding="utf-8-sig")

    rows = []
    for col in DIMENSION_COLS:
        rho, p = stats.spearmanr(num(df[col]), num(df["PotentialUse_CRITIC_percentile"]), nan_policy="omit")
        rows.append(
            {
                "dimension": col,
                "dimension_cn": DIMENSION_LABELS[col],
                "spearman_with_potentialuse": float(rho),
                "p_value": float(p),
            }
        )
    dim_to_score = save_table(pd.DataFrame(rows).sort_values("spearman_with_potentialuse", ascending=False), "potentialuse_dimension_correlations")
    return rank_corr, dim_corr.reset_index(names="dimension"), dim_to_score


def build_top_overlap_tables(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["PotentialUse_CRITIC_percentile", "PotentialUse_equal_percentile", "PotentialUse_entropy_percentile"]:
        work[col] = num(work[col])
    rows = []
    for metric in ["PotentialUse_equal_percentile", "PotentialUse_entropy_percentile"]:
        for pct in [0.01, 0.05, 0.10, 0.20, 0.30]:
            rows.append(top_overlap(work, "PotentialUse_CRITIC_percentile", metric, pct))
    return save_table(pd.DataFrame(rows), "potentialuse_top_overlap_robustness")


def build_weight_perturbation(df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260503)
    id_col = "数据集ID" if "数据集ID" in df.columns else next(col for col in df.columns if col.endswith("ID"))
    base_w = weights.set_index("dimension").loc[DIMENSION_COLS, "critic_weight"].to_numpy(dtype=float)
    dims = df[DIMENSION_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    base_score = num(df["PotentialUse_CRITIC"]).to_numpy(dtype=float)
    base_rank = pd.DataFrame({id_col: df[id_col], "score": base_score})
    base_top10 = set(base_rank.nlargest(int(np.ceil(len(df) * 0.10)), "score")[id_col])
    base_top30 = set(base_rank.nlargest(int(np.ceil(len(df) * 0.30)), "score")[id_col])
    rows = []
    for amplitude in [0.10, 0.20]:
        for i in range(200):
            jitter = rng.uniform(1 - amplitude, 1 + amplitude, size=len(base_w))
            w = base_w * jitter
            w = w / w.sum()
            score = dims @ w
            rho = stats.spearmanr(base_score, score, nan_policy="omit").statistic
            temp = pd.DataFrame({id_col: df[id_col], "score": score})
            top10 = set(temp.nlargest(int(np.ceil(len(temp) * 0.10)), "score")[id_col])
            top30 = set(temp.nlargest(int(np.ceil(len(temp) * 0.30)), "score")[id_col])
            rows.append(
                {
                    "amplitude": amplitude,
                    "iteration": i + 1,
                    "spearman_with_base": float(rho),
                    "top10_overlap_rate": len(base_top10 & top10) / len(base_top10),
                    "top30_overlap_rate": len(base_top30 & top30) / len(base_top30),
                    **{f"w_{dim}": w[j] for j, dim in enumerate(DIMENSION_COLS)},
                }
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(TABLE_DIR / "potentialuse_weight_perturbation_detail.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby("amplitude")[["spearman_with_base", "top10_overlap_rate", "top30_overlap_rate"]]
        .agg(["mean", "min", "median", "max"])
        .reset_index()
    )
    summary.columns = ["_".join([str(c) for c in col if str(c)]) for col in summary.columns.to_flat_index()]
    return save_table(summary, "potentialuse_weight_perturbation_summary")


def build_group_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col in ["数据资源类型", "开放属性", "数据领域", "更新频率"]:
        grouped = (
            df.groupby(group_col)
            .agg(
                count=("数据集ID", "count"),
                potential_median=("PotentialUse_CRITIC_percentile", lambda s: num(s).median()),
                potential_mean=("PotentialUse_CRITIC_percentile", lambda s: num(s).mean()),
                high_potential_count=("PotentialUse_CRITIC_percentile", lambda s: int(num(s).ge(0.70).sum())),
            )
            .reset_index()
        )
        grouped["high_potential_share"] = grouped["high_potential_count"] / grouped["count"]
        grouped.insert(0, "group_col", group_col)
        grouped = grouped.rename(columns={group_col: "group_value"})
        rows.append(grouped)
    out = pd.concat(rows, ignore_index=True)
    return save_table(out, "potentialuse_group_profile")


def build_high_potential_list(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "数据集ID",
        "数据资源名称",
        "数据领域",
        "数据资源类型",
        "开放属性",
        "PotentialUse_CRITIC",
        "PotentialUse_CRITIC_percentile",
        *DIMENSION_COLS,
        "ActualUse_type_percentile",
        "DormantScore_rule",
        "rule_dormant_candidate",
    ]
    out = df[df["PotentialUse_CRITIC_percentile"].astype(float).ge(0.70)][cols].sort_values("PotentialUse_CRITIC_percentile", ascending=False)
    out.to_csv(TABLE_DIR / "potentialuse_high_potential_resources.csv", index=False, encoding="utf-8-sig")
    return out


def build_figures(df: pd.DataFrame, weights: pd.DataFrame, rank_corr: pd.DataFrame, dim_corr_matrix: pd.DataFrame, group_profile: pd.DataFrame) -> None:
    weight_long = weights[["dimension", "critic_weight", "entropy_weight"]].copy()
    weight_long["dimension_cn"] = weight_long["dimension"].map(DIMENSION_LABELS)
    weight_long = weight_long.melt(id_vars=["dimension", "dimension_cn"], value_vars=["critic_weight", "entropy_weight"], var_name="weight_type", value_name="weight")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=weight_long, x="weight", y="dimension_cn", hue="weight_type")
    plt.title("PotentialUse 七维权重：CRITIC vs 熵权法")
    plt.xlabel("权重")
    plt.ylabel("")
    save_fig(FIG_DIR / "6_1_PotentialUse七维权重对比.png")

    dim_long = df[DIMENSION_COLS].apply(pd.to_numeric, errors="coerce").rename(columns=DIMENSION_LABELS).melt(var_name="维度", value_name="score")
    plt.figure(figsize=(11, 5.5))
    sns.boxplot(data=dim_long, x="score", y="维度", color="#7AA6C2")
    plt.title("PotentialUse 七个维度分数分布")
    plt.xlabel("0-1 维度分数")
    plt.ylabel("")
    save_fig(FIG_DIR / "6_2_PotentialUse七维分数分布.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(pd.to_numeric(df["PotentialUse_CRITIC"], errors="coerce"), bins=50, color="#4C78A8")
    plt.title("PotentialUse_CRITIC 综合分数分布")
    plt.xlabel("PotentialUse_CRITIC")
    plt.ylabel("频数")
    save_fig(FIG_DIR / "6_3_PotentialUse_CRITIC分布.png")

    corr = rank_corr.set_index("metric").apply(pd.to_numeric, errors="coerce")
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0)
    plt.title("PotentialUse 口径与 ActualUse/DormantScore 排名相关")
    save_fig(FIG_DIR / "6_4_PotentialUse稳健性排名相关矩阵.png")

    corr_matrix = dim_corr_matrix.set_index("dimension").apply(pd.to_numeric, errors="coerce")
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=False, cmap="RdBu_r", center=0)
    plt.title("PotentialUse 七维分数 Spearman 相关矩阵")
    save_fig(FIG_DIR / "6_5_PotentialUse七维相关矩阵.png")

    domain_profile = group_profile[group_profile["group_col"].eq("数据领域")].sort_values("potential_median", ascending=False).head(17)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=domain_profile.iloc[::-1], x="potential_median", y="group_value", color="#59A14F")
    plt.title("不同数据领域 PotentialUse 中位数")
    plt.xlabel("PotentialUse_CRITIC_percentile 中位数")
    plt.ylabel("")
    save_fig(FIG_DIR / "6_6_不同领域PotentialUse中位数.png")

    scatter = df.sample(min(3000, len(df)), random_state=20260503)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=scatter,
        x="ActualUse_type_percentile",
        y="PotentialUse_CRITIC_percentile",
        hue="数据资源类型",
        s=14,
        alpha=0.6,
        linewidth=0,
    )
    plt.axhline(0.70, color="red", linestyle="--", linewidth=1, label="高潜阈值 0.70")
    plt.title("PotentialUse 与 ActualUse 关系")
    plt.xlabel("ActualUse_type_percentile")
    plt.ylabel("PotentialUse_CRITIC_percentile")
    save_fig(FIG_DIR / "6_7_PotentialUse与ActualUse关系.png")


def make_report(
    df: pd.DataFrame,
    weights: pd.DataFrame,
    dim_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    rank_corr: pd.DataFrame,
    top_overlap_df: pd.DataFrame,
    perturb_summary: pd.DataFrame,
    group_profile: pd.DataFrame,
    high_potential: pd.DataFrame,
) -> str:
    critic_equal_corr = float(
        rank_corr.loc[rank_corr["metric"].eq("PotentialUse_CRITIC_percentile"), "PotentialUse_equal_percentile"].iloc[0]
    )
    critic_entropy_corr = float(
        rank_corr.loc[rank_corr["metric"].eq("PotentialUse_CRITIC_percentile"), "PotentialUse_entropy_percentile"].iloc[0]
    )
    high_count = len(high_potential)
    high_share = high_count / len(df)
    top_weight = weights.sort_values("critic_weight", ascending=False).iloc[0]

    report = []
    report.append("# 第6章 PotentialUse 潜在价值构造 V1.1")
    report.append("")
    report.append("## 6.1 PotentialUse 的理论定义")
    report.append("")
    report.append("PotentialUse 表示公共数据资源基于公共价值、内容质量、机器可读性、时效性、时空能力和可组合性所体现出的内在使用潜力。它回答的是：这个数据从自身属性看，本来是否值得被使用。")
    report.append("")
    report.append("本章严格避免将浏览量、下载量、接口调用量、ActualUse 或 DormantScore 等使用表现变量纳入 PotentialUse 构造。")
    report.append("")
    report.append("## 6.2 为什么需要 PotentialUse")
    report.append("")
    report.append("如果只看 ActualUse，会把所有低使用资源都当成问题；PotentialUse 的作用是区分“低潜低用”和“高潜低用”。只有高潜低用才是后续沉睡资产识别的核心对象。")
    report.append("")
    report.append("## 6.3 七个潜在价值维度")
    report.append("")
    report.append("七个维度的实际数据分布如下：")
    report.append("")
    report.append(markdown_table(dim_summary[["dimension_cn", "mean", "std", "min", "median", "max"]]))
    report.append("")
    report.append("![PotentialUse七维分数分布](figures/6_2_PotentialUse七维分数分布.png)")
    report.append("")
    report.append("七维之间的相关结构用于检查信息冗余，CRITIC 会同时考虑维度变异性与冲突性。")
    report.append("")
    report.append("![PotentialUse七维相关矩阵](figures/6_5_PotentialUse七维相关矩阵.png)")
    report.append("")
    report.append("## 6.4 为什么不能手动设定 PotentialUse 权重")
    report.append("")
    report.append("本项目不采用人工固定权重作为主口径。原因是七个维度的信息量和冗余程度不同，人工权重容易被质疑。V1.1 采用“理论维度定义 + 客观赋权”：理论决定哪些维度重要，数据决定各维度在样本中提供多少有效信息。")
    report.append("")
    report.append("## 6.5 PotentialUse 主口径：CRITIC 客观赋权")
    report.append("")
    report.append("CRITIC 权重如下：")
    report.append("")
    weight_show = weights.copy()
    weight_show["dimension_cn"] = weight_show["dimension"].map(DIMENSION_LABELS)
    report.append(markdown_table(weight_show[["dimension_cn", "std", "conflict", "information", "critic_weight", "entropy_weight"]]))
    report.append("")
    report.append(f"- CRITIC 权重最高维度：{DIMENSION_LABELS[top_weight['dimension']]}，权重 {float(top_weight['critic_weight']):.4f}。")
    report.append("- 权重没有被单一维度垄断，说明七维框架仍然保持综合评价性质。")
    report.append("")
    report.append("![PotentialUse七维权重对比](figures/6_1_PotentialUse七维权重对比.png)")
    report.append("")
    report.append("PotentialUse 综合分数摘要：")
    report.append("")
    report.append(markdown_table(score_summary))
    report.append("")
    report.append("![PotentialUse_CRITIC分布](figures/6_3_PotentialUse_CRITIC分布.png)")
    report.append("")
    report.append(f"- 高潜力资源阈值 `PotentialUse_CRITIC_percentile >= 0.70` 下，共 {high_count} 条，占主样本 {high_share:.2%}。")
    report.append("")
    report.append("## 6.6 PotentialUse 稳健性口径")
    report.append("")
    report.append("V1.1 保留等权法和熵权法作为稳健性检验，不替代 CRITIC 主口径。")
    report.append("")
    report.append("主口径与稳健性口径的排名相关：")
    report.append("")
    report.append(markdown_table(rank_corr))
    report.append("")
    report.append(f"- CRITIC vs 等权法 Spearman：{critic_equal_corr:.4f}")
    report.append(f"- CRITIC vs 熵权法 Spearman：{critic_entropy_corr:.4f}")
    report.append("")
    report.append("Top-N 高潜资源重合率：")
    report.append("")
    report.append(markdown_table(top_overlap_df, max_rows=20))
    report.append("")
    report.append("权重扰动检验：")
    report.append("")
    report.append(markdown_table(perturb_summary))
    report.append("")
    report.append("![PotentialUse稳健性排名相关矩阵](figures/6_4_PotentialUse稳健性排名相关矩阵.png)")
    report.append("")
    report.append("## 6.7 实际数据中的 PotentialUse 结构")
    report.append("")
    domain_profile = group_profile[group_profile["group_col"].eq("数据领域")].sort_values("potential_median", ascending=False).head(10)
    report.append("不同数据领域 PotentialUse 中位数 Top 10：")
    report.append("")
    report.append(markdown_table(domain_profile[["group_value", "count", "potential_median", "high_potential_count", "high_potential_share"]], max_rows=10))
    report.append("")
    report.append("![不同领域PotentialUse中位数](figures/6_6_不同领域PotentialUse中位数.png)")
    report.append("")
    report.append("![PotentialUse与ActualUse关系](figures/6_7_PotentialUse与ActualUse关系.png)")
    report.append("")
    report.append("## 6.8 本章结论")
    report.append("")
    report.append("- PotentialUse 已按 V1.1 文档构造为七维内在价值指标，不包含任何使用表现变量。")
    report.append("- CRITIC 主口径同时考虑维度差异性和冗余性，比人工权重更适合论文和建模报告表达。")
    report.append("- 等权法、熵权法和权重扰动结果用于稳健性检验，后续规则沉睡度仍以 `PotentialUse_CRITIC_percentile` 为正式口径。")
    report.append("- 下一步应进入第 8.1 的规则沉睡度正式整理：`DormantScore_rule = PotentialUse_CRITIC_percentile - ActualUse_type_percentile`，而不是直接跳到 ExpectedUse。")
    return "\n".join(report) + "\n"


def main() -> None:
    setup()
    df = pd.read_csv(FEATURE_PATH, dtype=str, encoding="utf-8-sig")
    weights = pd.read_csv(WEIGHT_PATH, encoding="utf-8-sig")

    dim_summary = build_dimension_summary(df)
    score_summary = build_score_summary(df)
    rank_corr, dim_corr_matrix, dim_to_score = build_correlation_tables(df)
    top_overlap_df = build_top_overlap_tables(df)
    perturb_summary = build_weight_perturbation(df, weights)
    group_profile = build_group_profile(df)
    high_potential = build_high_potential_list(df)
    build_figures(df, weights, rank_corr, dim_corr_matrix, group_profile)
    report = make_report(df, weights, dim_summary, score_summary, rank_corr, top_overlap_df, perturb_summary, group_profile, high_potential)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("potentialuse v11 done")
    print("rows", len(df))
    print("high_potential", len(high_potential))
    print("report", REPORT_PATH)
    print("figures", FIG_DIR)
    print("tables", TABLE_DIR)


if __name__ == "__main__":
    main()
