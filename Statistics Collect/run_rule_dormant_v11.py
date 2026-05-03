from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RULE_DIR = OUTPUT_DIR / "rule_dormant_v11"
FIG_DIR = RULE_DIR / "figures"
TABLE_DIR = RULE_DIR / "tables"

FEATURE_PATH = OUTPUT_DIR / "analysis_v11_features.csv"
REPORT_PATH = RULE_DIR / "rule_dormant_v11_report.md"

POTENTIAL_THRESHOLD = 0.70
ACTUAL_THRESHOLD = 0.50
DORMANT_SCORE_THRESHOLD = 0.30


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


def safe_value_counts(series: pd.Series, name: str) -> pd.DataFrame:
    counts = series.fillna("缺失").astype(str).value_counts(dropna=False).rename_axis(name).reset_index(name="count")
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts


def add_rule_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ActualUse_main_percentile"] = num(df["ActualUse_type_percentile"])
    df["PotentialUse_main_percentile"] = num(df["PotentialUse_CRITIC_percentile"])
    df["DormantScore_rule_v11"] = df["PotentialUse_main_percentile"] - df["ActualUse_main_percentile"]
    df["rule_dormant_candidate_v11"] = (
        (df["PotentialUse_main_percentile"] >= POTENTIAL_THRESHOLD)
        & (df["ActualUse_main_percentile"] <= ACTUAL_THRESHOLD)
        & (df["DormantScore_rule_v11"] >= DORMANT_SCORE_THRESHOLD)
    )

    potential_high = df["PotentialUse_main_percentile"] >= 0.50
    actual_high = df["ActualUse_main_percentile"] >= 0.50
    df["dormant_type_rule_v11"] = np.select(
        [
            potential_high & actual_high,
            potential_high & ~actual_high,
            ~potential_high & actual_high,
            ~potential_high & ~actual_high,
        ],
        ["高潜高用", "高潜低用", "低潜高用", "低潜低用"],
        default="未分类",
    )
    df["DormantScore_rule_rank_v11"] = df["DormantScore_rule_v11"].rank(ascending=False, method="min")
    return df


def build_threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = [
        {
            "condition": "全体主分析样本",
            "threshold": "analysis_main_sample",
            "count": total,
            "share": 1.0,
        },
        {
            "condition": "高潜力",
            "threshold": f"PotentialUse_CRITIC_percentile >= {POTENTIAL_THRESHOLD:.2f}",
            "count": int((df["PotentialUse_main_percentile"] >= POTENTIAL_THRESHOLD).sum()),
            "share": float((df["PotentialUse_main_percentile"] >= POTENTIAL_THRESHOLD).mean()),
        },
        {
            "condition": "低实际使用",
            "threshold": f"ActualUse_main_percentile <= {ACTUAL_THRESHOLD:.2f}",
            "count": int((df["ActualUse_main_percentile"] <= ACTUAL_THRESHOLD).sum()),
            "share": float((df["ActualUse_main_percentile"] <= ACTUAL_THRESHOLD).mean()),
        },
        {
            "condition": "规则沉睡差距",
            "threshold": f"DormantScore_rule >= {DORMANT_SCORE_THRESHOLD:.2f}",
            "count": int((df["DormantScore_rule_v11"] >= DORMANT_SCORE_THRESHOLD).sum()),
            "share": float((df["DormantScore_rule_v11"] >= DORMANT_SCORE_THRESHOLD).mean()),
        },
        {
            "condition": "规则沉睡候选",
            "threshold": (
                f"PotentialUse >= {POTENTIAL_THRESHOLD:.2f} & "
                f"ActualUse <= {ACTUAL_THRESHOLD:.2f} & "
                f"DormantScore >= {DORMANT_SCORE_THRESHOLD:.2f}"
            ),
            "count": int(df["rule_dormant_candidate_v11"].sum()),
            "share": float(df["rule_dormant_candidate_v11"].mean()),
        },
    ]
    return save_table(pd.DataFrame(rows), "8.1_规则沉睡阈值汇总")


def build_quadrant_counts(df: pd.DataFrame) -> pd.DataFrame:
    order = ["高潜高用", "高潜低用", "低潜低用", "低潜高用"]
    counts = (
        df["dormant_type_rule_v11"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .astype(int)
        .rename_axis("quadrant")
        .reset_index(name="count")
    )
    counts["share"] = counts["count"] / len(df)
    counts["rule_candidate_count"] = [
        int(((df["dormant_type_rule_v11"] == q) & df["rule_dormant_candidate_v11"]).sum()) for q in counts["quadrant"]
    ]
    return save_table(counts, "8.1_四象限数量")


def build_candidate_ranked(df: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        "序号",
        "数据集ID",
        "数据资源名称",
        "数据资源提供部门",
        "数据领域",
        "数据资源类型",
        "开放属性",
        "更新频率",
        "detail_url",
        "view_count_clean",
        "download_count_clean",
        "api_call_count_clean",
        "ActualUse_main_percentile",
        "PotentialUse_main_percentile",
        "DormantScore_rule_v11",
        "DormantScore_rule_rank_v11",
        "topic_public_value_score",
        "semantic_clarity_score",
        "data_richness_score",
        "machine_readability_score",
        "timeliness_score",
        "spatiotemporal_capability_score",
        "combinability_score",
        "format_count_clean",
        "field_count_clean",
        "field_description_count_clean",
        "record_count_log_winsor_p99",
        "data_size_log_winsor_p99",
        "has_meaningful_time_scope",
        "has_time_field_strict",
        "has_geo_field_strict",
    ]
    existing = [col for col in candidate_cols if col in df.columns]
    cand = (
        df.loc[df["rule_dormant_candidate_v11"], existing]
        .sort_values(["DormantScore_rule_v11", "PotentialUse_main_percentile"], ascending=[False, False])
        .reset_index(drop=True)
    )
    cand.insert(0, "rule_dormant_rank", np.arange(1, len(cand) + 1))
    cand.to_csv(TABLE_DIR / "8.1_规则沉睡候选名单_完整排序.csv", index=False, encoding="utf-8-sig")
    cand.head(100).to_csv(TABLE_DIR / "8.1_规则沉睡候选名单_Top100.csv", index=False, encoding="utf-8-sig")
    cand.head(300).to_csv(TABLE_DIR / "8.1_规则沉睡候选名单_Top300.csv", index=False, encoding="utf-8-sig")
    try:
        cand.to_excel(RULE_DIR / "8.1_规则沉睡候选名单_完整排序.xlsx", index=False)
    except Exception:
        pass
    return cand


def group_candidate_profile(df: pd.DataFrame, group_col: str, name: str, top_n: int = 30) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            total_count=("数据集ID", "count"),
            candidate_count=("rule_dormant_candidate_v11", "sum"),
            avg_actual=("ActualUse_main_percentile", "mean"),
            avg_potential=("PotentialUse_main_percentile", "mean"),
            avg_dormant_score=("DormantScore_rule_v11", "mean"),
        )
        .reset_index()
    )
    grouped["candidate_rate"] = grouped["candidate_count"] / grouped["total_count"]
    grouped = grouped.sort_values(["candidate_count", "candidate_rate"], ascending=[False, False]).head(top_n)
    grouped[group_col] = grouped[group_col].fillna("缺失").astype(str)
    return save_table(grouped, name)


def plot_scatter(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 7))
    base = df.loc[~df["rule_dormant_candidate_v11"]]
    cand = df.loc[df["rule_dormant_candidate_v11"]]
    plt.scatter(
        base["ActualUse_main_percentile"],
        base["PotentialUse_main_percentile"],
        s=12,
        alpha=0.18,
        c="#7a8a99",
        label="其他主分析样本",
    )
    plt.scatter(
        cand["ActualUse_main_percentile"],
        cand["PotentialUse_main_percentile"],
        s=16,
        alpha=0.70,
        c="#d14a3a",
        label="规则沉睡候选",
    )
    xs = np.linspace(0, 0.70, 100)
    plt.plot(xs, xs + DORMANT_SCORE_THRESHOLD, color="#2b5d8a", linewidth=1.5, linestyle="--", label="DormantScore = 0.30")
    plt.axvline(ACTUAL_THRESHOLD, color="#444444", linewidth=1.2, linestyle=":")
    plt.axhline(POTENTIAL_THRESHOLD, color="#444444", linewidth=1.2, linestyle=":")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("ActualUse_main_percentile")
    plt.ylabel("PotentialUse_CRITIC_percentile")
    plt.title("8.1 ActualUse 与 PotentialUse 的规则沉睡识别")
    plt.legend(loc="lower right")
    save_fig(FIG_DIR / "8.1_ActualUse与PotentialUse规则沉睡识别.png")


def plot_score_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5.5))
    sns.histplot(df["DormantScore_rule_v11"], bins=45, color="#5975a4", kde=True)
    plt.axvline(DORMANT_SCORE_THRESHOLD, color="#c44e52", linestyle="--", linewidth=1.6, label="候选阈值 0.30")
    plt.xlabel("DormantScore_rule")
    plt.ylabel("资源数量")
    plt.title("8.1 规则沉睡度分布")
    plt.legend()
    save_fig(FIG_DIR / "8.1_规则沉睡度分布.png")


def plot_quadrants(quadrants: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5.2))
    ax = sns.barplot(data=quadrants, x="quadrant", y="count", color="#4c78a8")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    plt.xlabel("四象限类型")
    plt.ylabel("资源数量")
    plt.title("8.1 潜力-使用四象限分布")
    save_fig(FIG_DIR / "8.1_四象限分布.png")


def plot_candidate_groups(domain_profile: pd.DataFrame, type_counts: pd.DataFrame, open_counts: pd.DataFrame) -> None:
    top_domain = domain_profile.head(15).copy()
    plt.figure(figsize=(9, 6.5))
    sns.barplot(data=top_domain, y="数据领域", x="candidate_count", color="#dd8452")
    plt.xlabel("规则沉睡候选数量")
    plt.ylabel("数据领域")
    plt.title("8.1 规则沉睡候选领域分布 Top15")
    save_fig(FIG_DIR / "8.1_规则沉睡候选领域分布Top15.png")

    merged = []
    for label, data in [("数据资源类型", type_counts), ("开放属性", open_counts)]:
        temp = data.copy()
        temp["group_kind"] = label
        temp = temp.rename(columns={label: "group_value"})
        merged.append(temp[["group_kind", "group_value", "candidate_count", "candidate_rate"]])
    profile = pd.concat(merged, ignore_index=True)
    plt.figure(figsize=(9, 5.8))
    ax = sns.barplot(data=profile, x="group_value", y="candidate_rate", hue="group_kind")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    plt.xlabel("分组")
    plt.ylabel("候选率")
    plt.title("8.1 规则沉睡候选率：资源类型与开放属性")
    plt.legend(title="")
    save_fig(FIG_DIR / "8.1_资源类型与开放属性候选率.png")


def build_report(
    df: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    quadrants: pd.DataFrame,
    candidates: pd.DataFrame,
    domain_profile: pd.DataFrame,
    type_profile: pd.DataFrame,
    open_profile: pd.DataFrame,
    dept_profile: pd.DataFrame,
) -> None:
    total = len(df)
    cand_count = int(df["rule_dormant_candidate_v11"].sum())
    high_potential = int((df["PotentialUse_main_percentile"] >= POTENTIAL_THRESHOLD).sum())
    low_actual = int((df["ActualUse_main_percentile"] <= ACTUAL_THRESHOLD).sum())
    gap_count = int((df["DormantScore_rule_v11"] >= DORMANT_SCORE_THRESHOLD).sum())
    quadrant_candidate_check = int(
        ((df["dormant_type_rule_v11"] == "高潜低用") & df["rule_dormant_candidate_v11"]).sum()
    )
    existing_mismatch = (
        df["rule_dormant_candidate"].astype(str).str.lower().isin(["true", "1"])
        != df["rule_dormant_candidate_v11"]
    ).sum() if "rule_dormant_candidate" in df.columns else 0

    top_cases = candidates[
        [
            "rule_dormant_rank",
            "数据集ID",
            "数据资源名称",
            "数据领域",
            "数据资源类型",
            "开放属性",
            "ActualUse_main_percentile",
            "PotentialUse_main_percentile",
            "DormantScore_rule_v11",
        ]
    ].head(15)

    lines = [
        "# 8.1 规则沉睡度识别报告",
        "",
        "## 1. 章节定位",
        "",
        "本报告对应《公共数据资产分析方案 V1.1》的 **8.1 规则沉睡度识别**。这一节只使用第 5 章构造的 ActualUse 与第 6 章构造的 PotentialUse，不引入 ExpectedUse 模型残差。因此，本节结论是规则口径下的沉睡资产候选，后续 8.2/8.3 还需要与模型残差口径交叉验证。",
        "",
        "## 2. 识别公式与阈值",
        "",
        "```text",
        "DormantScore_rule = PotentialUse_CRITIC_percentile - ActualUse_main_percentile",
        f"rule_dormant_candidate = PotentialUse_CRITIC_percentile >= {POTENTIAL_THRESHOLD:.2f}",
        f"                         & ActualUse_main_percentile <= {ACTUAL_THRESHOLD:.2f}",
        f"                         & DormantScore_rule >= {DORMANT_SCORE_THRESHOLD:.2f}",
        "```",
        "",
        "其中，`ActualUse_main_percentile` 采用第 5 章确定的类型适配 PCA 使用强度百分位，即当前特征表中的 `ActualUse_type_percentile`；`PotentialUse_CRITIC_percentile` 采用第 6 章七维潜在价值 CRITIC 赋权后的百分位。",
        "",
        "## 3. 总体识别结果",
        "",
        f"- 主分析样本数：{total} 条。",
        f"- 高潜力资源数：{high_potential} 条，占 {high_potential / total:.2%}。",
        f"- 低实际使用资源数：{low_actual} 条，占 {low_actual / total:.2%}。",
        f"- 规则沉睡差距达到 0.30 的资源数：{gap_count} 条，占 {gap_count / total:.2%}。",
        f"- 同时满足三项规则阈值的沉睡候选：{cand_count} 条，占 {cand_count / total:.2%}。",
        f"- 候选中落在“高潜低用”四象限的数量：{quadrant_candidate_check} 条。",
        f"- 与特征表既有 `rule_dormant_candidate` 的重新计算不一致数：{int(existing_mismatch)} 条。",
        "",
        "阈值汇总如下：",
        "",
        markdown_table(threshold_summary, digits=4),
        "",
        "## 4. 潜力-使用四象限",
        "",
        markdown_table(quadrants, digits=4),
        "",
        "规则候选是“高潜低用”的更严格子集：四象限使用 0.50/0.50 切分，而规则候选进一步要求潜力达到 0.70 且沉睡差距至少为 0.30。这样可以避免把一般低使用资源误判为沉睡资产。",
        "",
        "## 5. 规则沉睡候选结构",
        "",
        "### 5.1 按数据领域",
        "",
        markdown_table(domain_profile.head(15), digits=4),
        "",
        "### 5.2 按资源类型",
        "",
        markdown_table(type_profile, digits=4),
        "",
        "### 5.3 按开放属性",
        "",
        markdown_table(open_profile, digits=4),
        "",
        "### 5.4 按提供部门 Top 20",
        "",
        markdown_table(dept_profile.head(20), digits=4),
        "",
        "## 6. Top 规则沉睡候选示例",
        "",
        markdown_table(top_cases, max_rows=15, digits=4),
        "",
        "完整候选名单见：",
        "",
        "- `outputs/rule_dormant_v11/tables/8.1_规则沉睡候选名单_完整排序.csv`",
        "- `outputs/rule_dormant_v11/8.1_规则沉睡候选名单_完整排序.xlsx`",
        "- `outputs/rule_dormant_v11/tables/8.1_规则沉睡候选名单_Top100.csv`",
        "- `outputs/rule_dormant_v11/tables/8.1_规则沉睡候选名单_Top300.csv`",
        "",
        "## 7. 图表索引",
        "",
        "- `figures/8.1_ActualUse与PotentialUse规则沉睡识别.png`",
        "- `figures/8.1_规则沉睡度分布.png`",
        "- `figures/8.1_四象限分布.png`",
        "- `figures/8.1_规则沉睡候选领域分布Top15.png`",
        "- `figures/8.1_资源类型与开放属性候选率.png`",
        "",
        "## 8. 下一步",
        "",
        "本节完成的是规则口径沉睡资产识别。按照 V1.1 方案，下一步应进入 **ExpectedUse 模型期望使用强度构造**，用 K-fold OOF 预测生成 `ExpectedUse_model_percentile`，再在 8.2 中计算模型残差沉睡度，并在 8.3 与规则候选取交集形成高置信沉睡资产名单。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    df = pd.read_csv(FEATURE_PATH, dtype={"数据集ID": str}, encoding="utf-8-sig", low_memory=False)
    required = ["ActualUse_type_percentile", "PotentialUse_CRITIC_percentile", "数据集ID"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = add_rule_fields(df)
    threshold_summary = build_threshold_summary(df)
    quadrants = build_quadrant_counts(df)
    candidates = build_candidate_ranked(df)

    domain_profile = group_candidate_profile(df, "数据领域", "8.1_规则沉睡候选_领域画像", top_n=30)
    type_profile = group_candidate_profile(df, "数据资源类型", "8.1_规则沉睡候选_资源类型画像", top_n=10)
    open_profile = group_candidate_profile(df, "开放属性", "8.1_规则沉睡候选_开放属性画像", top_n=10)
    dept_profile = group_candidate_profile(df, "数据资源提供部门", "8.1_规则沉睡候选_提供部门画像Top30", top_n=30)

    plot_scatter(df)
    plot_score_distribution(df)
    plot_quadrants(quadrants)
    plot_candidate_groups(domain_profile, type_profile, open_profile)

    save_table(safe_value_counts(df["rule_dormant_candidate_v11"], "rule_dormant_candidate"), "8.1_规则沉睡候选布尔分布")
    build_report(df, threshold_summary, quadrants, candidates, domain_profile, type_profile, open_profile, dept_profile)

    print(f"rows={len(df)}")
    print(f"rule_dormant_candidates={int(df['rule_dormant_candidate_v11'].sum())}")
    print(f"report={REPORT_PATH}")
    print(f"candidate_csv={TABLE_DIR / '8.1_规则沉睡候选名单_完整排序.csv'}")


if __name__ == "__main__":
    main()
