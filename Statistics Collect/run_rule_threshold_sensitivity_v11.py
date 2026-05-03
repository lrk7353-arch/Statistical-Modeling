from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RULE_DIR = OUTPUT_DIR / "rule_dormant_v11"
SENS_DIR = RULE_DIR / "threshold_sensitivity"
FIG_DIR = SENS_DIR / "figures"
TABLE_DIR = SENS_DIR / "tables"

FEATURE_PATH = OUTPUT_DIR / "analysis_v11_features.csv"
REPORT_PATH = SENS_DIR / "8.1_阈值敏感性分析报告.md"

BASELINE = {
    "potential_cut": 0.70,
    "actual_cut": 0.50,
    "gap_cut": 0.30,
}

THRESHOLD_SETS = {
    "三点阈值组_0.20_0.50_0.80": [0.20, 0.50, 0.80],
    "四分位阈值组_0.25_0.50_0.75": [0.25, 0.50, 0.75],
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


def candidate_mask(df: pd.DataFrame, potential_cut: float, actual_cut: float, gap_cut: float) -> pd.Series:
    potential = num(df["PotentialUse_CRITIC_percentile"])
    actual = num(df["ActualUse_type_percentile"])
    gap = potential - actual
    return (potential >= potential_cut) & (actual <= actual_cut) & (gap >= gap_cut)


def pct_rank(series: pd.Series) -> pd.Series:
    values = num(series)
    n = values.notna().sum()
    if n <= 1:
        return pd.Series(0.0, index=series.index)
    return (values.rank(method="average", na_option="keep") - 1) / (n - 1)


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


def markdown_table(df: pd.DataFrame, max_rows: int = 20, digits: int = 4) -> str:
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return view.to_markdown(index=False)


def build_sensitivity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_col = "数据集ID"
    baseline_mask = candidate_mask(df, **BASELINE)
    baseline_ids = set(df.loc[baseline_mask, id_col].astype(str))
    rows = []
    memberships = pd.DataFrame({id_col: df[id_col].astype(str)})
    memberships["baseline_candidate"] = baseline_mask.astype(int)

    for set_name, values in THRESHOLD_SETS.items():
        for potential_cut in values:
            for actual_cut in values:
                for gap_cut in values:
                    mask = candidate_mask(df, potential_cut, actual_cut, gap_cut)
                    ids = set(df.loc[mask, id_col].astype(str))
                    key = f"{set_name}|P{potential_cut:.2f}|A{actual_cut:.2f}|G{gap_cut:.2f}"
                    memberships[key] = mask.astype(int)
                    intersection = len(ids & baseline_ids)
                    union = len(ids | baseline_ids)
                    rows.append(
                        {
                            "threshold_set": set_name,
                            "potential_cut": potential_cut,
                            "actual_cut": actual_cut,
                            "gap_cut": gap_cut,
                            "candidate_count": int(mask.sum()),
                            "candidate_share": float(mask.mean()),
                            "overlap_with_baseline_count": intersection,
                            "overlap_with_baseline_share": intersection / max(1, len(baseline_ids)),
                            "jaccard_with_baseline": intersection / union if union else 0.0,
                            "is_baseline_rule": (
                                potential_cut == BASELINE["potential_cut"]
                                and actual_cut == BASELINE["actual_cut"]
                                and gap_cut == BASELINE["gap_cut"]
                            ),
                        }
                    )

    detail = pd.DataFrame(rows).sort_values(
        ["threshold_set", "potential_cut", "actual_cut", "gap_cut"]
    )
    scenario_cols = [col for col in memberships.columns if "|P" in col]
    memberships["sensitivity_selected_count"] = memberships[scenario_cols].sum(axis=1)
    memberships["sensitivity_selected_share"] = memberships["sensitivity_selected_count"] / len(scenario_cols)

    robust = df[[id_col, "数据资源名称", "数据领域", "数据资源类型", "开放属性"]].copy()
    robust[id_col] = robust[id_col].astype(str)
    robust = robust.merge(memberships[[id_col, "baseline_candidate", "sensitivity_selected_count", "sensitivity_selected_share"]], on=id_col)
    robust["PotentialUse_CRITIC_percentile"] = num(df["PotentialUse_CRITIC_percentile"])
    robust["ActualUse_type_percentile"] = num(df["ActualUse_type_percentile"])
    robust["DormantScore_rule"] = robust["PotentialUse_CRITIC_percentile"] - robust["ActualUse_type_percentile"]
    robust["robustness_rank"] = (
        robust["sensitivity_selected_share"].rank(ascending=False, method="min")
        + (1 - pct_rank(robust["DormantScore_rule"]))
    )
    robust = robust.sort_values(
        ["sensitivity_selected_share", "DormantScore_rule", "PotentialUse_CRITIC_percentile"],
        ascending=[False, False, False],
    )

    return save_table(detail, "8.1_阈值敏感性_组合明细"), memberships, save_table(robust, "8.1_阈值敏感性_样本入选频率")


def plot_heatmaps(detail: pd.DataFrame) -> None:
    for set_name in detail["threshold_set"].unique():
        sub = detail[(detail["threshold_set"] == set_name) & (detail["actual_cut"] == 0.50)]
        count_pivot = sub.pivot(index="potential_cut", columns="gap_cut", values="candidate_count")
        overlap_pivot = sub.pivot(index="potential_cut", columns="gap_cut", values="overlap_with_baseline_share")

        plt.figure(figsize=(7.2, 5.2))
        sns.heatmap(count_pivot, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "候选数"})
        plt.title(f"8.1 阈值敏感性候选数：{set_name}，ActualUse<=0.50")
        plt.xlabel("DormantScore 阈值")
        plt.ylabel("PotentialUse 阈值")
        plt.savefig(FIG_DIR / f"8.1_阈值敏感性候选数_{set_name}_ActualUse050.png", dpi=180, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(7.2, 5.2))
        sns.heatmap(overlap_pivot, annot=True, fmt=".1%", cmap="Blues", cbar_kws={"label": "与基准候选重合率"})
        plt.title(f"8.1 与基准候选重合率：{set_name}，ActualUse<=0.50")
        plt.xlabel("DormantScore 阈值")
        plt.ylabel("PotentialUse 阈值")
        plt.savefig(FIG_DIR / f"8.1_阈值敏感性重合率_{set_name}_ActualUse050.png", dpi=180, bbox_inches="tight")
        plt.close()


def plot_candidate_count(detail: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    plot_data = detail.copy()
    plot_data["threshold_label"] = (
        "P>=" + plot_data["potential_cut"].map(lambda x: f"{x:.2f}")
        + ", A<=" + plot_data["actual_cut"].map(lambda x: f"{x:.2f}")
        + ", G>=" + plot_data["gap_cut"].map(lambda x: f"{x:.2f}")
    )
    plot_data = plot_data.sort_values("candidate_count", ascending=False)
    sns.scatterplot(
        data=plot_data,
        x="candidate_count",
        y="overlap_with_baseline_share",
        hue="threshold_set",
        size="gap_cut",
        sizes=(40, 180),
        alpha=0.8,
    )
    plt.axvline(1111, color="#c44e52", linestyle="--", linewidth=1.2, label="基准候选数 1111")
    plt.xlabel("候选数量")
    plt.ylabel("与基准候选重合率")
    plt.title("8.1 阈值组合敏感性：候选数量与基准重合")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.savefig(FIG_DIR / "8.1_阈值敏感性_候选数量与重合率.png", dpi=180, bbox_inches="tight")
    plt.close()


def write_report(detail: pd.DataFrame, robust: pd.DataFrame) -> None:
    baseline_rows = detail[
        (detail["potential_cut"] == 0.70)
        & (detail["actual_cut"] == 0.50)
        & (detail["gap_cut"] == 0.30)
    ]
    baseline_count = 1111
    focused = detail[
        (detail["actual_cut"] == 0.50)
        & (
            ((detail["potential_cut"].isin([0.20, 0.50, 0.80])) & (detail["gap_cut"].isin([0.20, 0.50, 0.80])))
            | ((detail["potential_cut"].isin([0.25, 0.50, 0.75])) & (detail["gap_cut"].isin([0.25, 0.50, 0.75])))
        )
    ].sort_values(["threshold_set", "potential_cut", "gap_cut"])

    summary = (
        detail.groupby("threshold_set")
        .agg(
            scenario_count=("candidate_count", "count"),
            candidate_min=("candidate_count", "min"),
            candidate_median=("candidate_count", "median"),
            candidate_max=("candidate_count", "max"),
            overlap_median=("overlap_with_baseline_share", "median"),
            jaccard_median=("jaccard_with_baseline", "median"),
        )
        .reset_index()
    )

    robust_baseline = robust[robust["baseline_candidate"].eq(1)].copy()
    robust_high = robust_baseline[robust_baseline["sensitivity_selected_share"] >= 0.50]
    robust_very_high = robust_baseline[robust_baseline["sensitivity_selected_share"] >= 0.75]
    robust_high.to_csv(TABLE_DIR / "8.1_基准候选中敏感性入选频率不低于50pct.csv", index=False, encoding="utf-8-sig")
    robust_very_high.to_csv(TABLE_DIR / "8.1_基准候选中敏感性入选频率不低于75pct.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 8.1 规则沉睡度阈值敏感性分析",
        "",
        "## 1. 分析目的",
        "",
        "本分析用于检验 `0.70/0.50/0.30` 规则阈值是否过度依赖人工设定。基准规则保持不变：",
        "",
        "```text",
        "PotentialUse_CRITIC_percentile >= 0.70",
        "ActualUse_type_percentile <= 0.50",
        "DormantScore_rule >= 0.30",
        "```",
        "",
        "在此基础上，按用户指定的两组阈值集合进行组合敏感性分析：",
        "",
        "- 三点阈值组：`0.20 / 0.50 / 0.80`",
        "- 四分位阈值组：`0.25 / 0.50 / 0.75`",
        "",
        "注意：这些组合是敏感性检验，不替代基准规则口径。特别是 `PotentialUse >= 0.20` 或 `ActualUse <= 0.80` 一类组合非常宽松，只用于观察边界变化。",
        "",
        "## 2. 阈值组总体结果",
        "",
        markdown_table(summary, digits=4),
        "",
        "## 3. ActualUse 固定为 0.50 时的重点比较",
        "",
        "为了和基准规则保持可比性，下表重点展示 `ActualUse <= 0.50` 时，不同 PotentialUse 与 DormantScore 阈值下的结果。",
        "",
        markdown_table(focused[[
            "threshold_set",
            "potential_cut",
            "actual_cut",
            "gap_cut",
            "candidate_count",
            "candidate_share",
            "overlap_with_baseline_share",
            "jaccard_with_baseline",
        ]], max_rows=30, digits=4),
        "",
        "## 4. 基准候选的稳定性",
        "",
        f"- 基准规则候选数：{baseline_count} 条。",
        f"- 在全部敏感性组合中，基准候选里入选频率不低于 50% 的样本：{len(robust_high)} 条。",
        f"- 在全部敏感性组合中，基准候选里入选频率不低于 75% 的样本：{len(robust_very_high)} 条。",
        "",
        "这说明 `0.70/0.50/0.30` 的 1111 条候选中，有一部分属于对阈值变化较敏感的边界样本；后续 8.2/8.3 应继续用 ExpectedUse OOF 模型残差交叉验证，而不是把 1111 条直接作为最终沉睡资产。",
        "",
        "## 5. 输出文件",
        "",
        "- `tables/8.1_阈值敏感性_组合明细.csv`",
        "- `tables/8.1_阈值敏感性_样本入选频率.csv`",
        "- `figures/8.1_阈值敏感性_候选数量与重合率.png`",
        "- `figures/8.1_阈值敏感性候选数_三点阈值组_0.20_0.50_0.80_ActualUse050.png`",
        "- `figures/8.1_阈值敏感性重合率_三点阈值组_0.20_0.50_0.80_ActualUse050.png`",
        "- `figures/8.1_阈值敏感性候选数_四分位阈值组_0.25_0.50_0.75_ActualUse050.png`",
        "- `figures/8.1_阈值敏感性重合率_四分位阈值组_0.25_0.50_0.75_ActualUse050.png`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    df = pd.read_csv(FEATURE_PATH, dtype={"数据集ID": str}, encoding="utf-8-sig", low_memory=False)
    detail, memberships, robust = build_sensitivity(df)
    memberships.to_csv(TABLE_DIR / "8.1_阈值敏感性_全场景入选矩阵.csv", index=False, encoding="utf-8-sig")
    plot_heatmaps(detail)
    plot_candidate_count(detail)
    write_report(detail, robust)
    print(f"scenarios={len(detail)}")
    print(f"report={REPORT_PATH}")
    print(f"tables={TABLE_DIR}")
    print(f"figures={FIG_DIR}")


if __name__ == "__main__":
    main()
