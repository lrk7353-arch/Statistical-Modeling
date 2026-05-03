from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
AUDIT_DIR = OUTPUT_DIR / "expecteduse_residual_audit_v11" / "tables"
EXPECTED_DIR = OUTPUT_DIR / "expecteduse_v11" / "tables"
DORMANT_DIR = OUTPUT_DIR / "dormant_82_83_v11"
FIG_DIR = DORMANT_DIR / "figures"
TABLE_DIR = DORMANT_DIR / "tables"

INPUT_PATH = OUTPUT_DIR / "analysis_v11_expecteduse_audited.csv"
REPORT_82_PATH = DORMANT_DIR / "8.2_model_residual_dormancy_v11_report.md"
REPORT_83_PATH = DORMANT_DIR / "8.3_high_confidence_dormant_v11_report.md"

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

LAYER_DEFS = [
    ("rule_dormant_candidate", "规则沉睡候选", "主识别池 / 第9章聚类主样本"),
    ("rule_candidate_model_enhanced_top20", "模型增强高置信Top20", "高优先级分析池 / 典型案例优先池"),
    ("strict_model_residual_candidate", "严格模型残差候选", "极端模型残差案例"),
    ("residual_top10_lowuse_candidate", "低使用且模型残差Top10%", "模型残差扩展观察池"),
    ("multi_model_strict_residual_any", "任一模型严格残差", "多模型极端验证案例"),
    ("multi_model_strict_residual_2plus", "至少两个模型严格残差", "多模型一致极端案例"),
    ("multi_model_strict_residual_3plus", "至少三个模型严格残差", "强一致极端案例"),
]

PROFILE_LAYERS = [
    "rule_dormant_candidate",
    "rule_candidate_model_enhanced_top20",
    "strict_model_residual_candidate",
    "multi_model_strict_residual_any",
]


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    font_candidates = [
        Path("/mnt/c/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("/mnt/c/Windows/Fonts/Noto Sans SC (TrueType).otf"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
        Path("/mnt/c/Windows/Fonts/simsun.ttc"),
    ]
    selected_font_name = None
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            selected_font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams["font.sans-serif"] = [
        selected_font_name or "DejaVu Sans",
        "Noto Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=selected_font_name or "DejaVu Sans")


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def find_table(directory: Path, required_cols: set[str], preferred_glob: str = "*.csv") -> pd.DataFrame:
    for path in sorted(directory.glob(preferred_glob)):
        try:
            df = read_csv(path)
        except Exception:
            continue
        if required_cols.issubset(set(df.columns)):
            return df
    for path in sorted(directory.glob("*.csv")):
        try:
            df = read_csv(path)
        except Exception:
            continue
        if required_cols.issubset(set(df.columns)):
            return df
    raise FileNotFoundError(f"Cannot find table with columns {sorted(required_cols)} in {directory}")


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def bool_mask(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).eq(1)


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


def markdown_table(df: pd.DataFrame, max_rows: int = 20, digits: int = 4) -> str:
    if df.empty:
        return "_无记录。_"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return view.to_markdown(index=False)


def build_layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label, role in LAYER_DEFS:
        mask = bool_mask(df, col)
        rows.append(
            {
                "layer_col": col,
                "layer_name": label,
                "role": role,
                "count": int(mask.sum()),
                "share": float(mask.mean()),
                "avg_actual": float(num(df.loc[mask, "ActualUse_type_percentile"]).mean()) if mask.any() else None,
                "avg_potential": float(num(df.loc[mask, "PotentialUse_CRITIC_percentile"]).mean()) if mask.any() else None,
                "avg_dormant_rule": float(num(df.loc[mask, "DormantScore_rule"]).mean()) if mask.any() else None,
                "avg_expected": float(num(df.loc[mask, "ExpectedUse_model_percentile"]).mean()) if mask.any() else None,
                "avg_dormant_model": float(num(df.loc[mask, "DormantScore_model"]).mean()) if mask.any() else None,
            }
        )
    return save_table(pd.DataFrame(rows), "8.3_高置信沉睡资产分层汇总")


def group_profile(df: pd.DataFrame, layer_cols: list[str]) -> pd.DataFrame:
    group_cols = ["数据领域", "数据资源类型", "开放属性", "更新频率"]
    rows = []
    for layer in layer_cols:
        mask = bool_mask(df, layer)
        sub = df.loc[mask]
        for group_col in group_cols:
            if group_col not in sub.columns:
                continue
            counts = sub[group_col].fillna("缺失").astype(str).value_counts(dropna=False)
            total = max(1, len(sub))
            for value, count in counts.items():
                rows.append(
                    {
                        "layer_col": layer,
                        "group_col": group_col,
                        "group_value": value,
                        "count": int(count),
                        "share_in_layer": float(count / total),
                    }
                )
    return save_table(pd.DataFrame(rows), "8.3_三层候选池分组画像")


def dimension_profile(df: pd.DataFrame, layer_cols: list[str]) -> pd.DataFrame:
    rows = []
    for layer in layer_cols:
        mask = bool_mask(df, layer)
        for col in DIMENSION_COLS:
            rows.append(
                {
                    "layer_col": layer,
                    "dimension": col,
                    "dimension_cn": DIMENSION_LABELS[col],
                    "mean": float(num(df.loc[mask, col]).mean()) if mask.any() else None,
                    "median": float(num(df.loc[mask, col]).median()) if mask.any() else None,
                    "std": float(num(df.loc[mask, col]).std()) if mask.sum() > 1 else None,
                }
            )
    return save_table(pd.DataFrame(rows), "8.3_三层候选池PotentialUse七维画像")


def score_distribution_summary(df: pd.DataFrame, layer_cols: list[str]) -> pd.DataFrame:
    score_cols = [
        "ActualUse_type_percentile",
        "PotentialUse_CRITIC_percentile",
        "DormantScore_rule",
        "ExpectedUse_model_percentile",
        "DormantScore_model",
    ]
    rows = []
    for layer in layer_cols:
        mask = bool_mask(df, layer)
        for score_col in score_cols:
            s = num(df.loc[mask, score_col])
            rows.append(
                {
                    "layer_col": layer,
                    "score": score_col,
                    "count": int(s.notna().sum()),
                    "mean": float(s.mean()) if s.notna().any() else None,
                    "median": float(s.median()) if s.notna().any() else None,
                    "p25": float(s.quantile(0.25)) if s.notna().any() else None,
                    "p75": float(s.quantile(0.75)) if s.notna().any() else None,
                    "min": float(s.min()) if s.notna().any() else None,
                    "max": float(s.max()) if s.notna().any() else None,
                }
            )
    return save_table(pd.DataFrame(rows), "8.3_三层候选池核心分数分布")


def strict_case_table(df: pd.DataFrame) -> pd.DataFrame:
    no_age_cols = [
        col
        for col in df.columns
        if col.startswith("strict_model_residual_candidate_") and col.endswith("_no_age")
    ]
    mask = bool_mask(df, "strict_model_residual_candidate") | bool_mask(df, "multi_model_strict_residual_any")
    if no_age_cols:
        mask = mask | df[no_age_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).gt(0)
    cols = [
        "数据集ID",
        "数据资源名称",
        "数据领域",
        "数据资源类型",
        "开放属性",
        "ActualUse_type_percentile",
        "PotentialUse_CRITIC_percentile",
        "DormantScore_rule",
        "ExpectedUse_model_percentile",
        "DormantScore_model",
        "rule_dormant_candidate",
        "strict_model_residual_candidate",
        "multi_model_strict_residual_vote_count",
        *no_age_cols,
        "detail_url",
    ]
    existing = [c for c in cols if c in df.columns]
    table = df.loc[mask, existing].copy()
    sort_cols = [c for c in ["multi_model_strict_residual_vote_count", "DormantScore_model"] if c in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return save_table(table, "8.2_严格及no-age模型残差极端案例")


def save_candidate_pools(df: pd.DataFrame) -> None:
    base_cols = [
        "数据集ID",
        "数据资源名称",
        "数据领域",
        "数据资源提供部门",
        "数据资源类型",
        "开放属性",
        "更新频率",
        "ActualUse_type_percentile",
        "PotentialUse_CRITIC_percentile",
        "DormantScore_rule",
        "ExpectedUse_model_percentile",
        "DormantScore_model",
        *DIMENSION_COLS,
        "detail_url",
    ]
    cols = [c for c in base_cols if c in df.columns]
    pools = [
        ("rule_dormant_candidate", "8.3_规则沉睡候选主池1111"),
        ("rule_candidate_model_enhanced_top20", "8.3_模型增强高置信Top20池223"),
        ("residual_top10_lowuse_candidate", "8.2_低使用且模型残差Top10扩展观察池"),
    ]
    for flag, name in pools:
        sub = df.loc[bool_mask(df, flag), cols].copy()
        sort_cols = [c for c in ["DormantScore_rule", "DormantScore_model"] if c in sub.columns]
        if sort_cols:
            sub = sub.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        save_table(sub, name)


def plot_layer_counts(layer_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5.8))
    data = layer_summary.sort_values("count", ascending=True)
    sns.barplot(data=data, x="count", y="layer_name", color="#4c78a8")
    plt.xlabel("数量")
    plt.ylabel("候选层")
    plt.title("8.3 高置信沉睡资产分层数量")
    plt.savefig(FIG_DIR / "8.3_高置信沉睡资产分层数量.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_score_box(df: pd.DataFrame) -> None:
    names = {
        "rule_dormant_candidate": "规则候选1111",
        "rule_candidate_model_enhanced_top20": "模型增强Top20",
        "strict_model_residual_candidate": "严格残差",
    }
    rows = []
    for layer, label in names.items():
        mask = bool_mask(df, layer)
        for score in ["DormantScore_rule", "DormantScore_model"]:
            for value in num(df.loc[mask, score]).dropna():
                rows.append({"layer": label, "score": score, "value": value})
    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(9, 5.5))
    sns.boxplot(data=plot_df, x="layer", y="value", hue="score")
    plt.xlabel("候选层")
    plt.ylabel("分数")
    plt.title("8.3 三层候选池规则沉睡度与模型残差分布")
    plt.legend(title="")
    plt.savefig(FIG_DIR / "8.3_三层候选池规则沉睡度与模型残差分布.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_dimension_compare(dimension_df: pd.DataFrame) -> None:
    data = dimension_df[
        dimension_df["layer_col"].isin(["rule_dormant_candidate", "rule_candidate_model_enhanced_top20"])
    ].copy()
    plt.figure(figsize=(10, 5.8))
    sns.barplot(data=data, x="dimension_cn", y="mean", hue="layer_col")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("PotentialUse 维度")
    plt.ylabel("均值")
    plt.title("8.3 规则候选与模型增强Top20的七维潜力对比")
    plt.legend(title="")
    plt.savefig(FIG_DIR / "8.3_规则候选与模型增强Top20七维潜力对比.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_top20_domain(group_df: pd.DataFrame) -> None:
    target = group_df[
        (group_df["layer_col"].eq("rule_candidate_model_enhanced_top20"))
        & (group_df["group_col"].eq("数据领域"))
    ].head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=target.iloc[::-1], x="count", y="group_value", color="#f28e2b")
    plt.xlabel("数量")
    plt.ylabel("数据领域")
    plt.title("8.3 模型增强Top20高置信池领域分布 Top15")
    plt.savefig(FIG_DIR / "8.3_模型增强Top20高置信池领域分布Top15.png", dpi=180, bbox_inches="tight")
    plt.close()


def write_report_82(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    no_age: pd.DataFrame,
    leakage: pd.DataFrame,
    quantiles: pd.DataFrame,
    layer_summary: pd.DataFrame,
    strict_cases: pd.DataFrame,
) -> None:
    leak_count = int(pd.to_numeric(leakage["any_leakage_flag"], errors="coerce").fillna(0).sum())
    strict_count = int(bool_mask(df, "strict_model_residual_candidate").sum())
    lowuse_top10 = int(bool_mask(df, "residual_top10_lowuse_candidate").sum())
    multi_any = int(bool_mask(df, "multi_model_strict_residual_any").sum())
    multi_2 = int(bool_mask(df, "multi_model_strict_residual_2plus").sum())
    multi_3 = int(bool_mask(df, "multi_model_strict_residual_3plus").sum())
    selected_model = str(df["ExpectedUse_selected_model_name"].iloc[0])
    selected_metrics = metrics[metrics["model"].eq(selected_model)].iloc[0]
    no_age_oof = no_age[no_age["fold"].astype(str).eq("OOF")].copy().sort_values("spearman_corr", ascending=False)
    layer_view = layer_summary[
        layer_summary["layer_col"].isin(
            [
                "strict_model_residual_candidate",
                "residual_top10_lowuse_candidate",
                "multi_model_strict_residual_any",
                "multi_model_strict_residual_2plus",
                "multi_model_strict_residual_3plus",
            ]
        )
    ]
    strict_view_cols = [
        c
        for c in [
            "数据集ID",
            "数据资源名称",
            "数据领域",
            "数据资源类型",
            "ActualUse_type_percentile",
            "ExpectedUse_model_percentile",
            "DormantScore_model",
            "multi_model_strict_residual_vote_count",
        ]
        if c in strict_cases.columns
    ]

    lines = [
        "# 8.2 模型残差沉睡度识别",
        "",
        "## 8.2.1 定义",
        "",
        "模型残差沉睡度用于衡量资源实际使用表现是否低于模型根据平台经验规律给出的期望使用表现：",
        "",
        "```text",
        "DormantScore_model = ExpectedUse_model_percentile - ActualUse_type_percentile",
        "```",
        "",
        f"当前 `ExpectedUse_model` 采用第 7 章综合 OOF 表现最优的 `{selected_model}`。本节引用第 7.10 审计结论作为方法依据，但第 8.2 的任务是正式识别模型残差沉睡度；模型残差不替代第 8.1 的规则沉睡候选，而用于经验验证、置信度增强和极端案例识别。",
        "",
        "## 8.2.2 ExpectedUse full 模型表现",
        "",
        markdown_table(metrics[["model", "r2", "mae", "rmse", "spearman_corr", "model_selection_score"]].sort_values("model_selection_score"), digits=4),
        "",
        f"- `{selected_model}` OOF R2：{selected_metrics['r2']:.4f}。",
        f"- `{selected_model}` OOF Spearman：{selected_metrics['spearman_corr']:.4f}。",
        "",
        "## 8.2.3 特征泄漏检查",
        "",
        f"- ExpectedUse 特征泄漏检查命中数：{leak_count}。",
        "- 模型未直接使用浏览量、下载量、接口调用量、ActualUse、PotentialUse、DormantScore、ID、URL 或采集过程变量。",
        "",
        "## 8.2.4 时间暴露效应与 no-age 稳健性",
        "",
        "去掉 `publication_age_days_clean`、`maintenance_span_days_clean`、`update_recency_days_clean` 后，各 no-age 模型 OOF 表现如下：",
        "",
        markdown_table(no_age_oof[["model", "r2", "mae", "rmse", "spearman_corr"]], digits=4),
        "",
        "no-age 结果表明：发布时间、维护跨度和更新近度解释了大量累计使用差异，但去除这些变量后模型排序相关仍然较高，说明资源类型、开放属性、部门、字段质量、机器可读性等资源自身属性仍具有稳定解释力。",
        "",
        "## 8.2.5 模型残差分布",
        "",
        markdown_table(quantiles, max_rows=20, digits=4),
        "",
        "`DormantScore_model` 的高分位数远低于规则沉睡度的 0.30 阈值，说明严格模型残差候选天然会非常少。这个现象并不表示模型失败，而是强模型已经较好解释了 ActualUse，且 0.30 对模型残差场景非常严格。",
        "",
        "## 8.2.6 模型残差候选画像",
        "",
        markdown_table(layer_view, digits=4),
        "",
        f"- 严格模型残差候选：{strict_count} 条。",
        f"- 低使用且模型残差 Top10%：{lowuse_top10} 条。",
        f"- 任一模型严格残差候选：{multi_any} 条。",
        f"- 至少两个模型同时识别的严格残差候选：{multi_2} 条。",
        f"- 至少三个模型同时识别的严格残差候选：{multi_3} 条。",
        "",
        "因此，模型残差适合作为规则候选的置信度增强器和极端验证案例来源，而不是替代规则候选的主筛选器。",
        "",
        "## 8.2.7 极端案例",
        "",
        markdown_table(strict_cases[strict_view_cols], max_rows=15, digits=4),
        "",
        "## 8.2.8 输出文件",
        "",
        "- `tables/8.2_严格及no-age模型残差极端案例.csv`",
        "- `tables/8.2_低使用且模型残差Top10扩展观察池.csv`",
        "- `tables/8.3_高置信沉睡资产分层汇总.csv`",
        "- `tables/8.3_三层候选池分组画像.csv`",
        "",
    ]
    REPORT_82_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_report_83(
    layer_summary: pd.DataFrame,
    group_df: pd.DataFrame,
    dimension_df: pd.DataFrame,
    score_df: pd.DataFrame,
) -> None:
    def count(layer: str) -> int:
        return int(layer_summary.loc[layer_summary["layer_col"].eq(layer), "count"].iloc[0])

    rule_count = count("rule_dormant_candidate")
    top20_count = count("rule_candidate_model_enhanced_top20")
    strict_any = count("multi_model_strict_residual_any")

    rule_domain = group_df[
        (group_df["layer_col"].eq("rule_dormant_candidate"))
        & (group_df["group_col"].eq("数据领域"))
    ].head(10)
    top20_domain = group_df[
        (group_df["layer_col"].eq("rule_candidate_model_enhanced_top20"))
        & (group_df["group_col"].eq("数据领域"))
    ].head(10)
    type_open = group_df[
        group_df["layer_col"].isin(["rule_dormant_candidate", "rule_candidate_model_enhanced_top20"])
        & group_df["group_col"].isin(["数据资源类型", "开放属性"])
    ]
    score_view = score_df[
        score_df["layer_col"].isin(
            ["rule_dormant_candidate", "rule_candidate_model_enhanced_top20", "strict_model_residual_candidate"]
        )
    ]
    dimension_view = dimension_df[
        dimension_df["layer_col"].isin(["rule_dormant_candidate", "rule_candidate_model_enhanced_top20"])
    ]

    lines = [
        "# 8.3 高置信沉睡资产分层",
        "",
        "## 8.3.1 分层定义",
        "",
        "第 8.3 不再把“规则候选与严格模型残差的硬交集”作为唯一高置信定义，因为严格残差样本过少，无法支撑画像和聚类。本文采用分层高置信口径：",
        "",
        markdown_table(layer_summary[["layer_name", "layer_col", "role", "count", "share", "avg_actual", "avg_potential", "avg_dormant_rule", "avg_dormant_model"]], digits=4),
        "",
        f"- 第一层：`rule_dormant_candidate`，{rule_count} 条，是主识别池和第 9 章聚类主样本。",
        f"- 第二层：`rule_candidate_model_enhanced_top20`，{top20_count} 条，是规则候选内部模型残差最高的 20%，作为高优先级 / 高置信分析池。",
        f"- 第三层：strict / no-age / multi-model residual cases，约 1-5 条，其中任一模型严格残差为 {strict_any} 条，是极端验证案例，不作为聚类基础。",
        "",
        "## 8.3.2 三层候选池核心分数对比",
        "",
        markdown_table(score_view, max_rows=30, digits=4),
        "",
        "## 8.3.3 数据领域画像",
        "",
        "规则候选领域 Top10：",
        "",
        markdown_table(rule_domain, digits=4),
        "",
        "模型增强 Top20 高置信池领域 Top10：",
        "",
        markdown_table(top20_domain, digits=4),
        "",
        "## 8.3.4 资源类型与开放属性画像",
        "",
        markdown_table(type_open, max_rows=30, digits=4),
        "",
        "## 8.3.5 PotentialUse 七维画像",
        "",
        markdown_table(dimension_view, max_rows=20, digits=4),
        "",
        "## 8.3.6 第 9 章聚类样本池",
        "",
        "第 9 章聚类主样本明确为：",
        "",
        "```text",
        "rule_dormant_candidate == 1",
        "样本数 = 1111",
        "```",
        "",
        "223 条 `rule_candidate_model_enhanced_top20` 不作为唯一聚类样本，而作为重点案例池、类型命名时的优先解释对象和策略建议的高优先级资源集合。严格模型残差 1-5 条只用于极端验证案例或附录。",
        "",
        "## 8.3.7 输出文件",
        "",
        "- `tables/8.3_高置信沉睡资产分层汇总.csv`",
        "- `tables/8.3_规则沉睡候选主池1111.csv`",
        "- `tables/8.3_模型增强高置信Top20池223.csv`",
        "- `tables/8.3_三层候选池分组画像.csv`",
        "- `tables/8.3_三层候选池PotentialUse七维画像.csv`",
        "- `tables/8.3_三层候选池核心分数分布.csv`",
        "- `figures/8.3_高置信沉睡资产分层数量.png`",
        "- `figures/8.3_三层候选池规则沉睡度与模型残差分布.png`",
        "- `figures/8.3_规则候选与模型增强Top20七维潜力对比.png`",
        "- `figures/8.3_模型增强Top20高置信池领域分布Top15.png`",
        "",
    ]
    REPORT_83_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    df = read_csv(INPUT_PATH, dtype={"数据集ID": str})
    metrics = find_table(EXPECTED_DIR, {"model", "r2", "mae", "rmse", "spearman_corr", "model_selection_score"})
    no_age = find_table(AUDIT_DIR, {"model", "fold", "r2", "mae", "rmse", "spearman_corr"})
    leakage = find_table(AUDIT_DIR, {"feature", "any_leakage_flag"})
    quantiles = find_table(AUDIT_DIR, {"quantile", "DormantScore_model"})

    layer_summary = build_layer_summary(df)
    group_df = group_profile(df, [x[0] for x in LAYER_DEFS])
    dimension_df = dimension_profile(df, PROFILE_LAYERS)
    score_df = score_distribution_summary(df, PROFILE_LAYERS)
    strict_cases = strict_case_table(df)
    save_candidate_pools(df)

    plot_layer_counts(layer_summary)
    plot_score_box(df)
    plot_dimension_compare(dimension_df)
    plot_top20_domain(group_df)

    write_report_82(df, metrics, no_age, leakage, quantiles, layer_summary, strict_cases)
    write_report_83(layer_summary, group_df, dimension_df, score_df)

    print(f"rows={len(df)}")
    print(f"rule_dormant_candidate={int(bool_mask(df, 'rule_dormant_candidate').sum())}")
    print(f"rule_candidate_model_enhanced_top20={int(bool_mask(df, 'rule_candidate_model_enhanced_top20').sum())}")
    print(f"strict_model_residual_candidate={int(bool_mask(df, 'strict_model_residual_candidate').sum())}")
    print(f"multi_model_strict_residual_any={int(bool_mask(df, 'multi_model_strict_residual_any').sum())}")
    print(REPORT_82_PATH)
    print(REPORT_83_PATH)


if __name__ == "__main__":
    main()
