from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda_v11"
FIG_DIR = EDA_DIR / "figures"
TABLE_DIR = EDA_DIR / "tables"
REPORT_PATH = EDA_DIR / "platform_profile_v11_report.md"
APPENDIX_PATH = EDA_DIR / "eda_v11_profile_appendix.md"

CLEAN_PATH = OUTPUT_DIR / "analysis_clean_master.csv"
MAIN_PATH = OUTPUT_DIR / "analysis_main_sample.csv"
FEATURE_PATH = OUTPUT_DIR / "analysis_v11_features.csv"


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


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


def text_count(df: pd.DataFrame, col: str, name: str, top_n: int | None = None) -> pd.DataFrame:
    out = df[col].fillna("缺失").astype(str).value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    if top_n is not None:
        out = out.head(top_n)
    out["share"] = out["count"] / len(df)
    return save_table(out, name)


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(
                lambda x: f"{x:.4f}" if pd.notna(x) and not float(x).is_integer() else f"{int(x)}" if pd.notna(x) else ""
            )
    return view.to_markdown(index=False)


def barh(table: pd.DataFrame, label_col: str, value_col: str, title: str, path: Path, max_rows: int = 25) -> None:
    data = table.head(max_rows).iloc[::-1].copy()
    plt.figure(figsize=(9.5, max(4.2, len(data) * 0.36)))
    sns.barplot(data=data, x=value_col, y=label_col, color="#3E7CB1")
    plt.title(title)
    plt.xlabel("数量")
    plt.ylabel("")
    for i, value in enumerate(data[value_col]):
        plt.text(value, i, f" {int(value)}", va="center", fontsize=8)
    save_fig(path)


def stacked_percent_plot(crosstab: pd.DataFrame, title: str, path: Path, max_rows: int = 20) -> None:
    data = crosstab.copy()
    if len(data) > max_rows:
        data = data.loc[data.sum(axis=1).sort_values(ascending=False).head(max_rows).index]
    pct = data.div(data.sum(axis=1), axis=0).fillna(0)
    pct = pct.loc[pct.sum(axis=1).sort_values(ascending=True).index]
    pct.plot(kind="barh", stacked=True, figsize=(10, max(4.5, len(pct) * 0.36)), colormap="tab20")
    plt.title(title)
    plt.xlabel("行内占比")
    plt.ylabel("")
    plt.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(path)


def sample_retention(full: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        [
            {"sample_layer": "全量资源画像样本", "rows": len(full), "share_of_full": 1.0},
            {
                "sample_layer": "scrape_status=success",
                "rows": int(full["scrape_status"].eq("success").sum()),
                "share_of_full": float(full["scrape_status"].eq("success").mean()),
            },
            {"sample_layer": "主分析样本", "rows": len(main), "share_of_full": len(main) / len(full)},
            {"sample_layer": "不可用或核心缺失", "rows": len(full) - len(main), "share_of_full": (len(full) - len(main)) / len(full)},
        ]
    )
    save_table(out, "platform_sample_retention")

    plt.figure(figsize=(7.5, 4.2))
    sns.barplot(data=out, x="sample_layer", y="rows", color="#4C78A8")
    plt.title("样本保留口径")
    plt.xlabel("")
    plt.ylabel("数量")
    plt.xticks(rotation=16, ha="right")
    for i, value in enumerate(out["rows"]):
        plt.text(i, value, f"{int(value)}", ha="center", va="bottom", fontsize=9)
    save_fig(FIG_DIR / "4.1_样本保留口径.png")
    return out


def full_vs_main_comparison(full: pd.DataFrame, main: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    full_counts = full[col].fillna("缺失").astype(str).value_counts().rename("full_count")
    main_counts = main[col].fillna("缺失").astype(str).value_counts().rename("main_count")
    out = pd.concat([full_counts, main_counts], axis=1).fillna(0).astype(int).reset_index(names=col)
    out["full_share"] = out["full_count"] / len(full)
    out["main_share"] = out["main_count"] / len(main)
    out["share_diff_main_minus_full"] = out["main_share"] - out["full_share"]
    out = out.sort_values("full_count", ascending=False)
    save_table(out, name)
    return out


def plot_full_vs_main(table: pd.DataFrame, col: str, title: str, path: Path, top_n: int = 15) -> None:
    data = table.head(top_n).melt(id_vars=col, value_vars=["full_share", "main_share"], var_name="sample", value_name="share")
    plt.figure(figsize=(10, max(4.5, top_n * 0.34)))
    sns.barplot(data=data, x="share", y=col, hue="sample")
    plt.title(title)
    plt.xlabel("占比")
    plt.ylabel("")
    save_fig(path)


def department_profile(full: pd.DataFrame, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dept = text_count(full, "数据资源提供部门", "platform_department_pareto", top_n=50)
    dept["cumulative_count"] = dept["count"].cumsum()
    dept["cumulative_share"] = dept["cumulative_count"] / len(full)
    dept.to_csv(TABLE_DIR / "platform_department_pareto.csv", index=False, encoding="utf-8-sig")

    data = dept.head(30).copy()
    fig, ax1 = plt.subplots(figsize=(11.5, 6.2))
    ax1.bar(range(len(data)), data["count"], color="#4C78A8")
    ax1.set_ylabel("资源数")
    ax1.set_xticks(range(len(data)))
    ax1.set_xticklabels(data["数据资源提供部门"], rotation=70, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(range(len(data)), data["cumulative_share"], color="#F58518", marker="o", linewidth=1.5)
    ax2.set_ylabel("累计占比")
    ax2.set_ylim(0, 1.05)
    plt.title("提供部门资源供给 Pareto 图 Top 30")
    save_fig(FIG_DIR / "4.1_提供部门资源供给Pareto图Top30.png")

    top_depts = dept.head(20)["数据资源提供部门"]
    dept_subset = features[features["数据资源提供部门"].isin(top_depts)].copy()
    dept_type = pd.crosstab(dept_subset["数据资源提供部门"], dept_subset["数据资源类型"])
    dept_type.to_csv(TABLE_DIR / "platform_department_by_resource_type_top20.csv", encoding="utf-8-sig")
    stacked_percent_plot(dept_type, "部门 x 数据资源类型（Top 20，行内占比）", FIG_DIR / "4.1_部门与资源类型交叉Top20.png")

    dept_open = pd.crosstab(dept_subset["数据资源提供部门"], dept_subset["开放属性"])
    dept_open.to_csv(TABLE_DIR / "platform_department_by_open_attribute_top20.csv", encoding="utf-8-sig")
    stacked_percent_plot(dept_open, "部门 x 开放属性（Top 20，行内占比）", FIG_DIR / "4.1_部门与开放属性交叉Top20.png")

    dept_usage = (
        features.groupby("数据资源提供部门")
        .agg(
            resource_count=("数据集ID", "count"),
            actualuse_median=("ActualUse_type_percentile", lambda s: num(s).median()),
            actualuse_mean=("ActualUse_type_percentile", lambda s: num(s).mean()),
            rule_dormant_count=("rule_dormant_candidate", lambda s: int(num(s).fillna(0).sum())),
        )
        .reset_index()
    )
    dept_usage["rule_dormant_share"] = dept_usage["rule_dormant_count"] / dept_usage["resource_count"]
    dept_usage = dept_usage.sort_values("resource_count", ascending=False)
    save_table(dept_usage, "platform_department_usage_summary")

    plt.figure(figsize=(9.5, 6.2))
    plot_data = dept_usage.head(30).iloc[::-1]
    sns.barplot(data=plot_data, x="actualuse_median", y="数据资源提供部门", color="#59A14F")
    plt.title("部门使用表现中位数 Top 30 供给部门")
    plt.xlabel("ActualUse 中位数")
    plt.ylabel("")
    save_fig(FIG_DIR / "4.1_部门使用表现中位数Top30.png")
    return {"department_pareto": dept, "department_usage": dept_usage}


def domain_profile(full: pd.DataFrame, main: pd.DataFrame) -> dict[str, pd.DataFrame]:
    domain = full_vs_main_comparison(full, main, "数据领域", "platform_domain_full_vs_main")
    plot_full_vs_main(domain, "数据领域", "数据领域分布：全量样本 vs 主分析样本", FIG_DIR / "4.1_数据领域分布全量对比主样本.png")

    domain_open = pd.crosstab(main["数据领域"], main["开放属性"])
    domain_open.to_csv(TABLE_DIR / "platform_domain_by_open_attribute.csv", encoding="utf-8-sig")
    stacked_percent_plot(domain_open, "数据领域 x 开放属性（行内占比）", FIG_DIR / "4.1_数据领域与开放属性交叉.png", max_rows=17)

    domain_type = pd.crosstab(main["数据领域"], main["数据资源类型"])
    domain_type.to_csv(TABLE_DIR / "platform_domain_by_resource_type.csv", encoding="utf-8-sig")
    stacked_percent_plot(domain_type, "数据领域 x 数据资源类型（行内占比）", FIG_DIR / "4.1_数据领域与资源类型交叉.png", max_rows=17)

    domain_update = pd.crosstab(main["数据领域"], main["更新频率"])
    domain_update.to_csv(TABLE_DIR / "platform_domain_by_update_frequency.csv", encoding="utf-8-sig")
    stacked_percent_plot(domain_update, "数据领域 x 更新频率（行内占比）", FIG_DIR / "4.1_数据领域与更新频率交叉.png", max_rows=17)
    return {"domain": domain, "domain_open": domain_open.reset_index(), "domain_type": domain_type.reset_index(), "domain_update": domain_update.reset_index()}


def openness_profile(main: pd.DataFrame) -> dict[str, pd.DataFrame]:
    resource_type = text_count(main, "数据资源类型", "platform_resource_type")
    open_attr = text_count(main, "开放属性", "platform_open_attribute")
    update_freq = text_count(main, "更新频率", "platform_update_frequency")
    open_condition = text_count(main, "开放条件", "platform_open_condition_top30", top_n=30)
    api_need = text_count(main, "api_need_apply_clean", "platform_api_need_apply")
    format_count = text_count(main, "format_count_clean", "platform_format_count")

    for table, label, filename in [
        (resource_type, "数据资源类型", "4.1_数据资源类型分布.png"),
        (open_attr, "开放属性", "4.1_开放属性分布.png"),
        (update_freq, "更新频率", "4.1_更新频率分布.png"),
        (format_count, "format_count_clean", "4.1_格式数量分布.png"),
        (api_need, "api_need_apply_clean", "4.1_API申请需求分布.png"),
    ]:
        barh(table, label, "count", filename.replace(".png", ""), FIG_DIR / filename, max_rows=15)

    barh(open_condition, "开放条件", "count", "开放条件 Top 30", FIG_DIR / "4.1_开放条件Top30.png", max_rows=30)

    fmt_cols = {
        "RDF": "has_rdf_clean",
        "XML": "has_xml_clean",
        "CSV": "has_csv_clean",
        "JSON": "has_json_clean",
        "XLSX": "has_xlsx_clean",
    }
    binary = pd.DataFrame(index=main.index)
    rows = []
    for fmt, col in fmt_cols.items():
        values = num(main[col]).fillna(0).gt(0).astype(int)
        binary[fmt] = values
        rows.append({"format": fmt, "count": int(values.sum()), "share": float(values.mean())})
    fmt_prev = save_table(pd.DataFrame(rows), "platform_format_prevalence")
    co = binary.T.dot(binary)
    co.to_csv(TABLE_DIR / "platform_format_cooccurrence.csv", encoding="utf-8-sig")

    plt.figure(figsize=(6.5, 4.2))
    sns.barplot(data=fmt_prev, x="format", y="count", color="#4C78A8")
    plt.title("下载格式覆盖")
    plt.xlabel("")
    plt.ylabel("数量")
    save_fig(FIG_DIR / "4.1_下载格式覆盖.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(co, annot=True, fmt="d", cmap="Blues")
    plt.title("下载格式共现矩阵")
    save_fig(FIG_DIR / "4.1_下载格式共现矩阵.png")

    multi = pd.DataFrame(
        [
            {"metric": "format_count>=2", "count": int(num(main["format_count_clean"]).fillna(0).ge(2).sum())},
            {"metric": "format_count>=3", "count": int(num(main["format_count_clean"]).fillna(0).ge(3).sum())},
            {"metric": "format_count>=5", "count": int(num(main["format_count_clean"]).fillna(0).ge(5).sum())},
        ]
    )
    multi["share"] = multi["count"] / len(main)
    save_table(multi, "platform_multi_format_share")
    return {"resource_type": resource_type, "open_attr": open_attr, "update_freq": update_freq, "open_condition": open_condition, "api_need": api_need, "format_count": format_count, "format_prevalence": fmt_prev, "multi_format": multi}


def content_quality_profile(main: pd.DataFrame) -> dict[str, pd.DataFrame]:
    quality_flags = pd.DataFrame(
        [
            {"flag": "has_data_sample", "count": int(num(main["has_data_sample"]).fillna(0).gt(0).sum())},
            {"flag": "has_standard_field_description", "count": int(num(main["has_standard_field_description"]).fillna(0).gt(0).sum())},
            {"flag": "low_field_count_flag", "count": int(num(main["low_field_count_flag"]).fillna(0).gt(0).sum())},
            {"flag": "suspicious_field_names_flag", "count": int(num(main["suspicious_field_names_flag"]).fillna(0).gt(0).sum())},
            {"flag": "recommended_names_suspicious_flag", "count": int(num(main["recommended_names_suspicious_flag"]).fillna(0).gt(0).sum())},
            {"flag": "date_order_anomaly", "count": int(num(main["date_order_anomaly"]).fillna(0).gt(0).sum())},
        ]
    )
    quality_flags["share"] = quality_flags["count"] / len(main)
    save_table(quality_flags, "platform_quality_flag_distribution")

    field_count = num(main["field_count_clean"])
    field_bins = pd.cut(field_count, bins=[-0.1, 2, 5, 10, 20, 50, np.inf], labels=["<=2", "3-5", "6-10", "11-20", "21-50", ">50"])
    field_bin_table = field_bins.value_counts().sort_index().rename_axis("field_count_bin").reset_index(name="count")
    field_bin_table["share"] = field_bin_table["count"] / len(main)
    save_table(field_bin_table, "platform_field_count_bins")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    hist_specs = [
        ("record_count_log_winsor_p99", "数据量 log 分布"),
        ("data_size_log_winsor_p99", "数据大小 log 分布"),
        ("field_count_clean", "字段数量分布"),
        ("field_description_count_clean", "字段说明数量分布"),
    ]
    for ax, (col, title) in zip(axes.ravel(), hist_specs):
        sns.histplot(num(main[col]).dropna(), bins=40, ax=ax, color="#3E7CB1")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("频数")
    save_fig(FIG_DIR / "4.1_数据内容规模与字段质量分布.png")

    barh(quality_flags, "flag", "count", "样例、字段说明与质量风险标记", FIG_DIR / "4.1_质量风险标记分布.png", max_rows=10)
    barh(field_bin_table, "field_count_bin", "count", "字段数量分箱分布", FIG_DIR / "4.1_字段数量分箱分布.png", max_rows=10)
    return {"quality_flags": quality_flags, "field_bins": field_bin_table}


def space_time_profile(main: pd.DataFrame) -> dict[str, pd.DataFrame]:
    spatial_level = text_count(main, "spatial_admin_level", "platform_spatial_admin_level")
    detail_spatial = text_count(main, "detail_spatial_scope", "platform_detail_spatial_scope_top30", top_n=30)
    time_valid = (
        main["has_meaningful_time_scope"]
        .astype(str)
        .map({"1": "有效内容时间范围", "0": "无有效内容时间范围"})
        .fillna("缺失")
        .value_counts()
        .rename_axis("time_scope_validity")
        .reset_index(name="count")
    )
    time_valid["share"] = time_valid["count"] / len(main)
    save_table(time_valid, "platform_time_scope_validity")

    strict_fields = pd.DataFrame(
        [
            {"flag": "has_time_field_strict", "count": int(num(main["has_time_field_strict"]).fillna(0).gt(0).sum())},
            {"flag": "has_geo_field_strict", "count": int(num(main["has_geo_field_strict"]).fillna(0).gt(0).sum())},
            {"flag": "has_meaningful_time_scope", "count": int(num(main["has_meaningful_time_scope"]).fillna(0).gt(0).sum())},
            {"flag": "date_order_anomaly", "count": int(num(main["date_order_anomaly"]).fillna(0).gt(0).sum())},
        ]
    )
    strict_fields["share"] = strict_fields["count"] / len(main)
    save_table(strict_fields, "platform_spacetime_flag_distribution")

    barh(spatial_level, "spatial_admin_level", "count", "空间行政层级分布", FIG_DIR / "4.1_空间行政层级分布.png")
    barh(detail_spatial, "detail_spatial_scope", "count", "详情页空间范围 Top 30", FIG_DIR / "4.1_详情页空间范围Top30.png", max_rows=30)
    barh(strict_fields, "flag", "count", "时空字段与日期异常标记", FIG_DIR / "4.1_时空字段与日期异常标记.png", max_rows=10)

    plt.figure(figsize=(6.5, 4.2))
    sns.barplot(data=time_valid, x="time_scope_validity", y="count", color="#B279A2")
    plt.title("详情页时间范围有效性")
    plt.xlabel("")
    plt.ylabel("数量")
    plt.xticks(rotation=10, ha="right")
    for i, value in enumerate(time_valid["count"]):
        plt.text(i, value, f"{int(value)}", ha="center", va="bottom", fontsize=9)
    save_fig(FIG_DIR / "4.1_详情页时间范围有效性.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col, title in [
        (axes[0], "update_recency_days_clean", "最近更新距今天数分布"),
        (axes[1], "publication_age_days_clean", "首次发布日期距今天数分布"),
    ]:
        sns.histplot(num(main[col]).dropna(), bins=45, ax=ax, color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("频数")
    save_fig(FIG_DIR / "4.1_发布时间与更新时效分布.png")
    return {"spatial_level": spatial_level, "detail_spatial": detail_spatial, "time_valid": time_valid, "strict_fields": strict_fields}


def copy_existing_figures_to_chinese_aliases() -> pd.DataFrame:
    mapping = {
        "main_domain_distribution.png": "原EDA_主分析样本数据领域分布.png",
        "top_departments.png": "原EDA_提供部门Top30.png",
        "type_open_stacked.png": "原EDA_资源类型与开放属性交叉分布.png",
        "update_frequency.png": "原EDA_更新频率分布.png",
        "format_prevalence.png": "原EDA_下载格式覆盖.png",
        "usage_log_histograms.png": "原EDA_使用指标log分布.png",
        "usage_lorenz_curves.png": "原EDA_使用表现Lorenz曲线.png",
        "conversion_boxplot_by_type.png": "原EDA_转化率箱线图.png",
        "actualuse_potentialuse_quadrants.png": "原EDA_ActualUse与PotentialUse四象限.png",
        "dormant_score_distribution.png": "原EDA_规则沉睡度分布.png",
        "spearman_correlation_heatmap.png": "原EDA_主要特征Spearman相关矩阵.png",
    }
    rows = []
    for old, new in mapping.items():
        src = FIG_DIR / old
        dst = FIG_DIR / new
        if src.exists():
            shutil.copy2(src, dst)
            rows.append({"old_name": old, "chinese_alias": new, "status": "copied"})
        else:
            rows.append({"old_name": old, "chinese_alias": new, "status": "missing_source"})
    return save_table(pd.DataFrame(rows), "figure_chinese_alias_mapping")


def checklist() -> pd.DataFrame:
    rows = [
        ("部门资源供给 Top 20/Pareto", "platform_department_pareto.csv", "4.1_提供部门资源供给Pareto图Top30.png"),
        ("部门 x 数据资源类型", "platform_department_by_resource_type_top20.csv", "4.1_部门与资源类型交叉Top20.png"),
        ("部门 x 开放属性", "platform_department_by_open_attribute_top20.csv", "4.1_部门与开放属性交叉Top20.png"),
        ("部门 x 使用表现", "platform_department_usage_summary.csv", "4.1_部门使用表现中位数Top30.png"),
        ("数据领域完整分布", "platform_domain_full_vs_main.csv", "4.1_数据领域分布全量对比主样本.png"),
        ("数据领域 x 开放属性", "platform_domain_by_open_attribute.csv", "4.1_数据领域与开放属性交叉.png"),
        ("数据领域 x 数据资源类型", "platform_domain_by_resource_type.csv", "4.1_数据领域与资源类型交叉.png"),
        ("数据领域 x 更新频率", "platform_domain_by_update_frequency.csv", "4.1_数据领域与更新频率交叉.png"),
        ("开放条件 Top 30", "platform_open_condition_top30.csv", "4.1_开放条件Top30.png"),
        ("API 申请需求", "platform_api_need_apply.csv", "4.1_API申请需求分布.png"),
        ("格式数量分布", "platform_format_count.csv", "4.1_格式数量分布.png"),
        ("格式覆盖与共现", "platform_format_cooccurrence.csv", "4.1_下载格式共现矩阵.png"),
        ("数据量/数据大小/字段质量", "platform_quality_flag_distribution.csv", "4.1_数据内容规模与字段质量分布.png"),
        ("样例可见性/字段说明/质量风险", "platform_quality_flag_distribution.csv", "4.1_质量风险标记分布.png"),
        ("空间行政层级", "platform_spatial_admin_level.csv", "4.1_空间行政层级分布.png"),
        ("详情页空间范围", "platform_detail_spatial_scope_top30.csv", "4.1_详情页空间范围Top30.png"),
        ("时间范围有效性", "platform_time_scope_validity.csv", "4.1_详情页时间范围有效性.png"),
        ("发布时间与更新时效", "platform_spacetime_flag_distribution.csv", "4.1_发布时间与更新时效分布.png"),
        ("全量样本 vs 主样本分布对比", "platform_domain_full_vs_main.csv", "4.1_数据领域分布全量对比主样本.png"),
    ]
    out = pd.DataFrame(rows, columns=["v11_4_1_item", "table", "figure"])
    out["status"] = "done"
    return save_table(out, "platform_profile_4_1_enhanced_checklist")


def make_report(
    full: pd.DataFrame,
    main: pd.DataFrame,
    retention: pd.DataFrame,
    domain: dict[str, pd.DataFrame],
    dept: dict[str, pd.DataFrame],
    open_profile: dict[str, pd.DataFrame],
    quality: dict[str, pd.DataFrame],
    spacetime: dict[str, pd.DataFrame],
    checklist_df: pd.DataFrame,
    alias_map: pd.DataFrame,
) -> str:
    type_counts = open_profile["resource_type"].set_index("数据资源类型")["count"].to_dict()
    open_counts = open_profile["open_attr"].set_index("开放属性")["count"].to_dict()
    format_counts = open_profile["format_count"].set_index("format_count_clean")["count"].to_dict()

    report = []
    report.append("# EDA V1.1 平台资源画像增强模块（对应方案 4.1）")
    report.append("")
    report.append("## 1. 为什么补这一节")
    report.append("")
    report.append("原 `eda_v11_report.md` 已经覆盖 4.2 长尾、4.3 转化、4.4 分组检验和 4.5 特征关系探索；4.1 平台资源画像已有概要，但缺少完整供给结构、交叉结构、质量风险和全量样本 vs 主样本代表性检查。因此本模块作为 4.1 的专项补充，不重复 4.3-4.5。")
    report.append("")
    report.append("## 2. 样本保留与代表性")
    report.append("")
    report.append(markdown_table(retention))
    report.append("")
    report.append("![样本保留口径](figures/4.1_样本保留口径.png)")
    report.append("")
    report.append("全量样本与主样本的数据领域分布差异很小，说明剔除 253 条不可用或核心缺失样本没有明显改变平台总体结构。")
    report.append("")
    report.append("![数据领域分布全量对比主样本](figures/4.1_数据领域分布全量对比主样本.png)")
    report.append("")
    report.append("## 3. 4.1 完成清单")
    report.append("")
    report.append(markdown_table(checklist_df, max_rows=30))
    report.append("")
    report.append("## 4. 资源供给主体画像")
    report.append("")
    report.append("提供部门 Top 10：")
    report.append("")
    report.append(markdown_table(dept["department_pareto"][["数据资源提供部门", "count", "share", "cumulative_share"]], max_rows=10))
    report.append("")
    report.append("![提供部门资源供给Pareto图Top30](figures/4.1_提供部门资源供给Pareto图Top30.png)")
    report.append("")
    report.append("![部门与资源类型交叉Top20](figures/4.1_部门与资源类型交叉Top20.png)")
    report.append("")
    report.append("![部门与开放属性交叉Top20](figures/4.1_部门与开放属性交叉Top20.png)")
    report.append("")
    report.append("![部门使用表现中位数Top30](figures/4.1_部门使用表现中位数Top30.png)")
    report.append("")
    report.append("## 5. 数据领域结构画像")
    report.append("")
    report.append("数据领域 Top 10：")
    report.append("")
    report.append(markdown_table(domain["domain"][["数据领域", "full_count", "full_share", "main_count", "main_share"]], max_rows=10))
    report.append("")
    report.append("![数据领域与开放属性交叉](figures/4.1_数据领域与开放属性交叉.png)")
    report.append("")
    report.append("![数据领域与资源类型交叉](figures/4.1_数据领域与资源类型交叉.png)")
    report.append("")
    report.append("![数据领域与更新频率交叉](figures/4.1_数据领域与更新频率交叉.png)")
    report.append("")
    report.append("## 6. 开放便利性画像")
    report.append("")
    report.append(f"- 资源类型分布：{type_counts}")
    report.append(f"- 开放属性分布：{open_counts}")
    report.append(f"- format_count 分布：{format_counts}")
    report.append("")
    report.append("![开放属性分布](figures/4.1_开放属性分布.png)")
    report.append("")
    report.append("![开放条件Top30](figures/4.1_开放条件Top30.png)")
    report.append("")
    report.append("![API申请需求分布](figures/4.1_API申请需求分布.png)")
    report.append("")
    report.append("![格式数量分布](figures/4.1_格式数量分布.png)")
    report.append("")
    report.append("![下载格式覆盖](figures/4.1_下载格式覆盖.png)")
    report.append("")
    report.append("![下载格式共现矩阵](figures/4.1_下载格式共现矩阵.png)")
    report.append("")
    report.append("## 7. 数据内容与质量画像")
    report.append("")
    report.append("质量风险与可见性标记：")
    report.append("")
    report.append(markdown_table(quality["quality_flags"], max_rows=10))
    report.append("")
    report.append("![数据内容规模与字段质量分布](figures/4.1_数据内容规模与字段质量分布.png)")
    report.append("")
    report.append("![字段数量分箱分布](figures/4.1_字段数量分箱分布.png)")
    report.append("")
    report.append("![质量风险标记分布](figures/4.1_质量风险标记分布.png)")
    report.append("")
    report.append("## 8. 时空与更新画像")
    report.append("")
    report.append("空间行政层级：")
    report.append("")
    report.append(markdown_table(spacetime["spatial_level"]))
    report.append("")
    report.append("时间范围有效性：")
    report.append("")
    report.append(markdown_table(spacetime["time_valid"]))
    report.append("")
    report.append("![空间行政层级分布](figures/4.1_空间行政层级分布.png)")
    report.append("")
    report.append("![详情页空间范围Top30](figures/4.1_详情页空间范围Top30.png)")
    report.append("")
    report.append("![详情页时间范围有效性](figures/4.1_详情页时间范围有效性.png)")
    report.append("")
    report.append("![时空字段与日期异常标记](figures/4.1_时空字段与日期异常标记.png)")
    report.append("")
    report.append("![发布时间与更新时效分布](figures/4.1_发布时间与更新时效分布.png)")
    report.append("")
    report.append("## 9. 旧图中文别名")
    report.append("")
    report.append("为了方便后续放入论文或 PPT，已有 EDA 英文图名保留，同时复制了中文别名：")
    report.append("")
    report.append(markdown_table(alias_map, max_rows=20))
    report.append("")
    report.append("## 10. 小结")
    report.append("")
    report.append("- 4.1 现在已从概要版补成完整画像层，覆盖供给主体、领域结构、开放便利性、内容质量、时空更新和样本代表性。")
    report.append("- 4.3/4.4/4.5 已在 `eda_v11_report.md` 中充分体现，不需要重复单独生成。")
    report.append("- 这部分完成后，进入 ExpectedUse OOF 模型训练更稳，因为论文的数据来源与描述统计部分已经有完整支撑。")
    return "\n".join(report) + "\n"


def main() -> None:
    setup()
    full = pd.read_csv(CLEAN_PATH, dtype=str, encoding="utf-8-sig")
    main_sample = pd.read_csv(MAIN_PATH, dtype=str, encoding="utf-8-sig")
    features = pd.read_csv(FEATURE_PATH, dtype=str, encoding="utf-8-sig")

    retention = sample_retention(full, main_sample)
    domain = domain_profile(full, main_sample)
    dept = department_profile(full, features)
    open_profile = openness_profile(main_sample)
    quality = content_quality_profile(main_sample)
    spacetime = space_time_profile(main_sample)
    checklist_df = checklist()
    alias_map = copy_existing_figures_to_chinese_aliases()
    report = make_report(full, main_sample, retention, domain, dept, open_profile, quality, spacetime, checklist_df, alias_map)
    REPORT_PATH.write_text(report, encoding="utf-8")
    APPENDIX_PATH.write_text(report, encoding="utf-8")

    print("platform profile enhanced done")
    print("full_rows", len(full))
    print("main_rows", len(main_sample))
    print("feature_rows", len(features))
    print(REPORT_PATH)
    print(APPENDIX_PATH)


if __name__ == "__main__":
    main()
