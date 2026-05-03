from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
AUDIT_DIR = OUTPUT_DIR / "expecteduse_residual_audit_v11"
FIG_DIR = AUDIT_DIR / "figures"
TABLE_DIR = AUDIT_DIR / "tables"

INPUT_PATH = OUTPUT_DIR / "analysis_v11_expecteduse.csv"
AUDITED_CSV_PATH = OUTPUT_DIR / "analysis_v11_expecteduse_audited.csv"
AUDITED_XLSX_PATH = OUTPUT_DIR / "analysis_v11_expecteduse_audited.xlsx"
REPORT_PATH = AUDIT_DIR / "expecteduse_residual_audit_v11_report.md"

TARGET_COL = "ActualUse_type_percentile"
RANDOM_STATE = 20260503
N_SPLITS = 5

TIME_EXPOSURE_FEATURES = [
    "publication_age_days_clean",
    "maintenance_span_days_clean",
    "update_recency_days_clean",
]

CATEGORICAL_FEATURES = [
    "数据资源提供部门",
    "数据领域",
    "开放条件",
    "数据资源类型",
    "数据资源状态",
    "更新频率",
    "开放属性",
    "type_code",
    "spatial_admin_level",
    "detail_spatial_scope",
    "detail_time_scope",
]

NUMERIC_FEATURES = [
    "title_len",
    "description_len",
    "description_has_field_hint",
    "scene_keyword_count",
    "publication_age_days_clean",
    "update_recency_days_clean",
    "maintenance_span_days_clean",
    "date_order_anomaly",
    "record_count_log_winsor_p99",
    "data_size_log_winsor_p99",
    "field_count_clean",
    "field_description_count_clean",
    "low_field_count_flag",
    "suspicious_field_names_flag",
    "has_time_field_strict",
    "has_geo_field_strict",
    "format_count_clean",
    "has_rdf_clean",
    "has_xml_clean",
    "has_csv_clean",
    "has_json_clean",
    "has_xlsx_clean",
    "api_need_apply_clean",
    "recommended_dataset_count_clean",
    "recommended_names_suspicious_flag",
    "has_meaningful_time_scope",
    "has_standard_field_description",
    "has_data_sample",
    "is_data_interface",
    "is_data_product",
    "is_conditional_open",
    "is_unconditional_open",
]

LEAKAGE_RULES = {
    "使用量相关": ["浏览量", "下载量", "接口调用量", "view_count", "download_count", "api_call_count", "log_view", "log_download", "log_api"],
    "ActualUse相关": ["ActualUse", "view_score", "download_score", "api_call_score"],
    "PotentialUse相关": ["PotentialUse", "topic_public_value_score", "semantic_clarity_score", "data_richness_score", "machine_readability_score", "timeliness_score", "spatiotemporal_capability_score", "combinability_score"],
    "DormantScore相关": ["DormantScore", "dormant", "candidate", "HighConfidence"],
    "ID_URL相关": ["数据集ID", "detail_url"],
    "采集过程变量": ["scrape_status", "scrape_error", "scraped_at", "web_overlay_source"],
    "评分评论和污染推荐": ["rating_score", "comment_count", "recommended_dataset_names"],
}

MODEL_RAW_COLS = {
    "XGBoost": "ExpectedUse_XGBoost_oof_raw",
    "LightGBM": "ExpectedUse_LightGBM_oof_raw",
    "HistGradientBoosting": "ExpectedUse_HistGradientBoosting_oof_raw",
    "RandomForest": "ExpectedUse_RandomForest_oof_raw",
    "CatBoostRegressor": "ExpectedUse_catboost_oof_raw",
    "Ridge": "ExpectedUse_Ridge_oof_raw",
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


def percentile_01(series: pd.Series) -> pd.Series:
    values = num(series)
    n = int(values.notna().sum())
    if n <= 1:
        return pd.Series(np.nan, index=series.index)
    return (values.rank(method="average", na_option="keep") - 1) / (n - 1)


def metric_dict(y_true: np.ndarray, pred: np.ndarray, model_name: str, fold: str | int = "OOF") -> dict[str, object]:
    mask = np.isfinite(y_true) & np.isfinite(pred)
    yt = y_true[mask]
    yp = pred[mask]
    spearman = stats.spearmanr(yt, yp)
    pearson = stats.pearsonr(yt, yp)
    return {
        "model": model_name,
        "fold": fold,
        "n": int(mask.sum()),
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(mean_squared_error(yt, yp) ** 0.5),
        "spearman_corr": float(spearman.correlation),
        "pearson_corr": float(pearson.statistic),
    }


def save_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    return df


def markdown_table(df: pd.DataFrame, max_rows: int = 20, digits: int = 4) -> str:
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return view.to_markdown(index=False)


def leakage_check(used_features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in used_features:
        row = {"feature": feature}
        any_flag = 0
        lower_feature = feature.lower()
        for rule_name, patterns in LEAKAGE_RULES.items():
            flag = int(any(pattern.lower() in lower_feature for pattern in patterns))
            row[rule_name] = flag
            any_flag = max(any_flag, flag)
        row["any_leakage_flag"] = any_flag
        rows.append(row)
    return save_table(pd.DataFrame(rows), "7.10_模型特征泄漏检查")


def prepare_frame(df: pd.DataFrame, cat_features: list[str], numeric_features: list[str]) -> pd.DataFrame:
    x = df[cat_features + numeric_features].copy()
    for col in cat_features:
        x[col] = x[col].fillna("缺失").astype(str)
        x[col] = x[col].mask(x[col].str.strip().eq(""), "缺失")
    for col in numeric_features:
        x[col] = num(x[col])
    return x


def oof_no_age_models(
    df: pd.DataFrame, cat_features: list[str], numeric_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_no_age = [col for col in numeric_features if col not in TIME_EXPOSURE_FEATURES]
    x_raw = prepare_frame(df, cat_features, numeric_no_age)
    y = num(df[TARGET_COL]).to_numpy(dtype=float)
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_no_age),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), cat_features),
        ],
        remainder="drop",
    )
    models = {
        "XGBoost_no_age": Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=650,
                        learning_rate=0.035,
                        max_depth=5,
                        min_child_weight=8,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.02,
                        reg_lambda=1.0,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=0,
                    ),
                ),
            ]
        ),
        "LightGBM_no_age": Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=650,
                        learning_rate=0.035,
                        num_leaves=31,
                        min_child_samples=25,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.05,
                        reg_lambda=0.25,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting_no_age": Pipeline(
            [
                ("pre", pre),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=350,
                        learning_rate=0.045,
                        l2_regularization=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pred_df = pd.DataFrame({"数据集ID": df["数据集ID"].astype(str)})
    fold_rows = []
    for model_name, pipe in models.items():
        pred = np.full(len(df), np.nan, dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(folds.split(x_raw), start=1):
            pipe.fit(x_raw.iloc[train_idx], y[train_idx])
            pred[valid_idx] = pipe.predict(x_raw.iloc[valid_idx])
            fold_rows.append(metric_dict(y[valid_idx], pred[valid_idx], model_name, fold))
        fold_rows.append(metric_dict(y, pred, model_name, "OOF"))
        pred_df[f"ExpectedUse_{model_name}_oof_raw"] = pred

    # CatBoost handles categorical features directly, so it is run outside the one-hot pipeline.
    cat_x = prepare_frame(df, cat_features, numeric_no_age)
    cat_indices = [cat_x.columns.get_loc(col) for col in cat_features]
    cat_pred = np.full(len(df), np.nan, dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(folds.split(cat_x), start=1):
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=900,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=8,
            random_seed=RANDOM_STATE,
            bootstrap_type="Bayesian",
            bagging_temperature=0.8,
            od_type="Iter",
            od_wait=80,
            allow_writing_files=False,
            verbose=False,
            thread_count=-1,
        )
        train_pool = Pool(cat_x.iloc[train_idx], y[train_idx], cat_features=cat_indices)
        valid_pool = Pool(cat_x.iloc[valid_idx], y[valid_idx], cat_features=cat_indices)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        cat_pred[valid_idx] = model.predict(valid_pool)
        fold_rows.append(metric_dict(y[valid_idx], cat_pred[valid_idx], "CatBoost_no_age", fold))
    fold_rows.append(metric_dict(y, cat_pred, "CatBoost_no_age", "OOF"))
    pred_df["ExpectedUse_CatBoost_no_age_oof_raw"] = cat_pred

    return pred_df, save_table(pd.DataFrame(fold_rows), "7.10_no_age模型OOF指标")


def add_model_residuals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for model, raw_col in MODEL_RAW_COLS.items():
        if raw_col not in out.columns:
            continue
        pct_col = f"ExpectedUse_{model}_percentile"
        score_col = f"DormantScore_model_{model}"
        strict_col = f"strict_model_residual_candidate_{model}"
        out[pct_col] = percentile_01(out[raw_col])
        out[score_col] = out[pct_col] - num(out[TARGET_COL])
        out[strict_col] = (
            (out[pct_col] >= 0.70)
            & (num(out[TARGET_COL]) <= 0.50)
            & (out[score_col] >= 0.30)
        ).astype(int)
    return out


def add_no_age_outputs(df: pd.DataFrame, no_age_predictions: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.merge(no_age_predictions, on="数据集ID", how="left")
    no_age_models = ["XGBoost_no_age", "LightGBM_no_age", "HistGradientBoosting_no_age", "CatBoost_no_age"]
    for model in no_age_models:
        raw_col = f"ExpectedUse_{model}_oof_raw"
        pct_col = f"ExpectedUse_{model}_percentile"
        score_col = f"DormantScore_model_{model}"
        strict_col = f"strict_model_residual_candidate_{model}"
        out[pct_col] = percentile_01(out[raw_col])
        out[score_col] = out[pct_col] - num(out[TARGET_COL])
        out[strict_col] = (
            (out[pct_col] >= 0.70)
            & (num(out[TARGET_COL]) <= 0.50)
            & (out[score_col] >= 0.30)
        ).astype(int)
    return out


def residual_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    score = num(df["DormantScore_model"])
    rows = []
    for q in [0, 0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99, 1.0]:
        rows.append({"quantile": q, "DormantScore_model": float(score.quantile(q))})
    return save_table(pd.DataFrame(rows), "7.10_DormantScore_model分位数")


def add_candidate_layers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    actual = num(out[TARGET_COL])
    score = num(out["DormantScore_model"])
    rule = pd.to_numeric(out.get("rule_dormant_candidate", 0), errors="coerce").fillna(0).astype(int).eq(1)
    low_use = actual <= 0.50

    top5_cut = score.quantile(0.95)
    top10_cut = score.quantile(0.90)
    top20_cut = score.quantile(0.80)
    top30_cut = score.quantile(0.70)
    rule_score = score[rule]
    rule_top10_cut = rule_score.quantile(0.90)
    rule_top20_cut = rule_score.quantile(0.80)
    rule_top30_cut = rule_score.quantile(0.70)

    out["strict_model_residual_candidate"] = (
        (num(out["ExpectedUse_model_percentile"]) >= 0.70) & low_use & (score >= 0.30)
    ).astype(int)
    out["residual_top5_lowuse_candidate"] = (low_use & (score >= top5_cut)).astype(int)
    out["residual_top10_lowuse_candidate"] = (low_use & (score >= top10_cut)).astype(int)
    out["residual_top20_lowuse_candidate"] = (low_use & (score >= top20_cut)).astype(int)
    out["residual_top30_lowuse_candidate"] = (low_use & (score >= top30_cut)).astype(int)
    out["rule_candidate_model_enhanced_top10"] = (rule & (score >= rule_top10_cut)).astype(int)
    out["rule_candidate_model_enhanced_top20"] = (rule & (score >= rule_top20_cut)).astype(int)
    out["rule_candidate_model_enhanced_top30"] = (rule & (score >= rule_top30_cut)).astype(int)

    strict_cols = [col for col in out.columns if col.startswith("strict_model_residual_candidate_")]
    out["multi_model_strict_residual_vote_count"] = out[strict_cols].sum(axis=1)
    out["multi_model_strict_residual_any"] = (out["multi_model_strict_residual_vote_count"] >= 1).astype(int)
    out["multi_model_strict_residual_2plus"] = (out["multi_model_strict_residual_vote_count"] >= 2).astype(int)
    out["multi_model_strict_residual_3plus"] = (out["multi_model_strict_residual_vote_count"] >= 3).astype(int)

    layer_defs = [
        ("strict_model_residual_candidate", "严格模型残差候选：ExpectedUse>=0.70, ActualUse<=0.50, DormantScore_model>=0.30"),
        ("residual_top5_lowuse_candidate", "低使用样本中，全样本模型残差 Top 5%"),
        ("residual_top10_lowuse_candidate", "低使用样本中，全样本模型残差 Top 10%"),
        ("residual_top20_lowuse_candidate", "低使用样本中，全样本模型残差 Top 20%"),
        ("residual_top30_lowuse_candidate", "低使用样本中，全样本模型残差 Top 30%"),
        ("rule_candidate_model_enhanced_top10", "规则候选内部模型残差 Top 10%"),
        ("rule_candidate_model_enhanced_top20", "规则候选内部模型残差 Top 20%"),
        ("rule_candidate_model_enhanced_top30", "规则候选内部模型残差 Top 30%"),
        ("multi_model_strict_residual_any", "任一模型严格残差候选"),
        ("multi_model_strict_residual_2plus", "至少两个模型同时识别的严格残差候选"),
        ("multi_model_strict_residual_3plus", "至少三个模型同时识别的严格残差候选"),
    ]
    rows = []
    ids_rule = set(out.loc[rule, "数据集ID"].astype(str))
    for col, definition in layer_defs:
        mask = out[col].astype(int).eq(1)
        ids = set(out.loc[mask, "数据集ID"].astype(str))
        intersection = len(ids & ids_rule)
        union = len(ids | ids_rule)
        rows.append(
            {
                "candidate_layer": col,
                "definition": definition,
                "count": int(mask.sum()),
                "share": float(mask.mean()),
                "intersection_with_rule": intersection,
                "jaccard_with_rule": intersection / union if union else 0.0,
            }
        )
    layer_summary = save_table(pd.DataFrame(rows), "7.10_残差与高置信候选分层汇总")
    return out, layer_summary


def group_profiles(df: pd.DataFrame, candidate_cols: list[str]) -> None:
    rows = []
    for col in candidate_cols:
        mask = df[col].astype(int).eq(1)
        sub = df.loc[mask]
        for group_col in ["数据领域", "数据资源类型", "开放属性"]:
            counts = sub[group_col].fillna("缺失").astype(str).value_counts().head(20)
            for value, count in counts.items():
                rows.append({"candidate_layer": col, "group_col": group_col, "group_value": value, "count": int(count)})
    save_table(pd.DataFrame(rows), "7.10_候选分层画像_领域类型开放属性")


def plot_outputs(df: pd.DataFrame, layer_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(8.5, 5.2))
    sns.histplot(num(df["DormantScore_model"]), bins=55, color="#4c78a8", kde=True)
    for q, label in [(0.80, "Top20%"), (0.90, "Top10%"), (0.95, "Top5%")]:
        plt.axvline(num(df["DormantScore_model"]).quantile(q), linestyle="--", linewidth=1.2, label=label)
    plt.xlabel("DormantScore_model")
    plt.ylabel("资源数量")
    plt.title("7.10 模型残差沉睡度分布与分位阈值")
    plt.legend()
    plt.savefig(FIG_DIR / "7.10_模型残差分布与分位阈值.png", dpi=180, bbox_inches="tight")
    plt.close()

    show = layer_summary.sort_values("count", ascending=False).copy()
    plt.figure(figsize=(10, 6.5))
    sns.barplot(data=show, y="candidate_layer", x="count", color="#59a14f")
    plt.xlabel("候选数量")
    plt.ylabel("候选层")
    plt.title("7.10 残差与高置信候选分层数量")
    plt.savefig(FIG_DIR / "7.10_残差与高置信候选分层数量.png", dpi=180, bbox_inches="tight")
    plt.close()

    vote_counts = df["multi_model_strict_residual_vote_count"].value_counts().sort_index().reset_index()
    vote_counts.columns = ["vote_count", "resource_count"]
    plt.figure(figsize=(7.5, 5))
    sns.barplot(data=vote_counts, x="vote_count", y="resource_count", color="#f28e2b")
    plt.xlabel("严格残差模型投票数")
    plt.ylabel("资源数量")
    plt.title("7.10 多模型严格残差一致性")
    plt.savefig(FIG_DIR / "7.10_多模型严格残差一致性.png", dpi=180, bbox_inches="tight")
    plt.close()


def write_report(
    df: pd.DataFrame,
    leakage: pd.DataFrame,
    no_age_metrics: pd.DataFrame,
    quantiles: pd.DataFrame,
    layer_summary: pd.DataFrame,
) -> None:
    leak_count = int(leakage["any_leakage_flag"].sum())
    no_age_oof = no_age_metrics[no_age_metrics["fold"].astype(str).eq("OOF")].copy()
    full_metrics = pd.read_csv(OUTPUT_DIR / "expecteduse_v11" / "tables" / "7_ExpectedUse_模型性能汇总.csv", encoding="utf-8-sig")
    xgb_full = full_metrics[full_metrics["model"].eq("XGBoost")].iloc[0]
    no_age_counts = []
    for model in ["XGBoost_no_age", "LightGBM_no_age", "HistGradientBoosting_no_age", "CatBoost_no_age"]:
        col = f"strict_model_residual_candidate_{model}"
        no_age_counts.append({"model": model, "strict_candidate_count": int(df[col].sum())})
    no_age_counts = pd.DataFrame(no_age_counts)
    rule_count = int(pd.to_numeric(df.get("rule_dormant_candidate", 0), errors="coerce").fillna(0).sum())
    cluster_pool_count = rule_count
    high20 = int(df["rule_candidate_model_enhanced_top20"].sum())
    strict = int(df["strict_model_residual_candidate"].sum())

    lines = [
        "# 7.10 ExpectedUse 残差口径审计与高置信分层",
        "",
        "## 1. 审计目的",
        "",
        "第 7 章多模型 OOF 表明 XGBoost、LightGBM、HGB、RandomForest 都能较好预测 ActualUse。严格模型残差候选数量很少，因此模型残差不应作为唯一沉睡资产筛选器，而应作为规则候选的经验验证和置信度增强指标。",
        "",
        "## 2. 特征泄漏检查",
        "",
        f"- 进入 ExpectedUse 模型的特征数：{len(leakage)}。",
        f"- 命中泄漏规则的特征数：{leak_count}。",
        "",
        "泄漏规则检查结果见 `tables/7.10_模型特征泄漏检查.csv`。若命中数为 0，说明当前 ExpectedUse 模型没有直接使用浏览、下载、调用、ActualUse、PotentialUse、DormantScore、ID、URL 或采集过程变量。",
        "",
        "## 3. no-age 稳健性模型",
        "",
        "去掉以下时间暴露变量后，重新训练 XGBoost OOF：",
        "",
        "```text",
        "\n".join(TIME_EXPOSURE_FEATURES),
        "```",
        "",
        "模型表现对比：",
        "",
        markdown_table(
            pd.concat(
                [
                    pd.DataFrame(
                        [
                            {
                                "model": "XGBoost_full",
                                "r2": xgb_full["r2"],
                                "mae": xgb_full["mae"],
                                "rmse": xgb_full["rmse"],
                                "spearman_corr": xgb_full["spearman_corr"],
                            }
                        ]
                    ),
                    no_age_oof[["model", "r2", "mae", "rmse", "spearman_corr"]],
                ],
                ignore_index=True,
            ).sort_values("spearman_corr", ascending=False),
            digits=4,
        ),
        "",
        "no-age 严格残差候选数量：",
        "",
        markdown_table(no_age_counts, digits=4),
        "",
        "若 no-age 模型性能明显下降，说明发布时间和维护时间确实解释了大量累计使用表现；这不是泄漏，而是累计使用量口径的时间暴露效应。",
        "",
        "## 4. DormantScore_model 分位数",
        "",
        markdown_table(quantiles, max_rows=20, digits=4),
        "",
        "## 5. 残差与高置信候选分层",
        "",
        markdown_table(layer_summary, max_rows=20, digits=4),
        "",
        "## 6. 后续聚类样本池",
        "",
        f"- 第 9 章聚类主样本池：`rule_dormant_candidate = 1`，共 {cluster_pool_count} 条。",
        f"- 高优先级/高置信分析池：`rule_candidate_model_enhanced_top20 = 1`，共 {high20} 条。",
        f"- 极严格模型残差候选：`strict_model_residual_candidate = 1`，共 {strict} 条，仅适合作为极端案例或附录。",
        "",
        "因此，聚类不使用严格交集样本，而以 1111 条规则候选为主；模型残差用于置信度排序、案例优先级和稳健性解释。",
        "",
        "## 7. 输出文件",
        "",
        "- `analysis_v11_expecteduse_audited.csv` / `analysis_v11_expecteduse_audited.xlsx`",
        "- `tables/7.10_模型特征泄漏检查.csv`",
        "- `tables/7.10_no_age模型OOF指标.csv`",
        "- `tables/7.10_DormantScore_model分位数.csv`",
        "- `tables/7.10_残差与高置信候选分层汇总.csv`",
        "- `tables/7.10_候选分层画像_领域类型开放属性.csv`",
        "- `figures/7.10_模型残差分布与分位阈值.png`",
        "- `figures/7.10_残差与高置信候选分层数量.png`",
        "- `figures/7.10_多模型严格残差一致性.png`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    df = pd.read_csv(INPUT_PATH, dtype={"数据集ID": str}, encoding="utf-8-sig", low_memory=False)
    cat_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    numeric_features = [col for col in NUMERIC_FEATURES if col in df.columns]
    used_features = cat_features + numeric_features
    leakage = leakage_check(used_features)
    no_age_predictions, no_age_metrics = oof_no_age_models(df, cat_features, numeric_features)
    df = add_model_residuals(df)
    df = add_no_age_outputs(df, no_age_predictions)
    quantiles = residual_quantiles(df)
    df, layer_summary = add_candidate_layers(df)
    group_profiles(
        df,
        [
            "strict_model_residual_candidate",
            "residual_top10_lowuse_candidate",
            "rule_candidate_model_enhanced_top20",
            "multi_model_strict_residual_2plus",
        ],
    )
    plot_outputs(df, layer_summary)
    write_report(df, leakage, no_age_metrics, quantiles, layer_summary)
    df.to_csv(AUDITED_CSV_PATH, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(AUDITED_XLSX_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="expecteduse_audited")
        layer_summary.to_excel(writer, index=False, sheet_name="candidate_layers")
        no_age_metrics.to_excel(writer, index=False, sheet_name="no_age_metrics")
        quantiles.to_excel(writer, index=False, sheet_name="residual_quantiles")
        leakage.to_excel(writer, index=False, sheet_name="leakage_check")
    print(f"rows={len(df)}")
    print(f"leakage_flags={int(leakage['any_leakage_flag'].sum())}")
    print(f"strict_model_residual_candidate={int(df['strict_model_residual_candidate'].sum())}")
    print(f"rule_candidate_model_enhanced_top20={int(df['rule_candidate_model_enhanced_top20'].sum())}")
    print(f"cluster_pool_rule_candidates={int(pd.to_numeric(df.get('rule_dormant_candidate',0), errors='coerce').fillna(0).sum())}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
