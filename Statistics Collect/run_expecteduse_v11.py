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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
EXPECTED_DIR = OUTPUT_DIR / "expecteduse_v11"
FIG_DIR = EXPECTED_DIR / "figures"
TABLE_DIR = EXPECTED_DIR / "tables"

FEATURE_PATH = OUTPUT_DIR / "analysis_v11_features.csv"
EXPECTED_CSV_PATH = OUTPUT_DIR / "analysis_v11_expecteduse.csv"
EXPECTED_XLSX_PATH = OUTPUT_DIR / "analysis_v11_expecteduse.xlsx"
REPORT_PATH = EXPECTED_DIR / "expecteduse_v11_report.md"
MODEL_FEATURES_PATH = TABLE_DIR / "7_ExpectedUse_模型特征清单.csv"
MODEL_METRICS_PATH = TABLE_DIR / "7_ExpectedUse_模型性能汇总.csv"
FOLD_METRICS_PATH = TABLE_DIR / "7_ExpectedUse_主模型OOF分折指标.csv"
ALL_MODEL_FOLD_METRICS_PATH = TABLE_DIR / "7_ExpectedUse_所有模型OOF分折指标.csv"
FEATURE_IMPORTANCE_PATH = TABLE_DIR / "7_ExpectedUse_CatBoost特征重要性.csv"
MODEL_CANDIDATE_PATH = TABLE_DIR / "7_ExpectedUse_各模型残差候选数量.csv"

TARGET_COL = "ActualUse_type_percentile"
N_SPLITS = 5
RANDOM_STATE = 20260503

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

FORBIDDEN_PATTERNS = [
    "浏览量",
    "下载量",
    "接口调用量",
    "view_count",
    "download_count",
    "api_call_count",
    "log_view",
    "log_download",
    "log_api",
    "view_score",
    "download_score",
    "api_call_score",
    "ActualUse",
    "PotentialUse",
    "DormantScore",
    "dormant",
    "candidate",
    "scrape_status",
    "scrape_error",
    "scraped_at",
    "数据集ID",
    "detail_url",
    "rating_score",
    "comment_count",
    "recommended_dataset_names",
]


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
    ranks = values.rank(method="average", na_option="keep")
    return (ranks - 1) / (n - 1)


def ensure_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    selected = cat + numeric
    forbidden = []
    for col in selected:
        lowered = col.lower()
        if any(pattern.lower() in lowered for pattern in FORBIDDEN_PATTERNS):
            forbidden.append(col)
    if forbidden:
        raise ValueError(f"Forbidden leakage features selected: {forbidden}")
    missing_target = TARGET_COL not in df.columns
    if missing_target:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    return cat, numeric


def prepare_catboost_frame(df: pd.DataFrame, cat_features: list[str], numeric_features: list[str]) -> pd.DataFrame:
    x = df[cat_features + numeric_features].copy()
    for col in cat_features:
        x[col] = x[col].fillna("缺失").astype(str)
        x[col] = x[col].mask(x[col].str.strip().eq(""), "缺失")
    for col in numeric_features:
        x[col] = num(x[col])
    return x


def metric_dict(y_true: np.ndarray, pred: np.ndarray, model_name: str, fold: str | int = "OOF") -> dict[str, object]:
    mask = np.isfinite(y_true) & np.isfinite(pred)
    yt = y_true[mask]
    yp = pred[mask]
    rmse = mean_squared_error(yt, yp) ** 0.5
    spearman = stats.spearmanr(yt, yp)
    pearson = stats.pearsonr(yt, yp)
    return {
        "model": model_name,
        "fold": fold,
        "n": int(mask.sum()),
        "r2": float(r2_score(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(rmse),
        "spearman_corr": float(spearman.correlation),
        "spearman_p": float(spearman.pvalue),
        "pearson_corr": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
    }


def make_catboost(iterations: int = 900, learning_rate: float = 0.035, depth: int = 6) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
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


def run_catboost_oof(
    df: pd.DataFrame,
    cat_features: list[str],
    numeric_features: list[str],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, CatBoostRegressor]:
    x = prepare_catboost_frame(df, cat_features, numeric_features)
    y = num(df[TARGET_COL]).to_numpy(dtype=float)
    cat_indices = [x.columns.get_loc(col) for col in cat_features]
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(df), np.nan, dtype=float)
    fold_rows = []
    importances = []

    for fold, (train_idx, valid_idx) in enumerate(folds.split(x), start=1):
        model = make_catboost()
        train_pool = Pool(x.iloc[train_idx], y[train_idx], cat_features=cat_indices)
        valid_pool = Pool(x.iloc[valid_idx], y[valid_idx], cat_features=cat_indices)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        pred = model.predict(valid_pool)
        oof[valid_idx] = pred
        row = metric_dict(y[valid_idx], pred, "CatBoostRegressor", fold)
        row["best_iteration"] = int(model.get_best_iteration() or model.tree_count_)
        fold_rows.append(row)
        imp = pd.DataFrame(
            {
                "feature": x.columns,
                f"fold_{fold}_importance": model.get_feature_importance(type="PredictionValuesChange"),
            }
        )
        importances.append(imp)

    full_model = make_catboost(iterations=1100, learning_rate=0.03, depth=6)
    full_pool = Pool(x, y, cat_features=cat_indices)
    full_model.fit(full_pool)

    imp_full = pd.DataFrame(
        {
            "feature": x.columns,
            "full_model_importance": full_model.get_feature_importance(type="PredictionValuesChange"),
        }
    )
    importance_df = imp_full.copy()
    for imp in importances:
        importance_df = importance_df.merge(imp, on="feature", how="left")
    fold_cols = [col for col in importance_df.columns if col.startswith("fold_")]
    importance_df["oof_mean_importance"] = importance_df[fold_cols].mean(axis=1)
    importance_df["oof_std_importance"] = importance_df[fold_cols].std(axis=1)
    importance_df = importance_df.sort_values("oof_mean_importance", ascending=False)

    return oof, pd.DataFrame(fold_rows), importance_df, full_model


def sklearn_oof_baselines(df: pd.DataFrame, cat_features: list[str], numeric_features: list[str]) -> pd.DataFrame:
    metrics, _, _ = sklearn_oof_model_predictions(df, cat_features, numeric_features)
    return metrics


def sklearn_oof_model_predictions(
    df: pd.DataFrame, cat_features: list[str], numeric_features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = df[cat_features + numeric_features].copy()
    y = num(df[TARGET_COL]).to_numpy(dtype=float)
    for col in cat_features:
        x[col] = x[col].fillna("缺失").astype(str).mask(x[col].fillna("").astype(str).str.strip().eq(""), "缺失")
    for col in numeric_features:
        x[col] = num(x[col])

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre_linear = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), cat_features),
        ],
        remainder="drop",
    )
    pre_tree = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]), cat_features),
        ],
        remainder="drop",
    )

    models = {
        "Ridge": Pipeline([("pre", pre_linear), ("model", Ridge(alpha=5.0))]),
        "RandomForest": Pipeline(
            [
                ("pre", pre_tree),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=450,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("pre", pre_tree),
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
        "LightGBM": Pipeline(
            [
                ("pre", pre_tree),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=650,
                        learning_rate=0.035,
                        num_leaves=31,
                        max_depth=-1,
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
        "XGBoost": Pipeline(
            [
                ("pre", pre_tree),
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
    }

    rows = []
    fold_rows = []
    pred_df = pd.DataFrame({"数据集ID": df["数据集ID"].astype(str)})
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for name, pipe in models.items():
        pred = np.full(len(df), np.nan, dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(folds.split(x), start=1):
            pipe.fit(x.iloc[train_idx], y[train_idx])
            pred[valid_idx] = pipe.predict(x.iloc[valid_idx])
            fold_rows.append(metric_dict(y[valid_idx], pred[valid_idx], name, fold))
        rows.append(metric_dict(y, pred, name, "OOF"))
        pred_df[f"ExpectedUse_{name}_oof_raw"] = pred
    return pd.DataFrame(rows), pred_df, pd.DataFrame(fold_rows)


def build_outputs(
    df: pd.DataFrame,
    cat_features: list[str],
    numeric_features: list[str],
    cat_oof: np.ndarray,
    fold_metrics: pd.DataFrame,
    all_model_fold_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    out["ExpectedUse_catboost_oof_raw"] = cat_oof
    out = out.merge(baseline_predictions, on="数据集ID", how="left")

    model_raw_cols = {
        "CatBoostRegressor": "ExpectedUse_catboost_oof_raw",
        "Ridge": "ExpectedUse_Ridge_oof_raw",
        "RandomForest": "ExpectedUse_RandomForest_oof_raw",
        "HistGradientBoosting": "ExpectedUse_HistGradientBoosting_oof_raw",
        "LightGBM": "ExpectedUse_LightGBM_oof_raw",
        "XGBoost": "ExpectedUse_XGBoost_oof_raw",
    }

    all_metrics = pd.concat(
        [pd.DataFrame([metric_dict(num(out[TARGET_COL]).to_numpy(float), cat_oof, "CatBoostRegressor", "OOF")]), baseline_metrics],
        ignore_index=True,
    )
    all_metrics["rank_spearman"] = all_metrics["spearman_corr"].rank(ascending=False, method="min")
    all_metrics["rank_rmse"] = all_metrics["rmse"].rank(ascending=True, method="min")
    all_metrics["rank_mae"] = all_metrics["mae"].rank(ascending=True, method="min")
    all_metrics["model_selection_score"] = all_metrics["rank_spearman"] + all_metrics["rank_rmse"] + all_metrics["rank_mae"]
    best_model = all_metrics.sort_values(
        ["model_selection_score", "spearman_corr", "rmse"],
        ascending=[True, False, True],
    ).iloc[0]["model"]
    best_raw_col = model_raw_cols[best_model]
    out["ExpectedUse_selected_model_name"] = best_model
    out["ExpectedUse_model"] = out[best_raw_col]
    out["ExpectedUse_model_percentile"] = percentile_01(out["ExpectedUse_model"])
    out["DormantScore_model"] = out["ExpectedUse_model_percentile"] - num(out[TARGET_COL])
    out["DormantScore_model_rank"] = out["DormantScore_model"].rank(ascending=False, method="min")
    out["model_residual_dormant_candidate"] = (
        (out["ExpectedUse_model_percentile"] >= 0.70)
        & (num(out[TARGET_COL]) <= 0.50)
        & (out["DormantScore_model"] >= 0.30)
    ).astype(int)
    out["HighConfidenceDormant"] = (
        (pd.to_numeric(out.get("rule_dormant_candidate", 0), errors="coerce").fillna(0).astype(int) == 1)
        & (out["model_residual_dormant_candidate"] == 1)
    ).astype(int)

    candidate_rows = []
    for model_name, raw_col in model_raw_cols.items():
        if raw_col not in out.columns:
            continue
        pct_col = f"ExpectedUse_{model_name}_percentile"
        score_col = f"DormantScore_model_{model_name}"
        cand_col = f"model_residual_dormant_candidate_{model_name}"
        out[pct_col] = percentile_01(out[raw_col])
        out[score_col] = out[pct_col] - num(out[TARGET_COL])
        out[cand_col] = (
            (out[pct_col] >= 0.70)
            & (num(out[TARGET_COL]) <= 0.50)
            & (out[score_col] >= 0.30)
        ).astype(int)
        candidate_rows.append(
            {
                "model": model_name,
                "expecteduse_raw_col": raw_col,
                "candidate_count": int(out[cand_col].sum()),
                "candidate_share": float(out[cand_col].mean()),
                "high_confidence_intersection_with_rule": int(
                    (
                        out[cand_col].eq(1)
                        & pd.to_numeric(out.get("rule_dormant_candidate", 0), errors="coerce").fillna(0).astype(int).eq(1)
                    ).sum()
                ),
                "is_selected_expecteduse_model": int(model_name == best_model),
            }
        )

    all_metrics.to_csv(MODEL_METRICS_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_rows).to_csv(MODEL_CANDIDATE_PATH, index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(FOLD_METRICS_PATH, index=False, encoding="utf-8-sig")
    all_fold_metrics = pd.concat([fold_metrics, all_model_fold_metrics], ignore_index=True)
    all_fold_metrics.to_csv(ALL_MODEL_FOLD_METRICS_PATH, index=False, encoding="utf-8-sig")
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    feature_table = pd.DataFrame(
        {
            "feature": cat_features + numeric_features,
            "feature_type": ["categorical"] * len(cat_features) + ["numeric"] * len(numeric_features),
            "used_in_catboost": 1,
        }
    )
    feature_table.to_csv(MODEL_FEATURES_PATH, index=False, encoding="utf-8-sig")

    out.to_csv(EXPECTED_CSV_PATH, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(EXPECTED_XLSX_PATH, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="expecteduse")
        all_metrics.to_excel(writer, index=False, sheet_name="model_metrics")
        pd.read_csv(MODEL_CANDIDATE_PATH, encoding="utf-8-sig").to_excel(
            writer, index=False, sheet_name="candidate_counts"
        )
        fold_metrics.to_excel(writer, index=False, sheet_name="catboost_folds")
        all_fold_metrics.to_excel(writer, index=False, sheet_name="all_model_folds")
        feature_importance.head(80).to_excel(writer, index=False, sheet_name="feature_importance")
        feature_table.to_excel(writer, index=False, sheet_name="model_features")
    return out


def plot_outputs(out: pd.DataFrame, metrics: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    selected_model = str(out["ExpectedUse_selected_model_name"].iloc[0])
    plt.figure(figsize=(6.5, 6))
    sns.scatterplot(
        x=num(out[TARGET_COL]),
        y=num(out["ExpectedUse_model_percentile"]),
        s=12,
        alpha=0.45,
        linewidth=0,
    )
    plt.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1)
    plt.xlabel("ActualUse_type_percentile")
    plt.ylabel("ExpectedUse_model_percentile")
    plt.title(f"7 {selected_model} OOF：实际使用与模型期望使用")
    plt.savefig(FIG_DIR / "7_最佳模型OOF实际使用与期望使用散点.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(num(out["DormantScore_model"]), bins=50, color="#4c78a8", kde=True)
    plt.axvline(0.30, color="#c44e52", linestyle="--", linewidth=1.4, label="模型残差候选阈值 0.30")
    plt.xlabel("DormantScore_model")
    plt.ylabel("资源数量")
    plt.title("7 模型残差沉睡度分布")
    plt.legend()
    plt.savefig(FIG_DIR / "7_模型残差沉睡度分布.png", dpi=180, bbox_inches="tight")
    plt.close()

    top_imp = feature_importance.head(25).iloc[::-1]
    plt.figure(figsize=(9, 8))
    sns.barplot(data=top_imp, x="oof_mean_importance", y="feature", color="#59a14f")
    plt.xlabel("CatBoost OOF 平均重要性")
    plt.ylabel("特征")
    plt.title("7 CatBoost 特征重要性 Top25")
    plt.savefig(FIG_DIR / "7_CatBoost特征重要性Top25.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    order = metrics.sort_values("spearman_corr", ascending=False)["model"].tolist()
    sns.barplot(data=metrics, x="model", y="spearman_corr", order=order, color="#f28e2b")
    plt.ylim(0, 1)
    plt.xlabel("模型")
    plt.ylabel("OOF Spearman")
    plt.title("7 ExpectedUse OOF 模型排序相关")
    plt.xticks(rotation=20, ha="right")
    plt.savefig(FIG_DIR / "7_ExpectedUse模型OOF排序相关对比.png", dpi=180, bbox_inches="tight")
    plt.close()


def markdown_table(df: pd.DataFrame, max_rows: int = 20, digits: int = 4) -> str:
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return view.to_markdown(index=False)


def make_report(
    out: pd.DataFrame,
    cat_features: list[str],
    numeric_features: list[str],
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    catboost_row = metrics[metrics["model"].eq("CatBoostRegressor")].iloc[0]
    selected_model = str(out["ExpectedUse_selected_model_name"].iloc[0])
    selected_row = metrics[metrics["model"].eq(selected_model)].iloc[0]
    candidate_counts = pd.read_csv(MODEL_CANDIDATE_PATH, encoding="utf-8-sig")
    all_fold_metrics = pd.read_csv(ALL_MODEL_FOLD_METRICS_PATH, encoding="utf-8-sig")
    model_candidate_count = int(out["model_residual_dormant_candidate"].sum())
    high_conf_count = int(out["HighConfidenceDormant"].sum())
    rule_count = int(pd.to_numeric(out.get("rule_dormant_candidate", 0), errors="coerce").fillna(0).sum())
    leakage_json = json.dumps(FORBIDDEN_PATTERNS, ensure_ascii=False)
    lines = [
        "# 第7章 ExpectedUse 模型期望使用构造 V1.1",
        "",
        "## 7.1 章节定位",
        "",
        "ExpectedUse 表示：基于公共数据资源自身属性，机器学习模型认为该资源在平台经验规律下应当达到的使用表现。本章先按用户要求以 CatBoostRegressor 为中心模型，同时加入 RandomForest、LightGBM、XGBoost、HistGradientBoosting 与 Ridge 做 OOF 对比；最终 `ExpectedUse_model` 采用 OOF 综合表现最优的模型。",
        "",
        "## 7.2 数据与目标变量",
        "",
        f"- 样本数：{len(out)} 条。",
        f"- 目标变量：`{TARGET_COL}`，即第 5 章确定的类型适配 ActualUse 百分位。",
        f"- 折数：{N_SPLITS} 折 OOF。",
        f"- CatBoost 分类特征数：{len(cat_features)}。",
        f"- CatBoost 数值特征数：{len(numeric_features)}。",
        "",
        "## 7.3 特征泄漏护栏",
        "",
        "本章禁止使用任何使用量、实际使用强度、潜在价值、沉睡度、采集过程变量、ID 与 URL 作为模型特征。禁用关键词如下：",
        "",
        f"`{leakage_json}`",
        "",
        "实际进入模型的特征清单见 `tables/7_ExpectedUse_模型特征清单.csv`。",
        "",
        "## 7.4 OOF 模型表现与主模型选择",
        "",
        markdown_table(metrics[["model", "fold", "n", "r2", "mae", "rmse", "spearman_corr", "pearson_corr", "model_selection_score"]].sort_values("model_selection_score"), digits=4),
        "",
        f"综合 Spearman 排名、RMSE 排名和 MAE 排名后，当前选择 `{selected_model}` 作为 `ExpectedUse_model` 的主口径。",
        "",
        "CatBoost 分折表现：",
        "",
        markdown_table(fold_metrics[["model", "fold", "n", "r2", "mae", "rmse", "spearman_corr", "best_iteration"]], digits=4),
        "",
        "所有模型分折表现摘要：",
        "",
        markdown_table(
            all_fold_metrics.groupby("model")
            .agg(
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                mae_mean=("mae", "mean"),
                mae_std=("mae", "std"),
                spearman_mean=("spearman_corr", "mean"),
                spearman_std=("spearman_corr", "std"),
            )
            .reset_index()
            .sort_values("spearman_mean", ascending=False),
            digits=4,
        ),
        "",
        "## 7.5 选定主模型与 CatBoost 对照",
        "",
        f"- 选定主模型：`{selected_model}`。",
        f"- 选定主模型 OOF R2：{selected_row['r2']:.4f}。",
        f"- 选定主模型 OOF MAE：{selected_row['mae']:.4f}。",
        f"- 选定主模型 OOF RMSE：{selected_row['rmse']:.4f}。",
        f"- 选定主模型 OOF Spearman：{selected_row['spearman_corr']:.4f}。",
        "",
        f"- CatBoost OOF R2：{catboost_row['r2']:.4f}。",
        f"- CatBoost OOF MAE：{catboost_row['mae']:.4f}。",
        f"- CatBoost OOF RMSE：{catboost_row['rmse']:.4f}。",
        f"- CatBoost OOF Spearman：{catboost_row['spearman_corr']:.4f}。",
        "",
        "主模型的 OOF 预测值被保存为：",
        "",
        "```text",
        "ExpectedUse_model",
        "ExpectedUse_model_percentile",
        "DormantScore_model = ExpectedUse_model_percentile - ActualUse_type_percentile",
        "```",
        "",
        "CatBoost 的 OOF 预测仍保留为 `ExpectedUse_catboost_oof_raw`，用于对照。",
        "",
        "## 7.6 初步模型残差候选",
        "",
        "虽然 8.2 才正式讨论模型残差沉睡度，但本章已同步生成必要字段，方便后续衔接：",
        "",
        f"- 规则沉睡候选：{rule_count} 条。",
        f"- 模型残差候选：{model_candidate_count} 条。",
        f"- 当前规则候选与模型残差候选交集 HighConfidenceDormant：{high_conf_count} 条。",
        "",
        "各模型残差候选数量如下：",
        "",
        markdown_table(candidate_counts.sort_values("candidate_count", ascending=False), digits=4),
        "",
        "注意：这些候选字段供第 8.2/8.3 使用，本章的主任务仍是 ExpectedUse OOF 构造。",
        "",
        "## 7.7 特征重要性 Top 25",
        "",
        markdown_table(feature_importance[["feature", "oof_mean_importance", "oof_std_importance", "full_model_importance"]].head(25), digits=4),
        "",
        "## 7.8 输出文件",
        "",
        "- `analysis_v11_expecteduse.csv` / `analysis_v11_expecteduse.xlsx`",
        "- `outputs/expecteduse_v11/tables/7_ExpectedUse_模型性能汇总.csv`",
        "- `outputs/expecteduse_v11/tables/7_ExpectedUse_CatBoost_OOF分折指标.csv`",
        "- `outputs/expecteduse_v11/tables/7_ExpectedUse_CatBoost特征重要性.csv`",
        "- `outputs/expecteduse_v11/figures/7_最佳模型OOF实际使用与期望使用散点.png`",
        "- `outputs/expecteduse_v11/figures/7_模型残差沉睡度分布.png`",
        "- `outputs/expecteduse_v11/figures/7_CatBoost特征重要性Top25.png`",
        "- `outputs/expecteduse_v11/figures/7_ExpectedUse模型OOF排序相关对比.png`",
        "",
        "## 7.9 下一步",
        "",
        "下一步应进入 8.2 模型残差沉睡度识别，正式解释 `DormantScore_model` 与 `model_residual_dormant_candidate`，再在 8.3 中与规则候选取交集形成高置信沉睡资产。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup()
    df = pd.read_csv(FEATURE_PATH, dtype={"数据集ID": str}, encoding="utf-8-sig", low_memory=False)
    cat_features, numeric_features = ensure_features(df)
    cat_oof, fold_metrics, feature_importance, _ = run_catboost_oof(df, cat_features, numeric_features)
    baseline_metrics, baseline_predictions, baseline_fold_metrics = sklearn_oof_model_predictions(
        df, cat_features, numeric_features
    )
    out = build_outputs(
        df,
        cat_features,
        numeric_features,
        cat_oof,
        fold_metrics,
        baseline_fold_metrics,
        feature_importance,
        baseline_metrics,
        baseline_predictions,
    )
    metrics = pd.read_csv(MODEL_METRICS_PATH, encoding="utf-8-sig")
    plot_outputs(out, metrics, feature_importance)
    make_report(out, cat_features, numeric_features, metrics, fold_metrics, feature_importance)
    print(f"rows={len(out)}")
    print(f"selected_model={out['ExpectedUse_selected_model_name'].iloc[0]}")
    print(f"cat_features={len(cat_features)} numeric_features={len(numeric_features)}")
    print(f"model_residual_dormant_candidate={int(out['model_residual_dormant_candidate'].sum())}")
    print(f"HighConfidenceDormant={int(out['HighConfidenceDormant'].sum())}")
    print(f"report={REPORT_PATH}")
    print(f"expected_csv={EXPECTED_CSV_PATH}")


if __name__ == "__main__":
    main()
