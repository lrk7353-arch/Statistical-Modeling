from __future__ import annotations

import ast
import json
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_PATH = OUTPUT_DIR / "analysis_v11_expecteduse_audited.csv"
EXPECTED_SCRIPT_PATH = BASE_DIR / "run_expecteduse_v11.py"

CHAPTER_DIR = OUTPUT_DIR / "usage_influence_v11"
TABLE_DIR = CHAPTER_DIR / "tables"
FIGURE_DIR = CHAPTER_DIR / "figures"
REPORT_PATH = CHAPTER_DIR / "9_usage_influence_explanation_v11_report.md"

TARGET_COL = "ActualUse_type_percentile"
RANDOM_STATE = 20260503
SHAP_SAMPLE_SIZE = 3000
PDP_SAMPLE_SIZE = 1800

KEY_VARIABLES = [
    "publication_age_days_clean",
    "maintenance_span_days_clean",
    "api_need_apply_clean",
    "is_unconditional_open",
    "is_data_interface",
    "format_count_clean",
    "field_description_count_clean",
    "update_recency_days_clean",
]

PLOT_LABELS = {
    "publication_age_days_clean": "发布年龄",
    "maintenance_span_days_clean": "维护跨度",
    "update_recency_days_clean": "更新近度",
    "api_need_apply_clean": "API申请门槛",
    "is_unconditional_open": "无条件开放",
    "is_conditional_open": "有条件开放",
    "is_data_interface": "数据接口",
    "is_data_product": "数据产品",
    "format_count_clean": "格式数量",
    "field_description_count_clean": "字段说明丰富度",
    "数据资源提供部门": "提供部门",
    "数据领域": "数据领域",
    "数据资源类型": "资源类型",
    "开放属性": "开放属性",
    "更新频率": "更新频率",
    "detail_spatial_scope": "详情页空间范围",
    "detail_time_scope": "详情页时间范围",
}


def ensure_dirs() -> None:
    for path in [CHAPTER_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_chinese_font() -> None:
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["font.sans-serif"] = candidates
    plt.rcParams["axes.unicode_minus"] = False


def parse_expecteduse_constants() -> tuple[list[str], list[str], list[str]]:
    source = EXPECTED_SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    values: dict[str, object] = {}
    wanted = {"CATEGORICAL_FEATURES", "NUMERIC_FEATURES", "FORBIDDEN_PATTERNS"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    values[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(values)
    if missing:
        raise RuntimeError(f"Missing constants in {EXPECTED_SCRIPT_PATH}: {sorted(missing)}")
    return (
        list(values["CATEGORICAL_FEATURES"]),
        list(values["NUMERIC_FEATURES"]),
        list(values["FORBIDDEN_PATTERNS"]),
    )


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_name(name: str) -> str:
    invalid = '<>:"/\\|?*'
    out = "".join("_" if ch in invalid else ch for ch in str(name))
    return out.replace(" ", "_")[:90]


def label(name: str) -> str:
    return PLOT_LABELS.get(name, name)


def make_feature_frame(df: pd.DataFrame, categorical: list[str], numeric: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    cat = [col for col in categorical if col in df.columns]
    nums = [col for col in numeric if col in df.columns]
    use_cols = cat + nums
    x = df[use_cols].copy()
    for col in cat:
        x[col] = x[col].astype("string").fillna("缺失").astype(str)
    for col in nums:
        x[col] = num(x[col])
    return x, cat, nums


def leakage_check(features: list[str], forbidden_patterns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        lowered = feature.lower()
        hits = [pattern for pattern in forbidden_patterns if pattern.lower() in lowered]
        rows.append(
            {
                "feature": feature,
                "leakage_flag": int(bool(hits)),
                "matched_patterns": ";".join(hits),
            }
        )
    return pd.DataFrame(rows)


def make_preprocessor(cat: list[str], nums: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), nums),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="缺失")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.25,
        verbose_feature_names_out=True,
    )


def make_xgb_model() -> XGBRegressor:
    return XGBRegressor(
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
        tree_method="hist",
        importance_type="gain",
    )


def get_feature_names(pre: ColumnTransformer) -> list[str]:
    names = pre.get_feature_names_out()
    return [str(x) for x in names]


def encoded_to_original(encoded_name: str, cat: list[str], nums: list[str]) -> str:
    if encoded_name.startswith("num__"):
        return encoded_name.replace("num__", "", 1)
    if encoded_name.startswith("cat__"):
        rest = encoded_name.replace("cat__", "", 1)
        for col in sorted(cat, key=len, reverse=True):
            if rest == col or rest.startswith(f"{col}_"):
                return col
    return encoded_name


def to_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    sp = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return {
        "R2_in_sample_full_fit": float(r2_score(y_true, y_pred)),
        "MAE_in_sample_full_fit": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_in_sample_full_fit": float(rmse),
        "Spearman_in_sample_full_fit": float(sp),
    }


def save_barplot(df: pd.DataFrame, value_col: str, name_col: str, title: str, path: Path, top_n: int = 30) -> None:
    plot_df = df.head(top_n).iloc[::-1].copy()
    plot_df[name_col] = plot_df[name_col].map(label)
    height = max(7, top_n * 0.28)
    plt.figure(figsize=(11, height))
    plt.barh(plot_df[name_col], plot_df[value_col], color="#3f7f93")
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def aggregate_shap(
    shap_values: np.ndarray,
    encoded_names: list[str],
    cat: list[str],
    nums: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded_df = pd.DataFrame(
        {
            "encoded_feature": encoded_names,
            "original_feature": [encoded_to_original(name, cat, nums) for name in encoded_names],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_signed_shap": shap_values.mean(axis=0),
            "std_shap": shap_values.std(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    original_df = (
        encoded_df.groupby("original_feature", as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "sum"),
            mean_signed_shap=("mean_signed_shap", "sum"),
            encoded_feature_count=("encoded_feature", "count"),
        )
        .sort_values("mean_abs_shap", ascending=False)
    )
    return encoded_df, original_df


def original_shap_matrix(shap_values: np.ndarray, encoded_names: list[str], original_features: list[str], cat: list[str], nums: list[str]) -> pd.DataFrame:
    mapping = [encoded_to_original(name, cat, nums) for name in encoded_names]
    out = {}
    for feature in original_features:
        idx = [i for i, base in enumerate(mapping) if base == feature]
        if idx:
            out[feature] = shap_values[:, idx].sum(axis=1)
    return pd.DataFrame(out)


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    x_dense: np.ndarray,
    encoded_names: list[str],
    encoded_summary: pd.DataFrame,
    path: Path,
    max_display: int = 25,
) -> None:
    top_features = encoded_summary.head(max_display)["encoded_feature"].tolist()
    indices = [encoded_names.index(f) for f in top_features if f in encoded_names]
    top_names = [encoded_names[i] for i in indices]
    display_names = [name.replace("num__", "").replace("cat__", "") for name in top_names]
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values[:, indices],
        features=x_dense[:, indices],
        feature_names=display_names,
        max_display=max_display,
        show=False,
        plot_size=(11, 8),
    )
    plt.title("SHAP蜂群图：编码特征Top25")
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def make_pdp(
    model: XGBRegressor,
    pre: ColumnTransformer,
    x_sample: pd.DataFrame,
    feature: str,
    path: Path,
) -> pd.DataFrame:
    series = x_sample[feature]
    if pd.api.types.is_numeric_dtype(series):
        clean = num(series).dropna()
        if clean.nunique() <= 2:
            grid = sorted(clean.unique().tolist())
        else:
            grid = np.unique(np.nanquantile(clean, np.linspace(0.05, 0.95, 21)))
    else:
        grid = series.astype(str).value_counts().head(12).index.tolist()

    rows = []
    for value in grid:
        temp = x_sample.copy()
        temp[feature] = value
        pred = model.predict(pre.transform(temp))
        rows.append({"feature": feature, "value": value, "mean_prediction": float(np.mean(pred))})

    pdp_df = pd.DataFrame(rows)
    plt.figure(figsize=(8.5, 5.2))
    if pd.api.types.is_numeric_dtype(series):
        plt.plot(pdp_df["value"], pdp_df["mean_prediction"], marker="o", color="#5867a6")
        plt.xlabel(label(feature))
    else:
        plt.bar(pdp_df["value"].astype(str), pdp_df["mean_prediction"], color="#5867a6")
        plt.xticks(rotation=35, ha="right")
        plt.xlabel(label(feature))
    plt.ylabel("模型预测 ExpectedUse")
    plt.title(f"局部影响：{label(feature)}")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return pdp_df


def direction_summary(x_sample: pd.DataFrame, shap_original: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for feature in variables:
        if feature not in x_sample.columns or feature not in shap_original.columns:
            continue
        s = num(x_sample[feature])
        sv = shap_original[feature]
        if s.nunique(dropna=True) <= 2:
            vals = sorted(s.dropna().unique().tolist())
            if len(vals) == 2:
                mean_low = float(sv[s == vals[0]].mean())
                mean_high = float(sv[s == vals[1]].mean())
                direction = "正向" if mean_high > mean_low else "负向"
                effect = mean_high - mean_low
            else:
                direction = "样本不足"
                effect = np.nan
        else:
            corr = pd.concat([s, sv], axis=1).dropna().corr(method="spearman").iloc[0, 1]
            effect = float(corr)
            if corr > 0.05:
                direction = "总体正相关"
            elif corr < -0.05:
                direction = "总体负相关"
            else:
                direction = "非单调或弱相关"
        rows.append(
            {
                "feature": feature,
                "feature_label": label(feature),
                "direction_or_rank_correlation": direction,
                "effect_statistic": effect,
                "mean_abs_shap": float(np.abs(sv).mean()),
                "mean_signed_shap": float(sv.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def group_shap_summary(df_sample: pd.DataFrame, shap_original: pd.DataFrame, group_col: str, top_n: int = 15) -> pd.DataFrame:
    if group_col not in df_sample.columns:
        return pd.DataFrame()
    rows = []
    joined = shap_original.copy()
    joined[group_col] = df_sample[group_col].astype(str).to_numpy()
    for group_value, sub in joined.groupby(group_col):
        shap_part = sub.drop(columns=[group_col])
        summary = np.abs(shap_part).mean(axis=0).sort_values(ascending=False).head(top_n)
        for rank, (feature, value) in enumerate(summary.items(), start=1):
            rows.append(
                {
                    "group_col": group_col,
                    "group_value": group_value,
                    "rank": rank,
                    "feature": feature,
                    "feature_label": label(feature),
                    "mean_abs_shap": float(value),
                    "n": int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    df: pd.DataFrame,
    metrics: dict[str, float],
    leakage: pd.DataFrame,
    xgb_importance: pd.DataFrame,
    shap_original: pd.DataFrame,
    direction: pd.DataFrame,
    group_summary: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    leak_count = int(leakage["leakage_flag"].sum())
    top_importance = xgb_importance.head(10)[["original_feature", "importance"]].copy()
    top_shap = shap_original.head(10)[["original_feature", "mean_abs_shap"]].copy()

    def md_table(frame: pd.DataFrame, max_rows: int = 10) -> str:
        if frame.empty:
            return "（无）"
        return frame.head(max_rows).to_markdown(index=False)

    group_text = []
    if not group_summary.empty:
        for (group_col, group_value), sub in group_summary.groupby(["group_col", "group_value"]):
            group_text.append(f"\n### {group_col} = {group_value}\n\n{sub.head(10)[['rank','feature_label','mean_abs_shap','n']].to_markdown(index=False)}")

    lines = [
        "# 9. 使用表现影响因素解释 V1.1",
        "",
        "## 9.0 章节定位",
        "",
        "本章承接第 7 章 ExpectedUse 模型，用全量训练后的 XGBoost 解释模型分析公共数据资源属性如何影响使用表现。按照 V1.1 方案，全量训练模型只用于特征重要性、SHAP 和策略解释；第 8 章 `DormantScore_model` 仍以 out-of-fold ExpectedUse 预测为准。",
        "",
        "本章不重新定义 ActualUse、PotentialUse 或 DormantScore，也不进行第 10 章的沉睡资产聚类分型。",
        "",
        "## 9.1 模型与样本口径",
        "",
        f"- 输入文件：`{INPUT_PATH.relative_to(PROJECT_DIR)}`",
        f"- 样本量：{len(df)}",
        f"- 目标变量：`{TARGET_COL}`",
        "- 解释模型：第 7 章主口径 XGBoost 全量拟合模型",
        f"- 泄漏检查命中特征数：{leak_count}",
        "",
        "全量拟合性能仅用于确认解释模型与第 7 章 OOF 模型方向一致，不作为模型泛化能力指标：",
        "",
        pd.DataFrame([metrics]).to_markdown(index=False),
        "",
        "## 9.2 全局特征重要性",
        "",
        "XGBoost 增益重要性 Top 10：",
        "",
        md_table(top_importance),
        "",
        "SHAP 原始特征聚合 Top 10：",
        "",
        md_table(top_shap),
        "",
        "解释重点：如果发布时间、维护跨度和更新近度位居前列，说明累计使用表现受到时间暴露影响；如果部门、领域、开放属性、API 申请和空间范围位居前列，说明平台资源组织方式和开放便利性也在显著影响使用表现。",
        "",
        "## 9.3 重点变量局部影响",
        "",
        md_table(direction, max_rows=20),
        "",
        "其中 `effect_statistic` 对连续变量表示变量值与该变量 SHAP 贡献的 Spearman 相关；对二元变量表示取值为 1 相比 0 的平均 SHAP 贡献差。",
        "",
        "## 9.4 分组 SHAP 机制差异",
        "",
        "本节比较数据产品/数据接口、有条件/无条件开放下的主要影响因素，避免把全平台平均机制误读为所有资源类型的共同机制。",
        "",
        "\n".join(group_text) if group_text else "（分组字段不足，未生成分组 SHAP。）",
        "",
        "## 9.5 输出文件",
        "",
        f"- 特征重要性：`{outputs['importance'].relative_to(PROJECT_DIR)}`",
        f"- SHAP 原始特征汇总：`{outputs['shap_original'].relative_to(PROJECT_DIR)}`",
        f"- 重点变量方向表：`{outputs['direction'].relative_to(PROJECT_DIR)}`",
        f"- 分组 SHAP：`{outputs['group'].relative_to(PROJECT_DIR)}`",
        f"- 图表目录：`{FIGURE_DIR.relative_to(PROJECT_DIR)}`",
        "",
        "## 9.6 后续衔接",
        "",
        "第 9 章回答“哪些资源属性影响公共数据被使用”；第 10 章再基于 1111 条规则沉睡候选进入 GMM/BGMM 聚类，回答“沉睡资产可以分成哪些成因类型”。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    setup_chinese_font()

    categorical, numeric, forbidden = parse_expecteduse_constants()
    df = pd.read_csv(INPUT_PATH, dtype={"数据集ID": str}, encoding="utf-8-sig")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    x, cat, nums = make_feature_frame(df, categorical, numeric)
    features = cat + nums
    leakage = leakage_check(features, forbidden)
    leakage_path = TABLE_DIR / "9_模型解释特征泄漏检查.csv"
    leakage.to_csv(leakage_path, index=False, encoding="utf-8-sig")
    if leakage["leakage_flag"].sum() > 0:
        raise RuntimeError(f"Leakage features detected: {leakage[leakage['leakage_flag'] == 1]['feature'].tolist()}")

    y = num(df[TARGET_COL]).to_numpy(dtype=float)
    valid = np.isfinite(y)
    df_valid = df.loc[valid].reset_index(drop=True)
    x = x.loc[valid].reset_index(drop=True)
    y = y[valid]

    pre = make_preprocessor(cat, nums)
    x_trans = pre.fit_transform(x)
    encoded_names = get_feature_names(pre)
    model = make_xgb_model()
    model.fit(x_trans, y)
    pred = model.predict(x_trans)
    metrics = metric_summary(y, pred)

    model_metrics_path = TABLE_DIR / "9_解释模型全量拟合指标.csv"
    pd.DataFrame([metrics]).to_csv(model_metrics_path, index=False, encoding="utf-8-sig")

    importance = pd.DataFrame(
        {
            "encoded_feature": encoded_names,
            "original_feature": [encoded_to_original(name, cat, nums) for name in encoded_names],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_original = (
        importance.groupby("original_feature", as_index=False)
        .agg(importance=("importance", "sum"), encoded_feature_count=("encoded_feature", "count"))
        .sort_values("importance", ascending=False)
    )
    importance_path = TABLE_DIR / "9_feature_importance_top30.csv"
    importance_original.to_csv(importance_path, index=False, encoding="utf-8-sig")
    importance.to_csv(TABLE_DIR / "9_编码特征重要性明细.csv", index=False, encoding="utf-8-sig")
    save_barplot(
        importance_original,
        "importance",
        "original_feature",
        "XGBoost特征重要性Top30",
        FIGURE_DIR / "9_特征重要性Top30.png",
    )

    sample_n = min(SHAP_SAMPLE_SIZE, len(x))
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(x), size=sample_n, replace=False)
    x_sample = x.iloc[sample_idx].reset_index(drop=True)
    df_sample = df_valid.iloc[sample_idx].reset_index(drop=True)
    x_sample_trans = pre.transform(x_sample)
    x_sample_dense = to_dense(x_sample_trans)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample_dense)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    shap_encoded, shap_original_summary = aggregate_shap(shap_values, encoded_names, cat, nums)
    shap_encoded_path = TABLE_DIR / "9_shap_summary_by_encoded_feature.csv"
    shap_original_path = TABLE_DIR / "9_shap_summary_by_original_feature.csv"
    shap_encoded.to_csv(shap_encoded_path, index=False, encoding="utf-8-sig")
    shap_original_summary.to_csv(shap_original_path, index=False, encoding="utf-8-sig")
    save_barplot(
        shap_original_summary,
        "mean_abs_shap",
        "original_feature",
        "SHAP原始特征影响强度Top30",
        FIGURE_DIR / "9_SHAP原始特征Top30条形图.png",
    )
    plot_shap_beeswarm(
        shap_values,
        x_sample_dense,
        encoded_names,
        shap_encoded,
        FIGURE_DIR / "9_SHAP编码特征Top25蜂群图.png",
    )

    shap_original_matrix = original_shap_matrix(shap_values, encoded_names, features, cat, nums)
    direction = direction_summary(x_sample, shap_original_matrix, KEY_VARIABLES)
    direction_path = TABLE_DIR / "9_key_variable_direction_summary.csv"
    direction.to_csv(direction_path, index=False, encoding="utf-8-sig")

    pdp_sample_n = min(PDP_SAMPLE_SIZE, len(x))
    pdp_idx = np.random.default_rng(RANDOM_STATE + 1).choice(len(x), size=pdp_sample_n, replace=False)
    x_pdp_sample = x.iloc[pdp_idx].reset_index(drop=True)
    pdp_rows = []
    for feature in KEY_VARIABLES:
        if feature in x_pdp_sample.columns:
            path = FIGURE_DIR / f"9_局部影响_{safe_name(label(feature))}.png"
            pdp_rows.append(make_pdp(model, pre, x_pdp_sample, feature, path))
    pdp_table = pd.concat(pdp_rows, ignore_index=True) if pdp_rows else pd.DataFrame()
    pdp_path = TABLE_DIR / "9_partial_dependence_key_variables.csv"
    pdp_table.to_csv(pdp_path, index=False, encoding="utf-8-sig")

    group_parts = []
    for group_col in ["数据资源类型", "开放属性"]:
        if group_col in df_sample.columns:
            group_parts.append(group_shap_summary(df_sample, shap_original_matrix, group_col))
    group_summary = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()
    group_path = TABLE_DIR / "9_group_shap_summary.csv"
    group_summary.to_csv(group_path, index=False, encoding="utf-8-sig")

    outputs = {
        "importance": importance_path,
        "shap_original": shap_original_path,
        "direction": direction_path,
        "group": group_path,
    }
    write_report(
        df_valid,
        metrics,
        leakage,
        importance_original,
        shap_original_summary,
        direction,
        group_summary,
        outputs,
    )

    manifest = {
        "input": str(INPUT_PATH),
        "rows": int(len(df_valid)),
        "features": int(len(features)),
        "categorical_features": cat,
        "numeric_features": nums,
        "shap_sample_size": int(sample_n),
        "pdp_sample_size": int(pdp_sample_n),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "report": str(REPORT_PATH),
        "metrics": metrics,
    }
    (CHAPTER_DIR / "9_usage_influence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
