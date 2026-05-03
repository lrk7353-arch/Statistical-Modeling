# 第7章 ExpectedUse 模型期望使用构造 V1.1

## 7.1 章节定位

ExpectedUse 表示：基于公共数据资源自身属性，机器学习模型认为该资源在平台经验规律下应当达到的使用表现。本章先按用户要求以 CatBoostRegressor 为中心模型，同时加入 RandomForest、LightGBM、XGBoost、HistGradientBoosting 与 Ridge 做 OOF 对比；最终 `ExpectedUse_model` 采用 OOF 综合表现最优的模型。

## 7.2 数据与目标变量

- 样本数：9540 条。
- 目标变量：`ActualUse_type_percentile`，即第 5 章确定的类型适配 ActualUse 百分位。
- 折数：5 折 OOF。
- CatBoost 分类特征数：11。
- CatBoost 数值特征数：32。

## 7.3 特征泄漏护栏

本章禁止使用任何使用量、实际使用强度、潜在价值、沉睡度、采集过程变量、ID 与 URL 作为模型特征。禁用关键词如下：

`["浏览量", "下载量", "接口调用量", "view_count", "download_count", "api_call_count", "log_view", "log_download", "log_api", "view_score", "download_score", "api_call_score", "ActualUse", "PotentialUse", "DormantScore", "dormant", "candidate", "scrape_status", "scrape_error", "scraped_at", "数据集ID", "detail_url", "rating_score", "comment_count", "recommended_dataset_names"]`

实际进入模型的特征清单见 `tables/7_ExpectedUse_模型特征清单.csv`。

## 7.4 OOF 模型表现与主模型选择

| model                | fold   |    n |     r2 |    mae |   rmse |   spearman_corr |   pearson_corr |   model_selection_score |
|:---------------------|:-------|-----:|-------:|-------:|-------:|----------------:|---------------:|------------------------:|
| XGBoost              | OOF    | 9540 | 0.9186 | 0.0565 | 0.0824 |          0.9577 |         0.9584 |                       5 |
| LightGBM             | OOF    | 9540 | 0.9184 | 0.0559 | 0.0825 |          0.9575 |         0.9583 |                       5 |
| HistGradientBoosting | OOF    | 9540 | 0.9181 | 0.056  | 0.0826 |          0.9574 |         0.9582 |                       9 |
| RandomForest         | OOF    | 9540 | 0.917  | 0.0573 | 0.0832 |          0.9574 |         0.9581 |                      11 |
| CatBoostRegressor    | OOF    | 9540 | 0.9125 | 0.06   | 0.0854 |          0.9545 |         0.9553 |                      15 |
| Ridge                | OOF    | 9540 | 0.8549 | 0.0809 | 0.11   |          0.9251 |         0.9246 |                      18 |

综合 Spearman 排名、RMSE 排名和 MAE 排名后，当前选择 `XGBoost` 作为 `ExpectedUse_model` 的主口径。

CatBoost 分折表现：

| model             |   fold |    n |     r2 |    mae |   rmse |   spearman_corr |   best_iteration |
|:------------------|-------:|-----:|-------:|-------:|-------:|----------------:|-----------------:|
| CatBoostRegressor |      1 | 1908 | 0.9024 | 0.0624 | 0.0903 |          0.9483 |              899 |
| CatBoostRegressor |      2 | 1908 | 0.9127 | 0.0586 | 0.0862 |          0.9539 |              898 |
| CatBoostRegressor |      3 | 1908 | 0.9176 | 0.0583 | 0.0811 |          0.9571 |              899 |
| CatBoostRegressor |      4 | 1908 | 0.9141 | 0.0597 | 0.0846 |          0.9547 |              899 |
| CatBoostRegressor |      5 | 1908 | 0.9155 | 0.0612 | 0.0845 |          0.9566 |              899 |

所有模型分折表现摘要：

| model                |   r2_mean |   r2_std |   rmse_mean |   rmse_std |   mae_mean |   mae_std |   spearman_mean |   spearman_std |
|:---------------------|----------:|---------:|------------:|-----------:|-----------:|----------:|----------------:|---------------:|
| XGBoost              |    0.9185 |   0.0061 |      0.0823 |     0.0035 |     0.0565 |    0.0017 |          0.9574 |         0.0038 |
| LightGBM             |    0.9183 |   0.0066 |      0.0824 |     0.0038 |     0.0559 |    0.0018 |          0.9572 |         0.004  |
| RandomForest         |    0.917  |   0.0054 |      0.0831 |     0.003  |     0.0573 |    0.0014 |          0.9571 |         0.0034 |
| HistGradientBoosting |    0.918  |   0.0058 |      0.0826 |     0.0034 |     0.056  |    0.0016 |          0.9571 |         0.0036 |
| CatBoostRegressor    |    0.9125 |   0.0059 |      0.0853 |     0.0033 |     0.06   |    0.0017 |          0.9541 |         0.0035 |
| Ridge                |    0.8547 |   0.013  |      0.1099 |     0.0046 |     0.0809 |    0.0032 |          0.9246 |         0.0084 |

## 7.5 选定主模型与 CatBoost 对照

- 选定主模型：`XGBoost`。
- 选定主模型 OOF R2：0.9186。
- 选定主模型 OOF MAE：0.0565。
- 选定主模型 OOF RMSE：0.0824。
- 选定主模型 OOF Spearman：0.9577。

- CatBoost OOF R2：0.9125。
- CatBoost OOF MAE：0.0600。
- CatBoost OOF RMSE：0.0854。
- CatBoost OOF Spearman：0.9545。

主模型的 OOF 预测值被保存为：

```text
ExpectedUse_model
ExpectedUse_model_percentile
DormantScore_model = ExpectedUse_model_percentile - ActualUse_type_percentile
```

CatBoost 的 OOF 预测仍保留为 `ExpectedUse_catboost_oof_raw`，用于对照。

## 7.6 初步模型残差候选

虽然 8.2 才正式讨论模型残差沉睡度，但本章已同步生成必要字段，方便后续衔接：

- 规则沉睡候选：1111 条。
- 模型残差候选：1 条。
- 当前规则候选与模型残差候选交集 HighConfidenceDormant：1 条。

各模型残差候选数量如下：

| model                | expecteduse_raw_col                      |   candidate_count |   candidate_share |   high_confidence_intersection_with_rule |   is_selected_expecteduse_model |
|:---------------------|:-----------------------------------------|------------------:|------------------:|-----------------------------------------:|--------------------------------:|
| CatBoostRegressor    | ExpectedUse_catboost_oof_raw             |                 3 |            0.0003 |                                        2 |                               0 |
| RandomForest         | ExpectedUse_RandomForest_oof_raw         |                 3 |            0.0003 |                                        1 |                               0 |
| LightGBM             | ExpectedUse_LightGBM_oof_raw             |                 2 |            0.0002 |                                        1 |                               0 |
| HistGradientBoosting | ExpectedUse_HistGradientBoosting_oof_raw |                 2 |            0.0002 |                                        1 |                               0 |
| Ridge                | ExpectedUse_Ridge_oof_raw                |                 1 |            0.0001 |                                        1 |                               0 |
| XGBoost              | ExpectedUse_XGBoost_oof_raw              |                 1 |            0.0001 |                                        1 |                               1 |

注意：这些候选字段供第 8.2/8.3 使用，本章的主任务仍是 ExpectedUse OOF 构造。

## 7.7 特征重要性 Top 25

| feature                         |   oof_mean_importance |   oof_std_importance |   full_model_importance |
|:--------------------------------|----------------------:|---------------------:|------------------------:|
| publication_age_days_clean      |               44.3034 |               1.8822 |                 44.0534 |
| maintenance_span_days_clean     |               10.0745 |               1.0782 |                 10.681  |
| detail_spatial_scope            |                4.7715 |               0.551  |                  3.3409 |
| 数据资源提供部门                        |                3.8476 |               0.4065 |                  4.8093 |
| api_need_apply_clean            |                3.6617 |               0.7184 |                  6.4878 |
| spatial_admin_level             |                3.1979 |               0.4648 |                  3.7902 |
| is_unconditional_open           |                2.8237 |               1.4016 |                  1.776  |
| 开放属性                            |                2.8064 |               0.7379 |                  1.4421 |
| is_conditional_open             |                2.3992 |               0.7541 |                  2.7051 |
| is_data_interface               |                2.05   |               0.6217 |                  0.9038 |
| update_recency_days_clean       |                1.8954 |               0.2301 |                  2.0161 |
| field_description_count_clean   |                1.7273 |               0.342  |                  2.043  |
| 数据领域                            |                1.691  |               0.093  |                  1.43   |
| 更新频率                            |                1.5555 |               0.1831 |                  1.3389 |
| recommended_dataset_count_clean |                1.395  |               0.3151 |                  1.2831 |
| has_meaningful_time_scope       |                1.0989 |               0.7166 |                  0.8413 |
| description_len                 |                1.0614 |               0.0695 |                  1.2311 |
| title_len                       |                1.0502 |               0.0672 |                  1.0852 |
| format_count_clean              |                0.9704 |               0.1762 |                  0.7291 |
| is_data_product                 |                0.9661 |               0.2737 |                  0.7918 |

## 7.8 输出文件

- `analysis_v11_expecteduse.csv` / `analysis_v11_expecteduse.xlsx`
- `outputs/expecteduse_v11/tables/7_ExpectedUse_模型性能汇总.csv`
- `outputs/expecteduse_v11/tables/7_ExpectedUse_CatBoost_OOF分折指标.csv`
- `outputs/expecteduse_v11/tables/7_ExpectedUse_CatBoost特征重要性.csv`
- `outputs/expecteduse_v11/figures/7_最佳模型OOF实际使用与期望使用散点.png`
- `outputs/expecteduse_v11/figures/7_模型残差沉睡度分布.png`
- `outputs/expecteduse_v11/figures/7_CatBoost特征重要性Top25.png`
- `outputs/expecteduse_v11/figures/7_ExpectedUse模型OOF排序相关对比.png`

## 7.9 下一步

下一步应进入 8.2 模型残差沉睡度识别，正式解释 `DormantScore_model` 与 `model_residual_dormant_candidate`，再在 8.3 中与规则候选取交集形成高置信沉睡资产。
