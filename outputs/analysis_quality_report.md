# analysis_clean_master 清洗报告

## 1. 数据源

- 原始增强主表：`\\wsl.localhost\Ubuntu\home\konglingrui\Statistic\Statistics Collect\outputs\dataset_master_enriched.xlsx`
- 第三轮 checkpoint：`\\wsl.localhost\Ubuntu\home\konglingrui\Statistic\Statistics Collect\outputs\web_supplement_checkpoint.csv`
- 第三轮残留质量清单：`\\wsl.localhost\Ubuntu\home\konglingrui\Statistic\Statistics Collect\outputs\round3_residual_missing_or_unavailable.csv`
- 清洗策略：以 `dataset_master_enriched.xlsx` 为底表，按 `数据集ID` 用第三轮 checkpoint 覆盖网页增强字段；原始 master 不修改。

## 2. 样本口径

- 全量资源画像样本：9793 条
- scrape_status 分布：{'success': 9667, 'unavailable': 126}
- 不可用样本：126 条
- success 但核心六字段任一缺失：127 条
- 主分析样本：9540 条
- 建模候选样本：9540 条

## 3. 核心字段缺失

| 字段 | 全量缺失 | success 中缺失 |
|---|---:|---:|
| `download_formats` | 218 | 92 |
| `record_count` | 158 | 32 |
| `data_size` | 157 | 31 |
| `field_names` | 208 | 82 |
| `detail_spatial_scope` | 157 | 31 |
| `detail_time_scope` | 157 | 31 |

## 4. 质量风险标记

- low_field_count_flag：816 条
- suspicious_field_names_flag：727 条
- has_meaningful_time_scope = 0：2976 条
- date_order_anomaly：848 条
- recommended_names_suspicious_flag：3321 条
- rating_comment_disabled_for_model：9793 条

## 5. 网页字段覆盖同步诊断

以下数字表示清洗时 checkpoint 覆盖 master 后发生变化的行数，用于说明第三轮最终 checkpoint 是网页字段权威来源。

| 字段 | 覆盖后变化行数 |
|---|---:|
| `download_formats` | 0 |
| `api_need_apply` | 9667 |
| `format_count` | 9793 |
| `record_count` | 0 |
| `data_size` | 0 |
| `field_names` | 796 |
| `field_count` | 9793 |
| `has_time_field` | 9793 |
| `has_geo_field` | 9793 |
| `field_description_count` | 9793 |
| `has_standard_field_description` | 9793 |
| `has_data_sample` | 9793 |
| `sample_field_headers` | 0 |
| `recommended_dataset_count` | 9793 |
| `recommended_dataset_names` | 0 |
| `rating_score` | 9356 |
| `comment_count` | 9637 |
| `detail_spatial_scope` | 0 |
| `detail_time_scope` | 0 |
| `scrape_status` | 0 |
| `scraped_at` | 0 |
| `has_rdf` | 9667 |
| `has_xml` | 9667 |
| `has_csv` | 9667 |
| `has_json` | 9667 |
| `has_xlsx` | 9667 |
| `备注` | 2936 |
| `detail_url` | 0 |
| `scrape_error` | 0 |

## 6. 残留质量清单 issue 计数

| issue | 条数 |
|---|---:|
| `field_terms_too_few` | 816 |
| `missing_data_size` | 157 |
| `missing_detail_spatial_scope` | 157 |
| `missing_detail_time_scope` | 157 |
| `missing_download_formats` | 218 |
| `missing_field_names` | 208 |
| `missing_record_count` | 158 |
| `suspicious_field_names` | 727 |
| `unavailable` | 126 |

## 7. 输出文件

- `analysis_clean_master.csv` / `analysis_clean_master.xlsx`：保留 9793 条全量行并增加清洗字段
- `analysis_main_sample.csv` / `analysis_main_sample.xlsx`：仅保留主分析样本
- `analysis_clean_field_dictionary.csv`：新增清洗字段说明

## 8. 建模口径提醒

- `field_terms_too_few`、`suspicious_field_names`、`has_meaningful_time_scope = 0` 是风险标记，不默认剔除。
- `rating_score` 与 `comment_count` 原始字段保留，但主模型禁用。
- `recommended_dataset_names` 原始字段保留，但疑似污染，推荐图网络暂不直接使用。
- `scrape_status` 只用于样本筛选与质量说明，不进入 ActualUse、PotentialUse 或 ExpectedUse 主模型。
