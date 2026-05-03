# analysis_v11_features 特征构造报告

## 1. 输入与输出

- 输入：`\\wsl.localhost\Ubuntu\home\konglingrui\Statistic\Statistics Collect\outputs\analysis_main_sample.csv`
- 输出：`\\wsl.localhost\Ubuntu\home\konglingrui\Statistic\Statistics Collect\outputs\analysis_v11_features.csv` / `analysis_v11_features.xlsx`
- 行数：9540
- 字段数：164

## 2. ActualUse PCA 诊断

| 资源类型 | 样本数 | 输入变量 | 解释方差占比 | 方向校正后相关 | 载荷 |
|---|---:|---|---:|---:|---|
| 数据产品 | 4623 | `view_score;download_score` | 0.9155 | 0.9996 | `view_score=0.707107;download_score=0.707107` |
| 数据接口 | 4917 | `view_score;api_call_score` | 0.7858 | 0.9987 | `view_score=0.707107;api_call_score=0.707107` |

## 3. CRITIC 权重

| 维度 | CRITIC 权重 | 熵权法权重 |
|---|---:|---:|
| `topic_public_value_score` | 0.0751 | 0.0262 |
| `semantic_clarity_score` | 0.1637 | 0.1952 |
| `data_richness_score` | 0.1263 | 0.2479 |
| `machine_readability_score` | 0.2474 | 0.1909 |
| `timeliness_score` | 0.1614 | 0.1694 |
| `spatiotemporal_capability_score` | 0.1412 | 0.1091 |
| `combinability_score` | 0.0848 | 0.0613 |

## 4. 规则沉睡识别

- rule_dormant_candidate：1111 条
- 四象限分布：{'高潜高用': 2675, '低潜低用': 2674, '低潜高用': 2096, '高潜低用': 2095}

规则阈值：

```text
PotentialUse_CRITIC_percentile >= 0.70
ActualUse_type_percentile <= 0.50
DormantScore_rule >= 0.30
```

## 5. 建模口径提醒

- 本文件已生成 V1.1 规则潜力与规则沉睡度，但尚未生成 `ExpectedUse_model` 和 `DormantScore_model`。
- 下一步 ExpectedUse 必须使用 K-fold out-of-fold 预测。
- 禁止把浏览量、下载量、接口调用量、ActualUse、DormantScore、scrape_status、ID、URL、评分评论等字段放入 ExpectedUse 特征矩阵。
