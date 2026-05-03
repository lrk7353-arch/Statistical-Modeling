# 7.10 ExpectedUse 残差口径审计与高置信分层

## 1. 审计目的

第 7 章多模型 OOF 表明 XGBoost、LightGBM、HGB、RandomForest 都能较好预测 ActualUse。严格模型残差候选数量很少，因此模型残差不应作为唯一沉睡资产筛选器，而应作为规则候选的经验验证和置信度增强指标。

## 2. 特征泄漏检查

- 进入 ExpectedUse 模型的特征数：43。
- 命中泄漏规则的特征数：0。

泄漏规则检查结果见 `tables/7.10_模型特征泄漏检查.csv`。若命中数为 0，说明当前 ExpectedUse 模型没有直接使用浏览、下载、调用、ActualUse、PotentialUse、DormantScore、ID、URL 或采集过程变量。

## 3. no-age 稳健性模型

去掉以下时间暴露变量后，重新训练 XGBoost OOF：

```text
publication_age_days_clean
maintenance_span_days_clean
update_recency_days_clean
```

模型表现对比：

| model                       |     r2 |    mae |   rmse |   spearman_corr |
|:----------------------------|-------:|-------:|-------:|----------------:|
| XGBoost_full                | 0.9186 | 0.0565 | 0.0824 |          0.9577 |
| HistGradientBoosting_no_age | 0.8996 | 0.063  | 0.0915 |          0.9478 |
| XGBoost_no_age              | 0.8994 | 0.0639 | 0.0916 |          0.9476 |
| LightGBM_no_age             | 0.8992 | 0.0629 | 0.0917 |          0.9474 |
| CatBoost_no_age             | 0.8987 | 0.065  | 0.0919 |          0.9473 |

no-age 严格残差候选数量：

| model                       |   strict_candidate_count |
|:----------------------------|-------------------------:|
| XGBoost_no_age              |                        4 |
| LightGBM_no_age             |                        5 |
| HistGradientBoosting_no_age |                        4 |
| CatBoost_no_age             |                        3 |

若 no-age 模型性能明显下降，说明发布时间和维护时间确实解释了大量累计使用表现；这不是泄漏，而是累计使用量口径的时间暴露效应。

## 4. DormantScore_model 分位数

|   quantile |   DormantScore_model |
|-----------:|---------------------:|
|       0    |              -0.7563 |
|       0.01 |              -0.2822 |
|       0.05 |              -0.1473 |
|       0.1  |              -0.0936 |
|       0.2  |              -0.0454 |
|       0.25 |              -0.0309 |
|       0.5  |               0.0077 |
|       0.75 |               0.0443 |
|       0.8  |               0.0545 |
|       0.9  |               0.0855 |
|       0.95 |               0.1129 |
|       0.99 |               0.1853 |
|       1    |               0.3806 |

## 5. 残差与高置信候选分层

| candidate_layer                     | definition                                                            |   count |   share |   intersection_with_rule |   jaccard_with_rule |
|:------------------------------------|:----------------------------------------------------------------------|--------:|--------:|-------------------------:|--------------------:|
| strict_model_residual_candidate     | 严格模型残差候选：ExpectedUse>=0.70, ActualUse<=0.50, DormantScore_model>=0.30 |       1 |  0.0001 |                        1 |              0.0009 |
| residual_top5_lowuse_candidate      | 低使用样本中，全样本模型残差 Top 5%                                                 |     308 |  0.0323 |                       90 |              0.0677 |
| residual_top10_lowuse_candidate     | 低使用样本中，全样本模型残差 Top 10%                                                |     587 |  0.0615 |                      152 |              0.0983 |
| residual_top20_lowuse_candidate     | 低使用样本中，全样本模型残差 Top 20%                                                |    1106 |  0.1159 |                      255 |              0.13   |
| residual_top30_lowuse_candidate     | 低使用样本中，全样本模型残差 Top 30%                                                |    1572 |  0.1648 |                      356 |              0.153  |
| rule_candidate_model_enhanced_top10 | 规则候选内部模型残差 Top 10%                                                    |     112 |  0.0117 |                      112 |              0.1008 |
| rule_candidate_model_enhanced_top20 | 规则候选内部模型残差 Top 20%                                                    |     223 |  0.0234 |                      223 |              0.2007 |
| rule_candidate_model_enhanced_top30 | 规则候选内部模型残差 Top 30%                                                    |     334 |  0.035  |                      334 |              0.3006 |
| multi_model_strict_residual_any     | 任一模型严格残差候选                                                            |       5 |  0.0005 |                        3 |              0.0027 |
| multi_model_strict_residual_2plus   | 至少两个模型同时识别的严格残差候选                                                     |       4 |  0.0004 |                        2 |              0.0018 |
| multi_model_strict_residual_3plus   | 至少三个模型同时识别的严格残差候选                                                     |       4 |  0.0004 |                        2 |              0.0018 |

## 6. 后续聚类样本池

- 第 9 章聚类主样本池：`rule_dormant_candidate = 1`，共 1111 条。
- 高优先级/高置信分析池：`rule_candidate_model_enhanced_top20 = 1`，共 223 条。
- 极严格模型残差候选：`strict_model_residual_candidate = 1`，共 1 条，仅适合作为极端案例或附录。

因此，聚类不使用严格交集样本，而以 1111 条规则候选为主；模型残差用于置信度排序、案例优先级和稳健性解释。

## 7. 输出文件

- `analysis_v11_expecteduse_audited.csv` / `analysis_v11_expecteduse_audited.xlsx`
- `tables/7.10_模型特征泄漏检查.csv`
- `tables/7.10_no_age模型OOF指标.csv`
- `tables/7.10_DormantScore_model分位数.csv`
- `tables/7.10_残差与高置信候选分层汇总.csv`
- `tables/7.10_候选分层画像_领域类型开放属性.csv`
- `figures/7.10_模型残差分布与分位阈值.png`
- `figures/7.10_残差与高置信候选分层数量.png`
- `figures/7.10_多模型严格残差一致性.png`
