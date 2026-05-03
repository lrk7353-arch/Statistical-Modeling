# 上海公共数据资产统计建模项目

本仓库用于保存“上海公共数据目录增强、沉睡资产识别与统计建模”项目的文档、采集/清洗/建模脚本和阶段性分析输出。

## 仓库结构

```text
文档与启动/
  项目启动文档、V1.0/V1.1 分析方案、竞赛材料与辅助图片

Statistics Collect/
  数据采集、重采、清洗、EDA、ActualUse、PotentialUse、ExpectedUse、沉睡识别等脚本

outputs/
  清洗主表、分析特征表、模型结果、质量审计、EDA 图表、8.2/8.3 结果与候选池明细
```

## 当前主线数据文件

- `outputs/analysis_clean_master.csv`：全量画像样本，保留 9793 条目录资源。
- `outputs/analysis_main_sample.csv`：主分析样本，9540 条。
- `outputs/analysis_v11_features.csv`：V1.1 特征表。
- `outputs/analysis_v11_expecteduse.csv`：ExpectedUse 多模型训练输出。
- `outputs/analysis_v11_expecteduse_audited.csv`：ExpectedUse 审计后主表，后续 8.2/8.3/9 章建议从此文件继续。

## 已完成的关键阶段

- 数据清洗与三层样本口径划分。
- EDA 与平台资源画像。
- 第 5 章：ActualUse 实际使用强度构造。
- 第 6 章：PotentialUse 潜在价值构造。
- 第 7 章：ExpectedUse 期望使用模型构造与残差审计。
- 第 8.1：规则沉睡度识别。
- 第 8.2：模型残差沉睡度识别。
- 第 8.3：高置信沉睡资产分层。

## 当前沉睡资产识别口径

- `rule_dormant_candidate`：1111 条，主沉睡资产识别池，也是第 9 章聚类主样本。
- `rule_candidate_model_enhanced_top20`：223 条，高优先级 / 高置信分析池。
- `strict_model_residual_candidate`：1 条，极端模型残差案例。
- 任一 no-age / 多模型严格残差极端案例：5 条。

## 下一步建议

从 `outputs/analysis_v11_expecteduse_audited.csv` 和 `outputs/dormant_82_83_v11/` 继续：

1. 第 9 章：以 1111 条 `rule_dormant_candidate` 为主样本做 GMM/BGMM 聚类分型。
2. 使用 223 条 `rule_candidate_model_enhanced_top20` 作为重点案例池。
3. 严格模型残差 1-5 条只作为极端验证案例或附录案例。

## 环境

Python 依赖见：

```text
Statistics Collect/requirements.txt
```

已排除上传内容：运行日志、备份目录、`__pycache__`、`.pyc`、Zone.Identifier 等临时噪声文件。
