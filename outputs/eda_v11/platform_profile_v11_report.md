# EDA V1.1 平台资源画像增强模块（对应方案 4.1）

## 1. 为什么补这一节

原 `eda_v11_report.md` 已经覆盖 4.2 长尾、4.3 转化、4.4 分组检验和 4.5 特征关系探索；4.1 平台资源画像已有概要，但缺少完整供给结构、交叉结构、质量风险和全量样本 vs 主样本代表性检查。因此本模块作为 4.1 的专项补充，不重复 4.3-4.5。

## 2. 样本保留与代表性

| sample_layer          |   rows |   share_of_full |
|:----------------------|-------:|----------------:|
| 全量资源画像样本              |   9793 |          1      |
| scrape_status=success |   9667 |          0.9871 |
| 主分析样本                 |   9540 |          0.9742 |
| 不可用或核心缺失              |    253 |          0.0258 |

![样本保留口径](figures/4.1_样本保留口径.png)

全量样本与主样本的数据领域分布差异很小，说明剔除 253 条不可用或核心缺失样本没有明显改变平台总体结构。

![数据领域分布全量对比主样本](figures/4.1_数据领域分布全量对比主样本.png)

## 3. 4.1 完成清单

| v11_4_1_item         | table                                           | figure                       | status   |
|:---------------------|:------------------------------------------------|:-----------------------------|:---------|
| 部门资源供给 Top 20/Pareto | platform_department_pareto.csv                  | 4.1_提供部门资源供给Pareto图Top30.png | done     |
| 部门 x 数据资源类型          | platform_department_by_resource_type_top20.csv  | 4.1_部门与资源类型交叉Top20.png       | done     |
| 部门 x 开放属性            | platform_department_by_open_attribute_top20.csv | 4.1_部门与开放属性交叉Top20.png       | done     |
| 部门 x 使用表现            | platform_department_usage_summary.csv           | 4.1_部门使用表现中位数Top30.png       | done     |
| 数据领域完整分布             | platform_domain_full_vs_main.csv                | 4.1_数据领域分布全量对比主样本.png        | done     |
| 数据领域 x 开放属性          | platform_domain_by_open_attribute.csv           | 4.1_数据领域与开放属性交叉.png          | done     |
| 数据领域 x 数据资源类型        | platform_domain_by_resource_type.csv            | 4.1_数据领域与资源类型交叉.png          | done     |
| 数据领域 x 更新频率          | platform_domain_by_update_frequency.csv         | 4.1_数据领域与更新频率交叉.png          | done     |
| 开放条件 Top 30          | platform_open_condition_top30.csv               | 4.1_开放条件Top30.png            | done     |
| API 申请需求             | platform_api_need_apply.csv                     | 4.1_API申请需求分布.png            | done     |
| 格式数量分布               | platform_format_count.csv                       | 4.1_格式数量分布.png               | done     |
| 格式覆盖与共现              | platform_format_cooccurrence.csv                | 4.1_下载格式共现矩阵.png             | done     |
| 数据量/数据大小/字段质量        | platform_quality_flag_distribution.csv          | 4.1_数据内容规模与字段质量分布.png        | done     |
| 样例可见性/字段说明/质量风险      | platform_quality_flag_distribution.csv          | 4.1_质量风险标记分布.png             | done     |
| 空间行政层级               | platform_spatial_admin_level.csv                | 4.1_空间行政层级分布.png             | done     |
| 详情页空间范围              | platform_detail_spatial_scope_top30.csv         | 4.1_详情页空间范围Top30.png         | done     |
| 时间范围有效性              | platform_time_scope_validity.csv                | 4.1_详情页时间范围有效性.png           | done     |
| 发布时间与更新时效            | platform_spacetime_flag_distribution.csv        | 4.1_发布时间与更新时效分布.png          | done     |
| 全量样本 vs 主样本分布对比      | platform_domain_full_vs_main.csv                | 4.1_数据领域分布全量对比主样本.png        | done     |

## 4. 资源供给主体画像

提供部门 Top 10：

| 数据资源提供部门    |   count |   share |   cumulative_share |
|:------------|--------:|--------:|-------------------:|
| 上海市虹口区人民政府  |     671 |  0.0685 |             0.0685 |
| 上海市黄浦区人民政府  |     598 |  0.0611 |             0.1296 |
| 上海市宝山区人民政府  |     569 |  0.0581 |             0.1877 |
| 上海市徐汇区人民政府  |     557 |  0.0569 |             0.2446 |
| 上海市大数据中心    |     535 |  0.0546 |             0.2992 |
| 上海市青浦区人民政府  |     527 |  0.0538 |             0.353  |
| 上海市浦东新区人民政府 |     510 |  0.0521 |             0.4051 |
| 上海市嘉定区人民政府  |     507 |  0.0518 |             0.4569 |
| 上海市普陀区人民政府  |     424 |  0.0433 |             0.5002 |
| 上海市闵行区人民政府  |     399 |  0.0407 |             0.5409 |

![提供部门资源供给Pareto图Top30](figures/4.1_提供部门资源供给Pareto图Top30.png)

![部门与资源类型交叉Top20](figures/4.1_部门与资源类型交叉Top20.png)

![部门与开放属性交叉Top20](figures/4.1_部门与开放属性交叉Top20.png)

![部门使用表现中位数Top30](figures/4.1_部门使用表现中位数Top30.png)

## 5. 数据领域结构画像

数据领域 Top 10：

| 数据领域   |   full_count |   full_share |   main_count |   main_share |
|:-------|-------------:|-------------:|-------------:|-------------:|
| 民生服务   |         2059 |       0.2103 |         2049 |       0.2148 |
| 经济建设   |         1573 |       0.1606 |         1544 |       0.1618 |
| 城市建设   |         1448 |       0.1479 |         1370 |       0.1436 |
| 教育科技   |          787 |       0.0804 |          775 |       0.0812 |
| 卫生健康   |          631 |       0.0644 |          623 |       0.0653 |
| 文化休闲   |          600 |       0.0613 |          515 |       0.054  |
| 公共安全   |          561 |       0.0573 |          557 |       0.0584 |
| 资源环境   |          543 |       0.0554 |          538 |       0.0564 |
| 机构团体   |          478 |       0.0488 |          468 |       0.0491 |
| 道路交通   |          330 |       0.0337 |          326 |       0.0342 |

![数据领域与开放属性交叉](figures/4.1_数据领域与开放属性交叉.png)

![数据领域与资源类型交叉](figures/4.1_数据领域与资源类型交叉.png)

![数据领域与更新频率交叉](figures/4.1_数据领域与更新频率交叉.png)

## 6. 开放便利性画像

- 资源类型分布：{'数据接口': 4917, '数据产品': 4623}
- 开放属性分布：{'无条件开放': 5522, '有条件开放': 4018}
- format_count 分布：{'5.0': 4693, '2.0': 3845, '4.0': 494, '1.0': 298, '3.0': 210}

![开放属性分布](figures/4.1_开放属性分布.png)

![开放条件Top30](figures/4.1_开放条件Top30.png)

![API申请需求分布](figures/4.1_API申请需求分布.png)

![格式数量分布](figures/4.1_格式数量分布.png)

![下载格式覆盖](figures/4.1_下载格式覆盖.png)

![下载格式共现矩阵](figures/4.1_下载格式共现矩阵.png)

## 7. 数据内容与质量画像

质量风险与可见性标记：

| flag                              |   count |   share |
|:----------------------------------|--------:|--------:|
| has_data_sample                   |    9206 |  0.965  |
| has_standard_field_description    |    7412 |  0.7769 |
| low_field_count_flag              |     729 |  0.0764 |
| suspicious_field_names_flag       |     722 |  0.0757 |
| recommended_names_suspicious_flag |    3305 |  0.3464 |
| date_order_anomaly                |     840 |  0.0881 |

![数据内容规模与字段质量分布](figures/4.1_数据内容规模与字段质量分布.png)

![字段数量分箱分布](figures/4.1_字段数量分箱分布.png)

![质量风险标记分布](figures/4.1_质量风险标记分布.png)

## 8. 时空与更新画像

空间行政层级：

| spatial_admin_level   |   count |   share |
|:----------------------|--------:|--------:|
| 区级                    |    6482 |  0.6795 |
| 市级                    |    3029 |  0.3175 |
| 其他/不确定                |      27 |  0.0028 |
| 跨区域/国家级               |       2 |  0.0002 |

时间范围有效性：

| time_scope_validity   |   count |   share |
|:----------------------|--------:|--------:|
| 有效内容时间范围              |    6739 |  0.7064 |
| 无有效内容时间范围             |    2801 |  0.2936 |

![空间行政层级分布](figures/4.1_空间行政层级分布.png)

![详情页空间范围Top30](figures/4.1_详情页空间范围Top30.png)

![详情页时间范围有效性](figures/4.1_详情页时间范围有效性.png)

![时空字段与日期异常标记](figures/4.1_时空字段与日期异常标记.png)

![发布时间与更新时效分布](figures/4.1_发布时间与更新时效分布.png)

## 9. 旧图中文别名

为了方便后续放入论文或 PPT，已有 EDA 英文图名保留，同时复制了中文别名：

| old_name                             | chinese_alias                      | status   |
|:-------------------------------------|:-----------------------------------|:---------|
| main_domain_distribution.png         | 原EDA_主分析样本数据领域分布.png               | copied   |
| top_departments.png                  | 原EDA_提供部门Top30.png                 | copied   |
| type_open_stacked.png                | 原EDA_资源类型与开放属性交叉分布.png             | copied   |
| update_frequency.png                 | 原EDA_更新频率分布.png                    | copied   |
| format_prevalence.png                | 原EDA_下载格式覆盖.png                    | copied   |
| usage_log_histograms.png             | 原EDA_使用指标log分布.png                 | copied   |
| usage_lorenz_curves.png              | 原EDA_使用表现Lorenz曲线.png              | copied   |
| conversion_boxplot_by_type.png       | 原EDA_转化率箱线图.png                    | copied   |
| actualuse_potentialuse_quadrants.png | 原EDA_ActualUse与PotentialUse四象限.png | copied   |
| dormant_score_distribution.png       | 原EDA_规则沉睡度分布.png                   | copied   |
| spearman_correlation_heatmap.png     | 原EDA_主要特征Spearman相关矩阵.png          | copied   |

## 10. 小结

- 4.1 现在已从概要版补成完整画像层，覆盖供给主体、领域结构、开放便利性、内容质量、时空更新和样本代表性。
- 4.3/4.4/4.5 已在 `eda_v11_report.md` 中充分体现，不需要重复单独生成。
- 这部分完成后，进入 ExpectedUse OOF 模型训练更稳，因为论文的数据来源与描述统计部分已经有完整支撑。
