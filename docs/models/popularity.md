<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# Popularity 技术档案

## 基本信息

- **ID**：`popularity`
- **分类**：召回模型（retrieval）
- **来源**：本项目自研
- **许可证**：项目自有
- **上游**：本项目（自研基线）
- **源码**：`src/recsys/models/popularity.py`

## 原理与适用场景

- **原理**：按训练集交互次数对物品排序，推荐全局最热且用户未看过的 Top-K。
- **适用场景**：基线对照；任何数据集的冷启动兜底。

## 运行方式

- **依赖 / Python**：无；Python >=3.11
- **训练入口**：无需训练；统计来自 PreparedData.train_interactions
- **推理入口**：python -m recsys.experiments.compare（或适配层 PopularityAdapter）
- **数据格式**：PreparedData（连续整数 user_idx/item_idx/timestamp）
- **保存/加载**：无参数，不需要保存；每次由训练集现算
- **适配层**：PopularityAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：无参数基线，已接入统一召回接口、推荐链路和 Rec Agent（无 checkpoint，现算热度）
- **已知局限**：只用训练集热度，无法个性化；冷门物品几乎不会被推荐，Coverage 明显偏低。

## 逐项状态

| 源码 | 已核源码 | 适配 | 可运行 | 业务接入 | Agent | 已评估 |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
