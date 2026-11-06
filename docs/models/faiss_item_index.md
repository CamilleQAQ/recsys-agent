<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# FAISS 物品索引封装（本项目） 技术档案

## 基本信息

- **ID**：`faiss_item_index`
- **分类**：基础设施（infrastructure）
- **来源**：本项目自研
- **许可证**：项目自有
- **上游**：本项目（自研）
- **源码**：`src/recsys/retrieval/faiss_index.py`、`src/recsys/evaluation/faiss_evaluator.py`

## 原理与适用场景

- **原理**：把双塔用户/物品向量写入 FAISS IndexFlatIP，校验 L2 归一化，并与暴力检索做一致性对比。
- **适用场景**：全量召回评测、检索性能基准、后续在线 serving 的检索底座。

## 运行方式

- **依赖 / Python**：faiss-cpu、numpy、torch；Python >=3.11
- **训练入口**：无（构建索引即 build_faiss_index）
- **推理入口**：python -m recsys.evaluation.faiss_evaluator --checkpoint artifacts/<dataset>/two_tower.pt
- **数据格式**：L2 归一化的 float32 向量（用户/物品同维度）
- **保存/加载**：FaissItemIndex 可落盘（index.write / read_index）
- **适配层**：FaissItemIndex

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：已实现暴力检索一致性校验（top_k_agreement）与性能基准
- **已知局限**：只封装 IndexFlatIP（精确检索）；数据量大时要换 IVF/HNSW 并重新校准召回率；索引不含物品 ID 元数据，只能靠向量行号对应。

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
