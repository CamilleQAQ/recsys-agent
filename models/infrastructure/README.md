<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# 基础设施（infrastructure）

本页由 `models/zoo.toml` 生成；源码仍在 `src/`（自研）或 `third_party/`（vendor），这里只做分类索引，不搬运代码，避免破坏上游相对导入。

让召回跑起来的检索与向量能力，本身不是推荐模型。

> FAISS 已接入并做过与暴力检索的一致性校验；TFRS 的 factorized_top_k 只作跨框架对照。

| id | 模型 | 来源 | 关键源码 | 完整度 |
| --- | --- | --- | --- | --- |
| `faiss` | FAISS（向量检索） | vendor:faiss | `third_party/faiss/faiss/IndexFlat.cpp` | 可直接运行 |
| `faiss_item_index` | FAISS 物品索引封装（本项目） | 自研 | `src/recsys/retrieval/faiss_index.py` | 可直接运行 |
| `tfrs_factorized_top_k` | TFRS factorized_top_k | vendor:recommenders | `third_party/recommenders/tensorflow_recommenders/layers/factorized_top_k.py` | 受依赖/权重限制 |
