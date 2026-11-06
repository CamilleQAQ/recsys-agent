<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# FAISS（向量检索） 技术档案

## 基本信息

- **ID**：`faiss`
- **分类**：基础设施（infrastructure）
- **来源**：vendor：third_party/faiss
- **许可证**：MIT
- **上游**：https://github.com/facebookresearch/faiss
- **Commit**：`9acf8b0c16a51aafa12b9d9bcf13998b9138a8c5`
- **源码**：`third_party/faiss/faiss/IndexFlat.cpp`、`third_party/faiss/faiss/IndexIVF.cpp`、`third_party/faiss/faiss/IndexHNSW.cpp`、`third_party/faiss/faiss/IndexPQ.cpp`

## 原理与适用场景

- **原理**：提供 Flat/IVF/PQ/HNSW/LSH 等近邻索引与聚类实现，对上游 embedding 做 Top-K 检索。
- **适用场景**：召回阶段的候选生成；大规模向量打分。本身不是推荐模型。

## 运行方式

- **依赖 / Python**：faiss-cpu；Python >=3.10
- **训练入口**：Index.train(xb) / add(xb)（建索引即可，无需拟合）
- **推理入口**：Index.search(xq, k)
- **数据格式**：float32 N×d 向量；索引必须与建库时的 embedding 模型配套
- **保存/加载**：index.write('x.faiss') / faiss.read_index('x.faiss')
- **适配层**：尚未接入

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：本项目通过 recsys.retrieval.faiss_index 调用已安装的 faiss-cpu，仓库源码仅作参考，无需本地编译
- **已知局限**：仓库源码需 C++ 编译，本项目实际用 pip 的 faiss-cpu；索引与 embedding 必须同版本，换模型要重建索引。

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
