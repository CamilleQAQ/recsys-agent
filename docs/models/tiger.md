<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# TIGER（生成式检索） 技术档案

## 基本信息

- **ID**：`tiger`
- **分类**：召回模型（retrieval）
- **来源**：vendor：third_party/genrec
- **许可证**：MIT
- **上游**：https://github.com/phonism/genrec
- **Commit**：`2676b5b08e2efc0b873ab4c3d66882611a62ca2e`
- **源码**：`third_party/genrec/genrec/models/tiger.py`、`third_party/genrec/genrec/models/rqvae.py`、`third_party/genrec/config/tiger/amazon/tiger.gin`

## 原理与适用场景

- **原理**：先用 RQ-VAE 把物品压成层级语义 ID，再用 Transformer 以序列到序列方式生成候选 ID，并用 trie 约束解码。
- **适用场景**：生成式召回研究；与向量召回做对照；物品语义表示实验。

## 运行方式

- **依赖 / Python**：torch==2.6、transformers、gin_config、wandb、sentence_transformers；Python >=3.9（genrec 声明）
- **训练入口**：python -m recsys.training.train_tiger --dataset kuasar_small --rqvae-checkpoint artifacts/kuasar_small/rqvae.pt
- **推理入口**：python -m recsys.adapters evaluate --model tiger --checkpoint artifacts/kuasar_small/tiger.pt
- **数据格式**：交互序列 CSV + 物品 embedding（用于训练 RQ-VAE）→ 语义 ID 序列
- **保存/加载**：artifacts/<dataset>/tiger.pt（Transformer + 码表 + RQ-VAE 来源 + provenance）
- **适配层**：TigerAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：按上游 trie 约束生成式检索重写；训练/beam 搜索/warm-start 冒烟已通过，效果未调参
- **已知局限**：暂无记录（接入实验后补充）

## 逐项状态

| 源码 | 已核源码 | 适配 | 可运行 | 业务接入 | Agent | 已评估 |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
