<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# RQ-VAE / Semantic ID（本项目适配） 技术档案

## 基本信息

- **ID**：`semantic_id_rqvae`
- **分类**：召回模型（retrieval）
- **来源**：本项目自研
- **许可证**：项目自有（上游参考 genrec / TorchEasyRec，MIT/Apache-2.0）
- **上游**：本项目（自研适配）；参考 https://github.com/phonism/genrec
- **Commit**：`2676b5b08e2efc0b873ab4c3d66882611a62ca2e`
- **源码**：`src/recsys/models/rqvae.py`、`src/recsys/training/train_rqvae.py`、`src/recsys/models/tiger.py`

## 原理与适用场景

- **原理**：编码物品内容向量 → 3 级残差向量量化 → 层级 Semantic ID；TIGER 消费码本做生成式召回。
- **适用场景**：生成式召回的物品离散表示；语义 ID 构建、前缀约束解码。

## 运行方式

- **依赖 / Python**：torch、numpy；Python >=3.11
- **训练入口**：python -m recsys.training.train_rqvae --dataset kuasar_small --epochs 8
- **推理入口**：from recsys.models.rqvae import load_rqvae, SemanticIDIndex
- **数据格式**：PreparedData.item_features → 固定随机投影内容向量 [num_items, 32]
- **保存/加载**：artifacts/<dataset>/rqvae.pt（编码器/解码器/码本 + 全量 codes + provenance）
- **适配层**：RQVAE 组件（非 RecommenderAdapter）

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：内容向量来自物品元数据（非视觉/文本 embedding）；训练与码表导出已冒烟，TIGER 已消费
- **已知局限**：本地 KuaiSAR 无封面/文本 embedding，语义 ID 的“多模态”输入实际是元数据向量；码本碰撞会限制可召回物品集合。

## 逐项状态

| 源码 | 已核源码 | 适配 | 可运行 | 业务接入 | Agent | 已评估 |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
