<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# Two-Tower（ID 双塔） 技术档案

## 基本信息

- **ID**：`two_tower`
- **分类**：召回模型（retrieval）
- **来源**：本项目自研
- **许可证**：项目自有
- **上游**：本项目（自研）
- **源码**：`src/recsys/models/two_tower.py`、`src/recsys/training/trainer.py`

## 原理与适用场景

- **原理**：用户塔/物品塔各自编码并 L2 归一化，用 batch 内负样本的交叉熵训练，内积即余弦相似度。
- **适用场景**：召回主干；可对接 FAISS 做全量 Top-K。

## 运行方式

- **依赖 / Python**：torch、faiss-cpu（可选）；Python >=3.11
- **训练入口**：python -m recsys.training.trainer --dataset kuairec
- **推理入口**：python -m recsys.evaluation.evaluator --checkpoint artifacts/<dataset>/two_tower.pt
- **数据格式**：PreparedData；物品词表必须与 checkpoint 一致
- **保存/加载**：artifacts/<dataset>/two_tower.pt（自带 model_config 与 split_strategy 校验）
- **适配层**：TwoTowerAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：既有训练、评测、FAISS 对照；Phase 1 接入统一推荐链路与 Rec Agent，默认双路之一
- **已知局限**：只吃 ID 特征，KuaiSAR 的行为/上下文特征尚未接入；batch 内负样本会带来假负例（mask_false_negatives 可缓解但无法消除）。

## 逐项状态

| 源码 | 已核源码 | 适配 | 可运行 | 业务接入 | Agent | 已评估 |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
