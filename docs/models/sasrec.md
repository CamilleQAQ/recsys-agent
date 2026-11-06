<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# SASRec（本项目实现） 技术档案

## 基本信息

- **ID**：`sasrec`
- **分类**：排序模型（ranking）
- **来源**：本项目自研
- **许可证**：项目自有（对照实现 Apache-2.0）
- **上游**：本项目（自研）；对照 https://github.com/pmixer/SASRec.pytorch
- **Commit**：`26a8fa880f312c3700f8dcca5512d5854b257f4e`
- **源码**：`src/recsys/models/sasrec.py`、`src/recsys/training/train_sasrec.py`、`third_party/SASRec.pytorch/python/model.py`

## 原理与适用场景

- **原理**：单向（因果）自注意力编码用户历史，按「下一个物品」目标训练，对全量物品打分排序。
- **适用场景**：序列推荐主力模型；召回或排序均可，本项目按下一物品预测评测。

## 运行方式

- **依赖 / Python**：torch；Python >=3.11
- **训练入口**：python -m recsys.training.train_sasrec --dataset kuairec
- **推理入口**：python -m recsys.evaluation.sasrec_evaluator --checkpoint artifacts/<dataset>/sasrec.pt
- **数据格式**：PreparedData → 右对齐/右补齐的 item 序列（token 0 为 padding）
- **保存/加载**：artifacts/<dataset>/sasrec.pt（含 max_seq_len 等结构配置）
- **适配层**：SASRecAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：训练/早停/评测（含 train-history-only 对照）已实现
- **已知局限**：只保留最近 max_seq_len 个物品，超长历史被截断；无侧信息；按用户构造样本，用户量级大时单 epoch 成本高。

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
