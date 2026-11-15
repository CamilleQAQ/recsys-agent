<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# HSTU（生成式序列推荐） 技术档案

## 基本信息

- **ID**：`hstu`
- **分类**：召回模型（retrieval）
- **来源**：vendor：third_party/TorchEasyRec
- **许可证**：Apache-2.0
- **上游**：https://github.com/alibaba/TorchEasyRec
- **Commit**：`2815a9732bed014219ee89826dd4987b67ccf6b0`
- **源码**：`third_party/TorchEasyRec/tzrec/models/hstu.py`、`third_party/TorchEasyRec/tzrec/models/ultra_hstu.py`

## 原理与适用场景

- **原理**：层次化序列转导单元：带时间偏置的注意力直接建模长行为序列。
- **适用场景**：长序列召回/排序统一建模；短视频场景。

## 运行方式

- **依赖 / Python**：torch、tzrec；Python >=3.10（Linux）
- **训练入口**：python -m recsys.training.train_sequence --model hstu --dataset kuasar_small --max-epochs 20
- **推理入口**：python -m recsys.adapters evaluate --model hstu --checkpoint artifacts/kuasar_small/hstu.pt
- **数据格式**：行为序列 + 时间戳 + 候选特征
- **保存/加载**：artifacts/<dataset>/hstu.pt（HSTU + 序列配置 + provenance）
- **适配层**：HSTUAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：按 HSTU 的相对偏置 + pointwise gating 重写；1 epoch 冒烟训练、全量打分与 checkpoint 往返通过
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
