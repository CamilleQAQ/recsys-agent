<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# BERT4Rec 技术档案

## 基本信息

- **ID**：`bert4rec`
- **分类**：排序模型（ranking）
- **来源**：vendor：third_party/BERT4Rec
- **许可证**：Apache-2.0
- **上游**：https://github.com/FeiSun/BERT4Rec
- **Commit**：`615eaf2004abecda487a38d5b0c72f3dcfcae5b3`
- **源码**：`third_party/BERT4Rec/modeling.py`、`third_party/BERT4Rec/run.py`、`third_party/BERT4Rec/gen_data_fin.py`

## 原理与适用场景

- **原理**：双向 Transformer + 掩码物品预测（Cloze），从两侧上下文推断被遮住的物品。
- **适用场景**：序列推荐；对双向上下文敏感的场景。

## 运行方式

- **依赖 / Python**：tensorflow==1.12、numpy、six；Python 2.7/3.6（TF1.12）
- **训练入口**：python -m recsys.training.train_sequence --model bert4rec --dataset kuasar_small --max-epochs 20
- **推理入口**：python -m recsys.adapters evaluate --model bert4rec --checkpoint artifacts/kuasar_small/bert4rec.pt
- **数据格式**：用户物品序列 → masked TFRecord + vocab/history pickle
- **保存/加载**：artifacts/<dataset>/bert4rec.pt（双向 Transformer + 序列配置 + provenance）
- **适配层**：Bert4RecAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：按上游 BERT4Rec 的 Cloze 目标重写为 PyTorch；1 epoch 冒烟训练、末尾 MASK 推理与 checkpoint 往返通过
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
