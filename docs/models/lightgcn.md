<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# LightGCN 技术档案

## 基本信息

- **ID**：`lightgcn`
- **分类**：召回模型（retrieval）
- **来源**：vendor：third_party/RecBole
- **许可证**：MIT（RecBole）/ GPL-3.0（MMRec）
- **上游**：https://github.com/RUCAIBox/RecBole
- **Commit**：`7b02be5ec80a88310f2d04a27a82adfcbb5dc211`
- **源码**：`third_party/RecBole/recbole/model/general_recommender/lightgcn.py`、`third_party/MMRec/src/models/lightgcn.py`

## 原理与适用场景

- **原理**：在用户-物品二部图上做多层无特征线性传播并加权求和，去掉非线性与自连接。
- **适用场景**：协同过滤召回；交互图稠密、无侧信息的场景。

## 运行方式

- **依赖 / Python**：torch、recbole；Python >=3.10
- **训练入口**：python -m recsys.training.train_lightgcn --dataset kuasar_small --epochs 10
- **推理入口**：python -m recsys.adapters evaluate --model lightgcn --checkpoint artifacts/kuasar_small/lightgcn.pt
- **数据格式**：RecBole atomic 文件（.inter/.user/.item，带类型表头）
- **保存/加载**：artifacts/<dataset>/lightgcn.pt（权重 + 模型配置 + 图边 + provenance）
- **适配层**：LightGCNAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：按上游前向逻辑重写为纯 PyTorch BPR 实现；1 epoch/2 batch 冒烟训练与 checkpoint 往返通过
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
