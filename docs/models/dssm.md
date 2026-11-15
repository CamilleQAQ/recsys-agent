<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# DSSM（双塔） 技术档案

## 基本信息

- **ID**：`dssm`
- **分类**：召回模型（retrieval）
- **来源**：vendor：third_party/TorchEasyRec
- **许可证**：Apache-2.0
- **上游**：https://github.com/alibaba/TorchEasyRec
- **Commit**：`2815a9732bed014219ee89826dd4987b67ccf6b0`
- **源码**：`third_party/TorchEasyRec/tzrec/models/dssm.py`、`third_party/TorchEasyRec/tzrec/models/dssm_v2.py`

## 原理与适用场景

- **原理**：用户侧/物品侧特征分别过 MLP 成向量，batch 内负采样 + 采样 softmax 训练。
- **适用场景**：工业级双塔召回参考实现（含特征工程与在线 serving 格式）。

## 运行方式

- **依赖 / Python**：torch、tzrec；Python >=3.10（上游 wheel 面向 3.10-3.12 Linux）
- **训练入口**：python -m recsys.training.train_dssm --dataset kuasar_small --epochs 10
- **推理入口**：python -m recsys.adapters evaluate --model dssm --checkpoint artifacts/kuasar_small/dssm.pt
- **数据格式**：CSV/Parquet/ODPS + pipeline protobuf 声明特征、标签与模型
- **保存/加载**：artifacts/<dataset>/dssm.pt（TwoTowerModel + 特征塔配置 + provenance）
- **适配层**：DSSMAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：复用 Two-Tower 的 MLP 特征塔实现 DSSM 结构；训练/推理/链路接入完成，未做效果调参
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
