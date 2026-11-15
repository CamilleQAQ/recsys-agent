<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# Ultra-HSTU（TorchEasyRec 排序版） 技术档案

## 基本信息

- **ID**：`ultra_hstu`
- **分类**：排序模型（ranking）
- **来源**：vendor：third_party/TorchEasyRec
- **许可证**：Apache-2.0
- **上游**：https://github.com/alibaba/TorchEasyRec
- **Commit**：`2815a9732bed014219ee89826dd4987b67ccf6b0`
- **源码**：`third_party/TorchEasyRec/tzrec/models/ultra_hstu.py`

## 原理与适用场景

- **原理**：在 HSTU 转导堆叠上接排序头，直接对候选做 CTR/时长预估。
- **适用场景**：长序列 + 生成式主干的精排。

## 运行方式

- **依赖 / Python**：torch、tzrec；Python >=3.10（Linux）
- **训练入口**：python -m recsys.training.train_neural_ranker --model ultra_hstu --dataset kuasar_small
- **推理入口**：python -m tzrec.predict --pipeline_config_path <config>
- **数据格式**：行为序列 + 候选特征 + 标签
- **保存/加载**：artifacts/<dataset>/ultra_hstu.pt（HSTU 主干 + 排序头 + provenance）
- **适配层**：UltraHSTUAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：HSTU 主干 + 候选排序头的纯 PyTorch 适配；训练/排序/链路接入完成
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
