<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# MMRec 多模态推荐框架 技术档案

## 基本信息

- **ID**：`mmrec_multimodal`
- **分类**：召回模型（retrieval）
- **来源**：vendor：third_party/MMRec
- **许可证**：GPL-3.0
- **上游**：https://github.com/enoche/MMRec
- **Commit**：`c68ee7c9b0de1ca7c5150258322edb4994be29d9`
- **源码**：`third_party/MMRec/src/models/lattice.py`、`third_party/MMRec/src/models/vbpr.py`、`third_party/MMRec/src/main.py`

## 原理与适用场景

- **原理**：用视觉/文本预提取特征增强用户-物品图模型（VBPR、LATTICE、DualGNN 等 22 个实现）。
- **适用场景**：有多模态物品特征的场景；KuaiSAR 的视频/封面特征可复用同一范式。

## 运行方式

- **依赖 / Python**：torch、torch_geometric、torch_scatter、torchvision；Python 3.7 时代实现，现代 Python 需移植
- **训练入口**：python -m recsys.training.train_mmrec --dataset kuasar_small --epochs 10
- **推理入口**：python -m recsys.adapters evaluate --model mmrec_multimodal --checkpoint artifacts/kuasar_small/mmrec_vbpr.pt
- **数据格式**：交互表 + 与物品 ID 对齐的视觉/文本特征文件
- **保存/加载**：artifacts/<dataset>/mmrec_vbpr.pt（ID + 内容投影 + 内容矩阵 + provenance）
- **适配层**：MMRecAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：VBPR 公式的纯 PyTorch 最小适配；训练/推理/链路接入完成。上游 LATTICE 等仍需 torch_geometric
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
