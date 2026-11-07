<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# MIND（多兴趣召回） 技术档案

## 基本信息

- **ID**：`mind`
- **分类**：召回模型（retrieval）
- **来源**：vendor：third_party/torch-rechub
- **许可证**：MIT
- **上游**：https://github.com/datawhalechina/torch-rechub
- **Commit**：`7ea1731a806ec3cb74a2c837b5bac8b04268dac6`
- **源码**：`third_party/torch-rechub/torch_rechub/models/matching/mind.py`、`third_party/torch-rechub/torch_rechub/basic/layers.py`、`src/recsys/adapters/rechub_mind.py`、`src/recsys/models/mind.py`、`src/recsys/training/train_mind.py`

## 原理与适用场景

- **原理**：在双塔结构上对用户行为序列做动态路由聚类，得到多个兴趣向量分别召回后合并。
- **适用场景**：用户兴趣多峰的短视频/电商场景，与 KuaiSAR 的短期兴趣建模契合。

## 运行方式

- **依赖 / Python**：torch、torch-rechub==1.0.0；Python >=3.10（纯 PyTorch，Windows 可用）
- **训练入口**：python -m recsys.training.train_mind --dataset kuasar_small --epochs 20
- **推理入口**：python -m recsys.adapters recommend --model mind --checkpoint artifacts/kuasar_small/mind.pt --users <user_id>
- **数据格式**：user_id + 历史 item 序列 + item_id（token 0 = padding，模型内部整体 +1）
- **保存/加载**：单个 .pt：state_dict + model_config + provenance（约 120 MB）
- **适配层**：MindAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：torch-rechub 1.0.0 + 本项目适配器；kuasar_small 上训练/评测通过（MIND 在测试集上弱于双塔，见实验记录）
- **已知局限**：暂无记录（接入实验后补充）

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
