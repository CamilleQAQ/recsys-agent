<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# BPR-MF（矩阵分解） 技术档案

## 基本信息

- **ID**：`matrix_factorization`
- **分类**：召回模型（retrieval）
- **来源**：本项目自研
- **许可证**：项目自有
- **上游**：本项目（自研基线）
- **源码**：`src/recsys/models/matrix_factorization.py`、`src/recsys/training/train_mf.py`

## 原理与适用场景

- **原理**：用户/物品各学一个隐向量，点积即偏好分数，用 BPR 负采样训练。
- **适用场景**：隐式反馈基线；召回通道的最小可用模型。

## 运行方式

- **依赖 / Python**：torch；Python >=3.11
- **训练入口**：python -m recsys.training.train_mf --dataset kuairec
- **推理入口**：python -m recsys.evaluation.evaluator --checkpoint artifacts/<dataset>/matrix_factorization.pt
- **数据格式**：PreparedData；评测用 seen-mask 过滤已曝光物品
- **保存/加载**：artifacts/<dataset>/matrix_factorization.pt（model_config + model_state_dict + train_config）
- **适配层**：MatrixFactorizationAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：训练/评测/链路/Agent 训练与召回均可运行；小样本冒烟已通过
- **已知局限**：只学 ID 隐向量，用不上特征与序列；负样本随机采样、未做热度校正，长尾物品学到的东西很少。

## 逐项状态

| 源码 | 已核源码 | 适配 | 可运行 | 业务接入 | Agent | 已评估 |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
