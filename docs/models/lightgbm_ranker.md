<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# LightGBM LambdaRank 技术档案

## 基本信息

- **ID**：`lightgbm_ranker`
- **分类**：排序模型（ranking）
- **来源**：vendor：third_party/LightGBM
- **许可证**：MIT
- **上游**：https://github.com/lightgbm-org/LightGBM
- **Commit**：`3ee03c26ba8c9e8ddc05fa8687717cae8a7d61d9`
- **源码**：`third_party/LightGBM/python-package/lightgbm/sklearn.py`、`third_party/LightGBM/src/objective/rank_objective.hpp`

## 原理与适用场景

- **原理**：GBDT 直接优化排序目标（LambdaRank / rank_xendcg），按 query 分组计算 NDCG 梯度。
- **适用场景**：精排基线；特征工程友好、训练成本低的强 baseline。

## 运行方式

- **依赖 / Python**：lightgbm、numpy、scipy、scikit-learn；Python >=3.11（python-package/pyproject.toml）
- **训练入口**：lightgbm.LGBMRanker.fit(X, y, group=...)（或 CLI：lightgbm config=train.conf）
- **推理入口**：Booster.predict / LGBMRanker.predict
- **数据格式**：稠密/稀疏特征矩阵 + label + 连续 query group（一个用户的候选为一个 group）
- **保存/加载**：booster.save_model('model.txt') / Booster(model_file=...)
- **适配层**：LightGBMRankerAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：既有排序适配器；Phase 1 接入统一链路的默认精排、默认配置与 Agent 训练/排序工具
- **已知局限**：暂无记录（接入实验后补充）

## 逐项状态

| 源码 | 已核源码 | 适配 | 可运行 | 业务接入 | Agent | 已评估 |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
