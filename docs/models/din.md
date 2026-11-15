<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# DIN（深度兴趣网络） 技术档案

## 基本信息

- **ID**：`din`
- **分类**：排序模型（ranking）
- **来源**：vendor：third_party/FuxiCTR
- **许可证**：Apache-2.0
- **上游**：https://github.com/reczoo/FuxiCTR
- **Commit**：`183da1c944ba8930fcdcd54114837c081959d748`
- **源码**：`third_party/FuxiCTR/model_zoo/DIN/src/DIN.py`、`third_party/DeepCTR-Torch/deepctr_torch/models/din.py`、`third_party/RecBole/recbole/model/sequential_recommender/din.py`

## 原理与适用场景

- **原理**：用注意力对用户历史行为按候选物品加权池化，刻画局部激活兴趣。
- **适用场景**：CTR 精排；行为序列 + 候选物品的注意力建模。

## 运行方式

- **依赖 / Python**：torch、fuxictr、pandas、pyarrow、ray[data]；Python >=3.10（FuxiCTR）
- **训练入口**：python -m recsys.training.train_neural_ranker --model din --dataset kuasar_small
- **推理入口**：python -m recsys.ads.predict --model din（广告）/ 推荐链路 rank_candidates(model=din)
- **数据格式**：广告候选：类目 4 列 + 数值 5 列 + 用户数值 2 列 + 请求前历史
- **保存/加载**：artifacts/<dataset>/ads/interface_weights/din.pt（固定初始化，interface_test）
- **适配层**：DINAdapter / 广告侧 ProjectAdAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：按上游注意力池化逻辑重写；推荐训练/排序/链路/Agent 已通过；Phase 3 在广告样本上复用同一实现（src/recsys/ads/models.py）
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
