<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# DIEN（深度兴趣演化） 技术档案

## 基本信息

- **ID**：`dien`
- **分类**：排序模型（ranking）
- **来源**：vendor：third_party/DeepCTR-Torch
- **许可证**：Apache-2.0
- **上游**：https://github.com/shenweichen/DeepCTR-Torch
- **Commit**：`9eccef78e5810c61d6a04ddc7d96279e3db9c970`
- **源码**：`third_party/DeepCTR-Torch/deepctr_torch/models/dien.py`

## 原理与适用场景

- **原理**：兴趣抽取层 + AUGRU 兴趣演化层，让兴趣向量随行为序列演进。
- **适用场景**：长序列 CTR；兴趣漂移明显的场景。

## 运行方式

- **依赖 / Python**：torch、scikit-learn、tqdm；Python >=3.8
- **训练入口**：python -m recsys.training.train_neural_ranker --model dien --dataset kuasar_small
- **推理入口**：python -m recsys.ads.predict --model dien（广告）/ 推荐链路 rank_candidates(model=dien)
- **数据格式**：广告候选：请求前历史 + 类目/数值特征 + click 标签
- **保存/加载**：artifacts/<dataset>/ads/interface_weights/dien.pt（固定初始化，interface_test）
- **适配层**：DIENAdapter / 广告侧 ProjectAdAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：按论文重写 AUGRU cell；推荐训练/排序/链路接入完成；Phase 3 在广告样本上复用同一实现，历史严格早于请求
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
