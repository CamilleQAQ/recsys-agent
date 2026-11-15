# 推荐全链路（Phase 1 v2）

统一入口：`python -m recsys.pipeline`。它把「多路召回 → 去重融合 → 候选打分 →
类目/多样性重排 → 输出 → 离线评估」收敛到一份 TOML 配置，默认配置复现既有
`recsys_mvp` 的双路（Two-Tower + MIND）、轮流融合、LightGBM LambdaRank、
类目上限 5 的行为。

## 1. 快速运行

```powershell
$env:PYTHONPATH = "src"
python -m recsys.pipeline list
python -m recsys.pipeline run --config configs/pipeline/kuasar_small_default.toml
python -m recsys.pipeline run --config configs/pipeline/kuasar_small_default.toml --users 16
python -m recsys.pipeline evaluate --artifacts artifacts/kuasar_small/pipeline --split validation
```

默认工件目录：`artifacts/kuasar_small/pipeline/`，包含：

| 文件 | 内容 |
| --- | --- |
| `recall_<channel>.json` | 每路召回的候选、通道内名次、模型分数 |
| `fusion.json` | 去重融合后的候选、融合名次、每个来源的名次/分数 |
| `ranked.json` | 排序模型对融合候选的真实分数与名次 |
| `final.json` | 最终 Top-K（原始 user_id / item_id），保留来源与各阶段分数 |
| `metrics.json` | 召回各通道 / 融合 / 排序 / 重排各阶段的指标 |
| `diagnostics.json` | 逐目标「最先丢失环节」与通道命中 |
| `run_meta.json` | 配置快照、数据指纹、checkpoint sha256、耗时、断言结果 |

## 2. 配置

示例见 [configs/pipeline/kuasar_small_default.toml](../configs/pipeline/kuasar_small_default.toml)
与 [configs/pipeline/kuasar_small_multi_recall.toml](../configs/pipeline/kuasar_small_multi_recall.toml)。

```toml
[run]
dataset = "kuasar_small"      # kuasar / kuasar_small / kuairec
split = "validation"          # validation / test
users = 128                   # 固定种子抽样的评测用户数
seed = 42
k_recall = 100
k_final = 20
include_validation_in_test_history = true   # test 时是否并入验证历史

[recall.two_tower]            # 通道名可自定义；每个通道一个模型
model = "two_tower"
checkpoint = "artifacts/kuasar_small/two_tower_features.pt"
k = 100
weight = 1.0

[recall.mind]
model = "mind"
checkpoint = "artifacts/kuasar_small/mind20.pt"
k = 100

[fusion]
mode = "round_robin"          # round_robin / rrf / quota
limit = 100
# quota 模式示例：quota = { two_tower = 50, mind = 50 }

[ranking]
model = "lightgbm_ranker"     # 或 din / dien / dcnv2 / ultra_hstu
checkpoint = "artifacts/kuasar_small/mvp/lgbm_ranker.txt"
train_if_missing = true       # LightGBM 缺失时用父表 train 请求训练
num_boost_round = 20
max_rows = 10000

[rerank]
category_cap = 5
author_cap = 0
diversity = true
```

`--user-ids 2401117,2401176` 可以指定原始用户，便于复现固定样本。

## 3. 各阶段实现

### 3.1 召回

`recsys.pipeline.recall.run_channel` 按模型类型选择真实打分路径，而不是只取
`recommend()` 的 item 列表：

| 模型类型 | 打分方式 | 保留分数 |
| --- | --- | --- |
| embedding（Two-Tower / DSSM / MF / LightGCN / MMRec） | `encode_users @ encode_items.T` | 点积/余弦分数 |
| MIND | `models.mind.score_users`，K 个兴趣取 max | 兴趣最大相似度 |
| Popularity | 训练集点击次数 | 热度计数 |
| SASRec / BERT4Rec / HSTU | `score_all_items` | 下一物品点积分数 |
| TIGER | 约束 beam search | 生成累计 log-prob |

所有通道都按 `seen_items_for_split` 过滤目标时刻前已曝光物品，并记录
`(item, score, rank)`。

### 3.2 融合

- `round_robin`：按各路原始名次轮流取、去重，达到 `limit` 停止；默认行为。
- `rrf`：`sum(weight / (rrf_k + rank))`，适合多路异构模型。
- `quota`：先按每路配额取，再轮流补位；配额之和不必等于 100，程序会补位。

每条融合候选保留 `sources: {channel: {rank, score, weight}}` 与 `fusion_rank`。

### 3.3 排序

`recsys.pipeline.ranking.rank_fused` 统一两种接口：

- LightGBM：使用与既有 MVP 完全一致的 9 列物品特征（`ITEM_FEATURE_COLUMNS`），
  `predict` 输出 LambdaRank 分数；
- 神经排序：适配器提供 `score_candidates(data, users, items_by_user, split)`，
  输入用户历史 + 候选物品 + 物品类目/数值特征 + 用户活跃度，输出 sigmoid 分数。

排序结果仍保留每个候选的融合名次与来源，便于回溯。

### 3.4 重排

`recsys.pipeline.rerank.diversify`：首遍每类目最多 `category_cap`、每作者最多
`author_cap`（0 表示不限制）；未知类目/作者不占配额；不足时按原序补齐并记录
`relaxations`。这是既有类目约束行为的一般化，默认 `author_cap=0` 时与旧行为一致。

### 3.5 评估

每个阶段都用同一套 `recsys.evaluation.metrics` 计算 Recall/NDCG/HitRate/
Precision@K + Coverage。指标口径与数据切分以 `run_meta.json` 的 `provenance`
为准；不同 split / 不同候选预算的结果不可直接混比。

### 3.6 历史口径（`include_validation_in_test_history`）

- `validation`：始终只用训练历史。
- `test` + `True`（默认）：训练 + 验证历史。
- `test` + `False`：只用训练历史。
- 开关同时作用于 SASRec/BERT4Rec/HSTU、MIND、TIGER、DIN/DIEN/Ultra-HSTU；
  LightGBM/DCNv2 不读序列历史，不受影响。
- 神经排序按每个用户最早留出目标时间过滤历史（保守上界，避免未来信息）；
  排序模型训练/评估按每条请求的时间戳过滤，同一请求的候选共享同一份历史。
- 已曝光过滤（seen items）规则不变。

## 4. 模型与索引的保存/加载

- 召回/排序模型通过各自 `save()`/`load()` 的 checkpoint 重建；适配器注册表见
  `recsys.adapters.registry` 与 `recsys.adapters.ranking_registry`。
- LightGBM 使用 `booster.save_model` / `Booster(model_file=...)`。
- 推荐运行结果是普通 JSON，`recsys.pipeline.chain.load_result(path)` 可读回；
  `python -m recsys.pipeline evaluate` 可对已落盘 `final.json` 重新计算指标。
- TIGER 的 Semantic ID 码表随 checkpoint 保存；RQ-VAE 码本/编码器/解码器在
  `rqvae.pt`，两者必须成套使用。

## 5. 与既有 `recsys_mvp` 的关系

`src/recsys/experiments/recsys_mvp.py` 保留不动，既有实验工件继续可复现。
新链路的默认配置与其超参一致（128 用户、100 召回、20 最终、LightGBM 20 轮/
10k 行、类目上限 5），因此可以逐阶段对照 `experiments/2026-11-08-kuasar_small-recsys-mvp`。
两者的差异：

1. 新链路的召回通道数可配置，MVP 固定双路；
2. 新链路保留每路模型分数，MVP 的 MIND 通道分数为 `null`；
3. 新链路把诊断写成 `diagnostics.json`，MVP 只写聚合命中数。

## 6. 已知限制

- **精排在完整测试集上是负贡献**（2026-11-10 效果专项）：同一批 100 条融合候选上，
  LightGBM 精排把 Top-20 命中从 25 压到 11，比随机重排还低；去掉精排与类目重排后
  最终 Recall@20 从 0.002175 提到 0.005967。若要保留精排，先把训练候选改成「召回产生的候选」。
  详见 [效果专项报告](../experiments/2026-11-10-effect-study/report.md)。
- 召回评测仍是「对全量物品精确打分」，没有走 FAISS 近似索引；454k 物品下
  128 用户可接受，全量用户需要用索引。
- 序列/生成式模型在评测时给所有物品打分或做 beam search，成本高于双塔。
- 神经排序的历史用训练集交互近似请求前历史（父表未保存请求时间戳）。
- 本地 KuaiSAR 没有视觉/文本 embedding，Semantic ID 与 MMRec 使用元数据向量。
