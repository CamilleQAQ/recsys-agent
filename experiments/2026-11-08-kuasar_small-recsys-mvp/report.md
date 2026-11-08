# 2026-11-08 · kuasar_small 离线推荐 MVP（双路召回 → 融合 → LightGBM → 类目重排）

> 本轮目标只是**链路跑通与口径正确**，不追求指标。样本为 128 个验证用户，Top-20 命中为 0 属样本量限制，指标不作结论。

## 1. 一次实际运行

```
python -m recsys.experiments.recsys_mvp --dataset kuasar_small --users 128 --seed 42 --threads 4
```

| 项 | 结果 |
| --- | --- |
| 评测用户 | 128 个验证用户（seed 42，名单：`users.json`） |
| 两路召回 | Two-Tower Top-100 + MIND Top-100（各算一次，缓存 JSON） |
| 融合 | 名次轮流、去重，最多 100（实际均值 100.0） |
| 排序 | LightGBM LambdaRank，20 轮，9,997 行 / 2,418 请求（正 6,387 / 负 3,610） |
| 重排 | Top-20，已知一级类目每类 ≤5，补齐次数 0 |
| 总耗时 | **10.0 s**（召回 1.0 + 3.9、融合 0.05、排序数据 1.8、训练 0.2、打分 0.1） |
| 峰值 RSS | **1,379 MiB**（预算 4 GiB） |
| 线程 | 4 |
| LightGBM 训练 | 0.2 s（预算 2 分钟） |

## 2. 数据与指纹

- 数据集 `kuasar_small`：2,000 用户 / 454,717 物品；切分 `per_user_last_two_requests`；
  split sha256 train `c62df14c…`、validation `00068116…`、test `d6b5c0e8…`；ID 映射 sha256 `b7d23062…`。
- 双塔 `two_tower_features.pt` sha256 `784a2ad6…`；MIND `mind20.pt` sha256 `eba01718…`；均复用、未重训。
- MIND 路由种子 `1234`，K=100；两路只为这 128 个用户打分（未对 2,000 用户全量打分）。

## 3. 指标（Top-20，128 用户验证样本）

| 阶段 | Recall@20 | HitRate@20 | NDCG@20 | Coverage |
| --- | --- | --- | --- | --- |
| Two-Tower | 0 | 0 | 0 | 0.005566 |
| 融合 | 0 | 0 | 0 | 0.005557 |
| LightGBM | 0 | 0 | 0 | 0.005395 |
| 重排 | 0 | 0 | 0 | 0.005412 |

**为什么全是 0**：128 个用户共有 362 个验证目标；Top-100 候选里命中 9 个（双塔 2、MIND 7、并集 9），
但这些命中都排在 20 名之后，所以 Top-20 为 0。这符合样本量（2,000 用户上双塔 Recall@20 也只有 0.0044）。
诊断数据写在 `metrics.json` 的 `diagnostics` 里，用来证明召回确实捞到了目标，而不是管线坏了。

## 4. 自动断言（全部通过）

- 融合候选无重复、无历史曝光泄漏、每个候选都有来源通道与原始名次；
- 融合 ≤100、最终 ≤20；LightGBM 排序与重排结果都来自同一融合池；
- 排序训练的分组长度之和 = 样本行数（9,997）；只使用 train 标签，未读取验证/测试标签；
- 训练与推理使用同一套特征列（9 列，顺序与 dtype 记录在 `features.json`）；
- 重载 LightGBM 对缓存候选重新打分，最大分数差 5.0e-07，排序一致。

## 5. 可追溯样例

`samples.json` 里 3 个用户的 Top-5 逐个包含：原始 user_id / item_id、来源通道及名次、融合名次、
LightGBM 分数与名次、最终名次。例如：

| 用户 | 物品 | 来源 | 融合名次 | LGBM 分数 | 最终名次 |
| --- | --- | --- | --- | --- | --- |
| 664570 | 3913948 | MIND rank 7 | 13 | 0.419849 | 1 |
| 664570 | 3256736 | Two-Tower rank 36 | 63 | 0.385514 | 2 |
| 664570 | 1578321 | Two-Tower rank 6 | 10 | 0.305339 | 3 |

（完整数据见 `samples.json`；ID 均为原始业务 ID，不是内部索引。）

## 6. 产物

`artifacts/kuasar_small/mvp/`：

| 文件 | 内容 |
| --- | --- |
| `users.json` | 128 个用户名单与 seed |
| `recall_two_tower.json` / `recall_mind.json` | 每路 Top-100（含分数/名次） |
| `fusion.json` | 融合池，含来源通道与名次 |
| `ranked.json` | LightGBM 打分与名次 |
| `final.json` | 重排后 Top-20 |
| `ranking_train_rows.npz` / `ranking_train_meta.json` | 排序训练样本与统计 |
| `lgbm_ranker.txt` | LightGBM 模型（sha256 `a79d9d45…`） |
| `features.json` | 特征列名、dtype、缺失值规则 |
| `samples.json` | Top-K 可追溯样例 |
| `metrics.json` / `run_meta.json` | 指标、诊断、耗时、RSS、指纹、断言 |

## 7. 预算核对

| 限制 | 实际 |
| --- | --- |
| ≤128 用户 | 128 |
| LightGBM 只训 1 次 | 1 次（20 轮，0.2 s） |
| ≤10,000 候选行 | 9,997（整请求，未截断） |
| ≤2 分钟训练 | 0.2 s |
| ≤4 线程 / 4 GiB | 4 线程 / 1,379 MiB |
| ≤15 分钟 | 10.0 s（编码另计） |
| 只用 kuasar_small | 入口显式校验 `--dataset kuasar_small`，否则退出 |

## 8. 未完成（留待后续）

- 完整验证集/测试集评估：本轮按预算只跑 128 用户，Top-20 指标无统计意义；
- 排序特征只用物品元数据，没有用户侧/交叉特征；LambdaRank 的 20 轮是冒烟预算，不是调参结果；
- 两路召回分数尺度不同，本轮只按名次融合、没有做分数校准；
- 未接 DIN/DCNv2，未做多路配额与消融实验（Phase 5/8 的正式内容）。
