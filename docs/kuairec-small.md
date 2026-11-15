# KuaiRec small_matrix 数据线（稠密小样本，正式）

> **定位**：`kuairec_small` 是 KuaiRec 官方 `small_matrix` 的正式接入线。
> 它和 `kuairec`（big_matrix）**共用同一个读取器、同一套标签与切分口径**，
> 区别只有矩阵文件本身。用途是「一台机器能完整跑完、每一步都能复核」的端到端回归：
> 数据接入 → 训练 → 验证集选型 → 测试集只跑一次 → 报告。
>
> 主数据线仍是 **KuaiSAR**（见 [kuasar-roadmap.md](kuasar-roadmap.md)）；这条线不改变主线的口径，
> 也不把它的 Recall 和 KuaiSAR 的直接横比（候选集规模差三个数量级，见下文「坑」）。

## 口径（与 `kuairec` 逐项相同）

| 项 | 值 | 定义位置 |
| --- | --- | --- |
| 正反馈 | 首次曝光 `watch_ratio >= 1.0` | `recsys.config.KUAIREC_WATCH_RATIO_THRESHOLD` |
| 重复 (user, video) | 合并成一条，保留最早一次曝光 | `KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS` |
| 切分 | 每个用户最后两个正反馈 = validation / test | `recsys.data.schema.temporal_train_validation_test_split` |
| 协议字符串 | `per_user_last_two_strict_timestamp_v2` | `recsys.config.SPLIT_STRATEGY` |
| 候选集 | 全量物品（3,317），打分后过滤 | 评测脚本统一行为 |
| 过滤 | 目标时刻前**真实曝光过**的视频（含未看完） | `PreparedData.test_seen_mask` |

## 数据事实（实测，可复现）

| 项 | 值 |
| --- | --- |
| 文件 | `data/kuairec_small/raw/small_matrix.csv`，387.34 MB |
| 来源 | Zenodo `KuaiRec.zip` 成员 `KuaiRec 2.0/data/small_matrix.csv` |
| 完整性 | CRC32 `77673810`（与官方 ZIP 成员一致）；建库时按行数/用户/物品三重核对 |
| 行数 | **4,676,570** |
| 用户 / 目录物品 | **1,411 / 3,327**（正样本词表 3,317，10 个视频没有正反馈） |
| 矩阵密度 | **95.7%**（4,494,578 个带时间的 `(user, video)` 对 / 4,694,397） |
| 重复对 | **0**——每个 `(user, video)` 只有一行，去重逻辑在这份数据上是 no-op |
| 缺 timestamp | **181,992 行（3.9%）**，每行都是独立的对、没有同时刻同对记录；其中 43,188 行 `watch_ratio >= 1.0` |
| 正样本 | **1,471,862** 对（占带时间对的 32.7%） |
| 切分 | train 1,469,040 / validation 1,411 / test 1,411 |
| 评测候选 | 曝光过滤后每用户 **29–470**（中位 149） |

审计原文在 `data/kuairec_small/processed/label_audit.json`（由 `prepare_kuairec_small` 生成），
本表数字全部来自它，不手写。

## 命令

```powershell
# 建库：CRC32 + 规模校验 → 流式解析 → 首次曝光 → 切分（实测 40.4 秒）
python -m recsys.data.prepare --dataset kuairec_small

# 三个基线（都写 artifacts/kuairec_small/）
python -m recsys.training.train_mf     --dataset kuairec_small   # 5 轮，约 3.5 分钟
python -m recsys.training.trainer      --dataset kuairec_small   # 双塔 9 轮
python -m recsys.training.train_sasrec --dataset kuairec_small   # 序列，按用户建样本，很快

# 正式收口：验证集选型 + 测试集只跑一次，结果写 artifacts/kuairec_small/results.json
python -m recsys.experiments.finalize_experiment --dataset kuairec_small
```

`finalize_experiment` 会核对 checkpoint 里记录的 `split_strategy` 与该数据集要求的协议；
换数据集但拿旧 checkpoint 来评测会直接报错，不会静默出数。

## 文件放哪

```
data/kuairec_small/
├── raw/small_matrix.csv        387 MB，保留原文件便于 CRC32 复核
├── raw/download.json           下载来源、成员路径、CRC32、耗时
└── processed/                  interactions.csv / id_mappings.json / protocol.json / label_audit.json
```

## 坑

- **缺时间的行只能丢**：181,992 行（3.9%）没有 `timestamp`，既无法定位先后、也进不了 `seen_mask`。
  本实现把它们计数后丢弃（`label_audit.rows_missing_timestamp_excluded`），**不猜时间**。
  代价是这 43,188 个正反馈不进训练集，且它们对应的曝光不在过滤集合里——
  理论上会让极少数「其实看过但无时间」的物品仍被推荐，报告里按这个口径解释数字。
- **去重是 no-op，但别删逻辑**：这份矩阵每对只有一行，`duplicate_events_on_positive_pairs = 0`；
  big_matrix 的重复率是 17.8%。两条线共用同一段合并代码，删掉会让 big_matrix 的评测出现泄漏。
- **候选集不是全目录**：稠密矩阵下「目标时刻前看过」几乎覆盖全目录，过滤后每用户只剩 29–470 个候选。
  因此 Recall@20 的天花板远低于 1，也不能和 KuaiSAR（4M 目录、~1 万级候选）的绝对值直接比较。
- **物品词表 3,317 ≠ 目录 3,327**：有 10 个视频没有任何 `watch_ratio >= 1.0` 的首次曝光，
  词表按正样本建，所以模型输出维度是 3,317。统计「多少物品」时先确认问的是目录还是词表。
- **阈值仍然必须在训练前定死**：换阈值 = 换词表 = 旧 checkpoint 全部作废，这条和 `kuairec` 一致。

## 首轮结果（2026-11-13，单种子 42）

| 模型 | 测试 Recall@10 | Recall@20 | NDCG@20 | 选中 epoch |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.199858 | 0.328845 | 0.141528 | — |
| MF | 0.222537 | 0.389794 | 0.151751 | 1 |
| Two-Tower | 0.206237 | 0.332388 | 0.131091 | 6 |
| SASRec | 0.795889 | 0.924167 | 0.478277 | 15 |

建库 + 训练 + 收口合计约 16.6 分钟（CPU，14 线程）。

读这张表之前先看两个约束：随机基线（精确期望，不是 `20/候选数` 的近似）在过滤后候选
（29–469，中位 149）上是 Recall@20 **0.156636** / NDCG@20 **0.055138**，所以 Popularity 是 2.10× 随机、
MF 2.49×、双塔 2.12×、SASRec 5.90×；数值高低主要由候选集大小和「序列任务是否成立」决定，
**不能与 KuaiSAR 的绝对值横比**。SASRec 用「只有训练历史」的受控输入复算，Recall@20 不变，
排除测试期历史泄漏；命中者平均名次 5.38（Popularity 8.73、MF 9.40、双塔 8.88）。详见
[实验报告](../experiments/2026-11-13-kuairec-small-formal/report.md) ·
[协议](../experiments/2026-11-13-kuairec-small-formal/protocol.json) ·
[结果](../experiments/2026-11-13-kuairec-small-formal/results.json) ·
[效果拆解](../experiments/2026-11-13-kuairec-small-formal/effect_analysis.json)。
