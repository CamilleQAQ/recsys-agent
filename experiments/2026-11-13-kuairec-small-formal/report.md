# KuaiRec small_matrix 正式接入与四模型基线

**结论**　`kuairec_small` 已作为一等数据集接入项目：`loader` 注册、`configs/datasets` 登记、`check_project` 交叉校验三处闭环，口径与既有 `kuairec` 逐项相同。四模型在同一份切分上走完「验证集选型 → 锁定 → 测试集只跑一次」。测试集 Recall@20：**SASRec 0.924167**、MF 0.389794、Two-Tower 0.332388、Popularity 0.328845。

SASRec 的领先不是历史泄漏：同一个 checkpoint 换成「只有训练历史」的受控输入后，测试 Recall@20 仍是 0.924167、NDCG@20 0.477264（标准输入 0.478277）。两个 ID-only 模型（MF / 双塔）在这份稠密矩阵上只比 Popularity 高一点，这是数据形态决定的，不是训练没跑对——见「这些数字能说明什么」。

## 数据与协议

| 项 | 值 |
| --- | --- |
| 原始文件 | `data/kuairec_small/raw/small_matrix.csv`，387.34 MB，CRC32 `77673810`（与官方 ZIP 成员一致） |
| 行数 / 用户 / 目录物品 | 4,676,570 / 1,411 / 3,327（正样本词表 3,317） |
| 矩阵密度 | 95.7%，每个 `(user, video)` 只有一行（重复对 0，与 big_matrix 的 17.8% 相反） |
| 缺 timestamp | 181,992 行（3.9%）计数后剔除，其中 43,188 行 `watch_ratio >= 1.0` |
| 正样本 | 1,471,862 对 |
| 切分 | train 1,469,040 / validation 1,411 / test 1,411，协议 `per_user_last_two_strict_timestamp_v2` |
| 评测候选 | 曝光过滤后每用户 29–470（中位 149） |
| 同时间戳歧义用户 | 0（验证与测试目标时间戳无并列） |
| 冷启动物品 | 0（每个物品在训练集都有正反馈，稠密矩阵的必然结果） |

标签审计原文见 [label_audit.json](label_audit.json)，由 `prepare_kuairec_small` 生成，本表数字全部取自它。

## 接入改动

| 文件 | 改了什么 | 为什么 |
| --- | --- | --- |
| [src/recsys/config.py](../../src/recsys/config.py) | 新增 `KUAIREC_SMALL_MATRIX_{ROWS,USERS,ITEMS}`、`KUAIREC_SMALL_POSITIVE_ITEMS` | 建库时按规模核对，挡住半截下载或换错矩阵 |
| [src/recsys/data/kuairec.py](../../src/recsys/data/kuairec.py) | `download_and_extract_kuairec` 加 `matrix_filename`；新增 `matrix_crc32`、`prepare_kuairec_small`、`expected_scale` 校验、`allow_missing_timestamp` | 同一读取器服务两份矩阵；缺时间的行显式记账后丢弃，而不是让整份数据解析失败 |
| [src/recsys/data/loader.py](../../src/recsys/data/loader.py) | `SUPPORTED_DATASETS` 加 `kuairec_small`；阈值/协议按 `kuairec` 家族分派 | 所有训练与评测入口通过 `--dataset` 自动可用 |
| [configs/datasets/kuairec_small.toml](../../configs/datasets/kuairec_small.toml) | 新增数据集登记（来源、布局、规模、标签、切分、评测口径） | 配置与代码常量双向可校验 |
| [scripts/check_project.py](../../scripts/check_project.py) | `check_dataset_configs` 增加 10 条 `kuairec_small` 交叉校验（阈值/去重/来源/成员/CRC32/行数/用户/物品/正样本物品/切分） | 文档漂移会直接让检查失败 |
| [src/recsys/experiments/finalize_experiment.py](../../src/recsys/experiments/finalize_experiment.py) | 结果从 `artifacts/results.json` 改为 `artifacts/<dataset>/results.json` | 原路径是单数据集时代的全局文件，第二个数据集收口会覆盖第一个 |
| [.gitignore](../../.gitignore) | 忽略 `data/kuairec_small/` | 387 MB 原始数据不入库 |

## 结果

验证集（只用于选 checkpoint，不用于报告结论）：

| 模型 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | HitRate@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.206945 | 0.103475 | 0.352941 | 0.140053 | 0.352941 |
| MF | 0.209780 | 0.096853 | 0.398299 | 0.144184 | 0.398299 |
| Two-Tower | 0.201984 | 0.093486 | 0.346563 | 0.129656 | 0.346563 |
| SASRec | 0.789511 | 0.438457 | 0.927002 | 0.473760 | 0.927002 |

测试集（配置与权重锁定后跑一次）：

| 模型 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | HitRate@20 | 选中 epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.199858 | 0.109265 | 0.328845 | 0.141528 | 0.328845 | — |
| MF | 0.222537 | 0.109700 | 0.389794 | 0.151751 | 0.389794 | 1 |
| Two-Tower | 0.206237 | 0.099231 | 0.332388 | 0.131091 | 0.332388 | 6 |
| SASRec | 0.795889 | 0.445587 | 0.924167 | 0.478277 | 0.924167 | 15 |

`HitRate@20` 与 `Recall@20` 相等，是因为每个评测用户只有一个留出目标（leave-one-out），命中即全召回。

## 训练预算与耗时（CPU，14 线程）

| 阶段 | 预算 | 实测 |
| --- | --- | ---: |
| 建库 + 切分 | — | 40.4 s |
| MF | 5 轮，batch 1024，dim 64 | 207.3 s（验证最好在第 1 轮） |
| Two-Tower | 9 轮，batch 256，dim 128 | 441.7 s（最好第 6 轮） |
| SASRec | ≤30 轮，patience 5，batch 256，dim 64 | 106.1 s（第 20 轮早停，取第 15 轮） |
| 收口评测（4 模型 × 2 切分 + FAISS） | — | 198.9 s |
| 合计 | — | **约 16.6 分钟** |

全程单种子（42）、单次运行。SASRec 快是因为它按用户建样本（1,411 条序列），不是按交互。

## 对照与检查

- **SASRec 受控对照**：同一个 checkpoint，测试输入从「train + validation」换成「只有 train」，Recall@20 仍 0.924167、NDCG@20 0.477264（ΔNDCG −0.001013）。测试阶段的历史补充没有贡献，说明 0.92 不是把验证目标喂进模型换来的。
- **FAISS 与暴力检索**：`IndexFlatIP` 与逐用户暴力点积的六项指标**完全一致**，100 个抽样用户 Top-20 顺序重合率 1.0。检索耗时 0.541 ms/用户 vs 暴力 0.282 ms/用户——这个规模下（3,317 × 128）精确 FAISS 反而慢 1.9×，和 KuaiSAR 上「IndexFlat 比 torch 暴力慢」的旧结论同向。
- **协议一致性**：三个 checkpoint 里的 `split_strategy` 都是 `per_user_last_two_strict_timestamp_v2`，评测时会按数据集校验；换成别的数据集或旧协议的权重会直接报错。
- **验证/测试无异常并列**：同时间戳歧义用户 0，不需要退回训练集处理。

## 效果拆解：命中名次与分层

只看 Recall/NDCG 不解释"效果好在哪"。把每个模型的 top-20 拿出来，记录目标物品的实际名次（`analyze_effect.py`，只读已锁定 checkpoint）：

| 模型 | Recall@20 | NDCG@20 | 命中者平均名次 | 名次=1 | 2–5 | 6–10 | 11–20 | 未命中 | 相对随机 R@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 随机（精确期望） | 0.156636 | 0.055138 | — | — | — | — | — | — | 1.00× |
| Popularity | 0.328845 | 0.141528 | 8.73 | 61 | 121 | 100 | 182 | 947 | 2.10× |
| MF | 0.389794 | 0.151751 | 9.40 | 48 | 125 | 141 | 236 | 861 | 2.49× |
| Two-Tower | 0.332388 | 0.131091 | 8.88 | 37 | 133 | 121 | 178 | 942 | 2.12× |
| SASRec | 0.924167 | 0.478277 | 5.38 | 219 | 613 | 291 | 181 | 107 | 5.90× |

随机基线是**精确期望**而不是 `20/候选数` 的近似：每个用户在 C 个候选里均匀取 top-20 的期望值，
对 C 的分布（29–469，中位 149）逐用户求和后平均，得 Recall@20 = 0.156636、NDCG@20 = 0.055138。

三个能直接读出来的事实：

1. **Popularity 在"第一名"上比 MF 更强**（61 vs 48 个用户目标排第 1），但 MF 在 11–20 名多捞了 54 个（236 vs 182）。所以 MF 的 Recall@20 领先（+0.061）远大于 NDCG@20 领先（+0.010）——它更像是"猜得更宽"，不是"排得更准"。
2. **双塔是这组里最弱的 ID 模型**：命中数只比 Popularity 多 5 个（469 vs 464），NDCG@20 反而低 0.010。在 1,411 个用户的稠密矩阵上，128 维用户塔没有足够信号可学。
3. **SASRec 不是靠"多捞"赢的**：219 个目标直接排在 1 位（占全部用户的 15.5%），命中者平均名次 5.38，且只有 107 个用户（7.6%）没进 top-20。

按「目标时刻前已曝光物品数」三分位切开（light / mid / heavy，即过滤后候选 168–469 / 125–168 / 29–125 个）：

| 模型 | light（候选 168–469） | mid（125–168） | heavy（29–125） |
| --- | ---: | ---: | ---: |
| Popularity | 0.2766 | 0.2638 | 0.4459 |
| MF | 0.3106 | 0.3106 | 0.5478 |
| Two-Tower | 0.2915 | 0.2872 | 0.4183 |
| SASRec | 0.8872 | 0.9170 | 0.9682 |

必须先说清这个分层实际在分层什么：稠密矩阵下每个用户已曝光 2,848–3,288 个物品（中位 3,168，目录 3,317），
所以三分位之间的差别只有约 340 个物品，**它等价于"过滤后还剩多少候选"**（候选越少 → 任务越容易）。
四个模型的 Recall 都随候选数下降而上升，这说明梯度主要来自候选集规模，而不是「模型更擅长服务重度用户」。
读数时把它当"指标对候选集大小有多敏感"的证据，而不是用户画像结论。

## 这些数字能说明什么

**能说明**：数据接入、切分、曝光过滤、四类模型（统计基线 / 矩阵分解 / 双塔 / 序列）在同一口径下可复现地跑通，且收口流程（验证选型 → 测试一次）没有被绕过。整条链路一台 CPU 机器 17 分钟跑完。

**不能说明**：

- **不能和 KuaiSAR 横比**。这里目录 3,317、过滤后候选中位 149；KuaiSAR 是 4M 目录、万级候选。随机基线 Recall@20 = 0.156636，所以 Popularity 是 2.10× 随机、MF 2.49×、SASRec 5.90×——数值高低主要反映候选集大小和任务形态，不反映"推荐系统强"。
- **SASRec 的高分是任务形态给的**。稠密矩阵下每个用户有近千条正反馈，序列模型拿到的是近乎完整的观看序列，而留出目标又在同一批物品里；这是「下一部看什么」这种强序列任务，不是冷启动或长尾召回。命中名次分布也印证：15.5% 的用户目标直接排第 1，说明很多目标有强前序信号。
- **ID-only 模型被数据形态压住了**。1,411 个用户 × 3,317 个物品、几乎填满，MF/双塔的 ID embedding 没有多少可学的区分度：双塔只比 Popularity 多命中 5 个、NDCG 还更低。想在这里做出差距，需要物品侧特征或序列信息，而不是调 ID 维度。
- **单种子、无方差**。项目在 KuaiSAR 上已经验证「种子方差可以大于组件效应」，这里每个模型只跑了一个种子，0.33 vs 0.39 这种量级差异不应被当成稳定结论。

## 复现

```powershell
python -m recsys.data.prepare --dataset kuairec_small
python -m recsys.training.train_mf     --dataset kuairec_small
python -m recsys.training.trainer      --dataset kuairec_small
python -m recsys.training.train_sasrec --dataset kuairec_small
python -m recsys.experiments.finalize_experiment --dataset kuairec_small
python experiments/2026-11-13-kuairec-small-formal/analyze_effect.py   # 命中名次/分层/随机基线
```

[实验协议与耗时](protocol.json) · [完整结果（含每模型配置与 checkpoint sha256）](results.json) ·
[效果拆解原始数据](effect_analysis.json) · [分析脚本](analyze_effect.py) · [标签审计](label_audit.json) ·
[数据线文档](../../docs/kuairec-small.md) · [数据契约](../../docs/data-contract.md)
