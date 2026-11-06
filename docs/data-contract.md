# 数据契约与评测口径

所有模型共用同一份 `PreparedData`。换数据集只换读取器，换模型只换适配器，
**契约本身不动**——这是不同模型指标能横向比较的前提。

## 1. 数据结构（`src/recsys/data/schema.py`）

```python
Interaction(user_idx: int, item_idx: int, timestamp: float)      # 已编码的正反馈
IdMappings(user_to_index, index_to_user, item_to_index, index_to_item)
PreparedData(train/validation/test_interactions, mappings,
             validation_seen_mask, test_seen_mask)
```

要点：

- `user_idx` / `item_idx` 是**连续整数**，从 0 开始；原始 ID 与索引的映射必须可逆（评测要能还原成业务 ID）；
- `timestamp` 用 float：不同数据集的精度不同（KuaiSAR/KuaiRec 是毫秒级），统一成 float 才能比较先后；
- 只保留**正反馈**进 `PreparedData`；负反馈不丢，进入 `*_seen_mask`；
- `seen_mask` 是 `[num_users, num_items]` 的 bool 矩阵，语义是「该时间点之前**真实曝光过**的物品」，
  包含没看完/没点击的曝光。它决定 Top-K 时过滤掉什么，比「只过滤正反馈」更严格，也更接近线上；
- 合成数据/小样例可以不给 mask，此时回退为「训练集（+验证集）里出现过的物品」。

## 2. 时间切分（`temporal_train_validation_test_split`）

```text
每个用户按 timestamp 升序排序
  正反馈数 < 3                       → 全部留在 train（不参与评测）
  最后三个时间戳不是严格递增         → 全部留在 train（无法确定先后）
  否则：倒数第 3 个及更早 → train
        倒数第 2 个        → validation
        最后 1 个          → test
```

约束与理由：

- 同一时间戳无法判定顺序，宁可把该用户退回训练集，也不猜；
- 测试阶段的历史 = train + validation（验证目标在测试时刻早已被观测到）；
- `build_evaluation_examples` 会断言「历史严格早于目标」，违反直接抛错，避免泄漏；
- 协议字符串写在 checkpoint 里（`per_user_last_two_strict_timestamp_v2`），
  换切分规则等于换协议 → 旧 checkpoint 一律拒绝加载。

## 3. 候选集与过滤

- 候选集 = 全部物品（全量排序），不是 sampled-100 负样本；
- 打分后把 `seen_mask` 命中的位置置 `-inf`，再取 Top-K；
- 因此 `Recall@K` 的分母是该用户留出的相关物品数（通常为 1，即 leave-one-out），
  这种协议下 `HitRate@K ≡ Recall@K`，见 `metrics.hit_rate_at_k` 的注释。

## 4. 指标（`src/recsys/evaluation/metrics.py`）

| 指标 | 定义 |
| --- | --- |
| `Recall@K` | 前 K 命中数 / 该用户相关物品数，对所有评测用户取平均 |
| `HitRate@K` | 前 K 至少命中一个相关物品的用户比例 |
| `NDCG@K` | 二值相关性下的折损累计增益 / 理想 DCG（IDCG 按 `min(相关物品数, K)` 计算） |
| `Precision@K` | 前 K 命中数 / K（分母固定，与 Recall 互补：候选集大小不同时结论可能不同） |
| `Coverage` | 推荐结果覆盖到的物品数 / 全部物品数（全局指标，不按 K 平均） |
| `AUC` / `LogLoss` | 排序/CTR 模型的分类侧指标，二分类概率输入（`compute_ranking_model_metrics`） |

召回指标与排序指标**不能混在一张表里比较**：召回看「能不能捞到」，排序看「排得准不准」。

报告里必须同时给出**数据集、阈值、切分协议、候选集规模**，否则数字不可比。

## 5. 各模型需要的数据形状

| 模型族 | 输入形状 | 转换位置 |
| --- | --- | --- |
| 双塔 / MF（本项目） | `(user_idx, item_idx)` 对 + in-batch 负样本 | `recsys.data.schema.InteractionDataset` |
| SASRec（本项目） | 右补齐 item 序列，token 0 = padding，正/负样本序列对齐 | `recsys.data.dataset.SASRecTrainDataset` |
| RecBole（LightGCN 等） | atomic 文件：`.inter`（user_id/item_id/timestamp 带类型表头）+ `.user`/`.item` | 待实现（`PreparedData → atomic`） |
| FuxiCTR / DeepCTR-Torch（DIN/DCNv2/…） | 特征表（CSV/Parquet 或特征名→数组字典）+ label + 序列字段 | 待实现（KuaiSAR 特征 > 本项目当前字段） |
| LightGBM Ranker | 特征矩阵 X + label + 连续 `group`（一个用户/请求一个 group） | `LightGBMRankerAdapter.fit(groups=...)` 已定义接口，特征仍待实现 |
| TorchEasyRec / genrec | CSV/Parquet + protobuf 或 gin 配置；genrec 另需物品 embedding → 语义 ID | 上游自有，未接入 |
| FAISS | L2 归一化 float32 向量（用户/物品同维） | `recsys.retrieval.faiss_index.encode_users_and_items` |


## 6. 接新数据集要满足什么

1. 能产出：正反馈三元组、完整曝光记录（用于 mask）、原始 ID → 连续索引映射；
2. 时间戳可比较（秒或毫秒都行，内部统一 float）；
3. 同一个 `(user, item)` 多次交互要先决定合并策略（KuaiRec 采用去重，见 `recsys.config`）；
4. 阈值写进 `configs/datasets/<name>.toml` 与 `recsys.config` 常量，两边由 `scripts/check_project.py` 校验一致；
5. 训练前必须固定阈值——**阈值变了，用户/物品词表就变了，历史 checkpoint 全部失效**。
