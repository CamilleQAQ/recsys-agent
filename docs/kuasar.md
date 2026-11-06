# KuaiSAR 2.0 数据说明

> **状态**：Phase 1.1（下载/校验/勘查）与 **Phase 1.2（全量画像）已完成**；Phase 1.3（正样本与切分定稿）待做。
> 下面所有数字都是**全量统计**（无采样），由 `python -m recsys.experiments.analyze_kuasar` 生成，
> 原始结果在 `artifacts/kuasar_profile.json`。
> **许可**：CC BY-NC-SA 4.0（非商业 + 相同方式共享）。数据不入库、不随代码分发。

## 1. 获取与复现

```powershell
python -m recsys.data.kuasar                      # 下载（805.9 MB）+ md5 校验 + 解压（3.5 GB）
python -m recsys.experiments.analyze_kuasar       # 全量画像 → artifacts/kuasar_profile.json
python -m recsys.data.kuasar --survey-only        # 离线复验目录与表头
```

- 来源：Zenodo record [8181109](https://zenodo.org/records/8181109)，论文 [arXiv 2306.07705](https://arxiv.org/abs/2306.07705)
- 压缩包：`KuaiSAR_v2.zip`，md5 `ed630f47fb013c5b31bb020c483ef262`（`recsys/data/kuasar.py` 校验）
- 解压位置：`data/kuasar/raw/KuaiSAR_v2/`

## 2. 文件与规模

| 文件 | 大小 | 行数 | 关键实体 |
| --- | --- | --- | --- |
| `rec_inter.csv` | 1102.6 MB | **14,605,716** | 25,877 用户 / 4,046,367 物品 |
| `src_inter.csv` | 527.8 MB | **5,059,169** | 25,705 用户 / 452,541 搜索会话 |
| `item_features.csv` | 1915.5 MB | **6,949,990** | 6,949,990 物品 / 2,399,630 作者 |
| `user_features.csv` | 0.4 MB | 25,877 | 覆盖全部推荐用户 |
| `social_network.csv` | < 0.1 MB | 612 | 关注边（极稀疏，暂不用） |

## 3. 字段（已核对表头）

### 3.1 `rec_inter.csv` — 推荐场景交互

```text
user_id, item_id, duration_ms, playing_time, timestamp, forward, like, follow,
search_photo_related, search, click, time
```

| 字段 | 含义 | 备注 |
| --- | --- | --- |
| `user_id` / `item_id` | 原始 ID | 需映射成连续索引 |
| `duration_ms` | 视频时长（毫秒） | 存在 `playing_time > duration_ms`（循环播放） |
| `playing_time` | 实际播放时长（毫秒） | 可算播放完成率 |
| `timestamp` | 毫秒级 epoch（float） | 与 `time` 冗余，优先用这个 |
| `time` | 日期时间字符串 | 人工排查用 |
| `forward` / `like` / `follow` | 转发 / 点赞 / 关注 | 0/1 |
| `search_photo_related` | 搜索图片相关 | 全量仅 **2 次**（可忽略） |
| `search` | 是否来自搜索 | 0/1，可区分渠道 |
| `click` | 是否点击 | 0/1 |

### 3.2 `src_inter.csv` — 搜索场景

```text
user_id, search_session_id, search_session_timestamp, search_session_source,
keyword, item_id, click_cnt, item_type, time
```

- `search_session_source` 实际有 **9 种取值**：`SUGGESTION`(150.0 万) / `USER_INPUT`(132.2 万) /
  `HISTORY`(84.9 万) / `UNKNOWN_PAGE`(52.2 万) / `TRENDING`(42.8 万) / `RELATED_FEED_QUERY`(26.7 万) /
  `PLACEHOLDER`(17.2 万)，另有 2 个极小枚举（19 与 17 行，疑似脏值 `UNKNOWN_ENUM_VALUE_…`）；
- `keyword`：分词 ID 列表（如 `[3345744, 3936332, 3847879]`），**不是原始文本**；
- `click_cnt`：该会话内该物品被点击次数 —— **88.4% 为 0**（曝光未点击），1 次占 11.0%，≥2 次占 0.6%；
- 与 `rec_inter` 没有共享交互 ID，只能按 user + item + 时间近似关联。

### 3.3 `item_features.csv` — 物品特征（18 列）

`item_id` + 四级类目（id/中文名/英文名）+ `caption`（分词 ID 列表）+ `author_id` + `item_type` +
`upload_time`（日期字符串 `YYYY-MM-DD`）+ `upload_type`。

### 3.4 `user_features.csv` / `social_network.csv`

- 用户：`onehot_feat1`、`onehot_feat2`、`search_active_level`、`rec_active_level`（活跃度分档）；
- 社交：`user_id → user_follow_id`，仅 612 条边，当前规划里不使用。

## 4. 全量统计（Phase 1.2 结论）

### 4.1 行为分布（14,605,716 行）

| 行为 | 次数 | 占比 |
| --- | --- | --- |
| `click` | 7,244,890 | **49.60%** |
| `like` | 459,741 | 3.15% |
| `follow` | 65,885 | 0.45% |
| `search` | 58,027 | 0.40% |
| `forward` | 44,752 | 0.31% |
| `search_photo_related` | 2 | 0.00% |
| **所有行为位全 0（纯曝光）** | **7,167,770** | **49.08%** |

- 点击与纯曝光几乎对半 → **数据集保留完整曝光流**，这是构造真实 `seen_mask` 的基础；
- `(user_id, item_id)` 重复率仅 **0.2111%**（14,574,889 个唯一对），说明记录基本是「按次曝光」，
  不需要像 KuaiRec 那样为去重单独设计策略；
- 时间跨度 **18.78 天**（`1684738200151` → `1686360599225` ms，2023-05-22 ~ 2023-06-10），
  适合按用户做时间切分，但测试期只有约 19 天。

### 4.2 用户与物品分布

| 维度 | 数值 |
| --- | --- |
| 用户数 | 25,877（与 `user_features` 完全一致，人人有画像） |
| 物品数（交互中出现） | 4,046,367 |
| 物品数（特征表） | 6,949,990（多出约 290 万未交互物品，可做冷启动池） |
| 每用户交互数 | 均值 564、中位 **171**、p90 1,439、p99 ≈ 5,623、最大 **32,885** |
| 物品流行度 | 均值 3.6、中位 **1**、p90 6、p99 38、最大 2,779 |
| 只出现 1 次的物品 | **2,486,466（61.4%）** |
| 交互物品的特征覆盖率 | **99.9999%**（4,046,363 / 4,046,367） |
| caption token | 中位 15、均值 23.2、最大 24,180（极端长文本需截断） |
| 物品类型 | NORMAL 6,861,074 / **AD 63,634** / UNKNOWN 25,259 / LIVEGAME_PLAYBACK 23 |

**解读**：典型「重目录、重长尾」结构——用户少而活跃（人均上百条），物品多且六成只出现一次。
这决定了两件事：

1. **全量排序的成本**：4.05M 物品 × 64 维 float32 ≈ **1 GB** embedding；单次全库打分 = 25,877 × 4.05M
   ≈ 1.05e11 次点积，CPU 上分钟级、必须分块或上 FAISS。Phase 2 要在「精确暴力评测」与「FAISS 近似」之间明确取舍；
2. **长尾物品**：61.4% 的物品只被看过一次，item 侧 embedding 学不动，
   冷启动/类目特征（Phase 1.5）不是可选项。

### 4.3 播放完成率（14.6M 行）

| `playing_time / duration_ms` | 行数 | 占比 |
| --- | --- | --- |
| `[0, 0.1)` | 5,100,858 | 34.9% |
| `[0.1, 0.25)` | 2,257,755 | 15.5% |
| `[0.25, 0.5)` | 1,545,177 | 10.6% |
| `[0.5, 0.75)` | 945,112 | 6.5% |
| `[0.75, 1.0)` | 1,054,929 | 7.2% |
| `[1.0, 1.5)` | 2,226,056 | 15.2% |
| `[1.5, 2.0)` | 398,413 | 2.7% |
| `[2.0, inf)` | 548,191 | 3.8% |

- **317 万行（21.7%）的 `playing_time > duration_ms`**（循环/重播），算完成率时要截断到 1.0 或单独标记；
- **52.9 万行 `duration_ms == 0`**，不能拿来做分母；
- 结论：`click` 是可靠的二值标签；`播放完成率` 可做辅助标签/特征，但必须先处理上面两类异常。

### 4.4 搜索侧（5,059,169 行）

- 452,541 个会话、25,705 个用户；平均每会话 11.2 条候选，中位 5 条，最大 926 条；
- 会话来源 9 种：`SUGGESTION`(150.0 万) / `USER_INPUT`(132.2 万) / `HISTORY`(84.9 万) /
  `UNKNOWN_PAGE`(52.2 万) / `TRENDING`(42.8 万) / `RELATED_FEED_QUERY`(26.7 万) /
  `PLACEHOLDER`(17.2 万)，另有 2 个极小脏值（19 与 17 行）；
- `click_cnt`：**88.4% 为 0**（曝光未点击），11.0% 为 1，0.6% ≥2；
- 每用户搜索行数：中位 68、p99 1,742、最大 16,083。

### 4.5 数据质量清单

| 问题 | 规模 | 处理 |
| --- | --- | --- |
| `upload_time` 存在 1970-01-01 哨兵值 | 待统计（全量最小值为该日期） | 视为缺失，不做时间特征 |
| `duration_ms == 0` | 529,225 行 | 不算完成率 |
| `playing_time > duration_ms` | 3,172,382 行 | 截断或标记 |
| 搜索来源脏枚举 | 36 行 | 归入 `UNKNOWN` |
| 广告类物品 `item_type == AD` | 63,634 个物品 | 是否排除由实验口径决定，需在 1.3 明确 |
| `(user,item)` 重复 | 0.2111% | 保留（按次曝光），若按对聚合需显式记录策略 |

## 5. 对设计的影响

1. **正样本**：MVP 用 `click == 1`；`like/follow/forward` 作为多任务标签（Phase 8）。
   KuaiSAR 没有 `watch_ratio`，不要套 KuaiRec 阈值。
2. **曝光 mask**：`rec_inter` 全部行（含 click=0）即曝光集合，按时间切分后构造 `seen_mask`，
   与 PROJECT_MASTER §6.1 的召回评估口径一致。
3. **切分**：推荐任务继续用 `per_user_last_two_strict_timestamp_v2`；
   搜索侧是会话结构，若要评估搜索任务需按 `search_session_id` 切分（Phase 1.3 决定，且不阻塞推荐主线）。
4. **渠道字段** `search`：默认不按渠道过滤，但保留字段，便于后续「搜索来自的交互是否提升推荐」的实验。
5. **特征**：`item_features` 6.95M 行、caption 是 token 列表，读取必须按列取、按需切分，不能整表进内存。
6. **规模**：物品词表 4M 级 → 训练侧的 in-batch 负采样没问题，评测侧必须规划分块/FAISS；
   `PreparedData` 目前用 Python list 存交互，14.6M 行会很吃内存（Phase 1.6 要换数组化/缓存）。

## 6. 尚未完成（Phase 1.3 / 1.4 / 1.5 / 1.6）

- [ ] 正样本与阈值定稿，写入 `recsys.config` 与 `configs/datasets/kuasar.toml`；
- [ ] 切分单位决定（推荐按用户 / 搜索按会话），并检查泄漏；
- [ ] ID 映射 + `PreparedData` 构造（`prepare_kuasar`），并接入 `loader.SUPPORTED_DATASETS`；
- [ ] 序列样本与候选+特征样本；
- [ ] 特征通道（类目/作者/活跃度）；
- [ ] `processed/` 缓存与 `protocol.json`（14.6M 行不做缓存，每次评测重解析不可接受）。
