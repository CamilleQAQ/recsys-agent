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
| 广告类物品 `item_type == AD` | 63,634 个物品 | **保留**（2026-11-06 决定），过滤能力等 1.5 特征通道 |
| `(user,item)` 重复 | 0.2111% | 正样本按对去重保留首次点击；曝光保留全部行 |

## 4.6 切分与 processed 存储（Phase 1.3）

### 为什么按「请求」切分，而不是按单条点击

KuaiSAR 的 `timestamp` 是**按请求记录**的：同一次推荐请求里点开的多个视频共享同一个毫秒时间戳。
按「最后三次点击严格递增」的旧规则，实测 **25,877 个用户里只有 3,752 个（14.5%）**能进评测
（14,684 人最后两次点击同毫秒、7,710 人最后三次都在同一毫秒）。

因此改为 **请求 = `(user_id, timestamp)`** 为单位切分：最后一个请求做测试、倒数第二个做验证。
这样既恢复了评测覆盖，也避免「同一请求内的物品被拆到 train/test」造成的泄漏。

### 全量切分结果（`data/kuasar/processed/`，253 MB / 17 个 npy）

| 项 | 数值 |
| --- | --- |
| 正样本（去重后） | 7,237,037（原始 7,244,890，按 `(user,item)` 去重 7,853 条） |
| 训练集 | 7,118,891 |
| 验证集（倒数第二个请求） | 67,517 行 |
| 测试集（最后一个请求） | 50,629 行 |
| 可评测用户 | **24,128 / 25,877（93.2%）** |
| 请求不足 3 个的用户 | 1,749 |
| 每用户请求数中位 | 33 |
| 曝光对（验证 / 测试） | 14,311,397 / 14,443,563 |
| 介于验证与测试之间被丢弃的行 | 0 |

**完整性校验**（脚本实跑）：

- 24,128 个可评测用户中，`train_max < val_min` 与 `val_max < test_min` 的比例均为 **100%**；
- 验证/测试目标物品**在训练历史中的比例均为 0.00%**（去重 + 请求切分共同保证无泄漏）。

### 存储布局

```
data/kuasar/processed/          （gitignore，可重建）
├── ids/{user_ids,item_ids}.npy             index -> 原始 ID（可逆）
├── split/{train,validation,test}_{users,items,timestamps}.npy
├── exposure/{validation,test}_{indptr,items}.npy   按用户 CSR，含 click=0 的曝光
├── features/*.npy                          12 列物品特征（1.5）
├── features/schema.json                    列语义、基数、填充率、每列 sha256
├── stats.json
└── protocol.json              口径、指纹（各数组 sha256）、生成时间
```

读取用 `recsys.data.kuasar_store.load_processed()`，默认 `mmap`，常驻内存接近 0。
烘焙一次 36 秒（14.6M 行解析 + 去重 + 切分 + 落盘）；特征表另需 46 秒、139 MB。

## 4.7 物品特征通道（Phase 1.5）

`item_features.csv` 有 6,949,990 行，但只保留**交互词表内的 4,046,363 个物品**
（与词表 4,046,367 差 4 个，覆盖率 99.9999%），按 item_idx 对齐存成 12 列 npy：

| 列 | 类型 | 说明 | 填充率 |
| --- | --- | --- | --- |
| `category_first_id` / `_code` | int32 | 一级类目（37 类） | 100% |
| `category_second_id` / `_code` | int32 | 二级类目（293 类） | 100% |
| `author_id` | int32 | 作者原始 ID | 100% |
| `author_freq` | int32 | 作者在词表内的作品数（中位 7） | 100% |
| `item_type_code` | int8 | NORMAL / AD / UNKNOWN（交互物品里出现 3 类） | 100% |
| `upload_type_code` | int8 | 上传来源（30 种） | 100% |
| `upload_days` | int32 | 上传日期（1970 起天数） | 99.73% |
| `upload_valid` | int8 | 日期是否有效（1970 哨兵 = 0） | 99.73% |
| `caption_tokens` | int32 | caption 的 token 数（不含内容） | 100% |
| `is_ad` | int8 | 广告物品标记：**10,265** 个（全表 63,634 个 AD 物品中进入交互词表的部分） | — |

用法：

```python
from recsys.data.loader import prepare_dataset
data = prepare_dataset("kuasar", Path("data"))
feats = data.item_features            # ItemFeatures | None
codes = feats.column("category_first_code")
```

要点：

- 特征表是**可选**的：没构建时 `item_features=None`，训练/评测不受影响；
- 长度必须与物品词表一致，否则 `prepare_kuasar` 直接报错（防止用旧缓存的特征配新词表）；
- 指纹写进 checkpoint 的 `provenance.item_features`，换特征表会立刻可见；
- 广告过滤能力已就位（`is_ad`），但**默认不过滤**（2026-11-06 决定，贴近线上曝光分布）。

## 4.8 数据版本与缓存校验（Phase 1.6）

`processed/` 与特征表都会在读取时校验，防止「代码换了、缓存还是旧的」：

| 机制 | 说明 |
| --- | --- |
| 文件指纹 | 每个 npy 的 sha256 记在 `protocol.json`（processed）与 `features/schema.json`（特征表） |
| `data_version` | 所有文件哈希的汇总短哈希（当前 `a643b5958660f7df`），便于人工对比 |
| 读取即校验 | `load_processed()` / `load_item_features()` 默认校验（约 1–2 秒 / 0.5 秒），不一致直接抛错 |
| 构建可复用 | 不带 `--force` 时先校验，通过则打印 `[reuse]` 直接返回；失败则提示并重建 |
| 命令 | `python -m recsys.data.kuasar_store --check` / `--force`；`python -m recsys.data.kuasar_features --check` / `--force` |

实测：把 `ids/user_ids.npy` 追加一个字节后，`verify_store` 返回「哈希不一致」；
删除文件返回「缺少文件」；正常状态下 `--check` 输出 `data_version`。

## 4.9 排序候选与特征样本（Phase 1.4 收尾）

召回用「正样本对」，排序用「一次请求的曝光候选 + 点击标签」。`rec_inter` 的一行就是一次曝光，
同一 `(user_id, timestamp)` 的所有行构成该请求的候选集：

```powershell
python -m recsys.data.kuasar_ranking            # 约 54 秒，产出 ranking/（约 250 MB）
python -m recsys.data.kuasar_ranking --check
```

| 切分 | 候选数 | 请求数 | 点击率 | 每请求候选（均值/中位/最大） |
| --- | --- | --- | --- | --- |
| train | 14,353,430 | 3,104,325 | 49.65% | 4.62 / 5 / 12 |
| validation | 113,902 | 24,128 | 59.32% | 4.72 / 5 / 10 |
| test | 88,417 | 24,128 | 57.29% | 3.66 / 4 / 10 |
| 丢弃（验证与测试之间） | 49,967 | — | — | — |

> 行数对账：14,353,430 + 113,902 + 88,417 + 49,967 = 14,605,716（一行不多、一行不少）。

**已知偏差**：验证/测试请求的点击率（59%/57%）明显高于训练集（50%）。原因是留出请求
本身就是「包含用户最后点击的那次请求」，天然点击更密；做 CTR 校准类分析时要记住这一点，
不要直接把验证集点击率当成线上先验。

数据布局：

```
data/kuasar/processed/ranking/
├── {train,validation,test}_items.npy    候选 item_idx（int32，按请求连续）
├── {train,validation,test}_labels.npy   点击标签（int8）
├── {train,validation,test}_users.npy    候选所属用户（int32）
├── {train,validation,test}_groups.npy   每个请求的候选数（int32，LightGBM 的 group）
├── user_levels.npy                      用户活跃度（num_users × 2，int8）
└── schema.json                          口径、计数、每文件 sha256
```

特征**不落盘**：候选矩阵在训练时按批 join 物品/用户特征（14M × 11 列 float32 ≈ 630 MB，
整表物化不划算）：

```python
import numpy as np
from pathlib import Path
from recsys.data.loader import prepare_dataset
from recsys.data.kuasar_ranking import load_ranking_samples, iter_candidate_batches
from recsys.data.kuasar_store import dataset_dir

root = dataset_dir(Path("data"))
data = prepare_dataset("kuasar", Path("data"))          # 物品特征挂在 data.item_features
samples = load_ranking_samples(root, "train")
user_levels = np.load(root / "processed" / "ranking" / "user_levels.npy", mmap_mode="r")

for X, y, groups in iter_candidate_batches(samples, data.item_features, user_levels=user_levels):
    ...   # X: [rows, 11]，y: [rows]，groups: 本批每请求的候选数
```

特征列：物品侧 9 列（类目×2、作者频次、item_type、upload_type、上传日期/有效性、caption token 数、`is_ad`）
+ 用户侧 2 列（搜索活跃度、推荐活跃度）。



## 5. 对设计的影响

1. **正样本**：MVP 用 `click == 1`；`like/follow/forward` 作为多任务标签（Phase 8）。
   KuaiSAR 没有 `watch_ratio`，不要套 KuaiRec 阈值。
2. **曝光 mask**：`rec_inter` 全部行（含 click=0）即曝光集合，已按用户存成 CSR，
   评测时按目标时刻过滤，与 PROJECT_MASTER §6.1 的召回评估口径一致。
3. **切分**：请求级 `per_user_last_two_requests`（不是 KuaiRec 的点击级规则）；
   **搜索侧不纳入 Phase 1**（2026-11-06 决定），`src_inter` 留到 Phase 5 搜推联合。
4. **渠道字段** `search`：默认不按渠道过滤，但保留字段，便于后续「搜索来自的交互是否提升推荐」的实验。
5. **特征**：`item_features` 6.95M 行、caption 是 token 列表，读取必须按列取、按需切分，不能整表进内存。
6. **规模（关键约束）**：物品词表 4M 级，当前的 `PreparedData` 表示方式放不下——
   7.24M 条 `Interaction` 对象约 1.2 GB，而**稠密 seen_mask = 25,877 × 4,046,367 bool ≈ 97.5 GiB**，
   完全不可行。因此 1.3 落的是「npy 数组 + 按用户稀疏曝光」的 processed 存储（见 4.6），
   `PreparedData` 的数组后端在 1.4 接。
7. **内存**：15.7 GB 总量；清理后台后可用 5.3–5.6 GB。全量烘焙一次约 36 秒，峰值约 1.5 GB；
   后续训练仍建议先确认可用内存 ≥ 6 GB。

## 6. 尚未完成（Phase 1.3 / 1.4 / 1.5 / 1.6）

- [x] 口径决定：正样本 `click == 1`；广告物品保留；搜索侧不纳入 Phase 1（2026-11-06）
- [x] **processed 存储**：`kuasar_store.py` 产出 npy + CSR 曝光 + `protocol.json`（253 MB，24,128 可评测用户）
- [x] `PreparedData` 数组后端：`ArrayInteractions` + `SparseExposure` + `SortedIndexLookup`，
      `prepare_dataset("kuasar")` 已可用（`prepare_kuasar`）
- [x] 切分策略按数据集参数化：`split_strategy_for()`，checkpoint 写 `prepared_data.split_strategy`
- [x] 召回样本（交互对）与序列样本可在数组后端上构造（`InteractionDataset` / `SASRecTrainDataset`）
- [x] **物品特征通道**：12 列 npy（类目/作者/上传类型/日期/caption token/`is_ad`），139 MB，
      `PreparedData.item_features` 已接通，指纹写进 provenance
- [x] **数据版本与缓存校验**：每个 npy 的 sha256 + `data_version`，读取时默认校验，
      构建可复用（`[reuse]`），`--check` / `--force` 两个入口（见 §4.8）
- [x] **排序候选 + 特征样本**：`kuasar_ranking.py` 按请求聚合曝光候选（train 14.35M / val 113.9k / test 88.4k），
      11 列特征按批 join，`groups` 直接可用作 LightGBM 分组（见 §4.9）
- [ ] 全量训练的资源规划：4.05M 物品 × 64 维 embedding ≈ 1 GB，Adam 状态再 2 GB，
      16 GB/CPU 机器上先做小维度试跑（Phase 2 正式定）


