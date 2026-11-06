# KuaiSAR 2.0 数据说明

> **状态**：Phase 1.1（下载/校验/结构勘查）已完成；Phase 1.2（字段核对与画像）进行中。
> 下面的字段来自真实表头，行为占比来自**前 30 万行样本**（不是全量统计），全量画像完成前不要引用为最终数字。
> **许可**：CC BY-NC-SA 4.0（非商业 + 相同方式共享）。数据不入库、不随代码分发。

## 1. 获取

```powershell
python -m recsys.data.kuasar                # 完整版 v2（805.9 MB 压缩包，解压 3.5 GB）
python -m recsys.data.kuasar --small        # KuaiSAR-small（488 MB）
python -m recsys.data.kuasar --survey-only  # 只勘查，不联网
```

- 来源：Zenodo record [8181109](https://zenodo.org/records/8181109)，论文 [arXiv 2306.07705](https://arxiv.org/abs/2306.07705)
- 压缩包：`KuaiSAR_v2.zip`，md5 `ed630f47fb013c5b31bb020c483ef262`（由 `recsys/data/kuasar.py` 校验）
- 解压后落在 `data/kuasar/raw/KuaiSAR_v2/`，勘查结果写入 `artifacts/kuasar_archive_survey.json`

## 2. 文件清单

| 文件 | 大小 | 内容 |
| --- | --- | --- |
| `rec_inter.csv` | 1102.6 MB | 推荐场景的曝光/交互记录 |
| `src_inter.csv` | 527.8 MB | 搜索场景的会话与点击记录 |
| `item_features.csv` | 1915.5 MB | 物品（视频）特征：类目、caption token、作者、上传信息 |
| `user_features.csv` | 0.4 MB | 用户 onehot 特征与搜/推活跃度 |
| `social_network.csv` | < 0.1 MB | 用户关注关系 |

## 3. 字段（已核对表头）

### 3.1 `rec_inter.csv` — 推荐场景交互

```text
user_id, item_id, duration_ms, playing_time, timestamp, forward, like, follow,
search_photo_related, search, click, time
```

| 字段 | 含义（据样本与列名） | 备注 |
| --- | --- | --- |
| `user_id` / `item_id` | 原始 ID | 需映射成连续索引 |
| `duration_ms` | 视频时长（毫秒） | 样本里出现 `playing_time > duration_ms`（循环播放） |
| `playing_time` | 实际播放时长（毫秒） | 可做「播放完成率」特征 |
| `timestamp` | 毫秒级 epoch（float） | 与 `time` 冗余，优先用这个 |
| `time` | 日期时间字符串 | 便于人工排查 |
| `forward` | 是否转发 | 0/1 |
| `like` | 是否点赞 | 0/1 |
| `follow` | 是否关注 | 0/1 |
| `search_photo_related` | 搜索图片相关 | 样本中全 0，需全量确认 |
| `search` | 是否来自搜索 | 0/1，可区分渠道 |
| `click` | 是否点击 | 0/1 |

**关键事实（30 万行样本）**：48.88% 的行所有行为位都是 0 —— 这些是**曝光未点击**记录。
数据集保留了曝光，因此 `seen_mask` 可以按「真实曝光」构造，而不是退化成「只看过正反馈」。

样本行为占比：click 49.59%、like 3.45%、follow 0.51%、forward 0.30%、search 0.43%。
样本内唯一 `(user_id, item_id)` 占 99.97%（重复率 0.03%），说明记录基本是「按次曝光」而非聚合行。

### 3.2 `src_inter.csv` — 搜索场景

```text
user_id, search_session_id, search_session_timestamp, search_session_source,
keyword, item_id, click_cnt, item_type, time
```

- `search_session_source`：样本值为 `USER_INPUT`，是否还有其他取值待全量确认；
- `keyword`：**不是原始文本**，而是分词后的 ID 列表（如 `[3345744, 3936332, 3847879]`）；
- `click_cnt`：该会话内该物品被点击次数；
- 与 `rec_inter` 的关联只能通过 `user_id` + `item_id` + 时间近似，**没有共享的交互 ID**。

### 3.3 `item_features.csv` — 物品特征（18 列）

```text
item_id, first/second/third/fourth_level_category_id + _name (+ 英文名列),
caption, author_id, item_type, upload_time, upload_type
```

- `caption` 是分词 ID 列表，不是自然语言文本；
- 没有视频/封面 embedding，要做多模态得自己从平台侧另行处理（本项目不做）；
- 可用于冷启动类目特征与 LightGBM/FuxiCTR 的类别特征。

### 3.4 `user_features.csv`

```text
user_id, onehot_feat1, onehot_feat2, search_active_level, rec_active_level
```

`*_active_level` 是活跃度分档（样本取值 0–5），可直接做用户侧特征或分层评测。

### 3.5 `social_network.csv`

```text
user_id, user_follow_id
```

可做社交增强召回（当前不在 Phase 1 范围）。

## 4. 对设计的影响

1. **正样本定义**：MVP 用 `click == 1`；`like/follow/forward` 留作多任务标签（Phase 8）。
   不要沿用 KuaiRec 的 `watch_ratio >= 1`——KuaiSAR 没有这个字段。
2. **曝光 mask**：`rec_inter` 的全部行（含 click=0）即曝光集合，按时间切分后可构造真实的 seen-mask，
   这比 KuaiRec/MovieLens 的口径更严格，也与 master §6.1 的召回评估要求一致。
3. **时间切分**：`timestamp` 是毫秒 float，可继续用 `per_user_last_two_strict_timestamp_v2`；
   但搜索侧 `src_inter` 是会话结构，若要评估搜索任务，需要 request/session 级切分（Phase 1.3 决定）。
4. **渠道**：`rec_inter.search` 可以区分「推荐流里的交互」与「从搜索进入的交互」，
   是否需要按渠道过滤，取决于实验问题（默认 MVP 不区分，但记录该字段）。
5. **特征**：`item_features` 1.9 GB、caption 是 token 列表，读取时要按需取列，不能整表进内存。

## 5. 尚未完成（Phase 1.2 / 1.3）

- [ ] 全量统计：行数、用户数、物品数、行为占比、每用户交互数分布、时间覆盖、缺失值；
- [ ] 确认 `search_photo_related` 是否全 0；`search_session_source` 的取值集合；
- [ ] `item_features` 与 `rec_inter` 的 item 覆盖关系（内外连接比例）；
- [ ] 正样本阈值/行为组合定稿，写入 `recsys.config` 与 `configs/datasets/kuasar.toml`；
- [ ] 切分单位决定（按用户 / 按会话），并检查数据泄漏；
- [ ] ID 映射与 `processed/` 缓存（带 `protocol.json` 校验）。
