# KuaiRec 数据线（流程模板 / 辅助数据线）

> **定位**：项目主数据线是 **KuaiSAR**（见 [kuasar-roadmap.md](kuasar-roadmap.md)）。
> KuaiRec 这条线保留两个用途：
> 1. **流程模板**——新的数据集读取器按它的结构写（下载 → md5 校验 → 多布局探测 → 流式解析 → `PreparedData`）；
> 2. **辅助实验线**——KuaiSAR 还没接入时，用它验证训练/评测/适配层是否正常。
>
> 因此下面的口径、命令仍然有效，但**不要**再把它当作「最终实验数据」。

KuaiRec 2.0（快手短视频，含 `watch_ratio` 等真实观看反馈）。

## 口径（改动会让历史 checkpoint 失效）

| 项 | 值 | 定义位置 |
| --- | --- | --- |
| 正反馈 | `watch_ratio >= 1.0` | `recsys.config.KUAIREC_WATCH_RATIO_THRESHOLD` |
| 重复 (user, video) | 合并成一条，保留最完整的观看记录 | `KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS` |
| 切分 | 每个用户最后两个正反馈 = validation / test | `recsys.data.schema.temporal_train_validation_test_split` |
| 候选集 | 全量物品 | 评测脚本统一行为 |
| 过滤 | 目标时刻前**真实曝光过**的视频（含未看完） | `PreparedData.test_seen_mask` |

`big_matrix.csv` 里约 17.8% 的行是重复的 `(user, video)`（代码注释记录），其中一部分是同一次播放被记多行，
另一部分是不同时间的重复播放。不合并的话，测试目标会出现在训练历史里，被「已看过就过滤」规则吃掉，
指标会虚高——所以合并是评测正确性问题，不是数据清洗偏好。

## 命令

```powershell
# 下载 + 校验 md5 + 解压（首次约 1 GB，之后走本地文件）
python -m recsys.data.prepare --dataset kuairec

# 数据体检：重复率、长尾分布、每用户交互数（输出 JSON）
python -m recsys.experiments.analyze_kuairec --data-dir data/kuairec --output artifacts/kuairec_data_stats.json

# 训练三个基线
python -m recsys.training.train_mf      --dataset kuairec
python -m recsys.training.trainer       --dataset kuairec
python -m recsys.training.train_sasrec  --dataset kuairec

# 评测（统一入口，换模型只改 --model）
python -m recsys.adapters evaluate --model sasrec --checkpoint artifacts/kuairec/sasrec.pt --split test
python -m recsys.evaluation.faiss_evaluator --checkpoint artifacts/kuairec/two_tower.pt --data-dir data
```

## 文件放哪

```
data/kuairec/
├── raw/KuaiRec.zip               原始压缩包（保留，便于校验 md5）
├── raw/big_matrix.csv
├── raw/small_matrix.csv
├── raw/item_categories.csv
├── raw/user_features.csv
└── processed/                    id_mappings.json / interactions.csv / protocol.json（人工核查用）
```

`resolve_kuairec_file` 同时接受 `data/kuairec/big_matrix.csv`、`raw/`、`raw/KuaiRec 2.0/data/`
等几种官方解压布局，换机器不用改代码。写 KuaiSAR 的读取器时直接照抄这套结构。

## 坑（写 KuaiSAR 读取器时会重复遇到）

- **下载校验**：`KUAIREC_ARCHIVE_MD5` 不匹配会直接抛错，别手工跳过——半截 zip 会解析出静默错误的数据；
- **内存**：`iter_matrix_rows` 是流式读取，但 `prepare_kuairec` 会把正反馈整体放进内存。
  KuaiSAR 解压后规模更大，必须换成分块/数组化（见 roadmap 第 7 节）；
- **时间戳单位不一**：KuaiRec 是毫秒级 float，MovieLens 是秒级整数；统一走 `Interaction.timestamp`（float）；
- **交互太少的用户**不参与评测（< 3 条正反馈），统计用户数时别用原始用户数；
- **阈值必须训练前定死**：改了阈值，用户/物品词表变化，所有 checkpoint 与已发布指标失效。

## 已知未做

- `processed/` 只写不读：评测每次重新解析原始 CSV。真要提速，需要加带版本号的缓存并校验 `protocol.json`；
- 没有 item/user 侧特征通道（`PreparedData` 只有 ID），多模态/特征类模型（MMRec、DeepCTR、FuxiCTR）暂时接不进来。
