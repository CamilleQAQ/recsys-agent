# 2026-11-09 · KuaiSAR Search MVP：词表核验、小样本与 BM25

> 本轮只做搜索链路第一段：核验 `keyword` / `caption` 词表，准备 30 个搜索会话的小样本，
> 建 BM25 索引，提供 `search(query_tokens, top_k)` 与独立 CLI。没有训练、没有全量检索评测、
> 没有 Search Agent、向量检索、排序或重排。

## 结论

- **词表**：archive 没有单独词表文件；论文说明 query 与 video title 共用分词/整数哈希；
  抽样中 keyword 与 caption 的 token 范围一致、交集是均匀独立基线的 **5.69×**，
  点击物品的 token 重合率 **51.16%**、相对随机 caption **73.32×**。结论：共用 token ID 空间，
  **直接使用 KuaiSAR 原始 token ID**，没有改用本地夹具。
- **样本**：扫描 `src_inter` 前 **200,000** 行（17,036 个完整会话）与 `item_features` 前
  **500,000** 行；保留 **30 个会话 / 120 个物品**，曝光行关联覆盖率 **55.814%**
  （120 / 215），点击行关联 **43 / 60**。两份输入均按上限截断，已标记。
- **索引**：BM25（k1=1.5, b=0.75），**120 docs / 823 vocab / avgdl 16.266667**，
  索引 76,077 B，sha256 `3ae2b858…`。
- **冒烟**：只运行一次 `python -m recsys.search.smoke`，有匹配 / 空查询 / 无匹配三组
  共 **9/9 检查通过**；有匹配时 4-token 查询返回 7 个匹配，Top-5 分数严格降序，
  点击物品 80675 排第 1。

## 1. 数据来源

| 项 | 值 |
| --- | --- |
| 数据集 | KuaiSAR 2.0（完整版 v2） |
| 来源 | Zenodo record [8181109](https://zenodo.org/records/8181109)；论文 [arXiv:2306.07705](https://arxiv.org/abs/2306.07705) |
| 压缩包 md5 | `ed630f47fb013c5b31bb020c483ef262`（`KUASAR_ARCHIVE_MD5`，与 `configs/datasets/kuasar.toml` 一致） |
| 搜索日志 | `data/kuasar/raw/KuaiSAR_v2/src_inter.csv`（553,388,618 B） |
| 物品特征 | `data/kuasar/raw/KuaiSAR_v2/item_features.csv`（2,008,531,649 B） |
| 许可 | CC BY-NC-SA 4.0；数据不入库 |

本轮只读上述两个 CSV 的前缀，没有读取/重写 `rec_inter.csv`，也没有改动
`data/kuasar/processed/` 下的既有产物。

## 2. 词表核验

### 2.1 证据

1. **仓库文档**：`docs/kuasar.md` 已记录 `keyword` 与 `caption` 都是分词 ID 列表；
   两边都不是原始文本。
2. **数据包元信息**：`KuaiSAR_v2.zip` 只有 5 个 CSV，没有 vocab/dictionary/tokenizer 配置。
3. **论文陈述**（§3.2 Data Construction）：

   > “Textual information, such as queries and video titles, underwent word segmentation
   > and sensitive word removal. Furthermore, terms in texts were randomly hashed into integers.”

4. **抽样观测**（200k 搜索行 / 500k 物品行）：

| 指标 | 数值 |
| --- | ---: |
| keyword 唯一 token | 15,339 |
| caption 唯一 token（500k 物品） | 666,385 |
| 交集 | **13,747** |
| 均匀独立词表期望交集 | 2,414.36 |
| 实际 / 期望 | **5.69×** |
| keyword token 范围 | 22 – 4,233,695 |
| caption token 范围 | 2 – 4,233,996 |
| 点击且已关联的 (query,item) 对 | 43 |
| 其中至少 1 个 token 重合 | 22（**51.16%**） |
| 已关联曝光 (query,item) 对 | 120 |
| 其中至少 1 个 token 重合 | 74（**61.67%**） |
| 随机 caption 基线（5,000 个，seed 42） | 0.012053 |
| 点击物品平均重合 / 随机基线 | **73.32×** |
| 按会话比较点击 vs 随机 | 19 胜 / 2 平 / 9 负 |

### 2.2 结论与边界

- **结论**：`keyword` 与 `caption` 使用同一套 KuaiSAR token ID；本轮用原始 token ID 建索引。
- **没有使用本地夹具**：证据满足脚本的 `use_kuasar_token_ids` 判定。
- **残余风险**：token 是随机哈希后的整数，无法反解为文本；没有官方词表文件。
  论文只说明 queries/video titles 共用哈希。若上游 tokenizer 变化，需要重跑
  `prepare.py` 并检查 `vocab_evidence.decision`。
- **未验证**：全量 KuaiSAR 检索质量、Recall/NDCG、人工相关性均未评估。

## 3. 采样与关联

### 3.1 扫描上限

| 输入 | 上限 | 实际 | 截断 |
| --- | ---: | ---: | --- |
| `src_inter.csv` | 200,000 行 | 200,000 行 | 是 |
| `item_features.csv` | 500,000 行 | 500,000 行 | 是 |

- 扫描上限落在会话中间时丢弃最后一个不完整会话；本轮 `partial_sessions_dropped = 0`。
- 同一会话 keyword 一致性检查通过：`keyword_mismatch_sessions = 0`。
- 先扫会话得到 163,902 个目标 `item_id`，再扫物品前缀，命中 20,407 个；
  其余目标物品保留为未关联记录。
- 现有 `data/kuasar/processed/features/caption_tokens.npy` 只有 token 数、没有 token 内容，
  无法直接复用来建 BM25；本轮读原始 `item_features.csv` 的 `caption` 列，未重建 processed。
- 前缀扫描偏向较小 `item_id`，这是已知偏差，不是随机抽样。

### 3.2 选择规则

1. keyword 非空且会话内一致；
2. 至少 2 个不同曝光物品；
3. 已关联 caption 的物品占比 ≥ 50%；
4. 优先有可关联点击物品；其次曝光物品更多、覆盖率更高、文件顺序更早；
5. 最多 30 个会话；物品池取已关联物品并集，按 item_id 排序，最多 1,000 个。

### 3.3 实际计数

| 项 | 数值 |
| --- | ---: |
| 完整搜索会话 | 17,036 |
| 候选会话 | 190 |
| **保留会话** | **30** |
| 有可关联点击的会话 | 30 |
| 曝光行 | 215 |
| 已关联曝光行 | 120（**55.814%**） |
| 唯一曝光物品 | 215 |
| 唯一已关联物品 | 120（**55.814%**） |
| 点击行 | 60 |
| 已关联点击行 | 43（71.67%） |
| 唯一点击物品 / 已关联 | 60 / 43 |
| 会话覆盖率 mean / min / max | 56.971% / 50% / 80% |
| 100% 覆盖会话 | 0 |
| **索引物品数** | **120**（上限 1,000，未截断） |

未关联物品保留在 `sample.json` 的 `exposures` 中并标记 `joined=false`；
索引只含 `joined=true` 的物品。逐条数据只保存这一份会话样本，物品 caption/类目只放在
`index.json`，没有重复导出物品 CSV/JSON。

## 4. BM25 索引

- 公式：`idf = ln(1 + (N - df + 0.5)/(df + 0.5))`；
  `score = Σ qtf·idf·tf·(k1+1) / (tf + k1·(1-b+b·|d|/avgdl))`。
- 参数：`k1 = 1.5`，`b = 0.75`；查询重复 token 计入 `qtf`。
- 排序：`(-score, item_id)`；空查询 / 无匹配返回 `[]`。
- 接口：`BM25Index.search(query_tokens, top_k)`，返回 `rank/item_id/score/category/...`。
- 索引加载时从 `items` 重算并核对 `num_docs` / `vocab_size` / `avgdl`。

| 项 | 值 |
| --- | ---: |
| docs | 120 |
| vocab | 823 |
| avgdl | 16.266667 |
| total tokens | 1,952 |
| doc len min / max | 1 / 79 |
| index size | 76,077 B |
| index sha256 | `3ae2b858d456eab4c4c21ed356812cd44260b56dd596639588a53fb1a6a52690` |
| sample size | 54,651 B |
| sample sha256 | `12a46f3c3a614a715e52f4a790ccac93eb0fc7519ab5ff4a0826ceace58a7daf` |
| build summary sha256 | `eba233325c6554b93384642c844b4a911e86d27d62cae7bc0a30772df2a38238` |

产物在 `artifacts/search/kuasar_bm25/`（Git 忽略）；验收摘要
[build_summary.json](build_summary.json) 与本报告入库。

## 5. 执行命令

```powershell
$env:PYTHONPATH = "src"

# 建索引（最终产物由本次成功的 --force 构建生成）
python -m recsys.search.prepare --force `
  --summary-out experiments/2026-11-09-kuasar-search-bm25/build_summary.json

# 查询 CLI（示例）
python -m recsys.search --tokens "3538762,3181091,4061567,3525929" --top-k 5

# 单次冒烟验收
python -m recsys.search.smoke --top-k 5 `
  --output experiments/2026-11-09-kuasar-search-bm25/smoke.json

# 产物校验
python -m recsys.search.prepare --check
```

最终构建耗时 **8.322 s**（200k + 500k 行扫描 + 索引落盘）。

## 6. 冒烟验收结果

**只运行一次**，9/9 检查通过：

| 检查 | 结果 |
| --- | --- |
| index_loaded | ok |
| match_found | ok |
| top_k_respected | ok |
| rank_contiguous | ok |
| score_order | ok |
| tie_order | ok（本轮 Top-5 无同分；规则由 `(-score, item_id)` 排序键保证） |
| fields_present | ok |
| empty_query | ok |
| no_match | ok |

有匹配的会话：`11208`，query `[3538762, 3181091, 4061567, 3525929]`，
可用匹配 7 个，Top-5：

| rank | item_id | score | category |
| ---: | ---: | ---: | --- |
| 1 | 80675 | 17.864051 | science / Experimental Invention |
| 2 | 135989 | 12.738221 | High tech digital / Technology digital products |
| 3 | 209624 | 8.432907 | High tech digital / Technology digital products |
| 4 | 350728 | 7.958622 | High tech digital / Technology digital products / Technology surrounding / Charging transfer |
| 5 | 135058 | 7.612740 | High tech digital / High tech digital teaching |

该会话点击物品为 80675、350728，均进入 Top-5；80675 排第 1。
空查询与 token `1000000000000` 均返回 `[]`。

完整结果：[smoke.json](smoke.json)（sha256 `56f1cd3b…`）。

## 7. 遇到的问题与处理

1. **第一次构建在写完产物后退出**：`_counts()` 调用漏传了会话选择统计，
   最终打印 `manifest["counts"]["sessions_selected"]` 时 `KeyError`。
   产物本身已写出，但 manifest 缺少 `sessions_selected` 等字段。
   修复：把 `selection_stats` 传入 `_counts()` 并合并；随后 `--force` 重建。
   采样规则、索引内容未变，index sha256 仍为 `3ae2b858…`。
2. **没有官方词表**：无法直接证明共用词表，改用论文陈述 + 抽样重合信号；
   结论和残余风险写进 `vocab_evidence` 与 `docs/search.md`。
3. **前缀扫描导致覆盖率只有 55.8%**：没有为了凑覆盖率去扫全量；
   未关联物品保留 `joined=false`，并在摘要中标记两个输入均截断。
4. **CLI 空查询参数**：`--tokens ""` 在 PowerShell 下可能被吞成 0 个参数，
   原来的 `nargs="+"` 会报错；改为 `nargs="*"`，省略 `--tokens` 或传空字符串
   都返回 `[]`。该修改只影响 CLI 参数解析，不影响索引与已完成的冒烟结果。
5. **Windows 控制台中文帮助乱码**：只是终端编码显示问题，不影响 UTF-8 文件与
   ASCII 输出；CLI 表格使用英文类目，JSON 文件为 UTF-8。

## 8. 遗留项

- [ ] Search Agent 调用 `search(query_tokens, top_k)` 的工具封装与预算控制；
- [ ] 向量检索 / 多路召回 / 排序 / 重排（本轮明确不做）；
- [ ] 全量或更大样本的检索评测（Recall/NDCG/人工相关性）；
- [ ] 提高关联覆盖率：扩大 `item_scan_rows` 或建立按 item_id 的随机访问索引；
- [ ] 若官方后续提供词表，补一次 `token → 文本` 的可读性核验。
