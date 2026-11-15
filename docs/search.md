# Search 1：KuaiSAR 词表核验、小样本与 BM25

> **状态**：2026-11-09。§1–§9 是 Search 1（数据准备 → BM25 建索引 → 查询接口/CLI）；
> §10 是 Phase 2 补齐的**搜索侧 FAISS 向量召回、RRF 混合检索与精排**（同一份 120 物品
> 小样本）。Search Agent 见 [search-agent.md](search-agent.md)。
>
> **一句话结论**：仓库/数据包没有单独的词表文件，但论文说明 query 与 video title
> 走同一套分词与整数哈希，且抽样观测显示 keyword 与 caption 的 token ID 范围一致、
> 点击/曝光物品的 token 重合显著高于随机基线。因此本轮用 **KuaiSAR 原始 token ID**
> 建索引，没有改用本地夹具；向量编码也只是同一套整数 token 的哈希表示，
> **不代表中文语义**。

## 1. 本轮交付

| 交付 | 位置 | 状态 |
| --- | --- | --- |
| 词表核验与结论 | 本文 §2；`build_summary.json` 的 `vocab_evidence` | 完成 |
| 采样与关联 | 本文 §3；[prepare.py](../src/recsys/search/prepare.py) | 完成 |
| BM25 索引 | 本文 §4；[bm25.py](../src/recsys/search/bm25.py) | 完成 |
| 查询接口 / CLI | 本文 §5；[__main__.py](../src/recsys/search/__main__.py) | 完成 |
| 单次冒烟验收 | 本文 §6；[smoke.py](../src/recsys/search/smoke.py) | 通过 |
| 配置 | [kuasar_search.toml](../configs/search/kuasar_search.toml) | 完成 |
| 开发记录与验收摘要 | [report.md](../experiments/2026-11-09-kuasar-search-bm25/report.md) | 完成 |

本轮**没有**训练模型、跑全量检索评测、做向量检索/排序/重排，也没有调用 Search Agent。
点击只用于样本检查与覆盖率；BM25 排序只使用 caption token。

## 2. 词表核验

### 2.1 先查已有文档与数据包元信息

- `docs/kuasar.md` 已记录：`src_inter.csv` 的 `keyword` 是分词 ID 列表，
  `item_features.csv` 的 `caption` 也是分词 ID 列表；两边都不是原始文本。
- `KuaiSAR_v2.zip` 解压后只有 5 个 CSV：
  `rec_inter.csv`、`src_inter.csv`、`item_features.csv`、`user_features.csv`、
  `social_network.csv`。**没有** `vocab.json` / `dictionary.txt` / tokenizer 配置等词表元信息。
- KuaiSAR 论文（[arXiv:2306.07705v4](https://arxiv.org/html/2306.07705v4)）
  §3.2 *Data Construction* 明确写到：

  > “Textual information, such as queries and video titles, underwent word segmentation
  > and sensitive word removal. Furthermore, terms in texts were randomly hashed into integers.”

  这句话说明 query 与 video title 走的是同一套分词/哈希流程，落到整数 token；
  数据包没有再给出两套独立映射的依据。

### 2.2 抽样观测

在 `src_inter.csv` 前 200,000 行、`item_features.csv` 前 500,000 行的扫描范围内：

| 指标 | 数值 |
| --- | ---: |
| keyword 唯一 token | 15,339 |
| caption 唯一 token（500k 物品） | 666,385 |
| 两边交集 | **13,747** |
| Jaccard | 0.02058 |
| keyword token 范围 | 22 – 4,233,695 |
| caption token 范围 | 2 – 4,233,996 |
| 按均匀独立词表估算的期望交集 | 2,414.36 |
| 实际交集 / 期望 | **5.69×** |
| keyword 中未在 caption 样本出现的 token | 1,592 |

更强的信号来自「查询 → 该会话曝光/点击物品」的重合：

| 指标 | 数值 |
| --- | ---: |
| 有点击且已关联 caption 的 (query, item) 对 | 43 |
| 其中至少 1 个 token 重合 | 22（**51.16%**） |
| 点击物品平均重合 token 数 | 0.8837 |
| 已关联曝光 (query, item) 对 | 120 |
| 其中至少 1 个 token 重合 | 74（**61.67%**） |
| 曝光物品平均重合 token 数 | 1.0250 |
| 随机 caption 基线（5,000 个物品，seed 42） | 0.012053 |
| 点击物品重合 / 随机基线 | **73.32×** |
| 按会话比较「点击物品 > 随机物品」 | 19 胜 / 2 平 / 9 负 |

如果 keyword 与 caption 是两套互不相干的整数空间，query token 在 caption 里
出现的机会应接近随机基线；实际观测到 5.69× 的全集富集、点击物品 51% 的 pair 命中、
以及相对随机 caption 73× 的平均重合，足以确认两边使用同一套 token ID 语义。

### 2.3 结论、决策与残余风险

- **结论**：`keyword` 与 `caption` 共用同一套 KuaiSAR token ID 空间。
- **决策**：BM25 直接使用原始 token ID；**不使用**本地夹具，也不需要重新编码/映射。
- **残余风险**：数据包没有官方的 `token → 文本` 词表，token 是随机哈希后的整数，
  无法反解或人工校验单条语义；论文只说明「queries and video titles」共用哈希，
  没有给出显式 vocab 文件。若上游更换 tokenizer/哈希，必须用同一套脚本重新核验。
- **未验证范围**：本轮只在这个 30 会话 / 120 物品小样本上验证查询链路；
  KuaiSAR 全量检索质量、召回率和端到端搜索效果**未验证**。

对应机器可读证据：`artifacts/search/kuasar_bm25/manifest.json` 的
`vocab_evidence`；验收副本见 `experiments/2026-11-09-kuasar-search-bm25/build_summary.json`。

## 3. 采样与 item 关联

### 3.1 先定扫描上限

| 输入 | 文件大小 | 扫描上限 | 实际扫描 | 是否截断 |
| --- | ---: | ---: | ---: | --- |
| `src_inter.csv` | 553,388,618 B | 200,000 行 | 200,000 行 | **是**（文件更大） |
| `item_features.csv` | 2,008,531,649 B | 500,000 行 | 500,000 行 | **是**（文件更大） |

实现细节：

- `src_inter` 按 `(user_id, search_session_id)` 聚合；扫描上限正好落在会话中间时，
  丢弃最后一个不完整会话。本轮 `partial_sessions_dropped = 0`，
  `keyword_mismatch_sessions = 0`（同一会话内 keyword 一致）。
- 先扫会话，收集出现过的 163,902 个目标 `item_id`；再扫 `item_features` 前缀，
  只把目标物品的 caption/类目读进内存，同时统计全部扫描行的 caption token 集合。
- 现有 `data/kuasar/processed/features/` 只保存 `caption_tokens.npy`（token 数），
  **没有 caption token 内容**；因此本轮无法直接复用 processed 特征建 BM25，
  必须读原始 `item_features.csv` 的 `caption` 列。没有重建/改写 processed 产物。
- `item_features` 前缀里命中 20,407 个目标物品；其余目标物品不在前缀内，
  作为**未关联**记录，不进入索引。
- 两个 `*_truncated = true` 已写进 `build_summary.json`。
  只读前缀会偏向较小的 `item_id`，这是本地小样本的已知偏差，不是随机抽样。

### 3.2 会话选择规则

固定规则（写在配置里，不按命中结果反调）：

1. keyword 非空、会话内 keyword 一致；
2. 至少 2 个不同曝光物品；
3. 已关联 caption 的物品占比 ≥ 50%；
4. 优先「有可关联点击物品」的会话，其次曝光物品更多、覆盖率更高、文件顺序更早；
5. 最多 30 个会话；物品池取这些会话已关联物品的并集，按 item_id 排序，
   最多 1,000 个（本轮未触发截断）。

### 3.3 实际样本

| 项 | 数值 |
| --- | ---: |
| 扫描到的完整搜索会话 | 17,036 |
| 满足规则的候选会话 | 190 |
| **保留会话** | **30**（上限 30） |
| 有可关联点击物品的会话 | 30 / 30 |
| 曝光行 | 215 |
| 已关联 caption 的曝光行 | 120（**55.814%**） |
| 唯一曝光物品 | 215 |
| 唯一已关联物品 | 120（**55.814%**） |
| 点击行 | 60 |
| 已关联的点击行 | 43（71.67%） |
| 唯一点击物品 / 已关联 | 60 / 43 |
| 会话覆盖率均值 / 最小 / 最大 | 56.971% / 50% / 80% |
| 100% 覆盖的会话 | 0 |
| **索引物品数** | **120**（≤ 1,000，未截断） |

会话覆盖率不是 100%：`item_features` 只扫了前缀，部分曝光物品没有 caption/类目。
这些物品保留在 `sample.json` 的 `exposures` 里并标记 `joined=false`，
用于追溯缺口；索引与 BM25 只包含 `joined=true` 的物品。

`sample.json` 只保存**一份**逐条数据：30 个会话的 query / 曝光 / 点击 / 关联标记。
索引物品的 caption 与类目只保存在 `index.json`，没有另外导出物品 CSV/JSON。

## 4. BM25 索引

### 4.1 公式与参数

```text
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q,d) = Σ_t qtf(t) · idf(t) · tf(t,d) · (k1 + 1)
             / (tf(t,d) + k1 · (1 - b + b · |d| / avgdl))
```

- `k1 = 1.5`，`b = 0.75`；
- 查询 token 的重复次数 `qtf` 计入分数，不做去重；
- 文档长度 `|d|` 是 caption token 数，`avgdl` 是样本平均长度；
- 排序键：`(-score, item_id)`，同分时 item_id 升序；
- 空查询 / 无匹配：返回 `[]`；`top_k <= 0` 也返回 `[]`。

### 4.2 可复用接口

```python
from recsys.search import BM25Index

index = BM25Index.load("artifacts/search/kuasar_bm25/index.json")
hits = index.search([3538762, 3181091, 4061567, 3525929], top_k=5)
for hit in hits:
    print(hit.rank, hit.item_id, hit.score, hit.category)
```

`SearchHit` 字段：`rank`、`item_id`、`score`、`category`（英文可读类目路径）、
`category_zh`、`item_type`。索引 JSON 同时保存 `items`、`doc_len`、`postings`、`idf`，
加载时会从 `items` 重算并核对 `num_docs` / `vocab_size` / `avgdl`。

### 4.3 索引规模与哈希

| 项 | 值 |
| --- | --- |
| 物品数 | 120 |
| 词表大小 | 823 |
| 平均文档长度 | 16.266667 |
| 总 token 数 | 1,952 |
| 文档长度 min / max | 1 / 79 |
| 索引文件 | `artifacts/search/kuasar_bm25/index.json` |
| 索引大小 | 76,077 B |
| 索引 sha256 | `3ae2b858d456eab4c4c21ed356812cd44260b56dd596639588a53fb1a6a52690` |
| 样本文件 | `artifacts/search/kuasar_bm25/sample.json` |
| 样本大小 | 54,651 B |
| 样本 sha256 | `12a46f3c3a614a715e52f4a790ccac93eb0fc7519ab5ff4a0826ceace58a7daf` |
| build summary / manifest sha256 | `eba233325c6554b93384642c844b4a911e86d27d62cae7bc0a30772df2a38238` |

`artifacts/` 已被 `.gitignore` 忽略；验收摘要与报告在
`experiments/2026-11-09-kuasar-search-bm25/`，可单独提交。

## 5. 查询 CLI

```powershell
$env:PYTHONPATH = "src"

# 有匹配：返回 rank / item_id / BM25 score / 可读类目
python -m recsys.search --tokens "3538762,3181091,4061567,3525929" --top-k 5

# 机器可读
python -m recsys.search --tokens 3538762 3181091 4061567 3525929 --top-k 5 --json

# 空查询 → []（省略 --tokens 或传空字符串均可）
python -m recsys.search --top-k 5
```

默认索引是 `artifacts/search/kuasar_bm25/index.json`，可用 `--index` 指向其他索引。
CLI 只读索引，不重新扫描原始 CSV。

## 6. 冒烟验收

**只运行一次**的验收命令：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.search.smoke --top-k 5 `
  --output experiments/2026-11-09-kuasar-search-bm25/smoke.json
```

同一次进程内检查三种情况，结果 **9/9 通过**：

| 检查 | 结果 |
| --- | --- |
| 索引可加载 | ok |
| 有匹配查询 | ok（会话 11208，4 个 token，7 个匹配） |
| Top-K 数量 | ok（5 / 5） |
| rank 连续 | ok（1..5） |
| score 降序 | ok |
| 同分 item_id 升序 | ok（本轮 Top-5 无同分；规则由 `(-score, item_id)` 排序键保证） |
| 返回字段齐全 | ok |
| 空查询 → `[]` | ok |
| 无匹配 token → `[]` | ok |

有匹配时 Top-5：

| rank | item_id | score | category |
| ---: | ---: | ---: | --- |
| 1 | 80675 | 17.864051 | science / Experimental Invention |
| 2 | 135989 | 12.738221 | High tech digital / Technology digital products |
| 3 | 209624 | 8.432907 | High tech digital / Technology digital products |
| 4 | 350728 | 7.958622 | High tech digital / Technology digital products / Technology surrounding / Charging transfer |
| 5 | 135058 | 7.612740 | High tech digital / High tech digital teaching |

该会话的点击物品是 80675 与 350728，均进入 Top-5；80675 排第 1。
这只说明 BM25 在这条小样本上能给出可解释的排序，**不是**检索质量评测。

完整验收 JSON：`experiments/2026-11-09-kuasar-search-bm25/smoke.json`
（sha256 `56f1cd3b6601db94b25ba7ef5677f11c57e3bb59c94183036e9f8a9cc3853cca`）。

## 7. 复现命令

```powershell
$env:PYTHONPATH = "src"

# 1) 采样 + 建索引（默认读 artifacts/search/kuasar_bm25/；重跑会校验哈希并复用）
python -m recsys.search.prepare `
  --summary-out experiments/2026-11-09-kuasar-search-bm25/build_summary.json

# 强制重建
python -m recsys.search.prepare --force `
  --summary-out experiments/2026-11-09-kuasar-search-bm25/build_summary.json

# 只校验产物哈希与索引可加载性
python -m recsys.search.prepare --check

# 2) 查询
python -m recsys.search --tokens "3538762,3181091,4061567,3525929" --top-k 5

# 3) 单次冒烟（本轮已执行一次，结果为通过；复跑会覆盖 smoke.json）
python -m recsys.search.smoke --top-k 5 `
  --output experiments/2026-11-09-kuasar-search-bm25/smoke.json
```

## 8. 已知限制与下一步

1. **样本是前缀截断的**：两个 CSV 都只读前若干行，`item_id` 偏小；
   120 个索引物品仅用于链路验收，不能代表 KuaiSAR 全量目录。
2. **覆盖率约 55.8%**：未关联物品保留在 `sample.json` 并标记 `joined=false`；
   若后续要提升覆盖率，可提高 `item_scan_rows` 或改为按目标 item_id 的随机访问索引。
3. **没有全量检索评测**：本轮不计算 Recall/NDCG，也没有人工相关性标注；
   `keyword`/`caption` 是哈希 token，无法直接阅读 query 文本。
4. **Search Agent 已另起一轮**：见 [search-agent.md](search-agent.md)；本轮接口只提供
   `search(query_tokens, top_k)` 与 CLI。向量检索 / 排序重排仍未做。
5. **词表结论依赖论文陈述与抽样观测**：若上游 tokenizer 变化，需要重新跑
   `prepare.py` 并检查 `vocab_evidence.decision` 是否仍为 `use_kuasar_token_ids`；
   若变为 `local_fixture_required`，脚本会拒绝建索引。

## 9. 后续：Search Agent

Search 2 复用本轮的 120 物品索引与 `BM25Index.search`，新增 `search_bm25` 原生工具调用、
查询子集校验、2 次检索 / 3 次请求预算和一次反馈决策。真实工作流与验收证据见
[search-agent.md](search-agent.md)。

## 10. Phase 2：搜索侧 FAISS 向量、混合检索与精排（2026-11-09）

> 与 §1–§9 使用同一份 120 物品 / 30 会话小样本；**没有重建 BM25、没有扫描原始 CSV、
> 没有训练或微调**。搜索向量索引独立于推荐侧 FAISS 索引。

### 10.1 复用的 SkimSearchAgent 组件（固定提交 `4794575`）

上游源码在 `third_party/skim-search-agent`，**未改动**；适配层
[`src/recsys/search/upstream.py`](../src/recsys/search/upstream.py) 只把它加入 `sys.path`
并再导出：

| 上游组件 | 用途 | 项目侧适配 |
| --- | --- | --- |
| `agent_search.retrievers.dense.vector_index` | 索引基类、`doc_ids.json` + `meta.json` 原子落盘、`load_index` / `index_exists` | 新增 `FaissFlatIndex`（`faiss.IndexFlatIP`）并登记进上游 backend 表；增加整数 `item_id` 映射与 `add_items` |
| `agent_search.retrievers.fusion.rrf.RRF` | RRF 融合（`1/(k+rank)`，只读名次） | 融合顺序直接调用上游；项目侧补来源/名次/分数 |
| `agent_search.retrievers.rerankers.Reranker` + `register_reranker` | 精排契约与注册表 | `TokenOverlapReranker`（`kuasar_token_overlap`） |
| `agent_search.retrievers.reranked.RerankedEngine` | `base retriever + reranker` 组合 | base 是混合检索适配器，先取候选池再精排 |

上游 `HybridEngine` / `RerankedEngine` 只返回 doc id，不保留各路来源与分数，
因此项目侧写薄适配；融合算法与精排流程仍由上游组件执行。

### 10.2 模块与调用

| 模块 | 作用 |
| --- | --- |
| [`encoders.py`](../src/recsys/search/encoders.py) | 可插拔 `SearchEncoder`；`TokenHashEncoder`（无训练）与 `TextSemanticEncoder`（有文本时启用） |
| [`vector_index.py`](../src/recsys/search/vector_index.py) | `FaissFlatIndex` + `SearchVectorIndex`：编码、Top-K、item_id 映射、保存/重载/更新 |
| [`hybrid.py`](../src/recsys/search/hybrid.py) | BM25 + 向量，按 item_id 去重，RRF 融合，来源追溯 |
| [`rerank.py`](../src/recsys/search/rerank.py) | 融合候选池精排（上游 `Reranker` + `RerankedEngine`） |
| [`service.py`](../src/recsys/search/service.py) | 四条检索入口的统一门面 |
| [`query.py`](../src/recsys/search/query.py) | 查询规范化与受约束改写 |
| [`build_vector_index.py`](../src/recsys/search/build_vector_index.py) | 从现有 BM25 索引构建搜索向量索引 |
| [`acceptance.py`](../src/recsys/search/acceptance.py) | 小样本验收（9 项） |

```python
from recsys.search import build_encoder
from recsys.search.service import SearchService

encoder = build_encoder({"name": "token_hash", "dim": 4096, "seed": 20261109})
service = SearchService.load(
    "artifacts/search/kuasar_bm25/index.json",
    "artifacts/search/kuasar_vector/vector",
    encoder,
)
query = [3538762, 3181091, 4061567, 3525929]
for hit in service.search("hybrid", query, top_k=5):
    print(hit.rank, hit.item_id, round(hit.score, 6), hit.category,
          [(s.source, s.rank, round(s.score, 4)) for s in hit.sources])
```

### 10.3 编码：无训练的整数 token 向量（查询/文档同配置）

- KuaiSAR 只有匿名整数 token（query 与 caption 同一哈希空间），没有原文、词表或
  预训练权重；`TokenHashEncoder` 用**稳定 blake2b** 把 `t:<id>` 映射到 4096 维带符号
  桶，次线性词频加权，L2 归一化，再建内积索引。
- **查询和文档使用同一个编码器实例与同一套配置**（dim / seed / 类别通道都相同）；
  索引 meta 记录 `encoder` 描述与配置哈希，重载时校验，换编码器必须重建索引。
- 文档侧可选类目词通道（`include_category`，权重 0.25）；查询侧没有类目。
- 查询侧过滤索引词表外的 OOV token（与 BM25 查 posting 的行为一致），避免哈希碰撞
  让「索引里没有的词」也召回文档。
- 整数 token 转成 `t:<id>` 只是哈希键，**不当作中文语义**；`describe()` 里
  `text_semantic=false` 明确标注。
- `TextSemanticEncoder`（sentence-transformers）接口已留：缺 KuaiSAR 原文、缺本地权重、
  当前环境也未安装 `sentence-transformers`，验收时给出具体报错；有文本/权重后换配置即可。

### 10.4 搜索侧 FAISS 索引

- 后端：`faiss.IndexFlatIP`（归一化向量上的精确余弦），**小样本也显式指定 FAISS**，
  不会因为规模自动退回 NumPy；`backend_actual`、`faiss_version` 与 `faiss_search_calls`
  都在 `describe()` 里可核对。
- 索引目录：`artifacts/search/kuasar_vector/vector/`
  （`doc_ids.json`、`meta.json`、`ann.faiss`）；`manifest.json` 记录来源 BM25 sha256、
  编码器配置、维度、物品数、payload sha256。索引 120 物品 / 4096 维 / 1,966,125 B。
- 物品 ID 映射：索引内部位置 ↔ 整数 `item_id`，搜索直接返回 `item_id`。
- 保存/重载：复用上游 `save_index` / `load_index`（`meta.json` 最后写入作为完成标记）。
- 更新：`SearchVectorIndex.add_items(items)` 追加物品并刷新词表；FAISS 直接 `index.add`，
  NumPy 后端重建；更新后显式 `save()`。验收里做了 119 → 120 的更新并重载查询。

```powershell
python -m recsys.search.build_vector_index          # 构建/复用
python -m recsys.search.build_vector_index --force  # 重建搜索向量索引（不动 BM25）
python -m recsys.search.build_vector_index --check  # 重载 + 查询 + 后端核对
```

### 10.5 混合检索与来源追溯

- 两路候选：BM25 Top-`pool` 与向量 Top-`pool`（默认 50）。
- 融合顺序直接调用上游 `RRF(k=60).fuse(...)`；项目侧按 `item_id` 去重，
  为每条结果保留 `sources=[{source, rank, score}]`，融合分 = Σ 1/(60+rank)。
- 分数不混用：BM25 分是关键词分，向量分是余弦，融合分是 RRF 分，三者分别记录。

```powershell
python -m recsys.search --mode bm25   --tokens "3538762,3181091" --top-k 5
python -m recsys.search --mode vector --tokens "3538762,3181091" --top-k 5
python -m recsys.search --mode hybrid --tokens "3538762,3181091" --top-k 5
python -m recsys.search --mode rerank --tokens "3538762,3181091" --top-k 5
```

### 10.6 精排

- 接在**融合候选池**后面：`RerankedEngine(base=混合检索, reranker=TokenOverlapReranker)`。
- 规则（不训练、不读点击标签）：
  `coverage(d) = Σ_{t∈Q} idf(t)·1[t∈d] / Σ_{t∈Q} idf(t)`（idf 取自现有 BM25 索引），
  `prior(d)` = 融合分在候选池内 min-max 归一化，
  `score = 0.65·coverage + 0.35·prior`；排序 `(-score, item_id)`。
- 精排对候选**重新打分**并返回最终 Top-K；结果保留 `coverage`、`prior_score`、
  `prior_rank` 与来源明细。

### 10.7 配置

[`configs/search/kuasar_search_phase2.toml`](../configs/search/kuasar_search_phase2.toml)：
编码器（`token_hash` / dim 4096 / seed 20261109 / 类别通道）、FAISS 后端、
混合（`rrf_k=60`、`pool=50`）、精排（`pool=50`、权重 0.65/0.35）。

### 10.8 验收结果（9/9，`python -m recsys.search.acceptance`）

样本查询：会话 11208 的 `[3538762, 3181091, 4061567, 3525929]`；点击物品（仅核对，
不参与打分）：`350728`、`80675`。

| 检查 | 结果 |
| --- | --- |
| 旧 BM25 库入口与 CLI 一致 | pass（Top-1=80675，score 17.8641） |
| FAISS 后端 + 查询计数 + 重载一致 | pass（`faiss.IndexFlatIP`，faiss 1.15.1，调用 +1，重载 Top-5 一致） |
| 索引更新（119 → 120）保存/重载 | pass |
| 混合去重 / 来源 / 融合分 / 上游 RRF 顺序 | pass（7 条，每条都有 bm25+vector 来源） |
| 精排最终 Top-5 + 分数可复算 | pass（1.000000 / 0.735169 / 0.495818 / 0.436689 / 0.379380） |
| Search Agent 脚本替身链路 | pass（`search_hybrid → search_rerank`，verified） |
| Search Agent 真实 LLM 链路（1 次） | pass（`deepseek-chat` 自主选择混合检索 → 精排，verified，17/17） |
| 空结果 / 空查询 / 首轮精排拒绝 | pass（无匹配 token：混合 0 条，Agent 链路 0 结果 verified；空查询与首轮精排被拒绝） |
| 文本语义编码缺口 | pass（明确报错，不是静默回退） |
| 只读约束（BM25/样本哈希） | pass |

Top-5（精排）：`80675`（1.000000）、`135989`（0.735169）、`209624`（0.495818）、
`350728`（0.436689）、`135058`（0.379380）；点击物品 80675、350728 都在 Top-5。
完整证据见 [`acceptance.json`](../experiments/2026-11-09-kuasar-search-phase2/acceptance.json)
与 [报告](../experiments/2026-11-09-kuasar-search-phase2/report.md)。

### 10.9 限制

1. **没有真实文本**：向量编码是 token 哈希，不是语义；当前样本上 BM25 与向量的候选集
   高度重合（同一 token 空间），混合检索的价值主要在 RRF 排序融合与来源追溯，
   语义扩展能力要等原文/预训练权重。
2. **未做检索质量评测**：没有 Recall/MRR/NDCG 或人工相关性标注；点击只用于核对。
3. **小样本**：120 物品 / 30 会话，覆盖率为前缀截断后的结果，不代表全量目录。
4. **精排是规则**：没有可用的预训练精排器；`kuasar_token_overlap` 只使用 token 覆盖率与
   融合先验。
5. **索引词表过滤**：OOV 过滤依赖建索引时的词表；新增物品后需要 `add_items` 或重建。
