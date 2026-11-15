# Phase 2：搜索侧 FAISS 向量、混合检索、精排与 Search Agent 四工具

> **日期**：2026-11-09
> **数据**：Search 1 的 KuaiSAR 小样本（30 会话 / 120 物品 / 823 token）
> **范围**：链路验收。训练 0、微调 0、BM25 重建 0、全量索引 0、原始 CSV 扫描 0。
> 验收脚本本身不调用 LLM；Agent 决策有两类证据：**脚本替身**（不联网）与**一次真实
> `deepseek-chat` 工作流**（3 次请求）。检索/精排/索引/落盘始终执行真实代码。

## 1. 结论

- 旧 BM25 入口（库 + CLI）保持可用，结果与改动前一致。
- 搜索侧 FAISS 向量索引可构建、保存、**重载**、查询、增量更新；`backend_actual`
  为 `faiss.IndexFlatIP`（faiss 1.15.1），查询时 FAISS 调用计数递增。
- 混合检索按 `item_id` 去重，每条结果保留 `bm25` / `vector` 两路来源、名次和实际分数，
  融合顺序与上游 `RRF.fuse` 完全一致。
- 精排接在融合候选池后面，重新打分并返回最终 Top-5，分数可按公式复算。
- Search Agent 注册四个工具，`search_hybrid → search_rerank → 最终回复` 链路
  `verified`；空结果、空查询、首轮精排等分支按预期处理。
- 验收脚本 **9/9 通过**；Agent 自检 **17/17**；本地回归 **45/45**（旧证据哈希不变）。
- 另外跑了 **1 次真实 `deepseek-chat` 工作流**（3 次请求）：模型自己选择
  `search_hybrid → search_rerank`，状态 `verified`，17/17 运行时检查通过，轨迹无候选 ID。

## 2. 复用的 SkimSearchAgent 组件（固定提交 `4794575`）

上游源码位于 `third_party/skim-search-agent`，**未修改**。适配层：
`src/recsys/search/upstream.py`。

| 上游组件 | 复用方式 | 项目侧适配 |
| --- | --- | --- |
| `agent_search.retrievers.dense.vector_index` | `VectorIndex` 契约、`save_index` / `load_index` / `index_exists`、原子写入 | 新增 `FaissFlatIndex`（`faiss.IndexFlatIP`）并登记进上游 backend 表；整数 `item_id` 映射；`add_items` |
| `agent_search.retrievers.fusion.rrf.RRF` | 融合顺序直接调用 `RRF(k=60).fuse(...)` | 收集两路候选、按 item_id 去重、补来源/名次/分数 |
| `agent_search.retrievers.rerankers.Reranker` / `register_reranker` | 精排契约与注册 | `TokenOverlapReranker`（`kuasar_token_overlap`） |
| `agent_search.retrievers.reranked.RerankedEngine` | `base retriever + reranker` 组合 | base 是混合检索；候选池来自融合结果 |

未复用（原因）：上游 `HybridEngine` / `RerankedEngine` 只返回 doc id，不保留各路来源
与分数，无法满足「保留来源、名次和实际分数」；上游 dense 编码器需要
sentence-transformers + 预训练权重 + 真实文本，当前三者都不具备。

## 3. 编码与索引

| 项 | 值 |
| --- | --- |
| 编码器 | `TokenHashEncoder`（无训练；blake2b 稳定哈希；查询/文档同一实例与配置） |
| 维度 / seed | 4096 / 20261109 |
| 文档侧 | caption 整数 token + 类目词通道（权重 0.25） |
| 查询侧 | 整数 token；过滤索引词表外 OOV token |
| 归一化 | L2；内积 = 余弦 |
| 索引 | 120 物品 / 4096 维 / 1,966,125 B |
| payload sha256 | `5d6983b148fc8b334c26d76ee67583f44a61aeb3601e462750608a418f7a97e3` |
| 后端 | `faiss.IndexFlatIP`，faiss 1.15.1，显式指定（小样本不退 NumPy） |
| 索引目录 | `artifacts/search/kuasar_vector/vector/`（独立于推荐索引） |

编码 meta 里记录 `encoder` 描述与配置哈希；重载时校验，换编码器必须重建。
整数 token 转成 `t:<id>` 只是哈希键，`text_semantic=false`，**不当作中文语义**。

## 4. 验收结果（`python -m recsys.search.acceptance`）

样本查询：会话 11208，`[3538762, 3181091, 4061567, 3525929]`。
点击物品（仅核对）：`350728`、`80675`。

| # | 检查 | 结果 |
| ---: | --- | --- |
| 1 | 旧 BM25 库入口与 CLI 一致 | pass（Top-1=80675，score 17.8641） |
| 2 | FAISS 后端 / 调用计数 / 重载一致 | pass（`faiss.IndexFlatIP`；调用 +1；重载 Top-5 一致） |
| 3 | 索引更新（119 → 120）保存/重载 | pass |
| 4 | 混合去重 / 来源 / 融合分 / 上游 RRF 顺序 | pass（7 条，每条含两路来源） |
| 5 | 精排最终 Top-5 + 分数可复算 | pass（1.000000 / 0.735169 / 0.495818 / 0.436689 / 0.379380） |
| 6 | Agent 脚本替身链路 | pass（`search_hybrid → search_rerank`，verified） |
| 7 | 空结果 / 空查询 / 首轮精排拒绝 | pass |
| 8 | 文本语义编码缺口 | pass（明确报错，不回退） |
| 9 | 只读约束（BM25 / 样本 sha256） | pass |

混合检索（Top-3，节选）：

| rank | item_id | RRF 分 | 来源 |
| ---: | ---: | ---: | --- |
| 1 | 80675 | 0.032787 | bm25#1 (17.864051) / vector#1 (0.619990) |
| 2 | 135989 | 0.032258 | bm25#2 (12.738221) / vector#2 (0.507980) |
| 3 | 209624 | 0.031746 | bm25#3 (8.432907) / vector#3 (0.382767) |

精排（Top-5）：`80675` 1.000000、`135989` 0.735169、`209624` 0.495818、
`350728` 0.436689、`135058` 0.379380。点击物品 80675、350728 都在 Top-5，
但点击标签没有参与打分。

Agent 脚本替身轨迹：3 次模型请求（首查 / 反馈 / 最终回复）、1 次检索 + 1 次精排；
`tool_sequence=[search_hybrid, search_rerank]`；`trace.json` 无候选 `item_id`、无模型
正文，只保留精排最终分数与计数。

真实 LLM 工作流（只跑 1 次，`driver=llm`，`deepseek-chat`）：

```text
run: experiments/2026-11-09-kuasar-search-phase2/agent_llm/run-20261109T134757Z-32924
tool_sequence: search_hybrid → search_rerank
request/search/rerank: 3 / 1 / 1
status: verified；17/17 运行时检查通过；耗时 2.83 s
精排分数: 1.0 / 0.735169 / 0.495818 / 0.436689 / 0.37938
trace: 无 item_id、无模型正文
```

模型在首轮从三个检索工具中选择了 `search_hybrid`，看到 5 条融合结果后调用
`search_rerank`，最后给出结束回复。这条证据只说明「工具可被真实模型选择、链路可跑通」，
不是检索质量评测。

## 5. 改动文件

新增：

- `src/recsys/search/upstream.py`（上游导入与再导出）
- `src/recsys/search/encoders.py`（可插拔编码器）
- `src/recsys/search/vector_index.py`（搜索侧 FAISS 索引）
- `src/recsys/search/hybrid.py`（RRF 混合 + 来源追溯）
- `src/recsys/search/rerank.py`（上游 Reranker / RerankedEngine 精排）
- `src/recsys/search/service.py`（四条入口门面）
- `src/recsys/search/query.py`（规范化 / 受约束改写）
- `src/recsys/search/build_vector_index.py`（索引构建 CLI）
- `src/recsys/search/acceptance.py`（验收脚本）
- `configs/search/kuasar_search_phase2.toml`

修改：

- `src/recsys/search/tool.py`（四个工具 schema + `SearchToolbox`）
- `src/recsys/search/__main__.py`（`--mode` 四模式，旧 BM25 用法不变）
- `src/recsys/search/__init__.py`（导出新接口）
- `src/recsys/experiments/search_agent.py`（四工具、精排节点、脚本替身驱动）
- `configs/experiments/search_agent.toml`（`[search_backend]`、精排预算）

未改：`third_party/skim-search-agent/**`、
`src/recsys/search/prepare.py`、`src/recsys/search/smoke.py`、
`configs/search/kuasar_search.toml`、推荐侧索引与配置。
`src/recsys/search/bm25.py` 只新增了一个只读 `idf` 属性供精排覆盖率加权使用，
`search` / 序列化 / 排序规则与索引文件均未改变（验收与旧冒烟均通过）。

## 6. 调用

```powershell
$env:PYTHONPATH = "src"

# 构建搜索向量索引（只读 BM25 小样本索引）
python -m recsys.search.build_vector_index --force
python -m recsys.search.build_vector_index --check

# 四种检索
python -m recsys.search --mode bm25   --tokens "3538762,3181091,4061567,3525929" --top-k 5
python -m recsys.search --mode vector --tokens "3538762,3181091,4061567,3525929" --top-k 5
python -m recsys.search --mode hybrid --tokens "3538762,3181091,4061567,3525929" --top-k 5
python -m recsys.search --mode rerank --tokens "3538762,3181091,4061567,3525929" --top-k 5

# 验收
python -m recsys.search.acceptance

# Agent：自检 / 回归 / 脚本替身链路
python -m recsys.experiments.search_agent --self-check
python -m recsys.experiments.search_agent --regression --tokens 2 `
  --output-root artifacts/search_agent_regression_phase2
python -m recsys.experiments.search_agent --scripted-plan hybrid_rerank `
  --output-root experiments/2026-11-09-kuasar-search-phase2/agent_scripted
```

## 7. 已知限制

1. **没有真实文本**：KuaiSAR 只有匿名整数 token；向量编码是哈希后的 token 重合，
   不是语义。当前样本上 BM25 与向量的候选集高度重合（同一 token 空间），
   混合检索的价值主要在 RRF 排序融合与来源追溯。
2. **没有预训练精排器**：`sentence-transformers` 未安装、无本地权重；精排是
   `0.65·coverage + 0.35·prior` 的规则版本，不读点击标签。
3. **未做检索质量评测**：没有 Recall/MRR/NDCG 或人工相关性标注；样本仍是前缀截断的
   120 物品，不代表全量目录。
4. **真实 LLM 只跑 1 次**：模型这次选择了 `search_hybrid → search_rerank`；「二次检索」
   与「空结果」分支由脚本替身和自检覆盖，不是真实模型的多次行为统计。
5. **OOV 过滤依赖建索引词表**：新增物品后需要 `add_items` 或重建向量索引。
