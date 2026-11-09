# Search 2 · Search Agent MVP

> **状态**：2026-11-09 完成一次真实工作流验收。只做 `search_bm25` 工具调用与一次反馈决策；
> 中文文本输入、向量检索、排序/重排和全量评测留到后续。
>
> **结论**：真实 `deepseek-chat` 工作流只启动一次，模型发出原生 `search_bm25` 调用，
> 收到 **5 条**结果后**主动停止**；共 **2 次模型请求 / 1 次 BM25 查询**，
> 状态 **verified**，11/11 验收检查通过。原 Search 1 索引前后 sha256 一致。

## 1. 本轮交付

| 交付 | 位置 |
| --- | --- |
| `search_bm25` 工具 | [tool.py](../src/recsys/search/tool.py) |
| Search Agent / LangGraph 工作流 | [search_agent.py](../src/recsys/experiments/search_agent.py) |
| 配置（索引身份、预算、输出目录） | [search_agent.toml](../configs/experiments/search_agent.toml) |
| 本地 self-check（不联网） | [self_check.json](../experiments/2026-11-09-kuasar-search-agent/self_check.json) |
| 真实工作流轨迹 | `experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/trace.json` |
| 验收指标 | `experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/metrics.json` |
| 开发记录 / 报告 | [report.md](../experiments/2026-11-09-kuasar-search-agent/report.md) |

复用 Search 1 的 `artifacts/search/kuasar_bm25/index.json` 与 `sample.json`；
**没有重新采样、没有重建索引、没有扫原始 CSV**。
索引身份在配置里固定为 sha256 `3ae2b858…`、120 docs / 823 vocab；
运行前后重新计算，见 §7。

## 2. 工具 `search_bm25`

工具 schema 由 `SearchBM25Tool` 注册，直接调用现有 `BM25Index.search`：

```python
from recsys.search import BM25Index, SearchBM25Tool

index = BM25Index.load("artifacts/search/kuasar_bm25/index.json")
tool = SearchBM25Tool(index, max_results=5)
result = tool.search([3538762, 3181091, 4061567, 3525929], top_k=5)
```

返回字段：

| 字段 | 说明 |
| --- | --- |
| `tool` | 固定 `search_bm25` |
| `query_tokens` | 实际查询 token |
| `requested_top_k` / `top_k` | 模型请求值 / 实际生效值；`top_k ≤ 5` |
| `count` | 实际返回数量 |
| `results` | 最多 5 条：`rank`、`item_id`、`score`、`category` |

工具只返回 item_id / BM25 分数 / 可读类目；不返回 caption token、原始文本或逐条候选之外的字段。
工具本身不做训练、不加载 CSV、不写盘。

## 3. 查询调整规则与预算

| 规则 | 实现 |
| --- | --- |
| 第一次查询 | 必须与 `original_query_tokens` **完全一致**（顺序也不能变） |
| 第二次查询 | 非空；只能使用原查询里的 token（multiset 子集）；必须至少去掉一个 token；不能与之前查询相同 |
| 样本外查询 | 只传 `--tokens` 时直接查现有索引；匹配不到样本会话也继续，`session_id=null`；匹配到时补充元信息 |
| 会话绑定 | `--session-id` 仍从样本加载；同时传 `--tokens` 时仍校验两者一致 |
| 工具参数 | `query_tokens` 非空、正整数、最多 64 个；`top_k` 1–5；未知字段拒绝 |
| BM25 查询预算 | 最多 **2** 次；执行前检查 |
| 模型请求预算 | 最多 **3** 次；每次调用前计数 |
| 查询预算用完后 | 只允许最终文本回复，工具不再下发；若模型仍请求工具，执行器拒绝 |
| 自动重试 | **关闭**；失败保留轨迹、状态和原因 |

程序在**执行工具前**完成参数、重复、子集、预算检查；不合法调用不会执行 BM25。
第二次查询合法时，`SearchBM25Tool` 仍只调用一次 `BM25Index.search`，不会重新加载索引。

## 4. LangGraph 工作流

```text
request_first → execute_first → request_feedback
                                   ├─ 停止 → finish
                                   └─ 第二次合法查询 → execute_second → request_final → finish
```

- `request_first`：模型必须发出原生 `search_bm25` tool_call，参数等于原查询；
- `execute_first`：校验通过后执行第一次检索，把 tool 消息按同一 `tool_call_id` 追加到对话；
- `request_feedback`：把第一次结果送回模型；模型可停止，或发起第二次合法查询；
- `execute_second`：执行第二次检索；预算用尽；
- `request_final`：`tools=None`，只允许最终回复；若再次请求工具则拒绝；
- `finish`：计算 checks、写 trace/metrics/run_meta。

状态：`verified` / `partial` / `incomplete` / `blocked` / `failed`。
失败不会走成功分支；最终回复截断或为空时保留已完成的检索轨迹并记 `partial`。

## 5. CLI

```powershell
$env:PYTHONPATH = "src"

# 不联网的本地自检（12 项，合成响应）
python -m recsys.experiments.search_agent --self-check

# 本地替身回归：候选不落盘、样本外查询、旧证据不变（不调用真实 LLM）
python -m recsys.experiments.search_agent `
  --regression --tokens 2 `
  --output-root artifacts/search_agent_regression_tokens

# 真实工作流（本轮只应启动一次；重复执行会创建新的 run 目录）
python -m recsys.experiments.search_agent `
  --tokens "3538762,3181091,4061567,3525929" `
  --output-root experiments/2026-11-09-kuasar-search-agent

# 也可以从 Search 1 样本会话加载
python -m recsys.experiments.search_agent --session-id 11208
```

只传 `--tokens` 时不再要求命中样本会话：能匹配就补 `session_id`，匹配不到以
`cli_tokens_out_of_sample` 继续查询。`--session-id` 仍从样本加载；两者同时传入时校验一致。
LLM 配置从仓库根目录 `.env` / 环境变量读取，密钥不写入轨迹。

## 6. 本地 self-check（12/12）

| 检查 | 结果 |
| --- | --- |
| 一次检索 + 主动停止 → verified | pass |
| 两次检索 + 最终回复 → verified | pass |
| 首查不是原查询 → 不执行工具 | pass |
| 第二查使用非原 token → 拒绝 | pass |
| 第二查与首查相同 → 拒绝 | pass |
| 第二查为空 → 拒绝 | pass |
| 预算用尽后再次请求工具 → 拒绝 | pass |
| 停止说明 `finish_reason=length` → partial | pass |
| tool_call_id 与 tool 消息一致 | pass |
| 请求 3/3、检索 2/2 未超限 | pass |
| 轨迹不保存候选明细 | pass |
| 模型最终回复正文不落盘 | pass |

完整结果：[regression.json](../experiments/2026-11-09-kuasar-search-agent-regression/regression.json)
的 `self_check` 字段；旧的 [self_check.json](../experiments/2026-11-09-kuasar-search-agent/self_check.json)
按“旧证据保留”未改写。

## 7. 真实工作流结果

**只启动一次**，使用已有 KuaiSAR 会话 11208：

```text
query tokens: [3538762, 3181091, 4061567, 3525929]
session_id:   11208
model:        deepseek-chat
tool:         search_bm25
tool_call_id: call_00_lnA1GQAShje8UOqfaZph1633
first return: 5 条（top_k=5）
```

模型收到第一次结果后**主动停止**，理由是：结果最高分 17.86、其余集中在高科技数码类目，
与查询意图相关性较好；剩余预算只有 1 次，去掉任一 token 都可能削弱语义覆盖，收益有限。
本轮没有发起第二次查询，因此实时路径只覆盖“查一次后停止”；第二次查询的合法/非法路径
由 self-check 覆盖。

| 项 | 值 |
| --- | ---: |
| 状态 | **verified** |
| 模型请求 | **2 / 3** |
| BM25 查询 | **1 / 2** |
| 返回数量 | 5 |
| 停止决策 | `model_stop` |
| 耗时 | 2.876 s |
| 验收检查 | **11 / 11** |
| 索引 sha256 前 / 后 | `3ae2b858…` / `3ae2b858…` |

轨迹里的工具调用 ID、返回数量、`tool_result_in_context.sent=true`、第二次请求的 tool 消息
和停止理由均已核对。完整文件：

- `experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/trace.json`
  （sha256 `05fd0438…`）
- `experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/metrics.json`
  （sha256 `f96a39f3…`）
- `experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/run_meta.json`
  （sha256 `48a07412…`）

> 该真实工件产生于 2026-11-09 修复前，按“旧证据保留”未改写。它不含候选 ID，
> 但仍保留了模型停止理由文本；修复后的代码不再落盘任何模型自由文本，见 §10。

## 8. 落盘边界（修复后）

- 候选/工具结果仍通过 `role=tool` 的消息回传给模型，供其决定停止或再查；
- 轨迹 `response` 只保留 `content_chars`、`content_present`、`finish_reason`、`tool_calls`，
  不写 `content_preview` / `content` / `final_reply`；
- 最终摘要只写 `stop_code`、`stop_type`、程序生成的 `stop_summary`、
  `final_reply_status`、`final_reply_chars`、返回数量和预算；
- `trace.json` / `metrics.json` / `run_meta.json` 标记
  `candidates_persisted=false`、`model_free_text_persisted=false`；
- 模型正文只在内存中用于 `finish_reason=stop` + 非空判断，随后丢弃；
  超时、截断、空回复仍走原有 `partial` 路径。

旧真实工件（sha256 见 §7）保持原样；新的回归工件写在 Git 忽略的 `artifacts/`。

## 9. 已知限制与下一步

- 实时只走了“查一次后停止”分支；两次查询的合法/非法分支由 self-check 的合成响应覆盖。
- 不是检索质量评测：没有 Recall/NDCG、人工相关性或全量数据。
- 未做中文文本输入、query→token 编码、向量检索、排序/重排。
- 本轮只读 Search 1 索引与样本；未重建索引、未扫原始 CSV。
- 后续可把 `SearchBM25Tool` 接到更大的索引，或扩展为多轮反馈/多路召回。

## 10. 2026-11-09 修复回归（本地替身）

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.search_agent `
  --regression --tokens 2 `
  --output-root artifacts/search_agent_regression_tokens
```

结果：**45/45** 检查通过，self-check **12/12**；真实 LLM 请求、索引重建、真实冒烟重跑均为 0。

| 用例 | session | 返回数量 | 结果 |
| --- | --- | ---: | --- |
| 样本会话 11208 | `11208` | 5 | verified |
| 样本外 token `[2]` | `null` | 1 | verified |
| 样本外无匹配 `[1000000000000]` | `null` | 0 | verified |
| `--session-id` + 不一致 token | — | — | blocked |

回归检查了实际序列化的 `trace.json` / `metrics.json` / `run_meta.json`：
候选 ID、模型原始回复均未出现；调用 ID、返回数量、停止类型与预算仍可核对。
旧真实工件与索引/样本哈希前后一致。

证据：[regression report](../experiments/2026-11-09-kuasar-search-agent-regression/report.md)、
[regression.json](../experiments/2026-11-09-kuasar-search-agent-regression/regression.json)
（sha256 `b2a24887…`）。
