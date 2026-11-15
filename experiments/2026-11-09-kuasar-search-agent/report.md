# 2026-11-09 · Search 2 · Search Agent MVP

> 本轮只做 `search_bm25` 工具调用与一次反馈决策。真实工作流**只启动一次**：
> 模型发出原生工具调用，收到 5 条结果后主动停止；没有训练、向量检索、排序或全量评测。

## 结论

- **通过**：`deepseek-chat` 真实工作流状态 **verified**，**2 次模型请求 / 1 次 BM25 查询**，
  11/11 验收检查通过。
- 模型第一次使用原查询（会话 11208 的 4 个 token），工具返回 **5 条**；
  模型判断相关性足够，且剩余 1 次预算下删 token 会削弱覆盖，于是**主动停止**。
- 本轮没有第二次查询；第二次查询的合法/非法路径由本地 self-check 的合成响应覆盖。
- 原 Search 1 索引前后 sha256 均为 `3ae2b858…`，未重建、未改写。

## 1. 复用与索引身份

| 项 | 值 |
| --- | --- |
| 索引 | `artifacts/search/kuasar_bm25/index.json` |
| 索引 sha256 | `3ae2b858d456eab4c4c21ed356812cd44260b56dd596639588a53fb1a6a52690` |
| 样本 | `artifacts/search/kuasar_bm25/sample.json` |
| 样本 sha256 | `12a46f3c3a614a715e52f4a790ccac93eb0fc7519ab5ff4a0826ceace58a7daf` |
| docs / vocab / avgdl | 120 / 823 / 16.266667 |
| 查询来源 | Search 1 样本会话 `11208` |

配置在 `configs/experiments/search_agent.toml` 固定索引/样本 hash 与规模；
加载时不一致直接 `blocked`。运行前后重新计算索引 hash，结果一致。
本轮没有重新采样、没有重建索引、没有读取 `src_inter.csv` / `item_features.csv`。

## 2. 工具与查询规则

`search_bm25` 直接调用 Search 1 的 `BM25Index.search`，返回最多 5 条：

```text
rank / item_id / score / category
```

工具参数检查（执行前）：

- `query_tokens` 非空、正整数、最多 64 个；
- `top_k` 1–5，超过 5 截断；
- 未知字段、非整数、空查询直接拒绝；
- 第一次必须与 `original_query_tokens` 完全一致（顺序也不能变）；
- 第二次必须非空、只能用原查询 token、至少去掉一个 token、且不能与之前查询相同；
- 最多 2 次 BM25 查询；执行前检查预算；不自动重试。

LangGraph 节点：`request_first → execute_first → request_feedback →`
（停止）`finish`，或（第二次合法查询）`execute_second → request_final → finish`。
查询预算用完后 `request_final` 不再下发工具；若模型仍请求工具，执行器拒绝。

## 3. 本地 self-check

12/12 通过，覆盖：

- 一次检索 + 主动停止 → verified；
- 两次检索 + 最终回复 → verified；
- 首查不是原查询、第二查非原 token / 重复 / 为空 → 拒绝且不执行；
- 最终回复再次请求工具、`finish_reason=length` → partial 且保留轨迹；
- tool_call_id 与 tool 消息一致；
- 请求 3/3、检索 2/2 预算；
- 轨迹不保存候选明细；最终回复中的逐条候选在落盘前截断。

证据：[self_check.json](self_check.json)。

## 4. 真实工作流

命令（只执行一次）：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.search_agent `
  --tokens "3538762,3181091,4061567,3525929" `
  --output-root experiments/2026-11-09-kuasar-search-agent
```

| 项 | 值 |
| --- | --- |
| 模型 | `deepseek-chat`（endpoint host `https://api.deepseek.com`） |
| 查询 | `[3538762, 3181091, 4061567, 3525929]`，会话 `11208` |
| 第一次 tool call | `search_bm25`，ID `call_00_lnA1GQAShje8UOqfaZph1633` |
| 第一次返回 | **5 条**（`top_k=5`） |
| 模型请求 2 | `finish_reason=stop`，选择停止 |
| 状态 | **verified** |
| 模型请求 | **2 / 3** |
| BM25 查询 | **1 / 2** |
| 耗时 | 2.876 s |
| 验收检查 | **11 / 11** |

轨迹核对：

- 第二次模型请求的输入里包含同一 `tool_call_id` 的 tool 消息；
- `tool_result_in_context.sent=true`，`count=5`；
- 模型没有发出第二次 tool_call，直接给出停止理由；
- 请求计数 2、检索计数 1，均未超预算。

停止理由（落盘版本，已去掉候选列表）：

> 第一次检索已返回 5 条结果，最高分 17.86（science / Experimental Invention），
> 其余集中在高科技数码类目，与查询意图相关性较好。剩余预算仅 1 次，
> 且去掉任一 token 都可能削弱语义覆盖，收益有限，故停止检索。

完整轨迹：trace.json（本地文件：`run-20261109T080657Z-19556/trace.json`）、
metrics.json（本地文件：`run-20261109T080657Z-19556/metrics.json`）、
run_meta.json（本地文件：`run-20261109T080657Z-19556/run_meta.json`）。

| 证据 | sha256 |
| --- | --- |
| `trace.json` | `05fd04386872e769544e2ae6717354e7c5b6ff719f993e9ad50cb5eff08ff14d` |
| `metrics.json` | `f96a39f3ec88f6ffb747dc044cda82c1bd272fb65c8da76157eed90340cb9f34` |
| `run_meta.json` | `48a074126b9da34242768f62756f0a03635d71fedec0c8fdccfa3b0f1f12be0f` |
| `self_check.json` | `041c55d28fed406e058ef74cece2775c6b15508d06e3fa53c7e5da7b7b1af9b2` |

## 5. 落盘边界

模型最终回复原始长度 **503 字符**，其中附带了逐条候选列表。按本轮交付边界，
落盘前只保留停止理由，候选列表截断；原始回复只留 sha256
`bc0963cef562925b1ed65ef0ae1e68f6dd3918b73ef9c13cfcd0bdec285171fb`，
不保存原文。轨迹中的 `tool_return` 只有数量、top_k、query token 和预算，
没有逐条候选；`candidates_persisted=false`、`final_reply_redacted=true`。

本地 self-check、本报告和开发日志中均不重复保存候选明细。

## 6. 预算与范围

| 项 | 实际 |
| --- | --- |
| 真实工作流 | 1 次 |
| LLM 请求 | 2（上限 3） |
| BM25 查询 | 1（上限 2） |
| 训练 / 重建索引 / 扫原始 CSV | 0 / 0 / 0 |
| 向量检索 / 排序 / 全量评测 | 0 / 0 / 0 |
| 自动重试 | 0 |

## 7. 遇到的问题与处理

1. **模型在最终回复里附带逐条候选**：原始回复 503 字符，包含 5 条 item_id/分数/类目。
   按“候选只在本次查询和决策中传递，默认不逐条落盘”的边界，落盘前截断为停止理由，
   并保留原始回复 sha256；代码中也加入同样的截断逻辑与 self-check 用例。
2. **第一次真实运行后才发现上述问题**：没有为了修正而重跑真实工作流；
   只对已生成的 trace/metrics 做一次落盘前截断，原始回复哈希保留。
3. **实时只覆盖“查一次后停止”**：模型判断不需要第二次查询；两次查询的合法/重复/
   非子集/空查询/最终工具调用等路径由 12 项本地 self-check 覆盖，不能写成真实运行过。

## 8. 遗留项

- [ ] 中文文本输入 / query→token 编码；
- [ ] 向量检索、排序/重排；
- [ ] 更多真实查询与两次查询路径的在线验收；
- [ ] 更大索引或全量数据上的检索评测；
- [ ] Search Agent 的多轮反馈与预算策略。
