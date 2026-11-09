# 2026-11-09 · Search Agent 修复回归：候选不落盘 + 样本外查询

> 本轮只修 `search_agent.py` 的两个问题，保留 CLI、BM25 接口、LangGraph 流程和预算规则。
> **本地替身回归 45/45 通过**；真实 LLM 请求 0、索引重建 0、真实冒烟重跑 0。
> 旧真实工件与索引哈希前后一致。

## 结论

- **候选/模型自由文本不落盘**：工具结果仍通过 tool 消息回传模型；轨迹、指标、事件和报告只写程序生成的结构化摘要（调用 ID、查询 token、返回数量、停止类型、预算、完成状态）。不再使用关键词截断，也不保存模型正文。
- **样本外查询可执行**：CLI 只传 `--tokens` 时直接用输入查现有索引；能匹配样本会话时补充 `session_id`，匹配不到也继续搜索。合法但无匹配的 token 正常返回 0 条。
- 回归覆盖：样本会话查询（5 条）、样本外命中（1 条）、样本外无匹配（0 条）、`--session-id` 与 `--tokens` 不一致拒绝。
- 本地 self-check **12/12**；索引/样本/旧真实工件/旧 self_check 哈希全部不变。

## 1. 改动位置

`src/recsys/experiments/search_agent.py`：

1. **停止摘要**
   - 删除 `_sanitize_reply` / `_remember_final_reply` 的关键词截断与正文保存；
   - 新增 `_record_final_reply_status`（只记 `ok/truncated/empty/...` 和字符数）与 `_stop_summary`（程序生成停止摘要）；
   - `trace.json` / `metrics.json` / 事件 / 最终摘要不再包含 `content_preview`、`final_reply`、模型原始回复或候选列表；
   - 完成状态仍按 `finish_reason=stop` + 非空正文判断；`length` / 空回复仍走原有 `partial` 失败路径。
2. **样本外查询**
   - `load_context` 的 `--tokens` 分支不再要求命中样本会话；匹配到时补 `session_id` 和来源，匹配不到时 `session_id=null`、`source=cli_tokens_out_of_sample`；
   - `--session-id`、默认会话、两者同时传入时的 token 一致性校验保持不变；
   - 首查仍必须原样使用输入 token；第二次查询仍执行原有子集/非空/不同/预算校验。

## 2. 回归命令

```powershell
$env:PYTHONPATH = "src"

# 通过 CLI 入口传入样本外 token；替身模型，不调用真实 LLM
python -m recsys.experiments.search_agent `
  --regression --tokens 2 `
  --output-root artifacts/search_agent_regression_tokens
```

回归写入 `artifacts/search_agent_regression_tokens/regression-20261109T082434Z-17568/`，
这是 Git 忽略的独立目录；本目录只保留不含候选的 [regression.json](regression.json)
（sha256 `b2a24887aad4c0d4f13604a3a4e0e4cd2c8da5a8593533ca0fb6aea5807f2d2c`）。

## 3. 实际结果

| 用例 | 查询 | session | 返回数量 | 状态 |
| --- | --- | --- | ---: | --- |
| 样本会话 | 会话 11208 的 4 个原 token | `11208` | 5 | verified |
| 样本外命中 | `[2]` | `null` | 1 | verified |
| 样本外无匹配 | `[1000000000000]` | `null` | 0 | verified |
| 会话/token 不一致 | 样本外 token + `--session-id 11208` | — | — | blocked |

每个 case 的 `trace.json` 都核对：

- 首查 tool_call 参数等于输入 token；
- tool_call ID、`tool_return.count`、`stop_code=model_stop` 可核对；
- 第二次请求通过同一 `tool_call_id` 的 tool 消息拿到结果；
- `response` 只有 `content_chars` / `content_present` / `finish_reason` / `tool_calls`，没有正文；
- `rules.model_free_text_persisted=false`、`candidates_persisted=false`；
- 序列化文件里没有替身模型列举的候选 ID，也没有原始回复句子。

## 4. 旧证据核对

| 文件 | sha256（前后一致） |
| --- | --- |
| Search 1 索引 | `3ae2b858d456eab4c4c21ed356812cd44260b56dd596639588a53fb1a6a52690` |
| Search 1 样本 | `12a46f3c3a614a715e52f4a790ccac93eb0fc7519ab5ff4a0826ceace58a7daf` |
| 旧真实 trace | `05fd04386872e769544e2ae6717354e7c5b6ff719f993e9ad50cb5eff08ff14d` |
| 旧真实 metrics | `f96a39f3ec88f6ffb747dc044cda82c1bd272fb65c8da76157eed90340cb9f34` |
| 旧真实 run_meta | `48a074126b9da34242768f62756f0a03635d71fedec0c8fdccfa3b0f1f12be0f` |
| 旧 self_check | `041c55d28fed406e058ef74cece2775c6b15508d06e3fa53c7e5da7b7b1af9b2` |

旧真实冒烟工件未改写；新的 self-check 写在 artifacts 下，不覆盖旧 `self_check.json`。

## 5. 验收

- `regression.json` 中 45 项检查全部 `pass`，`passed=true`。
- `self_check` 12/12。
- 三个替身 case 均 `verified`；样本外查询没有 blocked。
- 候选 ID / 模型自由文本泄漏检查：0。
- 真实 LLM 请求、索引重建、真实冒烟重跑均为 0。

## 6. 已知限制

- `--regression` 是本地替身入口，不验证真实模型是否遵守“不复述候选”的提示；
  真实模型仍可能复述，但代码不会把它的正文写入工件。
- 本轮没有重跑真实冒烟，也没有重新评估检索质量。
- 样本外 token `[2]` 只用于验证查询链路，不代表业务查询。
