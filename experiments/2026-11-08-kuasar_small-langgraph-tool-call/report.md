# 2026-11-08 · Phase 11：LangGraph 工具调用验收（通过）与失败路径回归（通过）

## 结论

1. **真实 LangGraph 运行通过**：图依次走过 `load_baseline → request_tool_call → execute_tool →
   send_result → finish`，模型通过原生 `tool_calls` 选择 **70/30（双塔 / MIND）**，
   程序执行真实评测并用同一 `tool_call_id` 回传，模型给出有效结论后结束。
2. **误报修复**：第二次请求失败时不再写 `verified`。回归用例（首次调用与工具执行成功、第二次请求超时）
   得到 `status: partial`、请求计数 2、工具执行 1、工具结果保留、CLI 非零，且**不出现** `verified`。

该配额输给了基准（命中 3 vs 4，Recall@100 0.009115 vs 0.011719），按规则保留基准 50/50；
新配额输了仍是合法验收结果。

## 图与状态

- 引擎：`langgraph==1.2.20`（`requirements/optional-agent.txt`）；
- 节点：`load_baseline`、`request_tool_call`、`execute_tool`、`send_result`、`finish`；
- 每个节点返回状态更新，条件边检查 `status/error`：`blocked`、`partial`、`failed` 直接进入 `finish`，
  不会落入成功分支；
- 状态字段：`status`、`stage`、`nodes`、`events`、`config`、`baseline`、`request_count`、
  `tool_calls`、`response_ids`、`tool_call_id`、`quota`、`tool_result`、`final_reply`、`error`、`conclusion`。

## 真实运行证据

| 项 | 值 |
| --- | --- |
| 状态 | **verified** |
| 图节点顺序 | `load_baseline → request_tool_call → execute_tool → send_result → finish` |
| 响应 ID | `1eac8b8c-…`（工具调用）、`8804ca67-…`（最终回复） |
| `tool_call_id` | `call_00_QJo2I5RFwJnSMQZrg8Ru7198` |
| 模型选择配额 | 70 / 30 |
| 请求次数 / 工具执行 | 2 / 1 |
| 耗时 | 2.81 s（工具内部计时），墙钟约 8.2 s |

| 指标 | 基准 50/50 | 本次 70/30 | 差值 |
| --- | --- | --- | --- |
| 命中目标数 | **4** | 3 | **−1** |
| Recall@100 | **0.011719** | 0.009115 | −0.002604 |
| Coverage | 0.026843 | 0.026905 | +0.000062 |
| 候选数（总/每用户） | 12,800 / 100 | 12,800 / 100 | 0 |

模型最终回复：

> 双塔 70 / MIND 30 的命中数为 3，低于基准的 4，Recall@100 为 0.009115，也低于基准的 0.011719，因此不如基准，停止评测。

`events.jsonl` 按顺序记录 `node_start/node_end` 与 `baseline_loaded`、`llm_request`、`llm_response`、
`tool_call_accepted`、`tool_executed`、`tool_result_sent`、`llm_final`、`finish`，可核对节点顺序、
调用 ID、参数与结果。

## 失败路径回归（不联网、不读缓存）

```powershell
python -m recsys.experiments.llm_tool_call --regression-partial `
  --experiment-dir experiments/2026-11-08-kuasar_small-langgraph-tool-call/regression_second_request_timeout
```

用注入的 `chat_fn` / `eval_fn` 让「首次调用 + 工具执行」成功、第二次请求抛 `TimeoutError`：

| 检查 | 结果 |
| --- | --- |
| `status == partial` | ✅ |
| 模拟 CLI 退出码 | **2**（非零） |
| 请求次数 | **2**（发出即计数） |
| 工具执行次数 | **1**，工具结果保留 |
| 出现 `verified` | ❌ 未出现 |
| 错误记录 | ✅ "第 2 次模型请求失败：simulated timeout on second request" |

证据：`regression_second_request_timeout/{metrics.json, events.jsonl, regression_partial.json}`。

## 实现调整与调试记录（如实）

- 接入 LangGraph 后第一次真实运行：回传的 assistant 消息缺少 `role` 字段，服务商返回 **HTTP 422**；
  该次被正确记为 `partial`（请求 2、工具 1、CLI 2），说明误报修复生效；
- 补上 `role: assistant` 后重跑成功（即上表 verified 运行）；
- 调试运行发生过工具执行，但配额与缓存完全相同、结果可复现；正式的候选工件写在独立目录
  `artifacts/kuasar_small/langgraph_tool_call/`，未覆盖上一轮 `artifacts/kuasar_small/llm_tool_call/`。

## 预算与约束

| 项 | 限制 | 实际 |
| --- | --- | --- |
| 模型请求 | ≤2 | 2 |
| 工具执行 | 1 | 1 |
| 耗时 | ≤180 s | 2.81 s |
| 训练 / 重新召回 | 0 / 0 | 0 / 0 |
| 测试标签 | 不使用 | 未读取 |
| 缓存与历史实验 | 只读、不覆盖 | 2.23 / 2.24 工件未改动 |

## 验收对照

| # | 验收项 | 结果 |
| --- | --- | --- |
| 1 | 真实 LangGraph 运行通过，节点顺序、同 ID 回传、最终回复与指标一致 | **通过** |
| 2 | 第二次请求失败回归：`partial`、CLI 非零、请求计数 2、工具结果保留、无 `verified` | **通过** |
| 3 | 预算与缓存约束（≤2 请求、1 工具、0 训练/召回、不覆盖旧产物） | **通过** |
| 4 | 结果可对账（配额、命中、Recall、差值、耗时、状态、证据路径） | **通过** |

## 复现

```powershell
python -m recsys.experiments.llm_tool_call --experiment-dir experiments/2026-11-08-kuasar_small-langgraph-tool-call
```

前置：`.env` 已填 `DSH_AGENT_LLM_API_KEY`；`.env` 被 gitignore，日志不含密钥与原始 ID。
