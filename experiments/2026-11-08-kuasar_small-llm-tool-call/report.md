# 2026-11-08 · Phase 11 验收：真实 LLM 原生 tool_calls（通过）

## 结论

**验收通过。** 真实服务商（`deepseek-chat`）返回了 1 条原生 `tool_calls`，模型选择的配额是
**70/30（双塔 / MIND）**；程序按参数执行了真实评测，并用同一个 `tool_call_id` 把结果回传给模型。
该配额**输给了基准**：命中 3 vs 4、Recall@100 0.009115 vs 0.011719，因此按规则保留基准 50/50。
新配额输了也算验收通过（协议允许），不做「必须赢」的预设。

## 证据链（`events.jsonl`，10 条）

| 顺序 | 事件 | 关键内容 |
| --- | --- | --- |
| 1 | `llm_config` | model=`deepseek-chat`，endpoint host=`https://api.deepseek.com`（不含密钥） |
| 2 | `baseline_loaded` | 50/50 基准：命中 4、Recall@100 0.011719、覆盖 128 用户 / 362 目标；指纹已核对 |
| 3 | `tools_registered` | `evaluate_fusion_quota`，`tool_choice=auto`，`parallel_tool_calls=false` |
| 4 | `llm_request` | 第 1 次请求（带 tools） |
| 5 | `llm_response` | 响应 ID `e74adcc6-…`；**1 条** `tool_calls`，ID `call_00_9LWE…`，参数 `{"two_tower_quota": 70, "mind_quota": 30}` |
| 6 | `tool_call_accepted` | 同 ID、配额 70/30 |
| 7 | `tool_executed` | 真实结果：命中 3、Recall@100 0.009115、候选 12,800（128×100） |
| 8 | `tool_result_sent` | 同 `tool_call_id`，第 2 次请求（不带 tools） |
| 9 | `llm_final` | 响应 ID `a2cb5d7c-…`：「70/30 命中 3 个目标、Recall@100 为 0.009115，均低于基准的 4 个和 0.011719，因此不采用该配额，停止评测。」 |
| 10 | `finish` | 结论写入 `metrics.json` |

## 数值对照

| 项 | 基准 50/50 | 本次 70/30 | 差值 |
| --- | --- | --- | --- |
| 用户 / 目标 | 128 / 362 | 128 / 362 | 一致 |
| 候选数（总数 / 每用户） | 12,800 / 100 | 12,800 / 100 | 0 |
| 补位数 | 0 | 0 | 0 |
| 命中目标数 | **4** | 3 | **−1** |
| Recall@100 | **0.011719** | 0.009115 | −0.002604 |
| Coverage | 0.026843 | 0.026905 | +0.000062 |
| 工具耗时 | — | 0.46 s | — |

结论：**保留基准 50/50**。本轮只评价 70/30 与基准的差异，不推广到其他配额或完整数据集。

## 预算

| 项 | 限制 | 实际 |
| --- | --- | --- |
| 模型请求 | ≤2 | 2 |
| LLM 发起的工具调用 | 恰好 1 | 1 |
| 耗时 | ≤180 s | 3.05 s（工具内部计时）/ 6.7 s 墙钟 |
| 训练 / 重新召回 | 0 / 0 | 0 / 0 |
| 测试标签 | 不使用 | 未读取 |

## 实现调整（如实记录）

- 第一次运行：代码 bug（`load_baseline` 在重构时被误删），在发请求前失败，未产生任何网络调用或结果；
- 第二次运行：真实服务商一次返回 **4 条并行 `tool_calls`**，被「恰好 1 条」的协议拒绝，未执行工具；
- 修正：请求加 `parallel_tool_calls=false`，并在 system 提示中要求「只发起 1 次工具调用」；
- 第三次运行即上表结果，是**唯一一次执行了工具**的运行；此前两次都没有看到任何配额指标，不存在挑结果的情况。

## 验收对照

| # | 验收项 | 结果 |
| --- | --- | --- |
| 1 | 真实响应含一条合法 `evaluate_fusion_quota` 调用，日志可确认参数来自模型 | **通过**（事件 5/6，含响应 ID、`tool_call_id`、参数） |
| 2 | 按参数执行一次真实评测，候选落盘、≤100 且不重复、指纹一致 | **通过**（`artifacts/kuasar_small/llm_tool_call/cache_eval/70_30/`） |
| 3 | 同 `tool_call_id` 回传，最终回复引用真实结果，差值可对账 | **通过**（事件 8/9；−1 命中、−0.002604 Recall） |
| 4 | 零训练、零重新召回、未用测试标签、未覆盖上轮产物 | **通过**（只读缓存；2.23 工件未改动） |
| 5 | 调用次数与耗时在预算内 | **通过**（2 次请求、1 次工具、3.05 s） |

## 复现

```powershell
python -m recsys.experiments.llm_tool_call --experiment-dir experiments/2026-11-08-kuasar_small-llm-tool-call
```

前置：`.env` 已填 `DSH_AGENT_LLM_API_KEY`（base URL / model 默认 DeepSeek）；`.env` 已被 gitignore。
