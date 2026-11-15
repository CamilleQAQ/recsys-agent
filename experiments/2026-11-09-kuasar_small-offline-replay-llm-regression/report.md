# 2026-11-09 · Phase 13 真实 LLM 离线回放：截断回复本地回归

## 状态

- 结果：**pass，13 / 13**
- 实际命令：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_llm --self-check
```

- 范围：只运行本地替身（合成响应 + LangGraph），**真实 LLM 请求 0、联网 0、训练 0、重新召回 0、真实评测 0**。
- 之前的真实运行目录 `experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/` 保持原样，未改写。

## 修复内容

`src/recsys/experiments/offline_replay_llm.py`：

1. 新增统一的回复校验 `_reply_status(response)` / `_finish_reason(response)`：
   - 只有 `finish_reason == "stop"`、正文非空、且没有 `tool_calls` 才返回 `ok`；
   - `finish_reason == "length"` 返回 `truncated`；
   - 其他非 `stop` 返回 `non_stop`；
   - 空正文返回 `empty`。
2. `request_feedback` 的“主动停止”路径与 `request_final` 的“最终回复”路径都使用该规则。
3. `finish_reason == "length"` 且已有成功揭晓时：
   - 状态改为 `partial`；
   - `stop_code = "reply_truncated"`；
   - 保留已揭晓指标、预算消耗和工具轨迹；
   - 不设置 `final_reply`，因此 `build_metrics` 的三方比较为 `not_judged`。
4. `stop` 轨迹保留 `finish_reason` 与 `reply_status`，可区分截断与正常完成；`node_finish` 合并而不是覆盖这些字段。

## 回归覆盖

| 场景 | 结果 |
| --- | --- |
| 揭晓一次后，停止说明 `finish_reason="length"` 且正文非空 | `partial` / `reply_truncated`，1 次揭晓，剩余预算 1，三方比较 `not_judged` |
| 揭晓两次后，最终回复 `finish_reason="length"` 且正文非空 | `partial` / `reply_truncated`，2 次揭晓，剩余预算 0，三方比较 `not_judged` |
| 正常主动停止 | 原有 `verified`，1 次揭晓 / 2 次请求 |
| 正常最终回复 | 原有 `verified`，2 次揭晓 / 3 次请求 |
| 最终回复超时、重复查询、超预算查询 | 原有回归继续通过 |

另保留原有首轮不可见隐藏指标、工具调用 ID 对应、非法配额、请求/揭晓上限等断言。

## 证据

- [self_check.json](self_check.json)：13 条断言逐条结果
- [config.toml](config.toml)：本次自检使用的配置副本
- [offline_replay_llm.py](../../src/recsys/experiments/offline_replay_llm.py)：状态判断与自检实现
- 修复前的真实运行记录：report.md（本地文件：`../2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/report.md`）

## 边界

本回归只验证截断状态判断和本地替身路径，不重新验证真实 LLM 行为；真实 LLM 的历史运行证据保持原样。
