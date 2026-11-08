# 2026-11-08 · Phase 11 验收：真实 LLM 工具调用（未验收）

## 结论先说

**未验收。** 本机没有配置 `DSH_AGENT_LLM_BASE_URL` / `DSH_AGENT_LLM_API_KEY` / `DSH_AGENT_LLM_MODEL`，
验收入口在发起任何网络请求之前就显式失败（exit 2），没有回落本地策略、没有伪造工具调用。

已交付的是**完整实现 + 边界验证**；真实服务商的 `tool_calls` 与 `tool_call_id` 回传轨迹，
需要在配置密钥后跑一次（命令见文末），届时应把 `metrics.json` 的 `status` 从 `blocked` 变为 `verified`。

## 实现了什么

新增 `src/recsys/experiments/llm_tool_call.py`（显式要求真实 LLM）：

1. **读基准**：加载上一轮已落盘的 `50/50` 指标，并重新核对 128 用户 / 362 目标、切分、ID 映射、
   两路缓存与 checkpoint 指纹；任一不一致即停止；
2. **原生工具调用**：请求注册 `evaluate_fusion_quota` 的 OpenAI tools schema（两个必填整数），
   `tool_choice="auto"`；解析服务商响应里的 `tool_calls`，要求恰好一条、函数名匹配、有 ID、参数合法；
3. **程序执行工具**：按模型给出的配额调用现有 `evaluate_fusion_quota`，真实读取缓存、计算候选池与指标，
   候选明细写到 `artifacts/kuasar_small/llm_tool_call/cache_eval/<quota>/`（独立目录，不覆盖 2.23 的工件）；
4. **同 ID 回传**：把工具结果作为 `role="tool"` 消息、携带原 `tool_call_id` 发回模型；
   第二次请求不开放工具，要求它对比基准并给出停止理由；
5. **日志**：`events.jsonl` 依次记录基准加载、注册的工具名、响应 ID、`tool_call_id` 与参数、
   工具真实返回值、回传动作、模型最终回复、停止原因；不含 Authorization、原始用户/物品 ID、候选明细与标签。

预算：最多 2 次模型请求、恰好 1 次工具调用、180 秒；请求超时取 `min(30s, 剩余预算)`，失败不重试。

## 边界验证（`--self-check`，已通过）

用合成响应验证解析与守卫，不访问网络：

```
接受合法的单条 tool_call 并解析参数
拒绝普通文本 JSON（无 tool_calls）
拒绝多条 tool_calls
拒绝未知工具
拒绝非法配额 60/40
拒绝重复基准 50/50
```

## 停点与证据

| 项 | 值 |
| --- | --- |
| 停点 | `llm_config()`：三个环境变量均未设置 |
| 网络请求 | 0 次 |
| 工具调用 | 0 次 |
| 训练 / 重新召回 | 0 / 0 |
| 基准复用 | 未执行（配置检查先于基准加载） |
| 耗时 | 0.0 s（预算 180 s） |
| 证据 | `events.jsonl`（`blocked` 事件）、`metrics.json`（`status: blocked`） |

## 配置密钥后的运行方式

推荐直接填仓库根目录的 `.env`（已被 `.gitignore` 忽略，不会提交；模板见 `.env.example`）：

```powershell
# 编辑 .env：把 DSH_AGENT_LLM_API_KEY= 后面填上 key
python -m recsys.experiments.llm_tool_call --experiment-dir experiments/2026-11-08-kuasar_small-llm-tool-call
```

也可以临时用环境变量（优先于 `.env`）：

```powershell
$env:DSH_AGENT_LLM_BASE_URL='https://api.deepseek.com/v1'
$env:DSH_AGENT_LLM_API_KEY='<你的 key>'
$env:DSH_AGENT_LLM_MODEL='deepseek-chat'
python -m recsys.experiments.llm_tool_call --experiment-dir experiments/2026-11-08-kuasar_small-llm-tool-call
```

入口会读取 `.env`，只补未设置的环境变量、不打印值；日志只记录 model 名与 endpoint host。

成功时 `metrics.json` 会包含：`status: verified`、`response_ids`、`tool_call_id`、
`baseline` 与 `chosen` 的命中数 / Recall@100 / Coverage、`delta_*`、模型最终回复与预算使用情况。
注意：配额输给基准也是合法结果，不做「必须赢」的预设。

## 验收对照

| # | 验收项 | 结果 |
| --- | --- | --- |
| 1 | 真实响应包含一条合法 `evaluate_fusion_quota` 的 `tool_calls`，日志可确认参数来自模型 | **未跑**（无配置，0 次请求） |
| 2 | 按该参数执行一次真实评测，候选落盘、每用户 ≤100 且不重复、指纹与基准一致 | **未跑**（无工具调用） |
| 3 | 用原 `tool_call_id` 回传结果，最终回复引用真实结果，差值可对账 | **未跑** |
| 4 | 零训练、零重新召回、未用测试标签、未覆盖上轮产物 | **通过**（本轮未执行任何评测/训练；入口只读） |
| 5 | 调用次数/耗时在预算内；缺少密钥时记录停点并标为未验收 | **通过**（0 请求、0.0 s、`blocked`，exit 2） |

本地策略、Mock 和普通 JSON 回复都不能替代本轮证据，因此不把 2.23 的本地 Agent 结果算作验收通过。
