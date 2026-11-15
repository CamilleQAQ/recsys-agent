# Ads Agent：复用 YieldAgent 结构的广告工具调用 Agent

> 固定本地源码：`third_party/YieldAgent`（commit `e79fda0`）。
> 本轮只复用它「工具调用 / 工具与业务分离 / 工具调用与结果记录」的结构与少量纯 Python
> 辅助函数；LinkedIn、投放草稿审批、网页和外部广告账户不在范围内。

## 1. 结构

```
Ads Agent（LangGraph StateGraph）
  prepare → plan ⇄ execute → finish
                │
                ├── planner=llm   ：原生 tool_calls，逐条校验并顺序执行（模型一次返回多条时排队）
                └── planner=local ：确定性计划，离线可复现
                                  │
                                  ▼
            recsys.ads.tools（工具注册表，薄函数，只做 schema/调度）
                                  │
                                  ▼
            recsys.ads.service（业务：预测、排序、竞价、预算、标签核对）
                                  │
                                  ▼
            AdsRuntime（数据 / 特征 / 历史 / 模型缓存）
```

工具不会 import 模型内部或竞价实现，业务函数不知道 LLM 存在。这与 YieldAgent
「工具只面向 `Connector` 契约、平台逻辑在 connector 里」的分层一致。

## 2. 具体复用了哪些代码 / 哪些按本项目改写

| YieldAgent 位置 | 内容 | 本项目处理 |
| --- | --- | --- |
| `agents/console/tools.py` | 每个工具是薄函数、docstring 即工具说明、底部 `CONSOLE_TOOLS` 列表统一注册；工具不 import 平台模块 | **结构复用**：`src/recsys/ads/tools.py` 的 `_tool_*` 薄函数 + `build_registry()` 注册；工具只调用 `recsys.ads.service` |
| `agents/console/tools.py` 的 `create_draft` | 失败时返回结构化 `{created:false, error, fixable}`，不抛到 UI | **语义复用**：`ToolRegistry.call` 把异常转成 `{ok:false, error, traceback}`；Agent 不把失败写成成功 |
| `agents/console/agent.py` | `get_console_agent` 按模型缓存 Agent；`ToolNode(..., handle_tool_errors=True)` 让工具错误回到模型而不是中断 | **改写**：本项目用 LangGraph `StateGraph` + 工具注册表；工具异常由 `ToolRegistry.call` 捕获并回传 `ok=false`。没有使用 `create_react_agent` / `ToolNode`（本环境未安装 LangChain） |
| `agents/console/runtime.py` | `_coerce`：把工具字符串结果恢复成 JSON/Python 对象；`_summarize`：结果截断 300 字；事件 `tool_call` / `tool_result` | **改写复用**：`src/recsys/ads/agent.py` 的 `coerce_payload` / `summarize_payload`；轨迹写 `tool_call` / `tool_result` 事件，模型结果按同一 `tool_call_id` 回传 |
| `connectors/base.py` | `Connector` Protocol、`ConnectorManifest` 自描述、`PublishError` 结构化错误 | **按广告改写**：`AdModelSpec`（模型 manifest：framework/task/needs_history/upstream）、`AdModelUnavailable`（缺依赖/权重/词表不一致的明确错误）。没有平台连接器，因此没有逐字实现 `Connector` Protocol |
| `agents/console/prompts.py` | 把平台约束写进系统提示 | **按广告改写**：系统提示要求只用工具、`interface_test` 不能当训练、CVR 不能拿 click 冒充、预算不足如实报告 |

**没有复用/不在范围：** `create_react_agent`、`@tool`、`MemorySaver`、`interrupt` 审批、
SSE 事件流、LinkedIn connector、`create_draft` 写操作、网页与外部广告账户。

LLM 配置没有另起一套：沿用 `recsys.experiments.llm_tool_call.llm_config()` 读取的
`DSH_AGENT_LLM_BASE_URL / DSH_AGENT_LLM_API_KEY / DSH_AGENT_LLM_MODEL`（与本项目
Rec Agent / Search Agent 相同），HTTP 客户端复用 `post_chat`。

## 3. 工具

| 工具 | 参数 | 真实执行的内容 |
| --- | --- | --- |
| `list_ad_models` | — | 六个模型的 framework/task/权重状态/checkpoint |
| `predict_ad_ctr` | `model`, `split`, `limit` | 加载权重、真实前向、按 click 标签算 AUC/LogLoss/CTR |
| `rank_ads` | `model`, `split`, `requests`, `min_candidates`, `top_k` | 真实打分 + 请求内排序 |
| `simulate_auction` | `model`, `split`, `requests`, `min_candidates`, `top_k`, `total_budget?`, `per_advertiser_budget?` | 二价 GSP、CPC 扣费、预算扣减；预算不足返回 `no_winner_reason` |
| `get_budget_state` | — | 最近一次竞价的 run_id、花费、剩余、广告主花费 |
| `check_label_availability` | — | 真实 click 与缺失的 conversion；synthetic 模拟样本位置 |
| `predict_multitask` | `model`（mmoe/ple）, `limit` | 在 synthetic 模拟样本上输出 CTR/CVR 两路结果 |

工具参数由 `ToolRegistry.validate` 校验：未知工具、额外参数、枚举外模型、负数/越界、
非整数全部拒绝；相同 `tool+args` 重复调用只允许一次。

## 4. 预算与状态语义

| 项 | 默认 | 说明 |
| --- | --- | --- |
| 工具调用 | 8 次 | 超过后 `status=partial` |
| LLM 请求 | 6 次 | 每次调用模型前先计数；真实 LLM 一轮通常 3–5 次 |
| 墙钟 | 180 s | 到期后 `status=partial` |
| 单次工具结果 | 结构化 | 轨迹只存 `result_summary`，明细在 `metrics.json` |

| 状态 | 条件 |
| --- | --- |
| `verified` | 所有已执行工具 `ok=true`（本地与真实 LLM 都是这个口径） |
| `partial` | 有成功也有失败；或工具/时间/LLM 预算耗尽；或最终回复前失败 |
| `failed` | 全部失败，或工具参数非法、未知工具 |
| `blocked` | 未进入执行（如显式 `--planner llm` 但没有 LLM 配置） |

## 5. 轨迹与落盘

每次运行在 `--output-root` 下新建 `run-<UTC>-<pid>/`：

| 文件 | 内容 |
| --- | --- |
| `events.jsonl` | `prepare` / `plan_next` / `tool_call` / `tool_end` / `tool_result` / `finish`；含工具名、参数、`tool_call_id`、ok、耗时 |
| `trace.json` | 步骤列表（序号、工具、参数、`tool_call_id`、结果摘要）、计划器、LLM 请求数、预算 |
| `metrics.json` | 每个工具的真实输出摘要（CTR AUC/LogLoss、排序统计、竞价花费/剩余、synthetic 标记） |
| `run_meta.json` | 状态、模型、工具序列、错误、耗时、六模型权重状态 |
| `report.md` | 人读报告：状态、轨迹表、指标、最终回复 |

## 6. 实际运行记录

### 6.1 本地确定性计划（离线，`planner=local`）

`experiments/2026-11-09-kuasar_small-ads-phase3/agent_local/run-20261109T143340Z-24812/`

- 模型：`din`；7 个工具全部成功，`verified`，4.66 s；
- 工具：`list_ad_models → predict_ad_ctr → check_label_availability → rank_ads → simulate_auction → get_budget_state → simulate_auction`；
- CTR：test 43 行，AUC 0.5833 / LogLoss 0.8322（固定初始化）；
- 排序：train 3 个多候选请求 / 9 个候选；
- 竞价：3 请求，赢家 3，点击 2，扣费 2.1672，剩余 17.8328 / 20；
- 0 预算：3 请求全部 `budget_exhausted_total`，扣费 0。

### 6.2 真实 LLM（`planner=llm`，`deepseek-chat`）

`experiments/2026-11-09-kuasar_small-ads-phase3/agent_llm/run-20261109T142641Z-25212/`

- 任务：「用 deepfm 模型对 kuasar_small 的广告候选做 CTR 预测和广告排序，并跑一次竞价与预算模拟，最后说明转化标签情况」；
- 模型自主选择 `deepfm`；4 个工具、3 次模型请求、21.25 s，`verified`；
- 工具：`predict_ad_ctr(test, 43) → rank_ads(train, 20 请求) → simulate_auction(train, 20 请求) → check_label_availability`；
- 竞价：20 请求全部成交，9 次计费点击，花费 9.9028，剩余 10.0972 / 20；
- 最终回复正确说明：权重 `fixed_init_interface_test`、未训练、CVR 无真实标签、未把 click 当转化。

**一次失败尝试（如实记录）：** 首次真实调用在 `max_llm_requests=4` 下用完 4 次请求完成
4 个工具，但没有第 5 次请求收尾，状态 `partial`（目录 `run-20261109T142616Z-30600/`）。
把配置改成 6 后重跑，得到上表的 `verified`。这是请求预算配置问题，不是工具执行失败。

## 7. 自检（14/14）

`python -m recsys.ads.agent --self-check`（不联网，除真实 LLM 外的路径全部本地）：

1. `list_ad_models`：六个模型且全部可用；
2. 未知模型被 schema 拒绝；
3. `predict_ad_ctr` 真实推理返回 metrics；
4. `rank_ads` 降序正确；
5. `simulate_auction` 有赢家、扣费>0、剩余一致；
6. `get_budget_state` 与最近 run_id 一致；
7. 0 预算：无赢家、无扣费、每个请求有原因；
8. `predict_multitask`：synthetic 双输出；
9. 标签核对：conversion 缺失、模拟样本 synthetic；
10. LLM 替身：同 `tool_call_id` 回传，2 请求 verified；
11. LLM 未知工具被拒，无工具执行；
12. 重复工具调用被拒（partial）；
13. 工具次数预算生效（partial，只执行 1 次）；
14. 本地固定计划端到端 verified，5 个工件齐全。

结果：`experiments/2026-11-09-kuasar_small-ads-phase3/self_check.json`。

## 8. 命令

```powershell
$env:PYTHONPATH = "src"

# 离线固定计划（可复现，不调用 LLM）
python -m recsys.ads.agent --planner local

# 真实 LLM（需要 .env 里的 DSH_AGENT_LLM_*）
python -m recsys.ads.agent --planner llm `
  --task "用 deepfm 模型对 kuasar_small 的广告候选做 CTR 预测和广告排序，并跑一次竞价与预算模拟，最后说明转化标签情况"

# 自检
python -m recsys.ads.agent --self-check
```

## 9. 限制

1. 没有训练：Agent 拿到的六个模型都是固定初始化权重，指标只能作为接口检查。
2. 真实 LLM 只跑过 1 次成功工作流（另 1 次因请求预算不足 partial）；温度固定 0，未做多次重复。
3. `kuasar_small` 的 test 只有 43 个广告候选，LLM 若按默认 test 做竞价，多数请求只有 1 个候选，
   竞价退化为保留价成交；报告中明确建议用 train 的多候选请求。
4. LinkedIn、投放草稿审批、网页、外部广告账户未接入（范围外）。
5. YieldAgent 的 LangChain `@tool` / `create_react_agent` 没有复用（依赖不在本环境），
   复用的是它的工具/业务分层、错误语义和结果记录方式；差异已在上表逐项列出。
