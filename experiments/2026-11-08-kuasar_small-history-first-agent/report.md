# 2026-11-08 · Phase 11.2：多工具注册与历史实验复用（真实运行通过）

## 结论

1. **真实历史复用通过**：一次真实 `deepseek-chat` 工作流先调用原生
   `search_experiment_history`（`call_00_EYwK4whr6AYjR48lZIYM3943`），命中 Phase 11
   已落盘的 **30/70** 记录；程序用同一 `tool_call_id` 回传结果，模型引用
   **5 个命中、Recall@100 0.013672**，相对 50/50 基准多 1 个目标后停止。
   `decision: reuse_history`，**融合评测执行 0 次**，指标全部来自读取工件。
2. **未命中分支通过（本地模拟）**：临时历史目录没有 30/70 记录时，图允许一次评测、
   同 ID 回传实际工具结果并收尾；查询前评测、多条 tool_calls、重复评测、命中后评测
   与超预算均被程序拦截。
3. **非默认缓存目录通过**：独立复制目录贯穿基准校验、历史匹配与真实融合评测；
   缺失目录和篡改 `recall_mind.json` 的副本均被拒绝（CLI 2），没有回退默认缓存。
4. **回传失败仍正确**：历史命中后最终请求超时、评测成功后最终请求超时，均为
   `partial`、模拟 CLI 2、工具结果保留、不写 `verified`。
5. **范围合规**：训练 0、重新召回 0、测试集评估 0；未覆盖旧缓存或旧实验。

## 范围与预算

| 项 | 限制 | 实际（真实运行） |
| --- | --- | --- |
| 模型请求 | ≤ 3 | **2** |
| 历史查询 | ≤ 1 | **1** |
| 配额评测 | ≤ 1（命中时必须 0） | **0** |
| 耗时 | ≤ 180 s | **2.21 s**（图内计时） |
| 用户 / 验证目标 | 128 / 362 | 128 / 362 |
| 每用户候选 | ≤ 100 | 12,800 / 128 = 100 |
| 训练 / 重新召回 / 测试集 | 0 / 0 / 0 | 0 / 0 / 0 |

## 实现

- **工具注册表** `src/recsys/experiments/agent_tools.py`：集中维护
  `search_experiment_history` / `evaluate_fusion_quota` 的 schema、参数校验和执行函数；
  参数只允许 `two_tower_quota`、`mind_quota` 两个整数，禁止路径/命令/自定义脚本；
  LangGraph 只通过 `ToolRegistry.execute` 执行注册过的工具。
- **历史匹配** `src/recsys/experiments/experiment_history.py`：逐项核对
  数据集/验证切分、固定用户与两路候选缓存文件 sha256、模型 checkpoint sha256、
  目标配额、每用户 100 预算、融合顺序/去重/补位/来源，以及指标口径；
  同时核对 `fusion.json` 工件的配额、预算、池大小和逐用户统计。旧记录无规则版本字段，
  用同一实验目录的 `config.toml` + 工件确认兼容性，结果标记 `legacy_record: true`。
  `partial`/失败/缺工件/指纹不符/配置不符一律 `miss` 并记录原因。
- **工作流** `src/recsys/experiments/history_first_agent.py`：
  `load_baseline → request_history → execute_history → send_history_result`；
  命中直接 `finish`，未命中且模型选择评测才进入
  `execute_eval → send_eval_result → finish`。第一次请求只开放历史工具，
  命中后评测工具不开放；程序校验请求数、历史查询数、评测数、工具顺序和同 ID 回传。
- **兼容上一轮入口**：`python -m recsys.experiments.llm_tool_call --experiment-dir ...`
  保持可用；该模块的 `--cache-dir` 也改为从 CLI 状态贯穿基准校验和工具执行。

## 真实运行证据

| 项 | 值 |
| --- | --- |
| 类型 | **真实 LLM**（非模拟） |
| 状态 / 决策 | **verified / reuse_history** |
| 图节点 | `load_baseline → request_history → execute_history → send_history_result → finish` |
| 响应 ID | `595b0020-f945-46a0-846b-3f01e6b729bb`（历史请求）、`04ce3279-7e95-4c2c-a277-ba05c5999c09`（最终回复） |
| `tool_call_id` | `call_00_EYwK4whr6AYjR48lZIYM3943` |
| 模型参数 | `{"two_tower_quota": 30, "mind_quota": 70}` |
| 请求 / 历史 / 评测 | **2 / 1 / 0** |
| 缓存目录 | `artifacts/kuasar_small/mvp`（默认，哈希已核对） |
| 历史命中 | `experiments/2026-11-08-kuasar_small-fusion-quota-agent/runs/30_70.json` |
| 证据工件 | `artifacts/kuasar_small/quota_agent/cache_eval/30_70/fusion.json` |

| 指标 | 基准 50/50 | 历史 30/70 | 差值 |
| --- | --- | --- | --- |
| 命中目标数 | 4 | **5** | **+1** |
| Recall@100 | 0.011719 | **0.013672** | +0.001953 |
| Coverage | 0.026843 | 0.026720 | −0.000123 |
| 候选数（总 / 每用户） | 12,800 / 100 | 12,800 / 100 | 0 |

模型最终回复：

> 历史结果显示，30/70 配额命中 5 个验证目标、Recall@100 为 0.013672，而 50/50 基准命中 4 个、Recall@100 为 0.011719，前者多保留 1 个目标且 Recall 提升 0.001953；由于该目标配额已有与当前缓存一致的可复用成功记录，按“命中目标数优先、Recall@100 次之”的规则已判定 30/70 更优，故停止，不再执行融合评测。

停止原因：**历史结果与当前输入和评估口径一致，直接复用，未执行融合评测**。
证据：[metrics.json](metrics.json)、[events.jsonl](events.jsonl)、
[decision.json](decision.json)、[baseline.json](baseline.json)、[config.toml](config.toml)。

## 本地分支验证（均不请求真实模型）

汇总证据：[local_checks.json](local_checks.json)。下面每一项也都保留了独立的 `metrics.json` / 检查 JSON。

| 分支 | 命令 | 结果 | 证据 |
| --- | --- | --- | --- |
| 解析/顺序/预算自检 | `--self-check` | 12 项全过：查询前评测、未知工具、路径/命令/脚本参数、配额不一致、命中复用、未命中一次评测、未命中停止、多条 tool_calls、命中后请求工具、重复评测、命中后评测、请求上限 | [self_check.json](self_check.json) |
| 未命中 → 一次评测 | `--regression-miss` | verified / evaluated / 3 请求 / 历史 1 / 评测 1（fake eval）；同 ID 回传 | [regression_miss.json](regression_miss/regression_miss.json) |
| 历史命中后最终请求失败 | `--regression-timeout history` | **partial** / CLI 2 / 2 请求 / 评测 0 / 历史结果保留 / 无 verified | [regression_timeout.json](regression_timeout_history/regression_timeout.json) |
| 评测成功后最终请求失败 | `--regression-timeout evaluation` | **partial** / CLI 2 / 3 请求 / 评测 1 / 评测结果保留 / 无 verified | [regression_timeout.json](regression_timeout_evaluation/regression_timeout.json) |
| 历史指纹不匹配 | 篡改副本的 `recall_mind_sha256` | **miss**，原因“指纹不一致 recall_mind_sha256” | [history_checks.json](cache_dir_check/history_checks.json) |
| 历史记录 partial | 副本写入 `status=partial` | **miss**，原因“运行状态不可复用：partial” | [history_checks.json](cache_dir_check/history_checks.json) |

## 非默认 `--cache-dir` 验证

独立复制目录：`artifacts/kuasar_small/history_first_agent/cache_dir_check/mvp_copy`
（只复制 `users.json` / `run_meta.json` / 两路召回；原缓存未改动）。

| 检查 | 命令 / 输入 | 结果 |
| --- | --- | --- |
| 基准 + 缓存身份 | `--check-cache --cache-dir .../mvp_copy` | `status: ok`，`cache_dir_source: cli`，哈希与默认目录一致 |
| 基准 + 历史 + 真实评测 | `--regression-miss --real-eval --cache-dir .../mvp_copy --output-dir .../output` | verified；评测命中 5；`evidence.cache`、历史 `cache_dir`、基准 `cache` 全部指向 `mvp_copy`；融合工件写到独立 output |
| 缺失目录 | `--check-cache --cache-dir .../does_not_exist` | **blocked**，CLI 退出码 **2**，未回退默认目录 |
| 篡改缓存 | 副本追加换行改变 `recall_mind.json` 哈希 | **blocked**，CLI **2**，“基准指纹与当前缓存不一致：['recall_mind_sha256']” |

证据：[cache_check.json](cache_dir_check/cache_check.json)、
[regression_miss/metrics.json](cache_dir_check/regression_miss/metrics.json)、
[missing/cache_check.json](cache_dir_check/missing/cache_check.json)、
[tampered/cache_check.json](cache_dir_check/tampered/cache_check.json)。

CLI 解析后的目录进入 LangGraph 状态字段 `cache_dir`；`node_load_baseline`、
`node_execute_history`、`node_execute_eval` 均从该状态取路径，节点内不再使用
`DEFAULT_CACHE`。省略参数时仍使用 `artifacts/kuasar_small/mvp`；显式目录缺失、
哈希不符或口径不符直接报错，不回退。

## 隐私与证据边界

- 发送给模型的只有任务配置、聚合指标、历史 `hit/miss` 与规则版本；**不发送**用户/物品
  原始 ID、候选明细、验证标签、文件路径、哈希或密钥。
- `events.jsonl` / `metrics.json` 中的证据路径与哈希只保存在本地实验目录。
- 未训练、未重新召回、未跑推荐全链路、未做测试集评估；旧实验与 `artifacts/kuasar_small/mvp`
  未被覆盖（新实验的候选只出现在本地 `cache_dir_check/output`）。
- 正式真实 LLM 工作流只跑 **1 次**；其余分支均用 fake chat / fake eval 或临时历史目录验证。

## 验收对照

| # | 验收项 | 结果 |
| --- | --- | --- |
| 1 | 真实历史复用：原生调用、同 ID 回传、5 命中 / 0.013672、比基准多 1、评测 0 次 | **通过** |
| 2 | 未命中分支：允许一次评测并回传；查询前评测/重复/越预算被拦截 | **通过（本地模拟）** |
| 3 | 非默认缓存路径：基准、历史、评测都用指定目录；缺失/不匹配被拒绝 | **通过（本地检查）** |
| 4 | 回传失败仍正确：partial、CLI 非零、工具结果保留 | **通过（本地模拟）** |
| 5 | 资源与证据：0 训练 / 0 重新召回 / 0 测试集；报告含请求/工具/耗时/停止原因 | **通过** |

## 复现

```powershell
$env:PYTHONPATH = "src"   # 未安装 editable 包时

# 真实运行（会请求 1 次真实 LLM 工作流，共 2 次模型请求）
python -m recsys.experiments.history_first_agent `
  --experiment-dir experiments/2026-11-08-kuasar_small-history-first-agent `
  --quota 30/70

# 本地分支
python -m recsys.experiments.history_first_agent --self-check
python -m recsys.experiments.history_first_agent --regression-miss
python -m recsys.experiments.history_first_agent --regression-timeout history
python -m recsys.experiments.history_first_agent --regression-timeout evaluation

# 独立缓存目录（先复制 users.json / run_meta.json / recall_two_tower.json / recall_mind.json）
python -m recsys.experiments.history_first_agent --check-cache --cache-dir <copy>
python -m recsys.experiments.history_first_agent --regression-miss --real-eval `
  --cache-dir <copy> --output-dir <output>
```
