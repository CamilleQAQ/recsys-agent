# 2026-11-08 · Phase 12 实验记忆 MVP（通过）

## 结论

**第一次查不到 → 真实评测一次 → 自动入库 → 第二次命中历史 → 零评测。**

两轮使用同一个配额 **80/20**、同一输入缓存、同一记忆目录，分别由两个独立 CLI 进程启动：

| 运行 | 决策 | 历史 | 评测次数 | 命中 | Recall@100 | 实验键 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 第一次 | `miss → evaluated → saved` | Phase 11 JSON 无 80/20；索引为空 | **1** | 2 | 0.005208 | `24cc580d…07962f` |
| 第二次 | `hit → reuse_history → stop` | 命中同一索引记录 | **0** | 2 | 0.005208 | `24cc580d…07962f` |

第二轮从磁盘重新读取 `memory/index.json`，命中第一轮的 run record 与
`80_20/fusion.json`；没有调用评测函数、没有生成新候选、没有改写第一轮结果。
总墙钟 **11.51 s**（run1 6.5 s + run2 5.01 s），低于 120 s 预算。

## 为什么选 80/20

`precheck.json` 核对了 Phase 11 `runs/*.json`、记忆索引和已有候选工件目录：

- Phase 11 runs 已有：50/50、30/70、20/80；
- 已有候选工件：20/80、30/70、50/50、70/30（70/30 来自 2.24/2.25，但没有进入本轮历史 JSON）；
- 未入库且没有候选工件：**80/20**；
- 80/20 的探针查询返回 `miss`，原因是 Phase 11 JSON 无记录且索引为空。

因此选择 **80/20**，不是把已有实验伪装成未命中。证据：[precheck.json](precheck.json)。

## 实验身份与索引

实验键只由以下字段的规范 JSON 计算 sha256：

- 数据集、用户/物品数、验证切分 `split_strategy` / `split_sha256`；
- `id_mapping_sha256`、`routing_seed`、`k_recall`；
- 双塔/MIND checkpoint sha256；
- `users.json`、`recall_two_tower.json`、`recall_mind.json` 的文件 sha256；
- 验证目标 `targets_sha256`（128 用户 → 362 目标，按用户排序后哈希）；
- 配额 80/20、每用户候选预算 100、融合规则版本 `fusion_quota_v1`、指标口径。

运行日期、实验目录名、耗时和模型服务商都不参与实验键。索引格式为
`schema_version=1` 的 JSON，`entries` 以实验键为 key；同一键覆盖，只保留一条可复用记录。
写入顺序是：真实评测工件 → run record（临时文件 + 原子替换）→ `index.json`
（临时文件 + 原子替换）。失败、`partial` 或缺少候选工件的结果不会发布。

读取时同时校验：身份匹配、`status=success`、指标记录完整、run record 存在、
`fusion.json` 可读，以及工件的配额/预算/池统计/指纹与当前缓存一致。旧 Phase 11 JSON
继续走原有兼容性检查。

## 两轮证据

### 第一次：未命中，真实评测并自动入库

- 图节点：`load_baseline → request_history → execute_history → send_history_result →
  execute_eval → send_eval_result → finish`
- 历史查询：`status=miss`，Phase 11 无 80/20 记录，索引无该键；
- 真实评测：命中 **2**、Recall@100 **0.005208**、Coverage 0.026891、候选 12,800（128×100）；
- 对照基准 50/50：命中 4、Recall@100 0.011719（80/20 输给基准，仍按规则记录）；
- 写入实验键：`24cc580d24b4d237a119ceba8901fccc305221ca4a01524a05ba2c1a7207962f`；
- `memory_saved=true`，索引 `entry_count=1`。

证据：[run1/metrics.json](run1/metrics.json)、[run1/events.jsonl](run1/events.jsonl)、
[run1/decision.json](run1/decision.json)。

### 第二次：跨进程命中，零评测

- 图节点：`load_baseline → request_history → execute_history → send_history_result → finish`
- 历史查询：`status=hit`、`source=memory_index`、`memory_key` 与第一轮相同；
- 读取第一轮 run record 和 `artifacts/kuasar_small/experiment_memory/cache_eval/80_20/fusion.json`；
- 复用命中 **2**、Recall@100 **0.005208**；
- `evaluation_count=0`，`evaluation=null`，events 中没有 `tool_executed` / `memory_saved`；
- `memory_saved=false`，索引 `updated_at` 未被第二轮改写。

证据：[run2/metrics.json](run2/metrics.json)、[run2/events.jsonl](run2/events.jsonl)、
[run2/decision.json](run2/decision.json)、[two_run_check.json](two_run_check.json)。

### 索引与工件

- 索引：[memory/index.json](memory/index.json)，1 条记录；
- run record：[24cc580d…07962f.json](memory/runs/24cc580d24b4d237a119ceba8901fccc305221ca4a01524a05ba2c1a7207962f.json)；
- 候选明细：[80_20/fusion.json](../../artifacts/kuasar_small/experiment_memory/cache_eval/80_20/fusion.json)；
- `fusion.json` sha256：`24f826fe8982f8ef20c1da7a23decb8c37dd79f8a0d7a20a43d79813e409554c`；
- 索引 sha256：`842353de7de7533d5bac25a359d5eb52865ba77f4b10316e58b69dae07fb2e84`；
- run record sha256：`c2ab46870eca5f397519267f0c5e5d692e219eaf34ccee67640df167a4b2619d`。

## 发布时机修复（2026-11-09）

**问题**：旧实现只要 `evaluate_fusion_quota` 成功，`node_execute_eval` 就立即调用
`save_evaluation()` 发布索引。若随后最终回复超时，运行会变成 `partial`，但索引已经把这次
实验标成可复用 `success`，下次查询会错误命中。

**修改位置**：

| 位置 | 修改 |
| --- | --- |
| `node_execute_eval`（[history_first_agent.py](../../src/recsys/experiments/history_first_agent.py)） | 只保留评测结果/工件/调用 ID，`memory_saved=false`、`eval_result_returned=false`，**不写索引** |
| `node_send_eval_result` | 只有收到有效最终回复后才置 `eval_result_returned=true` |
| `node_finish` / `_publish_evaluation` | 仅当 `status=verified`、新评测成功、同 `tool_call_id` 回传、最终回复有效时调用 `save_evaluation()` |
| `node_finish` / `_log_publish_skipped` | 未 verified 时写 `memory_publish_skipped` 事件，`memory_saved=false` |
| `_publish_evaluation` | 发布本身失败时 `status=failed`、`memory_failed` 事件、`memory_saved=false`，不显示入库成功 |

**本地回归**（隔离临时记忆目录；复用 `80_20/metrics.json` 只读夹具；真实 LangGraph、
真实索引查询/写入逻辑；不执行真实评测、不请求真实 LLM）：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.history_first_agent --regression-publish-gate `
  --experiment-dir experiments/2026-11-08-kuasar_small-experiment-memory `
  --cache-dir artifacts/kuasar_small/mvp --quota 80/20
```

执行顺序：临时历史 `miss` → 模拟评测成功（读取 `80_20/metrics.json`）→ 最终回复请求超时
→ 用相同输入重新查历史。实际结果：

| 断言 | 实际 |
| --- | --- |
| 最终状态 | **partial** |
| 模拟 CLI 退出码 | **2** |
| 评测次数 / 请求次数 | 1 / 3 |
| 工具结果 | 保留（命中 2、Recall@100 0.005208） |
| 候选证据路径 | 保留 `80_20/fusion.json` |
| 超时原因 | 保留 `simulated timeout after evaluation result` |
| `memory_saved` 事件 | **无** |
| `memory_publish_skipped` 事件 | 有 |
| 临时索引条目 / 本次键 | **0 / 未写入** |
| 重新查历史 | **miss**（未触发第二次评测） |
| 现有 80/20 verified 记录 | **hit**（`memory_index`，只读检查） |

18 项检查全部通过。上一轮修复回归产物保留在 `regression_publish_gate/`，未改写。
迁移后的复现命令与唯一输出目录见下节。

## 回归迁移（2026-11-09）

发布回归已从 `history_first_agent.py` 移到
[regression_publish_gate.py](../../src/recsys/experiments/regression_publish_gate.py)；旧 CLI
`--regression-publish-gate` 仅做委托，参数与退出码语义不变。回归每次创建唯一目录：
省略 `--experiment-dir` 时默认写到 `artifacts/kuasar_small/experiment_memory/regression_publish_gate/`；
显式传入时把它作为输出父目录，再创建 `run-<UTC>-<pid>/` 子目录，并打印实际证据路径。
不再使用 `shutil.rmtree`，不会删除或覆盖旧验收证据。

本次实际命令（显式父目录，且父目录预置了 `existing_regression/marker.txt`）：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.history_first_agent --regression-publish-gate `
  --experiment-dir artifacts/kuasar_small/experiment_memory/regression_publish_gate `
  --cache-dir artifacts/kuasar_small/mvp --quota 80/20
```

实际证据目录：
`artifacts/kuasar_small/experiment_memory/regression_publish_gate/run-20261108T162817Z-10980/`。
回归检查 18/18 通过；被测 Agent 运行仍为 `partial`（模拟 CLI 2），回归命令自身退出 0。
旧 `regression_publish_gate/` 递归 sha256 前后一致，预置 marker 内容未变，新结果写入另一个
唯一目录。证据：[refactor_check.json](refactor_check.json)、
[新回归 regression.json](../../artifacts/kuasar_small/experiment_memory/regression_publish_gate/run-20261108T162817Z-10980/regression.json)、
[新回归 metrics.json](../../artifacts/kuasar_small/experiment_memory/regression_publish_gate/run-20261108T162817Z-10980/metrics.json)。

## 匹配安全检查（不运行评测）

```powershell
python -m recsys.experiments.history_first_agent --memory-safety-check `
  --experiment-dir experiments/2026-11-08-kuasar_small-experiment-memory `
  --memory-dir experiments/2026-11-08-kuasar_small-experiment-memory/memory `
  --cache-dir artifacts/kuasar_small/mvp --quota 80/20
```

| 检查 | 结果 |
| --- | --- |
| 有效索引 | hit |
| 缓存指纹变化（`recall_mind_sha256` 改变） | miss |
| 规则版本不匹配（`fusion_quota_v2`） | miss |
| 候选工件缺失 | miss |
| 记录状态 `partial` | miss |

5 项全部通过，**未运行任何真实评测**。证据：
[memory_safety_checks.json](memory_safety_checks.json)。

## 范围与兼容性

| 项 | 实际 |
| --- | --- |
| 训练 | 0 |
| 重新召回 | 0 |
| 推荐 MVP / 排序 / 重排 | 0 |
| 测试集评估 | 0 |
| RAG / 向量库 | 0 |
| 旧缓存与旧实验 | 只读，未覆盖 |
| 真实 LLM | 本轮未跑；按任务要求使用本地确定性策略 |
| 工具与配额评测 | 真实执行（`evaluate_fusion_quota`，非 Mock 指标） |

兼容性：默认 `policy=llm` 与上一轮真实 LLM 路径不变；新增 `--local-policy` 仅用于本轮
本地验收。`search_experiment_history` 增加可选 `memory_dir`；`evaluate_fusion_quota`
未改。不带 `--memory-dir` 时不会自动入库，Phase 11 CLI 行为保持。
Phase 11 兼容回归 `--regression-miss`（30/70，fake eval）通过：
[regression_compat](regression_compat/regression_miss/regression_miss.json)。

## 复现

```powershell
$env:PYTHONPATH = "src"
$exp = "experiments/2026-11-08-kuasar_small-experiment-memory"

# 0) 选配额（首次运行前）
python -m recsys.experiments.history_first_agent --audit-quotas `
  --experiment-dir $exp --memory-dir "$exp/memory" `
  --output-dir artifacts/kuasar_small/experiment_memory --quota 80/20

# 1) 第一次独立进程：未命中 → 真实评测一次 → 自动入库
python -m recsys.experiments.history_first_agent --experiment-dir "$exp/run1" `
  --cache-dir artifacts/kuasar_small/mvp --memory-dir "$exp/memory" `
  --output-dir artifacts/kuasar_small/experiment_memory --quota 80/20 --local-policy

# 2) 第二次独立进程：命中同一实验键 → 零评测
python -m recsys.experiments.history_first_agent --experiment-dir "$exp/run2" `
  --cache-dir artifacts/kuasar_small/mvp --memory-dir "$exp/memory" `
  --output-dir artifacts/kuasar_small/experiment_memory --quota 80/20 --local-policy

# 3) 匹配安全检查（不评测）
python -m recsys.experiments.history_first_agent --memory-safety-check `
  --experiment-dir $exp --memory-dir "$exp/memory" `
  --cache-dir artifacts/kuasar_small/mvp --quota 80/20
```

## 验收对照

| # | 验收项 | 结果 |
| --- | --- | --- |
| 1 | 首次自动入库：历史 miss、真实评测 1 次、候选/指标/索引完整落盘 | **通过** |
| 2 | 跨进程零评测复用：第二次命中同一键、引用第一轮指标和工件、评测 0 | **通过** |
| 3 | 匹配安全：指纹变化/规则不匹配/工件缺失/partial 不误命中 | **通过（本地样例，0 评测）** |
| 4 | 范围合规：0 训练 / 0 重新召回 / 0 测试集；旧缓存与实验原样 | **通过** |
| 5 | 证据可对账：两轮日志显示查了什么、为何评测/复用、结果来源 | **通过** |
