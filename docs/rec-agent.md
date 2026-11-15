# Rec Agent（Phase 1 v2）

入口：`python -m recsys.experiments.rec_agent`。它在现有 LangGraph + 工具注册表
模式上扩展，工具全部落到真实训练/推理/评估函数；没有 LLM 时用确定性计划器，
有 LLM 时由模型通过原生 `tool_calls` 选择工具，程序负责校验、预算和失败处理。

## 1. 快速运行

```powershell
$env:PYTHONPATH = "src"

# 工具链自检：真实数据画像 / Model Zoo / 真实召回 / 真实链路 / 明确失败
python -m recsys.experiments.rec_agent --self-check

# 确定性工作流：双塔 + MIND 召回，LightGBM 排序
python -m recsys.experiments.rec_agent `
  --task "用双塔和 MIND 召回，LightGBM 排序并评估" `
  --users 16 --recall two_tower,mind --ranker lightgbm_ranker

# 先训练再使用（真实训练入口，受预算限制）
python -m recsys.experiments.rec_agent `
  --task "训练 LightGCN 并接入召回" `
  --train lightgcn --users 8 `
  --recall two_tower,mind,lightgcn --ranker lightgbm_ranker

# 真实 LLM 规划（需要 .env 里的 DSH_AGENT_LLM_*）
python -m recsys.experiments.rec_agent --llm --task "用可用模型做一次推荐并报告指标" --users 8
```

产物在 `experiments/<run>/`：`events.jsonl`（逐事件日志）、`summary.json`
（状态、工具轨迹、指标、预算、工件）、`tool_results.json`、`pipeline/<name>/`
（推荐链路工件）。

## 2. 工具清单

注册表：`src/recsys/experiments/rec_tools.py`。

| 工具 | 真实执行 | 说明 |
| --- | --- | --- |
| `read_data_profile` | `prepare_dataset` | 用户/物品/切分规模、留出目标、特征列、排序候选、checkpoint |
| `query_model_zoo` | `models/zoo.toml` | 模型能力、依赖、入口、逐项状态 |
| `list_artifacts` | 扫描 `artifacts/<dataset>` | 已有 checkpoint / 运行工件 |
| `train_model` | 各训练入口 | 支持 16 个模型；默认 1 epoch / 2 batch，硬上限 3/20 |
| `recall_models` | `adapters.load_adapter` + `pipeline.recall` | 真实召回、真实分数、通道名次、候选落盘 |
| `rank_candidates` | `adapters.ranking_registry.load_ranker` + `pipeline.ranking` | 对当前召回结果真实排序 |
| `rerank_candidates` | `pipeline.rerank.diversify` | 类目/作者多样性重排 |
| `evaluate_recommendations` | `evaluation.evaluator` + `metrics` | 按 `split`、`ks` 计算真实指标 |
| `run_pipeline` | `pipeline.chain.RecommendationPipeline` | 一次跑完召回→融合→排序→重排→评估 |
| `evaluate_fusion_quota` | 既有 `experiments.fusion_quota` | 复用双塔/MIND 缓存候选的融合配额评测 |
| `search_experiment_history` | 既有 `experiment_history` | 复用历史实验查询 |
| `list_experiments` | 扫描 `experiments/` | 历史实验目录 |

参数校验在 `ToolRegistry.validate`：只接受 schema 里的枚举/数值，路径由程序从
`artifacts/<dataset>/` 约定推导，模型不能传命令或任意路径。

## 3. LangGraph 流程

```text
START
  → plan        (确定性计划 或 LLM tool_calls)
  → execute     (校验 → 记预算 → 调真实函数 → 记录结果/错误)
  → plan        (循环，直到计划结束 / LLM 不再调用工具 / 预算耗尽)
  → finish      (汇总状态、指标、工件、错误)
  → END
```

状态字段（`AgentState`）包含 `plan/plan_index/pending/messages/tool_results/
errors/budget/status/final_answer`。节点与事件都写入 `events.jsonl`。

### 预算与防护

`ToolBudget`：`max_tool_calls`、`max_train_calls`、`max_seconds`、`max_epochs`、
`max_batches`。默认 10 次工具、2 次训练、600 秒、3 epoch、20 batch。超限时
工具被拒绝并记录 `ok=false`，不会静默继续。

- 重复 `(tool, arguments)` 被拒绝；
- `train_model` 的 epochs/batches 有 schema 上限和预算双重检查；
- 工具异常统一捕获为 `{"ok": false, "error": "..."}`，Agent 在 `summary.errors`
  和最终回复里报告；
- 不把失败结果写成完成。

### 训练权重隔离

`train_model` 的每次调用都会创建唯一目录：

```text
<experiment_dir>/training/<model>-<UTC时间>-<uuid>/
```

所有训练入口（包括 LightGBM、RQ-VAE/TIGER 的关联工件）都写入该目录；
业务默认 `artifacts/<dataset>/` 权重不会被覆盖。训练成功后
`summary.trained_checkpoints` 记录本次实际路径，后续 `recall_models` /
`rank_candidates` / `run_pipeline` 优先使用它；训练失败不更新引用。
其他 Agent 任务仍按现有业务配置加载默认权重。

### 历史开关

`recall_models`、`rank_candidates`、`run_pipeline` 都接受
`include_validation_in_test_history`（默认 True）。test 下 True 使用训练+验证历史，
False 只用训练历史；validation 始终只用训练历史。CLI 对应
`--no-validation-in-test-history`。

## 4. 选择模型与组合方案

确定性计划器：

1. `--train a,b` → 依次 `train_model`；
2. `--recall x,y --ranker r` → `run_pipeline(models=[x,y], ranker=r)`；
3. 自由文本任务按关键词识别模型名与动作（训练/召回/排序/评估/查询）；
4. 无明确模型时先 `read_data_profile` + `query_model_zoo`，再用默认双路链路。

LLM 计划器（`--llm`）：

- system prompt 要求只通过工具执行、失败如实报告；
- 每轮把工具结果作为 `tool` message 回传；
- 一次只允许 1 个 `tool_call`；
- 参数先过注册表校验，失败时不执行。

没有配置 LLM 或 `--force-local` 时，`summary.planner = "local"`，明确区分。

## 5. 失败语义

| 场景 | 行为 |
| --- | --- |
| checkpoint 缺失 | 工具返回 `ok=false` + 缺失模型/期望路径；Agent 停止该步骤并报告 |
| 未知模型/非法参数 | `ToolCallError`，工具不执行 |
| 训练预算超限 | `ToolBudgetExceeded`，记录拒绝原因 |
| 模型前向异常 | 捕获为 `ok=false` + traceback 摘要 |
| 重复调用 | 拒绝并记录 |
| LLM 超时/HTTP 错误 | 本轮标记 partial，保留已执行工具结果 |

## 6. 验证

- `--self-check` 覆盖：真实数据画像、真实 Model Zoo、未知模型拒绝、真实
  Popularity 召回、缺 checkpoint 的明确失败、Popularity + LightGBM 真实链路，
  共 7/7（确定性计划器，不联网）。
- 真实 LLM 计划器已验证：`deepseek-chat` 一次工作流（4 次请求 / 4 次工具调用 /
  14.5 s）依次调用 `read_data_profile`、`query_model_zoo`、`run_pipeline`、
  `evaluate_recommendations`，状态 `verified`，指标与工件落盘。
- 单独工具链已验证：`recall_models(two_tower)` → `rank_candidates(din)` →
  `rerank_candidates` → `evaluate_recommendations(final)`，状态在 `ToolContext`
  中串联，4 个工具都返回 `ok=true`。
- 融合配额工具已验证：`evaluate_fusion_quota(50,50)` 复用既有缓存候选，
  返回 hits / Recall@100 / Coverage（0.43 s）。
- 真实训练工具已验证：`--train din` 与 `--train matrix_factorization` 都进入训练
  入口、写出 checkpoint，并立即被后续排序/链路工具使用。
- 自检不跑长训练；长训练只做受预算约束的冒烟。用 scripted chat 代替真实 LLM 的
  本地自检在 `summary.planner=local` 中标记。

## 7. 已知限制

- 确定性计划器只按关键词/CLI 选模型，不做复杂的多轮实验设计；真正的
  实验搜索留给后续专项。
- `run_pipeline` 默认最多 256 个用户，避免 Agent 单次任务过重。
- 训练工具是同步执行；时间预算只能在调用前后检查，无法中断正在进行的训练。
- 工具注册表只读 `models/zoo.toml` 的状态字段，不自动改写它。
