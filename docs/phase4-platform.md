# Phase 4：统一 Agent 平台、共享实验平台与 RAG

本文件记录 Phase 4 的实际交付、调用入口、关键代码路径、验收结果和未验证边界。
对应 `PROJECT_MASTER_v2.md` §7 Phase 4，不做模型训练、不重建大索引、不跑全量。

## 1. 交付范围

- **Research Agent / Master Agent**：`recsys.platform.research_agent`，Phase 4 统一入口，
  同时承担总控调度；沿用项目现有 LangGraph 与 `recsys.experiments.llm_tool_call` 客户端。
- **三个业务入口薄适配**：推荐 `recsys.experiments.rec_agent`、搜索
  `recsys.experiments.search_agent`、广告 `recsys.ads.agent`；原 CLI 与单独运行方式不变。
- **共享实验平台**：推荐历史继续读 Phase 11 JSON 与实验记忆索引；新增推荐/搜索/广告
  统一 JSONL 记录、跨业务历史查询、复用校验、模型注册视图和跨进程写入锁。
- **统一任务预算**：一个研究任务共享 LLM 请求、工具调用、实际实验、历史查询、结果读取
  和总时限；总控与子 Agent 从同一预算扣减，子 Agent 上限只能更小。
- **本地 RAG**：独立索引目录 `artifacts/rag/phase4`；解析、索引、检索、引用、增量更新；
  Research Agent 与三个业务 Agent 都可调用 `retrieve_knowledge`。

## 2. 入口与关键代码路径

| 能力 | 路径 |
| --- | --- |
| 统一 CLI | `python -m recsys.platform`（`src/recsys/platform/__main__.py`） |
| Research / Master Agent | `src/recsys/platform/research_agent.py` |
| 三个业务薄适配 | `src/recsys/platform/adapters.py` |
| 统一预算 | `src/recsys/platform/budget.py` |
| 共享实验记录与历史查询 | `src/recsys/platform/shared_experiments.py` |
| RAG 索引、检索、引用 | `src/recsys/platform/rag.py` |
| 跨进程文件锁 | `src/recsys/platform/locks.py` |
| 小样本验收与自检 | `src/recsys/platform/acceptance.py` |
| 平台配置 | `configs/platform/research_agent.toml`、`configs/platform/rag.toml` |
| Rec Agent RAG 工具 | `src/recsys/experiments/rec_tools.py`（`retrieve_knowledge`） |
| Search Agent RAG 工具 | `src/recsys/experiments/search_agent.py`（可选 `[knowledge]`） |
| Ads Agent RAG 工具 | `src/recsys/ads/tools.py`（`retrieve_knowledge`） |

## 3. 统一 CLI

```powershell
$env:PYTHONPATH = "src"

# 自然语言任务（有 LLM 配置时 auto 走真实 LLM；否则脚本替身并明确标注）
python -m recsys.platform --task "查一下已有 30/70 融合配额推荐实验，能复用就复用"
python -m recsys.platform --task "推荐和搜索一起做" --planner scripted

# 历史查询（可在另一个进程执行）
python -m recsys.platform --history-query --quota 30,70
python -m recsys.platform --history-query --task-id task-...
python -m recsys.platform --read-record <record_id>   # 只读已有结果，计入 artifact 次数

# 三个业务模型注册与版本
python -m recsys.platform --registry

# 本地 RAG
python -m recsys.platform --rag-build
python -m recsys.platform --rag-update
python -m recsys.platform --rag-query "实验记忆 指纹 复用"

# 验收 / 自检
python -m recsys.platform --self-check
python -m recsys.platform --acceptance
python -m recsys.platform --acceptance --with-llm   # 只跑一次真实 LLM
python -m recsys.platform --acceptance --reuse-llm-result <已运行 acceptance.json>
```

任务产物默认写到
`experiments/2026-11-10-kuasar_small-platform-phase4/tasks/<task_id>/`：
`summary.json`、`trace.jsonl`、`tool_results.json`、`decisions.json`、`final_answer.md`。

## 4. Research Agent 决策与历史优先

### 4.1 历史优先

研究类任务（关键词或配额命中）先由策略执行一次
`search_experiment_history`，而不是让模型凭文本猜：

- 推荐历史：`lookup_history` 调用现有 `recsys.experiments.experiment_history`，
  继续校验缓存指纹、目标配额、候选预算、融合规则和工件；
- 共享库：校验状态、配置、数据/切分、模型来源和工件路径；
- **命中且可复用**：直接结束，`stop_reason=history_hit`，新实验次数为 0；
- 未命中/不匹配/工件失效：把真实 miss 原因和剩余预算交给 LLM（或离线脚本替身）
  选择下一步，允许的下一步包括 RAG、一个业务小样本检查、再次查历史或停止。

### 4.2 工具

Research Agent 注册 7 个工具：

1. `search_experiment_history`（跨业务历史；推荐配额仍走旧校验）
2. `read_experiment_result`（按 `record_id` 读已有结果）
3. `query_model_registry`（三个业务模型注册）
4. `retrieve_knowledge`（本地 RAG）
5. `run_recommendation_experiment`（Rec Agent）
6. `run_search_task`（Search Agent）
7. `run_ads_experiment`（Ads Agent）

一次模型响应只执行一个 `tool_call`；工具参数统一做类型/enum/范围校验，
相同工具+参数拒绝重复执行。

### 4.3 决策来源与状态

每个决策记录在 `decisions.json` 和 `trace.jsonl`，包含：

- `source`：`policy:history_first` / `llm` / `scripted_stand_in`；
- 模型可见状态：history 命中或 miss 原因、已有工具结果、剩余预算、任务目标；
- 选中的工具与参数，或停止原因；
- 工具返回摘要、耗时和预算变化。

状态按实际结果标记：`verified`、`partial`、`failed`、`blocked`。
预算用尽、超时、工具失败、应执行工具却零调用（`no_tool_call`）都不会被写成成功，
已完成的结果保留在 `tool_results.json`。

### 4.4 离线替身

离线验收只用替身做“决策”，工具仍执行真实业务代码；每条记录和汇总都会带
`decision_source=scripted_stand_in` / `scripted_decision_driver`，不会冒充 LLM 决策。

## 5. 共享实验平台

### 5.1 记录格式

共享库默认 `experiments/2026-11-10-kuasar_small-platform-phase4/shared/records.jsonl`，
每行一条 JSON 记录（只追加、不原地改写；更新用同一 `record_id` 的新 `revision`）：

```json
{
  "record_id": "...",
  "revision": 1,
  "task_id": "task-...",
  "business": "recommendation|search|ads|task",
  "status": "success|partial|failed|blocked",
  "config": {"task": "...", "split": "...", "users": 4},
  "data": {"dataset": "kuasar_small", "split": "validation", "split_role": "validation"},
  "model": {"models": ["popularity"], "weights_source": "existing_checkpoints"},
  "result": {"metrics": {}, "summary": "..."},
  "budget": {"counters": {"llm": 0, "tool": 4, "experiment": 1, "history": 1}},
  "artifacts": {"summary": "...", "trace": "..."},
  "evidence": {
    "validated": true,
    "reusable": true,
    "execution_mode": "real_tools",
    "decision_driver": "local|scripted|llm",
    "data_kind": "real_data|real_ads_data",
    "simulation": {"auction_money": true, "conversion_labels": "synthetic_fixture"},
    "artifact_checks": [{"name": "summary", "path": "...", "exists": true, "size": 123}],
    "warnings": []
  }
}
```

### 5.2 复用规则

只有以下条件全部满足才允许作为成功证据复用：

- `status=success`；
- `evidence.validated=true`、`evidence.reusable=true`；
- `execution_mode=real_tools`；
- 业务/数据/切分/模型/配置哈希与请求匹配；
- 记录中的工件仍存在且大小一致。

失败和部分完成也入库供排查，但查询 `--reusable` 时返回拒绝原因。验证集/测试集用
`split_role` 区分；真实工具与离线替身用 `decision_driver` 区分；广告金额与转化标签在
`simulation` 中单独标记，不当作真实出价/CVR。

### 5.3 历史查询

`query_records` 支持 `business`、`task_id`、`dataset`、`split`、`split_role`、`model`、
`status`、`driver`、`reusable`、文本筛选；返回匹配记录和拒绝原因。
推荐配额查询继续复用旧 `search_experiment_history`，不复制旧记录格式。

### 5.4 并发与跨进程

- JSONL 追加前对 `records.jsonl.lock` 加跨进程锁（Windows `msvcrt`，POSIX `fcntl`）；
- 读方跳过并发写入产生的半行，并在查询结果里报告 `bad_lines`；
- 验收用两个进程各追加 3 条，6 条全部可查、0 坏行。

### 5.5 模型注册

`python -m recsys.platform --registry` 汇总：

- 推荐：`models/zoo.toml` 的 26 个模型、类别、完整度、adapter 和 checkpoint 是否存在；
- 搜索：BM25 文档数/词表、FAISS 向量维度/后端版本、四个检索工具；
- 广告：六个模型、框架、任务、权重类型与 `interface_test` 状态。

## 6. 统一预算

`TaskBudget` 维护六类额度：

| 类型 | 含义 |
| --- | --- |
| `llm` | 真实 LLM 请求次数（脚本替身不计） |
| `tool` | 工具调用次数（总控工具 + 子 Agent 内部工具） |
| `experiment` | 真正执行的新实验次数 |
| `history` | 历史查询次数 |
| `artifact` | 已有结果读取次数 |
| `seconds` | 总时限 |

子 Agent 通过 `budget.child(name, policy)` 获得 `BudgetScope`，其上限在创建时被收紧到
父预算当前剩余额度；每次 `spend` 同时扣减父预算和子预算。启动子任务不会重置总额度，
额度不足时在分发前拒绝并记录 `budget_refusals`。

广告竞价金额仍由 Ads Agent 自己的模拟预算记账（`simulate_auction` 的
`budget.spent/remaining`），不纳入研究任务的 `llm/tool/experiment` 计数；
共享记录里以 `evidence.simulation` 单独标注。

## 7. 本地 RAG

- 索引目录：`artifacts/rag/phase4`，与推荐物品向量索引、搜索 FAISS 索引分开；
- 资料：项目 README、`PROJECT_MASTER_v2.md`、架构/三个业务 Agent 文档、
  Model Zoo 与模型说明、`third_party/SOURCES.toml`、历史实验报告（含失败/回归记录）；
  `docs/phase4/` 目录为增量资料入口；
- 切块：Markdown 按标题切分，其他文件按行窗口；每块保留
  `path + start_line + end_line + heading`；
- 检索：ASCII 词 + 中文二元组 BM25；返回 `citation=path:start-end`、片段和分数；
- 更新：按文件 sha256 增量重建，只重切内容变化的文件，未变文件复用旧片段；
- 工具：`retrieve_knowledge(query, top_k)`；Research Agent 与 Rec/Search/Ads Agent
  都注册并实际调用过；Search Agent 默认不启用，配置 `[knowledge] enabled=true` 时
  在首轮检索前可选调用一次（原有四工具、预算和检查不变）。

## 8. 实际验收

### 8.1 命令

```powershell
$env:PYTHONPATH = "src"
python -m recsys.platform --self-check
python -m recsys.platform --acceptance
python -m recsys.platform --acceptance --with-llm   # 本次实际只运行一次
```

最终汇总：

- 验收报告：`experiments/2026-11-10-kuasar_small-platform-phase4/acceptance-final/report.md`
- 验收 JSON：`.../acceptance-final/acceptance.json`
- 真实 LLM 原始运行：`.../acceptance-20261110T034618Z/acceptance.json`
  （`acceptance-final` 复用这份真实运行记录，没有再次调用模型）
- 共享实验库：`.../acceptance-final/shared/records.jsonl`

### 8.2 结果

| 验收项 | 结果 | 证据 |
| --- | --- | --- |
| 1. 已有 30/70 推荐实验历史命中 | PASS | `status=hit`、`source=phase11_json`、命中 5、新实验 0、`stop_reason=history_hit` |
| 2. 50/50 未命中后补一次小样本检查 | PASS | 历史真实 miss；调用一次推荐业务；`new_experiments=1`；记录见 `acceptance.json` 的 `2_history_miss_and_small_check` |
| 2c. 读取已有结果单独计数 | PASS | 只调用 `read_experiment_result`；`artifact=1`、`experiment=0`，任务 verified |
| 2b. 全局预算耗尽 | PASS | `stop_reason=budget_exhausted`、新实验 0、`budget_refusals` 有记录；直接用 0 实验父预算调用子 Agent 也被拒绝 |
| 3. 三业务入口 + 跨业务 | PASS | 跨业务任务同一 `task_id` 下 recommendation + search 两条记录；广告任务 verified |
| 3b. 共享实验平台 | PASS | 校验记录可复用；partial 记录入库但拒绝复用；split 不匹配返回原因；广告模拟单独标记 |
| 3c. 模型注册 | PASS | 26 推荐模型、4 搜索工具、6 广告模型 |
| 3d. 另起进程查记录 | PASS | 子进程按 `task_id` 查回 2 条同一任务记录 |
| 3e. RAG 引用定位 | PASS | `PROJECT_MASTER_v2.md:616-628` 可读回原文 |
| 3f. RAG 资料更新 | PASS | 写入 `docs/phase4/rag-update-note.md` 新内容后增量更新，新标记可检索，旧资料仍可检索 |
| 3g. 三业务 RAG 工具 | PASS | Rec / Ads 工具返回 ok；Search 脚本替身 `knowledge_bm25` 通过全部检查，1 次知识检索 + BM25 结果 |
| 3h. 并发写入 | PASS | 两进程各 3 条，6 条全部可查，0 坏行 |
| 4. 真实 LLM 任务 | PASS（一次） | 4 个决策：历史策略查 50/50 → miss → LLM 再查 → 调推荐小样本 → 停止；`llm_requests=3`、`tool=5`、`experiment=1`、`history=2`；状态 verified |

真实 LLM 任务
`task-20261110T114653-322c0b` 的决策轨迹保存在
`experiments/2026-11-10-kuasar_small-platform-phase4/acceptance-20261110T034618Z/real_llm/tasks/task-20261110T114653-322c0b/trace.jsonl`。
模型先看到 50/50 历史 miss，再额外查一次历史，然后选择 `run_recommendation_experiment`
（8 用户、two_tower + mind、LightGBM），最后停止并明确说明指标为 0 是小样本链路检查、
不能当作效果结论。

## 9. 尚未验证 / 明确边界

- **不训练、不调参、不跑全量、不重建大索引**：Phase 4 验收只做小样本链路检查；
  推荐指标为 0 属于小样本规模问题，不作为模型效果结论。
- **广告真实出价/预算/转化**：KuaiSAR 没有真实出价、预算和转化标签；
  竞价金额为配置模拟值，CVR 使用独立 synthetic fixture，已在记录和报告中标记。
- **真实 LLM 只运行一次**：没有做多轮/多场景 LLM 规划对比；真实 LLM 轨迹是单次实际记录。
- **检索语义能力**：KuaiSAR 查询是匿名整数 token；RAG 使用本地文本 BM25，
  搜索 Agent 的 `token_hash` 编码不是训练好的语义向量。
- **业务 Agent 的 RAG 入口**：Rec/Ads 默认注册工具；Search 只有显式配置
  `[knowledge] enabled=true` 时才暴露，默认行为与 Phase 2/3 完全一致。

## 10. 相关记录

- `experiments/2026-11-10-kuasar_small-platform-phase4/self_check/self_check.json`
- `experiments/2026-11-10-kuasar_small-platform-phase4/acceptance-final/acceptance.json`
- `experiments/2026-11-10-kuasar_small-platform-phase4/acceptance-20261110T034618Z/acceptance.json`
- `artifacts/rag/phase4/index.json`
- `artifacts/rag/phase4/.rag.lock`（并发更新锁）
