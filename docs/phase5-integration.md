# Phase 5：全项目集成与交付

本文记录 Phase 5 的实际交付：统一启动入口、演示配置、工件清单与恢复、严格「只加载」
路径、三业务协作演示、统一日志与缺工件失败用例。所有命令都在仓库根目录执行，
使用现有代码、现有权重和固定小样本，不训练、不扫全量、不重建大索引。

- 演示配置：[configs/platform/phase5_demo.toml](../configs/platform/phase5_demo.toml)
- 工件清单：[configs/platform/phase5_artifacts.toml](../configs/platform/phase5_artifacts.toml)
- 代码：`src/recsys/platform/phase5.py`、`src/recsys/platform/artifacts.py`、
  `src/recsys/utils/artifact_guard.py`

## 1. 统一入口

`python -m recsys.platform` 仍是唯一平台入口，原有三个业务模块 CLI 保持不变
（`python -m recsys.pipeline`、`python -m recsys.search`、`python -m recsys.platform --ads-*` 等）。

| 用途 | 命令 |
| --- | --- |
| 单独调用推荐 | `python -m recsys.platform --run-business recommendation` |
| 单独调用搜索 | `python -m recsys.platform --run-business search` |
| 单独调用广告 | `python -m recsys.platform --run-business ads` |
| 三业务协作演示 | `python -m recsys.platform --phase5-demo` |
| 独立进程工件恢复 | `python -m recsys.platform --phase5-restore` |
| 缺工件失败用例 | `python -m recsys.platform --phase5-missing` |
| 按任务 ID 查回结果 | `python -m recsys.platform --phase5-report --task-id <task_id>` |
| 原有 Research/Master 任务 | `python -m recsys.platform --task "..." --planner scripted` |

`--run-business` 与 `--phase5-demo` 都走同一套适配层（`platform/adapters.py`）：
共享实验库、统一预算、子 Agent 日志完全一致；单独调用只跑一个业务，演示跑三个。

## 2. 三业务演示（一条命令）

```powershell
$env:PYTHONPATH = "src"
python -m recsys.platform --phase5-demo
```

输出是统一摘要 JSON（同时写入文件）；退出码 `0=verified`、`2=blocked`、`1=其他`。

### 2.1 固定内容

| 项目 | 固定值 |
| --- | --- |
| 数据集 / 切分 | `kuasar_small` / `validation` |
| 推荐用户数 | 8（`select_users(seed=42)`，可复现） |
| 推荐召回 | `two_tower` + `mind` 既有 checkpoint |
| 推荐排序 | `lightgbm_ranker`，`artifacts/kuasar_small/mvp/lgbm_ranker.txt` |
| 推荐规模 | `k_recall=50`、`k_final=10`、`round_robin` 融合、类目上限 5 |
| 搜索查询 | 固定会话 `11208`，token `[3538762, 3181091, 4061567, 3525929]` |
| 搜索链路 | `search_hybrid`（BM25 + FAISS + RRF）→ `search_rerank` |
| 广告模型 | `din` 接口测试权重（`interface_test`，未训练） |
| 广告规模 | `test` 前 200 行做 CTR 预测；`train` 2 个请求做排序与二价竞价 |
| 决策来源 | `scripted_stand_in`（离线计划替身，逐条标注；工具是真实代码） |

### 2.2 预算（总控分发 + 子 Agent 内部工具）

全局预算写在演示配置里，不改 `configs/platform/research_agent.toml` 的默认值。

| 项 | 上限 | 实际 |
| --- | --- | --- |
| `max_experiments` | 3 | 3（每个业务 1 次） |
| `max_tool_calls` | 14 | 12（推荐 3 + 搜索 3 + 广告 6，含 3 次分发） |
| `max_llm_requests` | 0 | 0 |
| `max_history_lookups` | 0 | 0（`history_first=false`） |
| `max_seconds` | 600 | 约 11 s |

子 Agent 上限：推荐 5、搜索 4、广告 7 次工具；`TaskBudget.child()` 只会更小，
不可能重置或绕过总额度。额度不足时保留已完成业务并返回 `partial`。

### 2.3 状态语义

- `verified`：三路都是 `success`、三条业务记录都在、每路「实际使用的工件」与清单一致、
  推荐记录确认 `auto_train_disabled=true`；
- `partial`：至少一路成功、其余失败/预算不足（已完成结果保留）；
- `failed`：三路都没有成功；
- `blocked`：工件清单先校验失败，演示不启动任何业务。

## 3. 工件清单与恢复

### 3.1 清单内容

`configs/platform/phase5_artifacts.toml` 登记 13 个文件，覆盖四个部分：

- 推荐：`rec_two_tower`、`rec_mind`、`rec_lightgbm_ranker`
- 广告：`ads_din_weights`、`ads_weight_index`
- 搜索：BM25 索引/样本/清单、FAISS 载荷/`doc_ids`/`meta`/向量清单
- 平台：`rag_index`

每条包含 `business`、`role`、`path`、`sha256`、`size_bytes`、`source`、
`checksum_recorded_in`（已有校验出处）和 `metadata`（可机检的结构字段）。
校验时除存在性、大小和 sha256 外，还会复核：

- BM25 的 `num_docs=120`、`vocab_size=823`、`k1`、`b`；
- 样本会话数 30；
- FAISS 载荷哈希与向量清单 `index.payload_sha256`、`num_items=120`、`dim=4096`；
- DIN 权重哈希与 `interface_weights/index.json` 记录一致；
- LightGBM 权重可解析（20 棵树、9 个特征）；
- RAG 索引 `files=19 / chunks=308 / vocab=8049`。

任一条不满足都会报出具体文件并失败。

### 3.2 独立进程恢复

```powershell
python -m recsys.platform --phase5-restore
```

新进程从清单登记路径加载并各跑一次必要推理，再与清单逐项比对：

| 检查 | 内容 |
| --- | --- |
| `manifest` | 13 个文件全部通过存在性/大小/sha256/metadata |
| `recommendation` | 2 用户跑完整链路；`run_meta.checkpoints` 的 two_tower / mind / ranker sha256 与清单一致 |
| `search` | 按清单路径加载 BM25 + FAISS，跑 1 次混合检索 + 1 次精排；索引 sha256 与清单一致 |
| `ads` | `din` 权重路径/sha256 + `weight_kind=interface_test`；真实 CTR 推理 32 行 |
| `rag` | 检索返回可定位引用，且 `path:start-end` 能读回原文 |

结果写入 `experiments/2026-11-10-kuasar_small-platform-phase5/demo/restore/<restore-id>/restore.json`。

## 4. 只加载，不训练（LightGBM 重点修复）

原来的 `run_pipeline` 在找不到 LightGBM 排序权重时会**自动训练**到任务目录。Phase 5 增加
显式的只加载策略：

- `recsys/pipeline/ranking.py`：`train_if_missing=False` 且权重缺失时抛
  `ArtifactError`，错误信息包含完整路径；训练只在 `train_if_missing=True` 时发生。
- `recsys/pipeline/chain.py`：召回权重缺失时同样抛 `ArtifactError`，不再尝试别的模型。
- `recsys/experiments/rec_tools.py`：`run_pipeline` 新增
  `artifact_policy=existing_only|train_if_missing`、`ranker_checkpoint`、
  `expected_ranker_sha256`。`existing_only` 时在训练代码之前就返回
  `ok=false` + `missing_artifact` + `auto_train_attempted=false`。
- `recsys/platform/adapters.py`：推荐业务默认使用 `existing_only`，并把
  `run_meta.json#checkpoints` 里的真实权重路径/sha256 写进共享实验记录。
- 原有显式训练入口保留：`python -m recsys.experiments.rec_agent --train lightgbm_ranker`
  以及 `run_pipeline` 的 `artifact_policy=train_if_missing`（默认值，行为不变）。

演示配置同时固定 `ranker_checkpoint` 与 `expected_ranker_sha256`，权重被替换会直接失败。

## 5. 缺工件失败用例

```powershell
python -m recsys.platform --phase5-missing
```

5 个小用例，全部要求「明确失败 + 无副作用」：

1. 推荐：`existing_only` + 指定的缺失排序权重 → 失败信息含该路径，不训练；
2. 推荐：默认候选路径也不存在 → 失败信息列出所有检查过的路径；
3. 搜索：BM25 索引缺失 → Search Agent `blocked`，不重建索引；
4. 广告：权重目录为空 → `unavailable` + 明确提示，不构建新权重；
5. RAG：索引目录缺失 → `ok=false`，不重建索引。

用例前后会对权重、索引和 `artifacts/phase5_missing_do_not_create/` 做快照对比，
并扫描 `artifacts/` 是否出现新文件；全部为空才算通过。

## 6. 统一日志与结果

一次运行只有一个任务 ID，贯穿总控、三个子 Agent、共享实验记录、预算和工件来源：

```
experiments/2026-11-10-kuasar_small-platform-phase5/demo/
├── tasks/<task_id>/                  # 总控 + 三个子 Agent 的原始记录
│   ├── trace.jsonl                   # 节点/决策/工具/预算事件
│   ├── decisions.json                # 每步决策（scripted_stand_in）
│   ├── summary.json / tool_results.json / final_answer.md
│   ├── business_recommendation/      # Rec Agent: events.jsonl / pipeline/<run>/run_meta.json
│   ├── business_search/              # Search Agent: trace.json / metrics.json / run_meta.json
│   └── business_ads/                 # Ads Agent: trace.json / metrics.json / run_meta.json
├── runs/<task_id>/
│   ├── demo_summary.json             # 统一摘要（状态/工具/预算/工件/输出路径）
│   ├── demo_summary.md               # 同一内容的人类可读版
│   ├── artifacts_used.json           # 三路实际使用工件与清单核对明细
│   └── run.log                       # 统一文本日志（不含密钥/环境变量）
└── shared/records.jsonl              # 同一 task_id 下 3 条业务记录 + 1 条任务记录
```

> 演示之外单独执行 `--run-business` 会写入同一个共享库、但使用自己的 `task_id`；
> 本次另有一条 `--run-business search` 记录（`task-20261110T131120-4761d9`，
> `80ce1ccda0ac4db3aba1862a42a42976`），按 `task_id` 过滤即可与演示分开。

`--phase5-report --task-id <id>` 在另一个进程里查回统一摘要、三条业务记录及其工件路径，
不重跑任何实验。

## 7. 本次实际运行记录

| 项 | 值 |
| --- | --- |
| 演示任务 ID | `task-20261110T131020-3a865c` |
| 演示状态 | `verified`（planner=`scripted`，decision_source=`scripted_stand_in`） |
| 推荐记录 | `f09030a13a8042008bcd1b432ccd3bd4`（success，2 个工具） |
| 搜索记录 | `36490db7c7b04decb15ece3af47aa6c3`（success，2 个工具） |
| 广告记录 | `29dc0f764d5c469da4df4a654e3530f6`（success，5 个工具） |
| 任务记录 | `43469f61bae744f2afe77cb5520e4f66` |
| 恢复校验 | `restore-20261110T051036Z`（新进程 pid 31820，5/5 通过） |
| 缺工件用例 | `missing-20261110T051053Z`（5/5 通过，无新工件） |
| 单业务入口 | `--run-business search` → `task-20261110T131120-4761d9`（success，记录 `80ce1ccda0ac4db3aba1862a42a42976`） |

证据路径：

- `experiments/2026-11-10-kuasar_small-platform-phase5/demo/runs/task-20261110T131020-3a865c/demo_summary.json`
- `experiments/2026-11-10-kuasar_small-platform-phase5/demo/runs/task-20261110T131020-3a865c/demo_summary.md`
- `experiments/2026-11-10-kuasar_small-platform-phase5/demo/runs/task-20261110T131020-3a865c/artifacts_used.json`
- `experiments/2026-11-10-kuasar_small-platform-phase5/demo/restore/restore-20261110T051036Z/restore.json`
- `experiments/2026-11-10-kuasar_small-platform-phase5/demo/missing/missing-20261110T051053Z/missing_check.json`

小样本指标（仅链路检查，不作效果结论）：推荐 `k_final=10` 各阶段 Recall/NDCG/HitRate 为 0
（8 用户、已知数据规模问题，见 `docs/top20-diagnosis.md`）、Coverage 0.000176，4 项链路断言
（无重复、无曝光泄漏、排序/重排在候选池内）通过；搜索混合检索与精排各返回 5 条、
17/17 检查通过；广告 `din` 在 test 前 200 行上 AUC 0.5833 / LogLoss 0.8322，
2 个请求全部成交、模拟扣费 2.1672、总预算剩余 17.8328。

## 8. 已知边界（仍未验证）

1. **独立环境**：TensorFlow Recommenders（TFRS Two-Tower）、TorchEasyRec 原实现、
   OneRec 依赖 Linux / 未安装的 `verl`/`vllm`/`ray`/`transformers`，本机 Python 3.14
   无法运行；这些模型入口保持源码 + 适配说明，不在 Phase 5 演示范围内。
2. **真实 LLM 规划**：Phase 5 演示默认离线脚本替身；真实 LLM 只做过 Phase 4 的单次任务，
   未验证三业务多轮规划与重试。可用 `--phase5-planner llm` 显式开启（需 `.env` 配置
   `DSH_AGENT_LLM_*`，并把演示配置的 `[budget] max_llm_requests` 改为大于 0；会产生真实请求费用）。
3. **广告数据边界**：KuaiSAR 没有真实出价、预算和转化标签；竞价金额为模拟配置，
   转化标签为 synthetic fixture，CVR 不作为真实效果。DIN 权重是固定初始化
   `interface_test`，不是训练模型。
4. **搜索语义**：KuaiSAR 只有匿名整数 token，`token_hash` 编码不是训练好的语义向量；
   搜索只支持数据集内的 token 查询，不承诺任意自然语言搜索。
5. **推荐效果**：8 用户小样本 Top-10 命中为 0，属于既有数据规模/长尾问题，
   不是 Phase 5 的集成失败；效果专项另开。
6. **并发/断网**：多进程同时跑演示、LLM 断网回退、超长任务未在 Phase 5 覆盖。
