# Phase 5 三业务协作演示

- 任务 ID：`task-20261113T200727-d18758`
- 状态：**failed**（verified=False）
- 决策来源：`llm` / planner=`llm`
- 停止原因：`budget_exhausted`
- 任务记录：`7f90918052484c9085d6587fa7f7ec7b`

## 三业务结果

| 业务 | 状态 | 记录 ID | 内部工具 | 工件核对 | 失败步骤 |
| --- | --- | --- | --- | --- | --- |
| 推荐 | missing | `None` | - | 未通过 | dispatch：总控没有为该业务留下工具调用记录（预算不足或未分发） |
| 搜索 | missing | `None` | - | 未通过 | dispatch：总控没有为该业务留下工具调用记录（预算不足或未分发） |
| 广告 | missing | `None` | - | 未通过 | dispatch：总控没有为该业务留下工具调用记录（预算不足或未分发） |

## 预算（总控与子 Agent 共用）

- 上限：`{'max_llm_requests': 0, 'max_tool_calls': 14, 'max_experiments': 3, 'max_history_lookups': 0, 'max_artifact_reads': 0, 'max_seconds': 600.0}`
- 用量：`{'llm': 0, 'tool': 0, 'experiment': 0, 'history': 0, 'artifact': 0}`
- 剩余：`{'llm': 0, 'tool': 14, 'experiment': 3, 'history': 0, 'artifact': 0, 'seconds': 599.89}`

## 工件

- `rec_two_tower`（recommendation/recall_checkpoint）：`artifacts\kuasar_small\two_tower_features.pt` sha256=`784a2ad68972884d…`
- `rec_mind`（recommendation/recall_checkpoint）：`artifacts\kuasar_small\mind20.pt` sha256=`eba017182142a8d0…`
- `rec_lightgbm_ranker`（recommendation/ranking_checkpoint）：`artifacts\kuasar_small\mvp\lgbm_ranker.txt` sha256=`a79d9d451d383346…`
- `ads_din_weights`（ads/interface_test_weights）：`artifacts\kuasar_small\ads\interface_weights\din.pt` sha256=`3e1428d0ca67dfcb…`
- `ads_weight_index`（ads/weights_index）：`artifacts\kuasar_small\ads\interface_weights\index.json` sha256=`f156ddb88242a056…`
- `search_bm25_index`（search/keyword_index）：`artifacts\search\kuasar_bm25\index.json` sha256=`3ae2b858d456eab4…`
- `search_sample`（search/query_sample）：`artifacts\search\kuasar_bm25\sample.json` sha256=`12a46f3c3a614a71…`
- `search_bm25_manifest`（search/index_manifest）：`artifacts\search\kuasar_bm25\manifest.json` sha256=`eba233325c6554b9…`
- `search_vector_payload`（search/vector_index_payload）：`artifacts\search\kuasar_vector\vector\ann.faiss` sha256=`5d6983b148fc8b33…`
- `search_vector_doc_ids`（search/vector_index_metadata）：`artifacts\search\kuasar_vector\vector\doc_ids.json` sha256=`9a83c0384b1c54bc…`
- `search_vector_meta`（search/vector_index_metadata）：`artifacts\search\kuasar_vector\vector\meta.json` sha256=`28ff571006eedf9b…`
- `search_vector_manifest`（search/vector_index_manifest）：`artifacts\search\kuasar_vector\manifest.json` sha256=`a300e7abda8c9a59…`
- `rag_index`（platform/rag_index）：`artifacts\rag\phase4\index.json` sha256=`dff9c05db6f0854e…`

## 输出

- root: `experiments\2026-11-13-full-pipeline-agents-llm`
- task_dir: `experiments\2026-11-13-full-pipeline-agents-llm\tasks\task-20261113T200727-d18758`
- run_dir: `experiments\2026-11-13-full-pipeline-agents-llm\runs\task-20261113T200727-d18758`
- summary: `experiments\2026-11-13-full-pipeline-agents-llm\runs\task-20261113T200727-d18758\demo_summary.json`
- report: `experiments\2026-11-13-full-pipeline-agents-llm\runs\task-20261113T200727-d18758\demo_summary.md`
- log: `experiments\2026-11-13-full-pipeline-agents-llm\runs\task-20261113T200727-d18758\run.log`
- artifacts_used: `experiments\2026-11-13-full-pipeline-agents-llm\runs\task-20261113T200727-d18758\artifacts_used.json`
- shared_store: `experiments\2026-11-13-full-pipeline-agents-llm\shared\records.jsonl`

## 说明

- 决策由离线脚本替身给出（source=scripted_stand_in），业务工具是真实代码。
- 未完成业务：recommendation, search, ads；已完成的业务结果保留。
- 广告竞价金额与预算为模拟值，转化标签为 synthetic fixture，不代表真实效果。
