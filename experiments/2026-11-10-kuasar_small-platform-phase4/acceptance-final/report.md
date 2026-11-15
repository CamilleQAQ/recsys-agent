# Phase 4 小样本验收报告

- 生成时间：20261110T040035Z
- 总体状态：**通过**
- Python：3.14.3
- 真实 LLM：可用
- 共享实验库：`experiments\2026-11-10-kuasar_small-platform-phase4\acceptance-final\shared\records.jsonl`
- RAG 索引：`artifacts/rag/phase4`

## 验收项

| # | 项目 | required | 结果 | 耗时(s) | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1_history_hit | True | PASS | 0.33 | hit |
| 2 | 2_history_miss_and_small_check | True | PASS | 5.11 | miss |
| 3 | 2c_artifact_read_counted | True | PASS | 0.05 | verified |
| 4 | 2b_budget_exhaustion | True | PASS | 0.21 | budget_exhausted |
| 5 | 3_three_business_entries | True | PASS | 7.59 |  |
| 6 | 3b_shared_experiment_platform | True | PASS | 0.06 |  |
| 7 | 3c_model_registry | True | PASS | 0.02 |  |
| 8 | 3d_cross_process_records | True | PASS | 4.74 |  |
| 9 | 3e_rag_citation | True | PASS | 0.1 |  |
| 10 | 3f_rag_update | True | PASS | 0.09 |  |
| 11 | 3g_rag_business_agents | True | PASS | 0.14 |  |
| 12 | 3h_concurrent_writes | True | PASS | 4.69 | 两个进程各追加 3 条，JSONL 锁保证互不覆盖；记录只追加不改写。 |
| 13 | 4_real_llm_once | False | PASS | 0.0 | 只运行一次真实 LLM 研究任务；其余分支用离线替身。 |

## 关键证据

### 1_history_hit
- task_id: `task-20261110T120035-ba58fb`
- new_experiments: `0`
- command: `python -m recsys.platform --task "查一下已有 30/70 融合配额推荐实验..." --planner scripted`

### 2_history_miss_and_small_check
- task_id: `task-20261110T120035-afab6d`
- record_ids: `['5246988b1cfc4672a24df63814e94884']`
- decision_count: `3`
- new_experiments: `1`
- offline_trace: `experiments\2026-11-10-kuasar_small-platform-phase4\acceptance-final\history_miss\tasks\task-20261110T120035-afab6d\trace.jsonl`

### 2c_artifact_read_counted
- task_id: `task-20261110T120040-72c803`

### 2b_budget_exhaustion
- task_id: `task-20261110T120040-2bcb5b`
- new_experiments: `0`
- subagent_blocked: `True`

### 3_three_business_entries
- cross_task_id: `task-20261110T120040-33701f`
- ads_task_id: `task-20261110T120044-befb39`
- cross_record_ids: `['a1db6a3d2d774a65bf1adb8d6b75b46c', '09313ea391914c0183c02c93d963c44f']`

### 3d_cross_process_records
- command: `python -m recsys.platform --history-query --task-id task-20261110T120040-33701f --shared-store experiments\2026-11-10-kuasar_small-platform-phase4\acceptance-final\shared`

### 3e_rag_citation
- citation: `experiments\2026-11-08-kuasar_small-experiment-memory\report.md:1-2`
- command: `python -m recsys.platform --rag-query "实验记忆 指纹 复用"`

### 3f_rag_update
- citation: `docs\phase4\rag-update-note.md:1-5`
- command: `python -m recsys.platform --rag-update`

### 4_real_llm_once
- task_id: `task-20261110T114653-322c0b`
- decision_count: `4`
- trace: `experiments\2026-11-10-kuasar_small-platform-phase4\acceptance-20261110T034618Z\real_llm\tasks\task-20261110T114653-322c0b\trace.jsonl`

## 尚未验证/明确边界

- 完整 LLM 多轮规划（只按验收要求跑一次真实任务）
- 大规模训练/全量/调参（Phase 4 明确不做）
- 广告真实出价、预算与转化标签（KuaiSAR 数据缺口，使用模拟并单独标记）

## 复现命令

```powershell
$env:PYTHONPATH = "src"
python -m recsys.platform --acceptance
python -m recsys.platform --acceptance --with-llm   # 需要 DSH_AGENT_LLM_* 配置
```
