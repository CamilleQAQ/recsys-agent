# Phase 4 小样本验收报告

- 生成时间：20261110T034618Z
- 总体状态：**通过**
- Python：3.14.3
- 真实 LLM：可用
- 共享实验库：`experiments\2026-11-10-kuasar_small-platform-phase4\acceptance-20261110T034618Z\shared\records.jsonl`
- RAG 索引：`artifacts/rag/phase4`

## 验收项

| # | 项目 | required | 结果 | 耗时(s) | 说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1_history_hit | True | PASS | 0.28 | hit |
| 2 | 2_history_miss_and_small_check | True | PASS | 7.78 | miss |
| 3 | 2b_budget_exhaustion | True | PASS | 0.44 | budget_exhausted |
| 4 | 3_three_business_entries | True | PASS | 16.61 |  |
| 5 | 3b_cross_process_records | True | PASS | 4.41 |  |
| 6 | 3c_rag_citation | True | PASS | 0.23 |  |
| 7 | 3d_rag_update | True | PASS | 0.23 |  |
| 8 | 3e_rag_business_agents | True | PASS | 0.35 |  |
| 9 | 3f_concurrent_writes | True | PASS | 4.66 | 两个进程各追加 3 条，JSONL 锁保证互不覆盖；记录只追加不改写。 |
| 10 | 4_real_llm_once | True | PASS | 17.38 | 只运行一次真实 LLM 研究任务；其余分支用离线替身。 |

## 关键证据

### 1_history_hit
- task_id: `task-20261110T114618-1a8023`
- new_experiments: `0`
- command: `python -m recsys.platform --task "查一下已有 30/70 融合配额推荐实验..." --planner scripted`

### 2_history_miss_and_small_check
- task_id: `task-20261110T114619-13efcb`
- record_ids: `['033c90908c8b4c17b61745b5aa8a93ee']`
- decision_count: `3`
- new_experiments: `1`
- offline_trace: `experiments\2026-11-10-kuasar_small-platform-phase4\acceptance-20261110T034618Z\history_miss\tasks\task-20261110T114619-13efcb\trace.jsonl`

### 2b_budget_exhaustion
- task_id: `task-20261110T114626-65cf8e`
- new_experiments: `0`
- subagent_blocked: `True`

### 3_three_business_entries
- cross_task_id: `task-20261110T114627-5fabfc`
- ads_task_id: `task-20261110T114634-b6e43e`
- cross_record_ids: `['49a395f2b4ea4377aaa2178877b6ad2e', '6c42a1363e424c6e907638389af2a6d1']`

### 3b_cross_process_records
- command: `python -m recsys.platform --history-query --task-id task-20261110T114627-5fabfc --shared-store experiments\2026-11-10-kuasar_small-platform-phase4\acceptance-20261110T034618Z\shared`

### 3c_rag_citation
- citation: `PROJECT_MASTER_v2.md:616-628`
- command: `python -m recsys.platform --rag-query "实验记忆 指纹 复用"`

### 3d_rag_update
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
