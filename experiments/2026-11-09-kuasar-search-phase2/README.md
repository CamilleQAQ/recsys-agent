# Phase 2 搜索系统小样本验收

- **验收入口**：`python -m recsys.search.acceptance`
- **结果**：`acceptance.json`（9/9 通过）
- **报告**：[report.md](report.md)
- **Agent 脚本替身轨迹**：`agent_scripted/scripted-hybrid_rerank/`
- **Agent 真实 LLM 轨迹**（1 次）：`agent_llm/run-20261109T134757Z-32924/`
- **索引工件**：`artifacts/search/kuasar_vector/`（Git 忽略，需先跑
  `python -m recsys.search.build_vector_index`）

```powershell
$env:PYTHONPATH = "src"
python -m recsys.search.build_vector_index --force
python -m recsys.search.acceptance
python -m recsys.experiments.search_agent --scripted-plan hybrid_rerank `
  --output-root experiments/2026-11-09-kuasar-search-phase2/agent_scripted
```

范围：120 物品 / 30 会话小样本链路验收；不训练、不微调、不重建 BM25、不建全量索引、
不扫原始 CSV。验收脚本不调用 LLM；另单独跑 1 次真实 `deepseek-chat` 工作流（3 请求）。
点击标签只用于核对，不参与打分。
