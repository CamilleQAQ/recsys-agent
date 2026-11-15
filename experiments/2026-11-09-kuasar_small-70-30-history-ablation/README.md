# 2026-11-09 · Phase 13 `70/30` 历史信息消融

本目录是历史信息消融运行容器。每次执行默认新建 `run-<UTC>-<pid>/`，
不覆盖旧运行。对照组只看 `50/50`；历史组额外看到 `70/30`。`70/30` 在两组
都不可查询。

## 本次正式运行

- report.md（本地文件：`run-20261109T045034Z-15408/report.md`）
- metrics.json（本地文件：`run-20261109T045034Z-15408/metrics.json`）
- frozen/manifest.json（本地文件：`run-20261109T045034Z-15408/frozen/manifest.json`）
- 场景 A LLM 轨迹（本地文件：`run-20261109T045034Z-15408/scenarios/A/llm_trace.json`）
- 场景 B LLM 轨迹（本地文件：`run-20261109T045034Z-15408/scenarios/B/llm_trace.json`）
- self_check.json（本地文件：`run-20261109T045034Z-15408/self_check.json`）

## 结果

- **对照组（无 70/30）**：真实 LLM 首选 `30/70`，随后查 `80/20`；最佳
  `30/70`（命中 5 / Recall@100 0.013672），2 次揭晓 / 3 次请求 / 4.801 s。
- **历史组（可见 70/30）**：真实 LLM 首选 `30/70` 后主动停止；最佳同为
  `30/70`，1 次揭晓 / 2 次请求 / 3.674 s。
- **消融结论**：两组最终最佳相同，历史组少揭晓 1 次，按规则记为该场景
  节省 1 次查询。本地规则 Agent 与固定策略在两组均做出相同选择
  （`30/70 → 20/80` 与 `30/70 → 80/20`），未观察到历史带来的行为差异。
- 历史组最终回复明确引用了 `70/30`（命中 3 / Recall@100 0.009115）作为
  低于 `50/50` 的参照；但每组仅一次回放，不能据此断言因果。

两组真实 LLM 合计请求 5 / 6；训练/重新召回/真实评测均为 0；来源与冻结
基线哈希运行前后一致。

## 运行

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_scenarios `
  --config configs/experiments/offline_replay_70_30_ablation.toml `
  --output-root experiments/2026-11-09-kuasar_small-70-30-history-ablation
```
