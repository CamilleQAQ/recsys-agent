# 2026-11-09 · Phase 13 小规模多场景离线对照

本目录是多场景离线对照的运行容器。每次执行默认新建
`run-<UTC>-<pid>/`，不覆盖旧运行。先冻结场景 A/B 的本地规则 Agent 与
固定策略轨迹，再让真实 LLM 在每个场景各运行一次。

## 本次正式运行

- report.md（本地文件：`run-20261109T034321Z-4528/report.md`）
- metrics.json（本地文件：`run-20261109T034321Z-4528/metrics.json`）
- frozen/manifest.json（本地文件：`run-20261109T034321Z-4528/frozen/manifest.json`）
- 场景 A LLM 轨迹（本地文件：`run-20261109T034321Z-4528/scenarios/A/llm_trace.json`）
- 场景 B LLM 轨迹（本地文件：`run-20261109T034321Z-4528/scenarios/B/llm_trace.json`）
- self_check.json（本地文件：`run-20261109T034321Z-4528/self_check.json`）

## 结果

- **场景 A（1 次揭晓）**：真实 LLM 首选 `70/30`、最佳 `50/50`（1 次揭晓，
  2 次请求）；本地规则首选 `30/70`、最佳 `30/70`（1 次揭晓）；固定策略
  首选 `70/30`、最佳 `50/50`（1 次揭晓）。本地规则 Agent 胜出，真实 LLM
  输给最佳。
- **场景 B（2 次揭晓，带 70/30 历史）**：真实 LLM 首选 `30/70`、最佳
  `30/70`（2 次揭晓，3 次请求）；本地规则 `30/70 → 20/80`，固定策略
  `30/70 → 80/20`，三者最佳均为 `30/70`，三方最佳持平；各揭晓 2 次，
  没有查询次数节省。

真实 LLM 两场景合计请求 5 / 5，训练/重新召回/真实评测均为 0；来源与冻结
基线哈希运行前后一致。

## 运行

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_scenarios --self-check
python -m recsys.experiments.offline_replay_scenarios
```
