# 2026-11-09 · Phase 13 离线回放 MVP

本目录是 Phase 13 离线回放的运行容器。每次运行默认新建
`run-<UTC>-<pid>/`，不覆盖旧运行；每个运行目录自包含配置、来源清单、
双方轨迹、汇总指标、断言和报告。

## 本次正式回放

- report.md（本地文件：`run-20261109T020944Z-20800/report.md`）
- metrics.json（本地文件：`run-20261109T020944Z-20800/metrics.json`）
- sources.json（本地文件：`run-20261109T020944Z-20800/sources.json`）
- agent_trace.json（本地文件：`run-20261109T020944Z-20800/agent_trace.json`）
- fixed_trace.json（本地文件：`run-20261109T020944Z-20800/fixed_trace.json`）
- checks.json（本地文件：`run-20261109T020944Z-20800/checks.json`）

结论：Rec Agent 与固定策略的最佳配额均为 `30/70`（命中 5、Recall@100
0.013672、Coverage 0.026720），本轮对照记为**持平**。只覆盖
`kuasar_small` 的 128 个验证用户；不据此宣称正式测试集收益或策略泛化能力。

## 运行

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay --self-check
python -m recsys.experiments.offline_replay
```
