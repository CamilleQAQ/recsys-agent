# 2026-11-09 · Phase 13 历史信息节省查询的成对复验

本目录是 `70/30` 历史信息消融的成对复验运行容器。每次执行默认新建
`run-<UTC>-<pid>/`，不覆盖旧运行。3 对真实 LLM 回放按固定顺序
`A→B`、`B→A`、`A→B` 执行，每对两组各启动一次。

## 本次正式运行

- report.md（本地文件：`run-20261109T061138Z-30480/report.md`）
- metrics.json（本地文件：`run-20261109T061138Z-30480/metrics.json`）
- frozen/manifest.json（本地文件：`run-20261109T061138Z-30480/frozen/manifest.json`）
- self_check.json（本地文件：`run-20261109T061138Z-30480/self_check.json`）
- run_meta.json（本地文件：`run-20261109T061138Z-30480/run_meta.json`）

## 结果

- **3 / 3 对复现**：每对 A 均揭晓 2 次（`30/70 → 80/20`），B 均揭晓 1 次
  （`30/70` 后停止）；两组最佳均为 `30/70`（命中 5、Recall@100 0.013672）。
- 可判定对数 **3 / 3**；复现对数 / 可判定对数 **3 / 3**。
- 6 次工作流全部 `verified`；模型请求合计 **15 / 18**。
- 本地自检 **21 / 21**；训练 / 重新召回 / 真实评测 **0 / 0 / 0**；来源与冻结基线哈希前后一致。
- 实现问题与修复说明：[FIXES.md](FIXES.md)。首次运行工件保留，未重跑真实 LLM。
- 本地规则与固定策略沿用上一轮冻结轨迹，两组选择相同，未重跑。

## 运行

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_scenarios --self-check `
  --config configs/experiments/offline_replay_70_30_ablation.toml
python -m recsys.experiments.offline_replay_scenarios --paired `
  --config configs/experiments/offline_replay_70_30_ablation.toml `
  --output-root experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired
```
