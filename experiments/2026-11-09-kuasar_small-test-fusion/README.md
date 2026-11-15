# 2026-11-09 · kuasar_small 独立测试集融合验证

本目录是 `test` split 上固定配额的召回融合验证容器。每次执行默认新建
`run-<UTC>-<pid>/`，不覆盖旧运行；不读取也不改写验证集缓存。

## 本次正式运行

- report.md（本地文件：`run-20261109T070024Z-25952/report.md`）
- metrics.json（本地文件：`run-20261109T070024Z-25952/metrics.json`）
- run_meta.json（本地文件：`run-20261109T070024Z-25952/run_meta.json`）
- frozen_config.toml（本地文件：`run-20261109T070024Z-25952/frozen_config.toml`）
- frozen_users.json（本地文件：`run-20261109T070024Z-25952/frozen_users.json`）
- cache/test/cache_meta.json（本地文件：`run-20261109T070024Z-25952/cache/test/cache_meta.json`）

## 结果

- 样本：`test` split，seed `42`，**128 用户 / 269 目标**；用户集合
  `efccbd46…`，目标对 `bc08a916…`。
- 双塔权重 `784a2ad6…`，MIND 权重 `eba01718…`；路由 seed `1234`，Top-100。
- **50/50**：命中 6；Recall@100 **0.023698**；Coverage **0.026724**。
- **30/70**：命中 6；Recall@100 **0.023698**；Coverage **0.026570**。
- **差值**：命中 +0；Recall@100 +0.000000；Coverage −0.000154。
- 边界检查：269 个测试目标进入历史 0、进入已见过滤 0、时间违规 0。
- 正式执行 1 次；双塔/MIND 各召回 1 次；两组各评测 1 次；总耗时 12.298 s，
  低于 300 s；训练 / LightGBM / 重排 / LLM 均为 0。

## 运行

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.recsys_test_fusion --fixture-check `
  --config configs/experiments/kuasar_small_test_fusion.toml
python -m recsys.experiments.recsys_test_fusion `
  --config configs/experiments/kuasar_small_test_fusion.toml `
  --output-root experiments/2026-11-09-kuasar_small-test-fusion
```
