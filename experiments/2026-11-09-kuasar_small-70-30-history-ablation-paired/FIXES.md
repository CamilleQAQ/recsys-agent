# 2026-11-09 · 成对复验实现问题与修复

本目录保留首次 3 对真实回放的原始工件；以下问题在代码审查后修复，未重跑真实 LLM。

## 1. 成对模式重新执行了本地/固定基线

- 现象：首次 `--paired` 运行调用了 `freeze_local()`，重新执行并写出本地规则 Agent 与固定策略轨迹，而不是复用上一轮消融已冻结的基线。
- 影响：本次四份轨迹与上一轮冻结文件 sha256 完全一致，因此指标和判定未受影响；但执行范围表述不准确。
- 修复：`--paired` 改为从配置的 `[paired].reference_run` 读取 `frozen/manifest.json`，逐文件核对 sha256、起始历史、候选和预算；只把参照 manifest 写入新运行目录，不再执行本地策略。新运行记录 `frozen_reference_run`。
- 位置：`src/recsys/experiments/offline_replay_scenarios.py` 的 `load_reference_frozen` / `execute_paired`。

## 2. 写工件失败时模型请求漏计

- 现象：首次实现先写轨迹/events/metrics，成功后才把 `request_count` 累加到总数；若模型请求已完成但随后写工件抛错，该组请求数会漏计。
- 影响：异常路径下 `llm_request_total` 可能低于真实请求数，预算上限校验也可能放行。
- 修复：`_run_llm_workflow` 返回后立即把该组请求数计入总数，再执行任何工件写入；失败组在 `blocked.json` 与配对条目中保留 `request_count`。
- 本地替身回归：模拟首个 `llm_trace.json` 写入失败，总请求仍为 **15**（未修复时为 12）；该组记录 `blocked` 且 `request_count=3`。

## 3. 请求在工作流返回前失败仍会漏计

- 现象：上一版把请求数累加放在 `_run_llm_workflow()` 返回之后；如果模型调用已完成，但 `build_llm_trace()` 等工作流内部生成步骤抛错，外层拿到的是异常而不是结果，仍会记 0。
- 修复：把计数前移到**模型调用边界**。`offline_replay_llm.py` 的 `_call_model()` 在通过每组请求预算和全轮 18 次上限校验后、真正调用 `chat_fn` 前，写入工作流计数器与全轮计数器；客户端超时/报错仍计数，预算校验拒绝不计数。
- `execute_paired()` 使用这两个计数器填充每组 `request_count` 和全轮 `llm_request_total`，删除从返回结果累加总数的逻辑；异常处理从计数器读取已发请求数。
- 本地替身回归（均在第一对 A 组发出 3 次请求后注入）：
  - `build_llm_trace` 抛错：失败组 3 次，全轮 15 次，第一对不可判定；
  - 写 `llm_trace.json` 抛错：失败组 3 次，全轮 15 次，第一对不可判定；
  - 正常完成：每组计数与替身实际调用一致，合计 15 次，无重复累计。
- 全轮上限在调用客户端前检查；达到 18 次后拒绝请求，拒绝动作不增加计数器。
- 可复现回归脚本：`scripts/check_paired_request_accounting.py`。

```powershell
$env:PYTHONPATH = "src"
python -B scripts/check_paired_request_accounting.py
```

回归结果：正常 / `build_llm_trace` 抛错 / 写 `llm_trace.json` 抛错三组均为
全轮 15 次、逐组之和 15；两条故障的第一对 A 组记录 3 次且该对不可判定。

## 4. 现有工件

首次真实配对结果（3/3 复现、15/18 请求）保留在
`run-20261109T061138Z-30480/`，未重写。由于本地/固定轨迹与参照哈希相同，结果仍然有效；
后续 `--paired` 运行使用修复后的参照复用与请求计数路径。
