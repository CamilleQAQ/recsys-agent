# 2026-11-09 · Search Agent 修复回归

本目录只记录 `search_agent.py` 两个修复的本地替身回归：候选/模型自由文本不落盘、
`--tokens` 样本外查询可执行。未调用真实 LLM、未重建索引、未重跑真实冒烟。

## 文件

- [report.md](report.md)：改动位置、回归命令、实际结果与旧证据哈希。
- [regression.json](regression.json)：45 项检查的脱敏摘要（不含候选 ID / 模型正文）。

## 结果

- **45/45** 检查通过；本地 self-check **12/12**。
- 样本会话 `11208`：返回 5 条。
- 样本外 token `[2]`：`session_id=null`，返回 1 条。
- 样本外无匹配 token `[1000000000000]`：返回 0 条。
- `--session-id` 与 `--tokens` 不一致：blocked。
- 索引、样本、旧真实 trace/metrics/run_meta/self_check 哈希前后一致。

## 复现

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.search_agent `
  --regression --tokens 2 `
  --output-root artifacts/search_agent_regression_tokens
```

实现：[search_agent.py](../../src/recsys/experiments/search_agent.py)。
