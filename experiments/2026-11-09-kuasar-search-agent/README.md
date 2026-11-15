# 2026-11-09 · Search 2 · Search Agent MVP

本目录是 Search Agent 第一轮验收容器：只覆盖 `search_bm25` 工具调用与一次反馈决策，
不包含中文文本输入、向量检索、排序/重排或全量评测。

## 本次交付

- [report.md](report.md)：工具接口、查询调整规则、预算、真实决策、问题与遗留项。
- [self_check.json](self_check.json)：12 项本地合成检查（不联网）。
- `run-20261109T080657Z-19556/`：唯一一次真实工作流
  - trace.json（本地文件：`run-20261109T080657Z-19556/trace.json`）：工具调用 ID、查询 token、
    返回数量、继续/停止理由、请求计数。
  - metrics.json（本地文件：`run-20261109T080657Z-19556/metrics.json`）：验收 checks、预算、索引身份。
  - run_meta.json（本地文件：`run-20261109T080657Z-19556/run_meta.json`）：命令、模型、索引、查询来源。

## 结果速览

- 模型：`deepseek-chat`；真实工作流 **1 次**，状态 **verified**。
- 查询：会话 `11208`，token `[3538762, 3181091, 4061567, 3525929]`。
- `search_bm25` 第一次返回 **5 条**；模型收到结果后**主动停止**。
- **2 次模型请求 / 1 次 BM25 查询**（上限 3 / 2）；11/11 检查通过。
- 索引 sha256 `3ae2b858…` 前后一致；没有重建索引、没有扫原始 CSV。
- 候选明细没有逐条落盘；模型最终回复里的候选列表在落盘前截断，原始回复只留 sha256。

## 运行

```powershell
$env:PYTHONPATH = "src"

# 本地 self-check（不联网）
python -m recsys.experiments.search_agent --self-check

# 真实工作流（本轮已执行一次；复跑会新建 run 目录）
python -m recsys.experiments.search_agent `
  --tokens "3538762,3181091,4061567,3525929" `
  --output-root experiments/2026-11-09-kuasar-search-agent
```

实现与配置：[tool.py](../../src/recsys/search/tool.py)、
[search_agent.py](../../src/recsys/experiments/search_agent.py)、
[search_agent.toml](../../configs/experiments/search_agent.toml)。
