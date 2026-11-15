# 2026-11-09 · Phase 13 真实 LLM Rec Agent 离线回放

本目录是真实 LLM 三方离线回放的运行容器。每次运行默认新建
`run-<UTC>-<pid>/`，不覆盖旧运行；每个运行目录自包含配置、来源清单、
冻结基线引用、真实 LLM 轨迹、三方指标、自检和报告。

## 本次真实运行

- report.md（本地文件：`run-20261109T025055Z-22736/report.md`）
- llm_trace.json（本地文件：`run-20261109T025055Z-22736/llm_trace.json`）
- metrics.json（本地文件：`run-20261109T025055Z-22736/metrics.json`）
- sources.json（本地文件：`run-20261109T025055Z-22736/sources.json`）
- baselines.json（本地文件：`run-20261109T025055Z-22736/baselines.json`）
- self_check.json（本地文件：`run-20261109T025055Z-22736/self_check.json`）

## 结果

真实 LLM `deepseek-chat` 状态 **verified**：3 次模型请求 / 2 次成功揭晓 /
5.244 s。首查 `70/30`（命中 3、Recall@100 0.009115），看到反馈后查
`30/70`（命中 5、Recall@100 0.013672）；程序按 50/50 + 本侧揭晓结果
算出最佳 `30/70`。

本地规则 Agent 与固定策略直接引用上一轮已验收轨迹。三方最佳均为
`30/70`（命中 5、Recall@100 0.013672、Coverage 0.026720），三方最佳持平；
三方各揭晓 2 次，没有查询次数节省。首次达到最佳分别为 LLM 2、
本地规则 1、固定策略 2。

只覆盖 `kuasar_small` 的 128 个验证用户 / 362 个目标；不据此宣称正式测试集
收益或策略泛化能力。

截断回复修复的本地回归（13/13，不调用真实 LLM）见
[../2026-11-09-kuasar_small-offline-replay-llm-regression/report.md](../2026-11-09-kuasar_small-offline-replay-llm-regression/report.md)。

## 运行

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_llm --self-check
python -m recsys.experiments.offline_replay_llm
```
