# 代码修改型 Research Agent

基于 karpathy/autoresearch 的 program.md 改写研究循环；上游文件保存在
third_party/autoresearch，UPSTREAM.json 记录固定 commit 和文件哈希。
上游的 GPT 训练代码仅作来源保存，没有安装其依赖或执行训练。

## 当前可以做什么

Research Agent 调用 run_algorithm_research，按业务分发到现有 Rec Agent、Search Agent、
Ads Agent 的 run_research 入口。每个业务 Agent 使用真实 LLM 编写 Python 源码，实现
`rank(groups, context)`，执行后读取指标和冷热分组反馈，再编写下一轮算法。
不是从固定模型组合或配置菜单中选一个，也不是由脚本代替 LLM 决策。

当前研究对象是冻结模型推理结果上的融合、候选处理和排序。候选池固定，不能借此证明
新增召回覆盖、模型结构优化或训练目标改进。推荐结果、搜索点击和广告点击都来自实际数据。
广告离线排序不等于竞价收益、CVR 或线上效果。

每轮保存：algorithm.py、changes.diff、run.log、rankings.json、feedback.json、LLM 原始响应。
完整对话在 messages.json。best_algorithm.py 保存已观察到的验证最优实现；较差代码保留
在实验目录供复盘，不替换当前最佳，也不修改业务默认权重。

统一 TaskBudget 覆盖总控请求、子 Agent 请求、实验和历史读取。基线执行算一次实验，
默认每业务再进行两轮代码实验。循环是有限的 for 循环，没有轮询等待或自动重试；
每次算法执行只有一个子进程与一个执行超时。达到预算即停止，真实失败不报成功。

## 准备真实验证输入

使用已有推荐验证候选快照，以及现成搜索索引和广告权重：

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -B scripts/prepare_autoresearch_cases.py `
  --output artifacts/research_cases `
  --rec-channels experiments/2026-11-13-kuairec-fusion-research/diagnosis/validation_channels.json.gz `
  --rec-splits experiments/2026-11-13-kuairec-small `
  --rec-models two_tower,itemcf,popularity `
  --ads-models din,dcnv2
```

上面的 KuaiRec 路径只是现有数据示例，不是研究器的固定依赖。推荐快照路径与通道名可替换；
也可以直接按下面的格式提供其他数据集。搜索使用现有 BM25/FAISS 服务，广告通过 AdsRuntime
实际加载模型推理；缺工件会报错，不训练或重建。单业务可用 `--businesses search`。

搜索示例来自已有开发查询样本，明确用于验证，不是新建立的独立测试集。
推荐示例仍沿用之前已公开查看过的切分，不能作为新的盲测成绩。

## 从统一入口启动

```powershell
python -B -m recsys.platform `
  --research-cases artifacts/research_cases/manifest.json `
  --planner llm `
  --task "分别研究推荐、搜索、广告排序；各做两轮算法代码修改，根据上一轮反馈决定下一轮，不训练"
```

`--research-cases` 自动使用 configs/platform/autoresearch.toml 的统一预算；
也可指定 `--config`。原有普通业务入口保持原来的配置。
需要项目已有 .env 中的 LLM 配置，未配置时不会把本地脚本伪装成自主研究。

也可以直接调用单个业务 Agent：

```python
from recsys.experiments.rec_agent import run_research
# 搜索：recsys.experiments.search_agent.run_research
# 广告：recsys.ads.agent.run_research
result = run_research(
    "研究融合算法，根据实验反馈修改代码，不训练",
    case_path="artifacts/research_cases/recommendation.json",
    work_dir="experiments/my_research",
    store="experiments/autoresearch/shared",
    rounds=2,
)
```

## 输入约定

```json
{
  "business": "recommendation",
  "dataset": "your_dataset",
  "split": "validation",
  "objective": "NDCG@20",
  "ks": [20, 100],
  "cold_items": [9],
  "context": {"scope": "实际数据协议和特征含义"},
  "groups": [{
    "id": "user_or_query_or_request",
    "candidates": [
      {"item": 8, "cold": false, "ranks": {"model_a": 1}},
      {"item": 9, "cold": true, "ranks": {"model_b": 1}}
    ],
    "positives": [9, 10]
  }]
}
```

positives 保留未被召回的正例，不能只用候选中的正例做分母。
cold_items 用于对全部正例分组，包含未召回的冷正例；没有 cold_items 时使用候选的 cold 字段。
只有固定评估器读取 positives，算法子进程接收的输入没有这个字段。
原始源码可通过 source_files 提供；baseline_code 可提供现有算法的可执行版本，
默认基线保持输入候选顺序。当前目标为越大越好的现有排名指标。

共享记忆记录数据、评估器指纹、代码哈希、假设、反馈分析和验证结果。
只有业务、数据、评估协议一致且工件哈希有效的实验才作为对应历史读取。
研究模式不会因为命中历史就跳过新的研究任务。

## 定点验证

```powershell
python -B scripts/check_autoresearch.py
```

测试包括：两轮代码实际执行与结果反馈、较差版本不替换最佳、跨任务记忆读取、
父子预算共用、失败不误报成功、未召回正例计入冷热分组分母。
测试中的 LLM 是明确标注的测试替身；真实三业务运行证据另存于 experiments/2026-11-13-autoresearch。
