# Ads Agent 运行报告

- 状态：**failed**
- 计划器：`llm`（LLM 请求 1 次）
- 任务：调用不存在的工具
- 数据集：`kuasar_small`
- 选中模型：`None`
- 工具顺序：
- 耗时：0.04 s；工具预算：{'max_tool_calls': 2, 'max_train_calls': 2, 'max_seconds': 60.0, 'max_epochs': 3, 'max_batches': 20, 'calls': 0, 'train_calls': 0, 'elapsed_s': 0.04, 'remaining_s': 59.96}

## 工具调用轨迹

| # | 工具 | tool_call_id | ok | 耗时(s) | 参数 |
| --- | --- | --- | --- | --- | --- |

## 真实输出摘要

```json
{}
```

## 错误

- 未注册的工具：delete_database

## 最终回复

LLM 工具参数不合法：未注册的工具：delete_database
