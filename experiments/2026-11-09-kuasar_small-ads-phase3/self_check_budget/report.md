# Ads Agent 运行报告

- 状态：**partial**
- 计划器：`local`（LLM 请求 0 次）
- 任务：工具预算回归
- 数据集：`kuasar_small`
- 选中模型：`None`
- 工具顺序：list_ad_models
- 耗时：0.04 s；工具预算：{'max_tool_calls': 1, 'max_train_calls': 2, 'max_seconds': 120.0, 'max_epochs': 3, 'max_batches': 20, 'calls': 1, 'train_calls': 0, 'elapsed_s': 0.04, 'remaining_s': 119.96}

## 工具调用轨迹

| # | 工具 | tool_call_id | ok | 耗时(s) | 参数 |
| --- | --- | --- | --- | --- | --- |
| 1 | list_ad_models | - | True | 0.008 | `{}` |

## 真实输出摘要

```json
{
  "list_ad_models": {
    "calls": 1
  }
}
```

## 最终回复

工具调用次数预算耗尽，停止执行。
