# Ads Agent 运行报告

- 状态：**partial**
- 计划器：`local`（LLM 请求 0 次）
- 任务：重复调用回归
- 数据集：`kuasar_small`
- 选中模型：`None`
- 工具顺序：list_ad_models
- 耗时：0.06 s；工具预算：{'max_tool_calls': 4, 'max_train_calls': 2, 'max_seconds': 120.0, 'max_epochs': 3, 'max_batches': 20, 'calls': 1, 'train_calls': 0, 'elapsed_s': 0.06, 'remaining_s': 119.94}

## 工具调用轨迹

| # | 工具 | tool_call_id | ok | 耗时(s) | 参数 |
| --- | --- | --- | --- | --- | --- |
| 1 | list_ad_models | - | True | 0.013 | `{}` |

## 真实输出摘要

```json
{
  "list_ad_models": {
    "calls": 1
  }
}
```

## 错误

- 拒绝重复调用：list_ad_models {}

## 最终回复

拒绝重复调用 list_ad_models（相同参数已执行过）。
