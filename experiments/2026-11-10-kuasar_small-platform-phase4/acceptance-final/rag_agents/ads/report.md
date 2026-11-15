# Ads Agent 运行报告

- 状态：**verified**
- 计划器：`local`（LLM 请求 0 次）
- 任务：查广告模型来源
- 数据集：`kuasar_small`
- 选中模型：`None`
- 工具顺序：retrieve_knowledge
- 耗时：0.04 s；工具预算：{'max_tool_calls': 3, 'max_train_calls': 2, 'max_seconds': 60, 'max_epochs': 3, 'max_batches': 20, 'calls': 1, 'train_calls': 0, 'elapsed_s': 0.04, 'remaining_s': 59.96}

## 工具调用轨迹

| # | 工具 | tool_call_id | ok | 耗时(s) | 参数 |
| --- | --- | --- | --- | --- | --- |
| 1 | retrieve_knowledge | - | True | 0.015 | `{"query": "DeepFM CTR 广告", "top_k": 2}` |

## 真实输出摘要

```json
{
  "retrieve_knowledge": {
    "calls": 1
  }
}
```

## 最终回复

Ads Agent 执行完成。
使用模型：default（权重状态见 list_ad_models / 工具返回的 weight 字段）。
说明：本轮没有训练；interface_test 权重只用于接口验证。
