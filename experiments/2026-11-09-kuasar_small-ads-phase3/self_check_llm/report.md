# Ads Agent 运行报告

- 状态：**verified**
- 计划器：`llm`（LLM 请求 2 次）
- 任务：用 din 对广告候选排序
- 数据集：`kuasar_small`
- 选中模型：`din`
- 工具顺序：rank_ads
- 耗时：3.07 s；工具预算：{'max_tool_calls': 3, 'max_train_calls': 2, 'max_seconds': 120.0, 'max_epochs': 3, 'max_batches': 20, 'calls': 1, 'train_calls': 0, 'elapsed_s': 3.07, 'remaining_s': 116.93}

## 工具调用轨迹

| # | 工具 | tool_call_id | ok | 耗时(s) | 参数 |
| --- | --- | --- | --- | --- | --- |
| 1 | rank_ads | call_ads_1 | True | 2.967 | `{"model": "din", "split": "train", "requests": 1, "min_candidates": 2, "top_k": 3}` |

## 真实输出摘要

```json
{
  "rank_ads": {
    "calls": 1,
    "summaries": [
      {
        "arguments": {
          "model": "din",
          "split": "train",
          "requests": 1,
          "min_candidates": 2,
          "top_k": 3
        },
        "run_id": null,
        "summary": {
          "requests": 1,
          "candidates": 3,
          "mean_candidates_per_request": 3.0,
          "top1_ctr_mean": 0.375297,
          "top_k_clicks": 1,
          "requests_with_multiple_candidates": 1
        },
        "budget_overrides": null
      }
    ],
    "weight": {
      "weight_kind": "fixed_init_interface_test",
      "is_trained": false,
      "checkpoint": "artifacts\\kuasar_small\\ads\\interface_weights\\din.pt"
    }
  }
}
```

## 最终回复

排序完成，结果见工具返回。
