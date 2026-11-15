# Ads Agent 运行报告

- 状态：**verified**
- 计划器：`local`（LLM 请求 0 次）
- 任务：用 din 模型完成广告预测、排序、竞价和预算模拟
- 数据集：`kuasar_small`
- 选中模型：`din`
- 工具顺序：list_ad_models → predict_ad_ctr → check_label_availability → rank_ads → simulate_auction → get_budget_state → simulate_auction
- 耗时：4.57 s；工具预算：{'max_tool_calls': 9, 'max_train_calls': 2, 'max_seconds': 180.0, 'max_epochs': 3, 'max_batches': 20, 'calls': 7, 'train_calls': 0, 'elapsed_s': 4.57, 'remaining_s': 175.43}

## 工具调用轨迹

| # | 工具 | tool_call_id | ok | 耗时(s) | 参数 |
| --- | --- | --- | --- | --- | --- |
| 1 | list_ad_models | - | True | 0.01 | `{}` |
| 2 | predict_ad_ctr | - | True | 3.129 | `{"model": "din", "split": "test", "limit": 500}` |
| 3 | check_label_availability | - | True | 0.005 | `{}` |
| 4 | rank_ads | - | True | 1.338 | `{"model": "din", "split": "train", "requests": 3, "min_candidates": 2, "top_k": 3}` |
| 5 | simulate_auction | - | True | 0.007 | `{"model": "din", "split": "train", "requests": 3, "min_candidates": 2, "top_k": 3}` |
| 6 | get_budget_state | - | True | 0.002 | `{}` |
| 7 | simulate_auction | - | True | 0.007 | `{"model": "din", "split": "train", "requests": 3, "min_candidates": 2, "top_k": 3, "total_budget": 0.0, "per_advertiser_budget": 0.0}` |

## 真实输出摘要

```json
{
  "list_ad_models": {
    "calls": 1
  },
  "predict_ad_ctr": {
    "calls": 1,
    "metrics": {
      "rows": 43,
      "clicks": 36,
      "ctr": 0.8372093023255814,
      "predicted_ctr": 0.41497043468231376,
      "AUC": 0.5833333333333334,
      "LogLoss": 0.8321706875594654
    },
    "weight": {
      "weight_kind": "fixed_init_interface_test",
      "is_trained": false,
      "checkpoint": "artifacts\\kuasar_small\\ads\\interface_weights\\din.pt"
    }
  },
  "check_label_availability": {
    "calls": 1
  },
  "rank_ads": {
    "calls": 1,
    "summaries": [
      {
        "arguments": {
          "model": "din",
          "split": "train",
          "requests": 3,
          "min_candidates": 2,
          "top_k": 3
        },
        "run_id": null,
        "summary": {
          "requests": 3,
          "candidates": 9,
          "mean_candidates_per_request": 3.0,
          "top1_ctr_mean": 0.37227,
          "top_k_clicks": 6,
          "requests_with_multiple_candidates": 3
        },
        "budget_overrides": null
      }
    ],
    "weight": {
      "weight_kind": "fixed_init_interface_test",
      "is_trained": false,
      "checkpoint": "artifacts\\kuasar_small\\ads\\interface_weights\\din.pt"
    }
  },
  "simulate_auction": {
    "calls": 2,
    "summaries": [
      {
        "arguments": {
          "model": "din",
          "split": "train",
          "requests": 3,
          "min_candidates": 2,
          "top_k": 3
        },
        "run_id": "0810bd29ef5b",
        "summary": {
          "requests": 3,
          "winners": 3,
          "clicks": 2,
          "spend": 2.1672,
          "mean_charge": 0.7224,
          "ctr": 0.666667,
          "mean_clearing_cpc": 1.097,
          "no_winner_requests": 0,
          "no_winner_reasons": [],
          "partial_charges": 0,
          "budget": {
            "currency": "CNY",
            "total": 20.0,
            "spent": 2.1672,
            "remaining": 17.8328,
            "per_advertiser_budget": 8.0,
            "served": 3,
            "clicks": 2,
            "advertiser_spent_top": {
              "22253": 1.0943,
              "11827": 1.0729
            },
            "budget_exhausted_requests": 0
          },
          "pricing_rule": "second_price_gsp",
          "billing_unit": "CPC",
          "charge_timing": "click"
        },
        "budget_overrides": {
          "total": null,
          "per_advertiser": null
        }
      },
      {
        "arguments": {
          "model": "din",
          "split": "train",
          "requests": 3,
          "min_candidates": 2,
          "top_k": 3,
          "total_budget": 0.0,
          "per_advertiser_budget": 0.0
        },
        "run_id": "913052f3d977",
        "summary": {
          "requests": 3,
          "winners": 0,
          "clicks": 0,
          "spend": 0.0,
          "mean_charge": 0.0,
          "ctr": 0.0,
          "mean_clearing_cpc": 0.0,
          "no_winner_requests": 3,
          "no_winner_reasons": [
            "budget_exhausted_total",
            "budget_exhausted_total",
            "budget_exhausted_total"
          ],
          "partial_charges": 0,
          "budget": {
            "currency": "CNY",
            "total": 0.0,
            "spent": 0.0,
            "remaining": 0.0,
            "per_advertiser_budget": 0.0,
            "served": 0,
            "clicks": 0,
            "advertiser_spent_top": {},
            "budget_exhausted_requests": 3
          },
          "pricing_rule": "second_price_gsp",
          "billing_unit": "CPC",
          "charge_timing": "click"
        },
        "budget_overrides": {
          "total": 0.0,
          "per_advertiser": 0.0
        }
      }
    ],
    "weight": {
      "weight_kind": "fixed_init_interface_test",
      "is_trained": false,
      "checkpoint": "artifacts\\kuasar_small\\ads\\interface_weights\\din.pt"
    }
  },
  "get_budget_state": {
    "calls": 1,
    "summaries": [
      {
        "arguments": {},
        "run_id": "0810bd29ef5b",
        "summary": {
          "requests": 3,
          "winners": 3,
          "clicks": 2,
          "spend": 2.1672,
          "mean_charge": 0.7224,
          "ctr": 0.666667,
          "mean_clearing_cpc": 1.097,
          "no_winner_requests": 0,
          "no_winner_reasons": [],
          "partial_charges": 0,
          "budget": {
            "currency": "CNY",
            "total": 20.0,
            "spent": 2.1672,
            "remaining": 17.8328,
            "per_advertiser_budget": 8.0,
            "served": 3,
            "clicks": 2,
            "advertiser_spent_top": {
              "22253": 1.0943,
              "11827": 1.0729
            },
            "budget_exhausted_requests": 0
          },
          "pricing_rule": "second_price_gsp",
          "billing_unit": "CPC",
          "charge_timing": "click"
        },
        "budget_overrides": null
      }
    ]
  }
}
```

## 最终回复

Ads Agent 执行完成。
使用模型：din（权重状态见 list_ad_models / 工具返回的 weight 字段）。
CTR 预测：test 43 行，AUC=0.5833，LogLoss=0.8322，实际点击率=0.8372。
标签核对：真实 click 可用；转化标签缺失，CVR 仅在 synthetic 模拟样本上验证。
广告排序：3 个请求 / 9 个候选，Top1 平均 pCTR=0.37227。
竞价模拟：3 个请求，赢家 3，点击 2，模拟扣费 2.1672，总预算剩余 17.8328。
竞价模拟（预算不足验证）：3 个请求，赢家 0，点击 0，模拟扣费 0.0，总预算剩余 0.0。
其中 3 个请求未成交，原因：budget_exhausted_total, budget_exhausted_total, budget_exhausted_total。
说明：本轮没有训练；interface_test 权重只用于接口验证。
