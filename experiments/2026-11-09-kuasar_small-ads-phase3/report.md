# Phase 3：KuaiSAR 广告系统交付报告

日期：2026-11-09
数据集：`kuasar_small`（2,000 用户 / 454,717 物品）
范围：广告样本、六个广告模型、CTR/多任务预测、广告排序、二价竞价与预算模拟、Ads Agent。
不在本轮：训练/调参、全量数据、LinkedIn、投放草稿审批、网页、外部广告账户、Phase 4 平台能力。

## 1. 交付结果

| 项 | 结果 | 证据 |
| --- | --- | --- |
| 广告样本 | train 5,483 / validation 51 / test 43 个广告候选；复用请求级曝光与 `click` | `data/kuasar_small/processed/ads/schema.json` |
| 真实转化标签 | **不存在**（`rec_inter.csv` 无 conversion/purchase 字段），CVR 只在 synthetic 样本上验证 | `data/kuasar_small/processed/ads/schema.json` → `conversion_label` |
| 六个模型 | 6/6 可加载、可推理、保存/加载差值 0 | `model_check.json` |
| 广告排序 | 请求内 pCTR 降序 + 平局 item_id | `smoke.json` → `ranking` |
| 竞价/预算 | 二价 GSP + CPC + 点击扣费；正常与 0 预算两条路径 | `smoke.json` → `auction` |
| Ads Agent | 本地 7 工具 verified；真实 LLM 自主选 deepfm，4 工具 verified | `agent_local/`、`agent_llm/` |
| 自检 | 14/14（含 LLM 替身协议、未知工具、重复调用、预算） | `self_check.json` |
| 总验收 | `scripts/smoke_phase3_ads.py` 全项通过 | `smoke.json` |

## 2. 广告数据口径

- 候选：同一 `(user_id, timestamp)` 曝光请求内 `item_features.is_ad == 1` 的物品。
- 标签：`rec_inter.click`（0/1），是点击，不是转化。
- 切分：直接复用 `processed/ranking` 的请求级 train/validation/test，不重新定义。
- 序列模型：只读 `interaction_timestamp < request_timestamp` 的历史；同请求候选共享一份。
- 特征：复用 `ranking.features.FeatureContext`（类目 4 列 + 物品数值 5 列 + 用户数值 2 列）。

| 切分 | 广告候选 | 请求 | ≥2 广告请求 | 点击率 |
| --- | ---: | ---: | ---: | ---: |
| train | 5,483 | 5,186 | 278 | 0.7011 |
| validation | 51 | 49 | 2 | 0.6863 |
| test | 43 | 41 | 2 | 0.8372 |

数据边界：KuaiSAR 每次曝光通常只有 1 个广告，验证/测试的多广告候选请求各只有 2 个。
排序/竞价的功能验收主要在 train 的多候选请求上完成（只验证链路行为，不作效果结论）。

## 3. 六个模型（固定初始化权重，非训练）

`python -m recsys.ads.verify` 输出（`model_check.json`）：

| 模型 | framework | 任务 | 推理 | 保存/加载差 | test AUC* | test LogLoss* | 权重 |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| DeepFM | DeepCTR-Torch | CTR | ok | 0.0 | 0.6984 | 1.3524 | 16.4 MB |
| DIN | Phase 1 复用 | CTR | ok | 0.0 | 0.5833 | 0.8322 | 14.6 MB |
| DIEN | Phase 1 复用 | CTR | ok | 0.0 | 0.3690 | 0.6922 | 14.6 MB |
| DCNv2 | Phase 1 复用 | CTR | ok | 0.0 | 0.5278 | 0.6935 | <0.1 MB |
| MMoE | DeepCTR-Torch | CTR+CVR | ok | 0.0/0.0 | 0.5913 | 0.8444 | 14.6 MB |
| PLE | DeepCTR-Torch | CTR+CVR | ok | 0.0/0.0 | 0.6548 | 0.6064 | 14.6 MB |

\* 43 行 test 小样本 + 固定初始化权重，只证明接口可用；**不构成模型效果或训练结论**。
权重位于 `artifacts/kuasar_small/ads/interface_weights/`（业务默认目录 `.../ads/weights/` 未写入）。
MMoE/PLE 的 CVR 在 `ads_simulated`（304 行 / 50 合成转化）上验证，`synthetic=true`。

## 4. 竞价与预算（一次请求级模拟）

规则：`eCPM = bid_cpc × pCTR × 1000` 降序 → 二价 GSP 结算 →
`charge = clearing_cpc`（仅当样本 click=1）→ 扣总预算与广告主预算。
出价/预算/点击扣费口径全部标注来源（模拟配置 + 真实 click 标签）。

2026-11-09 实跑（`python -m recsys.ads.pipeline --model din --budget-check`）：

| 请求 | 候选 | 赢家 | 结算 CPC | 扣费 | 剩余预算 |
| --- | ---: | --- | ---: | ---: | ---: |
| 355 | 3 | 171598 | 1.0729 | 1.0729 | 18.9271 |
| 356 | 3 | 105649 | 1.0943 | 1.0943 | 17.8328 |
| 362 | 3 | 29367 | 1.1238 | 0.0（未点击） | 17.8328 |

0 预算：3 个请求全部 `budget_exhausted_total`，赢家 0、扣费 0、剩余 0；
`spend == Σ charge`、`remaining == total − spent` 均通过。

## 5. Ads Agent 轨迹

### 5.1 本地计划（`planner=local`，verified）

`agent_local/run-20261109T143340Z-24812/`

`list_ad_models → predict_ad_ctr → check_label_availability → rank_ads →
simulate_auction → get_budget_state → simulate_auction(0 预算)`

- 选中 `din`；7 个工具全部 ok；4.66 s；
- CTR：test 43 行，AUC 0.5833 / LogLoss 0.8322（固定初始化）；
- 排序：3 个多候选请求 / 9 个候选；
- 竞价：3 请求，赢家 3、点击 2、扣费 2.1672、剩余 17.8328；
- 0 预算：3 请求全部未成交。

### 5.2 真实 LLM（`planner=llm`，`deepseek-chat`，verified）

`agent_llm/run-20261109T142641Z-25212/`

任务：用 deepfm 做 CTR 预测、排序、竞价与预算模拟，并说明转化标签情况。

| # | tool_call_id | 工具 | 参数 | ok |
| --- | --- | --- | --- | --- |
| 1 | `call_00_9i5AOhWQkb4RKtjCE71e0582` | predict_ad_ctr | deepfm / test / 43 | ✓ |
| 2 | `call_01_rbOnddQIsWw4jbgi8bdy2067` | rank_ads | deepfm / train / 20 请求 / ≥2 候选 | ✓ |
| 3 | `call_00_oGax0YoCXpQ8cTw8Xo1G5202` | simulate_auction | deepfm / train / 20 请求 | ✓ |
| 4 | `call_01_oPppA4Bl5DL8veipg8fC3549` | check_label_availability | — | ✓ |

- 3 次模型请求 / 4 次工具 / 21.25 s；
- 竞价：20 请求全部成交、9 次计费点击、花费 9.9028、剩余 10.0972 / 20；
- 最终回复正确标注 `fixed_init_interface_test`、未训练、CVR 无真实标签、未把 click 当转化。

**失败尝试记录：** 第一次真实调用使用 `max_llm_requests=4`，4 次请求被 4 个工具用完后
没有余量收尾，状态 `partial`（`agent_llm/run-20261109T142616Z-30600/`）。把上限改为 6
后重跑得到 verified。原因在请求预算配置，不是工具执行失败。

## 6. YieldAgent 复用（固定版本 e79fda0）

| 复用 | 说明 |
| --- | --- |
| 工具 / 业务分离 | `recsys.ads.tools` 只做 schema/调度，`recsys.ads.service` 承载预测/排序/竞价；对应 YieldAgent `console/tools.py` 与 `connectors/base.py` 的分层 |
| 工具调用方式 | 模型返回的 tool_call 逐条校验参数、程序顺序执行，并用同一 `tool_call_id` 回传；由项目 LangGraph `StateGraph` 实现 |
| 工具错误语义 | 工具异常统一转 `{ok:false,error}` 返回给模型（对应 `ToolNode(handle_tool_errors=True)`） |
| 结果记录 | 工具调用/结果事件、`coerce_payload` / `summarize_payload`（改写自 `runtime._coerce` / `_summarize`）；轨迹落 `events.jsonl` / `trace.json` |

未复用：`create_react_agent`、`@tool`、`ToolNode`、checkpointer/interrupt、SSE、LinkedIn 与草稿审批
（`langchain` 未安装；投放审批不在本轮范围）。详见 `docs/ads-agent.md` 的逐项对照表。

## 7. 验收命令

```powershell
$env:PYTHONPATH = "src"
python -m recsys.ads.data --check
python -m recsys.ads.simulated --check
python -m recsys.ads.verify
python -m recsys.ads.pipeline --model din --budget-check
python -m recsys.ads.agent --self-check
python -m recsys.ads.agent --planner local
python scripts/smoke_phase3_ads.py
```

## 8. 阻碍与限制

1. **没有训练**：六模型均为固定初始化权重，`interface_test`；没有真实广告效果结论。
2. **没有真实 CVR**：KuaiSAR 无转化标签；只有 synthetic 模拟样本，不能作为业务效果。
3. **样本小**：test 43 个广告候选、2 个多广告请求；竞价演示主要用 train 278 个多候选请求。
4. **竞价参数为模拟值**：`base_cpc`、广告主预算、保留价来自 `configs/ads/ads_sim.toml`。
5. **无真实出价/账户**：不接 LinkedIn/外部广告账户，不做投放草稿审批（范围外）。
6. **真实 LLM**：仅 1 次成功工作流（另 1 次因请求预算不足 partial）；未做多次重复与温度扫描。

以上均不阻塞 Phase 3 的代码交付；后续若拿到真实转化/出价数据，可直接替换
`ads_simulated` 与竞价配置，模型和工具接口不变。
