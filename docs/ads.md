# Phase 3：KuaiSAR 广告系统

> 范围：广告样本管线、六个广告模型（DeepFM / DIN / DIEN / DCNv2 / MMoE / PLE）、
> CTR 与多任务预测、广告排序、二价竞价与预算模拟、Ads Agent。
> 不含：LinkedIn、投放草稿审批、网页、外部广告账户、Master Agent、跨业务实验平台和 RAG（Phase 4）。

## 1. 结论先行

| 能力 | 状态 | 入口 |
| --- | --- | --- |
| 广告样本（曝光请求内 `is_ad=1` 的候选 + click 标签） | 已构建，`data/kuasar_small/processed/ads/` | `python -m recsys.ads.data` |
| 六个广告模型 | 6/6 可构造、可推理、可保存/加载 | `python -m recsys.ads.verify` |
| CTR 预测 | 6 个模型统一接口 | `python -m recsys.ads.predict --model <name>` |
| 广告排序 | 请求内按 pCTR 降序 | `python -m recsys.ads.predict --mode rank` |
| 二价竞价 + CPC 预算扣减 | 已跑通，含预算不足分支 | `python -m recsys.ads.pipeline --budget-check` |
| Ads Agent | 7 工具 + LangGraph；本地 14/14 自检；真实 LLM 一次 verified | `python -m recsys.ads.agent` |
| CVR | **没有真实转化标签**；只在 synthetic 模拟样本上验证多任务输出 | `python -m recsys.ads.simulated` |
| 训练 | **本轮未训练**；权重是固定初始化，状态 `interface_test` | — |

**不能把本文的 AUC/LogLoss 当作广告效果。** 六个模型都没有训练，指标只用于确认
「模型能加载、能前向、输出在合理范围」。

## 2. 广告数据与标签

### 2.1 来源与复用

| 内容 | 来源 | 说明 |
| --- | --- | --- |
| 曝光、请求分组、时间戳 | `data/kuasar/processed/ranking/{split}_{items,labels,users,groups,timestamps}.npy` | 复用 Phase 1 请求级排序工件，`(user_id, timestamp)` 是一次曝光请求 |
| 点击标签 | 同上（`KuaiSAR_v2/rec_inter.csv` 的 `click`） | 0/1，真实标签 |
| 广告标记 | `data/<dataset>/processed/features/is_ad.npy` | KuaiSAR `item_type == AD` |
| 用户/物品特征 | `recsys.ranking.features.FeatureContext` | 类目 4 列 + 物品数值 5 列 + 用户数值 2 列；与推荐排序同一口径 |
| 行为序列历史 | `recsys.ranking.samples.histories_for_requests` | 严格 `interaction_timestamp < request_timestamp`，同请求候选共享一份历史 |

广告候选定义：**同一曝光请求中 `item_features.is_ad == 1` 的物品**；标签是该广告曝光是否被点击。
父数据集 ID 通过 `ranking.samples.map_parent_to_target` 映射到 `kuasar_small` 词表。

### 2.2 固定小样本（`kuasar_small`，2,000 用户）

| 切分 | 广告候选 | 含广告的请求 | 其中 ≥2 个广告的请求 | 点击率 |
| --- | ---: | ---: | ---: | ---: |
| train | 5,483 | 5,186 | 278 | 0.7011 |
| validation | 51 | 49 | 2 | 0.6863 |
| test | 43 | 41 | 2 | 0.8372 |

**数据边界：** KuaiSAR 每次曝光通常只展示 1 个广告，所以验证/测试里多广告候选请求极少。
广告排序/竞价的功能验收因此优先使用 train 切分的多候选请求（278 个），只验证链路行为，
不用于效果结论。测试集 43 行只做固定小样本接口检查。

### 2.3 真实转化标签核对

`audit_conversion_labels()` 读取 `KuaiSAR_v2/rec_inter.csv` 表头，实际字段：
`user_id, item_id, duration_ms, playing_time, timestamp, forward, like, follow,
search_photo_related, search, click, time`。**没有任何 conversion / purchase / order / pay 字段。**

| 标签 | 可用性 | 用途 |
| --- | --- | --- |
| click | 真实可用 | CTR、广告排序、竞价扣费模拟 |
| conversion | **缺失** | CVR 只在 `ads_simulated` 的 synthetic 样本上验证 |

模拟转化样本（`python -m recsys.ads.simulated`）：

- 目录 `data/kuasar_small/processed/ads_simulated/`，与真实广告样本分开；
- 304 行 / 211 点击 / 50 转化；转化只在 `click==1` 上生成；
- 规则 `blake2b("{user}:{item}:{seed}") % 100 < cvr_percent`（默认 20%），seed=42；
- `schema.json` 明确写 `synthetic=true / not_real_conversion=true / not_for_effect_claims=true`。

**禁止把 click 当转化。** MMoE/PLE 的 CVR 输出只能描述为「多任务接口验证」。

### 2.4 工件与校验

```
data/kuasar_small/processed/ads/
  {train,validation,test}_{items,labels,users,groups,timestamps}.npy
  schema.json        # 候选定义、标签、切分统计、conversion 缺口、每文件 sha256
data/kuasar_small/processed/ads_simulated/
  {users,items,click,conversion,groups,timestamps}.npy + schema.json
```

```powershell
python -m recsys.ads.data --check
python -m recsys.ads.simulated --check
```

## 3. 六个广告模型

### 3.1 复用关系

| 模型 | 实现来源 | 广告侧适配 | 任务 | 读序列历史 |
| --- | --- | --- | --- | --- |
| DeepFM | `third_party/DeepCTR-Torch/deepctr_torch/models/deepfm.py` | `DeepCTRAdAdapter`：5 稀疏 + 7 数值特征 → DeepCTR 输入 | CTR | 否 |
| DIN | `src/recsys/models/din.py`（Phase 1） | 直接 `build_model` 构造；广告数据/标签替换 | CTR | 是 |
| DIEN | `src/recsys/models/dien.py`（Phase 1） | 同上，AUGRU 历史严格早于请求 | CTR | 是 |
| DCNv2 | `src/recsys/models/dcnv2.py`（Phase 1） | 同上，不需要时间戳 | CTR | 否 |
| MMoE | `third_party/DeepCTR-Torch/deepctr_torch/models/multitask/mmoe.py` | 双任务 `ctr` / `cvr`，CVR 标签 synthetic | 多任务 | 否 |
| PLE | `third_party/DeepCTR-Torch/deepctr_torch/models/multitask/ple.py` | 同 MMoE，任务分层专家 | 多任务 | 否 |

DeepCTR-Torch 是本地源码，无需 pip 安装；`AdModelConfig.deepctr_path` 指向
`third_party/DeepCTR-Torch`，导入失败会抛 `AdModelUnavailable` 并说明原因。

### 3.2 统一接口

`src/recsys/ads/models.py`：

```python
adapter = build_ad_model("din", data, config)      # 构造
outputs = adapter.score(batch)                     # {"ctr": ndarray} / {"ctr","cvr"}
adapter.save(path, weight_kind=..., dataset=...)
adapter, payload = load_ad_model(path, data, config, expected_items=...)
```

- `AdModelConfig`：embedding 8 / hidden 32 / attention 16 / dropout 0.1 / max_history 20（接口验收规模，不是调参结果）。
- `AdModelRegistry`（`src/recsys/ads/registry.py`）解析权重并标注状态：
  - `integrated`：`artifacts/<dataset>/ads/weights/<model>.pt` 存在（真实训练产物才允许写这里）；
  - `interface_test`：只有 `artifacts/<dataset>/ads/interface_weights/<model>.pt`（固定初始化）；
  - `unavailable`：都没有，调用工具会明确报错。
- 词表大小不一致、checkpoint 格式不识别、缺少依赖都会抛出可执行的错误，不会静默返回随机结果。

### 3.3 接口验收结果（`experiments/2026-11-09-kuasar_small-ads-phase3/model_check.json`）

| 模型 | 状态 | 推理 | 保存/加载差值 | test AUC* | test LogLoss* | 权重体积 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| DeepFM | interface_test | ok | 0.0 | 0.6984 | 1.3524 | 16.4 MB |
| DIN | interface_test | ok | 0.0 | 0.5833 | 0.8322 | 14.6 MB |
| DIEN | interface_test | ok | 0.0 | 0.3690 | 0.6922 | 14.6 MB |
| DCNv2 | interface_test | ok | 0.0 | 0.5278 | 0.6935 | <0.1 MB |
| MMoE | interface_test | ok（ctr+cvr） | 0.0 / 0.0 | 0.5913 | 0.8444 | 14.6 MB |
| PLE | interface_test | ok（ctr+cvr） | 0.0 / 0.0 | 0.6548 | 0.6064 | 14.6 MB |

\* 固定初始化权重在 43 行 test 小样本上的数值，**只证明输出可用，不是效果**。
`save_load_max_abs_diff = 0.0` 表示保存后重新加载、同一批次前向结果逐元素一致。

## 4. 广告排序

`src/recsys/ads/ranking.py`：同一曝光请求内按 pCTR 降序，平局按 item_id 升序；
返回 `rank / item / ctr / click_label（诊断用）/ advertiser`。排序层不绑定模型。

## 5. 竞价与预算模拟

### 5.1 规则（`configs/ads/ads_sim.toml`）

| 项 | 规则 |
| --- | --- |
| 竞价模式 | 二价 GSP（`second_price`） |
| 出价 | `bid_cpc = base_cpc × (1 + step × (author_id % mod))` |
| 排序 | `eCPM = bid_cpc × pCTR × 1000` 降序，平局按 item_id |
| 结算价 | `max(reserve_cpc, 下一名 eCPM / (赢家 pCTR × 1000))`，且不超过赢家出价 |
| 计费单位 | **CPC（按点击）** |
| 扣费时机 | **点击发生时**；离线用样本 `click` 标签模拟是否产生点击 |
| 预算 | 总预算 + 单广告主预算（广告主代理 = 物品特征 `author_id`） |
| 预算不足 | 广告主剩余 < 保留价不参与竞价；实际扣费超过剩余时只扣到 0 并标记 `partial_charge` / `budget_exhausted` |

**出价与预算都是模拟值**：KuaiSAR 没有真实广告出价/预算，输出里固定带
`bid_source=simulated_config`、`budget_source=simulated_config`、`assumptions`。

### 5.2 一次请求的输出

`requests_detail[i]` 包含：候选完整排名（pCTR、bid、eCPM、eligible、click_label）、
赢家、`clearing_cpc`、`charge`、`billed_click`、`budget_remaining`、
`no_winner_reason`（如 `budget_exhausted_total`）。

### 5.3 预算不足分支（已实际验证）

`python -m recsys.ads.pipeline --budget-check` 会额外用 0 预算跑一次，断言：

- `winners == 0`、`spend == 0`、`remaining == 0`；
- 每个请求都有 `no_winner_reason=budget_exhausted_total`；
- 正常运行时 `spend == Σ charge`、`remaining == total − spent`。

2026-11-09 实跑：正常 3 个请求，赢家 3，计费点击 2，扣费 2.1672 CNY，剩余 17.8328 / 20；
0 预算 3 个请求全部未成交，扣费 0 —— 6/6 检查通过。

## 6. 命令

```powershell
$env:PYTHONPATH = "src"

# 数据
python -m recsys.ads.data --check
python -m recsys.ads.simulated --check

# 六个模型：构造固定初始化权重 + 推理 + 保存/加载往返（写入独立 interface_weights 目录）
python -m recsys.ads.verify

# CTR 预测 / 排序 / 多任务（simulated）
python -m recsys.ads.predict --model din --split test
python -m recsys.ads.predict --model deepfm --mode rank --split train --requests 3 --min-candidates 2
python -m recsys.ads.predict --model mmoe --mode multitask

# 竞价 + 预算 + 预算不足验证
python -m recsys.ads.pipeline --model din --split train --requests 3 --min-candidates 2 --budget-check
python -m recsys.ads.pipeline --model deepfm --budget 0 --split train

# Ads Agent（见 docs/ads-agent.md）
python -m recsys.ads.agent --planner local
python -m recsys.ads.agent --self-check

# 总验收
python scripts/smoke_phase3_ads.py
```

## 7. 关键文件

| 文件 | 作用 |
| --- | --- |
| `src/recsys/ads/data.py` | 广告样本构建、标签核对、CLI |
| `src/recsys/ads/simulated.py` | 独立模拟转化样本 |
| `src/recsys/ads/features.py` | 广告批次特征 + CTR 指标 |
| `src/recsys/ads/models.py` | 六模型适配器与保存/加载 |
| `src/recsys/ads/registry.py` | 模型注册与权重状态 |
| `src/recsys/ads/runtime.py` | 数据/特征/历史/模型缓存 |
| `src/recsys/ads/service.py` | 预测/排序/竞价业务函数 |
| `src/recsys/ads/ranking.py` | 请求内广告排序 |
| `src/recsys/ads/auction.py` | 二价竞价、预算状态、汇总 |
| `src/recsys/ads/pipeline.py` | 业务链路 CLI |
| `src/recsys/ads/verify.py` | 六模型接口验收 |
| `src/recsys/ads/tools.py` / `agent.py` | Ads Agent 工具与 LangGraph 工作流 |
| `configs/ads/ads_sim.toml` | 竞价/预算/模型规模（模拟值来源） |
| `configs/ads/ads_agent.toml` | Agent 预算与默认参数 |
| `scripts/smoke_phase3_ads.py` | Phase 3 总验收 |

## 8. 已知限制

1. **未训练**：六个模型都是固定初始化权重，`status=interface_test`；AUC/LogLoss 不能作为效果结论。
2. **CVR 无真实标签**：只在 synthetic 模拟样本上验证 MMoE/PLE 的双输出；不把 click 当转化。
3. **样本小且多广告请求少**：`kuasar_small` test 仅 43 个广告候选、2 个多广告请求；竞价演示主要用 train 的 278 个多候选请求，不代表线上分布。
4. **出价/预算为模拟值**：`base_cpc`、`author_multiplier`、总预算、单广告主预算都来自配置，不是真实广告主参数。
5. **扣费口径是离线模拟**：`realized_click` 用样本点击标签决定是否扣费，不是线上计费日志；也可切 `expected_click` 按期望扣费。
6. **DCNv2 不需要序列历史**：与 DIN/DIEN 的口径差异已在模型中注明；不是遗漏。
