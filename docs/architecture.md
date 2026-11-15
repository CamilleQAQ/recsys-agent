# 架构

## 分层

```
原始数据 (CSV/zip)
      │  recsys.data.kuasar / kuairec                   ← 只负责「读 + 清洗 + 编码」
      ▼
PreparedData（train/val/test + IdMappings + 曝光 mask）
      │  recsys.data.dataset                            ← 只负责「按模型需要切样本」
      ├── 序列样本  (SASRec / BERT4Rec / HSTU: 序列或掩码序列)
      ├── 交互样本  (双塔/MF/LightGCN/MMRec: (user, item) 对)
      └── 候选+特征样本（排序模型：请求级候选 + 物品/用户特征）
      ▼
模型 (recsys.models / third_party)
      │  recsys.training                               ← 训练循环、损失、负采样、早停
      ▼
checkpoint（权重 + model_config + 切分协议 + provenance + 指标历史）
      │  recsys.adapters                               ← 统一「加载 → 打分/编码 → 评测」
      ▼
多路召回（recsys.pipeline.recall）→ 融合（fusion）→ 排序（ranking）→ 重排（rerank）
      │  recsys.pipeline.chain                         ← 统一配置、工件、逐阶段指标
      ▼
最终 Top-K + metrics/diagnostics/run_meta → recsys.evaluation.metrics
```

Rec Agent（`recsys.experiments.rec_agent` + `rec_tools`）不绕过上述分层：
它调用的训练、召回、排序、重排、评估都落到同一批函数，只额外加预算、
工具参数校验与运行日志。

分层的硬边界：

- **data 层不 import 模型**，模型层不碰文件路径；
- **训练脚本只写 checkpoint**，不直接产出实验报告；报告由 evaluation / experiments 生成；
- **adapters 不重新实现算法**，只做协议转换（必要时复用 evaluator 里的加载逻辑）。

## 与 PROJECT_MASTER §3 模块的对应

| 逻辑模块 | 仓库里的位置 | 状态 |
| --- | --- | --- |
| 数据层 | `src/recsys/data/` | KuaiSAR 已接入（npy 数组后端 + 稀疏曝光 CSR）；KuaiRec 保留为流程模板 |
| 召回层 | `src/recsys/models/` + `third_party/` + `src/recsys/adapters/` | 11 个模型已接入（Two-Tower/MIND/Popularity/MF/LightGCN/DSSM/SASRec/BERT4Rec/HSTU/TIGER/MMRec） |
| 候选融合层 | `src/recsys/pipeline/fusion.py` | round_robin / rrf / quota，保留来源与通道内名次 |
| 排序层 | `src/recsys/adapters/ranking*.py` + `src/recsys/pipeline/ranking.py` | LightGBM + DIN/DIEN/DCNv2/Ultra-HSTU |
| 重排层 | `src/recsys/pipeline/rerank.py` | 类目上限 + 可选作者多样性 + 补位计数 |
| 评估层 | `src/recsys/evaluation/` + `src/recsys/pipeline/chain.py` | 召回/排序/重排逐阶段指标、逐目标丢失环节、AUC/LogLoss |
| Rec Agent | `src/recsys/experiments/rec_agent.py` + `rec_tools.py` | LangGraph + 12 个真实工具 + 预算/日志/失败上报 |
| 推荐链路 | `src/recsys/pipeline/` | 统一 TOML 配置、默认双路行为、工件保存/读回 |
| 搜索检索层 | `src/recsys/search/` | BM25 + FAISS 向量（`IndexFlatIP`）+ RRF 混合 + token 精排；搜索索引独立于推荐索引 |
| Search Agent | `src/recsys/experiments/search_agent.py` + `search/tool.py` | LangGraph + 4 个工具（BM25/向量/混合/精排）+ 预算/空结果/失败/落盘边界 |

## 关键契约

| 契约 | 位置 | 约束 |
| --- | --- | --- |
| 数据 | `recsys.data.schema.PreparedData` | 见 [data-contract.md](data-contract.md) |
| checkpoint | `SPLIT_STRATEGY` + `model_config` + `provenance` | 缺 `split_strategy` 或与当前常量不符 → 直接报错，不允许「凑合评测」 |
| 工件溯源 | `recsys.utils.provenance` | 数据指纹、ID 映射摘要、代码指纹、依赖版本，见下 |
| 评测 | `recsys.evaluation.metrics` | 所有模型共用；不同脚本不得各写一份指标 |
| 模型清单 | `models/zoo.toml` | 与 `third_party/` 路径一一对应，`scripts/zoo.py check` 校验 |

## 适配层设计（`src/recsys/adapters/`）

```
base.py       RecommenderAdapter（recommend/retrieve/encode_users/encode_items/evaluate/save）
              RankingAdapter（fit/predict/evaluate/save/load）+ AdapterError
builtin.py    TwoTowerAdapter / MatrixFactorizationAdapter / SASRecAdapter / PopularityAdapter
ranking.py    LightGBMRankerAdapter（LightGBM LambdaRank / 二分类）
registry.py   model_id → 适配器类；load_adapter(model_id, checkpoint, device)
__main__.py   python -m recsys.adapters list|evaluate
```

- `recommend()` / `retrieve()`：返回 `{user_idx: [item_idx, ...]}`，已按 seen-mask 过滤，长度恒为 k；
  两个名字指向同一件事——`retrieve` 是 PROJECT_MASTER §5.1 的召回语义，供多路融合按通道调用；
- `encode_users()` / `encode_items()`：导出 float32 向量，交给 FAISS；不支持向量表示的模型
  （Popularity、SASRec 的用户侧）明确抛 `AdapterError`，不返回伪造向量；
- `evaluate()`：复用 `evaluate_recommendations` 并补全局 `Coverage`，与模型专用脚本同口径；
- `RankingAdapter`：`fit(features, labels, groups)` → `predict(features)` → `evaluate`（AUC/LogLoss）；
  `groups` 给定时走 Lambdarank，否则走二分类；
- 无参数基线（Popularity）实现 `recommend` 即可，`save()` 明确报「不支持」而不是假装成功。

为什么 `third_party/` 不做物理分类（`retrieval/`、`ranking/` 各放一份）：

RecBole、FuxiCTR、DeepCTR、TorchEasyRec 每个仓库都同时覆盖召回和排序，硬拆会破坏包内相对导入，
也无法再和上游 diff。所以**分类只体现在 `models/` 索引层**，代码保持仓库完整。

## 工件溯源（PROJECT_MASTER §5.2）

每次训练写进 `checkpoint["provenance"]`：

| 字段 | 内容 | 用途 |
| --- | --- | --- |
| `dataset` / `positive_threshold` / `split_strategy` | 数据与口径 | 判断两个 checkpoint 能否比较 |
| `data.counts` / `split_sha256` | 三个切分的行数与内容哈希 | 数据换版本立刻能看出来 |
| `data.seen_masks` | mask 形状 / nnz / 哈希 | 评测过滤口径是否一致 |
| `id_mapping.sha256` + `sample` | 映射摘要与样例 | 训练/推理 ID 不一致时能定位（完整映射在 `processed/id_mappings.json`） |
| `code.sha256` / `package_version` | `src/recsys/**/*.py` 内容哈希 | 没有 git 也能发现代码漂移 |
| `environment` | Python / torch / numpy / faiss 版本 | 复现环境 |

CLI 评测时会先打印一行 `provenance: data=… code=… users=… items=… created_at=…`，
让「这次数字是哪份数据、哪份代码跑出来的」和指标出现在同一处。

## 扩展点

**加数据集**：在 `src/recsys/data/` 加读取器 → 产出 `PreparedData` → 在
`recsys.data.loader.SUPPORTED_DATASETS` 与 `prepare_dataset` 注册 → 加 `configs/datasets/*.toml`。

**加召回模型**：自研放 `src/recsys/models/` + `src/recsys/training/`；第三方保持 `third_party/` 原样，
写适配器把 `PreparedData` 转成上游格式，**不要改上游训练代码**。

**加排序模型**：实现 `RankingAdapter` 的 `fit/predict/save/load`，特征处理放在数据层而不是适配器里。

**加指标**：只在 `recsys.evaluation.metrics` 里加函数并在 `compute_ranking_metrics` / `compute_ranking_model_metrics` 汇总，
所有模型自动获得新指标。

## 已知限制

- KuaiSAR 读 `data/kuasar_small/processed/` 的 npy 存储（`kuasar_store --check` 可校验），
  KuaiRec 仍直接解析原始 CSV；
- 推荐/搜索/广告三套系统各有独立数据与索引目录，跨业务只通过平台层（实验记录、预算、RAG）连接；
- 需要 Linux GPU 的模型入口（TorchEasyRec 原实现、OpenOneRec、MMRec 上游、TFRS）仍未在本机环境运行，
  只保留源码、适配说明与替代实现，见 [dependencies.md](dependencies.md) 与 [phase5-integration.md](phase5-integration.md) §8；
- 推荐侧小样本的 Top-20 命中问题属效果专项（[top20-diagnosis.md](top20-diagnosis.md)），不是链路未打通。

> 本文档早于 Phase 1 v2 / Phase 5；候选融合、重排、统一推荐链路已经交付，见
> [recsys-pipeline.md](recsys-pipeline.md)；三业务集成与演示见 [phase5-integration.md](phase5-integration.md)。
