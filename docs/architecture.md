# 架构

## 分层

```
原始数据 (CSV/zip)
      │  recsys.data.kuairec / kuasar / movielens      ← 只负责「读 + 清洗 + 编码」
      ▼
PreparedData（train/val/test + IdMappings + 曝光 mask）
      │  recsys.data.dataset                            ← 只负责「按模型需要切样本」
      ├── 序列样本  (SASRec: input/positive/negative 序列)
      ├── 交互样本  (双塔/MF: (user, item) 对)
      └── 候选+特征样本（排序模型；Phase 1 未完成）
      ▼
模型 (recsys.models / third_party)
      │  recsys.training                               ← 训练循环、损失、负采样、早停
      ▼
checkpoint（权重 + model_config + 切分协议 + provenance + 指标历史）
      │  recsys.adapters                               ← 统一「加载 → 打分/编码 → 评测」
      ▼
Top-K 推荐 / 候选分数 → recsys.evaluation.metrics → 指标 JSON / 实验记录
```

分层的硬边界：

- **data 层不 import 模型**，模型层不碰文件路径；
- **训练脚本只写 checkpoint**，不直接产出实验报告；报告由 evaluation / experiments 生成；
- **adapters 不重新实现算法**，只做协议转换（必要时复用 evaluator 里的加载逻辑）。

## 与 PROJECT_MASTER §3 模块的对应

| 逻辑模块 | 仓库里的位置 | 状态 |
| --- | --- | --- |
| 数据层 | `src/recsys/data/` | 已有 KuaiRec/MovieLens；KuaiSAR 待 Phase 1 |
| 召回层 | `src/recsys/models/` + `third_party/` | 4 个模型已接入，另有 8 个 vendor 候选 |
| 候选融合层 | — | 未实现；适配层的 `retrieve()` 已按「通道」语义预留 |
| 排序层 | `LightGBMRankerAdapter` + FuxiCTR | 接口就绪，缺特征通道 |
| 重排层 | — | 未实现（Phase 9） |
| 评估层 | `src/recsys/evaluation/` | 召回指标 + AUC/LogLoss + 效率基准（faiss_evaluator） |
| Research Agent | — | 未实现（Phase 11+） |

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

- `prepare_dataset` 每次都重新解析原始文件（KuaiRec 约 1 GB），`processed/` 只写不读；
- 统一 CLI 会重新准备数据，跨模型批量评测需要外部循环；
- 没有候选融合与重排模块，端到端链路尚未打通（Phase 5 / 9）；
- TorchEasyRec / OpenOneRec / generative-recommenders / genrec 需要独立 Linux GPU 环境，本仓库只做源码归档与索引。
