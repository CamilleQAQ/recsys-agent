# Phase 1（v2）完整推荐系统与 Rec Agent — 总结报告

日期：2026-11-09
范围：`PROJECT_MASTER_v2.md` Phase 1（推荐系统 + Rec Agent）。搜索、广告、Master Agent 不在本轮。

## 0. 一句话结论

**功能交付完成：19 个推荐模型达到 `ready`，统一推荐链路和 Rec Agent 均已跑通真实模型；效果优化与 Top-20 修复未完成，已定位并记录，不冒充 Phase 1 效果验收。**

## 1. 本轮交付了什么

- 推荐模型清单全部有处理结果：19 个 `ready`、2 个 `upstream_ready`（DeepFM/MMoE，留给广告）、
  4 个 `blocked` 且有可执行替代（OneRec、TFRS Two-Tower、TFRS factorized_top_k、TorchEasyRec SID）。
- 统一推荐链路 `python -m recsys.pipeline`：可配置多路召回、融合、排序、重排、逐阶段评估。
- 默认配置复现既有双路（Two-Tower + MIND）行为，128 用户指标与旧 MVP 逐项一致。
- Rec Agent 12 个工具全部落到真实训练/推理/评估函数；确定性计划器与真实 LLM 计划器各跑通一次。
- Top-20 零命中用固定 128 用户样本定位到具体环节，记录在
  [../../docs/top20-diagnosis.md](../../docs/top20-diagnosis.md)。

## 2. 逐模型处理结果

| 模型 | 接入方式 | 训练/推理 | 链路 | Agent | 本轮证据 |
| --- | --- | --- | --- | --- | --- |
| Two-Tower | 既有 | 既有 checkpoint | ✓ 默认通道 | ✓ | 默认链路 128 用户通过 |
| MIND | 既有 torch-rechub | 既有 checkpoint | ✓ 默认通道 | ✓ | 默认链路 128 用户通过 |
| Popularity | 既有无参基线 | 现算热度 | ✓ | ✓ | Agent 自检真实召回 |
| BPR-MF | 既有 | Agent 训练 1 epoch/2 batch | ✓ | ✓ | checkpoint + 链路 verified |
| LightGCN | 新写 BPR 图传播 | 1 epoch/2 batch 冒烟 | ✓ | ✓ | `artifacts/kuasar_small/lightgcn*.pt` |
| DSSM | 复用 Two-Tower MLP 特征塔 | 1 epoch/2 batch 冒烟 | ✓ | ✓ | `dssm.pt` |
| SASRec | 既有 | 1 epoch/2 batch 冒烟 | ✓ | ✓ | `sasrec.pt` |
| BERT4Rec | 新写 PyTorch Cloze | 1 epoch/2 batch 冒烟 | ✓ | ✓ | `bert4rec.pt` |
| HSTU | 新写相对偏置 + gating | 1 epoch/2 batch 冒烟 | ✓ | ✓ | `hstu.pt` |
| TIGER | 新写 SID 生成 + trie beam | 1 epoch/2 batch 冒烟 | ✓ | ✓ | `tiger.pt`（43,040 SID） |
| RQ-VAE / Semantic ID | 新写 RQ-VAE + RQ-KMeans 兜底 | 4 epoch 冒烟 | ✓（TIGER 依赖） | ✓ | `rqvae.pt`；RQ-VAE 唯一 83、KMeans 43,040 |
| MMRec 多模态族 | VBPR 公式纯 PyTorch 适配 | 1 epoch/2 batch 冒烟 | ✓ | ✓ | `mmrec_vbpr.pt`；输入是元数据向量 |
| Ultra-HSTU | HSTU 主干 + 排序头 | 1 epoch/2 batch 冒烟 | ✓ | ✓ | `ultra_hstu.pt` |
| DIN | 纯 PyTorch 注意力池化 | 1 epoch/2 batch；AUC 0.5143 | ✓ | ✓ | `din.pt`；Agent 训练 + 排序 |
| DIEN | 纯 PyTorch AUGRU | 1 epoch/2 batch；AUC 0.5849 | ✓ | ✓ | `dien.pt` |
| DCNv2 | CrossNetV2 + DNN | 1 epoch/2 batch；AUC 0.5136 | ✓ | ✓ | `dcnv2.pt` |
| LightGBM LambdaRank | 既有 | 既有 checkpoint | ✓ 默认精排 | ✓ | 默认链路 + Agent 自检 |
| OneRec | 不接入 | — | — | — | blocked：缺外部权重/多卡；替代 TIGER |
| TFRS Two-Tower | 不接入 | — | — | — | blocked：TF2 冲突；替代项目 Two-Tower |
| TFRS factorized_top_k | 不接入 | — | — | — | blocked：同上 |
| TorchEasyRec SID | 不接入 | — | — | — | blocked：tzrec 平台；替代项目 RQ-VAE |

完整状态（源码/已核源码/适配/可运行/业务接入/Agent/已评估）见
[../../models/zoo.toml](../../models/zoo.toml) 与 [../../docs/model-zoo.md](../../docs/model-zoo.md)。

## 3. 推荐链路

```powershell
$env:PYTHONPATH = "src"
python -m recsys.pipeline list
python -m recsys.pipeline run --config configs/pipeline/kuasar_small_default.toml
python -m recsys.pipeline evaluate --artifacts artifacts/kuasar_small/pipeline --split validation
```

- 阶段：多路召回 → 去重融合（round_robin/rrf/quota）→ LightGBM/神经排序 → 类目/作者重排 → 输出/评估。
- 保留：候选来源、通道内名次、每阶段模型分数、融合名次、最终名次。
- 工件：`final.json` / `recall_<channel>.json` / `fusion.json` / `ranked.json` /
  `metrics.json` / `diagnostics.json` / `run_meta.json`（含 checkpoint sha256 与数据指纹）。
- 默认复现：128 用户、每路 100、融合 100、最终 20、LightGBM 20 轮/10k 行、类目上限 5。

实跑结果：

- 默认双路 128 用户：通过，6.3 s；128 用户固定样本与既有
  `artifacts/kuasar_small/mvp/metrics.json` **逐项一致**
  （Coverage 0.005566 / 0.005557 / 0.005395 / 0.005412；Top-100 命中 2/7/9）。
- 多路（Two-Tower + MIND + LightGCN → RRF → DIN → 类目/作者重排）4 用户：通过。
- 既有 `recsys_mvp.py` 与 `experiments/2026-11-08-kuasar_small-recsys-mvp/` 未被改动。

配置与实现说明见 [../../docs/recsys-pipeline.md](../../docs/recsys-pipeline.md)。

## 4. Rec Agent

```powershell
python -m recsys.experiments.rec_agent --self-check
python -m recsys.experiments.rec_agent `
  --task "用双塔和 MIND 召回，LightGBM 排序并评估" --users 16 `
  --recall two_tower,mind --ranker lightgbm_ranker
python -m recsys.experiments.rec_agent --llm `
  --task "查询数据画像和模型库，然后用 popularity 召回和 lightgbm 排序跑一次推荐链路并报告真实指标" `
  --users 4 --max-tool-calls 6
```

- 12 个工具：画像、Model Zoo、工件、训练、召回、排序、重排、评估、全链路、融合配额、
  历史实验、实验列表。
- LangGraph：plan → execute → plan → … → finish；预算、重复调用、失败原因落盘。
- 确定性计划器：自检 7/7；训练 DIN、BPR-MF、LightGCN 并跑链路均 verified。
- 单独工具链：`recall_models(two_tower)` → `rank_candidates(din)` → `rerank_candidates`
  → `evaluate_recommendations(final)`，4 个工具全部 `ok=true`，状态在 `ToolContext` 串联。
- 真实 LLM：`deepseek-chat` 4 请求 / 4 工具（画像 → Zoo → 链路 → 评估），14.5 s，verified
  （`experiments/2026-11-09-kuasar_small-rec-agent/run_llm_final/`）。

说明见 [../../docs/rec-agent.md](../../docs/rec-agent.md)。

## 5. Top-20 零命中

> 该样本只用于定位链路环节，**不是模型能力/代码正确性/接入验收的结论**。KuaiSAR 全量
> 14.6M 交互 / 4M 物品，本机无法为每次接入做全量训练；128 用户 + 长尾 + 2k 子集本身就会
> 导致零命中，数据规模与效果问题应进入专项。

固定 128 验证用户 / 362 目标：

- 353（97.51%）目标不在任何通道 Top-100；
- 5 个在通道并集但未进入融合池（通道名次 56/62/70/99/99）；
- 4 个进入融合池但被 Top-20 截断（融合名次 46/73/79/80）；
- 排序/重排没有再丢目标；
- 目标物品 41.4% 在训练集零点击，中位点击 1 次。

根因：ID-only 双塔/MIND 对长尾冷物品召回不足；轮流融合 `limit=100` 只覆盖每路约前 50；
融合后先截 Top-20 再排序。修复动作（扩大通道深度、先并集再去重、排序前不截断、增加内容/
Semantic ID 探索通道）留给效果专项。详见 [../../docs/top20-diagnosis.md](../../docs/top20-diagnosis.md)。

## 6. 检查记录

| 命令 | 结果 |
| --- | --- |
| `python scripts/zoo.py check` | 25 条；ready=19 / upstream=2 / blocked=4；通过 |
| `python scripts/check_project.py` | 0 错误、9 警告 |
| `python scripts/smoke_adapters.py` | 既有适配器、指标、溯源、LightGBM 通过 |
| `python scripts/smoke_phase1.py` | 11 召回适配器小样本推理/保存加载往返、5 排序器打分/往返、Agent 工具链、默认/多路链路、Agent 自检全部通过 |
| `python -m recsys.experiments.rec_agent --self-check` | 7/7 |
| `python -m recsys.experiments.rec_agent --llm ...` | 4 请求 / 4 工具，verified |
| `python scripts/diagnose_top20.py --users 128` | 完成并落盘 |
| `python -m recsys.pipeline evaluate ...` | 对已落盘 `final.json` 复算指标成功 |

新模型全部只做 1 epoch / 2 batch（RQ-VAE 4 epoch）冒烟，指标为 0 或接近随机，**不是效果结论**。

## 7. 阻碍

1. OneRec 缺外部权重/多卡；替代 TIGER。
2. TFRS 依赖 TensorFlow 2，与主环境冲突；替代项目 Two-Tower。
3. TorchEasyRec 运行栈（tzrec）不可安装；DSSM/HSTU/Ultra-HSTU/SID 用纯 PyTorch 重写，
   checkpoint 不通用。
4. MMRec 缺 torch_geometric 与视觉/文本特征；当前 VBPR 适配用元数据向量。
5. RQ-VAE 码本塌缩，已用 RQ-KMeans 导出可用码表，但 RQ-VAE 本身仍需调。
6. Top-20 零命中未修复，已定位并给出动作。
7. 新模型仅冒烟，未做正式训练/调参/多种子/Benchmark。
8. 真实 LLM 仅验证一次，限流/断网回退未覆盖。

## 8. 工件与日志

- 推荐链路：`artifacts/kuasar_small/pipeline/`（默认 128 用户）、
  `artifacts/kuasar_small/pipeline_smoke/`（16 用户冒烟）、
  `artifacts/kuasar_small/top20_diagnosis/`。
- 模型 checkpoint：`artifacts/kuasar_small/*.pt`、`artifacts/kuasar_small/mvp/lgbm_ranker.txt`。
- Agent 运行：`experiments/2026-11-09-kuasar_small-rec-agent/{run_default,run_din,run_mf,run_llm_final,self_check,quota_tool_check,train_tool_check}/`。
- 训练冒烟日志：`artifacts/_phase1_train_smokes*.log`、`artifacts/_train_dssm.log`。
- 汇总指标：[metrics.json](metrics.json)。
