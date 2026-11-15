# Phase 1 三个已确认问题修复记录

日期：2026-11-09
范围：只修以下三项，保留原有未提交改动与历史实验工件。

## 1. 排序训练历史泄漏

**改动：**

- [kuasar_ranking.py](../../src/recsys/data/kuasar_ranking.py)：排序工件为每个请求保存 `timestamps`
  （`{split}_timestamps.npy`，长度 = 请求数），schema 标记 `schema_version=2` /
  `has_request_timestamps=True`；`load_ranking_samples(..., require_timestamps=...)`
  对旧工件返回 `None` 或给出重建命令。
- [samples.py](../../src/recsys/ranking/samples.py)：`RankingView` 增加行级 `timestamps`；
  `build_history_source` / `histories_for_requests` / `histories_for_split_users`
  按 `(user, request_timestamp)` 用 `searchsorted(..., side="left")` 构造历史，
  严格排除同一时刻；`load_ranking_view(require_timestamps=True)` 对旧工件抛
  `RankingTimestampError`。
- [train_neural_ranker.py](../../src/recsys/training/train_neural_ranker.py)：DIN / DIEN /
  Ultra-HSTU 强制要求请求时间戳；训练视图和验证视图都按每条请求构造历史，同请求候选共享
  同一份；checkpoint 记录 `history_protocol="strictly_before_request"`。DCNv2 不需要时间戳，
  旧工件兼容；LightGBM 路径未改。
- [neural_ranking.py](../../src/recsys/adapters/neural_ranking.py)：推理时按任务给出的请求时间戳
  过滤历史；未给时使用用户最早留出目标时间（保守上界，避免未来信息）。

**数据重建（已执行）：**

```powershell
python -m recsys.data.kuasar_ranking --force
```

重建后行数不变：train 14,353,430 / validation 113,902 / test 88,417；请求数
3,104,325 / 24,128 / 24,128；新增时间戳数组与请求数严格对齐。

**验证：**

- 合成样例 t=1/2/3，请求 t=2：历史只含 t=1；t=1 无历史；t=4 含 t=1,2,3；
  同一请求的多个候选得到同一份历史。
- 旧工件（无 timestamps）：DIN 训练入口报 `RankingTimestampError` 并给出
  `python -m recsys.data.kuasar_ranking --force`；DCNv2 仍可训练/读取。
- 新训练 checkpoint 的 `history_protocol=strictly_before_request`。

**旧权重/旧指标：** `artifacts/kuasar_small/{din,dien,ultra_hstu}.pt` 仍是修复前逻辑训练的
权重，哈希未变；本轮修复验证权重写到独立文件名：
`artifacts/kuasar_small/repair_ranker/kuasar_small/{model}_history_fixed.pt`。
旧指标不能当作新逻辑结果。

## 2. include_validation_in_test_history 真正生效

**改动：**

- [pipeline/recall.py](../../src/recsys/pipeline/recall.py)：`run_channel` 增加开关，并传到
  SASRec/BERT4Rec/HSTU 的 `build_evaluation_examples`、MIND 的
  `build_observed_histories`、TIGER 的 `generate_tiger_scored`。
- [tiger_evaluator.py](../../src/recsys/evaluation/tiger_evaluator.py)：生成/评测函数增加开关。
- [adapters/rechub_mind.py](../../src/recsys/adapters/rechub_mind.py)、
  [adapters/tiger.py](../../src/recsys/adapters/tiger.py)：直接调用适配器时也支持开关。
- [pipeline/chain.py](../../src/recsys/pipeline/chain.py)：把 `run.include_validation_in_test_history`
  传给召回和神经排序；`rank_fused` 用同一开关和请求时间戳。
- [pipeline/ranking.py](../../src/recsys/pipeline/ranking.py)：神经排序 `score_candidates`
  接收开关与请求时间戳；LightGBM 分支不变。
- [adapters/neural_ranking.py](../../src/recsys/adapters/neural_ranking.py)：按开关选择
  train 或 train+validation 历史，并按请求时间过滤。
- [rec_tools.py](../../src/recsys/experiments/rec_tools.py)：`recall_models`、`rank_candidates`、
  `run_pipeline` 增加 `include_validation_in_test_history`；`run_pipeline` 写入
  `run_meta.json`。
- [rec_agent.py](../../src/recsys/experiments/rec_agent.py)：`--no-validation-in-test-history`
  传入确定性计划和工具调用。

**默认口径：** 默认 True；test 下 True = train+validation，False = train；
validation 始终只用 train；已曝光过滤规则不变。

**验证：**

- 同一用户 true/false：
  - 序列模型输入 `[1,2]` vs `[1]`；
  - MIND 历史 `[0,1]` vs `[0]`；
  - TIGER 历史 token 含验证物品 vs 不含；
  - DIN 输入含验证历史 vs 不含；
  - 推荐主链 sequence 通道的模型输入随开关变化；
  - Agent `recall_models` 的模型输入随开关变化；
  - Agent `run_pipeline` 的 `run_meta` 记录 False。

## 3. Rec Agent 训练权重隔离

**改动（[rec_tools.py](../../src/recsys/experiments/rec_tools.py)）：**

- 每次 `train_model` 创建唯一目录
  `<experiment_dir>/training/<model>-<UTC时间>-<uuid>/`，所有训练入口
  （Two-Tower/MF/SASRec/MIND/LightGCN/DSSM/BERT4Rec/HSTU/RQ-VAE/TIGER/MMRec/
  DIN/DIEN/DCNv2/Ultra-HSTU/LightGBM）都写到该目录。
- TIGER 优先使用同一任务内训练的 RQ-VAE 码表，否则才回退业务默认 `rqvae.pt`。
- 训练成功后才把 `model_id → 实际 checkpoint` 写入任务状态；后续
  `resolve_checkpoint` / `resolve_ranker_checkpoint` 优先使用任务内权重。
- 训练失败不更新引用；LightGBM 在 `run_pipeline` 内现训时也写唯一目录，不碰业务默认文件。
- Agent 总结新增 `trained_checkpoints`。

**验证：**

- 训练 DIN 与 LightGCN：各自生成唯一目录，返回实际路径；
- 训练前后 `artifacts/kuasar_small/din.pt`、`lightgcn.pt` 的 sha256 完全一致；
- 任务内 DIN 可重新加载，`rank_candidates` 返回的 checkpoint 就是本次训练的路径；
- 训练失败（如缺 RQ-VAE 的 TIGER）不改变任务内权重引用；
- 空任务上下文仍按业务默认权重加载。

**受影响权重是否重训：**

- 业务默认 DIN/DIEN/Ultra-HSTU **未重训**，仍为修复前逻辑；哈希保持原值。
- 修复逻辑的验证权重为 `repair_ranker/kuasar_small/*_history_fixed.pt`。
- LightGBM 默认 `mvp/lgbm_ranker.txt` 未改动。

## 4. 检查命令

```powershell
$env:PYTHONPATH = "src"
python scripts/test_phase1_fixes.py
python scripts/smoke_phase1.py
python -m recsys.experiments.rec_agent --self-check
python scripts/zoo.py check
python scripts/check_project.py
```

`test_phase1_fixes.py` 覆盖上述三组验证；`smoke_phase1.py` 覆盖模型接入回归。
