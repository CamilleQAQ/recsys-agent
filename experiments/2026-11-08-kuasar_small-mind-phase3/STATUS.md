# Phase 3 MIND 实验执行状态（2026-11-08，供 Codex 交接）

## 1. 协议核对（通过）

| 项 | 值 |
| --- | --- |
| 数据集 | `kuasar_small`（父数据集 kuasar） |
| 用户 / 物品 | 2,000 / 454,717 |
| 切分 | `per_user_last_two_requests` |
| 训练/验证/测试 | 748,674 / 5,775 / 4,275 |
| 切分 sha256 | train `c62df14c…`、val `00068116…`、test `d6b5c0e8…` |
| ID 映射 sha256 | `b7d230622f6b…` |
| 双塔基线 | `artifacts/kuasar_small/two_tower_features.pt`，sha256 `784a2ad68972884dac4b0a652d44bb4a4d802c5788517fec05b6a25c6c1f0c3b`，dataset/strategy/idmap 与当前数据一致 |

评测口径：全部 454,717 物品精确 Top-K；曝光过滤 `seen_items_for_split`；验证集历史 = train，测试集历史 = train + validation；K = 10/20。
分组规则（看测试指标前固定）：训练集正交互数 `<100` / `100–299` / `≥300`；用户 704 / 622 / 674，目标 1,493 / 1,346 / 1,436（合计 4,275 ✓）。

## 2. 实际执行

| 时间 | 事件 |
| --- | --- |
| 10:13 | 冒烟：1 epoch × 20 batch，`mind_p3_smoke.pt`；之后验证分群评测入口可用 |
| 10:18–12:03 | A/B/C 三组并行（各 6 线程）跑「40 epochs」；A 到 24 轮、B 到 25 轮、C 到 21 轮时发现三进程换页卡死（每轮验证分配 `[2000, 454717]` 打分矩阵 ≈ 3.6 GB/进程），杀掉 |
| 12:33 | 修复：新增 `rank_users()`（按用户批取 Top-K，不再物化全量分数），训练入口加 `epoch_history` 与计时；冒烟 `mind_p3_smoke2.pt` 通过 |
| 12:35–12:47 | 重启 A/B/C（同样 40 epochs），只跑到 epoch 1，随后为测速再次停止 |
| 12:47–14:00 | 测速与方案评估；当前 **0 个 python 进程在跑** |

## 3. 当前产物

**Checkpoint（artifacts/kuasar_small/）**

| 文件 | 轮次 | 测试 Recall@20 | 说明 |
| --- | --- | --- | --- |
| `mind.pt` | 8 | 0.002183 | 早期实验 |
| `mind20.pt` | 20 | **0.005675** | 配置 A（K=4, L=20, d64, b1024, lr1e-2）20 轮；best_epoch 20，验证 R@20 0.00765 |
| `mind_p3_smoke.pt` / `mind_p3_smoke2.pt` | 1 | — | 冒烟，不可用于结论 |

`mind_p3_A42/B42/C42.pt` **均不存在**：`train_mind.py` 只在训练结束时保存，中断即丢失权重。

**日志（experiments/2026-11-08-kuasar_small-mind-phase3/logs/）**

| 文件 | 最后完成轮次 | 观测到的最好验证 Recall@20 |
| --- | --- | --- |
| `A42.pre-restart.log` | 24 | 0.008758 |
| `B42.pre-restart.log` | 25 | 0.006892 |
| `C42.pre-restart.log` | 21 | 0.007133 |
| `A42/B42/C42.log`（重启后） | 1 | 0.000000 / 0.000000 / 0.000125 |

重启前后的曲线不可拼接；上述数字仅作「中断前观察」，不构成同预算对照。

**分群评测（artifacts/_eval/）**

- `two_tower_features_test.json`（已生成，可复现）：总体 R@20 0.004408 / HR@20 0.0085 / NDCG@20 0.002836；
  low R@20 0.003717、mid 0.003537、high 0.005935。
- `two_tower_features_validation.json`（已生成）。
- MIND 的分群评测**尚未生成**（等 checkpoint）。

## 4. 代码改动（未提交）

| 文件 | 改动 |
| --- | --- |
| `src/recsys/models/mind.py` | 新增 `rank_users()`（批内 Top-K，峰值 ~116 MB 而不是 3.6 GB）；`score_users` 向量化保留 |
| `src/recsys/training/train_mind.py` | 每轮记录 `epoch_history`（loss / 验证指标 / 秒数）与 `train_seconds`；验证改走 `rank_users` |
| `src/recsys/adapters/rechub_mind.py` | `recommend()` 改走 `rank_users` |
| `scripts/eval_groups.py` | 新增：按活跃度分群评测，输出 JSON（双塔/MIND 通用） |
| `scripts/mind_phase3_report.py` | 新增：汇总逐轮轨迹、三配置、三 seed、分群与双塔差值，写 `metrics.json` |
| `experiments/2026-11-08-kuasar_small-mind-phase3/` | `config.toml`、`protocol.json`、`logs/` |

最近提交：`b276246 feat(mind): 用 torch-rechub 接入 MIND 多兴趣召回`。

## 5. 实测成本（为什么停下来）

| 配置 | 每步 | 每 epoch（729 步） | 40 epochs |
| --- | --- | --- | --- |
| 单进程 16 线程（fp32） | 0.458–0.577 s | 5.6–7 min | 3.7–4.7 h |
| 三进程并行（各 6 线程） | ~0.82 s | ~10 min | **6.7 h** |

- `torch.compile`：0.462 s/step（无加速）；bf16 autocast：1.556 s/step（更慢）。
- 完整协议（A/B/C 40 轮 + 胜出配置 seed 43/44 各 40 轮）≈ **7–8 h 墙钟**。

## 6. 与验收标准的差距

| 验收项 | 状态 |
| --- | --- |
| 三组超参同预算对照 | ✗（只有 24/25/21 轮的部分日志，无权重） |
| 胜出配置三个 seed 的 checkpoint | ✗（0 个） |
| 收敛检查（40 轮平台期） | ✗（20 轮版本仍在上升，只能给「预算内未观察到平台期」的弱结论） |
| 分群评估（MIND vs 双塔） | 双塔已完成；MIND 未做 |
| metrics.json / report.md / checkpoint.txt | 未生成 |
| 台账、zoo 同步 | 未更新 |

## 7. 两条可选路径

**P1（省时，约 1–1.5 h）**：按 `docs/mind-phase3-stop-loss-prompt.md` 的止损方案执行——训练预算 0，用现有 `mind20.pt` + 双塔做分群评测，写阶段报告，明确未完成项。

**P2（完整，约 3 h）**：统一改成 20 epoch 预算——A 复用 `mind20.pt`；B/C 各跑 20 轮（并行约 1.5 h）；胜出配置补 seed 43/44（各 20 轮，约 1.5 h）；随后分群评测 + 汇总。可满足除「40 轮预算」外的全部验收项。

## 8. 复现命令

```powershell
# 协议核对 / 指纹
python -B scripts/eval_groups.py --model two_tower --checkpoint artifacts/kuasar_small/two_tower_features.pt --split test --output artifacts/_eval/two_tower_features_test.json

# 20 epoch 配置 A（已有 mind20.pt，可跳过）
python -u -B -m recsys.training.train_mind --dataset kuasar_small --epochs 20 --batch-size 1024 --embed-dim 64 --max-length 20 --interest-num 4 --seed 42 --checkpoint-name mind20

# B / C（20 轮）
python -u -B -m recsys.training.train_mind --dataset kuasar_small --epochs 20 --batch-size 1024 --embed-dim 64 --max-length 20 --interest-num 2 --seed 42 --checkpoint-name mind_p3_B42
python -u -B -m recsys.training.train_mind --dataset kuasar_small --epochs 20 --batch-size 1024 --embed-dim 64 --max-length 40 --interest-num 4 --seed 42 --checkpoint-name mind_p3_C42

# 分群评测 + 汇总
python -B scripts/eval_groups.py --model mind --checkpoint artifacts/kuasar_small/mind20.pt --split test --output artifacts/_eval/A42_test.json
python -B scripts/mind_phase3_report.py --winner A --seeds 42 43 44
```
