# 实验记录

代码在 `src/recsys/experiments/`，这里是**实验产物与结论**的归档位置。

## 目录约定

```
experiments/
└── YYYY-MM-DD-<dataset>-<model>/
    ├── config.toml       本次运行的配置快照（从 configs/ 复制后按实际参数覆盖）
    ├── metrics.json      Recall/Precision/NDCG/HitRate/Coverage（+ AUC/LogLoss）
    ├── checkpoint.txt    checkpoint 路径、sha256 与 provenance 摘要
    └── report.md         实验报告（模板见下）
```

命名一律小写连字符；同一天多组实验加序号后缀（`...-v2`）。

## report.md 模板（PROJECT_MASTER §9.3）

```markdown
# <日期> <数据集> <模型> — <一句话实验目的>

## 假设
这次想验证什么？为什么这个模型/参数组合可能更好？

## 设置
- 数据版本：<provenance.data.split_sha256.test 前 12 位>
- 划分与种子：last-two-strict / seed=42
- 候选集与负采样：全量物品 / in-batch（batch=256）
- 模型版本：<checkpoint 路径 + sha256>
- 运行环境：Python … / torch … / GPU …

## 结果
| 指标 | 本次 | 基线 | 变化 |
| --- | --- | --- | --- |
| NDCG@10 | | | |
| Recall@20 | | | |
| Coverage | | | |
| 训练耗时 / 单用户推理 | | | |

## 结论
- 结论必须能被数据支持；验证集结论不要写成测试集结论；
- 单次实验不要写成「稳定提升」；
- 计划中的功能不要写成已完成。

## 下一步
- 继续调参 / 换模型 / 结束（写清判断依据）
```

## 记录规则

- 指标必须来自 `recsys.adapters.evaluate` 或对应模型的评测脚本，不接受手抄；
- 必须写清：数据集、正反馈阈值、切分协议、候选集规模、设备——缺一项，这个数字以后就没法解释；
- 失败的实验也要记（`report.md` 里写清原因），避免重复踩坑；
- 权重不入库（`.gitignore` 已忽略 `*.pt/*.pth/*.ckpt`），只记路径、sha256 与 provenance。

## 当前状态

| 日期 | 运行 | 数据集 | 一句话结论 |
| --- | --- | --- | --- |
| 2026-11-07 | [kuasar_small-two_tower](2026-11-07-kuasar_small-two_tower/report.md) | kuasar_small（2,000 用户） | 链路跑通；测试 Recall@20 0.0035（≈37× 随机）；FAISS IndexFlat 比 torch 暴力检索慢 3.9× |
| 2026-11-07 | [kuasar_small-two-tower-features](2026-11-07-kuasar_small-two-tower-features/report.md) | kuasar_small（2,000 用户） | 物品侧特征入塔：Recall@10 **翻倍**（0.0016 → 0.0032）、Recall@20 +26.9%；验证/测试在 Recall@10 上结论不一致，需多 seed |
| 2026-11-07 | [kuasar_small-two-tower-sweep](2026-11-07-kuasar_small-two-tower-sweep/report.md) | kuasar_small（2,000 用户） | 超参扫描：选定 d64/b2048/lr1e-2；**8 epoch 扫描把 d128 排第一，20 epoch 后两者只差 6% 且 NDCG 反转**——短预算选参不可靠 |
| 2026-11-07 | [kuasar_small-mind](2026-11-07-kuasar_small-mind/report.md) | kuasar_small（2,000 用户） | MIND（torch-rechub）作为第二路召回：测试 Recall@20 **0.005675**，比双塔 +28.8%；20 epochs 未收敛，训练贵 3 倍 |
| 2026-11-08 | [kuasar_small-mind-phase3](2026-11-08-kuasar_small-mind-phase3/report.md) | kuasar_small（2,000 用户 / 128 用户分群） | 修复 MIND 路由随机性；分群：mid 组 +0.0058、high 组 −0.0040；收敛/多 seed 未完成（P1 止损） |
| 2026-11-08 | [kuasar_small-recsys-mvp](2026-11-08-kuasar_small-recsys-mvp/report.md) | kuasar_small（128 验证用户） | 双路召回→名次融合→LightGBM→类目重排全链路跑通，10.0 s / 1,379 MiB；Top-20 指标为 0（样本量），Top-100 命中 9 用于核对链路 |
| 2026-11-08 | [kuasar_small-fusion-quota-agent](2026-11-08-kuasar_small-fusion-quota-agent/report.md) | kuasar_small（128 验证用户） | Phase 11 Agent MVP：配额 50/50→30/70→20/80 三组实评，30/70 命中 5（基准 4）；本地策略跑通，真实 LLM 待验收 |
| 2026-11-08 | [kuasar_small-llm-tool-call](2026-11-08-kuasar_small-llm-tool-call/report.md) | kuasar_small（128 验证用户） | 真实 LLM 原生 tool_calls **验收通过**：deepseek-chat 选 70/30，程序评测（命中 3 vs 基准 4）并同 `tool_call_id` 回传；2 请求/1 工具/3.05 s |

权重不入库，只记路径 / sha256 / provenance（见各目录的 `checkpoint.txt`）。
下一个实验按上面的结构建目录，并把一句话结论回写到 `docs/models/<id>.md` 的「实验记录」表。
