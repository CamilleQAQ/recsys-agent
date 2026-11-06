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

仓库里没有正式实验记录（`artifacts/` 目前只有 KuaiSAR 的勘查与画像 JSON，以及少量冒烟产物）。
第一次正式实验请按上面的结构建目录，并把一句话结论回写到 `docs/models/<id>.md` 的「实验记录」表。
