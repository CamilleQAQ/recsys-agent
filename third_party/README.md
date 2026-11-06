# third_party：第三方原始代码

13 个推荐相关开源仓库，原样导入，**只读**。

- 不改上游代码（要打补丁请放 `patches/` 并在提交信息里说明原因）；
- 不在这里跑训练脚本的自动安装（依赖见 `requirements/` 与 [../docs/dependencies.md](../docs/dependencies.md)）；
- 原始 `.git` 已移除（浅克隆只有一个 commit，没有历史价值，且会干扰上层版本管理），
  来源与 commit 见 [SOURCES.toml](SOURCES.toml)。

## 仓库清单

| 目录 | 上游 | 分支 | Commit | 许可证 | 覆盖 | 可用性 |
| --- | --- | --- | --- | --- | --- | --- |
| [RecBole](RecBole/) | [RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole) | master | `7b02be5` | MIT | 协同过滤、图推荐、序列、CTR、知识图谱 | 独立环境装 recbole |
| [TorchEasyRec](TorchEasyRec/) | [alibaba/TorchEasyRec](https://github.com/alibaba/TorchEasyRec) | master | `2815a97` | Apache-2.0 | 双塔、MIND、HSTU、语义 ID、CTR | Linux-only wheel |
| [DeepCTR-Torch](DeepCTR-Torch/) | [shenweichen/DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch) | master | `9eccef7` | Apache-2.0 | CTR 交叉、DIN/DIEN、多任务 | 主环境可跑，缺特征通道 |
| [FuxiCTR](FuxiCTR/) | [reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR) | main | `183da1c` | Apache-2.0 | DIN、DIEN、DCNv2、DeepFM、MMoE/PLE | 需 parquet 特征表 |
| [LightGBM](LightGBM/) | [lightgbm-org/LightGBM](https://github.com/lightgbm-org/LightGBM) | main | `3ee03c2` | MIT | GBDT + LambdaRank | pip wheel 即可 |
| [faiss](faiss/) | [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | main | `9acf8b0` | MIT | Flat/IVF/PQ/HNSW 向量检索 | pip wheel 即可 |
| [genrec](genrec/) | [phonism/genrec](https://github.com/phonism/genrec) | main | `2676b5b` | MIT | TIGER、RPG、COBRA、LC-Rec、OneRec、RQ-VAE | torch 2.6 + 语义 ID 预处理 |
| [MMRec](MMRec/) | [enoche/MMRec](https://github.com/enoche/MMRec) | master | `c68ee7c` | GPL-3.0 | 多模态图推荐（22 个实现） | Python3.7 + torch1.11 |
| [SASRec.pytorch](SASRec.pytorch/) | [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch) | main | `26a8fa8` | Apache-2.0 | SASRec 第三方复现 | 代码可读，数据自备 |
| [BERT4Rec](BERT4Rec/) | [FeiSun/BERT4Rec](https://github.com/FeiSun/BERT4Rec) | master | `615eaf2` | Apache-2.0 | BERT4Rec 原作者实现 | TF1.12，需独立环境 |
| [generative-recommenders](generative-recommenders/) | [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders) | main | `b8e6ff0` | Apache-2.0 | HSTU、DLRMv3、ULTRA、MoL | Linux + CUDA + fbgemm_gpu |
| [OpenOneRec](OpenOneRec/) | [Kuaishou-OneRec/OpenOneRec](https://github.com/Kuaishou-OneRec/OpenOneRec) | main | `a969edc` | Apache-2.0（README 声明，**根 LICENSE 缺失**） | Qwen3 生成式推荐、ResKmeans | 多卡 Linux + 外部权重 |
| [recommenders](recommenders/) | [tensorflow/recommenders](https://github.com/tensorflow/recommenders) | main | `a8a90ae` | Apache-2.0 | TFRS：双塔任务、Top-K 检索层 | TF2 独立环境 |

commit 在上表做了缩写。完整 SHA、分支、许可、可用性结论、本地改动见
[SOURCES.toml](SOURCES.toml)（维护版）与 [repository_manifest.csv](repository_manifest.csv)
（Phase 0 静态审计快照，含实现文件与符号级证据，由 [build_audit.py](build_audit.py) 生成）。

## 目录里的其他文件

| 文件 | 作用 |
| --- | --- |
| [SOURCES.toml](SOURCES.toml) | **维护版**来源清单：目录、URL、分支、完整 commit、许可、覆盖、可用性、本地改动 |
| [repository_manifest.csv](repository_manifest.csv) | Phase 0 机读审计快照：URL、完整 commit、分支、许可、实现文件、符号（覆盖前 11 个仓库） |
| [model_mapping.md](model_mapping.md) | 逐文件的模型/组件映射与源码行号（跨仓库全量） |
| [audit_report.md](audit_report.md) | 静态审计：依赖、数据、训练/预测入口、权重与完整性 |
| [audit_evidence.json](audit_evidence.json) | 审计原始证据（依赖文件全文、定义行号、文件大小） |
| [inventory.json](inventory.json) | 克隆时全部跟踪文件索引 |
| [build_audit.py](build_audit.py) | 重新生成上述审计材料的脚本 |
| [LICENSES.md](LICENSES.md) | 许可证汇总与再分发注意事项 |

新增仓库时：在 `SOURCES.toml` 加一条 → 更新本页表格与 `LICENSES.md` →
`python scripts/check_project.py` 会校验「目录 ↔ 条目 ↔ 许可文件 ↔ commit 格式」是否一致。

## 本地改动记录

与上游相比，本目录只有以下差异，其余保持原样：

1. 删除全部 `.git`（浅克隆，commit 见 `SOURCES.toml`）；
2. 删除随仓库携带的大型示例语料（`BERT4Rec/data/` 与 `SASRec.pytorch/python/data/` 共 167 MB，
   清单见 [../docs/cleanup-report.md](../docs/cleanup-report.md)）；
3. 在上述两个 `data/` 目录各放一份 3 行说明，指向数据来源；
4. 删除 IDE 与构建缓存（`.idea`、`triton_aot/build`）。

没有任何源码文件被修改。如需修改上游实现，请在 `patches/` 存 diff 并在此处登记。

## 使用边界

- 这里是**参考实现与必要时的补丁基线**，不是默认安装来源：能用 pip wheel 就别编译本目录源码；
- 「克隆成功」不等于「能直接训练」：TorchEasyRec/OpenOneRec/generative-recommenders 需要 Linux GPU 环境，
  LightGBM/generative-recommenders 的子模块未拉取导致本地编译不完整；
- 各仓库许可不同（含 GPL-3.0），再分发或商用前请看 [LICENSES.md](LICENSES.md)。
