# 清理报告（2026-11-04，2026-11-05 更新）

## 结果

| 指标 | 清理前 | 清理后（2026-11-05） |
| --- | --- | --- |
| 第三方仓库数 | 11 | **13**（新增 FuxiCTR、genrec，见文末） |
| 第三方文件数 | 6,376 | **6,875** |
| 第三方体积 | **400.7 MB** | **108.8 MB** |
| 第三方仓库层级 | `third_party/recommender_repos/<repo>` | `third_party/<repo>` |
| 每个仓库的 .git | 11 个浅克隆 | 全部移除，commit 记入 `third_party/SOURCES.toml` |

> 另含不入库的运行时数据（`data/` 约 45 MB 的 MovieLens 下载与 `artifacts/` 的小 checkpoint），
> 由 `.gitignore` 排除，不属于交付内容。

回收 **296.7 MB / 26 个条目**，构成如下。

## 删了什么

### 1. 嵌套 VCS 元数据：129.7 MB

11 个浅克隆（depth=1）的 `.git` 目录。这些仓库只取了一个 commit，保留 `.git` 没有历史价值，
却会让根目录未来 `git add` 时把整个第三方仓库变成 gitlink（子模块），反而更难管理。

来源信息没有丢：URL、默认分支、完整 commit、许可证、克隆深度都在
`third_party/repository_manifest.csv` 与 `third_party/README.md` 里。
需要升级某个仓库时，按记录的 commit 重新克隆即可。

### 2. 随仓库携带的大型示例语料：167.0 MB

| 路径 | 大小 | 说明 |
| --- | --- | --- |
| `BERT4Rec/data/ml-20m.zip` | 50.5 MB | MovieLens-20M 原始包 |
| `BERT4Rec/data/steam.txt` | 39.8 MB | Steam 交互语料 |
| `BERT4Rec/data/ml-1m.txt` | 9.6 MB | 与 SASRec.pytorch 里的同名文件**完全重复** |
| `BERT4Rec/data/beauty.txt` | 4.1 MB | Amazon Beauty 语料 |
| `SASRec.pytorch/python/data/Steam.txt` | 44.1 MB | Steam 语料 |
| `SASRec.pytorch/python/data/ml-1m.txt` | 9.6 MB | 重复文件 |
| `SASRec.pytorch/python/data/Beauty.txt` | 4.7 MB | Amazon Beauty 语料 |
| `SASRec.pytorch/python/data/Video.txt` | 3.3 MB | 视频语料 |
| `SASRec.pytorch/python/data/wikipedia.txt` | 1.4 MB | 维基语料 |

保留：`gen_data*.py`、`run_*.sh`、`utils.py` 等**数据处理与运行脚本**；两个 `data/` 目录各留一份说明，
指向数据来源与替代方案。上游脚本仍可运行，只是需要自己放数据。

### 3. IDE / 构建缓存：< 0.1 MB

`generative-recommenders/.idea`、`MMRec/.idea`、`generative-recommenders/.../triton_aot/build`。

### 4. 重复与冗余

清理前全仓哈希扫描出 **377 组重复文件**。真正跨仓库的重复只有一份：`ml-1m.txt`
（BERT4Rec 与 SASRec.pytorch 各一份，同一内容），随示例语料一起删除。

其余重复属于**上游自身结构**，刻意保留：

- `RecBole/dataset/ml-100k` 与 `RecBole/recbole/dataset_example/ml-100k`：后者是随包发布的示例，测试与文档会直接引用；
- `LightGBM/examples/*` 下的 `rank.train`、`binary.train` 等：同一份数据被不同示例复用，是上游组织方式；
- `OpenOneRec/verl_rl` 与 `verl_distillation` 下的同名文件：上游把 verl 派生了两份，删任何一份都会破坏其中一个子项目。

结论：**只删「跨仓库重复 + 与代码无关的大文件」**，不动上游内部结构。

## 没动什么（以及为什么）

| 保留项 | 原因 |
| --- | --- |
| `.github/`、`.ci/`、`tests/`、`examples/`、`docs/` | 上游工程文件，用于理解构建/测试/用法；体积小 |
| `LICENSE`、`THIRD_PARTY_NOTICES` | 合规必需，逐个保留在仓库根 |
| 各仓库 `README.md` | 原始用法说明，是「训练入口」的权威来源 |
| 未初始化的子模块空目录（`faiss` 无、`LightGBM/external_libs/*`、`generative_recommenders/ops/cpp/cutlass`） | 保留空目录可提示「此处源码缺失」，删除反而掩盖问题 |
| `OpenOneRec` 等仓库的图片资源 | 文档插图，体积可控（单张 ≤ 3.7 MB） |

## 结构变化

1. `third_party/recommender_repos/*` → `third_party/*`（去掉无意义的一层嵌套）；
2. 原 `recommender_repos/` 下的审计材料（`model_mapping.md`、`audit_report.md`、`audit_evidence.json`、
   `inventory.json`、`build_audit.py`、`repository_manifest.csv`）随仓库一起上移到 `third_party/`，
   相对链接保持有效；
3. 原 `recommender_repos/README.md` 被重写为 `third_party/README.md`（补上本地改动、许可证、可用性分级）；
4. 新增 `docs/`、`configs/`、`models/`、`scripts/`、`experiments/`、`requirements/`、`data/`、`artifacts/`；
5. 自研代码位置不变（`src/recsys/`），仅新增 `src/recsys/adapters/`。

## 清理后的自检

```powershell
python scripts/zoo.py check        # 索引 ↔ 源码路径 ↔ 许可证
python scripts/check_project.py    # 配置一致性、文档链接、残留缓存
python scripts/smoke_adapters.py   # 适配层端到端（合成数据，无需真实数据集）
```

## 遗留问题

- `OpenOneRec` 根目录没有 `LICENSE`（README 声明 Apache-2.0），已在 `third_party/LICENSES.md` 标注「再分发前需确认」；
- `LightGBM`、`generative-recommenders` 的子模块未拉取，源码编译不完整（训练走 pip 安装即可，不依赖本地编译）；
- 空目录 `BERT4Rec/data`、`SASRec.pytorch/python/data` 各放了一份 3 行的说明文件（本项目对上游的唯一内容改动，已记录）。

## 后续新增（2026-11-05）

按 [PROJECT_MASTER.md](../PROJECT_MASTER.md) §4 点名的模型来源补齐两个缺失仓库（TorchRec 按决定后置）：

| 仓库 | 上游 | 分支 | Commit | 许可证 | 体积 | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| FuxiCTR | [reczoo/FuxiCTR](https://github.com/reczoo/FuxiCTR) | main | `183da1c944ba8930fcdcd54114837c081959d748` | Apache-2.0 | 2.6 MB / 593 文件 | Phase 7–8：DIN / DCNv2 的正式实现来源 |
| genrec | [phonism/genrec](https://github.com/phonism/genrec) | main | `2676b5b08e2efc0b873ab4c3d66882611a62ca2e` | MIT | 2.4 MB / 281 文件 | Phase 10：TIGER（含 RPG / COBRA / LC-Rec / RQ-VAE） |

处理方式与其余仓库一致：`git clone --depth 1` → 记录 commit/分支/许可证到 `third_party/SOURCES.toml`
→ 删除 `.git`。未安装依赖、未运行上游脚本。

**未补齐**：TorchRec（文档里 Two-Tower 的来源）。理由：TorchRec 主要解决大规模 embedding 分片与
分布式训练，Two-Tower 本身已有自研实现，且当前阶段用不上；等 Phase 2 需要规模化训练时再评估。

同时新增 `third_party/SOURCES.toml` 作为**维护版**来源清单（覆盖 13 个仓库），
`repository_manifest.csv` 保留为 Phase 0 审计快照；两者差异在 `third_party/README.md` 说明。

清理账目更新：第三方部分 **108.8 MB / 6,875 文件**（清理前 400.7 MB / 6,376 文件）。

