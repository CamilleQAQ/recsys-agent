# 项目进度与工作日志

> **用途**：记录「已经做完了什么 / 现在卡在哪 / 下一步做什么」。每轮开发后更新本文件。
> **与 [PROJECT_MASTER.md](PROJECT_MASTER.md) 的关系**：总档案是**大纲与基准**（定位、架构、13 个 Phase、
> 记录规范），不随开发进度改动；本文件是**执行台账**，只写事实、证据和下一步。
> **维护规则**：
> 1. 只写真实发生的事；没验证的不写成完成，验证集结果不写成测试集结果；
> 2. 每条「已完成」都要能点开证据（代码路径 / 文档 / 可复现命令）；
> 3. 待办用 `- [ ]`，完成的用 `- [x]` 就地勾选，不删除历史；
> 4. 每轮结束更新「当前状态快照」与「变更历史」两节。

**最后更新**：2026-11-06

---

## 一、当前状态快照

| 项目 | 状态 |
| --- | --- |
| 所处阶段 | **Phase 0 完成**（仓库审计 + Model Zoo + 工程底座），准备进入 **Phase 1**（KuaiSAR 数据层） |
| 版本控制 | git `main`，远端 [CamilleQAQ/recsys-agent](https://github.com/CamilleQAQ/recsys-agent)；首次提交 `0bef337`（2026-11-06） |
| 数据主线 | KuaiSAR 2.0：下载完成、**全量画像完成**（1.1/1.2）；1.3 正样本定稿待做 |
| 数据规模 | `rec_inter` 14,605,716 行 / 25,877 用户 / 4,046,367 物品；搜索 5,059,169 行 / 452,541 会话 |
| 磁盘 | E: 可用 33.3 GB（下载前 37.7 GB）；项目本体 203 MB（不含 `data/`） |
| 第三方仓库 | 13 个（[third_party/SOURCES.toml](third_party/SOURCES.toml)，含 commit / 许可 / 可用性） |
| 模型索引 | 24 个：12 召回 / 9 排序 / 3 基础设施（[models/zoo.toml](models/zoo.toml)） |
| 已接入统一接口 | 6 个 `ready`：Popularity、BPR-MF、Two-Tower、SASRec、FAISS、FAISS 索引封装 |
| 上游可用未接入 | 7 个 `upstream_ready`（LightGCN、DIN、DIEN、DCNv2、DeepFM、MMoE、LightGBM） |
| 环境受限 | 11 个 `blocked`（MIND/HSTU/OneRec/TFRS/MMRec/TIGER 等，见 [docs/model-zoo.md](docs/model-zoo.md)） |
| 代码规模 | 43 个 Python 模块（`src/recsys/`），全部可导入 |
| 第三方体积 | 108.8 MB / 6,875 文件（清理前 400.7 MB / 6,376 文件） |
| 质量闸门 | `zoo.py check` 通过；`check_project.py` 0 错误；`smoke_adapters.py` 全过 |

---

## 二、已完成

### 2.0 现在能直接用的能力

| 能力 | 命令 / 入口 | 状态 |
| --- | --- | --- |
| 数据准备（KuaiRec，流程模板） | `python -m recsys.data.prepare --dataset kuairec` | 可用（KuaiSAR 的 `prepare` 待 1.3） |
| 训练召回基线 | `python -m recsys.training.train_mf\|trainer --dataset kuairec` | 可用 |
| 训练序列模型 | `python -m recsys.training.train_sasrec --dataset kuairec` | 可用 |
| KuaiSAR 数据获取与画像 | `python -m recsys.data.kuasar` / `python -m recsys.experiments.analyze_kuasar` | 可用（1.1/1.2 完成） |
| 统一评测（多模型同一入口） | `python -m recsys.adapters evaluate --model <id> --checkpoint <path>` | 可用（4 个召回模型） |
| 向量导出 | `adapter.encode_users / encode_items` | 可用（embedding 类模型） |
| FAISS 检索与一致性校验 | `python -m recsys.evaluation.faiss_evaluator` | 可用 |
| 工件溯源 | checkpoint 的 `provenance` 字段 | 可用（三个训练入口已接入） |
| 排序模型接口 | `LightGBMRankerAdapter.fit/predict/evaluate` | 接口就绪，需 `pip install lightgbm` |
| 质量闸门 | `zoo.py check` / `check_project.py` / `smoke_adapters.py` | 通过 |
| 文档与索引生成 | `python scripts/zoo.py render` | 可用（24 条模型） |

### 2.1 Phase 0-A：仓库整理与来源管理（2026-11-04）

- [x] 11 个克隆仓库从 `third_party/recommender_repos/<repo>` 扁平化为 `third_party/<repo>`
- [x] 删除 11 个浅克隆的 `.git`（129.7 MB）——只有一个 commit，无历史价值，且会干扰上层版本管理
- [x] 删除随仓库携带的大型示例语料（167.0 MB）：BERT4Rec 的 ml-20m/steam/ml-1m/beauty、SASRec.pytorch 的 5 个 txt
- [x] 删除 IDE / 构建缓存（`.idea`、`triton_aot/build`）
- [x] 保留全部源码、许可证、README、examples、tests；两个 `data/` 目录各留 3 行说明指向数据来源
- [x] 来源与许可：`third_party/repository_manifest.csv`（Phase 0 审计快照）、`third_party/model_mapping.md`（逐文件行号）
- 证据：[docs/cleanup-report.md](docs/cleanup-report.md)

### 2.2 Phase 0-B：Model Zoo 与文档生成（2026-11-04 ~ 11-05）

- [x] `models/zoo.toml` 作为唯一事实来源，每条记录 20 个字段（来源/commit/许可/原理/场景/依赖/Python/训练入口/推理入口/数据格式/保存加载/完整度/适配器）
- [x] `scripts/zoo.py check` 校验必填字段、枚举、源码路径、vendor 仓库与许可证
- [x] `scripts/zoo.py render` 生成 `docs/model-zoo.md`、`models/<category>/README.md`、`docs/models/<id>.md`
- [x] 已接入的 6 个模型有技术档案（原理/入口/工件/已知局限/实验记录表）
- [x] 按 PROJECT_MASTER §4 补齐来源仓库：**FuxiCTR**（DIN/DCNv2，Apache-2.0）、**genrec**（TIGER，MIT）；TorchRec 后置
- [x] 纠正一处文档与事实的冲突：RecBole 没有 MIND 实现（99 个模型文件已核实），MIND 实际来源改为 TorchEasyRec
- 证据：[docs/model-zoo.md](docs/model-zoo.md)、[models/README.md](models/README.md)、[docs/models/](docs/models/README.md)

### 2.3 Phase 0-C：统一模型接口（2026-11-05）

- [x] 召回接口 `RecommenderAdapter`：`recommend` / `retrieve` / `encode_users` / `encode_items` / `evaluate` / `save`
- [x] 排序接口 `RankingAdapter`：`fit` / `predict` / `evaluate`（AUC、LogLoss）/ `save` / `load`
- [x] 具体实现：TwoTower、MatrixFactorization、SASRec、Popularity（召回）；LightGBMRanker（排序）
- [x] 注册表 + CLI：`python -m recsys.adapters list|evaluate`，有 checkpoint 的模型自动读回数据集与阈值
- [x] 不支持的操作明确报错（如 SASRec 无稳定用户向量、Popularity 无参数），不返回伪造结果
- 证据：[src/recsys/adapters/](src/recsys/adapters/base.py)、[docs/architecture.md](docs/architecture.md)

### 2.4 Phase 0-D：工件溯源与评估指标（2026-11-05）

- [x] `provenance`：数据指纹（三切分行数 + 内容哈希 + mask 哈希）、ID 映射摘要、代码指纹、依赖版本、创建时间
- [x] 三个训练入口（Two-Tower / MF / SASRec）写入 `provenance`；评测时打印 `provenance: data=… code=…`
- [x] 指标补齐：`Precision@K`、`Coverage`、`AUC`、`LogLoss`（不依赖 sklearn），并有手算用例
- [x] 端到端验证：曾在 MovieLens 快速线上跑通 MF 训练 → 统一接口评测（该数据线已于 2026-11-06 移除，见 2.9）
- 证据：[src/recsys/utils/provenance.py](src/recsys/utils/provenance.py)、[src/recsys/evaluation/metrics.py](src/recsys/evaluation/metrics.py)

### 2.5 Phase 0-E：配置、校验与文档体系（2026-11-04 ~ 11-05）

- [x] `configs/datasets/`（kuairec、kuasar）+ `configs/models/`（two_tower、sasrec、matrix_factorization），TOML 格式
  （movielens.toml 已随数据线移除，见 2.9）
- [x] `scripts/check_project.py`：配置与代码常量一致性、Markdown 链接、生成物是否过期、残留缓存、third_party 来源与许可证
- [x] `scripts/smoke_adapters.py`：4 个适配器 + 向量出口 + `retrieve` 一致性 + 指标手算 + provenance 往返 + 排序适配器
- [x] 文档：架构、数据契约、依赖、KuaiRec 流程模板、KuaiSAR 差距、清理报告
- [x] 依赖分层：`requirements/`（主环境 / 可选 / 遗留 TF1 / MMRec）
- 证据：运行 `python scripts/check_project.py` 应输出「0 个错误」

### 2.6 Phase 0-F：版本控制（2026-11-06）

- [x] `git init -b main`，首次提交 `0bef337`：6,941 个文件，压缩后 47.9 MB
- [x] 推送到 private 仓库 <https://github.com/CamilleQAQ/recsys-agent>，`main` 已跟踪 `origin/main`
- [x] `.gitattributes`：只声明二进制类型，不强制换行（避免整目录变成 modified）
- [x] `data/` 与 `artifacts/` 未被提交，仅保留各自的 `README.md`
- 注意：沙箱访问不到 Windows 证书存储，**每次 `git push` 需要一次完全权限授权**（本地 commit 不受影响）

---

### 2.7 Phase 1.1：KuaiSAR 下载与结构勘查（2026-11-06）

- [x] `src/recsys/data/kuasar.py`：下载（带 UA、进度）→ md5 校验 → 安全解压（拒绝越界路径）→ 结构勘查
- [x] `configs/datasets/kuasar.toml`：来源/md5/许可/文件布局/字段清单，`check_project.py` 校验配置与代码常量一致
- [x] 实际下载：`KuaiSAR_v2.zip` **805.9 MB**，md5 匹配，解压后 **3.5 GB / 5 个 CSV**
- [x] 勘查产物：`artifacts/kuasar_archive_survey.json`（文件清单 + 每个 CSV 的表头）
- [x] 字段说明与样本统计：`docs/kuasar.md`（30 万行样本，标注为采样而非全量）
- [x] 关键发现：`rec_inter.csv` 中 **48.9% 的行所有行为位为 0**，即保留了「曝光未点击」——可构造真实 seen-mask
- 磁盘：下载前 E: 可用 37.7 GB → 完成后 **33.3 GB**

---

### 2.8 Phase 1.2：KuaiSAR 全量画像（2026-11-06）

- [x] `src/recsys/experiments/analyze_kuasar.py`：纯标准库流式扫描 3.5 GB，产出 `artifacts/kuasar_profile.json`
      （内存策略：只保留 user/item 两列做去重与分布，其余边读边算）
- [x] 全量规模：`rec_inter` **14,605,716 行 / 25,877 用户 / 4,046,367 物品**；`src_inter` 5,059,169 行 / 452,541 会话；
      `item_features` 6,949,990 物品
- [x] 行为分布：click 49.60%、like 3.15%、follow 0.45%、forward 0.31%；**纯曝光 49.08%**；`search_photo_related` 仅 2 次
- [x] 长尾结构：物品流行度中位 **1**，**61.4% 的物品只出现 1 次**；每用户交互中位 171、最大 32,885
- [x] 覆盖率：交互物品中 **99.9999%** 能在 `item_features` 找到特征
- [x] 数据质量清单：`duration_ms==0` 52.9 万行、`playing_time>duration` 317 万行、`upload_time` 含 1970 哨兵、搜索来源 2 个脏枚举
- [x] 结论写入 `docs/kuasar.md`（全量统计取代之前的 30 万行样本结论）
- 影响：物品词表 4M 级 → 全量打分 1.05e11 次点积、64 维 embedding 约 1 GB，Phase 2 必须定「精确评测 vs FAISS 评测」策略

---

### 2.9 移除 MovieLens 数据线（2026-11-06）

**原因**：`PROJECT_MASTER.md` §2 只定义 KuaiSAR（主）+ KuaiRand（补充），从未要求 MovieLens。
MovieLens 是原始代码遗留的"快速上手"数据线，之前保留它属于工程便利，不是项目协议。

**保留 KuaiRec 的理由**：也非协议数据，但它是显式决定保留的「流程模板/辅助线」（2026-11-05 决策），
用于在没有 KuaiSAR 训练集时验证数据管线；MovieLens 没有这个作用。

**删除内容**

- `src/recsys/data/movielens.py`（下载/解析/编码，201 行）
- `configs/datasets/movielens.toml` 与 `check_project.py` 里对应的两条校验
- `recsys.config.DEFAULT_MOVIELENS_POSITIVE_THRESHOLD`
- 运行时数据 `data/raw`、`data/processed` 与 `artifacts/movielens/`
- 文档与注释中的 MovieLens 表述

**代码调整（不是简单删除）**

- `MovieLensDataset` 实为 `schema.InteractionDataset` 的别名 → 全部改用 `InteractionDataset`
- `popularity.py` / `trainer.py` / `train_mf.py` 改为从 `schema` 导入 `Interaction`
- `SUPPORTED_DATASETS` 只剩 `("kuairec",)`；`resolve_checkpoint_dataset` 缺 `dataset` 字段时改为**报错**（不再默认 MovieLens）
- `tune_two_tower.py` 增加 `--dataset`；`finalize_experiment.py` 泛化为 `--dataset <name>`（artifacts 子目录随数据集命名）
- `faiss_evaluator.py` 改为先从 checkpoint 读数据集再准备数据（与 `evaluator.py` 一致）
- `faiss_index.py` 的 `index_to_original_movie_id` / `to_original_movie_ids` 改名去掉 movie 语义

**验证**：44 个模块全部可导入；适配层冒烟测试通过；`check_project.py` 0 错误。

---

### 2.10 Phase 1.3 口径决定与数据表示约束（2026-11-06）

**已定口径**（写入 `configs/datasets/kuasar.toml` 的 `[scope]` 与 `docs/kuasar.md`）

| 项 | 决定 | 理由 |
| --- | --- | --- |
| 正样本 | `click == 1`；按 `(user, item)` 去重保留首次点击 | 重复播放不代表新兴趣；KuaiSAR 无 `watch_ratio` |
| 广告物品 | **保留**（63,634 个 `item_type == AD`） | 贴近工业曝光分布；过滤等 1.5 特征通道（`is_ad`）后再提供 |
| 搜索侧 | **不纳入 Phase 1**，`src_inter` 留到 Phase 5 | 会话结构与 `PreparedData` 契约不同，提前接入会连带改评测协议 |

**发现的两个硬约束（阻塞 1.3 编码）**

1. **稠密 mask 不可行**：`seen_mask` 现状是 `[num_users, num_items]` bool，
   KuaiSAR 是 25,877 × 4,046,367 ≈ **105 GB**；KuaiRec 只有 7k × 10k 才勉强可行。
2. **`list[Interaction]` 放不下**：14.6M 个 dataclass 实例约 1 GB+（含 list 指针与对象头），
   而当前机器 **15.7 GB 总量、可用仅约 1.4 GB（负载 91%）**，全量 prepare 会直接 OOM。

**拟定方案（待确认后实施）**：`processed/` 落 npy 数组（user int32 / item int32 / ts int64 / click bool / split int8），
曝光按用户存稀疏 CSR（约 60 MB），`PreparedData` 增加数组后端；ID 映射也存 npy，不再整表进 dict。

---

### 2.11 Phase 1.3 实施：processed npy 存储（2026-11-06）

内存清理后（可用 1.4 GB → 5.6 GB）按数组化方案落地 `src/recsys/data/kuasar_store.py`：

- [x] 流式解析 14,605,716 行 → 数组（int32/int64/int8），不构造 `Interaction` 对象
- [x] 正样本按 `(user, item)` 去重保留首次点击：7,244,890 → **7,237,037**
- [x] **切分单位从「点击」改为「请求」**：`timestamp` 是按请求记录的，旧规则只有 3,752 用户可评测；
      改为 `(user_id, timestamp)` 请求级切分后 **24,128 / 25,877（93.2%）** 用户可评测
- [x] 用户级时间边界校验：24,128 个用户 `train_max < val_min`、`val_max < test_min` 均为 **100%**；
      验证/测试目标物品在训练历史中的比例 **0.00%**
- [x] 曝光历史按用户 CSR（验证 14,311,397 对 / 测试 14,443,563 对，含 click=0）
- [x] 产出 `data/kuasar/processed/`：**253 MB / 17 个 npy**（`ids/`、`split/`、`exposure/` + `stats.json` + `protocol.json`）
- [x] 全量烘焙 **35.8 秒**，峰值内存约 1.5 GB（可用内存 5.3 GB，无压力）
- [x] `load_processed()` 用 mmap 读取，常驻内存接近 0

**结论**：数据层从「放不下」变成「253 MB 落盘 + 零常驻」，1.3 的存储部分完成。
剩余：把 npy 视图接进 `PreparedData`/loader（1.4），并让 checkpoint 的 `split_strategy` 按数据集取值。

---

## 三、下一步（按 PROJECT_MASTER 的 Phase 顺序）

### 3.1 Phase 1：KuaiSAR 数据层（当前优先级）

详见 [docs/kuasar-roadmap.md](docs/kuasar-roadmap.md)。字段与统计见 [docs/kuasar.md](docs/kuasar.md)。

- [x] **1.1 下载与校验**：`src/recsys/data/kuasar.py` + `configs/datasets/kuasar.toml`
      验收：下载 + md5 + 解压 + 勘查一次跑通；重跑复用本地压缩包（`--survey-only` 可离线复查）
- [x] **1.2 字段核对与数据画像**：全量扫描完成（`analyze_kuasar.py`），结论写入 `docs/kuasar.md`
      验收：画像脚本可复现（`artifacts/kuasar_profile.json`）；结论已用于 1.3 的标签与 Phase 2 的成本估算
- [x] **1.3 口径与 processed 存储**：正样本 `click == 1` + 按对去重；广告物品保留；搜索侧不纳入；
      请求级切分产出 npy/CSR（253 MB，24,128 可评测用户，见 2.11）
- [ ] **1.4 样本构建与 `PreparedData` 数组后端**：把 `processed/` 视图接进 `loader`，让训练/评测入口能读 KuaiSAR；
      召回交互对 / 序列样本 / 排序候选+特征（第三类尚无实现）
      另需：`split_strategy` 按数据集取值（KuaiSAR 是 `per_user_last_two_requests`）
- [ ] **1.5 特征通道**：`FeatureStore` 或 `PreparedData.item_features`（含 `is_ad`），特征版本写入 `provenance.extra`
- [ ] **1.6 数据版本与缓存**：`load_processed` 校验 `protocol.json` 指纹，防止读到旧缓存

### 3.2 Phase 2：Two-Tower + FAISS 在 KuaiSAR 上跑通

- [ ] 特征/样本换成 KuaiSAR，重训 Two-Tower
- [ ] FAISS 建索引 + 精确/近似检索对比（复用 `faiss_evaluator`）
- [ ] 补 Coverage 与效率指标（训练耗时、单用户推理延迟、索引构建时间）

### 3.3 Phase 3–5：多路召回

- [ ] Phase 3 MIND：**先决策**（见 4.1），再适配 KuaiSAR 序列
- [ ] Phase 4 LightGCN：`PreparedData → RecBole atomic` 桥接 + 适配器
- [ ] Phase 5 多路融合：统一「通道 → 候选（含来源与原始分数）」结构、去重、配额、消融

### 3.4 Phase 6–8：排序

- [ ] Phase 6 SASRec 迁移到 KuaiSAR（序列截断/泄漏检查）
- [ ] Phase 7 DIN（FuxiCTR）：依赖 1.5 特征通道
- [ ] Phase 8 DCNv2（FuxiCTR）+ LightGBM：统一 CTR 标签与划分，产出对比表（AUC / LogLoss / 成本）

### 3.5 Phase 9–13：链路、生成式召回、Agent

- [ ] Phase 9 召回→融合→排序→重排（MMR）端到端离线链路
- [ ] Phase 10 TIGER（genrec）：物品 embedding → RQ-VAE 语义 ID → trie 约束解码
- [ ] Phase 11–13 Research Agent、RAG、Agent 对照实验

---

## 四、待决策事项

### 4.1 MIND 的实现路径（阻塞 Phase 3）

- 事实：RecBole 无 MIND；唯一可用实现是 TorchEasyRec `tzrec/models/mind.py`，依赖 Linux-only 阿里 wheel。
- 选项：① 给一个 Linux GPU 环境跑 TorchEasyRec；② 换同类多兴趣实现（genrec 的 COBRA/LC-Rec 属生成式多兴趣）；
  ③ 先用 Two-Tower 多向量变体做最小对照。
- 需要决定：**选哪条**，以及是否接受「MIND 在 Windows 上永远跑不了」这一约束。

### 4.2 KuaiSAR 许可

- 事实：CC BY-NC-SA 4.0（非商业 + 相同方式共享）。
- 需要确认：本项目用途是否落在非商业范围；衍生数据/模型是否对外发布。

### 4.3 TorchRec 何时引入

- 当前后置：Two-Tower 已有自研实现，TorchRec 主要解决大规模分片与分布式训练。
- 触发条件：单机训练成为瓶颈（Phase 2 之后评估）。

### 4.4 仓库公开与第三方许可

- 现状：仓库为 **private**。`third_party/` 内含 GPL-3.0（MMRec）与根 LICENSE 缺失的 OpenOneRec，
  私有仓库不属于对外分发，暂无额外义务。
- 若要开源：先把第三方代码拆出去（按 `SOURCES.toml` 的 commit 用脚本重建或改用 submodule），
  只保留自研代码、索引与审计材料；否则 GPL 传染性与 OpenOneRec 授权范围说不清。

### 4.5 推送需要完全权限

- 事实：沙箱禁止访问 Windows 证书存储，`git push` 会报 `SEC_E_NO_CREDENTIALS`。
- 现状：每次推送需要一次「完全权限」授权（本地 commit 不受影响）。
- 可选缓解：配置 PAT + `http.sslBackend=openssl`（会绕过 Windows 证书存储，但凭据仍需可读）。

---

## 五、已知限制与风险

| 项 | 影响 | 现状 / 缓解 |
| --- | --- | --- |
| `processed/` 只写不读 | 每次评测重新解析原始 CSV | Phase 1.6 处理；KuaiSAR 规模更大，必须做 |
| 未安装 lightgbm / tensorflow / fuxictr / recbole / tzrec / genrec | 对应模型无法本地验证 | 已按 `blocked` / `upstream_ready` 标注，不谎报 ready |
| MIND 平台依赖 | Phase 3 无法在 Windows 推进 | 见 4.1 |
| OpenOneRec 根目录无 LICENSE | 再分发风险 | 已在 `third_party/LICENSES.md` 标注 |
| 召回侧 `fit()` 未接管训练 | 接口与训练入口仍分离 | 训练由 `recsys.training.*` 承担；等 Phase 2 真实调用后再重构 |
| 无候选融合与重排 | 端到端链路不通 | Phase 5 / 9 |

---

## 六、常用命令

```powershell
# 质量闸门
python scripts/zoo.py check          # 索引 ↔ 源码路径 ↔ 许可证
python scripts/check_project.py      # 配置一致性、文档链接、生成物漂移、第三方来源
python scripts/smoke_adapters.py     # 适配层 + 指标 + 溯源（合成数据，无需真实数据集）

# 文档重生成（改完 models/zoo.toml 后必跑）
python scripts/zoo.py render

# 当前可跑的数据线（KuaiRec 作流程模板；KuaiSAR 的 prepare 待 1.3）
python -m recsys.data.prepare --dataset kuairec
python -m recsys.training.train_mf --dataset kuairec --epochs 2
python -m recsys.adapters evaluate --model matrix_factorization `
    --checkpoint artifacts/kuairec/matrix_factorization.pt --split test
```

---

## 七、变更历史

| 日期 | 阶段 | 主要变化 | 关键文件 |
| --- | --- | --- | --- |
| 2026-11-04 | Phase 0 | 11 个仓库整理：扁平化、删 `.git`/示例语料/缓存；建 Model Zoo 索引与文档骨架 | `third_party/README.md`、`models/zoo.toml`、`docs/cleanup-report.md` |
| 2026-11-05 | Phase 0 | 补 FuxiCTR/genrec 与 `SOURCES.toml`；接口扩到 `retrieve/encode_*` 与 `RankingAdapter`；checkpoint 加 `provenance`；指标补 Precision/Coverage/AUC/LogLoss；建档案与校验脚本；数据主线改为 KuaiSAR | `src/recsys/adapters/`、`src/recsys/utils/provenance.py`、`src/recsys/evaluation/metrics.py`、`PROJECT_PROGRESS.md` |
| 2026-11-06 | Phase 0 | 接入版本控制：`git init -b main` + 首次提交 `0bef337`（6,941 文件 / 压缩后 47.9 MB）推到 `CamilleQAQ/recsys-agent`；新增 `.gitattributes` 固定二进制判定；`data/`、`artifacts/` 未入库 | `.gitattributes`、`.gitignore`、本文件 |
| 2026-11-06 | **Phase 1.1** | KuaiSAR 下载与结构勘查：新增 `kuasar.py`（下载/md5/安全解压/勘查）与 `kuasar.toml`；实测 805.9 MB、解压 3.5 GB；核对 5 个 CSV 字段并写入 `docs/kuasar.md`；发现 48.9% 行为纯曝光记录 | `src/recsys/data/kuasar.py`、`configs/datasets/kuasar.toml`、`docs/kuasar.md` |
| 2026-11-06 | **Phase 1.2** | 全量画像：新增 `analyze_kuasar.py`（流式，3.5 GB）；14.6M 行 / 25.9k 用户 / 4.05M 物品、61.4% 物品只出现 1 次、交互物品特征覆盖 99.9999%、含数据质量清单；更新 `kuasar.toml` 与台账 | `src/recsys/experiments/analyze_kuasar.py`、`docs/kuasar.md`、`configs/datasets/kuasar.toml` |
| 2026-11-06 | **Phase 0 收尾** | 移除 MovieLens 数据线（不在 PROJECT_MASTER §2 内）：删模块/配置/数据/checkpoint，`MovieLensDataset` → `InteractionDataset`，三个训练入口与评测脚本改为数据集无关；`SUPPORTED_DATASETS` 仅剩 kuairec | `src/recsys/data/`、`src/recsys/evaluation/`、`src/recsys/experiments/`、`scripts/check_project.py` |
| 2026-11-06 | **Phase 1.3（口径）** | 定稿：正样本 `click==1` 且按对去重、广告物品保留、搜索侧不纳入 Phase 1；同时发现 1.3 编码被数据表示阻塞（稠密 mask 105 GB、list 表示约 1 GB、机器仅剩 1.4 GB 内存） | `configs/datasets/kuasar.toml`、`docs/kuasar.md`、本文件 |
| 2026-11-06 | **Phase 1.3（存储）** | 新增 `kuasar_store.py`：14.6M 行 → npy + 按用户 CSR 曝光（253 MB / 17 文件，35.8 秒）；切分单位改为**请求级**，可评测用户 3,752 → **24,128（93.2%）**；时间边界与目标泄漏校验均通过 | `src/recsys/data/kuasar_store.py`、`configs/datasets/kuasar.toml`、`docs/kuasar.md` |

> 详细到「改了哪些文件、为什么这么改、测试输出」的记录，见本文件对应小节的证据链接；
> 结论性内容回写 [PROJECT_MASTER.md](PROJECT_MASTER.md)，本文件不重复架构说明。
