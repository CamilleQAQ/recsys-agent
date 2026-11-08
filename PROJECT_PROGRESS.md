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
| 所处阶段 | **Phase 1 完成**（数据层 1.1–1.6 + 排序样本）；**Phase 2 小规模验证完成**；下一步 Phase 5（多路融合 + 重排） |
| 版本控制 | git `main`，远端 [CamilleQAQ/recsys-agent](https://github.com/CamilleQAQ/recsys-agent)；首次提交 `0bef337`（2026-11-06） |
| 数据主线 | KuaiSAR 2.0：原始 3.5 GB → processed 253 MB + 特征 139 MB + 排序样本 250 MB；另有 2,000 用户子集 `kuasar_small`（23 MB） |
| 数据规模 | `rec_inter` 14,605,716 行 / 25,877 用户 / 4,046,367 物品；搜索 5,059,169 行 / 452,541 会话 |
| 磁盘 | E: 可用 60.4 GB；`data/` 4.9 GB（不入库）；`artifacts/` 111.5 MB（冒烟 checkpoint，不入库） |
| 第三方仓库 | 13 个（[third_party/SOURCES.toml](third_party/SOURCES.toml)，含 commit / 许可 / 可用性） |
| 模型索引 | 24 个：12 召回 / 9 排序 / 3 基础设施（[models/zoo.toml](models/zoo.toml)） |
| 已接入统一接口 | 6 个 `ready`：Popularity、BPR-MF、Two-Tower、SASRec、FAISS、FAISS 索引封装 |
| 上游可用未接入 | 7 个 `upstream_ready`（LightGCN、DIN、DIEN、DCNv2、DeepFM、MMoE、LightGBM） |
| 环境受限 | 11 个 `blocked`（MIND/HSTU/OneRec/TFRS/MMRec/TIGER 等，见 [docs/model-zoo.md](docs/model-zoo.md)） |
| 代码规模 | 48 个 Python 模块（`src/recsys/`），全部可导入 |
| 第三方体积 | 108.8 MB / 6,875 文件（清理前 400.7 MB / 6,376 文件） |
| 质量闸门 | `zoo.py check` 通过；`check_project.py` 0 错误；`smoke_adapters.py` 全过 |
| 面试材料 | [DEVELOPMENT_STORY.md](DEVELOPMENT_STORY.md)：10 个故事线（问题 → 根因 → 取舍 → 数字），重大决策后追加 |

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

### 2.12 Phase 1.4：`PreparedData` 数组后端与样本构建（2026-11-06）

- [x] `schema.py` 新增三种结构：`ArrayInteractions`（mmap 惰性视图）、`SparseExposure`（CSR）、
      `SortedIndexLookup`（405 万物品的二分查找，不建 dict）；`InteractionDataset` 不再复制成 list
- [x] `PreparedData` 增加 `validation_seen` / `test_seen` / `split_strategy`；`seen_items_for_split`
      返回 `SeenItemsView`（按用户惰性构造集合），稠密 mask 路径保留给 KuaiRec
- [x] `prepare_kuasar()` + `loader.prepare_dataset("kuasar")` 打通；`DEFAULT_DATASET` 改为 `kuasar`
- [x] 切分协议按数据集参数化：`split_strategy_for()`；三个训练入口写 `prepared_data.split_strategy`，
      四个评测/适配层入口按数据集校验协议（不再硬编码 KuaiRec 常量）
- [x] 稠密真值表加护栏：`num_users × num_items > 2e7` 时抛错并提示关闭 `mask_false_negatives`
      （KuaiSAR 需要 105 GB；批内假负例概率约 1e-5，关闭无实质影响）
- [x] 消除 `list(prepared_data.train_interactions)` 的隐式复制（`build_observed_histories`、
      `build_evaluation_examples` 改用 `itertools.chain`）
- [x] 验证：`prepare_dataset("kuasar")` 返回 train 7,118,891 / val 67,517 / test 50,629；
      `seen_items_for_split("test")` 覆盖 24,128 用户；`InteractionDataset` 与 `SASRecTrainDataset`
      可直接在数组后端上构造样本
- [x] 端到端冒烟：KuaiSAR 上 `train_mf --embedding-dim 8 --max-batches-per-epoch 50` 跑通，
      checkpoint 带 `dataset=kuasar` / `split_strategy=per_user_last_two_requests` / `provenance`
      （val Recall@20 = 0.000559，1 epoch + dim 8，**不作结论**；124 MB 的冒烟权重未保留）

**未完成（顺延到 1.5/Phase 2）**：排序候选+特征样本（依赖特征通道）、全量训练的显存/内存预算
（4.05M × 64 维 embedding ≈ 1 GB，Adam 状态再 2 GB）。

---

### 2.13 Phase 1.5：物品特征通道（2026-11-06）

- [x] `src/recsys/data/kuasar_features.py`：流式读 `item_features.csv`（6,949,990 行），
      只保留交互词表内的 **4,046,363 / 4,046,367（99.9999%）** 物品，按 item_idx 对齐
- [x] 产出 12 列 npy（139 MB）：类目 id/编码（一级 37 类、二级 293 类）、作者 id 与频次、
      item_type（3 类）、upload_type（30 种）、上传日期与有效性、caption token 数、`is_ad`
- [x] `ItemFeatures` 接入 `PreparedData.item_features`；`prepare_kuasar` 校验长度一致，
      特征指纹写入 `provenance.item_features`
- [x] 广告标记：交互词表内 **10,265** 个 AD 物品（全表 63,634 个），默认不过滤
- [x] 对齐抽查：随机 CSV 行 ↔ 特征数组逐列比对，5/5 一致
- [x] 构建耗时 **45.6 秒**，填充率：类目/作者/类型 100%、上传日期 99.73%（1970 哨兵）

**下一步**：排序候选+特征样本（用 `is_ad` + 类目/作者构造候选特征），然后 Phase 2 的资源规划。

---

### 2.14 Phase 1.6：数据版本与缓存校验（2026-11-06）

- [x] 指纹改为**文件级 sha256**（原来只对部分数组做字节哈希），覆盖全部 17 个 npy：
      `ids/*`、`split/{train,validation,test}_{users,items,timestamps}`、`exposure/{validation,test}_{indptr,items}`
- [x] 汇总出 `data_version`（当前 `a643b5958660f7df`），写进 `protocol.json` 并在构建时打印
- [x] `load_processed(verify=True)` 默认校验（1–2 秒），不一致抛错并提示 `--force`
- [x] 构建可复用：已存在且校验通过时打印 `[reuse]` 直接返回；失败则提示并重建
- [x] 新增 CLI：`--check`（只校验）、`--force`（强制重建）
- [x] 特征表同样处理：`schema.json` 存每列 sha256，`load_item_features(verify=True)` 默认校验，
      `kuasar_features --check/--force`
- [x] 篡改测试：给 `user_ids.npy` 追加 1 字节 → `哈希不一致`；删文件 → `缺少文件`；正常状态 `--check` 输出 data_version

**验证**：processed 重建 38.9 秒、特征表 51.6 秒，两次数据完全一致（7,118,891 / 67,517 / 50,629，
24,128 可评测用户）；`prepare_dataset("kuasar")` 在校验开启下正常工作。

---

### 2.15 Phase 1.4 收尾：排序候选与特征样本（2026-11-06）

- [x] `src/recsys/data/kuasar_ranking.py`：把 `rec_inter` 按 `(user_id, timestamp)` 聚成请求，
      一次请求的曝光物品即候选、`click` 为标签，切分与 processed 完全一致
- [x] 规模：train **14,353,430 候选 / 3,104,325 请求**（点击率 49.65%）、
      validation 113,902 / 24,128（59.32%）、test 88,417 / 24,128（57.29%）
- [x] 行数对账：train + val + test + 丢弃 49,967 = **14,605,716**（与原始行数一致，构建里有断言）
- [x] 产出约 250 MB：`items / labels / users / groups` 四类数组 + `user_levels.npy` + `schema.json`
- [x] 特征不落盘：`iter_candidate_batches()` 按请求分批 join 9 列物品特征 + 2 列用户活跃度，
      直接给出 `X / y / groups`（LightGBM 的 group 参数），避免 14M × 11 列 ≈ 630 MB 常驻
- [x] 缓存校验：`--check` / `--force`，schema 里记录每个文件 sha256
- [x] 验证：`--check` 通过；validation 批次实跑 `X (23644, 11)`，groups 求和等于行数
- 已知偏差（写进文档）：留出请求点击率 59%/57% 高于训练集 50%，属「留出请求天然含点击」的选择偏差

**Phase 1 至此全部完成**：1.1 下载 → 1.2 画像 → 1.3 口径/存储 → 1.4 数组后端+排序样本 →
1.5 特征 → 1.6 数据版本。下一步进 Phase 2，但需要先定训练资源策略。

---

### 2.16 Phase 2 小规模验证：kuasar_small + Two-Tower（2026-11-07）

- [x] `src/recsys/data/kuasar_subset.py`：从全量 processed 按用户抽子集（保留请求/曝光结构），
      产物与主 store 同构，`prepare_dataset("kuasar_small")` 直接可用
- [x] 子集：**2,000 用户 / 454,717 物品 / 758,724 正样本**（train 748,674 / val 5,775 / test 4,275），23.2 MB
- [x] 训练：Two-Tower（embedding 塔、64 维、batch 2048、20 epochs、in-batch 负样本），
      CPU 4 线程约十几分钟；checkpoint 111.5 MB，带 provenance
- [x] 测试集：Recall@10 0.0016 / Recall@20 **0.003475** / NDCG@20 0.001839 / Coverage 0.0716
      （随机基线约 9.4e-5，≈37 倍；绝对水平低，原因是长尾 ID 塔学不动）
- [x] 检索对照：暴力检索 1.875 s（0.94 ms/user）vs FAISS IndexFlatIP 7.227 s（3.61 ms/user），
      指标逐位相同、agreement 1.0 → **这个规模下 FAISS 精确索引不划算**
- [x] 首次实验记录：`experiments/2026-11-07-kuasar_small-two_tower/`（config / metrics / checkpoint / report）
- 全量外推：候选对约 115 倍 → 暴力检索约 3–4 分钟/次，物品向量约 1 GB

**结论**：链路（数据 → 训练 → 统一评测 → FAISS 一致性）已在 CPU 上闭环；下一步是
「词表裁剪 vs 换硬件」的全量决策，以及 ANN 对照。

---

### 2.17 Phase 2 清单收尾：ANN 对照与独立推理（2026-11-07）

对照 `PROJECT_MASTER` 的 12 项工作清单，**已完成 10 项**，剩第 3、5 项：

| # | 工作项 | 状态 |
| --- | --- | --- |
| 1 | 审计 Two-Tower 结构/损失/负采样 | ✅ in-batch 负采样 + 温度缩放 CE，见 `models/retrieval/two_tower.md` |
| 2 | 判断能否直接适配 KuaiSAR | ✅ 通过数组后端接入，未重写模型 |
| 3 | 明确用户塔/物品塔输入特征 | ⛔ 目前仅 ID embedding（特征表已就绪，未入塔） |
| 4 | 训练/验证/测试流程 | ✅ `--dataset kuasar_small` 一条命令跑通 |
| 5 | 维度/优化器/lr/batch/负采样策略 | ⚠️ 有一组手选配置（64 维、Adam 1e-2、batch 2048），**无扫描记录** |
| 6 | 记录训练损失与验证指标 | ✅ checkpoint 存 `epoch_losses` / `best_validation_metrics` |
| 7 | 生成用户/物品向量 | ✅ `adapter.encode_users/encode_items` |
| 8 | FAISS 建物品索引 | ✅ `IndexFlatIP`（`faiss_evaluator`） |
| 9 | Top-K 近邻检索 | ✅ `faiss_top_k_from_embeddings` |
| 10 | 精确 vs 近似检索对照 | ✅ 新增 `ann_benchmark`（IVF/HNSW 扫描） |
| 11 | 模型保存/加载/独立推理 | ✅ 新增 `python -m recsys.adapters recommend --users ...` |
| 12 | 固定协议评估 Recall/HR/Coverage | ✅ 测试集 Recall@20 0.003475、HR@20 0.0085、Coverage 7.16% |

**ANN 对照结果**（`python -m recsys.evaluation.ann_benchmark`，k_search=600，2,000 用户）：

| 索引 | 与精确重合 | Recall@20 | ms/用户 | 索引体积 |
| --- | --- | --- | --- | --- |
| 暴力检索 | 1.000 | 0.003475 | 1.137 | — |
| FAISS Flat | 1.000 | 0.003475 | 2.151 | 111 MB |
| IVF nprobe=8 | 0.497 | 0.003867 | 0.036 | 115 MB |
| IVF nprobe=128 | 0.874 | 0.003475 | 0.127 | 115 MB |
| HNSW ef=256 | 0.648 | 0.003808 | 0.140 | 174 MB |

结论：**离线评测继续用暴力检索**；**服务化用 IVF nprobe 32–128 或 HNSW ef 64–256**
（8–9 倍加速，指标不降）。指标 ±0.0003 在噪声区间，索引质量要看候选重合率。

**独立推理**：`recommend` 子命令支持「原始 user_id → Top-K 原始 item_id」，
不依赖评测脚本，实测输出 `user 97722: [1084492, 3304371, ...]`。

---

### 2.18 Phase 2 第 3 项：物品侧特征入塔（2026-11-07）

- [x] `TwoTowerModel` 支持侧特征：ID 表 ⊕ 4 个类目 embedding（`feature_dim=16`，编码 +1 留缺失桶）
      ⊕ 5 个数值特征（`log1p(author_freq)`、`log1p(caption_tokens)`、`upload_days/365.25`、`is_ad`、`upload_valid`）
- [x] 特征以 **buffer** 形式挂在模型上：checkpoint 自带特征，`encode_items(ids)` 独立可推理
- [x] 子集构建器现在会**切片全量特征表**（12 列），`prepare_dataset("kuasar_small")` 直接带特征
- [x] 训练入口新增 `--use-item-features` / `--feature-dim` / `--checkpoint-name`（变体不再互相覆盖）
- [x] 基线按原配置重训，指标**逐位复现**（Recall@20 0.003475、Coverage 0.071592）
- [x] 记录为独立实验：`experiments/2026-11-07-kuasar_small-two-tower-features/`

**测试集对照（4,275 个目标，单 seed）**：

| 指标 | ID-only | + 物品特征 | 变化 |
| --- | --- | --- | --- |
| Recall@10 | 0.001600 | **0.003200** | **+100.0%** |
| Recall@20 | 0.003475 | **0.004408** | +26.9% |
| NDCG@20 | 0.001839 | 0.002836 | +54.2% |
| Coverage | 0.071592 | 0.071134 | −0.6% |

参数量 29.23M → 33.35M，checkpoint 111.5 → 134.2 MB。验证集只有 5,775 个目标，
Recall@10 在验证上反而下降——**结论需多 seed 复核**，已写入实验报告「已知问题」。

**Phase 2 清单：11/12 完成**，只剩第 5 项（超参扫描记录）。

---

### 2.19 Phase 2 第 5 项：超参扫描（2026-11-07）

- [x] 4 组配置 × 8 epochs 扫描（dim 32/64/128、batch 1024/2048、lr 3e-3/1e-2，特征开启），
      指标取 checkpoint 里的 best-epoch 验证结果
- [x] 前两名跑满 20 epochs 复核（d64 与 d128），测试集确认
- [x] **选定配置**：`embedding_dim=output_dim=64`、`batch=2048`、`lr=0.01`、Adam、20 epochs、特征开启
- [x] 记录为独立实验：`experiments/2026-11-07-kuasar_small-two-tower-sweep/`

**扫描结果（验证集 best epoch）**：

| 配置 | best epoch | Recall@20 | 测试 Recall@20 | 测试 NDCG@20 |
| --- | --- | --- | --- | --- |
| d32 / b2048 / lr1e-2 | 2 | 0.001433 | — | — |
| d64 / b1024 / lr1e-2 | 5 | 0.001300 | — | — |
| d64 / b2048 / lr3e-3 | 4 | 0.000867 | — | — |
| d128 / b2048 / lr1e-2 | 7 | 0.002258 | 0.004825 | 0.002015 |
| **d64 / b2048 / lr1e-2（选定）** | 19 | 0.004133 | **0.004408** | **0.002836** |

**关键教训**：8 轮扫描把 d128 排成第一（是第二名的 1.7 倍），跑满 20 轮后 d64 与 d128 的
验证 Recall@20 只差 6%，而 NDCG@20 / Recall@10 的排序完全反转——**短预算扫描测的是「谁先动起来」，
不是「谁最终更好」；最终比较必须同预算、同 epoch 数**。这一条也写进了 `DEVELOPMENT_STORY.md`。

**Phase 2 清单：12/12 完成。**

---

### 2.20 Phase 3（部分）：MIND 多兴趣召回接入（2026-11-07）

**背景**：MIND 的参考实现只在 TorchEasyRec 里，而它的依赖 `fbgemm-gpu` / `torchrec`
只有 `manylinux` 的 cp310–312 wheel，Windows + Python 3.14 装不上（本轮已再次核实）。

**方案**：改用 `torch-rechub`（MIT，纯 PyTorch，30+ 模型），只借模型本体，
数据/训练/评测仍走本项目契约：

- 新增 `src/recsys/models/mind.py`（torch-rechub MIND 封装 + 打分/编码工具）
- 新增 `src/recsys/data/dataset.py::MindTrainDataset`（每「用户+位置」一条样本，746,674 条/epoch）
- 新增 `src/recsys/training/train_mind.py`（in-batch softmax，验证集选模，checkpoint 带 provenance）
- 新增 `src/recsys/adapters/rechub_mind.py`（`MindAdapter`，多兴趣取 max，接入统一评测与 `recommend` CLI）
- 依赖：`requirements/optional-rechub.txt`（`torch-rechub==1.0.0`）；源码 sparse-checkout 到
  `third_party/torch-rechub`（commit `7ea1731`）并登记 `SOURCES.toml`
- zoo：MIND 从 `blocked` → **`ready`**（`ready=7 / upstream_ready=7 / blocked=10`）

**结果（kuasar_small，测试集 4,275 个目标）**：

| 指标 | 双塔 + 物品特征 | MIND | 变化 |
| --- | --- | --- | --- |
| Recall@10 | 0.003200 | **0.003742** | +16.9% |
| Recall@20 | 0.004408 | **0.005675** | **+28.8%** |
| HitRate@20 | 0.008500 | **0.010500** | +23.5% |
| NDCG@20 | 0.002836 | 0.002712 | −4.4% |

- 20 epochs 验证集 Recall@20 = 0.007650，最后三轮仍在上升 → **未收敛**，当前不是上限
- 8 epochs 版本测试只有 0.002183 → **训练预算对 MIND 影响极大**（2.6 倍）
- 训练耗时约 60 分钟（CPU），是双塔的 3 倍；多路召回的总预算要算进去

实验记录：`experiments/2026-11-07-kuasar_small-mind/`。

---

### 2.21 Phase 3 阶段评估（P1 止损，2026-11-08）

**范围**：训练预算 0，只做离线评测；复用 `mind20.pt` 与固定双塔基线。

- [x] 核对 40-epoch 中断日志：A/B/C 分别到第 24/25/21 轮，日志保留在
      `experiments/2026-11-08-kuasar_small-mind-phase3/logs/*.pre-restart.log`；
      **没有权重**（`train_mind` 只在训练结束时保存），不构成收敛结论
- [x] 发现并修复评测可复现性问题：torch-rechub 的 `CapsuleNetwork` 用 `torch.randn`
      初始化路由权重，同一 checkpoint 每次评测指标都不同；新增 `deterministic_routing()`
      在评测时固定种子并恢复 RNG 状态，两次评测逐位一致
- [x] 分群评估（低/中/高活跃，<100 / 100–299 / ≥300；704/622/674 用户，目标 1,493/1,346/1,436）：
      双塔与 MIND 在同一批测试用户、候选范围、曝光过滤下对比
- [x] 旧报告差异已解释：旧值 0.005675 是未固定路由种子的单次抽样，落在本次 3 个路由种子的
      区间 0.004408–0.006300 内；本次报告均值与区间
- [x] 阶段报告 + metrics + checkpoint 记录：
      `experiments/2026-11-08-kuasar_small-mind-phase3/`（report.md / metrics.json / checkpoint.txt）

**结论**（测试集，MIND 为 3 个路由种子均值）：

| 组 | 双塔 Recall@20 | MIND Recall@20 | 差 |
| --- | --- | --- | --- |
| 总体 | 0.004408 | **0.005044** | +0.000636 |
| low（<100） | 0.003717 | 0.004198 | +0.000481 |
| mid（100–299） | 0.003537 | **0.009325** | +0.005788 |
| high（≥300） | **0.005935** | 0.001978 | −0.003957 |

MIND 的收益集中在中等活跃用户；高活跃用户因历史截断到 20 条而明显退化。
路由随机性带来约 ±20% 的 Recall@20 波动，单次评测不足以支撑结论。

**未完成（留待办）**：40 epoch 收敛检查、同预算 A/B/C 超参对照、训练 seed 43/44 复现、
三 seed 汇总脚本（`scripts/mind_phase3_report.py` 已就绪，等正式工件齐备后再跑）。

---

### 2.22 离线推荐 MVP：双路召回 → 融合 → LightGBM → 类目重排（2026-11-08）

**范围**：`kuasar_small` + 128 个验证用户，一次小样本端到端；只验收链路与口径，不看指标提升。

- [x] 新增离线入口 `python -m recsys.experiments.recsys_mvp`（显式校验 `--dataset kuasar_small`）
- [x] 两路召回复用现有权重：双塔 `two_tower_features.pt`、MIND `mind20.pt`（路由种子 1234），
      各算一次 Top-100 并缓存；只对这 128 个用户打分（未对 2,000 用户全量打分）
- [x] 名次轮流融合 + 去重，最多 100；每个候选记录来源通道与原始名次
- [x] LightGBM 只训 1 次：从父数据集 `data/kuasar/processed/ranking/` 只读 train 工件，
      按原始 ID 对齐父子词表，保留请求边界，**9,997 行 / 2,418 请求**（正 6,387 / 负 3,610），
      LambdaRank 20 轮；`LightGBMRankerAdapter.fit` 增加 `num_boost_round` 参数（默认 50 不变）
- [x] 重排：Top-20，已知一级类目每类 ≤5，补齐 0 次
- [x] 自动断言全部通过：无重复、无曝光泄漏、来源可追溯、融合 ≤100、最终 ≤20、
      排序/重排均来自融合池、分组长度=行数、训练/推理特征列一致、重载复验分数差 5e-07
- [x] 实测：**总耗时 10.0 s**（召回 1.0+3.9、排序数据 1.8、训练 0.2），**峰值 RSS 1,379 MiB**，
      4 线程；预算内完成

**指标（128 用户验证样本，Top-20）**：四阶段 Recall/HitRate/NDCG 均为 0——362 个目标中，
Top-100 候选命中 9 个（双塔 2 / MIND 7 / 并集 9），但都排在 20 名之后；属于样本量限制，
`metrics.json` 的 `diagnostics` 记录了这些命中，用来核对链路而非下结论。

**交付**：`experiments/2026-11-08-kuasar_small-recsys-mvp/`（config / metrics / run_meta / samples /
checkpoint / report）；缓存与模型在 `artifacts/kuasar_small/mvp/`。

**未完成**：完整验证/测试集评估、用户侧/交叉特征、分数校准、多路配额消融、DIN/DCNv2 对比。

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
- [x] **1.4 样本构建与 `PreparedData` 数组后端**：数组后端（`ArrayInteractions` / `SparseExposure` /
      `SortedIndexLookup`）接入 `loader`；`split_strategy` 按数据集取值；交互对与序列样本可直接构造
      剩余：排序候选+特征样本（依赖 1.5）
- [x] **1.5 特征通道**：`ItemFeatures` 接入 `PreparedData`；12 列物品特征（类目/作者/上传类型/日期/
      caption token/`is_ad`），139 MB，指纹进 provenance；排序候选样本依赖它，下一步做
- [x] **1.6 数据版本与缓存**：17 个 npy 全部有文件级 sha256 + `data_version`；
      读取默认校验、构建可复用、`--check` / `--force` 入口齐备（见 2.14）
- [x] **Phase 1 收尾**：排序候选+特征样本（train 14.35M 候选 / 3.10M 请求，11 列特征按批 join，见 2.15）

### 3.2 Phase 2：Two-Tower + FAISS 在 KuaiSAR 上跑通

- [x] 小规模验证（2,000 用户子集）：整条链路跑通，结论见
      [experiments/2026-11-07-kuasar_small-two_tower](experiments/2026-11-07-kuasar_small-two_tower/report.md)
- [x] ANN 对照（IVF/HNSW 扫描）：离线用暴力检索、服务化用 IVF/HNSW（见 2.17）
- [x] 独立推理入口：`python -m recsys.adapters recommend --model two_tower --checkpoint ... --users 97722`
- [x] **特征入塔**：物品侧 4 类目 + 5 数值特征；Recall@10 翻倍、Recall@20 +26.9%（见 2.18）
- [x] **超参选择记录**：4 组扫描 + 2 组 20 轮复核，选定 d64/b2048/lr1e-2；记录「短预算选参不可靠」（见 2.19）
- [ ] 多 seed 复核特征实验（验证/测试在 Recall@10 上方向不一致）
- [ ] 用户侧特征（活跃度）入塔——子集需切 `user_features.csv`
- [ ] 全量决策：词表裁剪（交互 ≥N 次）还是换 GPU/大内存；裁剪会改变评测口径，必须写清楚
- [ ] 补 Coverage 与效率指标（训练耗时、单用户推理延迟、索引构建时间）

### 3.3 Phase 3–5：多路召回

- [x] **Phase 3 MIND**：改用 `torch-rechub` 实现（TorchEasyRec 平台受限），已训练/评测；
      **阶段分群评估完成**：测试 Recall@20 0.005044（3 路由种子均值）vs 双塔 0.004408，
      mid 组 +0.0058、high 组 −0.0040（见 2.21）
- [ ] MIND 训练到收敛（30–40  epochs）、同预算 A/B/C 对照、训练 seed 43/44 复现
- [ ] Phase 4 LightGCN：`PreparedData → RecBole atomic` 桥接 + 适配器
- [x] Phase 5 多路融合（MVP 版）：名次轮流 + 去重 + 来源/名次可追溯，≤100 候选（见 2.22）
- [ ] Phase 5 完整版：分数校准、各通道配额、单路/多路消融、通道贡献与重叠率

### 3.4 Phase 6–8：排序

- [ ] Phase 6 SASRec 迁移到 KuaiSAR（序列截断/泄漏检查）
- [ ] Phase 7 DIN（FuxiCTR）：依赖 1.5 特征通道
- [x] Phase 8 LightGBM 排序（MVP）：复用 `LightGBMRankerAdapter`，train 工件 9,997 行 / 20 轮（见 2.22）
- [ ] Phase 8 完整版：DCNv2 对比、AUC/LogLoss、叶子数/学习率调参、特征贡献分析

### 3.5 Phase 9–13：链路、生成式召回、Agent

- [x] Phase 9 召回→融合→排序→重排端到端离线链路（MVP，128 用户；重排为类目配额，见 2.22）
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
| 2026-11-06 | **Phase 1.4** | `PreparedData` 数组后端（ArrayInteractions / SparseExposure / SortedIndexLookup）、`prepare_dataset("kuasar")` 打通、`split_strategy` 按数据集、稠密真值表护栏、去掉隐式 `list()` 复制；默认数据集改为 kuasar | `src/recsys/data/{schema,loader,dataset}.py`、`src/recsys/data/kuasar_store.py`、`src/recsys/training/*`、`src/recsys/evaluation/*` |
| 2026-11-06 | **Phase 1.5** | 物品特征通道：`kuasar_features.py` 把 695 万行宽表压成 12 列 npy（139 MB，覆盖 99.9999% 交互物品）；`ItemFeatures` 接入 `PreparedData`，指纹进 provenance；`is_ad` 覆盖交互词表内 10,265 个广告物品 | `src/recsys/data/kuasar_features.py`、`src/recsys/data/schema.py`、`configs/datasets/kuasar.toml`、`docs/kuasar.md` |
| 2026-11-06 | **Phase 1.6** | 数据版本与缓存校验：17 个 npy 的文件级 sha256 + `data_version`；`load_processed`/`load_item_features` 默认校验；构建可复用（`[reuse]`）；`--check`/`--force`；篡改测试通过 | `src/recsys/data/kuasar_store.py`、`src/recsys/data/kuasar_features.py`、`docs/kuasar.md` |
| 2026-11-06 | **Phase 1 收尾** | 排序候选与特征样本：`kuasar_ranking.py` 按请求聚合曝光候选（train 14.35M/3.10M 请求），产出 items/labels/users/groups + 用户活跃度；`iter_candidate_batches` 按批 join 11 列特征；行数对账 14,605,716 全平 | `src/recsys/data/kuasar_ranking.py`、`configs/datasets/kuasar.toml`、`docs/kuasar.md` |
| 2026-11-07 | **Phase 2 验证** | `kuasar_small`（2,000 用户 / 454,717 物品）+ Two-Tower 20 epochs：链路闭环；测试 Recall@20 0.003475（≈37× 随机）；FAISS IndexFlat 比 torch 暴力检索慢 3.9×；建立首个实验记录 | `src/recsys/data/kuasar_subset.py`、`experiments/2026-11-07-kuasar_small-two_tower/`、`docs/kuasar.md` |
| 2026-11-07 | **文档** | 新增 `DEVELOPMENT_STORY.md`：10 个开发故事线（仓库瘦身、文档生成、数据结构、切分口径、曝光语义、训练效率、FAISS 反直觉、缓存指纹、范围纪律、长尾），含 60 秒讲法与高频追问 | `DEVELOPMENT_STORY.md`、`README.md`、`docs/README.md` |
| 2026-11-07 | **Phase 2 清单** | 12 项完成 10 项：新增 ANN 对照（IVF nprobe=128 与精确指标相同、快 8.9×；HNSW ef=256 快 8×）与独立推理入口（原始 user_id → Top-K 原始 item_id）；剩余特征入塔与超参扫描 | `src/recsys/evaluation/ann_benchmark.py`、`src/recsys/adapters/__main__.py`、实验记录 |
| 2026-11-07 | **Phase 2 特征** | 物品侧特征入塔（4 类目 + 5 数值，buffer 随 checkpoint）：测试 Recall@10 0.0016 → 0.0032（翻倍）、Recall@20 +26.9%、NDCG@20 +54%；子集特征表切片与 `--checkpoint-name` 变体命名；基线重训逐位复现 | `src/recsys/models/two_tower.py`、`src/recsys/training/trainer.py`、`src/recsys/data/kuasar_subset.py`、`experiments/2026-11-07-kuasar_small-two-tower-features/` |
| 2026-11-07 | **Phase 2 超参** | 4 组 × 8 epochs 扫描 + 2 组 × 20 epochs 复核：选定 d64/b2048/lr1e-2；发现「8 轮扫描排名在 20 轮后反转」，记录短预算选参的陷阱；Phase 2 清单 12/12 | `experiments/2026-11-07-kuasar_small-two-tower-sweep/` |
| 2026-11-07 | **Phase 3 MIND** | 因 TorchEasyRec 的 Linux-only wheel 不可用，改用 torch-rechub 1.0.0 接入 MIND（模型/训练/适配器/CLI）；测试 Recall@20 0.005675（比双塔 +28.8%），NDCG@20 略低；20 epochs 未收敛，训练约 60 分钟 | `src/recsys/models/mind.py`、`src/recsys/training/train_mind.py`、`src/recsys/adapters/rechub_mind.py`、`third_party/torch-rechub`、`experiments/2026-11-07-kuasar_small-mind/` |
| 2026-11-08 | **Phase 3 阶段评估** | P1 止损：训练预算 0；发现并修复 MIND 评测随机性（胶囊路由 `torch.randn`），同一 checkpoint 固定种子后可逐位复现；分群评估完成（mid 组 +0.0058、high 组 −0.0040、总体 +0.0006）；旧 0.005675 解释为单次随机路由抽样；40-epoch 收敛/同预算对照/训练多 seed 留待办 | `src/recsys/models/mind.py`、`src/recsys/training/train_mind.py`、`scripts/eval_groups.py`、`experiments/2026-11-08-kuasar_small-mind-phase3/` |
| 2026-11-08 | **离线 MVP** | 新增 `recsys_mvp`：128 用户双路召回（双塔 + MIND）→ 名次融合（≤100，来源可追溯）→ LightGBM（父 train 工件 9,997 行 / 20 轮）→ 类目重排（每类 ≤5）；自动断言全过；10.0 s / 1,379 MiB；Top-20 指标为 0（样本量限制），Top-100 命中 9 记录在 diagnostics | `src/recsys/experiments/recsys_mvp.py`、`src/recsys/adapters/ranking.py`、`experiments/2026-11-08-kuasar_small-recsys-mvp/` |

> 详细到「改了哪些文件、为什么这么改、测试输出」的记录，见本文件对应小节的证据链接；
> 结论性内容回写 [PROJECT_MASTER.md](PROJECT_MASTER.md)，本文件不重复架构说明。
