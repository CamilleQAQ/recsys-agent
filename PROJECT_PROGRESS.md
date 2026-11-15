# 项目进度与工作日志

> **用途**：记录「已经做完了什么 / 现在卡在哪 / 下一步做什么」。每轮开发后更新本文件。
> **与 [PROJECT_MASTER.md](PROJECT_MASTER.md) 的关系**：总档案是**大纲与基准**（定位、架构、13 个 Phase、
> 记录规范），不随开发进度改动；本文件是**执行台账**，只写事实、证据和下一步。
> **维护规则**：
> 1. 只写真实发生的事；没验证的不写成完成，验证集结果不写成测试集结果；
> 2. 每条「已完成」都要能点开证据（代码路径 / 文档 / 可复现命令）；
> 3. 待办用 `- [ ]`，完成的用 `- [x]` 就地勾选，不删除历史；
> 4. 每轮结束更新「当前状态快照」与「变更历史」两节。

**最后更新**：2026-11-13

---

## 一、当前状态快照

| 项目 | 状态 |
| --- | --- |
| 所处阶段 | **Phase 1（v2）推荐系统 + Rec Agent 已交付**；**Phase 2 搜索系统 + Search Agent 已交付**（2.34–2.37）；**Phase 3 广告系统 + Ads Agent 已交付**（2.38）；**Phase 4 统一 Agent 平台已交付**（7.12）；Phase 11/12 MVP、Phase 13 离线回放保留 |
| 版本控制 | git `main`，远端 [CamilleQAQ/recsys-agent](https://github.com/CamilleQAQ/recsys-agent)；首次提交 `0bef337`（2026-11-06） |
| 数据主线 | KuaiSAR 2.0：原始 3.5 GB → processed 253 MB + 特征 139 MB + 排序样本 272 MB（含请求时间戳）；另有 2,000 用户子集 `kuasar_small`（23 MB） |
| 小样本正式线 | `kuairec_small`（KuaiRec 官方 small_matrix）：1,411 用户 / 3,317 物品 / 4,676,570 行，`SUPPORTED_DATASETS` 已注册，四模型在一台 CPU 机器 17 分钟跑完「验证选型 → 测试一次」（[docs/kuairec-small.md](docs/kuairec-small.md)） |
| 数据规模 | `rec_inter` 14,605,716 行 / 25,877 用户 / 4,046,367 物品；搜索 5,059,169 行 / 452,541 会话 |
| 磁盘 | E: 可用约 60 GB；`data/` 4.9 GB（不入库）；`artifacts/` 含本轮冒烟 checkpoint（不入库） |
| 第三方仓库 | 16 个（[third_party/SOURCES.toml](third_party/SOURCES.toml)，含 commit / 许可 / 可用性） |
| 模型索引 | 26 个：13 召回 / 10 排序 / 3 基础设施（[models/zoo.toml](models/zoo.toml)）；ready=22 / upstream_ready=0 / blocked=4 |
| 已接入统一接口 | 11 个召回 + 5 个排序（[src/recsys/adapters/](src/recsys/adapters/)）+ 6 个广告模型（[src/recsys/ads/models.py](src/recsys/ads/models.py)），逐模型状态见 [docs/model-zoo.md](docs/model-zoo.md) |
| 推荐链路 | `python -m recsys.pipeline`：召回→融合→排序→重排→评估（[docs/recsys-pipeline.md](docs/recsys-pipeline.md)） |
| 搜索链路 | `python -m recsys.search --mode bm25\|vector\|hybrid\|rerank`：BM25 + FAISS 向量 + RRF 混合 + token 精排（[docs/search.md](docs/search.md)） |
| 广告链路 | `python -m recsys.ads.pipeline`：请求内广告候选打分→排序→二价 GSP→CPC 扣费/预算（[docs/ads.md](docs/ads.md)） |
| Search Agent | 4 个工具（BM25 / 向量 / 混合 / 精排）+ LangGraph + 预算/空结果/失败落盘；真实 LLM 入口与本地替身驱动均可用（[docs/search-agent.md](docs/search-agent.md)） |
| Ads Agent | 7 个真实工具 + LangGraph；本地 14/14 自检，真实 `deepseek-chat` 一次 verified（[docs/ads-agent.md](docs/ads-agent.md)） |
| Rec Agent | 12 个真实工具 + LangGraph + 预算/日志/失败上报；确定性计划器与真实 LLM 均已跑通（[docs/rec-agent.md](docs/rec-agent.md)） |
| 环境受限 | 4 个 `blocked`：OneRec（缺权重/多卡）、TFRS Two-Tower、TFRS factorized_top_k、TorchEasyRec SID；均有可执行替代 |
| 代码规模 | 137 个 Python 文件（`src/recsys/`，本轮新增 `ads/` 16 个），核心模块可导入 |
| 质量闸门 | `zoo.py check` 通过；`check_project.py` 0 错误；`smoke_adapters.py`、`smoke_phase1.py`、`smoke_phase3_ads.py` 通过 |
| 效果问题 | **已做效果专项**（7.14）：精排是最大负贡献（去掉后 Recall@20 0.002175 → 0.005967）；排序上界 69 命中 vs 现在 25；放开 cap 稳定小赚，RRF 跨种子不复现 |
| 广告边界 | 六模型均为固定初始化权重（`interface_test`），未训练；KuaiSAR 无转化标签，CVR 仅 synthetic；出价/预算为配置模拟值 |

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
| 搜索检索（BM25 / 向量 / 混合 / 精排） | `python -m recsys.search --mode <mode> --tokens ...` | 可用（120 物品小样本；向量走 FAISS） |
| 搜索向量索引构建 / 校验 | `python -m recsys.search.build_vector_index [--check]` | 可用（独立于推荐索引） |
| 搜索系统验收 | `python -m recsys.search.acceptance` | 9/9 通过 |
| Search Agent（四工具） | `python -m recsys.experiments.search_agent --self-check / --scripted-plan hybrid_rerank` | 可用（17/17 自检；替身驱动验收） |
| Phase 4 统一平台（Research/Master + 共享实验 + 预算 + RAG） | `python -m recsys.platform --task ... / --history-query / --registry / --rag-query / --acceptance` | 可用（13/13 小样本验收；真实 LLM 单次 4 决策） |
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

### 2.23 Phase 11 Research Agent MVP：融合配额对照（2026-11-08）

**问题**：100 个候选/用户时，双塔与 MIND 怎么分配配额能保留更多验证目标？

- [x] 工具 `evaluate_fusion_quota`（`src/recsys/experiments/fusion_quota.py`），可脱离 LLM 独立运行；
      只读 MVP 缓存（0 训练、0 重新召回），核对 128 用户 / 362 目标、两路用户集合、名次、去重、
      曝光泄漏与文件指纹，不一致直接报错
- [x] 融合规则：双塔在前按配额交替，只有新物品消耗配额，重复跳过，一路耗尽由另一路补位并记录
- [x] Agent CLI `python -m recsys.experiments.quota_agent`：先强制 50/50，再根据结果选下一组，
      最多 3 组 / 180 秒；`--self-check` 覆盖非法配额、重复、超次数、超时间、补位与去重
- [x] 实跑 3 组：50/50 命中 4、30/70 命中 5、20/80 命中 5；结论 **30/70 最优**
      （Recall@100 0.013672 vs 基准 0.011719），总耗时 1.37 s
- [x] 决策与工具日志 `events.jsonl` 可核对「先基准、后选组」；候选明细落
      `artifacts/kuasar_small/quota_agent/cache_eval/`
- [ ] **真实 LLM 工具调用**：代码路径已实现（OpenAI 兼容，环境变量配置），本机无密钥，
      本轮为本地确定性策略；待配置后验收

**交付**：`experiments/2026-11-08-kuasar_small-fusion-quota-agent/`（config / metrics / runs / events / report）。

---

### 2.24 Phase 11 验收：真实 LLM 原生 tool_calls（2026-11-08，通过）

- [x] 新增 `src/recsys/experiments/llm_tool_call.py`：显式要求真实 LLM，注册
      `evaluate_fusion_quota` 的 OpenAI tools schema，解析服务商 `tool_calls`（恰好 1 条、
      函数名/调用 ID/参数校验），按参数执行真实评测，再用原 `tool_call_id` 回传结果、请求模型给停止理由
- [x] 预算与守卫：最多 2 次模型请求、1 次工具调用、180 秒；超时取 `min(30s, 剩余)` 且不重试；
      普通文本 JSON、多条 tool_calls、未知工具、非法配额、重复基准一律拒绝（`--self-check` 6 条全过）
- [x] **真实运行通过**（deepseek-chat）：响应 ID `e74adcc6…` 返回 1 条 `tool_calls`
      （`call_00_9LWE…`，参数 70/30）；程序执行评测（命中 3、Recall@100 0.009115），
      用同一 ID 回传，模型回复「低于基准，不采用该配额，停止」；共 2 次请求、1 次工具、3.05 s
- [x] 结论：70/30 输给基准 50/50（命中 3 vs 4，Recall 0.009115 vs 0.011719），按规则保留基准；
      新配额输了仍是合法验收结果
- 实现调整（如实记录）：先修复一次代码 bug（未发请求），再因服务商返回 4 条并行 `tool_calls`
  被协议拒绝（未执行工具）；加 `parallel_tool_calls=false` + 收紧提示后，第三次运行为唯一执行工具的一次

**交付**：`experiments/2026-11-08-kuasar_small-llm-tool-call/`（config / metrics(verified) / events /
baseline / report）；候选明细在 `artifacts/kuasar_small/llm_tool_call/cache_eval/70_30/`。

---

### 2.25 Phase 11：LangGraph 接入与失败路径回归（2026-11-08，通过）

- [x] `llm_tool_call.py` 改为 LangGraph 状态图：`load_baseline → request_tool_call → execute_tool →
      send_result → finish`，条件边按 `status/error` 分流，失败分支直接进 `finish`；CLI 调用方式不变
- [x] 依赖：`requirements/optional-agent.txt`（`langgraph==1.2.20`），作为 Agent 可选依赖
- [x] **修复误报**：只有「合法 tool_call → 工具执行成功 → 同 ID 回传 → 收到非空且不再请求工具的最终回复」
      才写 `verified`；第二次请求失败/空回复/再次请求工具写 `partial` 并返回非零
- [x] **真实 LangGraph 运行通过**：deepseek-chat 返回 1 条 `tool_calls`（70/30），
      2 次请求 / 1 次工具执行 / 2.81 s；节点顺序与 `tool_call_id` 回传均在 `events.jsonl` 可核对；
      结果：70/30 命中 3 vs 基准 4（Recall@100 0.009115 vs 0.011719），保留基准
- [x] **失败路径回归通过**（不联网、不读缓存）：注入「第二次请求超时」，
      得到 `status: partial`、模拟 CLI 退出码 2、请求计数 2、工具执行 1、工具结果保留、无 `verified`
- [x] 调试记录：首次真实运行因 assistant 消息缺 `role` 被服务商 422 拒绝，被正确记为 `partial`；
      修复后重跑成功。候选工件改写到独立目录 `artifacts/kuasar_small/langgraph_tool_call/`，不覆盖 2.24

**交付**：`experiments/2026-11-08-kuasar_small-langgraph-tool-call/`
（config / metrics / events / baseline / report / regression_second_request_timeout/）。

---

### 2.26 Phase 11.2：多工具注册与历史实验复用（2026-11-08，通过）

**目标**：扩展 LangGraph Agent 为「先查历史，未命中才评测」；历史与当前输入/口径一致时
直接复用并停止。同时修复 `--cache-dir` 不生效：显式目录必须贯穿基准校验、历史匹配和融合评测。

- [x] **工具注册表** `src/recsys/experiments/agent_tools.py`：集中维护
      `search_experiment_history` / `evaluate_fusion_quota` 的名称、schema、校验与执行函数；
      参数只允许 `two_tower_quota` / `mind_quota` 两个整数，路径/命令/自定义脚本一律拒绝；
      LangGraph 只通过注册表执行注册过的工具
- [x] **历史匹配** `src/recsys/experiments/experiment_history.py`：核对数据集/验证切分、
      128 固定用户与两路候选缓存文件 sha256、模型 checkpoint sha256、目标配额、
      每用户 100 候选预算、融合顺序/去重/补位/来源与指标口径；并核对 `fusion.json` 工件的
      配额、预算、池大小和逐用户统计。旧 run 无规则版本字段时，用同一实验目录的
      `config.toml` + 工件确认兼容（`legacy_record: true`）；`partial`/失败/证据不足返回 `miss`
- [x] **历史优先工作流** `src/recsys/experiments/history_first_agent.py`：
      `load_baseline → request_history → execute_history → send_history_result`；
      命中直接 `finish`；未命中且模型选择评测才进入 `execute_eval → send_eval_result → finish`。
      第一次请求只开放历史工具，命中后评测工具不开放；程序校验请求/查询/评测次数与同 ID 回传
- [x] **真实 LLM 运行通过（1 次）**：deepseek-chat 返回 1 条原生 `tool_calls`
      （`call_00_EYwK4whr6AYjR48lZIYM3943`，30/70），程序命中 Phase 11 已落盘记录并用同 ID 回传；
      模型引用 **5 个命中、Recall@100 0.013672**（基准 50/50 为 4 / 0.011719，多 1 个目标）后停止。
      `decision: reuse_history`、**评测 0 次**、2 次请求 / 1 次历史查询 / 2.21 s
- [x] **未命中分支（本地模拟）**：临时历史目录无 30/70 记录时，图允许一次评测、同 ID 回传并收尾
      （verified / evaluated / 3 请求 / 评测 1）；`--self-check` 12 项覆盖查询前评测、未知工具、
      夹带路径/命令/脚本、非法配额、多条 tool_calls、命中后评测、重复评测与请求上限
- [x] **非默认缓存目录（本地检查）**：独立复制 `mvp_copy` 后，基准校验、历史查询和真实融合评测
      全部使用该目录（`evidence.cache` 一致，评测命中 5）；缺失目录与篡改 `recall_mind.json`
      的副本均返回 blocked + CLI 2，未回退默认缓存
- [x] **回传失败（本地模拟）**：历史命中后最终请求超时、评测成功后最终请求超时，均为
      `partial`、模拟 CLI 2、工具结果保留、不写 `verified`
- [x] **旧入口保留**：`python -m recsys.experiments.llm_tool_call --experiment-dir ...` 调用方式不变；
      该模块的 `--cache-dir` 也改为从 CLI 状态贯穿基准校验与工具执行
- 范围：训练 0、重新召回 0、推荐全链路 0、测试集评估 0；未覆盖旧缓存/旧实验

**交付**：`experiments/2026-11-08-kuasar_small-history-first-agent/`
（config / events / metrics / decision / baseline / local_checks / report；
未命中与超时回归在 `regression_miss/`、`regression_timeout_*/`；
缓存目录检查在 `cache_dir_check/`）。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.history_first_agent `
  --experiment-dir experiments/2026-11-08-kuasar_small-history-first-agent --quota 30/70
python -m recsys.experiments.history_first_agent --self-check
python -m recsys.experiments.history_first_agent --regression-miss
python -m recsys.experiments.history_first_agent --regression-timeout history
python -m recsys.experiments.history_first_agent --regression-timeout evaluation
```

证据：[report.md](experiments/2026-11-08-kuasar_small-history-first-agent/report.md)、
[metrics.json](experiments/2026-11-08-kuasar_small-history-first-agent/metrics.json)、
events.jsonl（本地文件：`experiments/2026-11-08-kuasar_small-history-first-agent/events.jsonl`）、
[decision.json](experiments/2026-11-08-kuasar_small-history-first-agent/decision.json)、
[local_checks.json](experiments/2026-11-08-kuasar_small-history-first-agent/local_checks.json)。

---

### 2.27 Phase 12 实验记忆 MVP：新实验自动入库与跨进程零评测复用（2026-11-08，2026-11-09 修复发布时机，通过）

**目标**：在 Phase 11 历史优先流程上补齐写路径——第一次查不到 → 用已有候选缓存真实评测一次
→ 自动保存并更新 JSON 索引 → 第二次独立进程命中历史 → 零评测。

- [x] **选真正未入库的配额**：`precheck.json` 核对 Phase 11 `runs/*.json`、索引和已有候选工件；
      50/50、30/70、20/80 已有 runs，70/30 虽无索引但已有 2.24/2.25 工件；
      **80/20** 无 runs、无索引、无候选工件，探针返回 `miss`，故选 80/20
- [x] **稳定实验键** `src/recsys/experiments/experiment_memory.py`：由数据集/切分、
      `id_mapping`/`routing_seed`/`k_recall`、双塔与 MIND checkpoint sha256、
      `users.json`/两路召回文件 sha256、`targets_sha256`、配额、每用户 100 预算、
      融合规则版本和指标口径共同计算；日期/目录名/耗时/服务商不参与
- [x] **自动入库**：真实评测成功后先核验 `fusion.json` 与指标，再写 run record，
      最后用临时文件 + `os.replace` 原子更新 `index.json`；相同键只保留一条；
      失败/partial/缺工件不发布
- [x] **读取校验**：命中要求身份匹配、`status=success`、指标记录完整、run record 存在、
      融合工件可读且指纹/池统计一致；旧 Phase 11 JSON 继续走原兼容性检查
- [x] **第一次独立 CLI 进程**：本地确定性策略 + **真实评测**，历史 `miss` →
      `evaluated`，命中 2、Recall@100 0.005208（基准 4 / 0.011719），评测 **1** 次；
      自动写入索引，实验键 `24cc580d…07962f`，墙钟 6.5 s
- [x] **第二次独立 CLI 进程**：重新从磁盘读取索引，历史 `hit`（`source=memory_index`）→
      `reuse_history`，复用同键、同指标、同候选工件，评测 **0** 次；
      events 无 `tool_executed`/`memory_saved`，索引未被改写，墙钟 5.01 s
- [x] **匹配安全（不评测）**：有效索引 hit；缓存指纹变化、规则版本不匹配、
      候选工件缺失、记录 `partial` 均为 miss（5/5）
- [x] **范围**：训练 0、重新召回 0、推荐全链路 0、排序/重排 0、测试集 0、RAG 0；
      旧缓存与旧实验只读。本轮未跑真实 LLM（任务要求本地确定性策略），配额评测真实执行
- [x] **兼容性**：默认 `policy=llm` 与上一轮真实 LLM 路径不变；新增 `--local-policy`
      只用于本地验收；`search_experiment_history` 增加可选 `memory_dir`，
      `evaluate_fusion_quota` 未改
- [x] **发布时机修复（2026-11-09）**：评测成功后只保留候选/指标/调用 ID/证据，不再在
      `node_execute_eval` 发布索引；只有 `node_finish` 确认 `verified`、评测结果用同一
      `tool_call_id` 回传且收到有效最终回复，才调用 `save_evaluation()`。新增
      `--regression-publish-gate`：历史未命中 → 模拟评测成功 → 最终回复超时 →
      重查历史。回归结果 `partial`、模拟 CLI 2、工具结果/证据/超时原因保留、
      **无 `memory_saved` 事件、索引 0 条、重查仍 `miss`**；现有 80/20 verified 记录仍 hit。
      **失败结果不发布，Phase 12 实验记忆 MVP 验收完成。**
- [x] **发布回归整理（2026-11-09）**：将 `run_regression_publish_gate` 移到独立脚本
      `src/recsys/experiments/regression_publish_gate.py`；旧 CLI 仅委托，参数/退出码保留。
      每次在 `--experiment-dir` 父目录下创建唯一 `run-<UTC>-<pid>/`，省略时默认
      `artifacts/kuasar_small/experiment_memory/regression_publish_gate/`；移除
      `shutil.rmtree`。旧 `experiments/.../regression_publish_gate/` 递归 sha256 前后一致、
      预置 marker 未变；新回归 18/18 通过

**两次运行总墙钟 11.51 s**（预算 120 s）；索引 1 条记录，第二轮零评测。

**交付**：`experiments/2026-11-08-kuasar_small-experiment-memory/`
（config / precheck / memory/index.json / memory/runs/*.json / run1 / run2 /
two_run_check / memory_safety_checks / regression_publish_gate / refactor_check / report）；
独立回归脚本 `src/recsys/experiments/regression_publish_gate.py`。

**复现**：

```powershell
$env:PYTHONPATH = "src"
$exp = "experiments/2026-11-08-kuasar_small-experiment-memory"

python -m recsys.experiments.history_first_agent --audit-quotas `
  --experiment-dir $exp --memory-dir "$exp/memory" `
  --output-dir artifacts/kuasar_small/experiment_memory --quota 80/20

python -m recsys.experiments.history_first_agent --experiment-dir "$exp/run1" `
  --cache-dir artifacts/kuasar_small/mvp --memory-dir "$exp/memory" `
  --output-dir artifacts/kuasar_small/experiment_memory --quota 80/20 --local-policy

python -m recsys.experiments.history_first_agent --experiment-dir "$exp/run2" `
  --cache-dir artifacts/kuasar_small/mvp --memory-dir "$exp/memory" `
  --output-dir artifacts/kuasar_small/experiment_memory --quota 80/20 --local-policy

python -m recsys.experiments.history_first_agent --memory-safety-check `
  --experiment-dir $exp --memory-dir "$exp/memory" `
  --cache-dir artifacts/kuasar_small/mvp --quota 80/20

python -m recsys.experiments.history_first_agent --regression-publish-gate `
  --experiment-dir artifacts/kuasar_small/experiment_memory/regression_publish_gate `
  --cache-dir artifacts/kuasar_small/mvp --quota 80/20

# 独立脚本入口（同一实现）
python -m recsys.experiments.regression_publish_gate `
  --cache-dir artifacts/kuasar_small/mvp --quota 80/20
```

证据：[report.md](experiments/2026-11-08-kuasar_small-experiment-memory/report.md)、
[precheck.json](experiments/2026-11-08-kuasar_small-experiment-memory/precheck.json)、
run1/metrics.json（本地文件：`experiments/2026-11-08-kuasar_small-experiment-memory/run1/metrics.json`）、
run2/metrics.json（本地文件：`experiments/2026-11-08-kuasar_small-experiment-memory/run2/metrics.json`）、
[two_run_check.json](experiments/2026-11-08-kuasar_small-experiment-memory/two_run_check.json)、
[memory_safety_checks.json](experiments/2026-11-08-kuasar_small-experiment-memory/memory_safety_checks.json)、
regression.json（本地文件：`experiments/2026-11-08-kuasar_small-experiment-memory/regression_publish_gate/regression.json`）、
[refactor_check.json](experiments/2026-11-08-kuasar_small-experiment-memory/refactor_check.json)、
新回归（本地文件：`artifacts/kuasar_small/experiment_memory/regression_publish_gate/run-20261108T162817Z-10980/regression.json`）。

---

### 2.28 Phase 13 离线回放 MVP：融合配额受控对照（2026-11-09）

**目标**：在 `kuasar_small` 上，用已有双塔 + MIND 融合结果做一次受控回放：Rec Agent 与固定策略从同一份 `50/50` 历史出发，在各自 2 次揭晓预算下选择配额；每组结果只在该侧选中后揭晓。只读已有汇总工件，不重训、不重新召回、不重跑评测、不调用真实 LLM。

- [x] **独立入口** `src/recsys/experiments/offline_replay.py`：`python -m recsys.experiments.offline_replay`；默认每次在 `experiments/2026-11-09-kuasar_small-offline-replay/` 下新建 `run-<UTC>-<pid>/`，不覆盖旧验收证据
- [x] **配置** `configs/experiments/offline_replay.toml`：`50/50` 起始、4 个公开候选、每侧预算 2、最佳规则（Recall@100 → 离 50/50 距离 → 双塔配额升序）、固定顺序 `70/30 → 30/70`、Agent 首步/反馈/停止规则
- [x] **来源清单**：融合配额实验 3 组（`50/50`、`30/70`、`20/80`）、LangGraph 工具调用实验 1 组（`70/30`）、实验记忆实验 1 组（`80/20`）；另核对 5 条同配额重复记录，0 冲突；5 条记录共享同一缓存身份 `sha256:600fae5de84c561e…`、128 用户 / 362 目标 / 每用户 100 候选 / 同一 split 哈希
- [x] **共用执行器**：预算与配额校验、重复/非法/预算耗尽拒绝、单次拒绝即终止并标记失败；结果查询工具 `get_fusion_result(two_tower, mind)` 只在通过校验后返回公开指标，策略拿不到完整结果表、来源或对方轨迹
- [x] **实际回放（1 次）**：Agent `30/70 → 20/80`，最佳 `30/70`（命中 5 / Recall@100 0.013672 / Coverage 0.026720），揭晓 2、异常 0；固定策略 `70/30 → 30/70`，最佳同为 `30/70`，揭晓 2、异常 0；**结论：持平**。Agent 第 1 次揭晓即取得最佳，固定策略第 2 次取得最佳
- [x] **轻量本地断言 9/9**：初始隔离、反馈敏感（固定首步，仅替换首步指标：更好→`20/80`、更差→`70/30`、持平→停止）、重复请求、预算耗尽、非法配额、固定顺序、平局规则、双方隔离、轨迹完整性；全部使用合成结果，不触发真实评测
- [x] **范围**：训练 0、重新召回 0、真实评测 0、真实 LLM 0、网络 0；候选明细 / 缓存 / 完整数据集 / 实验记忆索引均未读取；输入汇总 JSON 运行前后 sha256 一致，旧验收证据未改写
- [ ] **Phase 13 完整版**：正式实验（更多配额与重复回放、策略泛化、测试集评估）仍保留为待办

**交付**：`experiments/2026-11-09-kuasar_small-offline-replay/run-20261109T020944Z-20800/`
（config / sources / agent_trace / fixed_trace / metrics / checks / run_meta / report）。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay --self-check
python -m recsys.experiments.offline_replay
```

证据：report.md（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay/run-20261109T020944Z-20800/report.md`）、
sources.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay/run-20261109T020944Z-20800/sources.json`）、
agent_trace.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay/run-20261109T020944Z-20800/agent_trace.json`）、
fixed_trace.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay/run-20261109T020944Z-20800/fixed_trace.json`）、
metrics.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay/run-20261109T020944Z-20800/metrics.json`）、
checks.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay/run-20261109T020944Z-20800/checks.json`）、
run_meta.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay/run-20261109T020944Z-20800/run_meta.json`）、
[入口](src/recsys/experiments/offline_replay.py)、
[配置](configs/experiments/offline_replay.toml)。

---

### 2.29 Phase 13 真实 LLM Rec Agent 离线回放（2026-11-09，verified）

**目标**：把 Phase 11 已验证的 LangGraph 原生工具调用接入 Phase 13 离线回放，对比真实 LLM Rec Agent、本地规则 Agent、固定策略。只读已有汇总 JSON；真实 LLM 工作流只跑一次，最多 3 次模型请求。

- [x] **独立入口** `src/recsys/experiments/offline_replay_llm.py`：`python -m recsys.experiments.offline_replay_llm`；默认每次在 `experiments/2026-11-09-kuasar_small-offline-replay-llm/` 下新建 `run-<UTC>-<pid>/`，保留旧证据
- [x] **复用范围**：来源校验与 `ReplaySession` 来自 `offline_replay.py`；HTTP/`.env` 读取与 OpenAI 兼容请求沿用 `llm_tool_call.py`；原生 `tool_calls` 校验按同一协议在薄适配层实现；未改动 Phase 11 / 12 入口或实验记忆发布逻辑
- [x] **LangGraph 节点**：`request_first → execute_first → request_feedback → execute_second → request_final → finish`；工具名 `get_fusion_result(two_tower, mind)`，每次最多 1 个 tool_call，校验配额、重复、剩余预算后才返回指标，并使用同一 `tool_call_id` 回传
- [x] **冻结基线**：本地规则 Agent 与固定策略直接引用上一轮 `run-20261109T020944Z-20800` 的轨迹与指标；启动真实工作流前核对来源 8 个文件 sha256、缓存身份 `sha256:600fae5de84c561e…`、起始历史、候选、预算、最佳规则和两条轨迹逐步结果，全部一致
- [x] **真实运行前本地替身检查 11/11**：首轮不可见隐藏指标、工具返回与调用 ID 对应、非法/重复/超预算拒绝、最终回复失败落 `partial`、一次/两次揭晓的 verified、固定顺序和请求/揭晓上限；全部合成、不联网
- [x] **截断回复修复（2026-11-09）**：统一“停止说明/最终回复”校验为 `finish_reason=="stop"` + 正文非空 + 无 tool_calls；`finish_reason=="length"` 且已有成功揭晓时改为 `partial` / `reply_truncated`，保留揭晓指标、预算和工具轨迹，三方比较 `not_judged`；本地回归扩展至 **13/13**，未调用真实 LLM
- [x] **真实 LLM 工作流（1 次）**：`deepseek-chat`；3 次模型请求 / 2 次成功揭晓 / 5.244 s；第 1 次请求 `70/30`（命中 3 / Recall@100 0.009115），看到反馈后第 2 次请求 `30/70`（命中 5 / 0.013672），预算用尽后完成最终回复；程序按 50/50 + 本侧揭晓结果算出最佳 `30/70`，状态 **verified**
- [x] **三方结果**：LLM、本地规则 Agent、固定策略最佳均为 `30/70`（命中 5 / Recall@100 0.013672 / Coverage 0.026720），**三方最佳持平**；三方各揭晓 2 次，没有查询次数节省；首次达到最佳分别为 LLM 2、本地规则 1、固定策略 2
- [x] **范围**：训练 0、重新召回 0、真实评测 0；只读 5 组 canonical 汇总 JSON 及重复核对节点（共 8 个文件）和两条冻结基线的轨迹/指标引用；候选明细 / 缓存 / 数据集 / 实验记忆索引未读取；除已配置 LLM 服务外其他网络请求 0；来源与基线 sha256 前后一致，旧工件未改写
- [ ] **Phase 13 完整版**：正式实验（更多配额与重复回放、其他模型/数据规模、策略泛化与测试集评估）仍待办

**交付**：`experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/`
（config / sources / baselines / llm_trace / events / metrics / self_check / run_meta / report）。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_llm --self-check
python -m recsys.experiments.offline_replay_llm
```

证据：report.md（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/report.md`）、
sources.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/sources.json`）、
baselines.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/baselines.json`）、
llm_trace.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/llm_trace.json`）、
events.jsonl（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/events.jsonl`）、
metrics.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/metrics.json`）、
self_check.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/self_check.json`）、
run_meta.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-llm/run-20261109T025055Z-22736/run_meta.json`）、
[截断修复回归](experiments/2026-11-09-kuasar_small-offline-replay-llm-regression/report.md)、
[13/13 自检](experiments/2026-11-09-kuasar_small-offline-replay-llm-regression/self_check.json)、
[入口](src/recsys/experiments/offline_replay_llm.py)、
[配置](configs/experiments/offline_replay.toml)。

---

### 2.30 Phase 13 小规模多场景离线对照（2026-11-09，A/B 均 verified）

**目标**：只读已有 5 组融合汇总 JSON，比较真实 LLM、本地规则 Agent、固定策略在两个场景中的表现。场景 A 只揭晓 1 组；场景 B 起始历史含 `70/30`，可再揭晓 2 组。本地基线先冻结，再各启动一次真实 LLM。

- [x] **双场景配置** `configs/experiments/offline_replay_scenarios.toml`：A 起始 `50/50`、预算 1、最多 2 次模型请求；B 起始 `50/50`+`70/30`、可揭晓 `30/70`/`80/20`/`20/80`、预算 2、最多 3 次请求；固定顺序 `70/30 → 30/70 → 80/20 → 20/80`，自动跳过起始历史
- [x] **复用范围**：`MultiStartReplaySession` 支持多条起始历史且不扣预算；真实 LLM 复用现有 LangGraph、`get_fusion_result` 原生工具调用、同 ID 回传、修复后的回复校验与请求/揭晓上限；单场景 CLI 仍可用，旧工件未改
- [x] **本地先冻结**：A 本地首选 `30/70`、固定首选 `70/30`；B 本地 `30/70 → 20/80`、固定 `30/70 → 80/20`；两条基线轨迹、摘要和 sha256 写入 `frozen/`，之后才启动真实 LLM
- [x] **场景 A 真实 LLM（1 次）**：`deepseek-chat`，2 次请求 / 1 次揭晓 / 3.234 s；首选 `70/30`（命中 3 / Recall@100 0.009115），最终最佳为起始 `50/50`；本地规则 Agent 的 `30/70` 为三方唯一最佳，**真实 LLM 输给最佳**
- [x] **场景 B 真实 LLM（1 次）**：3 次请求 / 2 次揭晓 / 3.928 s；先查 `30/70`（命中 5 / 0.013672），看到反馈后再查 `20/80`（5 / 0.013672）；三方最佳均为 `30/70`，**三方最佳持平**；三方各揭晓 2 次，没有查询次数节省
- [x] **自检 19/19**：单场景状态机既有 13 项 + 场景初始可见性、本地/固定策略、起始历史不扣预算、历史配额拒绝、A/B 正常与截断路径、请求上限；全部合成、不联网
- [x] **范围**：训练 0、重新召回 0、真实评测 0；两场景真实 LLM 请求合计 5/5；只读 8 个汇总 JSON 及冻结基线；候选明细 / 缓存 / 数据集 / 实验记忆索引未读取；来源与冻结基线哈希前后一致，旧工件未改写
- [ ] **Phase 13 完整版**：正式实验（更多场景/配额、其他模型/数据规模、策略泛化与测试集评估）仍待办

**交付**：`experiments/2026-11-09-kuasar_small-offline-replay-scenarios/run-20261109T034321Z-4528/`
（scenario_config / base_config / sources / frozen / scenarios/A / scenarios/B / metrics / self_check / run_meta / report）。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_scenarios --self-check
python -m recsys.experiments.offline_replay_scenarios
```

证据：report.md（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-scenarios/run-20261109T034321Z-4528/report.md`）、
metrics.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-scenarios/run-20261109T034321Z-4528/metrics.json`）、
frozen/manifest.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-scenarios/run-20261109T034321Z-4528/frozen/manifest.json`）、
场景 A LLM 轨迹（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-scenarios/run-20261109T034321Z-4528/scenarios/A/llm_trace.json`）、
场景 B LLM 轨迹（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-scenarios/run-20261109T034321Z-4528/scenarios/B/llm_trace.json`）、
self_check.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-scenarios/run-20261109T034321Z-4528/self_check.json`）、
run_meta.json（本地文件：`experiments/2026-11-09-kuasar_small-offline-replay-scenarios/run-20261109T034321Z-4528/run_meta.json`）、
[入口](src/recsys/experiments/offline_replay_scenarios.py)、
[配置](configs/experiments/offline_replay_scenarios.toml)。

---

### 2.31 Phase 13 `70/30` 历史信息消融（2026-11-09，两组均 verified）

**目标**：只改变“决策方能否看到已有 `70/30` 指标”，检查它是否改变首选、后续查询或停止时机。`70/30` 在两组都不可查询；两组候选、预算、工具、提示词模板和规则相同。

- [x] **冻结配置** `configs/experiments/offline_replay_70_30_ablation.toml`：对照组起始 `50/50`；历史组起始 `50/50`+`70/30`；两组均可查询 `30/70`/`80/20`/`20/80`，预算 2，最多 3 次模型请求；固定顺序 `30/70 → 80/20`
- [x] **复用**：同一 `offline_replay_scenarios` 入口、`MultiStartReplaySession`、LangGraph、`get_fusion_result` 和回复校验；仅新增 `history_ablation` 模式的最小校验/报告分支
- [x] **本地基线先冻结**：对照组与历史组的本地规则均为 `30/70 → 20/80`，固定策略均为 `30/70 → 80/20`；`70/30` 历史不扣预算，本地规则在两组做出相同选择
- [x] **真实 LLM 对照（各 1 次）**：对照组 3 请求 / 2 揭晓 / 4.801 s，首选 `30/70` 后查 `80/20`，最佳 `30/70`；历史组 2 请求 / 1 揭晓 / 3.674 s，首选 `30/70` 后主动停止，最佳同为 `30/70`
- [x] **消融结果**：两组最佳相同（5 / 0.013672 / 0.026720）；历史组少揭晓 1 次，按规则记为该场景 **节省 1 次查询**。历史组最终回复明确引用 `70/30`（3 / 0.009115）低于 `50/50`；但每组仅一次回放，不据此断言因果
- [x] **自检 20/20**：单场景状态机既有 13 项 + 两组初始可见性、本地/固定策略、历史不扣预算、历史配额不可查询、正常/截断路径、请求上限，以及“去掉 70/30 历史后首轮 payload 一致”
- [x] **范围**：训练 0、重新召回 0、真实评测 0；两组真实 LLM 请求合计 5/6；只读 8 个汇总 JSON 与冻结基线；候选明细 / 缓存 / 数据集 / 实验记忆索引未读取；来源与冻结基线哈希前后一致，旧工件未改写
- [ ] **Phase 13 完整版**：正式实验（更多历史/配额、其他模型/数据规模、策略泛化与测试集评估）仍待办

**交付**：`experiments/2026-11-09-kuasar_small-70-30-history-ablation/run-20261109T045034Z-15408/`
（scenario_config / base_config / sources / frozen / scenarios/A / scenarios/B / metrics / self_check / run_meta / report）。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_scenarios --self-check `
  --config configs/experiments/offline_replay_70_30_ablation.toml
python -m recsys.experiments.offline_replay_scenarios `
  --config configs/experiments/offline_replay_70_30_ablation.toml `
  --output-root experiments/2026-11-09-kuasar_small-70-30-history-ablation
```

证据：report.md（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation/run-20261109T045034Z-15408/report.md`）、
metrics.json（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation/run-20261109T045034Z-15408/metrics.json`）、
frozen/manifest.json（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation/run-20261109T045034Z-15408/frozen/manifest.json`）、
对照组 LLM 轨迹（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation/run-20261109T045034Z-15408/scenarios/A/llm_trace.json`）、
历史组 LLM 轨迹（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation/run-20261109T045034Z-15408/scenarios/B/llm_trace.json`）、
self_check.json（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation/run-20261109T045034Z-15408/self_check.json`）、
run_meta.json（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation/run-20261109T045034Z-15408/run_meta.json`）、
[入口](src/recsys/experiments/offline_replay_scenarios.py)、
[消融配置](configs/experiments/offline_replay_70_30_ablation.toml)。

---

### 2.32 Phase 13 历史信息节省查询的成对复验（2026-11-09，3/3 复现）

**目标**：复用 `70/30` 历史消融配置与来源，新增 3 对真实 LLM 回放，验证“最终最佳相同、历史组少揭晓 1 次”是否重复出现；上一轮结果不计入本轮。

- [x] **成对调度**：在同一 `offline_replay_scenarios` 入口新增 `--paired`，顺序固定 `A→B`、`B→A`、`A→B`；每组只启动 1 次，共 6 次工作流，每场景最多 3 请求 / 2 揭晓，合计上限 18
- [x] **对照条件不变**：A 起始 `50/50`，B 起始 `50/50`+`70/30`；两组候选均为 `30/70`/`80/20`/`20/80`，预算 2；`70/30` 在两组都不可查询，历史不扣预算
- [x] **本地先冻结**：复用上一轮消融的本地规则与固定策略轨迹；本轮只重复真实 LLM 对照，未重跑本地基线
- [x] **自检 21/21**：原有状态机/消融检查 + 配对顺序与复现/持平/退化/不可判定规则；本地自检先通过，真实请求在自检之后
- [x] **代码审查修复（2026-11-09）**：`--paired` 改为读取 `[paired].reference_run` 的冻结 manifest/轨迹并逐文件核对 sha256，不再重跑本地/固定策略；请求计数前移到模型调用边界（预算校验后、调用客户端前），客户端超时/报错与轨迹生成/写工件失败仍计数；全轮 18 次上限在调用前检查；回归脚本 `scripts/check_paired_request_accounting.py` 三条用例均 15/15（正常/轨迹生成失败/写工件失败），首次真实配对工件保留未重写，见 [FIXES.md](experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/FIXES.md)
- [x] **三对真实回放（6 次工作流，均 verified）**：
  - Pair 1 `A→B`：A `30/70→80/20`，2 揭晓/3 请求；B `30/70` 后停止，1 揭晓/2 请求
  - Pair 2 `B→A`：B `30/70` 后停止，1/2；A `30/70→80/20`，2/3
  - Pair 3 `A→B`：同 Pair 1
- [x] **结果**：3 对最佳均为 `30/70`（命中 5 / Recall@100 0.013672）；A 均 2 次揭晓、B 均 1 次；**复现 3/3，可判定 3/3，计划 3/3**；模型请求合计 **15/18**
- [x] **范围**：训练 0、重新召回 0、真实评测 0；只读 8 个汇总 JSON 与已冻结基线；候选明细 / 缓存 / 数据集 / 实验记忆索引未读取；来源与冻结基线哈希前后一致，旧工件未改写
- [ ] **Phase 13 完整版**：正式实验（更多历史/配额、其他模型/数据规模、策略泛化与测试集评估）仍待办

**交付**：`experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/run-20261109T061138Z-30480/`
（scenario_config / base_config / sources / frozen / pairs/pair-1..3 / metrics / self_check / run_meta / report）。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.offline_replay_scenarios --self-check `
  --config configs/experiments/offline_replay_70_30_ablation.toml
python -m recsys.experiments.offline_replay_scenarios --paired `
  --config configs/experiments/offline_replay_70_30_ablation.toml `
  --output-root experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired
```

证据：report.md（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/run-20261109T061138Z-30480/report.md`）、
metrics.json（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/run-20261109T061138Z-30480/metrics.json`）、
frozen/manifest.json（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/run-20261109T061138Z-30480/frozen/manifest.json`）、
self_check.json（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/run-20261109T061138Z-30480/self_check.json`）、
run_meta.json（本地文件：`experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/run-20261109T061138Z-30480/run_meta.json`）、
[FIXES.md](experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/FIXES.md)、
[入口](src/recsys/experiments/offline_replay_scenarios.py)、
[消融/配对配置](configs/experiments/offline_replay_70_30_ablation.toml)。

---

### 2.33 Phase 13 kuasar_small 独立测试集融合验证（2026-11-09）

**目标**：把验证集选定的 `30/70` 带到 `test` split，与 `50/50` 在冻结的 128 个测试用户上对照；只验证召回融合，不训练、不排序、不重排、不调用 LLM。

- [x] **先冻结配置**：`configs/experiments/kuasar_small_test_fusion.toml`；test split、seed 42、最多 128 人、Top-100、线程 4、墙钟上限 300 s；配额固定 `50/50` 与 `30/70`
- [x] **冻结样本**：128 用户 / **269 目标**；用户集合 sha256 `efccbd4679f5a2ccb3dd730f05ab73f14ad968ff29abd6d5b1849c576d5b4f69`；目标对 sha256 `bc08a91604b9ffec22c8239c71458460b49af78c7c166d407d036f9a1cd87c54`；未按命中筛用户、未换 seed
- [x] **时间边界**：269 个测试目标进入 train+validation 历史 0、进入 test 已见过滤 0、目标时间违规 0；本地夹具 8 用户 / 22 目标先通过，再启动正式命令
- [x] **权重复用**：双塔 `two_tower_features.pt` sha256 `784a2ad68972884dac4b0a652d44bb4a4d802c5788517fec05b6a25c6c1f0c3b`；MIND `mind20.pt` sha256 `eba017182142a8d09183ccf93de99d53535b32830de0a27bbdfad70d40ff9dac`；路由 seed `1234`
- [x] **一次召回、两组评测**：独立 `cache/test/` 中双塔/MIND 各召回一次（各 12,800 行），`50/50` 与 `30/70` 读取同一缓存各评测一次
- [x] **结果**：
  - `50/50`：命中 6；Recall@100 **0.023698**；Coverage **0.026724**
  - `30/70`：命中 6；Recall@100 **0.023698**；Coverage **0.026570**
  - 差值：命中 **+0**；Recall@100 **+0.000000**；Coverage **−0.000154**
  - 在这批测试用户上未观察到 `30/70` 相对 `50/50` 的提升；Recall 持平，Coverage 略低
- [x] **预算**：正式命令 1 次；双塔/MIND 各召回 1 次；两组各评测 1 次；训练 0、LightGBM 0、重排 0、LLM 0；总耗时 **12.298 s / 300 s**；实际 4 线程
- [x] **隔离**：测试缓存独立目录并标记 `split=test`，未读取/改写验证集缓存；原有 Phase 13 回放证据保留

**交付**：`experiments/2026-11-09-kuasar_small-test-fusion/run-20261109T070024Z-25952/`
（frozen_config / frozen_users / frozen_fingerprints / cache/test / evaluations/50_50 / evaluations/30_70 / metrics / run_meta / report）。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.recsys_test_fusion --fixture-check `
  --config configs/experiments/kuasar_small_test_fusion.toml
python -m recsys.experiments.recsys_test_fusion `
  --config configs/experiments/kuasar_small_test_fusion.toml `
  --output-root experiments/2026-11-09-kuasar_small-test-fusion
```

证据：report.md（本地文件：`experiments/2026-11-09-kuasar_small-test-fusion/run-20261109T070024Z-25952/report.md`）、
metrics.json（本地文件：`experiments/2026-11-09-kuasar_small-test-fusion/run-20261109T070024Z-25952/metrics.json`）、
run_meta.json（本地文件：`experiments/2026-11-09-kuasar_small-test-fusion/run-20261109T070024Z-25952/run_meta.json`）、
frozen_config.toml（本地文件：`experiments/2026-11-09-kuasar_small-test-fusion/run-20261109T070024Z-25952/frozen_config.toml`）、
frozen_users.json（本地文件：`experiments/2026-11-09-kuasar_small-test-fusion/run-20261109T070024Z-25952/frozen_users.json`）、
测试缓存元信息（本地文件：`experiments/2026-11-09-kuasar_small-test-fusion/run-20261109T070024Z-25952/cache/test/cache_meta.json`）、
[入口](src/recsys/experiments/recsys_test_fusion.py)、
[配置](configs/experiments/kuasar_small_test_fusion.toml)。

---

### 2.34 Search MVP：词表核验、30 会话小样本与 BM25（2026-11-09）

**目标**：完成搜索模块第一条链路：核验 `src_inter.keyword` 与 `item_features.caption` 是否共用 token ID，准备 ≤30 会话 / ≤1,000 物品的小样本，建 BM25 索引，提供 `search(query_tokens, top_k)` 与独立 CLI；只跑一次冒烟。不训练、不跑全量检索评测、不做 Search Agent/向量/排序。

- [x] **词表核验**：archive 只有 5 个 CSV，没有 vocab/dictionary；论文 §3.2 说明 query 与 video title 分词后随机哈希成整数。抽样（200k 搜索行 / 500k 物品行）keyword 与 caption token 范围 22–4,233,695 vs 2–4,233,996，交集 **13,747**（均匀独立基线 2,414.36 的 **5.69×**），点击物品 token 重合率 **51.16%**、相对随机 caption **73.32×**；结论共用 token 空间，直接使用 KuaiSAR token ID，**未用本地夹具**
- [x] **扫描上限**：先定死 `src_inter` 前 **200,000** 行、`item_features` 前 **500,000** 行；达到上限即停，两个输入均标记 `truncated=true`；`partial_sessions_dropped=0`、`keyword_mismatch_sessions=0`
- [x] **样本**：17,036 个完整会话 → 190 个候选 → **30 个会话 / 120 个物品**；曝光 215 行、关联 120 行（**55.814%**）；点击 60 行、关联 43 行；会话覆盖率 mean 56.971% / min 50% / max 80%；物品池上限 1,000 未触发
- [x] **索引**：`src/recsys/search/bm25.py`，k1=1.5、b=0.75；**120 docs / 823 vocab / avgdl 16.266667**；索引 76,077 B，sha256 `3ae2b858…`；排序 `(-score, item_id)`，空查询/无匹配返回 `[]`
- [x] **接口与 CLI**：`BM25Index.search(query_tokens, top_k)` + `python -m recsys.search --tokens ... --top-k ...`；返回 item_id、BM25 分数、可读类目；`python -m recsys.search.prepare --check` 校验产物
- [x] **单次冒烟 9/9**：会话 11208 的 4-token 查询（`[3538762, 3181091, 4061567, 3525929]`）有 7 个匹配，Top-5 分数降序、rank 连续、字段齐全，点击物品 80675 排第 1；空查询与 token `1000000000000` 均返回 `[]`
- [x] **范围**：正式训练 0、全量检索评测 0、Search Agent/向量/排序 0；只读 `src_inter.csv` / `item_features.csv` 前缀，未改写 `data/kuasar/processed/`；逐条样本只落一份 `sample.json`，索引另存 `index.json`
- [ ] Search Agent 工具封装、向量检索、排序/重排、更大样本或全量检索评测

**交付**：`experiments/2026-11-09-kuasar-search-bm25/`
（README / report / build_summary / smoke）；索引与样本在 `artifacts/search/kuasar_bm25/`（Git 忽略）。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.search.prepare --force `
  --summary-out experiments/2026-11-09-kuasar-search-bm25/build_summary.json
python -m recsys.search --tokens "3538762,3181091,4061567,3525929" --top-k 5
python -m recsys.search.smoke --top-k 5 `
  --output experiments/2026-11-09-kuasar-search-bm25/smoke.json
```

证据：[report.md](experiments/2026-11-09-kuasar-search-bm25/report.md)、
[build_summary.json](experiments/2026-11-09-kuasar-search-bm25/build_summary.json)、
[smoke.json](experiments/2026-11-09-kuasar-search-bm25/smoke.json)、
[词表/采样说明](docs/search.md)、
[prepare.py](src/recsys/search/prepare.py)、
[bm25.py](src/recsys/search/bm25.py)、
[查询 CLI](src/recsys/search/__main__.py)、
[冒烟脚本](src/recsys/search/smoke.py)、
[配置](configs/search/kuasar_search.toml)。

---

### 2.35 Search Agent MVP：search_bm25 工具调用与一次反馈决策（2026-11-09）

**目标**：复用 Search 1 的 120 物品 BM25 索引与 `BM25Index.search`，让 Agent 通过原生工具调用查一次，收到结果后自主停止或从原查询保留部分 token 再查一次；最多 2 次检索 / 3 次模型请求；只启动一次真实工作流。

- [x] **工具**：新增 `SearchBM25Tool` / `search_bm25` schema，直接调用 `BM25Index.search`，最多返回 5 条 `item_id/score/category`；不加载 CSV、不重建索引
- [x] **查询规则**：首查必须原样等于 `original_query_tokens`；第二查必须非空、只用原查询 token、至少去掉一个 token、且不与之前相同；参数/重复/预算均在执行工具前检查
- [x] **LangGraph**：`request_first → execute_first → request_feedback`；停止直接 `finish`，第二查走 `execute_second → request_final`；查询预算用完后 `tools=None`，再次请求工具会被拒绝
- [x] **预算与失败**：最多 2 次 BM25 / 3 次模型请求；请求在调用前计数；关闭自动重试；最终回复截断/为空/再次请求工具时保留已有轨迹并记 `partial`
- [x] **本地 self-check 12/12**（不联网）：一次停止、两次查询、首查不匹配、第二查非子集/重复/为空、最终再次请求工具、截断回复、tool_call_id 回传、预算上限、无候选落盘、最终回复候选截断
- [x] **唯一一次真实工作流**：`deepseek-chat`；查询为 Search 1 会话 11208 的 4 个 token；原生 tool call ID `call_00_lnA1GQAShje8UOqfaZph1633`；第一次返回 **5 条**；模型收到结果后**主动停止**（最高分 17.86、高科技数码类目、剩余 1 次预算下删 token 收益有限）；**2 次请求 / 1 次检索**，状态 **verified**，11/11 检查通过，耗时 2.876 s
- [x] **索引/样本不变**：运行前后索引 sha256 均为 `3ae2b858…`，样本 `12a46f3c…`；未重建索引、未扫原始 CSV
- [x] **落盘边界**：轨迹只保留 tool_call_id、查询 token、返回数量、停止理由与请求计数；模型最终回复原始 503 字符含逐条候选，落盘前截断为停止理由，原始回复仅留 sha256 `bc0963ce…`；`candidates_persisted=false`
- [ ] 实时第二次查询路径（本轮模型选择停止，二次规则仅由 self-check 合成覆盖）
- [ ] 中文文本输入 / query→token 编码、向量检索、排序/重排、全量检索评测

**交付**：`experiments/2026-11-09-kuasar-search-agent/`（README / report / self_check / 唯一 run 目录）。
**实现**：[tool.py](src/recsys/search/tool.py)、[search_agent.py](src/recsys/experiments/search_agent.py)、
[配置](configs/experiments/search_agent.toml)、[说明](docs/search-agent.md)。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.search_agent --self-check
python -m recsys.experiments.search_agent `
  --tokens "3538762,3181091,4061567,3525929" `
  --output-root experiments/2026-11-09-kuasar-search-agent
```

证据：[report.md](experiments/2026-11-09-kuasar-search-agent/report.md)、
[self_check.json](experiments/2026-11-09-kuasar-search-agent/self_check.json)、
trace.json（本地文件：`experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/trace.json`）、
metrics.json（本地文件：`experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/metrics.json`）、
run_meta.json（本地文件：`experiments/2026-11-09-kuasar-search-agent/run-20261109T080657Z-19556/run_meta.json`）。

---

### 2.36 Search Agent 修复：候选不落盘 + 样本外查询（2026-11-09）

**目标**：只修 `search_agent.py` 两个问题，保留 CLI、BM25 接口、LangGraph 与预算规则；只跑本地替身回归，不调用真实 LLM、不重建索引、不重跑真实冒烟。

- [x] **候选/模型自由文本不落盘**：删除 `_sanitize_reply` / `_remember_final_reply` 的关键词截断与正文保存；新增程序生成的 `_stop_summary`，轨迹/指标/事件只保留调用 ID、查询 token、返回数量、停止类型、预算、`final_reply_status` 与字符数；不再写 `content_preview` / `final_reply` / 原始回复。完成状态判断保留，`length`/空回复仍走 `partial`
- [x] **样本外查询**：`load_context` 的 `--tokens` 分支不再要求命中样本会话；匹配时补 `session_id`，匹配不到时 `session_id=null`、`source=cli_tokens_out_of_sample` 继续查询；`--session-id`、默认会话、两者同时传入的一致性校验不变；首查原样、第二查子集/预算规则不变
- [x] **回归入口**：`python -m recsys.experiments.search_agent --regression --tokens 2 ...`，本地 `CandidateRepeatingChat` 用普通句子复述工具返回的候选 ID；检查实际序列化的 trace/metrics/run_meta
- [x] **回归结果 45/45**：样本会话 5 条；样本外 `[2]` 返回 1 条且 `session_id=null`；样本外 `[1000000000000]` 返回 0 条；`--session-id` + 不一致 token 被拒绝；三个替身 case 均 verified
- [x] **落盘检查**：序列化文件无候选 ID、无替身模型原始回复、无 `content_preview`；`rules.model_free_text_persisted=false`、`candidates_persisted=false`；调用 ID/返回数量/停止类型仍可核对
- [x] **旧证据保留**：索引 `3ae2b858…`、样本 `12a46f3c…`、旧真实 trace/metrics/run_meta/self_check 哈希前后一致；新 self-check 12/12 写入 `artifacts/`，不覆盖旧 `self_check.json`；真实 LLM 请求 0、索引重建 0、真实冒烟重跑 0
- [ ] 真实模型是否遵守“不复述候选”的提示未验证；代码只保证即使复述也不落盘

**交付**：`experiments/2026-11-09-kuasar-search-agent-regression/`（README / report / regression.json）；
实际工件在 `artifacts/search_agent_regression_tokens/`（Git 忽略）。
**改动**：[search_agent.py](src/recsys/experiments/search_agent.py)。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.experiments.search_agent `
  --regression --tokens 2 `
  --output-root artifacts/search_agent_regression_tokens
```

证据：[regression report](experiments/2026-11-09-kuasar-search-agent-regression/report.md)、
[regression.json](experiments/2026-11-09-kuasar-search-agent-regression/regression.json)、
[说明](docs/search-agent.md)。

---

### 2.37 Phase 2 完整搜索系统：FAISS 向量 + RRF 混合 + 精排 + Search Agent 四工具（2026-11-09）

**目标**：在已有 BM25 与 Search Agent 上补齐搜索侧 FAISS 向量召回、混合检索、精排和
Query Understanding，并把四个工具接进现有 Agent 工作流；只做 120 物品 / 30 会话小样本
链路验收，不训练、不微调、不建全量索引、不跑分。

- [x] **复用 SkimSearchAgent（固定提交 4794575）**：`vector_index`（索引保存/加载/元数据）、
      `fusion.rrf.RRF`（融合）、`rerankers.Reranker` + `register_reranker`（精排契约）、
      `reranked.RerankedEngine`（检索+精排组合）；上游源码未改，适配层见
      [upstream.py](src/recsys/search/upstream.py)
- [x] **向量编码**：[encoders.py](src/recsys/search/encoders.py) 提供可插拔 `SearchEncoder`。
      当前用无训练的 `TokenHashEncoder`（4096 维、稳定 blake2b 哈希、次线性词频、L2 归一化），
      **查询与文档共用同一编码器实例和配置**；OOV token 在查询侧被过滤（与 BM25 posting 一致）。
      `TextSemanticEncoder` 接口已留：缺 KuaiSAR 原文、缺本地权重、缺
      `sentence-transformers`，验收时明确报错并记录缺口
- [x] **搜索侧 FAISS**：[vector_index.py](src/recsys/search/vector_index.py) 新增
      `FaissFlatIndex`（`faiss.IndexFlatIP`，精确余弦）并登记进上游 backend 表；
      实现整数 item_id 映射、Top-K、保存/重载（`doc_ids.json` + `meta.json` + `ann.faiss`）、
      `add_items` 增量更新。索引目录 `artifacts/search/kuasar_vector/`，**独立于推荐索引**
- [x] **混合检索**：[hybrid.py](src/recsys/search/hybrid.py) 用上游 RRF 融合 BM25 与向量候选，
      按 item_id 去重，保留每路来源、名次和实际分数；融合分 = Σ 1/(60+rank)
- [x] **精排**：[rerank.py](src/recsys/search/rerank.py) 的 `TokenOverlapReranker` 实现上游
      `Reranker` 契约，接在融合候选池后面：`0.65·idf 加权 token 覆盖率 + 0.35·融合先验`，
      排序 `(-score, item_id)`；点击标签只用于验收核对，不参与打分
- [x] **Query Understanding**：[query.py](src/recsys/search/query.py) 做规范化（正整数/去重/限长）
      和受约束改写（只删除 token）；原始查询与规范化/改写形态都写入 agent trace
- [x] **CLI**：`python -m recsys.search --mode bm25|vector|hybrid|rerank --tokens ... --top-k ...`；
      旧 BM25 用法不变；新增 `python -m recsys.search.build_vector_index`
- [x] **Search Agent**：注册 `search_bm25` / `search_vector` / `search_hybrid` / `search_rerank`；
      首轮必须是检索工具且用规范化原查询；反馈阶段三选一：停止、精排（查询必须是已检索过的）、
      一次受约束二次检索；预算 2 检索 + 1 精排 + 3 模型请求；空结果/拒绝/超时/截断路径保留
- [x] **验收 9/9**（[acceptance.json](experiments/2026-11-09-kuasar-search-phase2/acceptance.json)）：
      旧 BM25 库入口 + CLI 一致；FAISS 后端 `faiss.IndexFlatIP`、查询后调用计数 +1、重载后 Top-5 一致；
      119→120 的 `add_items` 更新可保存重载；混合去重/来源/融合分/顺序与上游 RRF 一致；
      精排最终 Top-5 分数可复算；Agent `search_hybrid → search_rerank` 链路 verified；
      无匹配返回 0 条、空查询与「首轮精排」被拒绝；BM25 索引/样本 sha256 前后一致
- [x] **Agent 自检 17/17 + 本地回归 45/45**：新增跨工具二次检索、混合→精排、精排前置拒绝、
      精排未知查询拒绝、混合来源、空结果等用例；旧真实工件与索引哈希不变
- [x] **真实 LLM 工作流 1 次**：`deepseek-chat` 在三个检索工具中自主选择 `search_hybrid`，
      看到 5 条融合结果后调用 `search_rerank`，最终回复结束；3 请求 / 1 检索 / 1 精排，
      **verified**，17/17 运行时检查，2.83 s，轨迹无候选 ID / 无模型正文
- [x] **范围**：训练 0、微调 0、BM25 重建 0、全量索引 0、原始 CSV 扫描 0；验收脚本本身
      不调用 LLM，另单跑 1 次真实工作流（3 次请求）；检索/精排/索引/落盘均为真实代码
- [ ] 真实 KuaiSAR 文本或可下载预训练权重（启用 `TextSemanticEncoder` / 语义精排）
- [ ] 更大样本或全量检索评测（Recall/MRR/NDCG）、查询延迟与索引构建时间

**交付**：`experiments/2026-11-09-kuasar-search-phase2/`
（acceptance.json / report.md / agent_scripted）；向量索引在 `artifacts/search/kuasar_vector/`（Git 忽略）。

**实现**：[upstream.py](src/recsys/search/upstream.py)、[encoders.py](src/recsys/search/encoders.py)、
[vector_index.py](src/recsys/search/vector_index.py)、[hybrid.py](src/recsys/search/hybrid.py)、
[rerank.py](src/recsys/search/rerank.py)、[query.py](src/recsys/search/query.py)、
[service.py](src/recsys/search/service.py)、[tool.py](src/recsys/search/tool.py)、
[build_vector_index.py](src/recsys/search/build_vector_index.py)、[acceptance.py](src/recsys/search/acceptance.py)、
[查询 CLI](src/recsys/search/__main__.py)、[search_agent.py](src/recsys/experiments/search_agent.py)、
[搜索配置](configs/search/kuasar_search_phase2.toml)、[Agent 配置](configs/experiments/search_agent.toml)。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.search.build_vector_index --force
python -m recsys.search --mode hybrid --tokens "3538762,3181091,4061567,3525929" --top-k 5
python -m recsys.search --mode rerank --tokens "3538762,3181091,4061567,3525929" --top-k 5
python -m recsys.search.acceptance
python -m recsys.experiments.search_agent --self-check
python -m recsys.experiments.search_agent --scripted-plan hybrid_rerank `
  --output-root experiments/2026-11-09-kuasar-search-phase2/agent_scripted
```

证据：[验收 JSON](experiments/2026-11-09-kuasar-search-phase2/acceptance.json)、
[报告](experiments/2026-11-09-kuasar-search-phase2/report.md)、
脚本替身轨迹（本地文件：`experiments/2026-11-09-kuasar-search-phase2/agent_scripted/scripted-hybrid_rerank/trace.json`）、
真实 LLM 轨迹（本地文件：`experiments/2026-11-09-kuasar-search-phase2/agent_llm/run-20261109T134757Z-32924/trace.json`）、
[搜索说明](docs/search.md)、[Search Agent 说明](docs/search-agent.md)。

---

### 2.38 Phase 3 完整广告系统：样本、六模型、排序、竞价预算与 Ads Agent（2026-11-09）

**目标**：把已有曝光/点击/`is_ad`/特征/请求分组接成广告样本，接入六个广告模型，跑通
CTR/多任务预测、广告排序、二价竞价与预算模拟，并复用 YieldAgent 结构交付独立 Ads Agent。
只做小样本接口验收，不训练、不跑全量、不调参。

- [x] **广告样本** `src/recsys/ads/data.py`：同一曝光请求 `(user_id, timestamp)` 内
      `is_ad == 1` 的物品为广告候选，`click` 为标签；ID 通过 `map_parent_to_target`
      映射到 `kuasar_small` 词表；切分沿用父数据集请求级切分。产物
      `data/kuasar_small/processed/ads/`（5 文件/切分 + schema + sha256）。
      train **5,483 候选 / 5,186 请求（278 个 ≥2 候选）**、validation **51 / 49 / 2**、
      test **43 / 41 / 2**；点击率 0.7011 / 0.6863 / 0.8372。
- [x] **序列历史**：DIN/DIEN 只读 `interaction_timestamp < request_timestamp` 的交互
      （`histories_for_requests`），同请求候选共享一份；DCNv2 不使用序列历史。
- [x] **真实转化标签核对** `audit_conversion_labels()`：`rec_inter.csv` 只有
      click/like/follow/forward/play_time 等字段，**没有 conversion/purchase**；
      schema 写 `available=false`、`cvr_mode=simulated`。未用 click 冒充转化。
- [x] **独立模拟样本** `src/recsys/ads/simulated.py`：`ads_simulated/` 304 行 / 211 点击 /
      50 合成转化；规则 `blake2b(user:item:seed)%100<20`；schema 标 `synthetic=true`、
      `not_real_conversion=true`、`not_for_effect_claims=true`。
- [x] **六个模型统一接口** `src/recsys/ads/models.py`：
      DIN/DIEN/DCNv2 直接复用 Phase 1 模型与 `build_model`；DeepFM/MMoE/PLE 复用本地
      DeepCTR-Torch（DeepFM 单任务，MMoE/PLE `ctr`+`cvr` 双任务）；
      `score / save / load_ad_model` 统一，词表不一致或缺依赖明确报错。
- [x] **权重隔离与验收** `src/recsys/ads/verify.py` + `registry.py`：固定初始化权重写
      `artifacts/kuasar_small/ads/interface_weights/`，业务默认目录 `.../ads/weights/` 未写入；
      六模型 **6/6 推理 ok、6/6 保存/加载差值 0.0**，MMoE/PLE 双输出在 synthetic 样本上验证；
      状态统一 `interface_test / is_trained=false`（见 `model_check.json`）。
- [x] **广告排序** `src/recsys/ads/ranking.py`：请求内 pCTR 降序、平局 item_id 升序；
      返回名次/分数/点击标签/广告主。
- [x] **二价竞价 + 预算** `src/recsys/ads/auction.py` + `pipeline.py`：
      `eCPM = bid×pCTR×1000`、GSP 二价结算、**CPC 计费、点击时扣费**、总预算 + 广告主预算；
      实跑 3 请求：赢家 3、扣费 2.1672、剩余 17.8328 / 20；0 预算 3/3 未成交、
      `budget_exhausted_total`、扣费 0；`spend==Σcharge`、`remaining==total-spent` 通过。
- [x] **Ads Agent** `src/recsys/ads/tools.py` + `agent.py`：7 个真实工具
      （list_ad_models / predict_ad_ctr / rank_ads / simulate_auction / get_budget_state /
      check_label_availability / predict_multitask），LangGraph `StateGraph`，
      LLM 原生 `tool_calls` + 同 ID 回传，本地确定性计划器可离线复现；
      预算 8 工具 / 6 LLM 请求 / 180 s；轨迹落 `events.jsonl`、`trace.json`、`metrics.json`、
      `run_meta.json`、`report.md`。
- [x] **YieldAgent 复用（e79fda0）**：工具/业务分离（tools 只调度、service 承载业务，
      对应 `console/tools.py` + `connectors/base.py` 分层）、工具错误回传
      `{ok:false}`（对应 `ToolNode(handle_tool_errors=True)`）、工具调用/结果记录
      （`coerce_payload`/`summarize_payload` 改写自 `runtime._coerce/_summarize`）。
      未用 `create_react_agent`/`@tool`（`langchain` 未安装）；LinkedIn/审批/网页/外部账户不在范围。
- [x] **本地 Agent 实跑**：`din`，7 工具全部 ok，**verified**，4.66 s；
      CTR test 43 行 AUC 0.5833（固定初始化）、排序 3 请求/9 候选、竞价扣费 2.1672、0 预算无成交。
- [x] **真实 LLM 实跑（1 次成功）**：`deepseek-chat` 自主选择 `deepfm`，4 工具 / 3 请求 /
      21.25 s，**verified**；竞价 20 请求全部成交、花费 9.9028、剩余 10.0972；最终回复
      正确说明未训练与 CVR 缺口。首次因 `max_llm_requests=4` 不够收尾而 `partial`
      （`run-20261109T142616Z-30600/`），改为 6 后重跑 verified。
- [x] **自检 14/14**：六模型/未知模型/预测/排序/竞价/预算状态/0 预算/多任务/标签核对/
      LLM 同 ID 回传/未知工具/重复调用/工具预算/本地端到端（5 工件齐全）。
- [x] **总验收** `scripts/smoke_phase3_ads.py`：data/models/ranking/auction/agent_self_check
      五项全 true；`zoo.py check` 26 模型 ready=22 / blocked=4；`check_project.py` 0 错误。
- [ ] 训练六个广告模型（需用户明确要求并单独安排）；真实转化/出价数据接入；更大样本排序效果

**交付**：`experiments/2026-11-09-kuasar_small-ads-phase3/`
（report / model_check / smoke / self_check / agent_local/run-* / agent_llm/run-*）。

**复现**：

```powershell
$env:PYTHONPATH = "src"
python -m recsys.ads.data --check
python -m recsys.ads.simulated --check
python -m recsys.ads.verify
python -m recsys.ads.pipeline --model din --budget-check
python -m recsys.ads.agent --self-check
python -m recsys.ads.agent --planner local
python scripts/smoke_phase3_ads.py
```

证据：[交付报告](experiments/2026-11-09-kuasar_small-ads-phase3/report.md)、
[模型验收](experiments/2026-11-09-kuasar_small-ads-phase3/model_check.json)、
[总验收](experiments/2026-11-09-kuasar_small-ads-phase3/smoke.json)、
[Agent 自检](experiments/2026-11-09-kuasar_small-ads-phase3/self_check.json)、
本地轨迹（本地文件：`experiments/2026-11-09-kuasar_small-ads-phase3/agent_local/run-20261109T143340Z-24812/trace.json`）、
真实 LLM 轨迹（本地文件：`experiments/2026-11-09-kuasar_small-ads-phase3/agent_llm/run-20261109T142641Z-25212/trace.json`）、
[广告说明](docs/ads.md)、[Ads Agent 说明](docs/ads-agent.md)。

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
- [x] Phase 11 Research Agent MVP：`evaluate_fusion_quota` 工具 + 配额决策闭环（本地策略，
      3 组评测，见 2.23）
- [x] Phase 11 真实 LLM tool-calling：**已验收**（deepseek-chat 返回 1 条 `tool_calls`，
      同 `tool_call_id` 回传，2 次请求 / 1 次工具 / 3.05 s，见 2.24）
- [x] Phase 11 LangGraph 工作流：**真实运行 + 失败路径回归通过**（见 2.25）；
      误报修复：第二次请求失败不再写 `verified`
- [x] Phase 11.2 多工具注册 + 历史实验复用：**真实运行通过**（30/70 命中历史，
      5 个命中 / Recall@100 0.013672，评测 0 次，见 2.26）；`--cache-dir` 已贯穿全流程
- [x] Phase 12 实验记忆 MVP：新实验自动入库 + 第二次独立进程零评测复用
      （80/20，2 命中 / Recall@100 0.005208，见 2.27）；**失败结果不发布**
      （发布时机修复 + 独立回归脚本 + 唯一输出目录，验收完成）
- [x] Phase 13 离线回放 MVP：**1 次受控对照完成**（Agent 与固定策略最佳均为 30/70，
      命中 5 / Recall@100 0.013672 / Coverage 0.026720，结论持平；见 2.28）
- [x] Phase 13 真实 LLM 离线回放：**verified**（deepseek-chat，3 请求 / 2 揭晓 / 5.244 s；
      真实 LLM、本地规则 Agent、固定策略最佳均为 30/70，三方最佳持平；见 2.29）
- [x] Phase 13 多场景离线对照：**A/B 均 verified**（场景 A 本地规则胜、真实 LLM 输；
      场景 B 三方最佳均为 30/70 持平；两场景合计 5 请求；见 2.30）
- [x] Phase 13 `70/30` 历史信息消融：**两组均 verified**；两组最佳均 30/70，
      历史组少揭晓 1 次（2 → 1），记为节省 1 次查询；见 2.31
- [x] Phase 13 历史信息节省查询成对复验：**3/3 对复现**（A 均 2 揭晓、B 均 1 揭晓，
      最佳均 30/70）；6 次工作流 / 15 请求；见 2.32
- [x] Phase 13 kuasar_small 独立测试集融合验证：128 测试用户 / 269 目标；
      `50/50` 与 `30/70` 均命中 6、Recall@100 0.023698；见 2.33
- [ ] Phase 13 完整版：正式实验（更多场景与配额、策略泛化、测试集评估）
- [ ] Phase 11 完整版：RAG、Agent 对照实验（工具注册与历史复用已完成）
- [ ] Phase 12–13 完整 RAG、Agent 对照实验（含 Phase 13 正式版）

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

## 七、Phase 1（v2）：完整推荐系统与 Rec Agent（2026-11-09）

> 本节对应 `PROJECT_MASTER_v2.md` Phase 1。范围只含推荐系统与 Rec Agent；搜索、广告、
> Master Agent 未在本轮改动。

### 7.1 交付概览

- 新增统一推荐链路 `python -m recsys.pipeline`：多路召回 → 去重融合 → 候选打分 →
  类目/多样性重排 → 最终输出 → 逐阶段离线评估。
- 模型注册表从 6 个召回 + 1 个排序扩展到 **11 个召回 + 5 个排序**；19 个模型达到
  `ready`，2 个保持 `upstream_ready`（DeepFM/MMoE，广告阶段用），4 个保持 `blocked`
  并记录可执行替代（OneRec / TFRS Two-Tower / TFRS factorized_top_k / TorchEasyRec SID）。
- Rec Agent 在既有 LangGraph + 工具注册表上扩展为 12 个真实工具，覆盖数据画像、
  Model Zoo、训练、召回、排序、重排、评估、历史实验查询。
- 默认双路链路（Two-Tower + MIND → 轮流融合 → LightGBM → 类目重排）行为保留；
  既有 `recsys_mvp.py` 和实验工件未改动。
- Top-20 零命中完成固定小样本定位，记录在 [docs/top20-diagnosis.md](docs/top20-diagnosis.md)。

### 7.2 逐模型状态

口径：源码 → 已核源码 → 项目适配 → 可运行 → 业务接入 → Agent 可调用 → 已评估。
`models/zoo.toml` 已逐项写入 `status`，`scripts/zoo.py render` 生成状态表。

| 模型 | 源码 | 适配 | 可运行 | 业务接入 | Agent | 已评估 | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 无参数基线；热度分数、链路、Agent 均可用 |
| BPR-MF | ✓ | ✓ | ✓ | ✓ | ✓ | — | 本轮用 Agent `train_model` 真实训练 1 epoch/2 batch；指标为冒烟值 |
| Two-Tower | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 默认召回通道；既有 checkpoint 复用 |
| MIND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 默认多兴趣通道；既有 checkpoint 复用 |
| LightGCN | ✓ | ✓ | ✓ | ✓ | ✓ | — | 新写纯 PyTorch BPR 实现；1 epoch/2 batch 训练 + 链路 + 往返通过 |
| DSSM | ✓ | ✓ | ✓ | ✓ | ✓ | — | 复用 Two-Tower MLP 特征塔；1 epoch/2 batch 训练通过 |
| SASRec | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 既有实现；本轮补 validation 子集参数并接入链路 |
| BERT4Rec | ✓ | ✓ | ✓ | ✓ | ✓ | — | 新写 PyTorch Cloze 实现；1 epoch/2 batch 训练通过 |
| HSTU | ✓ | ✓ | ✓ | ✓ | ✓ | — | 新写相对偏置 + gating 实现；1 epoch/2 batch 训练通过 |
| TIGER | ✓ | ✓ | ✓ | ✓ | ✓ | — | 新写 SID 生成 + trie beam；依赖 RQ-VAE 码表 |
| RQ-VAE / Semantic ID | ✓ | ✓ | ✓ | ✓ | ✓ | — | 元数据内容向量；RQ-VAE 塌缩时退化到 RQ-KMeans（见 7.4） |
| MMRec 多模态族 | ✓ | ✓ | ✓ | ✓ | ✓ | — | VBPR 公式的纯 PyTorch 适配；输入是元数据向量，不是视觉/文本 |
| Ultra-HSTU | ✓ | ✓ | ✓ | ✓ | ✓ | — | HSTU 主干 + 排序头；1 epoch/2 batch 训练通过 |
| DIN | ✓ | ✓ | ✓ | ✓ | ✓ | — | 纯 PyTorch 注意力池化；训练 + Agent 排序链路通过 |
| DIEN | ✓ | ✓ | ✓ | ✓ | ✓ | — | 纯 PyTorch AUGRU；1 epoch/2 batch 训练通过 |
| DCNv2 | ✓ | ✓ | ✓ | ✓ | ✓ | — | CrossNetV2 + DNN；1 epoch/2 batch 训练通过 |
| LightGBM LambdaRank | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 默认精排；既有 `mvp/lgbm_ranker.txt` 复用 |
| FAISS / 物品索引 | ✓ | ✓ | — | ✓ | — | ✓ | 基础设施；链路仍用精确打分，未切近似索引 |
| OneRec | ✓ | — | — | — | — | — | **blocked**：缺外部 Qwen3/OneRec 权重 + 多卡 Linux；替代为已接入的 TIGER |
| TFRS Two-Tower | ✓ | — | — | — | — | — | **blocked**：TF2 与主环境冲突；算法等价的项目 Two-Tower 已接入 |
| TFRS factorized_top_k | ✓ | — | — | — | — | — | **blocked**：同上，仅保留跨框架源码 |
| TorchEasyRec SID（sid_rqvae） | ✓ | — | — | — | — | — | **blocked**：tzrec 平台依赖；项目内 RQ-VAE 条目已可运行 |
| DeepFM / MMoE | ✓ | — | — | — | — | — | `upstream_ready`，按手册留给 Phase 3 广告 |

`python scripts/zoo.py check`：25 条模型，ready=19 / upstream_ready=2 / blocked=4，校验通过。

### 7.3 推荐链路

新增 `src/recsys/pipeline/`：

- `config.py`：TOML → `PipelineConfig`（run / recall / fusion / ranking / rerank）。
- `recall.py`：按模型类型选择真实打分路径；保留 `(item, score, rank)` 与来源。
- `fusion.py`：`round_robin` / `rrf` / `quota`，去重并保留每路名次/分数。
- `ranking.py`：LightGBM 与神经排序统一入口；LightGBM 缺失时可从父表 train 现训。
- `rerank.py`：类目上限 + 可选作者上限 + 补位计数。
- `chain.py`：一次运行、逐阶段指标、逐目标丢失环节、工件落盘与读回。
- `__main__.py`：`list` / `run` / `evaluate`。

默认配置 [configs/pipeline/kuasar_small_default.toml](configs/pipeline/kuasar_small_default.toml)
复现既有 MVP 的 128 用户 / 100 召回 / 20 最终 / LightGBM 20 轮 / 类目上限 5。
多路示例 [configs/pipeline/kuasar_small_multi_recall.toml](configs/pipeline/kuasar_small_multi_recall.toml)
为 Two-Tower + MIND + LightGCN → RRF → DIN → 类目/作者重排。

本轮实跑：

- 默认双路 16 用户：5 个阶段指标齐全，耗时 4.86 s（召回 0.71 + 3.97 s）。
- 128 用户固定样本与既有 `artifacts/kuasar_small/mvp/metrics.json` 逐项一致：
  two_tower Coverage 0.005566、fusion 0.005557、ranked 0.005395、rerank 0.005412；
  Top-100 命中 2/7/9、融合@20=0、重排@20=0。默认行为未被新链路改变。
- 多路 4 用户：3 路召回 + RRF + DIN + 重排通过（`scripts/smoke_phase1.py`）。
- 工件：`final.json` / `recall_<channel>.json` / `fusion.json` / `ranked.json` /
  `metrics.json` / `diagnostics.json` / `run_meta.json`，含 checkpoint sha256 与数据指纹。

### 7.4 RQ-VAE / Semantic ID 的码本塌缩处理

首次 RQ-VAE（2 epoch / codebook 32 / latent 8）只产生 **65 个唯一 SID**，属于码本
塌缩；如果直接给 TIGER，生成式召回只能覆盖 65 个物品。处理方式：

1. 提高配置重训（4 epoch / codebook 64 / latent 16）：RQ-VAE 唯一 SID = **83**；
2. 同时计算 RQ-KMeans（同属残差量化，MiniBatchKMeans 逐级聚类）：唯一 SID = **43,040**；
3. 导出码表选择唯一 SID 更多的 RQ-KMeans，并在 `rqvae.pt` 的 provenance 记录
   `code_method=rq_kmeans_fallback`、两种数量与选择原因；
4. TIGER 用 43,040 个 SID 重训（1 epoch/2 batch，冒烟 Recall@20=0）。

这不是把 RQ-VAE 冒充成可用；码本塌缩是真实发现，退化路径与原因都写进工件。
后续专项应继续调 RQ-VAE（码本初始化/commitment/训练轮数）并把唯一 SID 数作为验收项。

### 7.5 Rec Agent

新增 `src/recsys/experiments/rec_tools.py` 与 `rec_agent.py`，在既有 LangGraph 上：

- 12 个工具：`read_data_profile`、`query_model_zoo`、`list_artifacts`、`train_model`、
  `recall_models`、`rank_candidates`、`rerank_candidates`、`evaluate_recommendations`、
  `run_pipeline`、`evaluate_fusion_quota`、`search_experiment_history`、`list_experiments`。
- `train_model` 直接调用 16 个训练入口；`recall_models`/`rank_candidates` 调用真实
  适配器推理；`run_pipeline` 调用真实推荐链路；`evaluate_recommendations` 按 split 计算指标。
- 预算：10 次工具 / 2 次训练 / 600 秒 / 3 epoch / 20 batch；重复调用、非法参数、
  超预算、缺 checkpoint 都返回 `ok=false` 并进入 `summary.errors`。
- 有 LLM 时走原生 `tool_calls`，无 LLM/`--force-local` 时走确定性计划器，
  `summary.planner` 明确区分；本轮自检和实跑均为 `local`。

本轮真实执行记录：

- `--self-check`：7/7 通过（真实画像、真实 Zoo、未知模型拒绝、Popularity 召回、
  缺 DIN checkpoint 的明确失败、Popularity+LightGBM 真实链路）。
- `--train din --recall two_tower --ranker din`：训练 DIN（AUC 0.5143 / LogLoss
  0.6291 / NDCG@5 0.8970，1 epoch/2 batch）→ 真实链路 `ranked` 阶段使用 DIN → verified。
- `--train matrix_factorization --recall popularity --ranker lightgbm_ranker`：
  训练 BPR-MF → 真实链路 → verified。
- `--recall two_tower,mind --ranker lightgbm_ranker`：verified，`run_pipeline` 返回
  5 个阶段指标与工件路径。
- 单独工具链：`recall_models(two_tower)` → `rank_candidates(din)` →
  `rerank_candidates` → `evaluate_recommendations(final)` 全部 `ok=true`，状态在
  `ToolContext` 中串联（`scripts/smoke_phase1.py`）。
- 融合配额工具：`evaluate_fusion_quota(50,50)` 复用既有缓存，hits=4 /
  Recall@100=0.011719 / Coverage=0.026843，0.43 s。
- 真实 LLM 计划器已验证：`deepseek-chat` 一次工作流 **4 次请求 / 4 个工具**
  （`read_data_profile` → `query_model_zoo` → `run_pipeline` →
  `evaluate_recommendations`），耗时 14.5 s，`planner=llm` / `verified`；指标与
  `final.json`/`run_meta.json` 落盘在
  `experiments/2026-11-09-kuasar_small-rec-agent/run_llm_final/`。

### 7.6 Top-20 零命中

> **数据边界：** 该诊断是 128 用户 / 362 目标的固定小样本，只用于定位链路环节；KuaiSAR
> 全量 14.6M 交互 / 4M 物品，本机不做每次接入的全量训练。小样本 + 长尾 + 2k 子集本身就会
> 导致零命中，**不能据此判模型或代码错误，也不作为接入验收依据**。

固定 128 验证用户 / 362 目标，seed 42：

| 环节 | 目标数 |
| --- | --- |
| 不在任何通道 Top-100 | 353（97.51%） |
| 在通道并集、不在融合池 | 5 |
| 在融合池、不在融合 Top-20 | 4 |
| 融合 Top-20 之后丢失 | 0 |
| 最终命中 | 0 |

根因：目标物品 41.4% 在训练集零点击、中位点击 1 次；ID-only 双塔/MIND 无法稳定
召回；同时轮流融合 `limit=100` 只覆盖每路约前 50，深位候选（56/62/70/99）在融合
池被截掉；进入池的 4 个目标又被 Top-20 截断。排序/重排没有再丢目标。

可执行动作（留给效果专项）：扩大每路深度并先并集再去重；排序前不截到 20；增加
内容/Semantic ID/探索通道；扩大样本与多种子。详见 [docs/top20-diagnosis.md](docs/top20-diagnosis.md)。

### 7.7 本轮检查

| 检查 | 结果 |
| --- | --- |
| `python scripts/zoo.py check` | 25 条，ready=19 / upstream=2 / blocked=4，通过 |
| `python scripts/check_project.py` | 0 错误、9 警告（缓存目录/OpenOneRec LICENSE） |
| `python scripts/smoke_adapters.py` | 既有 4 个适配器 + 指标 + 溯源 + LightGBM 通过 |
| `python scripts/smoke_phase1.py` | 11 个召回适配器小样本推理/往返、5 个排序器打分/往返、Agent 工具链、默认与多路链路、Agent 自检全部通过 |
| `python -m recsys.pipeline run --users 16` | 默认双路链路通过 |
| `python scripts/diagnose_top20.py --users 128` | 固定样本诊断完成并落盘 |
| `python -m recsys.experiments.rec_agent --self-check` | 7/7 |
| 真实 LLM 计划器（`deepseek-chat`） | 4 请求 / 4 工具（画像 → Zoo → 链路 → 评估），verified，14.5 s |
| Agent 真实训练（DIN / BPR-MF） | 训练入口、checkpoint、排序/链路均真实执行 |
| 新模型训练冒烟 | LightGCN / DSSM / BERT4Rec / HSTU / RQ-VAE / TIGER / MMRec / DIN / DIEN / DCNv2 / Ultra-HSTU / SASRec 均产出 checkpoint |

说明：所有新模型只做了 1 epoch / 2 batch（部分 4 epoch）的**冒烟**，验证集指标为 0
或接近随机，不能作为效果结论；正式训练/调参/多种子/大规模 Benchmark 留给专项。

### 7.8 遗留阻碍与边界

1. **OneRec**：缺外部 Qwen3/OneRec 权重、RecIF 数据和多卡 Linux；本机 `verl` / `vllm` /
   `ray` 未安装，`transformers` 也未安装。影响：无法训练或推理 OneRec。替代：已接入的
   TIGER + Semantic ID（43,040 SID，冒烟级）；后续若提供权重，可在独立环境接
   `third_party/OpenOneRec` 的 Qwen3 主干。
2. **TFRS Two-Tower / factorized_top_k**：本机 `tensorflow` / `tf-keras` 不存在，
   Python 3.14 也没有可用 TF 轮子，与 PyTorch 主栈冲突。影响：TFRS checkpoint 不能
   在本环境加载。替代：算法等价的项目 Two-Tower（已接入、已评估），TFRS 只保留源码。
3. **TorchEasyRec SID / DSSM / HSTU / Ultra-HSTU 原实现**：`tzrec` 未安装，其
   graphlearn/pyfg/common_io 轮子只提供 Linux 版本。影响：不能直接跑上游训练/推理，
   checkpoint 不通用。替代：项目内纯 PyTorch 重写核心结构，已接入链路。
4. **MMRec 多模态**：`torch_geometric` / `torch_scatter` 未安装（Python 3.14 无可用
   预编译轮子），本地也没有视觉/文本 embedding。影响：上游 VBPR/LATTICE 不能原样跑。
   替代：VBPR 公式的纯 PyTorch 最小适配，用物品元数据内容向量；这不是真正的多模态。
5. **RQ-VAE 塌缩**：已用 RQ-KMeans 兜底导出可用码表（43,040 SID），但 RQ-VAE 本身
   只产出 83 个唯一 SID；唯一 SID 数应作为后续验收项。
6. **Top-20 零命中**：已定位为召回深度 + 融合截断 + 长尾冷物品，未在本轮修复；
   属于效果专项，不作为功能交付的阻塞。
7. **真实 LLM 计划器**：已验证一次 `deepseek-chat` 工作流（4 请求/4 工具，verified）；
   多次运行、限流、断网回退仍未覆盖。
8. **FAISS 索引**：推荐链路仍是精确打分；索引保存/加载能力在既有 `FaissItemIndex`，
   本轮未把链路切到近似检索。

### 7.9 本轮文件

新增：

- `src/recsys/pipeline/{__init__,__main__,config,recall,fusion,ranking,rerank,chain}.py`
- `src/recsys/models/{lightgcn,dssm,bert4rec,hstu,rqvae,tiger,mmrec_vbpr,din,dien,dcnv2,ultra_hstu}.py`
- `src/recsys/training/{train_lightgcn,train_dssm,train_sequence,train_rqvae,train_tiger,train_mmrec,train_neural_ranker,subsets}.py`
- `src/recsys/adapters/{graph,sequence,tiger,mmrec,neural_ranking,ranking_registry}.py`
- `src/recsys/ranking/{__init__,samples,features}.py`
- `src/recsys/evaluation/tiger_evaluator.py`
- `src/recsys/experiments/{rec_tools,rec_agent}.py`
- `configs/pipeline/{kuasar_small_default,kuasar_small_multi_recall}.toml`
- `scripts/{diagnose_top20,smoke_phase1}.py`
- `docs/{recsys-pipeline,rec-agent,top20-diagnosis}.md`

修改：

- `src/recsys/adapters/{builtin,registry,__init__,__main__,rechub_mind}.py`
- `src/recsys/training/{trainer,train_mf,train_mind,train_sasrec}.py`（新增验证用户子集参数）
- `models/zoo.toml`（逐模型 status、code、adapter、入口）
- `scripts/{zoo,check_project}.py`（status 校验/渲染；代码链接 `:line` 解析）
- `PROJECT_PROGRESS.md`、生成的 `docs/model-zoo.md`、`docs/models/*.md`、`models/*/README.md`

---

### 7.10 三个已确认问题修复（2026-11-09）

**1) 排序训练历史泄漏**

- 排序工件新增每请求 `{split}_timestamps.npy`（schema v2）；`load_ranking_samples` /
  `load_ranking_view` 支持时间戳，旧工件可读但无法给序列模型用。
- `ranking/samples.py` 新增按 `(user, request_timestamp)` 构造历史，严格
  `interaction_time < request_time`；同请求候选共享一份历史。
- `train_neural_ranker.py` 的 DIN/DIEN/Ultra-HSTU 强制使用请求时间戳；
  DCNv2 不需要时间戳，LightGBM 路径不变；旧工件报
  `RankingTimestampError` 并提示重建命令。
- 重建已执行：`python -m recsys.data.kuasar_ranking --force`；行数/请求数与重建前一致
  （train 14,353,430 / validation 113,902 / test 88,417）。
- 旧权重 `artifacts/kuasar_small/{din,dien,ultra_hstu}.pt` 仍是修复前逻辑，哈希未变；
  验证权重写到 `artifacts/kuasar_small/repair_ranker/kuasar_small/*_history_fixed.pt`。

**2) include_validation_in_test_history 生效**

- `run_channel` 把开关传到 SASRec/BERT4Rec/HSTU、MIND、TIGER；
  `rank_fused` 把开关和请求时间戳传到 DIN/DIEN/Ultra-HSTU。
- Rec Agent 的 `recall_models` / `rank_candidates` / `run_pipeline` 接受该参数；
  CLI 为 `--no-validation-in-test-history`；`run_meta.json` 记录实际值。
- 默认 True；test True=训练+验证，False=训练；validation 始终训练；seen 过滤不变。

**3) Rec Agent 训练权重隔离**

- 每次 `train_model` 写入
  `<experiment_dir>/training/<model>-<时间>-<uuid>/`，LightGBM 与 TIGER 的 RQ-VAE
  关联工件同样隔离；任务状态下只在成功后记录实际 checkpoint。
- 后续工具优先使用任务内权重；失败不更新引用；其他任务仍用业务默认。
- 验证：DIN/LightGCN 训练后默认 checkpoint sha256 不变；新权重可重载，且
  `rank_candidates` 返回的 checkpoint 是任务内路径。

**检查：** [scripts/test_phase1_fixes.py](scripts/test_phase1_fixes.py) 全部通过；
`smoke_phase1.py`、`rec_agent --self-check`、`zoo.py check`、`check_project.py` 通过。
详细记录见 [experiments/2026-11-09-kuasar_small-fix-three/report.md](experiments/2026-11-09-kuasar_small-fix-three/report.md)。

## 7.12 Phase 4 统一 Agent 平台（2026-11-10）

**交付**

- **Research Agent / Master Agent**：新增 `recsys.platform.research_agent`，沿用 LangGraph +
  现有 LLM 客户端；研究任务先查历史，命中且校验通过直接复用（新实验 0），未命中再把真实
  返回和剩余预算交给 LLM/离线替身选择 RAG、推荐/搜索/广告或停止。同一任务可跨业务汇总。
- **三业务薄适配**：`recsys.platform.adapters` 复用 `rec_agent` / `search_agent` / `ads.agent`
  原有入口，保留原 CLI 和单独运行方式；不训练、不写业务默认权重。
- **共享实验平台**：`shared_experiments.py` 新增推荐/搜索/广告统一 JSONL 记录（任务、业务、
  配置、数据/切分、模型/权重来源、状态、结果、预算、工件）；推荐旧历史继续走
  `experiment_history`；支持业务/数据/切分/模型/任务筛选、复用原因、验证集/测试集、
  真实工具/离线替身、广告模拟/真实数据标记；追加前加跨进程文件锁。
- **统一预算**：`budget.py` 在一个任务内统一管理 LLM 请求、工具调用、新实验、历史查询、
  已有结果读取和总时限；子 Agent 通过 `child()` 得到收紧后的上限，不能重置总额度。
- **本地 RAG**：`rag.py` 独立索引 `artifacts/rag/phase4`，收录项目文档、模型说明、
  开源来源、历史实验报告；返回 `path:start-end` 引用；按文件 sha256 增量更新；
  Research Agent 与三个业务 Agent 均可调用 `retrieve_knowledge`。
- **统一 CLI**：`python -m recsys.platform`（任务 / 历史查询 / 注册表 / RAG / 验收 / 自检）。

**验收**

- `python -m recsys.platform --self-check`：通过（预算共享、历史命中、RAG、模型注册、工具注册）。
- `python -m recsys.platform --acceptance`：13/13 通过，见
  `experiments/2026-11-10-kuasar_small-platform-phase4/acceptance-final/report.md`；
  覆盖历史命中 0 新实验、未命中后一次小样本检查、已有结果读取计数、预算耗尽、三业务入口、跨业务同任务、
  另起进程查回、RAG 引用/更新、三业务 RAG 工具、并发写入。
- 真实 LLM 只运行一次：任务 `task-20261110T114653-322c0b`，4 个决策（历史 50/50 miss →
  LLM 再查历史 → 调推荐小样本 → 停止），状态 verified，3 次 LLM 请求 / 1 次新实验；
  原始记录在 `experiments/2026-11-10-kuasar_small-platform-phase4/acceptance-20261110T034618Z/`。

**边界**

- 不训练、不调参、不跑全量、不重建大索引；小样本指标全 0 只作为链路检查，未作效果结论。
- 广告竞价金额为模拟配置，CVR 使用 synthetic fixture；KuaiSAR 无真实出价/预算/转化标签。
- 真实 LLM 只做单次任务，多轮规划对比未验证。
- Search Agent 的 RAG 工具默认关闭，需显式配置 `[knowledge] enabled=true`；默认行为不变。

**文档**：[docs/phase4-platform.md](docs/phase4-platform.md)。

## 7.13 Phase 5 全项目集成与交付（2026-11-10）

**交付**

- **统一启动入口**：`python -m recsys.platform` 新增 `--phase5-demo`（一条命令三业务）、
  `--run-business recommendation|search|ads`（单独调用）、`--phase5-restore`（独立进程工件恢复）、
  `--phase5-missing`（缺工件失败用例）、`--phase5-report --task-id`（按任务 ID 查回）；
  原有模块 CLI（`recsys.pipeline` / `recsys.search` / 各 Agent）保持不变。
- **演示配置**：`configs/platform/phase5_demo.toml` 固定 8 用户双塔 + MIND 召回、LightGBM 排序、
  会话 11208 混合检索 + 精排、DIN 广告预测/排序/竞价；全局预算 14 工具 / 3 实验 / 600 s，
  子 Agent 5 / 4 / 7 工具，未改 Phase 4 默认预算。
- **工件清单与恢复**：`configs/platform/phase5_artifacts.toml` 登记 13 个文件（3 推荐权重、
  2 广告权重、7 搜索索引/样本/清单、1 RAG 索引），含来源、大小、sha256、已有校验出处与可机检
  metadata；`recsys.platform.artifacts` 逐条校验，`recsys.platform.phase5.run_restore` 在新进程里
  加载并各跑一次推理，再用业务自己落盘的身份（run_meta checkpoints / metrics index / weight）核对。
- **只加载不训练**：`pipeline/ranking.py`、`pipeline/chain.py` 缺失权重直接抛 `ArtifactError`；
  `rec_tools.run_pipeline` 新增 `artifact_policy=existing_only|train_if_missing`、
  `ranker_checkpoint`、`expected_ranker_sha256`；推荐适配层默认 `existing_only`，并把
  `run_meta#checkpoints` 的真实权重身份写入共享记录。原有显式训练入口保留。
- **统一日志与摘要**：同一 `task_id` 下写 `tasks/<id>/`（总控 trace/decisions + 三个子 Agent 原始日志）、
  `runs/<id>/demo_summary.json|md`、`run.log`、`artifacts_used.json`、`shared/records.jsonl`
  （3 条业务 + 1 条任务）；缺工件/预算不足/子任务失败时保留已完成结果并返回 partial/failed。

**验收**

- `python -m recsys.platform --phase5-demo`：`task-20261110T131020-3a865c` verified，
  三业务 success（推荐 `f09030a1…`、搜索 `36490db7…`、广告 `29dc0f76…`），预算 12 工具 / 3 实验 / 0 LLM，
  工件核对 3/3 通过；耗时约 11 s。
- `python -m recsys.platform --phase5-restore`：`restore-20261110T051036Z`（新进程 pid 31820）5/5 通过；
  推荐 run_meta、搜索 index.sha256、DIN 权重、RAG 引用均与清单一致。
- `python -m recsys.platform --phase5-missing`：`missing-20261110T051053Z` 5/5 通过；
  `artifacts/` 无新增文件、无新训练产物、缺失目录未被创建。
- `python -m recsys.platform --run-business search`：`task-20261110T131120-4761d9` success。
- `python -m recsys.platform --phase5-report --task-id task-20261110T131020-3a865c`：另起进程查回
  3 条业务记录 + 1 条任务记录及工件路径。

**边界**

- 只加载现有权重与小样本；不训练、不扫全量、不重建大索引、不调参；小样本 Top-10 指标为 0
  属既有数据规模问题，未作效果结论。
- 演示默认离线脚本替身；真实 LLM 多轮规划未在 Phase 5 复验。
- 广告金额/预算为模拟值，DIN 为 `interface_test`；CVR 仅有 synthetic fixture。
- TFRS / TorchEasyRec 原实现 / OneRec 仍受独立环境与 Linux 依赖限制，见 `docs/phase5-integration.md` §8。

**文档**：[docs/phase5-integration.md](docs/phase5-integration.md)。

## 7.14 效果专项：排序器归因、排序上界与多种子（2026-11-10）

**为什么做**：Phase 1–5 的证据都是「链路跑通 + 小样本验收」，效果结论一律写着「留给效果专项」。
这一轮把排序器、融合、重排三段放到同一批 2,000 用户 / 4,275 目标的完整测试集上做受控对照。

**交付**

- 新增 5 个实验脚本 + 1 个对照配置，**未改 `src/`**：
  `experiments/2026-11-10-effect-study/`（排序器消融、召回分数组、上界与融合/重排、多种子、桩模型演示）、
  [configs/pipeline/kuasar_small_no_ranker.toml](configs/pipeline/kuasar_small_no_ranker.toml)。
- 报告：[experiments/2026-11-10-effect-study/report.md](experiments/2026-11-10-effect-study/report.md)
  （含两张图、三张表与边界说明）。

**关键数字**（同批候选、同批用户，只换一个组件）

| 配置 | Recall@20 | Top-20 命中 | 结论 |
| --- | ---: | ---: | --- |
| 现行：精排 + 类目重排 | 0.002175 | 12 | 基准 |
| 去掉精排 | 0.005592 | 23 | +0.003417 [+0.000942, +0.006317] |
| 精排与重排都去掉 | 0.005967 | 25 | +0.003792 [+0.001333, +0.006650] |
| RRF + cap≥10（无精排） | 0.006258 | 27 | 单批候选；跨种子 +2/−1/0，**不复现** |
| 排序上界（oracle） | 0.015325 | 69 | 池内命中 69，无人被 k=20 限制 |

**五个结论**

1. 现行 LightGBM 精排**显著差于不排序**，且与随机重排不可区分（11 vs 12.6 命中）；
   去掉它与类目重排，出厂候选池上最终 Recall@20 提升 **2.74 倍**。
   跨种子复验：同一份出厂权重在三个种子的候选池上 **3/3 都更差**，平均 −10 个 Top-20 命中
   （池子越好破坏越大：−14 / −13 / −3）。
2. **排序空间仍很大**：融合池里有 69 个命中，oracle 能全拿，现在只拿 25。该修的是排序器的
   训练分布（曝光候选 4.3 条/请求 → 召回候选 100 条/用户），不是删掉排序这一步。
3. 重训补不回：训练行扩到 62 倍（622,369 行）、加 5 列请求前交互先验、再加两路召回分数，
   在出厂池上只到「与不排序统计上不可区分」（R5 命中 24 vs 25）。
4. **但跨种子看，R5 是「稳定器」而不是「增益器」**：三个种子分别 24 / 29 / 27 命中，
   而融合是 25 / 25 / 13；平均比融合多约 6 个命中（Recall@20 +0.0010，区间跨零），波动从 6.9 个收窄到 2.5 个。
5. **种子方差是最大的一阶效应**：双塔跨种子 3–23 命中、融合 13–25，而组件效应只有
   ±0.001 量级。把两种不确定性放在同一把尺子上（`variance_decomposition.json`）：
   双塔的跨种子 std 是测试用户抽样 SE 的 **3.11 倍**、融合 1.46 倍，而排序器只有 0.79 倍
   （全链路最稳）。seed 44 的双塔只有 3 个命中，直接把融合拖到 13——降召回方差比继续调融合重要。
   但 seed 44 的双塔 best_epoch=2（seed 42 用全量验证是 19），说明 512 用户选模会把模型锁在早期，
   **种子方差里混着选模噪声**；要拆分需用全量验证重训一次（见报告 §5.1、§8）。

后处理另有两笔：配额融合（30/70、50/50）的排序是「按通道拼接」，top-20 全来自第一路，
@20 退化到单路水平、@100 却最高；`category_cap=5` 相对不限制少 2 个命中，且三个种子方向一致
（逐阶段 −0.0006，区间不含零）。

**边界**

- 2,000 用户子集，未外推全量 KuaiSAR（25,355 用户 / 4 百万物品）。
- 种子数 3（42 复用 20 轮既有权重、43/44 新训；45/46 因单种子约 1.5–1.8 小时、边际信息有限而停止）。3 个种子已能看出
  「种子方差 > 组件效应」，但 `±` 本身还不稳。
- 模型库 11 个召回适配器只有 2 个真实训练权重；桩模型演示（512 用户）显示塞进融合会让
  命中 3 → 1，因此「Agent 选模型」实验的前提是先训实候选池。
- 排序器训练管线未重建，本轮只证明「现在这个不行」，未证明「排序本身不行」。

**下一步**：①关掉精排 + 放开 cap（零训练成本，出厂池 2.74 倍；RRF 跨种子不复现，不建议改）；
②降召回方差（多种子集成 / 全量验证选模）——本轮最大发现；③按召回分布重建排序训练
（唯一能把 25 推向 69 的方向）；④再考虑补齐模型池与扩到全量。

## 八、变更历史

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
| 2026-11-08 | **Phase 11 Agent MVP** | 新增 `evaluate_fusion_quota` 工具与 `quota_agent`：先 50/50 基准，再按上下文选 30/70、20/80；3 组评测 1.37 s；结论 30/70（命中 5 vs 基准 4）；工具/决策/候选明细全落盘；真实 LLM 待配置 | `src/recsys/experiments/fusion_quota.py`、`src/recsys/experiments/quota_agent.py`、`experiments/2026-11-08-kuasar_small-fusion-quota-agent/` |
| 2026-11-08 | **Phase 11 验收（LLM）** | 真实 tool_calls 验收**通过**：deepseek-chat 返回 1 条 `tool_calls`（70/30），程序执行评测并同 ID 回传，模型判其输给基准；2 请求/1 工具/3.05 s；前两次尝试因代码 bug 与并行 tool_calls 被拒（未执行工具） | `src/recsys/experiments/llm_tool_call.py`、`experiments/2026-11-08-kuasar_small-llm-tool-call/` |
| 2026-11-08 | **Phase 11 LangGraph** | 入口改为 LangGraph 状态图；修复第二次请求失败仍写 verified 的误报（改为 partial + 非零退出）；真实运行通过（70/30，2 请求/1 工具/2.81 s）；失败路径回归通过（partial、请求 2、工具 1、无 verified） | `src/recsys/experiments/llm_tool_call.py`、`requirements/optional-agent.txt`、`experiments/2026-11-08-kuasar_small-langgraph-tool-call/` |
| 2026-11-09 | **Phase 13 回放 MVP** | 新增独立离线回放入口：只读 5 组已验收配额汇总 JSON，Rec Agent 与固定策略同起点、各 2 次揭晓；Agent 首步 30/70 → 反馈后 20/80，固定 70/30 → 30/70；双方最佳均 30/70（5 / 0.013672 / 0.026720），结论持平；9/9 合成断言通过；训练/召回/真实评测/LLM 均 0 | `src/recsys/experiments/offline_replay.py`、`configs/experiments/offline_replay.toml`、`experiments/2026-11-09-kuasar_small-offline-replay/` |
| 2026-11-09 | **Phase 13 真实 LLM** | 接入 LangGraph 原生工具调用 + `ReplaySession`：deepseek-chat 单次真实工作流，3 请求 / 2 揭晓 / 5.244 s；LLM 首查 70/30、反馈后查 30/70，最终最佳 30/70（5 / 0.013672 / 0.026720）；与本地规则 Agent、固定策略三方最佳持平；11/11 合成断言；训练/召回/真实评测 0 | `src/recsys/experiments/offline_replay_llm.py`、`experiments/2026-11-09-kuasar_small-offline-replay-llm/` |
| 2026-11-09 | **Phase 13 截断修复** | 统一回复校验：仅 `finish_reason="stop"` + 非空正文 + 无 tool_calls 可作为停止说明/最终回复；`length` 且已有揭晓时 `partial` / `reply_truncated`，保留指标/预算/轨迹，三方比较 `not_judged`；本地自检扩展至 13/13，未调用真实 LLM | `src/recsys/experiments/offline_replay_llm.py`、`experiments/2026-11-09-kuasar_small-offline-replay-llm-regression/` |
| 2026-11-09 | **Phase 13 多场景** | 双场景 A/B：A 一次揭晓，本地规则 30/70 胜、真实 LLM 70/30 输；B 带 70/30 历史、两次揭晓，三方最佳均为 30/70 持平；本地/固定基线先冻结，真实 LLM 两场景各 1 次、合计 5 请求；自检 19/19 | `src/recsys/experiments/offline_replay_scenarios.py`、`configs/experiments/offline_replay_scenarios.toml`、`experiments/2026-11-09-kuasar_small-offline-replay-scenarios/` |
| 2026-11-09 | **Phase 13 历史消融** | 只改变是否可见 70/30：对照组 30/70→80/20（2 揭晓/3 请求），历史组 30/70 后停止（1 揭晓/2 请求）；最佳均 30/70，历史组节省 1 次查询；本地/固定两组选择相同；自检 20/20 | `src/recsys/experiments/offline_replay_scenarios.py`、`configs/experiments/offline_replay_70_30_ablation.toml`、`experiments/2026-11-09-kuasar_small-70-30-history-ablation/` |
| 2026-11-09 | **Phase 13 成对复验** | 新增 `--paired`：3 对按 A→B、B→A、A→B 执行；6 次工作流全部 verified，15/18 请求；3/3 对复现“A 2 揭晓、B 1 揭晓、最佳均 30/70”；可判定 3/3 | `src/recsys/experiments/offline_replay_scenarios.py`、`configs/experiments/offline_replay_70_30_ablation.toml`、`experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/` |
| 2026-11-09 | **Phase 13 配对修正** | 成对模式改为读取 `reference_run` 冻结 manifest/轨迹，不再重跑本地/固定策略；请求计数前移到模型调用边界，轨迹生成/写工件失败仍计数，全轮 18 次上限在客户端调用前检查；两条故障回归均 15/15；首次真实配对工件保留并加 `FIXES.md` | `src/recsys/experiments/offline_replay_scenarios.py`、`src/recsys/experiments/offline_replay_llm.py`、`experiments/2026-11-09-kuasar_small-70-30-history-ablation-paired/FIXES.md` |
| 2026-11-09 | **Phase 13 独立测试集** | test split 128 用户 / 269 目标；双塔/MIND 各召回一次，50/50 与 30/70 共用缓存各评测一次；两组均命中 6、Recall@100 0.023698，Coverage 0.026724 vs 0.026570；总耗时 12.298 s / 300 s | `src/recsys/experiments/recsys_test_fusion.py`、`src/recsys/experiments/recsys_mvp.py`、`configs/experiments/kuasar_small_test_fusion.toml`、`experiments/2026-11-09-kuasar_small-test-fusion/` |
| 2026-11-09 | **Search MVP** | 核验 `keyword`/`caption` 共用 KuaiSAR token ID（论文 + 5.69× 富集 + 点击重合 51.16%）；扫描上限 200k/500k 行；30 会话 / 120 物品，曝光关联 55.814%；BM25 k1=1.5/b=0.75，120 docs / 823 vocab，索引 76,077 B；CLI + 单次冒烟 9/9 通过 | `src/recsys/search/`、`configs/search/kuasar_search.toml`、`docs/search.md`、`experiments/2026-11-09-kuasar-search-bm25/` |
| 2026-11-09 | **Search Agent MVP** | 新增 `search_bm25` 工具 + LangGraph 一次反馈决策；首查原样、第二查非空真子集、最多 2 检索/3 请求；本地 self-check 12/12；唯一一次真实 `deepseek-chat` 工作流返回 5 条后主动停止，2 请求/1 检索，verified 11/11；索引 sha256 前后一致；候选不逐条落盘 | `src/recsys/search/tool.py`、`src/recsys/experiments/search_agent.py`、`configs/experiments/search_agent.toml`、`docs/search-agent.md`、`experiments/2026-11-09-kuasar-search-agent/` |
| 2026-11-09 | **Search Agent 修复** | 模型自由文本/候选不再落盘（程序生成停止摘要）；`--tokens` 支持样本外查询；本地替身回归 45/45：样本会话 5 条、样本外 `[2]` 1 条、无匹配 0 条、会话/token 不一致拒绝；旧证据与索引哈希不变；真实 LLM/重建/真实冒烟 0 | `src/recsys/experiments/search_agent.py`、`experiments/2026-11-09-kuasar-search-agent-regression/`、`docs/search-agent.md` |
| 2026-11-09 | **Phase 1（v2）模型接入** | 新增 LightGCN/DSSM/BERT4Rec/HSTU/RQ-VAE/Semantic ID/TIGER/MMRec-VBPR/DIN/DIEN/DCNv2/Ultra-HSTU 的纯 PyTorch 适配、训练入口与适配器；召回注册表 11 个、排序注册表 5 个；19 个模型 ready，4 个 blocked 有替代方案 | `src/recsys/models/`、`src/recsys/training/`、`src/recsys/adapters/`、`models/zoo.toml` |
| 2026-11-09 | **Phase 1（v2）推荐链路** | 新增 `python -m recsys.pipeline`：可配置多路召回、round_robin/rrf/quota 融合、LightGBM/神经排序、类目/作者重排、逐阶段评估、工件保存/读回；默认配置复现既有双路行为 | `src/recsys/pipeline/`、`configs/pipeline/`、`docs/recsys-pipeline.md` |
| 2026-11-09 | **Phase 1（v2）Rec Agent** | 在 LangGraph + 工具注册表上新增 12 个真实工具（画像/Zoo/训练/召回/排序/重排/评估/链路/配额/历史/工件）；预算、重复调用、失败原因落盘；自检 7/7，真实训 DIN/BPR-MF 并跑链路 verified，真实 LLM 4 请求/4 工具 verified | `src/recsys/experiments/rec_tools.py`、`src/recsys/experiments/rec_agent.py`、`docs/rec-agent.md` |
| 2026-11-09 | **Phase 1（v2）Top-20 诊断** | 固定 128 用户/362 目标：97.51% 目标不在任何通道 Top-100；5 个在并集但未进融合池、4 个进池但未进 Top-20；排序/重排 0 丢失；记录 41.4% 目标训练零点击与融合深位截断 | `scripts/diagnose_top20.py`、`docs/top20-diagnosis.md`、`artifacts/kuasar_small/top20_diagnosis/` |
| 2026-11-09 | **Phase 1 修复三项** | 排序工件加请求时间戳；DIN/DIEN/Ultra-HSTU 按请求严格前置历史训练/评估，旧工件明确报错并给重建命令，DCNv2/LightGBM 兼容；`include_validation_in_test_history` 贯通序列/MIND/TIGER/神经排序/Agent；Agent 训练写入唯一目录，不覆盖业务默认 checkpoint，后续工具优先用任务内权重 | `src/recsys/data/kuasar_ranking.py`、`src/recsys/ranking/samples.py`、`src/recsys/training/train_neural_ranker.py`、`src/recsys/adapters/neural_ranking.py`、`src/recsys/pipeline/`、`src/recsys/experiments/rec_tools.py`、`scripts/test_phase1_fixes.py`、`experiments/2026-11-09-kuasar_small-fix-three/report.md` |
| 2026-11-09 | **Phase 3 广告数据** | 新增 `recsys.ads.data`：请求内 `is_ad=1` 候选 + click 标签；train 5,483/5,186/278、val 51/49/2、test 43/41/2；真实转化字段核对为缺失；独立 synthetic 转化样本 304 行/50 转化 | `src/recsys/ads/data.py`、`src/recsys/ads/simulated.py`、`data/kuasar_small/processed/ads/` |
| 2026-11-09 | **Phase 3 六模型** | DIN/DIEN/DCNv2 复用 Phase 1 实现；DeepFM/MMoE/PLE 复用本地 DeepCTR-Torch；统一 `score/save/load`；6/6 推理与保存加载通过，接口权重隔离在 `interface_weights/`，状态 `interface_test`，未训练 | `src/recsys/ads/models.py`、`src/recsys/ads/registry.py`、`src/recsys/ads/verify.py`、`experiments/2026-11-09-kuasar_small-ads-phase3/model_check.json` |
| 2026-11-09 | **Phase 3 排序/竞价/预算** | 请求内 pCTR 排序；二价 GSP + CPC + 点击扣费 + 总/广告主预算；正常 3 请求扣费 2.1672/剩余 17.8328；0 预算 3/3 未成交；`--budget-check` 6/6 | `src/recsys/ads/ranking.py`、`src/recsys/ads/auction.py`、`src/recsys/ads/pipeline.py`、`configs/ads/ads_sim.toml` |
| 2026-11-09 | **Phase 3 Ads Agent** | 7 个真实工具 + LangGraph；复用 YieldAgent 的工具/业务分离、错误语义、调用/结果记录；本地 14/14 自检；本地 `din` 7 工具 verified；真实 `deepseek-chat` 自主选 `deepfm`，4 工具/3 请求 verified；首次因 LLM 请求上限 4 收尾失败（partial）后调为 6 | `src/recsys/ads/tools.py`、`src/recsys/ads/agent.py`、`configs/ads/ads_agent.toml`、`docs/ads.md`、`docs/ads-agent.md`、`experiments/2026-11-09-kuasar_small-ads-phase3/` |
| 2026-11-09 | **Phase 3 验收** | `smoke_phase3_ads.py` data/models/ranking/auction/agent 全过；zoo 26 模型 ready=22/blocked=4；`check_project.py` 0 错误 | `scripts/smoke_phase3_ads.py`、`models/zoo.toml`、`experiments/2026-11-09-kuasar_small-ads-phase3/report.md` |
| 2026-11-10 | **Phase 4 统一平台** | 新增 Research/Master Agent（LangGraph + 真实 LLM 单次 4 决策）、三业务薄适配、共享实验 JSONL（跨进程锁/复用校验/模型注册）、统一任务预算、本地 RAG（独立索引/引用/增量更新）与统一 CLI；13/13 小样本验收通过 | `src/recsys/platform/`、`configs/platform/`、`docs/phase4-platform.md`、`experiments/2026-11-10-kuasar_small-platform-phase4/` |
| 2026-11-10 | **Phase 5 集成演示** | 统一入口新增三业务演示/单业务/恢复/缺工件/记录查询；13 个工件清单 + 逐文件 sha256/metadata 校验；LightGBM 排序改为只加载（缺失即失败，不自动训练）；任务 `task-20261110T131020-3a865c` verified（3 条业务记录、12 次工具含 3 次分发），恢复与缺工件用例各 5/5 | `src/recsys/platform/phase5.py`、`src/recsys/platform/artifacts.py`、`src/recsys/utils/artifact_guard.py`、`src/recsys/pipeline/ranking.py`、`src/recsys/experiments/rec_tools.py`、`configs/platform/phase5_*.toml`、`docs/phase5-integration.md`、`experiments/2026-11-10-kuasar_small-platform-phase5/` |
| 2026-11-10 | **效果专项** | 排序器归因：精排显著差于不排序（去掉精排+重排 Recall@20 0.002175 → 0.005967，2.74×）；重训 62 倍数据+交互先验+召回分数只追平；排序上界 69 命中 vs 现在 25；配额融合@20 退化、cap=5 掉 2 个命中；多种子（42/43）显示排序器符号翻转 | experiments/2026-11-10-effect-study/report.md、configs/pipeline/kuasar_small_no_ranker.toml |
| 2026-11-13 | **kuairec_small 正式接入** | KuaiRec 官方 `small_matrix`（1,411 用户 / 3,327 目录物品 / 4,676,570 行，密度 95.7%）注册为 `SUPPORTED_DATASETS` 一等数据集：同一读取器 + `matrix_filename` 参数、CRC32 `77673810`、建库按(行,用户,物品)三重核对、181,992 行缺 timestamp 计数后剔除（43,188 条正反馈一并剔除）；`configs/datasets/kuairec_small.toml` + `check_project` 10 条交叉校验；`finalize_experiment` 结果改为按数据集落盘。四模型走完「验证选型 → 测试一次」：测试 Recall@20 SASRec 0.924167 / MF 0.389794 / Two-Tower 0.332388 / Popularity 0.328845，SASRec 受控（只用训练历史）复算 0.924167 排除泄漏；全程 CPU 约 16.6 分钟；候选集中位 149，随机基线 Recall@20 0.156636（精确期望）→ Popularity 2.10× / MF 2.49× / SASRec 5.90×，命中者平均名次 5.38 vs 8.73–9.40，明确标注不可与 KuaiSAR 横比 | `src/recsys/data/kuairec.py`、`src/recsys/data/loader.py`、`src/recsys/config.py`、`configs/datasets/kuairec_small.toml`、`scripts/check_project.py`、`src/recsys/experiments/finalize_experiment.py`、`docs/kuairec-small.md`、`experiments/2026-11-13-kuairec-small-formal/` |

> 详细到「改了哪些文件、为什么这么改、测试输出」的记录，见本文件对应小节的证据链接；
> 结论性内容回写 [PROJECT_MASTER.md](PROJECT_MASTER.md)，本文件不重复架构说明。
