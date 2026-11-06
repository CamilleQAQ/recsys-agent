# Phase 1：接入 KuaiSAR 的工作清单

对应 [PROJECT_MASTER.md](../PROJECT_MASTER.md) §8 Phase 1。

**结论**：代码骨架已就位（`PreparedData → 训练 → 统一评测`），缺的是「数据层 + 标签口径 + 特征通道」三块。
KuaiRec 那条线（[kuairec.md](kuairec.md)）可直接当**读取器模板**，但它的单行为（`watch_ratio`）口径不能照搬到 KuaiSAR。

## 0. 已核实的公开信息

| 项 | 值 |
| --- | --- |
| 数据集 | KuaiSAR: A Unified Search And Recommendation Dataset（Kuaishou + 人大，CIKM 2023） |
| 下载 | Zenodo record [8181109](https://zenodo.org/records/8181109) |
| 文件 | `KuaiSAR_v2.zip` 845.1 MB（完整版，md5 `ed630f47fb013c5b31bb020c483ef262`）；`KuaiSAR.zip` 488.1 MB（small 版，md5 `daea8cbf605db6bd5841740f0e4a12d9`） |
| 许可 | CC BY-NC-SA 4.0（非商业 + 相同方式共享），Zenodo 标注 "Other (Non-Commercial)" |
| 内容 | 真实用户在**搜索**与**推荐**两个入口的行为、二者之间的跳转、以及多种反馈标签 |

> 论文：[arXiv 2306.07705](https://arxiv.org/abs/2306.07705)。上面的大小/md5 来自 Zenodo 页面；
> **列名、解压目录结构、是否提供曝光未点击记录**必须在拿到 zip 后核对一次，不要按猜测写解析器。

## 1. Phase 1 待办（按顺序）

### 1.1 数据获取与校验（0.5 天）

- `src/recsys/data/kuasar.py`：`KUASAR_ARCHIVE_URL / KUASAR_ARCHIVE_MD5`、`download_and_extract_kuasar()`、
  `resolve_kuasar_file()`，结构照抄 `recsys.data.kuairec`；
- `configs/datasets/kuasar.toml`：来源、md5、布局、许可、`status = "available"`；
- 数据不入库：`.gitignore` 已忽略 `data/`，且 CC BY-NC-SA 不适合随代码分发。

**验收**：`python -m recsys.data.prepare --dataset kuasar` 从零下载并通过 md5；重复运行走本地文件。

### 1.2 字段核对与数据画像（1 天）

先只 `head -1` 看列名，不要一次性加载全量：

| 需要 | KuaiSAR 里对应的未知项 | 处理 |
| --- | --- | --- |
| 用户、物品 ID | 交互表的 user / video 列 | 连续编码 + 保存可逆映射（`IdMappings`） |
| 时间 | 时间列（秒还是毫秒？） | 统一 float |
| **来源渠道**（搜索/推荐） | 是否有独立列 | 有则旁路保存，供后续搜索任务使用 |
| 反馈类型（点击/长播/点赞/关注） | 行为类型列 | 决定「哪些算正反馈」（见 1.3） |
| 曝光未点击 | **未知，可能没有** | 没有就让 `seen_mask` 降级并在文档标注口径变化 |

产出 `docs/kuasar.md`（数据说明 + 画像）：用户数、物品数、交互数、活跃度分布、物品流行度、
序列长度分布、时间覆盖、行为类型占比。可直接复用 `recsys.experiments.analyze_kuairec` 的统计骨架。

**验收**：`PreparedData.num_users / num_items` 与画像脚本统计一致；同一 `(user, video)` 的重复策略写进 config。

### 1.3 行为与正样本定义（1 天，必须先定死）

- KuaiSAR 是多行为数据集，不能像 KuaiRec 那样用单一 `watch_ratio >= 1`；
  **MVP 建议先用「点击/播放」单行为**，后续再扩多任务；
- 若交互来自搜索，user-item 对可能伴随 query；「最后一次交互」按用户切还是按请求切要显式决定
  （当前 `temporal_train_validation_test_split` 只按用户切）；
- 阈值与行为定义写进 `recsys.config` 常量 + `configs/datasets/kuasar.toml`，由 `scripts/check_project.py` 校验一致。

**验收**：能复现「同一份数据、同一阈值 → 相同词表规模、相同测试集大小」，并写进 `docs/kuasar.md`。

### 1.4 样本构建（1~2 天）

master §8 Phase 1 要求三类样本，目前只有前两类有实现路径：

| 样本 | 现状 | 要做 |
| --- | --- | --- |
| 召回交互对 + 负采样 | `InteractionDataset` + `samplers` 已有 | 换数据源即可 |
| 序列样本 | `SASRecTrainDataset` 已有 | 换数据源 + 截断规则 |
| 排序候选 + 特征 | **无** | 候选集构造（召回结果或曝光候选）+ 特征表（见 1.5） |

**验收**：三类样本都能从同一份 `PreparedData` 稳定生成，重复运行结果一致。

### 1.5 特征通道（3~5 天，排序模型的前置）

`PreparedData` 目前只有 ID，DIN/DCNv2（FuxiCTR）、MMRec 都要特征。

- 加载物品/用户特征表并与 ID 映射对齐；
- 新增 `FeatureStore`（稀疏/稠密列 + 名称 + 缺失值策略），或给 `PreparedData` 加可选 `item_features`；
- 特征版本写进 checkpoint（`provenance.extra` 已预留位置）。

**验收**：DIN 或 DCNv2 在 KuaiSAR 上跑通一次端到端（指标低没关系），特征名与列序写进工件。

### 1.6 数据版本与可复现（0.5 天）

- `processed/` 缓存要能**读**（当前只写不读，每次评测重新解析）；
- 缓存带 `protocol.json` 校验：切分协议/阈值/行数对不上就拒绝复用；
- 数据指纹已由 `recsys.utils.provenance.data_fingerprint` 产出，`prepare` 时应把同一指纹写进 `protocol.json`。

**验收**：`prepare` → 训练 → 评测 全链路可复现；改一个过滤条件，缓存会自动失效。

## 2. 后续 Phase 依赖项（Phase 1 之后才会碰到）

| 能力 | 现状 | 缺什么 |
| --- | --- | --- |
| 会话/请求级评测 | 无 request 概念 | 搜索任务需要 request 级样本与分组指标 |
| 多任务标签 | MMoE/PLE 代码已归档 | 多列 label 映射（点击/长播/点赞/关注） |
| 规模 | `list[Interaction]` 全量 | 数组化 / 分块 / 缓存 |
| 符号召回（TIGER） | genrec 已入库（`tiger` 条目） | 物品 embedding → RQ-VAE 语义 ID 流程 |

## 3. 最小可用路径

1. 下载 v2 并用 `head -1` 确认列名（不要一次加载全量）；
2. 用 1% 采样跑通 `PreparedData` → `train_mf` → `adapters evaluate`；
3. 再用全量跑一次 Two-Tower，确认内存与耗时；
4. 最后才做特征通道与多任务。

## 4. 与仓库现状的对应

| KuaiSAR 需要 | 仓库里已有什么 | 复用方式 |
| --- | --- | --- |
| 下载/校验/解压 | `recsys.data.kuairec` | 复制结构，改 URL/md5/布局 |
| ID 映射 | `IdMappings` + `save_processed_data` | 直接复用 |
| 时间切分 | `temporal_train_validation_test_split` | 直接复用（若改按请求切需新增函数） |
| 曝光 mask | `PreparedData.*_seen_mask` | 有曝光就用，没有则降级 |
| 指标 | `Recall/Precision/NDCG/HitRate/Coverage` | 直接复用 |
| 排序指标 | `AUC/LogLoss`（`RankingAdapter`） | 特征就位后接 FuxiCTR/LightGBM |
| 排序模型 | FuxiCTR（DIN/DCNv2）、LightGBM | `LightGBMRankerAdapter` 已就绪 |

## 5. 许可提醒

CC BY-NC-SA 4.0：**非商业** + **相同方式共享**。数据不入库、不随代码分发；
基于它的衍生数据/模型如果对外发布，需要按相同许可处理，先确认再动手。
