# 项目进度与工作日志

> **用途**：记录「已经做完了什么 / 现在卡在哪 / 下一步做什么」。每轮开发后更新本文件。
> **与 [PROJECT_MASTER.md](PROJECT_MASTER.md) 的关系**：总档案是**大纲与基准**（定位、架构、13 个 Phase、
> 记录规范），不随开发进度改动；本文件是**执行台账**，只写事实、证据和下一步。
> **维护规则**：
> 1. 只写真实发生的事；没验证的不写成完成，验证集结果不写成测试集结果；
> 2. 每条「已完成」都要能点开证据（代码路径 / 文档 / 可复现命令）；
> 3. 待办用 `- [ ]`，完成的用 `- [x]` 就地勾选，不删除历史；
> 4. 每轮结束更新「当前状态快照」与「变更历史」两节。

**最后更新**：2026-11-05

---

## 一、当前状态快照

| 项目 | 状态 |
| --- | --- |
| 所处阶段 | **Phase 0 完成**（仓库审计 + Model Zoo + 工程底座），准备进入 **Phase 1**（KuaiSAR 数据层） |
| 数据主线 | KuaiSAR（主，待接入）；KuaiRec / MovieLens 为流程模板与辅助线 |
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
| 数据准备（KuaiRec / MovieLens） | `python -m recsys.data.prepare --dataset movielens` | 可用（KuaiSAR 待 Phase 1） |
| 训练召回基线 | `python -m recsys.training.train_mf\|trainer --dataset movielens` | 可用 |
| 训练序列模型 | `python -m recsys.training.train_sasrec --dataset movielens` | 可用 |
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
- [x] 端到端验证：MovieLens 上 MF 训练 → 统一接口评测（指标仅证明管线通，不作为实验结论）
- 证据：[src/recsys/utils/provenance.py](src/recsys/utils/provenance.py)、[src/recsys/evaluation/metrics.py](src/recsys/evaluation/metrics.py)

### 2.5 Phase 0-E：配置、校验与文档体系（2026-11-04 ~ 11-05）

- [x] `configs/datasets/`（kuairec、movielens）+ `configs/models/`（two_tower、sasrec、matrix_factorization），TOML 格式
- [x] `scripts/check_project.py`：配置与代码常量一致性、Markdown 链接、生成物是否过期、残留缓存、third_party 来源与许可证
- [x] `scripts/smoke_adapters.py`：4 个适配器 + 向量出口 + `retrieve` 一致性 + 指标手算 + provenance 往返 + 排序适配器
- [x] 文档：架构、数据契约、依赖、KuaiRec 流程模板、KuaiSAR 差距、清理报告
- [x] 依赖分层：`requirements/`（主环境 / 可选 / 遗留 TF1 / MMRec）
- 证据：运行 `python scripts/check_project.py` 应输出「0 个错误」

---

## 三、下一步（按 PROJECT_MASTER 的 Phase 顺序）

### 3.1 Phase 1：KuaiSAR 数据层（当前优先级）

详见 [docs/kuasar-roadmap.md](docs/kuasar-roadmap.md)。

- [ ] **1.1 下载与校验**：`src/recsys/data/kuasar.py` + `configs/datasets/kuasar.toml`
      （Zenodo 8181109，`KuaiSAR_v2.zip` 845 MB，md5 `ed630f47fb013c5b31bb020c483ef262`）
      验收：`python -m recsys.data.prepare --dataset kuasar` 从零跑通，重复运行走本地
- [ ] **1.2 字段核对与数据画像**：先 `head -1` 看列名，产出 `docs/kuasar.md`
      验收：`PreparedData` 的规模与画像脚本一致
- [ ] **1.3 行为与正样本定义**：MVP 先用「点击/播放」单行为；阈值与切分写进 config 与 `recsys.config`
      验收：同数据同阈值 → 同词表规模、同测试集大小
- [ ] **1.4 样本构建**：召回交互对 / 序列样本 / 排序候选+特征（第三类尚无实现）
- [ ] **1.5 特征通道**：`FeatureStore` 或 `PreparedData.item_features`，特征版本写入 `provenance.extra`
- [ ] **1.6 数据版本与缓存**：`processed/` 可读 + `protocol.json` 校验（当前只写不读）

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

### 4.2 版本控制

- 事实：仓库目前**没有 `git init`**，「版本」靠 `SOURCES.toml` + provenance 的代码指纹表达。
- 建议：Phase 1 开始前初始化 git 并做首次提交，否则后续改动无法回溯（`.gitignore` 已就绪）。

### 4.3 KuaiSAR 许可

- 事实：CC BY-NC-SA 4.0（非商业 + 相同方式共享）。
- 需要确认：本项目用途是否落在非商业范围；衍生数据/模型是否对外发布。

### 4.4 TorchRec 何时引入

- 当前后置：Two-Tower 已有自研实现，TorchRec 主要解决大规模分片与分布式训练。
- 触发条件：单机训练成为瓶颈（Phase 2 之后评估）。

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

# 当前可跑的数据线（KuaiRec / MovieLens 作模板）
python -m recsys.data.prepare --dataset movielens
python -m recsys.training.train_mf --dataset movielens --epochs 2
python -m recsys.adapters evaluate --model matrix_factorization `
    --checkpoint artifacts/movielens/matrix_factorization.pt --split test
```

---

## 七、变更历史

| 日期 | 阶段 | 主要变化 | 关键文件 |
| --- | --- | --- | --- |
| 2026-11-04 | Phase 0 | 11 个仓库整理：扁平化、删 `.git`/示例语料/缓存；建 Model Zoo 索引与文档骨架 | `third_party/README.md`、`models/zoo.toml`、`docs/cleanup-report.md` |
| 2026-11-05 | Phase 0 | 补 FuxiCTR/genrec 与 `SOURCES.toml`；接口扩到 `retrieve/encode_*` 与 `RankingAdapter`；checkpoint 加 `provenance`；指标补 Precision/Coverage/AUC/LogLoss；建档案与校验脚本；数据主线改为 KuaiSAR | `src/recsys/adapters/`、`src/recsys/utils/provenance.py`、`src/recsys/evaluation/metrics.py`、`PROJECT_PROGRESS.md` |

> 详细到「改了哪些文件、为什么这么改、测试输出」的记录，见本文件对应小节的证据链接；
> 结论性内容回写 [PROJECT_MASTER.md](PROJECT_MASTER.md)，本文件不重复架构说明。
