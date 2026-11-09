# recsys-agent

短视频场景的智能搜索与推荐平台（工程底座）。把 13 个第三方推荐仓库和一套自研 PyTorch 基线，
整理成**同一套数据契约、同一套评估口径、同一个适配接口**的可复现工程。

项目规划、阶段划分与协作约定的基准文档是 [PROJECT_MASTER.md](PROJECT_MASTER.md)；
本文件只讲「现在仓库里有什么、怎么跑」。

- **数据契约**：`PreparedData`（连续整数 user/item 索引 + 时间戳 + 曝光 mask）
- **评估口径**：Recall / Precision / NDCG / HitRate / Coverage / AUC / LogLoss 只在 `recsys.evaluation.metrics` 实现
- **适配接口**：`recsys.adapters`，召回 `recommend/retrieve/encode_*/save`，排序 `fit/predict/save/load`
- **工件溯源**：checkpoint 带数据指纹、ID 映射摘要、代码指纹、依赖版本
- **第三方代码只读**：`third_party/` 保留原始源码与许可证，要改就放 `patches/`

## 现状（2026-11-09）

| 项目 | 状态 |
| --- | --- |
| 开发阶段 | **Phase 11/12 MVP 已验收**；Phase 13 离线回放/Agent 验证完成；**Search 1 + Search Agent MVP 第一轮完成**（见 [docs/search.md](docs/search.md)、[docs/search-agent.md](docs/search-agent.md)）；完整 Phase 13 正式实验待办 |
| 第三方仓库 | 13 个，来源见 [third_party/SOURCES.toml](third_party/SOURCES.toml) |
| 纳入索引的模型 | 24 个：12 召回 / 9 排序 / 3 基础设施 |
| 已接入统一接口 | 4 个召回模型（popularity、MF、Two-Tower、SASRec）+ FAISS 索引封装 |
| 数据主线 | **KuaiSAR 2.0**：下载 → 全量画像 → processed npy 存储 → `PreparedData` 接入（Phase 1.1–1.4 完成）；KuaiRec 为流程模板/辅助线 |
| 搜索 MVP | 2026-11-09：核验 `keyword`/`caption` 共用 KuaiSAR token ID；30 会话 / 120 物品 BM25 索引 + 查询 CLI；只跑一次冒烟（9/9 通过），见 [docs/search.md](docs/search.md) |
| Search Agent | 2026-11-09：`search_bm25` 原生工具调用 + 一次反馈决策；真实工作流 1 次，2 请求 / 1 检索后主动停止，verified，见 [docs/search-agent.md](docs/search-agent.md) |
| Search Agent 修复 | 2026-11-09：候选/模型自由文本不落盘；`--tokens` 支持样本外查询；本地替身回归 45/45，旧证据哈希不变 |
| 累计清理 | 400.7 MB → 108.8 MB（第三方部分），详见 [docs/cleanup-report.md](docs/cleanup-report.md) |

模型清单见 [docs/model-zoo.md](docs/model-zoo.md)（由 [models/zoo.toml](models/zoo.toml) 生成）。

## 目录

```
recsys-agent/
├── PROJECT_MASTER.md          项目总档案：定位、架构、13 个 Phase、记录规范（**大纲，不随进度改动**）
├── PROJECT_PROGRESS.md        进度台账：已完成 / 待决策 / 下一步（**每轮更新这一份**）
├── README.md                  本文件：现状、快速开始、约定
├── pyproject.toml             包定义（src 布局，Python ≥3.11）
├── configs/                   数据集、模型与搜索配置（TOML，可被脚本校验）
│   ├── datasets/              kuasar.toml / kuairec.toml
│   ├── experiments/           实验配置
│   ├── models/                已交付实验的默认超参
│   └── search/                kuasar_search.toml（Search MVP 采样/BM25）
├── data/                      运行时数据（不入库）
├── artifacts/                 训练产物：checkpoint、指标（不入库）
├── docs/                      架构、数据契约、Model Zoo、依赖、KuaiSAR 差距、清理报告、模型档案
├── experiments/               实验记录（每次一个目录，含 configuration/metrics/report）
├── models/                    Model Zoo 索引：zoo.toml + 三个分类页
├── patches/                   上游补丁登记（third_party 只读）
├── requirements/              依赖分层（主环境 / 可选 / 遗留独立环境）
├── scripts/                   zoo 渲染校验、项目校验、适配层冒烟测试
├── src/recsys/                自研代码（含 search/：BM25 索引、采样、查询 CLI）
└── third_party/               13 个第三方仓库（只读）+ 来源、许可证、审计
```

`src/recsys/` 内部：`adapters/`（统一接口）、`data/`（读取与 `PreparedData`）、`models/`（自研模型）、
`retrieval/`（FAISS）、`search/`（搜索采样 + BM25 + CLI）、`training/`（训练入口）、
`evaluation/`（指标与评测）、`experiments/`（对比与定稿）、`utils/`（种子、工件溯源）。

## 快速开始

```powershell
# 1. 环境（Python 3.11+）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements/optional-faiss.txt

# 2. 数据主线：下载 KuaiSAR → 构建 processed 存储 → 物品特征表
python -m recsys.data.kuasar          # 805.9 MB，带 md5 校验，解压 3.5 GB
python -m recsys.data.kuasar_store    # 约 36 秒，产出 processed（253 MB，24,128 可评测用户）
python -m recsys.data.kuasar_features # 约 46 秒，12 列物品特征（139 MB，含 is_ad）
python -m recsys.data.kuasar_ranking  # 约 54 秒，排序候选样本（约 250 MB，14.35M train 候选）
#   流程模板（辅助线）：python -m recsys.data.prepare --dataset kuairec

# 3. 训练（默认数据集已是 kuasar；4M 物品词表在 CPU/16GB 机器上很吃资源，
#    先用小维度 + 限制 batch 数验证通路，正式训练留到 Phase 2）
python -m recsys.training.train_mf --dataset kuasar --embedding-dim 16 --epochs 1 --max-batches-per-epoch 200
python -m recsys.training.trainer  --dataset kuasar --embedding-dim 32 --output-dim 16 --epochs 1 --max-batches-per-epoch 200
#   SASRec 需要按用户构造序列，KuaiSAR 上建议先加 --max-batches-per-epoch 试跑

# 4. 统一接口评测（自动读回 checkpoint 的数据集、阈值与溯源信息）
python -m recsys.adapters list
python -m recsys.adapters evaluate --model matrix_factorization `
    --checkpoint artifacts/kuasar/matrix_factorization.pt --split test

# 5. 搜索 MVP：小样本 BM25（只读原始 CSV 前缀，不扫全量）
python -m recsys.search.prepare        # 30 会话 / ≤1,000 物品，产物在 artifacts/search/
python -m recsys.search --tokens "3538762,3181091,4061567,3525929" --top-k 5
python -m recsys.search.smoke --top-k 5   # 有匹配 / 空查询 / 无匹配一次检查

# 6. Search Agent：复用上面的 BM25 索引（真实运行会调用 LLM，最多 3 次请求）
python -m recsys.experiments.search_agent --self-check       # 本地 12 项，不联网
python -m recsys.experiments.search_agent `
    --regression --tokens 2                                  # 本地替身回归，不联网
python -m recsys.experiments.search_agent `
    --tokens "3538762,3181091,4061567,3525929"
```

不装包时用 `$env:PYTHONPATH = "src"` 也能直接运行。校验与冒烟测试（不需要真实数据）：

```powershell
python scripts/zoo.py check           # 索引 ↔ 源码路径 ↔ 许可证
python scripts/check_project.py       # 配置一致性、文档链接、生成物漂移、残留缓存、第三方来源
python scripts/smoke_adapters.py      # 适配层 + 指标 + 工件溯源 + 排序适配器（合成数据）

# 数据缓存校验（1–2 秒；不一致会提示 --force 重建）
python -m recsys.data.kuasar_store --check
python -m recsys.data.kuasar_features --check
python -m recsys.data.kuasar_ranking --check
```

## 三条约定

1. **谁写的代码放哪**：`src/` 自研，`third_party/` 原样导入。要改上游就放 `patches/` 并登记，
   否则无法和上游 diff，也无法判断哪些行为是我们引入的。
2. **接一个新模型 = 三步**：写适配器（召回 `RecommenderAdapter` / 排序 `RankingAdapter`）→
   在 `models/zoo.toml` 加一条 → `python scripts/zoo.py render`。文档与档案由索引生成。
3. **数字只在一个口径下比较**：同一 `PreparedData`、同一时间切分、同一 seen-mask、同一指标实现；
   跨仓库的论文指标不能直接贴进实验表。

## 文档

| 文档 | 内容 |
| --- | --- |
| [PROJECT_MASTER.md](PROJECT_MASTER.md) | 项目总档案与 13 个 Phase（大纲） |
| [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) | 进度台账：已完成 / 待决策 / 下一步 / 变更历史 |
| [DEVELOPMENT_STORY.md](DEVELOPMENT_STORY.md) | **开发故事线**：遇到的问题、根因、取舍与数字（面试用） |
| [docs/architecture.md](docs/architecture.md) | 分层、数据流、适配层、工件溯源、扩展点 |
| [docs/data-contract.md](docs/data-contract.md) | `PreparedData` 契约、切分规则、模型输入映射 |
| [docs/model-zoo.md](docs/model-zoo.md) | 24 个模型的完整索引（生成物） |
| [docs/models/](docs/models/README.md) | 已接入模型的技术档案（生成物） |
| [docs/dependencies.md](docs/dependencies.md) | 依赖分层与版本冲突处理 |
| [docs/kuairec.md](docs/kuairec.md) | KuaiRec 流程模板：口径、命令、坑 |
| [docs/kuasar-roadmap.md](docs/kuasar-roadmap.md) | Phase 1：接入 KuaiSAR 的工作清单 |
| [docs/kuasar.md](docs/kuasar.md) | KuaiSAR 文件、字段与行为统计（数据说明） |
| [docs/search.md](docs/search.md) | Search 1：词表核验、30 会话小样本、BM25 接口与 CLI |
| [docs/search-agent.md](docs/search-agent.md) | Search 2：`search_bm25` 工具、查询调整规则、预算与真实工作流 |
| [docs/cleanup-report.md](docs/cleanup-report.md) | 仓库整理与清理账目 |
| [third_party/README.md](third_party/README.md) | 13 个仓库的来源、commit、许可、可用性 |

## 路线图

1. **Phase 1**：KuaiSAR loader + 字段核对 + 数据画像 + 行为定义 + 划分（见 [docs/kuasar-roadmap.md](docs/kuasar-roadmap.md)）；
2. **Phase 2**：Two-Tower + FAISS 在 KuaiSAR 上跑通，补齐 Coverage/效率指标；
3. **Phase 3–4**：MIND（需 Linux 环境）、LightGCN（`PreparedData → atomic` 桥接）；
4. **Phase 5**：多路召回融合（适配层已留 `retrieve` 通道语义）；
5. **Phase 6–8**：SASRec 迁移、DIN/DCNv2（FuxiCTR）、LightGBM 对比；
6. **Phase 9+**：完整链路、重排、TIGER（genrec）、Research Agent 与 RAG。
