# 配置

只放**结构化、可被脚本校验**的配置；训练脚本的真正参数仍以命令行 `--help` 为准。

```
configs/
├── datasets/          数据集来源、目录布局、正反馈阈值、切分协议、评测口径
│   ├── kuasar.toml    主数据线（Phase 1.1/1.2 完成，1.3 待接 prepare）
│   ├── kuairec.toml   流程模板 / 辅助线
│   └── kuairec_small.toml  KuaiRec small_matrix 正式小样本线（稠密 1,411 × 3,327）
├── experiments/       离线回放、融合验证、Search Agent 等实验配置
├── models/            已交付实验的默认超参（与训练脚本 dataclass 默认值一致）
│   ├── two_tower.toml
│   ├── sasrec.toml
│   └── matrix_factorization.toml
├── ads/               广告模型规模/竞价/预算（ads_sim.toml）与 Ads Agent 预算（ads_agent.toml）
├── pipeline/          推荐链路配置（kuasar_small_default.toml / multi_recall）
├── platform/          统一平台配置
│   ├── research_agent.toml   Phase 4 总控/子 Agent 预算（默认值，Phase 5 不改动）
│   ├── rag.toml              本地知识库来源与索引参数
│   ├── phase5_demo.toml      Phase 5 三业务演示：小样本、模型、查询、广告请求、预算
│   └── phase5_artifacts.toml Phase 5 工件清单：13 个文件的路径/sha256/来源/校验出处
└── search/            Search MVP 采样上限与 BM25 参数
    ├── kuasar_search.toml
    └── kuasar_search_phase2.toml（向量/混合/精排）
```

约定：

- 统一用 **TOML**（Python 3.11 起标准库 `tomllib` 可直接读，不再引入 PyYAML）；
- `configs/datasets/*` 里的阈值/切分/去重字段必须与 `src/recsys/config.py` 一致；
- `configs/models/*` 的 `[defaults]` 是训练 dataclass 默认值的镜像，允许缺省但**不允许冲突**；
- 两者都由 `python scripts/check_project.py` 校验，改代码不改配置会让校验失败。

新增数据集（例如 KuaiSAR）：复制 `datasets/kuairec.toml`，改 `source_url`、`layout`、
`positive_field`、`positive_threshold`，然后在 `recsys/data/loader.py` 注册读取器。
只改一边会失败：`check_dataset_configs` 校验取值一致，`check_dataset_registry` 校验
`[dataset].name` 已出现在 `SUPPORTED_DATASETS` 里。
