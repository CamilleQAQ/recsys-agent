# 配置

只放**结构化、可被脚本校验**的配置；训练脚本的真正参数仍以命令行 `--help` 为准。

```
configs/
├── datasets/          数据集来源、目录布局、正反馈阈值、切分协议、评测口径
│   ├── kuasar.toml    主数据线（Phase 1.1/1.2 完成，1.3 待接 prepare）
│   └── kuairec.toml   流程模板 / 辅助线
├── experiments/       离线回放、融合验证等实验配置
├── models/            已交付实验的默认超参（与训练脚本 dataclass 默认值一致）
│   ├── two_tower.toml
│   ├── sasrec.toml
│   └── matrix_factorization.toml
└── search/            Search MVP 采样上限与 BM25 参数
    └── kuasar_search.toml
```

约定：

- 统一用 **TOML**（Python 3.11 起标准库 `tomllib` 可直接读，不再引入 PyYAML）；
- `configs/datasets/*` 里的阈值/切分/去重字段必须与 `src/recsys/config.py` 一致；
- `configs/models/*` 的 `[defaults]` 是训练 dataclass 默认值的镜像，允许缺省但**不允许冲突**；
- 两者都由 `python scripts/check_project.py` 校验，改代码不改配置会让校验失败。

新增数据集（例如 KuaiSAR）：复制 `datasets/kuairec.toml`，改 `source_url`、`layout`、
`positive_field`、`positive_threshold`，然后在 `recsys/data/loader.py` 注册读取器。
