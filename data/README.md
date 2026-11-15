# data：运行时数据目录（不入库）

`.gitignore` 已忽略本目录下除 README 外的所有内容。原始数据体量大，且 KuaiSAR 等数据集
的许可（CC BY-NC-SA 4.0）不允许随代码分发。

## 布局

每个数据集住进自己的子目录，目录名由 `recsys.data.loader.resolve_dataset_data_dir` 固定：

```
data/
├── kuasar/                     KuaiSAR 2.0（主数据线）
│   ├── raw/KuaiSAR_v2/         rec_inter.csv / src_inter.csv / item_features.csv / user_features.csv
│   └── processed/              interactions/test/validation npy + features/ + ranking/
├── kuasar_small/               2,000 用户子集（协议与全量一致）
├── kuairec/                    KuaiRec 2.0 big_matrix（流程模板）
│   ├── raw/KuaiRec.zip
│   ├── raw/big_matrix.csv
│   └── processed/
└── kuairec_small/              KuaiRec 2.0 small_matrix（正式小样本线）
    ├── raw/small_matrix.csv    387 MB，CRC32 77673810
    ├── raw/download.json       下载来源 / 成员路径 / 校验值 / 耗时
    └── processed/              interactions.csv / id_mappings.json / protocol.json / label_audit.json
```

`processed/` 对 KuaiSAR 是**读取入口**（npy 存储，带 sha256 校验）；对 KuaiRec 两条线目前只写不读
（供人工核查映射与切分），详见 [../docs/architecture.md](../docs/architecture.md) 的「已知限制」。

## 下载 / 构建

```powershell
# KuaiSAR（主数据线）：下载 → processed → 特征 → 排序候选
python -m recsys.data.kuasar
python -m recsys.data.kuasar_store
python -m recsys.data.kuasar_features
python -m recsys.data.kuasar_ranking

# KuaiRec big_matrix（流程模板，约 1 GB，含 md5 校验）
python -m recsys.data.prepare --dataset kuairec

# KuaiRec small_matrix（正式小样本线，387 MB，含 CRC32 + 规模校验，约 40 秒）
python -m recsys.data.prepare --dataset kuairec_small
```

目录约定见 [../configs/datasets/](../configs/datasets/)、
[../docs/kuairec.md](../docs/kuairec.md)、[../docs/kuairec-small.md](../docs/kuairec-small.md)、
[../docs/kuasar.md](../docs/kuasar.md)。

## 注意

- 不要把 `data/` 提交进 git：体积、许可、以及「同一份数据不同版本的静默差异」都是风险；
- 换机器时重新下载，不要从别处拷贝半截文件——`prepare` 会做完整性校验（KuaiRec 两条线分别是
  整包 md5 与矩阵 CRC32，`kuairec_small` 另外核对行数/用户/物品三项规模）；
- 需要长期保留的小样例（用于测试）放到 `tests/fixtures/`，不要放这里。
