# data：运行时数据目录（不入库）

`.gitignore` 已忽略本目录下除 README 外的所有内容。原始数据体量大，且 KuaiSAR 等数据集
的许可（CC BY-NC-SA 4.0）不允许随代码分发。

## 布局

```
data/
├── raw/ml-1m/                  MovieLens-1M（ratings.dat 等，见 configs/datasets/movielens.toml）
├── processed/                  MovieLens 的处理结果
└── kuairec/                    KuaiRec 2.0
    ├── raw/KuaiRec.zip
    ├── raw/big_matrix.csv
    ├── raw/small_matrix.csv
    ├── raw/item_categories.csv
    ├── raw/user_features.csv
    └── processed/
```

MovieLens 是最早的数据线，处理结果一直放在 `data/` 根下；KuaiRec 有自己的子目录。
这个差异由 `recsys.data.loader.resolve_dataset_data_dir` 固化，不要手工改目录结构。

`processed/` 目前只写不读（供人工核查映射与切分），详见 [../docs/architecture.md](../docs/architecture.md) 的「已知限制」。

## 下载

```powershell
python -m recsys.data.prepare --dataset movielens   # ~25 MB
python -m recsys.data.prepare --dataset kuairec     # ~1 GB，含 md5 校验
```

MovieLens 解压到 `data/raw/`，KuaiRec 解压到 `data/kuairec/raw/`；两种数据集的目录约定见
[../configs/datasets/](../configs/datasets/) 与 [../docs/kuairec.md](../docs/kuairec.md)。

## 注意

- 不要把 `data/` 提交进 git：体积、许可、以及「同一份数据不同版本的静默差异」都是风险；
- 换机器时重新下载，不要从别处拷贝半截文件——`prepare` 会做完整性校验；
- 需要长期保留的小样例（用于测试）放到 `tests/fixtures/`，不要放这里。
