# 2026-11-09 · KuaiSAR Search MVP（词表核验 + BM25）

本目录是第一轮搜索链路验收的摘要容器：只覆盖数据准备、BM25 建索引和查询 CLI，
不包含 Search Agent、向量检索、排序/重排或全量检索评测。

## 本次交付

- [report.md](report.md)：词表判断、采样规则、计数/覆盖率、索引、命令与问题记录。
- [build_summary.json](build_summary.json)：构建汇总（来源、扫描上限、样本计数、覆盖率、
  词表证据、索引规模与 sha256）。与 `artifacts/search/kuasar_bm25/manifest.json` 同内容。
- [smoke.json](smoke.json)：单次冒烟验收结果（有匹配 / 空查询 / 无匹配，9/9 通过）。

## 结果速览

- 扫描上限：`src_inter` 200,000 行；`item_features` 500,000 行；两者均截断。
- 样本：**30 个搜索会话 / 120 个索引物品**；曝光行关联覆盖率 **55.814%**（120/215）。
- 索引：BM25 k1=1.5, b=0.75；120 docs / 823 vocab；76,077 B；
  sha256 `3ae2b858d456eab4c4c21ed356812cd44260b56dd596639588a53fb1a6a52690`。
- 词表：`keyword` 与 `caption` 共用 KuaiSAR token ID（论文陈述 + 抽样 5.69× 富集 +
  点击重合 51.16%）；使用原始 token ID，未使用本地夹具。
- 冒烟：会话 `11208` 的 4-token 查询返回 7 个匹配；Top-5 分数降序，点击物品 80675 排第 1。

## 运行

```powershell
$env:PYTHONPATH = "src"

# 采样 + 建索引（默认复用已校验产物；加 --force 重建）
python -m recsys.search.prepare `
  --summary-out experiments/2026-11-09-kuasar-search-bm25/build_summary.json

# 查询
python -m recsys.search --tokens "3538762,3181091,4061567,3525929" --top-k 5

# 单次冒烟
python -m recsys.search.smoke --top-k 5 `
  --output experiments/2026-11-09-kuasar-search-bm25/smoke.json
```

实现与配置：[prepare.py](../../src/recsys/search/prepare.py)、
[bm25.py](../../src/recsys/search/bm25.py)、
[查询 CLI](../../src/recsys/search/__main__.py)、
[冒烟脚本](../../src/recsys/search/smoke.py)、
[kuasar_search.toml](../../configs/search/kuasar_search.toml)。
