# 数据目录（上游示例语料已移除）

本目录原有 `beauty.txt`、`steam.txt`、`ml-1m.txt`、`ml-20m.zip`（合计约 104 MB），
已在本项目的仓库清理中删除，原因与清单见 [docs/cleanup-report.md](../../../docs/cleanup-report.md)。

需要数据时：

- MovieLens-1M：用本项目 `python -m recsys.data.prepare --dataset movielens`（会自动下载）；
- Beauty / Steam / ML-20M：按 [../README.md](../README.md) 的说明从原始来源获取，
  再用 `gen_data_fin.py` 生成 `*_data.pickle`。

注意：上游 `run_*.sh` 默认按 `data/<name>.txt` 找文件，放好数据后这些脚本可直接运行。
