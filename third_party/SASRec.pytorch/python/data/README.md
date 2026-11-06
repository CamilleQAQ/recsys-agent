# 数据目录（上游示例语料已移除）

本目录原有 `Steam.txt`、`ml-1m.txt`、`Beauty.txt`、`Video.txt`、`wikipedia.txt`（合计约 63 MB），
已在本项目的仓库清理中删除，原因与清单见 [docs/cleanup-report.md](../../../../docs/cleanup-report.md)。

需要数据时：在上游 README 里找到数据来源，下载后按 `data/<dataset>.txt` 命名，
每行 `user_id item_id`，按用户交互顺序排列；Train/Valid/Test 由 `main.py` 内部切分。

MovieLens-1M 可以直接用本项目的下载脚本获取，再复制到本目录：

```powershell
python -m recsys.data.prepare --dataset movielens
```
