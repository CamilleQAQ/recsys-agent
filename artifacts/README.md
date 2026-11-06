# artifacts：训练产物（不入库）

`.gitignore` 忽略了本目录下除 README 外的所有文件。权重与指标都可由代码 + 数据重新生成，
只需在 `experiments/<run>/checkpoint.txt` 记录路径与 sha256。

## 命名约定

| 产物 | 路径 |
| --- | --- |
| Two-Tower | `artifacts/<dataset>/two_tower.pt` |
| BPR-MF | `artifacts/<dataset>/matrix_factorization.pt` |
| SASRec | `artifacts/<dataset>/sasrec.pt` |
| 评测指标 | `artifacts/<dataset>/*_metrics.json` |
| 数据体检 | `artifacts/kuairec_data_stats.json` |

`<dataset>` 取 `movielens` / `kuairec`（与 `--dataset` 一致）。

## checkpoint 里有什么

```text
model_state_dict   torch state_dict
model_config       重建模型所需的结构参数
split_strategy     数据切分协议，例如 per_user_last_two_strict_timestamp_v2
dataset            数据集名
positive_threshold 正反馈阈值
train_config       完整训练参数
provenance         工件溯源（见下）
history / best_validation_metrics（SASRec / MF / Two-Tower）
```

规则：**缺 `split_strategy` 或与当前常量不符的 checkpoint 拒绝加载**。
不要手工改这些字段——要改就重训，否则指标无法解释。

## provenance（PROJECT_MASTER §5.2）

| 字段 | 含义 |
| --- | --- |
| `data.counts` / `data.split_sha256` | 三个切分的行数与内容哈希 |
| `data.seen_masks` | 曝光 mask 的形状 / nnz / 哈希 |
| `id_mapping.sha256` / `sample` | ID 映射摘要与样例（完整映射在 `data/<dataset>/processed/id_mappings.json`） |
| `code.sha256` / `package_version` | `src/recsys/**/*.py` 内容哈希，用于发现代码漂移 |
| `environment` | Python / torch / numpy / faiss 版本 |
| `created_at` | UTC 时间戳 |

两个 checkpoint 能不能比，先看 `dataset`、`positive_threshold`、`split_strategy`、`data.split_sha256` 是否一致。

## 复现一次评测

```powershell
python -m recsys.adapters evaluate --model sasrec `
    --checkpoint artifacts/kuairec/sasrec.pt --split test
```

输出开头会打印 `provenance: data=… code=… users=… items=… created_at=…`，
随后是 Recall / NDCG / HitRate / Precision / Coverage。有 checkpoint 时，数据集与阈值从 checkpoint 里读。
