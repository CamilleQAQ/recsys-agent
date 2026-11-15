<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# 召回模型（retrieval）

本页由 `models/zoo.toml` 生成；源码仍在 `src/`（自研）或 `third_party/`（vendor），这里只做分类索引，不搬运代码，避免破坏上游相对导入。

从全量物品里挑出候选集。Phase 1 已把双塔、图、序列、生成式与内容增强路线接进统一链路。

> 可直接跑：popularity / matrix_factorization / two_tower / dssm / mind / lightgcn / sasrec / bert4rec / hstu / tiger / mmrec_multimodal；RQ-VAE/Semantic ID 是 TIGER 的表示层组件。OneRec / TFRS / TorchEasyRec SID 受权重或运行时限制，见各条目。

| id | 模型 | 来源 | 关键源码 | 完整度 |
| --- | --- | --- | --- | --- |
| `popularity` | Popularity | 自研 | `src/recsys/models/popularity.py` | 可直接运行 |
| `matrix_factorization` | BPR-MF（矩阵分解） | 自研 | `src/recsys/models/matrix_factorization.py` | 可直接运行 |
| `two_tower` | Two-Tower（ID 双塔） | 自研 | `src/recsys/models/two_tower.py` | 可直接运行 |
| `lightgcn` | LightGCN | vendor:RecBole | `third_party/RecBole/recbole/model/general_recommender/lightgcn.py` | 可直接运行 |
| `dssm` | DSSM（双塔） | vendor:TorchEasyRec | `third_party/TorchEasyRec/tzrec/models/dssm.py` | 可直接运行 |
| `mind` | MIND（多兴趣召回） | vendor:torch-rechub | `third_party/torch-rechub/torch_rechub/models/matching/mind.py` | 可直接运行 |
| `sid_rqvae` | 语义 ID 召回（RQ-VAE，TorchEasyRec 路线） | vendor:TorchEasyRec | `third_party/TorchEasyRec/tzrec/models/sid_rqvae.py` | 受依赖/权重限制 |
| `tiger` | TIGER（生成式检索） | vendor:genrec | `third_party/genrec/genrec/models/tiger.py` | 可直接运行 |
| `hstu` | HSTU（生成式序列推荐） | vendor:TorchEasyRec | `third_party/TorchEasyRec/tzrec/models/hstu.py` | 可直接运行 |
| `onerec` | OneRec（Qwen3 生成式推荐） | vendor:OpenOneRec | `third_party/OpenOneRec/pretrain/onerec_llm/models/qwen3/modeling_qwen3.py` | 受依赖/权重限制 |
| `tfrs_two_tower` | TFRS Two-Tower（TensorFlow） | vendor:recommenders | `third_party/recommenders/tensorflow_recommenders/tasks/retrieval.py` | 受依赖/权重限制 |
| `mmrec_multimodal` | MMRec 多模态推荐框架 | vendor:MMRec | `third_party/MMRec/src/models/lattice.py` | 可直接运行 |
| `semantic_id_rqvae` | RQ-VAE / Semantic ID（本项目适配） | 自研 | `src/recsys/models/rqvae.py` | 可直接运行 |
