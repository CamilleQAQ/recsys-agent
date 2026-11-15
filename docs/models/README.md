<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# 模型技术档案

已接入统一接口（`completeness = ready`）的模型各有一份独立档案，记录原理、入口、工件、已知局限与实验记录；尚未接入的模型信息在 [../model-zoo.md](../model-zoo.md) 的卡片里。

规则（由 `scripts/zoo.py` 自动执行）：`models/zoo.toml` 里某条记录把 `completeness` 改成 `ready` 并重新 render，这里就会多一份档案；`limitations` 字段直接写进「已知局限」，没填时显示「暂无记录」——宁可留空，也不编。

| 模型 | 档案 |
| --- | --- |
| Popularity | [`popularity.md`](popularity.md) |
| BPR-MF（矩阵分解） | [`matrix_factorization.md`](matrix_factorization.md) |
| Two-Tower（ID 双塔） | [`two_tower.md`](two_tower.md) |
| LightGCN | [`lightgcn.md`](lightgcn.md) |
| DSSM（双塔） | [`dssm.md`](dssm.md) |
| MIND（多兴趣召回） | [`mind.md`](mind.md) |
| TIGER（生成式检索） | [`tiger.md`](tiger.md) |
| HSTU（生成式序列推荐） | [`hstu.md`](hstu.md) |
| MMRec 多模态推荐框架 | [`mmrec_multimodal.md`](mmrec_multimodal.md) |
| RQ-VAE / Semantic ID（本项目适配） | [`semantic_id_rqvae.md`](semantic_id_rqvae.md) |
| SASRec（本项目实现） | [`sasrec.md`](sasrec.md) |
| BERT4Rec | [`bert4rec.md`](bert4rec.md) |
| DIN（深度兴趣网络） | [`din.md`](din.md) |
| DIEN（深度兴趣演化） | [`dien.md`](dien.md) |
| DCNv2（Deep & Cross v2） | [`dcnv2.md`](dcnv2.md) |
| DeepFM | [`deepfm.md`](deepfm.md) |
| MMoE（多门混合专家） | [`mmoe.md`](mmoe.md) |
| PLE（渐进式分层提取） | [`ple.md`](ple.md) |
| LightGBM LambdaRank | [`lightgbm_ranker.md`](lightgbm_ranker.md) |
| Ultra-HSTU（TorchEasyRec 排序版） | [`ultra_hstu.md`](ultra_hstu.md) |
| FAISS（向量检索） | [`faiss.md`](faiss.md) |
| FAISS 物品索引封装（本项目） | [`faiss_item_index.md`](faiss_item_index.md) |
