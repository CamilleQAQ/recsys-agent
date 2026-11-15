<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# 排序模型（ranking）

本页由 `models/zoo.toml` 生成；源码仍在 `src/`（自研）或 `third_party/`（vendor），这里只做分类索引，不搬运代码，避免破坏上游相对导入。

对候选集做精排、CTR 预估或序列下一物品预测；LightGBM 与四个神经排序器都已接入链路。

> LightGBM 是默认精排；DIN / DIEN / DCNv2 / Ultra-HSTU 用项目特征视图与请求级候选训练。DeepFM / MMoE 留给 Phase 3 广告。

| id | 模型 | 来源 | 关键源码 | 完整度 |
| --- | --- | --- | --- | --- |
| `sasrec` | SASRec（本项目实现） | 自研 | `src/recsys/models/sasrec.py` | 可直接运行 |
| `bert4rec` | BERT4Rec | vendor:BERT4Rec | `third_party/BERT4Rec/modeling.py` | 可直接运行 |
| `din` | DIN（深度兴趣网络） | vendor:FuxiCTR | `third_party/FuxiCTR/model_zoo/DIN/src/DIN.py` | 可直接运行 |
| `dien` | DIEN（深度兴趣演化） | vendor:DeepCTR-Torch | `third_party/DeepCTR-Torch/deepctr_torch/models/dien.py` | 可直接运行 |
| `dcnv2` | DCNv2（Deep & Cross v2） | vendor:FuxiCTR | `third_party/FuxiCTR/model_zoo/DCNv2/src/DCNv2.py` | 可直接运行 |
| `deepfm` | DeepFM | vendor:DeepCTR-Torch | `third_party/DeepCTR-Torch/deepctr_torch/models/deepfm.py` | 可直接运行 |
| `mmoe` | MMoE（多门混合专家） | vendor:DeepCTR-Torch | `third_party/DeepCTR-Torch/deepctr_torch/models/multitask/mmoe.py` | 可直接运行 |
| `ple` | PLE（渐进式分层提取） | vendor:DeepCTR-Torch | `third_party/DeepCTR-Torch/deepctr_torch/models/multitask/ple.py` | 可直接运行 |
| `lightgbm_ranker` | LightGBM LambdaRank | vendor:LightGBM | `third_party/LightGBM/python-package/lightgbm/sklearn.py` | 可直接运行 |
| `ultra_hstu` | Ultra-HSTU（TorchEasyRec 排序版） | vendor:TorchEasyRec | `third_party/TorchEasyRec/tzrec/models/ultra_hstu.py` | 可直接运行 |
