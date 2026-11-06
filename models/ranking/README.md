<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# 排序模型（ranking）

本页由 `models/zoo.toml` 生成；源码仍在 `src/`（自研）或 `third_party/`（vendor），这里只做分类索引，不搬运代码，避免破坏上游相对导入。

对候选集做精排或序列下一物品预测。当前只有 SASRec 完成了端到端接入。

> DeepCTR 系列（DIN / DIEN / DCNv2 / DeepFM / MMoE）缺的是特征通道，不是模型代码；LightGBM 是最省事的精排 baseline。

| id | 模型 | 来源 | 关键源码 | 完整度 |
| --- | --- | --- | --- | --- |
| `sasrec` | SASRec（本项目实现） | 自研 | `src/recsys/models/sasrec.py` | 可直接运行 |
| `bert4rec` | BERT4Rec | vendor:BERT4Rec | `third_party/BERT4Rec/modeling.py` | 受依赖/权重限制 |
| `din` | DIN（深度兴趣网络） | vendor:FuxiCTR | `third_party/FuxiCTR/model_zoo/DIN/src/DIN.py` | 上游可用，未接入 |
| `dien` | DIEN（深度兴趣演化） | vendor:DeepCTR-Torch | `third_party/DeepCTR-Torch/deepctr_torch/models/dien.py` | 上游可用，未接入 |
| `dcnv2` | DCNv2（Deep & Cross v2） | vendor:FuxiCTR | `third_party/FuxiCTR/model_zoo/DCNv2/src/DCNv2.py` | 上游可用，未接入 |
| `deepfm` | DeepFM | vendor:DeepCTR-Torch | `third_party/DeepCTR-Torch/deepctr_torch/models/deepfm.py` | 上游可用，未接入 |
| `mmoe` | MMoE / PLE（多任务） | vendor:DeepCTR-Torch | `third_party/DeepCTR-Torch/deepctr_torch/models/multitask/mmoe.py` | 上游可用，未接入 |
| `lightgbm_ranker` | LightGBM LambdaRank | vendor:LightGBM | `third_party/LightGBM/python-package/lightgbm/sklearn.py` | 上游可用，未接入 |
| `ultra_hstu` | Ultra-HSTU（TorchEasyRec 排序版） | vendor:TorchEasyRec | `third_party/TorchEasyRec/tzrec/models/ultra_hstu.py` | 受依赖/权重限制 |
