<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# PLE（渐进式分层提取） 技术档案

## 基本信息

- **ID**：`ple`
- **分类**：排序模型（ranking）
- **来源**：vendor：third_party/DeepCTR-Torch
- **许可证**：Apache-2.0
- **上游**：https://github.com/shenweichen/DeepCTR-Torch
- **Commit**：`9eccef78e5810c61d6a04ddc7d96279e3db9c970`
- **源码**：`third_party/DeepCTR-Torch/deepctr_torch/models/multitask/ple.py`

## 原理与适用场景

- **原理**：共享专家 + 任务专属专家的分层网络，减少任务间负迁移。
- **适用场景**：点击 + 转化双任务；与 MMoE 同数据口径对照。

## 运行方式

- **依赖 / Python**：torch、scikit-learn、tqdm；Python >=3.8
- **训练入口**：（本轮只交付推理入口；广告训练未执行）
- **推理入口**：python -m recsys.ads.predict --model ple --mode multitask
- **数据格式**：广告候选特征字典 + click 标签（CVR 用 synthetic 模拟标签）
- **保存/加载**：artifacts/<dataset>/ads/interface_weights/ple.pt（固定初始化，interface_test）
- **适配层**：src/recsys/ads/models.py::DeepCTRAdAdapter

## 状态与局限

- **代码完整度**：可直接运行（ready）
- **依据**：复用本地 DeepCTR-Torch PLE；CTR/CVR 双输出与模拟转化样本已验证，权重为固定初始化接口测试，未训练
- **已知局限**：暂无记录（接入实验后补充）

## 逐项状态

| 源码 | 已核源码 | 适配 | 可运行 | 业务接入 | Agent | 已评估 |
| --- | --- | --- | --- | --- | --- | --- |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

## 实验记录

每次正式实验在本节追加一行：日期、数据版本、关键参数、指标、结论、对应 `experiments/<run>/` 目录。没有实验记录时不要写结论。

| 日期 | 数据集 | 参数摘要 | 指标 | 结论 | 记录目录 |
| --- | --- | --- | --- | --- | --- |
| — | — | — | — | 尚未开始系统性实验 | — |
