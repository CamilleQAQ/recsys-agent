<!-- 本文件由 scripts/zoo.py 从 models/zoo.toml 生成，请勿手改；改目录请编辑 zoo.toml 后运行 python scripts/zoo.py render -->

# Model Zoo 总索引

每个模型的名称、来源、原理、依赖、训练/推理入口、数据格式、保存方式和代码完整度。
召回/排序/基础设施三个分类的细分视图见 `models/<category>/README.md`。

## 统计

| 分类 | 模型数 | 可直接运行 | 上游可用，未接入 | 受依赖/权重限制 |
| --- | --- | --- | --- | --- |
| retrieval | 12 | 3 | 1 | 8 |
| ranking | 9 | 1 | 6 | 2 |
| infrastructure | 3 | 2 | 0 | 1 |
| **合计** | **24** | **6** | **7** | **11** |

## 一览

| id | 模型 | 分类 | 来源 | 完整度 | 适配层 |
| --- | --- | --- | --- | --- | --- |
| `popularity` | Popularity | retrieval | 自研 | 可直接运行 | PopularityAdapter |
| `matrix_factorization` | BPR-MF（矩阵分解） | retrieval | 自研 | 可直接运行 | MatrixFactorizationAdapter |
| `two_tower` | Two-Tower（ID 双塔） | retrieval | 自研 | 可直接运行 | TwoTowerAdapter |
| `lightgcn` | LightGCN | retrieval | vendor:RecBole | 上游可用，未接入 | — |
| `dssm` | DSSM（双塔） | retrieval | vendor:TorchEasyRec | 受依赖/权重限制 | — |
| `mind` | MIND（多兴趣召回） | retrieval | vendor:TorchEasyRec | 受依赖/权重限制 | — |
| `sid_rqvae` | 语义 ID 召回（RQ-VAE，TorchEasyRec 路线） | retrieval | vendor:TorchEasyRec | 受依赖/权重限制 | — |
| `tiger` | TIGER（生成式检索） | retrieval | vendor:genrec | 受依赖/权重限制 | — |
| `hstu` | HSTU（生成式序列推荐） | retrieval | vendor:TorchEasyRec | 受依赖/权重限制 | — |
| `onerec` | OneRec（Qwen3 生成式推荐） | retrieval | vendor:OpenOneRec | 受依赖/权重限制 | — |
| `tfrs_two_tower` | TFRS Two-Tower（TensorFlow） | retrieval | vendor:recommenders | 受依赖/权重限制 | — |
| `mmrec_multimodal` | MMRec 多模态推荐框架 | retrieval | vendor:MMRec | 受依赖/权重限制 | — |
| `sasrec` | SASRec（本项目实现） | ranking | 自研 | 可直接运行 | SASRecAdapter |
| `bert4rec` | BERT4Rec | ranking | vendor:BERT4Rec | 受依赖/权重限制 | — |
| `din` | DIN（深度兴趣网络） | ranking | vendor:FuxiCTR | 上游可用，未接入 | — |
| `dien` | DIEN（深度兴趣演化） | ranking | vendor:DeepCTR-Torch | 上游可用，未接入 | — |
| `dcnv2` | DCNv2（Deep & Cross v2） | ranking | vendor:FuxiCTR | 上游可用，未接入 | — |
| `deepfm` | DeepFM | ranking | vendor:DeepCTR-Torch | 上游可用，未接入 | — |
| `mmoe` | MMoE / PLE（多任务） | ranking | vendor:DeepCTR-Torch | 上游可用，未接入 | — |
| `lightgbm_ranker` | LightGBM LambdaRank | ranking | vendor:LightGBM | 上游可用，未接入 | — |
| `ultra_hstu` | Ultra-HSTU（TorchEasyRec 排序版） | ranking | vendor:TorchEasyRec | 受依赖/权重限制 | — |
| `faiss` | FAISS（向量检索） | infrastructure | vendor:faiss | 可直接运行 | — |
| `faiss_item_index` | FAISS 物品索引封装（本项目） | infrastructure | 自研 | 可直接运行 | FaissItemIndex |
| `tfrs_factorized_top_k` | TFRS factorized_top_k | infrastructure | vendor:recommenders | 受依赖/权重限制 | — |

## 召回模型（retrieval）

从全量物品里挑出候选集。先用自研基线把数据线跑通，再换更重的模型。

### Popularity

- **ID**：`popularity`
- **分类**：召回模型 / first_party
- **来源**：本项目自研（src/）
- **许可证**：项目自有
- **源码**：`src/recsys/models/popularity.py`
- **原理**：按训练集交互次数对物品排序，推荐全局最热且用户未看过的 Top-K。
- **适用场景**：基线对照；任何数据集的冷启动兜底。
- **依赖 / Python**：无；Python >=3.11
- **训练入口**：无需训练；统计来自 PreparedData.train_interactions
- **推理入口**：python -m recsys.experiments.compare（或适配层 PopularityAdapter）
- **数据格式**：PreparedData（连续整数 user_idx/item_idx/timestamp）
- **保存/加载**：无参数，不需要保存；每次由训练集现算
- **代码完整度**：可直接运行（ready）—— 已实现，并被 compare/finalize 脚本作为对照基线使用
- **适配层**：PopularityAdapter

### BPR-MF（矩阵分解）

- **ID**：`matrix_factorization`
- **分类**：召回模型 / first_party
- **来源**：本项目自研（src/）
- **许可证**：项目自有
- **源码**：`src/recsys/models/matrix_factorization.py`、`src/recsys/training/train_mf.py`
- **原理**：用户/物品各学一个隐向量，点积即偏好分数，用 BPR 负采样训练。
- **适用场景**：隐式反馈基线；召回通道的最小可用模型。
- **依赖 / Python**：torch；Python >=3.11
- **训练入口**：python -m recsys.training.train_mf --dataset kuairec
- **推理入口**：python -m recsys.evaluation.evaluator --checkpoint artifacts/<dataset>/matrix_factorization.pt
- **数据格式**：PreparedData；评测用 seen-mask 过滤已曝光物品
- **保存/加载**：artifacts/<dataset>/matrix_factorization.pt（model_config + model_state_dict + train_config）
- **代码完整度**：可直接运行（ready）—— 训练、checkpoint、评测入口齐全
- **适配层**：MatrixFactorizationAdapter

### Two-Tower（ID 双塔）

- **ID**：`two_tower`
- **分类**：召回模型 / first_party
- **来源**：本项目自研（src/）
- **许可证**：项目自有
- **源码**：`src/recsys/models/two_tower.py`、`src/recsys/training/trainer.py`
- **原理**：用户塔/物品塔各自编码并 L2 归一化，用 batch 内负样本的交叉熵训练，内积即余弦相似度。
- **适用场景**：召回主干；可对接 FAISS 做全量 Top-K。
- **依赖 / Python**：torch、faiss-cpu（可选）；Python >=3.11
- **训练入口**：python -m recsys.training.trainer --dataset kuairec
- **推理入口**：python -m recsys.evaluation.evaluator --checkpoint artifacts/<dataset>/two_tower.pt
- **数据格式**：PreparedData；物品词表必须与 checkpoint 一致
- **保存/加载**：artifacts/<dataset>/two_tower.pt（自带 model_config 与 split_strategy 校验）
- **代码完整度**：可直接运行（ready）—— 训练/暴力评测/FAISS 评测均已实现
- **适配层**：TwoTowerAdapter

### LightGCN

- **ID**：`lightgcn`
- **分类**：召回模型 / vendor
- **来源**：third_party/RecBole（上游 https://github.com/RUCAIBox/RecBole）
- **Commit**：`7b02be5ec80a88310f2d04a27a82adfcbb5dc211`
- **许可证**：MIT（RecBole）/ GPL-3.0（MMRec）
- **源码**：`third_party/RecBole/recbole/model/general_recommender/lightgcn.py`、`third_party/MMRec/src/models/lightgcn.py`
- **原理**：在用户-物品二部图上做多层无特征线性传播并加权求和，去掉非线性与自连接。
- **适用场景**：协同过滤召回；交互图稠密、无侧信息的场景。
- **依赖 / Python**：torch、recbole；Python >=3.10
- **训练入口**：cd third_party/RecBole && python run_recbole.py --model=LightGCN --dataset=<atomic数据集>
- **推理入口**：RecBole Trainer.evaluate / 模型 full_sort_predict
- **数据格式**：RecBole atomic 文件（.inter/.user/.item，带类型表头）
- **保存/加载**：RecBole saved/<model>/ 下的 .pth 与训练配置
- **代码完整度**：上游可用，未接入（upstream_ready）—— 上游实现完整；尚未接入 PreparedData 适配层，需要先做 atomic 文件或内存数据桥接
- **适配层**：尚未接入

### DSSM（双塔）

- **ID**：`dssm`
- **分类**：召回模型 / vendor
- **来源**：third_party/TorchEasyRec（上游 https://github.com/alibaba/TorchEasyRec）
- **Commit**：`2815a9732bed014219ee89826dd4987b67ccf6b0`
- **许可证**：Apache-2.0
- **源码**：`third_party/TorchEasyRec/tzrec/models/dssm.py`、`third_party/TorchEasyRec/tzrec/models/dssm_v2.py`
- **原理**：用户侧/物品侧特征分别过 MLP 成向量，batch 内负采样 + 采样 softmax 训练。
- **适用场景**：工业级双塔召回参考实现（含特征工程与在线 serving 格式）。
- **依赖 / Python**：torch、tzrec；Python >=3.10（上游 wheel 面向 3.10-3.12 Linux）
- **训练入口**：cd third_party/TorchEasyRec && python -m tzrec.train_eval --pipeline_config_path <config>
- **推理入口**：python -m tzrec.predict --pipeline_config_path <config>
- **数据格式**：CSV/Parquet/ODPS + pipeline protobuf 声明特征、标签与模型
- **保存/加载**：训练输出目录下的 model/（TorchScript 导出 + 配置快照）
- **代码完整度**：受依赖/权重限制（blocked）—— 依赖阿里 OSS 上的 Linux-only wheel（graphlearn/pyfg/common_io），Windows 无法直接安装，需要独立 Linux 环境
- **适配层**：尚未接入

### MIND（多兴趣召回）

- **ID**：`mind`
- **分类**：召回模型 / vendor
- **来源**：third_party/TorchEasyRec（上游 https://github.com/alibaba/TorchEasyRec）
- **Commit**：`2815a9732bed014219ee89826dd4987b67ccf6b0`
- **许可证**：Apache-2.0
- **源码**：`third_party/TorchEasyRec/tzrec/models/mind.py`
- **原理**：在双塔结构上对用户行为序列做动态路由聚类，得到多个兴趣向量分别召回后合并。
- **适用场景**：用户兴趣多峰的短视频/电商场景，与 KuaiSAR 的短期兴趣建模契合。
- **依赖 / Python**：torch、tzrec；Python >=3.10（Linux）
- **训练入口**：python -m tzrec.train_eval --pipeline_config_path examples/mind_taobao.config
- **推理入口**：python -m tzrec.predict --pipeline_config_path <config>
- **数据格式**：用户 item 序列特征 + 候选特征，pipeline protobuf 声明
- **保存/加载**：同 TorchEasyRec（model/ 目录）
- **代码完整度**：受依赖/权重限制（blocked）—— 同 DSSM，受 TorchEasyRec 平台依赖限制，需 Linux 环境验证
- **适配层**：尚未接入

### 语义 ID 召回（RQ-VAE，TorchEasyRec 路线）

- **ID**：`sid_rqvae`
- **分类**：召回模型 / vendor
- **来源**：third_party/TorchEasyRec（上游 https://github.com/alibaba/TorchEasyRec）
- **Commit**：`2815a9732bed014219ee89826dd4987b67ccf6b0`
- **许可证**：Apache-2.0
- **源码**：`third_party/TorchEasyRec/tzrec/models/sid_rqvae.py`、`third_party/TorchEasyRec/tzrec/models/sid_rqkmeans.py`
- **原理**：用残差量化把物品内容/协同向量压成层级语义 ID，再以生成式或前缀检索方式召回候选。
- **适用场景**：生成式召回的工业落地路径；冷启动与跨域迁移。
- **依赖 / Python**：torch、tzrec、transformers；Python >=3.10（Linux）
- **训练入口**：python -m tzrec.train_eval --pipeline_config_path <sid config>
- **推理入口**：python -m tzrec.predict --pipeline_config_path <config>
- **数据格式**：物品侧 embedding/多模态特征 → 语义 ID；生成式侧需序列化 token
- **保存/加载**：同 TorchEasyRec；语义 ID 码本必须与模型一起版本化
- **代码完整度**：受依赖/权重限制（blocked）—— 受 TorchEasyRec 平台依赖限制；TIGER 的正式实现在 genrec（见 tiger 条目），此处只保留工业语义 ID 路线
- **适配层**：尚未接入

### TIGER（生成式检索）

- **ID**：`tiger`
- **分类**：召回模型 / vendor
- **来源**：third_party/genrec（上游 https://github.com/phonism/genrec）
- **Commit**：`2676b5b08e2efc0b873ab4c3d66882611a62ca2e`
- **许可证**：MIT
- **源码**：`third_party/genrec/genrec/models/tiger.py`、`third_party/genrec/genrec/models/rqvae.py`、`third_party/genrec/config/tiger/amazon/tiger.gin`
- **原理**：先用 RQ-VAE 把物品压成层级语义 ID，再用 Transformer 以序列到序列方式生成候选 ID，并用 trie 约束解码。
- **适用场景**：生成式召回研究；与向量召回做对照；物品语义表示实验。
- **依赖 / Python**：torch==2.6、transformers、gin_config、wandb、sentence_transformers；Python >=3.9（genrec 声明）
- **训练入口**：cd third_party/genrec && python -m genrec.trainers.rqvae_trainer config/tiger/amazon/rqvae.gin && python -m genrec.trainers.tiger_trainer config/tiger/amazon/tiger.gin
- **推理入口**：genrec.trainers.tiger_trainer 的 evaluate/生成路径（trie 约束解码）
- **数据格式**：交互序列 CSV + 物品 embedding（用于训练 RQ-VAE）→ 语义 ID 序列
- **保存/加载**：genrec 输出目录：RQ-VAE 码本 + TIGER 权重；两者必须成套使用
- **代码完整度**：受依赖/权重限制（blocked）—— genrec 固定 torch==2.6 并发依赖，需独立环境；本项目还没有物品 embedding 与语义 ID 构建流程（见 docs/kuasar-roadmap.md 第 5 节）
- **适配层**：尚未接入

### HSTU（生成式序列推荐）

- **ID**：`hstu`
- **分类**：召回模型 / vendor
- **来源**：third_party/TorchEasyRec（上游 https://github.com/alibaba/TorchEasyRec）
- **Commit**：`2815a9732bed014219ee89826dd4987b67ccf6b0`
- **许可证**：Apache-2.0
- **源码**：`third_party/TorchEasyRec/tzrec/models/hstu.py`、`third_party/TorchEasyRec/tzrec/models/ultra_hstu.py`
- **原理**：层次化序列转导单元：带时间偏置的注意力直接建模长行为序列。
- **适用场景**：长序列召回/排序统一建模；短视频场景。
- **依赖 / Python**：torch、tzrec；Python >=3.10（Linux）
- **训练入口**：python -m tzrec.train_eval --pipeline_config_path <hstu config>
- **推理入口**：python -m tzrec.predict --pipeline_config_path <config>
- **数据格式**：行为序列 + 时间戳 + 候选特征
- **保存/加载**：同 TorchEasyRec
- **代码完整度**：受依赖/权重限制（blocked）—— 需要 TorchEasyRec 运行栈；Meta 官方研究实现（generative-recommenders）可作算法对照，但 checkpoint 不通用
- **适配层**：尚未接入

### OneRec（Qwen3 生成式推荐）

- **ID**：`onerec`
- **分类**：召回模型 / vendor
- **来源**：third_party/OpenOneRec（上游 https://github.com/Kuaishou-OneRec/OpenOneRec）
- **Commit**：`a969edcadd579a06c1966ae1db5984e02f48beff`
- **许可证**：Apache-2.0（README 声明；仓库根 LICENSE 缺失）
- **源码**：`third_party/OpenOneRec/pretrain/onerec_llm/models/qwen3/modeling_qwen3.py`、`third_party/OpenOneRec/tokenizer/res_kmeans.py`
- **原理**：把推荐当成序列到序列生成：残差 K-Means 构造语义 ID，Qwen3 主干做 SFT/蒸馏/GRPO 生成候选。
- **适用场景**：生成式召回/重排研究；与快手系数据（KuaiSAR/KuaiRand）同源生态。
- **依赖 / Python**：torch、transformers、verl、vllm、ray；Python >=3.10（多卡 Linux）
- **训练入口**：third_party/OpenOneRec/pretrain/recipes/train_qwen3.py；verl_rl/recipe/onerec/run_grpo.sh
- **推理入口**：tokenizer/infer_res_kmeans.py；benchmarks/scripts/ray-vllm/evaluate.py
- **数据格式**：Parquet chat/segments + itemic token；tokenizer 侧需物品 embedding 与 pid
- **保存/加载**：HuggingFace 格式权重目录（外部 Qwen3/OneRec 权重，不随仓库提供）
- **代码完整度**：受依赖/权重限制（blocked）—— 需外部权重与 RecIF 数据、多卡环境；根许可证缺失，再分发前须确认
- **适配层**：尚未接入

### TFRS Two-Tower（TensorFlow）

- **ID**：`tfrs_two_tower`
- **分类**：召回模型 / vendor
- **来源**：third_party/recommenders（上游 https://github.com/tensorflow/recommenders）
- **Commit**：`a8a90aee3f0b470cda96d47e26de99eedb7fc730`
- **许可证**：Apache-2.0
- **源码**：`third_party/recommenders/tensorflow_recommenders/tasks/retrieval.py`、`third_party/recommenders/tensorflow_recommenders/layers/factorized_top_k.py`
- **原理**：Keras 定义查询塔/候选塔，Retrieval task 以采样 softmax 训练，factorized_top_k 提供 BruteForce/Streaming/ScaNN 检索。
- **适用场景**：TF 技术栈的召回实现；跨框架对照。
- **依赖 / Python**：tensorflow、tf-keras；Python >=3.9
- **训练入口**：Keras Model.fit（docs/examples/*.ipynb，无统一训练 CLI）
- **推理入口**：tensorflow_recommenders/tasks/retrieval.py + layers/factorized_top_k.py
- **数据格式**：tf.data.Dataset 特征字典（query/candidate 分开）
- **保存/加载**：tf.saved_model / Keras .keras
- **代码完整度**：受依赖/权重限制（blocked）—— 需要独立 TF2 环境，与 PyTorch 主栈版本冲突，不做同环境共存
- **适配层**：尚未接入

### MMRec 多模态推荐框架

- **ID**：`mmrec_multimodal`
- **分类**：召回模型 / vendor
- **来源**：third_party/MMRec（上游 https://github.com/enoche/MMRec）
- **Commit**：`c68ee7c9b0de1ca7c5150258322edb4994be29d9`
- **许可证**：GPL-3.0
- **源码**：`third_party/MMRec/src/models/lattice.py`、`third_party/MMRec/src/models/vbpr.py`、`third_party/MMRec/src/main.py`
- **原理**：用视觉/文本预提取特征增强用户-物品图模型（VBPR、LATTICE、DualGNN 等 22 个实现）。
- **适用场景**：有多模态物品特征的场景；KuaiSAR 的视频/封面特征可复用同一范式。
- **依赖 / Python**：torch、torch_geometric、torch_scatter、torchvision；Python 3.7 时代实现，现代 Python 需移植
- **训练入口**：cd third_party/MMRec/src && python main.py --model=LATTICE --dataset=<name>
- **推理入口**：src/common/trainer.py + 各模型 full_sort_predict
- **数据格式**：交互表 + 与物品 ID 对齐的视觉/文本特征文件
- **保存/加载**：saved/<model>/<dataset>/ 下的 .pth
- **代码完整度**：受依赖/权重限制（blocked）—— requirements 固定 numpy1.21/pandas1.3.5/torch1.11；GPL-3.0 传染性需评估；特征数据需另行下载
- **适配层**：尚未接入


## 排序模型（ranking）

对候选集做精排或序列下一物品预测。当前只有 SASRec 完成了端到端接入。

### SASRec（本项目实现）

- **ID**：`sasrec`
- **分类**：排序模型 / first_party
- **来源**：本项目自研（src/）
- **Commit**：`26a8fa880f312c3700f8dcca5512d5854b257f4e`
- **许可证**：项目自有（对照实现 Apache-2.0）
- **源码**：`src/recsys/models/sasrec.py`、`src/recsys/training/train_sasrec.py`、`third_party/SASRec.pytorch/python/model.py`
- **原理**：单向（因果）自注意力编码用户历史，按「下一个物品」目标训练，对全量物品打分排序。
- **适用场景**：序列推荐主力模型；召回或排序均可，本项目按下一物品预测评测。
- **依赖 / Python**：torch；Python >=3.11
- **训练入口**：python -m recsys.training.train_sasrec --dataset kuairec
- **推理入口**：python -m recsys.evaluation.sasrec_evaluator --checkpoint artifacts/<dataset>/sasrec.pt
- **数据格式**：PreparedData → 右对齐/右补齐的 item 序列（token 0 为 padding）
- **保存/加载**：artifacts/<dataset>/sasrec.pt（含 max_seq_len 等结构配置）
- **代码完整度**：可直接运行（ready）—— 训练/早停/评测（含 train-history-only 对照）已实现
- **适配层**：SASRecAdapter

### BERT4Rec

- **ID**：`bert4rec`
- **分类**：排序模型 / vendor
- **来源**：third_party/BERT4Rec（上游 https://github.com/FeiSun/BERT4Rec）
- **Commit**：`615eaf2004abecda487a38d5b0c72f3dcfcae5b3`
- **许可证**：Apache-2.0
- **源码**：`third_party/BERT4Rec/modeling.py`、`third_party/BERT4Rec/run.py`、`third_party/BERT4Rec/gen_data_fin.py`
- **原理**：双向 Transformer + 掩码物品预测（Cloze），从两侧上下文推断被遮住的物品。
- **适用场景**：序列推荐；对双向上下文敏感的场景。
- **依赖 / Python**：tensorflow==1.12、numpy、six；Python 2.7/3.6（TF1.12）
- **训练入口**：python gen_data_fin.py && python run.py --train_input=<tfrecord> ...
- **推理入口**：run.py 的 Estimator.predict 路径（无现代 predict API）
- **数据格式**：用户物品序列 → masked TFRecord + vocab/history pickle
- **保存/加载**：TF Estimator checkpoint 目录
- **代码完整度**：受依赖/权重限制（blocked）—— 依赖 tf.contrib 等 TF1 API，无法与现代 TF2/PyTorch 环境共存，只能作算法参考
- **适配层**：尚未接入

### DIN（深度兴趣网络）

- **ID**：`din`
- **分类**：排序模型 / vendor
- **来源**：third_party/FuxiCTR（上游 https://github.com/reczoo/FuxiCTR）
- **Commit**：`183da1c944ba8930fcdcd54114837c081959d748`
- **许可证**：Apache-2.0
- **源码**：`third_party/FuxiCTR/model_zoo/DIN/src/DIN.py`、`third_party/DeepCTR-Torch/deepctr_torch/models/din.py`、`third_party/RecBole/recbole/model/sequential_recommender/din.py`
- **原理**：用注意力对用户历史行为按候选物品加权池化，刻画局部激活兴趣。
- **适用场景**：CTR 精排；行为序列 + 候选物品的注意力建模。
- **依赖 / Python**：torch、fuxictr、pandas、pyarrow、ray[data]；Python >=3.10（FuxiCTR）
- **训练入口**：cd third_party/FuxiCTR/model_zoo/DIN && python run_expid.py --config ./config --expid DIN_test --gpu 0
- **推理入口**：fuxictr BaseModel.predict + 保存的 checkpoint
- **数据格式**：特征表（CSV/Parquet）+ dataset_config.yaml 声明特征与标签
- **保存/加载**：FuxiCTR 输出目录下的 model.ckpt + 配置快照
- **代码完整度**：上游可用，未接入（upstream_ready）—— FuxiCTR（主档案指定来源）与 DeepCTR-Torch 都有完整实现；本项目尚未实现 KuaiSAR 特征表 → 两套输入的转换
- **适配层**：尚未接入

### DIEN（深度兴趣演化）

- **ID**：`dien`
- **分类**：排序模型 / vendor
- **来源**：third_party/DeepCTR-Torch（上游 https://github.com/shenweichen/DeepCTR-Torch）
- **Commit**：`9eccef78e5810c61d6a04ddc7d96279e3db9c970`
- **许可证**：Apache-2.0
- **源码**：`third_party/DeepCTR-Torch/deepctr_torch/models/dien.py`
- **原理**：兴趣抽取层 + AUGRU 兴趣演化层，让兴趣向量随行为序列演进。
- **适用场景**：长序列 CTR；兴趣漂移明显的场景。
- **依赖 / Python**：torch、scikit-learn、tqdm；Python >=3.8
- **训练入口**：python examples/run_dien.py
- **推理入口**：basemodel.predict
- **数据格式**：行为序列 + 序列长度字段（VarLenSparseFeat + seq_len）
- **保存/加载**：save_model/load_model
- **代码完整度**：上游可用，未接入（upstream_ready）—— 同上，缺数据转换层
- **适配层**：尚未接入

### DCNv2（Deep & Cross v2）

- **ID**：`dcnv2`
- **分类**：排序模型 / vendor
- **来源**：third_party/FuxiCTR（上游 https://github.com/reczoo/FuxiCTR）
- **Commit**：`183da1c944ba8930fcdcd54114837c081959d748`
- **许可证**：Apache-2.0
- **源码**：`third_party/FuxiCTR/model_zoo/DCNv2/src/DCNv2.py`、`third_party/DeepCTR-Torch/deepctr_torch/models/dcnmix.py`、`third_party/TorchEasyRec/tzrec/models/dcn_v2.py`
- **原理**：显式高阶特征交叉（矩阵式 cross layer + 低秩分解）与 DNN 并行，缓解 DCNv1 的标量权重限制。
- **适用场景**：CTR 精排主力结构；稠密特征交叉。
- **依赖 / Python**：torch、fuxictr、pandas、pyarrow、ray[data]；Python >=3.10（FuxiCTR）
- **训练入口**：cd third_party/FuxiCTR/model_zoo/DCNv2 && python run_expid.py --config ./config --expid DCNv2_test --gpu 0
- **推理入口**：fuxictr BaseModel.predict + 保存的 checkpoint
- **数据格式**：特征表（CSV/Parquet）+ dataset_config.yaml 声明特征与标签
- **保存/加载**：FuxiCTR 输出目录下的 model.ckpt + 配置快照（三个实现的 checkpoint 互不通用）
- **代码完整度**：上游可用，未接入（upstream_ready）—— FuxiCTR 为主来源；DeepCTR-Torch / TorchEasyRec 有等价实现，但互不通用，实验只选一条线
- **适配层**：尚未接入

### DeepFM

- **ID**：`deepfm`
- **分类**：排序模型 / vendor
- **来源**：third_party/DeepCTR-Torch（上游 https://github.com/shenweichen/DeepCTR-Torch）
- **Commit**：`9eccef78e5810c61d6a04ddc7d96279e3db9c970`
- **许可证**：Apache-2.0
- **源码**：`third_party/DeepCTR-Torch/deepctr_torch/models/deepfm.py`
- **原理**：FM 二阶交叉与 DNN 共享同一份 embedding，端到端联合训练。
- **适用场景**：CTR 基线；低阶+高阶交叉的通用起点。
- **依赖 / Python**：torch、scikit-learn、tqdm；Python >=3.8
- **训练入口**：python examples/run_classification_criteo.py
- **推理入口**：basemodel.predict
- **数据格式**：特征字典 + 标签
- **保存/加载**：save_model/load_model
- **代码完整度**：上游可用，未接入（upstream_ready）—— 上游样例自带 criteo 小样本，可直接跑 demo
- **适配层**：尚未接入

### MMoE / PLE（多任务）

- **ID**：`mmoe`
- **分类**：排序模型 / vendor
- **来源**：third_party/DeepCTR-Torch（上游 https://github.com/shenweichen/DeepCTR-Torch）
- **Commit**：`9eccef78e5810c61d6a04ddc7d96279e3db9c970`
- **许可证**：Apache-2.0
- **源码**：`third_party/DeepCTR-Torch/deepctr_torch/models/multitask/mmoe.py`、`third_party/DeepCTR-Torch/deepctr_torch/models/multitask/ple.py`
- **原理**：多专家 + 门控为每个任务组合专家输出；PLE 进一步做任务间分层隔离。
- **适用场景**：同时优化点击/时长/完播等多目标；KuaiSAR 的多行为标签可直接映射。
- **依赖 / Python**：torch、scikit-learn、tqdm；Python >=3.8
- **训练入口**：python examples/run_multitask_learning.py
- **推理入口**：basemodel.predict（多输出）
- **数据格式**：特征字典 + 多列标签
- **保存/加载**：save_model/load_model
- **代码完整度**：上游可用，未接入（upstream_ready）—— 多目标标签体系需按 KuaiSAR 行为重新定义
- **适配层**：尚未接入

### LightGBM LambdaRank

- **ID**：`lightgbm_ranker`
- **分类**：排序模型 / vendor
- **来源**：third_party/LightGBM（上游 https://github.com/lightgbm-org/LightGBM）
- **Commit**：`3ee03c26ba8c9e8ddc05fa8687717cae8a7d61d9`
- **许可证**：MIT
- **源码**：`third_party/LightGBM/python-package/lightgbm/sklearn.py`、`third_party/LightGBM/src/objective/rank_objective.hpp`
- **原理**：GBDT 直接优化排序目标（LambdaRank / rank_xendcg），按 query 分组计算 NDCG 梯度。
- **适用场景**：精排基线；特征工程友好、训练成本低的强 baseline。
- **依赖 / Python**：lightgbm、numpy、scipy、scikit-learn；Python >=3.11（python-package/pyproject.toml）
- **训练入口**：lightgbm.LGBMRanker.fit(X, y, group=...)（或 CLI：lightgbm config=train.conf）
- **推理入口**：Booster.predict / LGBMRanker.predict
- **数据格式**：稠密/稀疏特征矩阵 + label + 连续 query group（一个用户的候选为一个 group）
- **保存/加载**：booster.save_model('model.txt') / Booster(model_file=...)
- **代码完整度**：上游可用，未接入（upstream_ready）—— pip install lightgbm 即可用；本仓库源码仅在需要自定义编译时才需要
- **适配层**：尚未接入

### Ultra-HSTU（TorchEasyRec 排序版）

- **ID**：`ultra_hstu`
- **分类**：排序模型 / vendor
- **来源**：third_party/TorchEasyRec（上游 https://github.com/alibaba/TorchEasyRec）
- **Commit**：`2815a9732bed014219ee89826dd4987b67ccf6b0`
- **许可证**：Apache-2.0
- **源码**：`third_party/TorchEasyRec/tzrec/models/ultra_hstu.py`
- **原理**：在 HSTU 转导堆叠上接排序头，直接对候选做 CTR/时长预估。
- **适用场景**：长序列 + 生成式主干的精排。
- **依赖 / Python**：torch、tzrec；Python >=3.10（Linux）
- **训练入口**：python -m tzrec.train_eval --pipeline_config_path <ultra_hstu config>
- **推理入口**：python -m tzrec.predict --pipeline_config_path <config>
- **数据格式**：行为序列 + 候选特征 + 标签
- **保存/加载**：同 TorchEasyRec
- **代码完整度**：受依赖/权重限制（blocked）—— 平台依赖限制同 DSSM/MIND
- **适配层**：尚未接入


## 基础设施（infrastructure）

让召回跑起来的检索与向量能力，本身不是推荐模型。

### FAISS（向量检索）

- **ID**：`faiss`
- **分类**：基础设施 / vendor
- **来源**：third_party/faiss（上游 https://github.com/facebookresearch/faiss）
- **Commit**：`9acf8b0c16a51aafa12b9d9bcf13998b9138a8c5`
- **许可证**：MIT
- **源码**：`third_party/faiss/faiss/IndexFlat.cpp`、`third_party/faiss/faiss/IndexIVF.cpp`、`third_party/faiss/faiss/IndexHNSW.cpp`、`third_party/faiss/faiss/IndexPQ.cpp`
- **原理**：提供 Flat/IVF/PQ/HNSW/LSH 等近邻索引与聚类实现，对上游 embedding 做 Top-K 检索。
- **适用场景**：召回阶段的候选生成；大规模向量打分。本身不是推荐模型。
- **依赖 / Python**：faiss-cpu；Python >=3.10
- **训练入口**：Index.train(xb) / add(xb)（建索引即可，无需拟合）
- **推理入口**：Index.search(xq, k)
- **数据格式**：float32 N×d 向量；索引必须与建库时的 embedding 模型配套
- **保存/加载**：index.write('x.faiss') / faiss.read_index('x.faiss')
- **代码完整度**：可直接运行（ready）—— 本项目通过 recsys.retrieval.faiss_index 调用已安装的 faiss-cpu，仓库源码仅作参考，无需本地编译
- **适配层**：尚未接入

### FAISS 物品索引封装（本项目）

- **ID**：`faiss_item_index`
- **分类**：基础设施 / first_party
- **来源**：本项目自研（src/）
- **许可证**：项目自有
- **源码**：`src/recsys/retrieval/faiss_index.py`、`src/recsys/evaluation/faiss_evaluator.py`
- **原理**：把双塔用户/物品向量写入 FAISS IndexFlatIP，校验 L2 归一化，并与暴力检索做一致性对比。
- **适用场景**：全量召回评测、检索性能基准、后续在线 serving 的检索底座。
- **依赖 / Python**：faiss-cpu、numpy、torch；Python >=3.11
- **训练入口**：无（构建索引即 build_faiss_index）
- **推理入口**：python -m recsys.evaluation.faiss_evaluator --checkpoint artifacts/<dataset>/two_tower.pt
- **数据格式**：L2 归一化的 float32 向量（用户/物品同维度）
- **保存/加载**：FaissItemIndex 可落盘（index.write / read_index）
- **代码完整度**：可直接运行（ready）—— 已实现暴力检索一致性校验（top_k_agreement）与性能基准
- **适配层**：FaissItemIndex

### TFRS factorized_top_k

- **ID**：`tfrs_factorized_top_k`
- **分类**：基础设施 / vendor
- **来源**：third_party/recommenders（上游 https://github.com/tensorflow/recommenders）
- **Commit**：`a8a90aee3f0b470cda96d47e26de99eedb7fc730`
- **许可证**：Apache-2.0
- **源码**：`third_party/recommenders/tensorflow_recommenders/layers/factorized_top_k.py`
- **原理**：以 Keras 层封装 BruteForce/Streaming/ScaNN 三种候选检索策略，可在 tf.data 流水线内做 Top-K。
- **适用场景**：TF 技术栈检索；作为 FAISS 的对照实现。
- **依赖 / Python**：tensorflow、tf-keras、scann（可选）；Python >=3.9
- **训练入口**：无需训练（TopK 层直接调用）
- **推理入口**：layer(query_embeddings) → (scores, identifiers)
- **数据格式**：tf.Tensor 查询/候选向量
- **保存/加载**：随 Keras 模型一起保存（layer 本身无权重）
- **代码完整度**：受依赖/权重限制（blocked）—— 依赖 TF2，与主栈 PyTorch 环境隔离
- **适配层**：尚未接入
