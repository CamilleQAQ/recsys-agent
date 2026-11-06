# 静态代码审计报告

本轮仅独立浅克隆 11 个指定仓库（depth=1，single-branch，保留 .git 和原目录结构）。未安装依赖、运行第三方代码/安装脚本、初始化子模块或主动下载数据/权重。LFS smudge 已禁用。Git 跟踪的实验数据随仓库原样取得，详见数据体积清单。未增补或修改第三方文件。

## 结果与边界

11/11 克隆成功；最终失败清单为空。最初沙箱 Git TLS 凭据错误，经授权重试成功。默认分支由克隆时 origin/HEAD 记录，commit 为完整 SHA。浅克隆不包含全部历史、其他分支或 tag；未初始化的子模块会影响编译完整性。许可证为源码初筛，非法律结论；未验证外部权重服务、数据链接、wheel URL 或全部传递依赖的可用性。

LightGBM 原 URL 经 GitHub 官方重定向至 [lightgbm-org/LightGBM](https://github.com/lightgbm-org/LightGBM)，不是自行选择替代仓库。OpenOneRec 根 LICENSE 缺失；README 声明 Apache 2.0，子目录许可证分别保留，权重另有许可。无其他候选仓库被下载。

## RecBole

- 仓库：https://github.com/RUCAIBox/RecBole；默认分支 `master`；commit `7b02be5ec80a88310f2d04a27a82adfcbb5dc211`。
- 来源身份：框架官方；各论文模型为框架复现。
- 许可证：MIT；文件：`LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 存在；`environment.yml` 缺失；`setup.py` 存在；`pyproject.toml` 缺失。
- 依赖/构建文件：`requirements.txt`、`setup.py`。
- 实际模型覆盖：协同过滤、图推荐、序列、CTR、知识图谱；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：run_recbole.py; recbole/quick_start/quick_start.py。
- 预测/评估：recbole/trainer/trainer.py: fit/evaluate; 各模型 predict/full_sort_predict。
- 数据格式：带类型表头的制表符 atomic 文件 .inter/.user/.item；知识模型另需 .kg/.link。
- 依赖与兼容性：torch>=1.10、numpy、scipy、pandas、ray<=2.6.3；README 的 torch>=1.7 与依赖文件不一致，以实际依赖为准。setup.py 与 requirements 的 colorlog/colorama 约束不同。宽松 NumPy/PyTorch 上界不能保证现代版本兼容。
- 外部资源/完整性：数据通常来自 RecDatasets，部分路径可自动下载；预训练 embedding 是可选输入。默认实验数据不代表所有数据已具备。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：无 Git 子模块。
- 原始工作树：clean；已跟踪文件约 22.7 MiB（不含 .git）。
- 最大已跟踪文件：`recbole/dataset_example/ml-100k/ml-100k.kg` (3.44 MiB); `dataset/ml-100k/ml-100k.inter` (1.98 MiB); `recbole/dataset_example/ml-100k/ml-100k.inter` (1.98 MiB)。

## TorchEasyRec

- 仓库：https://github.com/alibaba/TorchEasyRec；默认分支 `master`；commit `2815a9732bed014219ee89826dd4987b67ccf6b0`。
- 来源身份：Alibaba 官方框架；论文模型为框架实现。
- 许可证：Apache-2.0；文件：`LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 存在；`environment.yml` 缺失；`setup.py` 存在；`pyproject.toml` 缺失。
- 依赖/构建文件：`requirements-cpu.txt`、`requirements-cu126.txt`、`requirements-cu130.txt`、`requirements-gpu.txt`、`requirements-ppu.txt`、`requirements.txt`、`setup.cfg`、`setup.py`。
- 实际模型覆盖：双塔召回、CTR、多任务、HSTU、生成式、语义ID；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：tzrec/train_eval.py。
- 预测/评估：tzrec/predict.py; tzrec/eval.py; tzrec/export.py。
- 数据格式：CSV/Parquet/ODPS/Kafka；pipeline protobuf 配置声明字段、标签、特征与模型。
- 依赖与兼容性：setup.py 声明 Python>=3.8，但当前 README/二进制 wheel 面向 3.10–3.12。runtime 要求 numpy>=2、transformers>=4.56；cpu/cu130 文件锁 torch==2.14.0，需后续核验发行渠道可用性。大量 OSS 直接 URL 和 Linux wheel，当前 Windows 不可直接假定兼容。
- 外部资源/完整性：样例可能依赖 OSS/ODPS；预训练 embedding/LLM 按配置外置。未访问云数据或执行安装。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：无 Git 子模块。
- 原始工作树：clean；已跟踪文件约 21.0 MiB（不含 .git）。
- 最大已跟踪文件：`docs/images/models/dlrm_hstu.png` (3.65 MiB); `data/test/tokenizer.json` (2.11 MiB); `docs/images/models/genrec.png` (1.56 MiB)。

## generative-recommenders

- 仓库：https://github.com/meta-recsys/generative-recommenders；默认分支 `main`；commit `b8e6ff07c5d4c3354b0807ead76b74d771557fd7`。
- 来源身份：Meta 官方研究实现。
- 许可证：Apache-2.0；文件：`LICENSE`、`generative_recommenders/research/cppilot/LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 存在；`environment.yml` 缺失；`setup.py` 存在；`pyproject.toml` 缺失。
- 依赖/构建文件：`generative_recommenders/dlrm_v3/inference/thirdparty/loadgen/pyproject.toml`、`generative_recommenders/dlrm_v3/inference/thirdparty/loadgen/requirements.txt`、`generative_recommenders/dlrm_v3/inference/thirdparty/loadgen/setup.py`、`generative_recommenders/ops/cpp/setup.py`、`generative_recommenders/research/cppilot/pyproject.toml`、`requirements.txt`、`setup.py`。
- 实际模型覆盖：HSTU、SASRec基线、DLRMv3/HSTU、ULTRA算子、MoL检索；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：main.py; generative_recommenders/research/trainer/train.py; generative_recommenders/dlrm_v3/train/train_ranker.py。
- 预测/评估：generative_recommenders/research/data/eval.py; generative_recommenders/dlrm_v3/inference/main.py。
- 数据格式：预处理的用户交互序列/时间戳/评分；Gin 配置；DLRMv3 特征/jagged tensor 路径另行处理。
- 依赖与兼容性：Python>=3.10、torch>=2.6、fbgemm_gpu>=1.1、torchrec>=1.1；README 测试 Ubuntu22.04/CUDA12.4。CUDA/Triton/C++ 扩展和 GPU 架构要求需独立环境；不能据此保证 Windows 运行。
- 外部资源/完整性：preprocess_public_data.py 涉及公开数据下载；fractal expansion 可生成巨量数据。CUTLASS 子模块未初始化，优化 CUDA 构建不完整；基础研究路径与优化路径分开判断。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：generative_recommenders/ops/cpp/cutlass。
- 原始工作树：clean；已跟踪文件约 4.0 MiB（不含 .git）。
- 最大已跟踪文件：`generative_recommenders/dlrm_v3/inference/thirdparty/loadgen/tools/mlperf-trace.ipynb` (0.15 MiB); `generative_recommenders/ops/cpp/hstu_attention/mainloop_bwd_sm90_tma_gmma_ws.h` (0.12 MiB); `generative_recommenders/dlrm_v3/inference/thirdparty/loadgen/docs/src/doxygen.cfg` (0.11 MiB)。

## OpenOneRec

- 仓库：https://github.com/Kuaishou-OneRec/OpenOneRec；默认分支 `main`；commit `a969edcadd579a06c1966ae1db5984e02f48beff`。
- 来源身份：Kuaishou-OneRec 官方项目；包含 Qwen3/VeRL 派生代码。
- 许可证：Apache-2.0 (README声明；根LICENSE缺失)；文件：`benchmarks/LICENSE`、`verl_distillation/LICENSE`、`verl_rl/LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 缺失；`requirements.txt` 缺失；`environment.yml` 缺失；`setup.py` 缺失；`pyproject.toml` 缺失。
- 依赖/构建文件：`benchmarks/pyproject.toml`、`benchmarks/requirements.txt`、`verl_distillation/docs/requirements-docs.txt`、`verl_distillation/pyproject.toml`、`verl_distillation/requirements-cuda.txt`、`verl_distillation/requirements-npu.txt`、`verl_distillation/requirements.txt`、`verl_distillation/requirements_sglang.txt`、`verl_distillation/requirements_transferqueue.txt`、`verl_distillation/setup.py`、`verl_rl/docs/requirements-docs.txt`、`verl_rl/pyproject.toml`、`verl_rl/requirements-npu.txt`、`verl_rl/requirements.txt`、`verl_rl/requirements_sglang.txt`、`verl_rl/setup.py`。
- 实际模型覆盖：Qwen3/OneRec生成式推荐、Residual K-Means tokenizer、SFT/蒸馏/GRPO流程；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：pretrain/recipes/train_qwen3.py; tokenizer/train_res_kmeans.py; verl_rl/recipe/onerec/run_grpo.sh。
- 预测/评估：tokenizer/infer_res_kmeans.py; benchmarks/scripts/ray-vllm/evaluate.py; benchmarks/benchmark/tasks/v1_0/。
- 数据格式：Parquet chat/messages 或 segments、itemic tokens；tokenizer 输入 embedding，推断另需 pid；参考 data/README.md。
- 依赖与兼容性：无根 requirements/setup/pyproject；依赖分散于 benchmarks、verl_rl、verl_distillation 与 pretrain README。Qwen3、Transformers、PyTorch FSDP、MPI/NCCL、Ray/vLLM 等；各模块版本需分别锁定，多卡 Linux 为主要路径。
- 外部资源/完整性：需要外部 Qwen3/OneRec 权重、Qwen3-Embedding-8B、tokenizer 及 RecIF 数据；未下载。Pro 使用的工业语料不能视为公开完整复现资源。根 README 的 coming soon 与已发布子模块文档不一致。根 LICENSE 缺失，子目录许可证不能自动覆盖全仓库，适配/再分发前应确认。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：无 Git 子模块。
- 原始工作树：clean；已跟踪文件约 15.7 MiB（不含 .git）。
- 最大已跟踪文件：`assets/main_framework.png` (1.58 MiB); `assets/framework.png` (0.55 MiB); `assets/benchmark.png` (0.49 MiB)。

## MMRec

- 仓库：https://github.com/enoche/MMRec；默认分支 `master`；commit `c68ee7c9b0de1ca7c5150258322edb4994be29d9`。
- 来源身份：MMRec 框架原仓库；论文模型包含作者实现及框架复现，不能统称各模型官方。
- 许可证：GPL-3.0 (见原LICENSE)；文件：`LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 存在；`environment.yml` 缺失；`setup.py` 缺失；`pyproject.toml` 缺失。
- 依赖/构建文件：`requirements.txt`。
- 实际模型覆盖：多模态图推荐、协同过滤、内容KNN；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：src/main.py; src/utils/quick_start.py。
- 预测/评估：src/common/trainer.py; src/utils/topk_evaluator.py; 模型 full_sort_predict。
- 数据格式：配置指定交互表 uid/iid/split 字段；视觉/文本特征及 ID 对齐；部分方法需用户邻接矩阵。
- 依赖与兼容性：requirements 固定 numpy1.21.5/pandas1.3.5/scipy1.7.3/torch1.11；python==3.7.11 是环境说明而非有效的常规 pip Python 安装方式。代码另有 torchvision/PIL/torch_geometric/torch_scatter/sklearn 等导入，根依赖清单不能视为完整。老版本与现代 Python/NumPy 不宜混装。
- 外部资源/完整性：data/README.md 指向 Google Drive 的 Baby/Sports/Elec 特征数据及 MicroLens；未下载。多模态预提取特征不是随仓库附带的模型权重。GPL 适配与分发边界需保留。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：无 Git 子模块。
- 原始工作树：clean；已跟踪文件约 0.9 MiB（不含 .git）。
- 最大已跟踪文件：`reports/MMRec-reports.pdf` (0.22 MiB); `images/MMRec.png` (0.10 MiB); `preprocessing/3feat-encoder.ipynb` (0.04 MiB)。

## DeepCTR-Torch

- 仓库：https://github.com/shenweichen/DeepCTR-Torch；默认分支 `master`；commit `9eccef78e5810c61d6a04ddc7d96279e3db9c970`。
- 来源身份：DeepCTR 作者维护的 PyTorch 库；论文模型为库内复现。
- 许可证：Apache-2.0；文件：`LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 缺失；`environment.yml` 缺失；`setup.py` 存在；`pyproject.toml` 缺失。
- 依赖/构建文件：`docs/requirements.readthedocs.txt`、`setup.py`。
- 实际模型覆盖：CTR特征交叉、DIN/DIEN、多任务；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：examples/ 下样例调用模型 fit。
- 预测/评估：deepctr_torch/models/basemodel.py: fit/predict/evaluate。
- 数据格式：特征名到数组的字典；SparseFeat/DenseFeat/VarLenSparseFeat，标签数组及序列长度/负采样字段。
- 依赖与兼容性：Python>=3.7；setup.py 按 Python<3.8 用 torch>=1.13，否则 torch>=2.4；tqdm/scikit-learn。样例的数据处理还需 pandas 等，setup 不等于全部样例依赖。
- 外部资源/完整性：示例 CTR/MovieLens 数据按样例准备；无强制通用预训练模型。README 的 FNN 不能单独计为现有模型模块：未发现 models/fnn.py。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：无 Git 子模块。
- 原始工作树：clean；已跟踪文件约 5.5 MiB（不含 .git）。
- 最大已跟踪文件：`docs/pics/mlr1.png` (0.52 MiB); `docs/pics/ONN.png` (0.38 MiB); `docs/pics/mlrvsdnn.png` (0.37 MiB)。

## LightGBM

- 仓库：https://github.com/lightgbm-org/LightGBM；默认分支 `main`；commit `3ee03c26ba8c9e8ddc05fa8687717cae8a7d61d9`。
- 来源身份：官方仓库已迁移至 lightgbm-org/LightGBM；原URL重定向。
- 许可证：MIT；文件：`LICENSE`、`R-package/LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 缺失；`environment.yml` 缺失；`setup.py` 缺失；`pyproject.toml` 缺失。
- 依赖/构建文件：`.ci/pip-envs/requirements-latest.txt`、`.ci/pip-envs/requirements-oldest.txt`、`.ci/pip-envs/requirements-test.txt`、`.ci/pip-envs/requirements-windows-arm64.txt`、`python-package/pyproject.toml`。
- 实际模型覆盖：GBDT/DART/GOSS/RF；LambdaRank、rank_xendcg排序目标；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：src/main.cpp; python-package/lightgbm/engine.py: train。
- 预测/评估：python-package/lightgbm/sklearn.py: LGBMRanker.fit; basic.py: Booster.predict。
- 数据格式：稠密/稀疏特征矩阵及 label；排序必须提供连续查询分组 group/query，非用户序列推荐器。
- 依赖与兼容性：当前 python-package/pyproject.toml 要求 Python>=3.11、narwhals>=2.26、numpy>=1.21.3、scipy；scikit-build-core/CMake/C++，可选 OpenMP/GPU。根无 Python setup.py 并非缺代码。
- 外部资源/完整性：5 个构建子模块未初始化，源码编译前需单独核验并获取；examples 包含少量实验数据。树模型通常自行训练，无预训练权重必需项。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：external_libs/compute, external_libs/eigen, external_libs/fast_double_parser, external_libs/fmt, external_libs/nanoarrow。
- 原始工作树：clean；已跟踪文件约 19.7 MiB（不含 .git）。
- 最大已跟踪文件：`examples/lambdarank/rank.train` (2.36 MiB); `examples/xendcg/rank.train` (2.36 MiB); `examples/multiclass_classification/multiclass.train` (1.23 MiB)。

## faiss

- 仓库：https://github.com/facebookresearch/faiss；默认分支 `main`；commit `9acf8b0c16a51aafa12b9d9bcf13998b9138a8c5`。
- 来源身份：Meta/FAIR 官方向量检索库。
- 许可证：MIT；文件：`LICENSE`、`THIRD_PARTY_NOTICES`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 缺失；`environment.yml` 缺失；`setup.py` 缺失；`pyproject.toml` 存在。
- 依赖/构建文件：`faiss/python/setup.py`、`pyproject-gpu-cuvs.toml`、`pyproject-gpu.toml`、`pyproject.toml`。
- 实际模型覆盖：Flat/IVF/PQ/HNSW/LSH、聚类；非端到端推荐模型；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：Index.train/add; tutorials/python/ 下索引示例。
- 预测/评估：Index.search; contrib/evaluation.py。
- 数据格式：连续 float32 的 N×d 向量和查询；部分二进制索引用 uint8，须由上游推荐模型生成向量。
- 依赖与兼容性：pyproject: Python>=3.10，运行 numpy>=1.25/packaging；构建 numpy>=2、scikit-build-core、SWIG>=4.2<5、CMake、BLAS/OpenMP。GPU 路径另需 CUDA，CPU 默认配置关闭 GPU。
- 外部资源/完整性：不附带推荐 embedding 或成品业务索引；benchmarks 的外部向量集不在本轮获取范围。THIRD_PARTY_NOTICES 需一起保留。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：无 Git 子模块。
- 原始工作树：clean；已跟踪文件约 11.3 MiB（不含 .git）。
- 最大已跟踪文件：`tests/torch_test_e_means.py` (0.16 MiB); `faiss/python/__init__.pyi` (0.15 MiB); `faiss/impl/index_read.cpp` (0.15 MiB)。

## SASRec.pytorch

- 仓库：https://github.com/pmixer/SASRec.pytorch；默认分支 `main`；commit `26a8fa880f312c3700f8dcca5512d5854b257f4e`。
- 来源身份：第三方复现：Zan Huang/pmixer，基于 kang205/SASRec 的论文作者 TensorFlow 实现。
- 许可证：Apache-2.0；文件：`LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 缺失；`environment.yml` 缺失；`setup.py` 缺失；`pyproject.toml` 缺失。
- 依赖/构建文件：未发现标准 Python 依赖或打包文件。
- 实际模型覆盖：SASRec；附Rust实现路径，非另一个独立模型；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：python/main.py (工作目录 python/)。
- 预测/评估：python/model.py: SASRec.predict; python/utils.py: evaluate/evaluate_valid; main inference_only。
- 数据格式：python/data/<dataset>.txt；每行 user_id item_id，按用户交互顺序，整数 ID；leave-last 验证/测试和负采样。
- 依赖与兼容性：无 requirements/environment/setup/pyproject；代码导入 torch/numpy，README 历史提及 PyTorch1.6，无完整锁定环境。附 rust/Cargo.toml 独立依赖未编译。
- 外部资源/完整性：Git 已跟踪多份实验文本数据；推断 state_dict_path 需自行训练或另备。README 记录 LayerNorm 更新，不能把当前实现直接当作原论文逐位复现。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：无 Git 子模块。
- 原始工作树：clean；已跟踪文件约 63.2 MiB（不含 .git）。
- 最大已跟踪文件：`python/data/Steam.txt` (44.09 MiB); `python/data/ml-1m.txt` (9.59 MiB); `python/data/Beauty.txt` (4.70 MiB)。

## BERT4Rec

- 仓库：https://github.com/FeiSun/BERT4Rec；默认分支 `master`；commit `615eaf2004abecda487a38d5b0c72f3dcfcae5b3`。
- 来源身份：论文作者 Fei Sun 原始实现；基于BERT架构代码。
- 许可证：Apache-2.0；文件：`LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 缺失；`environment.yml` 缺失；`setup.py` 缺失；`pyproject.toml` 缺失。
- 依赖/构建文件：未发现标准 Python 依赖或打包文件。
- 实际模型覆盖：BERT4Rec 双向序列推荐；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：gen_data_fin.py; run.py; run_ml-1m.sh 等。
- 预测/评估：run.py: model_fn_builder、EvalHooks、Estimator.train/evaluate；无现代统一 predict API。
- 数据格式：用户物品有序文本转 masked TFRecord；附 vocab/history pickle 与 bert_train JSON 配置。
- 依赖与兼容性：无 requirements/environment/setup/pyproject；README Python2.7+、TensorFlow1.12 GPU/CUDA；源码使用 tf.contrib/tf.layers/Estimator/旧 TFRecord API。不能与现代 TF2/TFRS 环境直接合并。six/numpy 等依赖需后续补锁定。
- 外部资源/完整性：已跟踪 beauty/steam/ml-1m 文本及 ml-20m.zip；无预训练 checkpoint。训练从任务数据构造，不应误认为下载通用BERT权重即能预测。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：无 Git 子模块。
- 原始工作树：clean；已跟踪文件约 104.1 MiB（不含 .git）。
- 最大已跟踪文件：`data/ml-20m.zip` (50.49 MiB); `data/steam.txt` (39.78 MiB); `data/ml-1m.txt` (9.59 MiB)。

## recommenders

- 仓库：https://github.com/tensorflow/recommenders；默认分支 `main`；commit `a8a90aee3f0b470cda96d47e26de99eedb7fc730`。
- 来源身份：TensorFlow 官方 TFRS 库。
- 许可证：Apache-2.0；文件：`LICENSE`。
- 根文件检查：`README.md` 存在；`LICENSE` 存在；`requirements.txt` 存在；`environment.yml` 缺失；`setup.py` 存在；`pyproject.toml` 缺失。
- 依赖/构建文件：`requirements.txt`、`setup.py`。
- 实际模型覆盖：双塔检索/排序任务、Cross/DCN层、BruteForce/Streaming/ScaNN检索接口；源码证据见 [model_mapping.md](model_mapping.md)。
- 训练入口：README 与 docs/examples/*.ipynb 使用 Keras Model.fit；无统一训练CLI。
- 预测/评估：tensorflow_recommenders/models/base.py; tasks/retrieval.py; tasks/ranking.py; layers/factorized_top_k.py。
- 数据格式：tf.data.Dataset 特征字典；query/candidate 特征与候选集合；标签/任务由调用方建模。
- 依赖与兼容性：requirements: tensorflow>=2.9（macOS 用 tensorflow-macos）、absl-py、tf-keras；无上界。ScaNN/tensorflow-ranking/tfds 为可选或样例依赖，需核对 TF/Keras/ABI 组合。
- 外部资源/完整性：样例 tfds.load 会下载 MovieLens；未运行。ScaNN 是外部后端，不能计为仓库内完整 ANN 算法；README 样例双塔架构不代表一组现成已训练模型。
- 预训练权重检查：已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式；LFS 指针 0 个。
- 未初始化子模块：无 Git 子模块。
- 原始工作树：clean；已跟踪文件约 2.2 MiB（不含 .git）。
- 最大已跟踪文件：`assets/parallel_deep_cross.png` (0.28 MiB); `assets/cross_features.gif` (0.22 MiB); `assets/feature_crossing.png` (0.20 MiB)。

## 后续统一适配建议

按 PyTorch 基线、旧 TF1、现代 TF2、工业 CUDA/分布式、C++ 检索/树排序分别隔离环境，先选小型合成数据验证接口，再决定依赖和外部资源下载。统一接口需保留 user/item ID 映射、序列顺序、候选集合、负采样策略、query group 与评估协议；采样 HR/NDCG 与全量检索指标不可直接横比。

OpenOneRec 的根许可证缺失、MMRec 的旧依赖/清单遗漏、Meta/LightGBM 的子模块，以及 TorchEasyRec 的平台 wheel 是优先处理项。静态发现不等于已复现性能或完成安全审计，未检查所有传递依赖和二进制供应链。

## 可核验证据

`audit_evidence.json` 保留依赖文件全文、模型定义行号/方法、根文件存在性、常见权重/LFS 扫描和最大文件清单。`inventory.json` 保留全部 Git 跟踪路径。`build_audit.py` 只用标准库读取文件与 Git 元数据，可重建本轮报告；人工审计说明固定于本次快照，更新代码后需重新审阅。
