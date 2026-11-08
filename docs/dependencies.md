# 依赖管理方案

## 原则

1. **一个主环境，多个隔离环境**。PyTorch 主栈（本项目自研 + LightGBM + FAISS）装在一个环境；
   TF1.12、TF2、TorchEasyRec、OpenOneRec、genrec、MMRec、FuxiCTR 各自独立，绝不混装。
2. **能用 wheel 就别编译源码**。`third_party/` 里的源码是参考实现与补丁基线，不是默认安装来源。
3. **依赖分三层记录**：`pyproject.toml`（包本身需要什么）、`requirements/*.txt`（环境怎么装）、
   `models/zoo.toml` 的 `deps` 字段（每个模型额外需要什么）。

## 环境矩阵

| 环境 | Python | 关键包 | 能跑什么 | 装法 |
| --- | --- | --- | --- | --- |
| **A 主环境** | 3.11–3.12 | torch>=2.0, numpy>=1.24, certifi, faiss-cpu, lightgbm | 自研 4 个模型、FAISS 评测、LightGBM 排序基线 | `pip install -e .` + `requirements/optional-*.txt` |
| B TF1 遗留 | 3.6–3.7 | tensorflow-gpu==1.12 | BERT4Rec 原实现（仅参考） | `requirements/legacy-bert4rec.txt` |
| C TF2 | 3.9–3.11 | tensorflow>=2.9, tf-keras | TFRS（recommenders） | 上游 README |
| D MMRec | 3.7 | numpy==1.21.5, torch==1.11 | MMRec 多模态模型 | `requirements/legacy-mmrec.txt` |
| E CTR（FuxiCTR） | 3.10+ | fuxictr, pandas, pyarrow, ray[data]>=2.52, polars, pydantic, h5py | DIN/DIEN/DCNv2/DeepFM/MMoE | `pip install fuxictr`，数据转 parquet |
| F Linux GPU（TorchEasyRec） | 3.10–3.12 | tzrec（阿里 OSS wheel）、CUDA | DSSM/MIND/HSTU/语义 ID | 上游 README，**Windows 不可用** |
| G Linux 多卡 | 3.10+ | torch>=2.6, verl, vllm, ray | OpenOneRec、generative-recommenders | 上游 README，需外部权重 |
| H 生成式（genrec） | 3.9+ | torch==2.6, transformers==4.57, gin_config, wandb, sentence_transformers | TIGER/RPG/COBRA/LC-Rec | 上游 README，需物品 embedding |
| **I torch-rechub** | 3.10+ | torch>=2.6, numpy<3, pandas, scikit-learn, tqdm | **MIND**（本机已跑通）；DIN/DIEN/DCNv2/DSSM 等 30+ 模型可按需接入 | `pip install torch-rechub==1.0.0` + `requirements/optional-rechub.txt` |
| **J Agent（LangGraph）** | 3.10+ | langgraph==1.2.20, langchain-core | Phase 11 工具调用工作流（`llm_tool_call`） | `requirements/optional-agent.txt`；不装时入口报缺依赖 |

## 版本冲突与处理

| 冲突 | 现象 | 处理 |
| --- | --- | --- |
| numpy 1.x vs 2.x | MMRec/RecBole 老版本依赖 numpy1.x；主环境用 2.x | 分环境；主环境不再降级 |
| torch 1.11 vs 2.x | MMRec 固定 torch==1.11 | 分环境；MMRec 只作算法参考 |
| torch==2.6 固定 | genrec 的 requirements 精确锁定 torch==2.6.0 | genrec 单独环境；不与主环境共享 |
| TF1.12 vs 现代 Python | `tf.contrib`、旧 TFRecord API 已删除 | BERT4Rec 固定 Python 3.6/3.7 |
| TensorFlow vs PyTorch | 体积大、CUDA/cuDNN 互相牵制 | TF 相关（recommenders、BERT4Rec）不进主环境 |
| ray 版本 | RecBole 要求 `ray<=2.6.3`；FuxiCTR 要求 `ray[data]>=2.52` | 二者独立环境；同时需要时先确认可兼容区间 |
| 阿里 OSS wheel | TorchEasyRec 依赖 graphlearn/pyfg/common_io 的 Linux wheel | Windows 无法安装；只在 Linux 构建机跑 |
| pandas 3 vs torch-rechub | torch-rechub 1.0.0 声明 `pandas<3`，但 Python 3.14 只有 pandas 3.1.1 的 wheel | 实测导入/MIND 前向正常，仅 pip 告警；若要用它的 `MatchTrainer` 再评估是否降级 |
| faiss-cpu 与 numpy | 部分 faiss-cpu wheel 对 numpy 2.x 支持滞后 | 按 `requirements/optional-faiss.txt`；失败时先降 numpy 再装 faiss |

## 本机实测（2026-11-05）

- Python 3.14.3 + torch 2.14.0+cpu + numpy 2.5.3 + faiss-cpu：可 import，
  适配层冒烟测试全过；KuaiSAR 下载 + md5 校验 + 全量画像（3.5 GB）跑通；
- 未安装：lightgbm、tensorflow、fuxictr、recbole、tzrec、genrec、verl/vllm
  （对应模型在 `zoo.toml` 中标为 `upstream_ready` 或 `blocked`）；
- 结论：主环境声明 `requires-python = ">=3.11"`，实测在 3.14 工作正常；上游更窄的限制只在各自环境里成立。

## 为什么不用 `requirements.txt` 一把梭

- 13 个仓库的依赖不可能收敛成一份：TF1.12 与 torch 2.x 无法共存，numpy 1.x/2.x 只能选一边；
- genrec 精确锁定 torch==2.6，和主环境的 torch 2.x 直接冲突；
- 精确锁定会锁死 CUDA 版本，换机器就装不上；
- 因此主环境用**下界约束 + pyproject extras**，需要复现某次实验时再 `pip freeze > requirements/lock-<date>.txt`。

## 新增依赖的规矩

1. 先判断能不能落在主环境：与本项目数据契约直接相关 → 主环境；只是某个上游模型的依赖 → 独立环境；
2. 主环境新增依赖必须同时更新 `pyproject.toml`；
3. 在 `models/zoo.toml` 对应模型的 `deps` 字段补一行，`python scripts/zoo.py check` 保证条目与路径一致；
4. 不引入只为「读配置文件」服务的重量级依赖（YAML 已用 TOML 替代）。
