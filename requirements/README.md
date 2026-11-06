# 依赖分层

主环境只装 `base.txt`（+ 按需 `optional-*.txt`）。其余文件对应**互相冲突、必须独立环境**的上游栈，
不要往同一个虚拟环境里装。

| 文件 | 用途 | 环境约束 |
| --- | --- | --- |
| `base.txt` | 本项目自研代码（torch / numpy / certifi） | Python 3.11+ |
| `optional-faiss.txt` | FAISS 召回评测 | 与 base 同环境 |
| `optional-lightgbm.txt` | LightGBM 精排基线 | 与 base 同环境 |
| `dev.txt` | ruff / pytest | 与 base 同环境 |
| `legacy-bert4rec.txt` | BERT4Rec 原实现（TF1.12） | Python 3.6/3.7，**独立** |
| `legacy-mmrec.txt` | MMRec 多模态框架 | Python 3.7，**独立** |

以下三个上游栈不在 pip 层管理，需要各自按上游 README 建环境（详见 `docs/dependencies.md`）：

- `third_party/TorchEasyRec`：依赖阿里 OSS 的 Linux-only wheel（graphlearn / pyfg / common_io），Windows 装不了；
- `third_party/OpenOneRec`：Qwen3 + verl + vLLM，多卡 Linux；
- `third_party/generative-recommenders`：torch>=2.6 + fbgemm_gpu + torchrec，CUDA 编译；
- `third_party/recommenders`：TF2 环境，与 PyTorch 主栈隔离。

原则：**能用 wheel 就不编译源码**；`third_party/` 里的源码是参考实现和必要时的补丁基线，不是默认安装来源。
