# recsys-agent

面向短视频场景的离线搜索、推荐与广告实验平台。使用统一数据契约、模型适配接口和评估指标，把模型实验与受预算约束的 Agent 工具调用串起来。

项目以 Python / PyTorch 为主，采用 `src` 包布局。第三方源码保留来源与许可证；数据、模型权重、索引和运行缓存需在本地准备。

## 能做什么

| 模块 | 已实现的能力 | 入口与文档 |
| --- | --- | --- |
| 推荐 | 多路召回、融合、排序、类目/作者重排、逐阶段评估 | `python -m recsys.pipeline` · [推荐链路](docs/recsys-pipeline.md) |
| 搜索 | BM25、FAISS 向量检索、RRF 混合、精排 | `python -m recsys.search` · [搜索说明](docs/search.md) |
| 广告 | CTR/多任务模型接口、广告排序、模拟竞价与预算 | [广告系统](docs/ads.md) · [Ads Agent](docs/ads-agent.md) |
| Agent 平台 | 业务工具调用、共享实验记录、预算、文档检索、三业务演示 | `python -m recsys.platform` · [平台说明](docs/phase4-platform.md) |
| 自动研究 | 候选实验、历史复用、结果验证与停止条件 | [自动研究](docs/autoresearch.md) |
| Model Zoo | 模型索引、召回/排序适配器、来源与接入状态 | [模型清单](docs/model-zoo.md) |

## 快速开始

在仓库根目录执行（Python 3.11+，以下为 Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# 无需下载数据，先查看模型与链路入口
python -m recsys.adapters list
python -m recsys.pipeline list

# 校验模型登记、配置与项目文档
python scripts/zoo.py check
python scripts/check_project.py
```

可选依赖按模块安装，见 [依赖说明](docs/dependencies.md) 和 `requirements/`。Agent 所需配置以 [.env.example](.env.example) 为模板；密钥仅写入本地 `.env`。

本机可跑的小样本推荐实验从 KuaiRec 开始：

```powershell
python -m recsys.data.prepare --dataset kuairec_small
python -m recsys.training.train_mf --dataset kuairec_small
```

完整训练与评测协议见 [KuaiRec 小样本说明](docs/kuairec-small.md)。KuaiSAR 数据处理、搜索索引、可选模型和平台演示的前置条件及命令见 [运行命令参考](docs/usage.md)。平台演示会加载已有工件，缺少权重或索引时需要先准备。

## 结果与边界

- 当前以离线实验和小样本链路验证为主，尚不能据此宣称线上收益或生产可用性。
- 搜索默认使用 KuaiSAR token ID；无训练 token 向量链路用于验证接口，语义理解能力需单独评测。
- 广告竞价使用模拟金额；部分模型只做固定初始化权重的接口验证，训练状态见 [广告说明](docs/ads.md)。
- Agent 能调用工具不等于效果更优。历史实验存在排序退化、召回瓶颈和种子方差，见 [Top-20 诊断](docs/top20-diagnosis.md) 与 [排序消融报告](experiments/2026-11-10-effect-study/report.md)。
- 验证集选型与测试集评估必须分开；已查看过的测试切分不再作为新的盲测证据。实验结果需连同数据、切分、候选集和种子一起解释。

## 仓库结构

```text
src/recsys/   数据、模型、训练、推荐/搜索/广告与 Agent 平台
configs/      数据集、模型、链路和实验配置
scripts/      一致性检查、诊断与冒烟验证
models/       Model Zoo 登记与分类索引
docs/         架构、数据契约、运行指南与模块说明
experiments/  实验脚本、配置、报告与汇总指标
third_party/  上游源码、来源版本和许可证
data/         本地数据（不随 Git 分发）
artifacts/    本地权重与索引（不随 Git 分发）
```

实验报告中引用的逐请求轨迹、缓存和大体积工件保留在本地。个人复习知识库与求职材料也不随仓库分发。

## 开发与文档

- [文档导航](docs/README.md) · [架构](docs/architecture.md) · [数据契约](docs/data-contract.md)
- [Rec Agent](docs/rec-agent.md) · [Search Agent](docs/search-agent.md) · [集成演示](docs/phase5-integration.md)
- [项目规划](PROJECT_MASTER_v2.md) · [进度记录](PROJECT_PROGRESS.md)（历史记录按各自日期解读）
- [第三方来源](third_party/SOURCES.toml) · [许可证说明](third_party/LICENSES.md)

模型登记以 `models/zoo.toml` 为准，修改后运行 `python scripts/zoo.py render` 同步生成文档。上游改动放入 `patches/` 并登记。指标统一在 `recsys.evaluation.metrics` 实现，同一数据协议下比较结果。
