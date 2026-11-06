# 项目文档

按「先看结论，再看细节」的顺序读：

0. [../PROJECT_MASTER.md](../PROJECT_MASTER.md) —— 项目总档案：定位、架构、13 个 Phase、记录规范（**大纲，不随进度改动**）；
1. [../PROJECT_PROGRESS.md](../PROJECT_PROGRESS.md) —— 进度台账：做完了什么、卡在哪、下一步做什么（**每轮更新这一份**）；
2. [../README.md](../README.md) —— 现状、目录、快速开始；
3. [architecture.md](architecture.md) —— 代码分层与数据流，改代码前先看这张图；
4. [data-contract.md](data-contract.md) —— 数据契约与评估口径，接新数据集/新模型必看；
5. [model-zoo.md](model-zoo.md) —— 24 个模型的索引（**生成物**，改 `models/zoo.toml` 后重新渲染）；
6. [models/](models/README.md) —— 已接入模型的技术档案（**生成物**，PROJECT_MASTER §9.2）；
7. [dependencies.md](dependencies.md) —— 哪套环境装什么、为什么必须隔离；
8. [kuairec.md](kuairec.md) —— KuaiRec 流程模板：口径、命令、坑；
9. [kuasar-roadmap.md](kuasar-roadmap.md) —— Phase 1：接入 KuaiSAR 的工作清单；
10. [cleanup-report.md](cleanup-report.md) —— 仓库整理与清理账目。

## 写文档的规矩

- 只写**能用起来**的信息：命令、路径、约束、坑；不重复代码注释里已有的细节。
- 涉及数字（阈值、体积、commit）必须能追溯到代码常量或 `third_party/SOURCES.toml`。
- `model-zoo.md`、`models/<category>/README.md`、`docs/models/*.md` 由 `scripts/zoo.py render` 生成，
  手改会在下次渲染时被覆盖；要改内容请改 `models/zoo.toml`。
- 进度只写进 `PROJECT_PROGRESS.md`（一份台账，按日期在「变更历史」追加）；架构级变化回写 `PROJECT_MASTER.md`；
  单个模型的技术细节写进 `docs/models/<id>.md`（由 zoo.toml 生成）。
- 新增/移动文件后跑一次 `python scripts/check_project.py`，它会检查文档里的相对链接是否还有效。
