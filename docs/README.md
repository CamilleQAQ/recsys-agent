# 项目文档

按「先看结论，再看细节」的顺序读：

0. [../PROJECT_MASTER.md](../PROJECT_MASTER.md) —— 项目总档案：定位、架构、13 个 Phase、记录规范（**大纲，不随进度改动**）；
1. [../PROJECT_PROGRESS.md](../PROJECT_PROGRESS.md) —— 进度台账：做完了什么、卡在哪、下一步做什么（**每轮更新这一份**）；
3. [../README.md](../README.md) —— 现状、目录、快速开始；
4. [architecture.md](architecture.md) —— 代码分层与数据流，改代码前先看这张图；
5. [data-contract.md](data-contract.md) —— 数据契约与评估口径，接新数据集/新模型必看；
6. [model-zoo.md](model-zoo.md) —— 26 个模型的索引（**生成物**，改 `models/zoo.toml` 后重新渲染）；
7. [models/](models/README.md) —— 已接入模型的技术档案（**生成物**，PROJECT_MASTER §9.2）；
8. [recsys-pipeline.md](recsys-pipeline.md) —— Phase 1 v2 统一推荐链路：配置、阶段、工件；
9. [rec-agent.md](rec-agent.md) —— Phase 1 v2 Rec Agent：工具、LangGraph、预算与失败语义；
10. [top20-diagnosis.md](top20-diagnosis.md) —— Top-20 零命中固定样本诊断；
11. [../experiments/2026-11-09-kuasar_small-phase1-v2/report.md](../experiments/2026-11-09-kuasar_small-phase1-v2/report.md) —— Phase 1 v2 总结报告（模型/链路/Agent/检查/阻碍）；
12. [../experiments/2026-11-09-kuasar_small-fix-three/report.md](../experiments/2026-11-09-kuasar_small-fix-three/report.md) —— 三个已确认问题修复记录；
13. [dependencies.md](dependencies.md) —— 哪套环境装什么、为什么必须隔离；
14. [kuairec.md](kuairec.md) —— KuaiRec 流程模板：口径、命令、坑；
15. [kuasar-roadmap.md](kuasar-roadmap.md) —— Phase 1：接入 KuaiSAR 的工作清单；
16. [kuasar.md](kuasar.md) —— KuaiSAR 字段说明、文件清单与样本统计（数据说明）；
17. [search.md](search.md) —— Search：keyword/caption 词表核验、小样本 BM25，以及 Phase 2 的 FAISS 向量 / RRF 混合 / 精排；
18. [search-agent.md](search-agent.md) —— Search Agent：BM25 / 向量 / 混合 / 精排四工具、查询规则、预算、落盘边界与验收；
19. [../experiments/2026-11-09-kuasar-search-phase2/report.md](../experiments/2026-11-09-kuasar-search-phase2/report.md) —— Phase 2 搜索系统小样本验收报告（复用组件、检查结果、限制）；
20. [cleanup-report.md](cleanup-report.md) —— 仓库整理与清理账目；
21. [ads.md](ads.md) —— Phase 3 广告系统：样本/标签、六模型、排序、二价竞价与预算、验收状态；
22. [ads-agent.md](ads-agent.md) —— Ads Agent：YieldAgent 复用对照、7 个工具、预算与轨迹、本地/真实 LLM 运行；
23. [../experiments/2026-11-09-kuasar_small-ads-phase3/report.md](../experiments/2026-11-09-kuasar_small-ads-phase3/report.md) —— Phase 3 总结报告（模型状态、工具轨迹、模拟输出、阻碍）；
24. [phase4-platform.md](phase4-platform.md) —— Phase 4 统一 Agent 平台：Research/Master 决策、共享实验、统一预算、RAG 与验收记录。
25. [phase5-integration.md](phase5-integration.md) —— Phase 5 全项目集成：统一入口、三业务演示、工件清单与独立进程恢复、
    只加载不训练（LightGBM）、统一日志与缺工件失败用例、本次运行证据。
26. [kuairec-small.md](kuairec-small.md) —— `kuairec_small` 正式小样本线：稠密矩阵事实、与 `kuairec` 相同的口径、
    建库/训练/收口命令，以及四个必须知道的坑（缺时间戳、去重 no-op、候选集规模、词表 3,317 ≠ 目录 3,327）。

## 运行与研究

- [运行命令参考](usage.md)：数据准备、训练、三业务入口和本地验收。
- [自动研究](autoresearch.md)：预算内的候选实验、历史复用与效果边界。

## 写文档的规矩

- 只写**能用起来**的信息：命令、路径、约束、坑；不重复代码注释里已有的细节。
- 涉及数字（阈值、体积、commit）必须能追溯到代码常量或 `third_party/SOURCES.toml`。
- `model-zoo.md`、`models/<category>/README.md`、`docs/models/*.md` 由 `scripts/zoo.py render` 生成，
  手改会在下次渲染时被覆盖；要改内容请改 `models/zoo.toml`。
- 进度只写进 `PROJECT_PROGRESS.md`（一份台账，按日期在「变更历史」追加）；架构级变化回写 `PROJECT_MASTER.md`；
  单个模型的技术细节写进 `docs/models/<id>.md`（由 zoo.toml 生成）。
- 新增/移动文件后跑一次 `python scripts/check_project.py`，它会检查文档里的相对链接是否还有效。
