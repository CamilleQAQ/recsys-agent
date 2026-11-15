# 运行命令参考

以下命令从仓库根目录执行。数据、权重、索引和逐请求运行记录均需本地生成，不随 Git 分发。
这些命令涵盖历史实验配置，并非从零一键安装脚本；请先阅读各模块的前置条件。

## 快速开始

```powershell
# 1. 环境（Python 3.11+）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements/optional-faiss.txt

# 2. 数据主线：下载 KuaiSAR → 构建 processed 存储 → 物品特征表
python -m recsys.data.kuasar          # 805.9 MB，带 md5 校验，解压 3.5 GB
python -m recsys.data.kuasar_store    # 约 36 秒，产出 processed（253 MB，24,128 可评测用户）
python -m recsys.data.kuasar_features # 约 46 秒，12 列物品特征（139 MB，含 is_ad）
python -m recsys.data.kuasar_ranking  # 约 54 秒，排序候选样本（约 250 MB，14.35M train 候选）
#   流程模板（辅助线）：python -m recsys.data.prepare --dataset kuairec

# 2b. 小样本正式线（一台机器可跑完的完整回归：1,411 用户 / 3,317 物品）
python -m recsys.data.prepare --dataset kuairec_small        # 约 40 秒，带 CRC32 + 规模校验
python -m recsys.training.train_mf     --dataset kuairec_small
python -m recsys.training.trainer      --dataset kuairec_small
python -m recsys.training.train_sasrec --dataset kuairec_small
python -m recsys.experiments.finalize_experiment --dataset kuairec_small

# 3. 训练（默认数据集已是 kuasar；4M 物品词表在 CPU/16GB 机器上很吃资源，
#    先用小维度 + 限制 batch 数验证通路，正式训练/调参/多种子留给效果专项）
python -m recsys.training.train_mf --dataset kuasar --embedding-dim 16 --epochs 1 --max-batches-per-epoch 200
python -m recsys.training.trainer  --dataset kuasar --embedding-dim 32 --output-dim 16 --epochs 1 --max-batches-per-epoch 200
#   SASRec 需要按用户构造序列，KuaiSAR 上建议先加 --max-batches-per-epoch 试跑

# 4. 统一接口评测（自动读回 checkpoint 的数据集、阈值与溯源信息）
python -m recsys.adapters list
python -m recsys.adapters evaluate --model matrix_factorization `
    --checkpoint artifacts/kuasar/matrix_factorization.pt --split test

# 5. 推荐全链路（Phase 1 v2）：双路召回 → 融合 → LightGBM → 类目重排
python -m recsys.pipeline list
python -m recsys.pipeline run --config configs/pipeline/kuasar_small_default.toml --users 16
python -m recsys.pipeline evaluate --artifacts artifacts/kuasar_small/pipeline --split validation

# 6. Rec Agent：真实工具调用（确定性计划器；加 --llm 走真实 LLM）
python -m recsys.experiments.rec_agent --self-check
python -m recsys.experiments.rec_agent `
    --task "用双塔和 MIND 召回，LightGBM 排序并评估" --users 16 `
    --recall two_tower,mind --ranker lightgbm_ranker
python -m recsys.experiments.rec_agent `
    --task "训练 DIN 并用于排序" --users 8 `
    --train din --recall two_tower --ranker din

# 7. Top-20 零命中固定样本诊断
python scripts/diagnose_top20.py --users 128 --seed 42

# 8. 搜索：小样本 BM25 + FAISS 向量 + RRF 混合 + 精排（只读原始 CSV 前缀，不扫全量）
python -m recsys.search.prepare        # 30 会话 / ≤1,000 物品，产物在 artifacts/search/
python -m recsys.search --tokens "3538762,3181091,4061567,3525929" --top-k 5
python -m recsys.search.smoke --top-k 5   # 有匹配 / 空查询 / 无匹配一次检查
python -m recsys.search.build_vector_index --force   # 搜索侧 FAISS 索引（独立于推荐索引）
python -m recsys.search --mode vector|hybrid|rerank `
    --tokens "3538762,3181091,4061567,3525929" --top-k 5
python -m recsys.search.acceptance     # Phase 2 验收（9 项）

# 9. Search Agent：四工具（BM25 / 向量 / 混合 / 精排）
python -m recsys.experiments.search_agent --self-check       # 本地 17 项，不联网
python -m recsys.experiments.search_agent `
    --regression --tokens 2                                  # 本地替身回归，不联网
python -m recsys.experiments.search_agent `
    --scripted-plan hybrid_rerank                            # 替身决策 + 真实检索/精排
python -m recsys.experiments.search_agent `
    --tokens "3538762,3181091,4061567,3525929"                # 真实 LLM

# 10. 广告系统（Phase 3）：样本 → 六模型 → 排序 → 竞价/预算 → Ads Agent
python -m recsys.ads.data                 # 请求内 is_ad 候选 + click 标签 → processed/ads/
python -m recsys.ads.simulated            # 独立 synthetic 转化样本（仅用于多任务接口验证）
python -m recsys.ads.verify               # 六模型固定初始化权重 + 推理 + 保存/加载
python -m recsys.ads.predict --model din --split test
python -m recsys.ads.pipeline --model din --split train --requests 3 --min-candidates 2 --budget-check
python -m recsys.ads.agent --planner local
python -m recsys.ads.agent --self-check
python scripts/smoke_phase3_ads.py        # Phase 3 总验收

# 11. Phase 4：Research/Master Agent + 共享实验 + 统一预算 + RAG
python -m recsys.platform --self-check
python -m recsys.platform --task "查一下已有 30/70 融合实验，能复用就复用"
python -m recsys.platform --task "推荐和搜索一起做" --planner scripted
python -m recsys.platform --history-query --quota 30,70
python -m recsys.platform --read-record <record_id>
python -m recsys.platform --registry
python -m recsys.platform --rag-build
python -m recsys.platform --rag-query "实验记忆 指纹 复用"
python -m recsys.platform --acceptance              # 12 项小样本验收
python -m recsys.platform --acceptance --with-llm    # 只跑一次真实 LLM

# 12. Phase 5：三业务协作演示 + 工件恢复 + 缺工件用例（只加载，不训练/不重建）
python -m recsys.platform --phase5-demo
python -m recsys.platform --run-business recommendation   # 单独调用（search / ads 同理）
python -m recsys.platform --phase5-restore                # 独立进程从清单恢复并推理
python -m recsys.platform --phase5-missing                # 缺工件必须失败，不自动训练
python -m recsys.platform --phase5-report --task-id <task_id>

# 13. 效果专项：排序器归因（只读已落盘候选，不训练、不改链路）
python experiments/2026-11-10-effect-study/run_ranker_ablation.py        # 排序器消融 + 随机锚点
python experiments/2026-11-10-effect-study/run_ranker_recall_features.py # 加两路召回分数
python experiments/2026-11-10-effect-study/run_fusion_rerank_ablation.py # 排序上界 + 融合 × 重排
python experiments/2026-11-10-effect-study/run_seed_study.py 42 --eval-only  # 多种子表（已有权重）
python -m recsys.pipeline run --config configs/pipeline/kuasar_small_no_ranker.toml  # 去掉精排的链路
```

### Phase 5 从零开始的可复制命令（Windows PowerShell）

```powershell
# 0) 仓库根目录，新建独立环境；Phase 5 演示所需的最小依赖：
#    base(torch/numpy) + torch-rechub(MIND) + lightgbm(排序) + faiss(搜索向量) + langgraph(Agent)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements/optional-rechub.txt
pip install -r requirements/optional-lightgbm.txt
pip install -r requirements/optional-faiss.txt
pip install -r requirements/optional-agent.txt
$env:PYTHONPATH = "src"

# 1) 现有小样本与工件：--check 只校验不写盘；下面三条只在对应产物缺失时才运行
#    （会重建 artifacts/search、artifacts/rag，不训练模型；已存在时不要执行以免覆盖）
python -m recsys.data.kuasar_store --check
python -m recsys.data.kuasar_features --check
python -m recsys.data.kuasar_ranking --check
python -m recsys.search.prepare                    # 仅当 artifacts/search/kuasar_bm25 缺失时
python -m recsys.search.build_vector_index         # 仅当 artifacts/search/kuasar_vector 缺失时
python -m recsys.platform --rag-build              # 仅当 artifacts/rag/phase4 缺失时

# 2) 工件清单核对（13 个文件：路径/大小/sha256/索引规模/权重类型/RAG 计数）
python -c "from recsys.platform.artifacts import load_manifest, verify_manifest; r=verify_manifest(load_manifest()); print(r['ok'], r['checked'])"

# 3) 一条命令跑三业务协作演示（约 15 秒，离线计划，无 LLM 请求）
python -m recsys.platform --phase5-demo

# 4) 独立进程恢复 + 缺工件失败用例
python -m recsys.platform --phase5-restore
python -m recsys.platform --phase5-missing

# 5) 按任务 ID 查回三条业务记录、工件和统一摘要（不重跑）
python -m recsys.platform --phase5-report --task-id <上一步输出的 task_id>
```

输出位置（相对仓库根目录）。如果运行时用了 `--phase5-output`，查询报告时也传同一路径。

- 演示根目录：`experiments/2026-11-10-kuasar_small-platform-phase5/demo/`
- 总控/子 Agent 原始日志：`demo/tasks/<task_id>/`（`trace.jsonl`、`decisions.json`、
  `business_recommendation/`、`business_search/`、`business_ads/`）
- 统一摘要：`demo/runs/<task_id>/demo_summary.json`、`demo_summary.md`、`run.log`、
  `artifacts_used.json`
- 共享实验库：`demo/shared/records.jsonl`（同一 `task_id` 下 3 条业务记录 + 1 条任务记录）
- 恢复/缺工件结果：`demo/restore/<restore-id>/restore.json`、
  `demo/missing/<missing-id>/missing_check.json`
- 本文档记录的一次实际运行：`task-20261110T131020-3a865c`，详见
  [docs/phase5-integration.md](phase5-integration.md) §7

说明：推荐排序权重使用 `artifact_policy=existing_only`，缺失时直接报出
`artifacts/kuasar_small/mvp/lgbm_ranker.txt` 并失败，不会自动训练；显式训练入口仍是
`python -m recsys.experiments.rec_agent --train lightgbm_ranker`。广告竞价金额为模拟值、
DIN 为 `interface_test` 固定初始化权重，不能当作真实出价或已训练模型。

不装包时用 `$env:PYTHONPATH = "src"` 也能直接运行。校验与冒烟测试（不需要真实数据）：

```powershell
python scripts/zoo.py check           # 索引 ↔ 源码路径 ↔ 许可证
python scripts/check_project.py       # 配置一致性、文档链接、生成物漂移、残留缓存、第三方来源
python scripts/smoke_adapters.py      # 适配层 + 指标 + 工件溯源 + 排序适配器（合成数据）
python scripts/smoke_phase1.py        # Phase 1：新模型小样本推理、链路、Agent 工具自检
python scripts/smoke_phase3_ads.py    # Phase 3：广告样本/六模型/排序/竞价/Agent 自检

# 数据缓存校验（1–2 秒；不一致会提示 --force 重建）
python -m recsys.data.kuasar_store --check
python -m recsys.data.kuasar_features --check
python -m recsys.data.kuasar_ranking --check
```
