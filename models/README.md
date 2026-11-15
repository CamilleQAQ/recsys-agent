# Model Zoo 索引层

这里**不放模型代码**，只放「按用途分类的模型索引」。源码在 `src/`（自研）或 `third_party/`（第三方）。

```
models/
├── README.md            本文件：分类规则、完整度定义、怎么加模型
├── zoo.toml             唯一事实来源（机读，25 条记录 + 3 个分类元数据）
├── retrieval/README.md  召回模型（生成）
├── ranking/README.md    排序模型（生成）
└── infrastructure/README.md  基础设施（生成）
```

`docs/model-zoo.md`、三个分类页、以及 `docs/models/<id>.md` 技术档案都由 `scripts/zoo.py render`
从 `zoo.toml` 生成。手改会被覆盖；要改内容请改 `zoo.toml`。

## 分类

| 分类 | 含义 | 典型模型 |
| --- | --- | --- |
| `retrieval` | 从全量物品里选出候选集（双塔、图召回、多兴趣、生成式召回） | Two-Tower、DSSM、MIND、LightGCN、TIGER |
| `ranking` | 对候选集精排 / CTR 预估 / 序列下一物品预测 | SASRec、DIN、DIEN、DCNv2、DeepFM、LightGBM |
| `infrastructure` | 不是推荐模型本身，而是让召回跑起来的基础设施 | FAISS、TFRS factorized_top_k |

分类按**模型**切，不按仓库切：RecBole 同时提供 LightGCN（召回）和 DIN（排序），
FuxiCTR 同时提供 DIN/DCNv2（排序）和 DSSM（召回），所以一个仓库会出现在多个分类页的「来源」列里。
若按仓库分类，要么把仓库拆碎（破坏上游导入），要么强行归类（失真）。

## 完整度

| 值 | 含义 | 判据 |
| --- | --- | --- |
| `ready` | 本项目可直接训练、评测 | 有训练入口 + checkpoint 契约 + 适配器 + 技术档案 |
| `upstream_ready` | 上游代码完整可跑，但还没接入本项目数据契约 | 上游有 requirements/入口/样例，缺 `PreparedData` 转换或适配器 |
| `blocked` | 代码在，但当前环境跑不了 | 依赖 TF1.12、Linux-only wheel、旧 numpy/torch、或需要外部权重 |

`blocked` 不等于「代码没价值」：它是**明确的已知限制**，防止有人照着文档装一晚上环境才发现装不上。

每条记录还有 7 个逐项状态（`status`），与完整度分开：

| 字段 | 含义 |
| --- | --- |
| `source_available` | 源码在 `third_party/` 或 `src/` |
| `source_reviewed` | 已定位上游入口、输入输出与依赖 |
| `adapter_ready` | 已有项目适配（适配器或组件 API） |
| `runnable` | 在指定环境中实际跑过基础训练或推理 |
| `integrated` | 已进入统一推荐链路（召回/排序） |
| `agent_callable` | Rec Agent 工具可调用 |
| `evaluated` | 有明确评估结果（不是冒烟指标） |

## 加一个模型

```powershell
# 1. 代码放对位置：自研 → src/recsys/models；第三方 → third_party/<repo>（原样）
# 2. 在 zoo.toml 追加一条 [[models]]
# 3. 若已接入统一接口：召回在 src/recsys/adapters/registry.py 注册，排序在 ranking_registry.py 注册
#    （具体实现按模型类型放 builtin/graph/sequence/tiger/mmrec/neural_ranking），并填 adapter 字段
# 4. 重新生成文档并校验
python scripts/zoo.py render
python scripts/zoo.py check
```

`check` 会验证：必填字段、枚举值、`code` 路径是否存在、vendor 模型对应仓库目录是否存在、
仓库根是否有许可证文件。这些是「索引不撒谎」的底线。

## 与 third_party 的关系

- `third_party/SOURCES.toml`：14 个仓库的**维护版**来源清单（URL、分支、完整 commit、许可、可用性、本地改动）；
- `third_party/repository_manifest.csv`：Phase 0 静态审计快照（含实现文件与符号级证据）；
- `third_party/model_mapping.md`：逐文件的模型定义与行号（跨仓库全量清单，比 `zoo.toml` 更细）；
- `models/zoo.toml`：**只收录本项目打算用的模型**，字段面向「怎么跑起来」而不是「源码在哪一行」。

所以：要找某个论文模型的实现位置 → 查 `model_mapping.md`；要知道这个模型在我们项目里怎么跑、缺什么 → 查 `docs/model-zoo.md`。
