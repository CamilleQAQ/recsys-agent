"""统一 Agent 平台：Research/Master Agent、共享实验、统一预算、RAG 与 Phase 5 集成。

模块职责：

* :mod:`recsys.platform.budget`：一个研究任务内共享的 LLM / 工具 / 实验 / 时间预算；
* :mod:`recsys.platform.shared_experiments`：推荐、搜索、广告统一实验记录与历史查询；
* :mod:`recsys.platform.rag`：本地资料索引、检索与引用；
* :mod:`recsys.platform.adapters`：三个业务 Agent 入口的薄适配；
* :mod:`recsys.platform.research_agent`：统一入口（Research Agent + 总控调度）；
* :mod:`recsys.platform.acceptance`：Phase 4 小样本验收；
* :mod:`recsys.platform.artifacts`：Phase 5 工件清单与逐文件校验；
* :mod:`recsys.platform.phase5`：三业务协作演示、独立进程恢复、缺工件失败用例。

这里只暴露轻量对象；重型模块由调用方按需导入，避免 ``import recsys`` 时加载 torch。
"""

from __future__ import annotations

from recsys.platform.budget import (
    BudgetExceeded,
    BudgetPolicy,
    BudgetScope,
    TaskBudget,
)
from recsys.platform.shared_experiments import (
    SharedExperimentStore,
    lookup_history,
    model_registry_view,
)

__all__ = [
    "BudgetExceeded",
    "BudgetPolicy",
    "BudgetScope",
    "SharedExperimentStore",
    "TaskBudget",
    "lookup_history",
    "model_registry_view",
]
