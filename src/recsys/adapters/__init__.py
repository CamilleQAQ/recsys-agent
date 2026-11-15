"""统一适配层：把不同模型家族收敛到同一套「推荐 → 评估 → 保存」接口。

设计边界：

* 适配层只做**协议转换**，不重新实现算法，也不改 `third_party/` 里的上游代码；
* 召回侧输入统一是 `recsys.data.schema.PreparedData`（连续整数索引 + 时间戳），
  输出统一是 `{user_idx: [item_idx, ...]}`，向量出口是 `encode_users/encode_items`；
* 排序侧输入是「候选特征 + 标签」，输出是分数，走 `RankingAdapter.predict`；
* 指标统一走 `recsys.evaluation.metrics`，保证不同模型的数字可以横向比较。

新增模型时：召回模型在 `builtin.py` 加一个 `RecommenderAdapter` 子类并在 `registry.py` 注册；
排序模型在 `ranking.py` 加一个 `RankingAdapter` 子类（或直接用 `LightGBMRankerAdapter`）。
"""

from recsys.adapters.base import (
    AdapterError,
    RankingAdapter,
    RecommenderAdapter,
    evaluation_users,
)
from recsys.adapters.neural_ranking import (
    DCNv2Adapter,
    DIENAdapter,
    DINAdapter,
    UltraHSTUAdapter,
)
from recsys.adapters.ranking import LightGBMRankerAdapter
from recsys.adapters.ranking_registry import (
    RANKERS,
    describe_rankers,
    load_ranker,
    ranker_ids,
)
from recsys.adapters.registry import (
    ADAPTERS,
    adapter_ids,
    describe,
    load_adapter,
)

__all__ = [
    "ADAPTERS",
    "AdapterError",
    "DCNv2Adapter",
    "DIENAdapter",
    "DINAdapter",
    "LightGBMRankerAdapter",
    "RANKERS",
    "RankingAdapter",
    "RecommenderAdapter",
    "UltraHSTUAdapter",
    "adapter_ids",
    "describe",
    "describe_rankers",
    "evaluation_users",
    "load_adapter",
    "load_ranker",
    "ranker_ids",
]
