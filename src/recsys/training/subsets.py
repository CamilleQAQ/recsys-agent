"""训练脚本的评测子集工具（仅用于 CI/冒烟，不改变默认全量评测）。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from recsys.data.schema import PreparedData


def limit_validation_users(
    data: PreparedData, count: int, seed: int = 42
) -> PreparedData:
    """只保留 ``count`` 个验证用户的留出交互；``count <= 0`` 时原样返回。

    seen/exposure 结构不动（仍是完整数据），因此指标口径与全量一致，只是
    评测用户更少。用于小样本训练冒烟，避免每次验证都给 454k 物品全量打分。
    """
    if count <= 0 or not data.validation_interactions:
        return data
    users = sorted({row.user_idx for row in data.validation_interactions})
    if count >= len(users):
        return data
    rng = np.random.default_rng(seed)
    chosen = set(int(user) for user in rng.choice(np.asarray(users), size=count, replace=False))
    rows = [row for row in data.validation_interactions if row.user_idx in chosen]
    return replace(data, validation_interactions=rows)


__all__ = ["limit_validation_users"]
