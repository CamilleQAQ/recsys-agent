"""DSSM 双塔：复用本项目 Two-Tower 的特征塔，作为独立模型入口。

上游参考：``third_party/TorchEasyRec/tzrec/models/dssm.py``（采样 softmax +
用户/物品两侧 MLP）。本项目不引入 TorchEasyRec 的 pipeline/ODPS 运行时，
而是用 ``TwoTowerModel(tower_type="mlp", use_item_features=True)`` 表达同一结构：

* 用户塔：user_id embedding -> MLP -> L2 归一化；
* 物品塔：item_id embedding + 类目/数值特征 -> MLP -> L2 归一化；
* 训练：in-batch 负样本交叉熵（等价于 batch 内采样 softmax）。

这样 DSSM 和 Two-Tower 共享经过测试的底层实现，只有特征与入口不同。
"""

from __future__ import annotations

from recsys.models.two_tower import TwoTowerModel

#: DSSM 在本项目里的模型结构就是「带侧特征 MLP 塔的双塔」。
DSSMModel = TwoTowerModel


def build_dssm_model(
    num_users: int,
    num_items: int,
    *,
    item_cat_cardinalities: list[int] | None = None,
    item_numeric_count: int = 0,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    output_dim: int = 64,
    temperature: float = 0.1,
    feature_dim: int = 16,
) -> TwoTowerModel:
    return TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        temperature=temperature,
        tower_type="mlp",
        item_cat_cardinalities=item_cat_cardinalities,
        item_numeric_count=item_numeric_count,
        feature_dim=feature_dim,
    )


__all__ = ["DSSMModel", "build_dssm_model"]
