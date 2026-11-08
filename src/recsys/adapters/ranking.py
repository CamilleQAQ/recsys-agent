"""排序模型适配器：目前接入 LightGBM LambdaRank。

为什么先做 LightGBM：它是 Phase 8 里唯一不依赖「特征通道之外一切就位」的排序基线，
训练成本低，能先验证「候选 + 特征 → 分数 → AUC/LogLoss」这条接口是否够用。
DIN / DCNv2 需要先把 KuaiSAR 的行为序列转成 VarLen 特征，等 Phase 1 之后再接。

依赖：``pip install lightgbm``（本项目主环境的可选依赖）。没装时 ``fit`` 会抛
``AdapterError`` 并给出安装提示，而不是 ImportError 直接崩在调用栈深处。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from recsys.adapters.base import AdapterError, RankingAdapter


def _import_lightgbm():
    try:
        import lightgbm  # noqa: PLC0415 - 延迟导入，避免主环境未装时污染 import
    except ImportError as error:  # pragma: no cover - 取决于本机是否安装
        raise AdapterError(
            "LightGBMRankerAdapter 需要 lightgbm：pip install lightgbm（或 -r requirements/optional-lightgbm.txt）"
        ) from error
    return lightgbm


def _as_matrix(features: Mapping[str, np.ndarray] | np.ndarray) -> np.ndarray:
    """特征既可以是二维数组，也可以是「特征名 → 列」的字典（按名字排序保证列序稳定）。"""
    if isinstance(features, Mapping):
        if not features:
            raise AdapterError("特征字典为空")
        columns = [np.asarray(features[name], dtype=np.float32) for name in sorted(features)]
        if len({column.shape[0] for column in columns}) != 1:
            raise AdapterError("特征字典各列长度不一致")
        return np.stack(columns, axis=1)
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise AdapterError("特征必须是二维数组 [num_samples, num_features]")
    return matrix


class LightGBMRankerAdapter(RankingAdapter):
    """LightGBM 排序/CTR 基线：``fit`` → ``predict`` → ``evaluate``（AUC / LogLoss）。"""

    model_id = "lightgbm_ranker"
    feature_names: tuple[str, ...] = ()

    def __init__(self, booster=None, params: dict | None = None) -> None:
        self.booster = booster
        self.params = params or {}

    def fit(
        self,
        features: Mapping[str, np.ndarray] | np.ndarray,
        labels: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        num_boost_round: int = 50,
    ) -> None:
        """``groups`` 给定时按排序目标（Lambdarank）训练，否则按二分类训练。

        排序时 ``groups`` 必须是连续的分组长度（一个用户/一个请求为一段候选）。
        ``num_boost_round`` 默认 50（保持既有调用行为），小样本冒烟时可调小。
        """
        lightgbm = _import_lightgbm()
        matrix = _as_matrix(features)
        target = np.asarray(labels).astype(np.int32)
        if matrix.shape[0] != target.shape[0]:
            raise AdapterError("特征行数与标签长度不一致")
        if isinstance(features, Mapping) and not self.feature_names:
            self.feature_names = tuple(sorted(features))
        if len(set(target.tolist())) < 2:
            raise AdapterError("标签只有单一取值，无法训练排序/CTR 模型")

        if groups is not None:
            group_sizes = np.asarray(groups).astype(np.int32)
            if int(group_sizes.sum()) != matrix.shape[0]:
                raise AdapterError("groups 之和必须等于样本数")
            dataset = lightgbm.Dataset(matrix, label=target, group=group_sizes.tolist())
            objective = "lambdarank"
            metric = "ndcg"
        else:
            dataset = lightgbm.Dataset(matrix, label=target)
            objective = "binary"
            metric = "auc"

        params = {
            "objective": objective,
            "metric": metric,
            "verbosity": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "num_threads": 2,
            **self.params,
        }
        self.booster = lightgbm.train(params, dataset, num_boost_round=num_boost_round)

    def predict(self, features: Mapping[str, np.ndarray] | np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise AdapterError("LightGBMRankerAdapter 还没有模型：先 fit() 或 load()")
        matrix = _as_matrix(features)
        return np.asarray(self.booster.predict(matrix), dtype=np.float64)

    def save(self, path: Path) -> None:
        if self.booster is None:
            raise AdapterError("没有可保存的模型")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "LightGBMRankerAdapter":
        if not path.exists():
            raise AdapterError(f"找不到模型文件：{path}")
        lightgbm = _import_lightgbm()
        return cls(lightgbm.Booster(model_file=str(path)))
