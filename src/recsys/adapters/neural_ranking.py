"""神经排序模型适配器：DIN / DIEN / DCNv2 / Ultra-HSTU。

与 LightGBM 的 ``predict(features)`` 不同，这些模型需要「用户历史 + 候选物品
+ 物品/用户特征」。因此除统一 ``RankingAdapter`` 接口外，另提供
``score_candidates`` 供推荐链路直接调用；``predict`` 接受已经组装好的
张量字典，便于单元检查和离线复用。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recsys.adapters.base import AdapterError, RankingAdapter
from recsys.data.loader import resolve_checkpoint_dataset, split_strategy_for
from recsys.data.schema import PreparedData
from recsys.models.dcnv2 import DCNv2
from recsys.models.dien import DIEN
from recsys.models.din import DIN
from recsys.models.ultra_hstu import UltraHSTU
from recsys.ranking.features import FeatureContext
from recsys.ranking.samples import (
    histories_for_split_users,
    target_request_timestamps,
)


class NeuralRankingAdapter(RankingAdapter):
    model_class: type[torch.nn.Module] = DIN
    model_type = "neural"

    def __init__(self, model: torch.nn.Module, checkpoint: dict, device: torch.device) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self._feature_context: FeatureContext | None = None
        self._feature_data_id: int | None = None

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "NeuralRankingAdapter":
        if not path.exists():
            raise AdapterError(f"找不到 checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        expected = split_strategy_for(resolve_checkpoint_dataset(checkpoint))
        if checkpoint.get("split_strategy") != expected:
            raise AdapterError(
                f"{cls.model_id} checkpoint 切分协议与数据集要求不一致（{expected}）"
            )
        model = cls.model_class(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(model, checkpoint, device)

    def _context(self, data: PreparedData) -> FeatureContext:
        if self._feature_context is None or self._feature_data_id != id(data):
            self._feature_context = FeatureContext(data)
            self._feature_data_id = id(data)
        return self._feature_context

    def _history_tokens(
        self,
        data: PreparedData,
        split: str,
        users: Sequence[int],
        request_timestamps: Mapping[int, int] | None,
        include_validation_in_test_history: bool,
    ) -> dict[int, np.ndarray]:
        max_history = int(
            self.checkpoint.get("train_config", {}).get("max_history")
            or self.checkpoint["model_config"].get("max_seq_len", 20)
        )
        timestamps = (
            dict(request_timestamps)
            if request_timestamps is not None
            else target_request_timestamps(data, split)
        )
        # 只对请求用户构造；同一用户同一请求只有一份历史。
        selected = {
            int(user): int(timestamps[int(user)])
            for user in users
            if int(user) in timestamps
        }
        histories = histories_for_split_users(
            data,
            split,
            selected,
            max_history=max_history,
            include_validation_in_test_history=include_validation_in_test_history,
        )
        output: dict[int, np.ndarray] = {}
        for user in users:
            output[int(user)] = histories.get(
                int(user), np.zeros(max_history, dtype=np.int64)
            )
        return output

    @torch.no_grad()
    def score_candidates(
        self,
        data: PreparedData,
        users: Sequence[int],
        items_by_user: Mapping[int, Sequence[int]],
        *,
        split: str = "test",
        batch_size: int = 2048,
        request_timestamps: Mapping[int, int] | None = None,
        include_validation_in_test_history: bool = True,
    ) -> dict[int, dict[int, float]]:
        self.model.eval()
        context = self._context(data)
        history_map = self._history_tokens(
            data,
            split,
            users,
            request_timestamps,
            include_validation_in_test_history,
        )
        output: dict[int, dict[int, float]] = {}
        flat_users: list[int] = []
        flat_items: list[int] = []
        for user in users:
            for item in items_by_user.get(int(user), []):
                flat_users.append(int(user))
                flat_items.append(int(item))
        if not flat_users:
            return {int(user): {} for user in users}
        users_array = np.asarray(flat_users, dtype=np.int64)
        items_array = np.asarray(flat_items, dtype=np.int64)
        scores = np.zeros(len(flat_users), dtype=np.float64)
        for start in range(0, len(flat_users), batch_size):
            end = start + batch_size
            batch_users = users_array[start:end]
            batch_items = items_array[start:end]
            history = np.stack([history_map[int(user)] for user in batch_users])
            item_cat, item_numeric = context.item_arrays(batch_items)
            user_numeric = context.user_array(batch_users)
            logits = self.model(
                history=torch.from_numpy(history).to(self.device),
                candidates=torch.from_numpy(batch_items).to(self.device),
                item_cat=torch.from_numpy(item_cat).to(self.device),
                item_numeric=torch.from_numpy(item_numeric).to(self.device),
                user_numeric=torch.from_numpy(user_numeric).to(self.device),
            )
            scores[start:end] = torch.sigmoid(logits).cpu().numpy()
        cursor = 0
        for user in users:
            count = len(items_by_user.get(int(user), []))
            output[int(user)] = {
                int(item): float(scores[cursor + offset])
                for offset, item in enumerate(items_by_user[int(user)])
            }
            cursor += count
        return output

    def predict(self, features: Any) -> np.ndarray:
        if not isinstance(features, Mapping):
            raise AdapterError(
                f"{self.model_id}.predict 需要包含 history/candidates/item_cat/item_numeric/"
                "user_numeric 的字典；推荐链路请用 score_candidates(data, ...)"
            )
        required = ("history", "candidates", "item_cat", "item_numeric", "user_numeric")
        missing = [key for key in required if key not in features]
        if missing:
            raise AdapterError(f"{self.model_id}.predict 缺少字段 {missing}")
        with torch.no_grad():
            logits = self.model(
                **{
                    key: torch.as_tensor(features[key]).to(self.device)
                    for key in required
                }
            )
            return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, path)


class DINAdapter(NeuralRankingAdapter):
    model_id = "din"
    model_class = DIN


class DIENAdapter(NeuralRankingAdapter):
    model_id = "dien"
    model_class = DIEN


class DCNv2Adapter(NeuralRankingAdapter):
    model_id = "dcnv2"
    model_class = DCNv2


class UltraHSTUAdapter(NeuralRankingAdapter):
    model_id = "ultra_hstu"
    model_class = UltraHSTU


__all__ = [
    "DCNv2Adapter",
    "DIENAdapter",
    "DINAdapter",
    "NeuralRankingAdapter",
    "UltraHSTUAdapter",
]
