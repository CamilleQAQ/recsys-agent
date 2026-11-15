"""本项目已接入统一接口的模型实现。

每个适配器都复用训练脚本已经写好的 checkpoint 格式，因此**不需要重新训练**
就能用统一接口评测已有产物：

    artifacts/<dataset>/two_tower.pt
    artifacts/<dataset>/matrix_factorization.pt
    artifacts/<dataset>/sasrec.pt

只有 ``PopularityAdapter`` 例外：它没有参数，排名直接从训练集现算。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recsys.adapters.base import AdapterError, RecommenderAdapter, evaluation_users
from recsys.data.loader import resolve_checkpoint_dataset, split_strategy_for
from recsys.data.schema import PreparedData, seen_items_for_split
from recsys.evaluation.evaluator import (
    generate_recommendations as generate_embedding_recommendations,
)
from recsys.evaluation.evaluator import load_model as load_two_tower_model
from recsys.evaluation.sasrec_evaluator import generate_sasrec_recommendations
from recsys.evaluation.sasrec_evaluator import load_sasrec_model
from recsys.models.matrix_factorization import MatrixFactorization
from recsys.models.popularity import (
    generate_popularity_recommendations,
    rank_items_by_popularity,
)


def _check_vocabulary(data: PreparedData, model_config: dict, model_id: str) -> None:
    """词表不一致时，用户/物品索引的含义就变了，指标没有意义。"""
    expected_items = model_config.get("num_items")
    if expected_items is not None and expected_items != data.num_items:
        raise AdapterError(
            f"{model_id}: checkpoint 物品数 {expected_items} 与数据 {data.num_items} 不一致"
        )
    expected_users = model_config.get("num_users")
    if expected_users is not None and expected_users != data.num_users:
        raise AdapterError(
            f"{model_id}: checkpoint 用户数 {expected_users} 与数据 {data.num_users} 不一致"
        )


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise AdapterError(
            f"找不到 checkpoint: {path}。先运行对应的训练入口，或确认 --checkpoint 路径。"
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    expected_strategy = split_strategy_for(resolve_checkpoint_dataset(checkpoint))
    if checkpoint.get("split_strategy") != expected_strategy:
        raise AdapterError(
            f"checkpoint 的切分协议（{checkpoint.get('split_strategy')}）与数据集要求的"
            f"（{expected_strategy}）不一致，需要重新训练"
        )
    return checkpoint


class EmbeddingCheckpointAdapter(RecommenderAdapter):
    """适用于任何提供 encode_users/encode_items 的模型（双塔、MF）。"""

    model_class: type[nn.Module]

    def __init__(self, model: nn.Module, checkpoint: dict, device: torch.device) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device) -> "EmbeddingCheckpointAdapter":
        checkpoint = _load_checkpoint(checkpoint_path, device)
        model_config = checkpoint.get("model_config")
        if not model_config:
            raise AdapterError(f"{checkpoint_path} 缺少 model_config，无法重建模型")
        model = cls.model_class(**model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, checkpoint, device)

    def recommend(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        k: int = 20,
        user_batch_size: int = 512,
        **_: object,
    ) -> dict[int, list[int]]:
        _check_vocabulary(data, self.checkpoint.get("model_config", {}), self.model_id)
        users = evaluation_users(data, split)
        if not users:
            raise AdapterError(f"{split} 切分里没有可评测用户")
        return generate_embedding_recommendations(
            model=self.model,
            user_indices=users,
            num_items=data.num_items,
            train_items_by_user=seen_items_for_split(data, split),
            k=k,
            user_batch_size=user_batch_size,
            device=self.device,
        )

    def _encode(
        self,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        ids: Sequence[int],
        batch_size: int,
        name: str,
    ) -> np.ndarray:
        if not ids:
            raise AdapterError(f"encode_{name}: id 列表为空")
        if batch_size <= 0:
            raise AdapterError("batch_size 必须为正数")
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(ids), batch_size):
                batch = torch.tensor(
                    [int(value) for value in ids[start : start + batch_size]],
                    dtype=torch.long,
                    device=self.device,
                )
                chunk = encoder(batch).detach().cpu().numpy().astype(np.float32)
                chunks.append(chunk)
        return np.concatenate(chunks, axis=0)

    def encode_users(self, user_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        return self._encode(self.model.encode_users, user_ids, batch_size, "users")

    def encode_items(self, item_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        return self._encode(self.model.encode_items, item_ids, batch_size, "items")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, path)


class TwoTowerAdapter(EmbeddingCheckpointAdapter):
    model_id = "two_tower"
    model_type = "embedding"

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device) -> "TwoTowerAdapter":
        # 直接复用评测脚本的加载逻辑，避免两处维护同一份重建规则。
        model, checkpoint = load_two_tower_model(checkpoint_path, device)
        return cls(model, checkpoint, device)


class MatrixFactorizationAdapter(EmbeddingCheckpointAdapter):
    model_id = "matrix_factorization"
    model_type = "embedding"
    model_class = MatrixFactorization


class DSSMAdapter(EmbeddingCheckpointAdapter):
    """DSSM = 带侧特征 MLP 塔的双塔，checkpoint 由 ``train_dssm`` 写出。

    结构与 Two-Tower 完全一致，因此直接复用其重建与向量出口；模型 id 不同，
    便于在推荐链路里把两种训练配置作为独立召回通道比较。
    """

    model_id = "dssm"
    model_type = "embedding"

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device) -> "DSSMAdapter":
        model, checkpoint = load_two_tower_model(checkpoint_path, device)
        return cls(model, checkpoint, device)


class SASRecAdapter(RecommenderAdapter):
    model_id = "sasrec"
    model_type = "sequence"

    def __init__(self, model: nn.Module, checkpoint: dict, device: torch.device) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device) -> "SASRecAdapter":
        model, checkpoint = load_sasrec_model(checkpoint_path, device)
        return cls(model, checkpoint, device)

    def recommend(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        k: int = 20,
        batch_size: int = 512,
        include_validation_in_test_history: bool = True,
        **_: object,
    ) -> dict[int, list[int]]:
        _check_vocabulary(data, self.checkpoint.get("model_config", {}), self.model_id)
        if not evaluation_users(data, split):
            raise AdapterError(f"{split} 切分里没有可评测用户")
        return generate_sasrec_recommendations(
            model=self.model,
            prepared_data=data,
            split=split,
            k=k,
            batch_size=batch_size,
            device=self.device,
            include_validation_in_test_history=include_validation_in_test_history,
        )

    def encode_items(self, item_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        """SASRec 的物品 token 是 item_idx+1（0 留给 padding），这里做一次偏移。"""
        if not item_ids:
            raise AdapterError("encode_items: id 列表为空")
        with torch.no_grad():
            weights = self.model.item_embedding.weight.detach().cpu().numpy()
        tokens = [int(item_id) + 1 for item_id in item_ids]
        if max(tokens) >= weights.shape[0]:
            raise AdapterError("encode_items: item_idx 超出 checkpoint 物品词表")
        return weights[tokens].astype(np.float32)

    def encode_users(self, user_ids: Sequence[int], *, batch_size: int = 1024) -> np.ndarray:
        raise AdapterError(
            "SASRec 没有稳定的用户向量：表示依赖输入序列，请用 recommend()/retrieve() 打分"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint, path)


class PopularityAdapter(RecommenderAdapter):
    """无参数基线：训练集热度排序 + 已曝光过滤。"""

    model_id = "popularity"
    model_type = "statistical"

    def __init__(self) -> None:
        self._ranking: list[int] | None = None
        self._ranking_key: tuple[int, int] | None = None

    def _ranking_for(self, data: PreparedData) -> list[int]:
        # 以 (物品数, 训练集长度) 作为缓存键，数据换了就重算。
        key = (data.num_items, len(data.train_interactions))
        if self._ranking is None or self._ranking_key != key:
            self._ranking = rank_items_by_popularity(
                data.train_interactions, data.num_items
            )
            self._ranking_key = key
        return self._ranking

    def recommend(
        self,
        data: PreparedData,
        *,
        split: str = "test",
        k: int = 20,
        **_: object,
    ) -> dict[int, list[int]]:
        users = evaluation_users(data, split)
        if not users:
            raise AdapterError(f"{split} 切分里没有可评测用户")
        return generate_popularity_recommendations(
            user_indices=users,
            popularity_ranking=self._ranking_for(data),
            train_items_by_user=seen_items_for_split(data, split),
            k=k,
        )


#: 供 registry 与脚本枚举用；顺序即 Model Zoo 里的推荐顺序。
BUILTIN_ADAPTERS: tuple[type[RecommenderAdapter], ...] = (
    TwoTowerAdapter,
    MatrixFactorizationAdapter,
    DSSMAdapter,
    SASRecAdapter,
    PopularityAdapter,
)

#: 序列模型在"测试集是否使用验证集历史"上需要显式控制，接口参数不同。
SEQUENCE_MODELS = frozenset(
    adapter.model_id for adapter in BUILTIN_ADAPTERS if adapter.model_type == "sequence"
)


def adapter_for(model_id: str, adapters: Sequence[type[RecommenderAdapter]]) -> type[RecommenderAdapter]:
    for adapter in adapters:
        if adapter.model_id == model_id:
            return adapter
    known = ", ".join(sorted(adapter.model_id for adapter in adapters))
    raise AdapterError(f"未知模型 {model_id!r}；已接入统一接口的模型：{known}")
