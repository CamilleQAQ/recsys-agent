"""广告业务运行时：数据、特征、历史、模型权重与推理入口的统一持有者。

设计目标（与 Rec Agent 的工具上下文一致）：一次 Agent 会话只加载一次数据集、
特征视图与模型权重；所有工具共享同一个 ``AdsRuntime``，避免重复构造模型和
重复读取 npy 工件。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from recsys.ads.data import AdView, ad_store_schema, load_ad_store
from recsys.ads.features import AdBatch, build_ad_batch
from recsys.ads.models import AdModelConfig, AdModelUnavailable, ProjectAdAdapter, load_ad_model
from recsys.ads.registry import resolve_weight
from recsys.ads.simulated import SimulatedAdView, load_simulated_fixture, simulated_schema
from recsys.config import DEFAULT_ARTIFACTS_DIR, DEFAULT_DATA_DIR
from recsys.data.loader import prepare_dataset
from recsys.ranking.features import FeatureContext
from recsys.ranking.samples import build_history_source, histories_for_requests


class AdsRuntime:
    def __init__(
        self,
        *,
        dataset: str = "kuasar_small",
        data_dir: Path = DEFAULT_DATA_DIR,
        artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
        model_config: AdModelConfig | None = None,
        checkpoints: Mapping[str, Path] | None = None,
    ) -> None:
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.model_config = model_config or AdModelConfig()
        self._checkpoints = dict(checkpoints or {})
        self._data = None
        self._context: FeatureContext | None = None
        self._views: dict[str, AdView] = {}
        self._histories: dict[str, dict[tuple[int, int], np.ndarray]] = {}
        self._models: dict[str, Any] = {}
        self._payloads: dict[str, dict] = {}
        self._simulated: SimulatedAdView | None = None
        self._schema_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ 数据
    def data(self):
        if self._data is None:
            self._data = prepare_dataset(self.dataset, self.data_dir)
        return self._data

    def context(self) -> FeatureContext:
        if self._context is None:
            self._context = FeatureContext(self.data())
        return self._context

    def view(self, split: str = "test") -> AdView:
        if split not in self._views:
            self._views[split] = load_ad_store(self.dataset, self.data_dir, split)
        return self._views[split]

    def histories(self, split: str) -> dict[tuple[int, int], np.ndarray]:
        """按请求构造严格早于请求时间的序列历史（缓存）。"""
        if split not in self._histories:
            view = self.view(split)
            starts = view.starts()
            request_users = np.asarray(view.users)[starts[:-1]]
            source = build_history_source(self.data(), split)
            self._histories[split] = histories_for_requests(
                source,
                zip(request_users.tolist(), view.timestamps.tolist()),
                max_history=self.model_config.max_history,
            )
        return self._histories[split]

    def batch(self, split: str = "test", rows: np.ndarray | None = None) -> AdBatch:
        return build_ad_batch(
            self.view(split),
            self.context(),
            self.histories(split),
            rows=rows,
            max_history=self.model_config.max_history,
        )

    def simulated_view(self) -> SimulatedAdView:
        if self._simulated is None:
            self._simulated = load_simulated_fixture(self.dataset, self.data_dir)
        return self._simulated

    def simulated_batch(self, rows: np.ndarray | None = None) -> AdBatch:
        view = self.simulated_view()
        if rows is None:
            rows = np.arange(len(view))
        # 模拟样本来自 train 切分，历史口径与 train 一致。
        return build_ad_batch(
            _SimulatedAsAdView(view),
            self.context(),
            self.histories("train"),
            rows=rows,
            max_history=self.model_config.max_history,
        )

    def ad_schema(self) -> dict:
        if "ads" not in self._schema_cache:
            self._schema_cache["ads"] = ad_store_schema(self.dataset, self.data_dir)
        return self._schema_cache["ads"]

    def simulated_schema(self) -> dict:
        if "simulated" not in self._schema_cache:
            self._schema_cache["simulated"] = simulated_schema(self.dataset, self.data_dir)
        return self._schema_cache["simulated"]

    # ------------------------------------------------------------------ 模型
    def weights(self, name: str) -> dict[str, Any]:
        if name in self._checkpoints:
            return {"path": self._checkpoints[name]}
        return resolve_weight(name, self.artifacts_dir, self.dataset)

    def model(self, name: str):
        """按模型名加载权重；不可用时抛出可执行的错误。"""
        if name in self._models:
            return self._models[name]
        weight = self.weights(name)
        if weight["path"] is None:
            raise AdModelUnavailable(
                f"{name} 没有可用权重；先构建接口测试权重："
                f"python -m recsys.ads.verify --build-interface-weights --dataset {self.dataset}"
            )
        adapter, payload = load_ad_model(
            Path(weight["path"]),
            self.data(),
            self.model_config,
            expected_items=int(self.data().num_items),
        )
        self._models[name] = adapter
        self._payloads[name] = payload
        return adapter

    def model_payload(self, name: str) -> dict:
        if name not in self._payloads:
            self.model(name)
        return self._payloads[name]

    def weight_info(self, name: str) -> dict[str, Any]:
        info = dict(self.weights(name))
        if info.get("path") is not None:
            info["path"] = str(info["path"])
        payload = self._payloads.get(name)
        if payload is not None:
            info["weight_kind"] = payload.get("weight_kind", info.get("weight_kind"))
            info["is_trained"] = payload.get("is_trained", info.get("is_trained"))
            info["created_at"] = payload.get("created_at")
            info["task"] = payload.get("task")
        return info

    def clear_models(self) -> None:
        self._models.clear()
        self._payloads.clear()

    # ------------------------------------------------------------------ 请求选择
    def request_rows(
        self,
        split: str,
        *,
        count: int,
        min_candidates: int = 1,
        prefer_large: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """选择固定的若干请求，返回 ``(request_indices, rows)``。

        ``prefer_large=True`` 时优先候选数多的请求（用于竞价/排序演示），
        平局按请求下标升序，保证可复现。
        """
        view = self.view(split)
        groups = np.asarray(view.groups)
        eligible = np.flatnonzero(groups >= max(1, int(min_candidates)))
        if eligible.size == 0:
            raise ValueError(
                f"{split} 切分没有候选数 >= {min_candidates} 的广告请求"
            )
        if prefer_large:
            order = np.lexsort((eligible, -groups[eligible]))
            eligible = eligible[order]
        chosen = np.sort(eligible[: max(1, int(count))])
        starts = view.starts()
        row_blocks = [
            np.arange(int(starts[index]), int(starts[index + 1])) for index in chosen
        ]
        rows = np.concatenate(row_blocks) if row_blocks else np.zeros(0, dtype=np.int64)
        return chosen, rows

    def advertiser_ids(self, items: np.ndarray) -> np.ndarray:
        """广告主代理：复用物品特征 ``author_id``（KuaiSAR 作者/广告主字段）。"""
        features = self.data().item_features
        if features is None:
            raise AdModelUnavailable("数据集没有 item_features，无法解析广告主")
        values = np.asarray(features.column("author_id"), dtype=np.int64)
        return values[np.asarray(items, dtype=np.int64)]


class _SimulatedAsAdView:
    """把模拟转化样本适配成 ``build_ad_batch`` 需要的视图形状。"""

    def __init__(self, view: SimulatedAdView) -> None:
        self._view = view
        self.users = view.users
        self.items = view.items
        self.labels = view.click
        self.groups = view.groups
        self.timestamps = view.timestamps

    def __len__(self) -> int:
        return len(self._view)


__all__ = ["AdsRuntime"]
