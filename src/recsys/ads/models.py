"""广告模型统一适配层：DeepFM / DIN / DIEN / DCNv2 / MMoE / PLE。

复用关系（不重写底层模型）：

* **DIN / DIEN / DCNv2**：直接复用 Phase 1 的 ``recsys.models.{din,dien,dcnv2}``，
  并用 ``recsys.training.train_neural_ranker.build_model`` 构造，保证与推荐排序
  场景是同一套实现、同一套特征口径（类目 4 列 + 数值 5 列 + 用户数值 2 列）。
  广告侧只替换数据、标签与预测任务。
* **DeepFM / MMoE / PLE**：复用本地 ``third_party/DeepCTR-Torch`` 的上游实现
  （``deepctr_torch.models``），本模块只做项目适配：把 ``AdBatch`` 转成
  DeepCTR 的 ``SparseFeat`` / ``DenseFeat`` 输入，并把多任务输出拆成 CTR/CVR。

统一接口：

* ``score(batch) -> {"ctr": np.ndarray, "cvr"?: np.ndarray}``；
* ``save(path, ...)`` / ``from_checkpoint(payload, data, config)``；
* ``needs_history`` 标识是否读取用户行为序列（DIN/DIEN 为 True）。

DeepCTR 训练/推理都用 ``torch.save(state_dict)``，不依赖它的文件格式；加载时按
``model_config`` 重建同结构再 ``load_state_dict``，便于固定初始化权重做接口验证。
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recsys.ads import AD_MODEL_NAMES
from recsys.models.dcnv2 import DCNv2
from recsys.models.dien import DIEN
from recsys.models.din import DIN
from recsys.ranking.features import (
    CATEGORICAL_CARDINALITIES,
    CATEGORICAL_COLUMNS,
    NUMERIC_ITEM_COLUMNS,
    NUMERIC_USER_COLUMNS,
)
from recsys.training.train_neural_ranker import (
    NeuralRankerTrainConfig,
    build_model as build_ranking_model,
    model_config_for,
)
from recsys.utils.seed import set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AdModelUnavailable(RuntimeError):
    """模型依赖或权重不可用；调用方必须原样报告，不能伪造结果。"""


@dataclass(frozen=True)
class AdModelSpec:
    name: str
    display: str
    framework: str  # project / deepctr_torch
    task: str  # ctr / multitask
    needs_history: bool
    upstream: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


AD_MODEL_SPECS: dict[str, AdModelSpec] = {
    "deepfm": AdModelSpec(
        name="deepfm",
        display="DeepFM",
        framework="deepctr_torch",
        task="ctr",
        needs_history=False,
        upstream="third_party/DeepCTR-Torch/deepctr_torch/models/deepfm.py",
        note="FM 二阶交叉 + DNN 共享 embedding；纯 CTR 任务。",
    ),
    "din": AdModelSpec(
        name="din",
        display="DIN（深度兴趣网络）",
        framework="project",
        task="ctr",
        needs_history=True,
        upstream="src/recsys/models/din.py（Phase 1 复用）",
        note="候选感知注意力池化；广告侧复用同一实现。",
    ),
    "dien": AdModelSpec(
        name="dien",
        display="DIEN（深度兴趣演化）",
        framework="project",
        task="ctr",
        needs_history=True,
        upstream="src/recsys/models/dien.py（Phase 1 复用）",
        note="GRU + AUGRU 兴趣演化；广告侧复用同一实现。",
    ),
    "dcnv2": AdModelSpec(
        name="dcnv2",
        display="DCNv2（Deep & Cross v2）",
        framework="project",
        task="ctr",
        needs_history=False,
        upstream="src/recsys/models/dcnv2.py（Phase 1 复用）",
        note="矩阵式 cross layer + DNN；不使用序列历史。",
    ),
    "mmoe": AdModelSpec(
        name="mmoe",
        display="MMoE（多门混合专家）",
        framework="deepctr_torch",
        task="multitask",
        needs_history=False,
        upstream="third_party/DeepCTR-Torch/deepctr_torch/models/multitask/mmoe.py",
        note="CTR + CVR 双任务；CVR 只在显式标注的模拟样本上验证。",
    ),
    "ple": AdModelSpec(
        name="ple",
        display="PLE（渐进式分层提取）",
        framework="deepctr_torch",
        task="multitask",
        needs_history=False,
        upstream="third_party/DeepCTR-Torch/deepctr_torch/models/multitask/ple.py",
        note="CTR + CVR 双任务；任务间分层隔离；CVR 为模拟标签。",
    ),
}

#: 六模型在广告链路中的统一顺序（也是 CLI / Agent 的默认枚举）。
AD_MODELS = AD_MODEL_NAMES

CHECKPOINT_FORMAT_PROJECT = "recsys.ads.project_model.v1"
CHECKPOINT_FORMAT_DEEPCTR = "recsys.ads.deepctr_torch.v1"


@dataclass(frozen=True)
class AdModelConfig:
    embedding_dim: int = 8
    hidden_dim: int = 32
    attention_dim: int = 16
    dropout: float = 0.1
    seed: int = 42
    device: str = "cpu"
    batch_size: int = 512
    max_history: int = 20
    deepctr_path: str = str(PROJECT_ROOT / "third_party" / "DeepCTR-Torch")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_spec(name: str) -> AdModelSpec:
    try:
        return AD_MODEL_SPECS[name]
    except KeyError as error:
        raise AdModelUnavailable(
            f"未知广告模型 {name!r}；可选 {', '.join(AD_MODELS)}"
        ) from error


def _import_deepctr(path: str):
    """按需把本地 DeepCTR-Torch 加入 sys.path 并导入；失败给出可执行原因。"""
    directory = Path(path)
    if not directory.exists():
        raise AdModelUnavailable(
            f"DeepCTR-Torch 源码目录不存在：{directory}；"
            "DeepFM/MMoE/PLE 需要本地 third_party/DeepCTR-Torch"
        )
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    try:
        from deepctr_torch.inputs import DenseFeat, SparseFeat
        from deepctr_torch.models import DeepFM, MMOE, PLE
    except Exception as error:  # noqa: BLE001 - 依赖问题必须暴露给调用方
        raise AdModelUnavailable(
            f"DeepCTR-Torch 导入失败（{type(error).__name__}: {error}）；"
            "DeepFM/MMoE/PLE 依赖 torch + 本地 third_party/DeepCTR-Torch"
        ) from error
    return {
        "SparseFeat": SparseFeat,
        "DenseFeat": DenseFeat,
        "DeepFM": DeepFM,
        "MMOE": MMOE,
        "PLE": PLE,
    }


class AdModelAdapter:
    """广告模型适配器基类。"""

    name: str
    spec: AdModelSpec

    def score(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def save(self, path: Path, *, weight_kind: str, dataset: str, extra: dict | None = None) -> Path:
        raise NotImplementedError

    @property
    def checkpoint_format(self) -> str:
        raise NotImplementedError


class ProjectAdAdapter(AdModelAdapter):
    """DIN / DIEN / DCNv2：复用 Phase 1 模型与构造入口。"""

    def __init__(self, name: str, data, config: AdModelConfig) -> None:
        self.spec = model_spec(name)
        if self.spec.framework != "project":
            raise ValueError(f"{name} 不是项目内模型")
        self.name = name
        self.config = config
        self.device = torch.device(config.device)
        set_random_seed(config.seed)
        train_config = NeuralRankerTrainConfig(
            model=name,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            max_history=config.max_history,
            seed=config.seed,
            device=config.device,
        )
        self.model = build_ranking_model(name, data, train_config).to(self.device)
        self.model.eval()

    def score(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        tensors = {
            "history": torch.as_tensor(
                np.asarray(batch.get("history", np.zeros((len(batch["items"]), self.config.max_history), dtype=np.int64))),
                dtype=torch.long,
                device=self.device,
            ),
            "candidates": torch.as_tensor(np.asarray(batch["items"]), dtype=torch.long, device=self.device),
            "item_cat": torch.as_tensor(np.asarray(batch["item_cat"]), dtype=torch.long, device=self.device),
            "item_numeric": torch.as_tensor(np.asarray(batch["item_numeric"]), dtype=torch.float32, device=self.device),
            "user_numeric": torch.as_tensor(np.asarray(batch["user_numeric"]), dtype=torch.float32, device=self.device),
        }
        self.model.eval()
        with torch.no_grad():
            logits = self.model(**tensors)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        return {"ctr": probabilities}

    def checkpoint_payload(self, *, weight_kind: str, dataset: str, extra: dict | None = None) -> dict:
        return {
            "format": CHECKPOINT_FORMAT_PROJECT,
            "model": self.name,
            "task": self.spec.task,
            "needs_history": self.spec.needs_history,
            "dataset": dataset,
            "weight_kind": weight_kind,
            "model_config": model_config_for(self.name, self.model),
            "ad_config": self.config.as_dict(),
            "state_dict": self.model.state_dict(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **(extra or {}),
        }

    @property
    def checkpoint_format(self) -> str:
        return CHECKPOINT_FORMAT_PROJECT

    def save(self, path: Path, *, weight_kind: str, dataset: str, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(weight_kind=weight_kind, dataset=dataset, extra=extra), path)
        return path

    @classmethod
    def from_checkpoint(
        cls, payload: dict, data, config: AdModelConfig, *, expected_items: int | None = None
    ) -> "ProjectAdAdapter":
        name = payload["model"]
        if name not in {"din", "dien", "dcnv2"}:
            raise AdModelUnavailable(f"{name} 不是项目内广告模型")
        if expected_items is not None and int(payload.get("model_config", {}).get("num_items", -1)) != int(expected_items):
            if name in {"din", "dien"}:
                raise AdModelUnavailable(
                    f"{name} 权重词表大小 {payload['model_config'].get('num_items')} "
                    f"与当前数据集 {expected_items} 不一致；请在当前数据集上重建权重"
                )
        adapter = cls.__new__(cls)
        adapter.spec = model_spec(name)
        adapter.name = name
        adapter.config = config
        adapter.device = torch.device(config.device)
        model_config = dict(payload["model_config"])
        model_config["cat_cardinalities"] = tuple(model_config.get("cat_cardinalities", CATEGORICAL_CARDINALITIES))
        set_random_seed(config.seed)
        if name == "dcnv2":
            model = DCNv2(**model_config)
        else:
            model = DIN(**model_config) if name == "din" else DIEN(**model_config)
        model.load_state_dict(payload["state_dict"])
        adapter.model = model.to(adapter.device)
        adapter.model.eval()
        return adapter


class DeepCTRAdAdapter(AdModelAdapter):
    """DeepFM / MMoE / PLE：复用本地 DeepCTR-Torch 上游实现。"""

    def __init__(self, name: str, data, config: AdModelConfig) -> None:
        self.spec = model_spec(name)
        if self.spec.framework != "deepctr_torch":
            raise ValueError(f"{name} 不是 DeepCTR 模型")
        self.name = name
        self.config = config
        self.num_items = int(data.num_items)
        self.device = torch.device(config.device)
        self._deps = _import_deepctr(config.deepctr_path)
        set_random_seed(config.seed)
        self.model = self._build()

    def _feature_columns(self):
        SparseFeat = self._deps["SparseFeat"]
        DenseFeat = self._deps["DenseFeat"]
        dim = int(self.config.embedding_dim)
        columns = [
            SparseFeat("item", self.num_items + 1, embedding_dim=dim),
        ]
        for name, cardinality in zip(CATEGORICAL_COLUMNS, CATEGORICAL_CARDINALITIES):
            columns.append(SparseFeat(name, int(cardinality), embedding_dim=dim))
        columns.append(
            DenseFeat(
                "numeric",
                len(NUMERIC_ITEM_COLUMNS) + len(NUMERIC_USER_COLUMNS),
            )
        )
        return columns

    def _build(self):
        columns = self._feature_columns()
        hidden = int(self.config.hidden_dim)
        if self.name == "deepfm":
            model = self._deps["DeepFM"](
                linear_feature_columns=columns,
                dnn_feature_columns=columns,
                dnn_hidden_units=(hidden, max(8, hidden // 2)),
                dnn_dropout=self.config.dropout,
                seed=self.config.seed,
                device=str(self.device),
            )
        elif self.name == "mmoe":
            model = self._deps["MMOE"](
                dnn_feature_columns=columns,
                num_experts=2,
                expert_dnn_hidden_units=(hidden,),
                gate_dnn_hidden_units=(self.config.attention_dim,),
                tower_dnn_hidden_units=(hidden,),
                task_types=("binary", "binary"),
                task_names=("ctr", "cvr"),
                seed=self.config.seed,
                device=str(self.device),
            )
        elif self.name == "ple":
            model = self._deps["PLE"](
                dnn_feature_columns=columns,
                shared_expert_num=1,
                specific_expert_num=1,
                num_levels=2,
                expert_dnn_hidden_units=(hidden,),
                gate_dnn_hidden_units=(self.config.attention_dim,),
                tower_dnn_hidden_units=(hidden,),
                task_types=("binary", "binary"),
                task_names=("ctr", "cvr"),
                seed=self.config.seed,
                device=str(self.device),
            )
        else:  # pragma: no cover - 由注册表约束
            raise AdModelUnavailable(f"DeepCTR 不支持 {self.name}")
        return model

    def _deepctr_input(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        items = np.asarray(batch["items"], dtype=np.int64)
        item_cat = np.asarray(batch["item_cat"], dtype=np.int64)
        item_numeric = np.asarray(batch["item_numeric"], dtype=np.float32)
        user_numeric = np.asarray(batch["user_numeric"], dtype=np.float32)
        features = {"item": items + 1}
        for column, name in enumerate(CATEGORICAL_COLUMNS):
            features[name] = item_cat[:, column].astype(np.int64)
        features["numeric"] = np.concatenate([item_numeric, user_numeric], axis=1)
        return features

    def score(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = np.asarray(self.model.predict(self._deepctr_input(batch), batch_size=self.config.batch_size))
        if output.ndim == 1:
            output = output.reshape(-1, 1)
        result = {"ctr": output[:, 0].astype(np.float64)}
        if self.spec.task == "multitask" and output.shape[1] >= 2:
            result["cvr"] = output[:, 1].astype(np.float64)
        return result

    def checkpoint_payload(self, *, weight_kind: str, dataset: str, extra: dict | None = None) -> dict:
        return {
            "format": CHECKPOINT_FORMAT_DEEPCTR,
            "model": self.name,
            "task": self.spec.task,
            "needs_history": self.spec.needs_history,
            "dataset": dataset,
            "weight_kind": weight_kind,
            "num_items": self.num_items,
            "model_config": {
                "embedding_dim": self.config.embedding_dim,
                "hidden_dim": self.config.hidden_dim,
                "attention_dim": self.config.attention_dim,
                "dropout": self.config.dropout,
                "seed": self.config.seed,
                "device": self.config.device,
                "num_items": self.num_items,
                "cat_cardinalities": list(CATEGORICAL_CARDINALITIES),
            },
            "state_dict": self.model.state_dict(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **(extra or {}),
        }

    @property
    def checkpoint_format(self) -> str:
        return CHECKPOINT_FORMAT_DEEPCTR

    def save(self, path: Path, *, weight_kind: str, dataset: str, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(weight_kind=weight_kind, dataset=dataset, extra=extra), path)
        return path

    @classmethod
    def from_checkpoint(
        cls, payload: dict, data, config: AdModelConfig, *, expected_items: int | None = None
    ) -> "DeepCTRAdAdapter":
        name = payload["model"]
        if name not in {"deepfm", "mmoe", "ple"}:
            raise AdModelUnavailable(f"{name} 不是 DeepCTR 广告模型")
        if expected_items is not None and int(payload.get("num_items", -1)) != int(expected_items):
            raise AdModelUnavailable(
                f"{name} 权重词表大小 {payload.get('num_items')} 与当前数据集 {expected_items} 不一致"
            )
        adapter = cls.__new__(cls)
        adapter.spec = model_spec(name)
        adapter.name = name
        adapter.config = config
        adapter.num_items = int(payload["num_items"])
        adapter.device = torch.device(config.device)
        adapter._deps = _import_deepctr(config.deepctr_path)
        set_random_seed(config.seed)
        adapter.model = adapter._build()
        adapter.model.load_state_dict(payload["state_dict"])
        adapter.model.to(adapter.device)
        adapter.model.eval()
        return adapter


def build_ad_model(name: str, data, config: AdModelConfig | None = None) -> AdModelAdapter:
    config = config or AdModelConfig()
    spec = model_spec(name)
    if spec.framework == "project":
        return ProjectAdAdapter(name, data, config)
    return DeepCTRAdAdapter(name, data, config)


def load_ad_model(
    path: Path,
    data,
    config: AdModelConfig | None = None,
    *,
    expected_items: int | None = None,
) -> tuple[AdModelAdapter, dict]:
    config = config or AdModelConfig()
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "format" not in payload:
        raise AdModelUnavailable(f"{path} 不是广告模型 checkpoint")
    if payload["format"] == CHECKPOINT_FORMAT_PROJECT:
        adapter = ProjectAdAdapter.from_checkpoint(
            payload, data, config, expected_items=expected_items
        )
    elif payload["format"] == CHECKPOINT_FORMAT_DEEPCTR:
        adapter = DeepCTRAdAdapter.from_checkpoint(
            payload, data, config, expected_items=expected_items
        )
    else:
        raise AdModelUnavailable(f"{path} 的格式 {payload['format']!r} 不受支持")
    return adapter, payload


__all__ = [
    "AD_MODELS",
    "AD_MODEL_SPECS",
    "AdModelAdapter",
    "AdModelConfig",
    "AdModelSpec",
    "AdModelUnavailable",
    "ProjectAdAdapter",
    "DeepCTRAdAdapter",
    "build_ad_model",
    "load_ad_model",
    "model_spec",
]
