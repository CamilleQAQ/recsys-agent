"""Rec Agent 的真实推荐工具注册表。

工具不是模拟器：每个 execute 都落到项目已有的训练入口、适配器推理或推荐链路。

    read_data_profile        读 PreparedData / 特征 / 候选规模
    query_model_zoo          读 models/zoo.toml 的能力与状态
    list_artifacts           列出当前数据集已有 checkpoint / 运行工件
    train_model              调用真实训练入口（可选小预算）
    recall_models            调用真实召回模型，保留分数与通道内名次
    rank_candidates          对上一轮召回结果做真实排序模型推理
    rerank_candidates        类目/作者多样性重排
    evaluate_recommendations 按指定数据切分计算真实指标
    run_pipeline             一次跑完召回→融合→排序→重排→评估
    search_experiment_history 复用既有实验历史查询
    list_experiments         列出历史实验目录

安全边界：模型只能传白名单里的枚举/数值，路径由程序从项目约定推导，
不接受模型传入任意命令或路径。
"""

from __future__ import annotations

import json
import time
import tomllib
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recsys.utils.artifact_guard import ArtifactError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ZOO_PATH = PROJECT_ROOT / "models" / "zoo.toml"

RECALL_MODELS = (
    "two_tower",
    "mind",
    "popularity",
    "matrix_factorization",
    "lightgcn",
    "dssm",
    "sasrec",
    "bert4rec",
    "hstu",
    "tiger",
    "mmrec_multimodal",
)
TRAINABLE_MODELS = (
    "two_tower",
    "matrix_factorization",
    "sasrec",
    "mind",
    "lightgcn",
    "dssm",
    "bert4rec",
    "hstu",
    "rqvae",
    "tiger",
    "mmrec_multimodal",
    "din",
    "dien",
    "dcnv2",
    "ultra_hstu",
    "lightgbm_ranker",
)
RANKER_MODELS = ("lightgbm_ranker", "din", "dien", "dcnv2", "ultra_hstu")


class ToolCallError(ValueError):
    """工具名、参数或阶段不合法（Agent 必须把它报告为失败，而不是继续假装成功）。"""


class ToolBudgetExceeded(RuntimeError):
    """工具调用次数 / 训练次数 / 时间预算耗尽。"""


# --------------------------------------------------------------------------- 预算


@dataclass
class ToolBudget:
    max_tool_calls: int = 10
    max_train_calls: int = 2
    max_seconds: float = 600.0
    max_epochs: int = 3
    max_batches: int = 20
    started: float = field(default_factory=time.perf_counter)
    calls: int = 0
    train_calls: int = 0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def remaining(self) -> float:
        return self.max_seconds - self.elapsed

    def check_call(self) -> None:
        if self.calls >= self.max_tool_calls:
            raise ToolBudgetExceeded(f"工具调用次数已达上限 {self.max_tool_calls}")
        if self.remaining <= 0:
            raise ToolBudgetExceeded("时间预算已耗尽")

    def check_train(self, epochs: int, max_batches: int) -> None:
        if self.train_calls >= self.max_train_calls:
            raise ToolBudgetExceeded(f"训练调用次数已达上限 {self.max_train_calls}")
        if epochs > self.max_epochs:
            raise ToolBudgetExceeded(f"epochs={epochs} 超过单次预算 {self.max_epochs}")
        if max_batches > self.max_batches:
            raise ToolBudgetExceeded(
                f"max_batches={max_batches} 超过单次预算 {self.max_batches}"
            )

    def as_dict(self) -> dict:
        return {
            "max_tool_calls": self.max_tool_calls,
            "max_train_calls": self.max_train_calls,
            "max_seconds": self.max_seconds,
            "max_epochs": self.max_epochs,
            "max_batches": self.max_batches,
            "calls": self.calls,
            "train_calls": self.train_calls,
            "elapsed_s": round(self.elapsed, 2),
            "remaining_s": round(max(0.0, self.remaining), 2),
        }


# --------------------------------------------------------------------------- 上下文


@dataclass
class ToolContext:
    dataset: str = "kuasar_small"
    data_dir: Path = DEFAULT_DATA_DIR
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR
    experiment_dir: Path | None = None
    split: str = "validation"
    users: int = 32
    seed: int = 42
    device: str = "cpu"
    budget: ToolBudget = field(default_factory=ToolBudget)
    state: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    executions: list[dict] = field(default_factory=list)
    _data: Any = None

    def data(self):
        if self._data is None:
            from recsys.data.loader import prepare_dataset

            self._data = prepare_dataset(self.dataset, self.data_dir)
        return self._data

    def log(self, payload: dict) -> None:
        self.events.append(payload)
        if self.experiment_dir is not None:
            path = Path(self.experiment_dir) / "events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def artifact(self, name: str) -> Path:
        base = Path(self.experiment_dir or self.artifacts_dir / self.dataset / "rec_agent")
        path = base / "tool_artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def select_users(self) -> list[int]:
        from recsys.pipeline.chain import select_users

        return select_users(self.data(), self.users, self.seed, self.split)

    def history_setting(self, arguments: dict) -> bool:
        key = "include_validation_in_test_history"
        setting = self.state.setdefault(key, arguments.get(key, True))
        if arguments.get(key, setting) != setting:
            raise ToolCallError(f"{key} 必须与任务设置一致（{setting}）")
        return setting


# --------------------------------------------------------------------------- 注册表


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    execute: Callable[[ToolContext, dict], dict]
    mutates: bool = False

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _object(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ToolCallError(f"重复注册工具 {spec.name}")
        self._specs[spec.name] = spec

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise ToolCallError(f"未注册的工具：{name}") from error

    def schemas(self) -> list[dict]:
        return [spec.schema() for spec in self._specs.values()]

    def validate(self, name: str, arguments: Any) -> dict:
        spec = self.get(name)
        if not isinstance(arguments, dict):
            raise ToolCallError("工具参数必须是 JSON 对象")
        schema = spec.parameters
        allowed = set(schema.get("properties", {}))
        extra = sorted(set(arguments) - allowed)
        if extra and not schema.get("additionalProperties", True):
            raise ToolCallError(f"{name} 不接受参数 {extra}")
        missing = [key for key in schema.get("required", []) if key not in arguments]
        if missing:
            raise ToolCallError(f"{name} 缺少参数 {missing}")
        for key, value in arguments.items():
            definition = schema["properties"].get(key)
            if not definition:
                continue
            expected = definition.get("type")
            if expected == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ToolCallError(f"{name}.{key} 必须是整数")
            elif expected == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ToolCallError(f"{name}.{key} 必须是数值")
            elif expected == "string":
                if not isinstance(value, str):
                    raise ToolCallError(f"{name}.{key} 必须是字符串")
            elif expected == "boolean":
                if not isinstance(value, bool):
                    raise ToolCallError(f"{name}.{key} 必须是布尔值")
            elif expected == "array":
                if not isinstance(value, list):
                    raise ToolCallError(f"{name}.{key} 必须是数组")
            if "enum" in definition and value not in definition["enum"]:
                raise ToolCallError(
                    f"{name}.{key}={value!r} 不在允许集合 {definition['enum']}"
                )
            if "minimum" in definition and value < definition["minimum"]:
                raise ToolCallError(f"{name}.{key} 小于最小值 {definition['minimum']}")
            if "maximum" in definition and value > definition["maximum"]:
                raise ToolCallError(f"{name}.{key} 大于最大值 {definition['maximum']}")
        return dict(arguments)

    def call(self, name: str, arguments: dict, context: ToolContext) -> dict:
        spec = self.get(name)
        validated = self.validate(name, arguments)
        context.budget.check_call()
        context.budget.calls += 1
        started = time.perf_counter()
        record = {"tool": name, "arguments": validated, "started": time.time()}
        context.log({"event": "tool_start", **record})
        try:
            result = spec.execute(context, validated)
            ok = True
            error = None
        except (ToolBudgetExceeded, ToolCallError):
            raise
        except Exception as exc:  # noqa: BLE001 - 工具失败必须被 Agent 看到
            ok = False
            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4),
            }
            error = result["error"]
        elapsed = round(time.perf_counter() - started, 2)
        result.setdefault("ok", ok)
        record.update(
            {
                "elapsed_s": elapsed,
                "ok": result.get("ok", ok),
                "error": error or result.get("error"),
            }
        )
        context.executions.append(record)
        context.log({"event": "tool_end", **record, "result_summary": _summarize(result)})
        return result


def _summarize(result: dict) -> dict:
    """只保留小字段，避免把整份候选写进日志。"""
    summary = {}
    for key, value in result.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            summary[key] = value
        elif key in {"metrics", "diagnostics", "artifacts"} and isinstance(value, dict):
            summary[key] = value
    return summary


# --------------------------------------------------------------------------- 路径约定


CHECKPOINT_CANDIDATES = {
    "two_tower": ("two_tower_features.pt", "two_tower.pt"),
    "matrix_factorization": ("matrix_factorization.pt",),
    "mind": ("mind20.pt", "mind.pt"),
    "sasrec": ("sasrec.pt",),
    "lightgcn": ("lightgcn.pt", "lightgcn_smoke.pt"),
    "dssm": ("dssm.pt",),
    "bert4rec": ("bert4rec.pt",),
    "hstu": ("hstu.pt",),
    "tiger": ("tiger.pt",),
    "mmrec_multimodal": ("mmrec_vbpr.pt", "mmrec_multimodal.pt"),
    "din": ("din.pt",),
    "dien": ("dien.pt",),
    "dcnv2": ("dcnv2.pt",),
    "ultra_hstu": ("ultra_hstu.pt",),
    "rqvae": ("rqvae.pt",),
}


def resolve_checkpoint(context: ToolContext, model_id: str) -> Path | None:
    if model_id == "popularity":
        return None
    trained = context.state.get("trained_checkpoints", {}).get(model_id)
    if trained and Path(trained).exists():
        return Path(trained)
    base = Path(context.artifacts_dir) / context.dataset
    for name in CHECKPOINT_CANDIDATES.get(model_id, (f"{model_id}.pt",)):
        path = base / name
        if path.exists():
            return path
    return None


def resolve_ranker_checkpoint(context: ToolContext, model_id: str) -> Path | None:
    trained = context.state.get("trained_checkpoints", {}).get(model_id)
    if trained and Path(trained).exists():
        return Path(trained)
    base = Path(context.artifacts_dir) / context.dataset
    if model_id == "lightgbm_ranker":
        for candidate in (
            base / "mvp" / "lgbm_ranker.txt",
            base / "lightgbm_ranker.txt",
        ):
            if candidate.exists():
                return candidate
        return None
    return resolve_checkpoint(context, model_id)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranker_checkpoint_candidates(context: ToolContext, model_id: str) -> list[Path]:
    """列出 ``resolve_ranker_checkpoint`` 会检查的默认路径（用于失败时报具体文件）。"""

    base = Path(context.artifacts_dir) / context.dataset
    if model_id == "lightgbm_ranker":
        return [base / "mvp" / "lgbm_ranker.txt", base / "lightgbm_ranker.txt"]
    names = CHECKPOINT_CANDIDATES.get(model_id, (f"{model_id}.pt",))
    return [base / name for name in names]


def _resolve_ranker_for_run(
    context: ToolContext,
    ranker: str,
    arguments: dict,
) -> tuple[Path | None, dict | None]:
    """解析排序权重；返回 ``(checkpoint, error_payload)``。

    ``artifact_policy="existing_only"`` 是 Phase 5 的只加载策略：缺失或哈希不匹配时
    返回带具体文件名的失败结果，不训练、不换模型、不改用随机权重。
    """

    policy = str(arguments.get("artifact_policy") or "train_if_missing")
    if policy not in {"existing_only", "train_if_missing"}:
        raise ToolCallError(
            f"artifact_policy={policy!r} 不支持；只允许 existing_only / train_if_missing"
        )
    explicit = arguments.get("ranker_checkpoint")
    expected = arguments.get("expected_ranker_sha256")
    if explicit:
        checkpoint = Path(str(explicit))
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
    else:
        checkpoint = resolve_ranker_checkpoint(context, ranker)

    if policy == "existing_only":
        if checkpoint is None or not Path(checkpoint).exists():
            checked = [str(path) for path in ranker_checkpoint_candidates(context, ranker)]
            if checkpoint is not None:
                checked = [str(checkpoint), *checked]
            return None, {
                "ok": False,
                "error": (
                    f"排序权重缺失：{checkpoint or '未找到'}；已检查 {checked}。"
                    "artifact_policy=existing_only 只加载已有权重，"
                    "不会自动训练、不会换模型、不会改用随机权重。"
                    "如需训练请显式运行 python -m recsys.experiments.rec_agent --train lightgbm_ranker"
                ),
                "missing_artifact": str(checkpoint) if checkpoint is not None else None,
                "checked_paths": checked,
                "artifact_policy": policy,
                "auto_train_attempted": False,
            }
        actual = _sha256_file(Path(checkpoint))
        if expected and actual.lower() != str(expected).lower():
            return None, {
                "ok": False,
                "error": (
                    f"排序权重校验不匹配：{checkpoint} 实际 sha256={actual}，"
                    f"清单记录={expected}。不会自动训练或重建"
                ),
                "missing_artifact": str(checkpoint),
                "actual_sha256": actual,
                "expected_sha256": expected,
                "artifact_policy": policy,
                "auto_train_attempted": False,
            }
        return Path(checkpoint), None

    # train_if_missing：保留原有显式训练行为。
    if checkpoint is None and ranker != "lightgbm_ranker":
        return None, {
            "ok": False,
            "error": f"排序模型 {ranker} 缺少 checkpoint；先调用 train_model",
            "missing_model": ranker,
        }
    if checkpoint is None:
        checkpoint = ranker_checkpoint_candidates(context, ranker)[0]
    if expected and Path(checkpoint).exists():
        actual = _sha256_file(Path(checkpoint))
        if actual.lower() != str(expected).lower():
            return None, {
                "ok": False,
                "error": (
                    f"排序权重校验不匹配：{checkpoint} 实际 sha256={actual}，"
                    f"清单记录={expected}"
                ),
                "actual_sha256": actual,
                "expected_sha256": expected,
            }
    return Path(checkpoint), None


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- 只读工具


def _profile(context: ToolContext, arguments: dict) -> dict:
    data = context.data()
    counts = {
        "users": data.num_users,
        "items": data.num_items,
        "train": len(data.train_interactions),
        "validation": len(data.validation_interactions),
        "test": len(data.test_interactions),
    }
    features = list(data.item_features.names) if data.item_features is not None else []
    ranking_dir = Path(context.data_dir) / "kuasar" / "processed" / "ranking"
    ranking_available = (ranking_dir / "schema.json").exists()
    checkpoints = {
        model: _relative(path) if (path := resolve_checkpoint(context, model)) else None
        for model in RECALL_MODELS
    }
    return {
        "ok": True,
        "dataset": context.dataset,
        "split_strategy": data.split_strategy,
        "counts": counts,
        "targets": {
            "validation_users": len({row.user_idx for row in data.validation_interactions}),
            "test_users": len({row.user_idx for row in data.test_interactions}),
            "validation_targets": len(data.validation_interactions),
            "test_targets": len(data.test_interactions),
        },
        "item_features": features,
        "ranking_samples_available": ranking_available,
        "ranking_dir": _relative(ranking_dir),
        "checkpoints": checkpoints,
        "note": "本地 KuaiSAR 没有视觉/文本 embedding；Semantic ID 使用物品元数据向量。",
    }


def _zoo(context: ToolContext, arguments: dict) -> dict:
    with ZOO_PATH.open("rb") as file:
        zoo = tomllib.load(file)
    category = arguments.get("category")
    status_key = arguments.get("status")
    models = []
    for model in zoo.get("models", []):
        if category and model.get("category") != category:
            continue
        status = model.get("status") or {}
        if status_key:
            flag = status_key.replace("agent_callable", "agent_callable")
            if not status.get(flag):
                continue
        models.append(
            {
                "id": model["id"],
                "name": model["name"],
                "category": model["category"],
                "completeness": model["completeness"],
                "adapter": model.get("adapter", ""),
                "status": status,
                "train_entry": model["train_entry"],
                "deps": model["deps"],
                "limitations": model.get("limitations", ""),
            }
        )
    return {
        "ok": True,
        "count": len(models),
        "models": models,
        "categories": list((zoo.get("categories") or {}).keys()),
    }


def _artifacts(context: ToolContext, arguments: dict) -> dict:
    base = Path(context.artifacts_dir) / context.dataset
    files = []
    if base.exists():
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in {".pt", ".txt", ".json"}:
                files.append(
                    {
                        "path": _relative(path),
                        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                        "mtime": int(path.stat().st_mtime),
                    }
                )
    return {"ok": True, "dataset": context.dataset, "base": _relative(base), "files": files}


# --------------------------------------------------------------------------- 训练工具


def _new_train_dir(context: ToolContext, model_id: str) -> Path:
    """每次训练一个唯一目录；并发/重复训练不会复用输出路径。"""
    base = Path(context.experiment_dir or context.artifacts_dir) / "training"
    run_id = f"{model_id}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = base / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _checkpoint_payload(path: Path, **extra: Any) -> dict:
    if not path.exists():
        raise RuntimeError(f"训练入口没有生成预期的 checkpoint：{path}")
    return {"ok": True, "checkpoint": _relative(path), **extra}


def _run_train(context: ToolContext, model_id: str, arguments: dict) -> dict:
    from recsys.training.trainer import TrainConfig, train

    dataset = context.dataset
    epochs = int(arguments.get("epochs", 1))
    max_batches = int(arguments.get("max_batches", 2))
    seed = int(arguments.get("seed", context.seed))
    device = context.device
    validation_users = int(arguments.get("validation_users", 8))
    embedding_dim = int(arguments.get("embedding_dim", 16))
    context.budget.check_train(epochs, max_batches)
    train_dir = _new_train_dir(context, model_id)
    output_dir = train_dir / dataset

    common = {
        "dataset": dataset,
        "data_dir": Path(context.data_dir),
        "artifacts_dir": train_dir,
        "epochs": epochs,
        "seed": seed,
        "device": device,
    }
    if model_id == "two_tower":
        from recsys.training.trainer import TrainConfig as Config

        config = Config(
            **common,
            max_batches_per_epoch=max_batches,
            embedding_dim=embedding_dim,
            hidden_dim=32,
            output_dim=16,
            batch_size=512,
            validation_users=validation_users,
            mask_false_negatives=False,
        )
        model, losses = train(config)
        return _checkpoint_payload(
            output_dir / "two_tower.pt",
            epochs=epochs,
            losses=[round(float(value), 6) for value in losses],
        )
    if model_id == "matrix_factorization":
        from recsys.training.train_mf import MFTrainConfig, train_mf

        config = MFTrainConfig(
            **common,
            max_batches_per_epoch=max_batches,
            embedding_dim=embedding_dim,
            batch_size=512,
            validation_users=validation_users,
        )
        _, losses = train_mf(config)
        return _checkpoint_payload(
            output_dir / "matrix_factorization.pt",
            losses=[round(float(value), 6) for value in losses],
        )
    if model_id == "sasrec":
        from recsys.training.train_sasrec import SASRecTrainConfig, train_sasrec

        config = SASRecTrainConfig(
            **common,
            max_epochs=epochs,
            patience=1,
            batch_size=128,
            embedding_dim=embedding_dim,
            max_seq_len=20,
            num_blocks=1,
            validation_users=validation_users,
        )
        _, checkpoint = train_sasrec(config)
        return _checkpoint_payload(
            output_dir / "sasrec.pt",
            metrics=checkpoint.get("best_validation_metrics", {}),
        )
    if model_id == "mind":
        from recsys.training.train_mind import MindTrainConfig, train_mind

        config = MindTrainConfig(
            dataset=dataset,
            data_dir=Path(context.data_dir),
            artifacts_dir=train_dir,
            epochs=epochs,
            batch_size=128,
            embed_dim=embedding_dim,
            max_length=20,
            interest_num=2,
            seed=seed,
            device=device,
            max_batches_per_epoch=max_batches,
            validation_users=validation_users,
        )
        _, checkpoint = train_mind(config)
        return _checkpoint_payload(
            output_dir / "mind.pt",
            metrics=checkpoint.get("best_validation_metrics", {}),
        )
    if model_id == "lightgcn":
        from recsys.training.train_lightgcn import LightGCNTrainConfig, train_lightgcn

        config = LightGCNTrainConfig(
            **common,
            max_batches_per_epoch=max_batches,
            embedding_dim=embedding_dim,
            num_layers=2,
            batch_size=1024,
            validation_users=validation_users,
        )
        _, losses = train_lightgcn(config)
        return _checkpoint_payload(output_dir / "lightgcn.pt", losses=losses)
    if model_id == "dssm":
        from recsys.training.train_dssm import dssm_config

        config = dssm_config(
            {
                "dataset": dataset,
                "data_dir": Path(context.data_dir),
                "artifacts_dir": train_dir,
                "epochs": epochs,
                "seed": seed,
                "device": device,
                "max_batches_per_epoch": max_batches,
                "embedding_dim": embedding_dim,
                "hidden_dim": 32,
                "output_dim": 16,
                "batch_size": 512,
                "validation_users": validation_users,
            }
        )
        train(config)
        return _checkpoint_payload(output_dir / "dssm.pt")
    if model_id in {"bert4rec", "hstu"}:
        from recsys.training.train_sequence import SequenceTrainConfig, train_sequence

        config = SequenceTrainConfig(
            model=model_id,
            dataset=dataset,
            data_dir=Path(context.data_dir),
            artifacts_dir=train_dir,
            max_epochs=epochs,
            patience=1,
            batch_size=128,
            embedding_dim=embedding_dim,
            max_seq_len=20,
            num_blocks=1,
            num_heads=2,
            seed=seed,
            device=device,
            max_batches_per_epoch=max_batches,
            validation_users=validation_users,
        )
        _, metrics = train_sequence(config)
        return _checkpoint_payload(output_dir / f"{model_id}.pt", metrics=metrics)
    if model_id == "rqvae":
        from recsys.training.train_rqvae import RQVAETrainConfig, train_rqvae

        config = RQVAETrainConfig(
            dataset=dataset,
            data_dir=Path(context.data_dir),
            artifacts_dir=train_dir,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        result = train_rqvae(config)
        return _checkpoint_payload(
            Path(result["checkpoint"]),
            unique_sids=result["unique_sids"],
            code_usage_per_level=result["code_usage_per_level"],
        )
    if model_id == "tiger":
        from recsys.training.train_tiger import TigerTrainConfig, train_tiger

        rqvae_checkpoint = (
            context.state.get("trained_checkpoints", {}).get("rqvae")
            or resolve_checkpoint(context, "rqvae")
        )
        if rqvae_checkpoint is None:
            raise ToolCallError(
                "tiger 需要 RQ-VAE 码表：先在同一任务里 train_model(rqvae)，"
                "或准备 artifacts/<dataset>/rqvae.pt"
            )
        config = TigerTrainConfig(
            dataset=dataset,
            data_dir=Path(context.data_dir),
            artifacts_dir=train_dir,
            rqvae_checkpoint=Path(rqvae_checkpoint),
            epochs=epochs,
            batch_size=128,
            seed=seed,
            device=device,
            max_batches_per_epoch=max_batches,
            val_users=min(64, context.users),
            beam_size=4,
        )
        result = train_tiger(config)
        return _checkpoint_payload(
            Path(result["checkpoint"]), metrics=result.get("metrics", {})
        )
    if model_id == "mmrec_multimodal":
        from recsys.training.train_mmrec import MMRecTrainConfig, train_mmrec

        config = MMRecTrainConfig(
            **common,
            max_batches_per_epoch=max_batches,
            embedding_dim=embedding_dim,
            content_dim=16,
            content_embed_dim=16,
            batch_size=1024,
            validation_users=validation_users,
        )
        _, result = train_mmrec(config)
        return _checkpoint_payload(
            output_dir / "mmrec_vbpr.pt", metrics=result.get("metrics", {})
        )
    if model_id in {"din", "dien", "dcnv2", "ultra_hstu"}:
        from recsys.training.train_neural_ranker import (
            NeuralRankerTrainConfig,
            train_neural_ranker,
        )

        config = NeuralRankerTrainConfig(
            model=model_id,
            dataset=dataset,
            data_dir=Path(context.data_dir),
            artifacts_dir=train_dir,
            epochs=epochs,
            batch_size=512,
            max_history=20,
            embedding_dim=embedding_dim,
            hidden_dim=32,
            max_train_rows=min(4000, int(arguments.get("max_rows", 4000))),
            max_valid_rows=2000,
            seed=seed,
            device=device,
            max_batches_per_epoch=max_batches,
        )
        result = train_neural_ranker(config)
        return _checkpoint_payload(
            Path(result["checkpoint"]), metrics=result.get("metrics", {})
        )
    if model_id == "lightgbm_ranker":
        from recsys.pipeline.ranking import train_lightgbm

        data = context.data()
        users = context.select_users()
        path = output_dir / "lightgbm_ranker.txt"
        train_lightgbm(
            data,
            users,
            checkpoint=path,
            max_rows=int(arguments.get("max_rows", 4000)),
            num_boost_round=int(arguments.get("num_boost_round", 10)),
            parent_ranking=Path(context.data_dir) / "kuasar" / "processed" / "ranking",
        )
        return _checkpoint_payload(path)
    raise ToolCallError(f"train_model 不支持 {model_id!r}")


def _train(context: ToolContext, arguments: dict) -> dict:
    model_id = arguments["model"]
    if model_id not in TRAINABLE_MODELS:
        raise ToolCallError(f"未知可训练模型 {model_id!r}")
    result = _run_train(context, model_id, arguments)
    context.budget.train_calls += 1
    result["model"] = model_id
    # 只有训练成功才更新任务内的权重引用；失败不会覆盖旧引用。
    checkpoint = result.get("checkpoint")
    if result.get("ok") and checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        context.state.setdefault("trained_checkpoints", {})[model_id] = str(path)
        result["checkpoint"] = _relative(path)
    return result


# --------------------------------------------------------------------------- 召回 / 排序 / 重排 / 评估


def _resolve_recall_models(arguments: dict) -> list[str]:
    models = arguments.get("models") or [arguments.get("model")]
    models = [model for model in models if model]
    if not models:
        raise ToolCallError("recall_models 需要 models")
    for model in models:
        if model not in RECALL_MODELS:
            raise ToolCallError(f"未知召回模型 {model!r}")
    return list(dict.fromkeys(models))


def _channel_artifacts(context: ToolContext, channels, users) -> dict[str, str]:
    from recsys.pipeline.chain import _json_dump

    paths = {}
    for name, mapping in channels.items():
        path = context.artifact(f"recall_{name}.json")
        _json_dump(
            path,
            {
                "channel": name,
                "users": [
                    {
                        "user_idx": int(user),
                        "items": [
                            {"item": candidate.item, "rank": candidate.rank, "score": candidate.score}
                            for candidate in mapping.get(user, [])
                        ],
                    }
                    for user in users
                ],
            },
        )
        paths[name] = _relative(path)
    return paths


def _recall(context: ToolContext, arguments: dict) -> dict:
    from recsys.adapters.registry import load_adapter
    from recsys.pipeline.recall import run_channel

    models = _resolve_recall_models(arguments)
    k = int(arguments.get("k", 100))
    split = arguments.get("split", context.split)
    history_setting = context.history_setting(arguments)
    users = context.select_users()
    data = context.data()
    channels = {}
    missing = []
    for model in models:
        checkpoint = resolve_checkpoint(context, model)
        if checkpoint is None and model != "popularity":
            missing.append(model)
            continue
        adapter = load_adapter(model, checkpoint, device_name=context.device)
        channels[model] = run_channel(
            adapter,
            data,
            users,
            split=split,
            k=k,
            include_validation_in_test_history=history_setting,
        )
    if not channels:
        return {"ok": False, "error": f"没有可加载的召回模型；缺少 checkpoint：{missing}"}
    context.state["channels"] = channels
    context.state["recall_split"] = split
    paths = _channel_artifacts(context, channels, users)
    targets = data.validation_interactions if split == "validation" else data.test_interactions
    user_set = set(users)
    target_by_user = {}
    for row in targets:
        if row.user_idx in user_set:
            target_by_user.setdefault(row.user_idx, set()).add(row.item_idx)
    summary = {}
    for name, mapping in channels.items():
        hits = sum(
            len({candidate.item for candidate in mapping.get(user, [])} & target_by_user.get(user, set()))
            for user in users
        )
        summary[name] = {
            "candidates": sum(len(mapping.get(user, [])) for user in users),
            "hits_at_k": hits,
            "targets": sum(len(values) for values in target_by_user.values()),
        }
    return {
        "ok": True,
        "split": split,
        "include_validation_in_test_history": history_setting,
        "users": len(users),
        "models": list(channels),
        "missing_checkpoints": missing,
        "channel_summary": summary,
        "artifacts": paths,
    }


def _fused_from_state(context: ToolContext, arguments: dict):
    from recsys.pipeline.config import FusionConfig
    from recsys.pipeline.fusion import fuse_channels

    channels = context.state.get("channels")
    if not channels:
        raise ToolCallError("当前会话还没有召回结果；先调用 recall_models")
    mode = arguments.get("fusion", "round_robin")
    config = FusionConfig(mode=mode, limit=int(arguments.get("limit", 100)))
    return fuse_channels(channels, config)


def _rank(context: ToolContext, arguments: dict) -> dict:
    from recsys.pipeline.ranking import rank_fused
    from recsys.pipeline.chain import _json_dump

    ranker_id = arguments["model"]
    if ranker_id not in RANKER_MODELS:
        raise ToolCallError(f"未知排序模型 {ranker_id!r}")
    history_setting = context.history_setting(arguments)
    checkpoint = resolve_ranker_checkpoint(context, ranker_id)
    if checkpoint is None:
        return {
            "ok": False,
            "error": f"排序模型 {ranker_id} 缺少 checkpoint；先调用 train_model",
            "expected_checkpoint": _relative(
                Path(context.artifacts_dir) / context.dataset / f"{ranker_id}.pt"
            ),
        }
    from recsys.adapters.ranking_registry import load_ranker

    ranker = load_ranker(ranker_id, checkpoint, device_name=context.device)
    fused = _fused_from_state(context, arguments)
    context.state["fused"] = fused
    data = context.data()
    split = context.state.get("recall_split", context.split)
    from recsys.ranking.samples import target_request_timestamps

    ranked = rank_fused(
        ranker,
        data,
        fused,
        split=split,
        request_timestamps=target_request_timestamps(data, split),
        include_validation_in_test_history=history_setting,
    )
    context.state["ranked"] = ranked
    path = context.artifact("ranked.json")
    _json_dump(
        path,
        {
            "rankers": ranker_id,
            "users": [
                {
                    "user_idx": int(user),
                    "items": [
                        {
                            "item": candidate.item,
                            "rank": candidate.rank,
                            "score": candidate.score,
                            "sources": candidate.sources,
                        }
                        for candidate in ranked.get(user, [])
                    ],
                }
                for user in sorted(ranked)
            ],
        },
    )
    return {
        "ok": True,
        "model": ranker_id,
        "checkpoint": _relative(checkpoint),
        "include_validation_in_test_history": history_setting,
        "users": len(ranked),
        "artifact": _relative(path),
    }


def _rerank(context: ToolContext, arguments: dict) -> dict:
    from recsys.pipeline.chain import _json_dump
    from recsys.pipeline.rerank import diversify

    ranked = context.state.get("ranked")
    if not ranked:
        raise ToolCallError("当前会话还没有排序结果；先调用 rank_candidates")
    data = context.data()
    categories = (
        np.asarray(data.item_features.column("category_first_code"))
        if data.item_features is not None
        else np.asarray([], dtype=np.int32)
    )
    author_cap = int(arguments.get("author_cap", 0))
    authors = (
        np.asarray(data.item_features.column("author_id"))
        if data.item_features is not None and author_cap > 0
        else None
    )
    final = {}
    relaxations = 0
    for user, rows in ranked.items():
        picked, relaxed = diversify(
            rows,
            categories=categories,
            authors=authors,
            k=int(arguments.get("k", 20)),
            category_cap=int(arguments.get("category_cap", 5)),
            author_cap=author_cap,
        )
        final[user] = picked
        relaxations += relaxed
    context.state["final"] = final
    path = context.artifact("final.json")
    _json_dump(
        path,
        {
            "users": [
                {
                    "user_idx": int(user),
                    "items": [
                        {
                            "item": candidate.item,
                            "rank": rank + 1,
                            "score": candidate.score,
                            "sources": candidate.sources,
                        }
                        for rank, candidate in enumerate(final.get(user, []))
                    ],
                }
                for user in sorted(final)
            ],
            "category_cap": int(arguments.get("category_cap", 5)),
            "author_cap": author_cap,
        },
    )
    return {
        "ok": True,
        "users": len(final),
        "relaxations": relaxations,
        "artifact": _relative(path),
    }


def _evaluate(context: ToolContext, arguments: dict) -> dict:
    from recsys.evaluation.evaluator import evaluate_recommendations
    from recsys.evaluation.metrics import catalog_coverage

    stage = arguments.get("stage", "final")
    if stage == "final":
        rows = context.state.get("final")
        source = "rerank_candidates"
    elif stage == "ranked":
        rows = context.state.get("ranked")
        source = "rank_candidates"
    elif stage == "recall":
        channels = context.state.get("channels") or {}
        model = arguments.get("model") or (next(iter(channels)) if channels else None)
        rows = channels.get(model) if model else None
        source = f"recall_models:{model}"
    else:
        raise ToolCallError(f"evaluate_recommendations.stage 不支持 {stage!r}")
    if not rows:
        raise ToolCallError(f"没有可评估的 {stage} 结果；先运行对应工具（{source}）")
    split = arguments.get("split", context.split)
    ks = tuple(int(value) for value in arguments.get("ks", [10, 20]))
    data = context.data()
    interactions = data.validation_interactions if split == "validation" else data.test_interactions
    user_set = {int(user) for user in rows}
    targets = [row for row in interactions if row.user_idx in user_set]
    recommendations = {
        int(user): (
            [candidate.item for candidate in values]
            if hasattr(values, "__iter__") and values and hasattr(next(iter(values)), "item")
            else [int(item) for item in values]
        )
        for user, values in rows.items()
    }
    metrics = evaluate_recommendations(recommendations, targets, ks)
    metrics["Coverage"] = catalog_coverage(recommendations, data.num_items)
    metrics = {key: round(float(value), 6) for key, value in metrics.items()}
    target_count = len(targets)
    return {
        "ok": True,
        "stage": stage,
        "source": source,
        "split": split,
        "users": len(recommendations),
        "targets": target_count,
        "ks": list(ks),
        "metrics": metrics,
        "targets_per_user": round(target_count / max(1, len(recommendations)), 3),
    }


def _run_pipeline(context: ToolContext, arguments: dict) -> dict:
    from dataclasses import replace

    from recsys.pipeline.chain import RecommendationPipeline
    from recsys.pipeline.config import (
        FusionConfig,
        PipelineConfig,
        RankingConfig,
        RecallChannelConfig,
        RerankConfig,
        RunConfig,
    )

    models = _resolve_recall_models(arguments)
    ranker = arguments.get("ranker", "lightgbm_ranker")
    if ranker not in RANKER_MODELS:
        raise ToolCallError(f"未知排序模型 {ranker!r}")
    split = arguments.get("split", context.split)
    history_setting = context.history_setting(arguments)
    users = context.users
    run = RunConfig(
        dataset=context.dataset,
        data_dir=Path(context.data_dir),
        artifacts_dir=Path(context.artifacts_dir),
        output_name=arguments.get("output_name", "pipeline"),
        split=split,
        users=min(users, 256),
        seed=context.seed,
        k_recall=int(arguments.get("k_recall", 100)),
        k_final=int(arguments.get("k_final", 20)),
        include_validation_in_test_history=history_setting,
        device=context.device,
    )
    channels = []
    for model in models:
        checkpoint = resolve_checkpoint(context, model)
        if checkpoint is None and model != "popularity":
            return {
                "ok": False,
                "error": f"召回模型 {model} 缺少 checkpoint；先调用 train_model",
                "missing_model": model,
            }
        channels.append(
            RecallChannelConfig(name=model, model=model, checkpoint=checkpoint)
        )
    ranker_checkpoint, ranker_error = _resolve_ranker_for_run(context, ranker, arguments)
    if ranker_error is not None:
        return ranker_error
    trained_lightgbm = bool(
        str(arguments.get("artifact_policy") or "train_if_missing") != "existing_only"
        and ranker == "lightgbm_ranker"
        and ranker_checkpoint is not None
        and not Path(ranker_checkpoint).exists()
    )
    existing_only = str(arguments.get("artifact_policy") or "train_if_missing") == "existing_only"
    if trained_lightgbm and ranker_checkpoint is not None:
        # LightGBM 随链路现训时也不能写业务默认路径。
        ranker_checkpoint = (
            Path(context.experiment_dir or context.artifacts_dir)
            / "training"
            / f"lightgbm-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
            / context.dataset
            / "lightgbm_ranker.txt"
        )
    ranker_sha256 = None
    if ranker_checkpoint is not None and Path(ranker_checkpoint).exists():
        ranker_sha256 = _sha256_file(Path(ranker_checkpoint))
    config = PipelineConfig(
        run=run,
        recall=tuple(channels),
        fusion=FusionConfig(
            mode=arguments.get("fusion", "round_robin"),
            limit=run.k_recall,
            quota=dict(arguments.get("quota") or {}),
        ),
        ranking=RankingConfig(
            model=ranker,
            checkpoint=ranker_checkpoint,
            # Phase 5：existing_only 时只加载已有权重，缺失由上面显式失败。
            train_if_missing=not existing_only,
            num_boost_round=int(arguments.get("num_boost_round", 10)),
            max_rows=int(arguments.get("max_rows", 4000)),
            parent_ranking=Path(context.data_dir) / "kuasar" / "processed" / "ranking",
        ),
        rerank=RerankConfig(
            category_cap=int(arguments.get("category_cap", 5)),
            author_cap=int(arguments.get("author_cap", 0)),
            diversity=bool(arguments.get("diversity", True)),
        ),
    )
    output_dir = Path(context.experiment_dir or context.artifacts_dir) / "pipeline" / arguments.get(
        "output_name", "pipeline"
    )
    try:
        pipeline = RecommendationPipeline(config)
        result = pipeline.run(split=split, write=True, output_dir=output_dir)
    except ArtifactError as error:
        return {
            "ok": False,
            "error": str(error),
            "missing_artifact": error.path,
            "artifact_policy": str(arguments.get("artifact_policy") or "train_if_missing"),
            "auto_train_attempted": False,
        }
    if trained_lightgbm and ranker_checkpoint is not None and Path(ranker_checkpoint).exists():
        context.state.setdefault("trained_checkpoints", {})[
            "lightgbm_ranker"
        ] = str(Path(ranker_checkpoint))
    context.state["channels"] = result.channels
    context.state["fused"] = result.fused
    context.state["ranked"] = result.ranked
    context.state["final"] = result.final
    context.state["last_pipeline"] = result
    return {
        "ok": True,
        "dataset": context.dataset,
        "split": split,
        "recall_models": models,
        "ranker": ranker,
        "include_validation_in_test_history": history_setting,
        "artifact_policy": str(arguments.get("artifact_policy") or "train_if_missing"),
        "ranker_checkpoint": _relative(Path(ranker_checkpoint)) if ranker_checkpoint else None,
        "ranker_checkpoint_sha256": ranker_sha256,
        "trained_ranker": bool(trained_lightgbm),
        "metrics": result.metrics,
        "diagnostics": {
            key: value
            for key, value in result.diagnostics.items()
            if key != "target_details"
        },
        "timings": result.timings,
        "artifacts": {
            "output_dir": _relative(output_dir),
            "final": _relative(output_dir / "final.json"),
            "metrics": _relative(output_dir / "metrics.json"),
            "run_meta": _relative(output_dir / "run_meta.json"),
        },
    }


# --------------------------------------------------------------------------- 既有实验平台复用


def _history(context: ToolContext, arguments: dict) -> dict:
    from recsys.experiments.agent_tools import _history_execute, ToolContext as LegacyContext
    from recsys.experiments.experiment_history import DEFAULT_HISTORY_ROOT

    legacy = LegacyContext(
        cache_dir=Path(context.artifacts_dir) / "kuasar_small" / "mvp",
        output_dir=Path(context.artifacts_dir) / "kuasar_small" / "rec_agent" / "cache_eval",
        history_root=DEFAULT_HISTORY_ROOT,
    )
    return _history_execute(
        {
            "two_tower_quota": int(arguments.get("two_tower_quota", 50)),
            "mind_quota": int(arguments.get("mind_quota", 50)),
        },
        legacy,
    )


def _quota_eval(context: ToolContext, arguments: dict) -> dict:
    """复用既有 fusion_quota：只读双塔/MIND 缓存候选，按配额重算指标。"""
    from recsys.experiments.fusion_quota import QuotaError, evaluate_fusion_quota, validate_quota

    two = int(arguments["two_tower_quota"])
    mind = int(arguments["mind_quota"])
    try:
        validate_quota(two, mind)
    except QuotaError as error:
        raise ToolCallError(str(error)) from error
    cache_dir = Path(context.artifacts_dir) / "kuasar_small" / "mvp"
    output_dir = Path(context.experiment_dir or context.artifacts_dir) / "cache_eval"
    result = evaluate_fusion_quota(
        two,
        mind,
        cache_dir=cache_dir,
        output_dir=output_dir,
        tag=f"{two}_{mind}",
    )
    return {"ok": True, **result}


def _list_experiments(context: ToolContext, arguments: dict) -> dict:
    root = PROJECT_ROOT / "experiments"
    entries = []
    if root.exists():
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            entries.append(
                {
                    "name": path.name,
                    "has_result": (path / "agent_result.json").exists(),
                    "has_metrics": (path / "metrics.json").exists(),
                    "files": len(list(path.glob("*.json"))),
                }
            )
    return {"ok": True, "count": len(entries), "experiments": entries[-50:]}


def _knowledge(context: ToolContext, arguments: dict) -> dict:
    """Phase 4：检索本地项目知识库；只读，不影响推荐链路。"""

    from recsys.platform.rag import retrieve_knowledge

    return retrieve_knowledge(
        str(arguments["query"]),
        top_k=int(arguments.get("top_k", 3)),
        index_dir=context.state.get("rag_index_dir"),
    )


# --------------------------------------------------------------------------- 组装


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "read_data_profile",
            "读取当前数据集画像：用户/物品/切分规模、留出目标、物品特征列、排序候选是否可用、各模型 checkpoint。",
            _object({}),
            _profile,
        )
    )
    registry.register(
        ToolSpec(
            "query_model_zoo",
            "查询 models/zoo.toml：模型能力、依赖、训练入口、完整度与逐项状态（runnable/integrated/agent_callable 等）。",
            _object(
                {
                    "category": {
                        "type": "string",
                        "enum": ["retrieval", "ranking", "infrastructure"],
                    },
                    "status": {"type": "string"},
                }
            ),
            _zoo,
        )
    )
    registry.register(
        ToolSpec(
            "list_artifacts",
            "列出当前数据集下已有的 checkpoint、索引与运行工件。",
            _object({}),
            _artifacts,
        )
    )
    registry.register(
        ToolSpec(
            "train_model",
            "调用真实训练入口训练一个模型（受预算限制：默认 1 epoch / 2 batch，最多 3 epoch / 20 batch）。",
            _object(
                {
                    "model": {"type": "string", "enum": list(TRAINABLE_MODELS)},
                    "epochs": {"type": "integer", "minimum": 1, "maximum": 3},
                    "max_batches": {"type": "integer", "minimum": 1, "maximum": 20},
                    "embedding_dim": {"type": "integer", "minimum": 4, "maximum": 64},
                    "max_rows": {"type": "integer", "minimum": 100, "maximum": 20000},
                    "num_boost_round": {"type": "integer", "minimum": 1, "maximum": 50},
                    "validation_users": {"type": "integer", "minimum": 0, "maximum": 256},
                    "seed": {"type": "integer", "minimum": 0, "maximum": 100000},
                },
                required=["model"],
            ),
            _train,
        )
    )
    registry.register(
        ToolSpec(
            "recall_models",
            "调用一个或多个真实召回模型，返回用户候选、通道内名次与模型分数，并落盘候选工件。",
            _object(
                {
                    "models": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(RECALL_MODELS)},
                    },
                    "k": {"type": "integer", "minimum": 1, "maximum": 200},
                    "split": {"type": "string", "enum": ["validation", "test"]},
                    "include_validation_in_test_history": {
                        "type": "boolean",
                        "description": "test 时是否把验证历史并入模型输入；validation 始终只用训练历史",
                    },
                },
                required=["models"],
            ),
            _recall,
        )
    )
    registry.register(
        ToolSpec(
            "rank_candidates",
            "对当前会话已召回的候选做真实排序模型推理（LightGBM/DIN/DIEN/DCNv2/Ultra-HSTU），保留模型分数。",
            _object(
                {
                    "model": {"type": "string", "enum": list(RANKER_MODELS)},
                    "fusion": {"type": "string", "enum": ["round_robin", "rrf", "quota"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                    "include_validation_in_test_history": {
                        "type": "boolean",
                        "description": "读取用户历史的神经排序模型沿用该设置构造历史",
                    },
                },
                required=["model"],
            ),
            _rank,
        )
    )
    registry.register(
        ToolSpec(
            "rerank_candidates",
            "对当前排序结果做类目/作者多样性重排，返回最终推荐与补位次数。",
            _object(
                {
                    "k": {"type": "integer", "minimum": 1, "maximum": 100},
                    "category_cap": {"type": "integer", "minimum": 1, "maximum": 50},
                    "author_cap": {"type": "integer", "minimum": 0, "maximum": 20},
                }
            ),
            _rerank,
        )
    )
    registry.register(
        ToolSpec(
            "evaluate_recommendations",
            "按指定数据切分计算真实指标（Recall/NDCG/HitRate/Precision/Coverage），可评估 final/ranked/recall 阶段。",
            _object(
                {
                    "stage": {"type": "string", "enum": ["final", "ranked", "recall"]},
                    "model": {"type": "string", "enum": list(RECALL_MODELS)},
                    "split": {"type": "string", "enum": ["validation", "test"]},
                    "ks": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 200}},
                }
            ),
            _evaluate,
        )
    )
    registry.register(
        ToolSpec(
            "run_pipeline",
            "一次跑完真实推荐链路：多路召回→去重融合→排序→类目/多样性重排→按切分评估，并落盘全部工件。",
            _object(
                {
                    "models": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(RECALL_MODELS)},
                    },
                    "ranker": {"type": "string", "enum": list(RANKER_MODELS)},
                    "fusion": {"type": "string", "enum": ["round_robin", "rrf", "quota"]},
                    "quota": {"type": "object"},
                    "split": {"type": "string", "enum": ["validation", "test"]},
                    "include_validation_in_test_history": {
                        "type": "boolean",
                        "description": "test 时是否使用训练+验证历史；False 只用训练历史。默认 True",
                    },
                    "k_recall": {"type": "integer", "minimum": 10, "maximum": 200},
                    "k_final": {"type": "integer", "minimum": 1, "maximum": 100},
                    "category_cap": {"type": "integer", "minimum": 1, "maximum": 50},
                    "author_cap": {"type": "integer", "minimum": 0, "maximum": 20},
                    "num_boost_round": {"type": "integer", "minimum": 1, "maximum": 50},
                    "max_rows": {"type": "integer", "minimum": 100, "maximum": 20000},
                    "output_name": {"type": "string"},
                    "artifact_policy": {
                        "type": "string",
                        "enum": ["existing_only", "train_if_missing"],
                        "description": (
                            "existing_only 只加载已有排序权重（缺失即失败，不训练）；"
                            "默认 train_if_missing 保留原有显式训练行为"
                        ),
                    },
                    "ranker_checkpoint": {
                        "type": "string",
                        "description": "显式指定排序权重文件；相对路径按仓库根目录解析",
                    },
                    "expected_ranker_sha256": {
                        "type": "string",
                        "description": "期望的排序权重 sha256；不匹配即失败",
                    },
                },
                required=["models"],
            ),
            _run_pipeline,
        )
    )
    registry.register(
        ToolSpec(
            "evaluate_fusion_quota",
            "复用既有融合配额评测：只读双塔/MIND 缓存候选，按配额重算命中数/Recall@100/Coverage，不训练、不重新召回。",
            _object(
                {
                    "two_tower_quota": {"type": "integer", "minimum": 0, "maximum": 100},
                    "mind_quota": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                required=["two_tower_quota", "mind_quota"],
            ),
            _quota_eval,
        )
    )
    registry.register(
        ToolSpec(
            "search_experiment_history",
            "复用既有融合配额实验历史查询（只读，不重跑评测）。",
            _object(
                {
                    "two_tower_quota": {"type": "integer", "minimum": 0, "maximum": 100},
                    "mind_quota": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                required=["two_tower_quota", "mind_quota"],
            ),
            _history,
        )
    )
    registry.register(
        ToolSpec(
            "retrieve_knowledge",
            "检索本地项目知识库（项目文档、模型说明、开源来源、历史实验报告），返回带 path:start-end 的来源片段。只用于查背景/实现，不能代替实验校验。",
            _object(
                {
                    "query": {"type": "string", "minLength": 1},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                required=["query"],
            ),
            _knowledge,
        )
    )
    registry.register(
        ToolSpec(
            "list_experiments",
            "列出 experiments/ 下的历史实验目录及其结果文件。",
            _object({}),
            _list_experiments,
        )
    )
    return registry


__all__ = [
    "CHECKPOINT_CANDIDATES",
    "PROJECT_ROOT",
    "RANKER_MODELS",
    "RECALL_MODELS",
    "TRAINABLE_MODELS",
    "ToolBudget",
    "ToolBudgetExceeded",
    "ToolCallError",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "build_registry",
    "resolve_checkpoint",
    "resolve_ranker_checkpoint",
]
