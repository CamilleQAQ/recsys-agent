"""Phase 11 工具注册表：集中维护工具名、参数 schema、校验与执行函数。

两个工具：

* ``search_experiment_history``：按目标配额 + 当前缓存指纹查已落盘历史；
* ``evaluate_fusion_quota``：复用现有融合配额评测（只读缓存、不训练、不重新召回）。

约束：

* 参数只有 ``two_tower_quota`` / ``mind_quota``，禁止路径、命令或自定义脚本；
* 参数必须恰好是这两个键、必须是整数、必须来自 ``ALLOWED_QUOTAS``；
* LangGraph 只通过本注册表执行工具，未知工具或未注册参数直接拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from recsys.experiments.experiment_history import (
    DEFAULT_HISTORY_ROOT,
    search_experiment_history,
)
from recsys.experiments.fusion_quota import (
    QuotaError,
    evaluate_fusion_quota,
    validate_quota,
)

TOOL_SEARCH_HISTORY = "search_experiment_history"
TOOL_EVALUATE = "evaluate_fusion_quota"

QUOTA_KEYS = ("two_tower_quota", "mind_quota")


class ToolCallError(ValueError):
    """工具名、参数或阶段不合法。"""


@dataclass(frozen=True)
class ToolContext:
    """程序从 CLI 状态注入的本地路径；模型不能传这些值。"""

    cache_dir: Path
    output_dir: Path
    history_root: Path = DEFAULT_HISTORY_ROOT
    memory_dir: Path | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    execute: Callable[[dict, ToolContext], dict]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def quota_parameters(description: str) -> dict:
    return {
        "type": "object",
        "properties": {
            "two_tower_quota": {
                "type": "integer",
                "description": "双塔配额（0-100），与 mind_quota 之和必须为 100",
            },
            "mind_quota": {
                "type": "integer",
                "description": "MIND 配额（0-100），与 two_tower_quota 之和必须为 100",
            },
        },
        "required": list(QUOTA_KEYS),
        "additionalProperties": False,
    }


class ToolRegistry:
    """小型注册表；执行前统一校验，执行时只调用注册过的函数。"""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ToolCallError(f"重复注册工具：{spec.name}")
        self._specs[spec.name] = spec

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise ToolCallError(f"未注册的工具：{name}") from error

    def schema(self, name: str) -> dict:
        return self.get(name).schema()

    def schemas(self, names: tuple[str, ...]) -> list[dict]:
        return [self.schema(name) for name in names]

    def validate(self, name: str, arguments: Any) -> dict:
        """只接受两个整数配额；多余键（路径/命令/脚本）一律拒绝。"""
        self.get(name)
        if not isinstance(arguments, dict):
            raise ToolCallError("工具参数必须是 JSON 对象")
        if set(arguments) != set(QUOTA_KEYS):
            extra = sorted(set(arguments) - set(QUOTA_KEYS))
            missing = sorted(set(QUOTA_KEYS) - set(arguments))
            raise ToolCallError(f"工具参数不合法：多 {extra}，少 {missing}")
        for key in QUOTA_KEYS:
            value = arguments[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolCallError(f"工具参数 {key} 必须是整数")
        try:
            quota = validate_quota(arguments["two_tower_quota"], arguments["mind_quota"])
        except QuotaError as error:
            raise ToolCallError(str(error)) from error
        return {"two_tower_quota": quota[0], "mind_quota": quota[1]}

    def execute(self, name: str, arguments: dict, *, context: ToolContext) -> dict:
        validated = self.validate(name, arguments)
        return self.get(name).execute(validated, context)


def _history_execute(arguments: dict, context: ToolContext) -> dict:
    return search_experiment_history(
        arguments["two_tower_quota"],
        arguments["mind_quota"],
        cache_dir=context.cache_dir,
        history_root=context.history_root,
        memory_dir=context.memory_dir,
    )


def _evaluate_execute(arguments: dict, context: ToolContext) -> dict:
    two = arguments["two_tower_quota"]
    mind = arguments["mind_quota"]
    return evaluate_fusion_quota(
        two,
        mind,
        cache_dir=context.cache_dir,
        output_dir=context.output_dir,
        tag=f"{two}_{mind}",
    )


def build_registry(
    *,
    history_execute: Callable[[dict, ToolContext], dict] = _history_execute,
    evaluate_execute: Callable[[dict, ToolContext], dict] = _evaluate_execute,
) -> ToolRegistry:
    """默认真实工具；本地模拟可替换 execute（schema/校验不变）。"""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=TOOL_SEARCH_HISTORY,
            description=(
                "按目标融合配额查询已落盘的历史实验：核对数据集/验证切分、固定用户和候选缓存"
                "文件哈希、每用户候选预算、融合规则与指标口径。命中返回聚合指标和 50/50 基准；"
                "未命中说明原因。不会重跑评测。"
            ),
            parameters=quota_parameters("目标配额"),
            execute=history_execute,
        )
    )
    registry.register(
        ToolSpec(
            name=TOOL_EVALUATE,
            description=(
                "在固定的 128 个验证用户、每用户 100 个候选的预算下，评测双塔与 MIND 的"
                "融合配额，返回命中目标数、Recall@100、Coverage 等真实指标。只读缓存，"
                "不训练、不重新召回。"
            ),
            parameters=quota_parameters("待评测配额"),
            execute=evaluate_execute,
        )
    )
    return registry


__all__ = [
    "QUOTA_KEYS",
    "ToolCallError",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "TOOL_EVALUATE",
    "TOOL_SEARCH_HISTORY",
    "build_registry",
    "quota_parameters",
]
