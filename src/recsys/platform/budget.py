"""Phase 4 统一任务预算。

一个研究任务只有一个 :class:`TaskBudget`。总控（Research Agent）和它调用的
三个业务 Agent 都从同一个对象扣减：

* ``llm``：真实 LLM 请求次数（脚本替身决策不计入）；
* ``tool``：工具调用次数（含总控工具与子 Agent 内部工具）；
* ``experiment``：真正执行的新实验次数；
* ``history``：历史实验查询次数；
* ``artifact``：读取已有结果/工件次数；
* ``seconds``：总时限。

子 Agent 通过 :meth:`TaskBudget.child` 得到带更小上限的 :class:`BudgetScope`，
scope 的每次扣减都会同步到父预算；启动子任务时用父预算的剩余额度推导内部上限，
因此不可能重置或绕过总额度。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

KINDS: tuple[str, ...] = ("llm", "tool", "experiment", "history", "artifact")

KIND_LABELS = {
    "llm": "LLM 请求",
    "tool": "工具调用",
    "experiment": "新实验",
    "history": "历史查询",
    "artifact": "已有结果读取",
    "seconds": "时间",
}

KIND_LIMIT_FIELDS = {
    "llm": "max_llm_requests",
    "tool": "max_tool_calls",
    "experiment": "max_experiments",
    "history": "max_history_lookups",
    "artifact": "max_artifact_reads",
}


class BudgetExceeded(RuntimeError):
    """预算不足；``kind`` 与 ``snapshot`` 用于状态记录和验收。"""

    def __init__(
        self,
        kind: str,
        reason: str,
        *,
        scope: str = "task",
        snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.kind = kind
        self.reason = reason
        self.scope = scope
        self.snapshot = dict(snapshot or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "scope": self.scope,
            "budget": self.snapshot,
        }


@dataclass
class BudgetPolicy:
    """一份预算上限。子预算沿用同一字段，但数值更小。"""

    max_llm_requests: int = 8
    max_tool_calls: int = 8
    max_experiments: int = 3
    max_history_lookups: int = 3
    max_artifact_reads: int = 3
    max_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "max_llm_requests",
            "max_tool_calls",
            "max_experiments",
            "max_history_lookups",
            "max_artifact_reads",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} 不能为负：{value}")
            setattr(self, name, value)
        self.max_seconds = max(0.0, float(self.max_seconds))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None, *, base: "BudgetPolicy | None" = None) -> "BudgetPolicy":
        if isinstance(payload, BudgetPolicy):
            return payload
        base = base or cls()
        payload = dict(payload or {})
        values: dict[str, Any] = {}
        for name in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            values[name] = payload.get(name, getattr(base, name))
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_llm_requests": self.max_llm_requests,
            "max_tool_calls": self.max_tool_calls,
            "max_experiments": self.max_experiments,
            "max_history_lookups": self.max_history_lookups,
            "max_artifact_reads": self.max_artifact_reads,
            "max_seconds": round(self.max_seconds, 3),
        }

    def tighten(self, remaining: Mapping[str, float]) -> "BudgetPolicy":
        """把预算收紧到父预算的剩余额度（子任务不得重置总额度）。"""

        return BudgetPolicy(
            max_llm_requests=int(min(self.max_llm_requests, remaining.get("llm", 0))),
            max_tool_calls=int(min(self.max_tool_calls, remaining.get("tool", 0))),
            max_experiments=int(min(self.max_experiments, remaining.get("experiment", 0))),
            max_history_lookups=int(min(self.max_history_lookups, remaining.get("history", 0))),
            max_artifact_reads=int(min(self.max_artifact_reads, remaining.get("artifact", 0))),
            max_seconds=float(min(self.max_seconds, remaining.get("seconds", 0.0))),
        )


class _CounterMixin:
    policy: BudgetPolicy
    counters: dict[str, int]
    refusals: list[dict[str, Any]]

    @property
    def elapsed(self) -> float:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    def _limit(self, kind: str) -> float:
        if kind == "seconds":
            return float(self.policy.max_seconds)
        field_name = KIND_LIMIT_FIELDS.get(kind)
        if field_name is None:
            raise KeyError(f"未知预算类型：{kind}")
        return float(getattr(self.policy, field_name))

    def _used(self, kind: str) -> float:
        if kind == "seconds":
            return float(self.elapsed)
        return float(self.counters.get(kind, 0))

    def remaining(self, kind: str) -> float:
        return max(0.0, self._limit(kind) - self._used(kind))

    def remaining_all(self) -> dict[str, float]:
        values = {kind: self.remaining(kind) for kind in KINDS}
        values["seconds"] = self.remaining("seconds")
        return values


class TaskBudget(_CounterMixin):
    """单个研究任务的全局预算。"""

    def __init__(
        self,
        task_id: str,
        policy: BudgetPolicy | None = None,
        *,
        started: float | None = None,
        clock=time.perf_counter,
    ) -> None:
        self.task_id = str(task_id)
        self.policy = policy or BudgetPolicy()
        self.counters = {kind: 0 for kind in KINDS}
        self.refusals: list[dict[str, Any]] = []
        self._clock = clock
        self._started = float(started if started is not None else clock())

    @property
    def elapsed(self) -> float:
        return max(0.0, self._clock() - self._started)

    def _refuse(self, kind: str, reason: str, *, scope: str = "task") -> None:
        self.refusals.append(
            {
                "kind": kind,
                "reason": reason,
                "scope": scope,
                "remaining": round(self.remaining(kind), 3)
                if kind != "seconds"
                else round(max(0.0, self.remaining("seconds")), 3),
                "at": round(self.elapsed, 3),
            }
        )

    def check(self, kind: str, *, scope: str = "task") -> None:
        if kind == "seconds":
            if self.remaining("seconds") <= 0:
                reason = f"总时限 {self.policy.max_seconds:.1f}s 已耗尽"
                self._refuse("seconds", reason, scope=scope)
                raise BudgetExceeded("seconds", reason, scope=scope, snapshot=self.snapshot())
            return
        field_name = KIND_LIMIT_FIELDS.get(kind)
        if field_name is None:
            raise KeyError(f"未知预算类型：{kind}")
        limit = int(getattr(self.policy, field_name))
        used = int(self.counters.get(kind, 0))
        if used >= limit:
            reason = f"{KIND_LABELS[kind]}已用完（{used}/{limit}）"
            self._refuse(kind, reason, scope=scope)
            raise BudgetExceeded(kind, reason, scope=scope, snapshot=self.snapshot())

    def spend(self, kind: str, amount: int = 1, *, scope: str = "task") -> None:
        if amount < 0:
            raise ValueError("amount 不能为负")
        if amount == 0:
            return
        self.check(kind, scope=scope)
        if kind != "seconds":
            limit = int(getattr(self.policy, KIND_LIMIT_FIELDS[kind]))
            used = int(self.counters.get(kind, 0))
            if used + int(amount) > limit:
                reason = (
                    f"{KIND_LABELS[kind]}额度不足：已用 {used}，"
                    f"本次需要 {amount}，上限 {limit}"
                )
                self._refuse(kind, reason, scope=scope)
                raise BudgetExceeded(kind, reason, scope=scope, snapshot=self.snapshot())
        self.counters[kind] = int(self.counters.get(kind, 0)) + int(amount)

    def refund(self, kind: str, amount: int = 1) -> None:
        """撤回预扣；只在确认没有实际消耗时使用。"""

        if kind not in self.counters:
            raise KeyError(kind)
        self.counters[kind] = max(0, int(self.counters[kind]) - max(0, int(amount)))

    def child(self, name: str, policy: BudgetPolicy | Mapping[str, Any] | None = None) -> "BudgetScope":
        if isinstance(policy, BudgetPolicy):
            base = policy
        elif policy is None:
            base = self.policy
        else:
            base = BudgetPolicy.from_mapping(policy, base=self.policy)
        # 关键：子预算永远不能超过父预算当前剩余额度，启动子任务不会重置总额度。
        tight = base.tighten(self.remaining_all())
        return BudgetScope(parent=self, name=str(name), policy=tight)

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "policy": self.policy.as_dict(),
            "counters": dict(self.counters),
            "remaining": {
                **{kind: int(round(self.remaining(kind))) for kind in KINDS},
                "seconds": round(self.remaining("seconds"), 2),
            },
            "elapsed_s": round(self.elapsed, 2),
            "refusals": list(self.refusals),
        }


class BudgetScope(_CounterMixin):
    """子 Agent 的预算视图：更小上限 + 同一父预算。"""

    def __init__(self, *, parent: TaskBudget, name: str, policy: BudgetPolicy) -> None:
        self.parent = parent
        self.name = str(name)
        self.policy = policy
        self.counters = {kind: 0 for kind in KINDS}
        self.refusals: list[dict[str, Any]] = []
        self._started_at = parent.elapsed

    @property
    def elapsed(self) -> float:
        return self.parent.elapsed - self._started_at

    def remaining(self, kind: str) -> float:
        own = super().remaining(kind)
        return max(0.0, min(own, self.parent.remaining(kind)))

    def check(self, kind: str) -> None:
        # 先看父预算（全局总额度），再看子预算的更小上限。
        try:
            self.parent.check(kind, scope=self.name)
        except BudgetExceeded as error:
            self.refusals.append({"kind": kind, "reason": error.reason, "scope": self.name})
            raise
        if kind == "seconds":
            if self.remaining("seconds") <= 0:
                reason = f"子任务 {self.name} 时间预算耗尽"
                self.refusals.append({"kind": "seconds", "reason": reason, "scope": self.name})
                raise BudgetExceeded("seconds", reason, scope=self.name, snapshot=self.snapshot())
            return
        limit = int(getattr(self.policy, KIND_LIMIT_FIELDS[kind]))
        used = int(self.counters.get(kind, 0))
        if used >= limit:
            reason = f"子任务 {self.name} 的{KIND_LABELS[kind]}已用完（{used}/{limit}）"
            self.refusals.append({"kind": kind, "reason": reason, "scope": self.name})
            raise BudgetExceeded(kind, reason, scope=self.name, snapshot=self.snapshot())

    def spend(self, kind: str, amount: int = 1) -> None:
        if amount <= 0:
            self.check(kind)
            return
        self.check(kind)
        if kind != "seconds":
            limit = int(getattr(self.policy, KIND_LIMIT_FIELDS[kind]))
            used = int(self.counters.get(kind, 0))
            if used + int(amount) > limit:
                reason = (
                    f"子任务 {self.name} 的{KIND_LABELS[kind]}额度不足："
                    f"已用 {used}，本次需要 {amount}，上限 {limit}"
                )
                self.refusals.append({"kind": kind, "reason": reason, "scope": self.name})
                raise BudgetExceeded(kind, reason, scope=self.name, snapshot=self.snapshot())
            parent_limit = int(getattr(self.parent.policy, KIND_LIMIT_FIELDS[kind]))
            parent_used = int(self.parent.counters.get(kind, 0))
            if parent_used + int(amount) > parent_limit:
                reason = (
                    f"全局{KIND_LABELS[kind]}额度不足：已用 {parent_used}，"
                    f"本次需要 {amount}，上限 {parent_limit}"
                )
                self.refusals.append({"kind": kind, "reason": reason, "scope": self.name})
                raise BudgetExceeded(kind, reason, scope=self.name, snapshot=self.snapshot())
        self.parent.counters[kind] = int(self.parent.counters.get(kind, 0)) + int(amount)
        self.counters[kind] = int(self.counters.get(kind, 0)) + int(amount)

    def snapshot(self) -> dict[str, Any]:
        return {
            "scope": self.name,
            "task_id": self.parent.task_id,
            "policy": self.policy.as_dict(),
            "counters": dict(self.counters),
            "global": self.parent.snapshot(),
        }


__all__ = [
    "BudgetExceeded",
    "BudgetPolicy",
    "BudgetScope",
    "KINDS",
    "KIND_LABELS",
    "TaskBudget",
]
