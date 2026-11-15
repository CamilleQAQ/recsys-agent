"""Phase 4 小样本验收与自检。

验收只覆盖 PROJECT_MASTER_v2 Phase 4 要求的必要分支：

1. 已有推荐实验历史命中 → 直接复用，新实验 0 次；
2. 历史未命中 → 真实工具返回 miss 后补一次小样本业务检查；另验预算耗尽停止；
3. 三个业务入口可由总控调用；跨两个业务进入同一任务记录；另起进程能查回；
   RAG 引用可定位；资料更新后可检索；
4. 有 LLM 配置时只跑一次真实 LLM 研究任务；配置不可用时记录具体卡点，
   离线替身与真实 LLM 状态分开标记。

不训练、不重建大索引、不跑全量、不调参、不覆盖现有权重。
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from recsys.platform.adapters import run_business
from recsys.platform.budget import (
    BudgetExceeded,
    BudgetPolicy,
    TaskBudget,
)
from recsys.platform.rag import RagIndex, load_rag_config
from recsys.platform.research_agent import (
    ResearchConfig,
    ResearchAgent,
    TOOLS,
)
from recsys.platform.shared_experiments import (
    REPO_ROOT,
    SharedExperimentStore,
    build_record,
    lookup_history,
    model_registry_view,
    query_records,
    reuse_reasons,
)

RAG_CONFIG = Path("configs/platform/rag.toml")
RAG_INDEX_DIR = Path("artifacts/rag/phase4")
UPDATE_NOTE = Path("docs/phase4/rag-update-note.md")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_json_subprocess(arguments: list[str], *, timeout: float = 180.0) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "recsys.platform", *arguments],
        cwd=str(REPO_ROOT),
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"子进程失败（exit={completed.returncode}）："
            f"{(completed.stderr or completed.stdout)[-800:]}"
        )
    return json.loads(completed.stdout)


def _fresh_config(base: ResearchConfig, root: Path, store: SharedExperimentStore) -> ResearchConfig:
    config = copy.deepcopy(base)
    config.experiment_root = root
    config.store_dir = store.root
    config.subagent_intent.setdefault("recommendation", {})
    config.subagent_intent["recommendation"].update(
        {"users": 4, "recall_models": ["popularity"], "k_recall": 20, "k_final": 10}
    )
    config.subagent_intent.setdefault("search", {})
    config.subagent_intent["search"].update({"mode": "bm25", "session_id": "11208"})
    config.subagent_intent.setdefault("ads", {})
    config.subagent_intent["ads"].update(
        {"model": "din", "split": "test", "requests": 2, "limit": 200, "simulate_auction": True}
    )
    return config


def _task_records(store: SharedExperimentStore, task_id: str) -> dict[str, Any]:
    result = query_records(store, task_id=task_id, reusable=None, limit=50)
    return result


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    fn: Callable[[], dict[str, Any]],
    *,
    required: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        detail = fn()
        ok = bool(detail.pop("ok", True))
    except Exception as error:  # noqa: BLE001 - 单项失败不阻塞其他验收
        detail = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        ok = False
    record = {
        "id": check_id,
        "required": required,
        "ok": ok,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "detail": detail,
    }
    checks.append(record)
    return record


# ---------------------------------------------------------------------- 验收项


def _check_history_hit(
    base: ResearchConfig, root: Path, store: SharedExperimentStore
) -> dict[str, Any]:
    config = _fresh_config(base, root / "history_hit", store)
    config.budget.max_experiments = 0
    config.budget.max_tool_calls = 4
    summary = ResearchAgent(config, planner="scripted", store=store).run(
        "查一下已有 30/70 融合配额推荐实验，校验通过就直接复用，不要重跑"
    )
    history = summary.get("history") or {}
    ok = (
        history.get("hit") is True
        and summary.get("status") == "verified"
        and summary.get("reuse_mode") == "history"
        and summary.get("new_experiments") == 0
        and not summary.get("businesses")
        and bool(history.get("metrics"))
    )
    return {
        "ok": ok,
        "history_status": history.get("status"),
        "history_source": history.get("source"),
        "history_metrics": history.get("metrics"),
        "new_experiments": summary.get("new_experiments"),
        "stop_reason": summary.get("stop_reason"),
        "task_id": summary.get("task_id"),
        "summary_path": (summary.get("artifacts") or {}).get("summary"),
        "command": 'python -m recsys.platform --task "查一下已有 30/70 融合配额推荐实验..." --planner scripted',
    }


def _check_history_miss(
    base: ResearchConfig, root: Path, store: SharedExperimentStore
) -> dict[str, Any]:
    config = _fresh_config(base, root / "history_miss", store)
    config.budget.max_experiments = 2
    config.budget.max_tool_calls = 8
    summary = ResearchAgent(config, planner="scripted", store=store).run(
        "查 50/50 融合配额历史，没有可复用结果就补做一次最小推荐检查"
    )
    history = summary.get("history") or {}
    records = _task_records(store, str(summary.get("task_id")))
    business_records = [item for item in records["matched"] if item.get("business") == "recommendation"]
    task_records = query_records(
        store, task_id=str(summary.get("task_id")), include_tasks=True, reusable=None, limit=20
    )["matched"]
    trace_path = (summary.get("artifacts") or {}).get("trace")
    decisions = summary.get("decisions") or []
    ok = (
        history.get("hit") is False
        and bool(history.get("reason"))
        and summary.get("new_experiments") == 1
        and summary.get("status") == "verified"
        and len(business_records) == 1
        and any(item.get("business") == "task" for item in task_records)
        and len(decisions) >= 2
    )
    return {
        "ok": ok,
        "history_status": history.get("status"),
        "history_reason": history.get("reason"),
        "new_experiments": summary.get("new_experiments"),
        "businesses": summary.get("businesses"),
        "record_ids": summary.get("record_ids"),
        "task_id": summary.get("task_id"),
        "decision_count": len(decisions),
        "decisions": decisions,
        "offline_trace": trace_path,
    }


def _check_artifact_read(
    base: ResearchConfig, root: Path, store: SharedExperimentStore, record_id: str | None
) -> dict[str, Any]:
    if not record_id:
        raise RuntimeError("没有可读取的记录 ID")
    config = _fresh_config(base, root / "artifact_read", store)
    config.budget.max_experiments = 0
    config.budget.max_tool_calls = 3
    summary = ResearchAgent(config, planner="scripted", store=store).run(
        "读取记录内容并给出摘要",
        intent={"read_record": record_id, "history_first": False},
    )
    counters = (summary.get("budget") or {}).get("counters") or {}
    tools = [item.get("tool") for item in summary.get("tool_results") or []]
    ok = (
        summary.get("status") == "verified"
        and tools == ["read_experiment_result"]
        and counters.get("artifact") == 1
        and counters.get("experiment") == 0
        and counters.get("tool") == 1
    )
    return {
        "ok": ok,
        "record_id": record_id,
        "status": summary.get("status"),
        "tools": tools,
        "counters": counters,
        "stop_reason": summary.get("stop_reason"),
        "task_id": summary.get("task_id"),
    }


def _check_budget_exhaustion(
    base: ResearchConfig, root: Path, store: SharedExperimentStore
) -> dict[str, Any]:
    config = _fresh_config(base, root / "budget", store)
    config.budget.max_tool_calls = 1
    config.budget.max_experiments = 1
    summary = ResearchAgent(config, planner="scripted", store=store).run(
        "查 50/50 融合配额历史，没有就补做一次推荐检查"
    )
    records = _task_records(store, str(summary.get("task_id")))
    business_records = [
        item for item in records["matched"] if item.get("business") in {"recommendation", "search", "ads"}
    ]
    refusals = summary.get("budget_refusals") or []
    # 子 Agent 不能绕过父预算：父 experiments=0 时直接拒绝，不产生实验记录。
    guard = TaskBudget(
        "acceptance-budget-guard",
        BudgetPolicy(max_experiments=0, max_tool_calls=4),
    )
    scope = guard.child("recommendation", BudgetPolicy(max_experiments=1, max_tool_calls=2))
    blocked = False
    try:
        run_business(
            "recommendation",
            "budget guard",
            task_id="acceptance-budget-guard",
            scope=scope,
            experiment_dir=root / "budget_guard",
            intent={"users": 2},
            store=store,
        )
    except BudgetExceeded:
        blocked = True
    ok = (
        summary.get("stop_reason") == "budget_exhausted"
        and summary.get("new_experiments") == 0
        and bool(refusals)
        and not business_records
        and blocked
        and guard.counters["experiment"] == 0
    )
    return {
        "ok": ok,
        "stop_reason": summary.get("stop_reason"),
        "stop_detail": summary.get("stop_detail"),
        "new_experiments": summary.get("new_experiments"),
        "budget_refusals": refusals,
        "business_records": len(business_records),
        "subagent_blocked": blocked,
        "guard_counters": dict(guard.counters),
        "task_id": summary.get("task_id"),
    }


def _check_three_business(
    base: ResearchConfig, root: Path, store: SharedExperimentStore
) -> dict[str, Any]:
    cross_config = _fresh_config(base, root / "cross", store)
    cross_config.budget.max_experiments = 3
    cross_config.budget.max_tool_calls = 9
    cross = ResearchAgent(cross_config, planner="scripted", store=store).run(
        "推荐和搜索一起做：先跑一个最小推荐检查，再用 BM25 检索会话 11208"
    )
    cross_records = _task_records(store, str(cross.get("task_id")))
    businesses = [item.get("business") for item in cross_records["matched"]]
    cross_ok = (
        cross.get("status") == "verified"
        and set(businesses) == {"recommendation", "search"}
        and cross.get("new_experiments") == 2
        and len(cross.get("record_ids") or []) == 2
    )

    ads_config = _fresh_config(base, root / "ads", store)
    ads_config.budget.max_experiments = 2
    ads_config.budget.max_tool_calls = 8
    ads = ResearchAgent(ads_config, planner="scripted", store=store).run(
        "用 din 做广告 CTR 预测和竞价模拟"
    )
    ads_records = _task_records(store, str(ads.get("task_id")))
    ads_ok = (
        ads.get("status") == "verified"
        and [item.get("business") for item in ads_records["matched"]] == ["ads"]
        and ads.get("new_experiments") == 1
    )
    return {
        "ok": bool(cross_ok and ads_ok),
        "cross_task_id": cross.get("task_id"),
        "cross_businesses": businesses,
        "cross_record_ids": cross.get("record_ids"),
        "cross_status": cross.get("status"),
        "ads_task_id": ads.get("task_id"),
        "ads_status": ads.get("status"),
        "ads_record_ids": ads.get("record_ids"),
        "command_cross": 'python -m recsys.platform --task "推荐和搜索一起做..." --planner scripted',
    }


def _check_experiment_platform(
    store: SharedExperimentStore,
    cross_task_id: str | None,
    ads_task_id: str | None,
) -> dict[str, Any]:
    partial = build_record(
        task_id=f"partial-{uuid.uuid4().hex[:8]}",
        business="search",
        status="partial",
        config={"task": "acceptance partial probe"},
        data={"dataset": "kuasar_small", "split": "validation", "split_role": "validation"},
        model={"models": ["bm25"], "weights_source": "existing_search_index"},
        result={"summary": "故意保留的 partial 记录，用于验证不可复用"},
        budget={"counters": {"tool": 1}},
        artifacts={},
        evidence={
            "validated": False,
            "reusable": False,
            "execution_mode": "real_tools",
            "decision_driver": "scripted",
            "warnings": ["acceptance_partial_probe"],
        },
        producer="recsys.platform.acceptance",
    )
    stored = store.append(partial)

    rejected_query = query_records(
        store,
        business="search",
        dataset="kuasar_small",
        split="validation",
        model="bm25",
        reusable=True,
        request={"dataset": "kuasar_small", "split": "validation", "model": "bm25"},
    )
    partial_query = query_records(
        store,
        business="search",
        status="partial",
        reusable=None,
        limit=10,
    )
    rec_query = query_records(
        store,
        business="recommendation",
        task_id=cross_task_id,
        reusable=True,
        request={"dataset": "kuasar_small", "split": "validation"},
        limit=10,
    )
    split_mismatch = query_records(
        store,
        business="recommendation",
        task_id=cross_task_id,
        reusable=True,
        request={"dataset": "kuasar_small", "split": "test"},
        limit=10,
    )
    ads_query = query_records(
        store,
        business="ads",
        task_id=ads_task_id,
        reusable=None,
        limit=10,
    )
    ads_record = (ads_query.get("matched") or [{}])[0]
    ads_simulation = ads_record.get("simulation") or {}
    ok = (
        rejected_query.get("count") == 0
        and bool(rejected_query.get("rejected"))
        and any("partial" in "；".join(item.get("reasons") or []) for item in rejected_query.get("rejected") or [])
        and stored.get("record_id") in {item.get("record_id") for item in partial_query.get("matched") or []}
        and rec_query.get("count") >= 1
        and bool(rec_query.get("matched"))
        and split_mismatch.get("count") == 0
        and bool(split_mismatch.get("rejected"))
        and bool(ads_record.get("validated"))
        and bool(ads_simulation.get("auction_money"))
    )
    return {
        "ok": ok,
        "partial_record_id": stored.get("record_id"),
        "partial_stored_and_queryable": stored.get("record_id")
        in {item.get("record_id") for item in partial_query.get("matched") or []},
        "partial_rejected_for_reuse": rejected_query.get("count") == 0
        and bool(rejected_query.get("rejected")),
        "partial_reject_reasons": (rejected_query.get("rejected") or [{}])[0].get("reasons"),
        "validated_record_reusable": rec_query.get("count"),
        "split_mismatch_rejected": split_mismatch.get("count") == 0
        and bool(split_mismatch.get("rejected")),
        "split_mismatch_reasons": (split_mismatch.get("rejected") or [{}])[0].get("reasons"),
        "ads_simulation_marked": ads_simulation,
        "ads_decision_driver": ads_record.get("decision_driver"),
        "ads_validated": ads_record.get("validated"),
    }


def _check_model_registry() -> dict[str, Any]:
    registry = model_registry_view()
    rec = registry.get("recommendation") or {}
    search = registry.get("search") or {}
    ads = registry.get("ads") or {}
    ok = (
        bool(rec.get("models"))
        and bool(search.get("tools"))
        and bool(search.get("bm25", {}).get("num_docs"))
        and bool(search.get("vector", {}).get("faiss_active"))
        and len(ads.get("models") or []) == 6
    )
    return {
        "ok": ok,
        "recommendation_models": rec.get("count"),
        "recommendation_updated": rec.get("updated"),
        "search_tools": search.get("tools"),
        "search_bm25_docs": search.get("bm25", {}).get("num_docs"),
        "search_vector_faiss": search.get("vector", {}).get("faiss_active"),
        "ads_models": [item.get("name") for item in ads.get("models") or []],
        "ads_statuses": {
            item.get("name"): item.get("status") for item in ads.get("models") or []
        },
    }


def _check_cross_process(
    store: SharedExperimentStore, cross_task_id: str | None
) -> dict[str, Any]:
    if not cross_task_id:
        raise RuntimeError("没有跨业务任务 ID，无法验证另起进程查询")
    payload = _run_json_subprocess(
        [
            "--history-query",
            "--task-id",
            cross_task_id,
            "--shared-store",
            str(store.root),
        ]
    )
    result = payload.get("result") or {}
    records = result.get("records") or []
    businesses = [item.get("business") for item in records]
    ok = (
        result.get("hit") is True
        and set(businesses) >= {"recommendation", "search"}
        and all(item.get("task_id") == cross_task_id for item in records)
    )
    return {
        "ok": ok,
        "query_task_id": cross_task_id,
        "count": len(records),
        "businesses": businesses,
        "store": str(store.records_path),
        "command": (
            "python -m recsys.platform --history-query --task-id "
            f"{cross_task_id} --shared-store {store.root}"
        ),
    }


def _ensure_rag_index(config: Mapping[str, Any], *, force: bool = False) -> RagIndex:
    index = RagIndex(config["index_dir"], config=config)
    index.build_or_update(force=force)
    return index


def _check_rag_citation(config: Mapping[str, Any]) -> dict[str, Any]:
    index = _ensure_rag_index(config)
    result = index.search("实验记忆 指纹 复用", top_k=3)
    hits = result.get("hits") or []
    if not hits:
        raise RuntimeError("RAG 检索没有命中")
    hit = hits[0]
    read = index.read_citation(str(hit["citation"]))
    snippet = _norm(hit["snippet"]).rstrip("…")
    body = _norm(read["text"])
    locatable = bool(snippet) and (snippet[:60] in body or body[:60] in snippet)
    path_ok = (REPO_ROOT / hit["path"]).exists() if not Path(hit["path"]).is_absolute() else Path(hit["path"]).exists()
    ok = (
        result.get("ok") is True
        and path_ok
        and read["start_line"] >= 1
        and read["end_line"] >= read["start_line"]
        and locatable
    )
    return {
        "ok": ok,
        "query": "实验记忆 指纹 复用",
        "citation": hit["citation"],
        "path": hit["path"],
        "start_line": hit["start_line"],
        "end_line": hit["end_line"],
        "snippet_head": _norm(hit["snippet"])[:120],
        "read_head": _norm(read["text"])[:120],
        "locatable": locatable,
        "index": index.describe(include_sources=False),
        "command": 'python -m recsys.platform --rag-query "实验记忆 指纹 复用"',
    }


def _check_rag_update(config: Mapping[str, Any]) -> dict[str, Any]:
    nonce = f"RAGUPDATE{uuid.uuid4().hex[:10]}"
    note = UPDATE_NOTE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# Phase 4 RAG 增量更新记录\n\n"
        f"唯一标记：{nonce}\n\n"
        "本文件用于验证本地知识库增量更新：资料内容变化后重新构建对应文件的片段，"
        "旧资料片段继续可检索。它是 Phase 4 的实际技术记录，不是临时测试数据。\n",
        encoding="utf-8",
    )
    index = RagIndex(config["index_dir"], config=config)
    result = index.build_or_update()
    changed_paths = [str(item).replace("\\", "/") for item in (result.get("added") or []) + (result.get("changed") or [])]
    updated = any(path.endswith("docs/phase4/rag-update-note.md") for path in changed_paths)
    nonce_hit = index.search(nonce, top_k=3)
    hits = nonce_hit.get("hits") or []
    hit_ok = bool(hits) and str(hits[0].get("path", "")).replace("\\", "/").endswith("docs/phase4/rag-update-note.md")
    old_hit = index.search("实验记忆 指纹 复用", top_k=2)
    ok = updated and hit_ok and bool(old_hit.get("hits"))
    return {
        "ok": ok,
        "nonce": nonce,
        "updated_paths": changed_paths,
        "update_seen": updated,
        "new_content_hit": hit_ok,
        "citation": hits[0].get("citation") if hits else None,
        "old_content_still_hits": bool(old_hit.get("hits")),
        "stats": result.get("stats"),
        "command": "python -m recsys.platform --rag-update",
    }


def _check_rag_business_agents(
    config: Mapping[str, Any], output_root: Path | None = None
) -> dict[str, Any]:
    from recsys.ads.agent import PlannedCall as AdsCall
    from recsys.ads.agent import run_agent as run_ads_agent
    from recsys.ads.tools import build_context as build_ads_context
    from recsys.experiments import search_agent
    from recsys.experiments.rec_agent import AgentSettings, PlannedCall as RecCall, run_agent as run_rec_agent

    root = (
        Path(output_root) if output_root is not None else Path(config["index_dir"]).parent.parent / "platform_acceptance"
    ) / "rag_agents"
    rec = run_rec_agent(
        AgentSettings(users=4, max_tool_calls=2, max_train_calls=0, max_seconds=60),
        task="查项目知识库中的实验记忆实现",
        plan=[RecCall("retrieve_knowledge", {"query": "实验记忆 指纹", "top_k": 2}, reason="RAG")],
        experiment_dir=root / "rec",
    )
    ads_context = build_ads_context(experiment_dir=root / "ads", max_tool_calls=3, max_seconds=60)
    ads = run_ads_agent(
        ads_context,
        task="查广告模型来源",
        experiment_dir=root / "ads",
        plan=[AdsCall("retrieve_knowledge", {"query": "DeepFM CTR 广告", "top_k": 2}, reason="RAG")],
        planner="local",
    )
    search_cfg = copy.deepcopy(search_agent.load_config(Path("configs/experiments/search_agent.toml")))
    search_cfg["knowledge"] = {"enabled": True, "index_dir": str(config["index_dir"]), "top_k": 2}
    search_cfg["search_agent"]["max_model_requests"] = 4
    search = search_agent.run_scripted(
        search_cfg,
        plan="knowledge_bm25",
        output_root=root / "search",
    )
    rec_ok = bool(rec.get("tool_results")) and rec["tool_results"][0]["tool"] == "retrieve_knowledge" and rec["tool_results"][0]["ok"]
    ads_ok = bool(ads.get("tool_results")) and ads["tool_results"][0]["tool"] == "retrieve_knowledge" and ads["tool_results"][0]["ok"]
    search_metrics = search.get("metrics") or {}
    search_summary = search_metrics.get("summary") or {}
    search_ok = (
        search_metrics.get("status") == "verified"
        and bool(search_metrics.get("passed"))
        and int(search_summary.get("knowledge_calls", 0)) == 1
        and bool(search_summary.get("knowledge_citations"))
        and bool(search_summary.get("return_counts"))
    )
    return {
        "ok": bool(rec_ok and ads_ok and search_ok),
        "rec": {"ok": rec_ok, "tools": [item["tool"] for item in rec.get("tool_results") or []]},
        "ads": {"ok": ads_ok, "tools": [item["tool"] for item in ads.get("tool_results") or []]},
        "search": {
            "ok": search_ok,
            "status": search_metrics.get("status"),
            "passed": search_metrics.get("passed"),
            "knowledge_calls": search_summary.get("knowledge_calls"),
            "citations": search_summary.get("knowledge_citations"),
            "return_counts": search_summary.get("return_counts"),
        },
    }


def _check_concurrency(store: SharedExperimentStore) -> dict[str, Any]:
    tags = [f"accept-{uuid.uuid4().hex[:6]}-A", f"accept-{uuid.uuid4().hex[:6]}-B"]
    processes = []
    for tag in tags:
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "recsys.platform",
                    "--append-probe-records",
                    "3",
                    "--probe-store",
                    str(store.root),
                    "--probe-tag",
                    tag,
                ],
                cwd=str(REPO_ROOT),
                env=_subprocess_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    codes = [process.wait(timeout=120) for process in processes]
    counts = {}
    for tag in tags:
        records = query_records(
            store,
            task_id=f"concurrency-{tag}",
            reusable=None,
            limit=20,
            include_tasks=True,
        )
        counts[tag] = records["count"]
    total = sum(counts.values())
    ok = codes == [0, 0] and all(count == 3 for count in counts.values()) and total == 6
    return {
        "ok": ok,
        "exit_codes": codes,
        "counts": counts,
        "total": total,
        "bad_lines": store.bad_lines(),
        "store": str(store.records_path),
        "note": "两个进程各追加 3 条，JSONL 锁保证互不覆盖；记录只追加不改写。",
    }


def _check_real_llm(
    base: ResearchConfig,
    root: Path,
    store: SharedExperimentStore,
    offline_decisions: list[dict],
    *,
    with_llm: bool,
) -> dict[str, Any]:
    from recsys.experiments.llm_tool_call import LLM_ENV, llm_config, load_dotenv

    load_dotenv()
    config = llm_config()
    available = config is not None
    if not with_llm:
        return {
            "ok": True,
            "status": "not_requested",
            "llm_available": available,
            "llm_blocker": None if available else "未请求真实 LLM；配置不可用",
            "offline_decisions": len(offline_decisions),
            "note": "本次验收未请求真实 LLM；加 --with-llm 可只跑一次真实任务。",
        }
    if not available:
        missing = [name for name in LLM_ENV if not os.environ.get(name, "").strip()]
        return {
            "ok": True,
            "status": "blocked",
            "llm_available": False,
            "llm_blocker": f"缺少/为空的环境变量：{', '.join(missing)}",
            "offline_decisions": len(offline_decisions),
            "offline_trace_sample": offline_decisions[:4],
            "note": "真实 LLM 配置不可用；离线替身已完成验证，真实 LLM 分支未运行。",
        }
    run_config = _fresh_config(base, root / "real_llm", store)
    run_config.budget.max_experiments = 2
    run_config.budget.max_tool_calls = 8
    summary = ResearchAgent(run_config, planner="llm", store=store).run(
        "查 50/50 融合配额历史。如果历史未命中，请调用一次最小推荐检查，然后停止。"
    )
    decisions = summary.get("decisions") or []
    trace = (summary.get("artifacts") or {}).get("trace")
    has_history_decision = any(
        (item.get("tool") == "search_experiment_history") for item in decisions
    )
    has_followup = any(
        (item.get("tool") in {"run_recommendation_experiment", "retrieve_knowledge", "run_search_task", "run_ads_experiment"})
        or item.get("action") == "stop"
        for item in decisions[1:]
    )
    ok = (
        summary.get("llm_used") is True
        and len(decisions) >= 2
        and has_history_decision
        and has_followup
        and summary.get("status") in {"verified", "partial"}
    )
    return {
        "ok": ok,
        "status": "ran",
        "llm_available": True,
        "llm_used": summary.get("llm_used"),
        "decision_count": len(decisions),
        "decisions": decisions,
        "task_id": summary.get("task_id"),
        "summary_status": summary.get("status"),
        "stop_reason": summary.get("stop_reason"),
        "trace": trace,
        "note": "只运行一次真实 LLM 研究任务；其余分支用离线替身。",
    }


# ---------------------------------------------------------------------- 主入口


def run_acceptance(
    *,
    config: ResearchConfig | None = None,
    with_llm: bool = False,
    output_root: Path | None = None,
    reuse_llm_result: Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    base = copy.deepcopy(config or ResearchConfig.load())
    stamp = _utc_stamp()
    root = output_root or (Path(base.experiment_root) / f"acceptance-{stamp}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    store = SharedExperimentStore(root / "shared")
    rag_config = load_rag_config(RAG_CONFIG)
    checks: list[dict[str, Any]] = []

    hit = _check(checks, "1_history_hit", lambda: _check_history_hit(base, root, store))
    miss = _check(checks, "2_history_miss_and_small_check", lambda: _check_history_miss(base, root, store))
    artifact = _check(
        checks,
        "2c_artifact_read_counted",
        lambda: _check_artifact_read(
            base,
            root,
            store,
            ((miss.get("detail") or {}).get("record_ids") or [None])[0],
        ),
    )
    budget = _check(checks, "2b_budget_exhaustion", lambda: _check_budget_exhaustion(base, root, store))
    three = _check(checks, "3_three_business_entries", lambda: _check_three_business(base, root, store))
    platform = _check(
        checks,
        "3b_shared_experiment_platform",
        lambda: _check_experiment_platform(
            store,
            (three.get("detail") or {}).get("cross_task_id"),
            (three.get("detail") or {}).get("ads_task_id"),
        ),
    )
    registry = _check(checks, "3c_model_registry", _check_model_registry)
    cross_process = _check(
        checks,
        "3d_cross_process_records",
        lambda: _check_cross_process(store, (three.get("detail") or {}).get("cross_task_id")),
    )
    rag_citation = _check(checks, "3e_rag_citation", lambda: _check_rag_citation(rag_config))
    rag_update = _check(checks, "3f_rag_update", lambda: _check_rag_update(rag_config))
    rag_agents = _check(
        checks,
        "3g_rag_business_agents",
        lambda: _check_rag_business_agents(rag_config, root),
    )
    concurrency = _check(checks, "3h_concurrent_writes", lambda: _check_concurrency(store))
    offline_decisions = (miss.get("detail") or {}).get("decisions") or []
    if reuse_llm_result is not None:
        if isinstance(reuse_llm_result, (str, Path)):
            prior_payload = json.loads(Path(reuse_llm_result).read_text(encoding="utf-8"))
            prior_path = str(reuse_llm_result)
        else:
            prior_payload = dict(reuse_llm_result)
            prior_path = prior_payload.get("_path") or "inline"
        prior_check = next(
            (item for item in prior_payload.get("checks") or [] if item.get("id") == "4_real_llm_once"),
            None,
        )
        if prior_check is None:
            raise RuntimeError(f"复用文件里没有 4_real_llm_once：{prior_path}")

        def _reused() -> dict[str, Any]:
            detail = copy.deepcopy(prior_check.get("detail") or {})
            detail["reused_from"] = prior_path
            detail["reused_note"] = (
                "真实 LLM 只运行一次；这里复用同一份实际运行记录，不重复调用模型。"
            )
            return detail

        real_llm = _check(checks, "4_real_llm_once", _reused, required=bool(with_llm))
    else:
        real_llm = _check(
            checks,
            "4_real_llm_once",
            lambda: _check_real_llm(
                base, root, store, offline_decisions, with_llm=with_llm
            ),
            required=bool(with_llm),
        )

    required = [item for item in checks if item.get("required")]
    ok = all(item.get("ok") for item in required)
    result = {
        "ok": ok,
        "phase": "Phase 4",
        "started_at": stamp,
        "repo_root": str(REPO_ROOT),
        "python": sys.version.split()[0],
        "llm_available": bool((real_llm.get("detail") or {}).get("llm_available")),
        "llm_status": (real_llm.get("detail") or {}).get("status"),
        "llm_blocker": (real_llm.get("detail") or {}).get("llm_blocker"),
        "checks": checks,
        "passed": [item["id"] for item in checks if item.get("ok")],
        "failed": [item["id"] for item in checks if not item.get("ok")],
        "unverified": [
            "完整 LLM 多轮规划（只按验收要求跑一次真实任务）",
            "大规模训练/全量/调参（Phase 4 明确不做）",
            "广告真实出价、预算与转化标签（KuaiSAR 数据缺口，使用模拟并单独标记）",
        ],
        "artifacts": {
            "acceptance": str(root / "acceptance.json"),
            "report": str(root / "report.md"),
            "shared_store": str(store.records_path),
            "rag_index": str(rag_config["index_dir"]),
        },
    }
    (root / "acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "report.md").write_text(_render_report(result), encoding="utf-8")
    return result


def _render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 4 小样本验收报告",
        "",
        f"- 生成时间：{result.get('started_at')}",
        f"- 总体状态：**{'通过' if result.get('ok') else '未通过'}**",
        f"- Python：{result.get('python')}",
        f"- 真实 LLM：{'可用' if result.get('llm_available') else '不可用/未运行'}",
        f"- 共享实验库：`{result['artifacts']['shared_store']}`",
        f"- RAG 索引：`{result['artifacts']['rag_index']}`",
        "",
        "## 验收项",
        "",
        "| # | 项目 | required | 结果 | 耗时(s) | 说明 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, check in enumerate(result.get("checks") or [], start=1):
        detail = check.get("detail") or {}
        summary = (
            detail.get("error")
            or detail.get("note")
            or detail.get("history_status")
            or detail.get("status")
            or detail.get("stop_reason")
            or ""
        )
        lines.append(
            f"| {index} | {check.get('id')} | {check.get('required')} | "
            f"{'PASS' if check.get('ok') else 'FAIL'} | {check.get('elapsed_s')} | {str(summary)[:120]} |"
        )
    lines += ["", "## 关键证据", ""]
    for check in result.get("checks") or []:
        detail = check.get("detail") or {}
        keys = [
            key
            for key in (
                "task_id",
                "cross_task_id",
                "ads_task_id",
                "citation",
                "record_ids",
                "cross_record_ids",
                "decision_count",
                "new_experiments",
                "subagent_blocked",
                "offline_trace",
                "trace",
                "command",
            )
            if detail.get(key) is not None
        ]
        if not keys:
            continue
        lines.append(f"### {check.get('id')}")
        for key in keys:
            lines.append(f"- {key}: `{detail.get(key)}`")
        lines.append("")
    lines += [
        "## 尚未验证/明确边界",
        "",
        *[f"- {item}" for item in result.get("unverified") or []],
        "",
        "## 复现命令",
        "",
        "```powershell",
        '$env:PYTHONPATH = "src"',
        "python -m recsys.platform --acceptance",
        "python -m recsys.platform --acceptance --with-llm   # 需要 DSH_AGENT_LLM_* 配置",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------- 自检


def run_self_check(config: ResearchConfig | None = None) -> dict[str, Any]:
    base = copy.deepcopy(config or ResearchConfig.load())
    root = Path(base.experiment_root) / "self_check"
    root.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def record(check_id: str, ok: bool, detail: Any) -> None:
        checks.append({"id": check_id, "ok": bool(ok), "detail": detail})

    budget = TaskBudget("self-check", BudgetPolicy(max_tool_calls=4, max_experiments=2, max_seconds=10))
    scope = budget.child("search", BudgetPolicy(max_tool_calls=2, max_experiments=1))
    scope.spend("tool")
    scope.spend("experiment")
    record(
        "budget_child_shares_parent",
        budget.counters["tool"] == 1 and budget.counters["experiment"] == 1,
        {"parent": dict(budget.counters), "scope": dict(scope.counters)},
    )
    tiny = TaskBudget("self-check-tiny", BudgetPolicy(max_tool_calls=1))
    tiny.spend("tool")
    exhausted = False
    try:
        tiny.spend("tool")
    except BudgetExceeded:
        exhausted = True
    record("budget_exhaustion", exhausted, tiny.snapshot())

    store = SharedExperimentStore(root / "shared")
    rec = build_record(
        task_id="self-check-record",
        business="recommendation",
        status="success",
        config={"task": "self-check"},
        data={"dataset": "kuasar_small", "split": "validation", "split_role": "validation"},
        model={"models": ["popularity"], "weights_source": "self-check"},
        result={"metrics": {"hits": 1}},
        budget={"counters": {}},
        artifacts={},
        evidence={"validated": True, "reusable": True, "execution_mode": "real_tools"},
    )
    stored = store.append(rec)
    fetched = store.get(stored["record_id"])
    record("shared_store_roundtrip", bool(fetched) and fetched.get("business") == "recommendation", stored["record_id"])
    record("reuse_validated_ok", not reuse_reasons(stored, {}), reuse_reasons(stored, {}))
    bad = dict(fetched or {})
    bad["status"] = "partial"
    record("reuse_rejects_partial", bool(reuse_reasons(bad, {})), reuse_reasons(bad, {}))

    legacy = lookup_history(
        business="recommendation",
        quota=(30, 70),
        store=store,
        legacy=base.history,
        request={"dataset": "kuasar_small"},
    )
    record("legacy_history_hit", bool(legacy.get("hit")), {"source": legacy.get("source"), "reason": legacy.get("reason")})

    rag = RagIndex(RAG_INDEX_DIR, config=load_rag_config(RAG_CONFIG))
    description = rag.describe(include_sources=False)
    record("rag_index_ready", bool(description.get("ok")), description)
    retrieval = rag.search("实验记忆 指纹 复用", top_k=2)
    record("rag_retrieval_with_citation", bool(retrieval.get("hits")), retrieval.get("hits", [])[:1])

    registry = model_registry_view()
    record(
        "model_registry_three_businesses",
        bool(registry.get("recommendation", {}).get("models"))
        and bool(registry.get("search", {}).get("tools"))
        and bool(registry.get("ads", {}).get("models")),
        {
            "recommendation": registry.get("recommendation", {}).get("count"),
            "search_tools": registry.get("search", {}).get("tools"),
            "ads": registry.get("ads", {}).get("count"),
        },
    )
    record("tool_registry", len(TOOLS) == 7, list(TOOLS))

    result = {
        "ok": all(item["ok"] for item in checks),
        "passed": all(item["ok"] for item in checks),
        "checks": checks,
        "root": str(root),
    }
    (root / "self_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


__all__ = ["run_acceptance", "run_self_check"]
