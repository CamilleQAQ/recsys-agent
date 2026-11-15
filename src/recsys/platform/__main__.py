"""Phase 4 统一 CLI。

常用命令（PowerShell，仓库根目录）::

    $env:PYTHONPATH = "src"

    # Research Agent / Master：自然语言任务
    python -m recsys.platform --task "查一下已有 30/70 融合实验，能复用就复用"
    python -m recsys.platform --task "推荐和搜索一起做" --planner scripted

    # 历史查询（另一个进程也能查回共享记录）
    python -m recsys.platform --history-query --task-id task-...

    # Phase 5：三业务协作演示（一条命令；默认离线计划）
    python -m recsys.platform --phase5-demo

    # Phase 5：单独调用一个业务，或按任务 ID 查回统一摘要
    python -m recsys.platform --run-business search
    python -m recsys.platform --phase5-report --task-id task-...

    # Phase 5：独立进程工件恢复 / 缺工件失败用例
    python -m recsys.platform --phase5-restore
    python -m recsys.platform --phase5-missing

    # 模型注册
    python -m recsys.platform --registry

    # 本地 RAG
    python -m recsys.platform --rag-build
    python -m recsys.platform --rag-query "实验记忆 指纹 复用"

    # Phase 4 小样本验收
    python -m recsys.platform --acceptance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from recsys.platform.rag import (
    DEFAULT_TOP_K,
    RagIndex,
    load_rag_config,
    retrieve_knowledge,
)
from recsys.platform.research_agent import (
    DEFAULT_CONFIG_PATH,
    ResearchConfig,
    ResearchAgent,
)
from recsys.platform.shared_experiments import (
    SharedExperimentStore,
    build_record,
    lookup_history,
    model_registry_view,
    query_records,
    resolve_path,
)

BUSINESSES = ("any", "recommendation", "search", "ads")


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _parse_quota(text: str | None) -> list[int] | None:
    if not text:
        return None
    cleaned = text.replace("/", ",").replace("：", ",").replace(":", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--quota 需要两个整数，例如 30,70；收到 {text!r}")
    try:
        return [int(parts[0]), int(parts[1])]
    except ValueError as error:
        raise SystemExit(f"--quota 不是整数：{text!r}") from error


def _config_from_args(args: argparse.Namespace) -> ResearchConfig:
    config_path = (Path("configs/platform/autoresearch.toml")
                   if args.research_cases and args.config == DEFAULT_CONFIG_PATH else args.config)
    config = ResearchConfig.load(config_path)
    overrides = {
        "max_llm_requests": args.max_llm_requests,
        "max_tool_calls": args.max_tool_calls,
        "max_experiments": args.max_experiments,
        "max_seconds": args.max_seconds,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config.budget, name, value)
    if args.experiment_root:
        config.experiment_root = Path(args.experiment_root)
    if args.shared_store:
        config.store_dir = Path(args.shared_store)
    if args.rag_index:
        config.rag_index_dir = str(args.rag_index)
    if args.max_steps is not None:
        config.max_steps = max(1, int(args.max_steps))
    return config


def _task_intent(args: argparse.Namespace) -> dict[str, Any]:
    intent: dict[str, Any] = {
        key: getattr(args, key) for key in ("dataset", "split", "model") if getattr(args, key)
    }
    if args.research_cases:
        manifest = json.loads(resolve_path(args.research_cases).read_text(encoding="utf-8"))
        intent["research_cases"] = manifest
        intent["businesses"] = list(manifest)
        intent["history_first"] = False
        intent["dataset"] = "research_cases"
    if args.session_id:
        intent["session_id"] = args.session_id
    if args.search_mode:
        intent["mode"] = args.search_mode
    quota = _parse_quota(args.quota)
    if quota:
        intent["quota"] = quota
    if args.business and args.business != "any":
        intent["businesses"] = [args.business]
    if args.task_id:
        intent["task_id_filter"] = args.task_id
    if args.read_record:
        intent["read_record"] = args.read_record
        intent.setdefault("history_first", False)
    if args.history_first is not None:
        intent["history_first"] = bool(args.history_first)
    return intent


def _run_task(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    agent = ResearchAgent(config, planner=args.planner)
    summary = agent.run(args.task, intent=_task_intent(args))
    _print_json(
        {
            key: summary.get(key)
            for key in (
                "task_id",
                "task",
                "status",
                "planner",
                "decision_source",
                "llm_used",
                "llm_available",
                "llm_blocker",
                "stop_reason",
                "stop_detail",
                "reuse_mode",
                "history",
                "new_experiments",
                "businesses",
                "record_ids",
                "task_record_id",
                "budget",
                "budget_refusals",
                "tool_results",
                "final_answer",
                "artifacts",
                "elapsed_s",
            )
        }
    )
    return 0 if summary.get("status") in {"verified", "partial", "blocked"} else 1


def _run_history_query(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    result = lookup_history(
        business=args.business,
        quota=_parse_quota(args.quota),
        task_id=args.task_id,
        dataset=args.dataset,
        split=args.split,
        model=args.model,
        store=resolve_path(args.shared_store) if args.shared_store else config.store_dir,
        legacy=config.history,
        request={
            "dataset": args.dataset, "split": args.split, "model": args.model,
            "session_id": args.session_id, "mode": args.search_mode,
        },
    )
    _print_json(
        {
            "query": {
                "business": args.business,
                "quota": result.get("quota"),
                "task_id": args.task_id,
                "dataset": args.dataset,
                "split": args.split,
                "model": args.model,
                "session_id": args.session_id,
                "mode": args.search_mode,
            },
            "result": result,
        }
    )
    return 0


def _run_registry(args: argparse.Namespace) -> int:
    _print_json(model_registry_view(args.business))
    return 0


def _rag(args: argparse.Namespace) -> int:
    config = load_rag_config(args.rag_config)
    index_dir = args.rag_index or config["index_dir"]
    if args.rag_build or args.rag_update:
        index = RagIndex(index_dir, config=config)
        result = index.build_or_update(force=bool(args.rag_build))
        result["describe"] = index.describe(include_sources=False)
        _print_json(result)
        return 0
    result = retrieve_knowledge(
        args.rag_query,
        top_k=args.rag_top_k or DEFAULT_TOP_K,
        index_dir=index_dir,
        config_path=args.rag_config,
    )
    _print_json(result)
    return 0 if result.get("ok") else 2


def _run_acceptance(args: argparse.Namespace) -> int:
    from recsys.platform.acceptance import run_acceptance

    result = run_acceptance(
        config=_config_from_args(args),
        with_llm=bool(args.with_llm),
        output_root=Path(args.acceptance_output) if args.acceptance_output else None,
        reuse_llm_result=Path(args.reuse_llm_result) if args.reuse_llm_result else None,
    )
    _print_json(result)
    return 0 if result.get("ok") else 1


# ---------------------------------------------------------------------- Phase 5


def _phase5_status_code(payload: dict[str, Any]) -> int:
    status = str(payload.get("status") or "")
    if status == "verified":
        return 0
    if status == "blocked":
        return 2
    return 1


def _phase5_demo(args: argparse.Namespace) -> int:
    from recsys.platform.phase5 import run_demo

    result = run_demo(
        args.phase5_config,
        task=args.phase5_task or args.task,
        planner=args.phase5_planner or (None if args.planner == "auto" else args.planner),
        output_root=args.phase5_output,
    )
    _print_json(result)
    return _phase5_status_code(result)


def _phase5_single(args: argparse.Namespace) -> int:
    from recsys.platform.phase5 import run_single_business

    result = run_single_business(
        args.run_business,
        args.phase5_config,
        task=args.phase5_task or args.task,
        output_root=args.phase5_output,
    )
    _print_json(result)
    return 0 if result.get("ok") else 1


def _phase5_restore(args: argparse.Namespace) -> int:
    from recsys.platform.phase5 import run_restore

    result = run_restore(args.phase5_config, output_root=args.phase5_output)
    _print_json(result)
    return _phase5_status_code(result)


def _phase5_missing(args: argparse.Namespace) -> int:
    from recsys.platform.phase5 import run_missing_check

    result = run_missing_check(args.phase5_config, output_root=args.phase5_output)
    _print_json(result)
    return 0 if result.get("ok") else 1


def _phase5_report(args: argparse.Namespace) -> int:
    from recsys.platform.phase5 import report

    task_id = args.task_id or args.phase5_task_id
    if not task_id:
        raise SystemExit("--phase5-report 需要 --task-id")
    result = report(task_id, args.phase5_config, output_root=args.phase5_output)
    _print_json(result)
    return 0 if result.get("found") else 1


def _append_probe_records(args: argparse.Namespace) -> int:
    store = SharedExperimentStore(resolve_path(args.probe_store))
    tag = args.probe_tag or f"probe-{os.getpid()}"
    created = []
    for index in range(max(1, int(args.append_probe_records))):
        record = build_record(
            task_id=f"concurrency-{tag}",
            business="recommendation",
            status="success",
            config={"probe_tag": tag, "probe_index": index, "task": f"concurrency {tag}"},
            data={"dataset": "kuasar_small", "split": "validation", "split_role": "validation"},
            model={"models": ["popularity"], "weights_source": "probe"},
            result={"metrics": {"probe": index}},
            budget={"counters": {"tool": 0}},
            artifacts={},
            evidence={
                "validated": False,
                "reusable": False,
                "execution_mode": "real_tools",
                "decision_driver": "probe",
                "warnings": ["concurrency_probe"],
            },
            producer="recsys.platform.concurrency_probe",
        )
        created.append(store.append(record)["record_id"])
    _print_json(
        {
            "ok": True,
            "store": str(store.records_path),
            "tag": tag,
            "count": len(created),
            "record_ids": created,
        }
    )
    return 0


def _list_records(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    result = query_records(
        resolve_path(args.shared_store) if args.shared_store else config.store_dir,
        business=args.business if args.business != "any" else None,
        task_id=args.task_id,
        dataset=args.dataset,
        split=args.split,
        model=args.model,
        status=args.status,
        reusable=args.reusable,
        driver=args.driver,
        limit=args.limit,
        include_tasks=bool(args.include_tasks),
    )
    _print_json(result)
    return 0


def _self_check(args: argparse.Namespace) -> int:
    from recsys.platform.acceptance import run_self_check

    result = run_self_check(_config_from_args(args))
    _print_json(result)
    return 0 if result.get("passed") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--task", default=None)
    parser.add_argument("--planner", choices=["auto", "llm", "scripted"], default="auto")
    parser.add_argument("--quota", default=None, help="融合配额，例如 30,70")
    parser.add_argument("--business", choices=BUSINESSES, default="any")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--read-record", default=None, help="读取共享实验库中的已有记录，不重跑")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--research-cases", help="业务名到真实 validation case 路径的 JSON；开启代码修改研究")
    parser.add_argument("--split", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--session-id", help="按 KuaiSAR 会话 ID 查询搜索历史")
    parser.add_argument("--search-mode", choices=["bm25", "vector", "hybrid", "rerank"])
    parser.add_argument("--history-first", dest="history_first", action="store_true", default=None)
    parser.add_argument("--no-history-first", dest="history_first", action="store_false")
    parser.add_argument("--experiment-root", default=None)
    parser.add_argument("--shared-store", default=None)
    parser.add_argument("--rag-index", default=None)
    parser.add_argument("--max-llm-requests", type=int, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--history-query", action="store_true")
    parser.add_argument("--registry", action="store_true")
    parser.add_argument("--list-records", action="store_true")
    parser.add_argument("--include-tasks", action="store_true")
    parser.add_argument("--status", default=None)
    parser.add_argument("--driver", default=None)
    parser.add_argument(
        "--reusable",
        action="store_true",
        default=None,
        help="只返回可复用记录；同时会给出拒绝原因",
    )
    parser.add_argument("--limit", type=int, default=20)

    parser.add_argument("--rag-build", action="store_true")
    parser.add_argument("--rag-update", action="store_true")
    parser.add_argument("--rag-query", default=None)
    parser.add_argument("--rag-top-k", type=int, default=None)
    parser.add_argument("--rag-config", type=Path, default=Path("configs/platform/rag.toml"))

    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument(
        "--reuse-llm-result",
        default=None,
        help="复用已实际运行过的 --with-llm 验收 JSON，避免重复调用真实 LLM",
    )
    parser.add_argument("--acceptance-output", default=None)
    parser.add_argument("--self-check", action="store_true")

    # Phase 5：统一集成演示 / 单业务调用 / 工件恢复 / 缺工件用例。
    parser.add_argument(
        "--phase5-demo",
        action="store_true",
        help="一条命令跑三业务协作演示（默认使用 configs/platform/phase5_demo.toml）",
    )
    parser.add_argument(
        "--phase5-config",
        type=Path,
        default=Path("configs/platform/phase5_demo.toml"),
        help="Phase 5 演示配置",
    )
    parser.add_argument(
        "--phase5-output",
        default=None,
        help="覆盖演示输出根目录（默认取配置里的 experiment_root）",
    )
    parser.add_argument("--phase5-task", default=None, help="覆盖演示任务文本")
    parser.add_argument(
        "--phase5-planner",
        choices=["auto", "llm", "scripted"],
        default=None,
        help="Phase 5 演示计划器；默认读取配置（scripted，离线）",
    )
    parser.add_argument(
        "--run-business",
        choices=["recommendation", "search", "ads"],
        default=None,
        help="从统一入口单独调用一个业务（仍走共享实验库与统一预算）",
    )
    parser.add_argument(
        "--phase5-restore",
        action="store_true",
        help="独立进程从工件清单加载并跑一次真实推理/查询",
    )
    parser.add_argument(
        "--phase5-missing",
        action="store_true",
        help="缺工件小用例：验证不会自动训练、换模型或重建索引",
    )
    parser.add_argument(
        "--phase5-report",
        action="store_true",
        help="按任务 ID 查回三业务统一摘要与记录（不重跑）",
    )
    parser.add_argument("--phase5-task-id", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--append-probe-records", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--probe-store", default="experiments/_probe/shared", help=argparse.SUPPRESS)
    parser.add_argument("--probe-tag", default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    # 统一 UTF-8 输出，保证另一个进程/脚本能稳定解析中文 JSON。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.append_probe_records:
        return _append_probe_records(args)
    if args.phase5_demo:
        return _phase5_demo(args)
    if args.run_business:
        return _phase5_single(args)
    if args.phase5_restore:
        return _phase5_restore(args)
    if args.phase5_missing:
        return _phase5_missing(args)
    if args.phase5_report:
        return _phase5_report(args)
    if args.acceptance:
        return _run_acceptance(args)
    if args.self_check:
        return _self_check(args)
    if args.rag_build or args.rag_update or args.rag_query:
        return _rag(args)
    if args.registry:
        return _run_registry(args)
    if args.history_query:
        return _run_history_query(args)
    if args.list_records:
        return _list_records(args)
    if args.task:
        return _run_task(args)
    if args.read_record:
        args.task = args.task or "读取记录内容并给出摘要"
        if args.planner == "auto":
            # 读取已有结果不需要模型重新规划；默认走脚本替身，避免无谓 LLM 请求。
            args.planner = "scripted"
        return _run_task(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
