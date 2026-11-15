"""Phase 2 搜索系统小样本验收（不训练、不重建 BM25、不扫原始 CSV、不联外网）。

    python -m recsys.search.acceptance
    python -m recsys.search.acceptance --output experiments/2026-11-09-kuasar-search-phase2/acceptance.json

检查项（全部执行真实代码）：

1. 旧 BM25 入口：``BM25Index.search`` 与 ``python -m recsys.search`` CLI 仍可用；
2. FAISS：向量索引构建后 ``backend_actual == faiss.IndexFlatIP``，搜索时 FAISS 调用计数
   增加；从磁盘重载后查询结果一致；
3. 索引更新：先用 N-1 个物品建索引，再 ``add_items`` 追加 1 个，保存/重载后仍是 N 个
   且可查询；
4. 混合检索：结果按 item_id 去重、每条保留 bm25/vector 来源与名次，融合分等于
   Σ 1/(rrf_k+rank)，融合顺序与上游 ``RRF.fuse`` 完全一致；
5. 精排：对融合候选池重新打分并返回最终 Top-K，分数等于
   ``coverage_weight*coverage + prior_weight*prior_norm``，排序为 ``(-score, item_id)``；
6. Search Agent 链路：本地脚本替身给出决策，检索/精排/索引读取/落盘执行真实代码；
7. 空结果与失败分支：无匹配 token 返回 0 条；首轮直接请求精排被拒绝；空查询被拒绝；
8. 文本语义编码缺口：``TextSemanticEncoder`` 在缺文本/缺权重/缺依赖时给出具体原因。

点击标签只用于验收时核对（记录命中情况），不参与任何打分或排序。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from recsys.search.bm25 import load_index
from recsys.search.build_vector_index import (
    DEFAULT_CONFIG_PATH,
    _sha256_file,
    load_config,
    paths_for,
)
from recsys.search.encoders import (
    EncoderUnavailable,
    TextSemanticEncoder,
    build_encoder,
)
from recsys.search.rerank import RERANKER_NAME
from recsys.search.service import SearchService
from recsys.search.tool import TOOL_HYBRID, TOOL_NAME, SearchToolbox, ToolRejected
from recsys.search.upstream import (
    RRF,
    UPSTREAM_SHORT,
    faiss_version,
)
from recsys.search.vector_index import SearchVectorIndex

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "2026-11-09-kuasar-search-phase2" / "acceptance.json"
)
SAMPLE_QUERY = [3538762, 3181091, 4061567, 3525929]
NO_MATCH_TOKEN = 10**12
CLICKED_JOINED = [350728, 80675]


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class Acceptance:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.evidence: dict[str, Any] = {}

    def check(self, check_id: str, fn: Callable[[], str], evidence: Any = None) -> None:
        try:
            detail = fn()
        except Exception as error:  # noqa: BLE001 - 失败也写入报告
            self.checks.append(
                {
                    "id": check_id,
                    "status": "fail",
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
        else:
            self.checks.append({"id": check_id, "status": "pass", "detail": detail})
            if evidence is not None:
                self.evidence[check_id] = evidence

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(item["status"] == "pass" for item in self.checks)


def _run_legacy_cli(tokens: list[int], top_k: int) -> dict[str, Any]:
    from recsys.search.__main__ import main as cli_main

    buffer = io.StringIO()
    argv = ["--tokens", ",".join(str(token) for token in tokens), "--top-k", str(top_k), "--json"]
    with contextlib.redirect_stdout(buffer):
        code = cli_main(argv)
    if code != 0:
        raise AssertionError(f"CLI 退出码 {code}")
    return json.loads(buffer.getvalue())


def _clicked_hits(item_ids: list[int], clicked: list[int]) -> dict[str, Any]:
    ranks = {item_id: index + 1 for index, item_id in enumerate(item_ids)}
    return {
        "clicked_items": clicked,
        "in_top_k": [item_id for item_id in clicked if item_id in ranks],
        "best_rank": min((ranks[item_id] for item_id in clicked if item_id in ranks), default=None),
        "note": "点击标签仅用于核对，不参与打分或排序",
    }


def run_acceptance(
    config_path: Path = DEFAULT_CONFIG_PATH, *, update_dir: Path | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = paths_for(config)
    encoder = build_encoder(config["encoder"])
    bm25_path = paths["bm25_index"]
    if not bm25_path.exists():
        raise SystemExit(f"[error] 缺少 {bm25_path}；先运行 python -m recsys.search.prepare")
    if not paths["vector_dir"].exists():
        raise SystemExit(
            f"[error] 缺少 {paths['vector_dir']}；先运行 python -m recsys.search.build_vector_index"
        )

    report = Acceptance()
    bm25_sha_before = _sha256_file(bm25_path)
    sample_sha_before = _sha256_file(paths["bm25_sample"])

    # 1. 旧 BM25 入口 ----------------------------------------------------------
    bm25 = load_index(bm25_path)
    bm25_hits = bm25.search(SAMPLE_QUERY, top_k=5)
    legacy_cli = _run_legacy_cli(SAMPLE_QUERY, 5)

    def legacy_bm25() -> str:
        assert len(bm25_hits) == 5, len(bm25_hits)
        assert [hit.item_id for hit in bm25_hits] == [
            row["item_id"] for row in legacy_cli["results"]
        ], "CLI 与库入口结果不一致"
        assert bm25_hits[0].item_id == 80675, bm25_hits[0].item_id
        assert bm25_hits == sorted(bm25_hits, key=lambda hit: (-hit.score, hit.item_id))
        return (
            f"BM25 库入口与 `python -m recsys.search` CLI 都返回 {len(bm25_hits)} 条；"
            f"Top-1={bm25_hits[0].item_id} score={bm25_hits[0].score:.4f}"
        )

    report.check(
        "legacy_bm25_entry",
        legacy_bm25,
        {
            "cli_mode": legacy_cli["mode"],
            "results": legacy_cli["results"],
            "clicked": _clicked_hits([hit["item_id"] for hit in legacy_cli["results"]], CLICKED_JOINED),
        },
    )

    # 2. FAISS 索引：构建/查询/重载 --------------------------------------------
    index_dir = paths["vector_dir"]
    loaded = SearchVectorIndex.load(index_dir, encoder)
    calls_before = loaded.faiss_search_calls
    first_hits = loaded.search_scored(SAMPLE_QUERY, top_k=5)
    calls_after = loaded.faiss_search_calls
    reloaded = SearchVectorIndex.load(index_dir, encoder)
    second_hits = reloaded.search_scored(SAMPLE_QUERY, top_k=5)

    def faiss_backend() -> str:
        assert loaded.faiss_active, "索引没有走 FAISS"
        assert loaded.backend_actual == "faiss.IndexFlatIP", loaded.backend_actual
        assert first_hits, "FAISS 查询没有结果"
        assert calls_after > calls_before, (calls_before, calls_after)
        assert [hit.item_id for hit in first_hits] == [hit.item_id for hit in second_hits], (
            "重载后 Top-K 不一致"
        )
        assert second_hits[0].item_id == 80675, second_hits[0].item_id
        assert all(row.score > 0 for row in second_hits), "余弦分数必须为正"
        assert [row.score for row in second_hits] == sorted(
            (row.score for row in second_hits), reverse=True
        )
        return (
            f"backend={loaded.backend_actual}（faiss {faiss_version()}）；"
            f"首次搜索 faiss 调用 +{calls_after - calls_before}；重载后 Top-5 一致"
        )

    report.check(
        "faiss_index_reload",
        faiss_backend,
        {
            "backend_actual": loaded.backend_actual,
            "faiss_version": faiss_version(),
            "faiss_search_calls_before": calls_before,
            "faiss_search_calls_after": calls_after,
            "num_items": loaded.num_items,
            "dim": loaded.dim,
            "encoder": loaded.encoder.describe(),
            "reload_identical": [hit.item_id for hit in first_hits]
            == [hit.item_id for hit in second_hits],
            "results": [
                {"rank": hit.rank, "item_id": hit.item_id, "score": round(hit.score, 6)}
                for hit in first_hits
            ],
            "clicked": _clicked_hits([hit.item_id for hit in first_hits], CLICKED_JOINED),
        },
    )

    # 3. 索引保存 / 更新 --------------------------------------------------------
    def index_update() -> str:
        target = update_dir or (paths["root"] / "update_check")
        target = Path(target)
        if target.exists():
            shutil.rmtree(target)
        items = list(bm25.items)
        assert len(items) >= 3
        partial = SearchVectorIndex.build(
            items[:-1], encoder, cache_dir=target, backend="faiss"
        )
        assert partial.num_items == len(items) - 1, partial.num_items
        added = partial.add_items([items[-1]])
        assert added == 1, added
        partial.save(target)
        again = SearchVectorIndex.load(target, encoder)
        assert again.num_items == len(items), again.num_items
        hits = again.search_scored(SAMPLE_QUERY, top_k=3)
        assert hits, "更新后的索引查询没有结果"
        assert again.faiss_active, "更新后的索引不是 FAISS"
        shutil.rmtree(target)
        return (
            f"先建 {len(items) - 1} 个物品的索引，add_items 追加 1 个 → 保存/重载后 "
            f"{again.num_items} 个物品，仍可查询（FAISS）"
        )

    report.check("vector_index_update", index_update)

    # 4. 混合检索：去重 + 来源 + RRF ------------------------------------------
    service = SearchService.load(
        bm25_path,
        index_dir,
        encoder,
        rrf_k=int(config["hybrid"]["rrf_k"]),
        hybrid_pool=int(config["hybrid"]["pool"]),
        rerank_method=config["rerank"].get("method", RERANKER_NAME),
        rerank_pool=int(config["rerank"]["pool"]),
        prior_weight=float(config["rerank"]["prior_weight"]),
        coverage_weight=float(config["rerank"]["coverage_weight"]),
    )
    hybrid_hits = service.search_hybrid(SAMPLE_QUERY, top_k=10)
    rankings = service.hybrid.rankings(SAMPLE_QUERY, pool=service.hybrid.pool)
    upstream_order = RRF(k=service.hybrid.rrf_k).fuse(list(rankings.values()), None)

    def hybrid_provenance() -> str:
        ids = [hit.item_id for hit in hybrid_hits]
        assert len(ids) == len(set(ids)), "混合结果没有按 item_id 去重"
        assert ids == [int(doc_id) for doc_id in upstream_order[: len(ids)]], (
            "融合顺序与上游 RRF 不一致"
        )
        for hit in hybrid_hits:
            assert hit.sources, hit
            recomputed = sum(
                1.0 / (service.hybrid.rrf_k + source.rank) for source in hit.sources
            )
            assert abs(recomputed - hit.score) < 1e-12, (hit.item_id, recomputed, hit.score)
            for source in hit.sources:
                assert source.source in {"bm25", "vector"}, source
                assert source.rank >= 1 and source.score != 0.0
        return (
            f"{len(hybrid_hits)} 条融合结果按 item_id 去重，来源/名次/分数齐全；"
            f"融合分等于 Σ1/({service.hybrid.rrf_k}+rank)，顺序与上游 RRF 一致"
        )

    report.check(
        "hybrid_rrf_provenance",
        hybrid_provenance,
        {
            "rrf_k": service.hybrid.rrf_k,
            "pool": service.hybrid.pool,
            "rankings": {
                source: [
                    {"item_id": item_id, "score": round(score, 6), "rank": rank}
                    for rank, (item_id, score) in enumerate(ranking, start=1)
                ]
                for source, ranking in rankings.items()
            },
            "results": [hit.as_dict() for hit in hybrid_hits[:10]],
            "clicked": _clicked_hits([hit.item_id for hit in hybrid_hits], CLICKED_JOINED),
        },
    )

    # 5. 精排：融合候选池重新打分 + 最终 Top-K --------------------------------
    pool_hits = service.search_hybrid(SAMPLE_QUERY, top_k=service.reranker.pool)
    rerank_hits = service.rerank(SAMPLE_QUERY, top_k=5)

    def rerank_final() -> str:
        assert rerank_hits, "精排没有结果"
        assert len(rerank_hits) <= 5, len(rerank_hits)
        assert [hit.rank for hit in rerank_hits] == list(range(1, len(rerank_hits) + 1))
        score_map = {hit.item_id: hit.score for hit in rerank_hits}
        assert [hit.item_id for hit in rerank_hits] == sorted(
            (hit.item_id for hit in rerank_hits),
            key=lambda item_id: (-score_map[item_id], item_id),
        ), "精排排序不是 (-score, item_id)"
        pool_ids = {hit.item_id for hit in pool_hits}
        prior = {hit.item_id: hit.score for hit in pool_hits}
        low, high = min(prior.values()), max(prior.values())
        span = high - low
        for hit in rerank_hits:
            assert hit.item_id in pool_ids, "精排返回了候选池之外的物品"
            coverage = float(hit.details.get("coverage", 0.0))
            assert hit.details.get("prior_rank"), "精排结果缺少融合先验名次"
            prior_norm = 0.5 if span <= 0 else (prior[hit.item_id] - low) / span
            expected = round(
                service.reranker.reranker.coverage_weight * coverage
                + service.reranker.reranker.prior_weight * prior_norm,
                9,
            )
            assert abs(expected - hit.score) < 1e-9, (hit.item_id, expected, hit.score)
        return (
            f"精排对 {len(pool_hits)} 个融合候选重新打分，返回最终 Top-{len(rerank_hits)}；"
            f"分数 = 0.65·coverage + 0.35·prior_norm，排序 (-score, item_id)"
        )

    report.check(
        "rerank_final_topk",
        rerank_final,
        {
            "reranker": service.reranker.describe(),
            "pool_size": len(pool_hits),
            "results": [hit.as_dict() for hit in rerank_hits],
            "clicked": _clicked_hits([hit.item_id for hit in rerank_hits], CLICKED_JOINED),
        },
    )

    # 6. Search Agent：脚本替身决策 + 真实检索/精排/落盘 ----------------------
    from recsys.experiments import search_agent as agent

    agent_config = agent.load_config(agent.DEFAULT_CONFIG_PATH)
    agent_root = DEFAULT_OUTPUT.parent / "agent_scripted"
    agent_result = agent.run_scripted(
        agent_config,
        plan="hybrid_rerank",
        output_root=agent_root,
    )
    agent_metrics = agent_result["metrics"]
    agent_trace_path = agent_root / "scripted-hybrid_rerank" / "trace.json"
    agent_trace = json.loads(agent_trace_path.read_text(encoding="utf-8"))
    serialized_trace = json.dumps(agent_trace, ensure_ascii=False)

    def agent_chain() -> str:
        assert agent_metrics["status"] == "verified", agent_metrics["status"]
        assert agent_metrics["driver"] == "scripted_driver", agent_metrics["driver"]
        assert agent_metrics["summary"]["tool_sequence"] == [TOOL_HYBRID, "search_rerank"]
        backend = agent_metrics["search_backend"]["vector"]
        assert backend["faiss_active"] is True and backend["backend_actual"] == "faiss.IndexFlatIP"
        query = agent_trace["query"]
        assert query["original_tokens"] == SAMPLE_QUERY
        assert query["normalized_tokens"] == SAMPLE_QUERY
        assert query["source"] == "config_default_session"
        scores = [
            score
            for step in agent_trace["steps"]
            for score in ((step.get("tool_return") or {}).get("rerank_scores") or [])
        ]
        assert scores and scores == sorted(scores, reverse=True), scores
        assert '"item_id"' not in serialized_trace, "轨迹里出现了候选 item_id"
        assert agent_trace["candidates_persisted"] is False
        assert agent_trace["model_free_text_persisted"] is False
        return (
            "scripted_driver：首轮 search_hybrid → 反馈阶段 search_rerank → 最终回复；"
            "状态 verified，FAISS 后端已核对，轨迹无候选 ID / 无模型正文"
        )

    report.check(
        "agent_end_to_end",
        agent_chain,
        {
            "driver": agent_metrics["driver"],
            "status": agent_metrics["status"],
            "tool_sequence": agent_metrics["summary"]["tool_sequence"],
            "request_count": agent_metrics["budget"]["request_count"],
            "search_calls": agent_metrics["budget"]["search_calls"],
            "rerank_calls": agent_metrics["budget"]["rerank_calls"],
            "rerank_result_count": agent_metrics["summary"]["rerank_result_count"],
            "rerank_scores": [
                score
                for step in agent_trace["steps"]
                for score in ((step.get("tool_return") or {}).get("rerank_scores") or [])
            ],
            "checks": agent_metrics["checks"],
            "trace_file": _display(agent_trace_path),
            "note": "决策由脚本替身给出；检索、精排、索引读取与落盘均为真实代码",
        },
    )

    # 7. 空结果与失败分支 ------------------------------------------------------
    toolbox = SearchToolbox(service, max_results=5)
    no_match = toolbox.run(TOOL_HYBRID, {"query_tokens": [NO_MATCH_TOKEN], "top_k": 5})
    try:
        toolbox.run(TOOL_NAME, {"query_tokens": []})
    except ToolRejected as error:
        empty_query_code = error.code
    else:
        empty_query_code = None
    rerank_first = agent.run_scripted(
        agent_config,
        plan="rerank_first",
        output_root=agent_root,
    )
    rerank_first_metrics = rerank_first["metrics"]
    no_match_agent = agent.run_scripted(
        agent_config,
        plan="hybrid_stop",
        tokens=[NO_MATCH_TOKEN],
        output_root=agent_root,
    )
    no_match_metrics = no_match_agent["metrics"]

    def empty_and_failure() -> str:
        assert no_match["count"] == 0, no_match
        assert empty_query_code == "empty_query", empty_query_code
        assert rerank_first_metrics["status"] == "incomplete", rerank_first_metrics["status"]
        assert "tool_not_allowed" in str(rerank_first_metrics["summary"]["stop_code"])
        assert no_match_metrics["status"] == "verified", no_match_metrics["status"]
        assert no_match_metrics["summary"]["return_counts"] == [0], no_match_metrics["summary"]
        return (
            "无匹配 token：混合检索返回 0 条，Agent 链路以 0 结果 verified 结束；"
            "空查询被拒绝（empty_query）；首轮直接请求精排被拒绝（tool_not_allowed → incomplete）"
        )

    report.check(
        "empty_and_failure_branch",
        empty_and_failure,
        {
            "no_match_hybrid": {"query_tokens": [NO_MATCH_TOKEN], "count": no_match["count"]},
            "empty_query_rejection": empty_query_code,
            "rerank_first": {
                "status": rerank_first_metrics["status"],
                "stop_code": rerank_first_metrics["summary"]["stop_code"],
            },
            "agent_no_match": {
                "status": no_match_metrics["status"],
                "return_counts": no_match_metrics["summary"]["return_counts"],
                "tool_sequence": no_match_metrics["summary"]["tool_sequence"],
            },
        },
    )

    # 8. 文本语义编码的已知缺口 -------------------------------------------------
    def text_encoder_gap() -> str:
        try:
            TextSemanticEncoder(model_path="")
        except EncoderUnavailable as error:
            message = str(error)
            assert "权重" in message or "token" in message, message
            return f"TextSemanticEncoder 按预期不可用：{message[:120]}…"
        raise AssertionError("缺少文本/权重时 TextSemanticEncoder 本应报错")

    report.check("text_encoder_blocked", text_encoder_gap)

    # 9. 只读约束 --------------------------------------------------------------
    def readonly_inputs() -> str:
        assert _sha256_file(bm25_path) == bm25_sha_before, "验收过程改写了 BM25 索引"
        assert _sha256_file(paths["bm25_sample"]) == sample_sha_before, "验收过程改写了样本"
        return "验收前后 BM25 索引与样本 sha256 一致；未重建 BM25、未扫描原始 CSV"

    report.check("readonly_inputs", readonly_inputs)

    return {
        "format": "recsys.search.phase2_acceptance.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "passed": report.passed,
        "total": len(report.checks),
        "passed_count": sum(1 for item in report.checks if item["status"] == "pass"),
        "checks": report.checks,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "faiss": faiss_version(),
        },
        "components_reused_from_skimsearchagent": [
            f"agent_search.retrievers.dense.vector_index (@{UPSTREAM_SHORT})：索引保存/加载/元数据",
            f"agent_search.retrievers.fusion.rrf (@{UPSTREAM_SHORT})：RRF 融合",
            f"agent_search.retrievers.rerankers (@{UPSTREAM_SHORT})：Reranker 契约与注册表",
            f"agent_search.retrievers.reranked (@{UPSTREAM_SHORT})：RerankedEngine 组合",
        ],
        "adapters": [
            "FaissFlatIndex：faiss.IndexFlatIP + 整数 item_id 映射（上游只有 numpy/hnsw/ivfpq）",
            "TokenHashEncoder：无训练的整数 token 哈希编码（查询/文档同配置）",
            "HybridSearcher：在 RRF 之外补来源/名次/分数的追溯",
            "TokenOverlapReranker：基于现有 token 的轻量精排规则",
            "SearchService / SearchToolbox / Search Agent 四工具注册",
        ],
        "evidence": report.evidence,
        "scope": {
            "training": 0,
            "fine_tuning": 0,
            "bm25_rebuild": 0,
            "full_corpus_index": 0,
            "raw_csv_scan": 0,
            "real_llm_requests": 0,
            "real_llm_note": (
                "验收脚本不联网；另有一次真实 LLM 工作流记录在 "
                "experiments/2026-11-09-kuasar-search-phase2/agent_llm/ 下"
            ),
            "agent_decisions": "scripted_driver",
            "retrieval_and_rerank_executed": True,
            "click_labels_used_for_scoring": False,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--update-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_acceptance(args.config, update_dir=args.update_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for item in result["checks"]:
        print(f"[acceptance] {item['id']}: {item['status']} — {item['detail']}")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "passed_count": result["passed_count"],
                "total": result["total"],
                "output": _display(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
