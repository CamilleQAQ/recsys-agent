"""Phase 5 工件清单：加载、逐文件校验、业务实际使用核对。

清单（``configs/platform/phase5_artifacts.toml``）登记演示与恢复真正加载的文件。
这里的校验刻意「只读 + 严格」：

* 文件不存在、大小变化、sha256 不匹配 → 记录到 ``failed``，并给出具体路径；
* metadata 里的结构字段（索引大小、向量维度、权重类型、RAG 计数）与文件内容
  不一致 → 同样失败；
* 不写任何文件、不训练、不重建索引。

``verify_business_usage`` 用业务运行自己落盘的工件身份（推荐 run_meta 的
checkpoints、搜索 metrics 的 index、广告工具返回的 weight）核对本次运行到底
加载了哪些文件，避免「配置写了但链路没用」的假证据。
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Mapping

from recsys.utils.artifact_guard import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = Path("configs/platform/phase5_artifacts.toml")
MANIFEST_VERSION = 1

REQUIRED_FIELDS = ("id", "business", "role", "path", "sha256")


class ManifestError(RuntimeError):
    """清单本身不可用（缺文件、格式错误、版本不支持）。"""


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(value: str | Path) -> str:
    path = resolve_path(value)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(value)


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    target = resolve_path(path or DEFAULT_MANIFEST_PATH)
    if not target.is_file():
        raise ManifestError(f"工件清单不存在：{target}")
    try:
        payload = tomllib.loads(target.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ManifestError(f"工件清单无法解析：{target}：{error}") from error
    version = int(payload.get("manifest_version") or 0)
    if version != MANIFEST_VERSION:
        raise ManifestError(
            f"工件清单版本 {version} 不受支持（需要 {MANIFEST_VERSION}）：{target}"
        )
    entries = payload.get("artifacts") or []
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"工件清单没有 [[artifacts]]：{target}")
    seen: set[str] = set()
    for entry in entries:
        missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
        if missing:
            raise ManifestError(f"工件 {entry.get('id') or '<no id>'} 缺少字段 {missing}")
        artifact_id = str(entry["id"])
        if artifact_id in seen:
            raise ManifestError(f"工件 id 重复：{artifact_id}")
        seen.add(artifact_id)
    payload["manifest_path"] = display_path(target)
    payload["manifest_sha256"] = sha256_file(target)
    return payload


def entries(manifest: Mapping[str, Any], *, business: str | None = None) -> list[dict[str, Any]]:
    items = [dict(item) for item in manifest.get("artifacts") or []]
    if business:
        items = [item for item in items if str(item.get("business")) == business]
    return items


def get_entry(manifest: Mapping[str, Any], artifact_id: str) -> dict[str, Any]:
    for item in manifest.get("artifacts") or []:
        if str(item.get("id")) == artifact_id:
            return dict(item)
    raise ManifestError(f"工件清单里没有 id={artifact_id}")


def identity(manifest: Mapping[str, Any], artifact_id: str) -> dict[str, Any]:
    entry = get_entry(manifest, artifact_id)
    return {
        "id": entry.get("id"),
        "business": entry.get("business"),
        "role": entry.get("role"),
        "path": display_path(entry["path"]),
        "sha256": entry.get("sha256"),
        "size_bytes": entry.get("size_bytes"),
        "source": entry.get("source"),
        "weight_kind": (entry.get("metadata") or {}).get("weight_kind"),
    }


# ---------------------------------------------------------------------- 校验


def _check_metadata(
    entry: Mapping[str, Any],
    path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """按 artifact id 做结构校验；只读已加载内容，不重建。"""

    artifact_id = str(entry.get("id"))
    metadata = dict(entry.get("metadata") or {})
    detail: dict[str, Any] = {"checked": [], "ok": True, "observed": {}}

    def fail(message: str) -> None:
        detail["ok"] = False
        detail.setdefault("errors", []).append(message)

    if artifact_id == "search_bm25_index":
        from recsys.search.bm25 import BM25Index

        index = BM25Index.load(path)
        detail["observed"].update(
            {"num_docs": index.num_docs, "vocab_size": index.vocab_size, "k1": index.k1, "b": index.b}
        )
        detail["checked"] += ["num_docs", "vocab_size", "k1", "b"]
        for key in ("num_docs", "vocab_size"):
            if key in metadata and int(metadata[key]) != int(detail["observed"][key]):
                fail(f"BM25 {key} 不一致：文件 {detail['observed'][key]}，清单 {metadata[key]}")
        for key in ("k1", "b"):
            if key in metadata and abs(float(metadata[key]) - float(detail["observed"][key])) > 1e-9:
                fail(f"BM25 {key} 不一致：文件 {detail['observed'][key]}，清单 {metadata[key]}")
    elif artifact_id == "search_sample":
        sample = json.loads(path.read_text(encoding="utf-8"))
        sessions = len(sample.get("sessions") or [])
        detail["observed"]["sessions"] = sessions
        detail["checked"].append("sessions")
        if "sessions" in metadata and int(metadata["sessions"]) != sessions:
            fail(f"样本会话数不一致：文件 {sessions}，清单 {metadata['sessions']}")
    elif artifact_id in {"search_vector_manifest", "search_vector_payload"}:
        manifest_path = (
            resolve_path(get_entry(manifest, "search_vector_manifest")["path"])
            if artifact_id == "search_vector_payload"
            else path
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = payload.get("index") or {}
        encoder = payload.get("encoder") or {}
        detail["observed"].update(
            {
                "num_items": index.get("num_items"),
                "dim": index.get("dim") or encoder.get("dim"),
                "encoder": encoder.get("encoder"),
                "payload_sha256": index.get("payload_sha256"),
            }
        )
        detail["checked"] += ["num_items", "dim", "encoder", "payload_sha256"]
        for key in ("num_items", "dim"):
            if key in metadata and int(metadata[key]) != int(detail["observed"][key] or -1):
                fail(f"向量索引 {key} 不一致：manifest {detail['observed'][key]}，清单 {metadata[key]}")
        if "encoder" in metadata and str(metadata["encoder"]) != str(detail["observed"]["encoder"]):
            fail(
                f"向量编码器不一致：manifest {detail['observed']['encoder']}，"
                f"清单 {metadata['encoder']}"
            )
        if artifact_id == "search_vector_payload":
            payload_sha = str(detail["observed"].get("payload_sha256") or "")
            if payload_sha and payload_sha != str(entry.get("sha256")):
                fail(
                    "faiss 载荷哈希与向量 manifest 记录不一致："
                    f"manifest {payload_sha}，清单 {entry.get('sha256')}"
                )
    elif artifact_id == "ads_weight_index":
        payload = json.loads(path.read_text(encoding="utf-8"))
        detail["observed"].update(
            {
                "weight_kind": payload.get("weight_kind"),
                "is_trained": payload.get("is_trained"),
                "num_items": payload.get("num_items"),
            }
        )
        detail["checked"] += ["weight_kind", "num_items"]
        if "weight_kind" in metadata and str(metadata["weight_kind"]) != str(
            detail["observed"]["weight_kind"]
        ):
            fail(
                f"权重类型不一致：索引 {detail['observed']['weight_kind']}，"
                f"清单 {metadata['weight_kind']}"
            )
        if "num_items" in metadata and int(metadata["num_items"]) != int(
            detail["observed"]["num_items"] or -1
        ):
            fail("权重索引 num_items 不一致")
        din = (payload.get("files") or {}).get("din") or {}
        if din.get("sha256"):
            din_entry = get_entry(manifest, "ads_din_weights")
            if str(din["sha256"]) != str(din_entry.get("sha256")):
                fail("din 权重哈希与权重索引记录不一致")
    elif artifact_id == "rag_index":
        from recsys.platform.rag import RagIndex

        description = RagIndex(path.parent).describe(include_sources=False)
        stats = description.get("stats") or {}
        detail["observed"].update(stats)
        detail["checked"] += ["files", "chunks", "vocab"]
        for key in ("files", "chunks", "vocab"):
            if key in metadata and int(metadata[key]) != int(stats.get(key) or -1):
                fail(f"RAG 索引 {key} 不一致：文件 {stats.get(key)}，清单 {metadata[key]}")
    elif artifact_id == "rec_lightgbm_ranker":
        try:
            import lightgbm

            booster = lightgbm.Booster(model_file=str(path))
            detail["observed"].update(
                {"num_trees": booster.num_trees(), "num_feature": booster.num_feature()}
            )
            detail["checked"] += ["num_trees", "num_feature"]
            for key in ("num_trees", "num_feature"):
                if key in metadata and int(metadata[key]) != int(detail["observed"][key]):
                    fail(
                        f"LightGBM {key} 不一致：文件 {detail['observed'][key]}，"
                        f"清单 {metadata[key]}"
                    )
        except Exception as error:  # noqa: BLE001 - 无法解析即为失败
            fail(f"LightGBM 权重无法解析：{type(error).__name__}: {error}")
    else:
        detail["checked"].append("size+sha256")
    return detail


def verify_manifest(
    manifest: Mapping[str, Any],
    *,
    check_metadata: bool = True,
    business: str | None = None,
) -> dict[str, Any]:
    """逐条校验清单；返回 ``ok`` / ``failed`` / 每条工件的观察值。"""

    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for entry in entries(manifest, business=business):
        path = resolve_path(entry["path"])
        item: dict[str, Any] = {
            "id": entry.get("id"),
            "business": entry.get("business"),
            "role": entry.get("role"),
            "path": display_path(path),
            "expected_sha256": entry.get("sha256"),
            "expected_size_bytes": entry.get("size_bytes"),
            "source": entry.get("source"),
            "checksum_recorded_in": entry.get("checksum_recorded_in"),
            "exists": path.is_file(),
            "ok": True,
        }
        if not path.exists():
            item["ok"] = False
            item["error"] = f"工件不存在：{path}"
            failed.append(item)
            results.append(item)
            continue
        if not path.is_file():
            item["ok"] = False
            item["error"] = f"工件不是文件：{path}"
            failed.append(item)
            results.append(item)
            continue
        actual_size = int(path.stat().st_size)
        item["size_bytes"] = actual_size
        if entry.get("size_bytes") is not None and int(entry["size_bytes"]) != actual_size:
            item["ok"] = False
            item.setdefault("errors", []).append(
                f"大小不一致：文件 {actual_size}，清单 {entry['size_bytes']}"
            )
        actual_sha = sha256_file(path)
        item["sha256"] = actual_sha
        if str(entry.get("sha256")).lower() != actual_sha.lower():
            item["ok"] = False
            item.setdefault("errors", []).append(
                f"sha256 不一致：文件 {actual_sha}，清单 {entry.get('sha256')}"
            )
        if check_metadata:
            detail = _check_metadata(entry, path, manifest)
            item["metadata_check"] = detail
            if not detail.get("ok"):
                item["ok"] = False
                item.setdefault("errors", []).extend(detail.get("errors") or [])
        if not item["ok"]:
            failed.append(item)
        results.append(item)

    return {
        "ok": not failed,
        "manifest_path": manifest.get("manifest_path"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "dataset": manifest.get("dataset"),
        "checked": len(results),
        "failed": failed,
        "artifacts": results,
    }


# ---------------------------------------------------------------------- 业务使用核对


def _sha_matches(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return False
    return str(expected).lower() == str(actual).lower()


def verify_artifact_usage(
    manifest: Mapping[str, Any], artifact_id: str, actual: Mapping[str, Any]
) -> dict[str, Any]:
    entry = get_entry(manifest, artifact_id)
    actual_path = actual.get("path")
    path_ok = bool(actual_path) and resolve_path(actual_path).resolve() == resolve_path(entry["path"]).resolve()
    sha_ok = _sha_matches(entry.get("sha256"), actual.get("sha256"))
    return {
        "artifact": artifact_id,
        "ok": path_ok and sha_ok,
        "path_ok": path_ok,
        "sha256_ok": sha_ok,
        "manifest_sha256": entry.get("sha256"),
        "actual_sha256": actual.get("sha256"),
        "actual_path": actual_path,
    }


def verify_business_usage(
    business: str,
    *,
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    require_vector: bool = True,
) -> dict[str, Any]:
    """核对一次业务运行实际加载的工件是否就是清单登记的那一个。"""

    checks: list[dict[str, Any]] = []
    ok = True

    def add(artifact_id: str, actual: Mapping[str, Any], *, note: str = "") -> None:
        nonlocal ok
        check = verify_artifact_usage(manifest, artifact_id, actual)
        ok = ok and check["ok"]
        checks.append({**check, "note": note})

    if business == "recommendation":
        model = record.get("model") or {}
        used = (model.get("used_artifacts") or {}) if isinstance(model, Mapping) else {}
        for name, artifact_id in (
            ("two_tower", "rec_two_tower"),
            ("mind", "rec_mind"),
            ("ranker", "rec_lightgbm_ranker"),
        ):
            actual = used.get(name) or {}
            add(artifact_id, actual, note=f"run_meta#checkpoints.{name}")
    elif business == "search":
        model = record.get("model") or {}
        used = (model.get("used_artifacts") or {}) if isinstance(model, Mapping) else {}
        index_actual = used.get("index") or {}
        add("search_bm25_index", index_actual, note="search_agent metrics#index")
        tools = ((record.get("result") or {}).get("metrics") or {}).get("tool_sequence") or []
        used_vector = bool({"search_vector", "search_hybrid", "search_rerank"}.intersection(tools))
        if require_vector or used_vector:
            actual_vector = (used.get("vector_payload") or {}) if used_vector else {}
            add("search_vector_payload", actual_vector, note="实际调用 FAISS 的索引文件")
    elif business == "ads":
        model = record.get("model") or {}
        used = (model.get("used_artifacts") or {}) if isinstance(model, Mapping) else {}
        weights = used.get("weights") or {}
        actual = {"path": weights.get("path"), "sha256": None}
        actual_path = weights.get("path")
        if actual_path:
            path = resolve_path(actual_path)
            if path.is_file():
                actual["sha256"] = sha256_file(path)
        add("ads_din_weights", actual, note="ads metrics#predict_ad_ctr.weight")
        # 权重索引里记的是 fixed_init_interface_test，registry 记的是 interface_test；
        # 两者都表示「固定初始化、未训练」，必须与 is_trained=false 同时成立。
        kind = str(weights.get("weight_kind") or "")
        kind_ok = kind in {"interface_test", "fixed_init_interface_test"} and not bool(
            weights.get("is_trained")
        )
        ok = ok and kind_ok
        checks.append(
            {
                "artifact": "ads_din_weights",
                "ok": kind_ok,
                "weight_kind": kind,
                "is_trained": weights.get("is_trained"),
                "note": "必须标注 interface_test / fixed_init_interface_test（未训练）",
            }
        )
    else:
        raise ManifestError(f"未知业务：{business}")

    return {"ok": bool(ok), "business": business, "checks": checks}


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "MANIFEST_VERSION",
    "ManifestError",
    "display_path",
    "entries",
    "get_entry",
    "identity",
    "load_manifest",
    "resolve_path",
    "verify_business_usage",
    "verify_artifact_usage",
    "verify_manifest",
]
