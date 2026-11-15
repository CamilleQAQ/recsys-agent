"""六个广告模型的接口验收：固定初始化权重 → 推理 → 保存/加载往返。

**没有训练。** 这里构建的是固定初始化权重（seed 固定），只用于证明：

1. 模型能在 KuaiSAR 广告样本上构造并完成一次真实前向推理；
2. 权重能保存到独立目录、重新加载后逐元素一致；
3. 多任务模型（MMoE/PLE）在显式标注的模拟转化样本上输出 CTR/CVR 两路结果。

因此输出状态统一写 ``interface_test``，``is_trained=false``；不能表述为已训练、
已收敛或有效果。测试权重只写 ``artifacts/<dataset>/ads/interface_weights/``，
绝不写业务默认目录 ``artifacts/<dataset>/ads/weights/``。

用法：

    python -m recsys.ads.verify --build-interface-weights
    python -m recsys.ads.verify --models deepfm,mmoe,ple
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from recsys.ads import AD_MODEL_NAMES
from recsys.ads.config import DEFAULT_ADS_CONFIG, ModelConfig, load_ads_config
from recsys.ads.features import build_click_metrics
from recsys.ads.models import AdModelConfig, AdModelUnavailable, build_ad_model, load_ad_model, model_spec
from recsys.ads.registry import interface_weights_root, weights_root
from recsys.ads.runtime import AdsRuntime
from recsys.config import DEFAULT_DATA_DIR

AD_MODELS = AD_MODEL_NAMES

DEFAULT_OUTPUT = Path("experiments") / "2026-11-09-kuasar_small-ads-phase3" / "model_check.json"
WEIGHT_KIND_INTERFACE = "fixed_init_interface_test"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ad_config(models: ModelConfig, device: str | None = None) -> AdModelConfig:
    return AdModelConfig(
        embedding_dim=models.embedding_dim,
        hidden_dim=models.hidden_dim,
        attention_dim=models.attention_dim,
        dropout=models.dropout,
        seed=models.seed,
        device=device or models.device,
        batch_size=models.batch_size,
        max_history=models.max_history,
    )


def build_interface_weights(
    runtime: AdsRuntime,
    *,
    models: tuple[str, ...] = AD_MODELS,
    force: bool = True,
) -> dict[str, Any]:
    target = interface_weights_root(runtime.artifacts_dir, runtime.dataset)
    default = weights_root(runtime.artifacts_dir, runtime.dataset)
    if target.resolve() == default.resolve():
        raise RuntimeError("接口测试权重目录不能与业务默认权重目录相同")
    target.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {}
    for name in models:
        path = target / f"{name}.pt"
        if path.exists() and not force:
            index[name] = {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "reused": True,
            }
            print(f"[reuse] {path}")
            continue
        adapter = build_ad_model(name, runtime.data(), runtime.model_config)
        adapter.save(
            path,
            weight_kind=WEIGHT_KIND_INTERFACE,
            dataset=runtime.dataset,
            extra={
                "is_trained": False,
                "purpose": "interface_test_only",
                "note": (
                    "固定初始化权重（seed=%d）：只用于接口/推理/保存加载检查，"
                    "不是训练产物，不代表效果。" % runtime.model_config.seed
                ),
                "num_items": int(runtime.data().num_items),
                "seed": runtime.model_config.seed,
            },
        )
        index[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "reused": False,
        }
        print(f"[ok] {path} ({index[name]['size_bytes'] / 1e6:.1f} MB)")
    (target / "index.json").write_text(
        json.dumps(
            {
                "weight_kind": WEIGHT_KIND_INTERFACE,
                "is_trained": False,
                "dataset": runtime.dataset,
                "num_items": int(runtime.data().num_items),
                "files": index,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return index


def _fixed_batch(runtime: AdsRuntime, rows: int = 32):
    view = runtime.view("test")
    row_index = np.arange(min(len(view), rows), dtype=np.int64)
    return runtime.batch("test", row_index), row_index


def verify_one(runtime: AdsRuntime, name: str) -> dict[str, Any]:
    spec = model_spec(name)
    weight_path = interface_weights_root(runtime.artifacts_dir, runtime.dataset) / f"{name}.pt"
    if not weight_path.exists():
        raise AdModelUnavailable(
            f"{name} 缺少接口测试权重：先运行 python -m recsys.ads.verify --build-interface-weights"
        )
    batch, _rows = _fixed_batch(runtime)
    adapter, payload = load_ad_model(
        weight_path,
        runtime.data(),
        runtime.model_config,
        expected_items=int(runtime.data().num_items),
    )
    outputs = adapter.score(batch.as_dict())
    reloaded, reload_payload = load_ad_model(
        weight_path,
        runtime.data(),
        runtime.model_config,
        expected_items=int(runtime.data().num_items),
    )
    reloaded_outputs = reloaded.score(batch.as_dict())
    differences = {
        task: float(np.max(np.abs(np.asarray(values) - np.asarray(reloaded_outputs[task]))))
        for task, values in outputs.items()
    }

    view = runtime.view("test")
    full_rows = np.arange(len(view), dtype=np.int64)
    full_batch = runtime.batch("test", full_rows)
    test_outputs = adapter.score(full_batch.as_dict())
    test_metrics = build_click_metrics(full_batch.labels, test_outputs["ctr"])

    multitask = None
    if spec.task == "multitask":
        simulated_view = runtime.simulated_view()
        simulated_rows = np.arange(len(simulated_view), dtype=np.int64)
        simulated_batch = runtime.simulated_batch(simulated_rows)
        simulated_outputs = adapter.score(simulated_batch.as_dict())
        multitask = {
            "rows": int(len(simulated_view)),
            "synthetic": True,
            "not_real_conversion": True,
            "tasks": sorted(simulated_outputs),
            "ctr_mean": round(float(np.mean(simulated_outputs["ctr"])), 6),
            "cvr_mean": round(float(np.mean(simulated_outputs["cvr"])), 6),
            "ctr_metrics": build_click_metrics(
                simulated_batch.labels, simulated_outputs["ctr"]
            ),
            "cvr_metrics": build_click_metrics(
                np.asarray(simulated_view.conversion, dtype=np.float64),
                simulated_outputs["cvr"],
            ),
        }

    return {
        "model": name,
        "display": spec.display,
        "framework": spec.framework,
        "task": spec.task,
        "needs_history": spec.needs_history,
        "status": "interface_test",
        "is_trained": False,
        "checkpoint": str(weight_path),
        "checkpoint_sha256": _sha256(weight_path),
        "checkpoint_size_bytes": weight_path.stat().st_size,
        "weight_kind": payload.get("weight_kind"),
        "inference_ok": True,
        "inference_rows": int(len(batch)),
        "save_load_max_abs_diff": differences,
        "save_load_ok": all(value <= 1e-6 for value in differences.values()),
        "test_metrics": test_metrics,
        "multitask_check": multitask,
        "note": "固定初始化权重（未训练）；只证明接口可用，不代表效果。",
    }


def verify_all(
    runtime: AdsRuntime,
    *,
    models: tuple[str, ...] = AD_MODELS,
    build: bool = True,
) -> dict[str, Any]:
    if build:
        build_interface_weights(runtime, models=models)
    results: dict[str, Any] = {}
    errors: list[str] = []
    for name in models:
        try:
            results[name] = verify_one(runtime, name)
            print(
                f"[verify] {name}: inference=ok load_diff="
                f"{results[name]['save_load_max_abs_diff']} "
                f"test_AUC={results[name]['test_metrics']['AUC']:.4f}"
            )
        except Exception as error:  # noqa: BLE001 - 逐个模型记录，不中断其余模型
            errors.append(f"{name}: {type(error).__name__}: {error}")
            results[name] = {
                "model": name,
                "status": "failed",
                "inference_ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"[verify] {name}: FAILED {type(error).__name__}: {error}")
    summary = {
        "models": len(models),
        "inference_ok": sum(1 for item in results.values() if item.get("inference_ok")),
        "save_load_ok": sum(1 for item in results.values() if item.get("save_load_ok")),
        "trained": 0,
        "multitask_ok": sum(
            1 for item in results.values() if item.get("multitask_check")
        ),
        "errors": errors,
    }
    return {
        "dataset": runtime.dataset,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "weight_dir": str(interface_weights_root(runtime.artifacts_dir, runtime.dataset)),
        "default_weight_dir": str(weights_root(runtime.artifacts_dir, runtime.dataset)),
        "weight_kind": WEIGHT_KIND_INTERFACE,
        "is_trained": False,
        "results": results,
        "summary": summary,
        "limits": [
            "没有训练：所有指标来自固定初始化权重，AUC/LogLoss 只用于确认输出可用。",
            "CVR 只在 synthetic 模拟样本上验证，KuaiSAR 没有真实转化标签。",
            "接口测试权重不写入业务默认权重目录。",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="kuasar_small")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=DEFAULT_ADS_CONFIG)
    parser.add_argument("--models", default=",".join(AD_MODELS))
    parser.add_argument("--build-interface-weights", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    models = tuple(part.strip() for part in args.models.split(",") if part.strip())
    config = load_ads_config(args.config)
    runtime = AdsRuntime(
        dataset=args.dataset,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        model_config=_ad_config(config.models, args.device),
    )
    report = verify_all(runtime, models=models, build=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"[ok] 报告写入 {args.output}")
    return 0 if report["summary"]["inference_ok"] == len(models) else 1


if __name__ == "__main__":
    raise SystemExit(main())
