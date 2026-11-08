"""汇总 Phase 3 MIND 实验：逐轮轨迹、三配置对照、三 seed 测试、分群结果。

在训练与分群评测都跑完之后执行：

    python scripts/mind_phase3_report.py

输入：
* artifacts/kuasar_small/mind_p3_{A42,B42,C42}.pt      三组超参（seed 42）
* artifacts/kuasar_small/mind_p3_{WINNER}{43,44}.pt    胜出配置的另外两个 seed
* artifacts/_eval/*.json                                scripts/eval_groups.py 的输出

输出：
* experiments/2026-11-08-kuasar_small-mind-phase3/metrics.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "2026-11-08-kuasar_small-mind-phase3"


def read_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    history = checkpoint.get("epoch_history", [])
    return {
        "path": str(path.relative_to(ROOT)),
        "model_config": checkpoint["model_config"],
        "best_epoch": checkpoint["best_epoch"],
        "best_validation": {
            key: round(float(value), 6)
            for key, value in checkpoint["best_validation_metrics"].items()
        },
        "train_seconds": checkpoint.get("train_seconds"),
        "epochs": len(history),
        "history": history,
        "provenance": checkpoint["provenance"]["data"]["counts"],
    }


def plateau_verdict(history: list[dict], best_epoch: int) -> dict:
    """按最后 5 轮走势与最佳轮次判断是否出现平台期。"""
    if len(history) < 5:
        return {"verdict": "轮次不足，无法判断", "last5": []}
    last5 = [round(entry["validation"]["Recall@20"], 6) for entry in history[-5:]]
    best = max(entry["validation"]["Recall@20"] for entry in history)
    tail_gain = last5[-1] - last5[0]
    return {
        "last5_recall20": last5,
        "best_epoch": best_epoch,
        "last5_gain": round(tail_gain, 6),
        "verdict": (
            "预算内未观察到平台期"
            if best_epoch >= len(history) - 2 or tail_gain > 0.0005
            else "出现平台期"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", default="A", help="胜出配置字母（A/B/C）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args(argv)

    metrics: dict = {"protocol": {}, "configs": {}, "winner": args.winner, "seeds": {}}

    for name in ("A", "B", "C"):
        path = ROOT / "artifacts" / "kuasar_small" / f"mind_p3_{name}42.pt"
        if not path.exists():
            print(f"[skip] 缺少 {path}")
            continue
        record = read_checkpoint(path)
        record["plateau"] = plateau_verdict(record["history"], record["best_epoch"])
        metrics["configs"][name] = record
        print(f"{name}: best_epoch={record['best_epoch']} "
              f"val_R@20={record['best_validation'].get('Recall@20')} "
              f"train={record['train_seconds']}s plateau={record['plateau']['verdict']}")

    for seed in args.seeds:
        path = ROOT / "artifacts" / "kuasar_small" / f"mind_p3_{args.winner}{seed}.pt"
        eval_path = ROOT / "artifacts" / "_eval" / f"{args.winner}{seed}_test.json"
        if not path.exists() or not eval_path.exists():
            print(f"[skip] seed {seed} 缺少 checkpoint 或评测结果")
            continue
        record = read_checkpoint(path)
        record["test"] = json.loads(eval_path.read_text(encoding="utf-8"))
        metrics["seeds"][str(seed)] = record

    baseline_eval = ROOT / "artifacts" / "_eval" / "two_tower_features_test.json"
    if baseline_eval.exists():
        metrics["baseline"] = json.loads(baseline_eval.read_text(encoding="utf-8"))

    # 三个 seed 的测试汇总（均值/样本标准差/相对基线的绝对差值）
    summary: dict = {}
    seed_entries = list(metrics["seeds"].values())
    if seed_entries:
        keys = [key for key in seed_entries[0]["test"]["overall"] if key.startswith(("Recall", "NDCG", "HitRate", "Coverage"))]
        for key in keys:
            values = [entry["test"]["overall"][key] for entry in seed_entries]
            summary[key] = {
                "mean": round(statistics.fmean(values), 6),
                "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
                "values": values,
            }
            if "baseline" in metrics:
                base = metrics["baseline"]["overall"].get(key)
                if base is not None:
                    summary[key]["baseline"] = base
                    summary[key]["delta_vs_baseline"] = round(statistics.fmean(values) - base, 6)
    metrics["test_summary"] = summary

    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ok] 写入 {EXPERIMENT / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
