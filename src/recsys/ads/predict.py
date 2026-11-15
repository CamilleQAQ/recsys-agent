"""广告 CTR / 多任务预测与排序 CLI。

    python -m recsys.ads.predict --model din --split test
    python -m recsys.ads.predict --model deepfm --mode rank --split train --requests 3 --min-candidates 2
    python -m recsys.ads.predict --model mmoe --mode multitask

所有结果都带 ``weight`` 元信息：``status=interface_test`` 表示固定初始化权重，
只证明接口可推理，不代表训练或效果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys.ads import AD_MODEL_NAMES
from recsys.ads.config import DEFAULT_ADS_CONFIG, load_ads_config
from recsys.ads.models import AdModelConfig, AdModelUnavailable
from recsys.ads.runtime import AdsRuntime
from recsys.ads.service import predict_ad_ctr, predict_multitask, rank_ad_candidates
from recsys.config import DEFAULT_DATA_DIR


def _model_config(config) -> AdModelConfig:
    models = config.models
    return AdModelConfig(
        embedding_dim=models.embedding_dim,
        hidden_dim=models.hidden_dim,
        attention_dim=models.attention_dim,
        dropout=models.dropout,
        seed=models.seed,
        max_history=models.max_history,
        batch_size=models.batch_size,
        device=models.device,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=AD_MODEL_NAMES, default="din")
    parser.add_argument("--mode", choices=("predict", "rank", "multitask"), default="predict")
    parser.add_argument("--dataset", default="kuasar_small")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=DEFAULT_ADS_CONFIG)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--preview", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_ads_config(args.config)
    runtime = AdsRuntime(
        dataset=args.dataset,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        model_config=_model_config(config),
    )
    try:
        if args.mode == "predict":
            payload = predict_ad_ctr(
                runtime,
                args.model,
                split=args.split,
                limit=args.limit,
                preview=args.preview,
            )
        elif args.mode == "rank":
            payload = rank_ad_candidates(
                runtime,
                args.model,
                split=args.split,
                requests=args.requests,
                min_candidates=args.min_candidates,
                top_k=args.top_k,
            )
        else:
            payload = predict_multitask(
                runtime, args.model, limit=args.limit, preview=args.preview
            )
    except AdModelUnavailable as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"[ok] 结果写入 {args.output}")
        summary = {
            key: payload.get(key)
            for key in ("ok", "model", "split", "rows", "metrics", "summary", "weight")
            if key in payload
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2)[:2000])
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
