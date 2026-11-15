"""出厂排序器为什么会把命中排掉：看它保留/挤掉的命中在物品热度上的差别。

    python diagnose_legacy_ranker.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO, load_data, load_fusion, targets_by_user  # noqa: E402

from recsys.adapters.ranking import LightGBMRankerAdapter  # noqa: E402
from recsys.data.kuasar_ranking import ITEM_FEATURE_COLUMNS  # noqa: E402

LEGACY = REPO / "artifacts" / "kuasar_small" / "mvp" / "lgbm_ranker.txt"
OUT = Path(__file__).resolve().parent / "legacy_ranker_diagnosis.json"


def main() -> int:
    data = load_data()
    ground_truth = targets_by_user(data, "test")
    pool = load_fusion()
    ranker = LightGBMRankerAdapter.load(LEGACY)
    clicks = np.zeros(data.num_items, dtype=np.int64)
    for row in data.train_interactions:
        clicks[row.item_idx] += 1
    author_freq = np.asarray(data.item_features.column("author_freq"), dtype=np.float64)

    kept, dropped, promoted = [], [], []
    for user, rows in pool.items():
        items = np.asarray([row["item"] for row in rows], dtype=np.int64)
        matrix = np.stack([
            np.asarray(data.item_features.column(name), dtype=np.float32)[items]
            for name in ITEM_FEATURE_COLUMNS
        ], axis=1)
        scores = ranker.predict(matrix)
        order = np.lexsort((np.arange(items.size), -scores))
        ranked = items[order]
        fusion_top = set(items[:20].tolist())
        ranked_top = set(ranked[:20].tolist())
        hits = fusion_top & ground_truth[user]
        kept.extend(hits & ranked_top)
        dropped.extend(hits - ranked_top)
        promoted.extend(ranked_top - hits)

    def summary(items):
        if not items:
            return {"n": 0}
        array = np.asarray(items, dtype=np.int64)
        return {
            "n": len(items),
            "mean_train_clicks": round(float(clicks[array].mean()), 2),
            "median_train_clicks": float(np.median(clicks[array])),
            "zero_click_share": round(float((clicks[array] == 0).mean()), 3),
            "mean_author_freq": round(float(author_freq[array].mean()), 1),
        }

    payload = {
        "fusion_top20_hits": len(kept) + len(dropped),
        "kept_hits": summary(kept),
        "dropped_hits": summary(dropped),
        "promoted_non_hits": summary(promoted),
        "reading": (
            "kept/dropped 都是融合 top-20 里的命中；promoted 是排序器放进 top-20 的非命中。"
            "若 promoted 的 train_clicks 显著高于 dropped，说明它用头部热度换掉了长尾命中。"
        ),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
