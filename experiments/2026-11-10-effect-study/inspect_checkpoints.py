import glob
import os

import torch

rows = []
for path in sorted(glob.glob("artifacts/kuasar_small/*.pt")):
    name = os.path.basename(path)
    if name.startswith(("sweep", "mind_p3", "mind_smoke", "lightgcn_smoke", "rqvae", "timing")):
        continue
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001
        rows.append((name, "ERR", str(error)[:50]))
        continue
    config = blob.get("train_config") or {}
    validation = blob.get("best_validation_metrics") or {}
    rows.append((
        name,
        f"epochs={config.get('epochs')} max_batches={config.get('max_batches_per_epoch')}",
        f"seed={config.get('seed')}",
        f"val Recall@20={validation.get('Recall@20')}",
    ))
for row in rows:
    print(" | ".join(str(value) for value in row))
