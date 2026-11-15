import glob
import json
import os

import torch

rows = []
for path in sorted(glob.glob("artifacts/effect_study/kuasar_small/*.pt")):
    name = os.path.basename(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = blob.get("train_config") or {}
    validation = blob.get("best_validation_metrics") or {}
    history = blob.get("epoch_history") or []
    curve = [
        round(float(entry.get("validation", {}).get("Recall@20", -1)), 6)
        for entry in history
        if entry.get("validation")
    ]
    rows.append({
        "checkpoint": name,
        "epochs": config.get("epochs"),
        "seed": config.get("seed"),
        "validation_users": config.get("validation_users"),
        "best_epoch": blob.get("best_epoch"),
        "best_val_recall20": validation.get("Recall@20"),
        "val_curve_tail": curve[-4:] if curve else None,
    })

print(json.dumps(rows, ensure_ascii=False, indent=2))
