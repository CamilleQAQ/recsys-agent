"""MMRec-VBPR 适配器：内容增强协同过滤接入统一召回接口。"""

from __future__ import annotations

from pathlib import Path

import torch

from recsys.adapters.base import AdapterError
from recsys.adapters.builtin import EmbeddingCheckpointAdapter, _load_checkpoint
from recsys.models.mmrec_vbpr import MMRecVBPR


class MMRecAdapter(EmbeddingCheckpointAdapter):
    model_id = "mmrec_multimodal"
    model_type = "embedding"
    model_class = MMRecVBPR

    @classmethod
    def load(cls, checkpoint_path: Path, device: torch.device) -> "MMRecAdapter":
        checkpoint = _load_checkpoint(checkpoint_path, device)
        config = checkpoint.get("model_config")
        if not config:
            raise AdapterError(f"{checkpoint_path} 缺少 model_config")
        model = MMRecVBPR(**config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(model, checkpoint, device)


__all__ = ["MMRecAdapter"]
