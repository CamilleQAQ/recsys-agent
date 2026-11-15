from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import json

from recsys.pipeline import ranking
from recsys.pipeline.chain import RecommendationPipeline, metrics_for
from recsys.pipeline.config import load_pipeline_config

root=Path("experiments/2026-11-10-formal-e2e")
out=root/"recommendation"
cfg_path=Path("experiments/2026-11-10-kuasar-small-test-512/run-20261110-134155-d7bc6c/config.toml")
cfg=load_pipeline_config(cfg_path)
assert cfg.ranking.train_if_missing is False
pipeline=RecommendationPipeline(cfg)
started=perf_counter()
data=pipeline.data
users=sorted({r.user_idx for r in data.test_interactions})
assert len(users)==2000 and len(data.test_interactions)==4275
with patch.object(ranking,"train_lightgbm",side_effect=AssertionError("existing weights only")):
    result=pipeline.run(split="test",users=users,output_dir=out)
for user in users:
    assert all(len(ch[user])==100 for ch in result.channels.values())
    union={x.item for ch in result.channels.values() for x in ch[user]}
    fused={x.item for x in result.fused[user]}
    ranked={x.item for x in result.ranked[user]}
    final=[x.item for x in result.final[user]]
    assert fused<=union and ranked==fused and set(final)<=ranked and len(final)==len(set(final))==20
fused=metrics_for({u:[c.item for c in row] for u,row in result.fused.items()},data.test_interactions,(100,))
payload={"users":len(users),"items":data.num_items,"targets":len(data.test_interactions),"split":"test",
         "recall":list(result.channels),"fusion":cfg.fusion.mode,"ranker":cfg.ranking.model,
         "fused_at_100":fused,"metrics":result.metrics,
         "diagnostics":{k:v for k,v in result.diagnostics.items() if k!="target_details"},
         "timings":result.timings,"wall_s":round(perf_counter()-started,3),
         "checkpoint_policy":"existing_only","stage_handoff_checks_passed":True}
(out/"result.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False),flush=True)
