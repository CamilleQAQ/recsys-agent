from pathlib import Path
import hashlib
import json
from recsys.experiments.rec_agent import AgentSettings,PlannedCall,run_agent

root=Path("experiments/2026-11-10-formal-e2e/training-check")
existing=Path("artifacts/kuasar_small/matrix_factorization.pt")
def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file,"sha256").hexdigest()
old=digest(existing)
s=run_agent(AgentSettings(users=16,max_tool_calls=3,max_train_calls=1,max_batches=4,max_epochs=1,max_seconds=180),
    task="验收 BPR-MF 训练、隔离保存、从新权重恢复并评估", force_local=True, experiment_dir=root,
    plan=[
        PlannedCall("train_model",{"model":"matrix_factorization","epochs":1,"max_batches":4,"validation_users":8},reason="一次有上限的真实梯度更新"),
        PlannedCall("run_pipeline",{"models":["matrix_factorization"],"ranker":"lightgbm_ranker","fusion":"round_robin",
                    "split":"validation","k_recall":50,"k_final":20,"artifact_policy":"existing_only"},reason="同一任务只加载新训练召回权重和既有排序权重"),
    ])
assert s["status"]=="verified" and s["budget"]["train_calls"]==1
train,pipeline=s["tool_results"]
checkpoint=Path(train["result"]["checkpoint"])
meta=json.loads(Path(pipeline["result"]["artifacts"]["run_meta"]).read_text(encoding="utf-8"))
loaded=meta["checkpoints"]["matrix_factorization"]
assert Path(loaded["path"]).resolve()==checkpoint.resolve()
assert loaded["sha256"]==digest(checkpoint) and digest(existing)==old
payload={"status":s["status"],"trainer":"BPR-MF","epochs":1,"max_batches":4,"eval_users":16,
         "losses":train["result"].get("losses"),"checkpoint":str(checkpoint),"loaded_sha256":loaded["sha256"],
         "default_checkpoint_unchanged":True,"validation_metrics":pipeline["result"]["metrics"],"budget":s["budget"],
         "tools":s["tools_called"]}
(root/"result.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False),flush=True)
