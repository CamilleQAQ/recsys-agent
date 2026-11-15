from pathlib import Path
from time import perf_counter
import gzip,json
import numpy as np
from recsys.evaluation.metrics import compute_ranking_metrics
root=Path("experiments/2026-11-13-kuairec-small")
result=json.loads((root/"result.json").read_text(encoding="utf-8"))
train=np.load(root/"train.npz")
test=np.load(root/"test.npz")
counts=np.bincount(train["items"],minlength=3327)
group_truth={}
for name,condition in (("warm",counts>0),("cold",counts==0)):
    g={}
    for u,i in zip(test["users"],test["items"]):
        if condition[i]:g.setdefault(int(u),set()).add(int(i))
    group_truth[name]=g
groups={}
for model in result["metrics"]:
    with gzip.open(root/f"{model}.json.gz","rt",encoding="utf-8") as f:recs={int(u):r for u,r in json.load(f).items()}
    groups[model]={}
    for name,g in group_truth.items():
        m=compute_ranking_metrics(recs,g,(20,100))
        groups[model][name]={"users":len(g),"targets":sum(map(len,g.values())),
                          **{key:m[key] for key in ("Recall@20","Recall@100","NDCG@20")}}
result["cold_warm_metrics"]=groups
result["download"]=json.loads(Path("data/kuairec_small/raw/download.json").read_text(encoding="utf-8"))
(root/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
rows=[]
for model,m in result["metrics"].items():
    rows.append(f"| {model} | {m['Recall@20']:.6f} | {m['Recall@100']:.6f} | {m['NDCG@20']:.6f} |")
p=result["protocol"]
report=[
"# KuaiRec small_matrix 首轮推荐实验","",
"使用官方 KuaiRec 2.0 small_matrix，1,411 用户、3,327 个视频。原始 4,676,570 行。通过官方 ZIP 中单个成员的范围下载取得数据，CRC32 校验通过。","",
"排除 181,992 个缺少时间戳的用户—视频对。第一次播放的 watch_ratio >= 1.0 为正例。同一用户按首次曝光的不同时间戳分成 80% / 10% / 10%，同时间戳归同侧。固定阈值和协议之后才运行模型，未按测试指标调参。","",
f"训练正例 {p['split_counts']['train']:,}，验证 {p['split_counts']['validation']:,}，测试 {p['split_counts']['test']:,}。测试有正例的用户 {p['eval_users']:,}。过滤测试开始前所有已曝光物品，每用户可选物品数最小/中位/最大为 {p['test_unseen_candidate_quantiles']}。全目录计算，再过滤已曝光物品。","",
"| 方法 | Recall@20 | Recall@100 | NDCG@20 |","| --- | ---: | ---: | ---: |",*rows,"",
f"双塔为项目 TwoTowerModel 的 32 维 ID 表示，Adam lr=0.01，批大小 1024，固定 5 轮。批内已知训练正例被屏蔽为假负例。按验证 NDCG@20 选第 {result['best_epoch']} 轮，随后仅在测试集评价一次。Popularity 按训练次数，ItemCF 按训练物品余弦共现。","",
f"原始数据观测密集不代表时间外推无冷启动。测试 {p['cold_test_targets']:,}/{p['split_counts']['test']:,} 个正例涉及训练期零正反馈物品。选择整段时间留出形成了较强的物品分布变化，不能将本次结果与 KuaiSAR 不同协议的 Recall 直接视作算法改进。","",
"本次协议是显式配置的首轮时间对照，区别于项目已有的 KuaiRec big_matrix 默认最后两条正反馈留出。首次尝试在数据检查阶段发现缺失时间，第二次在检查阶段发现验证和测试正例用户集合相差一人；两次均在训练前停止，之后采用全用户打分、各自正例用户计分。","",
f"数据处理、三基线及双塔训练耗时 {result['wall_s']:.2f} 秒，下载另计。只保存一个验证集选中的新权重。","",
"[实际结果和冷暖分组](result.json) · [切分与训练协议](protocol.json) · [运行脚本](../2026-11-13-kuairec-small-scripts/run_experiment.py) · [官方数据说明](https://kuairec.com/)","",
]
(root/"report.md").write_text("\n".join(report),encoding="utf-8")
print(json.dumps({"metrics":result["metrics"],"best_epoch":result["best_epoch"],"cold_warm":groups},indent=2))
