"""Verify saved outputs and write a report; never reruns selection or inference."""
from pathlib import Path
import gzip, hashlib, json
import numpy as np
import pandas as pd
from recsys.evaluation.metrics import compute_ranking_metrics
from recsys.platform.shared_experiments import SharedExperimentStore, build_record, reuse_reasons
ROOT=Path(__file__).resolve().parent
OLD=Path('experiments/2026-11-13-kuairec-small').resolve()
r=json.loads((ROOT/'result.json').read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OLD/'two_tower.pt')==r['protocol']['weights_sha256']
for s,h in r['protocol']['split_hashes'].items():assert sha(OLD/f'{s}.npz')==h
# Preserve each evaluation's timing as well as its metrics in an immutable record snapshot.
# Initial run used a shared per-config file, so snapshot and revise only evidence paths.
for arm,info in r['arms'].items():
    store=SharedExperimentStore(ROOT/arm/'store')
    for rec in store.latest():
        if rec['business']!='recommendation':continue
        cid=rec['config']['config_id']
        out=r['prior_validation'] if rec['task_id']=='historical-validation-only' else info['trials'][cid]
        snapshot=ROOT/'record_artifacts'/rec['task_id']/f'{cid}.json'
        save(snapshot,out)
        if rec['artifacts'].get('metrics')!=str(snapshot):
            updated=build_record(task_id=rec['task_id'],business=rec['business'],status=rec['status'],
                config=rec['config'],data=rec['data'],model=rec['model'],result=rec['result'],budget=rec['budget'],
                artifacts={**rec['artifacts'],'metrics':str(snapshot)},evidence=rec['evidence'],record_id=rec['record_id'])
            store.append(updated)
    for rec in store.latest():
        if rec['business']=='recommendation':assert not reuse_reasons(rec,{'dataset':'kuairec_small','split':'validation'})
df=pd.read_csv('data/kuairec_small/raw/small_matrix.csv',usecols=['user_id','video_id','timestamp','watch_ratio'])
uids=np.sort(df.user_id.unique()); iids=np.sort(df.video_id.unique()); nu,ni=len(uids),len(iids)
u=np.searchsorted(uids,df.user_id); i=np.searchsorted(iids,df.video_id)
t=df.timestamp.to_numpy(); keys=u*ni+i; valid=np.isfinite(t)
first=np.full(nu*ni,np.inf); np.minimum.at(first,keys[valid],t[valid])
same=t==first[keys]
positive=np.zeros(nu*ni,dtype=bool); negative=positive.copy()
positive[keys[same & (df.watch_ratio.to_numpy()>=1)]]=True
negative[keys[same & (df.watch_ratio.to_numpy()<1)]]=True
positive &= ~negative
missing=np.unique(keys[~valid]); first[missing]=np.inf; positive[missing]=False
first=first.reshape(nu,ni); positive=positive.reshape(nu,ni)
cuts=[]
for frac in [.8,.9]:
    cuts.append(np.array([np.unique(row[np.isfinite(row)])[int(frac*len(np.unique(row[np.isfinite(row)])))] for row in first]))
sv=first<cuts[0][:,None]; st=first<cuts[1][:,None]
for split,mask in {'train':positive&sv,'validation':positive&~sv&st,'test':positive&~st}.items():
    uu,ii=np.nonzero(mask); tt=first[uu,ii]; order=np.lexsort((ii,tt,uu)); old=np.load(OLD/f'{split}.npz')
    assert np.array_equal(uu[order],old['users']) and np.array_equal(ii[order],old['items']) and np.array_equal(tt[order],old['timestamps'])
labels={}
test=np.load(OLD/'test.npz')
for u,i in zip(test['users'],test['items']):labels.setdefault(int(u),set()).add(int(i))
for cid in set(r['selected'].values()):
    with gzip.open(ROOT/f'test_{cid}.json.gz','rt',encoding='utf-8') as f:rows={int(u):x for u,x in json.load(f).items()}
    assert set(rows)==set(range(nu))
    assert all(len(x)==100 and len(set(x))==100 and not st[u,x].any() for u,x in rows.items())
    measured=compute_ranking_metrics(rows,labels,(20,100))
    for arm,chosen in r['selected'].items():
        if chosen==cid:
            assert all(abs(measured[k]-v)<1e-12 for k,v in r['test_metrics'][arm].items())
checks={'weights_unchanged':True,'splits_hash_unchanged':True,'raw_reconstructed_splits_identical':True,
 'all_test_recommendations_unique_and_unseen':True,'all_test_metrics_recomputed_from_saved_predictions':True,
 'business_memory_record_artifact_hashes_verified':True,'test_evaluation_users':len(labels)}
save(ROOT/'checks.json',checks)
lines=['# KuaiRec 融合与 Research Agent 实验结果','',
'实际结论：两组 Agent 都在验证集选中了三路等权 RRF；测试集效果均低于固定轮询。有记忆组没有额外提升，也没有节省新实验次数。','',
'## 数据与比较规则','',
'- KuaiRec 2.0 small_matrix：1,411 用户、3,327 视频、4,676,570 条播放记录。首次播放 watch_ratio >= 1 为正例；无有效时间的用户物品对剔除。',
'- 按每个用户首次曝光时间顺序切 80%/10%/10%，同一时间不拆分。训练正交互 1,177,823；验证正例 140,775；测试正例 153,264，测试有正例的用户 1,410。不是全局时间切分。',
'- 候选来自相同双塔、ItemCF、Popularity，每路最多 100，过滤截止时刻之前的全部已曝光物品，融合输出 100。不使用负采样评测。',
'- 训练权重沿用上一轮 KuaiRec 最佳验证 epoch=1 的双塔，未重新训练。训练正交互覆盖 2,999 视频，测试正例中 120,308（78.50%）对应训练无正交互物品。',
'- 固定策略：三路固定轮询。Agent：从事先固定的 6 个融合配置中选择，以验证 NDCG@20 最大为目标，各最多 2 次新实验。',
'- 有记忆组额外得到一条真实、同协议的 rrf_equal 验证实验记录（一次额外历史成本），不是与无记忆组完全相同的累计信息预算。两组均可保留固定基线。',
'- 使用现有 ResearchAgent 的真实 LLM 工具决策、图、统一预算、实验存储和项目融合函数；本实验脚本仅在进程内替换为 KuaiRec 验证适配器和工具菜单。未调用子 RecAgent 的独立 LLM，也未使用学习型精排；不能据此宣称整个生产平台完成了 KuaiRec 通用适配。',
'- 先落盘 locked_plans.json 再读本次测试指标，不回看测试结果调参。上一轮已测过单模型测试集，故这是既有数据上的一次探索性对照，不是从未看过测试集的全新盲测。','',
'## 测试集实际结果','',
'| 策略 | 最终融合 | Recall@20 | Recall@100 | NDCG@20 |',
'|---|---|---:|---:|---:|']
names={'fixed':'固定策略','no_memory':'无记忆 Agent','memory':'有记忆 Agent'}
for arm,m in r['test_metrics'].items():
    lines.append(f"| {names[arm]} | {r['selected'][arm]} | {m['Recall@20']*100:.4f}% | {m['Recall@100']*100:.4f}% | {m['NDCG@20']:.6f} |")
lines+=['',f"两组相对固定策略：Recall@20 {r['gains_vs_fixed']['memory']['Recall@20']['relative_pct']:.2f}%，Recall@100 {r['gains_vs_fixed']['memory']['Recall@100']['relative_pct']:.2f}%，NDCG@20 {r['gains_vs_fixed']['memory']['NDCG@20']['relative_pct']:.2f}%。Memory 相比无 Memory 三项均为 0 提升。",'',
'## 实际决策与成本','',
'| 组别 | 历史读取 | 第一次新实验 | 第二次新实验 | 新实验数 | LLM 请求数 | 组内耗时 |',
'|---|---|---|---|---:|---:|---:|']
for arm,a in r['arms'].items():
    trials=list(a['trials']); c=a['summary']['budget']['counters']
    lines.append(f"| {names[arm]} | {'rrf_equal 验证指标' if arm=='memory' else '无'} | {trials[0]} | {trials[1]} | {c['experiment']} | {c['llm']} | {a['wall_s']:.2f}s |")
lines+=['',f"历史实验生成耗时 {r['prior_validation']['elapsed_s']:.3f}s（仅融合与评测，不含共享候选推理）；全脚本耗时 {r['wall_s']:.2f}s，不含首次数据下载及上一轮训练。组内耗时含 LLM 等待和工具执行，此为单次运行，不能当作稳定速度结论。",'',
'- 无记忆：先假设双塔加权有效，实测 rrf_tt3 低于基线；随后改为等权 RRF，在验证集变好，最终选 rrf_equal。',
'- 有记忆：先真实读取 rrf_equal 历史，再试双塔加权和去掉 ItemCF，两个新方案均较差，保留历史最优。它没有跳过新实验，未体现效率优势。','',
'## 验证集为何看起来变好，测试却变差','',
'| 配置 | 验证 Recall@20 | 验证 Recall@100 | 验证 NDCG@20 |',
'|---|---:|---:|---:|']
known={'fixed_rr':r['fixed_validation'],'rrf_equal':r['prior_validation']}
for a in r['arms'].values():known.update(a['trials'])
for cid,o in known.items():
    m=o['metrics'];lines.append(f"| {cid} | {m['Recall@20']*100:.4f}% | {m['Recall@100']*100:.4f}% | {m['NDCG@20']:.6f} |")
lines+=['',
'等权 RRF 在验证集优于固定轮询，但未泛化到后续时间块。测试冷物品正例占比很高，且 ID 双塔没有冷物品语义特征，这些是已知限制；本次不能仅凭汇总数断言是哪一因素导致下降。',
'作为上轮已存在的参考，单双塔测试 Recall@20=6.1280%、Recall@100=27.0159%、NDCG@20=0.330743，均优于本次两种融合。因此本次既没有证明融合优于单模型，也没有证明 Research Agent 或记忆带来测试收益。',
'LLM 回答中“显著优于”“ItemCF 必要”等措辞是其自行解释，不是统计显著性检验或严格消融结论；本报告只采用工具返回数值。','',
'## 保存证据','',
'- run.py：实验配置、融合与真实 Agent 调用；finish.py：只校验已有结果并生成本报告。',
'- protocol.json、locked_plans.json：数据指纹、预算和冻结后的选择。',
'- result.json、checks.json：完整结果与定点检查，原始数据重建切分完全一致，工件哈希通过，保存的推荐无重复或已曝光物品。',
'- no_memory/ 和 memory/：llm_responses.json、arm.json、tasks/*/trace.jsonl、decisions.json、tool_results.json、summary.json。',
'- */store/：可校验的验证集实验记忆；record_artifacts/：每次实验独立结果快照，避免不同组同名配置覆盖运行耗时。',
'- test_*.json.gz：最终 100 条推荐列表，可直接复算指标。','',
'本次只有一次 LLM 运行/组，没有多 seed、显著性检验、继续调参或重训。三组效果对照完成；结论为这次 Agent 选择没有带来测试提升。','']
(ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False))
