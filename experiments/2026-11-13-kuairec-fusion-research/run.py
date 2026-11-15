"""One bounded KuaiRec fusion study. Production source and model weights stay unchanged.

The ResearchAgent graph/budget/store and production rank fusion are reused.
Only this process substitutes its KuaiSAR business adapter with KuaiRec validation.
No training; two new validation evaluations per agent, one prior for memory.
"""
from pathlib import Path
import gc, gzip, hashlib, json, time
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from recsys.models.two_tower import TwoTowerModel
from recsys.evaluation.metrics import compute_ranking_metrics
from recsys.pipeline.recall import Candidate
from recsys.pipeline.config import FusionConfig
from recsys.pipeline.fusion import fuse_channels
from recsys.platform import research_agent as ra, adapters
from recsys.platform.budget import BudgetPolicy
from recsys.platform.shared_experiments import SharedExperimentStore, build_record, reuse_reasons

ROOT=Path(__file__).resolve().parent
OLD=Path('experiments/2026-11-13-kuairec-small').resolve()
KEYS=['Recall@20','Recall@100','NDCG@20']
START=time.perf_counter()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def truth(split):
    z=np.load(OLD/f'{split}.npz'); out={}
    for u,i in zip(z['users'],z['items']): out.setdefault(int(u),set()).add(int(i))
    return out
def measure(recs,labels):
    m=compute_ranking_metrics(recs,labels,(20,100))
    return {k:m[k] for k in KEYS}

CONFIGS={
 'fixed_rr':{'mode':'round_robin','channels':['two_tower','itemcf','popularity'],'weights':{}},
 'rrf_equal':{'mode':'rrf','channels':['two_tower','itemcf','popularity'],'weights':{}},
 'rrf_tt3':{'mode':'rrf','channels':['two_tower','itemcf','popularity'],'weights':{'two_tower':3,'itemcf':1,'popularity':1}},
 'rrf_tt6':{'mode':'rrf','channels':['two_tower','itemcf','popularity'],'weights':{'two_tower':6,'itemcf':1,'popularity':1}},
 'rrf_tt_pop':{'mode':'rrf','channels':['two_tower','popularity'],'weights':{'two_tower':3,'popularity':1}},
 'rr_tt_itemcf':{'mode':'round_robin','channels':['two_tower','itemcf'],'weights':{}},
}
protocol={'dataset':'kuairec_small','source_protocol':json.loads((OLD/'protocol.json').read_text()),
 'configs':CONFIGS,'objective':'validation NDCG@20; tie: Recall@100 then config_id',
 'new_experiments_per_agent':2,'memory_prior':'one real validation evaluation: rrf_equal',
 'prior_cost_counted_separately':True,'test_previously_evaluated_for_single_models':True,
 'selection_test_access':'No test metrics/labels supplied to LLM; config menu frozen before this study.',
 'scope':'rank fusion only; existing ResearchAgent graph with experimental KuaiRec business adapter, no child RecAgent LLM or learned reranker',
 'weights_sha256':sha(OLD/'two_tower.pt'),
 'split_hashes':{s:sha(OLD/f'{s}.npz') for s in ['train','validation','test']}}
save(ROOT/'protocol.json',protocol)
assert not (ROOT/'result.json').exists(), 'Completed study exists: do not rerun'
torch.set_num_threads(4)
df=pd.read_csv('data/kuairec_small/raw/small_matrix.csv',usecols=['user_id','video_id','timestamp','watch_ratio'])
uids=np.sort(df.user_id.unique()); iids=np.sort(df.video_id.unique()); nu,ni=len(uids),len(iids)
us=np.searchsorted(uids,df.user_id); it=np.searchsorted(iids,df.video_id)
ts=df.timestamp.to_numpy(); keys=us*ni+it; valid=np.isfinite(ts)
first=np.full(nu*ni,np.inf); np.minimum.at(first,keys[valid],ts[valid])
first[np.unique(keys[~valid])]=np.inf; first=first.reshape(nu,ni)
val_times=np.array([np.unique(row[np.isfinite(row)])[int(.8*len(np.unique(row[np.isfinite(row)])))] for row in first])
seen=first<val_times[:,None]
del df,us,it,ts,keys,valid,first; gc.collect()
train=np.load(OLD/'train.npz'); train_u,train_i=train['users'],train['items']
assert np.all(seen[train_u,train_i])
val_truth=truth('validation')
assert all(not seen[u,list(items)].any() for u,items in val_truth.items())
def top(scores,positive_only=False):
    recs={}
    for u,row in enumerate(scores):
        eligible=np.flatnonzero(~seen[u] & np.isfinite(row) & ((row>0) if positive_only else True))
        recs[u]=eligible[np.lexsort((eligible,-row[eligible]))[:100]].tolist()
    return recs
counts=np.bincount(train_i,minlength=ni)
val_lists={'popularity':top(np.broadcast_to(counts,(nu,ni)))}
inv=np.zeros(ni,dtype=np.float32); np.divide(1,np.sqrt(counts),out=inv,where=counts>0)
z=csr_matrix((inv[train_i],(train_u,train_i)),shape=(nu,ni))
val_lists['itemcf']=top(((z@z.T)@z).toarray(),True)
del z; gc.collect()
model=TwoTowerModel(nu,ni,embedding_dim=32,hidden_dim=64,output_dim=32,temperature=.1,tower_type='embedding')
checkpoint=torch.load(OLD/'two_tower.pt',map_location='cpu',weights_only=False)
model.load_state_dict(checkpoint['model_state_dict']); model.eval()
with torch.no_grad():
    scores=model.encode_users(torch.arange(nu)).numpy()@model.encode_items(torch.arange(ni)).numpy().T
val_lists['two_tower']=top(scores)
del model,checkpoint,scores; gc.collect()
def candidates(lists):
    return {name:{int(u):[Candidate(int(i),None,k+1) for k,i in enumerate(items)] for u,items in rows.items()} for name,rows in lists.items()}
val_channels=candidates(val_lists)
def fused(config_id,channels):
    c=CONFIGS[config_id]
    rows=fuse_channels({k:channels[k] for k in c['channels']},FusionConfig(mode=c['mode'],limit=100,rrf_k=60),weights=c['weights'])
    return {u:[x.item for x in rows] for u,rows in rows.items()}
evaluations={}
def evaluate(config_id):
    t=time.perf_counter(); metrics=measure(fused(config_id,val_channels),val_truth)
    out={'config_id':config_id,'metrics':metrics,'elapsed_s':time.perf_counter()-t,'split':'validation'}
    evaluations[config_id]=out
    print('VALIDATION',json.dumps(out),flush=True)
    save(ROOT/'validation'/f'{config_id}.json',out)
    return out
baseline=evaluate('fixed_rr')
prior=evaluate('rrf_equal')
def record(store,task_id,cid,out):
    snapshot=ROOT/'record_artifacts'/task_id/f'{cid}.json'
    save(snapshot,out)
    return store.append(build_record(task_id=task_id,business='recommendation',status='success',
        config={'config_id':cid,**CONFIGS[cid],'weights_sha256':protocol['weights_sha256']},
        data={'dataset':'kuairec_small','split':'validation'},model={'models':CONFIGS[cid]['channels']},
        result={'metrics':{'config_id':cid,**out['metrics']},'summary':f'KuaiRec validation fusion {cid}'},
        artifacts={'metrics':str(snapshot),'weights':str(OLD/'two_tower.pt')},
        evidence={'validated':True,'execution_mode':'real_tools','decision_driver':'llm'}))

original_post=ra.post_chat
active={}
def logged_post(*args,**kwargs):
    response=original_post(*args,**kwargs)
    active['llm'].append(response)
    save(ROOT/active['arm']/'llm_responses.json',active['llm'])
    print('LLM',active['arm'],json.dumps(response.get('choices'),ensure_ascii=False)[:1700],flush=True)
    return response
ra.post_chat=logged_post
# Fix the dataset tag only inside this isolated process, including task records.
def scoped_build_record(**kw):
    kw['data']={**kw.get('data',{}),'dataset':'kuairec_small'}
    return build_record(**kw)
ra.build_record=scoped_build_record
ra.TOOLS={'run_recommendation_experiment':ra.ResearchTool('run_recommendation_experiment',
    'Evaluate one declared fusion config on all KuaiRec VALIDATION users using real cached model predictions. Costs one new experiment.',
    {'type':'object','properties':{'config_id':{'type':'string','enum':list(CONFIGS)}},'required':['config_id'],'additionalProperties':False},'business'),
    'read_experiment_result':ra.TOOLS['read_experiment_result']}
def messages(state):
    context={'dataset':'kuairec_small','split':'validation','configurations':CONFIGS,'baseline':baseline,
        'historical_record_id':active['prior_id'],'new_experiment_limit':2,'budget':state['budget'].snapshot()}
    return [{'role':'system','content':ra.SYSTEM_PROMPT},
        {'role':'user','content':state['task']}, {'role':'user','content':json.dumps(context,ensure_ascii=False)}]
ra._base_messages=messages
def business(business,task,task_id,scope,*,experiment_dir,intent,store):
    assert business=='recommendation'
    adapters._spend_dispatch(scope,business=business,min_internal_tools=1)
    cid=intent['config_id']; assert cid not in active['trials']
    scope.spend('tool',1)
    out=evaluate(cid); active['trials'][cid]=out
    rec=record(store,task_id,cid,out)
    return {'ok':True,'status':'success','business':business,'record_id':rec['record_id'],
        'metrics':{'config_id':cid,**out['metrics']}}
adapters.run_business=business
TASK='''研究 KuaiRec 多路召回融合。目标是在验证集最大化 NDCG@20（平手选 Recall@100）。
你必须先解释假设，选一个融合配置调用工具，看反馈后决定第二个不同配置，最多两个新实验，然后停止。
不要重跑已知 fixed_rr 或历史配置；有历史记录时先调用 read_experiment_result，基于指标形成下一步。
所有方案至少融合两个模型，不得改模型权重、训练或接触测试集。只用给定菜单，不要虚构指标。
第一轮反馈后在第二轮选择中体现反馈；允许保留固定基线。最终给出 selected_config_id 和理由。
系统用你实际观察过的验证 NDCG 最优方案作为 incumbent，若新方案更差则保留此前最好方案。'''
arms={}; selected={'fixed':'fixed_rr'}
for arm in ['no_memory','memory']:
    store=SharedExperimentStore(ROOT/arm/'store')
    prior_id=None
    if arm=='memory':
        rec=record(store,'historical-validation-only','rrf_equal',prior)
        assert not reuse_reasons(rec,{'dataset':'kuairec_small','split':'validation'})
        prior_id=rec['record_id']
    active.update(arm=arm,prior_id=prior_id,trials={},llm=[])
    cfg=ra.ResearchConfig(experiment_root=ROOT/arm,store_dir=ROOT/arm/'store',history={},
        budget=BudgetPolicy(max_llm_requests=6,max_tool_calls=10,max_experiments=2,max_history_lookups=0,max_artifact_reads=2,max_seconds=240),max_steps=6)
    t=time.perf_counter()
    summary=ra.ResearchAgent(cfg,planner='llm',store=store).run(TASK,intent={'businesses':['recommendation'],'history_first':False,'dataset':'kuairec_small','split':'validation'})
    observed={'fixed_rr':baseline,**active['trials']}
    if arm=='memory':observed['rrf_equal']=prior
    cid=sorted(observed,key=lambda k:(-observed[k]['metrics']['NDCG@20'],-observed[k]['metrics']['Recall@100'],k))[0]
    selected[arm]=cid
    arms[arm]={'selected_config_id':cid,'selection':'best validation incumbent among actual observations',
        'trials':active['trials'],'history_prior_count':int(arm=='memory'),'wall_s':time.perf_counter()-t,'summary':summary}
    save(ROOT/arm/'arm.json',arms[arm])
    print('ARM_DONE',arm,cid,summary.get('status'),flush=True)
save(ROOT/'locked_plans.json',{'selected':selected,'arms':arms,'locked_before_test_evaluation':True})
# Only now load the held-out labels/predictions. Prior single-model tests are disclosed.
test_truth=truth('test'); test_lists={}
for name in ['two_tower','itemcf','popularity']:
    with gzip.open(OLD/f'{name}.json.gz','rt',encoding='utf-8') as f:test_lists[name]=json.load(f)
test_channels=candidates(test_lists)
test_metrics={}; tested={}
for arm,cid in selected.items():
    if cid not in tested:
        recs=fused(cid,test_channels)
        assert all(len(r)==len(set(r)) and len(r)<=100 for r in recs.values())
        tested[cid]=measure(recs,test_truth)
        with gzip.open(ROOT/f'test_{cid}.json.gz','wt',encoding='utf-8') as f:json.dump(recs,f)
    test_metrics[arm]=tested[cid]
gains={arm:{k:{'absolute':m[k]-test_metrics['fixed'][k],
 'relative_pct':100*(m[k]/test_metrics['fixed'][k]-1)} for k in KEYS} for arm,m in test_metrics.items() if arm!='fixed'}
result={'protocol':protocol,'selected':selected,'test_metrics':test_metrics,'gains_vs_fixed':gains,
 'eval_users':len(test_truth),'test_positive_count':sum(map(len,test_truth.values())),
 'arms':arms,'prior_validation':prior,'fixed_validation':baseline,'wall_s':time.perf_counter()-START}
save(ROOT/'result.json',result)
print('FINAL',json.dumps({'selected':selected,'test_metrics':test_metrics,'gains':gains},ensure_ascii=False),flush=True)
