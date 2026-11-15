"""One frozen-weight, globally chronological, matched-budget code-research study."""
from pathlib import Path
from datetime import datetime, timezone
import gc,gzip,hashlib,json,shutil,time
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from recsys.models.two_tower import TwoTowerModel
from recsys.platform import autoresearch as research
from recsys.experiments.rec_agent import run_research
from recsys.platform.budget import BudgetPolicy,TaskBudget
from recsys.platform.shared_experiments import SharedExperimentStore,reuse_reasons

ROOT=Path(__file__).resolve().parent
OLD=Path('experiments/2026-11-13-kuairec-small').resolve()
START=time.perf_counter()
KEYS=['Recall@20','Recall@100','NDCG@20']
save=research.write_json
BASE_EVALUATE=research.evaluate
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc(t):return datetime.fromtimestamp(float(t),timezone.utc).isoformat()
def dump_gz(path,value):
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(value,f)

assert not (ROOT/'result.json').exists(),'Study is complete; do not rerun or tune against test'
torch.set_num_threads(4)
train=dict(np.load(OLD/'train.npz'));old_val=dict(np.load(OLD/'validation.npz'))
T0=float(train['timestamps'].max());T1=float(old_val['timestamps'].max())
protocol={
 'dataset':'KuaiRec 2.0 small_matrix','training':'frozen existing checkpoint; no optimizer or fitting',
 'T0':T0,'T1':T1,'T0_utc':utc(T0),'T1_utc':utc(T1),
 'validation':'first exposure in (T0,T1]','test':'first exposure in (T1,last_valid_timestamp]',
 'boundary_reason':'T0 after every old training event; T1 after all old checkpoint-selection validation events',
 'availability':'global first valid exposure <= window start, a conservative proxy for publish time',
 'labels':'first exposure watch_ratio>=1; discard any user-item pair with missing timestamp and ambiguous first label',
 'seen':'all valid first exposures <= window start, including negatives; unknown-time pairs excluded from candidates',
 'main_targets':'positive first exposures to items already in the start-of-window catalog',
 'secondary_targets':'all window positives, including newly arriving items, counted as unavailable misses',
 'users':'all 1411 mapped users for inference; macro metrics on users with positives, same cohort in all arms',
 'warm':'at least one positive in original training snapshot; original train-only models and counts',
 'cold':'no original train positive; exclude random TwoTower cold embeddings from recall',
 'cold_order':'decreasing distinct-user first-exposure count before window start; ties item index ascending',
 'cold_merge':'same in every arm: at rank r reserve floor(r*cold_pool_size/total_pool_size) cold positions; fill empty lane from other lane',
 'warm_pool':'union of top100 warm-only TwoTower, ItemCF and train Popularity; fixed pool for all arms',
 'fixed':'round-robin warm channels + shared cold merge',
 'objective':'validation NDCG@20; strict best-so-far, no test-driven selection',
 'agent_budget':{'code_experiments':3,'llm_requests':3,'max_seconds':240,'max_tools':8,'max_history_reads':2},
 'memory_accounting':'memory arm spends one of its three code trials on a real prior and two on continuation; no-memory uses three consecutive trials',
 'shared_baseline':'computed once and reused in all phases, not charged as a newly generated algorithm',
 'test_status':'exploratory re-analysis: dataset test results were inspected in previous conversations; not a new blind benchmark',
 'weights_sha256':sha(OLD/'two_tower.pt'),
 'original_splits_sha256':{s:sha(OLD/f'{s}.npz') for s in ['train','validation','test']},
 'study_rounds_fixed_before_feedback':True,
}
save(ROOT/'protocol.json',protocol)

df=pd.read_csv('data/kuairec_small/raw/small_matrix.csv',usecols=['user_id','video_id','timestamp','watch_ratio'])
uids=np.sort(df.user_id.unique());iids=np.sort(df.video_id.unique());nu,ni=len(uids),len(iids)
assert (nu,ni)==(1411,3327)
u=np.searchsorted(uids,df.user_id);i=np.searchsorted(iids,df.video_id)
t=df.timestamp.to_numpy();keys=u*ni+i;valid=np.isfinite(t)
first=np.full(nu*ni,np.inf);np.minimum.at(first,keys[valid],t[valid]);same=t==first[keys]
pos=np.zeros(nu*ni,dtype=bool);neg=pos.copy()
pos[keys[same & (df.watch_ratio.to_numpy()>=1)]]=True
neg[keys[same & (df.watch_ratio.to_numpy()<1)]]=True
ambiguous=pos&neg;pos&=~neg
missing=np.unique(keys[~valid]);unknown=np.zeros(nu*ni,dtype=bool);unknown[missing]=True
first[missing]=np.inf;pos[missing]=False
first=first.reshape(nu,ni);positive=pos.reshape(nu,ni);unknown=unknown.reshape(nu,ni)
del df,u,i,t,keys,valid,same,neg;gc.collect()
launch=first.min(axis=0);counts=np.bincount(train['items'],minlength=ni);cold=counts==0
assert np.all(first[train['users'],train['items']]<=T0)
inv=np.zeros(ni,dtype=np.float32);np.divide(1,np.sqrt(counts),out=inv,where=counts>0)
z=csr_matrix((inv[train['items']],(train['users'],train['items'])),shape=(nu,ni))
itemcf_scores=((z@z.T)@z).toarray();del z
model=TwoTowerModel(nu,ni,embedding_dim=32,hidden_dim=64,output_dim=32,temperature=.1,tower_type='embedding')
ck=torch.load(OLD/'two_tower.pt',map_location='cpu',weights_only=False)
model.load_state_dict(ck['model_state_dict']);model.eval()
with torch.no_grad():tower_scores=model.encode_users(torch.arange(nu)).numpy()@model.encode_items(torch.arange(ni)).numpy().T
del model,ck;gc.collect()
NAMES=['two_tower','itemcf','popularity']

def make_case(split,cutoff,end):
    available=launch<=cutoff
    eligible=(first>cutoff)&available[None,:]&~unknown
    all_targets=positive&(first>cutoff)&(first<=end)&np.isfinite(first)
    targets=all_targets&available[None,:]
    history_exposures=(first<=cutoff).sum(axis=0)
    groups=[];cold_orders={};full_truth={};stats={
        'split':split,'cutoff':cutoff,'cutoff_utc':utc(cutoff),'catalog_items':int(available.sum()),
        'cold_catalog_items':int((available&cold).sum()),'positive_count':int(targets.sum()),
        'cold_positive_count':int((targets&cold).sum()),'positive_users':int(targets.any(axis=1).sum()),
        'all_window_positive_count':int(all_targets.sum()),
        'excluded_new_arrival_positives':int((all_targets&~available).sum()),
        'unknown_time_pairs_excluded_from_candidates':int((unknown&available).sum()),
        'eligible_candidate_quantiles':np.quantile(eligible[targets.any(axis=1)].sum(axis=1),[0,.5,1]).tolist(),
        'ambiguous_first_labels_excluded':int(ambiguous.sum()),
    }
    warm_slots=[];cold_slots=[]
    for user in range(nu):
        warm_ids=np.flatnonzero(eligible[user]&~cold)
        lists={}
        for name,scores in [('two_tower',tower_scores[user]),('itemcf',itemcf_scores[user]),('popularity',counts)]:
            ids=warm_ids[scores[warm_ids]>0] if name=='itemcf' else warm_ids
            lists[name]=ids[np.lexsort((ids,-scores[ids]))[:100]].tolist()
        order=[];rows={}
        for rank in range(100):
            for name in NAMES:
                if rank>=len(lists[name]):continue
                item=lists[name][rank]
                if item not in rows:
                    order.append(item)
                    rows[item]={'item':item,'cold':False,'train_count':int(counts[item]),'ranks':{},'scores':{}}
                rows[item]['ranks'][name]=rank+1
                rows[item]['scores'][name]=float(tower_scores[user,item] if name=='two_tower' else itemcf_scores[user,item] if name=='itemcf' else counts[item])
        cold_ids=np.flatnonzero(eligible[user]&cold)
        cold_order=cold_ids[np.lexsort((cold_ids,-history_exposures[cold_ids]))].tolist()
        cold_orders[str(user)]=cold_order
        groups.append({'id':str(user),'candidates':[rows[j] for j in order],
                       'positives':np.flatnonzero(targets[user]).tolist()})
        full_truth[str(user)]=np.flatnonzero(all_targets[user]).tolist()
        warm_slots.append(len(rows));cold_slots.append(len(cold_order))
    stats['cold_positive_fraction']=stats['cold_positive_count']/stats['positive_count']
    stats['warm_pool_quantiles']=np.quantile(warm_slots,[0,.5,1]).tolist()
    stats['cold_pool_quantiles']=np.quantile(cold_slots,[0,.5,1]).tolist()
    case={'business':'recommendation','dataset':'kuairec_global_time_frozen','split':split,
          'objective':'NDCG@20','ks':[20,100],'models':NAMES,'groups':groups,
          'cold_items':np.flatnonzero(cold).tolist(),'cold_orders':cold_orders,
          'context':{'protocol':protocol,'window_statistics':stats,
            'algorithm_scope':'Modify warm fusion/ordering in Python; common cold lane and merge are fixed in evaluator for all arms. Do not train or access test.',
            'baseline':'Stable round robin warm TwoTower/ItemCF/Popularity; then evaluator inserts shared cold lane'},
          'source_files':['src/recsys/pipeline/fusion.py']}
    return case,full_truth,stats

def merge_cold(warm_rankings,case):
    result={}
    for group in case['groups']:
        key=str(group['id']);warm=warm_rankings[key];cold_lane=case['cold_orders'][key]
        eligible_warm={row['item'] for row in group['candidates']}
        if len(set(warm))!=len(warm) or not set(warm)<=eligible_warm:
            raise ValueError('Generated algorithm changed the fixed warm candidate identity')
        if len(warm)<min(100,len(eligible_warm)):raise ValueError('Generated ranking is too short')
        # Denominator uses the fixed pool, not the algorithm's returned length.
        nw,nc=len(eligible_warm),len(cold_lane);total=nw+nc
        row=[];wi=ci=0
        for rank in range(1,min(100,total)+1):
            cold_due=(rank*nc)//total if total else 0
            if ci<len(cold_lane) and (ci<cold_due or wi>=len(warm)):
                row.append(cold_lane[ci]);ci+=1
            else:
                row.append(warm[wi]);wi+=1
        result[key]=row
    return result

def full_case(case,truth=None):
    return {**case,'groups':[{**g,'candidates':g['candidates']+[
        {'item':i,'cold':True} for i in case['cold_orders'][str(g['id'])]],
        'positives':truth[str(g['id'])] if truth is not None else g['positives']} for g in case['groups']]}

def validation_feedback(case,warm_rankings):
    """Same frozen cold implementation for the baseline and every agent proposal."""
    if set(warm_rankings)!={str(g['id']) for g in case['groups']}:raise ValueError('Group IDs changed')
    return BASE_EVALUATE(full_case(case),merge_cold(warm_rankings,case))

# Prepare validation only; test outcomes are built after code selection is locked.
val_case,unused,val_stats=make_case('validation',T0,T1)
save(ROOT/'validation_case.json',val_case)
save(ROOT/'validation_statistics.json',val_stats)
base_warm={str(g['id']):[r['item'] for r in g['candidates']] for g in val_case['groups']}
baseline_feedback=validation_feedback(val_case,base_warm)
save(ROOT/'shared_baseline.json',baseline_feedback)
print('VALIDATION_PROTOCOL',json.dumps(val_stats),flush=True)
print('BASELINE',json.dumps(baseline_feedback['metrics']),flush=True)
research.evaluate=validation_feedback
policy=BudgetPolicy(max_llm_requests=3,max_tool_calls=8,max_experiments=3,max_history_lookups=2,max_artifact_reads=0,max_seconds=240)
TASK='''在冻结模型和固定候选池上研究暖物品融合算法，最大化验证 NDCG@20，结合 Recall@20/100、分组指标和反馈分析。
你可以编写新的融合公式、分数校准、通道一致性和用户自适应算法，不限于配置或权重选择。
输入候选已剔除所有未训练冷物品；统一冷启动队列由评估器在你排序后插入，各组完全相同，不能修改。
第一轮先分析基线；后续必须指出上一轮哪条真实反馈改变了判断，再提交改动。
有历史时必须结合历史源码、结果和失败方向继续研究，不要重复已经执行的代码。
完整输出至少前100条暖候选（不足100则全部），保留item身份，不读取评估标签或文件，不训练。
最终按已观察到的验证 NDCG 最优保留算法；新算法不如基线就保留基线。'''
results={};selected={}
for arm in ['no_memory','memory']:
    budget=TaskBudget('fair-'+arm,policy)
    t=time.perf_counter()
    store=SharedExperimentStore(ROOT/arm/'memory')
    phases=[]
    if arm=='memory':
        prior_store=SharedExperimentStore(ROOT/arm/'prior_source')
        prior=run_research(TASK,case_path=ROOT/'validation_case.json',work_dir=ROOT/arm/'prior',
            scope=budget,store=prior_store,rounds=1,baseline_feedback=baseline_feedback)
        prior_attempt=prior['attempts'][1]
        rec=prior_store.get(prior_attempt['record_id'])
        store.append(rec)
        phases.append(prior)
    online=run_research(TASK,case_path=ROOT/'validation_case.json',work_dir=ROOT/arm/'online',
        scope=budget,store=store,rounds=3 if arm=='no_memory' else 2,baseline_feedback=baseline_feedback)
    phases.append(online)
    candidates=[a for phase in phases for a in phase['attempts'] if a['status']!='crash']
    best=max(candidates,key=lambda a:a['value'])
    chosen=ROOT/arm/'selected_algorithm.py';shutil.copyfile(best['algorithm'],chosen)
    selected[arm]={'algorithm':str(chosen),'sha256':sha(chosen),'validation_metrics':best['feedback']['metrics'],
                   'source_record_id':best['record_id']}
    results[arm]={'phases':phases,'budget':budget.snapshot(),'wall_s':time.perf_counter()-t}
    save(ROOT/arm/'arm.json',results[arm])
    assert budget.counters['experiment']==3 and budget.counters['llm']==3
    print('ARM_LOCKED',arm,json.dumps(selected[arm]),flush=True)
fixed=ROOT/'fixed_algorithm.py';fixed.write_text(research.IDENTITY,encoding='utf-8')
selected['fixed']={'algorithm':str(fixed),'sha256':sha(fixed),'validation_metrics':baseline_feedback['metrics']}
save(ROOT/'locked_selection.json',{'selected':selected,'locked_at_utc':datetime.now(timezone.utc).isoformat(),
                                 'test_outcomes_not_evaluated':True})

test_case,test_all_truth,test_stats=make_case('test',T1,np.inf)
save(ROOT/'test_statistics.json',test_stats)
dump_gz(ROOT/'test_case.json.gz',test_case)
test_public={'groups':[{k:v for k,v in g.items() if k!='positives'} for g in test_case['groups']],
             'context':{k:v for k,v in test_case['context'].items() if k!='window_statistics'}}
test_metrics={};secondary={};detail={}
for arm in ['fixed','no_memory','memory']:
    algorithm=selected[arm]['algorithm'];assert sha(algorithm)==selected[arm]['sha256']
    warm=research.execute(algorithm,test_public,ROOT/'test_runs'/arm,timeout=60)
    final=merge_cold(warm,test_case)
    assessed=BASE_EVALUATE(full_case(test_case),final)
    all_targets=BASE_EVALUATE(full_case(test_case,test_all_truth),final)
    test_metrics[arm]={k:assessed['metrics'][k] for k in KEYS}
    secondary[arm]={k:all_targets['metrics'][k] for k in KEYS}
    detail[arm]=assessed
    dump_gz(ROOT/'test_runs'/f'{arm}_final.json.gz',final)
    save(ROOT/'test_runs'/arm/'assessment.json',assessed)
    print('TEST',arm,json.dumps(test_metrics[arm]),flush=True)
gains={arm:{k:{'absolute':m[k]-test_metrics['fixed'][k],
                   'relative_pct':100*(m[k]/test_metrics['fixed'][k]-1)} for k in KEYS}
       for arm,m in test_metrics.items() if arm!='fixed'}
checks={'weights_unchanged':sha(OLD/'two_tower.pt')==protocol['weights_sha256'],
        'old_splits_unchanged':all(sha(OLD/f'{s}.npz')==h for s,h in protocol['original_splits_sha256'].items()),
        'equal_total_code_trials':all(r['budget']['counters']['experiment']==3 for r in results.values()),
        'equal_total_llm_calls':all(r['budget']['counters']['llm']==3 for r in results.values()),
        'test_after_all_old_training_and_checkpoint_selection_events':T1>=T0,
        'cold_queue_and_position_rule_shared':True,'training_performed':False}
assert checks['weights_unchanged'] and checks['old_splits_unchanged']
result={'protocol':protocol,'validation_statistics':val_stats,'test_statistics':test_stats,'selected':selected,
        'test_metrics':test_metrics,'all_arrival_targets_secondary_metrics':secondary,'gains_vs_fixed':gains,
        'test_segments':{arm:r['segments'] for arm,r in detail.items()},'arms':results,'checks':checks,
        'wall_s':time.perf_counter()-START}
save(ROOT/'checks.json',checks);save(ROOT/'result.json',result)
print('FINISHED',json.dumps({'test':test_metrics,'gains':gains,'wall_s':result['wall_s']}),flush=True)
