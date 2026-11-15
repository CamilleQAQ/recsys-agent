"""Post-hoc diagnosis only: frozen weights/configs, no training or selection."""
from pathlib import Path
import gzip,json,time,gc
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from recsys.models.two_tower import TwoTowerModel
from recsys.pipeline.fusion import fuse_channels
from recsys.pipeline.config import FusionConfig
from recsys.pipeline.recall import Candidate
from recsys.evaluation.metrics import compute_ranking_metrics

ROOT=Path(__file__).resolve().parent
OLD=Path('experiments/2026-11-13-kuairec-small')
OUT=ROOT/'diagnosis';OUT.mkdir(exist_ok=True)
start=time.perf_counter();torch.set_num_threads(4)
df=pd.read_csv('data/kuairec_small/raw/small_matrix.csv',usecols=['user_id','video_id','timestamp','watch_ratio'])
uids=np.sort(df.user_id.unique());iids=np.sort(df.video_id.unique());nu,ni=len(uids),len(iids)
uu=np.searchsorted(uids,df.user_id);ii=np.searchsorted(iids,df.video_id)
ts=df.timestamp.to_numpy();keys=uu*ni+ii;valid=np.isfinite(ts)
first=np.full(nu*ni,np.inf);np.minimum.at(first,keys[valid],ts[valid])
first[np.unique(keys[~valid])]=np.inf;first=first.reshape(nu,ni)
cuts=[np.array([np.unique(row[np.isfinite(row)])[int(q*len(np.unique(row[np.isfinite(row)])))] for row in first]) for q in [.8,.9]]
seens={'validation':first<cuts[0][:,None],'test':first<cuts[1][:,None]}
del df,uu,ii,keys,valid,ts;gc.collect()
arrays={s:dict(np.load(OLD/f'{s}.npz')) for s in ['train','validation','test']}
train=arrays['train'];counts=np.bincount(train['items'],minlength=ni);cold=counts==0
exposure_cold=~seens['validation'].any(axis=0)
def truth(a):
    d={}
    for u,i in zip(a['users'],a['items']):d.setdefault(int(u),set()).add(int(i))
    return d
def top(scores,seen,onlypos=False):
    d={}
    for u,row in enumerate(scores):
        e=np.flatnonzero(~seen[u] & np.isfinite(row) & ((row>0) if onlypos else True))
        d[u]=e[np.lexsort((e,-row[e]))[:100]].tolist()
    return d
inv=np.zeros(ni,dtype=np.float32);np.divide(1,np.sqrt(counts),out=inv,where=counts>0)
z=csr_matrix((inv[train['items']],(train['users'],train['items'])),shape=(nu,ni))
item_scores=((z@z.T)@z).toarray();del z
model=TwoTowerModel(nu,ni,embedding_dim=32,hidden_dim=64,output_dim=32,temperature=.1,tower_type='embedding')
ck=torch.load(OLD/'two_tower.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict']);model.eval()
with torch.no_grad():scores=model.encode_users(torch.arange(nu)).numpy()@model.encode_items(torch.arange(ni)).numpy().T
def metrics(rows,g):return {k:v for k,v in compute_ranking_metrics(rows,g,(20,100)).items() if k in ['Recall@20','Recall@100','NDCG@20']}
def detail(rows,g):
    out={'metrics':metrics(rows,g)}
    for k in [20,100]:
        hits={'cold':0,'warm':0};slots={'cold':0,'warm':0};contrib={'cold':0.,'warm':0.}
        for u,targets in g.items():
            chosen=rows[u][:k];hit=set(chosen)&targets
            for kind,is_cold in [('cold',True),('warm',False)]:
                hits[kind]+=sum(bool(cold[i])==is_cold for i in hit)
                slots[kind]+=sum(bool(cold[i])==is_cold for i in chosen)
                contrib[kind]+=sum(bool(cold[i])==is_cold for i in hit)/len(targets)/len(g)
        out[f'top{k}']={'hits':hits,'recommendation_slots':slots,'macro_recall_contribution':contrib}
    for kind,is_cold in [('cold',True),('warm',False)]:
        part={u:{i for i in targets if bool(cold[i])==is_cold} for u,targets in g.items()}
        part={u:s for u,s in part.items() if s}
        out[kind+'_only_metrics']={'users':len(part),**metrics(rows,part)} if part else {}
    return out
results={}; all_lists={};checks={}
for split,seen in seens.items():
    a=arrays[split];g=truth(a)
    lists={'two_tower':top(scores,seen),'itemcf':top(item_scores,seen,True),'popularity':top(np.broadcast_to(counts,(nu,ni)),seen)}
    if split=='test':
        for name,rows in lists.items():
            with gzip.open(OLD/f'{name}.json.gz','rt',encoding='utf-8') as f:old={int(u):v for u,v in json.load(f).items()}
            assert rows==old,name
        checks['recomputed_test_all_three_channels_equal_saved']=True
    channels={n:{u:[Candidate(i,None,r+1) for r,i in enumerate(items)] for u,items in rows.items()} for n,rows in lists.items()}
    for mode in ['round_robin','rrf']:
        name='fixed_rr' if mode=='round_robin' else 'rrf_equal'
        rows=fuse_channels(channels,FusionConfig(mode=mode,limit=100,rrf_k=60))
        lists[name]={u:[x.item for x in xs] for u,xs in rows.items()}
        if split=='test':
            with gzip.open(ROOT/f'test_{name}.json.gz','rt',encoding='utf-8') as f:old={int(u):v for u,v in json.load(f).items()}
            assert lists[name]==old
        # Independent rank-sum implementation and independent round-robin.
        reference={}
        for u in range(nu):
            source=[lists[n][u] for n in ['two_tower','itemcf','popularity']]
            if mode=='rrf':
                summed={};first_rank={}
                for row in source:
                    for rank,i in enumerate(row,1):
                        summed[i]=summed.get(i,0)+1/(60+rank);first_rank.setdefault(i,rank)
                reference[u]=sorted(summed,key=lambda i:(-summed[i],first_rank[i],i))[:100]
            else:
                order=[row[r] for r in range(100) for row in source if len(row)>r]
                reference[u]=list(dict.fromkeys(order))[:100]
        assert lists[name]==reference
    checks[split+'_fusion_matches_independent_reference']=True
    positives_cold=int(cold[a['items']].sum())
    d={'positive_count':len(a['items']),'positive_users':len(g),'positive_cold_count':positives_cold,'positive_cold_fraction':positives_cold/len(a['items']),
       'exposure_cold_positive_count':int(exposure_cold[a['items']].sum()),
       'candidate_count_quantiles':np.quantile((~seen).sum(axis=1),[0,.5,1]).tolist(),
       'warm_candidate_count_quantiles':np.quantile(((~seen)&~cold).sum(axis=1),[0,.5,1]).tolist(),
       'cold_candidate_count_quantiles':np.quantile(((~seen)&cold).sum(axis=1),[0,.5,1]).tolist(),
       'positive_time_quantiles':np.quantile(a['timestamps'],[0,.25,.5,.75,1]).tolist(),
       'models':{n:detail(rows,g) for n,rows in lists.items()}}
    agreement={kind:{'union':0,'two_or_more_channels':0,'three_channels':0,'target_union_hits':0} for kind in ['cold','warm']}
    for u,targets in g.items():
        source=[set(lists[n][u]) for n in ['two_tower','itemcf','popularity']]
        for i in set.union(*source):
            entry=agreement['cold' if cold[i] else 'warm'];support=sum(i in s for s in source)
            entry['union']+=1;entry['two_or_more_channels']+=int(support>=2);entry['three_channels']+=int(support==3)
            entry['target_union_hits']+=int(i in targets)
    d['channel_agreement']=agreement
    d['top20_rrf_vs_fixed_hit_difference']={k:d['models']['rrf_equal']['top20']['hits'][k]-d['models']['fixed_rr']['top20']['hits'][k] for k in ['warm','cold']}
    results[split]=d;all_lists[split]=lists
    with gzip.open(OUT/f'{split}_channels.json.gz','wt',encoding='utf-8') as f:json.dump(lists,f)
# Are train-positive-cold embeddings untouched initialization? Recreate the original seed.
torch.manual_seed(42)
initial=TwoTowerModel(nu,ni,embedding_dim=32,hidden_dim=64,output_dim=32,temperature=.1,tower_type='embedding')
initial_state=initial.state_dict()
cold_embeddings=[]
for name,trained in ck['model_state_dict'].items():
    if 'item_tower' in name and trained.ndim==2 and trained.shape[0]==ni:
        unchanged=torch.equal(trained[cold],initial_state[name][cold])
        cold_embeddings.append({'parameter':name,'cold_rows_identical_to_seed42_initialization':unchanged,
            'warm_changed_rows':int((trained[~cold]!=initial_state[name][~cold]).any(dim=1).sum())})
checks['cold_embeddings']=cold_embeddings
train_end=np.full(nu,-np.inf);np.maximum.at(train_end,train['users'],train['timestamps'])
validation_start=np.full(nu,np.inf);np.minimum.at(validation_start,arrays['validation']['users'],arrays['validation']['timestamps'])
test_start=np.full(nu,np.inf);np.minimum.at(test_start,arrays['test']['users'],arrays['test']['timestamps'])
checks['per_user_train_before_validation']=bool((train_end<validation_start).all())
val_end=np.full(nu,-np.inf);np.maximum.at(val_end,arrays['validation']['users'],arrays['validation']['timestamps'])
checks['per_user_validation_before_test']=bool((val_end<test_start).all())
output={'checks':checks,'cold_items':int(cold.sum()),'no_train_exposure_items':int(exposure_cold.sum()),
    'splits':results,'global_time_ranges_overlap':bool(train['timestamps'].max()>arrays['validation']['timestamps'].min()),
    'diagnostic_inference_only_wall_s':time.perf_counter()-start}
(OUT/'result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(output,ensure_ascii=False),flush=True)
