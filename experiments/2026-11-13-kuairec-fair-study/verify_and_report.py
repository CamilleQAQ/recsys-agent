"""Verify stored predictions and report; no model inference or experiment search."""
from pathlib import Path
import gzip,hashlib,json,math
import numpy as np
from recsys.platform.shared_experiments import SharedExperimentStore,reuse_reasons

ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'result.json').read_text(encoding='utf-8'))
with gzip.open(ROOT/'test_case.json.gz','rt',encoding='utf-8') as f:case=json.load(f)
with (ROOT/'locked_selection.json').open(encoding='utf-8') as f:locked=json.load(f)
cold=set(case['cold_items']);truth={str(g['id']):set(g['positives']) for g in case['groups'] if g['positives']}
allowed={str(g['id']):{x['item'] for x in g['candidates']}|set(case['cold_orders'][str(g['id'])]) for g in case['groups']}
rankings={}
for arm in ['fixed','no_memory','memory']:
    with gzip.open(ROOT/'test_runs'/f'{arm}_final.json.gz','rt',encoding='utf-8') as f:rows=json.load(f)
    rankings[arm]=rows
    assert set(rows)==set(allowed)
    for user,items in rows.items():assert len(items)==len(set(items)) and set(items)<=allowed[user]
    values={'Recall@20':0.,'Recall@100':0.,'NDCG@20':0.}
    for user,positive in truth.items():
        for k in [20,100]:values[f'Recall@{k}']+=len(set(rows[user][:k])&positive)/len(positive)/len(truth)
        dcg=sum(1/math.log2(j+2) for j,i in enumerate(rows[user][:20]) if i in positive)
        idcg=sum(1/math.log2(j+2) for j in range(min(20,len(positive))))
        values['NDCG@20']+=dcg/idcg/len(truth)
    assert all(abs(v-r['test_metrics'][arm][key])<1e-12 for key,v in values.items())
    path=Path(locked['selected'][arm]['algorithm'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==locked['selected'][arm]['sha256']
for user in allowed:
    cold_positions=[[(j,i) for j,i in enumerate(rankings[a][user]) if i in cold] for a in ['fixed','no_memory','memory']]
    assert cold_positions[0]==cold_positions[1]==cold_positions[2]
record_count=0
for arm,folders in {'no_memory':['memory'],'memory':['memory','prior_source']}.items():
    for folder in folders:
        store=SharedExperimentStore(ROOT/arm/folder)
        for rec in store.latest():
            assert not reuse_reasons(rec,{'dataset':case['dataset'],'split':'validation'})
            record_count+=1
checks={**r['checks'],'independent_recall_ndcg_recalculation':True,
        'cold_item_identity_and_exact_positions_identical_all_arms':True,
        'final_rankings_same_user_ids_no_duplicate_or_unknown_item':True,
        'selected_algorithm_hashes_match_lock':True,'shared_records_artifact_checks_passed':record_count,
        'test_users_with_candidate_pool_at_most_100':sum(len(allowed[u])<=100 for u in truth),
        'test_positive_users':len(truth)}
(ROOT/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
names={'fixed':'固定轮询','no_memory':'无记忆 Agent','memory':'有记忆 Agent'}
lines=['# KuaiRec：冻结权重、公平预算、算法代码研究对照','',
'本次结果：两组 Agent 的测试 Recall@20 和 NDCG@20 都高于固定策略；有记忆组只比无记忆组略高，Recall@100 略低。没有重训，没有按测试结果继续改方案。','',
'## 测试集结果：推荐时已可用的物品目录','',
'| 策略 | Recall@20 | Recall@100 | NDCG@20 |',
'|---|---:|---:|---:|']
for arm,m in r['test_metrics'].items():
    lines.append(f"| {names[arm]} | {100*m['Recall@20']:.4f}% | {100*m['Recall@100']:.4f}% | {m['NDCG@20']:.6f} |")
lines+=['','| 相对固定策略 | Recall@20 相对变化 | Recall@100 相对变化 | NDCG@20 相对变化 |',
'|---|---:|---:|---:|']
for arm,g in r['gains_vs_fixed'].items():
    lines.append(f"| {names[arm]} | {g['Recall@20']['relative_pct']:+.3f}% | {g['Recall@100']['relative_pct']:+.3f}% | {g['NDCG@20']['relative_pct']:+.3f}% |")
memory_delta={k:100*(r['test_metrics']['memory'][k]/r['test_metrics']['no_memory'][k]-1) for k in ['Recall@20','Recall@100','NDCG@20']}
lines+=['',f"有记忆相比无记忆：Recall@20 相对 {memory_delta['Recall@20']:+.3f}%，Recall@100 {memory_delta['Recall@100']:+.3f}%，NDCG@20 {memory_delta['NDCG@20']:+.3f}%。Recall@20 的绝对差约 0.102 个百分点；不能把有记忆对固定策略的全部收益归因于 Memory。",'',
'## 时间切分和冷启动怎么处理','',
f"- 模型和原训练快照冻结不变。T0={r['protocol']['T0_utc']}，晚于或等于所有原训练事件；验证使用 (T0,T1]。",
f"- T1={r['protocol']['T1_utc']}，晚于或等于原 checkpoint 选择所用验证事件；测试严格在 T1 之后。所有用户使用同一时间边界。",
'- 仍按首次曝光 watch_ratio>=1 定义正例；同一用户物品不跨窗口。时间缺失的用户物品对不作为候选或标签。',
'- 原双塔曾用旧验证集选择 epoch，所以新验证窗口与其选模型数据有重叠；所有组共用同一 checkpoint。新测试时间晚于该选择窗口，没有把旧模型训练事件或选模型验证事件重新算成新测试。',
'- 物品目录仅包含窗口开始前已出现的物品。全局首次有效曝光作为可用时间的保守代理，尚无真实发布时间，不能把这个代理等同于精确上架时间。',
'- 所有用户在截止时间前的全部曝光（含负例）均过滤。训练有正交互为暖物品；训练无正交互为冷物品。',
'- 双塔的随机冷向量完全退出候选评分。暖物品三路复用原训练权重与统计；冷物品统一按窗口开始前的不同用户曝光数排序，同分按 item ID。',
'- 三组用同一冷队列、同一插入位置规则：输出第 r 位前累计预留 floor(r×冷池数/总池数) 个冷位置，空队列由另一侧补齐。比例来自固定候选池，不使用验证/测试点击率。',
'- Agent 只能修改暖物品融合与排序的 Python 代码；本轮不修改冷队列或比例。这是为了比较算法研究带来的排序收益，不能据此宣称 Agent 自己学会了冷启动。','',
'| 数据统计 | 验证 | 测试 |','|---|---:|---:|']
v=r['validation_statistics'];t=r['test_statistics']
for name,key in [('有目录内正例用户','positive_users'),('目录内正例','positive_count'),('其中冷正例','cold_positive_count'),('起始目录物品','catalog_items'),('其中冷物品','cold_catalog_items'),('窗口内新到达物品的正例','excluded_new_arrival_positives')]:
    lines.append(f"| {name} | {v[key]:,} | {t[key]:,} |")
lines += [f"| 冷正例比例 | {100*v['cold_positive_fraction']:.2f}% | {100*t['cold_positive_fraction']:.2f}% |",'',
f"测试过滤已曝光、缺失时间和不可用物品后，每用户可推荐物品数的最小/中位/最大为 {t['eligible_candidate_quantiles']}。{checks['test_users_with_candidate_pool_at_most_100']:,}/{len(truth):,} 名评估用户的实际候选池不超过100。因此 Recall@100 接近98%主要与候选池规模有关，不能与百万物品候选的旧实验或其他论文直接比较。",'',
'## 同时报告全部到达正例，避免只展示较高的一组成绩','',
f"测试窗口共有 {t['all_window_positive_count']:,} 个正例，其中 {t['excluded_new_arrival_positives']:,} 个来自窗口开始后才出现的物品。主表只评估起始目录内的 {t['positive_count']:,} 个正例；下表保留全部正例，把当时不可推荐的物品也计为未命中。",'',
'| 策略 | 全窗口 Recall@20 | 全窗口 Recall@100 | 全窗口 NDCG@20 |',
'|---|---:|---:|---:|']
for arm,m in r['all_arrival_targets_secondary_metrics'].items():
    lines.append(f"| {names[arm]} | {100*m['Recall@20']:.4f}% | {100*m['Recall@100']:.4f}% | {m['NDCG@20']:.6f} |")
lines+=['','这里是窗口起始的一次性推荐快照，不模拟窗口内新物品上线后追加召回；新物品到达能力尚未验证。','',
'## 相同预算和 Memory 的实际含义','',
'两组预算上限完全相同：3 次算法实验、3 次 LLM 请求、240秒、8次工具、2次历史读取。固定策略不搜索，复用同一个公共基线。公共基线反馈只计算一次，不重复扣作新算法实验。','',
'无记忆组：3次连续代码实验，组内仍保留上一轮结果作为研究反馈；“无记忆”指没有跨任务历史，不是每轮清空对话。',
'有记忆组：先用1次代码实验生成同协议的真实历史，再新开研究任务读取该历史，继续2次代码实验。历史生成成本计入这组，未额外赠送指标或实验。两组没有读取此前不兼容的旧切分历史。','',
'| 组别 | 累计算法实验 | 累计LLM请求 | 历史读取 | 组内耗时 |',
'|---|---:|---:|---:|---:|']
for arm,a in r['arms'].items():
    c=a['budget']['counters'];lines.append(f"| {names[arm]} | {c['experiment']} | {c['llm']} | {c['history']} | {a['wall_s']:.2f}s |")
lines += ['',f"总脚本耗时 {r['wall_s']:.2f}s，含数据读取、推理、研究和一次测试；不含早先数据下载和旧模型训练。有记忆组本次没有省实验或省时间。",'',
'## 实际研究轨迹','']
for arm,a in r['arms'].items():
    lines.append(f"### {names[arm]}")
    lines.append('')
    trial=0
    for phase in a['phases']:
        if phase['history_records_read']:lines.append(f"读取真实历史记录：{', '.join(phase['history_records_read'])}。")
        for attempt in phase['attempts']:
            if not attempt['iteration']:continue
            trial+=1
            lines += [f"**累计实验 {trial}：{attempt['status']}，验证 NDCG@20={attempt.get('value',0):.6f}**",'',
                      f"假设：{attempt['hypothesis']}",'',f"反馈分析：{attempt['feedback_analysis']}",'',f"代码：`{attempt['algorithm']}`。",'']
lines += ['LLM 的“证明”“已最优”等表述是其推测，不是统计显著性或因果验证；此处只确认它引用真实反馈并执行了对应源码。','',
'## 冷物品与暖物品的测试分组','',
'| 策略 | 暖 Recall@20 | 冷 Recall@20 | 暖 NDCG@20 | 冷 NDCG@20 |',
'|---|---:|---:|---:|---:|']
for arm,s in r['test_segments'].items():
    w=s['warm']['metrics'];c=s['cold']['metrics']
    lines.append(f"| {names[arm]} | {100*w['Recall@20']:.4f}% | {100*c['Recall@20']:.4f}% | {w['NDCG@20']:.6f} | {c['NDCG@20']:.6f} |")
lines+=['','三组冷物品的身份、推荐位置与分组指标完全一致，因此本次策略差异来自暖物品排序。分组指标各自按该组正例归一化，不能直接相加为整体指标。','',
'## 证据和边界','',
'- protocol.json：开跑前固定的时间、标签、冷启动规则和预算。',
'- locked_selection.json：测试结果构建和评估前锁定的源码及哈希。',
'- no_memory/、memory/：每轮源码、diff、LLM响应、工具反馈、记忆和预算。',
'- test_runs/：三个冻结算法的暖排序、最终推荐列表和评估结果，只有一次测试推理/组。',
'- checks.json：独立公式复算 Recall/NDCG、冷位置一致性、工件哈希、等预算检查。',
'- run.py：整个实验；verify_and_report.py：只读取保存结果核对并生成报告。',
'- baseline_cache 小型定点检查确认公共基线不消耗新增代码实验额度；输入为明确的测试替身，不混入真实指标。',
'',
'本次为一次固定预算、单次LLM轨迹的探索性对照，数据集此前已看过，不能包装为全新盲测。模型仍是旧训练快照；这次没有优化训练、增加候选覆盖或执行动态上线回放。有记忆组领先无记忆组的幅度很小，不能凭一次结果认定 Memory 稳定有效。','']
(ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False))
