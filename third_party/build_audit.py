"""Read local source and Git metadata; never imports or executes third-party code."""
import ast
import csv
import json
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
ORDER = ['RecBole', 'TorchEasyRec', 'generative-recommenders', 'OpenOneRec', 'MMRec', 'DeepCTR-Torch', 'LightGBM', 'faiss', 'SASRec.pytorch', 'BERT4Rec', 'recommenders']
INFO = {
 'RecBole': ('MIT', '框架官方；各论文模型为框架复现', '协同过滤、图推荐、序列、CTR、知识图谱', 'run_recbole.py; recbole/quick_start/quick_start.py', 'recbole/trainer/trainer.py: fit/evaluate; 各模型 predict/full_sort_predict', '带类型表头的制表符 atomic 文件 .inter/.user/.item；知识模型另需 .kg/.link', 'torch>=1.10、numpy、scipy、pandas、ray<=2.6.3；README 的 torch>=1.7 与依赖文件不一致，以实际依赖为准。setup.py 与 requirements 的 colorlog/colorama 约束不同。宽松 NumPy/PyTorch 上界不能保证现代版本兼容。', '数据通常来自 RecDatasets，部分路径可自动下载；预训练 embedding 是可选输入。默认实验数据不代表所有数据已具备。'),
 'TorchEasyRec': ('Apache-2.0', 'Alibaba 官方框架；论文模型为框架实现', '双塔召回、CTR、多任务、HSTU、生成式、语义ID', 'tzrec/train_eval.py', 'tzrec/predict.py; tzrec/eval.py; tzrec/export.py', 'CSV/Parquet/ODPS/Kafka；pipeline protobuf 配置声明字段、标签、特征与模型', 'setup.py 声明 Python>=3.8，但当前 README/二进制 wheel 面向 3.10–3.12。runtime 要求 numpy>=2、transformers>=4.56；cpu/cu130 文件锁 torch==2.14.0，需后续核验发行渠道可用性。大量 OSS 直接 URL 和 Linux wheel，当前 Windows 不可直接假定兼容。', '样例可能依赖 OSS/ODPS；预训练 embedding/LLM 按配置外置。未访问云数据或执行安装。'),
 'generative-recommenders': ('Apache-2.0', 'Meta 官方研究实现', 'HSTU、SASRec基线、DLRMv3/HSTU、ULTRA算子、MoL检索', 'main.py; generative_recommenders/research/trainer/train.py; generative_recommenders/dlrm_v3/train/train_ranker.py', 'generative_recommenders/research/data/eval.py; generative_recommenders/dlrm_v3/inference/main.py', '预处理的用户交互序列/时间戳/评分；Gin 配置；DLRMv3 特征/jagged tensor 路径另行处理', 'Python>=3.10、torch>=2.6、fbgemm_gpu>=1.1、torchrec>=1.1；README 测试 Ubuntu22.04/CUDA12.4。CUDA/Triton/C++ 扩展和 GPU 架构要求需独立环境；不能据此保证 Windows 运行。', 'preprocess_public_data.py 涉及公开数据下载；fractal expansion 可生成巨量数据。CUTLASS 子模块未初始化，优化 CUDA 构建不完整；基础研究路径与优化路径分开判断。'),
 'OpenOneRec': ('Apache-2.0 (README声明；根LICENSE缺失)', 'Kuaishou-OneRec 官方项目；包含 Qwen3/VeRL 派生代码', 'Qwen3/OneRec生成式推荐、Residual K-Means tokenizer、SFT/蒸馏/GRPO流程', 'pretrain/recipes/train_qwen3.py; tokenizer/train_res_kmeans.py; verl_rl/recipe/onerec/run_grpo.sh', 'tokenizer/infer_res_kmeans.py; benchmarks/scripts/ray-vllm/evaluate.py; benchmarks/benchmark/tasks/v1_0/', 'Parquet chat/messages 或 segments、itemic tokens；tokenizer 输入 embedding，推断另需 pid；参考 data/README.md', '无根 requirements/setup/pyproject；依赖分散于 benchmarks、verl_rl、verl_distillation 与 pretrain README。Qwen3、Transformers、PyTorch FSDP、MPI/NCCL、Ray/vLLM 等；各模块版本需分别锁定，多卡 Linux 为主要路径。', '需要外部 Qwen3/OneRec 权重、Qwen3-Embedding-8B、tokenizer 及 RecIF 数据；未下载。Pro 使用的工业语料不能视为公开完整复现资源。根 README 的 coming soon 与已发布子模块文档不一致。根 LICENSE 缺失，子目录许可证不能自动覆盖全仓库，适配/再分发前应确认。'),
 'MMRec': ('GPL-3.0 (见原LICENSE)', 'MMRec 框架原仓库；论文模型包含作者实现及框架复现，不能统称各模型官方', '多模态图推荐、协同过滤、内容KNN', 'src/main.py; src/utils/quick_start.py', 'src/common/trainer.py; src/utils/topk_evaluator.py; 模型 full_sort_predict', '配置指定交互表 uid/iid/split 字段；视觉/文本特征及 ID 对齐；部分方法需用户邻接矩阵', 'requirements 固定 numpy1.21.5/pandas1.3.5/scipy1.7.3/torch1.11；python==3.7.11 是环境说明而非有效的常规 pip Python 安装方式。代码另有 torchvision/PIL/torch_geometric/torch_scatter/sklearn 等导入，根依赖清单不能视为完整。老版本与现代 Python/NumPy 不宜混装。', 'data/README.md 指向 Google Drive 的 Baby/Sports/Elec 特征数据及 MicroLens；未下载。多模态预提取特征不是随仓库附带的模型权重。GPL 适配与分发边界需保留。'),
 'DeepCTR-Torch': ('Apache-2.0', 'DeepCTR 作者维护的 PyTorch 库；论文模型为库内复现', 'CTR特征交叉、DIN/DIEN、多任务', 'examples/ 下样例调用模型 fit', 'deepctr_torch/models/basemodel.py: fit/predict/evaluate', '特征名到数组的字典；SparseFeat/DenseFeat/VarLenSparseFeat，标签数组及序列长度/负采样字段', 'Python>=3.7；setup.py 按 Python<3.8 用 torch>=1.13，否则 torch>=2.4；tqdm/scikit-learn。样例的数据处理还需 pandas 等，setup 不等于全部样例依赖。', '示例 CTR/MovieLens 数据按样例准备；无强制通用预训练模型。README 的 FNN 不能单独计为现有模型模块：未发现 models/fnn.py。'),
 'LightGBM': ('MIT', '官方仓库已迁移至 lightgbm-org/LightGBM；原URL重定向', 'GBDT/DART/GOSS/RF；LambdaRank、rank_xendcg排序目标', 'src/main.cpp; python-package/lightgbm/engine.py: train', 'python-package/lightgbm/sklearn.py: LGBMRanker.fit; basic.py: Booster.predict', '稠密/稀疏特征矩阵及 label；排序必须提供连续查询分组 group/query，非用户序列推荐器', '当前 python-package/pyproject.toml 要求 Python>=3.11、narwhals>=2.26、numpy>=1.21.3、scipy；scikit-build-core/CMake/C++，可选 OpenMP/GPU。根无 Python setup.py 并非缺代码。', '5 个构建子模块未初始化，源码编译前需单独核验并获取；examples 包含少量实验数据。树模型通常自行训练，无预训练权重必需项。'),
 'faiss': ('MIT', 'Meta/FAIR 官方向量检索库', 'Flat/IVF/PQ/HNSW/LSH、聚类；非端到端推荐模型', 'Index.train/add; tutorials/python/ 下索引示例', 'Index.search; contrib/evaluation.py', '连续 float32 的 N×d 向量和查询；部分二进制索引用 uint8，须由上游推荐模型生成向量', 'pyproject: Python>=3.10，运行 numpy>=1.25/packaging；构建 numpy>=2、scikit-build-core、SWIG>=4.2<5、CMake、BLAS/OpenMP。GPU 路径另需 CUDA，CPU 默认配置关闭 GPU。', '不附带推荐 embedding 或成品业务索引；benchmarks 的外部向量集不在本轮获取范围。THIRD_PARTY_NOTICES 需一起保留。'),
 'SASRec.pytorch': ('Apache-2.0', '第三方复现：Zan Huang/pmixer，基于 kang205/SASRec 的论文作者 TensorFlow 实现', 'SASRec；附Rust实现路径，非另一个独立模型', 'python/main.py (工作目录 python/)', 'python/model.py: SASRec.predict; python/utils.py: evaluate/evaluate_valid; main inference_only', 'python/data/<dataset>.txt；每行 user_id item_id，按用户交互顺序，整数 ID；leave-last 验证/测试和负采样', '无 requirements/environment/setup/pyproject；代码导入 torch/numpy，README 历史提及 PyTorch1.6，无完整锁定环境。附 rust/Cargo.toml 独立依赖未编译。', 'Git 已跟踪多份实验文本数据；推断 state_dict_path 需自行训练或另备。README 记录 LayerNorm 更新，不能把当前实现直接当作原论文逐位复现。'),
 'BERT4Rec': ('Apache-2.0', '论文作者 Fei Sun 原始实现；基于BERT架构代码', 'BERT4Rec 双向序列推荐', 'gen_data_fin.py; run.py; run_ml-1m.sh 等', 'run.py: model_fn_builder、EvalHooks、Estimator.train/evaluate；无现代统一 predict API', '用户物品有序文本转 masked TFRecord；附 vocab/history pickle 与 bert_train JSON 配置', '无 requirements/environment/setup/pyproject；README Python2.7+、TensorFlow1.12 GPU/CUDA；源码使用 tf.contrib/tf.layers/Estimator/旧 TFRecord API。不能与现代 TF2/TFRS 环境直接合并。six/numpy 等依赖需后续补锁定。', '已跟踪 beauty/steam/ml-1m 文本及 ml-20m.zip；无预训练 checkpoint。训练从任务数据构造，不应误认为下载通用BERT权重即能预测。'),
 'recommenders': ('Apache-2.0', 'TensorFlow 官方 TFRS 库', '双塔检索/排序任务、Cross/DCN层、BruteForce/Streaming/ScaNN检索接口', 'README 与 docs/examples/*.ipynb 使用 Keras Model.fit；无统一训练CLI', 'tensorflow_recommenders/models/base.py; tasks/retrieval.py; tasks/ranking.py; layers/factorized_top_k.py', 'tf.data.Dataset 特征字典；query/candidate 特征与候选集合；标签/任务由调用方建模', 'requirements: tensorflow>=2.9（macOS 用 tensorflow-macos）、absl-py、tf-keras；无上界。ScaNN/tensorflow-ranking/tfds 为可选或样例依赖，需核对 TF/Keras/ABI 组合。', '样例 tfds.load 会下载 MovieLens；未运行。ScaNN 是外部后端，不能计为仓库内完整 ANN 算法；README 样例双塔架构不代表一组现成已训练模型。'),
}

def git(p, *args):
 return subprocess.check_output(['git', '-C', str(p), *args], text=True, encoding='utf-8').strip()

def read(p):
 return p.read_text(encoding='utf-8', errors='replace')

def model_files(n, fs):
 if n=='RecBole': return [f for f in fs if re.match(r'recbole/model/[^/]+_recommender/[^/]+\.py$',f) and not f.endswith('__init__.py')]
 if n=='TorchEasyRec': return [f for f in fs if re.match(r'tzrec/models/[^/]+\.py$',f) and not f.endswith(('_test.py','__init__.py')) and Path(f).stem not in ['model','rank_model','match_model','multi_task_rank','sid_model','genrec_model']]
 if n=='MMRec': return [f for f in fs if re.match(r'src/models/[^/]+\.py$',f)]
 if n=='DeepCTR-Torch': return [f for f in fs if f.startswith('deepctr_torch/models/') and f.endswith('.py') and not f.endswith(('__init__.py','basemodel.py'))]
 select={
  'generative-recommenders':['generative_recommenders/research/modeling/sequential/hstu.py','generative_recommenders/research/modeling/sequential/sasrec.py','generative_recommenders/modules/hstu_transducer.py','generative_recommenders/modules/dlrm_hstu.py','generative_recommenders/ops/pytorch/pt_hstu_ultra.py','generative_recommenders/research/rails/similarities/mol/similarity_fn.py'],
  'OpenOneRec':['pretrain/onerec_llm/models/qwen3/modeling_qwen3.py','tokenizer/res_kmeans.py'],
  'LightGBM':['python-package/lightgbm/sklearn.py','python-package/lightgbm/engine.py','src/objective/rank_objective.hpp','src/boosting/gbdt.cpp'],
  'faiss':['faiss/IndexFlat.cpp','faiss/IndexIVF.cpp','faiss/IndexPQ.cpp','faiss/IndexHNSW.cpp','faiss/IndexLSH.cpp','faiss/Clustering.cpp'],
  'SASRec.pytorch':['python/model.py','rust/src/model.rs'],
  'BERT4Rec':['modeling.py','run.py'],
  'recommenders':['tensorflow_recommenders/tasks/retrieval.py','tensorflow_recommenders/tasks/ranking.py','tensorflow_recommenders/layers/dcn.py','tensorflow_recommenders/layers/factorized_top_k.py','tensorflow_recommenders/experimental/models/ranking.py'],
 }
 return [f for f in select[n] if f in fs]

rows=[]; mapping=['# 模型与代码映射','', '审计日期：2026-11-04（Asia/Shanghai）。依据当前 commit 的实际源码定义、方法和函数体静态检查；未导入、训练或验证数值正确性。框架实现不等于论文作者官方实现；组件不等于可直接训练的完整推荐模型。','']
report=['# 静态代码审计报告','', '本轮仅独立浅克隆 11 个指定仓库（depth=1，single-branch，保留 .git 和原目录结构）。未安装依赖、运行第三方代码/安装脚本、初始化子模块或主动下载数据/权重。LFS smudge 已禁用。Git 跟踪的实验数据随仓库原样取得，详见数据体积清单。未增补或修改第三方文件。','', '## 结果与边界','', '11/11 克隆成功；最终失败清单为空。最初沙箱 Git TLS 凭据错误，经授权重试成功。默认分支由克隆时 origin/HEAD 记录，commit 为完整 SHA。浅克隆不包含全部历史、其他分支或 tag；未初始化的子模块会影响编译完整性。许可证为源码初筛，非法律结论；未验证外部权重服务、数据链接、wheel URL 或全部传递依赖的可用性。','', 'LightGBM 原 URL 经 GitHub 官方重定向至 [lightgbm-org/LightGBM](https://github.com/lightgbm-org/LightGBM)，不是自行选择替代仓库。OpenOneRec 根 LICENSE 缺失；README 声明 Apache 2.0，子目录许可证分别保留，权重另有许可。无其他候选仓库被下载。','']
evidence={}
for n in ORDER:
 p=BASE/n; fs=git(p,'ls-files').splitlines(); lic,origin,coverage,train,predict,data,compat,external=INFO[n]
 selected=model_files(n,fs); symbols=[]; ev=[]
 for f in selected:
  t=read(p/f); defs=[]
  if f.endswith('.py'):
   try:
    tree=ast.parse(t)
    for node in tree.body:
     if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
      methods=[x.name for x in getattr(node,'body',[]) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]
      defs.append({'name':node.name,'line':node.lineno,'methods':methods})
   except SyntaxError:
    defs=[{'name':m.group(2),'line':t[:m.start()].count('\n')+1,'methods':[]} for m in re.finditer(r'^(class|def)\s+(\w+)',t,re.M)]
  else:
   defs=[{'name':line.strip()[:140],'line':i,'methods':[]} for i,line in enumerate(t.splitlines(),1) if re.search(r'^(class |struct |pub struct |pub fn )|class .*: public',line)]
  ev.append({'path':f,'definitions':defs})
  symbols.extend(x['name'] for x in defs if not x['name'].startswith('_'))
 mapping.extend([f'## {n}', '', f'- 来源：{origin}。', f'- 覆盖：{coverage}。', '- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：','', '| 文件 | 源码定义（行号） |', '|---|---|'])
 for entry in ev:
  ds='; '.join(f"{x['name']} L{x['line']}" for x in entry['definitions']) or 'C++/Rust 索引或模型实现文件，非 Python 类'
  mapping.append(f"| [{entry['path']}]({n}/{entry['path']}) | {ds.replace('|', '/')} |")
 mapping.append('')
 deps=[f for f in fs if re.search(r'(^|/)(requirements[^/]*\.(txt|in)|environment[^/]*\.ya?ml|setup\.(py|cfg)|pyproject[^/]*\.toml)$',f)]
 roots={f: f in fs for f in ['README.md','LICENSE','requirements.txt','environment.yml','setup.py','pyproject.toml']}
 licenses=[f for f in fs if Path(f).name.lower().startswith(('license','copying','notice')) or f=='THIRD_PARTY_NOTICES']
 stage=git(p,'ls-files','--stage').splitlines();subs=[s.split('\t',1)[1] for s in stage if s.startswith('160000 ')]
 weights=[f for f in fs if re.search(r'\.(pt|pth|ckpt|safetensors|h5|onnx|pb|bin)$|\.ckpt\.',f,re.I)]
 lfs=[f for f in fs if (p/f).is_file() and (p/f).stat().st_size<1024 and (p/f).read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1')]
 sizes=sorted([(f,(p/f).stat().st_size) for f in fs if (p/f).is_file()],key=lambda x:x[1],reverse=True)
 status=git(p,'status','--porcelain');assert not status,(n,status)
 canonical='https://github.com/lightgbm-org/LightGBM' if n=='LightGBM' else git(p,'remote','get-url','origin').removesuffix('.git')
 row=dict(repository=n,requested_url=git(p,'remote','get-url','origin').removesuffix('.git'),canonical_url=canonical,status='cloned',commit=git(p,'rev-parse','HEAD'),default_branch=git(p,'symbolic-ref','refs/remotes/origin/HEAD').removeprefix('refs/remotes/origin/'),license=lic,license_files='; '.join(licenses),provenance=origin,model_coverage=coverage,implementation_files='; '.join(selected),source_symbols='; '.join(dict.fromkeys(symbols)),clone_depth=1,is_shallow=git(p,'rev-parse','--is-shallow-repository'),submodules_not_initialized='; '.join(subs),working_tree='clean',audit_date='2026-11-04')
 rows.append(row)
 evidence[n]={'metadata':row,'root_file_presence':roots,'dependency_files':{f:read(p/f) for f in deps},'model_definitions':ev,'weight_filename_candidates':weights,'lfs_pointers':lfs,'largest_tracked_files':sizes[:12],'tracked_file_bytes':sum(s for _,s in sizes)}
 report.extend([f'## {n}', '', f'- 仓库：{canonical}；默认分支 `{row["default_branch"]}`；commit `{row["commit"]}`。', f'- 来源身份：{origin}。', f'- 许可证：{lic}；文件：'+ ('、'.join(f'`{f}`' for f in licenses) or '无独立许可证文件')+'。', '- 根文件检查：'+'；'.join(f'`{f}` '+('存在' if present else '缺失') for f,present in roots.items())+'。', '- 依赖/构建文件：'+('、'.join(f'`{f}`' for f in deps) or '未发现标准 Python 依赖或打包文件')+'。', f'- 实际模型覆盖：{coverage}；源码证据见 [model_mapping.md](model_mapping.md)。', f'- 训练入口：{train}。', f'- 预测/评估：{predict}。', f'- 数据格式：{data}。', f'- 依赖与兼容性：{compat}', f'- 外部资源/完整性：{external}', '- 预训练权重检查：'+('发现候选文件 '+', '.join(weights) if weights else '已跟踪文件中未发现常见权重扩展名（pt/pth/ckpt/safetensors/h5/onnx/pb/bin）；不能排除其他命名或归档形式')+'；LFS 指针 '+str(len(lfs))+' 个。', '- 未初始化子模块：'+(', '.join(subs) if subs else '无 Git 子模块')+'。', f'- 原始工作树：clean；已跟踪文件约 {sum(s for _,s in sizes)/1048576:.1f} MiB（不含 .git）。', '- 最大已跟踪文件：'+ '; '.join(f'`{f}` ({s/1048576:.2f} MiB)' for f,s in sizes[:3])+'。',''])

mapping.extend(['## 跨仓库映射与避免误计','', '| 模型/任务 | 主要源码位置 | 说明 |','|---|---|---|','| SASRec | RecBole/recbole/model/sequential_recommender/sasrec.py；SASRec.pytorch/python/model.py；Meta research/modeling/sequential/sasrec.py | 框架复现、第三方复现、研究基线；不重复克隆 |','| BERT4Rec | BERT4Rec/modeling.py + run.py；RecBole/recbole/model/sequential_recommender/bert4rec.py | 原作者 TF1 与框架 PyTorch 复现 |','| LightGCN/BPR | RecBole 与 MMRec 相应 model 文件 | 多仓库已有模型不新增下载 |','| DeepFM/DCN/DIN/多任务 | DeepCTR-Torch、TorchEasyRec、RecBole 对应实现 | 各库支持范围以逐文件列表为准 |','| HSTU | Meta research/modules 与 TorchEasyRec/tzrec/models | 架构与训练栈不同，不能互换 checkpoint |','| OneRec | OpenOneRec/pretrain；tokenizer；verl_rl/verl_distillation | 1.7B/8B/Pro 是外部权重版本，不是四份独立源码实现 |','| LambdaRank/rank_xendcg | LightGBM/src/objective/rank_objective.hpp | 排序目标；不把 GBDT 分类器都计为推荐算法 |','| ANN | faiss/Index*.cpp | 检索基础设施，需要上游 embedding |','| TFRS | tasks、layers、experimental/models 与 notebooks | 任务/层/教程组合，不宣称完整论文模型集 |','', 'OpenOneRec README 比较表中的 TIGER、LC-Rec 等不能直接计为该仓库实现；Qwen3 的通用分类/问答头也不是新增推荐模型。MMRec 的 MG 对应训练策略，不能当成另一个模型文件。模型表中出现外部库封装（例如 RecBole LightGBM/XGBoost）仅表示适配器存在，底层算法需相应外部实现。',''])
report.extend(['## 后续统一适配建议','', '按 PyTorch 基线、旧 TF1、现代 TF2、工业 CUDA/分布式、C++ 检索/树排序分别隔离环境，先选小型合成数据验证接口，再决定依赖和外部资源下载。统一接口需保留 user/item ID 映射、序列顺序、候选集合、负采样策略、query group 与评估协议；采样 HR/NDCG 与全量检索指标不可直接横比。','', 'OpenOneRec 的根许可证缺失、MMRec 的旧依赖/清单遗漏、Meta/LightGBM 的子模块，以及 TorchEasyRec 的平台 wheel 是优先处理项。静态发现不等于已复现性能或完成安全审计，未检查所有传递依赖和二进制供应链。','', '## 可核验证据','', '`audit_evidence.json` 保留依赖文件全文、模型定义行号/方法、根文件存在性、常见权重/LFS 扫描和最大文件清单。`inventory.json` 保留全部 Git 跟踪路径。`build_audit.py` 只用标准库读取文件与 Git 元数据，可重建本轮报告；人工审计说明固定于本次快照，更新代码后需重新审阅。',''])
with (BASE/'repository_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(BASE/'model_mapping.md').write_text('\n'.join(mapping),encoding='utf-8')
(BASE/'audit_report.md').write_text('\n'.join(report),encoding='utf-8')
(BASE/'audit_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
overview=['# 推荐系统开源代码库','', '收集日期：2026-11-04（Asia/Shanghai）。11 个指定仓库均已独立克隆成功，最终失败清单：无。未下载其他候选仓库。','', '仅下载与静态审计；没有安装依赖、运行第三方训练/安装脚本或额外下载数据/权重。浅克隆默认分支，保留原目录及独立 .git；子模块未递归克隆、LFS smudge 已关闭。源码内已跟踪的数据随 clone 原样保留，体积详见审计报告。','', '## 下载清单','', '| 本地目录 | 仓库 | 分支 | commit | 许可证 |','|---|---|---|---|---|']
for r in rows:overview.append(f'| [{r["repository"]}]({r["repository"]}/) | [{r["repository"]}]({r["canonical_url"]}) | {r["default_branch"]} | `{r["commit"]}` | {r["license"]} |')
overview.extend(['', '## 交付索引','', '- [repository_manifest.csv](repository_manifest.csv)：URL、完整 commit、分支、许可、身份与代码覆盖。','- [model_mapping.md](model_mapping.md)：逐文件模型/组件映射与源码定义行号。','- [audit_report.md](audit_report.md)：依赖、数据、训练/预测/评估、权重与完整性问题。','- [audit_evidence.json](audit_evidence.json)：静态审计证据及原始依赖文本。','- [inventory.json](inventory.json)：全部跟踪文件索引。','', '## 覆盖与限制','', '覆盖协同过滤、图推荐、序列推荐、CTR、多任务、多模态、知识图谱、生成式推荐、Learning-to-Rank 与 ANN 检索。FAISS 属检索基础设施，TFRS 属建模任务/组件库；均不伪计为一组已训练推荐模型。SASRec.pytorch 明确为第三方复现；框架内其他论文模型也不一概标为原作者官方。','', 'OpenOneRec 根 LICENSE 缺失：README 声明 Apache 2.0，不能据此忽略许可文件缺口。LightGBM 原 microsoft URL 自动跳转到官方迁移后的 lightgbm-org。Meta/CUTLASS 和 LightGBM 构建子模块未取回，因此“克隆成功”不代表“可立即编译训练”。11 个工作树均保持 clean。',''])
(BASE/'README.md').write_text('\n'.join(overview),encoding='utf-8')
print(json.dumps({'repositories':len(rows),'model_evidence_files':sum(len(v['model_definitions']) for v in evidence.values()),'all_worktrees_clean':True},ensure_ascii=False))
