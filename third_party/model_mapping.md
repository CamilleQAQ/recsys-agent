# 模型与代码映射

> **快照说明**：本文件是 2026-11-04 对当时 11 个仓库的静态审计结果，之后新增了
> `FuxiCTR`（DIN/DCNv2）与 `genrec`（TIGER）两个仓库，本文未重新扫描。
> 模型归属、完整性状态与入口以 `models/zoo.toml`（及生成的 `docs/model-zoo.md`）为准；
> TIGER 的实现位于 `genrec/genrec/models/tiger.py`。

审计日期：2026-11-04（Asia/Shanghai）。依据当前 commit 的实际源码定义、方法和函数体静态检查；未导入、训练或验证数值正确性。框架实现不等于论文作者官方实现；组件不等于可直接训练的完整推荐模型。

## RecBole

- 来源：框架官方；各论文模型为框架复现。
- 覆盖：协同过滤、图推荐、序列、CTR、知识图谱。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [recbole/model/context_aware_recommender/afm.py](RecBole/recbole/model/context_aware_recommender/afm.py) | AFM L23 |
| [recbole/model/context_aware_recommender/autoint.py](RecBole/recbole/model/context_aware_recommender/autoint.py) | AutoInt L24 |
| [recbole/model/context_aware_recommender/dcn.py](RecBole/recbole/model/context_aware_recommender/dcn.py) | DCN L30 |
| [recbole/model/context_aware_recommender/dcnv2.py](RecBole/recbole/model/context_aware_recommender/dcnv2.py) | DCNV2 L27 |
| [recbole/model/context_aware_recommender/deepfm.py](RecBole/recbole/model/context_aware_recommender/deepfm.py) | DeepFM L26 |
| [recbole/model/context_aware_recommender/dssm.py](RecBole/recbole/model/context_aware_recommender/dssm.py) | DSSM L22 |
| [recbole/model/context_aware_recommender/eulernet.py](RecBole/recbole/model/context_aware_recommender/eulernet.py) | EulerNet L25; EulerInteractionLayer L95 |
| [recbole/model/context_aware_recommender/ffm.py](RecBole/recbole/model/context_aware_recommender/ffm.py) | FFM L30; FieldAwareFactorizationMachine L164 |
| [recbole/model/context_aware_recommender/fignn.py](RecBole/recbole/model/context_aware_recommender/fignn.py) | GraphLayer L29; FiGNN L53 |
| [recbole/model/context_aware_recommender/fm.py](RecBole/recbole/model/context_aware_recommender/fm.py) | FM L26 |
| [recbole/model/context_aware_recommender/fnn.py](RecBole/recbole/model/context_aware_recommender/fnn.py) | FNN L21 |
| [recbole/model/context_aware_recommender/fwfm.py](RecBole/recbole/model/context_aware_recommender/fwfm.py) | FwFM L22 |
| [recbole/model/context_aware_recommender/kd_dagfm.py](RecBole/recbole/model/context_aware_recommender/kd_dagfm.py) | KD_DAGFM L25; DAGFM L116; CrossNet L186; CINComp L223; CIN L238 |
| [recbole/model/context_aware_recommender/lr.py](RecBole/recbole/model/context_aware_recommender/lr.py) | LR L20 |
| [recbole/model/context_aware_recommender/nfm.py](RecBole/recbole/model/context_aware_recommender/nfm.py) | NFM L21 |
| [recbole/model/context_aware_recommender/pnn.py](RecBole/recbole/model/context_aware_recommender/pnn.py) | PNN L27; InnerProductLayer L125; OuterProductLayer L164 |
| [recbole/model/context_aware_recommender/widedeep.py](RecBole/recbole/model/context_aware_recommender/widedeep.py) | WideDeep L21 |
| [recbole/model/context_aware_recommender/xdeepfm.py](RecBole/recbole/model/context_aware_recommender/xdeepfm.py) | xDeepFM L31 |
| [recbole/model/exlib_recommender/lightgbm.py](RecBole/recbole/model/exlib_recommender/lightgbm.py) | LightGBM L15 |
| [recbole/model/exlib_recommender/xgboost.py](RecBole/recbole/model/exlib_recommender/xgboost.py) | XGBoost L15 |
| [recbole/model/general_recommender/admmslim.py](RecBole/recbole/model/general_recommender/admmslim.py) | soft_threshold L21; zero_mean_columns L25; add_noise L29; ADMMSLIM L33 |
| [recbole/model/general_recommender/asymknn.py](RecBole/recbole/model/general_recommender/asymknn.py) | ComputeSimilarity L9; AsymKNN L135 |
| [recbole/model/general_recommender/bpr.py](RecBole/recbole/model/general_recommender/bpr.py) | BPR L27 |
| [recbole/model/general_recommender/cdae.py](RecBole/recbole/model/general_recommender/cdae.py) | CDAE L24 |
| [recbole/model/general_recommender/convncf.py](RecBole/recbole/model/general_recommender/convncf.py) | ConvNCFBPRLoss L26; ConvNCF L52 |
| [recbole/model/general_recommender/dgcf.py](RecBole/recbole/model/general_recommender/dgcf.py) | sample_cor_samples L35; DGCF L56 |
| [recbole/model/general_recommender/diffrec.py](RecBole/recbole/model/general_recommender/diffrec.py) | ModelMeanType L30; DNN L35; DiffRec L83; betas_from_linear_variance L544; betas_for_alpha_bar L553; normal_kl L576; mean_flat L606; timestep_embedding L613 |
| [recbole/model/general_recommender/dmf.py](RecBole/recbole/model/general_recommender/dmf.py) | DMF L28 |
| [recbole/model/general_recommender/ease.py](RecBole/recbole/model/general_recommender/ease.py) | EASE L16 |
| [recbole/model/general_recommender/enmf.py](RecBole/recbole/model/general_recommender/enmf.py) | ENMF L23 |
| [recbole/model/general_recommender/fism.py](RecBole/recbole/model/general_recommender/fism.py) | FISM L24 |
| [recbole/model/general_recommender/gcmc.py](RecBole/recbole/model/general_recommender/gcmc.py) | GCMC L34; GcEncoder L233; BiDecoder L444; orthogonal L510 |
| [recbole/model/general_recommender/itemknn.py](RecBole/recbole/model/general_recommender/itemknn.py) | ComputeSimilarity L22; ItemKNN L167 |
| [recbole/model/general_recommender/ldiffrec.py](RecBole/recbole/model/general_recommender/ldiffrec.py) | AutoEncoder L31; LDiffRec L200; compute_loss L342 |
| [recbole/model/general_recommender/lightgcn.py](RecBole/recbole/model/general_recommender/lightgcn.py) | LightGCN L32 |
| [recbole/model/general_recommender/line.py](RecBole/recbole/model/general_recommender/line.py) | NegSamplingLoss L27; LINE L35 |
| [recbole/model/general_recommender/macridvae.py](RecBole/recbole/model/general_recommender/macridvae.py) | MacridVAE L31 |
| [recbole/model/general_recommender/multidae.py](RecBole/recbole/model/general_recommender/multidae.py) | MultiDAE L24 |
| [recbole/model/general_recommender/multivae.py](RecBole/recbole/model/general_recommender/multivae.py) | MultiVAE L23 |
| [recbole/model/general_recommender/nais.py](RecBole/recbole/model/general_recommender/nais.py) | NAIS L30 |
| [recbole/model/general_recommender/nceplrec.py](RecBole/recbole/model/general_recommender/nceplrec.py) | NCEPLRec L25 |
| [recbole/model/general_recommender/ncl.py](RecBole/recbole/model/general_recommender/ncl.py) | NCL L22 |
| [recbole/model/general_recommender/neumf.py](RecBole/recbole/model/general_recommender/neumf.py) | NeuMF L27 |
| [recbole/model/general_recommender/ngcf.py](RecBole/recbole/model/general_recommender/ngcf.py) | NGCF L35 |
| [recbole/model/general_recommender/nncf.py](RecBole/recbole/model/general_recommender/nncf.py) | NNCF L29 |
| [recbole/model/general_recommender/pop.py](RecBole/recbole/model/general_recommender/pop.py) | Pop L26 |
| [recbole/model/general_recommender/ract.py](RecBole/recbole/model/general_recommender/ract.py) | RaCT L24 |
| [recbole/model/general_recommender/random.py](RecBole/recbole/model/general_recommender/random.py) | Random L20 |
| [recbole/model/general_recommender/recvae.py](RecBole/recbole/model/general_recommender/recvae.py) | swish L28; log_norm_pdf L37; CompositePrior L41; Encoder L78; RecVAE L107 |
| [recbole/model/general_recommender/sgl.py](RecBole/recbole/model/general_recommender/sgl.py) | SGL L26 |
| [recbole/model/general_recommender/simplex.py](RecBole/recbole/model/general_recommender/simplex.py) | SimpleX L26 |
| [recbole/model/general_recommender/slimelastic.py](RecBole/recbole/model/general_recommender/slimelastic.py) | SLIMElastic L23 |
| [recbole/model/general_recommender/spectralcf.py](RecBole/recbole/model/general_recommender/spectralcf.py) | SpectralCF L27 |
| [recbole/model/knowledge_aware_recommender/cfkg.py](RecBole/recbole/model/knowledge_aware_recommender/cfkg.py) | CFKG L22; InnerProductLoss L125 |
| [recbole/model/knowledge_aware_recommender/cke.py](RecBole/recbole/model/knowledge_aware_recommender/cke.py) | CKE L23 |
| [recbole/model/knowledge_aware_recommender/kgat.py](RecBole/recbole/model/knowledge_aware_recommender/kgat.py) | Aggregator L28; KGAT L75 |
| [recbole/model/knowledge_aware_recommender/kgcn.py](RecBole/recbole/model/knowledge_aware_recommender/kgcn.py) | KGCN L27 |
| [recbole/model/knowledge_aware_recommender/kgin.py](RecBole/recbole/model/knowledge_aware_recommender/kgin.py) | Aggregator L33; GraphConv L85; KGIN L250 |
| [recbole/model/knowledge_aware_recommender/kgnnls.py](RecBole/recbole/model/knowledge_aware_recommender/kgnnls.py) | KGNNLS L30 |
| [recbole/model/knowledge_aware_recommender/ktup.py](RecBole/recbole/model/knowledge_aware_recommender/ktup.py) | KTUP L27; orthogonalLoss L270; alignLoss L277 |
| [recbole/model/knowledge_aware_recommender/mcclk.py](RecBole/recbole/model/knowledge_aware_recommender/mcclk.py) | Aggregator L29; GraphConv L103; MCCLK L299 |
| [recbole/model/knowledge_aware_recommender/mkr.py](RecBole/recbole/model/knowledge_aware_recommender/mkr.py) | MKR L25; CrossCompressUnit L219 |
| [recbole/model/knowledge_aware_recommender/ripplenet.py](RecBole/recbole/model/knowledge_aware_recommender/ripplenet.py) | RippleNet L26 |
| [recbole/model/sequential_recommender/bert4rec.py](RecBole/recbole/model/sequential_recommender/bert4rec.py) | BERT4Rec L33 |
| [recbole/model/sequential_recommender/caser.py](RecBole/recbole/model/sequential_recommender/caser.py) | Caser L32 |
| [recbole/model/sequential_recommender/core.py](RecBole/recbole/model/sequential_recommender/core.py) | TransNet L21; CORE L105 |
| [recbole/model/sequential_recommender/dien.py](RecBole/recbole/model/sequential_recommender/dien.py) | DIEN L39; InterestExtractorNetwork L197; InterestEvolvingLayer L276; AGRUCell L393; AUGRUCell L438; DynamicRNN L482 |
| [recbole/model/sequential_recommender/din.py](RecBole/recbole/model/sequential_recommender/din.py) | DIN L32 |
| [recbole/model/sequential_recommender/fdsa.py](RecBole/recbole/model/sequential_recommender/fdsa.py) | FDSA L28 |
| [recbole/model/sequential_recommender/fearec.py](RecBole/recbole/model/sequential_recommender/fearec.py) | FEARec L36; HybridAttention L405; FeedForward L719; FEABlock L777; FEAEncoder L828 |
| [recbole/model/sequential_recommender/fossil.py](RecBole/recbole/model/sequential_recommender/fossil.py) | FOSSIL L25 |
| [recbole/model/sequential_recommender/fpmc.py](RecBole/recbole/model/sequential_recommender/fpmc.py) | FPMC L28 |
| [recbole/model/sequential_recommender/gcsan.py](RecBole/recbole/model/sequential_recommender/gcsan.py) | GNN L28; GCSAN L98 |
| [recbole/model/sequential_recommender/gru4rec.py](RecBole/recbole/model/sequential_recommender/gru4rec.py) | GRU4Rec L28 |
| [recbole/model/sequential_recommender/gru4reccpr.py](RecBole/recbole/model/sequential_recommender/gru4reccpr.py) | gelu L38; GRU4RecCPR L46 |
| [recbole/model/sequential_recommender/gru4recf.py](RecBole/recbole/model/sequential_recommender/gru4recf.py) | GRU4RecF L25 |
| [recbole/model/sequential_recommender/gru4reckg.py](RecBole/recbole/model/sequential_recommender/gru4reckg.py) | GRU4RecKG L23 |
| [recbole/model/sequential_recommender/hgn.py](RecBole/recbole/model/sequential_recommender/hgn.py) | HGN L25 |
| [recbole/model/sequential_recommender/hrm.py](RecBole/recbole/model/sequential_recommender/hrm.py) | HRM L27 |
| [recbole/model/sequential_recommender/ksr.py](RecBole/recbole/model/sequential_recommender/ksr.py) | KSR L24 |
| [recbole/model/sequential_recommender/lightsans.py](RecBole/recbole/model/sequential_recommender/lightsans.py) | LightSANs L23 |
| [recbole/model/sequential_recommender/narm.py](RecBole/recbole/model/sequential_recommender/narm.py) | NARM L31 |
| [recbole/model/sequential_recommender/nextitnet.py](RecBole/recbole/model/sequential_recommender/nextitnet.py) | NextItNet L28; ResidualBlock_a L155; ResidualBlock_b L211 |
| [recbole/model/sequential_recommender/npe.py](RecBole/recbole/model/sequential_recommender/npe.py) | NPE L27 |
| [recbole/model/sequential_recommender/repeatnet.py](RecBole/recbole/model/sequential_recommender/repeatnet.py) | RepeatNet L29; Repeat_Explore_Mechanism L180; Repeat_Recommendation_Decoder L220; Explore_Recommendation_Decoder L261 |
| [recbole/model/sequential_recommender/s3rec.py](RecBole/recbole/model/sequential_recommender/s3rec.py) | S3Rec L30 |
| [recbole/model/sequential_recommender/sasrec.py](RecBole/recbole/model/sequential_recommender/sasrec.py) | SASRec L26 |
| [recbole/model/sequential_recommender/sasreccpr.py](RecBole/recbole/model/sequential_recommender/sasreccpr.py) | gelu L36; SASRecCPR L44 |
| [recbole/model/sequential_recommender/sasrecf.py](RecBole/recbole/model/sequential_recommender/sasrecf.py) | SASRecF L20 |
| [recbole/model/sequential_recommender/shan.py](RecBole/recbole/model/sequential_recommender/shan.py) | SHAN L25 |
| [recbole/model/sequential_recommender/sine.py](RecBole/recbole/model/sequential_recommender/sine.py) | SINE L28 |
| [recbole/model/sequential_recommender/srgnn.py](RecBole/recbole/model/sequential_recommender/srgnn.py) | GNN L29; SRGNN L98 |
| [recbole/model/sequential_recommender/stamp.py](RecBole/recbole/model/sequential_recommender/stamp.py) | STAMP L28 |
| [recbole/model/sequential_recommender/transrec.py](RecBole/recbole/model/sequential_recommender/transrec.py) | TransRec L24 |

## TorchEasyRec

- 来源：Alibaba 官方框架；论文模型为框架实现。
- 覆盖：双塔召回、CTR、多任务、HSTU、生成式、语义ID。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [tzrec/models/dat.py](TorchEasyRec/tzrec/models/dat.py) | _update_dict_tensor L29; DATTower L39; DAT L109 |
| [tzrec/models/dbmtl.py](TorchEasyRec/tzrec/models/dbmtl.py) | DBMTL L28 |
| [tzrec/models/dc2vr.py](TorchEasyRec/tzrec/models/dc2vr.py) | DC2VR L28 |
| [tzrec/models/dcn.py](TorchEasyRec/tzrec/models/dcn.py) | DCNV1 L26 |
| [tzrec/models/dcn_v2.py](TorchEasyRec/tzrec/models/dcn_v2.py) | DCNV2 L26 |
| [tzrec/models/deepfm.py](TorchEasyRec/tzrec/models/deepfm.py) | DeepFM L26 |
| [tzrec/models/dlrm.py](TorchEasyRec/tzrec/models/dlrm.py) | DLRM L26 |
| [tzrec/models/dlrm_hstu.py](TorchEasyRec/tzrec/models/dlrm_hstu.py) | _fx_construct_payload L52; DlrmHSTU L70 |
| [tzrec/models/dlrm_hstu_onerank.py](TorchEasyRec/tzrec/models/dlrm_hstu_onerank.py) | DlrmHSTUOneRank L29 |
| [tzrec/models/dssm.py](TorchEasyRec/tzrec/models/dssm.py) | _update_dict_tensor L28; DSSMTower L38; DSSM L86 |
| [tzrec/models/dssm_v2.py](TorchEasyRec/tzrec/models/dssm_v2.py) | DSSMTower L30; DSSMV2 L120 |
| [tzrec/models/genrec_causal_lm_model.py](TorchEasyRec/tzrec/models/genrec_causal_lm_model.py) | GenRecCausalLMModel L42; _fx_wrapped_forward L288; _fx_wrapped_generate L308 |
| [tzrec/models/hstu.py](TorchEasyRec/tzrec/models/hstu.py) | _fx_filter_tensor_dict L44; HSTUUserTower L55; HSTUMatchItemTower L165; HSTUMatch L287 |
| [tzrec/models/masknet.py](TorchEasyRec/tzrec/models/masknet.py) | MaskNet L25 |
| [tzrec/models/mind.py](TorchEasyRec/tzrec/models/mind.py) | MINDUserTower L28; MINDItemTower L197; MIND L255 |
| [tzrec/models/mmoe.py](TorchEasyRec/tzrec/models/mmoe.py) | MMoE L26 |
| [tzrec/models/multi_tower.py](TorchEasyRec/tzrec/models/multi_tower.py) | MultiTower L25 |
| [tzrec/models/multi_tower_din.py](TorchEasyRec/tzrec/models/multi_tower_din.py) | MultiTowerDIN L26 |
| [tzrec/models/pepnet.py](TorchEasyRec/tzrec/models/pepnet.py) | PEPNet L27 |
| [tzrec/models/ple.py](TorchEasyRec/tzrec/models/ple.py) | PLE L27 |
| [tzrec/models/rocket_launching.py](TorchEasyRec/tzrec/models/rocket_launching.py) | RocketLaunching L28 |
| [tzrec/models/sid_rqkmeans.py](TorchEasyRec/tzrec/models/sid_rqkmeans.py) | SidRqkmeans L37 |
| [tzrec/models/sid_rqvae.py](TorchEasyRec/tzrec/models/sid_rqvae.py) | SidRqvae L40 |
| [tzrec/models/tdm.py](TorchEasyRec/tzrec/models/tdm.py) | TDM L28; TDMEmbedding L101 |
| [tzrec/models/ultra_hstu.py](TorchEasyRec/tzrec/models/ultra_hstu.py) | _HSTUTransducerStack L26; UltraHSTU L47 |
| [tzrec/models/wide_and_deep.py](TorchEasyRec/tzrec/models/wide_and_deep.py) | WideAndDeep L25 |
| [tzrec/models/wukong.py](TorchEasyRec/tzrec/models/wukong.py) | WuKong L26 |
| [tzrec/models/xdeepfm.py](TorchEasyRec/tzrec/models/xdeepfm.py) | xDeepFM L26 |

## generative-recommenders

- 来源：Meta 官方研究实现。
- 覆盖：HSTU、SASRec基线、DLRMv3/HSTU、ULTRA算子、MoL检索。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [generative_recommenders/research/modeling/sequential/hstu.py](generative-recommenders/generative_recommenders/research/modeling/sequential/hstu.py) | RelativeAttentionBiasModule L50; RelativePositionalBias L65; RelativeBucketedTimeAndPositionBasedBias L86; _hstu_attention_maybe_from_cache L149; SequentialTransductionUnitJagged L225; HSTUJagged L448; HSTU L544 |
| [generative_recommenders/research/modeling/sequential/sasrec.py](generative-recommenders/generative_recommenders/research/modeling/sequential/sasrec.py) | StandardAttentionFF L49; SASRec L84 |
| [generative_recommenders/modules/hstu_transducer.py](generative-recommenders/generative_recommenders/modules/hstu_transducer.py) | default_seq_payload L45; HSTUTransducer L54 |
| [generative_recommenders/modules/dlrm_hstu.py](generative-recommenders/generative_recommenders/modules/dlrm_hstu.py) | SequenceEmbedding L63; DlrmHSTUConfig L69; _get_supervision_labels_and_weights L108; DlrmHSTU L127 |
| [generative_recommenders/ops/pytorch/pt_hstu_ultra.py](generative-recommenders/generative_recommenders/ops/pytorch/pt_hstu_ultra.py) | get_hstu_ultra_valid_attn_mask L26; _pad_qkv L85; pytorch_hstu_ultra_attention L112 |
| [generative_recommenders/research/rails/similarities/mol/similarity_fn.py](generative-recommenders/generative_recommenders/research/rails/similarities/mol/similarity_fn.py) | _softmax_dropout_combiner_fn L36; _load_balancing_mi_loss_fn L54; SoftmaxDropoutCombiner L70; MoLGatingFn L103; MoLSimilarity L210 |

## OpenOneRec

- 来源：Kuaishou-OneRec 官方项目；包含 Qwen3/VeRL 派生代码。
- 覆盖：Qwen3/OneRec生成式推荐、Residual K-Means tokenizer、SFT/蒸馏/GRPO流程。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [pretrain/onerec_llm/models/qwen3/modeling_qwen3.py](OpenOneRec/pretrain/onerec_llm/models/qwen3/modeling_qwen3.py) | Qwen3RMSNorm L55; Qwen3MLP L75; rotate_half L91; apply_rotary_pos_emb L98; repeat_kv L125; eager_attention_forward L137; Qwen3Attention L163; Qwen3DecoderLayer L253; Qwen3RotaryEmbedding L312; Qwen3PreTrainedModel L367; Qwen3Model L464; KwargsForCausalLM L766; Qwen3ForCausalLM L769; Qwen3ForSequenceClassification L909; Qwen3ForTokenClassification L1002; Qwen3ForQuestionAnswering L1084 |
| [tokenizer/res_kmeans.py](OpenOneRec/tokenizer/res_kmeans.py) | ResKmeans L4 |

## MMRec

- 来源：MMRec 框架原仓库；论文模型包含作者实现及框架复现，不能统称各模型官方。
- 覆盖：多模态图推荐、协同过滤、内容KNN。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [src/models/bm3.py](MMRec/src/models/bm3.py) | BM3 L23 |
| [src/models/bpr.py](MMRec/src/models/bpr.py) | BPR L20 |
| [src/models/damrs.py](MMRec/src/models/damrs.py) | DAMRS L14 |
| [src/models/dragon.py](MMRec/src/models/dragon.py) | DRAGON L20; User_Graph_sample L327; GCN L344; Base_gcn L386 |
| [src/models/dualgnn.py](MMRec/src/models/dualgnn.py) | DualGNN L21; User_Graph_sample L252; GCN L269; Base_gcn L318 |
| [src/models/freedom.py](MMRec/src/models/freedom.py) | FREEDOM L22 |
| [src/models/grcn.py](MMRec/src/models/grcn.py) | SAGEConv L26; GATConv L46; EGCN L80; CGCN L112; GRCN L169 |
| [src/models/itemknncbf.py](MMRec/src/models/itemknncbf.py) | ItemKNNCBF L25 |
| [src/models/lattice.py](MMRec/src/models/lattice.py) | LATTICE L26 |
| [src/models/layergcn.py](MMRec/src/models/layergcn.py) | LayerGCN L15 |
| [src/models/lgmrec.py](MMRec/src/models/lgmrec.py) | LGMRec L18; HGNNLayer L202 |
| [src/models/lightgcn.py](MMRec/src/models/lightgcn.py) | LightGCN L23 |
| [src/models/mgcn.py](MMRec/src/models/mgcn.py) | MGCN L22 |
| [src/models/mmgcf.py](MMRec/src/models/mmgcf.py) | MMGCF L23 |
| [src/models/mmgcn.py](MMRec/src/models/mmgcn.py) | MMGCN L22; GCN L108; BaseModel L191 |
| [src/models/msca.py](MMRec/src/models/msca.py) | MSCA L22 |
| [src/models/mvgae.py](MMRec/src/models/mvgae.py) | MVGAE L27; GCN L183; ProductOfExperts L285; BaseModel L304 |
| [src/models/pgl.py](MMRec/src/models/pgl.py) | PGL L22 |
| [src/models/selfcfed_lgn.py](MMRec/src/models/selfcfed_lgn.py) | SELFCFED_LGN L28 |
| [src/models/slmrec.py](MMRec/src/models/slmrec.py) | SLMRec L20 |
| [src/models/smore.py](MMRec/src/models/smore.py) | SMORE L24 |
| [src/models/vbpr.py](MMRec/src/models/vbpr.py) | VBPR L20 |

## DeepCTR-Torch

- 来源：DeepCTR 作者维护的 PyTorch 库；论文模型为库内复现。
- 覆盖：CTR特征交叉、DIN/DIEN、多任务。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [deepctr_torch/models/afm.py](DeepCTR-Torch/deepctr_torch/models/afm.py) | AFM L15 |
| [deepctr_torch/models/afn.py](DeepCTR-Torch/deepctr_torch/models/afn.py) | AFN L17 |
| [deepctr_torch/models/autoint.py](DeepCTR-Torch/deepctr_torch/models/autoint.py) | AutoInt L16 |
| [deepctr_torch/models/ccpm.py](DeepCTR-Torch/deepctr_torch/models/ccpm.py) | CCPM L21 |
| [deepctr_torch/models/dcn.py](DeepCTR-Torch/deepctr_torch/models/dcn.py) | DCN L18 |
| [deepctr_torch/models/dcnmix.py](DeepCTR-Torch/deepctr_torch/models/dcnmix.py) | DCNMix L18 |
| [deepctr_torch/models/deepfm.py](DeepCTR-Torch/deepctr_torch/models/deepfm.py) | DeepFM L14 |
| [deepctr_torch/models/dien.py](DeepCTR-Torch/deepctr_torch/models/dien.py) | DIEN L16; InterestExtractor L181; InterestEvolving L276 |
| [deepctr_torch/models/difm.py](DeepCTR-Torch/deepctr_torch/models/difm.py) | DIFM L14 |
| [deepctr_torch/models/din.py](DeepCTR-Torch/deepctr_torch/models/din.py) | DIN L15 |
| [deepctr_torch/models/fibinet.py](DeepCTR-Torch/deepctr_torch/models/fibinet.py) | FiBiNET L15 |
| [deepctr_torch/models/ifm.py](DeepCTR-Torch/deepctr_torch/models/ifm.py) | IFM L14 |
| [deepctr_torch/models/mlr.py](DeepCTR-Torch/deepctr_torch/models/mlr.py) | MLR L17 |
| [deepctr_torch/models/multitask/esmm.py](DeepCTR-Torch/deepctr_torch/models/multitask/esmm.py) | ESMM L17 |
| [deepctr_torch/models/multitask/mmoe.py](DeepCTR-Torch/deepctr_torch/models/multitask/mmoe.py) | MMOE L17 |
| [deepctr_torch/models/multitask/ple.py](DeepCTR-Torch/deepctr_torch/models/multitask/ple.py) | PLE L17 |
| [deepctr_torch/models/multitask/sharedbottom.py](DeepCTR-Torch/deepctr_torch/models/multitask/sharedbottom.py) | SharedBottom L17 |
| [deepctr_torch/models/nfm.py](DeepCTR-Torch/deepctr_torch/models/nfm.py) | NFM L16 |
| [deepctr_torch/models/onn.py](DeepCTR-Torch/deepctr_torch/models/onn.py) | Interac L14; ONN L37 |
| [deepctr_torch/models/pnn.py](DeepCTR-Torch/deepctr_torch/models/pnn.py) | PNN L17 |
| [deepctr_torch/models/wdl.py](DeepCTR-Torch/deepctr_torch/models/wdl.py) | WDL L14 |
| [deepctr_torch/models/xdeepfm.py](DeepCTR-Torch/deepctr_torch/models/xdeepfm.py) | xDeepFM L15 |

## LightGBM

- 来源：官方仓库已迁移至 lightgbm-org/LightGBM；原URL重定向。
- 覆盖：GBDT/DART/GOSS/RF；LambdaRank、rank_xendcg排序目标。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [python-package/lightgbm/sklearn.py](LightGBM/python-package/lightgbm/sklearn.py) | _get_group_from_constructed_dataset L127; _get_label_from_constructed_dataset L137; _get_weight_from_constructed_dataset L147; _ObjectiveFunctionWrapper L157; _EvalFunctionWrapper L244; _extract_evaluation_meta_data L325; _validate_eval_set_Xy L349; LGBMModel L395; LGBMRegressor L1428; LGBMClassifier L1771; LGBMRanker L2399 |
| [python-package/lightgbm/engine.py](LightGBM/python-package/lightgbm/engine.py) | _choose_num_iterations L52; train L115; CVBooster L367; _make_n_folds L536; _agg_cv_result L609; cv L649 |
| [src/objective/rank_objective.hpp](LightGBM/src/objective/rank_objective.hpp) | class RankingObjective : public ObjectiveFunction { L26; class LambdarankNDCG : public RankingObjective { L139; class RankXENDCG : public RankingObjective { L385 |
| [src/boosting/gbdt.cpp](LightGBM/src/boosting/gbdt.cpp) | C++/Rust 索引或模型实现文件，非 Python 类 |

## faiss

- 来源：Meta/FAIR 官方向量检索库。
- 覆盖：Flat/IVF/PQ/HNSW/LSH、聚类；非端到端推荐模型。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [faiss/IndexFlat.cpp](faiss/faiss/IndexFlat.cpp) | struct FlatL2Dis : FlatCodesDistanceComputer { L141; struct FlatIPDis : FlatCodesDistanceComputer { L247; struct FlatL2WithNormsDis : FlatCodesDistanceComputer { L341 |
| [faiss/IndexIVF.cpp](faiss/faiss/IndexIVF.cpp) | C++/Rust 索引或模型实现文件，非 Python 类 |
| [faiss/IndexPQ.cpp](faiss/faiss/IndexPQ.cpp) | struct PreSortedArray { L414; struct ArgSort { L439; struct SortedArray { L450; struct SemiSortedArray { L524; struct MinSumK { L615 |
| [faiss/IndexHNSW.cpp](faiss/faiss/IndexHNSW.cpp) | struct SinkScan { L933; struct IndexHNSWLinkSample : HNSWLinkSampleSearch { L1093 |
| [faiss/IndexLSH.cpp](faiss/faiss/IndexLSH.cpp) | C++/Rust 索引或模型实现文件，非 Python 类 |
| [faiss/Clustering.cpp](faiss/faiss/Clustering.cpp) | C++/Rust 索引或模型实现文件，非 Python 类 |

## SASRec.pytorch

- 来源：第三方复现：Zan Huang/pmixer，基于 kang205/SASRec 的论文作者 TensorFlow 实现。
- 覆盖：SASRec；附Rust实现路径，非另一个独立模型。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [python/model.py](SASRec.pytorch/python/model.py) | PointWiseFeedForward L5; SASRec L25 |
| [rust/src/model.rs](SASRec.pytorch/rust/src/model.rs) | pub struct PointWiseFeedForward { L4; pub struct MultiHeadAttention { L31; pub struct TransformerBlock { L113; pub struct MLP { L142; pub struct LayerNorm { L161; pub struct SASRec { L176; pub struct Args { L281 |

## BERT4Rec

- 来源：论文作者 Fei Sun 原始实现；基于BERT架构代码。
- 覆盖：BERT4Rec 双向序列推荐。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [modeling.py](BERT4Rec/modeling.py) | BertConfig L18; BertModel L94; gelu L257; get_activation L273; get_assignment_map_from_checkpoint L310; dropout L337; layer_norm L355; layer_norm_and_dropout L364; create_initializer L371; embedding_lookup L376; embedding_postprocessor L425; create_attention_mask_from_input_mask L522; attention_layer L556; transformer_model L754; get_shape_list L900; reshape_to_matrix L937; reshape_from_matrix L951; assert_rank L964 |
| [run.py](BERT4Rec/run.py) | EvalHooks L123; model_fn_builder L232; get_masked_lm_output L360; gather_indexes L408; input_fn_builder L424; _decode_record L481; main L496 |

## recommenders

- 来源：TensorFlow 官方 TFRS 库。
- 覆盖：双塔检索/排序任务、Cross/DCN层、BruteForce/Streaming/ScaNN检索接口。
- 实现证据（完整实现类/函数与辅助组件均列出；不按类数量统计模型）：

| 文件 | 源码定义（行号） |
|---|---|
| [tensorflow_recommenders/tasks/retrieval.py](recommenders/tensorflow_recommenders/tasks/retrieval.py) | Retrieval L29; _cross_replica_concat L238 |
| [tensorflow_recommenders/tasks/ranking.py](recommenders/tensorflow_recommenders/tasks/ranking.py) | Ranking L26 |
| [tensorflow_recommenders/layers/factorized_top_k.py](recommenders/tensorflow_recommenders/layers/factorized_top_k.py) | _wrap_batch_too_small_error L35; _take_along_axis L57; _exclude L83; _check_candidates_with_identifiers L118; TopK L140; Streaming L336; BruteForce L515; ScaNN L613 |
| [tensorflow_recommenders/experimental/models/ranking.py](recommenders/tensorflow_recommenders/experimental/models/ranking.py) | Ranking L27 |

## 跨仓库映射与避免误计

| 模型/任务 | 主要源码位置 | 说明 |
|---|---|---|
| SASRec | RecBole/recbole/model/sequential_recommender/sasrec.py；SASRec.pytorch/python/model.py；Meta research/modeling/sequential/sasrec.py | 框架复现、第三方复现、研究基线；不重复克隆 |
| BERT4Rec | BERT4Rec/modeling.py + run.py；RecBole/recbole/model/sequential_recommender/bert4rec.py | 原作者 TF1 与框架 PyTorch 复现 |
| LightGCN/BPR | RecBole 与 MMRec 相应 model 文件 | 多仓库已有模型不新增下载 |
| DeepFM/DCN/DIN/多任务 | DeepCTR-Torch、TorchEasyRec、RecBole 对应实现 | 各库支持范围以逐文件列表为准 |
| HSTU | Meta research/modules 与 TorchEasyRec/tzrec/models | 架构与训练栈不同，不能互换 checkpoint |
| OneRec | OpenOneRec/pretrain；tokenizer；verl_rl/verl_distillation | 1.7B/8B/Pro 是外部权重版本，不是四份独立源码实现 |
| LambdaRank/rank_xendcg | LightGBM/src/objective/rank_objective.hpp | 排序目标；不把 GBDT 分类器都计为推荐算法 |
| ANN | faiss/Index*.cpp | 检索基础设施，需要上游 embedding |
| TFRS | tasks、layers、experimental/models 与 notebooks | 任务/层/教程组合，不宣称完整论文模型集 |

OpenOneRec README 比较表中的 TIGER、LC-Rec 等不能直接计为该仓库实现；Qwen3 的通用分类/问答头也不是新增推荐模型。MMRec 的 MG 对应训练策略，不能当成另一个模型文件。模型表中出现外部库封装（例如 RecBole LightGBM/XGBoost）仅表示适配器存在，底层算法需相应外部实现。
