"""项目级配置常量。

数据主线是 KuaiSAR（Phase 1 接入中）；KuaiRec 作为流程模板/辅助线保留常量。
只记录当前在用数据线的常量：数据集增删时同步改这里与 configs/datasets/。
"""

from __future__ import annotations

from pathlib import Path

# 数据切分策略。会写进 checkpoint，并在评测时校验「数据是否按同一套规则切分」。
# 具体做法见 recsys.data.schema.temporal_train_validation_test_split：
# 每个用户的最后两次交互依次作为验证集与测试集。
SPLIT_STRATEGY = "per_user_last_two_strict_timestamp_v2"

# 默认目录。CLI 参数默认值同样使用这两个路径。
DEFAULT_DATA_DIR = Path("data")
DEFAULT_ARTIFACTS_DIR = Path("artifacts")

# 默认数据集：KuaiSAR 是主数据线（需先跑 python -m recsys.data.kuasar_store 构建 processed）。
DEFAULT_DATASET = "kuasar"

# KuaiRec 的正反馈阈值：watch_ratio >= 1.0（完整播放）记为「喜欢」。
# 这个值必须在训练之前定死：正样本集合决定用户/物品词表，
# 之后换阈值会让所有已交付的 checkpoint 和指标失效。
KUAIREC_WATCH_RATIO_THRESHOLD = 1.0

# 是否把同一个 (user, video) 的多次播放合并成一次。
# KuaiRec 的 big_matrix 有 17.8% 的重复行，其中一部分是同一次播放被记了多行，
# 另一部分是同一用户在不同时间重复播放同一个视频。
# 不合并会让评测目标出现在训练历史里，从而被 Top-K 的「已看过就不推荐」规则过滤掉。
KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS = True

# KuaiRec small_matrix：与 big_matrix 同一套解析与标签口径，只是矩阵是稠密的
# （1,411 用户 × 3,327 视频，4,676,570 行播放记录，填充率 95.7%）。
# 这三个数用于建库时校验「拿到的是完整文件而不是半截下载」；
# 改动它们等于换了数据集，必须同时改 configs/datasets/kuairec_small.toml。
KUAIREC_SMALL_MATRIX_ROWS = 4_676_570
KUAIREC_SMALL_MATRIX_USERS = 1_411
KUAIREC_SMALL_MATRIX_ITEMS = 3_327
# 正样本覆盖到的物品数：有 10 个视频没有任何 watch_ratio >= 1.0 的首次曝光，
# 因此模型物品词表是 3,317 而不是原始目录大小 3,327。
KUAIREC_SMALL_POSITIVE_ITEMS = 3_317
