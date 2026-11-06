"""适配层冒烟测试：不需要真实数据集，也不训练，只验证接口契约。

构造一份小型合成 PreparedData，按各训练脚本的 checkpoint 格式写入临时文件，
再通过统一接口加载、打分、评估，确认：

1. 四个已接入模型都能返回 {user_idx: [item_idx, ...]}；
2. 推荐结果里不含该用户已曝光的物品；
3. 统一评估能算出 Recall/NDCG/HitRate；
4. checkpoint 保存/重新加载后结果一致。

运行：python scripts/smoke_adapters.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.adapters import AdapterError, adapter_ids, load_adapter  # noqa: E402
from recsys.adapters.registry import CHECKPOINT_REQUIRED  # noqa: E402
from recsys.config import SPLIT_STRATEGY  # noqa: E402
from recsys.data.schema import (  # noqa: E402
    IdMappings,
    Interaction,
    PreparedData,
    seen_items_for_split,
)
from recsys.models.matrix_factorization import MatrixFactorization  # noqa: E402
from recsys.models.sasrec import SASRec  # noqa: E402
from recsys.models.two_tower import TwoTowerModel  # noqa: E402

NUM_USERS = 12
NUM_ITEMS = 10
TRAIN_PER_USER = 4


def build_synthetic_data() -> PreparedData:
    """每个用户 4 条训练交互 + 1 条验证 + 1 条测试，时间严格递增。"""
    train: list[Interaction] = []
    validation: list[Interaction] = []
    test: list[Interaction] = []
    validation_mask = np.zeros((NUM_USERS, NUM_ITEMS), dtype=bool)
    test_mask = np.zeros((NUM_USERS, NUM_ITEMS), dtype=bool)

    for user in range(NUM_USERS):
        watched = [(user + offset) % NUM_ITEMS for offset in range(TRAIN_PER_USER)]
        for step, item in enumerate(watched):
            train.append(Interaction(user, item, float(step)))
            validation_mask[user, item] = True
            test_mask[user, item] = True
        validation_item = (user + TRAIN_PER_USER) % NUM_ITEMS
        test_item = (user + TRAIN_PER_USER + 1) % NUM_ITEMS
        validation.append(Interaction(user, validation_item, 100.0))
        test.append(Interaction(user, test_item, 200.0))
        validation_mask[user, validation_item] = True
        test_mask[user, validation_item] = True
        test_mask[user, test_item] = True

    mappings = IdMappings(
        user_to_index={user: user for user in range(NUM_USERS)},
        index_to_user=list(range(NUM_USERS)),
        item_to_index={item: item for item in range(NUM_ITEMS)},
        index_to_item=list(range(NUM_ITEMS)),
    )
    return PreparedData(
        train_interactions=train,
        validation_interactions=validation,
        test_interactions=test,
        mappings=mappings,
        validation_seen_mask=validation_mask,
        test_seen_mask=test_mask,
    )


def checkpoint_payload(model_config: dict, state_dict: dict) -> dict:
    # 夹具借用 kuairec 这个「已登记数据集」的名字，只为让切分协议校验能走通；
    # 数据本身是合成的小样本，词表不一致时适配器会自己报错。
    return {
        "model_state_dict": state_dict,
        "split_strategy": SPLIT_STRATEGY,
        "dataset": "kuairec",
        "positive_threshold": 1.0,
        "model_config": model_config,
        "train_config": {"dataset": "kuairec"},
    }


def write_checkpoints(directory: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    torch.manual_seed(0)

    two_tower = TwoTowerModel(num_users=NUM_USERS, num_items=NUM_ITEMS, output_dim=8)
    payload = checkpoint_payload(
        {
            "num_users": NUM_USERS,
            "num_items": NUM_ITEMS,
            "embedding_dim": 64,
            "hidden_dim": 128,
            "output_dim": 8,
            "temperature": 0.2,
            "tower_type": "mlp",
        },
        two_tower.state_dict(),
    )
    paths["two_tower"] = directory / "two_tower.pt"
    torch.save(payload, paths["two_tower"])

    mf = MatrixFactorization(num_users=NUM_USERS, num_items=NUM_ITEMS, embedding_dim=8)
    payload = checkpoint_payload(
        {"num_users": NUM_USERS, "num_items": NUM_ITEMS, "embedding_dim": 8},
        mf.state_dict(),
    )
    paths["matrix_factorization"] = directory / "matrix_factorization.pt"
    torch.save(payload, paths["matrix_factorization"])

    sasrec = SASRec(num_items=NUM_ITEMS, max_seq_len=5, embedding_dim=8, num_heads=2)
    payload = checkpoint_payload(
        {
            "num_items": NUM_ITEMS,
            "max_seq_len": 5,
            "embedding_dim": 8,
            "num_blocks": 2,
            "num_heads": 2,
            "dropout": 0.2,
        },
        sasrec.state_dict(),
    )
    paths["sasrec"] = directory / "sasrec.pt"
    torch.save(payload, paths["sasrec"])
    return paths


def check_metrics() -> list[str]:
    """指标要有已知答案：用一小份手算结果验证实现没写反。"""
    from recsys.evaluation.metrics import (
        catalog_coverage,
        compute_ranking_metrics,
        roc_auc,
        log_loss,
    )

    failures: list[str] = []
    recommendations = {0: [1, 2, 3], 1: [4, 5, 6]}
    ground_truth = {0: {2}, 1: {7}}
    metrics = compute_ranking_metrics(recommendations, ground_truth, ks=(3,), num_items=10)

    expected = {
        "Recall@3": 0.5,
        "HitRate@3": 0.5,
        "Precision@3": (1 / 3) / 2,
        "NDCG@3": 0.5 * (1 / np.log2(3)),
        "Coverage": 0.6,
    }
    for key, value in expected.items():
        if abs(metrics[key] - value) > 1e-9:
            failures.append(f"指标 {key}={metrics[key]}，期望 {value}")
    if abs(catalog_coverage(recommendations, 10) - 0.6) > 1e-9:
        failures.append("Coverage 计算错误")
    if abs(roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) - 0.75) > 1e-9:
        failures.append("AUC 计算错误（期望 0.75）")
    if abs(log_loss([0.5, 0.5], [0, 1]) - np.log(2)) > 1e-9:
        failures.append("LogLoss 计算错误（期望 ln2）")
    try:
        roc_auc([0.5, 0.5], [1, 1])
        failures.append("AUC 在只有正样本时应当报错")
    except ValueError:
        pass
    if not failures:
        print("[ok] 指标（Recall/Precision/NDCG/HitRate/Coverage/AUC/LogLoss）与手算一致")
    return failures


def check_provenance(data, workdir: Path) -> list[str]:
    """工件溯源：确定性、可序列化、可被 weights_only 读回。"""
    from recsys.utils.provenance import build_provenance, describe_provenance

    failures: list[str] = []
    first = build_provenance(data, dataset="synthetic", positive_threshold=1.0)
    second = build_provenance(data, dataset="synthetic", positive_threshold=1.0)
    for key in ("data", "id_mapping", "code", "environment"):
        if key not in first:
            failures.append(f"provenance 缺字段 {key}")
    if first["data"] != second["data"]:
        failures.append("同一份数据两次指纹不一致（不可复现）")
    if first["id_mapping"]["sha256"] == "":
        failures.append("ID 映射摘要为空")

    path = workdir / "provenance.pt"
    torch.save({"provenance": first}, path)
    loaded = torch.load(path, weights_only=True)
    if loaded["provenance"]["data"] != first["data"]:
        failures.append("provenance 经 torch.save/load(weights_only=True) 后不一致")
    path.unlink(missing_ok=True)

    if "data=" not in describe_provenance(first):
        failures.append("describe_provenance 输出格式异常")
    if not failures:
        print("[ok] 工件溯源（数据指纹/映射摘要/代码指纹/依赖版本）可复现且可读回")
    return failures


def check_ranking_adapter() -> list[str]:
    """排序适配器：未安装 lightgbm 时要报可读错误，安装后要能 fit/predict/evaluate。"""
    from recsys.adapters import LightGBMRankerAdapter, RankingAdapter

    failures: list[str] = []
    random = np.random.RandomState(0)
    features = random.rand(40, 4).astype(np.float32)
    labels = (np.arange(40) % 2).astype(np.int32)

    try:
        RankingAdapter().predict(features)  # type: ignore[abstract]
    except TypeError:
        pass  # 抽象基类不可实例化，符合预期

    adapter = LightGBMRankerAdapter()
    try:
        adapter.fit(features, labels)
    except AdapterError as error:
        print(f"[skip] 未安装 lightgbm，排序适配器只验证错误路径：{error}")
        try:
            adapter.predict(features)
            failures.append("未 fit 的排序适配器应当抛 AdapterError")
        except AdapterError:
            pass
        return failures

    scores = adapter.predict(features)
    metrics = adapter.evaluate(features, labels)
    if scores.shape != (40,):
        failures.append(f"排序适配器 predict 形状异常 {scores.shape}")
    for key in ("AUC", "LogLoss"):
        if key not in metrics:
            failures.append(f"排序适配器缺指标 {key}")
    print(f"[ok] LightGBMRankerAdapter fit/predict/evaluate（AUC={metrics['AUC']:.3f}）")
    return failures


def main() -> int:
    data = build_synthetic_data()
    workdir = PROJECT_ROOT / "artifacts" / "_smoke"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    failures: list[str] = []

    try:
        checkpoints = write_checkpoints(workdir)
        for model_id in adapter_ids():
            try:
                adapter = load_adapter(
                    model_id,
                    checkpoints.get(model_id),
                    device_name="cpu",
                )
                recommendations = adapter.recommend(data, split="test", k=3)
                metrics = adapter.evaluate(data, split="test", ks=(2, 3))
            except AdapterError as error:
                failures.append(f"{model_id}: {error}")
                continue

            users = set(recommendations)
            expected_users = {row.user_idx for row in data.test_interactions}
            if users != expected_users:
                failures.append(f"{model_id}: 用户集合不完整 {len(users)}/{len(expected_users)}")
            for user, items in recommendations.items():
                if len(items) != 3:
                    failures.append(f"{model_id}: 用户 {user} 只返回 {len(items)} 个物品")
                seen = seen_items_for_split(data, "test").get(user, set())
                leaked = set(items) & seen
                if leaked:
                    failures.append(f"{model_id}: 用户 {user} 推荐了已曝光物品 {sorted(leaked)}")
            if set(metrics) != {
                "Recall@2",
                "NDCG@2",
                "HitRate@2",
                "Precision@2",
                "Recall@3",
                "NDCG@3",
                "HitRate@3",
                "Precision@3",
                "Coverage",
            }:
                failures.append(f"{model_id}: 指标键不完整 {sorted(metrics)}")

            # 向量出口：embedding 类模型必须能导出 [n, dim]，且与 item 数一致
            try:
                item_vectors = adapter.encode_items(list(range(min(4, NUM_ITEMS))))
                if item_vectors.shape[0] != min(4, NUM_ITEMS) or item_vectors.ndim != 2:
                    failures.append(f"{model_id}: encode_items 形状异常 {item_vectors.shape}")
            except AdapterError:
                if model_id != "popularity":
                    failures.append(f"{model_id}: encode_items 不可用")
            try:
                user_vectors = adapter.encode_users([0, 1])
                if user_vectors.shape != (2, item_vectors.shape[1]):
                    failures.append(
                        f"{model_id}: encode_users 形状 {user_vectors.shape} 与物品维度不一致"
                    )
            except AdapterError:
                # SASRec（序列表示）与 Popularity（无参数）不提供用户向量，属于预期
                if model_id not in {"sasrec", "popularity"}:
                    failures.append(f"{model_id}: encode_users 不可用")

            # retrieve 与 recommend 必须是同一件事（多路融合按通道调用 retrieve）
            if adapter.retrieve(data, split="test", k=3) != recommendations:
                failures.append(f"{model_id}: retrieve 与 recommend 结果不一致")

            # 保存后重新加载，推荐结果应当完全一致（checkpoint 往返一致）
            if model_id in CHECKPOINT_REQUIRED:
                roundtrip = workdir / f"{model_id}_roundtrip.pt"
                try:
                    adapter.save(roundtrip)
                    reloaded = load_adapter(model_id, roundtrip, device_name="cpu")
                    again = reloaded.recommend(data, split="test", k=3)
                except AdapterError as error:
                    failures.append(f"{model_id}: 往返失败 {error}")
                else:
                    if again != recommendations:
                        failures.append(f"{model_id}: checkpoint 往返后推荐结果不一致")

            print(
                f"[ok] {model_id:<22} users={len(recommendations)} "
                f"Recall@2={metrics['Recall@2']:.4f} NDCG@3={metrics['NDCG@3']:.4f}"
            )

        # 负例：词表不一致必须报 AdapterError，而不是静默算出错误指标
        try:
            adapter = load_adapter("two_tower", checkpoints["two_tower"], device_name="cpu")
            bigger = PreparedData(
                train_interactions=data.train_interactions,
                validation_interactions=data.validation_interactions,
                test_interactions=data.test_interactions,
                mappings=IdMappings(
                    {user: user for user in range(NUM_USERS)},
                    list(range(NUM_USERS)),
                    {item: item for item in range(NUM_ITEMS + 5)},
                    list(range(NUM_ITEMS + 5)),
                ),
            )
            adapter.recommend(bigger, split="test", k=3)
            failures.append("two_tower: 词表不一致时没有抛 AdapterError")
        except AdapterError:
            print("[ok] 词表不一致的用例按预期抛出 AdapterError")

        failures.extend(check_metrics())
        failures.extend(check_provenance(data, workdir))
        failures.extend(check_ranking_adapter())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if failures:
        print("\n失败项：")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("\n全部适配器冒烟测试通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
