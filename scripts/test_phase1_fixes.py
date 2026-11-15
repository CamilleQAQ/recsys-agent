"""Phase 1 三个已确认修复的针对性检查（小样本 / 合成 fixture）。

    python scripts/test_phase1_fixes.py

覆盖：
1. 排序历史只取“请求前”，同一时刻排除；
2. 缺请求时间戳的旧工件：序列排序模型报错，DCNv2 仍兼容；
3. include_validation_in_test_history 在序列/MIND/TIGER/神经排序/推荐链路/Agent 入口生效；
4. Agent 训练写入本次运行唯一目录，不覆盖业务默认 checkpoint，新权重可被后续工具使用。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recsys.adapters.neural_ranking import NeuralRankingAdapter  # noqa: E402
from recsys.adapters.rechub_mind import MindAdapter  # noqa: E402
from recsys.data.loader import prepare_dataset  # noqa: E402
from recsys.data.schema import IdMappings, Interaction, PreparedData  # noqa: E402
from recsys.evaluation.tiger_evaluator import generate_tiger_scored  # noqa: E402
from recsys.experiments.rec_tools import (  # noqa: E402
    ToolBudget,
    ToolCallError,
    ToolContext,
    build_registry,
    resolve_checkpoint,
    resolve_ranker_checkpoint,
)
from recsys.models.rqvae import SemanticIDIndex  # noqa: E402
from recsys.pipeline.recall import run_channel  # noqa: E402
from recsys.ranking.samples import (  # noqa: E402
    RankingTimestampError,
    build_history_source,
    histories_for_requests,
    load_ranking_view,
)
from recsys.training.train_neural_ranker import (  # noqa: E402
    NeuralRankerTrainConfig,
    train_neural_ranker,
)

DATASET = "kuasar_small"
ARTIFACTS = PROJECT_ROOT / "artifacts"
TEST_ROOT = ARTIFACTS / "_fix_three_tests"
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: object = "") -> None:
    if condition:
        print(f"[ok] {name}")
    else:
        FAILURES.append(f"{name}: {detail}")
        print(f"[FAIL] {name}: {detail}")


def synthetic_data() -> PreparedData:
    train = [Interaction(0, 0, 1.0)]
    validation = [Interaction(0, 1, 2.0)]
    test = [Interaction(0, 2, 3.0)]
    mappings = IdMappings({0: 0}, [0], {i: i for i in range(10)}, list(range(10)))
    return PreparedData(train, validation, test, mappings)


def synthetic_history_data() -> PreparedData:
    """用户 0 在 t=1,2,3 有点击；用来验证请求前/同一时刻/请求后。"""
    train = [Interaction(0, 0, 1.0), Interaction(0, 1, 2.0), Interaction(0, 2, 3.0)]
    validation = [Interaction(0, 3, 4.0)]
    test = [Interaction(0, 4, 5.0)]
    mappings = IdMappings({0: 0}, [0], {i: i for i in range(10)}, list(range(10)))
    return PreparedData(train, validation, test, mappings)


def test_strict_before() -> None:
    data = synthetic_history_data()
    source = build_history_source(data, "train")
    before = histories_for_requests(source, [(0, 2)], max_history=3)[(0, 2)].tolist()
    same = histories_for_requests(source, [(0, 1)], max_history=3)[(0, 1)].tolist()
    after = histories_for_requests(source, [(0, 4)], max_history=3)[(0, 4)].tolist()
    check("请求前/同一时刻/请求后：t=2 只含 t=1", before == [0, 0, 1], before)
    check("请求前/同一时刻/请求后：t=1 无历史", same == [0, 0, 0], same)
    check("请求前/同一时刻/请求后：t=4 含 t=1,2,3", after == [1, 2, 3], after)
    shared = histories_for_requests(source, [(0, 2), (0, 2)], max_history=3)
    check("同一请求的候选共享同一份历史", len(shared) == 1, shared)


def _old_ranking_dir(root: Path, data: PreparedData) -> Path:
    """复制 ids + validation 数组，但不带 timestamps，模拟旧工件。"""
    shutil.rmtree(root, ignore_errors=True)
    (root / "ranking").mkdir(parents=True)
    (root / "ids").mkdir()
    ids = Path("data/kuasar/processed/ids")
    shutil.copy(ids / "user_ids.npy", root / "ids" / "user_ids.npy")
    shutil.copy(ids / "item_ids.npy", root / "ids" / "item_ids.npy")
    full_users = np.load(ids / "user_ids.npy")
    full_items = np.load(ids / "item_ids.npy")
    raw_user = int(data.mappings.index_to_user[0])
    items = [int(data.mappings.index_to_item[i]) for i in range(2)]
    user_index = int(np.searchsorted(full_users, raw_user))
    item_indices = [int(np.searchsorted(full_items, item)) for item in items]
    for split in ("train", "validation"):
        np.save(root / "ranking" / f"{split}_users.npy", np.array([user_index, user_index], dtype=np.int32))
        np.save(root / "ranking" / f"{split}_items.npy", np.array(item_indices, dtype=np.int32))
        np.save(root / "ranking" / f"{split}_labels.npy", np.array([1, 0], dtype=np.int8))
        np.save(root / "ranking" / f"{split}_groups.npy", np.array([2], dtype=np.int32))
    return root / "ranking"


def test_old_artifacts() -> None:
    data = prepare_dataset(DATASET, PROJECT_ROOT / "data")
    ranking = _old_ranking_dir(TEST_ROOT / "old_ranking", data)
    compatible = load_ranking_view(
        data, ranking, split="validation", max_rows=100, require_both_labels=False,
        require_timestamps=False,
    )
    check("旧工件：DCNv2/LightGBM 路径可读且 timestamps=None", compatible.timestamps is None)
    try:
        load_ranking_view(
            data, ranking, split="validation", max_rows=100, require_both_labels=False,
            require_timestamps=True,
        )
    except RankingTimestampError as error:
        check("旧工件：序列排序模型明确报错并给重建命令", "kuasar_ranking --force" in str(error))
    else:
        check("旧工件：序列排序模型明确报错并给重建命令", False, "no error")
    try:
        train_neural_ranker(
            NeuralRankerTrainConfig(
                model="din",
                dataset=DATASET,
                data_dir=PROJECT_ROOT / "data",
                parent_ranking=ranking,
                epochs=1,
                batch_size=2,
                max_train_rows=100,
                max_valid_rows=100,
                max_batches_per_epoch=1,
                artifacts_dir=TEST_ROOT / "old_train",
            )
        )
    except RankingTimestampError as error:
        check("旧工件：DIN 训练入口报错且包含模型名", "din" in str(error))
    else:
        check("旧工件：DIN 训练入口报错且包含模型名", False, "no error")
    dcnv2 = train_neural_ranker(
        NeuralRankerTrainConfig(
            model="dcnv2",
            dataset=DATASET,
            data_dir=PROJECT_ROOT / "data",
            parent_ranking=ranking,
            epochs=1,
            batch_size=2,
            max_train_rows=100,
            max_valid_rows=100,
            max_batches_per_epoch=1,
            artifacts_dir=TEST_ROOT / "old_train",
        )
    )
    check("旧工件：DCNv2 训练兼容", Path(dcnv2["checkpoint"]).exists(), dcnv2["checkpoint"])


def test_history_switch() -> None:
    data = synthetic_data()
    for flag in (True, False):
        from recsys.data.dataset import build_evaluation_examples, build_observed_histories

        example = build_evaluation_examples(
            data, "test", 5, include_validation_in_test_history=flag
        )[0]
        history = build_observed_histories(data, "test", flag)[0]
        if flag:
            check("SASRec/BERT4Rec/HSTU 输入含验证历史", example.input_sequence[:2] == [1, 2], example.input_sequence)
            check("MIND 输入含验证历史", history == [0, 1], history)
        else:
            check("SASRec/BERT4Rec/HSTU 输入不含验证历史", example.input_sequence[:2] == [1, 0], example.input_sequence)
            check("MIND 输入不含验证历史", history == [0], history)

    mind = MindAdapter(torch.nn.Identity(), {"model_config": {"max_length": 5}}, torch.device("cpu"))
    with patch("recsys.adapters.rechub_mind.rank_users", return_value={0: [2]}) as score_mind:
        mind.evaluate(data, split="test", ks=(1,), include_validation_in_test_history=False)
    history = score_mind.call_args.args[2]
    check("MIND evaluate(False) 不使用验证历史", history == {0: [0]}, history)

    adapter = NeuralRankingAdapter(
        torch.nn.Identity(),
        {"train_config": {"max_history": 5}, "model_config": {}},
        torch.device("cpu"),
    )
    true_history = adapter._history_tokens(data, "test", [0], None, True)[0].tolist()
    false_history = adapter._history_tokens(data, "test", [0], None, False)[0].tolist()
    check("DIN 输入含验证历史", true_history[-2:] == [1, 2], true_history)
    check("DIN 输入不含验证历史", false_history[-1:] == [1] and 2 not in false_history, false_history)

    codes = np.stack([np.arange(10), np.arange(10) % 3, np.arange(10) % 2], axis=1).astype(np.int32)
    index = SemanticIDIndex(codes)

    class FakeTiger:
        max_history_tokens = 10

        def __init__(self) -> None:
            self.seen = None

        def eval(self) -> None:
            pass

        def parameters(self):
            return iter([torch.zeros(1)])

        def generate_user(self, tokens, sid_index, **kwargs):
            self.seen = tokens.tolist()
            return []

    for flag in (True, False):
        model = FakeTiger()
        generate_tiger_scored(
            model, data, "test", codes, index, users=[0],
            include_validation_in_test_history=flag,
        )
        contains_validation = 3 in model.seen
        check(f"TIGER 输入 {'含' if flag else '不含'}验证历史", contains_validation == flag, model.seen)

    # 推荐主链：sequence 通道把 flag 传到模型输入
    class FakeModel:
        max_seq_len = 5

        def __init__(self) -> None:
            self.seen = None

        def eval(self) -> None:
            pass

        def score_all_items(self, sequences):
            self.seen = sequences.tolist()
            return torch.zeros(sequences.shape[0], data.num_items)

    class FakeSequenceAdapter:
        model_id = "sasrec"
        checkpoint = {"model_config": {"max_seq_len": 5}}
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.model = FakeModel()

    for flag in (True, False):
        fake = FakeSequenceAdapter()
        run_channel(
            fake, data, [0], split="test", k=2,
            include_validation_in_test_history=flag,
        )
        check(
            f"推荐主链 sequence flag={flag} 到达模型输入",
            (2 in fake.model.seen[0]) == flag,
            fake.model.seen,
        )

    # Agent 召回入口
    import recsys.adapters.registry as registry_module

    created: list[FakeSequenceAdapter] = []

    def fake_load(model_id, checkpoint, device_name="auto"):
        adapter = FakeSequenceAdapter()
        created.append(adapter)
        return adapter

    with patch.object(registry_module, "load_adapter", fake_load):
        for flag in (True, False):
            context = ToolContext(
                dataset=DATASET,
                experiment_dir=TEST_ROOT / "agent_recall",
                users=1,
                split="test",
                budget=ToolBudget(max_tool_calls=2, max_seconds=60),
            )
            context._data = data
            result = build_registry().call(
                "recall_models",
                {"models": ["sasrec"], "split": "test", "k": 2,
                 "include_validation_in_test_history": flag},
                context,
            )
            check(
                f"Agent recall flag={flag} 到达模型输入",
                result.get("ok") and ((2 in created[-1].model.seen[0]) == flag),
                created[-1].model.seen,
            )


def test_agent_history_conflicts() -> None:
    key = "include_validation_in_test_history"
    context = ToolContext(split="test", state={key: False})
    registry = build_registry()
    for tool, arguments in (
        ("recall_models", {"models": ["sasrec"]}),
        ("rank_candidates", {"model": "din"}),
        ("run_pipeline", {"models": ["sasrec"]}),
    ):
        try:
            registry.call(tool, {**arguments, key: True}, context)
        except ToolCallError as error:
            check(f"{tool} 拒绝更改任务历史设置", key in str(error) and context.state[key] is False, error)
        else:
            check(f"{tool} 拒绝更改任务历史设置", False)


def test_agent_pipeline_flag() -> None:
    context = ToolContext(
        dataset=DATASET,
        experiment_dir=TEST_ROOT / "agent_pipeline",
        users=2,
        split="test",
        budget=ToolBudget(max_tool_calls=3, max_seconds=180),
    )
    result = build_registry().call(
        "run_pipeline",
        {
            "models": ["popularity"],
            "ranker": "lightgbm_ranker",
            "split": "test",
            "k_recall": 20,
            "k_final": 10,
            "include_validation_in_test_history": False,
            "output_name": "history_false",
        },
        context,
    )
    meta_path = TEST_ROOT / "agent_pipeline" / "pipeline" / "history_false" / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    check(
        "Agent run_pipeline flag 写入配置与 run_meta",
        result.get("ok")
        and result.get("include_validation_in_test_history") is False
        and meta.get("config", {}).get("run", {}).get("include_validation_in_test_history") is False,
        {"result": result.get("include_validation_in_test_history"), "meta": meta.get("config", {})},
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agent_training_isolation() -> None:
    import torch

    defaults = {
        "din": ARTIFACTS / DATASET / "din.pt",
        "lightgcn": ARTIFACTS / DATASET / "lightgcn.pt",
    }
    before = {name: _sha256(path) for name, path in defaults.items() if path.exists()}
    context = ToolContext(
        dataset=DATASET,
        experiment_dir=TEST_ROOT / "agent_train",
        users=3,
        budget=ToolBudget(
            max_tool_calls=5, max_train_calls=2, max_epochs=1, max_batches=1, max_seconds=300
        ),
    )
    registry = build_registry()
    din = registry.call(
        "train_model",
        {"model": "din", "epochs": 1, "max_batches": 1, "embedding_dim": 8,
         "validation_users": 3, "max_rows": 1000},
        context,
    )
    lightgcn = registry.call(
        "train_model",
        {"model": "lightgcn", "epochs": 1, "max_batches": 1, "embedding_dim": 8,
         "validation_users": 3},
        context,
    )
    after = {name: _sha256(path) for name, path in defaults.items() if path.exists()}
    check("Agent 训练返回实际 checkpoint", din.get("ok") and lightgcn.get("ok"), (din, lightgcn))
    check("业务默认 checkpoint 哈希未变", before == after, {"before": before, "after": after})
    din_path = Path(din["checkpoint"])
    lightgcn_path = Path(lightgcn["checkpoint"])
    check(
        "重复/并发训练目录唯一",
        din_path.parent != lightgcn_path.parent
        and "training" in din_path.parts
        and "training" in lightgcn_path.parts,
        (din_path, lightgcn_path),
    )
    check("任务内权重引用已更新", set(context.state.get("trained_checkpoints", {})) == {"din", "lightgcn"})
    from recsys.adapters.ranking_registry import load_ranker

    loaded = load_ranker("din", resolve_ranker_checkpoint(context, "din"), "cpu")
    check("任务内新 DIN 可重新加载", loaded is not None)
    check("resolve 使用任务内权重", resolve_ranker_checkpoint(context, "din") == din_path.resolve())
    # 后续排序工具使用这次训练的 DIN
    registry.call("recall_models", {"models": ["popularity"], "k": 3}, context)
    ranked = registry.call("rank_candidates", {"model": "din"}, context)
    check("后续 rank 工具使用任务内 DIN", ranked.get("ok") and ranked.get("checkpoint") == din["checkpoint"], ranked)

    # 失败训练不得更新引用
    empty = ToolContext(
        dataset=DATASET,
        artifacts_dir=TEST_ROOT / "empty_artifacts",
        experiment_dir=TEST_ROOT / "failure",
        users=1,
        budget=ToolBudget(max_tool_calls=2, max_train_calls=1),
    )
    empty.state["trained_checkpoints"] = {"din": str(din_path)}
    try:
        registry.call("train_model", {"model": "tiger", "epochs": 1, "max_batches": 1}, empty)
    except ToolCallError:
        pass
    check(
        "训练失败不更新权重引用",
        set(empty.state.get("trained_checkpoints", {})) == {"din"},
        empty.state.get("trained_checkpoints"),
    )


def main() -> int:
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        test_strict_before()
        test_old_artifacts()
        test_history_switch()
        test_agent_history_conflicts()
        test_agent_pipeline_flag()
        test_agent_training_isolation()
    finally:
        shutil.rmtree(TEST_ROOT, ignore_errors=True)
    if FAILURES:
        print("\n失败项：")
        for failure in FAILURES:
            print(f" - {failure}")
        return 1
    print("\n三个修复的针对性检查全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
