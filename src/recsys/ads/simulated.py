"""独立模拟转化样本（真实转化标签缺失时的多任务接口验证夹具）。

**这不是 KuaiSAR 的真实转化数据。** ``recsys.ads.data.audit_conversion_labels()``
已核对 ``rec_inter.csv`` 只有 click / like / follow / forward / play_time 等行为，
没有 conversion / purchase 字段。为了验证 MMoE / PLE 的多任务输入输出，这里生成
一份**显式标注为 synthetic** 的独立样本：

* 来源：``kuasar_small`` 广告样本 train 切分的前 ``requests`` 个请求（固定顺序）；
* 点击标签：沿用真实 ``click``（0/1）；
* 转化标签：只在 ``click == 1`` 的行上生成，
  ``blake2b("{user}:{item}:{seed}") % 100 < cvr_percent`` 记 ``conversion = 1``；
* 生成规则、seed、CVR 比例全部写进 ``schema.json``，并标记
  ``synthetic=true / not_real_conversion=true / not_for_effect_claims=true``。

产物目录：``data/<dataset>/processed/ads_simulated/``，与真实广告样本目录分离，
不会覆盖或混淆真实工件。

用法：

    python -m recsys.ads.simulated           # 构建
    python -m recsys.ads.simulated --check   # 校验
    python -m recsys.ads.simulated --force   # 重建
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recsys.ads.data import SPLITS, _sha256, ads_dir, load_ad_store
from recsys.config import DEFAULT_DATA_DIR
from recsys.data.loader import resolve_dataset_data_dir

SIMULATED_SCHEMA_VERSION = 1
DEFAULT_CVR_PERCENT = 20
DEFAULT_REQUESTS = 300


@dataclass(frozen=True)
class SimulatedAdView:
    """模拟转化样本视图；``click`` 为真实点击，``conversion`` 为合成标签。"""

    dataset: str
    users: np.ndarray
    items: np.ndarray
    click: np.ndarray
    conversion: np.ndarray
    groups: np.ndarray
    timestamps: np.ndarray
    synthetic: bool = True

    def __len__(self) -> int:
        return int(self.items.size)

    @property
    def requests(self) -> int:
        return int(self.groups.size)


def simulated_dir(dataset_root: Path) -> Path:
    return Path(dataset_root) / "processed" / "ads_simulated"


def _conversion_hash(user: int, item: int, seed: int) -> int:
    raw = f"{int(user)}:{int(item)}:{int(seed)}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "big")


def build_simulated_fixture(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    dataset: str = "kuasar_small",
    split: str = "train",
    requests: int = DEFAULT_REQUESTS,
    cvr_percent: int = DEFAULT_CVR_PERCENT,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    if not 1 <= cvr_percent <= 100:
        raise ValueError("cvr_percent 必须在 1..100")
    dataset_root = resolve_dataset_data_dir(dataset, Path(data_dir))
    target = simulated_dir(dataset_root)
    if (target / "schema.json").exists() and not force:
        print(f"[reuse] {target} 模拟转化样本已存在；加 --force 重建")
        return json.loads((target / "schema.json").read_text(encoding="utf-8"))

    view = load_ad_store(dataset, data_dir, split)
    groups = np.asarray(view.groups)
    starts = np.concatenate(([0], np.cumsum(groups)))
    take = min(int(requests), int(groups.size))
    if take <= 0:
        raise ValueError(f"{split} 切分没有广告请求")
    rows = np.arange(int(starts[0]), int(starts[take]))

    items = np.asarray(view.items[rows], dtype=np.int32)
    users = np.asarray(view.users[rows], dtype=np.int32)
    click = np.asarray(view.labels[rows], dtype=np.int8)
    conversion = np.zeros(click.size, dtype=np.int8)
    for index, (user, item) in enumerate(zip(users.tolist(), items.tolist())):
        if click[index] and _conversion_hash(user, item, seed) % 100 < cvr_percent:
            conversion[index] = 1
    if conversion.sum() == 0:
        raise RuntimeError("模拟转化样本没有正例；调大 cvr_percent 或 requests")

    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "users": users,
        "items": items,
        "click": click,
        "conversion": conversion,
        "groups": groups[:take].astype(np.int32),
        "timestamps": np.asarray(view.timestamps[:take], dtype=np.int64),
    }
    fingerprints: dict[str, str] = {}
    for name, values in payload.items():
        path = target / f"{name}.npy"
        np.save(path, values)
        fingerprints[name] = _sha256(path)

    schema = {
        "schema_version": SIMULATED_SCHEMA_VERSION,
        "dataset": dataset,
        "source_split": split,
        "source_rows": int(rows.size),
        "requests": int(take),
        "synthetic": True,
        "not_real_conversion": True,
        "not_for_effect_claims": True,
        "label_provenance": {
            "click": "KuaiSAR rec_inter.click（真实标签）",
            "conversion": (
                "synthetic：click==1 且 blake2b(user:item:seed) % 100 < cvr_percent；"
                "仅用于验证 MMoE/PLE 多任务输入输出，不能当作真实 CVR"
            ),
            "cvr_percent": int(cvr_percent),
            "seed": int(seed),
        },
        "counts": {
            "rows": int(click.size),
            "clicks": int(click.sum()),
            "conversions": int(conversion.sum()),
            "click_rate": round(float(click.mean()), 6),
            "conversion_rate_on_clicks": round(
                float(conversion.sum() / max(1, int(click.sum()))), 6
            ),
        },
        "fingerprints": fingerprints,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (target / "schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[ok] 写入 {target}：{schema['counts']['rows']} 行 / "
        f"{schema['counts']['clicks']} 点击 / {schema['counts']['conversions']} 模拟转化"
    )
    return schema


def verify_simulated_fixture(target: Path, schema: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for name, expected in (schema.get("fingerprints") or {}).items():
        path = Path(target) / f"{name}.npy"
        if not path.exists():
            problems.append(f"缺少 {name}.npy")
        elif _sha256(path) != expected:
            problems.append(f"{name}.npy 哈希不一致")
    return problems


def load_simulated_fixture(
    dataset: str = "kuasar_small",
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    verify: bool = True,
) -> SimulatedAdView:
    dataset_root = resolve_dataset_data_dir(dataset, Path(data_dir))
    target = simulated_dir(dataset_root)
    schema_path = target / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"{target} 不存在；先运行 python -m recsys.ads.simulated --dataset {dataset}"
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if verify:
        problems = verify_simulated_fixture(target, schema)
        if problems:
            raise ValueError(
                f"模拟转化样本校验失败：{problems[:3]}；重建命令：python -m recsys.ads.simulated --force"
            )
    return SimulatedAdView(
        dataset=dataset,
        users=np.load(target / "users.npy"),
        items=np.load(target / "items.npy"),
        click=np.load(target / "click.npy"),
        conversion=np.load(target / "conversion.npy"),
        groups=np.load(target / "groups.npy"),
        timestamps=np.load(target / "timestamps.npy"),
    )


def simulated_schema(
    dataset: str = "kuasar_small", data_dir: Path = DEFAULT_DATA_DIR
) -> dict[str, Any]:
    dataset_root = resolve_dataset_data_dir(dataset, Path(data_dir))
    path = simulated_dir(dataset_root) / "schema.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在；先运行 python -m recsys.ads.simulated")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", default="kuasar_small")
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--cvr-percent", type=int, default=DEFAULT_CVR_PERCENT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = resolve_dataset_data_dir(args.dataset, args.data_dir)
    target = simulated_dir(dataset_root)
    if args.check:
        schema_path = target / "schema.json"
        if not schema_path.exists():
            print(f"[error] 模拟转化样本不存在：{target}")
            return 2
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        problems = verify_simulated_fixture(target, schema)
        if problems:
            print(f"[fail] {problems[:5]}")
            return 1
        print(f"[ok] 模拟转化样本有效：{schema['counts']}")
        return 0
    schema = build_simulated_fixture(
        args.data_dir,
        dataset=args.dataset,
        split=args.split,
        requests=args.requests,
        cvr_percent=args.cvr_percent,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(schema["counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
