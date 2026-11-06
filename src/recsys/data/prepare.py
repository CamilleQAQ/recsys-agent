"""Prepare either supported dataset using the common interaction schema."""

from __future__ import annotations

import argparse

from recsys.config import DEFAULT_DATASET, DEFAULT_DATA_DIR
from recsys.data.loader import SUPPORTED_DATASETS, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--positive-threshold",
        type=float,
        help="positive threshold for the selected dataset (KuaiRec: watch_ratio)",
    )
    args = parser.parse_args()
    from pathlib import Path

    prepared = prepare_dataset(
        args.dataset, Path(args.data_dir), positive_threshold=args.positive_threshold
    )
    print(
        f"dataset={args.dataset} users={prepared.num_users} items={prepared.num_items} "
        f"train={len(prepared.train_interactions)} "
        f"validation={len(prepared.validation_interactions)} "
        f"test={len(prepared.test_interactions)}"
    )


if __name__ == "__main__":
    main()
