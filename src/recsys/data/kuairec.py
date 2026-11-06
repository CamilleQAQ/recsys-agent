"""KuaiRec 2.0 parsing and conversion to the common interaction schema."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import urllib.request
import zipfile
import numpy as np
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from recsys.config import (
    KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS,
    KUAIREC_WATCH_RATIO_THRESHOLD,
)
from recsys.data.schema import (
    IdMappings,
    Interaction,
    PreparedData,
    save_processed_data,
    temporal_train_validation_test_split,
)


KUAIREC_ARCHIVE_URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
KUAIREC_ARCHIVE_MD5 = "261550d472c48eff4990fb13c0e5bcf7"
MATRIX_REQUIRED_COLUMNS = {
    "user_id",
    "video_id",
    "play_duration",
    "video_duration",
    "timestamp",
    "watch_ratio",
}


@dataclass(frozen=True)
class RawKuaiRecInteraction:
    """Fields needed by the first KuaiRec data-stage experiments."""

    original_user_id: int
    original_item_id: int
    play_duration: int
    video_duration: int
    timestamp: float
    watch_ratio: float


def resolve_kuairec_file(data_dir: Path, filename: str) -> Path:
    """Find a KuaiRec CSV under supported official extraction layouts."""
    candidates = (
        data_dir / filename,
        data_dir / "raw" / filename,
        data_dir / "raw" / "KuaiRec 2.0" / "data" / filename,
        data_dir / "KuaiRec 2.0" / "data" / filename,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Checked:\n{attempted}")


def download_and_extract_kuairec(data_dir: Path) -> Path:
    """Download the official archive when needed and return ``big_matrix.csv``."""
    try:
        return resolve_kuairec_file(data_dir, "big_matrix.csv")
    except FileNotFoundError:
        pass

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "KuaiRec.zip"
    if not archive_path.exists():
        temporary_path = archive_path.with_suffix(".zip.part")
        print(f"Downloading KuaiRec from {KUAIREC_ARCHIVE_URL}")
        with urllib.request.urlopen(KUAIREC_ARCHIVE_URL, timeout=300) as response:
            with temporary_path.open("wb") as file:
                shutil.copyfileobj(response, file)
        temporary_path.replace(archive_path)

    digest = hashlib.md5()
    with archive_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    archive_md5 = digest.hexdigest()
    if archive_md5 != KUAIREC_ARCHIVE_MD5:
        raise ValueError(
            f"KuaiRec archive checksum mismatch: {archive_md5} != {KUAIREC_ARCHIVE_MD5}"
        )
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(raw_dir)
    return resolve_kuairec_file(data_dir, "big_matrix.csv")


def iter_matrix_rows(matrix_path: Path) -> Iterator[RawKuaiRecInteraction]:
    """Stream a KuaiRec matrix without materializing the one-gigabyte CSV."""
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = MATRIX_REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{matrix_path} is missing required columns: {sorted(missing)}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                parsed = RawKuaiRecInteraction(
                    original_user_id=int(row["user_id"]),
                    original_item_id=int(row["video_id"]),
                    play_duration=int(row["play_duration"]),
                    video_duration=int(row["video_duration"]),
                    timestamp=float(row["timestamp"]),
                    watch_ratio=float(row["watch_ratio"]),
                )
                if not math.isfinite(parsed.timestamp) or not math.isfinite(parsed.watch_ratio):
                    raise ValueError("timestamp and watch_ratio must be finite")
                yield parsed
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid KuaiRec matrix row at line {line_number}: {error}"
                ) from error


def read_matrix(matrix_path: Path) -> list[RawKuaiRecInteraction]:
    """Read a matrix into memory; intended for tests and small subsets."""
    rows = list(iter_matrix_rows(matrix_path))
    if not rows:
        raise ValueError(f"No interactions found in {matrix_path}")
    return rows


@dataclass(frozen=True)
class PairAggregate:
    """A pair's first exposure and repeat-play audit data."""

    original_user_id: int
    original_item_id: int
    first_timestamp: float
    first_watch_ratio: float
    latest_timestamp: float
    events: int
    positive_events: int
    latest_watch_ratio: float
    ambiguous_first_label: bool


@dataclass(frozen=True)
class LabelAudit:
    """标签定义的可核对统计量；写进审计文件，避免「凭印象说阈值合理」。"""

    raw_rows: int
    rows_kept_as_positive: int
    unique_positive_pairs: int
    duplicate_events_on_positive_pairs: int
    mixed_signal_pairs: int
    negative_only_pairs: int
    latest_event_below_threshold_pairs: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "raw_rows": self.raw_rows,
            "raw_rows_meeting_threshold": self.rows_kept_as_positive,
            "unique_positive_pairs": self.unique_positive_pairs,
            "duplicate_events_on_positive_pairs": (
                self.duplicate_events_on_positive_pairs
            ),
            "mixed_signal_pairs": self.mixed_signal_pairs,
            "negative_only_pairs": self.negative_only_pairs,
            "latest_timestamp_has_below_threshold_event_pairs": (
                self.latest_event_below_threshold_pairs
            ),
            "raw_rows_meeting_threshold_ratio": (
                self.rows_kept_as_positive / self.raw_rows if self.raw_rows else 0.0
            ),
            "replays_per_positive_raw_row": (
                self.duplicate_events_on_positive_pairs / self.rows_kept_as_positive
                if self.rows_kept_as_positive
                else 0.0
            ),
        }


def aggregate_pairs(
    rows: Iterator[RawKuaiRecInteraction] | Sequence[RawKuaiRecInteraction],
    watch_ratio_threshold: float,
) -> tuple[dict[tuple[int, int], PairAggregate], LabelAudit]:
    """Keep first exposure per pair; audit later replays without relabeling it.

    Equal-time first rows with conflicting labels are excluded as ambiguous.
    Input may be in arbitrary row order; timestamps, not CSV order, determine
    the first exposure. The event's own watch ratio and time stay together.
    """
    if not math.isfinite(watch_ratio_threshold) or watch_ratio_threshold < 0:
        raise ValueError("watch_ratio_threshold must be finite and non-negative")
    aggregates: dict[tuple[int, int], PairAggregate] = {}
    raw_rows = 0
    rows_positive = 0

    for row in rows:
        raw_rows += 1
        pair = (row.original_user_id, row.original_item_id)
        is_positive = row.watch_ratio >= watch_ratio_threshold
        if is_positive:
            rows_positive += 1
        previous = aggregates.get(pair)
        if previous is None:
            aggregates[pair] = PairAggregate(
                original_user_id=row.original_user_id,
                original_item_id=row.original_item_id,
                first_timestamp=row.timestamp,
                first_watch_ratio=row.watch_ratio,
                latest_timestamp=row.timestamp,
                events=1,
                positive_events=1 if is_positive else 0,
                latest_watch_ratio=row.watch_ratio,
                ambiguous_first_label=False,
            )
            continue
        first_is_new = row.timestamp < previous.first_timestamp
        same_first_time = row.timestamp == previous.first_timestamp
        first_ratio = (
            row.watch_ratio if first_is_new else
            min(row.watch_ratio, previous.first_watch_ratio) if same_first_time else
            previous.first_watch_ratio
        )
        ambiguous = (False if first_is_new else previous.ambiguous_first_label)
        if same_first_time and is_positive != (previous.first_watch_ratio >= watch_ratio_threshold):
            ambiguous = True
        latest_is_new = row.timestamp > previous.latest_timestamp
        aggregates[pair] = PairAggregate(
            original_user_id=previous.original_user_id,
            original_item_id=previous.original_item_id,
            first_timestamp=min(row.timestamp, previous.first_timestamp),
            first_watch_ratio=first_ratio,
            latest_timestamp=max(row.timestamp, previous.latest_timestamp),
            events=previous.events + 1,
            positive_events=previous.positive_events + int(is_positive),
            latest_watch_ratio=(
                row.watch_ratio if latest_is_new else
                min(row.watch_ratio, previous.latest_watch_ratio)
                if row.timestamp == previous.latest_timestamp else
                previous.latest_watch_ratio
            ),
            ambiguous_first_label=ambiguous,
        )
    positive_pairs = [
        aggregate for aggregate in aggregates.values()
        if aggregate.first_watch_ratio >= watch_ratio_threshold
        and not aggregate.ambiguous_first_label
    ]
    audit = LabelAudit(
        raw_rows=raw_rows,
        rows_kept_as_positive=rows_positive,
        unique_positive_pairs=len(positive_pairs),
        duplicate_events_on_positive_pairs=sum(row.events - 1 for row in positive_pairs),
        mixed_signal_pairs=sum(
            1
            for aggregate in aggregates.values()
            if 0 < aggregate.positive_events < aggregate.events
        ),
        negative_only_pairs=sum(
            1 for aggregate in aggregates.values() if aggregate.positive_events == 0
        ),
        # 这一对整体是正样本，但时间上最后一次播放没达标。
        latest_event_below_threshold_pairs=sum(
            1
            for aggregate in aggregates.values()
            if aggregate.first_watch_ratio >= watch_ratio_threshold
            and aggregate.latest_watch_ratio < watch_ratio_threshold
        ),
    )
    return aggregates, audit


def encode_aggregated_pairs(
    aggregates: dict[tuple[int, int], PairAggregate],
    watch_ratio_threshold: float,
    deduplicate_user_item_pairs: bool = True,
) -> tuple[list[Interaction], IdMappings]:
    """把合并结果编码成 ``Interaction``，并按原 ID 建双向映射。"""
    usable = [
        aggregate
        for aggregate in aggregates.values()
        if aggregate.first_watch_ratio >= watch_ratio_threshold
        and not aggregate.ambiguous_first_label
    ]
    if not usable:
        raise ValueError("No positive interactions remain after watch-ratio filtering")
    if not deduplicate_user_item_pairs:
        raise ValueError(
            "KuaiRec 目前只支持合并后的 (user, video) 正样本；"
            "保留重复行会让留出目标出现在训练历史里，"
            "评测时被「已看过就不推荐」规则过滤掉。"
        )

    original_user_ids = sorted({row.original_user_id for row in usable})
    original_item_ids = sorted({row.original_item_id for row in usable})
    mappings = IdMappings(
        user_to_index={value: index for index, value in enumerate(original_user_ids)},
        index_to_user=original_user_ids,
        item_to_index={value: index for index, value in enumerate(original_item_ids)},
        index_to_item=original_item_ids,
    )
    interactions = [
        Interaction(
            user_idx=mappings.user_to_index[row.original_user_id],
            item_idx=mappings.item_to_index[row.original_item_id],
            timestamp=row.first_timestamp,
        )
        for row in usable
    ]
    interactions.sort(key=lambda row: (row.user_idx, row.timestamp, row.item_idx))
    return interactions, mappings


def encode_positive_interactions(
    rows: Sequence[RawKuaiRecInteraction],
    watch_ratio_threshold: float,
    deduplicate_user_item_pairs: bool = True,
) -> tuple[list[Interaction], IdMappings]:
    """Encode pairs whose unambiguous first playback reaches the threshold."""
    aggregates, _ = aggregate_pairs(rows, watch_ratio_threshold)
    return encode_aggregated_pairs(
        aggregates,
        watch_ratio_threshold,
        deduplicate_user_item_pairs=deduplicate_user_item_pairs,
    )


def prepare_kuairec_from_file(
    matrix_path: Path,
    processed_dir: Path,
    watch_ratio_threshold: float,
    deduplicate_user_item_pairs: bool = True,
    audit_path: Path | None = None,
) -> PreparedData:
    """Parse first exposures, threshold them, split, and save KuaiRec.

    整个 big_matrix（约 1.03 GB、1250 万行）只流式读一遍，原始行不驻留内存；
    只在内存里保留按 ``(user, video)`` 合并后的结果。
    """
    aggregates, audit = aggregate_pairs(
        iter_matrix_rows(matrix_path), watch_ratio_threshold
    )
    interactions, mappings = encode_aggregated_pairs(
        aggregates,
        watch_ratio_threshold,
        deduplicate_user_item_pairs=deduplicate_user_item_pairs,
    )
    train, validation, test = temporal_train_validation_test_split(interactions)
    validation_times = {row.user_idx: row.timestamp for row in validation}
    test_times = {row.user_idx: row.timestamp for row in test}
    validation_seen = np.zeros((len(mappings.index_to_user), len(mappings.index_to_item)), dtype=np.bool_)
    test_seen = np.zeros_like(validation_seen)
    for row in aggregates.values():
        user_idx = mappings.user_to_index.get(row.original_user_id)
        item_idx = mappings.item_to_index.get(row.original_item_id)
        if user_idx is None or item_idx is None:
            continue
        if row.first_timestamp < validation_times.get(user_idx, float("-inf")):
            validation_seen[user_idx, item_idx] = True
        if row.first_timestamp < test_times.get(user_idx, float("-inf")):
            test_seen[user_idx, item_idx] = True
    save_processed_data(
        processed_dir, train, validation, test, mappings,
        protocol_metadata={
            "positive_rule": "first exposure watch_ratio >= threshold",
            "positive_threshold": watch_ratio_threshold,
            "seen_filter": "all playbacks strictly before target time",
        },
    )

    # 标签定义的证据：正样本从多少行来、合并了多少重复、有多少对信号是混的。
    if audit_path is not None:
        payload = {
            "dataset": "KuaiRec 2.0",
            "matrix": str(matrix_path),
            "positive_rule": "first exposure watch_ratio >= threshold (inclusive)",
            "seen_filter": "all first exposures strictly before target time, including non-positive plays",
            "watch_ratio_threshold": watch_ratio_threshold,
            "duplicate_handling": (
                "first exposure per (user, video); later replays excluded"
                if deduplicate_user_item_pairs
                else "keep every raw row"
            ),
            "label_audit": audit.as_dict(),
            "split": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
                "users": len(mappings.index_to_user),
                "items": len(mappings.index_to_item),
            },
        }
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
    return PreparedData(train, validation, test, mappings, validation_seen, test_seen)


def prepare_kuairec(
    data_dir: Path,
    watch_ratio_threshold: float = KUAIREC_WATCH_RATIO_THRESHOLD,
    deduplicate_user_item_pairs: bool = KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS,
    audit_path: Path | None = None,
) -> PreparedData:
    """Prepare first-exposure KuaiRec positives with the configured threshold."""
    matrix_path = download_and_extract_kuairec(data_dir)
    return prepare_kuairec_from_file(
        matrix_path,
        data_dir / "processed",
        watch_ratio_threshold=watch_ratio_threshold,
        deduplicate_user_item_pairs=deduplicate_user_item_pairs,
        audit_path=audit_path,
    )
