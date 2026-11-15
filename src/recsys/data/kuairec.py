"""KuaiRec 2.0 parsing and conversion to the common interaction schema.

同一个读取器服务两份矩阵：

- ``big_matrix.csv``（12M 级、稀疏）→ ``prepare_kuairec``；
- ``small_matrix.csv``（1,411 用户 × 3,327 视频的稠密矩阵）→ ``prepare_kuairec_small``。

两者的解析、标签与切分口径完全一致，差异只在矩阵文件名和「建库时的规模校验」。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import urllib.request
import zipfile
import zlib
import numpy as np
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from recsys.config import (
    KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS,
    KUAIREC_SMALL_MATRIX_ROWS,
    KUAIREC_SMALL_MATRIX_USERS,
    KUAIREC_SMALL_POSITIVE_ITEMS,
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
KUAIREC_BIG_MATRIX_FILENAME = "big_matrix.csv"
KUAIREC_SMALL_MATRIX_FILENAME = "small_matrix.csv"
#: small_matrix 在官方 ZIP 里的成员路径（download.json 记录的原始出处）。
KUAIREC_SMALL_ARCHIVE_MEMBER = "KuaiRec 2.0/data/small_matrix.csv"
#: 官方 ZIP 成员解压后的 CRC32，用于确认 small_matrix.csv 不是半截文件。
KUAIREC_SMALL_MATRIX_CRC32 = 77_673_810
#: 建库时要比对的规模：(原始行数, 正样本用户数, 正样本物品数)。
KUAIREC_SMALL_MATRIX_SCALE = (
    KUAIREC_SMALL_MATRIX_ROWS,
    KUAIREC_SMALL_MATRIX_USERS,
    KUAIREC_SMALL_POSITIVE_ITEMS,
)
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


def matrix_crc32(matrix_path: Path) -> int:
    """分块计算矩阵文件的 CRC32（与官方 ZIP 成员记录的校验值对齐）。"""
    checksum = 0
    with matrix_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum


def download_and_extract_kuairec(
    data_dir: Path,
    matrix_filename: str = KUAIREC_BIG_MATRIX_FILENAME,
) -> Path:
    """Download the official archive when needed and return the requested matrix.

    已经存在的那份矩阵直接复用；缺哪份就从官方 ZIP 解压哪份，两份矩阵共用同一个压缩包。
    """
    try:
        return resolve_kuairec_file(data_dir, matrix_filename)
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
    return resolve_kuairec_file(data_dir, matrix_filename)


def iter_matrix_rows(
    matrix_path: Path,
    allow_missing_timestamp: bool = False,
) -> Iterator[RawKuaiRecInteraction]:
    """Stream a KuaiRec matrix without materializing the one-gigabyte CSV.

    ``small_matrix`` 里有 181,992 行缺少 ``timestamp``（占 3.9%）：这些行既无
    先后顺序、也无法参与「目标时刻之前是否曝光过」的判断，只能丢弃。它们恰好
    每行都是独立的 ``(user, video)`` 对、没有带时间的同对记录，因此「丢行」
    等价于「丢对」。``allow_missing_timestamp=True`` 时把这类行解析成
    ``timestamp=nan`` 交给调用方统一记账；默认严格模式（big_matrix 走这条）。
    """
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
                timestamp_text = (row["timestamp"] or "").strip()
                if timestamp_text:
                    timestamp = float(timestamp_text)
                    if not math.isfinite(timestamp):
                        raise ValueError("timestamp must be finite")
                elif allow_missing_timestamp:
                    timestamp = math.nan
                else:
                    raise ValueError("timestamp is empty")
                parsed = RawKuaiRecInteraction(
                    original_user_id=int(row["user_id"]),
                    original_item_id=int(row["video_id"]),
                    play_duration=int(row["play_duration"]),
                    video_duration=int(row["video_duration"]),
                    timestamp=timestamp,
                    watch_ratio=float(row["watch_ratio"]),
                )
                if not math.isfinite(parsed.watch_ratio):
                    raise ValueError("watch_ratio must be finite")
                yield parsed
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid KuaiRec matrix row at line {line_number}: {error}"
                ) from error


def read_matrix(
    matrix_path: Path,
    allow_missing_timestamp: bool = False,
) -> list[RawKuaiRecInteraction]:
    """Read a matrix into memory; intended for tests and small subsets."""
    rows = list(
        iter_matrix_rows(matrix_path, allow_missing_timestamp=allow_missing_timestamp)
    )
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
    rows_missing_timestamp: int
    rows_kept_as_positive: int
    unique_positive_pairs: int
    duplicate_events_on_positive_pairs: int
    mixed_signal_pairs: int
    negative_only_pairs: int
    latest_event_below_threshold_pairs: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "raw_rows": self.raw_rows,
            "rows_missing_timestamp_excluded": self.rows_missing_timestamp,
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
    rows_missing_timestamp = 0

    for row in rows:
        raw_rows += 1
        is_positive = row.watch_ratio >= watch_ratio_threshold
        if is_positive:
            rows_positive += 1
        if not math.isfinite(row.timestamp):
            # 没有时间的行无法定位先后，也进不了 seen_mask：显式丢弃并记账，
            # 不做「猜一个时间」这种会污染 leak 检查的处理。
            rows_missing_timestamp += 1
            continue
        pair = (row.original_user_id, row.original_item_id)
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
        rows_missing_timestamp=rows_missing_timestamp,
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
    expected_scale: tuple[int, int, int] | None = None,
    expected_scale_threshold: float | None = None,
    dataset_label: str = "KuaiRec 2.0",
    allow_missing_timestamp: bool = False,
) -> PreparedData:
    """Parse first exposures, threshold them, split, and save KuaiRec.

    整个 big_matrix（约 1.03 GB、1250 万行）只流式读一遍，原始行不驻留内存；
    只在内存里保留按 ``(user, video)`` 合并后的结果。

    ``expected_scale`` 给出「(原始行数, 正样本用户数, 正样本物品数)」的期望值；
    给出时在切分前核对，用来挡住半截下载或换错矩阵文件。行数与阈值无关，永远校验；
    用户/物品数随阈值变化，只在 ``watch_ratio_threshold == expected_scale_threshold``
    （即登记的正式口径）时校验，避免调阈值做实验时误报。
    """
    aggregates, audit = aggregate_pairs(
        iter_matrix_rows(
            matrix_path, allow_missing_timestamp=allow_missing_timestamp
        ),
        watch_ratio_threshold,
    )
    interactions, mappings = encode_aggregated_pairs(
        aggregates,
        watch_ratio_threshold,
        deduplicate_user_item_pairs=deduplicate_user_item_pairs,
    )
    observed_scale = (
        audit.raw_rows,
        len(mappings.index_to_user),
        len(mappings.index_to_item),
    )
    if expected_scale is not None:
        # 行数与阈值无关；用户/物品数会随阈值变化，只有正式口径下才当固定值核对。
        counts_are_fixed = (
            expected_scale_threshold is None
            or watch_ratio_threshold == expected_scale_threshold
        )
        rows_match = observed_scale[0] == expected_scale[0]
        counts_match = observed_scale[1:] == expected_scale[1:]
        if not rows_match or (counts_are_fixed and not counts_match):
            raise ValueError(
                f"{dataset_label} 规模与登记值不一致："
                f"(行, 用户, 物品) 实际 {observed_scale} ≠ 期望 {expected_scale}。"
                "文件可能被截断或是另一份矩阵；请重新下载后重跑。"
                "（若是刻意改了 --positive-threshold，用户/物品词表本就会变，"
                "请同时更新 configs/datasets 里的登记值。）"
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
            "dataset": dataset_label,
            "matrix": str(matrix_path),
            "positive_rule": "first exposure watch_ratio >= threshold (inclusive)",
            "seen_filter": "all first exposures strictly before target time, including non-positive plays",
            "watch_ratio_threshold": watch_ratio_threshold,
            "duplicate_handling": (
                "first exposure per (user, video); later replays excluded"
                if deduplicate_user_item_pairs
                else "keep every raw row"
            ),
            "observed_scale": {
                "raw_rows": observed_scale[0],
                "positive_users": observed_scale[1],
                "positive_items": observed_scale[2],
            },
            "expected_scale": (
                {
                    "raw_rows": expected_scale[0],
                    "positive_users": expected_scale[1],
                    "positive_items": expected_scale[2],
                }
                if expected_scale is not None
                else None
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
    """Prepare first-exposure KuaiRec big_matrix positives with the configured threshold."""
    matrix_path = download_and_extract_kuairec(data_dir, KUAIREC_BIG_MATRIX_FILENAME)
    return prepare_kuairec_from_file(
        matrix_path,
        data_dir / "processed",
        watch_ratio_threshold=watch_ratio_threshold,
        deduplicate_user_item_pairs=deduplicate_user_item_pairs,
        audit_path=audit_path,
        dataset_label="KuaiRec 2.0 big_matrix",
    )


def prepare_kuairec_small(
    data_dir: Path,
    watch_ratio_threshold: float = KUAIREC_WATCH_RATIO_THRESHOLD,
    deduplicate_user_item_pairs: bool = KUAIREC_DEDUPLICATE_USER_ITEM_PAIRS,
    audit_path: Path | None = None,
    verify_checksum: bool = True,
) -> PreparedData:
    """Prepare the dense KuaiRec ``small_matrix`` with the same labels and split.

    与 big_matrix 的差别只有三处：矩阵文件、建库前的规模校验、CRC32 校验。
    审计文件默认写到 ``processed/label_audit.json``，让「正样本是从多少行来的」
    跟着数据目录走，报告里可以直接引用而不必重算。
    """
    matrix_path = download_and_extract_kuairec(
        data_dir, KUAIREC_SMALL_MATRIX_FILENAME
    )
    if verify_checksum:
        checksum = matrix_crc32(matrix_path)
        if checksum != KUAIREC_SMALL_MATRIX_CRC32:
            raise ValueError(
                f"{matrix_path} 的 CRC32 校验失败：{checksum} ≠ {KUAIREC_SMALL_MATRIX_CRC32}。"
                f"官方 ZIP 成员为 {KUAIREC_SMALL_ARCHIVE_MEMBER}，请重新下载后重跑。"
            )
    return prepare_kuairec_from_file(
        matrix_path,
        data_dir / "processed",
        watch_ratio_threshold=watch_ratio_threshold,
        deduplicate_user_item_pairs=deduplicate_user_item_pairs,
        audit_path=audit_path or (data_dir / "processed" / "label_audit.json"),
        expected_scale=KUAIREC_SMALL_MATRIX_SCALE,
        expected_scale_threshold=KUAIREC_WATCH_RATIO_THRESHOLD,
        dataset_label="KuaiRec 2.0 small_matrix",
        # small_matrix 有 181,992 行缺 timestamp（3.9%），big_matrix 没有；
        # 这些行单独记账后丢弃，而不是让整份数据解析失败。
        allow_missing_timestamp=True,
    )
