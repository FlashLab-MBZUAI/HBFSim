#!/usr/bin/env python3
"""Build the canonical Qwen-Bailian production-window suite for Frontier.

The adapter stops at the serving-request boundary. Qwen-Bailian records
arrivals, token counts, conversation ancestry, and anonymized 16-token prefix
hashes; it does not contain GPU memory addresses. The output is therefore a
digest-bound set of Frontier request CSVs, not an HBFSim memory trace.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, TextIO


ADAPTER_SCHEMA_NAME = "hbfsim.qwen_bailian_frontier_adapter"
ADAPTER_SCHEMA_VERSION = 2
SUITE_CONFIG_SCHEMA_NAME = "hbfsim.qwen_bailian_frontier_suite_config"
SUITE_CONFIG_SCHEMA_VERSION = 1
SUITE_MANIFEST_SCHEMA_NAME = "hbfsim.qwen_bailian_frontier_suite"
SUITE_MANIFEST_SCHEMA_VERSION = 1
HASH_BLOCK_TOKENS = 16
REQUIRED_WINDOWS = ("steady", "burst", "long_context_decode_tail")
REQUIRED_RUNTIME_BEHAVIORS = (
    "multi_request_batching",
    "prefix_kv_lifecycle",
    "steady_burst_long_context_decode_tail",
)
SELECTOR_ALGORITHMS = {
    "steady": "robust_center_mad_v1",
    "burst": "minimum_arrival_span_v1",
    "long_context_decode_tail": "balanced_input_output_tail_v1",
}
OUTPUT_COLUMNS = (
    "arrived_at",
    "num_prefill_tokens",
    "num_decode_tokens",
    "session_id",
    "block_hash_ids",
    "source_chat_id",
    "parent_chat_id",
    "turn",
    "request_type",
)
EXPECTED_STATISTIC_KEYS = {
    "start_index",
    "end_index_exclusive",
    "source_start_timestamp_s",
    "source_end_timestamp_s",
    "duration_s",
    "prefill_tokens",
    "decode_tokens",
    "total_tokens",
    "max_prefill_tokens",
    "max_decode_tokens",
    "max_total_tokens",
}


class WorkloadError(ValueError):
    """The suite config or Qwen-Bailian source violates its contract."""


class _DuplicateJsonKey(ValueError):
    """A JSON object contains the same key more than once."""


@dataclass(frozen=True)
class RequestRecord:
    chat_id: int
    parent_chat_id: int
    timestamp: Decimal
    input_length: int
    output_length: int
    turn: int
    request_type: str
    hash_ids: tuple[int, ...]
    session_id: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _decode_json(text: str, description: str) -> Any:
    try:
        return json.loads(
            text,
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except UnicodeError as error:
        raise WorkloadError(f"{description} is not valid UTF-8") from error
    except _DuplicateJsonKey as error:
        raise WorkloadError(
            f"{description} contains duplicate key {error.args[0]!r}"
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise WorkloadError(f"invalid JSON in {description}: {error}") from error


def _exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise WorkloadError(f"{path} keys mismatch: missing={missing}, extra={extra}")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkloadError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkloadError(f"{path} must be a non-empty string")
    return value.strip()


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkloadError(f"{path} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise WorkloadError(f"{path} must be boolean")
    return value


def _lower_hex(value: Any, path: str, *, length: int) -> str:
    text = _string(value, path)
    if (
        len(text) != length
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise WorkloadError(
            f"{path} must be {length} lowercase hexadecimal digits"
        )
    return text


def _decimal(value: Any, path: str, *, minimum: Decimal = Decimal(0)) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal, str)):
        raise WorkloadError(f"{path} must be a finite decimal >= {minimum}")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise WorkloadError(
            f"{path} must be a finite decimal >= {minimum}"
        ) from error
    if not result.is_finite() or result < minimum:
        raise WorkloadError(f"{path} must be a finite decimal >= {minimum}")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _load_suite_config(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except OSError as error:
        raise WorkloadError(f"cannot read suite config {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise WorkloadError(f"suite config is not valid UTF-8: {path}") from error
    config = _mapping(_decode_json(text, "suite config"), "suite config")
    _validate_suite_config(config)
    return config, payload


def _validate_suite_config(config: dict[str, Any]) -> None:
    _exact_keys(
        config,
        {
            "schema",
            "suite_id",
            "intended_use",
            "source",
            "frontier_workload",
            "window_policy",
        },
        "suite config",
    )
    schema = _mapping(config["schema"], "schema")
    if schema != {
        "name": SUITE_CONFIG_SCHEMA_NAME,
        "version": SUITE_CONFIG_SCHEMA_VERSION,
    }:
        raise WorkloadError(
            "suite config schema must be "
            f"{SUITE_CONFIG_SCHEMA_NAME} v{SUITE_CONFIG_SCHEMA_VERSION}"
        )
    _string(config["suite_id"], "suite_id")

    intended = _mapping(config["intended_use"], "intended_use")
    _exact_keys(
        intended,
        {"role", "paper_result_eligible_by_itself", "claims_boundary"},
        "intended_use",
    )
    _string(intended["role"], "intended_use.role")
    if _boolean(
        intended["paper_result_eligible_by_itself"],
        "intended_use.paper_result_eligible_by_itself",
    ):
        raise WorkloadError(
            "a request-window suite is not paper-result eligible by itself"
        )
    _string(intended["claims_boundary"], "intended_use.claims_boundary")

    source = _mapping(config["source"], "source")
    _exact_keys(
        source,
        {
            "repository",
            "revision",
            "license",
            "artifact",
            "sha256",
            "bytes",
            "records",
            "hash_block_tokens",
            "statistics",
        },
        "source",
    )
    for key in ("repository", "license", "artifact"):
        _string(source[key], f"source.{key}")
    _lower_hex(source["revision"], "source.revision", length=40)
    _lower_hex(source["sha256"], "source.sha256", length=64)
    _integer(source["bytes"], "source.bytes", minimum=1)
    _integer(source["records"], "source.records", minimum=1)
    if _integer(
        source["hash_block_tokens"], "source.hash_block_tokens", minimum=1
    ) != HASH_BLOCK_TOKENS:
        raise WorkloadError(
            f"source.hash_block_tokens must be {HASH_BLOCK_TOKENS}"
        )
    source_statistics = _mapping(source["statistics"], "source.statistics")
    _exact_keys(
        source_statistics,
        {
            "duration_s",
            "prefill_tokens",
            "decode_tokens",
            "max_prefill_tokens",
            "max_decode_tokens",
        },
        "source.statistics",
    )
    _decimal(source_statistics["duration_s"], "source.statistics.duration_s")
    for key in (
        "prefill_tokens",
        "decode_tokens",
        "max_prefill_tokens",
        "max_decode_tokens",
    ):
        _integer(source_statistics[key], f"source.statistics.{key}", minimum=1)

    workload = _mapping(config["frontier_workload"], "frontier_workload")
    _exact_keys(
        workload,
        {
            "model",
            "primary_precision_profile",
            "sensitivity_precision_profiles",
            "required_runtime_behaviors",
        },
        "frontier_workload",
    )
    model = _mapping(workload["model"], "frontier_workload.model")
    _exact_keys(
        model,
        {"identity", "frontier_name"},
        "frontier_workload.model",
    )
    _string(model["identity"], "frontier_workload.model.identity")
    _string(model["frontier_name"], "frontier_workload.model.frontier_name")
    _string(
        workload["primary_precision_profile"],
        "frontier_workload.primary_precision_profile",
    )
    sensitivities = workload["sensitivity_precision_profiles"]
    if not isinstance(sensitivities, list) or not sensitivities:
        raise WorkloadError(
            "frontier_workload.sensitivity_precision_profiles "
            "must be a non-empty list"
        )
    sensitivity_values = [
        _string(value, f"frontier_workload.sensitivity_precision_profiles[{index}]")
        for index, value in enumerate(sensitivities)
    ]
    if len(set(sensitivity_values)) != len(sensitivity_values):
        raise WorkloadError(
            "frontier_workload.sensitivity_precision_profiles must be unique"
        )
    behaviors = workload["required_runtime_behaviors"]
    if behaviors != list(REQUIRED_RUNTIME_BEHAVIORS):
        raise WorkloadError(
            "frontier_workload.required_runtime_behaviors must equal "
            f"{list(REQUIRED_RUNTIME_BEHAVIORS)}"
        )

    policy = _mapping(config["window_policy"], "window_policy")
    _exact_keys(
        policy,
        {"request_count", "required_windows", "selectors"},
        "window_policy",
    )
    _integer(policy["request_count"], "window_policy.request_count", minimum=2)
    if policy["required_windows"] != list(REQUIRED_WINDOWS):
        raise WorkloadError(
            f"window_policy.required_windows must equal {list(REQUIRED_WINDOWS)}"
        )
    selectors = _mapping(policy["selectors"], "window_policy.selectors")
    if set(selectors) != set(REQUIRED_WINDOWS):
        raise WorkloadError(
            "window_policy.selectors must contain exactly "
            f"{list(REQUIRED_WINDOWS)}"
        )
    for window_id in REQUIRED_WINDOWS:
        selector = _mapping(
            selectors[window_id], f"window_policy.selectors.{window_id}"
        )
        _exact_keys(
            selector,
            {"algorithm", "expected"},
            f"window_policy.selectors.{window_id}",
        )
        if selector["algorithm"] != SELECTOR_ALGORITHMS[window_id]:
            raise WorkloadError(
                f"{window_id} selector algorithm must be "
                f"{SELECTOR_ALGORITHMS[window_id]}"
            )
        expected = _mapping(
            selector["expected"],
            f"window_policy.selectors.{window_id}.expected",
        )
        _exact_keys(
            expected,
            EXPECTED_STATISTIC_KEYS,
            f"window_policy.selectors.{window_id}.expected",
        )
        _integer(expected["start_index"], f"{window_id}.expected.start_index")
        _integer(
            expected["end_index_exclusive"],
            f"{window_id}.expected.end_index_exclusive",
            minimum=1,
        )
        for key in (
            "source_start_timestamp_s",
            "source_end_timestamp_s",
            "duration_s",
        ):
            _decimal(expected[key], f"{window_id}.expected.{key}")
        for key in (
            "prefill_tokens",
            "decode_tokens",
            "total_tokens",
            "max_prefill_tokens",
            "max_decode_tokens",
            "max_total_tokens",
        ):
            _integer(expected[key], f"{window_id}.expected.{key}", minimum=1)


def _require_record_int(
    record: dict[str, Any], key: str, line_no: int, *, minimum: int
) -> int:
    return _integer(record.get(key), f"line {line_no}: {key}", minimum=minimum)


def _read_source(path: Path) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    roots: dict[int, int] = {}
    turns: dict[int, int] = {}
    last_timestamp: Decimal | None = None
    try:
        handle = path.open("rb")
    except OSError as error:
        raise WorkloadError(f"cannot read source trace {path}: {error}") from error
    with handle:
        for line_no, raw_line in enumerate(handle, start=1):
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise WorkloadError(
                    f"line {line_no}: source is not valid UTF-8"
                ) from error
            if not text.strip():
                raise WorkloadError(f"line {line_no}: blank lines are forbidden")
            value = _decode_json(text, f"source line {line_no}")
            record = _mapping(value, f"source line {line_no}")
            _exact_keys(
                record,
                {
                    "chat_id",
                    "parent_chat_id",
                    "timestamp",
                    "input_length",
                    "output_length",
                    "type",
                    "turn",
                    "hash_ids",
                },
                f"source line {line_no}",
            )
            chat_id = _require_record_int(record, "chat_id", line_no, minimum=0)
            parent_chat_id = _require_record_int(
                record, "parent_chat_id", line_no, minimum=-1
            )
            timestamp = _decimal(
                record.get("timestamp"), f"line {line_no}: timestamp"
            )
            input_length = _require_record_int(
                record, "input_length", line_no, minimum=1
            )
            output_length = _require_record_int(
                record, "output_length", line_no, minimum=1
            )
            turn = _require_record_int(record, "turn", line_no, minimum=1)
            request_type = _string(record.get("type"), f"line {line_no}: type")
            hash_values = record.get("hash_ids")
            if not isinstance(hash_values, list):
                raise WorkloadError(f"line {line_no}: hash_ids must be a list")
            hashes = tuple(
                _integer(
                    item,
                    f"line {line_no}: hash_ids[{index}]",
                    minimum=0,
                )
                for index, item in enumerate(hash_values)
            )
            expected_hashes = (
                input_length + HASH_BLOCK_TOKENS - 1
            ) // HASH_BLOCK_TOKENS
            if len(hashes) != expected_hashes:
                raise WorkloadError(
                    f"line {line_no}: hash_ids has {len(hashes)} entries; "
                    f"ceil(input_length/{HASH_BLOCK_TOKENS}) requires "
                    f"{expected_hashes}"
                )
            if chat_id in roots:
                raise WorkloadError(f"line {line_no}: duplicate chat_id {chat_id}")
            if parent_chat_id == -1:
                if turn != 1:
                    raise WorkloadError(
                        f"line {line_no}: a root request must have turn=1"
                    )
                session_id = chat_id
            else:
                if parent_chat_id not in roots:
                    raise WorkloadError(
                        f"line {line_no}: parent_chat_id {parent_chat_id} "
                        "must refer to an earlier record"
                    )
                if turn != turns[parent_chat_id] + 1:
                    raise WorkloadError(
                        f"line {line_no}: turn must equal parent turn + 1"
                    )
                session_id = roots[parent_chat_id]
            if last_timestamp is not None and timestamp < last_timestamp:
                raise WorkloadError(
                    f"line {line_no}: timestamps must be nondecreasing "
                    f"({_decimal_text(timestamp)} < "
                    f"{_decimal_text(last_timestamp)})"
                )
            roots[chat_id] = session_id
            turns[chat_id] = turn
            last_timestamp = timestamp
            records.append(
                RequestRecord(
                    chat_id=chat_id,
                    parent_chat_id=parent_chat_id,
                    timestamp=timestamp,
                    input_length=input_length,
                    output_length=output_length,
                    turn=turn,
                    request_type=request_type,
                    hash_ids=hashes,
                    session_id=session_id,
                )
            )
    if not records:
        raise WorkloadError("source trace contains no records")
    return records


def _median(values: Iterable[Fraction]) -> Fraction:
    ordered = sorted(values)
    if not ordered:
        raise WorkloadError("cannot take the median of an empty population")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _robust_scale(values: list[Fraction], center: Fraction) -> Fraction:
    deviations = [abs(value - center) for value in values]
    median_deviation = _median(deviations)
    if median_deviation:
        return median_deviation
    maximum_deviation = max(deviations)
    return maximum_deviation if maximum_deviation else Fraction(1)


def _window_metrics(
    records: list[RequestRecord], request_count: int
) -> list[tuple[Fraction, int, int]]:
    prefix_prefill = [0]
    prefix_decode = [0]
    for record in records:
        prefix_prefill.append(prefix_prefill[-1] + record.input_length)
        prefix_decode.append(prefix_decode[-1] + record.output_length)
    metrics = []
    for start in range(len(records) - request_count + 1):
        end = start + request_count
        duration = Fraction(records[end - 1].timestamp - records[start].timestamp)
        metrics.append(
            (
                duration,
                prefix_prefill[end] - prefix_prefill[start],
                prefix_decode[end] - prefix_decode[start],
            )
        )
    return metrics


def _select_steady(metrics: list[tuple[Fraction, int, int]]) -> int:
    columns = [
        [Fraction(metric[index]) for metric in metrics] for index in range(3)
    ]
    centers = [_median(column) for column in columns]
    scales = [
        _robust_scale(column, center)
        for column, center in zip(columns, centers, strict=True)
    ]

    def score(start: int) -> tuple[Fraction, Fraction, int]:
        deviations = [
            abs(Fraction(metrics[start][index]) - centers[index]) / scales[index]
            for index in range(3)
        ]
        return max(deviations), sum(deviations, Fraction()), start

    return min(range(len(metrics)), key=score)


def _select_burst(metrics: list[tuple[Fraction, int, int]]) -> int:
    return min(range(len(metrics)), key=lambda start: (metrics[start][0], start))


def _select_balanced_tail(
    records: list[RequestRecord], request_count: int
) -> int:
    global_max_input = max(record.input_length for record in records)
    global_max_output = max(record.output_length for record in records)

    def score(start: int) -> tuple[Fraction, Fraction, int]:
        window = records[start : start + request_count]
        input_share = Fraction(
            max(record.input_length for record in window), global_max_input
        )
        output_share = Fraction(
            max(record.output_length for record in window), global_max_output
        )
        return -min(input_share, output_share), -(input_share + output_share), start

    return min(range(len(records) - request_count + 1), key=score)


def _statistics(
    records: list[RequestRecord], start: int, request_count: int
) -> dict[str, Any]:
    window = records[start : start + request_count]
    prefill = sum(record.input_length for record in window)
    decode = sum(record.output_length for record in window)
    hash_occurrences = sum(len(record.hash_ids) for record in window)
    unique_hashes = {
        hash_id for record in window for hash_id in record.hash_ids
    }
    request_types: dict[str, int] = {}
    for record in window:
        request_types[record.request_type] = (
            request_types.get(record.request_type, 0) + 1
        )
    return {
        "start_index": start,
        "end_index_exclusive": start + request_count,
        "records": request_count,
        "source_start_timestamp_s": _decimal_text(window[0].timestamp),
        "source_end_timestamp_s": _decimal_text(window[-1].timestamp),
        "duration_s": _decimal_text(window[-1].timestamp - window[0].timestamp),
        "prefill_tokens": prefill,
        "decode_tokens": decode,
        "total_tokens": prefill + decode,
        "max_prefill_tokens": max(record.input_length for record in window),
        "max_decode_tokens": max(record.output_length for record in window),
        "max_total_tokens": max(
            record.input_length + record.output_length for record in window
        ),
        "hash_identity_occurrences": hash_occurrences,
        "unique_hash_identities": len(unique_hashes),
        "repeated_hash_identity_occurrences": hash_occurrences - len(unique_hashes),
        "request_types": dict(sorted(request_types.items())),
        "hash_repetition_is_cache_hit_rate": False,
    }


def _selection_statistics(statistics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: statistics[key]
        for key in (
            "start_index",
            "end_index_exclusive",
            "source_start_timestamp_s",
            "source_end_timestamp_s",
            "duration_s",
            "prefill_tokens",
            "decode_tokens",
            "total_tokens",
            "max_prefill_tokens",
            "max_decode_tokens",
            "max_total_tokens",
        )
    }


def _validate_source_identity(
    source: Path,
    records: list[RequestRecord],
    config_source: dict[str, Any],
) -> dict[str, Any]:
    actual = {
        "sha256": _sha256_file(source),
        "bytes": source.stat().st_size,
        "records": len(records),
        "statistics": {
            "duration_s": _decimal_text(
                records[-1].timestamp - records[0].timestamp
            ),
            "prefill_tokens": sum(record.input_length for record in records),
            "decode_tokens": sum(record.output_length for record in records),
            "max_prefill_tokens": max(record.input_length for record in records),
            "max_decode_tokens": max(record.output_length for record in records),
        },
    }
    for key in ("sha256", "bytes", "records", "statistics"):
        if actual[key] != config_source[key]:
            raise WorkloadError(
                f"source {key} mismatch: config={config_source[key]!r}, "
                f"actual={actual[key]!r}"
            )
    return {
        "repository": config_source["repository"],
        "revision": config_source["revision"],
        "license": config_source["license"],
        "artifact": config_source["artifact"],
        "sha256": actual["sha256"],
        "bytes": actual["bytes"],
        "records": actual["records"],
        "hash_block_tokens": HASH_BLOCK_TOKENS,
        "statistics": actual["statistics"],
    }


def _open_text(path: Path) -> TextIO:
    return path.open("w", encoding="utf-8", newline="")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with _open_text(path) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_window_csv(
    path: Path, records: list[RequestRecord], start: int, request_count: int
) -> None:
    window = records[start : start + request_count]
    first_timestamp = window[0].timestamp
    with _open_text(path) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in window:
            writer.writerow(
                {
                    "arrived_at": _decimal_text(record.timestamp - first_timestamp),
                    "num_prefill_tokens": record.input_length,
                    "num_decode_tokens": record.output_length,
                    "session_id": record.session_id,
                    "block_hash_ids": "|".join(map(str, record.hash_ids)),
                    "source_chat_id": record.chat_id,
                    "parent_chat_id": record.parent_chat_id,
                    "turn": record.turn,
                    "request_type": record.request_type,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())


def prepare_suite(
    *,
    source: Path,
    suite_config: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = source.resolve()
    suite_config = suite_config.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file():
        raise WorkloadError(f"source trace does not exist: {source}")
    if not suite_config.is_file():
        raise WorkloadError(f"suite config does not exist: {suite_config}")
    if output_dir.exists():
        raise WorkloadError(f"output directory already exists: {output_dir}")
    if source == suite_config:
        raise WorkloadError("source trace and suite config must be distinct")
    if output_dir in source.parents or output_dir in suite_config.parents:
        raise WorkloadError("output directory must not contain an input")

    config, config_bytes = _load_suite_config(suite_config)
    records = _read_source(source)
    config_source = _mapping(config["source"], "source")
    source_identity = _validate_source_identity(source, records, config_source)
    request_count = int(config["window_policy"]["request_count"])
    if len(records) < request_count:
        raise WorkloadError(
            f"source has {len(records)} records, fewer than request_count="
            f"{request_count}"
        )

    metrics = _window_metrics(records, request_count)
    starts = {
        "steady": _select_steady(metrics),
        "burst": _select_burst(metrics),
        "long_context_decode_tail": _select_balanced_tail(
            records, request_count
        ),
    }
    statistics = {
        window_id: _statistics(records, starts[window_id], request_count)
        for window_id in REQUIRED_WINDOWS
    }
    selectors = config["window_policy"]["selectors"]
    for window_id in REQUIRED_WINDOWS:
        actual = _selection_statistics(statistics[window_id])
        expected = selectors[window_id]["expected"]
        if actual != expected:
            raise WorkloadError(
                f"{window_id} selector drift: expected={expected!r}, "
                f"actual={actual!r}"
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        suite_id = config["suite_id"]
        config_identity = {
            "suite_id": suite_id,
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
            "bytes": len(config_bytes),
        }
        window_artifacts: dict[str, Any] = {}
        for window_id in REQUIRED_WINDOWS:
            csv_name = f"{window_id}.frontier.csv"
            manifest_name = f"{window_id}.adapter-manifest.json"
            csv_path = temporary / csv_name
            manifest_path = temporary / manifest_name
            _write_window_csv(
                csv_path,
                records,
                starts[window_id],
                request_count,
            )
            window_manifest = {
                "schema": {
                    "name": ADAPTER_SCHEMA_NAME,
                    "version": ADAPTER_SCHEMA_VERSION,
                },
                "suite_config": config_identity,
                "adapter_semantics": {
                    "boundary": "serving_request",
                    "output_is_hbfsim_memory_trace": False,
                    "timestamp": (
                        "source seconds normalized to the first selected request; "
                        "Frontier arrived_at remains in seconds"
                    ),
                    "prefill_tokens": "Qwen-Bailian input_length",
                    "decode_tokens": "Qwen-Bailian output_length",
                    "session_id": "root of the parent_chat_id chain",
                    "prefix_identity": (
                        "ordered salted 16-token hash IDs serialized with | for "
                        "Frontier block_hash_ids"
                    ),
                },
                "source": source_identity,
                "frontier_workload": config["frontier_workload"],
                "selection": {
                    "window_id": window_id,
                    "algorithm": SELECTOR_ALGORITHMS[window_id],
                    **_selection_statistics(statistics[window_id]),
                    "records": request_count,
                },
                "output": {
                    "format": "frontier_trace_replay_csv",
                    "path": csv_name,
                    "sha256": _sha256_file(csv_path),
                    "bytes": csv_path.stat().st_size,
                    "columns": list(OUTPUT_COLUMNS),
                },
                "statistics": statistics[window_id],
                "eligibility": {
                    "production_request_window_validated": True,
                    "frontier_replay_attached": False,
                    "paper_result_eligible": False,
                    "reason": (
                        "request provenance alone does not validate Frontier "
                        "execution, memory export, HBFSim replay, or timing"
                    ),
                },
            }
            _write_json(manifest_path, window_manifest)
            window_artifacts[window_id] = {
                "request_csv": {
                    "path": csv_name,
                    "sha256": _sha256_file(csv_path),
                    "bytes": csv_path.stat().st_size,
                },
                "request_manifest": {
                    "path": manifest_name,
                    "sha256": _sha256_file(manifest_path),
                    "bytes": manifest_path.stat().st_size,
                },
                "selection": window_manifest["selection"],
                "statistics": statistics[window_id],
            }

        suite_manifest = {
            "schema": {
                "name": SUITE_MANIFEST_SCHEMA_NAME,
                "version": SUITE_MANIFEST_SCHEMA_VERSION,
            },
            "suite_id": suite_id,
            "suite_config": config_identity,
            "source": source_identity,
            "frontier_workload": config["frontier_workload"],
            "window_policy": {
                "request_count": request_count,
                "required_windows": list(REQUIRED_WINDOWS),
                "selectors": {
                    window_id: SELECTOR_ALGORITHMS[window_id]
                    for window_id in REQUIRED_WINDOWS
                },
            },
            "windows": window_artifacts,
            "eligibility": {
                "production_request_suite_complete": True,
                "all_required_windows_present": True,
                "frontier_replays_attached": False,
                "paper_result_eligible": False,
                "reason": (
                    "the suite is a qualified workload input, not a simulator "
                    "result or timing calibration"
                ),
            },
        }
        _write_json(temporary / "suite-manifest.json", suite_manifest)
        os.replace(temporary, output_dir)
        return suite_manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare_suite(
            source=args.input,
            suite_config=args.suite_config,
            output_dir=args.output_dir,
        )
    except (OSError, WorkloadError) as error:
        parser.error(str(error))
    print(
        f"prepared suite {result['suite_id']}: "
        + ", ".join(
            f"{window_id}=start"
            f"{result['windows'][window_id]['selection']['start_index']}"
            for window_id in REQUIRED_WINDOWS
        )
    )
    print(f"suite_manifest={(args.output_dir / 'suite-manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
