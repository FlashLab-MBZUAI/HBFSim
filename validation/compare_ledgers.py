"""Strict canonical-ledger validation and first-divergence reporting."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from validation.contracts import ContractError, LEDGER_SCHEMA
from validation.ledger_reducer import reduce_ledger


@dataclass(frozen=True)
class Difference:
    record_index: int
    record_id: str
    field_path: str
    expected: Any
    actual: Any
    absolute_difference: float | None = None
    relative_difference: float | None = None

    def format(self, *, case_id: str, replay: str) -> str:
        numeric = ""
        if self.absolute_difference is not None:
            numeric = (
                f"\nabsolute_difference={self.absolute_difference!r}"
                f"\nrelative_difference={self.relative_difference!r}"
            )
        return (
            f"validation ledger mismatch: case={case_id}\n"
            f"record_index={self.record_index} id={self.record_id}\n"
            f"field={self.field_path}\n"
            f"expected={self.expected!r}\n"
            f"actual={self.actual!r}"
            f"{numeric}\n"
            f"replay={replay}"
        )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_ledger(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ContractError(f"{path}: {error}") from error
    if not lines:
        raise ContractError(f"{path}: empty ledger")
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ContractError(f"{path}:{line_number}: blank ledger line")
        try:
            record = json.loads(line, object_pairs_hook=_strict_object)
        except (json.JSONDecodeError, ContractError) as error:
            raise ContractError(f"{path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ContractError(
                f"{path}:{line_number}: record must be an object")
        records.append(record)
    validate_ledger(records, source=str(path))
    return records


def _walk_numbers(value: Any, path: str = "$") -> Iterator[tuple[str, float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_numbers(child, f"{path}[{index}]")


def validate_ledger(
    records: list[dict[str, Any]],
    *,
    source: str = "<memory>",
) -> None:
    seen: set[str] = set()
    seen_kinds: dict[str, str] = {}
    case_id: str | None = None
    for index, record in enumerate(records):
        common = {
            "schema", "record_index", "kind", "id", "parent_id", "case_id"}
        if not common <= record.keys():
            missing = common - record.keys()
            raise ContractError(
                f"{source}: record {index} missing keys {sorted(missing)}")
        if record["schema"] != LEDGER_SCHEMA:
            raise ContractError(
                f"{source}: record {index} has unsupported schema")
        if record["record_index"] != index:
            raise ContractError(
                f"{source}: record index {record['record_index']!r} "
                f"does not equal position {index}")
        if record["kind"] not in {
            "header", "request", "event", "state", "completion", "summary"
        }:
            raise ContractError(
                f"{source}: record {index} has invalid kind")
        fields_by_kind = {
            "header": (
                {"model", "producer"},
                set(),
            ),
            "request": (
                {
                    "request_id", "model", "action", "arrival_ns",
                    "address_space", "addr", "bytes",
                },
                set(),
            ),
            "event": (
                {
                    "request_id", "model", "resource", "category", "action",
                    "start_ns", "finish_ns", "critical",
                },
                {"diagnostic_detail"},
            ),
            "state": (
                {
                    "request_id", "model", "resource", "action", "time_ns",
                    "attributes", "state_hash",
                },
                set(),
            ),
            "completion": (
                {
                    "request_id", "model", "action", "arrival_ns", "start_ns",
                    "finish_ns", "logical_bytes", "physical_bytes", "resource",
                    "result",
                },
                set(),
            ),
            "summary": (
                {
                    "model", "action", "config", "address_observations",
                    "counters",
                },
                set(),
            ),
        }
        required_kind, optional_kind = fields_by_kind[record["kind"]]
        missing_kind = required_kind - record.keys()
        unknown_kind = record.keys() - common - required_kind - optional_kind
        if missing_kind:
            raise ContractError(
                f"{source}: record {index} missing "
                f"{record['kind']} keys {sorted(missing_kind)}")
        if unknown_kind:
            raise ContractError(
                f"{source}: record {index} has unknown "
                f"{record['kind']} keys {sorted(unknown_kind)}")
        record_id = record["id"]
        if not isinstance(record_id, str) or not record_id:
            raise ContractError(
                f"{source}: record {index} has invalid ID")
        if record_id in seen:
            raise ContractError(
                f"{source}: duplicate record ID {record_id!r}")
        parent = record["parent_id"]
        if parent is not None and parent not in seen:
            raise ContractError(
                f"{source}: record {record_id!r} references an absent "
                f"or later parent {parent!r}")
        if record["kind"] in {"event", "state", "completion"} and parent is None:
            raise ContractError(
                f"{source}: {record['kind']} {record_id!r} needs a parent")
        if (
            record["kind"] in {"event", "state", "completion"}
            and parent is not None
            and seen_kinds[parent] != "request"
        ):
            raise ContractError(
                f"{source}: {record['kind']} {record_id!r} parent "
                f"{parent!r} is not a request")
        if index == 0 and (
            record["kind"] != "header" or record_id != "ledger"
        ):
            raise ContractError(
                f"{source}: first record must be header ledger")
        if index and record["kind"] == "header":
            raise ContractError(f"{source}: multiple headers")
        if case_id is None:
            case_id = record["case_id"]
        elif record["case_id"] != case_id:
            raise ContractError(
                f"{source}: record {record_id!r} changes case_id")
        for number_path, number in _walk_numbers(record):
            if not math.isfinite(number):
                raise ContractError(
                    f"{source}: {record_id} {number_path} is not finite")
        if record["kind"] == "event":
            start = record.get("start_ns")
            finish = record.get("finish_ns")
            if (
                isinstance(start, bool)
                or isinstance(finish, bool)
                or not isinstance(start, (int, float))
                or not isinstance(finish, (int, float))
                or start > finish
            ):
                raise ContractError(
                    f"{source}: event {record_id!r} has invalid interval")
        seen.add(record_id)
        seen_kinds[record_id] = record["kind"]
    if records[-1]["kind"] != "summary":
        raise ContractError(f"{source}: final record must be summary")
    reduce_ledger(records, source=source)


def _visible_keys(value: dict[str, Any]) -> set[str]:
    return {
        key for key in value
        if not key.startswith("diagnostic_") and key != "producer"
    }


def _first_value_difference(
    expected: Any,
    actual: Any,
    *,
    path: str,
    abs_tolerance: float,
    rel_tolerance: float,
) -> tuple[str, Any, Any, float | None, float | None] | None:
    if (
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and isinstance(actual, int)
        and not isinstance(actual, bool)
    ):
        if expected != actual:
            return (
                path,
                expected,
                actual,
                float(abs(expected - actual)),
                (
                    abs(expected - actual) / max(abs(expected), abs(actual))
                    if expected or actual
                    else 0.0
                ),
            )
        return None
    if (
        not isinstance(expected, bool)
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and isinstance(actual, (int, float))
    ):
        expected_number = float(expected)
        actual_number = float(actual)
        absolute = abs(expected_number - actual_number)
        scale = max(abs(expected_number), abs(actual_number))
        relative = absolute / scale if scale else 0.0
        if absolute <= max(abs_tolerance, rel_tolerance * scale):
            return None
        return path, expected, actual, absolute, relative
    if type(expected) is not type(actual):
        return path, expected, actual, None, None
    if isinstance(expected, dict):
        expected_keys = _visible_keys(expected)
        actual_keys = _visible_keys(actual)
        if expected_keys != actual_keys:
            return (
                f"{path}.<keys>",
                sorted(expected_keys),
                sorted(actual_keys),
                None,
                None,
            )
        for key in sorted(expected_keys):
            difference = _first_value_difference(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return (
                f"{path}.length",
                len(expected),
                len(actual),
                None,
                None,
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            difference = _first_value_difference(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
            if difference:
                return difference
        return None
    if expected != actual:
        return path, expected, actual, None, None
    return None


def first_difference(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> Difference | None:
    shared = min(len(expected), len(actual))
    for index in range(shared):
        difference = _first_value_difference(
            expected[index],
            actual[index],
            path="$",
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )
        if difference:
            path, expected_value, actual_value, absolute, relative = difference
            return Difference(
                record_index=index,
                record_id=str(
                    expected[index].get(
                        "id", actual[index].get("id", "<unknown>"))),
                field_path=path,
                expected=expected_value,
                actual=actual_value,
                absolute_difference=absolute,
                relative_difference=relative,
            )
    if len(expected) != len(actual):
        index = shared
        return Difference(
            record_index=index,
            record_id=(
                str(expected[index]["id"])
                if index < len(expected)
                else str(actual[index]["id"])
            ),
            field_path="$.record_count",
            expected=len(expected),
            actual=len(actual),
        )
    return None
