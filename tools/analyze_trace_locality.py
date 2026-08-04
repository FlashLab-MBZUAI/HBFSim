#!/usr/bin/env python3
"""Measure spatial and temporal locality in an HBFSim request trace.

The analyzer accepts the same address/op ordering, request-size fields,
semantic-kind aliases, comments, and arrival-time fields as
``scenario_compare``.  It intentionally rejects ambiguous or malformed lines
instead of guessing.  All address intervals are half-open and every request
fragment is charged byte-exactly to each 4 KiB page that it overlaps.

The implementation is exact and deterministic.  Its work is
``O(ops log ops + request-page fragments)``; explicit limits prevent an
accidental enormous request or trace from consuming unbounded host memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NoReturn


UINT64_MAX = (1 << 64) - 1
DEFAULT_LINE_SIZE = 64
DEFAULT_PAGE_SIZE = 4096
DEFAULT_MAX_OPS = 10_000_000
DEFAULT_MAX_PAGE_FRAGMENTS = 50_000_000

SCHEMA_NAME = "hbfsim.trace_locality"
SCHEMA_VERSION = 2

SEMANTIC_KIND_ORDER = (
    "unknown",
    "model_weights",
    "shared_context",
    "generated_context",
    "scratch",
    "metadata",
)

KIND_ALIASES = {
    "": "unknown",
    "unknown": "unknown",
    "none": "unknown",
    "model": "model_weights",
    "weights": "model_weights",
    "model_weight": "model_weights",
    "model_weights": "model_weights",
    "weight": "model_weights",
    "shared": "shared_context",
    "shared_context": "shared_context",
    "shared_kv": "shared_context",
    "prompt_context": "shared_context",
    "prefill_context": "shared_context",
    "generated": "generated_context",
    "generated_context": "generated_context",
    "generated_kv": "generated_context",
    "decode_context": "generated_context",
    "kv_cache": "generated_context",
    "scratch": "scratch",
    "activation": "scratch",
    "activations": "scratch",
    "temp": "scratch",
    "temporary": "scratch",
    "metadata": "metadata",
    "meta": "metadata",
    "page_table": "metadata",
    "mapping": "metadata",
}

OP_ALIASES = {
    "R": "R",
    "READ": "R",
    "LD": "R",
    "LOAD": "R",
    "W": "W",
    "WRITE": "W",
    "ST": "W",
    "STORE": "W",
}


class TraceInputError(ValueError):
    """Raised when the trace cannot be analyzed without guessing."""


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(".", "_")


def _parse_u64(value: str, *, line_no: int, field_name: str) -> int:
    if not value or value.startswith("-"):
        raise TraceInputError(
            f"trace line {line_no}: invalid unsigned {field_name}: {value!r}")
    is_hex = len(value) > 2 and value[:2].lower() == "0x"
    syntax = r"0[xX][0-9a-fA-F]+" if is_hex else r"\+?[0-9]+"
    if re.fullmatch(syntax, value) is None:
        raise TraceInputError(
            f"trace line {line_no}: invalid unsigned {field_name}: {value!r}")
    base = 16 if is_hex else 10
    try:
        parsed = int(value, base)
    except ValueError as error:
        raise TraceInputError(
            f"trace line {line_no}: invalid unsigned {field_name}: {value!r}"
        ) from error
    if parsed < 0 or parsed > UINT64_MAX:
        raise TraceInputError(
            f"trace line {line_no}: {field_name} is outside uint64: {value!r}")
    return parsed


def _try_parse_u64(value: str, *, line_no: int, field_name: str) -> int | None:
    """Return an integer for numeric-looking tokens, otherwise ``None``.

    A malformed numeric-looking token is an error rather than an arbitrary
    label.  This is the fail-closed distinction needed to catch values such as
    ``4096oops``.
    """
    if not value:
        return None
    numeric_looking = value[0].isdigit() or value[0] in "+-"
    if not numeric_looking:
        return None
    return _parse_u64(value, line_no=line_no, field_name=field_name)


def _parse_nonnegative_float(
    value: str, *, line_no: int, field_name: str
) -> float:
    if re.fullmatch(
        r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
        value,
    ) is None:
        raise TraceInputError(
            f"trace line {line_no}: invalid {field_name}: {value!r}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise TraceInputError(
            f"trace line {line_no}: invalid {field_name}: {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise TraceInputError(
            f"trace line {line_no}: invalid {field_name}: {value!r}")
    return parsed


def _parse_arrival(value: str, *, line_no: int) -> float:
    return _parse_nonnegative_float(
        value, line_no=line_no, field_name="arrival time")


@dataclass(frozen=True)
class Request:
    address: int
    op: str
    bytes: int
    kind: str
    line_no: int

    @property
    def end(self) -> int:
        return self.address + self.bytes


def parse_trace_line(
    raw_line: str, *, line_no: int, line_size: int
) -> Request | None:
    """Parse one HBFSim trace line, returning ``None`` for comments/blanks."""
    line = raw_line.split("#", 1)[0].strip()
    if not line:
        return None
    tokens = line.split()
    if len(tokens) < 2:
        raise TraceInputError(
            f"trace line {line_no}: expected <addr> <R|W> or <R|W> <addr>")

    first_op = OP_ALIASES.get(tokens[0].upper())
    second_op = OP_ALIASES.get(tokens[1].upper())
    if second_op is not None:
        address_index, op_index, op = 0, 1, second_op
    elif first_op is not None:
        address_index, op_index, op = 1, 0, first_op
    else:
        raise TraceInputError(
            f"trace line {line_no}: no valid R/W operation token")

    address = _parse_u64(
        tokens[address_index], line_no=line_no, field_name="address")
    request_bytes = line_size
    saw_size = False
    kind = "unknown"
    saw_kind = False
    saw_arrival = False
    saw_label = False
    saw_phase = False
    saw_layer = False
    saw_compute = False

    for index, token in enumerate(tokens):
        if index in (address_index, op_index):
            continue
        key_text, separator, value = token.partition("=")
        key = _normalize_token(key_text) if separator else ""
        value = value if separator else token

        if key in ("bytes", "size"):
            if saw_size:
                raise TraceInputError(
                    f"trace line {line_no}: duplicate request-size field")
            request_bytes = _parse_u64(
                value, line_no=line_no, field_name="request size")
            saw_size = True
            continue

        if key in ("at", "arrival"):
            if saw_arrival:
                raise TraceInputError(
                    f"trace line {line_no}: duplicate arrival-time field")
            _parse_arrival(value, line_no=line_no)
            saw_arrival = True
            continue

        if key in ("label", "region", "name"):
            if saw_label or not value:
                raise TraceInputError(
                    f"trace line {line_no}: invalid or duplicate label field")
            saw_label = True
            continue

        if key == "phase":
            if saw_phase:
                raise TraceInputError(
                    f"trace line {line_no}: duplicate phase field")
            _parse_u64(value, line_no=line_no, field_name="phase id")
            saw_phase = True
            continue

        if key == "layer":
            if saw_layer:
                raise TraceInputError(
                    f"trace line {line_no}: duplicate layer field")
            _parse_u64(value, line_no=line_no, field_name="layer id")
            saw_layer = True
            continue

        if key in ("compute_ns", "compute"):
            if saw_compute:
                raise TraceInputError(
                    f"trace line {line_no}: duplicate compute_ns field")
            _parse_nonnegative_float(
                value, line_no=line_no, field_name="compute_ns")
            saw_compute = True
            continue

        if key in ("kind", "semantic", "type"):
            if saw_kind:
                raise TraceInputError(
                    f"trace line {line_no}: duplicate semantic-kind field")
            canonical = KIND_ALIASES.get(_normalize_token(value))
            if canonical is None:
                raise TraceInputError(
                    f"trace line {line_no}: unknown semantic kind: {value!r}")
            kind = canonical
            saw_kind = True
            continue

        if separator:
            raise TraceInputError(
                f"trace line {line_no}: unsupported field: {key_text!r}")

        numeric = _try_parse_u64(
            value, line_no=line_no, field_name="request size")
        if numeric is not None:
            if saw_size:
                raise TraceInputError(
                    f"trace line {line_no}: duplicate request-size field")
            request_bytes = numeric
            saw_size = True
            continue

        canonical = KIND_ALIASES.get(_normalize_token(value))
        if canonical is not None:
            if saw_kind:
                raise TraceInputError(
                    f"trace line {line_no}: duplicate semantic-kind field")
            kind = canonical
            saw_kind = True
            continue

        # scenario_compare accepts one free-form label.  It does not affect
        # semantic placement, so the locality kind remains ``unknown``.
        if not saw_label:
            saw_label = True
            continue
        raise TraceInputError(
            f"trace line {line_no}: unsupported token: {token!r}")

    if request_bytes == 0:
        raise TraceInputError(
            f"trace line {line_no}: zero-byte memory operation")
    if request_bytes - 1 > UINT64_MAX - address:
        raise TraceInputError(
            f"trace line {line_no}: address range overflows uint64")
    return Request(address, op, request_bytes, kind, line_no)


def _merge_unique_bytes(ranges: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(ranges)
    if not ordered:
        return 0
    total = 0
    begin, end = ordered[0]
    for next_begin, next_end in ordered[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return total + end - begin


def _nearest_rank(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


@dataclass(slots=True)
class _OrderNode:
    key: int
    priority: int
    left: _OrderNode | None = None
    right: _OrderNode | None = None
    size: int = 1


def _node_size(node: _OrderNode | None) -> int:
    return 0 if node is None else node.size


def _refresh_node(node: _OrderNode) -> None:
    node.size = 1 + _node_size(node.left) + _node_size(node.right)


def _priority_for_position(position: int) -> int:
    # SplitMix64 makes the treap shape deterministic and independent of Python
    # hash randomization while retaining expected logarithmic depth.
    value = (position + 0x9E3779B97F4A7C15) & UINT64_MAX
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MAX
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MAX
    return value ^ (value >> 31)


def _rotate_left(node: _OrderNode) -> _OrderNode:
    root = node.right
    if root is None:
        raise AssertionError("left rotation lacks a right child")
    node.right = root.left
    root.left = node
    _refresh_node(node)
    _refresh_node(root)
    return root


def _rotate_right(node: _OrderNode) -> _OrderNode:
    root = node.left
    if root is None:
        raise AssertionError("right rotation lacks a left child")
    node.left = root.right
    root.right = node
    _refresh_node(node)
    _refresh_node(root)
    return root


def _insert_position(
    root: _OrderNode | None,
    node: _OrderNode,
) -> _OrderNode:
    if root is None:
        return node
    if node.key == root.key:
        raise AssertionError("reuse-distance treap inserted a duplicate key")
    if node.key < root.key:
        root.left = _insert_position(root.left, node)
        if root.left.priority < root.priority:
            root = _rotate_right(root)
    else:
        root.right = _insert_position(root.right, node)
        if root.right.priority < root.priority:
            root = _rotate_left(root)
    _refresh_node(root)
    return root


def _erase_position(root: _OrderNode | None, key: int) -> _OrderNode | None:
    if root is None:
        raise AssertionError("reuse-distance treap erased an absent key")
    if key < root.key:
        root.left = _erase_position(root.left, key)
    elif key > root.key:
        root.right = _erase_position(root.right, key)
    elif root.left is None:
        return root.right
    elif root.right is None:
        return root.left
    elif root.left.priority < root.right.priority:
        root = _rotate_right(root)
        root.right = _erase_position(root.right, key)
    else:
        root = _rotate_left(root)
        root.left = _erase_position(root.left, key)
    _refresh_node(root)
    return root


def _count_greater(root: _OrderNode | None, key: int) -> int:
    if root is None:
        return 0
    if root.key <= key:
        return _count_greater(root.right, key)
    return 1 + _node_size(root.right) + _count_greater(root.left, key)


@dataclass
class TemporalPageAccumulator:
    hbm_capacity_pages: int | None
    touches: int = 0
    cold_touches: int = 0
    reuse_touches: int = 0
    reuse_within_hbm_pages: int = 0
    last_position: dict[int, int] = field(default_factory=dict)
    distance_counts: dict[int, int] = field(default_factory=dict)
    root: _OrderNode | None = None
    max_reuse_distance_pages: int | None = None

    def add(self, page: int) -> None:
        position = self.touches
        self.touches += 1
        previous = self.last_position.get(page)
        if previous is None:
            self.cold_touches += 1
        else:
            distance = _count_greater(self.root, previous)
            self.reuse_touches += 1
            self.distance_counts[distance] = (
                self.distance_counts.get(distance, 0) + 1)
            self.max_reuse_distance_pages = (
                distance if self.max_reuse_distance_pages is None
                else max(self.max_reuse_distance_pages, distance))
            if (
                self.hbm_capacity_pages is not None
                and distance < self.hbm_capacity_pages
            ):
                self.reuse_within_hbm_pages += 1
            self.root = _erase_position(self.root, previous)
        self.root = _insert_position(
            self.root,
            _OrderNode(
                key=position,
                priority=_priority_for_position(position),
            ),
        )
        self.last_position[page] = position

    def percentile(self, quantile: float) -> int | None:
        if self.reuse_touches == 0:
            return None
        target = math.ceil(quantile * self.reuse_touches)
        cumulative = 0
        for distance, count in sorted(self.distance_counts.items()):
            cumulative += count
            if cumulative >= target:
                return distance
        raise AssertionError("reuse-distance histogram lost observations")

    def result(self) -> dict[str, object]:
        return {
            "page_touches": self.touches,
            "cold_page_touches": self.cold_touches,
            "reuse_page_touches": self.reuse_touches,
            "reuse_touch_ratio": (
                None if self.touches == 0
                else self.reuse_touches / self.touches),
            "reuse_distance_pages_p50": self.percentile(0.50),
            "reuse_distance_pages_p95": self.percentile(0.95),
            "reuse_distance_pages_p99": self.percentile(0.99),
            "reuse_distance_pages_max": self.max_reuse_distance_pages,
            "hbm_capacity_pages": self.hbm_capacity_pages,
            "reuse_within_hbm_pages": (
                None if self.hbm_capacity_pages is None
                else self.reuse_within_hbm_pages),
            "reuse_within_hbm_ratio": (
                None
                if self.hbm_capacity_pages is None or self.reuse_touches == 0
                else self.reuse_within_hbm_pages / self.reuse_touches),
            "definition": (
                "LRU stack distance: distinct pages touched since the same "
                "page's previous touch; distance < HBM pages is capacity-fit"),
        }


@dataclass
class LocalityAccumulator:
    page_size: int
    ops: int = 0
    reads: int = 0
    writes: int = 0
    total_bytes: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    address_min: int | None = None
    address_max_exclusive: int | None = None
    ranges: list[tuple[int, int]] = field(default_factory=list)
    page_traffic_bytes: dict[int, int] = field(default_factory=dict)
    page_touches: int = 0
    signatures: set[tuple[int, int]] = field(default_factory=set)
    repeated_signatures: int = 0
    sequential_adjacencies: int = 0
    stride_abs_deltas: list[int] = field(default_factory=list)
    previous_address: int | None = None
    previous_end: int | None = None

    def add(
        self,
        request: Request,
        temporal: TemporalPageAccumulator | None = None,
    ) -> None:
        self.ops += 1
        self.total_bytes += request.bytes
        if request.op == "R":
            self.reads += 1
            self.read_bytes += request.bytes
        else:
            self.writes += 1
            self.write_bytes += request.bytes

        self.address_min = (
            request.address if self.address_min is None
            else min(self.address_min, request.address))
        self.address_max_exclusive = (
            request.end if self.address_max_exclusive is None
            else max(self.address_max_exclusive, request.end))
        self.ranges.append((request.address, request.end))

        signature = (request.address, request.bytes)
        if signature in self.signatures:
            self.repeated_signatures += 1
        else:
            self.signatures.add(signature)

        if self.previous_address is not None:
            self.stride_abs_deltas.append(
                abs(request.address - self.previous_address))
            if request.address == self.previous_end:
                self.sequential_adjacencies += 1
        self.previous_address = request.address
        self.previous_end = request.end

        cursor = request.address
        while cursor < request.end:
            page = cursor // self.page_size
            fragment_end = min(request.end, (page + 1) * self.page_size)
            fragment_bytes = fragment_end - cursor
            self.page_traffic_bytes[page] = (
                self.page_traffic_bytes.get(page, 0) + fragment_bytes)
            self.page_touches += 1
            if temporal is not None:
                temporal.add(page)
            cursor = fragment_end

    def result(self) -> dict[str, object]:
        unique_pages = len(self.page_traffic_bytes)
        unique_bytes = _merge_unique_bytes(self.ranges)
        ordered_page_traffic = sorted(
            self.page_traffic_bytes.values(), reverse=True)

        def top_share(fraction: float) -> tuple[int, float | None]:
            if not ordered_page_traffic:
                return 0, None
            count = max(1, math.ceil(fraction * unique_pages))
            return count, sum(ordered_page_traffic[:count]) / self.total_bytes

        top_1_count, top_1_share = top_share(0.01)
        top_10_count, top_10_share = top_share(0.10)
        opportunities = max(0, self.ops - 1)
        max_exclusive = self.address_max_exclusive
        address_min = self.address_min

        return {
            "ops": self.ops,
            "reads": self.reads,
            "writes": self.writes,
            "bytes": {
                "total": self.total_bytes,
                "read": self.read_bytes,
                "write": self.write_bytes,
            },
            "address": {
                "min": address_min,
                "max_inclusive": (
                    None if max_exclusive is None else max_exclusive - 1),
                "max_exclusive": max_exclusive,
                "span_bytes": (
                    0 if address_min is None or max_exclusive is None
                    else max_exclusive - address_min),
            },
            "unique_occupied_bytes": unique_bytes,
            "unique_occupied_pages": unique_pages,
            "page_touches": self.page_touches,
            "reuse_ratio": (
                None if self.page_touches == 0
                else (self.page_touches - unique_pages) / self.page_touches),
            "byte_coverage_reuse_ratio": (
                None if self.total_bytes == 0
                else (self.total_bytes - unique_bytes) / self.total_bytes),
            "exact_address_signature_repeat_ops": self.repeated_signatures,
            "exact_address_signature_repeat_share": (
                None if self.ops == 0 else self.repeated_signatures / self.ops),
            "top_1_percent_pages": top_1_count,
            "top_1_percent_page_traffic_share": top_1_share,
            "top_10_percent_pages": top_10_count,
            "top_10_percent_page_traffic_share": top_10_share,
            "adjacency_opportunities": opportunities,
            "sequential_adjacencies": self.sequential_adjacencies,
            "sequential_adjacency_fraction": (
                None if opportunities == 0
                else self.sequential_adjacencies / opportunities),
            "stride_abs_delta_bytes_p50": _nearest_rank(
                self.stride_abs_deltas, 0.50),
            "stride_abs_delta_bytes_p95": _nearest_rank(
                self.stride_abs_deltas, 0.95),
            "page_touches_per_unique_page": (
                None if unique_pages == 0
                else self.page_touches / unique_pages),
            "address_span_pages": (
                0 if address_min is None or max_exclusive is None
                else (
                    (max_exclusive - 1) // self.page_size
                    - address_min // self.page_size + 1)),
            "address_span_to_occupied_page_ratio": (
                None if unique_pages == 0
                else (
                    (max_exclusive - 1) // self.page_size
                    - address_min // self.page_size + 1) / unique_pages),
        }


def analyze_trace(
    path: Path,
    *,
    line_size: int = DEFAULT_LINE_SIZE,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_ops: int = DEFAULT_MAX_OPS,
    max_page_fragments: int = DEFAULT_MAX_PAGE_FRAGMENTS,
    hbm_capacity_bytes: int | None = None,
) -> dict[str, object]:
    """Analyze ``path`` and return a JSON-serializable deterministic report."""
    if line_size <= 0 or line_size > UINT64_MAX:
        raise TraceInputError("line size must be in [1, UINT64_MAX]")
    if page_size <= 0 or page_size > UINT64_MAX:
        raise TraceInputError("page size must be in [1, UINT64_MAX]")
    if max_ops <= 0 or max_page_fragments <= 0:
        raise TraceInputError("analysis limits must be positive")
    if hbm_capacity_bytes is not None:
        if hbm_capacity_bytes <= 0:
            raise TraceInputError("HBM capacity must be positive when supplied")
        if hbm_capacity_bytes % page_size != 0:
            raise TraceInputError(
                "HBM capacity must be page-size aligned")

    overall = LocalityAccumulator(page_size)
    temporal = TemporalPageAccumulator(
        None if hbm_capacity_bytes is None
        else hbm_capacity_bytes // page_size)
    by_kind = {
        kind: LocalityAccumulator(page_size) for kind in SEMANTIC_KIND_ORDER}
    page_fragments = 0
    digest = hashlib.sha256()

    try:
        handle = path.open("rb")
    except OSError as error:
        raise TraceInputError(f"cannot open trace {path}: {error}") from error

    try:
        with handle:
            for line_no, raw_bytes in enumerate(handle, start=1):
                digest.update(raw_bytes)
                raw_line = raw_bytes.decode("utf-8", errors="strict")
                request = parse_trace_line(
                    raw_line, line_no=line_no, line_size=line_size)
                if request is None:
                    continue
                if overall.ops >= max_ops:
                    raise TraceInputError(
                        f"trace exceeds --max-ops={max_ops} at line {line_no}")
                request_page_fragments = (
                    (request.end - 1) // page_size
                    - request.address // page_size + 1)
                if request_page_fragments > max_page_fragments - page_fragments:
                    raise TraceInputError(
                        "trace exceeds --max-page-fragments="
                        f"{max_page_fragments} at line {line_no}")
                page_fragments += request_page_fragments
                overall.add(request, temporal)
                by_kind[request.kind].add(request)
    except UnicodeError as error:
        raise TraceInputError(f"trace is not valid UTF-8: {error}") from error

    if overall.ops == 0:
        raise TraceInputError("trace contains no memory operations")

    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        # This digest covers exactly the byte stream parsed above; it cannot
        # silently describe a later second read of a changing trace file.
        "trace_sha256": digest.hexdigest(),
        "parameters": {
            "line_size_bytes": line_size,
            "page_size_bytes": page_size,
            "max_ops": max_ops,
            "max_page_fragments": max_page_fragments,
        },
        "overall": overall.result(),
        "temporal_page_locality": temporal.result(),
        "per_kind": {kind: by_kind[kind].result()
                     for kind in SEMANTIC_KIND_ORDER},
        "definitions": {
            "reuse_ratio": (
                "(request-page touches - unique occupied pages) / "
                "request-page touches"),
            "byte_coverage_reuse_ratio": (
                "(request bytes - merged unique occupied bytes) / request bytes"),
            "exact_address_signature": (
                "(request start address, request byte count); operation and kind "
                "are intentionally ignored"),
            "page_traffic_share": (
                "byte-exact traffic in the hottest ceil(N*fraction) unique pages "
                "divided by request bytes"),
            "sequential_adjacency": (
                "current request start equals previous request end; per-kind "
                "results use the kind-filtered request stream"),
            "stride_percentile": (
                "nearest-rank percentile of absolute deltas between consecutive "
                "request start addresses"),
            "address_span_to_occupied_page_ratio": (
                "inclusive page span from minimum to maximum touched address "
                "divided by the number of actually occupied pages"),
        },
    }


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _same_path(left: Path, right: Path) -> bool:
    """Detect normalized aliases, including existing symlinks/hardlinks."""
    try:
        return left.samefile(right)
    except (FileNotFoundError, OSError):
        return left.resolve() == right.resolve()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--line-size", type=_positive_int,
                        default=DEFAULT_LINE_SIZE)
    parser.add_argument("--page-size", type=_positive_int,
                        default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-ops", type=_positive_int,
                        default=DEFAULT_MAX_OPS)
    parser.add_argument("--max-page-fragments", type=_positive_int,
                        default=DEFAULT_MAX_PAGE_FRAGMENTS)
    parser.add_argument("--hbm-capacity-bytes", type=_positive_int)
    args = parser.parse_args()

    # Validate this before reading or creating anything.  In particular, a
    # successful analysis must never atomically replace its own input trace.
    if args.output is not None and _same_path(args.trace, args.output):
        _fail("trace and output paths must be different")

    try:
        report = analyze_trace(
            args.trace,
            line_size=args.line_size,
            page_size=args.page_size,
            max_ops=args.max_ops,
            max_page_fragments=args.max_page_fragments,
            hbm_capacity_bytes=args.hbm_capacity_bytes,
        )
    except TraceInputError as error:
        _fail(str(error))
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        _write_atomic(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
