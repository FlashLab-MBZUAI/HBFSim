#!/usr/bin/env python3
"""Render a logical-address heatmap directly from an HBFSim text trace.

The output is one self-contained HTML file with two views of the same selected
traffic: the full configured address capacity and a zoom over the observed
working-set span.  Both views use exactly ``--bins`` bins.  Requests crossing
bin boundaries are split by byte overlap, so every input byte is conserved.

Examples::

    python3 tools/plot_trace_heatmap.py \
        --trace workload.trace --capacity-bytes 0x10000000000 \
        --bins 1024 --output workload.logical.html

    python3 tools/plot_trace_heatmap.py \
        --trace workload.trace --capacity-bytes 4_398_046_511_104 \
        --kind model_weights --kind shared_context -o static.logical.html

Trace syntax is a strict, auditable subset of ``scenario_compare``:
``<addr> <R|W>`` or ``<R|W> <addr>``, followed by optional bytes, semantic
kind, one label, and ``at=<ns>`` fields.  Duplicate fields, unknown keyed
fields, and malformed numeric-looking tokens are rejected.  Two-token records
use ``--line-size`` bytes.
"""

from __future__ import annotations

import argparse
import html
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, NoReturn, Sequence


UINT64_MAX = (1 << 64) - 1
ADDRESS_SPACE_SIZE = 1 << 64
MAX_BINS = 8192

KINDS = (
    "unknown",
    "model_weights",
    "shared_context",
    "generated_context",
    "scratch",
    "metadata",
)
KIND_INDEX = {kind: index for index, kind in enumerate(KINDS)}

KIND_ALIASES = {
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

READ_TOKENS = frozenset(("R", "READ", "LD", "LOAD"))
WRITE_TOKENS = frozenset(("W", "WRITE", "ST", "STORE"))

READ_COLORS = {
    "unknown": "#2563eb",
    "model_weights": "#1d4ed8",
    "shared_context": "#0284c7",
    "generated_context": "#0891b2",
    "scratch": "#4f46e5",
    "metadata": "#7c3aed",
}
WRITE_COLORS = {
    "unknown": "#ea580c",
    "model_weights": "#dc2626",
    "shared_context": "#e11d48",
    "generated_context": "#f97316",
    "scratch": "#d97706",
    "metadata": "#be123c",
}

_DECIMAL = re.compile(r"\+?[0-9]+\Z")
_HEX = re.compile(r"\+?0[xX][0-9a-fA-F]+\Z")
_ARRIVAL = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


class TraceHeatmapError(ValueError):
    """A trace or plotting option violates the logical-heatmap contract."""


def _fail(message: str) -> NoReturn:
    raise TraceHeatmapError(message)


@dataclass(frozen=True)
class Operation:
    addr: int
    end: int
    op: str
    kind: str
    line_no: int

    @property
    def bytes(self) -> int:
        return self.end - self.addr


@dataclass
class DirectionTotals:
    read_bytes: int = 0
    write_bytes: int = 0
    read_accesses: int = 0
    write_accesses: int = 0

    def add(self, operation: Operation, context: str) -> None:
        if operation.op == "read":
            self.read_bytes = _checked_add(
                self.read_bytes, operation.bytes, f"{context} read bytes")
            self.read_accesses = _checked_add(
                self.read_accesses, 1, f"{context} read accesses")
        else:
            self.write_bytes = _checked_add(
                self.write_bytes, operation.bytes, f"{context} write bytes")
            self.write_accesses = _checked_add(
                self.write_accesses, 1, f"{context} write accesses")


@dataclass
class TraceSummary:
    parsed_operations: int = 0
    selected_operations: int = 0
    observed_begin: int | None = None
    observed_end: int | None = None
    totals: DirectionTotals = field(default_factory=DirectionTotals)
    per_kind: dict[str, DirectionTotals] = field(
        default_factory=lambda: {kind: DirectionTotals() for kind in KINDS})

    def count_parsed(self) -> None:
        self.parsed_operations = _checked_add(
            self.parsed_operations, 1, "parsed operation count")

    def add_selected(self, operation: Operation) -> None:
        self.selected_operations = _checked_add(
            self.selected_operations, 1, "selected operation count")
        self.observed_begin = (
            operation.addr if self.observed_begin is None
            else min(self.observed_begin, operation.addr))
        self.observed_end = (
            operation.end if self.observed_end is None
            else max(self.observed_end, operation.end))
        self.totals.add(operation, "selected traffic")
        self.per_kind[operation.kind].add(
            operation, f"{operation.kind} traffic")


@dataclass
class BinnedView:
    name: str
    begin: int
    end: int
    bin_count: int
    read: list[list[int]] = field(init=False)
    write: list[list[int]] = field(init=False)

    def __post_init__(self) -> None:
        if self.begin < 0 or self.end <= self.begin:
            _fail(f"{self.name}: invalid half-open view range")
        self.read = [[0] * self.bin_count for _ in KINDS]
        self.write = [[0] * self.bin_count for _ in KINDS]

    @property
    def extent(self) -> int:
        return self.end - self.begin

    def boundary(self, index: int) -> int:
        """Return ceil(index * extent / bins), offset by view begin."""
        if index == 0:
            return self.begin
        return self.begin + (
            index * self.extent + self.bin_count - 1) // self.bin_count

    def add(self, operation: Operation) -> None:
        if operation.addr < self.begin or operation.end > self.end:
            _fail(
                f"trace line {operation.line_no}: range "
                f"[{operation.addr:#x}, {operation.end:#x}) falls outside "
                f"{self.name} [{self.begin:#x}, {self.end:#x})")

        local_begin = operation.addr - self.begin
        local_last = operation.end - 1 - self.begin
        first_bin = local_begin * self.bin_count // self.extent
        last_bin = local_last * self.bin_count // self.extent
        kind_index = KIND_INDEX[operation.kind]
        target = self.read if operation.op == "read" else self.write
        allocated = 0
        for bin_index in range(first_bin, last_bin + 1):
            bin_begin = self.boundary(bin_index)
            bin_end = self.boundary(bin_index + 1)
            overlap = min(operation.end, bin_end) - max(operation.addr, bin_begin)
            if overlap <= 0:
                continue
            target[kind_index][bin_index] = _checked_add(
                target[kind_index][bin_index], overlap,
                f"{self.name} bin byte count")
            allocated += overlap
        if allocated != operation.bytes:
            _fail(
                f"internal {self.name} accounting error on trace line "
                f"{operation.line_no}: allocated {allocated} of "
                f"{operation.bytes} bytes")

    def direction_totals(self, direction: str) -> list[int]:
        values = self.read if direction == "read" else self.write
        return [sum(values[kind][index] for kind in range(len(KINDS)))
                for index in range(self.bin_count)]

    def occupied_bins(self) -> int:
        reads = self.direction_totals("read")
        writes = self.direction_totals("write")
        return sum(1 for read, write in zip(reads, writes) if read or write)


@dataclass(frozen=True)
class TraceAnalysis:
    summary: TraceSummary
    full: BinnedView
    zoom: BinnedView
    selected_kinds: tuple[str, ...]


@dataclass(frozen=True)
class BinTrafficStats:
    """Distribution of byte-overlap traffic across one complete bin view."""

    bin_count: int
    occupied_bins: int
    minimum_bytes: int
    total_bytes: int
    maximum_bytes: int
    population_cv: float | None


def _checked_add(current: int, increment: int, context: str) -> int:
    result = current + increment
    if result > UINT64_MAX:
        _fail(f"{context} overflows uint64")
    return result


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(".", "_")


def canonical_kind(value: str) -> str | None:
    return KIND_ALIASES.get(_normalize(value))


def _parse_uint(token: str, context: str, maximum: int = UINT64_MAX) -> int:
    if not (_DECIMAL.fullmatch(token) or _HEX.fullmatch(token)):
        _fail(f"{context}: invalid non-negative integer {token!r}")
    try:
        value = int(token, 16 if token.lstrip("+").lower().startswith("0x") else 10)
    except ValueError as error:  # Defensive: the regular expressions are stricter.
        raise TraceHeatmapError(
            f"{context}: invalid non-negative integer {token!r}") from error
    if value > maximum:
        _fail(f"{context}: value {value} exceeds maximum {maximum}")
    return value


def _parse_op(token: str) -> str | None:
    upper = token.upper()
    if upper in READ_TOKENS:
        return "read"
    if upper in WRITE_TOKENS:
        return "write"
    return None


def _try_parse_uint(token: str, context: str) -> int | None:
    """Parse numeric-looking bare fields, rejecting malformed spellings."""
    if not token:
        return None
    if not (token[0].isdigit() or token[0] in "+-"):
        return None
    return _parse_uint(token, context)


def _parse_arrival(token: str, line_no: int) -> None:
    if _ARRIVAL.fullmatch(token) is None:
        _fail(f"trace line {line_no}: invalid arrival time {token!r}")
    try:
        arrival = float(token)
    except ValueError:  # Defensive: the regular expression is stricter.
        arrival = math.nan
    if not math.isfinite(arrival) or arrival < 0:
        _fail(f"trace line {line_no}: invalid arrival time {token!r}")


def parse_trace_line(line: str, line_no: int, line_size: int) -> Operation | None:
    """Parse one line from the strict, auditable trace-grammar subset."""
    tokens = line.split("#", 1)[0].split()
    if not tokens:
        return None
    if len(tokens) < 2:
        _fail(
            f"trace line {line_no}: expected Ramulator-compatible "
            "<addr> <R|W>")

    second_op = _parse_op(tokens[1])
    first_op = _parse_op(tokens[0])
    if second_op is not None:
        operation = second_op
        addr_index, op_index = 0, 1
    elif first_op is not None:
        operation = first_op
        addr_index, op_index = 1, 0
    else:
        _fail(f"trace line {line_no}: has no R/W operation token")

    addr = _parse_uint(tokens[addr_index], f"trace line {line_no} address")
    byte_count = line_size
    kind = "unknown"
    saw_size = False
    saw_kind = False
    saw_arrival = False
    saw_label = False
    for index, token in enumerate(tokens):
        if index in (addr_index, op_index):
            continue
        key, separator, value = token.partition("=")
        normalized_key = _normalize(key) if separator else ""
        field_value = value if separator else token

        if normalized_key in ("bytes", "size"):
            if saw_size:
                _fail(f"trace line {line_no}: duplicate request-size field")
            byte_count = _parse_uint(
                field_value, f"trace line {line_no} byte count")
            saw_size = True
            continue

        if normalized_key in ("at", "arrival"):
            if saw_arrival:
                _fail(f"trace line {line_no}: duplicate arrival-time field")
            _parse_arrival(field_value, line_no)
            saw_arrival = True
            continue

        if normalized_key in ("label", "region", "name"):
            if saw_label or not field_value:
                _fail(f"trace line {line_no}: invalid or duplicate label field")
            saw_label = True
            continue

        if normalized_key in ("kind", "semantic", "type"):
            if saw_kind:
                _fail(f"trace line {line_no}: duplicate semantic-kind field")
            parsed_kind = canonical_kind(field_value)
            if parsed_kind is None:
                _fail(
                    f"trace line {line_no}: unknown semantic kind "
                    f"{field_value!r}")
            kind = parsed_kind
            saw_kind = True
            continue

        if separator:
            _fail(
                f"trace line {line_no}: unsupported field {key!r}")

        parsed_bytes = _try_parse_uint(
            field_value, f"trace line {line_no} byte count")
        if parsed_bytes is not None:
            if saw_size:
                _fail(f"trace line {line_no}: duplicate request-size field")
            byte_count = parsed_bytes
            saw_size = True
            continue

        parsed_kind = canonical_kind(field_value)
        if parsed_kind is not None:
            if saw_kind:
                _fail(f"trace line {line_no}: duplicate semantic-kind field")
            kind = parsed_kind
            saw_kind = True
            continue

        # One bare, non-semantic token is the optional free-form label.
        if not saw_label:
            saw_label = True
            continue
        _fail(f"trace line {line_no}: unsupported token {token!r}")

    if byte_count == 0:
        _fail(f"trace line {line_no}: zero-byte memory operation")
    if byte_count - 1 > UINT64_MAX - addr:
        _fail(f"trace line {line_no}: address range overflows uint64")
    return Operation(
        addr=addr,
        end=addr + byte_count,
        op=operation,
        kind=kind,
        line_no=line_no,
    )


def iter_operations(path: Path, line_size: int) -> Iterator[Operation]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                operation = parse_trace_line(line, line_no, line_size)
                if operation is not None:
                    yield operation
    except OSError as error:
        _fail(f"cannot read trace {path}: {error}")
    except UnicodeError as error:
        _fail(f"trace {path} is not valid UTF-8: {error}")


def _conservation_check(view: BinnedView, summary: TraceSummary) -> None:
    read_bytes = sum(sum(kind_bins) for kind_bins in view.read)
    write_bytes = sum(sum(kind_bins) for kind_bins in view.write)
    if read_bytes != summary.totals.read_bytes:
        _fail(
            f"internal {view.name} conservation error: binned read bytes "
            f"{read_bytes} != selected read bytes {summary.totals.read_bytes}")
    if write_bytes != summary.totals.write_bytes:
        _fail(
            f"internal {view.name} conservation error: binned write bytes "
            f"{write_bytes} != selected write bytes {summary.totals.write_bytes}")
    for kind_index, kind in enumerate(KINDS):
        if sum(view.read[kind_index]) != summary.per_kind[kind].read_bytes:
            _fail(f"internal {view.name} {kind} read conservation error")
        if sum(view.write[kind_index]) != summary.per_kind[kind].write_bytes:
            _fail(f"internal {view.name} {kind} write conservation error")


def analyze_trace(
    path: Path,
    capacity_bytes: int,
    bin_count: int,
    line_size: int = 64,
    selected_kinds: Sequence[str] | None = None,
) -> TraceAnalysis:
    """Parse and bin one trace without retaining its request stream in memory."""
    if not 0 < capacity_bytes <= ADDRESS_SPACE_SIZE:
        _fail(
            f"capacity bytes must be in [1, {ADDRESS_SPACE_SIZE}], got "
            f"{capacity_bytes}")
    if not 1 <= bin_count <= MAX_BINS:
        _fail(f"bin count must be in [1, {MAX_BINS}], got {bin_count}")
    if not 1 <= line_size <= UINT64_MAX:
        _fail(f"line size must be in [1, {UINT64_MAX}], got {line_size}")

    if selected_kinds is None:
        selected = tuple(KINDS)
    else:
        invalid = [kind for kind in selected_kinds if kind not in KIND_INDEX]
        if invalid:
            _fail(f"unknown selected semantic kind {invalid[0]!r}")
        selected = tuple(dict.fromkeys(selected_kinds))
        if not selected:
            _fail("at least one semantic kind must be selected")
    selected_set = set(selected)

    before = _file_identity(path)
    summary = TraceSummary()
    full = BinnedView("full-capacity view", 0, capacity_bytes, bin_count)
    for operation in iter_operations(path, line_size):
        summary.count_parsed()
        if operation.kind not in selected_set:
            continue
        if operation.end > capacity_bytes:
            _fail(
                f"trace line {operation.line_no}: selected range "
                f"[{operation.addr:#x}, {operation.end:#x}) exceeds capacity "
                f"[0x0, {capacity_bytes:#x})")
        summary.add_selected(operation)
        full.add(operation)

    if summary.parsed_operations == 0:
        _fail("trace contains no memory operations")
    if summary.selected_operations == 0:
        _fail(
            "trace contains no operations matching selected kinds: "
            + ", ".join(selected))
    assert summary.observed_begin is not None and summary.observed_end is not None
    zoom = BinnedView(
        "observed-working-set zoom",
        summary.observed_begin,
        summary.observed_end,
        bin_count,
    )

    second_parsed = 0
    second_selected = DirectionTotals()
    second_selected_operations = 0
    for operation in iter_operations(path, line_size):
        second_parsed = _checked_add(
            second_parsed, 1, "second-pass operation count")
        if operation.kind not in selected_set:
            continue
        second_selected_operations = _checked_add(
            second_selected_operations, 1, "second-pass selected count")
        second_selected.add(operation, "second-pass selected traffic")
        zoom.add(operation)
    after = _file_identity(path)
    if before != after or second_parsed != summary.parsed_operations:
        _fail("trace changed while it was being analyzed")
    if (second_selected_operations != summary.selected_operations or
            second_selected != summary.totals):
        _fail("trace selected traffic changed between analysis passes")

    _conservation_check(full, summary)
    _conservation_check(zoom, summary)
    return TraceAnalysis(summary, full, zoom, selected)


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        _fail(f"cannot stat trace {path}: {error}")
    if not path.is_file():
        _fail(f"trace path is not a regular file: {path}")
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")
    amount = float(value)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(amount) < 1024.0 or candidate == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{value:,} B"
    return f"{amount:.3f} {unit} ({value:,} B)"


def _format_percent(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.6f}%"


def _bin_traffic_stats(values: Sequence[int]) -> BinTrafficStats:
    """Return all-bin traffic statistics, including zero-valued bins."""
    if not values:
        _fail("cannot summarize an empty bin view")
    total = sum(values)
    population_cv: float | None = None
    if total != 0:
        # This algebraic form keeps the sums exact until the final ratio:
        # CV^2 = N * sum(x_i^2) / sum(x_i)^2 - 1.
        cv_squared = (
            len(values) * sum(value * value for value in values)
            / (total * total) - 1.0
        )
        population_cv = math.sqrt(max(0.0, cv_squared))
    return BinTrafficStats(
        bin_count=len(values),
        occupied_bins=sum(value != 0 for value in values),
        minimum_bytes=min(values),
        total_bytes=total,
        maximum_bytes=max(values),
        population_cv=population_cv,
    )


def _format_mean_bytes(total: int, bin_count: int) -> str:
    quotient, remainder = divmod(total, bin_count)
    if remainder == 0:
        return _format_bytes(quotient)

    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")
    mean_bytes = total / bin_count
    amount = mean_bytes
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(amount) < 1024.0 or candidate == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{mean_bytes:,.3f} B"
    return f"{amount:.3f} {unit} ({mean_bytes:,.3f} B)"


def _svg_view(view: BinnedView, description: str) -> str:
    width = 920.0
    height = 386.0
    plot_x = 54.0
    plot_width = 812.0
    baseline = 188.0
    half_height = 108.0
    reads = view.direction_totals("read")
    writes = view.direction_totals("write")
    maximum = max((*reads, *writes), default=0)
    log_maximum = math.log1p(maximum) if maximum else 1.0
    bin_width = plot_width / view.bin_count

    parts = [
        (f'<svg viewBox="0 0 {width:g} {height:g}" role="img" '
         f'aria-label="{html.escape(view.name)}">'),
        f"<title>{html.escape(view.name)}</title>",
        f'<text x="{plot_x:g}" y="27" class="svg-title">'
        f'{html.escape(view.name)}</text>',
        f'<text x="{plot_x:g}" y="50" class="svg-note">'
        f'{html.escape(description)}</text>',
        f'<rect x="{plot_x:g}" y="{baseline - half_height:g}" '
        f'width="{plot_width:g}" height="{2 * half_height:g}" '
        'class="plot-bg"/>',
        f'<line x1="{plot_x:g}" y1="{baseline:g}" '
        f'x2="{plot_x + plot_width:g}" y2="{baseline:g}" '
        'class="baseline"/>',
        f'<text x="12" y="102" class="direction read-label">R</text>',
        f'<text x="12" y="286" class="direction write-label">W</text>',
    ]

    for index in range(view.bin_count):
        if not reads[index] and not writes[index]:
            continue
        x = plot_x + index * bin_width
        bin_begin = view.boundary(index)
        bin_end = view.boundary(index + 1)
        for direction, total, values, colors, sign in (
            ("read", reads[index], view.read, READ_COLORS, -1),
            ("write", writes[index], view.write, WRITE_COLORS, 1),
        ):
            if total == 0:
                continue
            total_height = math.log1p(total) / log_maximum * half_height
            cursor = baseline
            for kind_index, kind in enumerate(KINDS):
                value = values[kind_index][index]
                if value == 0:
                    continue
                segment_height = total_height * value / total
                y = cursor - segment_height if sign < 0 else cursor
                parts.append(
                    f'<rect x="{x:.6f}" y="{y:.6f}" '
                    f'width="{bin_width:.6f}" height="{segment_height:.6f}" '
                    f'fill="{colors[kind]}"><title>'
                    f'bin {index} [{bin_begin:#x}, {bin_end:#x}); '
                    f'{direction} {kind}: {value:,} B; '
                    f'{direction} bin total: {total:,} B'
                    f'</title></rect>')
                cursor = y if sign < 0 else cursor + segment_height

    for tick in range(5):
        fraction = tick / 4
        x = plot_x + fraction * plot_width
        address = view.begin + view.extent * tick // 4
        anchor = "start" if tick == 0 else "end" if tick == 4 else "middle"
        parts.extend((
            f'<line x1="{x:.3f}" y1="{baseline - half_height:g}" '
            f'x2="{x:.3f}" y2="{baseline + half_height:g}" class="grid"/>',
            f'<text x="{x:.3f}" y="326" text-anchor="{anchor}" '
            f'class="tick">{address:#x}</text>',
        ))
    parts.extend((
        f'<text x="{plot_x:g}" y="353" class="svg-note">'
        f'Range [{view.begin:#x}, {view.end:#x}) · {_format_bytes(view.extent)}'
        f'</text>',
        f'<text x="{plot_x:g}" y="374" class="svg-note">'
        f'{view.occupied_bins():,}/{view.bin_count:,} occupied byte-overlap '
        'traffic bins · '
        'log1p height; kind segments retain byte proportions</text>',
        "</svg>",
    ))
    return "\n".join(parts)


def _legend(analysis: TraceAnalysis) -> str:
    rows = []
    for kind in analysis.selected_kinds:
        totals = analysis.summary.per_kind[kind]
        if not (totals.read_accesses or totals.write_accesses):
            continue
        rows.append(
            '<tr>'
            f'<th scope="row">{html.escape(kind)}</th>'
            f'<td><span class="swatch" style="background:{READ_COLORS[kind]}">'
            '</span>read</td>'
            f'<td>{totals.read_accesses:,}</td>'
            f'<td>{_format_bytes(totals.read_bytes)}</td>'
            f'<td><span class="swatch" style="background:{WRITE_COLORS[kind]}">'
            '</span>write</td>'
            f'<td>{totals.write_accesses:,}</td>'
            f'<td>{_format_bytes(totals.write_bytes)}</td>'
            '</tr>')
    return (
        '<table><caption>Semantic-kind traffic and color legend</caption>'
        '<thead><tr><th>Kind</th><th>Direction</th><th>Accesses</th>'
        '<th>Bytes</th><th>Direction</th><th>Accesses</th><th>Bytes</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')


def _full_traffic_distribution_table(view: BinnedView) -> str:
    reads = view.direction_totals("read")
    writes = view.direction_totals("write")
    rows = []
    for label, values in (
        ("All traffic", [read + write for read, write in zip(reads, writes)]),
        ("Read traffic", reads),
        ("Write traffic", writes),
    ):
        stats = _bin_traffic_stats(values)
        cv = ("N/A" if stats.population_cv is None
              else f"{stats.population_cv:.6f}")
        rows.append(
            '<tr>'
            f'<th scope="row">{label}</th>'
            f'<td>{stats.occupied_bins:,}/{stats.bin_count:,}</td>'
            f'<td>{_format_bytes(stats.minimum_bytes)}</td>'
            f'<td>{_format_mean_bytes(stats.total_bytes, stats.bin_count)}</td>'
            f'<td>{_format_bytes(stats.maximum_bytes)}</td>'
            f'<td>{cv}</td>'
            '</tr>')
    return (
        '<table class="distribution">'
        '<caption>Full-view byte-overlap traffic distribution</caption>'
        '<thead><tr><th>Traffic</th><th>Occupied bins</th>'
        '<th>Min bytes/bin (all bins)</th><th>Mean bytes/bin</th>'
        '<th>Max bytes/bin</th><th>Population CV</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
        '<p class="footnote distribution-note">Occupied bins and byte '
        'statistics use exact request overlap with each full-view bin. The '
        'minimum includes empty bins. Population CV is standard deviation '
        'divided by the all-bin mean; N/A means that direction has no traffic.'
        '</p>')


def render_html(analysis: TraceAnalysis, trace_path: Path) -> str:
    """Return deterministic offline HTML for an analyzed logical trace."""
    summary = analysis.summary
    assert summary.observed_begin is not None and summary.observed_end is not None
    observed_span = summary.observed_end - summary.observed_begin
    full_occupied = analysis.full.occupied_bins()
    zoom_occupied = analysis.zoom.occupied_bins()
    selected = ", ".join(analysis.selected_kinds)
    full_description = (
        f"{analysis.full.bin_count:,} fixed bins across configured capacity; "
        f"{full_occupied:,} contain selected traffic")
    zoom_description = (
        f"the same {analysis.zoom.bin_count:,} bins over the selected "
        f"observed span; {zoom_occupied:,} contain traffic")

    style = """
html { color-scheme: light; background:#f3f6fb; color:#172033; }
body { font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,
       "Segoe UI",sans-serif; margin:0 auto; max-width:1900px; padding:28px; }
h1 { font-size:26px; margin:0 0 5px; }
.subtitle { color:#596579; margin:0 0 22px; overflow-wrap:anywhere; }
.cards { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
.card,.summary,table { background:#fff; border:1px solid #d9e0eb;
  border-radius:11px; box-shadow:0 2px 8px #18223510; }
.card { padding:10px; overflow-x:auto; }
.card svg { display:block; min-width:650px; width:100%; height:auto; }
.summary { display:grid; grid-template-columns:repeat(4,minmax(160px,1fr));
  gap:0; margin:18px 0; overflow:hidden; }
.metric { border-right:1px solid #e6eaf1; border-bottom:1px solid #e6eaf1;
  padding:13px 16px; }
.metric b { display:block; font-size:17px; margin-top:3px; }
.metric span { color:#667085; font-size:12px; text-transform:uppercase;
  letter-spacing:.04em; }
table { border-collapse:separate; border-spacing:0; width:100%; overflow:hidden; }
.distribution { margin-bottom:0; }
caption { font-weight:700; font-size:16px; text-align:left; padding:14px 16px; }
th,td { border-top:1px solid #e6eaf1; padding:9px 12px; text-align:right; }
th:first-child,td:first-child { text-align:left; }
thead th { color:#566176; font-size:12px; text-transform:uppercase; }
.swatch { display:inline-block; width:10px; height:10px; border-radius:2px;
  margin-right:6px; vertical-align:-1px; }
.footnote { color:#596579; margin-top:14px; }
.distribution-note { margin:8px 2px 18px; }
.svg-title { fill:#172033; font-size:18px; font-weight:700; }
.svg-note { fill:#637086; font-size:12px; }
.plot-bg { fill:#f8fafc; stroke:#d9e0eb; }
.baseline { stroke:#65738a; stroke-width:1.2; }
.grid { stroke:#dce3ed; stroke-width:.7; }
.direction { font-size:15px; font-weight:800; }
.read-label { fill:#1d4ed8; }.write-label { fill:#dc2626; }
.tick { fill:#5b6678; font:10px ui-monospace,SFMono-Regular,Consolas,monospace; }
@media (max-width:1100px) { .cards { grid-template-columns:1fr; }
  .summary { grid-template-columns:repeat(2,minmax(150px,1fr)); } }
""".strip()

    metrics = (
        ("Parsed operations", f"{summary.parsed_operations:,}"),
        ("Selected operations", f"{summary.selected_operations:,}"),
        ("Read traffic", _format_bytes(summary.totals.read_bytes)),
        ("Write traffic", _format_bytes(summary.totals.write_bytes)),
        ("Observed range", f"[{summary.observed_begin:#x}, {summary.observed_end:#x})"),
        ("Observed span", _format_bytes(observed_span)),
        ("Span / capacity", _format_percent(observed_span, analysis.full.extent)),
        ("Selected kinds", selected),
        ("Full byte-overlap occupied bins",
         f"{full_occupied:,}/{analysis.full.bin_count:,}"),
        ("Zoom byte-overlap occupied bins",
         f"{zoom_occupied:,}/{analysis.zoom.bin_count:,}"),
        ("Full capacity", _format_bytes(analysis.full.extent)),
        ("Bin count per view", f"{analysis.full.bin_count:,}"),
    )
    metric_html = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span>'
        f'<b>{html.escape(value)}</b></div>' for label, value in metrics)
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>HBFSim logical trace heatmap</title>\n'
        f'<style>{style}</style>\n</head>\n<body>\n'
        '<h1>HBFSim logical trace heatmap</h1>\n'
        f'<p class="subtitle">Trace: {html.escape(str(trace_path))}</p>\n'
        '<div class="cards">\n'
        f'<div class="card">{_svg_view(analysis.full, full_description)}</div>\n'
        f'<div class="card">{_svg_view(analysis.zoom, zoom_description)}</div>\n'
        '</div>\n'
        f'<section class="summary">{metric_html}</section>\n'
        f'{_full_traffic_distribution_table(analysis.full)}\n'
        f'{_legend(analysis)}\n'
        '<p class="footnote">Each request is split by exact half-open byte '
        'overlap at bin boundaries. Read bars grow upward in blue hues; write '
        'bars grow downward in orange/red hues. Bar height uses log1p scaling '
        'per view, while stacked kind segments preserve their exact share of '
        'the bin direction total. Empty bins remain visible.</p>\n'
        '</body>\n</html>\n')


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except (FileNotFoundError, OSError):
        return left.resolve() == right.resolve()


def _atomic_write(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass
        _fail(f"cannot write output {path}: {error}")


def _cli_uint(value: str) -> int:
    try:
        return _parse_uint(value.replace("_", ""), "command-line integer",
                           ADDRESS_SPACE_SIZE)
    except TraceHeatmapError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _cli_kind(value: str) -> str:
    result = canonical_kind(value)
    if result is None:
        raise argparse.ArgumentTypeError(
            f"unknown semantic kind {value!r}; use one of: {', '.join(KINDS)}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True,
                        help="HBFSim/Ramulator-compatible text trace")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="self-contained HTML output")
    parser.add_argument("--capacity-bytes", type=_cli_uint, required=True,
                        help="logical capacity/exclusive address-space end")
    parser.add_argument("--bins", type=int, default=1024,
                        help=f"fixed bins in each view (default 1024, max {MAX_BINS})")
    parser.add_argument("--line-size", type=_cli_uint, default=64,
                        help="bytes for two-token trace records (default 64)")
    parser.add_argument(
        "--kind", action="append", type=_cli_kind,
        help="include one semantic kind (repeatable; default includes all)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if _same_path(args.trace, args.output):
            _fail("trace and output paths must be different")
        analysis = analyze_trace(
            args.trace,
            args.capacity_bytes,
            args.bins,
            args.line_size,
            args.kind,
        )
        output = render_html(analysis, args.trace)
        _atomic_write(args.output, output)
    except TraceHeatmapError as error:
        parser.error(str(error))
    print(
        f"wrote {args.output}: {analysis.summary.selected_operations:,} selected "
        f"operations, span [{analysis.summary.observed_begin:#x}, "
        f"{analysis.summary.observed_end:#x}), {args.bins} bins/view")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
