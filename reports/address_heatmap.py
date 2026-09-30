#!/usr/bin/env python3
"""Render an HBFSim address-traffic heatmap as one offline HTML file.

Input may be either a versioned ``hbfsim.address_heatmap.v1`` JSON artifact or
an ``hbfsim.simulation.summary`` v19 artifact containing
``scenario.address_heatmap``.  The five address domains and all traffic-source
accounting are validated before an output file is touched.  The generated HTML
contains only inline HTML, CSS, and SVG; it has no script, font, CDN, or network
dependency.

Usage::

    python3 reports/address_heatmap.py --input heatmap.json --output heatmap.html

    python3 reports/address_heatmap.py --input summary.json \
        --scenario hbf-streaming --output heatmap.html

All address intervals use the half-open convention ``[begin, end)``. The final
exclusive boundary may be ``2^64`` so byte address ``UINT64_MAX`` is visible.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence


SCHEMA = "hbfsim.address_heatmap.v1"
SUMMARY_SCHEMA = "hbfsim.simulation.summary"
SUMMARY_SCHEMA_VERSION = 19
MAX_BIN_COUNT = 8192
UINT64_MAX = (1 << 64) - 1
ADDRESS_SPACE_SIZE = 1 << 64

DOMAIN_ORDER = (
    "workload_logical",
    "hbm_physical",
    "hbf_logical",
    "hbf_physical",
    "external_physical",
)

DOMAIN_LABELS = {
    "workload_logical": "Workload logical",
    "hbm_physical": "HBM physical",
    "hbf_logical": "HBF logical",
    "hbf_physical": "HBF physical",
    "external_physical": "External physical",
}

TRAFFIC_SOURCES = (
    "workload",
    "direct",
    "mapping",
    "prepopulate",
    "cooperative_buffer",
    "demand_fill",
    "prefetch_fill",
    "streaming_install",
    "destage",
    "garbage_collection",
    "maintenance",
)

DIRECTIONS = ("read", "write", "erase")
METRICS = tuple(
    f"{direction}_{suffix}"
    for direction in DIRECTIONS
    for suffix in ("bytes", "accesses")
)
SOURCE_ARRAY_FIELDS = tuple(f"source_{metric}" for metric in METRICS)

REGION_KINDS = (
    "workload",
    "cooperative_buffer",
    "layer_buffer",
    "static_data",
    "mapped_data",
    "reserved",
    "other",
)

ROOT_KEYS = frozenset(("schema", "bin_count", "traffic_sources", "domains"))
DOMAIN_KEYS = frozenset(
    ("domain", "size_bytes", "totals", "source_totals", "regions", "bins")
)
TOTAL_KEYS = frozenset(METRICS)
SOURCE_TOTAL_KEYS = frozenset(("source", *METRICS))
REGION_KEYS = frozenset(("name", "kind", "begin", "end"))
BIN_KEYS = frozenset(("begin", "end", *METRICS, *SOURCE_ARRAY_FIELDS))


class HeatmapInputError(ValueError):
    """Raised when an input artifact violates the v1 heatmap contract."""


@dataclass(frozen=True)
class Region:
    name: str
    kind: str
    begin: int
    end: int


@dataclass(frozen=True)
class Bin:
    begin: int
    end: int
    metrics: dict[str, int]
    source_metrics: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class SourceTotal:
    source: str
    metrics: dict[str, int]


@dataclass(frozen=True)
class Domain:
    name: str
    size_bytes: int
    totals: dict[str, int]
    source_totals: tuple[SourceTotal, ...]
    regions: tuple[Region, ...]
    bins: tuple[Bin, ...]


@dataclass(frozen=True)
class Heatmap:
    bin_count: int
    domains: tuple[Domain, ...]


def _fail(path: str, message: str) -> NoReturn:
    raise HeatmapInputError(f"{path}: {message}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return value


def _exact_keys(value: dict[str, Any], expected: frozenset[str], path: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    details = []
    if missing:
        details.append("missing " + ", ".join(repr(item) for item in missing))
    if unknown:
        details.append("unknown " + ", ".join(repr(item) for item in unknown))
    if details:
        _fail(path, "; ".join(details))


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected a non-negative integer")
    if value < 0:
        _fail(path, "must be non-negative")
    return value


def _positive_integer(value: Any, path: str) -> int:
    result = _nonnegative_integer(value, path)
    if result == 0:
        _fail(path, "must be greater than zero")
    return result


def _uint64(value: Any, path: str) -> int:
    result = _nonnegative_integer(value, path)
    if result > UINT64_MAX:
        _fail(path, f"must not exceed uint64 maximum {UINT64_MAX}")
    return result


def _address_boundary(value: Any, path: str) -> int:
    result = _nonnegative_integer(value, path)
    if result > ADDRESS_SPACE_SIZE:
        _fail(path, f"must not exceed uint64 address-space end {ADDRESS_SPACE_SIZE}")
    return result


def _positive_address_extent(value: Any, path: str) -> int:
    result = _address_boundary(value, path)
    if result == 0:
        _fail(path, "must be greater than zero")
    return result


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "expected a non-empty string")
    return value


def _validate_metric_presence(metrics: dict[str, int], path: str) -> None:
    for direction in DIRECTIONS:
        has_bytes = metrics[f"{direction}_bytes"] != 0
        has_accesses = metrics[f"{direction}_accesses"] != 0
        if has_bytes != has_accesses:
            _fail(
                path,
                f"{direction} bytes and accesses must either both be zero or "
                "both be non-zero",
            )


def _metrics(value: Any, path: str) -> dict[str, int]:
    obj = _object(value, path)
    _exact_keys(obj, TOTAL_KEYS, path)
    result = {
        metric: _uint64(obj[metric], f"{path}.{metric}")
        for metric in METRICS
    }
    _validate_metric_presence(result, path)
    return result


def _parse_source_totals(value: Any, path: str) -> tuple[SourceTotal, ...]:
    entries = _array(value, path)
    if len(entries) != len(TRAFFIC_SOURCES):
        _fail(path, f"expected {len(TRAFFIC_SOURCES)} entries")

    parsed = []
    for index, (entry_value, expected_source) in enumerate(
            zip(entries, TRAFFIC_SOURCES)):
        entry_path = f"{path}[{index}]"
        entry = _object(entry_value, entry_path)
        _exact_keys(entry, SOURCE_TOTAL_KEYS, entry_path)
        source = _nonempty_string(entry["source"], f"{entry_path}.source")
        if source != expected_source:
            _fail(
                f"{entry_path}.source",
                f"expected {expected_source!r}, got {source!r}",
            )
        metrics = {
            metric: _uint64(
                entry[metric], f"{entry_path}.{metric}")
            for metric in METRICS
        }
        _validate_metric_presence(metrics, entry_path)
        parsed.append(SourceTotal(source=source, metrics=metrics))
    return tuple(parsed)


def _parse_regions(value: Any, size_bytes: int, path: str) -> tuple[Region, ...]:
    parsed = []
    for index, region_value in enumerate(_array(value, path)):
        region_path = f"{path}[{index}]"
        region = _object(region_value, region_path)
        _exact_keys(region, REGION_KEYS, region_path)
        name = _nonempty_string(region["name"], f"{region_path}.name")
        kind = _nonempty_string(region["kind"], f"{region_path}.kind")
        if kind not in REGION_KINDS:
            _fail(
                f"{region_path}.kind",
                f"unknown kind {kind!r}; expected one of {', '.join(REGION_KINDS)}",
            )
        begin = _uint64(region["begin"], f"{region_path}.begin")
        end = _address_boundary(region["end"], f"{region_path}.end")
        if begin >= end:
            _fail(region_path, "region must satisfy begin < end")
        if end > size_bytes:
            _fail(
                f"{region_path}.end",
                f"{end} exceeds domain size_bytes {size_bytes}",
            )
        parsed.append(Region(name=name, kind=kind, begin=begin, end=end))
    # Region order does not carry meaning.  Canonicalizing it makes output
    # stable even if a producer gathers regions from an unordered container.
    return tuple(sorted(parsed, key=lambda item: (
        item.begin, item.end, item.kind, item.name)))


def _parse_bins(
        value: Any, bin_count: int, size_bytes: int, path: str) -> tuple[Bin, ...]:
    entries = _array(value, path)
    if len(entries) != bin_count:
        _fail(path, f"expected bin_count={bin_count} entries, got {len(entries)}")

    parsed = []
    for index, bin_value in enumerate(entries):
        bin_path = f"{path}[{index}]"
        item = _object(bin_value, bin_path)
        _exact_keys(item, BIN_KEYS, bin_path)
        begin = _uint64(item["begin"], f"{bin_path}.begin")
        end = _address_boundary(item["end"], f"{bin_path}.end")
        expected_begin = size_bytes * index // bin_count
        expected_end = size_bytes * (index + 1) // bin_count
        if begin != expected_begin:
            _fail(
                f"{bin_path}.begin",
                f"expected canonical begin {expected_begin}, got {begin}",
            )
        if end != expected_end:
            _fail(
                f"{bin_path}.end",
                f"expected canonical end {expected_end}, got {end}",
            )

        metrics = {
            metric: _uint64(item[metric], f"{bin_path}.{metric}")
            for metric in METRICS
        }
        source_metrics: dict[str, tuple[int, ...]] = {}
        for metric, field in zip(METRICS, SOURCE_ARRAY_FIELDS):
            raw_values = _array(item[field], f"{bin_path}.{field}")
            if len(raw_values) != len(TRAFFIC_SOURCES):
                _fail(
                    f"{bin_path}.{field}",
                    f"expected {len(TRAFFIC_SOURCES)} entries",
                )
            values = tuple(
                _uint64(raw, f"{bin_path}.{field}[{source_index}]")
                for source_index, raw in enumerate(raw_values)
            )
            if sum(values) != metrics[metric]:
                _fail(
                    f"{bin_path}.{field}",
                    f"sum {sum(values)} does not match {metric}={metrics[metric]}",
                )
            source_metrics[metric] = values

        _validate_metric_presence(metrics, bin_path)
        for source_index, source in enumerate(TRAFFIC_SOURCES):
            for direction in DIRECTIONS:
                has_bytes = source_metrics[f"{direction}_bytes"][source_index] != 0
                has_accesses = (
                    source_metrics[f"{direction}_accesses"][source_index] != 0)
                if has_bytes != has_accesses:
                    _fail(
                        f"{bin_path}.source_{direction}_accesses[{source_index}]",
                        f"source {source!r} {direction} bytes and accesses must "
                        "either both be zero or both be non-zero",
                    )

        parsed.append(Bin(
            begin=begin,
            end=end,
            metrics=metrics,
            source_metrics=source_metrics,
        ))
    return tuple(parsed)


def validate_heatmap(value: Any, path: str = "root") -> Heatmap:
    """Validate and normalize a decoded v1 heatmap JSON value."""
    root = _object(value, path)
    _exact_keys(root, ROOT_KEYS, path)
    if root["schema"] != SCHEMA:
        _fail(f"{path}.schema", f"expected {SCHEMA!r}, got {root['schema']!r}")

    bin_count = _positive_integer(root["bin_count"], f"{path}.bin_count")
    if bin_count > MAX_BIN_COUNT:
        _fail(
            f"{path}.bin_count",
            f"must not exceed contract maximum {MAX_BIN_COUNT}",
        )
    raw_sources = _array(
        root["traffic_sources"], f"{path}.traffic_sources")
    if raw_sources != list(TRAFFIC_SOURCES):
        _fail(
            f"{path}.traffic_sources",
            "must exactly match canonical order: " + ", ".join(TRAFFIC_SOURCES),
        )

    domain_values = _array(root["domains"], f"{path}.domains")
    if len(domain_values) != len(DOMAIN_ORDER):
        _fail(
            f"{path}.domains",
            f"expected exactly {len(DOMAIN_ORDER)} domains")

    domains = []
    for index, (domain_value, expected_name) in enumerate(
            zip(domain_values, DOMAIN_ORDER)):
        domain_path = f"{path}.domains[{index}]"
        obj = _object(domain_value, domain_path)
        _exact_keys(obj, DOMAIN_KEYS, domain_path)
        name = _nonempty_string(obj["domain"], f"{domain_path}.domain")
        if name != expected_name:
            _fail(
                f"{domain_path}.domain",
                f"expected {expected_name!r}, got {name!r}",
            )
        size_bytes = _positive_address_extent(
            obj["size_bytes"], f"{domain_path}.size_bytes")
        if size_bytes < bin_count:
            _fail(
                f"{domain_path}.size_bytes",
                f"must be at least bin_count={bin_count}",
            )
        totals = _metrics(obj["totals"], f"{domain_path}.totals")
        source_totals = _parse_source_totals(
            obj["source_totals"], f"{domain_path}.source_totals")
        regions = _parse_regions(
            obj["regions"], size_bytes, f"{domain_path}.regions")
        bins = _parse_bins(
            obj["bins"], bin_count, size_bytes, f"{domain_path}.bins")

        for metric in METRICS:
            bin_sum = sum(item.metrics[metric] for item in bins)
            source_sum = sum(item.metrics[metric] for item in source_totals)
            if totals[metric] != source_sum:
                _fail(
                    f"{domain_path}.totals.{metric}",
                    f"value {totals[metric]} does not match source_totals sum "
                    f"{source_sum}",
                )
            # Bytes are split across touched bins and must conserve exactly.
            # An access, however, is counted once in every bin it touches, so
            # bin-scope access counts intentionally include spatial fan-out.
            if metric.endswith("_bytes"):
                if totals[metric] != bin_sum:
                    _fail(
                        f"{domain_path}.totals.{metric}",
                        f"value {totals[metric]} does not match bin sum {bin_sum}",
                    )
            elif not totals[metric] <= bin_sum <= totals[metric] * bin_count:
                _fail(
                    f"{domain_path}.totals.{metric}",
                    f"bin sum {bin_sum} is outside fan-out range "
                    f"[{totals[metric]}, {totals[metric] * bin_count}]",
                )
            for source_index, source in enumerate(TRAFFIC_SOURCES):
                binned_source_sum = sum(
                    item.source_metrics[metric][source_index] for item in bins)
                declared_source_sum = source_totals[source_index].metrics[metric]
                if metric.endswith("_bytes") and (
                        declared_source_sum != binned_source_sum):
                    _fail(
                        f"{domain_path}.source_totals[{source_index}].{metric}",
                        f"value {declared_source_sum} does not match {source!r} "
                        f"bin sum {binned_source_sum}",
                    )
                if metric.endswith("_accesses") and not (
                        declared_source_sum <= binned_source_sum <=
                        declared_source_sum * bin_count):
                    _fail(
                        f"{domain_path}.source_totals[{source_index}].{metric}",
                        f"{source!r} bin sum {binned_source_sum} is outside "
                        f"fan-out range [{declared_source_sum}, "
                        f"{declared_source_sum * bin_count}]",
                    )

        domains.append(Domain(
            name=name,
            size_bytes=size_bytes,
            totals=totals,
            source_totals=source_totals,
            regions=regions,
            bins=bins,
        ))
    return Heatmap(bin_count=bin_count, domains=tuple(domains))


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HeatmapInputError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_number(token: str) -> NoReturn:
    raise HeatmapInputError(f"JSON contains non-standard number {token!r}")


def select_heatmap(value: Any, scenario: str | None = None) -> Heatmap:
    """Select and validate a direct heatmap or one scenario in a v7 summary."""
    root = _object(value, "root")
    if scenario is not None and (not isinstance(scenario, str) or not scenario):
        _fail("--scenario", "must be a non-empty exact scenario name")

    schema = root.get("schema")
    if schema == SCHEMA:
        if scenario is not None:
            _fail(
                "--scenario",
                "is only valid for an hbfsim.simulation.summary input",
            )
        return validate_heatmap(root)

    if not isinstance(schema, dict):
        _fail(
            "root.schema",
            f"expected {SCHEMA!r} or a {SUMMARY_SCHEMA!r} v9 schema object",
        )
    _exact_keys(schema, frozenset(("name", "version")), "root.schema")
    schema_name = _nonempty_string(schema["name"], "root.schema.name")
    schema_version = _nonnegative_integer(
        schema["version"], "root.schema.version")
    if schema_name != SUMMARY_SCHEMA:
        _fail(
            "root.schema.name",
            f"expected {SUMMARY_SCHEMA!r}, got {schema_name!r}",
        )
    if schema_version != SUMMARY_SCHEMA_VERSION:
        _fail(
            "root.schema.version",
            f"expected {SUMMARY_SCHEMA_VERSION}, got {schema_version}",
        )
    if "scenarios" not in root:
        _fail("root", "missing 'scenarios'")
    raw_scenarios = _array(root["scenarios"], "root.scenarios")
    if not raw_scenarios:
        _fail("root.scenarios", "expected at least one scenario")

    scenarios: list[tuple[str, dict[str, Any], str]] = []
    seen_names: set[str] = set()
    for index, raw_scenario in enumerate(raw_scenarios):
        scenario_path = f"root.scenarios[{index}]"
        block = _object(raw_scenario, scenario_path)
        if "name" not in block:
            _fail(scenario_path, "missing 'name'")
        name = _nonempty_string(block["name"], f"{scenario_path}.name")
        if name in seen_names:
            _fail(
                f"{scenario_path}.name",
                f"duplicate scenario name {name!r}",
            )
        seen_names.add(name)
        scenarios.append((name, block, scenario_path))

    if scenario is None:
        if len(scenarios) != 1:
            available = ", ".join(repr(name) for name, _, _ in scenarios)
            _fail(
                "--scenario",
                "is required for a summary with multiple scenarios; "
                f"available scenarios: {available}",
            )
        selected_name, selected, selected_path = scenarios[0]
    else:
        matches = [entry for entry in scenarios if entry[0] == scenario]
        if not matches:
            available = ", ".join(repr(name) for name, _, _ in scenarios)
            _fail(
                "--scenario",
                f"unknown scenario {scenario!r}; available scenarios: {available}",
            )
        selected_name, selected, selected_path = matches[0]

    if "address_heatmap" not in selected:
        _fail(
            selected_path,
            f"scenario {selected_name!r} is missing 'address_heatmap'",
        )
    heatmap_path = f"{selected_path}.address_heatmap"
    return validate_heatmap(selected["address_heatmap"], heatmap_path)


def load_heatmap(path: Path, scenario: str | None = None) -> Heatmap:
    """Load and select a heatmap from a direct artifact or summary v19 JSON."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise HeatmapInputError(f"cannot read {path}: {error}") from error
    try:
        decoded = json.loads(
            source,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    except json.JSONDecodeError as error:
        raise HeatmapInputError(
            f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    return select_heatmap(decoded, scenario)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")
    unit_index = 0
    scaled = float(value)
    while scaled >= 1024.0 and unit_index < len(units) - 1:
        scaled /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{value} B"
    if scaled >= 100.0:
        rendered = f"{scaled:.0f}"
    elif scaled >= 10.0:
        rendered = f"{scaled:.1f}"
    else:
        rendered = f"{scaled:.2f}"
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return f"{rendered} {units[unit_index]}"


def _format_count(value: int) -> str:
    return f"{value:,}"


def _format_address(value: int) -> str:
    return f"0x{value:x}"


def _direction_title(direction: str) -> str:
    return direction.capitalize()


def _bin_tooltip(domain: Domain, item: Bin, direction: str) -> str:
    byte_metric = f"{direction}_bytes"
    access_metric = f"{direction}_accesses"
    lines = [
        f"{DOMAIN_LABELS[domain.name]} / {_direction_title(direction)}",
        f"[{_format_address(item.begin)}, {_format_address(item.end)})",
        f"{_format_count(item.metrics[byte_metric])} bytes; "
        f"{_format_count(item.metrics[access_metric])} accesses",
    ]
    source_bytes = item.source_metrics[byte_metric]
    source_accesses = item.source_metrics[access_metric]
    for index, source in enumerate(TRAFFIC_SOURCES):
        if source_bytes[index] or source_accesses[index]:
            lines.append(
                f"{source}: {_format_count(source_bytes[index])} bytes, "
                f"{_format_count(source_accesses[index])} accesses"
            )
    return "\n".join(lines)


def _opacity(value: int, accesses: int, maximum: int) -> float:
    if value == 0 and accesses == 0:
        return 0.0
    if value == 0 or maximum == 0:
        return 0.18
    return 0.16 + 0.84 * math.log1p(value) / math.log1p(maximum)


def _svg_rect(
        *, x: float, y: float, width: float, height: float, fill: str,
        opacity: float | None = None, css_class: str = "",
        extra: str = "", title: str | None = None) -> str:
    attributes = [
        f'x="{x:.3f}"', f'y="{y:.3f}"',
        f'width="{max(width, 0.0):.3f}"', f'height="{height:.3f}"',
        f'fill="{fill}"',
    ]
    if opacity is not None:
        attributes.append(f'opacity="{opacity:.6f}"')
    if css_class:
        attributes.append(f'class="{_escape(css_class)}"')
    if extra:
        attributes.append(extra)
    if title is None:
        return "<rect " + " ".join(attributes) + "></rect>"
    return (
        "<rect " + " ".join(attributes) + ">"
        f"<title>{_escape(title)}</title></rect>"
    )


def _render_svg(heatmap: Heatmap) -> str:
    width = 1280.0
    left = 160.0
    right = 42.0
    plot_width = width - left - right
    header_height = 112.0
    panel_height = 222.0
    footer_height = 84.0
    total_height = header_height + panel_height * len(heatmap.domains) + footer_height
    lane_height = 25.0
    lane_gap = 9.0

    colors = {
        "read": "var(--read-color)",
        "write": "var(--write-color)",
        "erase": "var(--erase-color)",
    }
    patterns = {
        "write": "url(#write-hatch)",
        "erase": "url(#erase-crosshatch)",
    }
    region_styles = {
        "workload": ("var(--region-workload)", ""),
        "cooperative_buffer": ("var(--region-cooperative-buffer)", "7 3"),
        "layer_buffer": ("var(--region-layer-buffer)", "7 3"),
        "static_data": ("var(--region-static)", "3 2"),
        "mapped_data": ("var(--region-mapped)", "9 3 2 3"),
        "reserved": ("var(--region-reserved)", "2 2"),
        "other": ("var(--region-other)", "10 4"),
    }

    maxima = {}
    for direction in DIRECTIONS:
        metric = f"{direction}_bytes"
        maxima[direction] = max(
            (item.metrics[metric]
             for domain in heatmap.domains
             for item in domain.bins),
            default=0,
        )

    lines = [
        (f'<svg viewBox="0 0 {width:.0f} {total_height:.0f}" '
         'role="img" aria-labelledby="heatmap-svg-title heatmap-svg-desc">'),
        '<title id="heatmap-svg-title">HBFSim address traffic heatmap</title>',
        ('<desc id="heatmap-svg-desc">Five aligned address-domain panels. '
         'Each panel has separate read, write, and erase traffic lanes; writes '
         'are diagonally hatched and erases are cross-hatched.</desc>'),
        '<defs>',
        ('<pattern id="write-hatch" patternUnits="userSpaceOnUse" width="8" '
         'height="8"><path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" '
         'stroke="var(--write-pattern)" stroke-width="1.5"></path></pattern>'),
        ('<pattern id="erase-crosshatch" patternUnits="userSpaceOnUse" '
         'width="8" height="8"><path d="M0,0 L8,8 M8,0 L0,8" '
         'stroke="var(--erase-pattern)" stroke-width="1.25"></path></pattern>'),
        '</defs>',
        ('<rect x="0" y="0" width="1280" height="100%" '
         'fill="var(--svg-background)"></rect>'),
        '<text x="42" y="38" class="svg-title">HBFSim address traffic heatmap</text>',
        (f'<text x="42" y="65" class="svg-note">{heatmap.bin_count} bins per '
         'domain · global per-direction log intensity · hover a mark for exact '
         'source accounting</text>'),
    ]

    legend_x = 42.0
    legend_y = 82.0
    for direction in DIRECTIONS:
        lines.append(_svg_rect(
            x=legend_x,
            y=legend_y,
            width=28,
            height=15,
            fill=colors[direction],
            opacity=0.92,
            css_class=f"legend-{direction}",
        ))
        if direction in patterns:
            lines.append(_svg_rect(
                x=legend_x,
                y=legend_y,
                width=28,
                height=15,
                fill=patterns[direction],
            ))
        lines.append(
            f'<text x="{legend_x + 36:.1f}" y="{legend_y + 12:.1f}" '
            f'class="svg-legend">{_direction_title(direction)}</text>'
        )
        legend_x += 132.0
    lines.append(
        '<text x="455" y="94" class="svg-note">Solid / diagonal hatch / '
        'cross-hatch encode direction independently of color.</text>')

    for domain_index, domain in enumerate(heatmap.domains):
        panel_y = header_height + domain_index * panel_height
        plot_top = panel_y + 62.0
        plot_bottom = plot_top + len(DIRECTIONS) * lane_height + (
            len(DIRECTIONS) - 1) * lane_gap
        lines.extend([
            (f'<g class="domain-panel" data-domain="{_escape(domain.name)}">'),
            _svg_rect(
                x=24,
                y=panel_y + 5,
                width=width - 48,
                height=panel_height - 10,
                fill=("var(--panel-even)" if domain_index % 2 == 0
                      else "var(--panel-odd)"),
                css_class="panel-background",
            ),
            (f'<text x="42" y="{panel_y + 31:.1f}" class="panel-title">'
             f'{_escape(DOMAIN_LABELS[domain.name])}</text>'),
            (f'<text x="{left:.1f}" y="{panel_y + 31:.1f}" class="svg-note">'
             f'range [0, {_escape(_format_address(domain.size_bytes))}) · '
             f'{_escape(_format_bytes(domain.size_bytes))}</text>'),
            (f'<text x="{left:.1f}" y="{panel_y + 50:.1f}" class="svg-note">'
             f'R {_escape(_format_bytes(domain.totals["read_bytes"]))} / '
             f'{_escape(_format_count(domain.totals["read_accesses"]))} accesses · '
             f'W {_escape(_format_bytes(domain.totals["write_bytes"]))} / '
             f'{_escape(_format_count(domain.totals["write_accesses"]))} accesses · '
             f'E {_escape(_format_bytes(domain.totals["erase_bytes"]))} / '
             f'{_escape(_format_count(domain.totals["erase_accesses"]))} accesses'
             '</text>'),
        ])

        for direction_index, direction in enumerate(DIRECTIONS):
            lane_y = plot_top + direction_index * (lane_height + lane_gap)
            lines.append(
                f'<text x="{left - 13:.1f}" y="{lane_y + 17:.1f}" '
                f'class="lane-label" text-anchor="end">'
                f'{_direction_title(direction)}</text>'
            )
            lines.append(_svg_rect(
                x=left,
                y=lane_y,
                width=plot_width,
                height=lane_height,
                fill="var(--lane-background)",
                css_class="lane-background",
            ))
            byte_metric = f"{direction}_bytes"
            access_metric = f"{direction}_accesses"
            for bin_index, item in enumerate(domain.bins):
                traffic_bytes = item.metrics[byte_metric]
                accesses = item.metrics[access_metric]
                opacity = _opacity(traffic_bytes, accesses, maxima[direction])
                if opacity == 0.0:
                    continue
                x = left + plot_width * item.begin / domain.size_bytes
                end_x = left + plot_width * item.end / domain.size_bytes
                tooltip = _bin_tooltip(domain, item, direction)
                extra = (
                    f'data-bin="{bin_index}" data-direction="{direction}" '
                    f'data-begin="{item.begin}" data-end="{item.end}"'
                )
                lines.append(_svg_rect(
                    x=x,
                    y=lane_y,
                    width=end_x - x,
                    height=lane_height,
                    fill=colors[direction],
                    opacity=opacity,
                    css_class=f"traffic-mark traffic-{direction}",
                    extra=extra,
                    title=tooltip,
                ))
                if direction in patterns:
                    lines.append(_svg_rect(
                        x=x,
                        y=lane_y,
                        width=end_x - x,
                        height=lane_height,
                        fill=patterns[direction],
                        opacity=min(1.0, opacity + 0.18),
                        css_class=f"traffic-pattern traffic-{direction}-pattern",
                        extra='pointer-events="none"',
                    ))

        for region_index, region in enumerate(domain.regions):
            region_x = left + plot_width * region.begin / domain.size_bytes
            region_end_x = left + plot_width * region.end / domain.size_bytes
            color, dash = region_styles[region.kind]
            extra = (
                f'stroke="{color}" stroke-width="1.8" fill-opacity="0.075" '
                'pointer-events="stroke"'
            )
            if dash:
                extra += f' stroke-dasharray="{dash}"'
            lines.append(_svg_rect(
                x=region_x,
                y=plot_top - 4,
                width=region_end_x - region_x,
                height=plot_bottom - plot_top + 8,
                fill=color,
                css_class="region-overlay",
                extra=(extra + f' data-region-index="{region_index}" '
                       f'data-region-kind="{_escape(region.kind)}"'),
                title=(f"Region: {region.name}\nKind: {region.kind}\n"
                       f"[{_format_address(region.begin)}, "
                       f"{_format_address(region.end)})"),
            ))
            label_x = min(max(region_x + 3.0, left + 3.0), left + plot_width - 5.0)
            lines.append(
                f'<text x="{label_x:.3f}" y="{plot_top - 9:.3f}" '
                f'class="region-label" fill="{color}">'
                f'{_escape(region.name)} [{_escape(region.kind)}]</text>'
            )

        axis_y = plot_bottom + 21.0
        lines.append(
            f'<line x1="{left:.1f}" y1="{axis_y - 13:.1f}" '
            f'x2="{left + plot_width:.1f}" y2="{axis_y - 13:.1f}" '
            'stroke="var(--axis-color)" stroke-width="1"></line>'
        )
        for tick in range(5):
            x = left + plot_width * tick / 4
            address = domain.size_bytes * tick // 4
            anchor = "start" if tick == 0 else "end" if tick == 4 else "middle"
            lines.append(
                f'<line x1="{x:.3f}" y1="{axis_y - 13:.1f}" '
                f'x2="{x:.3f}" y2="{axis_y - 8:.1f}" '
                'stroke="var(--axis-color)"></line>'
            )
            lines.append(
                f'<text x="{x:.3f}" y="{axis_y + 5:.1f}" text-anchor="{anchor}" '
                f'class="axis-label">{_escape(_format_address(address))} · '
                f'{_escape(_format_bytes(address))}</text>'
            )
        lines.append('</g>')

    footer_y = header_height + panel_height * len(heatmap.domains) + 25
    # Avoid a long visually dense line while keeping the exact legend readable.
    region_legend = [
        "workload: solid green",
        "cooperative/layer buffers: dashed magenta",
        "static_data: dotted blue",
        "mapped_data: dash-dot orange",
        "reserved: dotted gray",
        "other: dashed black",
    ]
    lines.append(
        f'<text x="42" y="{footer_y:.1f}" class="svg-note">Region outlines · '
        f'{_escape(" · ".join(region_legend))}</text>'
    )
    lines.append(
        f'<text x="42" y="{footer_y + 25:.1f}" class="svg-note">'
        f'Contract: {SCHEMA}. Intervals are half-open [begin, end). '
        'Empty lanes represent zero bytes and zero accesses.</text>'
    )
    lines.append('</svg>')
    return "\n".join(line for line in lines if line)


def _render_totals_table(heatmap: Heatmap) -> str:
    header = "".join(
        f"<th scope=\"col\">{_escape(metric)}</th>" for metric in METRICS)
    rows = []
    for domain in heatmap.domains:
        values = "".join(
            f'<td data-metric="{_escape(metric)}">'
            f'{_escape(_format_count(domain.totals[metric]))}</td>'
            for metric in METRICS
        )
        rows.append(
            f'<tr><th scope="row">{_escape(DOMAIN_LABELS[domain.name])}</th>'
            f"{values}</tr>"
        )
    return (
        '<section aria-labelledby="totals-heading"><h2 id="totals-heading">'
        'Exact domain totals</h2><div class="table-scroll"><table>'
        f'<thead><tr><th scope="col">Domain</th>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></section>'
    )


def _render_source_tables(heatmap: Heatmap) -> str:
    sections = []
    metric_headers = "".join(
        f'<th scope="col">{_escape(metric)}</th>' for metric in METRICS)
    for domain in heatmap.domains:
        rows = []
        for source_total in domain.source_totals:
            values = "".join(
                f'<td>{_escape(_format_count(source_total.metrics[metric]))}</td>'
                for metric in METRICS
            )
            rows.append(
                f'<tr><th scope="row">{_escape(source_total.source)}</th>'
                f'{values}</tr>')
        sections.append(
            '<details><summary>'
            f'{_escape(DOMAIN_LABELS[domain.name])} source totals</summary>'
            '<div class="table-scroll"><table><thead><tr>'
            f'<th scope="col">Source</th>{metric_headers}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></details>'
        )
    return (
        '<section aria-labelledby="sources-heading"><h2 id="sources-heading">'
        'Traffic-source accounting</h2>' + "".join(sections) + '</section>'
    )


def _render_region_tables(heatmap: Heatmap) -> str:
    sections = []
    for domain in heatmap.domains:
        if domain.regions:
            rows = "".join(
                '<tr>'
                f'<th scope="row">{_escape(region.name)}</th>'
                f'<td>{_escape(region.kind)}</td>'
                f'<td>{_escape(_format_address(region.begin))}</td>'
                f'<td>{_escape(_format_address(region.end))}</td>'
                '</tr>'
                for region in domain.regions
            )
        else:
            rows = '<tr><td colspan="4">No regions declared</td></tr>'
        sections.append(
            '<details><summary>'
            f'{_escape(DOMAIN_LABELS[domain.name])} regions</summary>'
            '<div class="table-scroll"><table><thead><tr>'
            '<th scope="col">Name</th><th scope="col">Kind</th>'
            '<th scope="col">Begin</th><th scope="col">End (exclusive)</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div></details>'
        )
    return (
        '<section aria-labelledby="regions-heading"><h2 id="regions-heading">'
        'Region overlays</h2>' + "".join(sections) + '</section>'
    )


def render_html(heatmap: Heatmap) -> str:
    """Return deterministic, self-contained HTML for a validated heatmap."""
    style = """
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
      --page-background: #f2f4f7;
      --surface: #ffffff;
      --svg-background: #ffffff;
      --panel-even: #f8fafc;
      --panel-odd: #ffffff;
      --lane-background: #eef1f4;
      --border: #cfd6dd;
      --border-subtle: #d8dee5;
      --border-panel: #d9dfe5;
      --border-lane: #d7dde3;
      --text-primary: #17202a;
      --text-secondary: #4e5a66;
      --text-lane: #34404b;
      --table-heading: #eef2f5;
      --axis-color: #67727e;
      --label-halo: #ffffff;
      --shadow: rgba(0, 0, 0, 0.07);
      --read-color: #0072b2;
      --write-color: #d55e00;
      --erase-color: #5f6368;
      --write-pattern: #3a2700;
      --erase-pattern: #ffffff;
      --region-workload: #007f5f;
      --region-cooperative-buffer: #a83f83;
      --region-layer-buffer: #7048b8;
      --region-static: #087eaa;
      --region-mapped: #9b6500;
      --region-reserved: #61676d;
      --region-other: #111111;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --page-background: #0f1419;
        --surface: #171d23;
        --svg-background: #171d23;
        --panel-even: #1d252d;
        --panel-odd: #171d23;
        --lane-background: #28323c;
        --border: #46515c;
        --border-subtle: #3d4853;
        --border-panel: #3b4651;
        --border-lane: #4a5662;
        --text-primary: #f1f5f8;
        --text-secondary: #bdc8d2;
        --text-lane: #dce4eb;
        --table-heading: #27313b;
        --axis-color: #9ba9b6;
        --label-halo: #171d23;
        --shadow: rgba(0, 0, 0, 0.35);
        --read-color: #56b4e9;
        --write-color: #e69f00;
        --erase-color: #c3cbd3;
        --write-pattern: #302000;
        --erase-pattern: #20272e;
        --region-workload: #38d39f;
        --region-cooperative-buffer: #ee8dcb;
        --region-layer-buffer: #c4b5fd;
        --region-static: #6fc9f4;
        --region-mapped: #f0b84b;
        --region-reserved: #aeb8c2;
        --region-other: #f2f5f7;
      }
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--text-primary); background: var(--page-background); }
    main { max-width: 1440px; margin: 0 auto; padding: 20px; }
    .visualization { background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; box-shadow: 0 2px 7px var(--shadow); overflow: hidden; }
    svg { display: block; width: 100%; height: auto; }
    .svg-title { font-size: 24px; font-weight: 700; fill: var(--text-primary); }
    .panel-title { font-size: 17px; font-weight: 700; fill: var(--text-primary); }
    .svg-note, .svg-legend { font-size: 12px; fill: var(--text-secondary); }
    .lane-label { font-size: 12px; font-weight: 700; fill: var(--text-lane); }
    .axis-label { font-size: 10px; fill: var(--text-secondary); }
    .region-label { font-size: 9px; font-weight: 700; paint-order: stroke;
      stroke: var(--label-halo); stroke-width: 2px; stroke-linejoin: round; }
    .panel-background { stroke: var(--border-panel); stroke-width: 1; }
    .lane-background { stroke: var(--border-lane); stroke-width: 1; }
    .traffic-mark, .region-overlay { shape-rendering: crispEdges; }
    section { margin-top: 22px; background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px; }
    h1 { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
      overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
    h2 { margin: 0 0 12px; font-size: 18px; }
    details { border-top: 1px solid var(--border-subtle); padding: 9px 0; }
    summary { cursor: pointer; font-weight: 650; }
    .table-scroll { overflow-x: auto; margin-top: 9px; }
    table { border-collapse: collapse; min-width: 850px; width: 100%;
      font-variant-numeric: tabular-nums; }
    th, td { border: 1px solid var(--border-subtle); padding: 6px 8px; text-align: right;
      white-space: nowrap; font-size: 12px; }
    th:first-child, td:first-child { text-align: left; }
    thead th { background: var(--table-heading); }
    footer { margin: 16px 2px 4px; color: var(--text-secondary); font-size: 12px; }
    """.strip()
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>HBFSim address traffic heatmap</title>\n'
        f'<style>\n{style}\n</style>\n</head>\n<body>\n<main>\n'
        '<h1>HBFSim address traffic heatmap</h1>\n'
        f'<div class="visualization">\n{_render_svg(heatmap)}\n</div>\n'
        f'{_render_totals_table(heatmap)}\n'
        f'{_render_source_tables(heatmap)}\n'
        f'{_render_region_tables(heatmap)}\n'
        f'<footer>Validated input schema: {_escape(SCHEMA)}. Generated '
        'deterministically without external assets or network dependencies.</footer>\n'
        '</main>\n</body>\n</html>\n'
    )


def write_html(path: Path, content: str) -> None:
    """Atomically publish UTF-8 HTML, preserving any old file on failure."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass
        raise HeatmapInputError(f"cannot write {path}: {error}") from error


def _argument_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.exit(2, f"{parser.prog}: error: {message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="v1 address heatmap JSON or simulator summary v19 JSON")
    parser.add_argument(
        "--scenario",
        help=("exact scenario name when --input is a simulator summary; "
              "required only when that summary contains multiple scenarios"),
    )
    parser.add_argument(
        "-o", "--output", "--out", type=Path, required=True,
        help="self-contained HTML output",
    )
    args = parser.parse_args(argv)

    try:
        if args.input.resolve() == args.output.resolve():
            raise HeatmapInputError("input and output paths must be different")
        heatmap = load_heatmap(args.input, args.scenario)
        try:
            output = render_html(heatmap)
        except (ArithmeticError, ValueError) as error:
            raise HeatmapInputError(
                f"validated heatmap cannot be rendered: {error}") from error
        write_html(args.output, output)
    except HeatmapInputError as error:
        _argument_error(parser, str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
