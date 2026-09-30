"""Turn a ``hbfsim.simulation.summary`` JSON into a few comparable numbers.

The summary keeps every counter the simulator produces (hundreds per
scenario). The headline metrics below are the ones most questions start
with; everything else stays one ``summary["scenarios"][i][...]`` away.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from hbfsim.workspace import HbfsimError
from reports.address_heatmap import SUMMARY_SCHEMA as _SCHEMA_NAME
from reports.address_heatmap import SUMMARY_SCHEMA_VERSION


SUMMARY_SCHEMA = {"name": _SCHEMA_NAME, "version": SUMMARY_SCHEMA_VERSION}


def _get(row: Mapping[str, Any], *path: str) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, Mapping) or value.get(key) is None:
            return None
        value = value[key]
    return value


def _scaled(scale: float, *path: str) -> Callable[[Mapping[str, Any]], float | None]:
    def extract(row: Mapping[str, Any]) -> float | None:
        value = _get(row, *path)
        return None if value is None else float(value) * scale
    return extract


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    help: str
    extract: Callable[[Mapping[str, Any]], Any]


METRICS: tuple[Metric, ...] = (
    Metric("makespan_us", "makespan", "us",
           "Time from the first request until all work (including background drain) finished.",
           _scaled(1e-3, "time_breakdown", "wall_clock_ns", "makespan_ns")),
    Metric("throughput_GBps", "throughput", "GB/s",
           "User bytes divided by the time the last user request completed.",
           _scaled(1.0, "user_completion_throughput_GBps")),
    Metric("mean_latency_us", "mean lat", "us",
           "Mean request latency from offered arrival to completion.",
           _scaled(1e-3, "time_breakdown", "latency_work", "average_ns")),
    Metric("p95_latency_us", "p95 lat", "us",
           "95th-percentile request latency.",
           _scaled(1e-3, "time_breakdown", "latency_work", "p95_ns")),
    Metric("hbm_accesses", "HBM ops", "",
           "Accesses served by HBM, including background copies.",
           lambda row: _get(row, "hbm_accesses")),
    Metric("hbf_accesses", "HBF ops", "",
           "Accesses served by HBF, including background copies.",
           lambda row: _get(row, "hbf_accesses")),
    Metric("external_accesses", "ext ops", "",
           "Accesses served by the external backing device (CXL, NVMe, host DRAM...).",
           lambda row: _get(row, "external_accesses")),
    Metric("hbf_waf", "HBF WAF", "",
           "HBF write amplification: physical programmed bytes / logical written bytes.",
           lambda row: _get(row, "hbf_stats", "waf")),
    Metric("hbf_block_erases", "erases", "",
           "HBF block erases performed during the run (wear).",
           lambda row: _get(row, "hbf_stats", "block_erases")),
    Metric("hbf_max_block_erases", "max P/E", "",
           "Erase count of the most-worn HBF block; it reaches the endurance limit first.",
           lambda row: _get(row, "hbf_stats", "max_block_erase_count")),
    Metric("hbf_gc_relocations", "GC copies", "",
           "Valid pages garbage collection copied to reclaim blocks.",
           lambda row: _get(row, "hbf_stats", "gc_relocations")),
)
METRIC_KEYS = tuple(metric.key for metric in METRICS)
DEFAULT_TABLE_METRICS = (
    "makespan_us", "throughput_GBps", "mean_latency_us", "p95_latency_us",
    "hbm_accesses", "hbf_accesses", "hbf_waf",
)


def metric(key: str) -> Metric:
    for candidate in METRICS:
        if candidate.key == key:
            return candidate
    raise HbfsimError(f"unknown metric {key!r}; choose from {', '.join(METRIC_KEYS)}")


@dataclass(frozen=True)
class ScenarioResult:
    """Headline metrics for one scenario plus the complete raw summary row."""

    name: str
    metrics: Mapping[str, Any]
    warnings: tuple[str, ...]
    raw: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.metrics[key]


@dataclass(frozen=True)
class RunResult:
    """One completed reference-runner invocation."""

    summary_path: Path
    summary: Mapping[str, Any]
    scenarios: tuple[ScenarioResult, ...]
    command: tuple[str, ...] = ()
    labels: Mapping[str, str] | None = None

    @property
    def sanity(self) -> str:
        return str(self.summary.get("sanity", "UNKNOWN"))

    @property
    def ok(self) -> bool:
        return self.sanity == "PASS"

    def scenario(self, name: str) -> ScenarioResult:
        for scenario in self.scenarios:
            if scenario.name == name:
                return scenario
        raise KeyError(name)

    def rows(self) -> list[dict[str, Any]]:
        """Flat records (labels + scenario + metrics) for tables, CSV or pandas."""

        records = []
        for scenario in self.scenarios:
            record: dict[str, Any] = dict(self.labels or {})
            record["scenario"] = scenario.name
            record.update(scenario.metrics)
            record["warnings"] = len(scenario.warnings)
            records.append(record)
        return records

    def table(self, metrics: Sequence[str] = DEFAULT_TABLE_METRICS) -> str:
        return format_table(self.rows(), metrics=metrics)


def load_summary(path: str | Path, *, labels: Mapping[str, str] | None = None,
                 command: Sequence[str] = ()) -> RunResult:
    """Load ``summary.json`` (or a run directory containing one)."""

    location = Path(path)
    if location.is_dir():
        location = location / "summary.json"
    try:
        summary = json.loads(location.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HbfsimError(f"no summary JSON at {location}") from error
    except json.JSONDecodeError as error:
        raise HbfsimError(f"{location} is not valid JSON: {error}") from error
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise HbfsimError(
            f"{location} is not an {SUMMARY_SCHEMA['name']} v{SUMMARY_SCHEMA['version']} "
            f"summary (found {summary.get('schema')!r})"
        )
    scenarios = []
    for row in summary.get("scenarios", []):
        values = {item.key: item.extract(row) for item in METRICS}
        scenarios.append(ScenarioResult(
            name=str(row.get("name")),
            metrics=values,
            warnings=tuple(str(item) for item in row.get("warnings") or ()),
            raw=row,
        ))
    return RunResult(location, summary, tuple(scenarios), tuple(command), labels)


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1e6 or magnitude < 1e-3:
            return f"{value:.3e}"
        if magnitude >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4g}"
    return str(value)


def _header(key: str) -> str:
    try:
        item = metric(key)
    except HbfsimError:
        return key
    return f"{item.label} ({item.unit})" if item.unit else item.label


def format_table(rows: Sequence[Mapping[str, Any]], *, metrics: Sequence[str] = DEFAULT_TABLE_METRICS,
                 style: str = "text") -> str:
    """Render records as an aligned text table, Markdown, CSV, or JSON."""

    if not rows:
        return "(no results)"
    label_keys = [key for key in rows[0] if key not in METRIC_KEYS and key != "warnings"]
    columns = label_keys + [key for key in metrics if key not in label_keys]
    if style == "json":
        return json.dumps([{key: row.get(key) for key in columns} for row in rows], indent=2)
    if style == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().rstrip("\n")
    headers = [_header(key) for key in columns]
    cells = [[format_value(row.get(key)) for key in columns] for row in rows]
    if style == "markdown":
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join("---" if index < len(label_keys) else "---:"
                                for index in range(len(columns))) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in cells]
        return "\n".join(lines)
    if style != "text":
        raise HbfsimError(f"unknown table style {style!r}")
    widths = [max(len(headers[i]), *(len(row[i]) for row in cells)) for i in range(len(columns))]

    def render(values: Iterable[str]) -> str:
        parts = []
        for index, value in enumerate(values):
            if index < len(label_keys):
                parts.append(value.ljust(widths[index]))
            else:
                parts.append(value.rjust(widths[index]))
        return "  ".join(parts).rstrip()

    lines = [render(headers), render("-" * width for width in widths)]
    lines += [render(row) for row in cells]
    return "\n".join(lines)


def metric_legend(metrics: Sequence[str] = DEFAULT_TABLE_METRICS) -> str:
    return "\n".join(f"  {_header(key):<20} {metric(key).help}" for key in metrics)
