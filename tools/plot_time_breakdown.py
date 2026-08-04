#!/usr/bin/env python3
"""Render an auditable HBFSim timing dashboard from schema-v9/v16 summaries.

The dashboard keeps the timing contract visible instead of drawing one
misleading stacked "latency" bar:

* wall-clock segments are the only additive elapsed-time quantities;
* phase dependency, admission, and service latency are per-user-operation
  work averages;
* device/controller stage work can overlap and is never presented as elapsed
  time; and
* resource utilization is capacity-normalized independently from stage work.

The output is a self-contained HTML document with no network dependencies.
Every input is first validated by :mod:`time_breakdown_report`, so malformed,
failed, or internally inconsistent summaries cannot produce a plausible-looking
chart.

Example::

    python3 tools/plot_time_breakdown.py \
        --summary baseline=out/ec0-ec1.summary.json \
        --summary ec2=out/ec2.summary.json \
        --alias baseline/all-HBM=EC0 \
        --alias baseline/all-HBF=EC1 \
        --output out/ec0-ec2.time-breakdown.html
"""

from __future__ import annotations

import argparse
import html
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, NoReturn, Sequence

from time_breakdown_report import (
    BASE_DIE_LINK_COMPONENTS,
    ARCHIVED_DEVICE_STAGE_COMPONENTS,
    DEVICE_STAGE_COMPONENTS,
    RESOURCE_NAMES,
    LAYER_STREAMING_CONTROLLER_FIELDS,
    SummaryInput,
    TimeBreakdownError,
    TimeBreakdownReport,
    build_time_breakdown_report,
    load_summary_input,
)


PRIMARY_STAGE_ROLES = frozenset((
    "overlapping_work_component",
    "controller_parent",
))
NESTED_ATTRIBUTION_ROLES = frozenset((
    "attribution_child",
    "attribution_subset",
))


@dataclass(frozen=True)
class StageGroup:
    key: str
    label: str
    css_color: str


STAGE_GROUPS = (
    StageGroup("hbm_queue", "HBM queue", "--stage-hbm-queue"),
    StageGroup("hbm_core", "HBM core", "--stage-hbm-core"),
    StageGroup("hbm_transfer", "HBM transfer", "--stage-hbm-transfer"),
    StageGroup("hbf_queue", "HBF queue", "--stage-hbf-queue"),
    StageGroup("hbf_metadata", "HBF mapping / metadata", "--stage-hbf-metadata"),
    StageGroup("hbf_array", "HBF array", "--stage-hbf-array"),
    StageGroup("hbf_transfer", "HBF data path", "--stage-hbf-transfer"),
    StageGroup("hbf_ecc", "HBF ECC", "--stage-hbf-ecc"),
    StageGroup("external_queue", "External queue", "--stage-external-queue"),
    StageGroup(
        "external_controller",
        "External controller",
        "--stage-external-controller",
    ),
    StageGroup("external_media", "External media", "--stage-external-media"),
    StageGroup("external_link", "External host link", "--stage-external-link"),
    StageGroup("base_die", "Base-die link", "--stage-base-die"),
    StageGroup("streaming", "Layer streaming", "--stage-streaming"),
    StageGroup("cooperative", "Cooperative write", "--stage-cooperative"),
    StageGroup("other", "Other", "--stage-other"),
)
STAGE_GROUP_BY_KEY = {group.key: group for group in STAGE_GROUPS}


COMPONENT_LABELS = {
    "ingress_queue_wait_work_ns": "Ingress queue wait",
    "scheduler_queue_wait_work_ns": "Scheduler queue wait",
    "address_mapping_work_ns": "Address mapping",
    "translation_work_ns": "Translation",
    "metadata_cache_work_ns": "Metadata cache",
    "mapping_dram_work_ns": "Resident mapping DRAM",
    "dram_wait_work_ns": "DRAM issue wait",
    "refresh_stall_work_ns": "Refresh stall",
    "precharge_work_ns": "Precharge",
    "activation_work_ns": "Activation",
    "command_work_ns": "Command",
    "array_read_work_ns": "Array read",
    "array_program_work_ns": "Array program",
    "array_erase_work_ns": "Array erase",
    "program_verify_work_ns": "Program verify",
    "media_lane_transfer_work_ns": "Media-lane transfer",
    "page_buffer_work_ns": "Page buffer",
    "sram_staging_work_ns": "Logic SRAM staging",
    "channel_transfer_work_ns": "Channel transfer",
    "tsv_transfer_work_ns": "TSV transfer",
    "hb_io_transfer_work_ns": "HB I/O transfer",
    "transport_latency_work_ns": "Transport propagation",
    "ecc_queue_wait_work_ns": "ECC queue wait",
    "ecc_response_latency_work_ns": "ECC response",
    "maintenance_work_ns": "Maintenance",
    "slot_wait_work_ns": "Finite-slot wait",
    "state_wait_work_ns": "Causal-state wait",
    "read_queue_wait_work_ns": "Read queue wait",
    "write_queue_wait_work_ns": "Write queue wait",
    "read_serialization_work_ns": "Read serialization",
    "write_serialization_work_ns": "Write serialization",
    "read_fixed_latency_work_ns": "Read fixed latency",
    "write_fixed_latency_work_ns": "Write fixed latency",
    "demand_stall_work_ns": "Demand stall",
    "install_admission_wait_work_ns": "Install admission wait",
    "full_wait_work_ns": "Full-buffer wait",
}


RESOURCE_LABELS = {
    "hbm_data_bus": "HBM data bus",
    "hbf_logic_ingress": "HBF logic ingress",
    "hbf_plane_media": "HBF plane media",
    "hbf_media_lane": "HBF media lane",
    "hbf_subarray": "HBF subarray",
    "hbf_page_buffer_bank": "HBF page-buffer bank",
    "hbf_flash_source_queue": "HBF flash source queue",
    "hbf_channel_command": "HBF channel command",
    "hbf_channel_data": "HBF channel data",
    "hbf_tsv": "HBF TSV",
    "hbf_sram": "HBF logic SRAM",
    "hbf_hbio_command": "HBF HB-I/O command",
    "hbf_hbio_data": "HBF HB-I/O data",
    "hbf_sequencer": "HBF sequencer",
    "hbf_ecc_issue": "HBF ECC issue",
    "base_die_link_read": "Base-die read link",
    "base_die_link_write": "Base-die write link",
    "external_controller": "External-backing controller issue",
    "external_media": "External-backing media",
    "external_link_m2s": "External-backing M2S link",
    "external_link_s2m": "External-backing S2M link",
}


class TimeBreakdownVisualizationError(ValueError):
    """A validated timing report cannot be represented unambiguously."""


@dataclass(frozen=True)
class StageCost:
    key: str
    label: str
    group: str
    value_us_per_op: float


@dataclass(frozen=True)
class ResourceCost:
    key: str
    label: str
    utilization: float
    average_parallelism: float
    resource_count: int


@dataclass(frozen=True)
class VisualScenario:
    key: str
    label: str
    run_label: str
    scenario: str
    config: str
    ops: int
    reads: int
    writes: int
    throughput_GBps: float | None
    hbm_accesses: int
    hbf_accesses: int
    external_accesses: int
    offered_ms: float
    completion_tail_ms: float
    drain_ms: float
    makespan_ms: float
    admission_us_per_op: float
    service_us_per_op: float
    offered_us_per_op: float
    phase_dependency_us_per_op: float
    source_us_per_op: float
    p50_us: float
    p95_us: float
    max_us: float
    stages: tuple[StageCost, ...]
    attributions: tuple[StageCost, ...]
    resources: tuple[ResourceCost, ...]


@dataclass(frozen=True)
class VisualizationModel:
    scenarios: tuple[VisualScenario, ...]
    stage_keys: tuple[str, ...]
    attribution_keys: tuple[str, ...]
    resource_keys: tuple[str, ...]
    omitted_zero_stage_keys: tuple[str, ...]


def _fail(message: str) -> NoReturn:
    raise TimeBreakdownVisualizationError(message)


def _finite_nonnegative(value: object, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        _fail(f"{context} must be numeric: {error}")
    if not math.isfinite(parsed) or parsed < 0.0:
        _fail(f"{context} must be finite and non-negative")
    return parsed


def _integer_nonnegative(value: object, context: str) -> int:
    if isinstance(value, bool):
        _fail(f"{context} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        _fail(f"{context} must be a non-negative integer: {error}")
    if parsed < 0 or parsed != value:
        _fail(f"{context} must be a non-negative integer")
    return parsed


def _stage_group(scope: str, component: str) -> str:
    if scope in {"hbf.resident_mapping", "hbf.write_buffer"}:
        return "hbf_queue"
    if scope.startswith("hbf.ecc."):
        return "hbf_ecc"
    if scope == "hbm":
        if component in {
            "ingress_queue_wait_work_ns",
            "scheduler_queue_wait_work_ns",
        }:
            return "hbm_queue"
        if component in {
            "channel_transfer_work_ns",
            "tsv_transfer_work_ns",
            "hb_io_transfer_work_ns",
        }:
            return "hbm_transfer"
        return "hbm_core"
    if scope == "hbf":
        if component in {
            "ingress_queue_wait_work_ns",
            "scheduler_queue_wait_work_ns",
        }:
            return "hbf_queue"
        if component in {
            "address_mapping_work_ns",
            "translation_work_ns",
            "metadata_cache_work_ns",
            "mapping_dram_work_ns",
            "command_work_ns",
        }:
            return "hbf_metadata"
        if component in {
            "array_read_work_ns",
            "array_program_work_ns",
            "array_erase_work_ns",
            "program_verify_work_ns",
            "maintenance_work_ns",
        }:
            return "hbf_array"
        if component in {
            "ecc_queue_wait_work_ns",
            "ecc_response_latency_work_ns",
        }:
            return "hbf_ecc"
        return "hbf_transfer"
    if scope == "external_backing":
        if component in {
            "ingress_queue_wait_work_ns",
            "scheduler_queue_wait_work_ns",
        }:
            return "external_queue"
        if component == "command_work_ns":
            return "external_controller"
        if component in {
            "hb_io_transfer_work_ns",
            "transport_latency_work_ns",
        }:
            return "external_link"
        return "external_media"
    if scope == "base_die_link":
        return "base_die"
    if scope == "layer_streaming_controller":
        return "streaming"
    if scope == "cooperative_write_controller":
        return "cooperative"
    return "other"


def _component_label(scope: str, component: str) -> str:
    scope_label = {
        "hbm": "HBM",
        "hbf": "HBF",
        "hbf.write_buffer": "HBF write buffer",
        "hbf.resident_mapping": "HBF resident mapping",
        "hbf.ecc.decode": "HBF ECC decode",
        "hbf.ecc.encode": "HBF ECC encode",
        "external_backing": "External backing",
        "base_die_link": "Base-die link",
        "layer_streaming_controller": "Layer streaming controller",
        "cooperative_write_controller": "Cooperative write controller",
    }.get(scope, scope.replace("_", " ").title())
    external_labels = {
        "ingress_queue_wait_work_ns": "Per-channel queue-depth wait",
        "scheduler_queue_wait_work_ns": "Media / link serialization wait",
        "command_work_ns": "Host-link fixed latency",
        "array_read_work_ns": "Media read fixed latency",
        "array_program_work_ns": "Media write fixed latency",
        "channel_transfer_work_ns": "Media transfer",
        "hb_io_transfer_work_ns": "Host-link transfer",
    }
    component_label = (
        external_labels.get(component)
        if scope == "external_backing"
        else COMPONENT_LABELS.get(component)
    )
    if component_label is None:
        component_label = component.removesuffix("_work_ns").replace("_", " ").title()
    return f"{scope_label} · {component_label}"


def _row_identity(row: Mapping[str, object]) -> tuple[str, str]:
    return str(row["run_label"]), str(row["scenario"])


def _scenario_blocks(
    summaries: Sequence[SummaryInput],
) -> dict[tuple[str, str], Mapping[str, object]]:
    blocks: dict[tuple[str, str], Mapping[str, object]] = {}
    for source in summaries:
        raw_scenarios = source.summary.get("scenarios")
        if not isinstance(raw_scenarios, list):
            _fail(f"{source.label}: scenarios must be a list")
        for index, raw in enumerate(raw_scenarios):
            if not isinstance(raw, Mapping):
                _fail(f"{source.label}.scenarios[{index}] must be an object")
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                _fail(f"{source.label}.scenarios[{index}].name must be non-empty")
            identity = (source.label, name)
            if identity in blocks:
                _fail(f"duplicate scenario identity {source.label}/{name}")
            blocks[identity] = raw
    return blocks


def _display_labels(
    report: TimeBreakdownReport,
    aliases: Mapping[str, str],
) -> tuple[dict[tuple[str, str], str], set[str]]:
    per_run_count: dict[str, int] = {}
    for overview in report.overviews:
        per_run_count[overview.run_label] = per_run_count.get(overview.run_label, 0) + 1

    used_aliases: set[str] = set()
    provisional: dict[tuple[str, str], str] = {}
    for overview in report.overviews:
        identity = (overview.run_label, overview.scenario)
        selector = f"{overview.run_label}/{overview.scenario}"
        alias = aliases.get(selector)
        if alias is None and overview.case:
            alias = aliases.get(overview.case)
            if alias is not None:
                used_aliases.add(overview.case)
        elif alias is not None:
            used_aliases.add(selector)
        if alias is not None:
            label = alias
        elif overview.case:
            label = overview.case.upper()
        elif per_run_count[overview.run_label] == 1:
            label = overview.run_label
        else:
            label = overview.scenario
        provisional[identity] = label

    counts: dict[str, int] = {}
    for label in provisional.values():
        counts[label] = counts.get(label, 0) + 1
    for overview in report.overviews:
        identity = (overview.run_label, overview.scenario)
        if counts[provisional[identity]] > 1:
            provisional[identity] = (
                f"{overview.run_label} · {overview.scenario}"
            )
    return provisional, used_aliases


def build_visualization_model(
    summaries: Iterable[SummaryInput],
    aliases: Mapping[str, str] | None = None,
) -> VisualizationModel:
    """Validate summaries and convert their canonical report into chart data."""

    inputs = tuple(summaries)
    if not inputs:
        _fail("at least one summary is required")
    aliases = dict(aliases or {})
    report = build_time_breakdown_report(inputs)
    blocks = _scenario_blocks(inputs)
    labels, used_aliases = _display_labels(report, aliases)
    unused_aliases = sorted(set(aliases) - used_aliases)
    if unused_aliases:
        _fail(f"alias selector(s) matched no scenario: {', '.join(unused_aliases)}")

    rows_by_identity: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in report.rows:
        rows_by_identity.setdefault(_row_identity(row), []).append(row)

    overview_by_identity = {
        (overview.run_label, overview.scenario): overview
        for overview in report.overviews
    }
    ordered_overviews = []
    for source in inputs:
        raw_scenarios = source.summary["scenarios"]
        for raw in raw_scenarios:
            identity = (source.label, str(raw["name"]))
            overview = overview_by_identity.get(identity)
            if overview is None:
                _fail(f"validated report lost scenario {source.label}/{raw['name']}")
            ordered_overviews.append(overview)
    if len(ordered_overviews) != len(report.overviews):
        _fail("validated report scenario count changed during ordering")

    scenarios: list[VisualScenario] = []
    all_stage_keys: set[str] = set()
    nonzero_stage_keys: set[str] = set()
    all_attribution_keys: set[str] = set()
    all_resource_keys: set[str] = set()
    for overview in ordered_overviews:
        identity = (overview.run_label, overview.scenario)
        block = blocks.get(identity)
        if block is None:
            _fail(f"missing scenario block for {overview.run_label}/{overview.scenario}")
        rows = rows_by_identity.get(identity, [])
        by_metric = {
            (str(row["domain"]), str(row["scope"]), str(row["component"]),
             str(row["role"])): row
            for row in rows
        }

        def value_ns(
            domain: str, scope: str, component: str, role: str
        ) -> float:
            row = by_metric.get((domain, scope, component, role))
            if row is None:
                _fail(
                    f"{overview.run_label}/{overview.scenario}: missing "
                    f"{domain}.{scope}.{component} ({role})"
                )
            return _finite_nonnegative(
                row["value_ns"],
                f"{overview.run_label}/{overview.scenario}.{component}",
            )

        user_ops = overview.user_ops
        if user_ops <= 0:
            _fail(f"{overview.run_label}/{overview.scenario}: user_ops must be positive")
        admission_work = value_ns(
            "latency_work", "user_operations",
            "front_end_admission_wait_work_ns", "additive_work_component")
        service_work = value_ns(
            "latency_work", "user_operations",
            "service_to_user_completion_sum_work_ns", "additive_work_component")
        offered_work = value_ns(
            "latency_work", "user_operations",
            "offered_to_user_completion_sum_work_ns", "work_total")
        phase_dependency_work = value_ns(
            "latency_work", "source_operations",
            "phase_dependency_wait_work_ns", "additive_work_component")
        source_work = value_ns(
            "latency_work", "source_operations",
            "source_to_user_completion_sum_work_ns", "source_work_total")

        stage_costs: list[StageCost] = []
        for row in rows:
            if row["domain"] != "stage_work" or row["role"] not in PRIMARY_STAGE_ROLES:
                continue
            scope = str(row["scope"])
            component = str(row["component"])
            key = f"{scope}.{component}"
            value_us = _finite_nonnegative(
                row["per_user_op_ns"], f"{identity}.{key}.per_user_op_ns") / 1000.0
            group = _stage_group(scope, component)
            stage_costs.append(StageCost(
                key=key,
                label=_component_label(scope, component),
                group=group,
                value_us_per_op=value_us,
            ))
            all_stage_keys.add(key)
            if value_us > 0.0:
                nonzero_stage_keys.add(key)

        attribution_costs: list[StageCost] = []
        for row in rows:
            if (
                row["domain"] != "stage_work"
                or row["role"] not in NESTED_ATTRIBUTION_ROLES
            ):
                continue
            scope = str(row["scope"])
            component = str(row["component"])
            key = f"{scope}.{component}"
            value_us = _finite_nonnegative(
                row["per_user_op_ns"],
                f"{identity}.{key}.per_user_op_ns",
            ) / 1000.0
            attribution_costs.append(StageCost(
                key=key,
                label=_component_label(scope, component),
                group=_stage_group(scope, component),
                value_us_per_op=value_us,
            ))
            all_attribution_keys.add(key)

        resources: list[ResourceCost] = []
        for row in rows:
            if row["domain"] != "resource_busy" or row["role"] != "exclusive_resource_busy":
                continue
            key = str(row["scope"])
            utilization = _finite_nonnegative(
                row["utilization"], f"{identity}.{key}.utilization")
            if utilization > 1.0 + 1e-9:
                _fail(f"{identity}.{key}.utilization exceeds 1")
            resources.append(ResourceCost(
                key=key,
                label=RESOURCE_LABELS.get(key, key.replace("_", " ").title()),
                utilization=min(utilization, 1.0),
                average_parallelism=_finite_nonnegative(
                    row["average_parallelism"],
                    f"{identity}.{key}.average_parallelism"),
                resource_count=_integer_nonnegative(
                    row["resource_count"], f"{identity}.{key}.resource_count"),
            ))
            all_resource_keys.add(key)

        throughput_raw = block.get("user_completion_throughput_GBps")
        throughput = None if throughput_raw is None else _finite_nonnegative(
            throughput_raw, f"{identity}.user_completion_throughput_GBps")
        reads = _integer_nonnegative(block.get("reads", 0), f"{identity}.reads")
        writes = _integer_nonnegative(block.get("writes", 0), f"{identity}.writes")
        hbm_accesses = _integer_nonnegative(
            block.get("hbm_accesses", 0), f"{identity}.hbm_accesses")
        hbf_accesses = _integer_nonnegative(
            block.get("hbf_accesses", 0), f"{identity}.hbf_accesses")
        external_accesses = _integer_nonnegative(
            block.get("external_accesses", 0),
            f"{identity}.external_accesses")
        scenarios.append(VisualScenario(
            key=f"scenario-{len(scenarios)}",
            label=labels[identity],
            run_label=overview.run_label,
            scenario=overview.scenario,
            config=overview.config,
            ops=user_ops,
            reads=reads,
            writes=writes,
            throughput_GBps=throughput,
            hbm_accesses=hbm_accesses,
            hbf_accesses=hbf_accesses,
            external_accesses=external_accesses,
            offered_ms=overview.offered_arrival_span_ns / 1e6,
            completion_tail_ms=overview.post_offer_user_completion_tail_ns / 1e6,
            drain_ms=overview.drain_tail_ns / 1e6,
            makespan_ms=overview.makespan_ns / 1e6,
            admission_us_per_op=admission_work / user_ops / 1000.0,
            service_us_per_op=service_work / user_ops / 1000.0,
            offered_us_per_op=offered_work / user_ops / 1000.0,
            phase_dependency_us_per_op=(
                phase_dependency_work / user_ops / 1000.0
            ),
            source_us_per_op=source_work / user_ops / 1000.0,
            p50_us=value_ns(
                "latency_work", "offered_distribution", "p50_ns",
                "distribution_statistic") / 1000.0,
            p95_us=value_ns(
                "latency_work", "offered_distribution", "p95_ns",
                "distribution_statistic") / 1000.0,
            max_us=value_ns(
                "latency_work", "offered_distribution", "max_ns",
                "distribution_statistic") / 1000.0,
            stages=tuple(stage_costs),
            attributions=tuple(attribution_costs),
            resources=tuple(resources),
        ))

    if not scenarios:
        _fail("no scenarios were found")
    if len({scenario.label for scenario in scenarios}) != len(scenarios):
        _fail("scenario display labels must be unique")

    scope_order = {
        "hbm": 0,
        "hbf": 1,
        "base_die_link": 2,
        "layer_streaming_controller": 3,
        "cooperative_write_controller": 4,
    }
    component_order = {
        component: index for index, component in enumerate(
            DEVICE_STAGE_COMPONENTS
            + tuple(
                component
                for component in ARCHIVED_DEVICE_STAGE_COMPONENTS
                if component not in DEVICE_STAGE_COMPONENTS
            )
            + BASE_DIE_LINK_COMPONENTS
            + LAYER_STREAMING_CONTROLLER_FIELDS
            + ("full_wait_work_ns",)
        )
    }
    def stage_sort_key(key: str) -> tuple[int, int, str]:
        scope, component = key.split(".", 1)
        return (
            scope_order.get(scope, len(scope_order)),
            component_order.get(component, len(component_order)),
            key,
        )
    visible_stage_keys = tuple(sorted(
        nonzero_stage_keys,
        key=stage_sort_key,
    ))
    omitted = tuple(sorted(all_stage_keys - nonzero_stage_keys))
    attribution_keys = tuple(sorted(
        all_attribution_keys,
        key=lambda key: (
            0 if key.startswith("hbf.write_buffer.") else
            1 if key.startswith("hbf.cmt.") else
            2 if key.startswith("hbf.ecc.decode.") else
            3 if key.startswith("hbf.ecc.encode.") else
            4,
            key,
        ),
    ))
    resource_order = {name: index for index, name in enumerate(RESOURCE_NAMES)}
    resource_keys = tuple(sorted(
        all_resource_keys,
        key=lambda key: (resource_order.get(key, len(resource_order)), key),
    ))
    return VisualizationModel(
        scenarios=tuple(scenarios),
        stage_keys=visible_stage_keys,
        attribution_keys=attribution_keys,
        resource_keys=resource_keys,
        omitted_zero_stage_keys=omitted,
    )


def _fmt(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0:
        return f"{value:,.1f}"
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    if abs(value) >= 10.0:
        return f"{value:.2f}"
    if abs(value) >= 1.0:
        return f"{value:.3f}"
    if abs(value) >= 0.01:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _stage_maps(
    model: VisualizationModel,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    labels: dict[str, str] = {}
    groups: dict[str, str] = {}
    colors: dict[str, str] = {}
    for scenario in model.scenarios:
        for cost in scenario.stages + scenario.attributions:
            labels[cost.key] = cost.label
            groups[cost.key] = cost.group
            colors[cost.key] = STAGE_GROUP_BY_KEY[cost.group].css_color
    return labels, groups, colors


def _bar_lane(
    label: str,
    segments: Sequence[tuple[str, float, str]],
    maximum: float,
    total_label: str,
    aria_description: str,
    *,
    segment_unit: str | None = None,
) -> str:
    marks: list[str] = []
    cursor = 0.0
    for segment_label, value, css_color in segments:
        width = 0.0 if maximum <= 0.0 else 100.0 * value / maximum
        if value > 0.0:
            marks.append(
                '<span class="bar-segment" '
                f'style="--left:{cursor:.8f}%;--width:{width:.8f}%;'
                f'--segment-color:var({css_color})" '
                f'aria-label="{_esc(segment_label)} {_esc(_fmt(value))}"></span>'
            )
        cursor += width
    visible_values = ""
    if segment_unit is not None:
        visible_values = (
            '<div class="lane-details">'
            + "".join(
                '<span>'
                f'{_esc(segment_label)}: '
                f'<strong>{_esc(_fmt(value))} {_esc(segment_unit)}</strong>'
                '</span>'
                for segment_label, value, _css_color in segments
            )
            + '</div>'
        )
    return (
        '<div class="bar-lane">'
        f'<div class="lane-label">{_esc(label)}</div>'
        f'<div class="bar-track" role="img" aria-label="{_esc(aria_description)}">'
        + "".join(marks) + '</div>'
        f'<div class="lane-value">{_esc(total_label)}</div>'
        + visible_values
        + '</div>'
    )


def _legend(items: Sequence[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        '<span class="legend-item">'
        f'<span class="legend-swatch" style="--swatch:var({color})"></span>'
        f'{_esc(label)}</span>'
        for label, color in items
    ) + '</div>'


def render_html(
    model: VisualizationModel,
    *,
    title: str = "HBFSim time-cost visualization",
) -> str:
    """Return one self-contained, responsive HTML dashboard."""

    scenarios = model.scenarios
    stage_labels, stage_groups, stage_colors = _stage_maps(model)
    stage_values = {
        (scenario.key, cost.key): cost.value_us_per_op
        for scenario in scenarios for cost in scenario.stages
    }
    attribution_values = {
        (scenario.key, cost.key): cost.value_us_per_op
        for scenario in scenarios for cost in scenario.attributions
    }
    resource_values = {
        (scenario.key, resource.key): resource
        for scenario in scenarios for resource in scenario.resources
    }

    maximum_wall = max(scenario.makespan_ms for scenario in scenarios)
    maximum_latency = max(scenario.source_us_per_op for scenario in scenarios)
    group_totals: dict[tuple[str, str], float] = {}
    for scenario in scenarios:
        for cost in scenario.stages:
            group_totals[(scenario.key, cost.group)] = (
                group_totals.get((scenario.key, cost.group), 0.0)
                + cost.value_us_per_op
            )
    maximum_stage = max(
        (sum(group_totals.get((scenario.key, group.key), 0.0)
             for group in STAGE_GROUPS) for scenario in scenarios),
        default=0.0,
    )
    maximum_component = max(stage_values.values(), default=0.0)

    summary_rows: list[str] = []
    for scenario in scenarios:
        routed = (
            scenario.hbm_accesses
            + scenario.hbf_accesses
            + scenario.external_accesses
        )
        if routed:
            route = (
                f"{100.0 * scenario.hbm_accesses / routed:.1f}% / "
                f"{100.0 * scenario.hbf_accesses / routed:.1f}% / "
                f"{100.0 * scenario.external_accesses / routed:.1f}%"
            )
        else:
            route = "—"
        highest = max(
            scenario.resources,
            key=lambda item: item.utilization,
            default=None,
        )
        bottleneck = "—" if highest is None else (
            f"{highest.label} {_pct(highest.utilization)}"
        )
        throughput = "—" if scenario.throughput_GBps is None else (
            _fmt(scenario.throughput_GBps)
        )
        summary_rows.append(
            '<tr>'
            f'<th scope="row">{_esc(scenario.label)}</th>'
            f'<td>{_esc(scenario.scenario)}</td>'
            f'<td class="num">{scenario.ops:,}</td>'
            f'<td class="num">{_esc(route)}</td>'
            f'<td class="num">{_esc(throughput)}</td>'
            f'<td class="num">{_esc(_fmt(scenario.makespan_ms))}</td>'
            f'<td class="num">{_esc(_fmt(scenario.offered_us_per_op))}</td>'
            f'<td class="num">{_esc(_fmt(scenario.p95_us))}</td>'
            f'<td class="num">{_esc(_fmt(scenario.source_us_per_op))}</td>'
            f'<td>{_esc(bottleneck)}</td>'
            '</tr>'
        )

    wall_lanes = "".join(
        _bar_lane(
            scenario.label,
            (
                ("Offered-arrival span", scenario.offered_ms, "--wall-offered"),
                ("Post-offer completion tail", scenario.completion_tail_ms,
                 "--wall-completion"),
                ("Drain tail", scenario.drain_ms, "--wall-drain"),
            ),
            maximum_wall,
            f"{_fmt(scenario.makespan_ms)} ms",
            (
                f"{scenario.label}: offered span {_fmt(scenario.offered_ms)} ms, "
                f"completion tail {_fmt(scenario.completion_tail_ms)} ms, "
                f"drain {_fmt(scenario.drain_ms)} ms, total "
                f"{_fmt(scenario.makespan_ms)} ms"
            ),
            segment_unit="ms",
        )
        for scenario in scenarios
    )
    latency_lanes = "".join(
        _bar_lane(
            scenario.label,
            (
                ("Phase dependency wait", scenario.phase_dependency_us_per_op,
                 "--latency-phase"),
                ("Front-end admission wait", scenario.admission_us_per_op,
                 "--latency-admission"),
                ("Service (first credit to completion)",
                 scenario.service_us_per_op, "--latency-service"),
            ),
            maximum_latency,
            f"{_fmt(scenario.source_us_per_op)} µs/op",
            (
                f"{scenario.label}: phase dependency "
                f"{_fmt(scenario.phase_dependency_us_per_op)} microseconds per "
                f"operation plus admission {_fmt(scenario.admission_us_per_op)} "
                f"microseconds per operation plus service latency "
                f"{_fmt(scenario.service_us_per_op)} microseconds per operation"
            ),
            segment_unit="µs/op",
        )
        for scenario in scenarios
    )

    active_groups = [
        group for group in STAGE_GROUPS
        if any(group_totals.get((scenario.key, group.key), 0.0) > 0.0
               for scenario in scenarios)
    ]
    stage_lanes = "".join(
        _bar_lane(
            scenario.label,
            tuple(
                (
                    group.label,
                    group_totals.get((scenario.key, group.key), 0.0),
                    group.css_color,
                )
                for group in active_groups
            ),
            maximum_stage,
            (
                f"{_fmt(sum(group_totals.get((scenario.key, group.key), 0.0) for group in active_groups))} "
                "µs/op work"
            ),
            (
                f"{scenario.label}: accumulated overlapping stage work, "
                f"{_fmt(sum(group_totals.get((scenario.key, group.key), 0.0) for group in active_groups))} "
                "microseconds per user operation"
            ),
        )
        for scenario in scenarios
    )

    stage_rows: list[str] = []
    for key in model.stage_keys:
        label = stage_labels[key]
        color = stage_colors[key]
        group_label = STAGE_GROUP_BY_KEY[stage_groups[key]].label
        cells: list[str] = []
        for scenario in scenarios:
            value = stage_values.get((scenario.key, key), 0.0)
            intensity = (
                0.0 if value <= 0.0 or maximum_component <= 0.0 else
                math.log1p(value) / math.log1p(maximum_component)
            )
            cells.append(
                '<td class="heat-cell num" '
                f'style="--heat:{100.0 * intensity:.6f}%;'
                f'--heat-color:var({color})" '
                f'aria-label="{_esc(scenario.label)} { _esc(label)} '
                f'{_esc(_fmt(value))} microseconds per operation">'
                f'<span>{_esc(_fmt(value))}</span></td>'
            )
        stage_rows.append(
            '<tr>'
            f'<th scope="row"><span class="component-group">{_esc(group_label)}</span>'
            f'{_esc(label)}</th>'
            + "".join(cells) + '</tr>'
        )

    maximum_attribution = max(attribution_values.values(), default=0.0)
    attribution_rows: list[str] = []
    for key in model.attribution_keys:
        label = stage_labels[key]
        color = stage_colors[key]
        parent = (
            "HBF scheduler queue wait"
            if key.startswith((
                "hbf.write_buffer.",
                "hbf.resident_mapping.",
            ))
            else "HBF ECC parent"
        )
        cells = []
        for scenario in scenarios:
            value = attribution_values.get((scenario.key, key), 0.0)
            intensity = (
                0.0
                if value <= 0.0 or maximum_attribution <= 0.0
                else math.log1p(value) / math.log1p(maximum_attribution)
            )
            cells.append(
                '<td class="heat-cell num" '
                f'style="--heat:{100.0 * intensity:.6f}%;'
                f'--heat-color:var({color})" '
                f'aria-label="{_esc(scenario.label)} {_esc(label)} '
                f'{_esc(_fmt(value))} microseconds per operation; '
                f'already included in {_esc(parent)}">'
                f'<span>{_esc(_fmt(value))}</span></td>'
            )
        attribution_rows.append(
            '<tr>'
            f'<th scope="row"><span class="component-group">'
            f'Included in {_esc(parent)}</span>{_esc(label)}</th>'
            + "".join(cells) + '</tr>'
        )

    resource_labels: dict[str, str] = {}
    for scenario in scenarios:
        for resource in scenario.resources:
            resource_labels[resource.key] = resource.label
    resource_rows: list[str] = []
    for key in model.resource_keys:
        cells = []
        for scenario in scenarios:
            resource = resource_values.get((scenario.key, key))
            if resource is None:
                cells.append('<td class="num empty">—</td>')
                continue
            cells.append(
                '<td class="util-cell num" '
                f'style="--util:{100.0 * resource.utilization:.6f}%" '
                f'aria-label="{_esc(scenario.label)} {_esc(resource.label)} '
                f'utilization {_esc(_pct(resource.utilization))}, average parallelism '
                f'{_esc(_fmt(resource.average_parallelism))} of {resource.resource_count}">'
                f'<span>{_esc(_pct(resource.utilization))}</span>'
                f'<small>P={_esc(_fmt(resource.average_parallelism))}/{resource.resource_count}</small>'
                '</td>'
            )
        resource_rows.append(
            f'<tr><th scope="row">{_esc(resource_labels[key])}</th>'
            + "".join(cells) + '</tr>'
        )

    scenario_headers = "".join(
        f'<th scope="col">{_esc(scenario.label)}</th>' for scenario in scenarios
    )
    omitted_note = ""
    if model.omitted_zero_stage_keys:
        omitted_note = (
            '<p class="muted zero-note">Universally zero primary stages omitted: '
            f'{len(model.omitted_zero_stage_keys)}. They remain present in the canonical CSV.</p>'
        )

    css = r"""
    :root {
      color-scheme: light dark;
      --bg: #fbfcfe; --fg: #172033; --muted: #5f6b7d; --line: #d7deea;
      --panel: #ffffff; --track: #edf1f6; --table-alt: #f6f8fb;
      --wall-offered: #6b7a90; --wall-completion: #2563a8; --wall-drain: #b45309;
      --latency-phase: #b45309; --latency-admission: #7c3aed;
      --latency-service: #0891b2;
      --stage-hbm-queue: #1d4ed8; --stage-hbm-core: #3b82f6;
      --stage-hbm-transfer: #7dd3fc; --stage-hbf-queue: #c2410c;
      --stage-hbf-metadata: #f97316; --stage-hbf-array: #f59e0b;
      --stage-hbf-transfer: #fbbf24; --stage-hbf-ecc: #db2777;
      --stage-external-queue: #0e7490; --stage-external-media: #06b6d4;
      --stage-external-link: #67e8f9;
      --stage-base-die: #0f766e; --stage-streaming: #7c3aed;
      --stage-cooperative: #9333ea; --stage-other: #64748b;
      --util-color: #059669;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f1420; --fg: #e7edf7; --muted: #a8b3c5; --line: #344057;
        --panel: #151c2a; --track: #273247; --table-alt: #192232;
        --wall-offered: #94a3b8; --wall-completion: #60a5fa; --wall-drain: #fb923c;
        --latency-phase: #fb923c; --latency-admission: #a78bfa;
        --latency-service: #22d3ee;
        --stage-hbm-queue: #60a5fa; --stage-hbm-core: #93c5fd;
        --stage-hbm-transfer: #bae6fd; --stage-hbf-queue: #fb923c;
        --stage-hbf-metadata: #fdba74; --stage-hbf-array: #fbbf24;
        --stage-hbf-transfer: #fde68a; --stage-hbf-ecc: #f472b6;
        --stage-external-queue: #22d3ee; --stage-external-media: #67e8f9;
        --stage-external-link: #a5f3fc;
        --stage-base-die: #5eead4; --stage-streaming: #c4b5fd;
        --stage-cooperative: #d8b4fe; --stage-other: #94a3b8;
        --util-color: #34d399;
      }
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: 100%; padding: 24px clamp(14px, 3vw, 44px) 48px; }
    h1, h2 { font-weight: 500; margin: 0; }
    h1 { font-size: clamp(1.45rem, 3vw, 2rem); }
    h2 { font-size: 1.15rem; margin-top: 30px; }
    p { margin: 7px 0 0; }
    .subtitle, .muted { color: var(--muted); }
    .contract { margin-top: 16px; padding: 12px 14px; border-left: 4px solid var(--wall-completion); background: var(--panel); }
    .table-wrap { overflow-x: auto; margin-top: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }
    thead th { color: var(--muted); font-weight: 500; white-space: nowrap; }
    tbody th { font-weight: 500; }
    tbody tr:nth-child(even) { background: var(--table-alt); }
    .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .chart { margin-top: 13px; display: grid; gap: 7px; }
    .bar-lane { display: grid; grid-template-columns: minmax(78px, 135px) minmax(150px, 1fr) minmax(92px, 140px); gap: 10px; align-items: center; min-height: 28px; }
    .lane-label { font-weight: 500; overflow-wrap: anywhere; }
    .lane-value { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
    .lane-details { grid-column: 2 / -1; display: flex; flex-wrap: wrap; gap: 3px 16px; color: var(--muted); font-size: 0.76rem; font-variant-numeric: tabular-nums; }
    .lane-details strong { color: var(--fg); font-weight: 500; }
    .bar-track { position: relative; height: 18px; background: var(--track); border-radius: 3px; overflow: hidden; }
    .bar-segment { position: absolute; left: var(--left); width: max(var(--width), 1px); height: 100%; background: var(--segment-color); }
    .legend { display: flex; flex-wrap: wrap; gap: 7px 16px; margin-top: 10px; color: var(--muted); font-size: 0.83rem; }
    .legend-item { display: inline-flex; gap: 6px; align-items: center; }
    .legend-swatch { width: 12px; height: 12px; background: var(--swatch); border-radius: 2px; }
    .component-group { display: block; color: var(--muted); font-size: 0.73rem; font-weight: 400; }
    .heat-cell, .util-cell { position: relative; isolation: isolate; min-width: 88px; }
    .heat-cell::before, .util-cell::before { content: ""; position: absolute; z-index: -1; inset: 3px; border-radius: 2px; transform-origin: left; }
    .heat-cell::before { width: var(--heat); background: var(--heat-color); opacity: 0.26; }
    .util-cell::before { width: var(--util); background: var(--util-color); opacity: 0.23; }
    .util-cell small { display: block; color: var(--muted); font-size: 0.72rem; }
    .empty { color: var(--muted); }
    .zero-note { font-size: 0.82rem; }
    footer { margin-top: 30px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--line); padding-top: 10px; }
    @media (max-width: 560px) {
      main { padding-inline: 12px; }
      .bar-lane { grid-template-columns: 72px minmax(100px, 1fr); gap: 7px; }
      .lane-value { grid-column: 2; text-align: left; font-size: 0.8rem; }
      .lane-details { grid-column: 2; }
    }
    @media print {
      :root { --bg: white; --fg: black; --panel: white; --track: #eee; --line: #bbb; --table-alt: #f5f5f5; }
      main { padding: 10mm; }
      .table-wrap { overflow: visible; }
    }
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)}</title>
  <style>{css}</style>
</head>
<body>
<main>
  <header>
    <h1>{_esc(title)}</h1>
    <p class="subtitle">{len(scenarios)} validated scenario(s) · time-breakdown contract v2</p>
    <p class="contract"><strong>Read the three views separately.</strong> Wall-clock segments are additive. Source-to-completion work contains phase dependency wait, front-end admission wait, and first-credit-to-completion service latency; it is additive only within its own per-operation bar. The primary online latency is offer-to-completion. Stage work can overlap across requests and resources and must not be summed into elapsed time.</p>
  </header>

  <section aria-labelledby="overview-title">
    <h2 id="overview-title">Outcome and latency</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Case</th><th>Scenario</th><th class="num">Ops</th><th class="num">HBM / HBF / External</th><th class="num">GB/s</th><th class="num">Makespan ms</th><th class="num">Offer→done avg µs</th><th class="num">P95 µs</th><th class="num">Source→done avg µs</th><th>Bottleneck</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table></div>
  </section>

  <section aria-labelledby="wall-title">
    <h2 id="wall-title">Additive wall clock</h2>
    {_legend((("Offered-arrival span", "--wall-offered"), ("Post-offer completion", "--wall-completion"), ("Drain", "--wall-drain")))}
    <div class="chart">{wall_lanes}</div>
  </section>

  <section aria-labelledby="latency-title">
    <h2 id="latency-title">Per-operation source-to-completion work</h2>
    {_legend((("Phase dependency wait", "--latency-phase"), ("Front-end admission wait", "--latency-admission"), ("Service (first credit to completion)", "--latency-service")))}
    <div class="chart">{latency_lanes}</div>
  </section>

  <section aria-labelledby="stage-title">
    <h2 id="stage-title">Overlapping stage work</h2>
    <p class="muted">Accumulated internal work normalized by user operations; this is demand on the model, not serialized latency.</p>
    {_legend(tuple((group.label, group.css_color) for group in active_groups))}
    <div class="chart">{stage_lanes}</div>
  </section>

  <section aria-labelledby="detail-title">
    <h2 id="detail-title">Stage-cost detail · µs per user operation</h2>
    <p class="muted">Fill uses one global logarithmic scale so small control-path costs remain visible; printed values are exact rounded values.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Primary stage</th>{scenario_headers}</tr></thead>
      <tbody>{''.join(stage_rows)}</tbody>
    </table></div>
    {omitted_note}
  </section>

  <section aria-labelledby="attribution-title">
    <h2 id="attribution-title">Nested wait / direction attribution · µs per user operation</h2>
    <p class="muted">Diagnostic children are already included in the named primary parent. Use them to explain the parent; never add them to the primary-stage total.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Non-additive attribution</th>{scenario_headers}</tr></thead>
      <tbody>{''.join(attribution_rows)}</tbody>
    </table></div>
  </section>

  <section aria-labelledby="resource-title">
    <h2 id="resource-title">Resource utilization and average parallelism</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Resource</th>{scenario_headers}</tr></thead>
      <tbody>{''.join(resource_rows)}</tbody>
    </table></div>
  </section>

  <footer>Self-contained HBFSim report. Zero write/GC cost remains visible as zero in the canonical CSV even when omitted from the stage heatmap.</footer>
</main>
</body>
</html>
"""


def _atomic_write(path: Path, payload: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_visualization(
    summaries: Iterable[SummaryInput],
    output: Path,
    *,
    aliases: Mapping[str, str] | None = None,
    title: str = "HBFSim time-cost visualization",
) -> VisualizationModel:
    """Validate inputs, render the dashboard, and atomically publish it."""

    model = build_visualization_model(summaries, aliases)
    _atomic_write(Path(output), render_html(model, title=title))
    return model


def _parse_summary_spec(spec: str) -> SummaryInput:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        if not label or not raw_path:
            raise argparse.ArgumentTypeError("summary must be [LABEL=]PATH")
        return load_summary_input(Path(raw_path), label=label)
    return load_summary_input(Path(spec))


def _parse_aliases(specs: Sequence[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise argparse.ArgumentTypeError("alias must be SELECTOR=DISPLAY")
        selector, display = spec.split("=", 1)
        if not selector or not display:
            raise argparse.ArgumentTypeError("alias must be SELECTOR=DISPLAY")
        if selector in aliases:
            raise argparse.ArgumentTypeError(f"duplicate alias selector: {selector}")
        aliases[selector] = display
    return aliases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        metavar="[LABEL=]PATH",
        help="repeat for every schema-v9/v16 summary in the comparison",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="SELECTOR=DISPLAY",
        help=(
            "rename a case or RUN/SCENARIO in the visual; repeat as needed"
        ),
    )
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--title", default="HBFSim time-cost visualization",
        help="document title shown above the charts",
    )
    args = parser.parse_args()
    try:
        aliases = _parse_aliases(args.alias)
        summaries = [_parse_summary_spec(spec) for spec in args.summary]
        model = write_visualization(
            summaries, args.output, aliases=aliases, title=args.title)
    except (TimeBreakdownError, TimeBreakdownVisualizationError) as error:
        parser.error(str(error))
    print(
        f"wrote {args.output}: {len(model.scenarios)} scenarios, "
        f"{len(model.stage_keys)} non-zero primary stages, "
        f"{len(model.attribution_keys)} nested attribution rows, "
        f"{len(model.resource_keys)} resources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
