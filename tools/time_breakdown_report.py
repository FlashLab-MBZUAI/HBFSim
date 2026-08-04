#!/usr/bin/env python3
"""Render auditable, suite-level time-breakdown reports from summary v9/v16.

The canonical simulator summary deliberately keeps four different clocks apart:

* wall-clock spans are the only elapsed-time terms that may be added;
* latency work is summed across user operations and may overlap;
* stage work is accumulated device/controller work and may overlap;
* resource busy time is interpreted against resource capacity-time.

This module validates that contract before it writes anything.  It is both an
importable API for E2E runners and a small CLI for post-processing one or more
existing summaries.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, NoReturn


SUMMARY_SCHEMA = {"name": "hbfsim.scenario_compare.summary", "version": 16}
ARCHIVED_SUMMARY_SCHEMA = {
    "name": "hbfsim.scenario_compare.summary",
    "version": 9,
}
TIME_CONTRACT_VERSION = 2
# Timing work is accumulated independently by several simulator owners and is
# then reduced into parent totals.  At long-trace scale, changing the reduction
# order can legitimately move the last few decimal places even though both
# values originate from the same events.  Keep the identity check fail-closed,
# but allow 0.1 parts per billion of scale-dependent roundoff.  For example, a
# 4.5e8 ns total gets a tolerance below 0.05 ns, so millinanosecond reduction
# noise passes while a 1 ns accounting error still fails.
IDENTITY_REL_TOL = 1e-10
IDENTITY_ABS_TOL = 1e-9
AVERAGE_IDENTITY_REL_TOL = 1e-9
AVERAGE_IDENTITY_ABS_TOL = 1e-9
TIME_SEMANTICS = {
    "wall_clock": "additive_non_overlapping_spans",
    "latency_work": "sum_across_user_operations_may_overlap",
    "stage_work":
        "hierarchical_device_work_may_overlap_do_not_sum_parent_and_children",
    "controller_work": "parent_totals_and_attribution_views_may_overlap",
    "resource_busy": "exclusive_per_resource_then_aggregated",
}

WALL_FRONTIERS = (
    "trace_origin_ns",
    "last_offered_arrival_ns",
    "last_user_completion_ns",
    "quiescent_finish_ns",
)
WALL_SEGMENTS = (
    "offered_arrival_span_ns",
    "post_offer_user_completion_tail_ns",
    "drain_tail_ns",
)
WALL_TOTALS = (
    "user_completion_span_ns",
    "makespan_ns",
)
LATENCY_WORK_COMPONENTS = (
    "service_to_user_completion_sum_work_ns",
    "front_end_admission_wait_work_ns",
)
LATENCY_TOTAL = "offered_to_user_completion_sum_work_ns"
PHASE_DEPENDENCY_FIELDS = (
    "phase_barriers",
    "phase_dependency_waited_ops",
    "phase_dependency_wait_work_ns",
    "phase_dependency_max_wait_ns",
    "source_to_user_completion_sum_work_ns",
)
SOURCE_LATENCY_TOTAL = "source_to_user_completion_sum_work_ns"
LATENCY_STATISTICS = (
    "average_ns",
    "p50_ns",
    "p95_ns",
    "max_ns",
)
SERVICE_LATENCY_STATISTICS = (
    "service_average_ns",
    "service_p50_ns",
    "service_p95_ns",
    "service_max_ns",
)
SOURCE_LATENCY_STATISTICS = (
    "source_average_ns",
    "source_p50_ns",
    "source_p95_ns",
    "source_max_ns",
)
DEVICE_STAGE_COMPONENTS = (
    "ingress_queue_wait_work_ns",
    "scheduler_queue_wait_work_ns",
    "address_mapping_work_ns",
    "translation_work_ns",
    "mapping_dram_work_ns",
    "refresh_stall_work_ns",
    "precharge_work_ns",
    "activation_work_ns",
    "command_work_ns",
    "array_read_work_ns",
    "array_program_work_ns",
    "array_erase_work_ns",
    "program_verify_work_ns",
    "media_lane_transfer_work_ns",
    "page_buffer_work_ns",
    "sram_staging_work_ns",
    "channel_transfer_work_ns",
    "tsv_transfer_work_ns",
    "hb_io_transfer_work_ns",
    "transport_latency_work_ns",
    "ecc_queue_wait_work_ns",
    "ecc_response_latency_work_ns",
    "maintenance_work_ns",
)
# Schema v9 results remain useful because long sweeps are expensive to replay,
# and they already carry the same timing contract v2.  Validate their native
# metadata-cache stage instead of synthesizing v16 resident-mapping metrics.
ARCHIVED_DEVICE_STAGE_COMPONENTS = tuple(
    "metadata_cache_work_ns" if component == "mapping_dram_work_ns" else component
    for component in DEVICE_STAGE_COMPONENTS
    if component != "transport_latency_work_ns"
)
DEVICE_STAGE_TOTAL = "total_overlapping_work_ns"
BASE_DIE_LINK_COMPONENTS = (
    "read_queue_wait_work_ns",
    "write_queue_wait_work_ns",
    "read_serialization_work_ns",
    "write_serialization_work_ns",
    "read_fixed_latency_work_ns",
    "write_fixed_latency_work_ns",
)
LAYER_STREAMING_CONTROLLER_FIELDS = (
    "backing_admission_wait_work_ns",
    "backing_admission_max_wait_ns",
    "user_wait_work_ns",
    "user_max_wait_ns",
    "exposed_prefetch_ns",
    "hidden_prefetch_ns",
    "buffer_reuse_wait_work_ns",
)
LAYER_STREAMING_CONTROLLER_PARENTS = {
    "backing_admission_wait_work_ns",
    "user_wait_work_ns",
    "exposed_prefetch_ns",
    "hidden_prefetch_ns",
    "buffer_reuse_wait_work_ns",
}
LAYER_STREAMING_CONTROLLER_STATISTICS = {
    "backing_admission_max_wait_ns",
    "user_max_wait_ns",
}
COOPERATIVE_CONTROLLER_FIELDS = ("full_wait_work_ns",)
RESOURCE_NAMES = (
    "hbm_data_bus",
    "hbf_logic_ingress",
    "hbf_mapping_dram_issue",
    "hbf_plane_media",
    "hbf_media_lane",
    "hbf_subarray",
    "hbf_page_buffer_bank",
    "hbf_flash_source_queue",
    "hbf_channel_command",
    "hbf_channel_data",
    "hbf_tsv",
    "hbf_sram",
    "hbf_hbio_command",
    "hbf_hbio_data",
    "hbf_sequencer",
    "hbf_ecc_issue",
    "base_die_link_read",
    "base_die_link_write",
)
OPTIONAL_RESOURCE_NAMES = (
    "external_controller",
    "external_media",
    "external_link_m2s",
    "external_link_s2m",
)
ARCHIVED_RESOURCE_NAMES = tuple(
    name for name in RESOURCE_NAMES if name != "hbf_mapping_dram_issue"
)

CSV_FIELDS = (
    "run_label",
    "case",
    "config",
    "source_summary",
    "scenario",
    "user_ops",
    "domain",
    "scope",
    "component",
    "role",
    "value_ns",
    "value_ms",
    "per_user_op_ns",
    "share",
    "share_basis",
    "resource_count",
    "active_span_ns",
    "capacity_time_ns",
    "utilization",
    "average_parallelism",
    "note",
)


class TimeBreakdownError(ValueError):
    """A summary cannot be interpreted under the canonical timing contract."""


@dataclass(frozen=True)
class SummaryInput:
    """One summary plus runner context used to identify its scenario rows.

    ``scenario_metadata`` may provide ``case`` and ``config`` strings for each
    scenario name.  This supports summaries that contain multiple experiment
    cases without copying or mutating the canonical JSON.
    """

    label: str
    summary: Mapping[str, object]
    source: str = ""
    scenario_metadata: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict)


@dataclass(frozen=True)
class ScenarioOverview:
    run_label: str
    case: str
    config: str
    source_summary: str
    scenario: str
    user_ops: int
    offered_arrival_span_ns: float
    post_offer_user_completion_tail_ns: float
    user_completion_span_ns: float
    drain_tail_ns: float
    makespan_ns: float
    average_ns: float
    p95_ns: float
    largest_stage: str | None
    largest_stage_work_ns: float
    highest_utilization_resource: str | None
    highest_resource_utilization: float


@dataclass(frozen=True)
class TimeBreakdownReport:
    rows: tuple[Mapping[str, object], ...]
    overviews: tuple[ScenarioOverview, ...]


def _fail(path: str, message: str) -> NoReturn:
    raise TimeBreakdownError(f"{path}: {message}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(path, "must be an object")
    return value


def _required_mapping(
    container: Mapping[str, object], key: str, path: str
) -> Mapping[str, object]:
    if key not in container:
        _fail(path, f"missing {key}")
    return _mapping(container[key], f"{path}.{key}")


def _required_list(
    container: Mapping[str, object], key: str, path: str
) -> list[object]:
    if key not in container or not isinstance(container[key], list):
        _fail(path, f"{key} must be an array")
    return container[key]


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _fail(path, "must be finite and non-negative")
    return result


def _required_number(
    container: Mapping[str, object], key: str, path: str
) -> float:
    if key not in container:
        _fail(path, f"missing {key}")
    return _number(container[key], f"{path}.{key}")


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(path, "must be a non-negative integer")
    return value


def _required_integer(
    container: Mapping[str, object], key: str, path: str
) -> int:
    if key not in container:
        _fail(path, f"missing {key}")
    return _integer(container[key], f"{path}.{key}")


def _required_bool(
    container: Mapping[str, object], key: str, path: str
) -> bool:
    if key not in container or not isinstance(container[key], bool):
        _fail(path, f"{key} must be a boolean")
    return container[key]


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=IDENTITY_REL_TOL,
        abs_tol=IDENTITY_ABS_TOL,
    )


def _expect_close(left: float, right: float, path: str, identity: str) -> None:
    if not _close(left, right):
        _fail(path, f"{identity} ({left!r} != {right!r})")


def _expect_average_close(
    left: float, right: float, path: str, identity: str
) -> None:
    if not math.isclose(
        left,
        right,
        rel_tol=AVERAGE_IDENTITY_REL_TOL,
        abs_tol=AVERAGE_IDENTITY_ABS_TOL,
    ):
        _fail(path, f"{identity} ({left!r} != {right!r})")


def _device_stage_components(schema_version: int) -> tuple[str, ...]:
    if schema_version == ARCHIVED_SUMMARY_SCHEMA["version"]:
        return ARCHIVED_DEVICE_STAGE_COMPONENTS
    return DEVICE_STAGE_COMPONENTS


def _resource_names(schema_version: int) -> tuple[str, ...]:
    if schema_version == ARCHIVED_SUMMARY_SCHEMA["version"]:
        return ARCHIVED_RESOURCE_NAMES
    return RESOURCE_NAMES


def _validate_wall(time: Mapping[str, object], path: str) -> Mapping[str, float]:
    wall = _required_mapping(time, "wall_clock_ns", path)
    values = {
        key: _required_number(wall, key, f"{path}.wall_clock_ns")
        for key in WALL_FRONTIERS + WALL_SEGMENTS + WALL_TOTALS
    }
    origin = values["trace_origin_ns"]
    last_offer = values["last_offered_arrival_ns"]
    last_user = values["last_user_completion_ns"]
    finish = values["quiescent_finish_ns"]
    if not origin <= last_offer <= last_user <= finish:
        _fail(f"{path}.wall_clock_ns", "frontiers violate causal order")
    _expect_close(
        last_offer - origin,
        values["offered_arrival_span_ns"],
        f"{path}.wall_clock_ns",
        "last offered arrival minus trace origin must equal offered span",
    )
    _expect_close(
        last_user - last_offer,
        values["post_offer_user_completion_tail_ns"],
        f"{path}.wall_clock_ns",
        "last user completion minus last offer must equal post-offer tail",
    )
    _expect_close(
        last_user - origin,
        values["user_completion_span_ns"],
        f"{path}.wall_clock_ns",
        "last user completion minus trace origin must equal user span",
    )
    _expect_close(
        finish - last_user,
        values["drain_tail_ns"],
        f"{path}.wall_clock_ns",
        "quiescent finish minus last user completion must equal drain",
    )
    _expect_close(
        finish - origin,
        values["makespan_ns"],
        f"{path}.wall_clock_ns",
        "quiescent finish minus trace origin must equal makespan",
    )
    _expect_close(
        values["offered_arrival_span_ns"]
        + values["post_offer_user_completion_tail_ns"],
        values["user_completion_span_ns"],
        f"{path}.wall_clock_ns",
        "offered span plus post-offer tail must equal user span",
    )
    _expect_close(
        values["user_completion_span_ns"] + values["drain_tail_ns"],
        values["makespan_ns"],
        f"{path}.wall_clock_ns",
        "user span plus drain must equal makespan",
    )
    return values


def _validate_latency(
    time: Mapping[str, object], expected_ops: int, path: str
) -> Mapping[str, float | int | str]:
    latency = _required_mapping(time, "latency_work", path)
    if latency.get("basis") != "offered_to_user_completion":
        _fail(f"{path}.latency_work.basis", "unsupported latency basis")
    user_count = _required_integer(latency, "user_count", f"{path}.latency_work")
    if user_count != expected_ops:
        _fail(
            f"{path}.latency_work.user_count",
            f"{user_count} does not match scenario ops {expected_ops}",
        )
    waited = _required_integer(
        latency, "front_end_admission_waited_ops", f"{path}.latency_work"
    )
    if waited > user_count:
        _fail(
            f"{path}.latency_work.front_end_admission_waited_ops",
            "cannot exceed user_count",
        )
    values: dict[str, float | int | str] = {
        "basis": "offered_to_user_completion",
        "user_count": user_count,
        "front_end_admission_waited_ops": waited,
    }
    for key in (
        LATENCY_WORK_COMPONENTS
        + (LATENCY_TOTAL, "front_end_admission_max_wait_ns")
        + LATENCY_STATISTICS
        + SERVICE_LATENCY_STATISTICS
        + SOURCE_LATENCY_STATISTICS
    ):
        values[key] = _required_number(latency, key, f"{path}.latency_work")
    service = float(values["service_to_user_completion_sum_work_ns"])
    admission = float(values["front_end_admission_wait_work_ns"])
    offered = float(values[LATENCY_TOTAL])
    _expect_close(
        service + admission,
        offered,
        f"{path}.latency_work",
        "service work plus admission wait must equal offered work",
    )
    phase_barriers = _required_integer(
        latency, "phase_barriers", f"{path}.latency_work"
    )
    phase_waited = _required_integer(
        latency, "phase_dependency_waited_ops", f"{path}.latency_work"
    )
    if phase_waited > user_count:
        _fail(
            f"{path}.latency_work.phase_dependency_waited_ops",
            "cannot exceed user_count",
        )
    phase_wait = _required_number(
        latency, "phase_dependency_wait_work_ns", f"{path}.latency_work"
    )
    phase_max = _required_number(
        latency, "phase_dependency_max_wait_ns", f"{path}.latency_work"
    )
    source_total = _required_number(
        latency, SOURCE_LATENCY_TOTAL, f"{path}.latency_work"
    )
    if (phase_waited == 0) != (phase_wait == 0.0):
        _fail(
            f"{path}.latency_work",
            "phase waited-op count and wait work disagree",
        )
    if (phase_waited == 0) != (phase_max == 0.0):
        _fail(
            f"{path}.latency_work",
            "phase waited-op count and maximum wait disagree",
        )
    if phase_max > phase_wait and not _close(phase_max, phase_wait):
        _fail(
            f"{path}.latency_work.phase_dependency_max_wait_ns",
            "cannot exceed phase dependency wait work",
        )
    _expect_close(
        offered + phase_wait,
        source_total,
        f"{path}.latency_work",
        "offered work plus phase dependency wait must equal source work",
    )
    values.update({
        "phase_barriers": phase_barriers,
        "phase_dependency_waited_ops": phase_waited,
        "phase_dependency_wait_work_ns": phase_wait,
        "phase_dependency_max_wait_ns": phase_max,
        SOURCE_LATENCY_TOTAL: source_total,
    })
    expected_average = offered / user_count if user_count else 0.0
    _expect_average_close(
        expected_average,
        float(values["average_ns"]),
        f"{path}.latency_work.average_ns",
        "average must equal offered work divided by user_count",
    )
    expected_service_average = service / user_count if user_count else 0.0
    _expect_average_close(
        expected_service_average,
        float(values["service_average_ns"]),
        f"{path}.latency_work.service_average_ns",
        "service average must equal service work divided by user_count",
    )
    expected_source_average = source_total / user_count if user_count else 0.0
    _expect_average_close(
        expected_source_average,
        float(values["source_average_ns"]),
        f"{path}.latency_work.source_average_ns",
        "source average must equal source work divided by user_count",
    )
    for prefix in ("", "service_", "source_"):
        average = float(values[f"{prefix}average_ns"])
        p50 = float(values[f"{prefix}p50_ns"])
        p95 = float(values[f"{prefix}p95_ns"])
        maximum = float(values[f"{prefix}max_ns"])
        if p50 > p95 or p95 > maximum or average > maximum:
            _fail(
                f"{path}.latency_work",
                f"{prefix or 'offered_'}latency statistics are not ordered",
            )
    if (
        float(values["service_average_ns"]) > float(values["average_ns"])
        or float(values["average_ns"]) > float(values["source_average_ns"])
    ):
        _fail(
            f"{path}.latency_work",
            "service/offered/source average latencies are not causal",
        )
    return values


def _validate_device_stage(
    stage: Mapping[str, object],
    device: str,
    path: str,
    schema_version: int,
) -> Mapping[str, object]:
    components = _device_stage_components(schema_version)
    values: dict[str, object] = {
        key: _required_number(stage, key, path) for key in components
    }
    total = _required_number(stage, DEVICE_STAGE_TOTAL, path)
    _expect_close(
        sum(float(values[key]) for key in components),
        total,
        f"{path}.{DEVICE_STAGE_TOTAL}",
        "device component work must sum to total_overlapping_work_ns",
    )
    values[DEVICE_STAGE_TOTAL] = total
    if device != "hbf":
        return values

    write_buffer = _required_mapping(stage, "write_buffer", path)
    slot_waited_ops = _required_integer(
        write_buffer, "slot_waited_ops", f"{path}.write_buffer"
    )
    slot_wait = _required_number(
        write_buffer, "slot_wait_work_ns", f"{path}.write_buffer"
    )
    if slot_wait > float(values["scheduler_queue_wait_work_ns"]) and not _close(
        slot_wait, float(values["scheduler_queue_wait_work_ns"])
    ):
        _fail(
            f"{path}.write_buffer.slot_wait_work_ns",
            "named subset exceeds scheduler queue-wait parent",
        )
    values["write_buffer"] = {
        "slot_waited_ops": slot_waited_ops,
        "slot_wait_work_ns": slot_wait,
    }

    if schema_version == SUMMARY_SCHEMA["version"]:
        resident = _required_mapping(stage, "resident_mapping", path)
        resident_values: dict[str, float | int] = {
            "table_bytes": _required_integer(
                resident, "table_bytes", f"{path}.resident_mapping"
            ),
            "table_bytes_per_stack": _required_integer(
                resident, "table_bytes_per_stack", f"{path}.resident_mapping"
            ),
            "pages_per_stack": _required_integer(
                resident, "pages_per_stack", f"{path}.resident_mapping"
            ),
            "lookup_ops": _required_integer(
                resident, "lookup_ops", f"{path}.resident_mapping"
            ),
            "user_lookup_ops": _required_integer(
                resident, "user_lookup_ops", f"{path}.resident_mapping"
            ),
            "gc_lookup_ops": _required_integer(
                resident, "gc_lookup_ops", f"{path}.resident_mapping"
            ),
            "update_ops": _required_integer(
                resident, "update_ops", f"{path}.resident_mapping"
            ),
            "user_update_ops": _required_integer(
                resident, "user_update_ops", f"{path}.resident_mapping"
            ),
            "gc_update_ops": _required_integer(
                resident, "gc_update_ops", f"{path}.resident_mapping"
            ),
            "dram_waited_ops": _required_integer(
                resident, "dram_waited_ops", f"{path}.resident_mapping"
            ),
            "dram_wait_work_ns": _required_number(
                resident, "dram_wait_work_ns", f"{path}.resident_mapping"
            ),
            "dram_wait_max_ns": _required_number(
                resident, "dram_wait_max_ns", f"{path}.resident_mapping"
            ),
            "dram_issue_busy_ns": _required_number(
                resident, "dram_issue_busy_ns", f"{path}.resident_mapping"
            ),
            "dram_resources": _required_integer(
                resident, "dram_resources", f"{path}.resident_mapping"
            ),
        }
        if resident_values["dram_resources"] == 0:
            _fail(
                f"{path}.resident_mapping.dram_resources",
                "resident mapping requires at least one per-stack DRAM resource",
            )
        if (
            resident_values["table_bytes"]
            != resident_values["table_bytes_per_stack"]
            * resident_values["dram_resources"]
        ):
            _fail(
                f"{path}.resident_mapping.table_bytes",
                "resident table bytes must equal bytes/stack times stack resources",
            )
        if (
            resident_values["lookup_ops"]
            != resident_values["user_lookup_ops"]
            + resident_values["gc_lookup_ops"]
        ):
            _fail(
                f"{path}.resident_mapping.lookup_ops",
                "resident mapping lookup sources do not conserve total",
            )
        if (
            resident_values["update_ops"]
            != resident_values["user_update_ops"]
            + resident_values["gc_update_ops"]
        ):
            _fail(
                f"{path}.resident_mapping.update_ops",
                "resident mapping update sources do not conserve total",
            )
        dram_wait = float(resident_values["dram_wait_work_ns"])
        scheduler_parent = float(values["scheduler_queue_wait_work_ns"])
        if dram_wait > scheduler_parent and not _close(
            dram_wait, scheduler_parent
        ):
            _fail(
                f"{path}.resident_mapping.dram_wait_work_ns",
                "mapping DRAM wait exceeds scheduler queue-wait parent",
            )
        dram_wait_ops = int(resident_values["dram_waited_ops"])
        dram_wait_max = float(resident_values["dram_wait_max_ns"])
        if dram_wait_max > dram_wait and not _close(dram_wait_max, dram_wait):
            _fail(
                f"{path}.resident_mapping.dram_wait_max_ns",
                "mapping DRAM maximum wait exceeds total wait work",
            )
        if (
            (dram_wait_ops == 0) != (dram_wait == 0.0)
            or (dram_wait_ops == 0) != (dram_wait_max == 0.0)
        ):
            _fail(
                f"{path}.resident_mapping.dram_waited_ops",
                "mapping DRAM wait count/work/max zero state is inconsistent",
            )
        mapping_ops = (
            int(resident_values["lookup_ops"])
            + int(resident_values["update_ops"])
        )
        issue_busy = float(resident_values["dram_issue_busy_ns"])
        if (mapping_ops == 0) != (issue_busy == 0.0):
            _fail(
                f"{path}.resident_mapping.dram_issue_busy_ns",
                "mapping DRAM issue work must be present exactly when accesses exist",
            )
        values["resident_mapping"] = resident_values

    directional = _required_mapping(stage, "ecc_directional", path)
    directions: dict[str, Mapping[str, object]] = {}
    for direction in ("decode", "encode"):
        item = _required_mapping(
            directional, direction, f"{path}.ecc_directional"
        )
        directions[direction] = {
            "queue_wait_work_ns": _required_number(
                item, "queue_wait_work_ns", f"{path}.ecc_directional.{direction}"
            ),
            "response_latency_work_ns": _required_number(
                item,
                "response_latency_work_ns",
                f"{path}.ecc_directional.{direction}",
            ),
            "issue_busy_ns": _required_number(
                item, "issue_busy_ns", f"{path}.ecc_directional.{direction}"
            ),
            "ops": _required_integer(
                item, "ops", f"{path}.ecc_directional.{direction}"
            ),
            "codeword_bytes": _required_integer(
                item, "codeword_bytes", f"{path}.ecc_directional.{direction}"
            ),
        }
    _expect_close(
        sum(float(directions[d]["queue_wait_work_ns"]) for d in directions),
        float(values["ecc_queue_wait_work_ns"]),
        f"{path}.ecc_directional",
        "directional ECC queue wait must sum to its parent",
    )
    _expect_close(
        sum(
            float(directions[d]["response_latency_work_ns"])
            for d in directions
        ),
        float(values["ecc_response_latency_work_ns"]),
        f"{path}.ecc_directional",
        "directional ECC response latency must sum to its parent",
    )
    values["ecc_directional"] = directions
    return values


def _validate_stage_work(
    time: Mapping[str, object], path: str, schema_version: int
) -> Mapping[str, object]:
    stage = _required_mapping(time, "stage_work", path)
    if stage.get("scope") != "all_device_work_including_background_and_drain":
        _fail(f"{path}.stage_work.scope", "unsupported stage-work scope")
    values: dict[str, object] = {
        "scope": "all_device_work_including_background_and_drain"
    }
    for device in ("hbm", "hbf"):
        if device not in stage:
            _fail(f"{path}.stage_work", f"missing {device}")
        item = stage[device]
        values[device] = None if item is None else _validate_device_stage(
            _mapping(item, f"{path}.stage_work.{device}"),
            device,
            f"{path}.stage_work.{device}",
            schema_version,
        )
    external = stage.get("external_backing")
    values["external_backing"] = (
        None
        if external is None
        else _validate_device_stage(
            _mapping(external, f"{path}.stage_work.external_backing"),
            "external_backing",
            f"{path}.stage_work.external_backing",
            schema_version,
        )
    )

    link = _required_mapping(stage, "base_die_link", f"{path}.stage_work")
    values["base_die_link"] = {
        key: _required_number(link, key, f"{path}.stage_work.base_die_link")
        for key in BASE_DIE_LINK_COMPONENTS
    }

    streaming = _required_mapping(
        stage, "layer_streaming_controller", f"{path}.stage_work"
    )
    streaming_present = _required_bool(
        streaming, "present", f"{path}.stage_work.layer_streaming_controller"
    )
    streaming_values = {
        key: _required_number(
            streaming,
            key,
            f"{path}.stage_work.layer_streaming_controller",
        )
        for key in LAYER_STREAMING_CONTROLLER_FIELDS
    }
    if not streaming_present and any(
            value != 0.0 for value in streaming_values.values()):
        _fail(
            f"{path}.stage_work.layer_streaming_controller",
            "absent controller has non-zero work",
        )
    values["layer_streaming_controller"] = {
        "present": streaming_present,
        **streaming_values,
    }

    cooperative = _required_mapping(
        stage, "cooperative_write_controller", f"{path}.stage_work"
    )
    cooperative_present = _required_bool(
        cooperative, "present", f"{path}.stage_work.cooperative_write_controller"
    )
    full_waited_ops = _required_integer(
        cooperative,
        "full_waited_ops",
        f"{path}.stage_work.cooperative_write_controller",
    )
    full_wait = _required_number(
        cooperative,
        "full_wait_work_ns",
        f"{path}.stage_work.cooperative_write_controller",
    )
    if not cooperative_present and (full_waited_ops != 0 or full_wait != 0.0):
        _fail(
            f"{path}.stage_work.cooperative_write_controller",
            "absent controller has non-zero work",
        )
    values["cooperative_write_controller"] = {
        "present": cooperative_present,
        "full_waited_ops": full_waited_ops,
        "full_wait_work_ns": full_wait,
    }
    return values


def _validate_resources(
    time: Mapping[str, object], path: str, schema_version: int
) -> Mapping[str, Mapping[str, float | int]]:
    resources = _required_mapping(time, "resource_busy", path)
    values: dict[str, Mapping[str, float | int]] = {}
    for name in _resource_names(schema_version):
        metric = _required_mapping(resources, name, f"{path}.resource_busy")
        busy = _required_number(metric, "busy_ns", f"{path}.resource_busy.{name}")
        count = _required_integer(
            metric, "resource_count", f"{path}.resource_busy.{name}"
        )
        span = _required_number(
            metric, "active_span_ns", f"{path}.resource_busy.{name}"
        )
        capacity = _required_number(
            metric, "capacity_time_ns", f"{path}.resource_busy.{name}"
        )
        utilization = _required_number(
            metric, "utilization", f"{path}.resource_busy.{name}"
        )
        _expect_close(
            count * span,
            capacity,
            f"{path}.resource_busy.{name}.capacity_time_ns",
            "resource_count times active_span_ns must equal capacity_time_ns",
        )
        if busy > capacity and not _close(busy, capacity):
            _fail(
                f"{path}.resource_busy.{name}.busy_ns",
                "busy time exceeds capacity-time",
            )
        expected_utilization = busy / capacity if capacity else 0.0
        _expect_close(
            expected_utilization,
            utilization,
            f"{path}.resource_busy.{name}.utilization",
            "utilization must equal busy_ns divided by capacity_time_ns",
        )
        if utilization > 1.0 and not _close(utilization, 1.0):
            _fail(f"{path}.resource_busy.{name}.utilization", "exceeds 1")
        values[name] = {
            "busy_ns": busy,
            "resource_count": count,
            "active_span_ns": span,
            "capacity_time_ns": capacity,
            "utilization": utilization,
        }
    for name in OPTIONAL_RESOURCE_NAMES:
        if name not in resources:
            continue
        metric = _required_mapping(resources, name, f"{path}.resource_busy")
        busy = _required_number(
            metric, "busy_ns", f"{path}.resource_busy.{name}")
        count = _required_integer(
            metric, "resource_count", f"{path}.resource_busy.{name}")
        span = _required_number(
            metric, "active_span_ns", f"{path}.resource_busy.{name}")
        capacity = _required_number(
            metric, "capacity_time_ns", f"{path}.resource_busy.{name}")
        utilization = _required_number(
            metric, "utilization", f"{path}.resource_busy.{name}")
        _expect_close(
            count * span,
            capacity,
            f"{path}.resource_busy.{name}.capacity_time_ns",
            "resource_count times active_span_ns must equal capacity_time_ns",
        )
        if busy > capacity and not _close(busy, capacity):
            _fail(
                f"{path}.resource_busy.{name}.busy_ns",
                "busy time exceeds capacity-time",
            )
        expected_utilization = busy / capacity if capacity else 0.0
        _expect_close(
            expected_utilization,
            utilization,
            f"{path}.resource_busy.{name}.utilization",
            "utilization must equal busy_ns divided by capacity_time_ns",
        )
        if utilization > 1.0 and not _close(utilization, 1.0):
            _fail(f"{path}.resource_busy.{name}.utilization", "exceeds 1")
        values[name] = {
            "busy_ns": busy,
            "resource_count": count,
            "active_span_ns": span,
            "capacity_time_ns": capacity,
            "utilization": utilization,
        }
    return values


def _validate_scenario(
    scenario: Mapping[str, object], path: str, schema_version: int
) -> tuple[
    int,
    Mapping[str, float],
    Mapping[str, float | int | str],
    Mapping[str, object],
    Mapping[str, Mapping[str, float | int]],
]:
    ops = _required_integer(scenario, "ops", path)
    if ops == 0:
        _fail(f"{path}.ops", "must be positive")
    time = _required_mapping(scenario, "time_breakdown", path)
    version = _required_integer(time, "contract_version", f"{path}.time_breakdown")
    if version != TIME_CONTRACT_VERSION:
        _fail(
            f"{path}.time_breakdown.contract_version",
            f"unsupported version {version}",
        )
    semantics = _required_mapping(time, "semantics", f"{path}.time_breakdown")
    if dict(semantics) != TIME_SEMANTICS:
        _fail(f"{path}.time_breakdown.semantics", "does not match contract v1")
    wall = _validate_wall(time, f"{path}.time_breakdown")
    latency = _validate_latency(time, ops, f"{path}.time_breakdown")
    stage = _validate_stage_work(
        time, f"{path}.time_breakdown", schema_version
    )
    resources = _validate_resources(
        time, f"{path}.time_breakdown", schema_version
    )
    hbf = stage["hbf"]
    if hbf is not None:
        directional = _mapping(hbf, f"{path}.time_breakdown.stage_work.hbf")[
            "ecc_directional"
        ]
        issue_busy = sum(
            float(_mapping(directional, "ecc_directional")[direction]["issue_busy_ns"])
            for direction in ("decode", "encode")
        )
        _expect_close(
            issue_busy,
            float(resources["hbf_ecc_issue"]["busy_ns"]),
            f"{path}.time_breakdown.stage_work.hbf.ecc_directional",
            "directional ECC issue busy must sum to hbf_ecc_issue busy_ns",
        )
    external = stage["external_backing"]
    if external is not None:
        for resource_name in OPTIONAL_RESOURCE_NAMES:
            if resource_name not in resources:
                _fail(
                    f"{path}.time_breakdown.resource_busy",
                    f"external-backing stage requires {resource_name}",
                )
        external_work = _mapping(
            external, f"{path}.time_breakdown.stage_work.external_backing")
        _expect_close(
            float(external_work["channel_transfer_work_ns"]),
            float(resources["external_media"]["busy_ns"]),
            f"{path}.time_breakdown.stage_work.external_backing",
            "media transfer work must equal external_media busy_ns",
        )
        _expect_close(
            float(external_work["hb_io_transfer_work_ns"]),
            float(resources["external_link_m2s"]["busy_ns"]) +
            float(resources["external_link_s2m"]["busy_ns"]),
            f"{path}.time_breakdown.stage_work.external_backing",
            "host-link transfer work must equal directional link busy_ns",
        )
        external_stats = _required_mapping(
            scenario, "external_backing_stats", path
        )
        controller_processing = _required_number(
            external_stats,
            "controller_processing_work_ns",
            f"{path}.external_backing_stats",
        )
        _expect_close(
            float(external_work["command_work_ns"]),
            float(resources["external_controller"]["busy_ns"])
            + controller_processing,
            f"{path}.time_breakdown.stage_work.external_backing",
            "controller command work must equal issue busy plus processing work",
        )
        propagation_work = _required_number(
            external_stats,
            "transport_propagation_work_ns",
            f"{path}.external_backing_stats",
        )
        _expect_close(
            float(external_work["transport_latency_work_ns"]),
            propagation_work,
            f"{path}.time_breakdown.stage_work.external_backing",
            "transport latency work must equal directional propagation work",
        )
    return ops, wall, latency, stage, resources


def _row(
    *,
    overview: ScenarioOverview,
    domain: str,
    scope: str,
    component: str,
    role: str,
    value_ns: float,
    per_user_op: bool,
    share: float | None = None,
    share_basis: str = "",
    resource_count: int | None = None,
    active_span_ns: float | None = None,
    capacity_time_ns: float | None = None,
    utilization: float | None = None,
    average_parallelism: float | None = None,
    note: str = "",
) -> dict[str, object]:
    return {
        "run_label": overview.run_label,
        "case": overview.case,
        "config": overview.config,
        "source_summary": overview.source_summary,
        "scenario": overview.scenario,
        "user_ops": overview.user_ops,
        "domain": domain,
        "scope": scope,
        "component": component,
        "role": role,
        "value_ns": value_ns,
        "value_ms": value_ns / 1e6,
        "per_user_op_ns": value_ns / overview.user_ops if per_user_op else None,
        "share": share,
        "share_basis": share_basis,
        "resource_count": resource_count,
        "active_span_ns": active_span_ns,
        "capacity_time_ns": capacity_time_ns,
        "utilization": utilization,
        "average_parallelism": average_parallelism,
        "note": note,
    }


def _scenario_rows(
    *,
    source: SummaryInput,
    scenario: Mapping[str, object],
    metadata: Mapping[str, object],
    path: str,
    schema_version: int,
) -> tuple[list[dict[str, object]], ScenarioOverview]:
    name = scenario["name"]
    assert isinstance(name, str)
    ops, wall, latency, stage, resources = _validate_scenario(
        scenario, path, schema_version
    )
    stage_components = _device_stage_components(schema_version)
    case = metadata.get("case", "")
    config = metadata.get("config", "")
    if not isinstance(case, str) or not isinstance(config, str):
        _fail(path, "scenario metadata case/config must be strings")

    primary_stages: list[tuple[str, float]] = []
    for device in ("hbm", "hbf", "external_backing"):
        work = stage[device]
        if work is not None:
            work_map = _mapping(work, f"{path}.time_breakdown.stage_work.{device}")
            primary_stages.extend(
                (f"{device}.{component}", float(work_map[component]))
                for component in stage_components
            )
    link_map = _mapping(stage["base_die_link"], "base_die_link")
    link_present = any(float(link_map[key]) > 0.0 for key in BASE_DIE_LINK_COMPONENTS)
    if link_present:
        primary_stages.extend(
            (f"base_die_link.{component}", float(link_map[component]))
            for component in BASE_DIE_LINK_COMPONENTS
        )
    streaming = _mapping(
        stage["layer_streaming_controller"], "layer-streaming controller")
    if streaming["present"]:
        primary_stages.extend(
            (f"layer_streaming_controller.{component}",
             float(streaming[component]))
            for component in LAYER_STREAMING_CONTROLLER_PARENTS
        )
    cooperative = _mapping(
        stage["cooperative_write_controller"], "cooperative controller"
    )
    if cooperative["present"]:
        primary_stages.append(
            (
                "cooperative_write_controller.full_wait_work_ns",
                float(cooperative["full_wait_work_ns"]),
            )
        )
    positive_stages = [item for item in primary_stages if item[1] > 0.0]
    largest_stage, largest_stage_work = (
        max(positive_stages, key=lambda item: (item[1], item[0]))
        if positive_stages
        else (None, 0.0)
    )
    active_resources = [
        (resource, float(metric["utilization"]))
        for resource, metric in resources.items()
        if int(metric["resource_count"]) > 0
    ]
    highest_resource, highest_utilization = (
        max(active_resources, key=lambda item: (item[1], item[0]))
        if active_resources
        else (None, 0.0)
    )
    overview = ScenarioOverview(
        run_label=source.label,
        case=case,
        config=config,
        source_summary=source.source,
        scenario=name,
        user_ops=ops,
        offered_arrival_span_ns=wall["offered_arrival_span_ns"],
        post_offer_user_completion_tail_ns=wall[
            "post_offer_user_completion_tail_ns"
        ],
        user_completion_span_ns=wall["user_completion_span_ns"],
        drain_tail_ns=wall["drain_tail_ns"],
        makespan_ns=wall["makespan_ns"],
        average_ns=float(latency["average_ns"]),
        p95_ns=float(latency["p95_ns"]),
        largest_stage=largest_stage,
        largest_stage_work_ns=largest_stage_work,
        highest_utilization_resource=highest_resource,
        highest_resource_utilization=highest_utilization,
    )

    rows: list[dict[str, object]] = []
    makespan = wall["makespan_ns"]
    for component in WALL_FRONTIERS:
        rows.append(_row(
            overview=overview,
            domain="wall_clock",
            scope="frontier",
            component=component,
            role="absolute_frontier",
            value_ns=wall[component],
            per_user_op=False,
            note="absolute simulated timestamp; never add to elapsed spans",
        ))
    for component in WALL_SEGMENTS:
        rows.append(_row(
            overview=overview,
            domain="wall_clock",
            scope="makespan",
            component=component,
            role="additive_segment",
            value_ns=wall[component],
            per_user_op=False,
            share=wall[component] / makespan if makespan else 0.0,
            share_basis="makespan_ns",
            note="the three additive_segment rows sum to makespan_ns",
        ))
    for component in WALL_TOTALS:
        rows.append(_row(
            overview=overview,
            domain="wall_clock",
            scope="makespan",
            component=component,
            role="derived_total",
            value_ns=wall[component],
            per_user_op=False,
            note="derived total; do not add to additive_segment rows",
        ))

    offered_latency = float(latency[LATENCY_TOTAL])
    phase_dependency_wait = float(latency["phase_dependency_wait_work_ns"])
    source_latency = float(latency[SOURCE_LATENCY_TOTAL])
    for component in LATENCY_WORK_COMPONENTS:
        value = float(latency[component])
        rows.append(_row(
            overview=overview,
            domain="latency_work",
            scope="user_operations",
            component=component,
            role="additive_work_component",
            value_ns=value,
            per_user_op=True,
            share=value / offered_latency if offered_latency else 0.0,
            share_basis=LATENCY_TOTAL,
            note="sum across user operations; may overlap in wall time",
        ))
    rows.append(_row(
        overview=overview,
        domain="latency_work",
        scope="source_operations",
        component="phase_dependency_wait_work_ns",
        role="additive_work_component",
        value_ns=phase_dependency_wait,
        per_user_op=True,
        share=phase_dependency_wait / source_latency if source_latency else 0.0,
        share_basis=SOURCE_LATENCY_TOTAL,
        note=(
            "source-ready to dependency-gated offer wait; sum across user "
            "operations and may overlap in wall time"
        ),
    ))
    rows.append(_row(
        overview=overview,
        domain="latency_work",
        scope="user_operations",
        component=LATENCY_TOTAL,
        role="work_total",
        value_ns=offered_latency,
        per_user_op=True,
        note=(
            "post-dependency offer to completion subtotal; excludes phase "
            "dependency wait"
        ),
    ))
    rows.append(_row(
        overview=overview,
        domain="latency_work",
        scope="source_operations",
        component=SOURCE_LATENCY_TOTAL,
        role="source_work_total",
        value_ns=source_latency,
        per_user_op=True,
        note=(
            "phase dependency wait plus offer-to-completion work; not an "
            "elapsed-time decomposition"
        ),
    ))
    for component in LATENCY_STATISTICS:
        rows.append(_row(
            overview=overview,
            domain="latency_work",
            scope="offered_distribution",
            component=component,
            role="distribution_statistic",
            value_ns=float(latency[component]),
            per_user_op=False,
            note="offer-to-completion per-operation statistic; do not add",
        ))
    for component in SERVICE_LATENCY_STATISTICS:
        rows.append(_row(
            overview=overview,
            domain="latency_work",
            scope="service_distribution",
            component=component,
            role="distribution_statistic",
            value_ns=float(latency[component]),
            per_user_op=False,
            note="first-credit-to-completion statistic; do not add",
        ))
    for component in SOURCE_LATENCY_STATISTICS:
        rows.append(_row(
            overview=overview,
            domain="latency_work",
            scope="source_distribution",
            component=component,
            role="distribution_statistic",
            value_ns=float(latency[component]),
            per_user_op=False,
            note="source-ready-to-completion statistic; do not add",
        ))
    rows.append(_row(
        overview=overview,
        domain="latency_work",
        scope="admission",
        component="front_end_admission_max_wait_ns",
        role="distribution_statistic",
        value_ns=float(latency["front_end_admission_max_wait_ns"]),
        per_user_op=False,
        note="largest offer-to-first-credit wait; do not add",
    ))
    rows.append(_row(
        overview=overview,
        domain="latency_work",
        scope="dependency",
        component="phase_dependency_max_wait_ns",
        role="distribution_statistic",
        value_ns=float(latency["phase_dependency_max_wait_ns"]),
        per_user_op=False,
        note="largest source-ready to dependency-gated offer wait; do not add",
    ))

    for device in ("hbm", "hbf", "external_backing"):
        work = stage[device]
        if work is None:
            continue
        work_map = _mapping(work, f"{path}.time_breakdown.stage_work.{device}")
        total = float(work_map[DEVICE_STAGE_TOTAL])
        for component in stage_components:
            value = float(work_map[component])
            rows.append(_row(
                overview=overview,
                domain="stage_work",
                scope=device,
                component=component,
                role="overlapping_work_component",
                value_ns=value,
                per_user_op=True,
                share=value / total if total else 0.0,
                share_basis=f"{device}.{DEVICE_STAGE_TOTAL}",
                note="component work may overlap in elapsed time",
            ))
        rows.append(_row(
            overview=overview,
            domain="stage_work",
            scope=device,
            component=DEVICE_STAGE_TOTAL,
            role="scope_work_total",
            value_ns=total,
            per_user_op=True,
            note="sum of this scope's primary components; may exceed makespan",
        ))
        if device == "hbf":
            write_buffer = _mapping(work_map["write_buffer"], "hbf.write_buffer")
            slot_wait = float(write_buffer["slot_wait_work_ns"])
            parent = float(work_map["scheduler_queue_wait_work_ns"])
            rows.append(_row(
                overview=overview,
                domain="stage_work",
                scope="hbf.write_buffer",
                component="slot_wait_work_ns",
                role="attribution_subset",
                value_ns=slot_wait,
                per_user_op=True,
                share=slot_wait / parent if parent else 0.0,
                share_basis="hbf.scheduler_queue_wait_work_ns",
                note="named subset; already included in its scheduler parent",
            ))
            if "resident_mapping" in work_map:
                resident = _mapping(
                    work_map["resident_mapping"],
                    "hbf.resident_mapping",
                )
                value = float(resident["dram_wait_work_ns"])
                rows.append(_row(
                    overview=overview,
                    domain="stage_work",
                    scope="hbf.resident_mapping",
                    component="dram_wait_work_ns",
                    role="attribution_subset",
                    value_ns=value,
                    per_user_op=True,
                    share=value / parent if parent else 0.0,
                    share_basis="hbf.scheduler_queue_wait_work_ns",
                    note=(
                        "per-stack mapping-DRAM issue wait; already included "
                        "in scheduler queue wait"
                    ),
                ))
            directional = _mapping(work_map["ecc_directional"], "hbf ECC")
            for direction in ("decode", "encode"):
                diagnostic = _mapping(directional[direction], f"hbf ECC {direction}")
                for component, parent_component in (
                    ("queue_wait_work_ns", "ecc_queue_wait_work_ns"),
                    (
                        "response_latency_work_ns",
                        "ecc_response_latency_work_ns",
                    ),
                ):
                    value = float(diagnostic[component])
                    parent = float(work_map[parent_component])
                    rows.append(_row(
                        overview=overview,
                        domain="stage_work",
                        scope=f"hbf.ecc.{direction}",
                        component=component,
                        role="attribution_child",
                        value_ns=value,
                        per_user_op=True,
                        share=value / parent if parent else 0.0,
                        share_basis=f"hbf.{parent_component}",
                        note="directional child; already included in its ECC parent",
                    ))

    if link_present:
        for component in BASE_DIE_LINK_COMPONENTS:
            rows.append(_row(
                overview=overview,
                domain="stage_work",
                scope="base_die_link",
                component=component,
                role="overlapping_work_component",
                value_ns=float(link_map[component]),
                per_user_op=True,
                note="link work may overlap with device and user work",
            ))
    if streaming["present"]:
        for component in LAYER_STREAMING_CONTROLLER_FIELDS:
            if component in LAYER_STREAMING_CONTROLLER_STATISTICS:
                role = "distribution_statistic"
                per_op = False
            elif component in LAYER_STREAMING_CONTROLLER_PARENTS:
                role = "controller_parent"
                per_op = True
            else:
                role = "controller_attribution"
                per_op = True
            rows.append(_row(
                overview=overview,
                domain="stage_work",
                scope="layer_streaming_controller",
                component=component,
                role=role,
                value_ns=float(streaming[component]),
                per_user_op=per_op,
                note="controller parent and attribution views may overlap",
            ))
    if cooperative["present"]:
        rows.append(_row(
            overview=overview,
            domain="stage_work",
            scope="cooperative_write_controller",
            component="full_wait_work_ns",
            role="controller_parent",
            value_ns=float(cooperative["full_wait_work_ns"]),
            per_user_op=True,
            note="controller work may overlap across waiting operations",
        ))

    for resource, metric in resources.items():
        count = int(metric["resource_count"])
        if count == 0:
            continue
        busy = float(metric["busy_ns"])
        span = float(metric["active_span_ns"])
        rows.append(_row(
            overview=overview,
            domain="resource_busy",
            scope=resource,
            component="busy_ns",
            role="exclusive_resource_busy",
            value_ns=busy,
            per_user_op=False,
            resource_count=count,
            active_span_ns=span,
            capacity_time_ns=float(metric["capacity_time_ns"]),
            utilization=float(metric["utilization"]),
            average_parallelism=busy / span if span else 0.0,
            note="utilization is busy_ns / (resource_count * active_span_ns)",
        ))
    if stage["hbf"] is not None:
        hbf_map = _mapping(stage["hbf"], "hbf")
        directional = _mapping(hbf_map["ecc_directional"], "hbf ECC")
        ecc_metric = resources["hbf_ecc_issue"]
        ecc_busy = float(ecc_metric["busy_ns"])
        for direction in ("decode", "encode"):
            diagnostic = _mapping(directional[direction], f"hbf ECC {direction}")
            value = float(diagnostic["issue_busy_ns"])
            rows.append(_row(
                overview=overview,
                domain="resource_busy",
                scope="hbf_ecc_issue",
                component=f"{direction}_issue_busy_ns",
                role="attribution_child",
                value_ns=value,
                per_user_op=False,
                share=value / ecc_busy if ecc_busy else 0.0,
                share_basis="hbf_ecc_issue.busy_ns",
                note="directional child; already included in resource busy_ns",
            ))
    return rows, overview


def build_time_breakdown_report(
    summaries: Iterable[SummaryInput],
) -> TimeBreakdownReport:
    """Validate summaries and return deterministic long-form report data."""

    rows: list[Mapping[str, object]] = []
    overviews: list[ScenarioOverview] = []
    identities: set[tuple[str, str]] = set()
    inputs = list(summaries)
    if not inputs:
        raise TimeBreakdownError("at least one summary is required")
    for summary_index, source in enumerate(inputs):
        root_path = source.source or f"summary[{summary_index}]"
        summary = _mapping(source.summary, root_path)
        schema = summary.get("schema")
        if schema == SUMMARY_SCHEMA:
            schema_version = int(SUMMARY_SCHEMA["version"])
        elif schema == ARCHIVED_SUMMARY_SCHEMA:
            schema_version = int(ARCHIVED_SUMMARY_SCHEMA["version"])
        else:
            _fail(root_path, "unsupported summary schema; expected v9 or v16")
        if summary.get("sanity") != "PASS":
            _fail(root_path, f"summary sanity is {summary.get('sanity')!r}")
        if not source.label:
            _fail(root_path, "run label must not be empty")
        scenario_items = _required_list(summary, "scenarios", root_path)
        if not scenario_items:
            _fail(f"{root_path}.scenarios", "must not be empty")
        names: list[str] = []
        for index, value in enumerate(scenario_items):
            scenario = _mapping(value, f"{root_path}.scenarios[{index}]")
            name = scenario.get("name")
            if not isinstance(name, str) or not name:
                _fail(f"{root_path}.scenarios[{index}].name", "must be a string")
            names.append(name)
            identity = (source.label, name)
            if identity in identities:
                _fail(root_path, f"duplicate report identity {identity!r}")
            identities.add(identity)
            metadata = source.scenario_metadata.get(name, {})
            if not isinstance(metadata, Mapping):
                _fail(root_path, f"metadata for {name!r} must be an object")
            scenario_rows, overview = _scenario_rows(
                source=source,
                scenario=scenario,
                metadata=metadata,
                path=f"{root_path}.scenarios[{index}]({name})",
                schema_version=schema_version,
            )
            rows.extend(scenario_rows)
            overviews.append(overview)
        if len(names) != len(set(names)):
            _fail(f"{root_path}.scenarios", "contains duplicate scenario names")
        unknown_metadata = set(source.scenario_metadata) - set(names)
        if unknown_metadata:
            _fail(root_path, f"metadata names unknown scenarios {sorted(unknown_metadata)}")
    overviews.sort(key=lambda item: (item.run_label, item.case, item.config, item.scenario))
    row_order = {id(item): index for index, item in enumerate(rows)}
    rows.sort(key=lambda item: (
        str(item["run_label"]),
        str(item["case"]),
        str(item["config"]),
        str(item["scenario"]),
        row_order[id(item)],
    ))
    return TimeBreakdownReport(tuple(rows), tuple(overviews))


def _csv_scalar(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def render_time_breakdown_csv(report: TimeBreakdownReport) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in report.rows:
        writer.writerow({field: _csv_scalar(row.get(field)) for field in CSV_FIELDS})
    return handle.getvalue()


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_ms(value_ns: float) -> str:
    return f"{value_ns / 1e6:.6f}"


def _format_us(value_ns: float) -> str:
    return f"{value_ns / 1e3:.3f}"


def render_time_breakdown_markdown(report: TimeBreakdownReport) -> str:
    lines = [
        "# HBFSim time breakdown",
        "",
        "Only the three wall-clock segments are additive. Latency and stage "
        "work can overlap across operations/resources; parent totals and "
        "attribution children must not be added twice. Resource utilization "
        "uses `busy_ns / (resource_count * active_span_ns)`.",
        "",
        "## Overview",
        "",
        "| Run | Case | Config | Scenario | Ops | Offered ms | Post-offer ms | "
        "User ms | Drain ms | Makespan ms | Avg us | P95 us | Largest primary "
        "stage (work ms) | Highest resource utilization |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report.overviews:
        largest = (
            f"{item.largest_stage} ({_format_ms(item.largest_stage_work_ns)})"
            if item.largest_stage else "n/a"
        )
        resource = (
            f"{item.highest_utilization_resource} "
            f"({item.highest_resource_utilization * 100.0:.3f}%)"
            if item.highest_utilization_resource else "n/a"
        )
        cells = (
            item.run_label,
            item.case or "-",
            item.config or "-",
            item.scenario,
            item.user_ops,
            _format_ms(item.offered_arrival_span_ns),
            _format_ms(item.post_offer_user_completion_tail_ns),
            _format_ms(item.user_completion_span_ns),
            _format_ms(item.drain_tail_ns),
            _format_ms(item.makespan_ns),
            _format_us(item.average_ns),
            _format_us(item.p95_ns),
            largest,
            resource,
        )
        lines.append("| " + " | ".join(_markdown_cell(cell) for cell in cells) + " |")

    rows_by_identity: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in report.rows:
        rows_by_identity.setdefault(
            (str(row["run_label"]), str(row["scenario"])), []
        ).append(row)
    for item in report.overviews:
        identity = (item.run_label, item.scenario)
        scenario_rows = rows_by_identity[identity]
        lines.extend((
            "",
            f"## {_markdown_cell(item.run_label)} / {_markdown_cell(item.scenario)}",
            "",
            "### Additive wall clock",
            "",
            "| Segment | ms | % makespan |",
            "|---|---:|---:|",
        ))
        for row in scenario_rows:
            if row["domain"] != "wall_clock" or row["role"] != "additive_segment":
                continue
            lines.append(
                f"| {_markdown_cell(row['component'])} | "
                f"{float(row['value_ms']):.6f} | {float(row['share']) * 100.0:.3f}% |"
            )
        lines.extend((
            f"| **makespan_ns (total)** | **{_format_ms(item.makespan_ns)}** | **100.000%** |",
            "",
            "### Latency work and distribution",
            "",
            "| Component | Role | work ms | per user op us |",
            "|---|---|---:|---:|",
        ))
        for row in scenario_rows:
            if row["domain"] != "latency_work":
                continue
            per_op = row["per_user_op_ns"]
            lines.append(
                f"| {_markdown_cell(row['component'])} | {_markdown_cell(row['role'])} | "
                f"{float(row['value_ms']):.6f} | "
                f"{float(per_op) / 1e3:.3f} |" if per_op is not None else
                f"| {_markdown_cell(row['component'])} | {_markdown_cell(row['role'])} | "
                f"{float(row['value_ms']):.6f} | - |"
            )
        lines.extend((
            "",
            "### Stage work (non-zero rows)",
            "",
            "| Scope | Component | Role | work ms | per user op us |",
            "|---|---|---|---:|---:|",
        ))
        nonzero_stage_rows = [
            row for row in scenario_rows
            if row["domain"] == "stage_work" and float(row["value_ns"]) > 0.0
        ]
        for row in nonzero_stage_rows:
            per_op = row["per_user_op_ns"]
            per_op_text = f"{float(per_op) / 1e3:.3f}" if per_op is not None else "-"
            lines.append(
                f"| {_markdown_cell(row['scope'])} | {_markdown_cell(row['component'])} | "
                f"{_markdown_cell(row['role'])} | {float(row['value_ms']):.6f} | "
                f"{per_op_text} |"
            )
        if not nonzero_stage_rows:
            lines.append("| - | - | - | 0.000000 | - |")
        lines.extend((
            "",
            "### Resource capacity",
            "",
            "| Resource | busy ms | count | active span ms | utilization | avg parallelism |",
            "|---|---:|---:|---:|---:|---:|",
        ))
        resource_rows = [
            row for row in scenario_rows
            if row["domain"] == "resource_busy"
            and row["role"] == "exclusive_resource_busy"
        ]
        resource_rows.sort(
            key=lambda row: (-float(row["utilization"]), str(row["scope"]))
        )
        for row in resource_rows:
            lines.append(
                f"| {_markdown_cell(row['scope'])} | {float(row['value_ms']):.6f} | "
                f"{int(row['resource_count'])} | "
                f"{float(row['active_span_ns']) / 1e6:.6f} | "
                f"{float(row['utilization']) * 100.0:.3f}% | "
                f"{float(row['average_parallelism']):.3f} |"
            )
        if not resource_rows:
            lines.append("| - | 0.000000 | 0 | 0.000000 | 0.000% | 0.000 |")
        lines.extend((
            "",
            "The CSV retains zero-valued stages and hierarchy roles. The Markdown "
            "stage table omits zero rows for readability.",
        ))
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_time_breakdown_report(
    summaries: Iterable[SummaryInput],
    csv_path: Path,
    markdown_path: Path,
) -> TimeBreakdownReport:
    """Validate all inputs, then atomically write CSV and Markdown artifacts."""

    report = build_time_breakdown_report(summaries)
    csv_payload = render_time_breakdown_csv(report)
    markdown_payload = render_time_breakdown_markdown(report)
    _atomic_write(Path(csv_path), csv_payload)
    _atomic_write(Path(markdown_path), markdown_payload)
    return report


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TimeBreakdownError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_summary_input(
    path: Path,
    *,
    label: str | None = None,
    scenario_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> SummaryInput:
    path = Path(path).resolve()
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError) as error:
        raise TimeBreakdownError(f"cannot load {path}: {error}") from error
    return SummaryInput(
        label=label or path.stem,
        summary=_mapping(data, str(path)),
        source=str(path),
        scenario_metadata=scenario_metadata or {},
    )


def _parse_summary_spec(spec: str) -> SummaryInput:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        if not label or not raw_path:
            raise argparse.ArgumentTypeError("summary must be [LABEL=]PATH")
        return load_summary_input(Path(raw_path), label=label)
    return load_summary_input(Path(spec))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        metavar="[LABEL=]PATH",
        help="repeat for every schema-v9/v16 summary in the suite",
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    try:
        summaries = [_parse_summary_spec(spec) for spec in args.summary]
        report = write_time_breakdown_report(summaries, args.csv, args.markdown)
    except TimeBreakdownError as error:
        parser.error(str(error))
    print(
        f"wrote {args.csv} and {args.markdown}: "
        f"{len(report.overviews)} scenarios, {len(report.rows)} detail rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
