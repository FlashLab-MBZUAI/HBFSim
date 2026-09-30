#!/usr/bin/env python3
"""Summarize one fixed-footprint execution without model semantics in HBFSim.

HBFSim reports target-level latency and device-level traffic.  This module
joins those receipts with the object class attached to each submitted trace
phase.  Object identity therefore stays outside the simulator execution API.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


METRICS_SCHEMA = {
    "name": "hbfsim.fixed_footprint_metrics",
    "version": 2,
}

TARGETS_BY_MEDIUM = {
    "hbm": ("HBM",),
    "hbf": ("HBF_LOGICAL", "HBF_STATIC", "HBF_PHYSICAL"),
    "external": ("EXTERNAL",),
    "interconnect": (
        "D2D_HBF_TO_HBM",
        "D2D_HBM_TO_HBF",
        "DIRECT_HBF_TO_EXTERNAL",
        "DIRECT_EXTERNAL_TO_HBF",
    ),
}


class FixedFootprintMetricsError(ValueError):
    """A simulation receipt cannot be reduced to fixed-footprint metrics."""


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FixedFootprintMetricsError(f"{description} must be an object")
    return value


def _number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixedFootprintMetricsError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise FixedFootprintMetricsError(
            f"{description} must be finite and nonnegative"
        )
    return result


def _integer(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FixedFootprintMetricsError(f"{description} must be an integer")
    if isinstance(value, float) and (
        not math.isfinite(value) or value < 0 or not value.is_integer()
    ):
        raise FixedFootprintMetricsError(f"{description} must be an integer")
    if isinstance(value, int) and value < 0:
        raise FixedFootprintMetricsError(f"{description} must be nonnegative")
    return int(value)


def _rate(byte_count: int, elapsed_ns: float) -> float | None:
    if elapsed_ns == 0:
        return None if byte_count else 0.0
    # Decimal GB/s: bytes/ns is numerically equal to GB/s.
    return byte_count / elapsed_ns


def _thermal_summary(
    device_totals: Mapping[str, Any],
    *,
    thermal_identity: Mapping[str, Any] | None,
    active_hbf_stacks: int,
    completion_ns: float,
) -> dict[str, Any]:
    """Reduce the HBF thermal governor telemetry for one topology row.

    The simulator is the only source of truth for temperatures and paced
    work; the resolved config identity contributes the pacing budget so the
    row can bind measured average power against its declared envelope.
    """

    hbf_totals = device_totals.get("hbf")
    if hbf_totals is None:
        if thermal_identity is not None and thermal_identity.get("enabled"):
            raise FixedFootprintMetricsError(
                "thermal identity declares an enabled model but the run "
                "reports no HBF device totals"
            )
        return {"enabled": False}
    hbf = _mapping(hbf_totals, "HBF device totals")
    state = _mapping(hbf.get("state", {}), "HBF device state")
    enabled = bool(state.get("thermal_enabled", False))
    if thermal_identity is not None and thermal_identity.get("enabled"):
        if not enabled:
            raise FixedFootprintMetricsError(
                "resolved config enables the HBF thermal model but the "
                "device reports it disabled"
            )
    if not enabled:
        return {"enabled": False}
    energy_j = _number(
        hbf.get("thermal_media_energy_j", 0), "thermal media energy"
    )
    span_ns = _number(
        hbf.get("thermal_throttled_span_ns", 0), "thermal throttled span"
    )
    seconds = completion_ns / 1e9
    average_power_w = energy_j / seconds if seconds > 0 else None
    per_stack_power_w = (
        average_power_w / active_hbf_stacks
        if active_hbf_stacks and average_power_w is not None
        else None
    )
    summary: dict[str, Any] = {
        "enabled": True,
        "boot_temperature_c": _number(
            state.get("thermal_boot_temperature_c", 0),
            "thermal boot temperature",
        ),
        "peak_temperature_c": _number(
            state.get("thermal_peak_temperature_c", 0),
            "thermal peak temperature",
        ),
        "final_temperature_c": _number(
            state.get("thermal_final_temperature_c", 0),
            "thermal final temperature",
        ),
        "throttled_stacks_at_finish": _integer(
            state.get("thermal_throttled_stacks", 0),
            "thermal throttled stacks",
        ),
        "throttle_engagements": _integer(
            hbf.get("thermal_throttle_engagements", 0),
            "thermal throttle engagements",
        ),
        "throttled_media_ops": _integer(
            hbf.get("thermal_throttled_media_ops", 0),
            "thermal throttled media ops",
        ),
        "throttle_wait_ns": _number(
            hbf.get("thermal_throttle_wait_work_ns", 0),
            "thermal throttle wait",
        ),
        "pacing_busy_ns": _number(
            hbf.get("thermal_pacing_busy_ns", 0), "thermal pacing busy"
        ),
        "throttled_span_ns": span_ns,
        "throttled_span_fraction": (
            span_ns / (completion_ns * active_hbf_stacks)
            if completion_ns > 0 and active_hbf_stacks
            else None
        ),
        "media_energy_j": energy_j,
        "average_media_power_w": average_power_w,
        "average_media_power_w_per_stack": per_stack_power_w,
    }
    if thermal_identity is not None and thermal_identity.get("enabled"):
        pacing_power = float(thermal_identity["pacing_power_w"])
        summary["start_state"] = thermal_identity["start_state"]
        summary["boundary_temperature_c"] = thermal_identity[
            "boundary_temperature_c"
        ]
        summary["pacing_power_w_per_stack"] = pacing_power
        summary["sustainable_read_GBps_per_stack"] = thermal_identity[
            "sustainable_read_GBps_per_stack"
        ]
        # Pacing may only bound power; a violation means governor pacing
        # was bypassed somewhere and the row cannot be trusted. The 5%
        # allowance covers boundary partial slots at the window edges.
        if (
            per_stack_power_w is not None
            and per_stack_power_w > pacing_power * 1.05
            and summary["throttled_span_fraction"] is not None
            and summary["throttled_span_fraction"] > 0.99
        ):
            raise FixedFootprintMetricsError(
                "average per-stack media power exceeds the pacing budget "
                "while fully throttled"
            )
        summary["per_stack_power_within_pacing_budget"] = (
            per_stack_power_w is None
            or per_stack_power_w <= pacing_power * 1.05
        )
    return summary


def _latency_summary(
    matrix: Mapping[str, Any], targets: Iterable[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for operation in ("read", "write"):
        transactions = 0
        logical_bytes = 0
        physical_bytes = 0
        latency_work_ns = 0.0
        queue_wait_work_ns = 0.0
        service_work_ns = 0.0
        minima: list[float] = []
        maxima: list[float] = []
        for target in targets:
            target_row = _mapping(
                matrix.get(target), f"latency matrix target {target}"
            )
            row = _mapping(
                target_row.get(operation),
                f"latency matrix {target}.{operation}",
            )
            count = _integer(
                row.get("transactions"),
                f"latency matrix {target}.{operation}.transactions",
            )
            transactions += count
            logical_bytes += _integer(
                row.get("logical_bytes"),
                f"latency matrix {target}.{operation}.logical_bytes",
            )
            physical_bytes += _integer(
                row.get("physical_bytes"),
                f"latency matrix {target}.{operation}.physical_bytes",
            )
            latency_work_ns += _number(
                row.get("latency_work_ns"),
                f"latency matrix {target}.{operation}.latency_work_ns",
            )
            queue_wait_work_ns += _number(
                row.get("queue_wait_work_ns"),
                f"latency matrix {target}.{operation}.queue_wait_work_ns",
            )
            service_work_ns += _number(
                row.get("service_work_ns"),
                f"latency matrix {target}.{operation}.service_work_ns",
            )
            if count:
                minima.append(
                    _number(
                        row.get("min_latency_ns"),
                        f"latency matrix {target}.{operation}.min_latency_ns",
                    )
                )
                maxima.append(
                    _number(
                        row.get("max_latency_ns"),
                        f"latency matrix {target}.{operation}.max_latency_ns",
                    )
                )
        result[operation] = {
            "request_transactions": transactions,
            "logical_bytes": logical_bytes,
            "physical_bytes": physical_bytes,
            "mean_latency_ns": (
                latency_work_ns / transactions if transactions else None
            ),
            "mean_queue_wait_ns": (
                queue_wait_work_ns / transactions if transactions else None
            ),
            "mean_service_ns": (
                service_work_ns / transactions if transactions else None
            ),
            "min_latency_ns": min(minima) if minima else None,
            "max_latency_ns": max(maxima) if maxima else None,
            "latency_work_ns": latency_work_ns,
            "queue_wait_work_ns": queue_wait_work_ns,
            "service_work_ns": service_work_ns,
        }
    return result


def _device_traffic(
    devices: Mapping[str, Any], *, external_kind: str | None
) -> dict[str, dict[str, Any]]:
    hbm_raw = devices.get("hbm")
    hbf_raw = devices.get("hbf")
    external_raw = devices.get("external")
    link_raw = devices.get("base_die_link")
    hbm = {} if hbm_raw is None else _mapping(hbm_raw, "HBM device totals")
    hbf = {} if hbf_raw is None else _mapping(hbf_raw, "HBF device totals")
    external = (
        {}
        if external_raw is None
        else _mapping(external_raw, "external device totals")
    )
    link = (
        {}
        if link_raw is None
        else _mapping(link_raw, "base-die-link totals")
    )

    def integer(row: Mapping[str, Any], key: str, description: str) -> int:
        return _integer(row.get(key, 0), f"{description}.{key}")

    return {
        "hbm": {
            "label": "HBM",
            "read_bytes": integer(hbm, "read_bytes", "HBM device totals"),
            "write_bytes": integer(hbm, "write_bytes", "HBM device totals"),
            "media_read_operations": None,
            "media_write_operations": None,
        },
        "hbf": {
            "label": "HBF",
            "read_bytes": integer(
                hbf, "physical_read_bytes", "HBF device totals"
            ),
            "write_bytes": integer(
                hbf, "physical_write_bytes", "HBF device totals"
            ),
            "logical_read_bytes": integer(
                hbf, "logical_read_bytes", "HBF device totals"
            ),
            "logical_write_bytes": integer(
                hbf, "logical_write_bytes", "HBF device totals"
            ),
            "media_read_operations": integer(
                hbf, "page_reads", "HBF device totals"
            ),
            "media_write_operations": integer(
                hbf, "page_programs", "HBF device totals"
            ),
        },
        "external": {
            "label": external_kind or "external",
            "read_bytes": integer(
                external, "read_bytes", "external device totals"
            ),
            "write_bytes": integer(
                external, "write_bytes", "external device totals"
            ),
            "media_read_operations": integer(
                external, "read_requests", "external device totals"
            ),
            "media_write_operations": integer(
                external, "write_requests", "external device totals"
            ),
            "read_wire_bytes": integer(
                external, "s2m_wire_bytes", "external device totals"
            ),
            "write_wire_bytes": integer(
                external, "m2s_wire_bytes", "external device totals"
            ),
        },
        "interconnect": {
            "label": "base-die-link",
            "read_bytes": integer(
                link, "read_bytes", "base-die-link totals"
            ),
            "write_bytes": integer(
                link, "write_bytes", "base-die-link totals"
            ),
            "media_read_operations": integer(
                link, "read_transfers", "base-die-link totals"
            ),
            "media_write_operations": integer(
                link, "write_transfers", "base-die-link totals"
            ),
        },
    }


def _summarize_scope(
    *,
    latency_matrix: Mapping[str, Any],
    device_totals: Mapping[str, Any],
    elapsed_ns: float,
    external_kind: str | None,
) -> dict[str, dict[str, Any]]:
    traffic = _device_traffic(device_totals, external_kind=external_kind)
    result: dict[str, dict[str, Any]] = {}
    for medium, targets in TARGETS_BY_MEDIUM.items():
        latency = _latency_summary(latency_matrix, targets)
        row = {**traffic[medium], "transactions": latency}
        read_bytes = int(row["read_bytes"])
        write_bytes = int(row["write_bytes"])
        total_bytes = read_bytes + write_bytes
        row["total_bytes"] = total_bytes
        row["read_fraction"] = read_bytes / total_bytes if total_bytes else None
        row["write_fraction"] = write_bytes / total_bytes if total_bytes else None
        row["read_throughput_GBps"] = _rate(read_bytes, elapsed_ns)
        row["write_throughput_GBps"] = _rate(write_bytes, elapsed_ns)
        row["total_throughput_GBps"] = _rate(
            read_bytes + write_bytes, elapsed_ns
        )
        result[medium] = row
    return result


def _object_distribution(
    phase_rows: Iterable[Mapping[str, Any]],
    scenario_media: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate exact submitted transactions by externally bound object class."""

    grouped: dict[str, dict[str, Any]] = {}
    for raw_phase in phase_rows:
        phase = _mapping(raw_phase, "object phase row")
        object_class = str(phase.get("object_class", ""))
        if not object_class:
            raise FixedFootprintMetricsError("object phase has no object class")
        group = grouped.setdefault(
            object_class,
            {
                "object_class": object_class,
                "phase_ids": [],
                "objects": [],
                "source_logical_read_bytes": 0,
                "source_logical_write_bytes": 0,
                "medium": {
                    medium: {
                        operation: {
                            "request_transactions": 0,
                            "mapped_logical_bytes": 0,
                            "mapped_physical_bytes": 0,
                            "latency_work_ns": 0.0,
                            "queue_wait_work_ns": 0.0,
                            "service_work_ns": 0.0,
                        }
                        for operation in ("read", "write")
                    }
                    for medium in TARGETS_BY_MEDIUM
                },
            },
        )
        group["phase_ids"].append(str(phase.get("phase_id", "")))
        group["objects"] = list(
            dict.fromkeys(
                [*group["objects"], *(str(item) for item in phase.get("objects", []))]
            )
        )
        group["source_logical_read_bytes"] += int(
            phase["source_logical_read_bytes"]
        )
        group["source_logical_write_bytes"] += int(
            phase["source_logical_write_bytes"]
        )
        phase_media = _mapping(phase.get("media"), "object phase media")
        for medium in TARGETS_BY_MEDIUM:
            medium_row = _mapping(
                phase_media.get(medium), f"object phase medium {medium}"
            )
            transactions = _mapping(
                medium_row.get("transactions"),
                f"object phase medium {medium} transactions",
            )
            for operation in ("read", "write"):
                source = _mapping(
                    transactions.get(operation),
                    f"object phase {medium}.{operation}",
                )
                target = group["medium"][medium][operation]
                target["request_transactions"] += int(
                    source["request_transactions"]
                )
                target["mapped_logical_bytes"] += int(source["logical_bytes"])
                target["mapped_physical_bytes"] += int(source["physical_bytes"])
                for key in (
                    "latency_work_ns",
                    "queue_wait_work_ns",
                    "service_work_ns",
                ):
                    target[key] += float(source[key])

    result: list[dict[str, Any]] = []
    for group in grouped.values():
        source_total = (
            group["source_logical_read_bytes"]
            + group["source_logical_write_bytes"]
        )
        group["source_logical_total_bytes"] = source_total
        for medium in TARGETS_BY_MEDIUM:
            total_mapped = 0
            scenario_row = _mapping(
                scenario_media.get(medium), f"scenario medium {medium}"
            )
            scenario_transactions = _mapping(
                scenario_row.get("transactions"),
                f"scenario medium {medium} transactions",
            )
            for operation in ("read", "write"):
                row = group["medium"][medium][operation]
                count = int(row["request_transactions"])
                row["mean_latency_ns"] = (
                    row["latency_work_ns"] / count if count else None
                )
                row["mean_queue_wait_ns"] = (
                    row["queue_wait_work_ns"] / count if count else None
                )
                row["mean_service_ns"] = (
                    row["service_work_ns"] / count if count else None
                )
                denominator = int(
                    _mapping(
                        scenario_transactions.get(operation),
                        f"scenario medium {medium}.{operation}",
                    )["logical_bytes"]
                )
                row["fraction_of_medium_mapped_bytes"] = (
                    row["mapped_logical_bytes"] / denominator
                    if denominator
                    else None
                )
                total_mapped += row["mapped_logical_bytes"]
            group["medium"][medium]["mapped_total_bytes"] = total_mapped
        result.append(group)
    return result


def build_fixed_footprint_metrics(
    *,
    final_measurement: Mapping[str, Any],
    logical_read_bytes: int,
    logical_write_bytes: int,
    phase_measurements: Iterable[Mapping[str, Any]],
    external_kind: str | None,
    active_hbf_stacks: int = 0,
    thermal_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one topology row and its object-attributed traffic table."""

    final = _mapping(final_measurement, "final measurement")
    completion_ns = _number(
        final.get("drained_frontier_ns"), "final drained frontier"
    )
    source_read = _integer(logical_read_bytes, "source logical read bytes")
    source_write = _integer(logical_write_bytes, "source logical write bytes")
    source_total = source_read + source_write
    if source_total == 0:
        raise FixedFootprintMetricsError("source trace has no logical traffic")
    final_latency = _mapping(
        final.get("transaction_latency_by_target"), "final latency matrix"
    )
    final_devices = _mapping(
        final.get("device_workload_totals"), "final device totals"
    )
    media = _summarize_scope(
        latency_matrix=final_latency,
        device_totals=final_devices,
        elapsed_ns=completion_ns,
        external_kind=external_kind,
    )
    physical_media_bytes = sum(
        int(media[name]["total_bytes"]) for name in ("hbm", "hbf", "external")
    )

    phase_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(phase_measurements):
        phase = _mapping(raw, f"phase measurement {index}")
        completion = _mapping(
            phase.get("completion"), f"phase measurement {index}.completion"
        )
        elapsed = _number(
            completion.get("elapsed_ns"),
            f"phase measurement {index}.completion.elapsed_ns",
        )
        phase_rows.append(
            {
                "phase_id": str(phase.get("phase_id", "")),
                "stage": str(phase.get("stage", "")),
                "layer": phase.get("layer"),
                "object_class": str(phase.get("object_class", "")),
                "objects": list(phase.get("objects", [])),
                "source_logical_read_bytes": _integer(
                    phase.get("source_logical_read_bytes", 0),
                    f"phase measurement {index}.source read bytes",
                ),
                "source_logical_write_bytes": _integer(
                    phase.get("source_logical_write_bytes", 0),
                    f"phase measurement {index}.source write bytes",
                ),
                "phase_elapsed_time_ns": elapsed,
                "media": _summarize_scope(
                    latency_matrix=_mapping(
                        completion.get("transaction_latency_by_target"),
                        f"phase measurement {index}.latency matrix",
                    ),
                    device_totals=_mapping(
                        completion.get("device_delta"),
                        f"phase measurement {index}.device delta",
                    ),
                    elapsed_ns=elapsed,
                    external_kind=external_kind,
                ),
            }
        )

    return {
        "schema": METRICS_SCHEMA,
        "final_drain_time_ns": completion_ns,
        "source_logical_traffic": {
            "read_bytes": source_read,
            "write_bytes": source_write,
            "total_bytes": source_total,
            "read_fraction": source_read / source_total,
            "write_fraction": source_write / source_total,
            "effective_trace_throughput_GBps": _rate(
                source_total, completion_ns
            ),
        },
        "media_traffic": {
            "actual_memory_media_bytes": physical_media_bytes,
            "source_logical_bytes": source_total,
            "actual_to_logical_ratio": physical_media_bytes / source_total,
            "definition": (
                "sum_of_HBM_HBF_and_external_media_bytes_divided_by_"
                "source_trace_logical_bytes"
            ),
            "interconnect_bytes_excluded": True,
        },
        "media": media,
        "thermal": _thermal_summary(
            final_devices,
            thermal_identity=thermal_identity,
            active_hbf_stacks=active_hbf_stacks,
            completion_ns=completion_ns,
        ),
        "object_distribution": _object_distribution(phase_rows, media),
        "phase_attribution": phase_rows,
        "semantics": {
            "final_drain_time": (
                "trace_start_through_all_trace_caused_work_and_final_drain"
            ),
            "medium_throughput_denominator": "final_drain_time_ns",
            "mean_latency": "target_transaction_completion_latency",
            "object_attribution": (
                "exact_submitted_target_transactions_grouped_by_phase_identity_"
                "outside_the_semantic_blind_simulator"
            ),
            "device_physical_traffic_attribution": (
                "reported_by_medium_for_the_whole_run_because_asynchronous_"
                "media_work_can_outlive_its_source_phase"
            ),
            "effective_trace_throughput": (
                "fixed_source_trace_logical_bytes_divided_by_final_drain_time"
            ),
        },
    }
