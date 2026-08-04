#!/usr/bin/env python3
"""Independently reduce canonical events into accounting invariants.

The reducer deliberately consumes only the public ledger contract.  It does
not import either oracle or production scheduling/accounting helpers, and it
never parses diagnostic text.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable

from validation.contracts import ContractError


# Busy work is reconstructed by subtracting serialized absolute timestamps.
# At multi-millisecond frontiers that subtraction can accumulate a few
# nanoseconds of round-off even though every event endpoint compares at 1e-9.
ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-12


def _events(
    records: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    selected = [record for record in records if record["kind"] == "event"]
    if predicate is not None:
        selected = [record for record in selected if predicate(record)]
    return selected


def _duration(event: dict[str, Any]) -> float:
    return float(event["finish_ns"]) - float(event["start_ns"])


def _sum_duration(events: list[dict[str, Any]]) -> float:
    return math.fsum(_duration(event) for event in events)


def _count_action(
    events: list[dict[str, Any]],
    action: str,
) -> int:
    return sum(event["action"] == action for event in events)


def _counter(
    counters: dict[str, Any],
    key: str,
    *,
    source: str,
) -> Any:
    if key not in counters:
        raise ContractError(
            f"{source}: summary.counters.{key} is required by ledger reducer")
    return counters[key]


def _expect_equal(
    actual: Any,
    expected: Any,
    path: str,
    *,
    source: str,
) -> None:
    if (
        type(actual) is not type(expected)
        or actual != expected
    ):
        raise ContractError(
            f"{source}: {path} is not ledger-reducible: "
            f"expected {expected!r}, observed {actual!r}")


def _expect_number(
    actual: Any,
    expected: int | float,
    path: str,
    *,
    source: str,
) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isclose(
            float(actual),
            float(expected),
            rel_tol=REL_TOLERANCE,
            abs_tol=ABS_TOLERANCE,
        )
    ):
        raise ContractError(
            f"{source}: {path} is not ledger-reducible: "
            f"expected {expected!r}, observed {actual!r}")


def _expect_counter(
    counters: dict[str, Any],
    key: str,
    expected: int | float | bool,
    *,
    source: str,
) -> None:
    actual = _counter(counters, key, source=source)
    path = f"summary.counters.{key}"
    if isinstance(expected, bool):
        _expect_equal(actual, expected, path, source=source)
    else:
        _expect_number(actual, expected, path, source=source)


def _request_and_completion_audit(
    records: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    requests = [
        record for record in records if record["kind"] == "request"]
    completions = [
        record for record in records if record["kind"] == "completion"]
    request_by_record_id = {record["id"]: record for record in requests}
    request_by_id: dict[str, dict[str, Any]] = {}
    completions_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    events_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for request in requests:
        request_id = request["request_id"]
        if request_id in request_by_id:
            raise ContractError(
                f"{source}: duplicate request_id {request_id!r}")
        request_by_id[request_id] = request

    for record in records:
        if record["kind"] not in {"event", "state", "completion"}:
            continue
        parent = request_by_record_id.get(record["parent_id"])
        if parent is None:
            continue
        if record["request_id"] != parent["request_id"]:
            raise ContractError(
                f"{source}: {record['id']} request_id does not match parent")
        if record["kind"] == "event":
            events_by_request[record["request_id"]].append(record)
        elif record["kind"] == "completion":
            completions_by_request[record["request_id"]].append(record)

    for request in requests:
        request_id = request["request_id"]
        matching = completions_by_request[request_id]
        if len(matching) != 1:
            raise ContractError(
                f"{source}: request {request_id!r} has "
                f"{len(matching)} completions, expected exactly one")
        completion = matching[0]
        _expect_number(
            completion["arrival_ns"],
            request["arrival_ns"],
            f"{completion['id']}.arrival_ns",
            source=source,
        )
        _expect_number(
            completion["logical_bytes"],
            request["bytes"],
            f"{completion['id']}.logical_bytes",
            source=source,
        )
        if completion["finish_ns"] < completion["arrival_ns"]:
            raise ContractError(
                f"{source}: {completion['id']} finishes before arrival")
        critical_finishes = [
            event["finish_ns"]
            for event in events_by_request[request_id]
            if event["critical"]
        ]
        causal_finish = max(
            critical_finishes,
            default=float(request["arrival_ns"]),
        )
        _expect_number(
            completion["finish_ns"],
            causal_finish,
            f"{completion['id']}.finish_ns(max-critical-event)",
            source=source,
        )

    if len(completions) != len(requests):
        raise ContractError(
            f"{source}: completion count does not equal request count")

    finish_ns = max(
        (float(record["finish_ns"]) for record in completions),
        default=0.0,
    )
    return {
        "requests": requests,
        "request_by_id": request_by_id,
        "completions": completions,
        "events_by_request": events_by_request,
        "finish_ns": finish_ns,
    }


def _hbm_metrics(
    events: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, int | float]:
    hbm_events = [event for event in events if event["model"] == "hbm"]
    actions = Counter(event["action"] for event in hbm_events)
    burst_bytes = (
        int(config["channel_width_bits"])
        // int(config["pseudo_channels_per_channel"])
        // 8
        * int(config["burst_length"])
    )
    activations = actions["ACT"]
    precharges = actions["PRE"]
    column_commands = actions["RD"] + actions["WR"]
    bus_events = [
        event for event in hbm_events if event["category"] == "hbm_bus"]
    accesses_by_pch = Counter(event["resource"] for event in bus_events)
    return {
        "read_bytes": actions["read_burst"] * burst_bytes,
        "write_bytes": actions["write_burst"] * burst_bytes,
        "row_hits": column_commands - activations,
        "row_misses": activations - precharges,
        "row_conflicts": precharges,
        "activations": activations,
        "precharges": precharges,
        "refresh_count": 0,
        "bus_busy_ns": _sum_duration(bus_events),
        "finish_ns": max(
            (float(event["finish_ns"]) for event in hbm_events),
            default=0.0,
        ),
        "pseudo_channels": (
            int(config["stacks"])
            * int(config["channels_per_stack"])
            * int(config["pseudo_channels_per_channel"])
        ),
        "active_pseudo_channels": len(accesses_by_pch),
        "max_pseudo_channel_accesses": max(
            accesses_by_pch.values(), default=0),
        "burst_bytes": burst_bytes,
    }


def _plane_resource(resource: str) -> str:
    marker = "/subarray"
    return resource.partition(marker)[0]


def _hbf_metrics(
    events: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    hbf_events = [event for event in events if event["model"] == "hbf"]
    actions = Counter(event["action"] for event in hbf_events)
    page_size = int(config["page_size_bytes"])
    wire_bytes = page_size + int(config["oob_bytes_per_page"])
    page_reads = actions["user/array_read"] + actions["gc/array_read"]
    data_programs = actions["user/array_program"]
    mapping_programs = actions["mapping/array_program"]
    relocations = actions["gc/array_program"]
    page_programs = data_programs + mapping_programs + relocations
    gc_mapping_relocations = actions["gc_mapping_checkpoint_relocate"]
    metadata_ops = (
        actions["resident_mapping_lookup"]
        + actions["resident_mapping_update_access"]
    )

    channel_events = [
        event for event in hbf_events
        if event["category"] == "flash_channel"
    ]
    channel_command_events = [
        event for event in channel_events
        if "cmd_addr_channel" in event["action"]
    ]
    channel_data_events = [
        event for event in channel_events
        if "cmd_addr_channel" not in event["action"]
    ]
    hbio_events = [
        event for event in hbf_events if event["category"] == "hbio"]
    hbio_command_events = [
        event for event in hbio_events
        if event["action"] == "user/request_hbio"
    ]
    hbio_data_events = [
        event for event in hbio_events
        if event["action"] != "user/request_hbio"
    ]
    flash_array_events = [
        event for event in hbf_events
        if event["category"] == "flash_array"
    ]
    media_lane_events = [
        event for event in hbf_events
        if event["category"] == "media_lane"
    ]
    page_buffer_events = [
        event for event in hbf_events
        if event["category"] == "page_buffer"
    ]
    sequencer_events = [
        event for event in hbf_events
        if event["category"] == "sequencer"
    ]

    return {
        "physical_read_bytes": page_reads * page_size,
        "physical_write_bytes": page_programs * page_size,
        "data_program_payload_bytes": data_programs * page_size,
        "mapping_program_payload_bytes": mapping_programs * page_size,
        "gc_relocation_payload_bytes": relocations * page_size,
        "page_reads": page_reads,
        "data_programs": data_programs,
        "page_programs": page_programs,
        "mapping_page_programs": mapping_programs,
        "block_erases": actions["gc/block_erase"],
        "gc_runs": actions["gc/block_erase"],
        "gc_relocations": relocations,
        "gc_data_relocations": relocations - gc_mapping_relocations,
        "gc_mapping_relocations": gc_mapping_relocations,
        "mapping_lookup_ops": actions["resident_mapping_lookup"],
        "mapping_update_ops": actions["resident_mapping_update_access"],
        "mapping_gc_update_ops": actions["gc_resident_mapping_update"],
        "mapping_dram_issue_busy_ns": (
            metadata_ops * float(config["ctrl_dram_issue_ns"])
        ),
        "flash_scheduler_enqueues": len(sequencer_events),
        "flash_scheduler_issues": len(sequencer_events),
        "ecc_decode_ops": (
            actions["user/ecc_decode_issue"]
            + actions["gc/ecc_decode_issue"]
        ),
        "ecc_decode_codeword_bytes": (
            (
                actions["user/ecc_decode_issue"]
                + actions["gc/ecc_decode_issue"]
            )
            * wire_bytes
        ),
        "ecc_encode_ops": (
            actions["user/ecc_encode_issue"]
            + actions["mapping/ecc_encode_issue"]
            + actions["gc/ecc_encode_issue"]
        ),
        "ecc_encode_codeword_bytes": (
            (
                actions["user/ecc_encode_issue"]
                + actions["mapping/ecc_encode_issue"]
                + actions["gc/ecc_encode_issue"]
            )
            * wire_bytes
        ),
        "ecc_issue_busy_ns": _sum_duration([
            event for event in hbf_events
            if event["category"] == "ecc_issue"
        ]),
        "media_busy_ns": _sum_duration(flash_array_events),
        "channel_command_busy_ns": _sum_duration(channel_command_events),
        "channel_data_busy_ns": _sum_duration(channel_data_events),
        "tsv_busy_ns": _sum_duration([
            event for event in hbf_events
            if event["category"] == "tsv"
        ]),
        "sram_busy_ns": _sum_duration([
            event for event in hbf_events
            if event["category"] == "sram"
        ]),
        "hb_io_command_busy_ns": _sum_duration(hbio_command_events),
        "hb_io_data_busy_ns": _sum_duration(hbio_data_events),
        "finish_ns": max(
            (float(event["finish_ns"]) for event in hbf_events),
            default=0.0,
        ),
        "total_pages": (
            int(config["stacks"])
            * int(config["channels_per_stack"])
            * int(config["dies_per_channel"])
            * int(config["planes_per_die"])
            * int(config["blocks_per_plane"])
            * int(config["pages_per_block"])
        ),
        "active_planes": len({
            _plane_resource(event["resource"])
            for event in flash_array_events
        }),
        "active_media_lanes": len({
            event["resource"] for event in media_lane_events
        }),
        "active_subarrays": len({
            event["resource"]
            for event in flash_array_events
            if "/subarray" in event["resource"]
        }),
        "active_page_buffer_banks": len({
            event["resource"] for event in page_buffer_events
        }),
        "active_channels": len({
            event["resource"] for event in channel_events
        }),
        "active_dies": len({
            event["resource"] for event in sequencer_events
        }),
    }


def _external_metrics(
    common: dict[str, Any],
    events: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    external_events = [
        event for event in events if event["model"] == "external"
    ]
    requests = common["requests"]
    completions = common["completions"]
    event_by_request = common["events_by_request"]

    def selected(action_suffix: str) -> list[dict[str, Any]]:
        return [
            event for event in external_events
            if event["action"].endswith(action_suffix)
        ]

    reads = [request for request in requests if request["action"] == "read"]
    writes = [
        request for request in requests if request["action"] == "write"
    ]
    request_count = len(requests)
    read_bytes = sum(int(request["bytes"]) for request in reads)
    write_bytes = sum(int(request["bytes"]) for request in writes)
    controller_events = selected("_controller_issue")
    media_events = selected("_media_transfer")
    m2s_events = selected("_m2s_transfer")
    s2m_events = selected("_s2m_transfer")

    outstanding_wait = 0.0
    m2s_wait = 0.0
    controller_wait = 0.0
    media_wait = 0.0
    s2m_wait = 0.0
    max_outstanding = 0
    prior_finishes: list[float] = []
    active_channels: set[str] = set()
    for request, completion in zip(requests, completions, strict=True):
        request_id = request["request_id"]
        owned = event_by_request[request_id]

        def one(suffix: str) -> dict[str, Any] | None:
            matches = [
                event for event in owned
                if event["action"].endswith(suffix)
            ]
            if len(matches) > 1:
                raise ContractError(
                    f"{source}: {request_id} has duplicate {suffix} events"
                )
            return matches[0] if matches else None

        m2s = one("_m2s_transfer")
        media = one("_media_transfer")
        s2m = one("_s2m_transfer")
        if m2s is None or media is None or s2m is None:
            raise ContractError(
                f"{source}: {request_id} lacks a mandatory transfer event"
            )
        controller = one("_controller_issue")
        admitted = float(completion["start_ns"])
        outstanding_wait += admitted - float(request["arrival_ns"])
        m2s_wait += float(m2s["start_ns"]) - admitted
        controller_ready = (
            float(m2s["finish_ns"])
            + float(config["one_way_propagation_ns"])
        )
        controller_start = (
            float(controller["start_ns"])
            if controller is not None
            else controller_ready
        )
        controller_wait += controller_start - controller_ready
        media_ready = (
            controller_start
            + float(config["controller_issue_ns"])
            + float(config["controller_processing_ns"])
            + float(
                config[
                    "media_read_latency_ns"
                    if request["action"] == "read"
                    else "media_write_latency_ns"
                ]
            )
        )
        media_wait += float(media["start_ns"]) - media_ready
        s2m_wait += (
            float(s2m["start_ns"]) - float(media["finish_ns"])
        )
        active_channels.add(str(media["resource"]))
        active_before = sum(
            finish > admitted for finish in prior_finishes
        )
        max_outstanding = max(max_outstanding, active_before + 1)
        prior_finishes.append(float(completion["finish_ns"]))
        _expect_number(
            completion["physical_bytes"],
            request["bytes"],
            f"{completion['id']}.physical_bytes",
            source=source,
        )

    command_bytes = int(config["command_bytes"])
    completion_bytes = int(config["completion_bytes"])
    metrics: dict[str, Any] = {
        "read_requests": len(reads),
        "write_requests": len(writes),
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "media_channels": int(config["media_channels"]),
        "active_media_channels": len(active_channels),
        "max_device_outstanding": max_outstanding,
        "outstanding_wait_ns": outstanding_wait,
        "controller_queue_wait_ns": controller_wait,
        "controller_issue_busy_ns": _sum_duration(controller_events),
        "controller_processing_work_ns": (
            request_count * float(config["controller_processing_ns"])
        ),
        "media_queue_wait_ns": media_wait,
        "media_read_latency_work_ns": (
            len(reads) * float(config["media_read_latency_ns"])
        ),
        "media_write_latency_work_ns": (
            len(writes) * float(config["media_write_latency_ns"])
        ),
        "media_read_busy_ns": _sum_duration([
            event for event in media_events
            if event["action"].startswith("external_read_")
        ]),
        "media_write_busy_ns": _sum_duration([
            event for event in media_events
            if event["action"].startswith("external_write_")
        ]),
        "m2s_payload_bytes": write_bytes,
        "m2s_protocol_bytes": request_count * command_bytes,
        "m2s_wire_bytes": write_bytes + request_count * command_bytes,
        "s2m_payload_bytes": read_bytes,
        "s2m_protocol_bytes": request_count * completion_bytes,
        "s2m_wire_bytes": read_bytes + request_count * completion_bytes,
        "m2s_queue_wait_ns": m2s_wait,
        "s2m_queue_wait_ns": s2m_wait,
        "m2s_busy_ns": _sum_duration(m2s_events),
        "s2m_busy_ns": _sum_duration(s2m_events),
        "transport_propagation_work_ns": (
            request_count
            * 2
            * float(config["one_way_propagation_ns"])
        ),
        "finish_ns": common["finish_ns"],
    }
    return metrics


def _validate_hbm_summary(
    summary: dict[str, Any],
    common: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    counters = summary["counters"]
    metrics = _hbm_metrics(events, summary["config"])
    for key in (
        "read_bytes",
        "write_bytes",
        "row_hits",
        "row_misses",
        "row_conflicts",
        "activations",
        "precharges",
        "refresh_count",
        "bus_busy_ns",
        "pseudo_channels",
        "active_pseudo_channels",
        "max_pseudo_channel_accesses",
    ):
        _expect_counter(counters, key, metrics[key], source=source)
    _expect_counter(
        counters, "finish_ns", common["finish_ns"], source=source)
    derived = summary["config"].get("derived")
    if not isinstance(derived, dict):
        raise ContractError(
            f"{source}: summary.config.derived is required by reducer")
    _expect_number(
        derived.get("burst_bytes"),
        metrics["burst_bytes"],
        "summary.config.derived.burst_bytes",
        source=source,
    )
    return metrics


def _validate_external_summary(
    summary: dict[str, Any],
    common: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    metrics = _external_metrics(
        common,
        events,
        summary["config"],
        source=source,
    )
    counters = summary["counters"]
    for key, expected in metrics.items():
        _expect_counter(counters, key, expected, source=source)
    if (
        metrics["m2s_wire_bytes"]
        != metrics["m2s_payload_bytes"] + metrics["m2s_protocol_bytes"]
        or metrics["s2m_wire_bytes"]
        != metrics["s2m_payload_bytes"] + metrics["s2m_protocol_bytes"]
    ):
        raise ContractError(
            f"{source}: external directional wire accounting does not conserve"
        )
    return metrics


def _validate_final_hbf_state(
    records: list[dict[str, Any]],
    counters: dict[str, Any],
    metrics: dict[str, Any],
    *,
    source: str,
) -> None:
    states = [record for record in records if record["kind"] == "state"]
    if not states:
        return
    attributes = states[-1]["attributes"]
    if not attributes.get("quiescent"):
        return
    blocks = attributes.get("blocks")
    if not isinstance(blocks, list):
        raise ContractError(
            f"{source}: final quiescent state lacks block accounting")
    reduced = {
        "free_pages": sum(int(block["free_pages"]) for block in blocks),
        "valid_pages": sum(int(block["valid_pages"]) for block in blocks),
        "invalid_pages": sum(
            int(block["invalid_pages"]) for block in blocks),
        "pending_program_pages": sum(
            int(block["pending_program_pages"]) for block in blocks),
        "pending_mapping_publications": sum(
            int(block["pending_mapping_publications"]) for block in blocks),
    }
    for key, value in reduced.items():
        _expect_counter(counters, key, value, source=source)
    partition = (
        reduced["free_pages"]
        + reduced["valid_pages"]
        + reduced["invalid_pages"]
        + reduced["pending_program_pages"]
    )
    if partition != metrics["total_pages"]:
        raise ContractError(
            f"{source}: final HBF page-state partition is {partition}, "
            f"expected {metrics['total_pages']}")
    _expect_number(
        attributes.get("free_pages"),
        reduced["free_pages"],
        "final_state.attributes.free_pages",
        source=source,
    )


def _validate_hbf_summary(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    common: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    counters = summary["counters"]
    metrics = _hbf_metrics(events, summary["config"])
    requests = common["requests"]
    read_requests = [
        request for request in requests if request["action"] == "read"]
    write_requests = [
        request for request in requests if request["action"] == "write"]
    metrics.update({
        "read_requests": len(read_requests),
        "program_requests": len(write_requests),
        "logical_read_bytes": sum(
            int(request["bytes"])
            for request in read_requests
            if request["address_space"] == "logical"
        ),
        "logical_write_bytes": sum(
            int(request["bytes"])
            for request in write_requests
            if request["address_space"] == "logical"
        ),
        "page_read_admission_events": len(read_requests),
    })
    metrics["mapping_user_lookup_ops"] = metrics["mapping_lookup_ops"]
    metrics["mapping_gc_lookup_ops"] = 0
    metrics["mapping_user_update_ops"] = (
        metrics["mapping_update_ops"]
        - metrics["mapping_gc_update_ops"]
    )
    metrics["write_amplification"] = (
        metrics["physical_write_bytes"]
        / metrics["logical_write_bytes"]
        if metrics["logical_write_bytes"]
        else None
    )

    for key in (
        "read_requests",
        "program_requests",
        "logical_read_bytes",
        "logical_write_bytes",
        "physical_read_bytes",
        "physical_write_bytes",
        "data_program_payload_bytes",
        "mapping_program_payload_bytes",
        "gc_relocation_payload_bytes",
        "page_reads",
        "data_programs",
        "page_programs",
        "mapping_page_programs",
        "block_erases",
        "gc_runs",
        "gc_relocations",
        "gc_data_relocations",
        "gc_mapping_relocations",
        "mapping_lookup_ops",
        "mapping_user_lookup_ops",
        "mapping_gc_lookup_ops",
        "mapping_update_ops",
        "mapping_user_update_ops",
        "mapping_gc_update_ops",
        "mapping_dram_issue_busy_ns",
        "page_read_admission_events",
        "flash_scheduler_enqueues",
        "flash_scheduler_issues",
        "ecc_decode_ops",
        "ecc_decode_codeword_bytes",
        "ecc_encode_ops",
        "ecc_encode_codeword_bytes",
        "ecc_issue_busy_ns",
        "media_busy_ns",
        "channel_command_busy_ns",
        "channel_data_busy_ns",
        "tsv_busy_ns",
        "sram_busy_ns",
        "hb_io_command_busy_ns",
        "hb_io_data_busy_ns",
        "total_pages",
        "active_planes",
        "active_media_lanes",
        "active_subarrays",
        "active_page_buffer_banks",
        "active_channels",
        "active_dies",
    ):
        _expect_counter(counters, key, metrics[key], source=source)
    _expect_counter(
        counters, "finish_ns", common["finish_ns"], source=source)
    _expect_counter(counters, "accounting_verified", True, source=source)

    component_bytes = (
        metrics["data_program_payload_bytes"]
        + metrics["mapping_program_payload_bytes"]
        + metrics["gc_relocation_payload_bytes"]
    )
    if component_bytes != metrics["physical_write_bytes"]:
        raise ContractError(
            f"{source}: HBF physical-write components total "
            f"{component_bytes}, expected {metrics['physical_write_bytes']}")
    _validate_final_hbf_state(
        records, counters, metrics, source=source)
    return metrics


def _validate_hybrid_summary(
    summary: dict[str, Any],
    common: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    counters = summary["counters"]
    requests = common["requests"]
    completions = common["completions"]
    reads = [request for request in requests if request["action"] == "read"]
    writes = [request for request in requests if request["action"] == "write"]
    event_models_by_request: dict[str, set[str]] = defaultdict(set)
    for event in events:
        if event["model"] in {"hbm", "hbf"}:
            event_models_by_request[event["request_id"]].add(event["model"])
    hbm_accesses = sum(
        "hbm" in event_models_by_request[request["request_id"]]
        for request in requests
    )
    hbf_accesses = sum(
        "hbf" in event_models_by_request[request["request_id"]]
        for request in requests
    )
    latencies = [
        float(completions[index]["finish_ns"])
        - float(request["arrival_ns"])
        for index, request in enumerate(requests)
    ]
    metrics: dict[str, Any] = {
        "ops": len(requests),
        "reads": len(reads),
        "writes": len(writes),
        "logical_bytes": sum(int(request["bytes"]) for request in requests),
        "physical_bytes": sum(
            int(completion["physical_bytes"])
            for completion in completions
        ),
        "child_completions": sum(
            len(models) for models in event_models_by_request.values()),
        "hbm_user_accesses": hbm_accesses,
        "hbf_user_accesses": hbf_accesses,
        "finish_ns": common["finish_ns"],
        "first_offered_arrival_ns": min(
            (float(request["arrival_ns"]) for request in requests),
            default=0.0,
        ),
        "last_offered_arrival_ns": max(
            (float(request["arrival_ns"]) for request in requests),
            default=0.0,
        ),
        "service_latencies_ns": latencies,
    }
    for key in (
        "ops",
        "reads",
        "writes",
        "logical_bytes",
        "physical_bytes",
        "child_completions",
        "hbm_user_accesses",
        "hbf_user_accesses",
        "finish_ns",
        "user_finish_ns",
        "first_offered_arrival_ns",
        "last_offered_arrival_ns",
    ):
        expected = (
            metrics["finish_ns"] if key == "user_finish_ns"
            else metrics[key]
        )
        _expect_counter(counters, key, expected, source=source)
    for key in (
        "service_latencies_ns",
        "offered_latencies_ns",
        "source_latencies_ns",
    ):
        _expect_equal(
            _counter(counters, key, source=source),
            latencies,
            f"summary.counters.{key}",
            source=source,
        )

    hbm = _hbm_metrics(events, summary["config"]["hbm"])
    for key in (
        "read_bytes",
        "write_bytes",
        "row_hits",
        "row_misses",
        "row_conflicts",
        "activations",
        "precharges",
        "refresh_count",
        "bus_busy_ns",
        "finish_ns",
        "pseudo_channels",
        "active_pseudo_channels",
        "max_pseudo_channel_accesses",
    ):
        _expect_number(
            _counter(counters["hbm"], key, source=source),
            hbm[key],
            f"summary.counters.hbm.{key}",
            source=source,
        )

    hbf = _hbf_metrics(events, summary["config"]["hbf"])
    hbf_read_requests = hbf_accesses
    hbf_logical_read_bytes = round(
        hbf["hb_io_data_busy_ns"]
        * float(summary["config"]["hbf"]["hb_io_bandwidth_GBps"])
    )
    hbf_expected = {
        "read_requests": hbf_read_requests,
        "program_requests": 0,
        "logical_read_bytes": hbf_logical_read_bytes,
        "logical_write_bytes": 0,
        "physical_read_bytes": hbf["physical_read_bytes"],
        "physical_write_bytes": hbf["physical_write_bytes"],
        "page_reads": hbf["page_reads"],
        "page_programs": hbf["page_programs"],
        "mapping_lookup_ops": hbf["mapping_lookup_ops"],
        "mapping_user_lookup_ops": hbf["mapping_lookup_ops"],
        "mapping_gc_lookup_ops": 0,
        "finish_ns": hbf["finish_ns"],
        "total_pages": hbf["total_pages"],
        "accounting_verified": True,
        "active_planes": hbf["active_planes"],
        "active_channels": hbf["active_channels"],
        "active_dies": hbf["active_dies"],
    }
    for key, expected in hbf_expected.items():
        actual = _counter(counters["hbf"], key, source=source)
        path = f"summary.counters.hbf.{key}"
        if isinstance(expected, bool):
            _expect_equal(actual, expected, path, source=source)
        else:
            _expect_number(actual, expected, path, source=source)
    metrics["hbm"] = hbm
    metrics["hbf"] = hbf
    return metrics


def reduce_ledger(
    records: list[dict[str, Any]],
    *,
    source: str = "<memory>",
) -> dict[str, Any]:
    """Validate and return summary-independent metrics for one ledger."""

    if not records or records[-1].get("kind") != "summary":
        raise ContractError(f"{source}: reducer requires a final summary")
    summary = records[-1]
    common = _request_and_completion_audit(records, source=source)
    events = _events(records)
    model = summary["model"]
    if model == "hbm":
        model_metrics = _validate_hbm_summary(
            summary, common, events, source=source)
    elif model == "hbf":
        model_metrics = _validate_hbf_summary(
            records, summary, common, events, source=source)
    elif model == "hybrid":
        model_metrics = _validate_hybrid_summary(
            summary, common, events, source=source)
    elif model == "external":
        model_metrics = _validate_external_summary(
            summary, common, events, source=source)
    else:
        raise ContractError(
            f"{source}: ledger reducer does not support model {model!r}")
    return {
        "model": model,
        "request_count": len(common["requests"]),
        "completion_count": len(common["completions"]),
        "finish_ns": common["finish_ns"],
        "metrics": model_metrics,
    }
