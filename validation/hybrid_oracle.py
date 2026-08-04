"""Independent direct-composition oracle over the HBM and HBF oracles.

The hybrid layer deliberately owns only routing, page-boundary splitting, and
parent aggregation.  Tier timing and state come from the already independent
tiny HBM/HBF models; no production composition helper is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from validation.contracts import LEDGER_SCHEMA
from validation.hbf_oracle import Oracle as HbfOracle
from validation.hbm_oracle import Oracle as HbmOracle


@dataclass(frozen=True)
class Child:
    tier: str
    completion: dict[str, Any]


def _split_requests(
    case: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    page_size = case["config"]["hbf"]["page_size_bytes"]
    boundary = case["config"]["policy"]["read_boundary"]
    split: list[list[dict[str, Any]]] = []
    for request in case["requests"]:
        cursor = request["addr"]
        remaining = request["bytes"]
        children: list[dict[str, Any]] = []
        while remaining:
            segment_bytes = min(
                remaining,
                page_size - cursor % page_size,
            )
            tier = "hbm" if cursor < boundary else "hbf"
            children.append({
                "tier": tier,
                "id": request["id"],
                "arrival_ns": request["arrival_ns"],
                "op": request["op"],
                "address_space": "logical",
                "addr": cursor,
                "bytes": segment_bytes,
            })
            cursor += segment_bytes
            remaining -= segment_bytes
        split.append(children)
    return split


def _parent_resource(children: list[Child]) -> str:
    tiers = sorted({child.tier for child in children})
    return (
        f"hybrid/{'+'.join(tiers)}/"
        f"{len(children)}-child"
    )


def _parent_result(children: list[Child]) -> str:
    return (
        "split-tier-request"
        if len({child.tier for child in children}) > 1
        else "single-tier-request"
    )


def _hbm_counters(counters: dict[str, Any]) -> dict[str, Any]:
    keys = (
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
        "max_queue_occupancy",
    )
    return {key: counters[key] for key in keys}


def _hbf_counters(counters: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "read_requests",
        "program_requests",
        "logical_read_bytes",
        "logical_write_bytes",
        "physical_read_bytes",
        "physical_write_bytes",
        "page_reads",
        "page_programs",
        "mapping_lookup_ops",
        "mapping_user_lookup_ops",
        "mapping_gc_lookup_ops",
        "finish_ns",
        "total_pages",
        "free_pages",
        "valid_pages",
        "invalid_pages",
        "accounting_verified",
        "active_planes",
        "active_channels",
        "active_dies",
    )
    return {key: counters[key] for key in keys}


def build_ledger(case: dict[str, Any]) -> list[dict[str, Any]]:
    split_requests = _split_requests(case)
    prepopulate_lpns = sorted({
        child["addr"] // case["config"]["hbf"]["page_size_bytes"]
        for children in split_requests
        for child in children
        if child["tier"] == "hbf"
    })
    hbm = HbmOracle({
        "config": case["config"]["hbm"],
        "initial_state": {},
    })
    hbf = HbfOracle({
        "config": case["config"]["hbf"],
        "initial_state": {
            "prepopulate_lpns": prepopulate_lpns,
        },
    })
    records: list[dict[str, Any]] = []

    def append(
        kind: str,
        record_id: str,
        parent_id: str | None,
        **fields: Any,
    ) -> None:
        records.append({
            "schema": LEDGER_SCHEMA,
            "record_index": len(records),
            "kind": kind,
            "id": record_id,
            "parent_id": parent_id,
            "case_id": case["case_id"],
            **fields,
        })

    append(
        "header",
        "ledger",
        None,
        model="hybrid",
        producer="oracle",
    )
    hbm_accesses = 0
    hbf_accesses = 0
    physical_bytes = 0
    service_latencies: list[float] = []
    finish_ns = 0.0

    for request, child_specs in zip(
        case["requests"],
        split_requests,
        strict=True,
    ):
        request_record_id = f"request/{request['id']}"
        append(
            "request",
            request_record_id,
            None,
            request_id=request["id"],
            model="hybrid",
            action=request["op"],
            arrival_ns=request["arrival_ns"],
            address_space=request["address_space"],
            addr=request["addr"],
            bytes=request["bytes"],
        )
        children: list[Child] = []
        for child_spec in child_specs:
            tier = child_spec["tier"]
            completion = (
                hbm.issue(child_spec)
                if tier == "hbm"
                else hbf.issue(child_spec)
            )
            children.append(Child(tier, completion))
            if tier == "hbm":
                hbm_accesses += 1
            else:
                hbf_accesses += 1

        spans = [
            (child.tier, span)
            for child in children
            for span in child.completion["spans"]
        ]
        spans.sort(key=lambda item: (
            item[1].start_ns,
            item[1].finish_ns,
            item[1].category,
            item[1].name,
            item[1].entity,
            item[0],
        ))
        for index, (tier, span) in enumerate(spans):
            append(
                "event",
                f"{request_record_id}/event/{index:04d}",
                request_record_id,
                request_id=request["id"],
                model=tier,
                resource=span.entity,
                category=span.category,
                action=span.name,
                start_ns=span.start_ns,
                finish_ns=span.finish_ns,
                critical=span.critical,
            )

        parent_start = min(
            child.completion["start_ns"] for child in children)
        parent_finish = max(
            child.completion["finish_ns"] for child in children)
        parent_physical_bytes = sum(
            child.completion["physical_bytes"] for child in children)
        physical_bytes += parent_physical_bytes
        service_latencies.append(
            parent_finish - request["arrival_ns"])
        finish_ns = max(finish_ns, parent_finish)
        append(
            "completion",
            f"{request_record_id}/completion",
            request_record_id,
            request_id=request["id"],
            model="hybrid",
            action="complete",
            arrival_ns=request["arrival_ns"],
            start_ns=parent_start,
            finish_ns=parent_finish,
            logical_bytes=request["bytes"],
            physical_bytes=parent_physical_bytes,
            resource=_parent_resource(children),
            result=_parent_result(children),
        )

    hbm.finalize_counters()
    hbf.finalize()
    append(
        "summary",
        "summary",
        None,
        model="hybrid",
        action="final",
        config={
            "hbm": hbm.derived_config(),
            "hbf": hbf.derived_config(),
            "policy": case["config"]["policy"],
            "knobs": case["config"]["knobs"],
        },
        address_observations=[],
        counters={
            "ops": len(case["requests"]),
            "reads": len(case["requests"]),
            "writes": 0,
            "logical_bytes": sum(
                request["bytes"] for request in case["requests"]),
            "physical_bytes": physical_bytes,
            "child_completions": hbm_accesses + hbf_accesses,
            "hbm_user_accesses": hbm_accesses,
            "hbf_user_accesses": hbf_accesses,
            "hbf_direct_user_ops": 0,
            "hbf_static_read_bytes": 0,
            "finish_ns": finish_ns,
            "user_finish_ns": finish_ns,
            "first_offered_arrival_ns": min(
                request["arrival_ns"] for request in case["requests"]),
            "last_offered_arrival_ns": max(
                request["arrival_ns"] for request in case["requests"]),
            "service_latencies_ns": service_latencies,
            "offered_latencies_ns": service_latencies,
            "source_latencies_ns": service_latencies,
            "front_end_admission_waited_ops": 0,
            "front_end_admission_wait_work_ns": 0.0,
            "front_end_admission_max_wait_ns": 0.0,
            "phase_barriers": 0,
            "phase_dependency_waited_ops": 0,
            "phase_dependency_wait_work_ns": 0.0,
            "phase_dependency_max_wait_ns": 0.0,
            "warnings": 0,
            "hbm": _hbm_counters(hbm.counters),
            "hbf": _hbf_counters(hbf.counters),
        },
    )
    return records
