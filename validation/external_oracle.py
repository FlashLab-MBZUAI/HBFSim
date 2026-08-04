"""Independent oracle for the external-backing protocol pipeline.

The implementation intentionally uses linear interval scans, Python heap
admission, and locally derived accounting.  It imports no production
scheduler, timeline, profile, or statistics helper.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

from validation.contracts import LEDGER_SCHEMA


@dataclass(frozen=True)
class Reservation:
    start_ns: float
    finish_ns: float

    @property
    def busy_ns(self) -> float:
        return self.finish_ns - self.start_ns


class Timeline:
    """Tiny exact earliest-gap serial calendar implemented by a full scan."""

    def __init__(self) -> None:
        self.intervals: list[Reservation] = []

    def reserve(self, ready_ns: float, busy_ns: float) -> Reservation:
        candidate = ready_ns
        for interval in self.intervals:
            if candidate + busy_ns <= interval.start_ns:
                break
            if candidate < interval.finish_ns:
                candidate = interval.finish_ns
        result = Reservation(candidate, candidate + busy_ns)
        position = 0
        while (
            position < len(self.intervals)
            and self.intervals[position].start_ns < result.start_ns
        ):
            position += 1
        self.intervals.insert(position, result)
        return result


def _span(
    spans: list[dict[str, Any]],
    *,
    name: str,
    category: str,
    entity: str,
    start_ns: float,
    finish_ns: float,
) -> None:
    if finish_ns <= start_ns:
        return
    spans.append({
        "name": name,
        "category": category,
        "entity": entity,
        "start_ns": start_ns,
        "finish_ns": finish_ns,
        "critical": True,
    })


class Oracle:
    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case
        self.config = case["config"]
        self.m2s = Timeline()
        self.s2m = Timeline()
        self.controller = Timeline()
        self.media = [
            Timeline() for _ in range(self.config["media_channels"])
        ]
        self.inflight: list[float] = []
        self.active_media_channels: set[int] = set()
        self.counters: dict[str, int | float] = {
            "read_requests": 0,
            "write_requests": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "media_channels": self.config["media_channels"],
            "active_media_channels": 0,
            "max_device_outstanding": 0,
            "outstanding_wait_ns": 0.0,
            "controller_queue_wait_ns": 0.0,
            "controller_issue_busy_ns": 0.0,
            "controller_processing_work_ns": 0.0,
            "media_queue_wait_ns": 0.0,
            "media_read_latency_work_ns": 0.0,
            "media_write_latency_work_ns": 0.0,
            "media_read_busy_ns": 0.0,
            "media_write_busy_ns": 0.0,
            "m2s_payload_bytes": 0,
            "m2s_protocol_bytes": 0,
            "m2s_wire_bytes": 0,
            "s2m_payload_bytes": 0,
            "s2m_protocol_bytes": 0,
            "s2m_wire_bytes": 0,
            "m2s_queue_wait_ns": 0.0,
            "s2m_queue_wait_ns": 0.0,
            "m2s_busy_ns": 0.0,
            "s2m_busy_ns": 0.0,
            "transport_propagation_work_ns": 0.0,
            "finish_ns": 0.0,
        }

    def _admit(self, arrival_ns: float) -> float:
        while self.inflight and self.inflight[0] <= arrival_ns:
            heapq.heappop(self.inflight)
        admitted_ns = arrival_ns
        while (
            len(self.inflight)
            >= self.config["max_outstanding_requests"]
        ):
            admitted_ns = self.inflight[0]
            while self.inflight and self.inflight[0] <= admitted_ns:
                heapq.heappop(self.inflight)
        return admitted_ns

    def issue(self, request: dict[str, Any]) -> dict[str, Any]:
        read = request["op"] == "read"
        admitted_ns = self._admit(request["arrival_ns"])
        m2s_payload = 0 if read else request["bytes"]
        m2s_protocol = self.config["command_bytes"]
        m2s_wire = m2s_payload + m2s_protocol
        s2m_payload = request["bytes"] if read else 0
        s2m_protocol = self.config["completion_bytes"]
        s2m_wire = s2m_payload + s2m_protocol

        m2s = self.m2s.reserve(
            admitted_ns,
            m2s_wire / self.config["m2s_bandwidth_GBps"],
        )
        m2s_arrival_ns = (
            m2s.finish_ns + self.config["one_way_propagation_ns"]
        )
        controller = self.controller.reserve(
            m2s_arrival_ns,
            self.config["controller_issue_ns"],
        )
        controller_finish_ns = (
            controller.finish_ns
            + self.config["controller_processing_ns"]
        )
        media_latency = self.config[
            "media_read_latency_ns" if read
            else "media_write_latency_ns"
        ]
        media_ready_ns = controller_finish_ns + media_latency
        channel = (
            request["addr"] // self.config["page_size_bytes"]
        ) % self.config["media_channels"]
        aggregate_media_bandwidth = self.config[
            "media_read_bandwidth_GBps" if read
            else "media_write_bandwidth_GBps"
        ]
        media = self.media[channel].reserve(
            media_ready_ns,
            request["bytes"]
            / (
                aggregate_media_bandwidth
                / self.config["media_channels"]
            ),
        )
        s2m = self.s2m.reserve(
            media.finish_ns,
            s2m_wire / self.config["s2m_bandwidth_GBps"],
        )
        finish_ns = (
            s2m.finish_ns + self.config["one_way_propagation_ns"]
        )
        heapq.heappush(self.inflight, finish_ns)

        count_key = "read_requests" if read else "write_requests"
        byte_key = "read_bytes" if read else "write_bytes"
        self.counters[count_key] += 1
        self.counters[byte_key] += request["bytes"]
        self.active_media_channels.add(channel)
        self.counters["active_media_channels"] = len(
            self.active_media_channels
        )
        self.counters["max_device_outstanding"] = max(
            self.counters["max_device_outstanding"],
            len(self.inflight),
        )
        self.counters["outstanding_wait_ns"] += (
            admitted_ns - request["arrival_ns"]
        )
        self.counters["m2s_queue_wait_ns"] += (
            m2s.start_ns - admitted_ns
        )
        self.counters["controller_queue_wait_ns"] += (
            controller.start_ns - m2s_arrival_ns
        )
        self.counters["media_queue_wait_ns"] += (
            media.start_ns - media_ready_ns
        )
        self.counters["s2m_queue_wait_ns"] += (
            s2m.start_ns - media.finish_ns
        )
        self.counters["controller_issue_busy_ns"] += controller.busy_ns
        self.counters["controller_processing_work_ns"] += self.config[
            "controller_processing_ns"
        ]
        latency_key = (
            "media_read_latency_work_ns" if read
            else "media_write_latency_work_ns"
        )
        busy_key = (
            "media_read_busy_ns" if read
            else "media_write_busy_ns"
        )
        self.counters[latency_key] += media_latency
        self.counters[busy_key] += media.busy_ns
        for prefix, payload, protocol, wire, reservation in (
            ("m2s", m2s_payload, m2s_protocol, m2s_wire, m2s),
            ("s2m", s2m_payload, s2m_protocol, s2m_wire, s2m),
        ):
            self.counters[f"{prefix}_payload_bytes"] += payload
            self.counters[f"{prefix}_protocol_bytes"] += protocol
            self.counters[f"{prefix}_wire_bytes"] += wire
            self.counters[f"{prefix}_busy_ns"] += reservation.busy_ns
        self.counters["transport_propagation_work_ns"] += (
            2 * self.config["one_way_propagation_ns"]
        )
        self.counters["finish_ns"] = max(
            self.counters["finish_ns"], finish_ns
        )

        prefix = "external_read" if read else "external_write"
        resource = (
            f"external/{self.config['kind']}/media-channel{channel}"
        )
        spans: list[dict[str, Any]] = []
        _span(
            spans,
            name=f"{prefix}_m2s_transfer",
            category="external_link",
            entity="external/m2s",
            start_ns=m2s.start_ns,
            finish_ns=m2s.finish_ns,
        )
        _span(
            spans,
            name=f"{prefix}_m2s_propagation",
            category="external_link",
            entity="external/m2s",
            start_ns=m2s.finish_ns,
            finish_ns=m2s_arrival_ns,
        )
        _span(
            spans,
            name=f"{prefix}_controller_issue",
            category="external_controller",
            entity="external/controller",
            start_ns=controller.start_ns,
            finish_ns=controller.finish_ns,
        )
        _span(
            spans,
            name=f"{prefix}_controller_processing",
            category="external_controller",
            entity="external/controller",
            start_ns=controller.finish_ns,
            finish_ns=controller_finish_ns,
        )
        _span(
            spans,
            name=f"{prefix}_media_latency",
            category="external_backing",
            entity=resource,
            start_ns=controller_finish_ns,
            finish_ns=media_ready_ns,
        )
        _span(
            spans,
            name=f"{prefix}_media_transfer",
            category="external_backing",
            entity=resource,
            start_ns=media.start_ns,
            finish_ns=media.finish_ns,
        )
        _span(
            spans,
            name=f"{prefix}_s2m_transfer",
            category="external_link",
            entity="external/s2m",
            start_ns=s2m.start_ns,
            finish_ns=s2m.finish_ns,
        )
        _span(
            spans,
            name=f"{prefix}_s2m_propagation",
            category="external_link",
            entity="external/s2m",
            start_ns=s2m.finish_ns,
            finish_ns=finish_ns,
        )
        spans.sort(key=lambda span: (
            span["start_ns"],
            span["finish_ns"],
            span["category"],
            span["name"],
            span["entity"],
        ))
        return {
            "start_ns": admitted_ns,
            "finish_ns": finish_ns,
            "logical_bytes": request["bytes"],
            "physical_bytes": request["bytes"],
            "resource": resource,
            "result": (
                "external-backing-to-hbm"
                if read
                else "hbm-to-external-backing"
            ),
            "spans": spans,
        }


def build_ledger(case: dict[str, Any]) -> list[dict[str, Any]]:
    oracle = Oracle(case)
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
        model="external",
        producer="oracle",
    )
    for request in case["requests"]:
        request_record_id = f"request/{request['id']}"
        append(
            "request",
            request_record_id,
            None,
            request_id=request["id"],
            model="external",
            action=request["op"],
            arrival_ns=request["arrival_ns"],
            address_space=request["address_space"],
            addr=request["addr"],
            bytes=request["bytes"],
        )
        completion = oracle.issue(request)
        for index, span in enumerate(completion["spans"]):
            append(
                "event",
                f"{request_record_id}/event/{index:04d}",
                request_record_id,
                request_id=request["id"],
                model="external",
                resource=span["entity"],
                category=span["category"],
                action=span["name"],
                start_ns=span["start_ns"],
                finish_ns=span["finish_ns"],
                critical=span["critical"],
            )
        append(
            "completion",
            f"{request_record_id}/completion",
            request_record_id,
            request_id=request["id"],
            model="external",
            action="complete",
            arrival_ns=request["arrival_ns"],
            start_ns=completion["start_ns"],
            finish_ns=completion["finish_ns"],
            logical_bytes=completion["logical_bytes"],
            physical_bytes=completion["physical_bytes"],
            resource=completion["resource"],
            result=completion["result"],
        )
    observations = [
        {
            "address": address,
            "media_channel": (
                address // case["config"]["page_size_bytes"]
            ) % case["config"]["media_channels"],
        }
        for address in case["inspect_addresses"]
    ]
    append(
        "summary",
        "summary",
        None,
        model="external",
        action="final",
        config=dict(case["config"]),
        address_observations=observations,
        counters=oracle.counters,
    )
    return records
