"""Small synchronous reference for channel-aggregate-v2.

Each burst is decoded independently to its pseudo-channel lane. Consecutive
lanes (``service_group_channels`` of them) share one service group whose data
calendar is an explicit interval list. A request's demand on a group is its
busiest lane's burst count, reserved in at most one service quantum at a time:
an isolated transfer keeps its lane bandwidth while disjoint concurrent lanes
inside a group serialize. Lane byte and access counters stay per pseudo
channel. This checks production grouping and causal coalescing; it is a model
cross-check, not independent hardware calibration. Native tests cover online
admission, finite queues and HBF DMA arrival boundaries.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from verification.core.contracts import LEDGER_SCHEMA

MASK64 = (1 << 64) - 1
DEFAULT_INTERLEAVE_BYTES = 256
MAXIMUM_CYCLE = 2.0 ** 48


def _pseudo_channel_hash(stripe: int) -> int:
    return ((stripe * 0x9E3779B97F4A7C15) & MASK64) >> 32


def _bank_group_hash(bank_row: int) -> int:
    return ((bank_row * 0xD1B54A32D192ED03) & MASK64) >> 32


def _lower_bound(values: list[int], target: int) -> int:
    """First index whose value is >= target (values sorted ascending)."""
    index = 0
    while index < len(values) and values[index] < target:
        index += 1
    return index


def command_clock_cycles(time_ns: float, tck_ns: float) -> int:
    """Round causally to the command grid with bounded edge-roundoff tolerance."""
    if not math.isfinite(time_ns) or time_ns < 0.0:
        raise ValueError("time must be finite and non-negative")
    cycles = time_ns / tck_ns
    if not cycles < MAXIMUM_CYCLE:
        raise ValueError("time exceeds the representable clock horizon")

    def covered_by_edge(edge_ns: float) -> bool:
        if edge_ns >= time_ns:
            return True
        tolerance = min(
            tck_ns * 1e-6,
            16.0 * max(
                math.nextafter(time_ns, math.inf) - time_ns,
                math.nextafter(edge_ns, math.inf) - edge_ns,
                math.ulp(1.0) * max(1.0, abs(time_ns)),
            ),
        )
        return time_ns - edge_ns <= tolerance

    eligible = math.floor(cycles)
    while not covered_by_edge(eligible * tck_ns):
        eligible += 1
    while eligible and covered_by_edge((eligible - 1) * tck_ns):
        eligible -= 1
    if eligible >= MAXIMUM_CYCLE:
        raise ValueError("time exceeds the representable clock horizon")
    return eligible


def effective_interleave_bytes(config: dict[str, Any]) -> int:
    row = config["channel_row_size_bytes"] // config["pseudo_channels_per_channel"]
    burst = (
        config["channel_width_bits"] // config["pseudo_channels_per_channel"] // 8
        * config["burst_length"]
    )
    explicit = int(config.get("interleave_bytes", 0))
    if explicit:
        if explicit % burst or explicit > row or row % explicit:
            raise ValueError("interleave_bytes is not a legal interleave")
        return explicit
    bursts_per_row = row // burst
    best = burst
    for bursts in range(1, bursts_per_row + 1):
        if bursts_per_row % bursts:
            continue
        if bursts > DEFAULT_INTERLEAVE_BYTES // burst:
            break
        best = bursts * burst
    return best


@dataclass
class Address:
    stack: int
    channel: int
    pseudo_channel: int
    bank_group: int
    bank: int
    row: int
    offset: int

    def path(self) -> str:
        return (
            f"stack{self.stack}/ch{self.channel}/pch{self.pseudo_channel}"
            f"/bg{self.bank_group}/bank{self.bank}/row{self.row}"
            f"/off{self.offset}"
        )

    def bus_entity(self) -> str:
        return f"stack{self.stack}/ch{self.channel}/pch{self.pseudo_channel}"


@dataclass
class Span:
    name: str
    category: str
    entity: str
    start_ns: float
    finish_ns: float
    critical: bool = False
    physical_bytes: int = 0

class Bus:
    def __init__(self):
        self.intervals = []
    def reserve(self, earliest_ns, duration_ns):
        start = earliest_ns
        for interval in self.intervals:
            if start + duration_ns <= interval.start_ns:
                break
            start = max(start, interval.finish_ns)
        result = SimpleNamespace(start_ns=start, finish_ns=start+duration_ns)
        self.intervals.append(result)
        self.intervals.sort(key=lambda x: x.start_ns)
        return result

class Oracle:
    def __init__(self, case):
        self.case = case
        self.config = dict(case["config"])
        config = self.config
        self.tck = (
            config["data_rate_per_command_clock"] / config["pin_rate_Gbps"])
        self.pseudo_channel_width_bits = (
            config["channel_width_bits"] // config["pseudo_channels_per_channel"])
        self.burst_bytes = (
            self.pseudo_channel_width_bits // 8 * config["burst_length"])
        self.row_size_bytes = (
            config["channel_row_size_bytes"] // config["pseudo_channels_per_channel"])
        self.interleave_bytes = effective_interleave_bytes(config)
        self.bursts_per_unit = self.interleave_bytes // self.burst_bytes
        self.units_per_row = self.row_size_bytes // self.interleave_bytes
        self.pseudo_channels_per_stack = (
            config["channels_per_stack"] * config["pseudo_channels_per_channel"])
        self.total_pseudo_channels = (
            config["stacks"] * self.pseudo_channels_per_stack)
        self.stripe_bytes = self.total_pseudo_channels * self.interleave_bytes
        self.bank_groups = config["bank_groups_per_pseudo_channel"]
        self.banks_per_group = config["banks_per_group"]
        self.banks_per_pseudo_channel = self.bank_groups * self.banks_per_group
        self.burst_cycles = (
            config["burst_length"] // config["data_rate_per_command_clock"])
        self.pseudo_channels = [SimpleNamespace(accesses=0, bus_busy_cycles=0)
                                for _ in range(self.total_pseudo_channels)]
        # Consecutive lanes arbitrate as one service group; the last group may
        # be narrower. A composed oracle may replace a group's data_bus with
        # its own interval list so DMA and application traffic share it.
        self.service_group_width = min(
            config["service_group_channels"], self.total_pseudo_channels)
        self.total_service_groups = -(
            -self.total_pseudo_channels // self.service_group_width)
        self.service_groups = [SimpleNamespace(floor=0, last_op=None, data_bus=Bus())
                               for _ in range(self.total_service_groups)]
        self.total_bus_busy_cycles = 0
        self.counters = dict(read_bytes=0, write_bytes=0, bus_busy_ns=0.0, finish_ns=0.0,
            pseudo_channels=self.total_pseudo_channels, active_pseudo_channels=0,
            max_pseudo_channel_accesses=0, max_queue_occupancy=0)

    def _cycles(self, value):
        return command_clock_cycles(value, self.tck)
    def ns(self, cycles):
        return cycles * self.tck
    def service_cycles(self, bursts):
        return math.ceil(bursts * self.burst_cycles / self.config["bandwidth_efficiency"])

    def service_group(self, lane: int) -> int:
        return lane // self.service_group_width

    def lanes_in_group(self, group: int) -> int:
        return min(self.service_group_width,
                   self.total_pseudo_channels - group * self.service_group_width)

    def group_demand(self, addr: int, byte_count: int) -> dict[int, dict[int, int]]:
        """Burst counts per lane, keyed by service group, for one transfer."""
        groups: dict[int, dict[int, int]] = {}
        first = addr - addr % self.burst_bytes
        for burst_addr in range(first, addr + byte_count, self.burst_bytes):
            lane = self.pc_index(self.decode(burst_addr))
            lanes = groups.setdefault(self.service_group(lane), {})
            lanes[lane] = lanes.get(lane, 0) + 1
        return groups

    def account_lanes(self, lanes: dict[int, int]) -> None:
        for lane, count in lanes.items():
            pc = self.pseudo_channels[lane]
            pc.accesses += count
            pc.bus_busy_cycles += count * self.burst_cycles
            self.total_bus_busy_cycles += count * self.burst_cycles

    def derived_config(self):
        config = dict(self.config)
        config["interleave_bytes"] = self.interleave_bytes
        config["timing_model"] = "channel-aggregate-v2"
        config["derived"] = dict(pseudo_channel_width_bits=self.pseudo_channel_width_bits,
            row_size_bytes=self.row_size_bytes, burst_bytes=self.burst_bytes,
            channel_bandwidth_GBps=config["pin_rate_Gbps"]*config["channel_width_bits"]/8,
            pseudo_channel_bandwidth_GBps=config["pin_rate_Gbps"]*self.pseudo_channel_width_bits/8,
            command_clock_period_ns=self.tck, burst_duration_ns=config["burst_length"]/config["pin_rate_Gbps"],
            stripe_bytes=self.stripe_bytes)
        return config

    def _local(self, stripe: int) -> tuple[int, int, int, int]:
        """(bank_group, bank, row, column_unit) of a stripe index."""
        lane_bg = stripe % self.bank_groups
        g1 = stripe // self.bank_groups
        column_unit = g1 % self.units_per_row
        g2 = g1 // self.units_per_row
        bank = g2 % self.banks_per_group
        row = g2 // self.banks_per_group
        bank_group = (lane_bg + _bank_group_hash(g2) % self.bank_groups) % self.bank_groups
        return bank_group, bank, row, column_unit

    def _assign_pseudo_channel(self, linear: int) -> tuple[int, int, int]:
        stack = linear // self.pseudo_channels_per_stack
        within = linear % self.pseudo_channels_per_stack
        return (
            stack,
            within // self.config["pseudo_channels_per_channel"],
            within % self.config["pseudo_channels_per_channel"],
        )

    def decode(self, addr: int) -> Address:
        if addr >= self.config["capacity_bytes"]:
            raise ValueError("address out of capacity")
        global_unit, unit_offset = divmod(addr, self.interleave_bytes)
        stripe, lane = divmod(global_unit, self.total_pseudo_channels)
        linear = (
            lane + _pseudo_channel_hash(stripe) % self.total_pseudo_channels
        ) % self.total_pseudo_channels
        bank_group, bank, row, column_unit = self._local(stripe)
        stack, channel, pseudo_channel = self._assign_pseudo_channel(linear)
        return Address(
            stack=stack,
            channel=channel,
            pseudo_channel=pseudo_channel,
            bank_group=bank_group,
            bank=bank,
            row=row,
            offset=column_unit * self.interleave_bytes + unit_offset,
        )

    def encode(self, address: Address) -> int:
        g2 = address.row * self.banks_per_group + address.bank
        lane_bg = (
            address.bank_group - _bank_group_hash(g2) % self.bank_groups
        ) % self.bank_groups
        column_unit, unit_offset = divmod(address.offset, self.interleave_bytes)
        g1 = g2 * self.units_per_row + column_unit
        stripe = g1 * self.bank_groups + lane_bg
        linear = self.pc_index(address)
        lane = (
            linear - _pseudo_channel_hash(stripe) % self.total_pseudo_channels
        ) % self.total_pseudo_channels
        global_unit = stripe * self.total_pseudo_channels + lane
        return global_unit * self.interleave_bytes + unit_offset

    def pc_index(self, address: Address) -> int:
        return (
            (address.stack * self.config["channels_per_stack"] + address.channel)
            * self.config["pseudo_channels_per_channel"]
            + address.pseudo_channel
        )

    def issue(self, request):
        config = self.config
        arrival = request["arrival_ns"]
        mapped = arrival + config["address_mapping_ns"]
        ready = self._cycles(mapped) + self._cycles(config["read_latency_ns"] if request["op"] == "read" else config["write_latency_ns"])
        latency = max(mapped, self.ns(ready)) - mapped
        groups = self.group_demand(request["addr"], request["bytes"])
        start_ns, finish_ns = math.inf, arrival
        spans = []
        quantum = config["service_quantum_bytes"] // self.burst_bytes
        physical = 0
        for index, lanes in sorted(groups.items()):
            group = self.service_groups[index]
            demand = max(lanes.values())
            earliest = max(ready, group.floor)
            if group.last_op is not None and group.last_op != request["op"]:
                turn = config["write_to_read_ns"] if request["op"] == "read" else config["read_to_write_ns"]
                earliest = max(earliest, group.floor + self._cycles(turn))
            first_start = None
            served = 0
            local_spans = []
            while served < demand:
                take = min(demand - served, quantum)
                slot = group.data_bus.reserve(earliest, self.service_cycles(take))
                begin, end = self.ns(slot.start_ns), self.ns(slot.finish_ns)
                if first_start is None: first_start = begin
                # Every lane of the group advances by its own remaining bursts
                # within this quantum; the span carries the group's bytes.
                quantum_bytes = sum(min(max(count - served, 0), take) for count in lanes.values()) * self.burst_bytes
                if local_spans and local_spans[-1].finish_ns == begin:
                    local_spans[-1].finish_ns = end
                    local_spans[-1].physical_bytes += quantum_bytes
                else:
                    local_spans.append(Span("hbm_channel_"+request["op"], "hbm_channel_service", f"hbm/group{index}",
                                            begin, end, True, quantum_bytes))
                earliest = group.floor = slot.finish_ns
                group.last_op = request["op"]
                served += take
            spans.extend(local_spans)
            start_ns = min(start_ns, max(mapped, first_start-latency))
            finish_ns = max(finish_ns, self.ns(group.floor))
            self.account_lanes(lanes)
            physical += sum(lanes.values()) * self.burst_bytes
        self.counters[request["op"]+"_bytes"] += physical
        self.counters["bus_busy_ns"] = self.ns(self.total_bus_busy_cycles)
        self.counters["finish_ns"] = max(self.counters["finish_ns"], finish_ns)
        self.counters["max_queue_occupancy"] = 1
        spans.sort(key=lambda s: (s.start_ns,s.finish_ns,s.category,s.name,s.entity))
        return dict(start_ns=start_ns,finish_ns=finish_ns,logical_bytes=request["bytes"],physical_bytes=physical,
                    resource="hbm/channel-aggregate", result="channel-aggregate-v2", spans=spans)

    def finalize_counters(self):
        self.counters["active_pseudo_channels"] = sum(pc.accesses>0 for pc in self.pseudo_channels)
        self.counters["max_pseudo_channel_accesses"] = max((pc.accesses for pc in self.pseudo_channels), default=0)

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
        model="hbm",
        producer="oracle",
    )
    for request in case["requests"]:
        request_record_id = f"request/{request['id']}"
        append(
            "request",
            request_record_id,
            None,
            request_id=request["id"],
            model="hbm",
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
                model="hbm",
                resource=span.entity,
                category=span.category,
                action=span.name,
                start_ns=span.start_ns,
                finish_ns=span.finish_ns,
                critical=span.critical,
                **({"physical_bytes": span.physical_bytes} if span.physical_bytes else {}),
            )
        append(
            "completion",
            f"{request_record_id}/completion",
            request_record_id,
            request_id=request["id"],
            model="hbm",
            action="complete",
            arrival_ns=request["arrival_ns"],
            start_ns=completion["start_ns"],
            finish_ns=completion["finish_ns"],
            logical_bytes=completion["logical_bytes"],
            physical_bytes=completion["physical_bytes"],
            resource=completion["resource"],
            result=completion["result"],
        )
    oracle.finalize_counters()
    observations = []
    for address in case["inspect_addresses"]:
        decoded = oracle.decode(address)
        observations.append({
            "address": address,
            "stack": decoded.stack,
            "channel": decoded.channel,
            "pseudo_channel": decoded.pseudo_channel,
            "bank_group": decoded.bank_group,
            "bank": decoded.bank,
            "row": decoded.row,
            "offset": decoded.offset,
            "round_trip_address": oracle.encode(decoded),
        })
    append(
        "summary",
        "summary",
        None,
        model="hbm",
        action="final",
        config=oracle.derived_config(),
        address_observations=observations,
        counters=oracle.counters,
    )
    return records
