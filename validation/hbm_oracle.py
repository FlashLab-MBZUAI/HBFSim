"""Independent, deliberately slow HBM oracle for tiny validation cases.

The implementation favors explicit state and scans over reuse or performance.
It does not import, execute, or translate any HBFSim production-model helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from validation.contracts import LEDGER_SCHEMA


MASK64 = (1 << 64) - 1
ACT, PRE, PREA, RD, WR, RDA, WRA, REFAB, REFSB = range(9)


def _align(time_ns: float, tck_ns: float) -> float:
    cycles = time_ns / tck_ns
    aligned_cycles = math.floor(cycles)
    aligned_ns = aligned_cycles * tck_ns
    if aligned_ns >= time_ns:
        return aligned_ns
    while aligned_ns < time_ns:
        next_cycles = aligned_cycles + 1.0
        if next_cycles == aligned_cycles:
            raise ValueError("time exceeds exact cycle indexing")
        aligned_cycles = next_cycles
        aligned_ns = aligned_cycles * tck_ns
    return aligned_ns


def _pseudo_hash(group: int) -> int:
    return ((group * 0x9E3779B97F4A7C15) & MASK64) >> 32


def _bank_hash(row_column: int) -> int:
    return ((row_column * 0xD1B54A32D192ED03) & MASK64) >> 32


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


@dataclass
class Bank:
    open_row: int | None = None
    ready: list[float] = field(default_factory=lambda: [0.0] * 9)


@dataclass
class BankGroup:
    ready: list[float] = field(default_factory=lambda: [0.0] * 9)


@dataclass
class PseudoChannel:
    banks: list[Bank]
    groups: list[BankGroup]
    ready: list[float] = field(default_factory=lambda: [0.0] * 9)
    bus_ready_ns: float = 0.0
    bus_busy_ns: float = 0.0
    accesses: int = 0
    recent_activations: list[float] = field(default_factory=list)


@dataclass
class Span:
    name: str
    category: str
    entity: str
    start_ns: float
    finish_ns: float
    critical: bool = True


@dataclass
class ChildCompletion:
    start_ns: float
    finish_ns: float
    physical_bytes: int
    resource: str
    result: str
    spans: list[Span]


@dataclass
class QueuedChild:
    addr: int
    bypass_count: int = 0


class Oracle:
    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case
        self.config = case["config"]
        self.tck = (
            self.config["data_rate_per_command_clock"]
            / self.config["pin_rate_Gbps"]
        )
        self.burst_bytes = (
            (
                self.config["channel_width_bits"]
                // self.config["pseudo_channels_per_channel"]
            )
            // 8
            * self.config["burst_length"]
        )
        self.row_size_bytes = (
            self.config["channel_row_size_bytes"]
            // self.config["pseudo_channels_per_channel"]
        )
        pc_count = (
            self.config["stacks"]
            * self.config["channels_per_stack"]
            * self.config["pseudo_channels_per_channel"]
        )
        banks_per_pc = (
            self.config["bank_groups_per_pseudo_channel"]
            * self.config["banks_per_group"]
        )
        self.pseudo_channels = [
            PseudoChannel(
                banks=[Bank() for _ in range(banks_per_pc)],
                groups=[
                    BankGroup()
                    for _ in range(
                        self.config["bank_groups_per_pseudo_channel"])
                ],
            )
            for _ in range(pc_count)
        ]
        self.counters: dict[str, int | float] = {
            "read_bytes": 0,
            "write_bytes": 0,
            "row_hits": 0,
            "row_misses": 0,
            "row_conflicts": 0,
            "activations": 0,
            "precharges": 0,
            "refresh_count": 0,
            "bus_busy_ns": 0.0,
            "finish_ns": 0.0,
            "pseudo_channels": pc_count,
            "active_pseudo_channels": 0,
            "max_pseudo_channel_accesses": 0,
            "max_queue_occupancy": 0,
        }
        if self.config["refresh_enabled"]:
            raise ValueError(
                "independent oracle v1 intentionally excludes refresh cases")

    def derived_config(self) -> dict[str, Any]:
        config = dict(self.config)
        config["derived"] = {
            "pseudo_channel_width_bits": (
                config["channel_width_bits"]
                // config["pseudo_channels_per_channel"]
            ),
            "row_size_bytes": self.row_size_bytes,
            "burst_bytes": self.burst_bytes,
            "channel_bandwidth_GBps": (
                config["pin_rate_Gbps"] * config["channel_width_bits"] / 8.0
            ),
            "pseudo_channel_bandwidth_GBps": (
                config["pin_rate_Gbps"]
                * (
                    config["channel_width_bits"]
                    // config["pseudo_channels_per_channel"]
                )
                / 8.0
            ),
            "command_clock_period_ns": self.tck,
            "burst_duration_ns": (
                config["burst_length"] / config["pin_rate_Gbps"]
            ),
            "tCCD_S_ns": config["tCCD_S_cycles"] * self.tck,
            "tCCD_L_ns": config["tCCD_L_cycles"] * self.tck,
        }
        return config

    def decode(self, addr: int) -> Address:
        burst_linear, burst_offset = divmod(addr, self.burst_bytes)
        total_pc = len(self.pseudo_channels)
        group, lane = divmod(burst_linear, total_pc)
        pseudo_linear = (lane + _pseudo_hash(group) % total_pc) % total_pc
        banks_per_pc = (
            self.config["bank_groups_per_pseudo_channel"]
            * self.config["banks_per_group"]
        )
        after_bank, bank_lane = divmod(group, banks_per_pc)
        bank_interleave = (
            bank_lane + _bank_hash(after_bank) % banks_per_pc
        ) % banks_per_pc
        row_bursts = self.row_size_bytes // self.burst_bytes
        column = after_bank % row_bursts
        pc_per_stack = (
            self.config["channels_per_stack"]
            * self.config["pseudo_channels_per_channel"]
        )
        stack, within_stack = divmod(pseudo_linear, pc_per_stack)
        channel, pseudo_channel = divmod(
            within_stack, self.config["pseudo_channels_per_channel"])
        bank, bank_group = divmod(
            bank_interleave,
            self.config["bank_groups_per_pseudo_channel"],
        )
        return Address(
            stack=stack,
            channel=channel,
            pseudo_channel=pseudo_channel,
            bank_group=bank_group,
            bank=bank,
            row=after_bank // row_bursts,
            offset=column * self.burst_bytes + burst_offset,
        )

    def encode(self, address: Address) -> int:
        total_pc = len(self.pseudo_channels)
        banks_per_pc = (
            self.config["bank_groups_per_pseudo_channel"]
            * self.config["banks_per_group"]
        )
        pseudo_linear = (
            (
                address.stack * self.config["channels_per_stack"]
                + address.channel
            )
            * self.config["pseudo_channels_per_channel"]
            + address.pseudo_channel
        )
        bank_interleave = (
            address.bank
            * self.config["bank_groups_per_pseudo_channel"]
            + address.bank_group
        )
        column, burst_offset = divmod(address.offset, self.burst_bytes)
        row_bursts = self.row_size_bytes // self.burst_bytes
        row_column = address.row * row_bursts + column
        bank_lane = (
            bank_interleave - _bank_hash(row_column) % banks_per_pc
        ) % banks_per_pc
        group = row_column * banks_per_pc + bank_lane
        lane = (pseudo_linear - _pseudo_hash(group) % total_pc) % total_pc
        return (group * total_pc + lane) * self.burst_bytes + burst_offset

    def pc_index(self, address: Address) -> int:
        return (
            (
                address.stack * self.config["channels_per_stack"]
                + address.channel
            )
            * self.config["pseudo_channels_per_channel"]
            + address.pseudo_channel
        )

    def bank_index(self, address: Address) -> int:
        return (
            address.bank_group * self.config["banks_per_group"]
            + address.bank
        )

    def timing(self, name: str) -> float:
        return _align(float(self.config[name]), self.tck)

    def column_latency(self, op: str) -> float:
        return self.timing("tCL_ns" if op == "read" else "tCWL_ns")

    def command(
        self,
        pc: PseudoChannel,
        address: Address,
        op: str,
        command: int,
        earliest_ns: float,
        spans: list[Span],
    ) -> tuple[float, float]:
        bank = pc.banks[self.bank_index(address)]
        group = pc.groups[address.bank_group]
        issue = _align(
            max(
                earliest_ns,
                pc.ready[command],
                group.ready[command],
                bank.ready[command],
            ),
            self.tck,
        )
        if command == PRE:
            name = "PRE"
            duration = self.timing("tRP_ns")
        elif command == ACT:
            name = "ACT"
            duration = self.timing(
                "tRCDRD_ns" if op == "read" else "tRCDWR_ns")
        elif command == RD:
            name = "RD"
            duration = self.column_latency("read")
        elif command == WR:
            name = "WR"
            duration = self.column_latency("write")
        else:
            raise AssertionError("unsupported oracle command")
        spans.append(Span(
            name=name,
            category="hbm_command",
            entity=address.path(),
            start_ns=issue,
            finish_ns=issue + duration,
        ))
        self.apply_command(pc, address, command, issue)
        return issue, issue + duration

    def apply_command(
        self,
        pc: PseudoChannel,
        address: Address,
        command: int,
        issue_ns: float,
    ) -> None:
        bank = pc.banks[self.bank_index(address)]
        group = pc.groups[address.bank_group]
        burst = self.config["burst_length"] / self.config["pin_rate_Gbps"]
        if command == PRE:
            bank.open_row = None
            bank.ready[ACT] = max(
                bank.ready[ACT], issue_ns + self.timing("tRP_ns"))
        elif command == ACT:
            bank.open_row = address.row
            bank.ready[PRE] = max(
                bank.ready[PRE], issue_ns + self.timing("tRAS_ns"))
            bank.ready[ACT] = max(
                bank.ready[ACT],
                issue_ns
                + max(
                    self.timing("tRC_ns"),
                    self.timing("tRAS_ns") + self.timing("tRP_ns"),
                ),
            )
            bank.ready[RD] = max(
                bank.ready[RD], issue_ns + self.timing("tRCDRD_ns"))
            bank.ready[WR] = max(
                bank.ready[WR], issue_ns + self.timing("tRCDWR_ns"))
            pc.ready[ACT] = max(
                pc.ready[ACT], issue_ns + self.timing("tRRD_S_ns"))
            group.ready[ACT] = max(
                group.ready[ACT], issue_ns + self.timing("tRRD_L_ns"))
            pc.recent_activations.append(issue_ns)
            pc.recent_activations = pc.recent_activations[-4:]
            if len(pc.recent_activations) == 4:
                pc.ready[ACT] = max(
                    pc.ready[ACT],
                    pc.recent_activations[0] + self.timing("tFAW_ns"),
                )
        elif command == RD:
            bank.ready[PRE] = max(
                bank.ready[PRE], issue_ns + self.timing("tRTP_ns"))
            pc.ready[RD] = max(
                pc.ready[RD],
                issue_ns + self.config["tCCD_S_cycles"] * self.tck,
            )
            group.ready[RD] = max(
                group.ready[RD],
                issue_ns + self.config["tCCD_L_cycles"] * self.tck,
            )
            pc.ready[WR] = max(
                pc.ready[WR], issue_ns + self.timing("tRTW_ns"))
        elif command == WR:
            write_data_end = (
                issue_ns + self.column_latency("write") + burst
            )
            bank.ready[PRE] = max(
                bank.ready[PRE],
                write_data_end + self.timing("tWR_ns"),
            )
            pc.ready[WR] = max(
                pc.ready[WR],
                issue_ns + self.config["tCCD_S_cycles"] * self.tck,
            )
            group.ready[WR] = max(
                group.ready[WR],
                issue_ns + self.config["tCCD_L_cycles"] * self.tck,
            )
            pc.ready[RD] = max(
                pc.ready[RD],
                write_data_end + self.timing("tWTR_S_ns"),
            )
            group.ready[RD] = max(
                group.ready[RD],
                write_data_end + self.timing("tWTR_L_ns"),
            )

    def service_child(
        self,
        request: dict[str, Any],
        child_addr: int,
    ) -> ChildCompletion:
        address = self.decode(child_addr)
        pc = self.pseudo_channels[self.pc_index(address)]
        bank = pc.banks[self.bank_index(address)]
        spans: list[Span] = []
        ready = request["arrival_ns"] + self.config["address_mapping_ns"]
        if self.config["address_mapping_ns"] > 0.0:
            spans.append(Span(
                name="address_map",
                category="mapping",
                entity=address.path(),
                start_ns=request["arrival_ns"],
                finish_ns=ready,
            ))
        ready = _align(ready, self.tck)
        first_command: float | None = None
        precharged = False
        activated = False
        if bank.open_row is not None and bank.open_row != address.row:
            issue, ready = self.command(
                pc, address, request["op"], PRE, ready, spans)
            first_command = issue
            precharged = True
            self.counters["precharges"] += 1
        if bank.open_row is None:
            issue, ready = self.command(
                pc, address, request["op"], ACT, ready, spans)
            if first_command is None:
                first_command = issue
            activated = True
            self.counters["activations"] += 1

        final = RD if request["op"] == "read" else WR
        ready = max(ready, pc.bus_ready_ns - self.column_latency(request["op"]))
        issue, ready = self.command(
            pc, address, request["op"], final, ready, spans)
        if first_command is None:
            first_command = issue
        bus_start = max(ready, pc.bus_ready_ns)
        burst_ns = (
            self.config["burst_length"] / self.config["pin_rate_Gbps"])
        finish = bus_start + burst_ns
        bus_entity = (
            f"stack{address.stack}/ch{address.channel}"
            f"/pch{address.pseudo_channel}"
        )
        spans.append(Span(
            name=(
                "read_burst"
                if request["op"] == "read"
                else "write_burst"
            ),
            category="hbm_bus",
            entity=bus_entity,
            start_ns=bus_start,
            finish_ns=finish,
        ))
        pc.bus_ready_ns = finish
        pc.bus_busy_ns += burst_ns
        pc.accesses += 1
        self.counters["bus_busy_ns"] += burst_ns
        self.counters["finish_ns"] = max(
            float(self.counters["finish_ns"]), finish)
        counter = "read_bytes" if request["op"] == "read" else "write_bytes"
        self.counters[counter] += self.burst_bytes
        if precharged:
            result = "row-conflict"
            self.counters["row_conflicts"] += 1
        elif activated:
            result = "row-miss"
            self.counters["row_misses"] += 1
        else:
            result = "row-hit"
            self.counters["row_hits"] += 1
        return ChildCompletion(
            start_ns=first_command,
            finish_ns=finish,
            physical_bytes=self.burst_bytes,
            resource=address.path(),
            result=result,
            spans=spans,
        )

    def preview_first_command(
        self,
        request: dict[str, Any],
        child_addr: int,
    ) -> tuple[float, bool]:
        address = self.decode(child_addr)
        pc = self.pseudo_channels[self.pc_index(address)]
        bank = pc.banks[self.bank_index(address)]
        group = pc.groups[address.bank_group]
        final = RD if request["op"] == "read" else WR
        if bank.open_row is None:
            command = ACT
        elif bank.open_row != address.row:
            command = PRE
        else:
            command = final
        issue = _align(
            max(
                request["arrival_ns"]
                + self.config["address_mapping_ns"],
                pc.ready[command],
                group.ready[command],
                bank.ready[command],
            ),
            self.tck,
        )
        if command == final:
            issue = _align(
                max(
                    issue,
                    pc.bus_ready_ns
                    - self.column_latency(request["op"]),
                ),
                self.tck,
            )
        return issue, command == final

    def pick_next_child(
        self,
        request: dict[str, Any],
        queue: list[QueuedChild],
    ) -> int:
        if len(queue) == 1:
            return 0
        oldest_issue, _ = self.preview_first_command(
            request, queue[0].addr)
        # Every child in this queue belongs to the same parent and therefore
        # has the same arrival. The production time-based starvation cap is
        # vacuous here; the independent bypass-count cap still applies.
        for index, candidate in enumerate(queue):
            if (
                index != 0
                and queue[index - 1].bypass_count
                >= self.config["queue_depth"]
            ):
                break
            issue, row_hit = self.preview_first_command(
                request, candidate.addr)
            if issue <= oldest_issue and row_hit:
                return index
        return 0

    def issue(self, request: dict[str, Any]) -> dict[str, Any]:
        first_burst = request["addr"] - request["addr"] % self.burst_bytes
        last_byte = request["addr"] + request["bytes"] - 1
        last_burst = last_byte - last_byte % self.burst_bytes
        queues: dict[int, list[QueuedChild]] = {}
        routes: list[int] = []
        cursor = request["addr"]
        remaining = request["bytes"]
        while remaining:
            child_bytes = min(
                remaining, self.burst_bytes - cursor % self.burst_bytes)
            address = self.decode(cursor)
            pc_index = self.pc_index(address)
            if pc_index not in queues:
                queues[pc_index] = []
                routes.append(pc_index)
            queue = queues[pc_index]
            queue.append(QueuedChild(cursor))
            self.counters["max_queue_occupancy"] = max(
                int(self.counters["max_queue_occupancy"]), len(queue))
            remaining -= child_bytes
            cursor += child_bytes
        expected_bursts = (
            (last_burst - first_burst) // self.burst_bytes + 1)
        if sum(map(len, queues.values())) != expected_bursts:
            raise AssertionError("oracle burst split lost a child")

        children: list[ChildCompletion] = []
        while any(queues.values()):
            for route in routes:
                if queues[route]:
                    queue = queues[route]
                    picked = self.pick_next_child(request, queue)
                    for queued in queue[:picked]:
                        queued.bypass_count += 1
                    child = queue.pop(picked)
                    children.append(
                        self.service_child(request, child.addr))
        critical = children[0]
        for child in children[1:]:
            if child.finish_ns > critical.finish_ns:
                critical = child
        spans = [span for child in children for span in child.spans]
        spans.sort(key=lambda span: (
            span.start_ns,
            span.finish_ns,
            span.category,
            span.name,
            span.entity,
        ))
        split = len(children) > 1
        return {
            "start_ns": min(child.start_ns for child in children),
            "finish_ns": max(child.finish_ns for child in children),
            "logical_bytes": request["bytes"],
            "physical_bytes": sum(
                child.physical_bytes for child in children),
            "resource": (
                f"hbm/split/{len(children)}-bursts/"
                f"{len(routes)}-pseudochannels"
                if split else critical.resource
            ),
            "result": "split-burst-request" if split else critical.result,
            "spans": spans,
        }

    def finalize_counters(self) -> None:
        active = [
            pc for pc in self.pseudo_channels
            if pc.accesses or pc.bus_busy_ns > 0.0
        ]
        self.counters["active_pseudo_channels"] = len(active)
        self.counters["max_pseudo_channel_accesses"] = max(
            (pc.accesses for pc in active), default=0)


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
