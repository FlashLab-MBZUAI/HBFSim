"""Independent tiny-geometry oracle for the HBF state and resource paths.

The oracle uses explicit interval lists and simple full scans; it shares no
calendars, placement helpers, or statistics code with the production device.
Its intentionally small scope is expanded only through reviewed validation
cases, so unsupported behaviors fail closed instead of silently borrowing
production logic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from verification.core.contracts import LEDGER_SCHEMA, hbf_buffer_hbm_config
from verification.oracles.hbm import Oracle as HbmAddressOracle, command_clock_cycles


MASK64 = (1 << 64) - 1


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


@dataclass(frozen=True)
class Address:
    stack: int = 0
    channel: int = 0
    die: int = 0
    plane: int = 0
    block: int = 0
    page: int = 0
    offset: int = 0

    def path(self) -> str:
        return (
            f"stack{self.stack}/ch{self.channel}/die{self.die}"
            f"/plane{self.plane}/block{self.block}/page{self.page}"
            f"/off{self.offset}"
        )

    def channel_path(self) -> str:
        return f"stack{self.stack}/ch{self.channel}"

    def die_path(self) -> str:
        return f"{self.channel_path()}/die{self.die}"

    def plane_path(self) -> str:
        return f"{self.die_path()}/plane{self.plane}"


@dataclass(frozen=True)
class Reservation:
    start_ns: float
    finish_ns: float


class Timeline:
    """Exact earliest-gap reservation for a tiny interval set."""

    def __init__(self) -> None:
        self.intervals: list[Reservation] = []
        self.reserved_work_ns = 0.0

    def reserve(self, earliest_ns: float, duration_ns: float) -> Reservation:
        candidate = self.earliest(earliest_ns, duration_ns)
        reservation = Reservation(candidate, candidate + duration_ns)
        position = 0
        while (
            position < len(self.intervals)
            and self.intervals[position].start_ns < reservation.start_ns
        ):
            position += 1
        self.intervals.insert(position, reservation)
        self.reserved_work_ns += duration_ns
        return reservation

    def earliest(self, earliest_ns: float, duration_ns: float) -> float:
        candidate = earliest_ns
        for interval in self.intervals:
            if candidate + duration_ns <= interval.start_ns:
                break
            if candidate < interval.finish_ns:
                candidate = interval.finish_ns
        return candidate


@dataclass
class LogicResources:
    ingress: Timeline = field(default_factory=Timeline)
    tsv: Timeline = field(default_factory=Timeline)
    sram: Timeline = field(default_factory=Timeline)
    mapping_issue: Timeline = field(default_factory=Timeline)
    write_buffer_dram_issue: Timeline = field(default_factory=Timeline)


@dataclass
class ChannelResources:
    external_command: Timeline = field(default_factory=Timeline)
    external_rx: Timeline = field(default_factory=Timeline)
    external_tx: Timeline = field(default_factory=Timeline)
    command: Timeline = field(default_factory=Timeline)
    data: Timeline = field(default_factory=Timeline)
    command_count: int = 0
    data_count: int = 0


@dataclass
class DieResources:
    source_user: Timeline = field(default_factory=Timeline)
    source_mapping: Timeline = field(default_factory=Timeline)
    source_gc: Timeline = field(default_factory=Timeline)
    sequencer: Timeline = field(default_factory=Timeline)
    ecc_issue: Timeline = field(default_factory=Timeline)
    transactions: int = 0


@dataclass
class PlaneResources:
    subarrays: list[Timeline]
    lanes: list[Timeline]
    page_buffers: list[Timeline]
    read_count: int = 0
    program_count: int = 0
    erase_count: int = 0
    subarray_reads: list[int] = field(default_factory=list)
    lane_reads: list[int] = field(default_factory=list)
    page_buffer_reads: list[int] = field(default_factory=list)
    media_intervals: list[Reservation] = field(default_factory=list)


@dataclass
class Span:
    name: str
    category: str
    entity: str
    start_ns: float
    finish_ns: float
    critical: bool = True
    physical_bytes: int = 0


@dataclass
class Block:
    role: str = "free"
    valid_pages: int = 0
    invalid_pages: int = 0
    free_pages: int = 0
    next_page: int = 0
    erase_count: int = 0
    pending_program_pages: int = 0
    pending_mapping_publications: int = 0
    epoch: int = 0
    erase_pending: bool = False


@dataclass
class Page:
    status: str = "erased"
    owner: str = "unassigned"
    logical_key: int = 0
    block_epoch: int = 0


@dataclass
class WriteBufferEntry:
    ranges: list[tuple[int, int]] = field(default_factory=list)
    ready_ns: float = 0.0


class Oracle:
    def __init__(self, case: dict[str, Any], *, buffer_hbm: HbmAddressOracle | None = None) -> None:
        self.case = case
        self.config = dict(case["config"])
        # The address map is the independent Python HBM oracle. Buffer data
        # arbitration below uses only this oracle's explicit interval lists,
        # one per HBM service group so DMA and application traffic share it.
        self.buffer_hbm = buffer_hbm or HbmAddressOracle({"config": hbf_buffer_hbm_config()})
        self.buffer_buses = [Timeline() for _ in range(self.buffer_hbm.total_service_groups)]
        for group, bus in zip(self.buffer_hbm.service_groups, self.buffer_buses, strict=True):
            group.data_bus = bus
        self.buffer_cursors = [0] * self.config["stacks"]
        # Host mapping control, publication and codec work run on a pool of
        # compute workers shared across stacks (one worker per stack), separate
        # from the per-stack pipelined metadata-memory issue ports.
        self.mapping_workers = [Timeline() for _ in range(self.config["stacks"])]
        # The production calendars reclaim history behind each request's
        # arrival; the "most recently freed" worker tie-break observes an idle
        # interval as beginning no earlier than that watermark.
        self.causal_watermark_ns = 0.0
        self.effective_subarrays = 1
        self.config["hb_io_bandwidth_GBps"] = (
            {1:48.0, 2:96.0, 3:192.0}[self.config["speed_grade"]]
            * self.config["channels_per_stack"])
        self.total_channels = (
            self.config["stacks"] * self.config["channels_per_stack"])
        self.total_dies = (
            self.total_channels * self.config["dies_per_channel"])
        self.total_planes = (
            self.total_dies * self.config["planes_per_die"])
        self.total_pages = (
            self.total_planes
            * self.config["blocks_per_plane"]
            * self.config["pages_per_block"]
        )
        self.channel_payload_bw = {1:48.0, 2:96.0, 3:192.0}[self.config["speed_grade"]]
        self.logic = [
            LogicResources() for _ in range(self.config["stacks"])]
        self.channels = [
            ChannelResources() for _ in range(self.total_channels)]
        self.dies = [DieResources() for _ in range(self.total_dies)]
        self.planes = [
            PlaneResources(
                subarrays=[
                    Timeline() for _ in range(self.effective_subarrays)],
                lanes=[
                    Timeline()
                    for _ in range(self.config["media_lanes_per_plane"])
                ],
                page_buffers=[
                    Timeline()
                    for _ in range(
                        self.config["page_buffer_banks_per_plane"])
                ],
                subarray_reads=[0] * self.effective_subarrays,
                lane_reads=[0] * self.config["media_lanes_per_plane"],
                page_buffer_reads=[
                    0
                    for _ in range(
                        self.config["page_buffer_banks_per_plane"])
                ],
            )
            for _ in range(self.total_planes)
        ]
        self.lpn_to_ppn: dict[int, int] = {}
        self.mapping_vpn_to_ppn: dict[int, int] = {}
        self.free_blocks_by_plane = [
            [
                plane * self.config["blocks_per_plane"] + block
                for block in range(self.config["blocks_per_plane"])
            ]
            for plane in range(self.total_planes)
        ]
        self.blocks = [
            Block(free_pages=self.config["pages_per_block"])
            for _ in range(
                self.total_planes * self.config["blocks_per_plane"])
        ]
        self.pages: dict[int, Page] = {}
        self.cache_events: dict[int, list[tuple[float, int, int, int]]] = {}
        self.cache_sequence = 0
        self.last_decoded_ready = 0.0
        self.active_blocks: dict[tuple[str, int], int] = {}
        self.allocation_cursors = {
            "data": [0] * self.config["stacks"],
            "mapping": [0] * self.config["stacks"],
            "gc": [0] * self.config["stacks"],
        }
        self.dirty_mapping_events: dict[
            int, list[tuple[float, int]]
        ] = {}
        self.next_dirty_mapping_sequence = 0
        self.lpn_ready_ns: dict[int, float] = {}
        self.materialized_lpn_ready_ns: dict[int, float] = {}
        self.write_buffers: list[dict[int, WriteBufferEntry]] = [
            {} for _ in range(self.config["stacks"])
        ]
        self.background_finish_ns = 0.0
        self.valid_pages = 0
        self.counters: dict[str, int | float | bool] = {
            "read_requests": 0,
            "program_requests": 0,
            "logical_read_bytes": 0,
            "logical_write_bytes": 0,
            "physical_read_bytes": 0,
            "physical_write_bytes": 0,
            "data_program_payload_bytes": 0,
            "mapping_program_payload_bytes": 0,
            "gc_relocation_payload_bytes": 0,
            "page_reads": 0,
            "data_programs": 0,
            "page_programs": 0,
            "mapping_page_programs": 0,
            "invalidations": 0,
            "write_buffer_hits": 0,
            "write_buffer_misses": 0,
            "write_buffer_flushes": 0,
            "write_buffer_merged_bytes": 0,
            "write_buffer_slot_wait_ops": 0,
            "write_buffer_slot_wait_ns": 0.0,
            "block_erases": 0,
            "gc_runs": 0,
            "gc_relocations": 0,
            "gc_data_relocations": 0,
            "gc_mapping_relocations": 0,
            "gc_reclaimed_invalid_pages": 0,
            "gc_user_blocked_runs": 0,
            "mapping_lookup_ops": 0,
            "mapping_user_lookup_ops": 0,
            "mapping_gc_lookup_ops": 0,
            "mapping_update_ops": 0,
            "mapping_user_update_ops": 0,
            "mapping_gc_update_ops": 0,
            "mapping_dram_issue_busy_ns": 0.0,
            "mapping_compute_work_ns": 0.0,
            "mapping_compute_ops": 0,
            "page_read_admission_events": 0,
            "flash_scheduler_enqueues": 0,
            "flash_scheduler_issues": 0,
            "ecc_decode_ops": 0,
            "ecc_decode_codeword_bytes": 0,
            "ecc_encode_ops": 0,
            "ecc_encode_codeword_bytes": 0,
            "ecc_issue_busy_ns": 0.0,
            "media_busy_ns": 0.0,
            "channel_command_busy_ns": 0.0,
            "channel_data_busy_ns": 0.0,
            "tsv_busy_ns": 0.0,
            "sram_busy_ns": 0.0,
            "hb_io_command_busy_ns": 0.0,
            "hb_io_data_busy_ns": 0.0,
            "finish_ns": 0.0,
            "total_pages": self.total_pages,
            "free_pages": self.total_pages,
            "valid_pages": 0,
            "invalid_pages": 0,
            "pending_program_pages": 0,
            "pending_mapping_publications": 0,
            "accounting_verified": True,
            "active_planes": 0,
            "active_media_lanes": 0,
            "active_subarrays": 0,
            "active_page_buffer_banks": 0,
            "active_channels": 0,
            "active_dies": 0,
        }
        self.prepopulate(case["initial_state"].get("prepopulate_lpns", []))

    def channel_index(self, address: Address) -> int:
        return (
            address.stack * self.config["channels_per_stack"]
            + address.channel
        )

    def die_index(self, address: Address) -> int:
        return (
            self.channel_index(address) * self.config["dies_per_channel"]
            + address.die
        )

    def plane_index(self, address: Address) -> int:
        return (
            self.die_index(address) * self.config["planes_per_die"]
            + address.plane
        )

    def planes_per_stack(self) -> int:
        return self.total_planes // self.config["stacks"]

    def encode_ppn(self, address: Address) -> int:
        unit = address.stack
        unit = unit * self.config["channels_per_stack"] + address.channel
        unit = unit * self.config["dies_per_channel"] + address.die
        unit = unit * self.config["planes_per_die"] + address.plane
        unit = unit * self.config["blocks_per_plane"] + address.block
        return unit * self.config["pages_per_block"] + address.page

    def encode(self, address: Address) -> int:
        return (
            self.encode_ppn(address) * self.config["page_size_bytes"]
            + address.offset
        )

    def decode(self, byte_address: int) -> Address:
        unit, offset = divmod(
            byte_address, self.config["page_size_bytes"])
        unit, page = divmod(unit, self.config["pages_per_block"])
        unit, block = divmod(unit, self.config["blocks_per_plane"])
        unit, plane = divmod(unit, self.config["planes_per_die"])
        unit, die = divmod(unit, self.config["dies_per_channel"])
        stack, channel = divmod(unit, self.config["channels_per_stack"])
        return Address(
            stack=stack,
            channel=channel,
            die=die,
            plane=plane,
            block=block,
            page=page,
            offset=offset,
        )

    def decode_ppn(self, ppn: int) -> Address:
        return self.decode(ppn * self.config["page_size_bytes"])

    def mapping_vpn(self, lpn: int) -> int:
        stacks = self.config["stacks"]
        stripe, lane = divmod(lpn, stacks)
        group = stripe // self.config["mapping_entries_per_page"]
        rotation = _mix64(group) % stacks
        owner = (lane + rotation) % stacks
        return group * stacks + owner

    def stack_for_lpn(self, lpn: int) -> int:
        return self.mapping_vpn(lpn) % self.config["stacks"]

    def mapping_plane(self, vpn: int) -> int:
        stack = vpn % self.config["stacks"]
        return (
            stack * self.planes_per_stack()
            + (vpn // self.config["stacks"]) % self.planes_per_stack()
        )

    def mark_mapping_dirty(self, vpn: int, at_ns: float) -> None:
        event = (at_ns, self.next_dirty_mapping_sequence)
        self.next_dirty_mapping_sequence += 1
        events = self.dirty_mapping_events.setdefault(vpn, [])
        events.append(event)
        events.sort()

    def allocate_page(
        self,
        role: str,
        stack: int,
        preferred_plane: int | None = None,
    ) -> int:
        if preferred_plane is None:
            local_plane = self.allocation_cursors[role][stack]
            self.allocation_cursors[role][stack] = (
                self.allocation_cursors[role][stack] + 1
            ) % self.planes_per_stack()
            plane = stack * self.planes_per_stack() + local_plane
        else:
            plane = preferred_plane
            active = self.active_blocks.get((role, plane))
            if role != "gc" and (active is None or self.blocks[active].free_pages == 0):
                self.allocation_cursors[role][stack] = (self.allocation_cursors[role][stack] + 1) % self.planes_per_stack()
        key = (role, plane)
        block = self.active_blocks.get(key)
        if (
            block is None
            or self.blocks[block].free_pages == 0
        ):
            if not self.free_blocks_by_plane[plane]:
                raise ValueError("HBF oracle exhausted a plane")
            block = self.free_blocks_by_plane[plane].pop(0)
            self.active_blocks[key] = block
            self.blocks[block].role = role
        block_state = self.blocks[block]
        page = block_state.next_page
        block_state.next_page += 1
        block_state.free_pages -= 1
        block_state.pending_program_pages += 1
        if block_state.free_pages == 0:
            del self.active_blocks[key]
        ppn = block * self.config["pages_per_block"] + page
        self.pages[ppn] = Page(block_epoch=block_state.epoch)
        self.counters["free_pages"] = (
            int(self.counters["free_pages"]) - 1)
        return ppn

    def mark_programmed(
        self,
        ppn: int,
        logical_key: int,
        owner: str,
    ) -> None:
        page = self.pages[ppn]
        block = self.blocks[ppn // self.config["pages_per_block"]]
        if page.status != "erased" or block.pending_program_pages == 0:
            raise ValueError("HBF oracle program commit lost its reservation")
        page.status = "valid"
        page.owner = owner
        page.logical_key = logical_key
        block.pending_program_pages -= 1
        block.valid_pages += 1
        self.valid_pages += 1

    def invalidate(self, ppn: int) -> None:
        page = self.pages[ppn]
        if page.status != "valid":
            raise ValueError("HBF oracle invalidated a non-valid page")
        block = self.blocks[ppn // self.config["pages_per_block"]]
        page.status = "invalid"
        block.valid_pages -= 1
        block.invalid_pages += 1
        self.valid_pages -= 1
        self.counters["invalidations"] += 1

    def prepopulate(self, lpns: list[int]) -> None:
        for lpn in lpns:
            ppn = self.allocate_page(
                "data", self.stack_for_lpn(lpn))
            self.lpn_to_ppn[lpn] = ppn
            self.mark_programmed(ppn, lpn, "logical")
            vpn = self.mapping_vpn(lpn)
            if vpn not in self.mapping_vpn_to_ppn:
                mapping_ppn = self.allocate_page(
                    "mapping",
                    vpn % self.config["stacks"],
                    self.mapping_plane(vpn),
                )
                self.mapping_vpn_to_ppn[vpn] = mapping_ppn
                self.mark_programmed(
                    mapping_ppn, (1 << 63) | vpn, "mapping")
        self.counters["valid_pages"] = self.valid_pages

    def derived_config(self) -> dict[str, Any]:
        config = dict(self.config)
        pages_per_stack = self.total_pages // config["stacks"]
        mapping_pages_per_stack = (
            pages_per_stack + config["mapping_entries_per_page"] - 1
        ) // config["mapping_entries_per_page"]
        mapping_bytes_per_stack = (
            mapping_pages_per_stack * config["page_size_bytes"])
        config["derived"] = {
            "effective_subarrays_per_plane": self.effective_subarrays,
            "page_wire_bytes": (
                config["page_size_bytes"] + config["oob_bytes_per_page"]),
            "total_channels": self.total_channels,
            "total_dies": self.total_dies,
            "total_planes": self.total_planes,
            "total_pages": self.total_pages,
            "capacity_bytes": (
                self.total_pages * config["page_size_bytes"]),
            "resident_mapping_pages_per_stack": mapping_pages_per_stack,
            "resident_mapping_bytes_per_stack": mapping_bytes_per_stack,
            "resident_mapping_total_bytes": (
                mapping_bytes_per_stack * config["stacks"]),
        }
        return config

    def reserve(
        self,
        timeline: Timeline,
        earliest_ns: float,
        duration_ns: float,
    ) -> Reservation:
        return timeline.reserve(earliest_ns, duration_ns)

    def external_command(self, ppn: int, ready: float, source: str, spans: list[Span]) -> float:
        address = self.decode_ppn(ppn)
        transfer = self.channels[self.channel_index(address)].external_command.reserve(
            ready, self.config["command_address_bytes"] / self.channel_payload_bw)
        spans.append(Span(f"{source}/request_hbio", "hbio", address.channel_path()+"/hbio/command",
            transfer.start_ns, transfer.finish_ns))
        return transfer.finish_ns

    def select_mapping_worker(self, ready_ns: float, work_ns: float) -> int:
        """Earliest-start worker; equal starts prefer the most recently freed.

        A worker's idle interval is observed as beginning at the later of the
        preceding reservation's finish and the causal watermark, because the
        production calendars reclaim idle history behind that watermark.
        """
        selected = 0
        best_start = float("inf")
        best_idle_begin = -1.0
        for index, worker in enumerate(self.mapping_workers):
            start = worker.earliest(ready_ns, work_ns)
            preceding_finish = max(
                (
                    interval.finish_ns
                    for interval in worker.intervals
                    if interval.finish_ns <= start
                ),
                default=0.0,
            )
            idle_begin = max(preceding_finish, self.causal_watermark_ns)
            if start < best_start or (
                start == best_start and idle_begin > best_idle_begin
            ):
                selected = index
                best_start = start
                best_idle_begin = idle_begin
        return selected

    def service_mapping_compute(
        self,
        work_ns: float,
        ready_ns: float,
        name: str,
        spans: list[Span],
    ) -> float:
        if work_ns == 0:
            return ready_ns
        worker = self.select_mapping_worker(ready_ns, work_ns)
        work = self.reserve(self.mapping_workers[worker], ready_ns, work_ns)
        spans.append(Span(
            name,
            "mapping_compute",
            f"host/mapping/worker{worker}",
            work.start_ns,
            work.finish_ns,
        ))
        self.counters["mapping_compute_work_ns"] += work_ns
        self.counters["mapping_compute_ops"] += 1
        return work.finish_ns

    def access_mapping(
        self,
        lpn: int,
        ready_ns: float,
        kind: str,
        spans: list[Span],
        *,
        source: str = "user",
    ) -> float:
        """One host L2P access: control compute, metadata DRAM, then publish.

        Lookups and updates first pay ``mapping_control_compute_ns`` on the
        shared host worker pool, then one pipelined issue on the stack's
        metadata-memory port whose entry transfer shares the HBM buffer
        calendars. An update (user or GC alike) then publishes for
        ``mapping_update_ns`` on the worker pool again.
        """
        stack = self.stack_for_lpn(lpn)
        logic = self.logic[stack]
        control_finish = self.service_mapping_compute(
            self.config["mapping_control_compute_ns"],
            ready_ns,
            (
                "mapping_lookup_compute"
                if kind == "lookup"
                else "mapping_update_compute"
            ),
            spans,
        )
        mapping_issue = self.reserve(
            logic.mapping_issue,
            control_finish,
            self.config["ctrl_dram_issue_ns"],
        )
        data_finish = self.host_memory_transfer(stack,
            self.config["page_size_bytes"] // self.config["mapping_entries_per_page"],
            kind != "lookup", mapping_issue.start_ns, spans)
        mapping_finish = max(
            mapping_issue.finish_ns,
            data_finish,
            mapping_issue.start_ns + self.config["ctrl_dram_latency_ns"],
        )
        spans.append(Span(
            (
                "resident_mapping_lookup"
                if kind == "lookup"
                else "resident_mapping_update_access"
            ),
            "metadata",
            f"host/mapping/partition{stack}",
            mapping_issue.start_ns,
            mapping_finish,
        ))
        if kind == "lookup":
            self.counters["mapping_lookup_ops"] += 1
            self.counters[
                (
                    "mapping_user_lookup_ops"
                    if source == "user"
                    else "mapping_gc_lookup_ops"
                )
            ] += 1
            return mapping_finish
        self.counters["mapping_update_ops"] += 1
        self.counters[
            (
                "mapping_user_update_ops"
                if source == "user"
                else "mapping_gc_update_ops"
            )
        ] += 1
        return self.service_mapping_compute(
            self.config["mapping_update_ns"],
            mapping_finish,
            "mapping_publish_compute",
            spans,
        )

    def host_memory_transfer(self, stack: int, bytes_: int, write: bool,
                             ready: float, spans: list[Span]) -> float:
        memory = self.buffer_hbm
        burst = memory.burst_bytes
        reserved = (self.config["ctrl_dram_bytes"] + burst - 1) // burst * burst
        partition = self.config["ctrl_dram_bytes"] // self.config["stacks"] // burst * burst
        rounded = (bytes_ + burst - 1) // burst * burst
        if not partition or rounded > partition:
            raise ValueError("HBF oracle buffer transfer exceeds its HBM partition")
        if self.buffer_cursors[stack] + rounded > partition:
            self.buffer_cursors[stack] = 0
        addr = memory.config["capacity_bytes"] - reserved + stack * partition + self.buffer_cursors[stack]
        self.buffer_cursors[stack] += rounded
        offered = command_clock_cycles(ready, memory.tck)
        finish = ready
        # Controller DMA is a data-bus transfer whose data is already ready:
        # each touched service group serves its busiest lane's bursts once,
        # sharing the group calendar with application accesses but paying no
        # second access latency and no direction-change ordering.
        for index, lanes in sorted(memory.group_demand(addr, rounded).items()):
            demand = max(lanes.values())
            interval = memory.service_groups[index].data_bus.reserve(
                offered, memory.service_cycles(demand))
            start_ns = max(ready, interval.start_ns * memory.tck)
            finish_ns = max(start_ns, interval.finish_ns * memory.tck)
            spans.append(Span("hbf_buffer_write" if write else "hbf_buffer_read", "hbm_buffer_bus",
                f"hbm/group{index}", start_ns, finish_ns, False, sum(lanes.values()) * burst))
            memory.account_lanes(lanes)
            finish = max(finish, finish_ns)
        memory.counters["write_bytes" if write else "read_bytes"] += rounded
        memory.counters["bus_busy_ns"] = memory.ns(memory.total_bus_busy_cycles)
        memory.counters["finish_ns"] = max(memory.counters["finish_ns"], finish)
        return finish

    def access_write_buffer_dram(
        self,
        stack: int,
        ready_ns: float,
        bytes_: int,
        name: str,
        spans: list[Span],
    ) -> float:
        if bytes_ <= 0 or bytes_ > self.config["page_size_bytes"]:
            raise ValueError(
                "HBF oracle write-buffer DRAM access must contain "
                "1..page_size bytes")
        issue = self.reserve(
            self.logic[stack].write_buffer_dram_issue,
            ready_ns,
            self.config["ctrl_dram_issue_ns"],
        )
        data_finish = self.host_memory_transfer(stack, bytes_, name == "write_buffer_stage_dram",
                                                 issue.start_ns, spans)
        finish = max(
            issue.finish_ns,
            data_finish,
            issue.start_ns + self.config["ctrl_dram_latency_ns"],
        )
        spans.append(Span(
            name,
            "controller_dram",
            f"stack{stack}/logic",
            issue.start_ns,
            finish,
        ))
        return finish

    def schedule_read_page(
        self,
        ppn: int,
        earliest_ns: float,
        external_payload_bytes: int,
        spans: list[Span],
        *,
        internal: bool = False,
        source: str = "user",
    ) -> float:
        address = self.decode_ppn(ppn)
        logic = self.logic[address.stack]
        channel = self.channels[self.channel_index(address)]
        die = self.dies[self.die_index(address)]
        plane = self.planes[self.plane_index(address)]
        page_in_plane = (
            address.block * self.config["pages_per_block"] + address.page)
        subarray_index = page_in_plane % self.effective_subarrays
        lane_index = (
            page_in_plane % self.config["media_lanes_per_plane"])
        page_buffer_index = (
            page_in_plane % self.config["page_buffer_banks_per_plane"])
        source_timeline = {
            "user": die.source_user,
            "gc": die.source_gc,
        }[source]

        earliest_ns = self.external_command(ppn, earliest_ns, source, spans)
        command_bytes = self.config["command_address_bytes"]
        command_tsv = self.reserve(
            logic.tsv,
            earliest_ns,
            command_bytes / self.config["tsv_bandwidth_GBps"],
        )
        spans.append(Span(
            f"{source}/cmd_addr_tsv",
            "tsv",
            f"stack{address.stack}/tsv",
            command_tsv.start_ns,
            command_tsv.finish_ns,
        ))
        command_channel = self.reserve(
            channel.command,
            command_tsv.finish_ns,
            command_bytes / self.config["channel_bandwidth_GBps"],
        )
        channel.command_count += 1
        spans.append(Span(
            f"{source}/cmd_addr_channel",
            "flash_channel",
            address.channel_path(),
            command_channel.start_ns,
            command_channel.finish_ns,
        ))

        source_slot = self.reserve(
            source_timeline,
            command_channel.finish_ns,
            self.config["flash_tsu_issue_ns"],
        )
        sequencer = self.reserve(
            die.sequencer,
            source_slot.start_ns,
            self.config["flash_tsu_issue_ns"],
        )
        die.transactions += 1
        spans.append(Span(
            f"{source}/flash_scheduler_issue_read",
            "sequencer",
            address.die_path(),
            sequencer.start_ns,
            sequencer.finish_ns,
        ))

        sense = self.reserve(
            plane.subarrays[subarray_index],
            sequencer.finish_ns,
            self.config["t_read_page_ns"],
        )
        plane.subarray_reads[subarray_index] += 1
        plane.media_intervals.append(sense)
        spans.append(Span(
            f"{source}/array_read",
            "flash_array",
            f"{address.plane_path()}/subarray{subarray_index}",
            sense.start_ns,
            sense.finish_ns,
        ))

        wire_bytes = (
            self.config["page_size_bytes"]
            + self.config["oob_bytes_per_page"]
        )
        lane = self.reserve(
            plane.lanes[lane_index],
            sense.finish_ns,
            wire_bytes / self.config["media_lane_bandwidth_GBps"],
        )
        plane.lane_reads[lane_index] += 1
        spans.append(Span(
            f"{source}/array_to_page_buffer",
            "media_lane",
            f"{address.plane_path()}/lane{lane_index}",
            lane.start_ns,
            lane.finish_ns,
        ))
        page_buffer = self.reserve(
            plane.page_buffers[page_buffer_index],
            lane.finish_ns,
            wire_bytes / self.config["page_buffer_bandwidth_GBps"],
        )
        plane.page_buffer_reads[page_buffer_index] += 1
        spans.append(Span(
            f"{source}/page_buffer_out",
            "page_buffer",
            f"{address.plane_path()}/page_buffer_bank{page_buffer_index}",
            page_buffer.start_ns,
            page_buffer.finish_ns,
        ))
        plane.read_count += 1

        data_channel = self.reserve(
            channel.data,
            page_buffer.finish_ns,
            wire_bytes / self.config["channel_bandwidth_GBps"],
        )
        channel.data_count += 1
        spans.append(Span(
            f"{source}/data_out_channel",
            "flash_channel",
            address.channel_path(),
            data_channel.start_ns,
            data_channel.finish_ns,
        ))
        data_tsv = self.reserve(
            logic.tsv,
            data_channel.finish_ns,
            wire_bytes / self.config["tsv_bandwidth_GBps"],
        )
        spans.append(Span(
            f"{source}/data_out_tsv",
            "tsv",
            f"stack{address.stack}/tsv",
            data_tsv.start_ns,
            data_tsv.finish_ns,
        ))
        ecc_initiation = (
            wire_bytes
            / self.config["ecc_decode_raw_bandwidth_GBps_per_die"]
        )
        ecc_issue = self.reserve(
            die.ecc_issue, data_tsv.finish_ns, ecc_initiation)
        ecc_finish = (
            ecc_issue.start_ns + self.config["ecc_decode_latency_ns"])
        spans.append(Span(
            f"{source}/ecc_decode_issue",
            "ecc_issue",
            f"{address.die_path()}/ecc_issue_port",
            ecc_issue.start_ns,
            ecc_issue.finish_ns,
            critical=False,
        ))
        spans.append(Span(
            f"{source}/ecc_decode_latency",
            "ecc_latency",
            f"{address.die_path()}/ecc_decode_pipeline",
            ecc_issue.start_ns,
            ecc_finish,
        ))
        sram = self.reserve(
            logic.sram,
            ecc_finish,
            self.config["page_size_bytes"]
            / self.config["logic_sram_bandwidth_GBps"],
        )
        spans.append(Span(
            f"{source}/sram_stage_read",
            "sram",
            f"stack{address.stack}/logic",
            sram.start_ns,
            sram.finish_ns,
        ))
        self.counters["flash_scheduler_enqueues"] += 1
        self.counters["flash_scheduler_issues"] += 1
        self.counters["ecc_decode_ops"] += 1
        self.counters["ecc_decode_codeword_bytes"] += wire_bytes
        self.counters["ecc_issue_busy_ns"] += ecc_initiation
        self.last_decoded_ready = sram.finish_ns
        egress = self.reserve(
            channel.external_tx,
            sram.finish_ns,
            (self.config["page_size_bytes"] if internal else external_payload_bytes) / self.channel_payload_bw,
        )
        spans.append(Span(
            "ocp_read_response",
            "hbio",
            address.channel_path()+"/hbio/tx",
            egress.start_ns,
            egress.finish_ns,
        ))
        return (self.host_memory_transfer(address.stack, self.config["page_size_bytes"],
                                          True, egress.finish_ns, spans)
                if internal else egress.finish_ns)

    @staticmethod
    def _common_gap_start(
        timelines: list[Timeline],
        earliest_ns: float,
        duration_ns: float,
    ) -> float:
        candidate = earliest_ns
        intervals = sorted(
            (
                interval
                for timeline in timelines
                for interval in timeline.intervals
            ),
            key=lambda interval: (interval.start_ns, interval.finish_ns),
        )
        for interval in intervals:
            if candidate + duration_ns <= interval.start_ns:
                break
            if candidate < interval.finish_ns:
                candidate = interval.finish_ns
        return candidate

    def schedule_program_page(
        self,
        ppn: int,
        earliest_ns: float,
        source: str,
        spans: list[Span],
    ) -> float:
        address = self.decode_ppn(ppn)
        logic = self.logic[address.stack]
        channel = self.channels[self.channel_index(address)]
        die = self.dies[self.die_index(address)]
        plane = self.planes[self.plane_index(address)]
        source_timeline = {
            "user": die.source_user,
            "mapping": die.source_mapping,
            "gc": die.source_gc,
        }[source]

        command_ready = self.external_command(ppn, earliest_ns, source, spans)
        erase_ready = command_ready
        if address.page == 0:
            erase_ready = self.schedule_erase_block(ppn // self.config["pages_per_block"],
                command_ready, spans, source=source)
        source_ready = (self.host_memory_transfer(address.stack, self.config["page_size_bytes"],
                                                  False, earliest_ns, spans)
                        if source in {"mapping", "gc"} else earliest_ns)
        payload = channel.external_rx.reserve(max(command_ready, source_ready),
            self.config["page_size_bytes"] / self.channel_payload_bw)
        spans.append(Span("ocp_program_payload", "hbio", address.channel_path()+"/hbio/rx",
            payload.start_ns, payload.finish_ns))
        ingress = logic.sram.reserve(payload.finish_ns,
            self.config["page_size_bytes"] / self.config["logic_sram_bandwidth_GBps"])
        spans.append(Span("ocp_page_ingress", "sram", f"stack{address.stack}/logic",
            ingress.start_ns, ingress.finish_ns))
        earliest_ns = max(erase_ready, ingress.finish_ns)
        command_bytes = self.config["command_address_bytes"]
        command_tsv = self.reserve(
            logic.tsv,
            earliest_ns,
            command_bytes / self.config["tsv_bandwidth_GBps"],
        )
        spans.append(Span(
            f"{source}/cmd_addr_tsv",
            "tsv",
            f"stack{address.stack}/tsv",
            command_tsv.start_ns,
            command_tsv.finish_ns,
        ))
        command_channel = self.reserve(
            channel.command,
            command_tsv.finish_ns,
            command_bytes / self.config["channel_bandwidth_GBps"],
        )
        channel.command_count += 1
        spans.append(Span(
            f"{source}/cmd_addr_channel",
            "flash_channel",
            address.channel_path(),
            command_channel.start_ns,
            command_channel.finish_ns,
        ))

        sram = self.reserve(
            logic.sram,
            command_channel.finish_ns,
            self.config["page_size_bytes"]
            / self.config["logic_sram_bandwidth_GBps"],
        )
        spans.append(Span(
            f"{source}/sram_stage_write",
            "sram",
            f"stack{address.stack}/logic",
            sram.start_ns,
            sram.finish_ns,
        ))

        wire_bytes = (
            self.config["page_size_bytes"]
            + self.config["oob_bytes_per_page"])
        ecc_initiation = (
            wire_bytes
            / self.config["ecc_encode_raw_bandwidth_GBps_per_die"])
        ecc_issue = self.reserve(
            die.ecc_issue, sram.finish_ns, ecc_initiation)
        ecc_finish = (
            ecc_issue.start_ns + self.config["ecc_encode_latency_ns"])
        spans.append(Span(
            f"{source}/ecc_encode_issue",
            "ecc_issue",
            f"{address.die_path()}/ecc_issue_port",
            ecc_issue.start_ns,
            ecc_issue.finish_ns,
            critical=False,
        ))
        spans.append(Span(
            f"{source}/ecc_encode_latency",
            "ecc_latency",
            f"{address.die_path()}/ecc_encode_pipeline",
            ecc_issue.start_ns,
            ecc_finish,
        ))

        data_tsv = self.reserve(
            logic.tsv,
            ecc_finish,
            wire_bytes / self.config["tsv_bandwidth_GBps"],
        )
        spans.append(Span(
            f"{source}/data_in_tsv",
            "tsv",
            f"stack{address.stack}/tsv",
            data_tsv.start_ns,
            data_tsv.finish_ns,
        ))
        data_channel = self.reserve(
            channel.data,
            data_tsv.finish_ns,
            wire_bytes / self.config["channel_bandwidth_GBps"],
        )
        channel.data_count += 1
        spans.append(Span(
            f"{source}/data_in_channel",
            "flash_channel",
            address.channel_path(),
            data_channel.start_ns,
            data_channel.finish_ns,
        ))

        source_slot = self.reserve(
            source_timeline,
            data_channel.finish_ns,
            self.config["flash_tsu_issue_ns"],
        )
        sequencer = self.reserve(
            die.sequencer,
            source_slot.start_ns,
            self.config["flash_tsu_issue_ns"],
        )
        die.transactions += 1
        spans.append(Span(
            f"{source}/flash_scheduler_issue_program",
            "sequencer",
            address.die_path(),
            sequencer.start_ns,
            sequencer.finish_ns,
        ))

        barrier_ns = self.config["t_program_page_ns"]
        barrier_timelines = (
            plane.subarrays + plane.lanes + plane.page_buffers)
        media_start = self._common_gap_start(
            barrier_timelines, sequencer.finish_ns, barrier_ns)
        reservations = [
            timeline.reserve(media_start, barrier_ns)
            for timeline in barrier_timelines
        ]
        if any(item.start_ns != media_start for item in reservations):
            raise ValueError(
                "HBF oracle full-plane program calendars diverged")
        program_done = media_start + barrier_ns
        spans.append(Span(
            f"{source}/array_program",
            "flash_array",
            address.plane_path(),
            media_start,
            program_done,
        ))
        plane.media_intervals.append(
            Reservation(media_start, program_done))
        plane.program_count += 1

        self.counters["flash_scheduler_enqueues"] += 1
        self.counters["flash_scheduler_issues"] += 1
        self.counters["ecc_encode_ops"] += 1
        self.counters["ecc_encode_codeword_bytes"] += wire_bytes
        self.counters["ecc_issue_busy_ns"] += ecc_initiation
        return program_done

    def schedule_erase_block(
        self,
        block_index: int,
        earliest_ns: float,
        spans: list[Span],
        *, source: str,
    ) -> float:
        address = self.decode_ppn(
            block_index * self.config["pages_per_block"])
        logic = self.logic[address.stack]
        channel = self.channels[self.channel_index(address)]
        die = self.dies[self.die_index(address)]
        plane = self.planes[self.plane_index(address)]
        command_bytes = self.config["command_address_bytes"]

        command_tsv = self.reserve(
            logic.tsv,
            earliest_ns,
            command_bytes / self.config["tsv_bandwidth_GBps"],
        )
        spans.append(Span(
            f"{source}/cmd_addr_tsv",
            "tsv",
            f"stack{address.stack}/tsv",
            command_tsv.start_ns,
            command_tsv.finish_ns,
        ))
        command_channel = self.reserve(
            channel.command,
            command_tsv.finish_ns,
            command_bytes / self.config["channel_bandwidth_GBps"],
        )
        channel.command_count += 1
        spans.append(Span(
            f"{source}/cmd_addr_channel",
            "flash_channel",
            address.channel_path(),
            command_channel.start_ns,
            command_channel.finish_ns,
        ))
        source_slot = self.reserve(
            {"user":die.source_user,"mapping":die.source_mapping,"gc":die.source_gc}[source],
            command_channel.finish_ns,
            self.config["flash_tsu_issue_ns"],
        )
        sequencer = self.reserve(
            die.sequencer,
            source_slot.start_ns,
            self.config["flash_tsu_issue_ns"],
        )
        die.transactions += 1
        spans.append(Span(
            f"{source}/flash_scheduler_issue_erase",
            "sequencer",
            address.die_path(),
            sequencer.start_ns,
            sequencer.finish_ns,
        ))
        duration = self.config["t_erase_block_ns"]
        barrier_timelines = (
            plane.subarrays + plane.lanes + plane.page_buffers)
        media_start = self._common_gap_start(
            barrier_timelines, sequencer.finish_ns, duration)
        reservations = [
            timeline.reserve(media_start, duration)
            for timeline in barrier_timelines
        ]
        if any(item.start_ns != media_start for item in reservations):
            raise ValueError(
                "HBF oracle full-plane erase calendars diverged")
        erase_done = media_start + duration
        spans.append(Span(
            f"{source}/block_erase",
            "flash_array",
            address.plane_path(),
            media_start,
            erase_done,
        ))
        plane.media_intervals.append(
            Reservation(media_start, erase_done))
        plane.erase_count += 1
        self.blocks[block_index].erase_count += 1
        self.counters["block_erases"] += 1
        self.counters["flash_scheduler_enqueues"] += 1
        self.counters["flash_scheduler_issues"] += 1
        return erase_done

    def _plane_for_block(self, block_index: int) -> int:
        return block_index // self.config["blocks_per_plane"]

    def _stack_for_plane(self, plane: int) -> int:
        return plane // self.planes_per_stack()

    def _gc_headroom(
        self,
        stack: int,
        allocation_role: str,
    ) -> tuple[int, int, int]:
        relocation_pages = 0
        foreground_pages = 0
        reserve_target = 0
        first_plane = stack * self.planes_per_stack()
        for plane in range(
            first_plane, first_plane + self.planes_per_stack()
        ):
            active_gc = self.active_blocks.get(("gc", plane))
            if active_gc is not None:
                relocation_pages += self.blocks[active_gc].free_pages
            active_role = self.active_blocks.get((allocation_role, plane))
            if active_role is not None:
                foreground_pages += self.blocks[active_role].free_pages
            free_blocks = len(self.free_blocks_by_plane[plane])
            relocation_pages += (
                free_blocks * self.config["pages_per_block"])
            allocatable_blocks = max(
                0,
                free_blocks
                - self.config["gc_reserved_free_blocks_per_plane"],
            )
            foreground_pages += (
                allocatable_blocks * self.config["pages_per_block"])
            reserve_target += (
                self.config["gc_reserved_free_blocks_per_plane"]
                * self.config["pages_per_block"])
        return relocation_pages, foreground_pages, reserve_target

    def _can_allocate_on_plane(self, plane: int, role: str) -> bool:
        active = self.active_blocks.get((role, plane))
        if active is not None and self.blocks[active].free_pages > 0:
            return True
        free_blocks = len(self.free_blocks_by_plane[plane])
        return (
            free_blocks > 0
            and (
                role == "gc"
                or free_blocks
                > self.config["gc_reserved_free_blocks_per_plane"]
            )
        )

    def _preview_allocation_plane(
        self,
        stack: int,
        role: str,
        preferred_plane: int | None,
    ) -> int | None:
        if (
            preferred_plane is not None
            and self._can_allocate_on_plane(preferred_plane, role)
        ):
            return preferred_plane
        first = self.allocation_cursors[role][stack]
        base = stack * self.planes_per_stack()
        for offset in range(self.planes_per_stack()):
            plane = base + (first + offset) % self.planes_per_stack()
            if self._can_allocate_on_plane(plane, role):
                return plane
        return None

    @staticmethod
    def _has_headroom(
        available: int,
        required: int,
        watermark: int,
    ) -> bool:
        return (
            available > required
            and available - required > watermark
        )

    def _gc_pressure(
        self,
        stack: int,
        allocation_role: str,
        preferred_plane: int | None,
    ) -> tuple[bool, bool]:
        relocation, foreground, reserve_target = self._gc_headroom(stack, allocation_role)
        active_pages = sum(self.blocks[b].free_pages
            for (role, plane), b in self.active_blocks.items()
            if role == allocation_role and self._stack_for_plane(plane) == stack)
        preventive = active_pages + max(0, relocation - reserve_target)
        hard_ok = foreground >= 1 + self.config["gc_hard_watermark_pages"]
        soft_ok = preventive > 1 + (self.config["gc_low_watermark_pages"] or self.config["pages_per_block"])
        return (True if allocation_role == "mapping" and hard_ok else soft_ok), hard_ok

    def choose_gc_victim(self, stack: int) -> int | None:
        relocation, _, _ = self._gc_headroom(stack, "data")
        begin = (
            stack
            * self.planes_per_stack()
            * self.config["blocks_per_plane"])
        end = (
            (stack + 1)
            * self.planes_per_stack()
            * self.config["blocks_per_plane"])
        active = set(self.active_blocks.values())
        best: int | None = None
        best_score = float("-inf")
        best_valid = 1 << 32
        best_erase = 1 << 32
        for index in range(begin, end):
            block = self.blocks[index]
            if (
                block.role in {"free", "static_read_only", "raw_physical"}
                or block.erase_pending
                or block.pending_program_pages
                or block.pending_mapping_publications
                or block.free_pages != 0
                or block.invalid_pages == 0
                or index in active
                or block.valid_pages > relocation
            ):
                continue
            used = block.valid_pages + block.invalid_pages
            invalid_ratio = block.invalid_pages / used if used else 0.0
            score = (
                invalid_ratio * 1000.0
                + block.invalid_pages * 10.0
                - block.valid_pages
                - block.erase_count
                * self.config["gc_wear_leveling_weight"]
            )
            if (
                score > best_score
                or (
                    score == best_score
                    and block.valid_pages < best_valid
                )
                or (
                    score == best_score
                    and block.valid_pages == best_valid
                    and block.erase_count < best_erase
                )
            ):
                best = index
                best_score = score
                best_valid = block.valid_pages
                best_erase = block.erase_count
        return best

    def release_reclaimed_block(self, block_index: int) -> None:
        block = self.blocks[block_index]
        old_epoch = block.epoch
        old_erase_count = block.erase_count
        begin = block_index * self.config["pages_per_block"]
        for ppn in range(begin, begin + self.config["pages_per_block"]):
            self.pages.pop(ppn, None)
        self.counters["free_pages"] += (
            self.config["pages_per_block"] - block.free_pages)
        self.blocks[block_index] = Block(
            role="free",
            free_pages=self.config["pages_per_block"],
            erase_count=old_erase_count,
            epoch=old_epoch + 1,
        )
        for key, active in list(self.active_blocks.items()):
            if active == block_index:
                del self.active_blocks[key]
        self.free_blocks_by_plane[
            self._plane_for_block(block_index)
        ].append(block_index)

    def relocate_and_reclaim_block(
        self,
        victim: int,
        at_ns: float,
        spans: list[Span],
    ) -> float:
        victim_block = self.blocks[victim]
        original_invalid = victim_block.invalid_pages
        select_done = at_ns + self.config["host_gc_decision_ns"]
        spans.append(Span(
            "gc_victim_select",
            "maintenance",
            "host/gc",
            at_ns,
            select_done,
        ))
        at_ns = select_done
        begin = victim * self.config["pages_per_block"]
        valid_pages = [
            (ppn, self.pages[ppn].logical_key)
            for ppn in range(
                begin, begin + self.config["pages_per_block"])
            if ppn in self.pages and self.pages[ppn].status == "valid"
        ]
        stack = self._stack_for_plane(self._plane_for_block(victim))
        for old_ppn, logical_key in valid_pages:
            read_done = self.schedule_read_page(
                old_ppn,
                at_ns,
                0,
                spans,
                internal=True,
                source="gc",
            )
            allocation_done = (
                read_done + self.config["free_page_allocation_ns"])
            spans.append(Span(
                "free_page_alloc",
                "translation",
                "logic/free_page_allocator",
                read_done,
                allocation_done,
            ))
            new_ppn = self.allocate_page(
                "gc",
                stack,
                self._plane_for_block(victim),
            )
            program_done = self.schedule_program_page(
                new_ppn, allocation_done, "gc", spans)
            metadata = bool(logical_key & (1 << 63))
            if metadata:
                mapping_done = (
                    program_done + self.config["mapping_update_ns"])
                spans.append(Span(
                    "gc_mapping_checkpoint_relocate",
                    "translation",
                    "logic/mapping_table",
                    program_done,
                    mapping_done,
                ))
            else:
                mapping_done = self.access_mapping(
                    logical_key,
                    program_done,
                    "update",
                    spans,
                    source="gc",
                )
            self.invalidate(old_ppn)
            if metadata:
                vpn = logical_key & ~(1 << 63)
                self.mapping_vpn_to_ppn[vpn] = new_ppn
                owner = "mapping"
            else:
                self.lpn_to_ppn[logical_key] = new_ppn
                self.lpn_ready_ns[logical_key] = mapping_done
                vpn = self.mapping_vpn(logical_key)
                self.mark_mapping_dirty(vpn, mapping_done)
                owner = "logical"
            self.mark_programmed(new_ppn, logical_key, owner)
            self.counters["physical_read_bytes"] += (
                self.config["page_size_bytes"])
            self.counters["physical_write_bytes"] += (
                self.config["page_size_bytes"])
            self.counters["gc_relocation_payload_bytes"] += (
                self.config["page_size_bytes"])
            self.counters["page_reads"] += 1
            self.counters["page_programs"] += 1
            self.counters["gc_relocations"] += 1
            self.counters[
                (
                    "gc_mapping_relocations"
                    if metadata
                    else "gc_data_relocations"
                )
            ] += 1
            at_ns = max(at_ns, mapping_done)

        self.release_reclaimed_block(victim)
        self.counters["gc_runs"] += 1
        self.counters["gc_reclaimed_invalid_pages"] += original_invalid
        return at_ns

    def maybe_run_gc(
        self,
        at_ns: float,
        stack: int,
        allocation_role: str,
        preferred_plane: int | None,
        spans: list[Span],
    ) -> float:
        if not self.config["auto_gc_enabled"]:
            return at_ns
        progress = 0
        while True:
            soft_ok, hard_ok = self._gc_pressure(
                stack, allocation_role, preferred_plane)
            if soft_ok and hard_ok:
                return at_ns
            victim = self.choose_gc_victim(stack)
            if victim is None:
                return at_ns
            invalid = self.blocks[victim].invalid_pages
            begin_ns = at_ns
            reclaim_done = self.relocate_and_reclaim_block(victim, at_ns, spans)
            self.background_finish_ns = max(self.background_finish_ns, reclaim_done)
            if hard_ok:
                return at_ns
            at_ns = reclaim_done
            if not hard_ok:
                spans.append(Span(
                    "gc/foreground_block",
                    "maintenance",
                    "host/gc",
                    begin_ns,
                    at_ns,
                ))
                self.counters["gc_user_blocked_runs"] += 1
            progress += invalid
            if progress > self.total_pages:
                raise ValueError("HBF oracle GC failed to make progress")

    def cache_event(self, ppn: int, at: float, kind: int) -> bool:
        bank = ppn // (self.config["pages_per_block"] * self.config["blocks_per_plane"])
        history = self.cache_events.setdefault(bank, [])
        def replay(events, end=float("inf")):
            resident = []
            for when, sequence, page, action in sorted(events):
                if when > end:
                    break
                if action == 0 and page not in resident:
                    return resident, False
                if page in resident:
                    resident.remove(page)
                if action >= 0:
                    resident.append(page)
                resident = resident[-2:]
            return resident, True
        if kind == 0 and ppn not in replay(history, at)[0]:
            return False
        event = (at, self.cache_sequence, ppn, kind)
        self.cache_sequence += 1
        if not replay(history + [event])[1]:
            return False
        history.append(event)
        return True

    def cached_read(self, ppn: int, bytes_: int, ready: float, spans: list[Span]) -> float:
        address = self.decode_ppn(ppn)
        ready = self.external_command(ppn, ready, "user", spans)
        stage = self.logic[address.stack].sram.reserve(ready,
            bytes_ / self.config["logic_sram_bandwidth_GBps"])
        spans.append(Span("read_buffer_hit", "sram", f"stack{address.stack}/logic",
            stage.start_ns, stage.finish_ns))
        response = self.channels[self.channel_index(address)].external_tx.reserve(
            stage.finish_ns, bytes_ / self.channel_payload_bw)
        spans.append(Span("read_buffer_hit_hbio_out", "hbio", address.channel_path()+"/hbio/tx",
            response.start_ns, response.finish_ns))
        return response.finish_ns

    def issue_read(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["bytes"] > self.config["page_size_bytes"]:
            raise ValueError(
                "HBF oracle v1 read cases must be at most one page")
        physical = request["address_space"] == "physical"
        if physical:
            target = self.decode(request["addr"])
            ppn = self.encode_ppn(target)
            request_stack = target.stack
            range_offset = target.offset
        else:
            lpn, range_offset = divmod(
                request["addr"], self.config["page_size_bytes"])
            if range_offset + request["bytes"] > self.config["page_size_bytes"]:
                raise ValueError(
                    "HBF oracle v1 excludes cross-page reads")
            if lpn not in self.lpn_to_ppn:
                raise ValueError(
                    "HBF oracle v1 requires mapped logical reads")
            ppn = self.lpn_to_ppn[lpn]
            request_stack = self.stack_for_lpn(lpn)
        logic = self.logic[request_stack]
        spans: list[Span] = []
        ingress_ready = request["arrival_ns"]
        if not physical:
            ingress_ready = max(
                ingress_ready,
                self.materialized_lpn_ready_ns.get(lpn, 0.0),
            )
        logic_issue = self.reserve(
            logic.ingress,
            ingress_ready,
            self.config["logic_scheduler_issue_ns"],
        )
        spans.append(Span(
            "logic_scheduler_issue",
            "logic",
            f"stack{request_stack}/logic",
            logic_issue.start_ns,
            logic_issue.finish_ns,
        ))
        ready = logic_issue.finish_ns
        if not physical:
            prior_lpn_ready = self.lpn_ready_ns.get(lpn, 0.0)
            ready = max(ready, prior_lpn_ready)
            if prior_lpn_ready:
                self.materialized_lpn_ready_ns[lpn] = max(
                    self.materialized_lpn_ready_ns.get(lpn, 0.0),
                    prior_lpn_ready,
                )
            ready = self.access_mapping(lpn, ready, "lookup", spans)

        wire_bytes = ((range_offset + request["bytes"] + 63) // 64 - range_offset // 64) * 64
        hit = self.cache_event(ppn, ready, 0)
        if hit:
            finish = self.cached_read(ppn, wire_bytes, ready, spans)
        else:
            finish = self.schedule_read_page(ppn, ready, wire_bytes, spans)
            self.cache_event(ppn, self.last_decoded_ready, 1)
            self.counters["physical_read_bytes"] += self.config["page_size_bytes"]
            self.counters["page_reads"] += 1
        self.counters["page_read_admission_events"] += 1
        self.counters["read_requests"] += 1
        if not physical:
            self.counters["logical_read_bytes"] += request["bytes"]
        self.counters["finish_ns"] = max(
            float(self.counters["finish_ns"]), finish)
        spans.sort(key=lambda span: (
            span.start_ns,
            span.finish_ns,
            span.category,
            span.name,
            span.entity,
        ))
        path = (
            self.decode(request["addr"]).path()
            if physical
            else f"lpn{lpn}->{self.decode_ppn(ppn).path()}"
        )
        return {
            "start_ns": logic_issue.start_ns,
            "finish_ns": finish,
            "logical_bytes": request["bytes"],
            "physical_bytes": 0 if hit else self.config["page_size_bytes"],
            "resource": path,
            "result": "read-buffer-hit" if hit else "mapped-page-read",
            "spans": spans,
        }

    @staticmethod
    def _merge_dirty_range(
        ranges: list[tuple[int, int]],
        begin: int,
        end: int,
    ) -> int:
        old_covered = sum(hi - lo for lo, hi in ranges)
        merged: list[tuple[int, int]] = []
        next_begin = begin
        next_end = end
        inserted = False
        for lo, hi in ranges:
            if hi < next_begin:
                merged.append((lo, hi))
            elif next_end < lo:
                if not inserted:
                    merged.append((next_begin, next_end))
                    inserted = True
                merged.append((lo, hi))
            else:
                next_begin = min(next_begin, lo)
                next_end = max(next_end, hi)
        if not inserted:
            merged.append((next_begin, next_end))
        ranges[:] = merged
        new_covered = sum(hi - lo for lo, hi in ranges)
        return old_covered + (end - begin) - new_covered

    @staticmethod
    def _covered_bytes(entry: WriteBufferEntry) -> int:
        return sum(hi - lo for lo, hi in entry.ranges)

    def issue_write(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["address_space"] != "logical":
            raise ValueError(
                "HBF oracle v1 write cases require logical addresses")
        lpn, offset = divmod(
            request["addr"], self.config["page_size_bytes"])
        if (
            request["bytes"] > self.config["page_size_bytes"]
            or offset + request["bytes"] > self.config["page_size_bytes"]
        ):
            raise ValueError(
                "HBF oracle v1 excludes cross-page writes")
        stack = self.stack_for_lpn(lpn)
        logic = self.logic[stack]
        spans: list[Span] = []
        physical_before = (
            int(self.counters["physical_read_bytes"])
            + int(self.counters["physical_write_bytes"])
        )
        ingress_ready = max(
            request["arrival_ns"],
            self.materialized_lpn_ready_ns.get(lpn, 0.0),
        )
        logic_issue = self.reserve(
            logic.ingress,
            ingress_ready,
            self.config["logic_scheduler_issue_ns"],
        )
        spans.append(Span(
            "logic_scheduler_issue",
            "logic",
            f"stack{stack}/logic",
            logic_issue.start_ns,
            logic_issue.finish_ns,
        ))

        prior_lpn_ready = self.lpn_ready_ns.get(lpn, 0.0)
        ready = max(logic_issue.finish_ns, prior_lpn_ready)
        if prior_lpn_ready:
            self.materialized_lpn_ready_ns[lpn] = max(
                self.materialized_lpn_ready_ns.get(lpn, 0.0),
                prior_lpn_ready,
            )
        ready = self.access_mapping(lpn, ready, "lookup", spans)
        old_ppn = self.lpn_to_ppn.get(lpn)
        full_page = (
            offset == 0 and request["bytes"] == self.config["page_size_bytes"])
        if self.config["write_coalescing_enabled"]:
            buffer = self.write_buffers[stack]
            entry = buffer.get(lpn)
            if entry is None:
                if len(buffer) >= self.config["write_buffer_pages"]:
                    raise ValueError(
                        "HBF oracle v1 excludes write-buffer eviction")
                entry = WriteBufferEntry()
                buffer[lpn] = entry
                self.counters["write_buffer_misses"] += 1
            else:
                self.counters["write_buffer_hits"] += 1
            overlap = self._merge_dirty_range(
                entry.ranges, offset, offset + request["bytes"])
            self.counters["write_buffer_merged_bytes"] += overlap

            stage_done = self.access_write_buffer_dram(
                stack,
                ready,
                request["bytes"],
                "write_buffer_stage_dram",
                spans,
            )
            entry.ready_ns = max(entry.ready_ns, stage_done)
            self.counters["program_requests"] += 1
            self.counters["logical_write_bytes"] += request["bytes"]
            self.counters["finish_ns"] = max(
                float(self.counters["finish_ns"]), stage_done)
            spans.sort(key=lambda span: (
                span.start_ns,
                span.finish_ns,
                span.category,
                span.name,
                span.entity,
            ))
            return {
                "start_ns": logic_issue.start_ns,
                "finish_ns": stage_done,
                "logical_bytes": request["bytes"],
                "physical_bytes": 0,
                "resource": f"lpn{lpn}->write_buffer",
                "result": "write-buffer-stage",
                "spans": spans,
            }
        if not full_page:
            if old_ppn is not None:
                merge_finish = ready + self.config["mapping_update_ns"]
                spans.append(Span(
                    "partial_page_merge",
                    "translation",
                    f"stack{stack}/logic",
                    ready,
                    merge_finish,
                ))
                ready = self.schedule_read_page(
                    old_ppn,
                    merge_finish,
                    0,
                    spans,
                    internal=True,
                )
                self.counters["physical_read_bytes"] += (
                    self.config["page_size_bytes"])
                self.counters["page_reads"] += 1

            ready = self.host_memory_transfer(stack,
                request["bytes"] if old_ppn is not None else self.config["page_size_bytes"],
                True, ready, spans)
            ready = self.host_memory_transfer(stack, self.config["page_size_bytes"], False, ready, spans)

        allocation_ready = self.maybe_run_gc(
            ready,
            stack,
            "data",
            None,
            spans,
        )
        allocation_finish = (
            allocation_ready
            + self.config["free_page_allocation_ns"])
        spans.append(Span(
            "free_page_alloc",
            "translation",
            "logic/free_page_allocator",
            allocation_ready,
            allocation_finish,
        ))
        new_ppn = self.allocate_page("data", stack)
        program_done = self.schedule_program_page(
            new_ppn, allocation_finish, "user", spans)
        self.mark_programmed(new_ppn, lpn, "logical")
        mapping_done = self.access_mapping(
            lpn, program_done, "update", spans)
        if old_ppn is not None and old_ppn != new_ppn:
            self.invalidate(old_ppn)
        self.lpn_to_ppn[lpn] = new_ppn
        self.lpn_ready_ns[lpn] = mapping_done
        vpn = self.mapping_vpn(lpn)
        self.mark_mapping_dirty(vpn, mapping_done)

        self.counters["program_requests"] += 1
        self.counters["logical_write_bytes"] += request["bytes"]
        self.counters["physical_write_bytes"] += self.config["page_size_bytes"]
        self.counters["data_program_payload_bytes"] += (
            self.config["page_size_bytes"])
        self.counters["data_programs"] += 1
        self.counters["page_programs"] += 1
        self.counters["finish_ns"] = max(
            float(self.counters["finish_ns"]), mapping_done)
        spans.sort(key=lambda span: (
            span.start_ns,
            span.finish_ns,
            span.category,
            span.name,
            span.entity,
        ))
        return {
            "start_ns": logic_issue.start_ns,
            "finish_ns": mapping_done,
            "logical_bytes": request["bytes"],
            "physical_bytes": (
                int(self.counters["physical_read_bytes"])
                + int(self.counters["physical_write_bytes"])
                - physical_before
            ),
            "resource": f"lpn{lpn}->{self.decode_ppn(new_ppn).path()}",
            "result": "page-program-map-update",
            "spans": spans,
        }

    def audit_attributes(self) -> dict[str, Any]:
        free_pages_per_stack = [0] * self.config["stacks"]
        blocks_per_stack = (
            self.config["channels_per_stack"]
            * self.config["dies_per_channel"]
            * self.config["planes_per_die"]
            * self.config["blocks_per_plane"]
        )
        for index, block in enumerate(self.blocks):
            free_pages_per_stack[index // blocks_per_stack] += block.free_pages
        write_buffer_entries = sum(
            len(buffer) for buffer in self.write_buffers)
        return {
            "quiescent": (
                not self.dirty_mapping_events
                and write_buffer_entries == 0
            ),
            "free_pages": int(self.counters["free_pages"]),
            "free_pages_per_stack": free_pages_per_stack,
            "allocation_cursors": {
                "data": list(self.allocation_cursors["data"]),
                "mapping": list(self.allocation_cursors["mapping"]),
                "gc": list(self.allocation_cursors["gc"]),
            },
            "logical_mappings": [
                {"lpn": lpn, "ppn": ppn}
                for lpn, ppn in sorted(self.lpn_to_ppn.items())
            ],
            "mapping_pages": [
                {"vpn": vpn, "ppn": ppn}
                for vpn, ppn in sorted(self.mapping_vpn_to_ppn.items())
            ],
            "pages": [
                {
                    "ppn": ppn,
                    "status": page.status,
                    "owner": page.owner,
                    "logical_key": page.logical_key,
                    "block_epoch": page.block_epoch,
                }
                for ppn, page in sorted(self.pages.items())
                if page.status == "valid"
            ],
            "blocks": [
                {
                    "block": index,
                    "role": block.role,
                    "valid_pages": block.valid_pages,
                    "invalid_pages": block.invalid_pages,
                    "free_pages": block.free_pages,
                    "next_page": block.next_page,
                    "erase_count": block.erase_count,
                    "pending_program_pages": block.pending_program_pages,
                    "pending_mapping_publications":
                        block.pending_mapping_publications,
                    "epoch": block.epoch,
                    "erase_pending": block.erase_pending,
                }
                for index, block in enumerate(self.blocks)
            ],
            "dirty_mapping_vpns": sorted(self.dirty_mapping_events),
            "pending": {
                "dirty_mapping_events": sum(
                    len(events)
                    for events in self.dirty_mapping_events.values()
                ),
                "lpn_updates": 0,
                "vpn_updates": 0,
                "commits": 0,
                "write_buffer_entries": write_buffer_entries,
                "inflight_buffered_generations": 0,
                "physical_programs": 0,
                "block_transitions": 0,
            },
        }

    @staticmethod
    def canonical_audit_state(attributes: dict[str, Any]) -> str:
        lines = [
            "hbfsim.hbf.audit.v1",
            f"quiescent\t{int(attributes['quiescent'])}",
            f"free_pages\t{attributes['free_pages']}",
        ]

        def vector(name: str, values: list[int]) -> None:
            suffix = "".join(f"\t{value}" for value in values)
            lines.append(f"{name}{suffix}")

        vector(
            "free_pages_per_stack",
            attributes["free_pages_per_stack"],
        )
        vector(
            "data_allocation_cursors",
            attributes["allocation_cursors"]["data"],
        )
        vector(
            "mapping_allocation_cursors",
            attributes["allocation_cursors"]["mapping"],
        )
        vector(
            "gc_allocation_cursors",
            attributes["allocation_cursors"]["gc"],
        )
        vector(
            "dirty_mapping_vpns",
            attributes["dirty_mapping_vpns"],
        )
        pending = attributes["pending"]
        lines.append(
            "pending"
            f"\t{pending['dirty_mapping_events']}"
            f"\t{pending['lpn_updates']}"
            f"\t{pending['vpn_updates']}"
            f"\t{pending['commits']}"
            f"\t{pending['write_buffer_entries']}"
            f"\t{pending['inflight_buffered_generations']}"
            f"\t{pending['physical_programs']}"
            f"\t{pending['block_transitions']}"
        )
        for mapping in attributes["logical_mappings"]:
            lines.append(f"l2p\t{mapping['lpn']}\t{mapping['ppn']}")
        for mapping in attributes["mapping_pages"]:
            lines.append(f"vpn\t{mapping['vpn']}\t{mapping['ppn']}")
        for page in attributes["pages"]:
            lines.append(
                "page"
                f"\t{page['ppn']}"
                f"\t{page['status']}"
                f"\t{page['owner']}"
                f"\t{page['logical_key']}"
                f"\t{page['block_epoch']}"
            )
        for block in attributes["blocks"]:
            lines.append(
                "block"
                f"\t{block['block']}"
                f"\t{block['role']}"
                f"\t{block['valid_pages']}"
                f"\t{block['invalid_pages']}"
                f"\t{block['free_pages']}"
                f"\t{block['next_page']}"
                f"\t{block['erase_count']}"
                f"\t{block['pending_program_pages']}"
                f"\t{block['pending_mapping_publications']}"
                f"\t{block['epoch']}"
                f"\t{int(block['erase_pending'])}"
            )
        return "\n".join(lines) + "\n"

    def drain(self, request: dict[str, Any]) -> dict[str, Any]:
        spans: list[Span] = []
        initial_drain_start = max(
            request["arrival_ns"], self.background_finish_ns)
        finish = initial_drain_start
        physical_before = (
            int(self.counters["physical_read_bytes"])
            + int(self.counters["physical_write_bytes"])
        )
        for stack, buffer in enumerate(self.write_buffers):
            for lpn in sorted(list(buffer)):
                entry = buffer[lpn]
                entry_ns = max(initial_drain_start, entry.ready_ns)
                logic = self.logic[stack]
                entry_ns = self.access_write_buffer_dram(
                    stack,
                    entry_ns,
                    self._covered_bytes(entry),
                    "write_buffer_flush_dram",
                    spans,
                )
                current_ppn = self.lpn_to_ppn.get(lpn)
                covered = self._covered_bytes(entry)
                if (
                    current_ppn is not None
                    and covered != self.config["page_size_bytes"]
                ):
                    merge_finish = (
                        entry_ns + self.config["mapping_update_ns"])
                    spans.append(Span(
                        "partial_page_merge_on_flush",
                        "translation",
                        f"stack{stack}/logic",
                        entry_ns,
                        merge_finish,
                    ))
                    entry_ns = self.schedule_read_page(
                        current_ppn,
                        merge_finish,
                        0,
                        spans,
                        internal=True,
                    )
                    self.counters["physical_read_bytes"] += (
                        self.config["page_size_bytes"])
                    self.counters["page_reads"] += 1

                if covered != self.config["page_size_bytes"]:
                    entry_ns = self.host_memory_transfer(
                        stack,
                        covered if current_ppn is not None else self.config["page_size_bytes"],
                        True, entry_ns, spans,
                    )
                    entry_ns = self.host_memory_transfer(
                        stack, self.config["page_size_bytes"], False, entry_ns, spans,
                    )

                entry_ns = self.maybe_run_gc(
                    entry_ns,
                    stack,
                    "data",
                    None,
                    spans,
                )
                allocation_finish = (
                    entry_ns + self.config["free_page_allocation_ns"])
                spans.append(Span(
                    "free_page_alloc",
                    "translation",
                    "logic/free_page_allocator",
                    entry_ns,
                    allocation_finish,
                ))
                new_ppn = self.allocate_page("data", stack)
                program_done = self.schedule_program_page(
                    new_ppn, allocation_finish, "user", spans)
                self.mark_programmed(new_ppn, lpn, "logical")
                mapping_done = self.access_mapping(
                    lpn, program_done, "update", spans)
                if current_ppn is not None and current_ppn != new_ppn:
                    self.invalidate(current_ppn)
                self.lpn_to_ppn[lpn] = new_ppn
                self.lpn_ready_ns[lpn] = mapping_done
                vpn = self.mapping_vpn(lpn)
                self.mark_mapping_dirty(vpn, mapping_done)
                del buffer[lpn]
                self.counters["physical_write_bytes"] += (
                    self.config["page_size_bytes"])
                self.counters["data_program_payload_bytes"] += (
                    self.config["page_size_bytes"])
                self.counters["data_programs"] += 1
                self.counters["page_programs"] += 1
                self.counters["write_buffer_flushes"] += 1
                finish = max(finish, mapping_done)

        wave = 0
        while self.dirty_mapping_events:
            dirty_vpns = sorted(self.dirty_mapping_events)
            wave_start = finish
            wave_finish = finish
            for vpn in dirty_vpns:
                events = self.dirty_mapping_events.get(vpn)
                if not events:
                    self.dirty_mapping_events.pop(vpn, None)
                    continue
                entry_ns = max(wave_start, events[-1][0])
                eligible = [
                    event for event in events
                    if event[0] <= entry_ns
                ]
                if not eligible:
                    continue
                snapshot = eligible[-1]
                preferred_plane = self.mapping_plane(vpn)
                entry_ns = self.maybe_run_gc(
                    entry_ns,
                    vpn % self.config["stacks"],
                    "mapping",
                    preferred_plane,
                    spans,
                )
                allocation_finish = (
                    entry_ns + self.config["free_page_allocation_ns"])
                spans.append(Span(
                    "free_page_alloc",
                    "translation",
                    "logic/free_page_allocator",
                    entry_ns,
                    allocation_finish,
                ))
                new_ppn = self.allocate_page(
                    "mapping",
                    vpn % self.config["stacks"],
                    preferred_plane,
                )
                program_done = self.schedule_program_page(
                    new_ppn, allocation_finish, "mapping", spans)
                self.mark_programmed(
                    new_ppn, (1 << 63) | vpn, "mapping")
                old_ppn = self.mapping_vpn_to_ppn.get(vpn)
                if old_ppn is not None and old_ppn != new_ppn:
                    self.invalidate(old_ppn)
                self.mapping_vpn_to_ppn[vpn] = new_ppn
                remaining = [
                    event
                    for event in self.dirty_mapping_events.get(vpn, [])
                    if event > snapshot
                ]
                if remaining:
                    self.dirty_mapping_events[vpn] = remaining
                else:
                    self.dirty_mapping_events.pop(vpn, None)
                self.counters["physical_write_bytes"] += (
                    self.config["page_size_bytes"])
                self.counters["mapping_program_payload_bytes"] += (
                    self.config["page_size_bytes"])
                self.counters["page_programs"] += 1
                self.counters["mapping_page_programs"] += 1
                wave_finish = max(wave_finish, program_done)
            finish = wave_finish
            wave += 1
            if wave > len(self.blocks) + 1:
                raise ValueError(
                    "HBF oracle mapping drain did not converge")
        self.counters["finish_ns"] = max(
            float(self.counters["finish_ns"]), finish)
        spans.sort(key=lambda span: (
            span.start_ns,
            span.finish_ns,
            span.category,
            span.name,
            span.entity,
        ))
        attributes = self.audit_attributes()
        if not attributes["quiescent"]:
            raise ValueError("HBF oracle drain did not reach quiescence")
        canonical_state = self.canonical_audit_state(attributes)
        return {
            "start_ns": request["arrival_ns"],
            "finish_ns": finish,
            "logical_bytes": 0,
            "physical_bytes": (
                int(self.counters["physical_read_bytes"])
                + int(self.counters["physical_write_bytes"])
                - physical_before
            ),
            "resource": "logic/write_buffer+mapping_table",
            "result": (
                "drained-pending-hbf-state"
                if finish > request["arrival_ns"]
                else "no-pending-hbf-state"
            ),
            "spans": spans,
            "state": {
                "attributes": attributes,
                "state_hash": (
                    "sha256:"
                    + hashlib.sha256(
                        canonical_state.encode("utf-8")
                    ).hexdigest()
                ),
            },
        }

    def issue(self, request: dict[str, Any]) -> dict[str, Any]:
        # Every issue and drain is a causal barrier: nothing is scheduled
        # behind its arrival and calendar history before it is reclaimed.
        self.causal_watermark_ns = max(
            self.causal_watermark_ns, float(request["arrival_ns"]))
        if request["op"] == "read":
            return self.issue_read(request)
        if request["op"] == "write":
            return self.issue_write(request)
        if request["op"] == "drain":
            return self.drain(request)
        raise ValueError(f"unsupported HBF oracle action: {request['op']}")

    @staticmethod
    def _union_length(intervals: list[Reservation]) -> float:
        if not intervals:
            return 0.0
        ordered = sorted(intervals, key=lambda interval: interval.start_ns)
        begin = ordered[0].start_ns
        end = ordered[0].finish_ns
        total = 0.0
        for interval in ordered[1:]:
            if interval.start_ns <= end:
                end = max(end, interval.finish_ns)
            else:
                total += end - begin
                begin, end = interval.start_ns, interval.finish_ns
        return total + end - begin

    def finalize(self) -> None:
        self.counters["mapping_dram_issue_busy_ns"] = sum(
            logic.mapping_issue.reserved_work_ns for logic in self.logic)
        self.counters["channel_command_busy_ns"] = sum(
            channel.command.reserved_work_ns for channel in self.channels)
        self.counters["channel_data_busy_ns"] = sum(
            channel.data.reserved_work_ns for channel in self.channels)
        self.counters["tsv_busy_ns"] = sum(
            logic.tsv.reserved_work_ns for logic in self.logic)
        self.counters["sram_busy_ns"] = sum(
            logic.sram.reserved_work_ns for logic in self.logic)
        self.counters["hb_io_command_busy_ns"] = sum(
            channel.external_command.reserved_work_ns for channel in self.channels)
        self.counters["hb_io_data_busy_ns"] = sum(
            channel.external_rx.reserved_work_ns+channel.external_tx.reserved_work_ns for channel in self.channels)
        self.counters["media_busy_ns"] = sum(
            self._union_length(plane.media_intervals)
            for plane in self.planes
        )
        self.counters["active_planes"] = sum(
            bool(
                plane.read_count
                or plane.program_count
                or plane.erase_count
            )
            for plane in self.planes
        )
        self.counters["active_media_lanes"] = sum(
            count > 0
            for plane in self.planes
            for count in plane.lane_reads
        )
        self.counters["active_subarrays"] = sum(
            count > 0
            for plane in self.planes
            for count in plane.subarray_reads
        )
        self.counters["active_page_buffer_banks"] = sum(
            count > 0
            for plane in self.planes
            for count in plane.page_buffer_reads
        )
        self.counters["active_channels"] = sum(
            bool(channel.command_count or channel.data_count)
            for channel in self.channels
        )
        self.counters["active_dies"] = sum(
            die.transactions > 0 for die in self.dies)
        self.counters["free_pages"] = sum(
            block.free_pages for block in self.blocks)
        self.counters["valid_pages"] = sum(
            block.valid_pages for block in self.blocks)
        self.counters["invalid_pages"] = sum(
            block.invalid_pages for block in self.blocks)
        self.counters["pending_program_pages"] = sum(
            block.pending_program_pages for block in self.blocks)
        self.counters["pending_mapping_publications"] = sum(
            block.pending_mapping_publications for block in self.blocks)


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
        model="hbf",
        producer="oracle",
    )
    for request in case["requests"]:
        request_record_id = f"request/{request['id']}"
        append(
            "request",
            request_record_id,
            None,
            request_id=request["id"],
            model="hbf",
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
                model="hbf",
                resource=span.entity,
                category=span.category,
                action=span.name,
                start_ns=span.start_ns,
                finish_ns=span.finish_ns,
                critical=span.critical,
                **({"physical_bytes": span.physical_bytes} if span.physical_bytes else {}),
            )
        if "state" in completion:
            append(
                "state",
                f"{request_record_id}/state/quiescent",
                request_record_id,
                request_id=request["id"],
                model="hbf",
                resource="hbf/ftl",
                action="quiescent_checkpoint",
                time_ns=completion["finish_ns"],
                attributes=completion["state"]["attributes"],
                state_hash=completion["state"]["state_hash"],
            )
        append(
            "completion",
            f"{request_record_id}/completion",
            request_record_id,
            request_id=request["id"],
            model="hbf",
            action="complete",
            arrival_ns=request["arrival_ns"],
            start_ns=completion["start_ns"],
            finish_ns=completion["finish_ns"],
            logical_bytes=completion["logical_bytes"],
            physical_bytes=completion["physical_bytes"],
            resource=completion["resource"],
            result=completion["result"],
        )
    oracle.finalize()
    observations = []
    for address in case["inspect_addresses"]:
        decoded = oracle.decode(address)
        observations.append({
            "address": address,
            "stack": decoded.stack,
            "channel": decoded.channel,
            "die": decoded.die,
            "plane": decoded.plane,
            "block": decoded.block,
            "page": decoded.page,
            "offset": decoded.offset,
            "round_trip_address": oracle.encode(decoded),
        })
    append(
        "summary",
        "summary",
        None,
        model="hbf",
        action="final",
        config=oracle.derived_config(),
        address_observations=observations,
        counters=oracle.counters,
    )
    return records
