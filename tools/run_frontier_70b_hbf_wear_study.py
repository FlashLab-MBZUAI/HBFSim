#!/usr/bin/env python3
"""Run the canonical Frontier 70B HBF write/wear sensitivity study.

The study deliberately separates three contracts:

1. The audited Frontier ledger determines *logical* mutable-KV write demand.
2. Digest-bound placement receipts determine which physical KV block IDs are
   cold and therefore write back to HBF for each HBM/HBF topology.
3. A capacity-scaled HBF replay measures media WAF for the exact cold block,
   token-offset, batch, request, and layer write order.

The scaled replay samples block IDs at a predeclared stride and runs every
predeclared residue.  It keeps the production stack/channel/die/plane topology,
the 4 KiB Llama 3.1 70B KV page, the 80-layer layout, and approximately the
same mutable-data occupancy after the immutable-weight reservation.  It is a
WAF/GC sensitivity experiment, not a time-throughput experiment.

There is one WAF definition throughout:

    physical NAND payload program bytes / logical bytes submitted to HBF

The lifetime table is a first-order uniform-wear sensitivity over explicitly
assumed media-cycle budgets and KV-append token rates.  HBFSim does not model
P/E failure, bad-block growth, retention, disturb, or calibrated wear leveling;
the table must never be presented as a device lifetime prediction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, BinaryIO, TextIO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prepare_frontier_residency_config import (  # noqa: E402
    _compile_kv_partition,
    prepare_residency_config,
)
from validation.certificate import (  # noqa: E402
    EXPLORATORY_VALIDATION,
    CertificateError,
    VerifiedCertificate,
    attach_certificate_to_summary_file,
    ensure_exploratory_summary,
    verify_certificate,
)
from validation.contracts import load_json_strict  # noqa: E402


STUDY_SCHEMA = {
    "name": "hbfsim.frontier_70b_hbf_wear_study_config",
    "version": 1,
}
AUDIT_SCHEMA = {
    "name": "hbfsim.frontier_replay_audit",
    "version": 7,
}
RECEIPT_SCHEMA = {
    "name": "hbfsim.frontier_residency_binding",
    "version": 3,
}
SUMMARY_SCHEMA = {
    "name": "hbfsim.scenario_compare.summary",
    "version": 16,
}
OUTPUT_SCHEMA = {
    "name": "hbfsim.frontier_70b_hbf_wear_study",
    "version": 1,
}
WAF_DEFINITION = "physical_write_bytes/logical_write_bytes"
PAGE_SIZE = 4096
MAPPING_ENTRIES_PER_PAGE = 512
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


class WearStudyError(ValueError):
    """The study input or output violates its evidence contract."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WearStudyError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise WearStudyError(f"{name} must be an array")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WearStudyError(f"{name} must be an integer")
    if value < minimum:
        raise WearStudyError(f"{name} must be >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise WearStudyError(f"{name} must be finite")
    result = float(value)
    if result < minimum:
        raise WearStudyError(f"{name} must be >= {minimum}")
    return result


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WearStudyError(f"{name} must be a non-empty string")
    return value


def _expect(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise WearStudyError(
            f"{name}: expected {expected!r}, got {actual!r}"
        )


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise WearStudyError("ceil division requires non-negative/positive inputs")
    return (numerator + denominator - 1) // denominator


def _placement_mix64(value: int) -> int:
    """Mirror the HBF controller's stateless SplitMix64 lane rotation."""
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def _mapping_vpn_for_lpn(lpn: int, stacks: int) -> int:
    if lpn < 0 or stacks <= 0:
        raise WearStudyError("mapping VPN requires non-negative/positive inputs")
    stripe = lpn // stacks
    lane = lpn % stacks
    group = stripe // MAPPING_ENTRIES_PER_PAGE
    rotation = _placement_mix64(group) % stacks
    stack = (
        lane - (stacks - rotation)
        if lane >= stacks - rotation
        else lane + rotation
    )
    return group * stacks + stack


def _static_weight_reserved_blocks(
    *,
    weight_begin_page: int,
    weight_end_page: int,
    stacks: int,
    planes_per_stack: int,
    blocks_per_plane: int,
    pages_per_block: int,
) -> int:
    """Count the exact whole blocks fenced by static immutable weights.

    This mirrors LayerStreamingComposition's full-block fast path and its two
    edge extents.  Edge work is bounded by two global block coordinates, so it
    does not expand the complete 70B weight population page by page.
    """
    if not 0 <= weight_begin_page <= weight_end_page:
        raise WearStudyError("invalid immutable-weight page interval")
    planes = stacks * planes_per_stack
    total_pages = planes * blocks_per_plane * pages_per_block
    if weight_end_page > total_pages:
        raise WearStudyError("immutable weights exceed HBF payload capacity")
    pages_per_global_block = planes * pages_per_block
    first_complete = _ceil_div(weight_begin_page, pages_per_global_block)
    last_complete = weight_end_page // pages_per_global_block

    edge_blocks: set[int] = set()

    def add_edge_page(source_page: int) -> None:
        stack = _mapping_vpn_for_lpn(source_page, stacks) % stacks
        stack_page_index = source_page // stacks
        local_plane = stack_page_index % planes_per_stack
        page_in_plane = stack_page_index // planes_per_stack
        block = page_in_plane // pages_per_block
        edge_blocks.add(
            (stack * planes_per_stack + local_plane) * blocks_per_plane
            + block
        )

    first_complete_page = first_complete * pages_per_global_block
    for source_page in range(
        weight_begin_page,
        min(weight_end_page, first_complete_page),
    ):
        add_edge_page(source_page)
    complete_blocks = max(last_complete - first_complete, 0) * planes
    last_complete_page = last_complete * pages_per_global_block
    tail_begin = max(
        min(weight_end_page, first_complete_page),
        last_complete_page,
    )
    for source_page in range(tail_begin, weight_end_page):
        add_edge_page(source_page)
    return complete_blocks + len(edge_blocks)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_csv_atomic(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8-sig",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _resolve_path(value: Any, *, owner: Path, name: str) -> Path:
    raw = _string(value, name)
    path = Path(raw)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _verify_artifact(
    record_value: Any,
    *,
    owner: Path,
    name: str,
) -> Path:
    record = _mapping(record_value, name)
    path = _resolve_path(record.get("path"), owner=owner, name=f"{name}.path")
    if not path.is_file():
        raise WearStudyError(f"{name} does not exist: {path}")
    expected_bytes = _integer(record.get("bytes"), f"{name}.bytes", minimum=1)
    expected_sha = _string(record.get("sha256"), f"{name}.sha256")
    if len(expected_sha) != 64:
        raise WearStudyError(f"{name}.sha256 is not SHA-256")
    _expect(path.stat().st_size, expected_bytes, f"{name} bytes")
    _expect(_sha256(path), expected_sha, f"{name} SHA-256")
    return path


def _load_study_config(path: Path) -> dict[str, Any]:
    data = load_json_strict(path)
    _expect(data.get("schema"), STUDY_SCHEMA, "study config schema")
    workload = _mapping(data.get("workload"), "study.workload")
    _expect(workload.get("model"), "llama31_70b", "study model")
    _expect(
        workload.get("precision_profile"),
        "w8a16-kv-bf16",
        "study precision profile",
    )
    _expect(workload.get("window"), "burst", "study window")
    _expect(workload.get("requests"), 256, "study request count")
    _expect(workload.get("num_layers"), 80, "study layer count")
    _expect(workload.get("kv_block_tokens"), 16, "study KV block tokens")
    _expect(
        workload.get("kv_bytes_per_token_per_layer"),
        PAGE_SIZE,
        "study KV bytes/token/layer",
    )
    sampling = _mapping(data.get("sampling"), "study.sampling")
    stride = _integer(
        sampling.get("block_stride"),
        "sampling block stride",
        minimum=2,
    )
    residues = [
        _integer(value, f"sampling residue[{index}]")
        for index, value in enumerate(
            _list(sampling.get("residues"), "sampling.residues")
        )
    ]
    if not residues or len(set(residues)) != len(residues):
        raise WearStudyError("sampling residues must be non-empty and unique")
    if any(value >= stride for value in residues):
        raise WearStudyError("sampling residue must be smaller than stride")
    epoch_counts = [
        _integer(value, f"epoch_counts[{index}]", minimum=1)
        for index, value in enumerate(
            _list(data.get("epoch_counts"), "study.epoch_counts")
        )
    ]
    if (
        not epoch_counts
        or epoch_counts != sorted(set(epoch_counts))
        or epoch_counts[0] != 1
    ):
        raise WearStudyError(
            "epoch_counts must be unique, increasing, and begin with 1"
        )
    topologies = _list(data.get("topologies"), "study.topologies")
    if not topologies:
        raise WearStudyError("study.topologies must be non-empty")
    topology_ids: set[str] = set()
    for index, value in enumerate(topologies):
        topology = _mapping(value, f"study.topologies[{index}]")
        topology_id = _string(topology.get("id"), f"topology[{index}].id")
        if topology_id in topology_ids:
            raise WearStudyError(f"duplicate topology {topology_id}")
        topology_ids.add(topology_id)
        _integer(
            topology.get("physical_hbm_capacity_bytes"),
            f"{topology_id} HBM capacity",
            minimum=1,
        )
        _integer(
            topology.get("expected_normalized_mutable_blocks_per_plane"),
            f"{topology_id} expected normalized mutable blocks/plane",
            minimum=8,
        )
        _integer(
            topology.get("usable_hbf_payload_capacity_bytes"),
            f"{topology_id} usable HBF payload capacity",
            minimum=1,
        )
    sensitivity = _mapping(
        data.get("lifetime_sensitivity"),
        "study.lifetime_sensitivity",
    )
    for field in ("kv_append_tokens_per_second", "assumed_media_cycles"):
        values = _list(sensitivity.get(field), f"lifetime.{field}")
        if not values:
            raise WearStudyError(f"lifetime.{field} must be non-empty")
        for index, value in enumerate(values):
            _number(value, f"lifetime.{field}[{index}]", minimum=1.0)
    return data


def _read_key_value_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            key, separator, value = line.partition("=")
            if not separator or not key or not value or key in values:
                raise WearStudyError(
                    f"{path}:{line_number}: invalid or duplicate config entry"
                )
            values[key] = value
    return values


def _config_integer(
    values: dict[str, str],
    key: str,
    name: str,
    *,
    minimum: int = 1,
) -> int:
    try:
        value = int(values[key], 10)
    except (KeyError, ValueError) as error:
        raise WearStudyError(f"{name} is missing or not an integer") from error
    if value < minimum:
        raise WearStudyError(f"{name} must be >= {minimum}")
    return value


def _verify_source_audit(audit_path: Path) -> dict[str, Any]:
    audit = load_json_strict(audit_path)
    _expect(audit.get("schema"), AUDIT_SCHEMA, "Frontier audit schema")
    _expect(audit.get("result"), "pass", "Frontier audit result")
    frontier = _mapping(audit.get("frontier"), "audit.frontier")
    _expect(frontier.get("model"), "llama31_70b", "Frontier model")
    memory = _mapping(audit.get("model_memory"), "audit.model_memory")
    _expect(memory.get("profile_id"), "w8a16-kv-bf16", "memory profile")
    requests = _mapping(audit.get("requests"), "audit.requests")
    _expect(requests.get("count"), 256, "audited request count")
    request_suite = _mapping(
        audit.get("request_suite"),
        "audit.request_suite",
    )
    _expect(request_suite.get("window_id"), "burst", "request window")
    selection = _mapping(
        request_suite.get("selection"),
        "audit.request_suite.selection",
    )
    _expect(selection.get("records"), 256, "request selection records")
    eligibility = _mapping(audit.get("eligibility"), "audit.eligibility")
    for field in (
        "production_request_window_verified",
        "scheduler_ledger_valid",
        "kv_block_lifecycle_valid",
        "hybrid_residency_plan_valid",
    ):
        _expect(eligibility.get(field), True, f"audit eligibility {field}")
    _expect(
        eligibility.get("eligible_for_ttft_tpot_slo_claims"),
        False,
        "TTFT/TPOT/SLO eligibility",
    )
    _expect(
        eligibility.get("frontier_timing_kind"),
        "dummy",
        "Frontier timing kind",
    )
    ledger = _mapping(audit.get("ledger"), "audit.ledger")
    lifecycle = _mapping(audit.get("kv_lifecycle"), "audit.kv_lifecycle")
    plan = _mapping(audit.get("residency_plan"), "audit.residency_plan")
    _expect(ledger.get("memory_contract_schema_version"), 4, "ledger schema")
    _expect(lifecycle.get("schema_version"), 1, "lifecycle schema")
    _expect(plan.get("kv_mutability"), "mutable", "KV mutability")
    _expect(plan.get("weight_mutability"), "read_only", "weight mutability")
    _expect(plan.get("kv_block_size_tokens"), 16, "KV block tokens")
    _expect(plan.get("kv_page_bytes_per_layer"), 65536, "KV layer bytes")
    _expect(plan.get("num_layers"), 80, "model layers")
    _expect(plan.get("kv_block_stride_bytes"), 80 * 65536, "KV stride")
    artifacts = _mapping(audit.get("artifacts"), "audit.artifacts")
    ledger_path = _verify_artifact(
        artifacts.get("stage_batch_ledger"),
        owner=audit_path,
        name="stage batch ledger",
    )
    lifecycle_path = _verify_artifact(
        artifacts.get("kv_block_lifecycle"),
        owner=audit_path,
        name="KV lifecycle",
    )
    return {
        "audit": audit,
        "audit_record": _artifact_record(audit_path),
        "ledger_path": ledger_path,
        "lifecycle_path": lifecycle_path,
        "ledger_rows": _integer(ledger.get("rows"), "ledger rows", minimum=1),
        "lifecycle_events": _integer(
            lifecycle.get("events"),
            "lifecycle events",
            minimum=1,
        ),
        "logical_kv_blocks": _integer(
            plan.get("num_logical_kv_blocks"),
            "logical KV blocks",
            minimum=1,
        ),
        "total_request_tokens": _integer(
            requests.get("total_tokens"),
            "total request tokens",
            minimum=1,
        ),
        "request_duration_s": _number(
            float(_string(selection.get("duration_s"), "request duration")),
            "request duration",
            minimum=0.0,
        ),
    }


def _maximum_active_cold_context(
    *,
    source: dict[str, Any],
    hot_thresholds: dict[str, int],
    block_tokens: int,
) -> dict[str, dict[str, int]]:
    """Replay request residency and find the largest active cold-KV union."""
    logical_blocks = source["logical_kv_blocks"]
    request_blocks: dict[str, list[int]] = {}
    request_tokens: dict[str, int] = {}
    token_refs = [0] * (logical_blocks * block_tokens)
    active = {topology: 0 for topology in hot_thresholds}
    maxima = {
        topology: {"tokens": 0, "batch": 0}
        for topology in hot_thresholds
    }
    event_index = 0

    def block_id(value: Any, name: str) -> int:
        result = _integer(value, name)
        if result >= logical_blocks:
            raise WearStudyError(
                f"{name}={result} exceeds {logical_blocks} blocks"
            )
        return result

    def change_token(request_id: str, position: int, delta: int) -> None:
        blocks = request_blocks[request_id]
        block_index = position // block_tokens
        if block_index >= len(blocks):
            raise WearStudyError(
                f"request {request_id} token {position} exceeds its blocks"
            )
        current_block = blocks[block_index]
        index = current_block * block_tokens + position % block_tokens
        before = token_refs[index]
        after = before + delta
        if after < 0:
            raise WearStudyError("active KV token reference underflow")
        token_refs[index] = after
        if before == 0 and after > 0:
            for topology, hot in hot_thresholds.items():
                if current_block >= hot:
                    active[topology] += 1
        elif before > 0 and after == 0:
            for topology, hot in hot_thresholds.items():
                if current_block >= hot:
                    active[topology] -= 1

    def remove_request(request_id: str) -> None:
        if request_id not in request_blocks:
            raise WearStudyError(f"release of unknown request {request_id}")
        tokens = request_tokens.pop(request_id, 0)
        for position in range(tokens):
            change_token(request_id, position, -1)
        request_blocks.pop(request_id)

    def apply_event(event: dict[str, Any]) -> None:
        nonlocal event_index
        _expect(event.get("event_index"), event_index, "lifecycle index")
        event_index += 1
        event_type = _string(event.get("event"), "lifecycle event")
        request_value = event.get("request_id")
        request_id = None if request_value is None else str(request_value)
        if event_type in {
            "prefix_lookup",
            "prefix_admission",
            "cache_assign",
            "evict",
        }:
            if event_type == "evict":
                block_id(event.get("block_id"), "evict block")
            return
        if request_id is None:
            raise WearStudyError(f"{event_type} has no request_id")
        if event_type == "allocate":
            request_blocks.setdefault(request_id, []).append(
                block_id(event.get("block_id"), "allocate block")
            )
            return
        if event_type == "touch":
            transitions = _list(event.get("blocks"), "touch blocks")
            if not transitions:
                raise WearStudyError("touch has no blocks")
            target = request_blocks.setdefault(request_id, [])
            for index, raw in enumerate(transitions):
                transition = _mapping(raw, f"touch block[{index}]")
                target.append(
                    block_id(
                        transition.get("block_id"),
                        f"touch block[{index}].block_id",
                    )
                )
            return
        if event_type == "release":
            remove_request(request_id)
            return
        raise WearStudyError(f"unsupported lifecycle event {event_type!r}")

    with source["lifecycle_path"].open(encoding="utf-8") as event_handle:
        with source["ledger_path"].open(encoding="utf-8") as ledger_handle:
            batch_count = 0
            for batch_index, raw in enumerate(ledger_handle):
                try:
                    row = json.loads(raw)
                except (json.JSONDecodeError, ValueError) as error:
                    raise WearStudyError(
                        f"invalid ledger JSON at row {batch_index}: {error}"
                    ) from error
                row = _mapping(row, f"ledger row {batch_index}")
                _expect(row.get("batch_id"), batch_index, "ledger batch ID")
                contract = _mapping(
                    row.get("hbfsim_memory_contract"),
                    f"batch {batch_index} memory contract",
                )
                cursor = _integer(
                    contract.get("lifecycle_event_cursor"),
                    f"batch {batch_index} lifecycle cursor",
                )
                if cursor < event_index:
                    raise WearStudyError("lifecycle cursor moved backward")
                while event_index < cursor:
                    apply_event(_read_event(event_handle, event_index))
                snapshots = _list(
                    contract.get("requests"),
                    f"batch {batch_index} requests",
                )
                for request_index, raw_snapshot in enumerate(snapshots):
                    snapshot = _mapping(
                        raw_snapshot,
                        f"batch {batch_index} request[{request_index}]",
                    )
                    request_id = str(snapshot.get("request_id"))
                    blocks = request_blocks.get(request_id)
                    if blocks is None:
                        raise WearStudyError(
                            f"batch {batch_index} request {request_id} "
                            "has no allocator state"
                        )
                    _expect(
                        len(blocks),
                        snapshot.get("allocated_block_count"),
                        f"batch {batch_index} request {request_id} blocks",
                    )
                    previous = request_tokens.get(request_id, 0)
                    current = _integer(
                        snapshot.get("kv_tokens_after"),
                        "request KV tokens after",
                    )
                    if current < previous:
                        raise WearStudyError(
                            f"request {request_id} KV frontier moved backward"
                        )
                    for position in range(previous, current):
                        change_token(request_id, position, 1)
                    request_tokens[request_id] = current
                for topology, count in active.items():
                    if count > maxima[topology]["tokens"]:
                        maxima[topology] = {
                            "tokens": count,
                            "batch": batch_index,
                        }
                batch_count += 1
        while event_index < source["lifecycle_events"]:
            apply_event(_read_event(event_handle, event_index))
        if event_handle.readline():
            raise WearStudyError("lifecycle contains uncounted events")
    _expect(batch_count, source["ledger_rows"], "ledger row count")
    if request_blocks or request_tokens or any(active.values()):
        raise WearStudyError("active-context replay did not drain")
    if any(token_refs):
        raise WearStudyError("active-context token references did not drain")
    return maxima


def _compile_full_lifecycle_placements(
    *,
    source: dict[str, Any],
    placements: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Solve active-buffer/hot-KV placement against the complete ledger."""
    block_tokens = 16
    transformer_bytes = _integer(
        source["audit"]["model_memory"]["weight_streaming"]["objects"]
        ["transformer_layer"]["memory_bytes"],
        "transformer layer bytes",
        minimum=1,
    )
    transformer_pages = (
        transformer_bytes // PAGE_SIZE
        + int(transformer_bytes % PAGE_SIZE != 0)
    )
    buffers: dict[str, int] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    for placement in placements:
        contract = placement["population_receipt"]["contract"]
        buffers[placement["id"]] = _integer(
            contract.get("source_active_weight_buffer_pages_per_slot"),
            "source active buffer pages",
            minimum=1,
        )
        histories[placement["id"]] = []

    final_partitions: dict[str, dict[str, int]] = {}
    for iteration in range(1, 65):
        partitions: dict[str, dict[str, int]] = {}
        thresholds: dict[str, int] = {}
        for placement in placements:
            topology_id = placement["id"]
            contract = placement["population_receipt"]["contract"]
            partition = _compile_kv_partition(
                physical_hbm_capacity_bytes=_integer(
                    contract.get("physical_hbm_capacity_bytes"),
                    f"{topology_id} HBM capacity",
                    minimum=1,
                ),
                page_size=PAGE_SIZE,
                metadata_pages=_integer(
                    contract.get("metadata_resident_pages"),
                    f"{topology_id} metadata pages",
                    minimum=1,
                ),
                active_buffer_pages=buffers[topology_id],
                logical_blocks=source["logical_kv_blocks"],
                kv_stride=_integer(
                    contract.get("kv_block_stride_bytes"),
                    f"{topology_id} KV stride",
                    minimum=1,
                ),
            )
            partitions[topology_id] = partition
            thresholds[topology_id] = partition["hot_blocks"]
        maxima = _maximum_active_cold_context(
            source=source,
            hot_thresholds=thresholds,
            block_tokens=block_tokens,
        )
        converged = True
        for placement in placements:
            topology_id = placement["id"]
            required = max(
                _integer(
                    placement["population_receipt"]["contract"].get(
                        "source_active_weight_buffer_pages_per_slot"
                    ),
                    "source active buffer pages",
                    minimum=1,
                ),
                transformer_pages + maxima[topology_id]["tokens"],
            )
            histories[topology_id].append({
                "iteration": iteration,
                "active_buffer_pages": buffers[topology_id],
                "hot_kv_blocks": partitions[topology_id]["hot_blocks"],
                "cold_kv_blocks": partitions[topology_id]["cold_blocks"],
                "max_unique_active_cold_kv_tokens": (
                    maxima[topology_id]["tokens"]
                ),
                "max_active_batch": maxima[topology_id]["batch"],
                "required_active_buffer_pages": required,
            })
            if required > buffers[topology_id]:
                buffers[topology_id] = required
                converged = False
        final_partitions = partitions
        if converged:
            break
    else:
        raise WearStudyError("full-lifecycle placement did not converge")

    for placement in placements:
        topology_id = placement["id"]
        partition = final_partitions[topology_id]
        final_history = histories[topology_id][-1]
        population_contract = placement["population_receipt"]["contract"]
        immutable_weight_pages = _integer(
            population_contract.get("immutable_weight_pages"),
            f"{topology_id} immutable weight pages",
            minimum=1,
        )
        static_weight_resident_pages = min(
            immutable_weight_pages,
            partition["unused_hbm_pages"],
        )
        model_weight_backing_pages = (
            immutable_weight_pages - static_weight_resident_pages
        )
        receipt = {
            "schema": {
                "name": "hbfsim.frontier_70b_wear_placement",
                "version": 1,
            },
            "result": "pass",
            "inputs": {
                "frontier_audit": source["audit_record"],
                "population_receipt": placement["population_receipt_record"],
            },
            "policy": "full_lifecycle_active_buffer_fixed_point_v1",
            "contract": {
                "physical_hbm_capacity_bytes": (
                    placement["population_receipt"]["contract"]
                    ["physical_hbm_capacity_bytes"]
                ),
                "logical_kv_blocks": source["logical_kv_blocks"],
                "hot_kv_blocks": partition["hot_blocks"],
                "cold_kv_blocks": partition["cold_blocks"],
                "active_buffer_pages_per_slot": buffers[topology_id],
                "active_buffer_bytes_per_slot": (
                    buffers[topology_id] * PAGE_SIZE
                ),
                "max_unique_active_cold_kv_tokens": (
                    final_history["max_unique_active_cold_kv_tokens"]
                ),
                "max_active_batch": final_history["max_active_batch"],
                "transformer_weight_pages": transformer_pages,
                "active_buffer_count": 2,
                "immutable_weight_pages": immutable_weight_pages,
                "static_weight_resident_pages": (
                    static_weight_resident_pages
                ),
                "model_weight_backing_pages": model_weight_backing_pages,
                "unused_hbm_pages_after_static_weights": (
                    partition["unused_hbm_pages"]
                    - static_weight_resident_pages
                ),
                "population_preserved_across_placement": True,
            },
            "fixed_point_history": histories[topology_id],
        }
        receipt_path = (
            out_dir
            / "placements"
            / topology_id.lower()
            / "wear-placement.receipt.json"
        )
        _write_json_atomic(receipt_path, receipt)
        placement.update({
            "receipt": receipt,
            "receipt_path": receipt_path,
            "receipt_record": _artifact_record(receipt_path),
            "hot_kv_blocks": partition["hot_blocks"],
            "cold_kv_blocks": partition["cold_blocks"],
            "active_buffer_pages": buffers[topology_id],
            "active_buffer_bytes": buffers[topology_id] * PAGE_SIZE,
            "static_weight_resident_pages": static_weight_resident_pages,
            "model_weight_backing_pages": model_weight_backing_pages,
        })


def _compile_placements(
    *,
    source: dict[str, Any],
    study: dict[str, Any],
    manifest_path: Path,
    object_map_path: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    placements: list[dict[str, Any]] = []
    for value in _list(study.get("topologies"), "study.topologies"):
        topology = _mapping(value, "study topology")
        topology_id = _string(topology.get("id"), "topology id")
        hardware_config = _resolve_path(
            topology.get("hardware_config"),
            owner=ROOT / "placeholder",
            name=f"{topology_id} hardware config",
        )
        if not hardware_config.is_file():
            raise WearStudyError(
                f"{topology_id} hardware config does not exist: "
                f"{hardware_config}"
            )
        placement_dir = out_dir / "placements" / topology_id.lower()
        config_path = placement_dir / "residency.cfg"
        receipt_path = placement_dir / "residency.receipt.json"
        receipt = prepare_residency_config(
            manifest_path=manifest_path,
            object_map_path=object_map_path,
            output_config_path=config_path,
            receipt_path=receipt_path,
            physical_hbm_capacity_bytes=_integer(
                topology.get("physical_hbm_capacity_bytes"),
                f"{topology_id} physical HBM capacity",
                minimum=1,
            ),
        )
        _expect(receipt.get("schema"), RECEIPT_SCHEMA, "placement schema")
        _expect(receipt.get("result"), "pass", "placement result")
        contract = _mapping(receipt.get("contract"), "placement contract")
        _expect(
            contract.get("logical_kv_blocks"),
            83205,
            f"{topology_id} logical KV blocks",
        )
        values = _read_key_value_config(hardware_config)
        stacks = _config_integer(values, "hbf-stacks", "HBF stacks")
        channels = _config_integer(values, "hbf-channels", "HBF channels")
        dies = _config_integer(
            values,
            "hbf-dies-per-channel",
            "HBF dies/channel",
        )
        planes = _config_integer(
            values,
            "hbf-planes-per-die",
            "HBF planes/die",
        )
        full_blocks = _config_integer(
            values,
            "hbf-blocks-per-plane",
            "HBF full blocks/plane",
        )
        pages_per_block = _config_integer(
            values,
            "hbf-pages-per-block",
            "HBF pages/block",
        )
        gc_reserved_blocks = _config_integer(
            values,
            "hbf-gc-reserved-free-blocks-per-plane",
            "HBF GC reserved blocks/plane",
            minimum=0,
        )
        gc_low_watermark_pages = _config_integer(
            values,
            "hbf-gc-low-watermark-pages",
            "HBF GC low watermark pages",
            minimum=0,
        )
        gc_hard_watermark_pages = _config_integer(
            values,
            "hbf-gc-hard-watermark-pages",
            "HBF GC hard watermark pages",
            minimum=0,
        )
        page_size = _config_integer(values, "hbf-page-size", "HBF page size")
        _expect(page_size, PAGE_SIZE, "HBF page size")
        usable_payload_capacity = (
            stacks
            * channels
            * dies
            * planes
            * full_blocks
            * pages_per_block
            * page_size
        )
        _expect(
            usable_payload_capacity,
            topology.get("usable_hbf_payload_capacity_bytes"),
            f"{topology_id} usable HBF payload capacity",
        )
        population_receipt_record = _artifact_record(receipt_path)
        placements.append({
            "id": topology_id,
            "hardware_config": hardware_config,
            "hardware_config_record": _artifact_record(hardware_config),
            "population_receipt": receipt,
            "population_receipt_path": receipt_path,
            "population_receipt_record": population_receipt_record,
            "expected_normalized_mutable_blocks_per_plane": _integer(
                topology.get(
                    "expected_normalized_mutable_blocks_per_plane"
                ),
                f"{topology_id} expected normalized mutable blocks/plane",
                minimum=8,
            ),
            "usable_hbf_payload_capacity_bytes": usable_payload_capacity,
            "stacks": stacks,
            "channels_per_stack": channels,
            "dies_per_channel": dies,
            "planes_per_die": planes,
            "full_blocks_per_plane": full_blocks,
            "pages_per_block": pages_per_block,
            "full_gc_reserved_blocks_per_plane": gc_reserved_blocks,
            "full_gc_low_watermark_pages": gc_low_watermark_pages,
            "full_gc_hard_watermark_pages": gc_hard_watermark_pages,
        })
    _compile_full_lifecycle_placements(
        source=source,
        placements=placements,
        out_dir=out_dir,
    )
    stride = _integer(
        _mapping(study.get("sampling"), "study.sampling").get(
            "block_stride"
        ),
        "sampling block stride",
        minimum=2,
    )
    for placement in placements:
        topology_id = placement["id"]
        total_planes = (
            placement["stacks"]
            * placement["channels_per_stack"]
            * placement["dies_per_channel"]
            * placement["planes_per_die"]
        )
        static_blocks = _static_weight_reserved_blocks(
            weight_begin_page=placement["static_weight_resident_pages"],
            weight_end_page=(
                placement["static_weight_resident_pages"]
                + placement["model_weight_backing_pages"]
            ),
            stacks=placement["stacks"],
            planes_per_stack=(
                placement["channels_per_stack"]
                * placement["dies_per_channel"]
                * placement["planes_per_die"]
            ),
            blocks_per_plane=placement["full_blocks_per_plane"],
            pages_per_block=placement["pages_per_block"],
        )
        full_blocks = total_planes * placement["full_blocks_per_plane"]
        gc_blocks = (
            total_planes
            * placement["full_gc_reserved_blocks_per_plane"]
        )
        writable_blocks = full_blocks - static_blocks - gc_blocks
        if writable_blocks <= 0:
            raise WearStudyError(
                f"{topology_id} has no writable HBF block pool"
            )
        denominator = stride * total_planes
        normalized_blocks_per_plane = (
            writable_blocks + denominator // 2
        ) // denominator
        _expect(
            normalized_blocks_per_plane,
            placement["expected_normalized_mutable_blocks_per_plane"],
            f"{topology_id} normalized mutable blocks/plane",
        )
        normalized_gc_reserved_blocks = (
            placement["full_gc_reserved_blocks_per_plane"] + stride // 2
        ) // stride
        normalized_gc_low_watermark_pages = _ceil_div(
            placement["full_gc_low_watermark_pages"],
            stride,
        )
        normalized_gc_hard_watermark_pages = _ceil_div(
            placement["full_gc_hard_watermark_pages"],
            stride,
        )
        normalized_capacity = (
            total_planes
            * normalized_blocks_per_plane
            * placement["pages_per_block"]
            * PAGE_SIZE
        )
        placement.update({
            "full_static_weight_reserved_blocks": static_blocks,
            "full_writable_pool_blocks": writable_blocks,
            "full_writable_pool_payload_bytes": (
                writable_blocks * placement["pages_per_block"] * PAGE_SIZE
            ),
            "normalized_blocks_per_plane": normalized_blocks_per_plane,
            "normalized_gc_reserved_blocks_per_plane": (
                normalized_gc_reserved_blocks
            ),
            "normalized_gc_low_watermark_pages": (
                normalized_gc_low_watermark_pages
            ),
            "normalized_gc_hard_watermark_pages": (
                normalized_gc_hard_watermark_pages
            ),
            "normalized_mutable_pool_capacity_bytes": normalized_capacity,
            "normalized_mutable_pool_block_error_at_full_scale": (
                normalized_blocks_per_plane * denominator - writable_blocks
            ),
        })
    return placements


class _ReplayState:
    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.request_blocks: dict[str, list[int]] = {}
        self.event_index = 0

    def _block_id(self, value: Any, name: str) -> int:
        block_id = _integer(value, name)
        if block_id >= self.num_blocks:
            raise WearStudyError(
                f"{name}={block_id} exceeds {self.num_blocks} blocks"
            )
        return block_id

    def apply(self, event: dict[str, Any]) -> None:
        _expect(event.get("event_index"), self.event_index, "lifecycle index")
        self.event_index += 1
        event_type = _string(event.get("event"), "lifecycle event")
        request_value = event.get("request_id")
        request_id = None if request_value is None else str(request_value)
        if event_type in {
            "prefix_lookup",
            "prefix_admission",
            "cache_assign",
            "evict",
        }:
            if event_type == "evict":
                self._block_id(event.get("block_id"), "evict block")
            return
        if request_id is None:
            raise WearStudyError(f"{event_type} has no request_id")
        if event_type == "allocate":
            block_id = self._block_id(event.get("block_id"), "allocate block")
            self.request_blocks.setdefault(request_id, []).append(block_id)
            return
        if event_type == "touch":
            transitions = _list(event.get("blocks"), "touch blocks")
            if not transitions:
                raise WearStudyError("touch has no blocks")
            target = self.request_blocks.setdefault(request_id, [])
            for index, value in enumerate(transitions):
                transition = _mapping(value, f"touch block[{index}]")
                target.append(
                    self._block_id(
                        transition.get("block_id"),
                        f"touch block[{index}].block_id",
                    )
                )
            return
        if event_type == "release":
            transitions = _list(event.get("blocks"), "release blocks")
            if not transitions:
                raise WearStudyError("release has no blocks")
            if request_id not in self.request_blocks:
                raise WearStudyError(f"release of unknown request {request_id}")
            self.request_blocks.pop(request_id)
            return
        raise WearStudyError(f"unsupported lifecycle event {event_type!r}")


def _read_event(handle: TextIO, expected_index: int) -> dict[str, Any]:
    raw = handle.readline()
    if not raw:
        raise WearStudyError(
            f"KV lifecycle ended before event {expected_index}"
        )
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as error:
        raise WearStudyError(
            f"invalid lifecycle JSON at event {expected_index}: {error}"
        ) from error
    return _mapping(value, f"lifecycle event {expected_index}")


def _write_initial_image(
    path: Path,
    *,
    sampled_blocks: int,
    pages_per_kv_block: int,
    stacks: int,
) -> dict[str, Any]:
    records = sampled_blocks * pages_per_kv_block
    mapping_vpns: set[int] = set()
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# sampled cold-KV initial image; population only\n")
        for lpn in range(records):
            handle.write(f"0x{lpn * PAGE_SIZE:x} R {PAGE_SIZE}\n")
            mapping_vpns.add(_mapping_vpn_for_lpn(lpn, stacks))
    record = _artifact_record(path)
    record["records"] = records
    record["mapping_pages"] = len(mapping_vpns)
    return record


def _generate_base_traces(
    *,
    source: dict[str, Any],
    placements: list[dict[str, Any]],
    sampling: dict[str, Any],
    workload: dict[str, Any],
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stride = _integer(sampling.get("block_stride"), "sample stride", minimum=2)
    residues = [int(value) for value in sampling["residues"]]
    num_layers = _integer(workload.get("num_layers"), "layers", minimum=1)
    block_tokens = _integer(
        workload.get("kv_block_tokens"),
        "KV block tokens",
        minimum=1,
    )
    kv_bytes = _integer(
        workload.get("kv_bytes_per_token_per_layer"),
        "KV bytes/token/layer",
        minimum=1,
    )
    _expect(kv_bytes, PAGE_SIZE, "KV write page size")
    logical_blocks = source["logical_kv_blocks"]
    pages_per_kv_block = num_layers * block_tokens

    cells: dict[str, dict[str, Any]] = {}
    handles: dict[str, TextIO] = {}
    full_stats: dict[str, dict[str, Any]] = {}
    try:
        for placement in placements:
            topology_id = placement["id"]
            hot = placement["hot_kv_blocks"]
            full_stats[topology_id] = {
                "hot_kv_blocks": hot,
                "cold_kv_blocks": placement["cold_kv_blocks"],
                "cold_kv_token_writes": 0,
                "prefill_cold_kv_token_writes": 0,
                "decode_cold_kv_token_writes": 0,
                "first_cold_write_batch": None,
                "last_cold_write_batch": None,
                "touched_blocks": set(),
                "token_masks": {},
            }
            for residue in residues:
                selected = list(range(hot + residue, logical_blocks, stride))
                if not selected:
                    raise WearStudyError(
                        f"{topology_id}/r{residue} samples no cold blocks"
                    )
                mapping = {block_id: index for index, block_id in enumerate(selected)}
                cell_id = f"{topology_id}.r{residue}"
                cell_dir = out_dir / "media" / topology_id.lower() / f"r{residue}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                initial_path = cell_dir / "initial-image.trace"
                base_path = cell_dir / "writes-1e.trace"
                initial_record = _write_initial_image(
                    initial_path,
                    sampled_blocks=len(selected),
                    pages_per_kv_block=pages_per_kv_block,
                    stacks=placement["stacks"],
                )
                handle = base_path.open("w", encoding="utf-8")
                handle.write(
                    "# page-expanded sampled Frontier cold-KV write order; "
                    f"topology={topology_id} stride={stride} residue={residue} "
                    "epochs=1\n"
                )
                handles[cell_id] = handle
                cells[cell_id] = {
                    "cell_id": cell_id,
                    "topology": topology_id,
                    "stacks": placement["stacks"],
                    "residue": residue,
                    "sample_stride": stride,
                    "selected_blocks": selected,
                    "block_to_slot": mapping,
                    "sampled_blocks": len(selected),
                    "sampled_token_writes_per_epoch": 0,
                    "sampled_prefill_tokens_per_epoch": 0,
                    "sampled_decode_tokens_per_epoch": 0,
                    "dirty_mapping_vpns": set(),
                    "base_trace_path": base_path,
                    "initial_image_path": initial_path,
                    "initial_image": initial_record,
                    "pages_per_kv_block": pages_per_kv_block,
                }

        state = _ReplayState(logical_blocks)
        batch_count = 0
        scheduled_tokens = 0
        with source["lifecycle_path"].open(encoding="utf-8") as event_handle:
            with source["ledger_path"].open(encoding="utf-8") as ledger_handle:
                for batch_index, raw in enumerate(ledger_handle):
                    try:
                        row = json.loads(raw)
                    except (json.JSONDecodeError, ValueError) as error:
                        raise WearStudyError(
                            f"invalid ledger JSON at row {batch_index}: {error}"
                        ) from error
                    row = _mapping(row, f"ledger row {batch_index}")
                    _expect(row.get("batch_id"), batch_index, "ledger batch ID")
                    contract = _mapping(
                        row.get("hbfsim_memory_contract"),
                        f"batch {batch_index} memory contract",
                    )
                    _expect(contract.get("schema_version"), 4, "memory schema")
                    cursor = _integer(
                        contract.get("lifecycle_event_cursor"),
                        f"batch {batch_index} lifecycle cursor",
                    )
                    if cursor < state.event_index:
                        raise WearStudyError("lifecycle cursor moved backward")
                    while state.event_index < cursor:
                        state.apply(_read_event(event_handle, state.event_index))

                    row_events: dict[str, list[tuple[int, int]]] = {
                        cell_id: [] for cell_id in cells
                    }
                    snapshots = _list(
                        contract.get("requests"),
                        f"batch {batch_index} requests",
                    )
                    if not snapshots:
                        raise WearStudyError(f"batch {batch_index} is empty")
                    for request_index, raw_snapshot in enumerate(snapshots):
                        snapshot = _mapping(
                            raw_snapshot,
                            f"batch {batch_index} request[{request_index}]",
                        )
                        request_id = str(snapshot.get("request_id"))
                        blocks = state.request_blocks.get(request_id)
                        if blocks is None:
                            raise WearStudyError(
                                f"batch {batch_index} request {request_id} "
                                "has no allocator state"
                            )
                        _expect(
                            len(blocks),
                            snapshot.get("allocated_block_count"),
                            f"batch {batch_index} request {request_id} blocks",
                        )
                        begin = _integer(
                            snapshot.get("kv_tokens_before"),
                            "KV token begin",
                        )
                        count = _integer(
                            snapshot.get("scheduled_tokens"),
                            "scheduled tokens",
                            minimum=1,
                        )
                        phase = _string(snapshot.get("phase"), "request phase")
                        if phase not in {"prefill", "decode"}:
                            raise WearStudyError(f"unsupported phase {phase!r}")
                        scheduled_tokens += count
                        for position in range(begin, begin + count):
                            block_index = position // block_tokens
                            if block_index >= len(blocks):
                                raise WearStudyError(
                                    f"request {request_id} token {position} "
                                    "exceeds allocated blocks"
                                )
                            block_id = blocks[block_index]
                            token_offset = position % block_tokens
                            for placement in placements:
                                topology_id = placement["id"]
                                stats = full_stats[topology_id]
                                if block_id < placement["hot_kv_blocks"]:
                                    continue
                                stats["cold_kv_token_writes"] += 1
                                stats[f"{phase}_cold_kv_token_writes"] += 1
                                if stats["first_cold_write_batch"] is None:
                                    stats["first_cold_write_batch"] = batch_index
                                stats["last_cold_write_batch"] = batch_index
                                stats["touched_blocks"].add(block_id)
                                old_mask = stats["token_masks"].get(block_id, 0)
                                stats["token_masks"][block_id] = (
                                    old_mask | (1 << token_offset)
                                )
                                residue = (block_id - placement["hot_kv_blocks"]) % stride
                                cell_id = f"{topology_id}.r{residue}"
                                if cell_id not in row_events:
                                    continue
                                cell = cells[cell_id]
                                slot = cell["block_to_slot"].get(block_id)
                                if slot is None:
                                    raise WearStudyError(
                                        f"sampled block {block_id} lost its slot"
                                    )
                                row_events[cell_id].append((slot, token_offset))
                                cell["sampled_token_writes_per_epoch"] += 1
                                cell[f"sampled_{phase}_tokens_per_epoch"] += 1

                    for cell_id, events in row_events.items():
                        if not events:
                            continue
                        cell = cells[cell_id]
                        handle = handles[cell_id]
                        for layer in range(num_layers):
                            layer_offset = layer * block_tokens
                            for slot, token_offset in events:
                                lpn = (
                                    slot * pages_per_kv_block
                                    + layer_offset
                                    + token_offset
                                )
                                handle.write(
                                    f"0x{lpn * PAGE_SIZE:x} W {PAGE_SIZE}\n"
                                )
                                cell["dirty_mapping_vpns"].add(
                                    _mapping_vpn_for_lpn(
                                        lpn,
                                        cell["stacks"],
                                    )
                                )
                    batch_count += 1

            while state.event_index < source["lifecycle_events"]:
                state.apply(_read_event(event_handle, state.event_index))
            if event_handle.readline():
                raise WearStudyError("lifecycle contains uncounted events")
        _expect(batch_count, source["ledger_rows"], "ledger row count")
        _expect(
            state.event_index,
            source["lifecycle_events"],
            "lifecycle event count",
        )
        if state.request_blocks:
            raise WearStudyError("lifecycle ended with referenced requests")
    finally:
        for handle in handles.values():
            handle.close()

    output_full: dict[str, Any] = {}
    for topology_id, stats in full_stats.items():
        token_masks: dict[int, int] = stats.pop("token_masks")
        touched: set[int] = stats.pop("touched_blocks")
        unique_token_slots = sum(mask.bit_count() for mask in token_masks.values())
        cold_tokens = stats["cold_kv_token_writes"]
        logical_bytes = cold_tokens * num_layers * kv_bytes
        output_full[topology_id] = {
            **stats,
            "touched_cold_kv_blocks": len(touched),
            "unique_block_token_positions": unique_token_slots,
            "repeated_block_token_writes": cold_tokens - unique_token_slots,
            "cold_kv_token_fraction": cold_tokens / scheduled_tokens,
            "logical_hbf_write_bytes": logical_bytes,
            "logical_hbf_write_bytes_per_kv_append_token": (
                logical_bytes / scheduled_tokens
            ),
            "logical_hbf_write_bytes_per_request": logical_bytes / 256,
            "logical_hbf_write_bytes_over_usable_hbf_capacity": (
                logical_bytes
                / next(
                    placement["usable_hbf_payload_capacity_bytes"]
                    for placement in placements
                    if placement["id"] == topology_id
                )
            ),
        }

    output_cells: dict[str, Any] = {}
    for cell_id, cell in cells.items():
        base_record = _artifact_record(cell["base_trace_path"])
        expected_records = cell["sampled_token_writes_per_epoch"] * num_layers
        base_record["records"] = expected_records
        output_cells[cell_id] = {
            key: value
            for key, value in cell.items()
            if key not in {
                "selected_blocks",
                "block_to_slot",
                "base_trace_path",
                "initial_image_path",
                "dirty_mapping_vpns",
            }
        }
        output_cells[cell_id].update({
            "base_trace_path": cell["base_trace_path"],
            "initial_image_path": cell["initial_image_path"],
            "base_trace": base_record,
            "expected_mapping_checkpoint_pages": len(
                cell["dirty_mapping_vpns"]
            ),
            "sampling_estimated_full_cold_tokens": (
                cell["sampled_token_writes_per_epoch"] * stride
            ),
            "sampling_error_vs_exact_tokens": (
                cell["sampled_token_writes_per_epoch"] * stride
                - output_full[cell["topology"]]["cold_kv_token_writes"]
            ),
        })
    return ({
        "ledger_batches": batch_count,
        "scheduled_kv_append_tokens": scheduled_tokens,
        "full_population": output_full,
    }, output_cells)


def _copy_epochs(
    *,
    base_path: Path,
    output_path: Path,
    epochs: int,
    cell_id: str,
) -> dict[str, Any]:
    if epochs == 1:
        record = _artifact_record(base_path)
        return {**record, "epochs": 1}
    with base_path.open("rb") as handle:
        lines = handle.readlines()
    body = b"".join(line for line in lines if not line.startswith(b"#"))
    with output_path.open("wb") as handle:
        handle.write(
            f"# sampled Frontier cold-KV write order; cell={cell_id} "
            f"epochs={epochs}\n".encode()
        )
        for _ in range(epochs):
            handle.write(body)
    record = _artifact_record(output_path)
    return {**record, "epochs": epochs}


def _verify_media_summary(
    *,
    summary_path: Path,
    trace_record: dict[str, Any],
    expected_records: int,
    expected_initial_pages: int,
    expected_initial_mapping_pages: int,
    expected_mapping_programs: int,
    validation_certificate: VerifiedCertificate | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if validation_certificate is None:
        summary = load_json_strict(summary_path)
        ensure_exploratory_summary(summary)
    else:
        summary = attach_certificate_to_summary_file(
            summary_path,
            validation_certificate,
        )
    _expect(summary.get("schema"), SUMMARY_SCHEMA, "summary schema")
    _expect(summary.get("sanity"), "PASS", "summary sanity")
    workload = _mapping(summary.get("workload"), "summary.workload")
    digest = _mapping(workload.get("trace_digest"), "summary trace digest")
    _expect(digest.get("algorithm"), "sha256", "trace digest algorithm")
    _expect(digest.get("value"), trace_record["sha256"], "trace digest")
    _expect(workload.get("trace_file_bytes"), trace_record["bytes"], "trace bytes")
    scenarios = _list(summary.get("scenarios"), "summary.scenarios")
    matches = [value for value in scenarios if value.get("name") == "all-HBF"]
    if len(matches) != 1:
        raise WearStudyError(f"expected one all-HBF scenario, got {len(matches)}")
    scenario = _mapping(matches[0], "all-HBF scenario")
    _expect(scenario.get("ops"), expected_records, "scenario operations")
    _expect(scenario.get("writes"), expected_records, "scenario writes")
    _expect(scenario.get("reads"), 0, "scenario reads")
    logical_bytes = expected_records * PAGE_SIZE
    _expect(scenario.get("logical_bytes"), logical_bytes, "scenario bytes")
    stats = _mapping(scenario.get("hbf_stats"), "HBF stats")
    _expect(stats.get("logical_write_bytes"), logical_bytes, "logical writes")
    _expect(stats.get("logical_read_bytes"), 0, "logical reads")
    _expect(stats.get("mapping_entries"), expected_initial_pages, "mapping entries")
    _expect(stats.get("accounting_verified"), True, "HBF accounting audit")
    _expect(stats.get("pending_program_pages"), 0, "pending programs")
    _expect(
        stats.get("pending_mapping_publications"),
        0,
        "pending mapping publications",
    )
    page_programs = _integer(stats.get("page_programs"), "page programs")
    data_programs = _integer(stats.get("data_programs"), "data programs")
    mapping_programs = _integer(
        stats.get("mapping_page_programs"),
        "mapping programs",
    )
    relocations = _integer(stats.get("gc_relocations"), "GC relocations")
    physical_bytes = _integer(
        stats.get("physical_write_bytes"),
        "physical write bytes",
    )
    _expect(
        page_programs,
        data_programs + mapping_programs + relocations,
        "page-program conservation",
    )
    _expect(physical_bytes, page_programs * PAGE_SIZE, "physical write bytes")
    for key, programs in (
        ("data_program_payload_bytes", data_programs),
        ("mapping_program_payload_bytes", mapping_programs),
        ("gc_relocation_payload_bytes", relocations),
    ):
        _expect(stats.get(key), programs * PAGE_SIZE, key)
    _expect(stats.get("waf_definition"), WAF_DEFINITION, "WAF definition")
    expected_waf = physical_bytes / logical_bytes
    reported_waf = _number(stats.get("waf"), "reported WAF")
    if not math.isclose(reported_waf, expected_waf, rel_tol=1e-12, abs_tol=1e-12):
        raise WearStudyError(
            f"reported WAF {reported_waf} != {expected_waf}"
        )
    config = _mapping(summary.get("config"), "summary.config")
    hbf_config = _mapping(config.get("hbf"), "summary.config.hbf")
    capacity = _integer(
        hbf_config.get("capacity_bytes"),
        "normalized HBF capacity",
        minimum=1,
    )
    pages_per_block = _integer(
        hbf_config.get("pages_per_block"),
        "normalized HBF pages/block",
        minimum=1,
    )
    _expect(data_programs, expected_records, "foreground data programs")
    _expect(mapping_programs, expected_mapping_programs, "mapping checkpoints")
    _expect(
        stats.get("initial_logical_data_pages"),
        expected_initial_pages,
        "initial logical data pages",
    )
    _expect(
        stats.get("initial_mapping_pages"),
        expected_initial_mapping_pages,
        "initial mapping pages",
    )
    gc_runs = _integer(stats.get("gc_runs"), "GC runs")
    block_erases = _integer(stats.get("block_erases"), "block erases")
    _expect(block_erases, gc_runs, "GC erase conservation")
    _expect(
        _integer(stats.get("gc_data_relocations"), "GC data relocations")
        + _integer(
            stats.get("gc_mapping_relocations"),
            "GC mapping relocations",
        ),
        relocations,
        "GC relocation-class conservation",
    )
    _expect(
        relocations
        + _integer(
            stats.get("gc_reclaimed_invalid_pages"),
            "GC reclaimed invalid pages",
        ),
        gc_runs * pages_per_block,
        "GC victim-page conservation",
    )
    page_reads = _integer(stats.get("page_reads"), "page reads")
    _expect(page_reads, relocations, "GC relocation reads")
    _expect(
        stats.get("physical_read_bytes"),
        page_reads * PAGE_SIZE,
        "physical read bytes",
    )
    expected_flash_transactions = page_reads + page_programs + block_erases
    _expect(
        stats.get("flash_scheduler_enqueues"),
        expected_flash_transactions,
        "flash scheduler enqueues",
    )
    _expect(
        stats.get("flash_scheduler_issues"),
        expected_flash_transactions,
        "flash scheduler issues",
    )
    _expect(stats.get("ecc_decode_ops"), page_reads, "ECC decodes")
    _expect(stats.get("ecc_encode_ops"), page_programs, "ECC encodes")
    _expect(stats.get("static_unmaterialized_pages"), 0, "static pages")
    total_pages = _integer(stats.get("total_pages"), "HBF total pages")
    free_pages = _integer(stats.get("free_pages"), "free pages")
    valid_pages = _integer(stats.get("valid_pages"), "valid pages")
    invalid_pages = _integer(stats.get("invalid_pages"), "invalid pages")
    _expect(
        free_pages + valid_pages + invalid_pages,
        total_pages,
        "HBF page-state conservation",
    )
    _expect(
        total_pages * PAGE_SIZE,
        capacity,
        "HBF total capacity",
    )
    result = {
        "logical_write_bytes": logical_bytes,
        "physical_write_bytes": physical_bytes,
        "waf": expected_waf,
        "data_programs": data_programs,
        "mapping_page_programs": mapping_programs,
        "gc_relocations": relocations,
        "gc_runs": gc_runs,
        "block_erases": block_erases,
        "free_pages": free_pages,
        "valid_pages": valid_pages,
        "invalid_pages": invalid_pages,
        "normalized_hbf_capacity_bytes": capacity,
        "normalized_gc_reserved_blocks_per_plane": _integer(
            hbf_config.get("gc_reserved_free_blocks_per_plane"),
            "normalized HBF GC reserve",
        ),
        "normalized_gc_low_watermark_pages": _integer(
            hbf_config.get("gc_low_watermark_pages"),
            "normalized HBF GC low watermark",
        ),
        "normalized_gc_hard_watermark_pages": _integer(
            hbf_config.get("gc_hard_watermark_pages"),
            "normalized HBF GC hard watermark",
        ),
        "physical_write_bytes_over_normalized_capacity": (
            physical_bytes / capacity
        ),
        "accounting_verified": True,
    }
    return summary, result


def _run_media_cells(
    *,
    binary: Path,
    placements: list[dict[str, Any]],
    cells: dict[str, Any],
    epoch_counts: list[int],
    validation_certificate: VerifiedCertificate | None,
) -> list[dict[str, Any]]:
    placement_by_id = {value["id"]: value for value in placements}
    results: list[dict[str, Any]] = []
    for cell_id in sorted(cells):
        cell = cells[cell_id]
        placement = placement_by_id[cell["topology"]]
        cell_dir = Path(cell["base_trace_path"]).parent
        for epochs in epoch_counts:
            trace_path = (
                Path(cell["base_trace_path"])
                if epochs == 1
                else cell_dir / f"writes-{epochs}e.trace"
            )
            trace_record = _copy_epochs(
                base_path=Path(cell["base_trace_path"]),
                output_path=trace_path,
                epochs=epochs,
                cell_id=cell_id,
            )
            expected_records = (
                cell["sampled_token_writes_per_epoch"] * 80 * epochs
            )
            trace_record["records"] = expected_records
            summary_path = cell_dir / f"summary-{epochs}e.json"
            resolved_path = cell_dir / f"resolved-{epochs}e.cfg"
            stdout_path = cell_dir / f"stdout-{epochs}e.log"
            stderr_path = cell_dir / f"stderr-{epochs}e.log"
            command = [
                str(binary.resolve()),
                "--config",
                str(placement["hardware_config"]),
                "--trace",
                str(trace_path.resolve()),
                "--initial-image-trace",
                str(Path(cell["initial_image_path"]).resolve()),
                "--scenarios",
                "all-HBF",
                "--hbf-blocks-per-plane",
                str(placement["normalized_blocks_per_plane"]),
                "--hbf-gc-reserved-free-blocks-per-plane",
                str(placement["normalized_gc_reserved_blocks_per_plane"]),
                "--hbf-gc-low-watermark-pages",
                str(placement["normalized_gc_low_watermark_pages"]),
                "--hbf-gc-hard-watermark-pages",
                str(placement["normalized_gc_hard_watermark_pages"]),
                "--hbf-hbm-write-buffer-bytes",
                "0",
                "--max-outstanding-requests",
                "1",
                "--summary-json",
                str(summary_path.resolve()),
                "--config-out",
                str(resolved_path.resolve()),
            ]
            print(
                f"[run] {cell_id} epochs={epochs} "
                f"ops={expected_records:,}",
                flush=True,
            )
            started = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            wall_seconds = time.monotonic() - started
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise WearStudyError(
                    f"{cell_id}/{epochs}e exited {completed.returncode}; "
                    f"see {stderr_path}"
                )
            summary, result = _verify_media_summary(
                summary_path=summary_path,
                trace_record=trace_record,
                expected_records=expected_records,
                expected_initial_pages=(
                    cell["sampled_blocks"] * cell["pages_per_kv_block"]
                ),
                expected_initial_mapping_pages=(
                    cell["initial_image"]["mapping_pages"]
                ),
                expected_mapping_programs=(
                    cell["expected_mapping_checkpoint_pages"]
                ),
                validation_certificate=validation_certificate,
            )
            _expect(
                result["normalized_hbf_capacity_bytes"],
                placement["normalized_mutable_pool_capacity_bytes"],
                f"{cell_id}/{epochs}e normalized capacity",
            )
            _expect(
                result["normalized_gc_reserved_blocks_per_plane"],
                placement["normalized_gc_reserved_blocks_per_plane"],
                f"{cell_id}/{epochs}e normalized GC reserve",
            )
            _expect(
                result["normalized_gc_low_watermark_pages"],
                placement["normalized_gc_low_watermark_pages"],
                f"{cell_id}/{epochs}e normalized GC low watermark",
            )
            _expect(
                result["normalized_gc_hard_watermark_pages"],
                placement["normalized_gc_hard_watermark_pages"],
                f"{cell_id}/{epochs}e normalized GC hard watermark",
            )
            result.update({
                "cell_id": cell_id,
                "topology": cell["topology"],
                "residue": cell["residue"],
                "sample_stride": cell["sample_stride"],
                "sampled_blocks": cell["sampled_blocks"],
                "sampled_token_writes_per_epoch": (
                    cell["sampled_token_writes_per_epoch"]
                ),
                "epochs": epochs,
                "operations": expected_records,
                "wall_seconds": wall_seconds,
                "trace": trace_record,
                "initial_image": cell["initial_image"],
                "summary": _artifact_record(summary_path),
                "resolved_config": _artifact_record(resolved_path),
                "validation": summary.get("validation"),
            })
            results.append(result)
            print(
                f"[pass] {cell_id} epochs={epochs} "
                f"WAF={result['waf']:.9f} "
                f"GC={result['gc_runs']:,} "
                f"reloc={result['gc_relocations']:,} "
                f"wall={wall_seconds:.1f}s",
                flush=True,
            )
    return results


def _aggregate(
    *,
    study: dict[str, Any],
    demand: dict[str, Any],
    placements: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    max_epochs = max(int(value) for value in study["epoch_counts"])
    placement_by_id = {value["id"]: value for value in placements}
    topology_rows: list[dict[str, Any]] = []
    lifetime_rows: list[dict[str, Any]] = []
    sensitivity = study["lifetime_sensitivity"]
    for topology_id, full in demand["full_population"].items():
        main = [
            value for value in results
            if value["topology"] == topology_id
            and value["epochs"] == max_epochs
        ]
        if not main:
            raise WearStudyError(f"{topology_id} has no main-epoch results")
        missing_gc = [
            value["cell_id"]
            for value in main
            if int(value["gc_runs"]) == 0
        ]
        if missing_gc:
            raise WearStudyError(
                f"{topology_id} main epochs did not exercise GC: "
                + ", ".join(missing_gc)
            )
        waf_values = [float(value["waf"]) for value in main]
        aggregate_logical = sum(
            int(value["logical_write_bytes"]) for value in main
        )
        aggregate_physical = sum(
            int(value["physical_write_bytes"]) for value in main
        )
        waf_mean = aggregate_physical / aggregate_logical
        logical_per_token = float(
            full["logical_hbf_write_bytes_per_kv_append_token"]
        )
        physical_per_token = logical_per_token * waf_mean
        placement = placement_by_id[topology_id]
        capacity = placement["usable_hbf_payload_capacity_bytes"]
        projected_window_physical = (
            full["logical_hbf_write_bytes"] * waf_mean
        )
        topology_rows.append({
            "topology": topology_id,
            "hbf_stacks": placement["stacks"],
            "usable_hbf_capacity_bytes": capacity,
            "hot_kv_blocks": full["hot_kv_blocks"],
            "cold_kv_blocks": full["cold_kv_blocks"],
            "cold_kv_token_writes": full["cold_kv_token_writes"],
            "cold_kv_token_fraction": full["cold_kv_token_fraction"],
            "logical_hbf_write_bytes": full["logical_hbf_write_bytes"],
            "logical_write_bytes_per_kv_append_token": logical_per_token,
            "measured_waf_min": min(waf_values),
            "measured_waf_mean": waf_mean,
            "measured_waf_mean_kind": "logical_byte_weighted",
            "measured_waf_max": max(waf_values),
            "projected_physical_write_bytes": projected_window_physical,
            "projected_physical_write_bytes_per_kv_append_token": (
                physical_per_token
            ),
            "projected_physical_write_bytes_per_request": (
                projected_window_physical / 256
            ),
            "projected_physical_write_over_usable_hbf_capacity": (
                projected_window_physical / capacity
            ),
            "main_epochs": max_epochs,
            "sampling_residues": len(main),
            "gc_runs_sum": sum(int(value["gc_runs"]) for value in main),
            "gc_relocations_sum": sum(
                int(value["gc_relocations"]) for value in main
            ),
        })
        for raw_rate in sensitivity["kv_append_tokens_per_second"]:
            rate = float(raw_rate)
            capacity_writes_per_day = (
                physical_per_token * rate * 86400 / capacity
            )
            for raw_cycles in sensitivity["assumed_media_cycles"]:
                cycles = float(raw_cycles)
                years = (
                    cycles / capacity_writes_per_day / 365.25
                    if capacity_writes_per_day > 0
                    else None
                )
                lifetime_rows.append({
                    "topology": topology_id,
                    "kv_append_tokens_per_second": rate,
                    "assumed_media_cycles": cycles,
                    "physical_write_bytes_per_second": (
                        physical_per_token * rate
                    ),
                    "usable_capacity_writes_per_day": capacity_writes_per_day,
                    "uniform_wear_years": years,
                    "status": "uncalibrated_first_order_sensitivity",
                })
    return topology_rows, lifetime_rows


def _write_report(
    path: Path,
    topology_rows: list[dict[str, Any]],
    lifetime_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Frontier 70B HBF WAF and Wear-Demand Study",
        "",
        "The workload demand is exact for the audited 256-request burst.  "
        "Media WAF is the logical-byte-weighted mean over all predeclared "
        "capacity-scaled block residues at the longest epoch count.",
        "",
        "| Topology | Cold KV tokens | Cold fraction | Logical HBF writes | "
        "WAF min/weighted mean/max | Projected physical writes | "
        "Usable-HBF-capacity writes/window |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in topology_rows:
        lines.append(
            f"| {row['topology']} | {row['cold_kv_token_writes']:,} | "
            f"{100 * row['cold_kv_token_fraction']:.2f}% | "
            f"{row['logical_hbf_write_bytes'] / 2**30:.3f} GiB | "
            f"{row['measured_waf_min']:.6f}/"
            f"{row['measured_waf_mean']:.6f}/"
            f"{row['measured_waf_max']:.6f} | "
            f"{row['projected_physical_write_bytes'] / 2**30:.3f} GiB | "
            f"{row['projected_physical_write_over_usable_hbf_capacity']:.6f} |"
        )
    lines.extend([
        "",
        "## Lifetime sensitivity",
        "",
        "`uniform_wear_years = assumed_media_cycles × usable_capacity / "
        "physical_write_rate`. These rows are not device lifetime predictions.",
        "",
        "| Topology | New KV tokens/s | Assumed cycles | Capacity writes/day | "
        "Uniform-wear years |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in lifetime_rows:
        lines.append(
            f"| {row['topology']} | "
            f"{row['kv_append_tokens_per_second']:.0f} | "
            f"{row['assumed_media_cycles']:.0f} | "
            f"{row['usable_capacity_writes_per_day']:.6f} | "
            f"{row['uniform_wear_years']:.3f} |"
        )
    lines.extend([
        "",
        "## Claim boundary",
        "",
        "- Canonical WAF includes data programs, mapping checkpoints, and GC "
        "relocations; it excludes OOB/ECC/link bytes.",
        "- Read-only weights contribute zero logical HBF writes.",
        "- The scaled media geometry removes the exact immutable-weight "
        "block reservation, and scales GC reserve/watermarks mechanically; "
        "it does not treat read-only weights as mutable FTL pages.",
        "- Frontier timing is dummy; this study reports no TTFT, TPOT, SLO, "
        "or end-to-end token throughput.",
        "- The media replay is capacity-scaled and HBF timing/endurance is "
        "uncalibrated. Lifetime rows assume uniform wear and omit retention, "
        "disturb, bad-block growth, and P/E failure physics.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_study(
    *,
    binary: Path,
    audit_path: Path,
    manifest_path: Path,
    object_map_path: Path,
    config_path: Path,
    out_dir: Path,
    validation_certificate: VerifiedCertificate | None,
) -> dict[str, Any]:
    study = _load_study_config(config_path)
    source = _verify_source_audit(audit_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    placements = _compile_placements(
        source=source,
        study=study,
        manifest_path=manifest_path,
        object_map_path=object_map_path,
        out_dir=out_dir,
    )
    demand, cells = _generate_base_traces(
        source=source,
        placements=placements,
        sampling=_mapping(study.get("sampling"), "study.sampling"),
        workload=_mapping(study.get("workload"), "study.workload"),
        out_dir=out_dir,
    )
    _expect(
        demand["scheduled_kv_append_tokens"],
        source["audit"]["accounting"]["scheduled_tokens"],
        "scheduled KV-append tokens",
    )
    results = _run_media_cells(
        binary=binary,
        placements=placements,
        cells=cells,
        epoch_counts=[int(value) for value in study["epoch_counts"]],
        validation_certificate=validation_certificate,
    )
    topology_rows, lifetime_rows = _aggregate(
        study=study,
        demand=demand,
        placements=placements,
        results=results,
    )
    demand_csv = out_dir / "write-demand.csv"
    _write_csv_atomic(
        demand_csv,
        (
            "topology",
            "hot_kv_blocks",
            "cold_kv_blocks",
            "cold_kv_token_writes",
            "cold_kv_token_fraction",
            "logical_hbf_write_bytes",
            "logical_hbf_write_bytes_per_kv_append_token",
            "logical_hbf_write_bytes_per_request",
            "logical_hbf_write_bytes_over_usable_hbf_capacity",
            "prefill_cold_kv_token_writes",
            "decode_cold_kv_token_writes",
            "first_cold_write_batch",
            "last_cold_write_batch",
            "touched_cold_kv_blocks",
            "unique_block_token_positions",
            "repeated_block_token_writes",
        ),
        [
            {"topology": topology, **row}
            for topology, row in demand["full_population"].items()
        ],
    )
    media_csv = out_dir / "media-waf.csv"
    media_fields = (
        "cell_id",
        "topology",
        "residue",
        "sample_stride",
        "sampled_blocks",
        "sampled_token_writes_per_epoch",
        "epochs",
        "operations",
        "logical_write_bytes",
        "physical_write_bytes",
        "waf",
        "data_programs",
        "mapping_page_programs",
        "gc_relocations",
        "gc_runs",
        "block_erases",
        "normalized_hbf_capacity_bytes",
        "normalized_gc_reserved_blocks_per_plane",
        "normalized_gc_low_watermark_pages",
        "normalized_gc_hard_watermark_pages",
        "physical_write_bytes_over_normalized_capacity",
        "wall_seconds",
        "accounting_verified",
    )
    _write_csv_atomic(
        media_csv,
        media_fields,
        [{field: row[field] for field in media_fields} for row in results],
    )
    lifetime_csv = out_dir / "lifetime-sensitivity.csv"
    lifetime_fields = (
        "topology",
        "kv_append_tokens_per_second",
        "assumed_media_cycles",
        "physical_write_bytes_per_second",
        "usable_capacity_writes_per_day",
        "uniform_wear_years",
        "status",
    )
    _write_csv_atomic(lifetime_csv, lifetime_fields, lifetime_rows)
    report_path = out_dir / "report.md"
    _write_report(report_path, topology_rows, lifetime_rows)
    receipt = {
        "schema": OUTPUT_SCHEMA,
        "result": "pass",
        "validation": (
            validation_certificate.summary_block()
            if validation_certificate is not None
            else EXPLORATORY_VALIDATION
        ),
        "claim_scope": {
            "eligible": "hbf_write_demand_and_uncalibrated_wear_sensitivity",
            "paper_result_eligible": False,
            "time_based_throughput": False,
            "ttft_tpot_slo": False,
            "absolute_device_lifetime": False,
        },
        "waf_definition": WAF_DEFINITION,
        "method": {
            "logical_demand": "complete_audited_frontier_kv_lifecycle",
            "media_replay": "capacity_scaled_block_stratified_exact_order",
            "initial_image_is_workload_write": False,
            "sample_stride_blocks": study["sampling"]["block_stride"],
            "sample_residues": study["sampling"]["residues"],
            "epoch_counts": study["epoch_counts"],
            "scaled_static_weight_policy": (
                "remove_exact_full_system_static_weight_block_reservation"
            ),
            "scaled_gc_policy": (
                "nearest_reserved_blocks_and_ceil_watermarks_by_stride"
            ),
            "lifetime_model": "uniform_wear_first_order_sensitivity_only",
        },
        "inputs": {
            "study_config": _artifact_record(config_path),
            "frontier_audit": source["audit_record"],
            "stage_batch_ledger": _artifact_record(source["ledger_path"]),
            "kv_block_lifecycle": _artifact_record(source["lifecycle_path"]),
            "placement_manifest": _artifact_record(manifest_path),
            "object_map": _artifact_record(object_map_path),
            "scenario_compare": _artifact_record(binary),
        },
        "source_workload": {
            "requests": 256,
            "total_request_tokens": source["total_request_tokens"],
            "scheduled_kv_append_tokens": demand["scheduled_kv_append_tokens"],
            "arrival_span_s": source["request_duration_s"],
            "arrival_span_used_as_transaction_timing": False,
        },
        "placements": [
            {
                "topology": value["id"],
                "receipt": value["receipt_record"],
                "hardware_config": value["hardware_config_record"],
                "hot_kv_blocks": value["hot_kv_blocks"],
                "cold_kv_blocks": value["cold_kv_blocks"],
                "usable_hbf_payload_capacity_bytes": (
                    value["usable_hbf_payload_capacity_bytes"]
                ),
                "model_weight_backing_pages": (
                    value["model_weight_backing_pages"]
                ),
                "full_static_weight_reserved_blocks": (
                    value["full_static_weight_reserved_blocks"]
                ),
                "full_writable_pool_blocks": (
                    value["full_writable_pool_blocks"]
                ),
                "full_writable_pool_payload_bytes": (
                    value["full_writable_pool_payload_bytes"]
                ),
                "normalized_blocks_per_plane": (
                    value["normalized_blocks_per_plane"]
                ),
                "normalized_gc_reserved_blocks_per_plane": (
                    value["normalized_gc_reserved_blocks_per_plane"]
                ),
                "normalized_gc_low_watermark_pages": (
                    value["normalized_gc_low_watermark_pages"]
                ),
                "normalized_gc_hard_watermark_pages": (
                    value["normalized_gc_hard_watermark_pages"]
                ),
                "normalized_mutable_pool_capacity_bytes": (
                    value["normalized_mutable_pool_capacity_bytes"]
                ),
                "normalized_mutable_pool_block_error_at_full_scale": (
                    value[
                        "normalized_mutable_pool_block_error_at_full_scale"
                    ]
                ),
            }
            for value in placements
        ],
        "write_demand": demand,
        "media_results": results,
        "topology_summary": topology_rows,
        "lifetime_sensitivity": lifetime_rows,
        "artifacts": {
            "write_demand_csv": _artifact_record(demand_csv),
            "media_waf_csv": _artifact_record(media_csv),
            "lifetime_sensitivity_csv": _artifact_record(lifetime_csv),
            "report_markdown": _artifact_record(report_path),
        },
        "limitations": [
            "media replay is block-stratified and capacity-scaled by 64",
            "integer block/plane scaling introduces a recorded bounded "
            "capacity error",
            "Frontier memory traffic is model-conditioned, not measured GPU IO",
            "Frontier compute and communication timing is dummy",
            "HBF timing and endurance parameters are not L4 calibrated",
            "no retention, disturb, bad-block growth, P/E failure, thermal, "
            "power, or calibrated wear-leveling model",
        ],
    }
    receipt_path = out_dir / "study.receipt.json"
    _write_json_atomic(receipt_path, receipt)
    print(f"wrote: {receipt_path}", flush=True)
    print(f"wrote: {report_path}", flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--placement-manifest", type=Path, required=True)
    parser.add_argument("--object-map", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--validation-certificate", type=Path)
    args = parser.parse_args()
    binary = args.scenario_compare.resolve()
    if not binary.is_file():
        parser.error(f"scenario_compare does not exist: {binary}")
    validation_certificate: VerifiedCertificate | None = None
    if args.validation_certificate is not None:
        try:
            validation_certificate = verify_certificate(
                args.validation_certificate,
                repository=ROOT,
                scenario_compare=binary,
            )
        except CertificateError as error:
            parser.error(f"invalid validation certificate: {error}")
    try:
        run_study(
            binary=binary,
            audit_path=args.audit.resolve(),
            manifest_path=args.placement_manifest.resolve(),
            object_map_path=args.object_map.resolve(),
            config_path=args.study_config.resolve(),
            out_dir=args.out_dir.resolve(),
            validation_certificate=validation_certificate,
        )
    except (OSError, WearStudyError, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
