"""Strict validation-case and ledger contracts.

This module is deliberately limited to data validation and serialization.  It
contains no production-model scheduling, mapping, or accounting helpers; the
independent oracle implements those rules separately.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


CASE_SCHEMA = {"name": "hbfsim.verification.case", "version": 1}
LEDGER_SCHEMA = {"name": "hbfsim.verification.ledger", "version": 1}
CASE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
UINT64_MAX = (1 << 64) - 1


def hbf_buffer_hbm_config() -> dict[str, Any]:
    """Explicit shared-buffer hardware for standalone HBF oracle cases."""
    return resolve_hbm_config({
        "capacity_bytes": 48 * 1024**3,
        "channels_per_stack": 32,
        "bank_groups_per_pseudo_channel": 16,
        "pin_rate_Gbps": 8.0,
    })

HBM_DEFAULTS: dict[str, int | float | bool] = {
    "capacity_bytes": 128 * 1024 * 1024 * 1024,
    "stacks": 1,
    "channels_per_stack": 8,
    "pseudo_channels_per_channel": 2,
    "bank_groups_per_pseudo_channel": 4,
    "banks_per_group": 4,
    "channel_row_size_bytes": 2048,
    "channel_width_bits": 64,
    "burst_length": 8,
    "pin_rate_Gbps": 6.4,
    "data_rate_per_command_clock": 4,
    "address_mapping_ns": 0.0,
    "read_latency_ns": 42.0,
    "write_latency_ns": 38.0,
    "read_to_write_ns": 8.0,
    "write_to_read_ns": 8.0,
    "bandwidth_efficiency": 0.94,
    "queue_depth": 32,
    "interleave_bytes": 0,
    "service_quantum_bytes": 4096,
    # channel-aggregate-v2: consecutive pseudo-channel lanes share one service
    # group calendar; 1 is the fine-grained sensitivity setting.
    "service_group_channels": 4,
}

HBM_INTEGER_FIELDS = {
    "capacity_bytes", "stacks", "channels_per_stack", "pseudo_channels_per_channel",
    "bank_groups_per_pseudo_channel", "banks_per_group", "channel_row_size_bytes",
    "channel_width_bits", "burst_length", "data_rate_per_command_clock",
    "queue_depth", "service_quantum_bytes", "service_group_channels",
}
HBM_U64_FIELDS = {"capacity_bytes", "channel_row_size_bytes", "service_quantum_bytes"}
HBM_BOOL_FIELDS = set()
HBM_NONNEGATIVE_FLOAT_FIELDS = {"address_mapping_ns", "read_latency_ns", "write_latency_ns", "read_to_write_ns", "write_to_read_ns"}

HBF_DEFAULTS: dict[str, int | float | bool] = {
    "speed_grade": 2,
    "host_gc_decision_ns": 50.0,
    "stacks": 1,
    "channels_per_stack": 16,
    "dies_per_channel": 1,
    "planes_per_die": 16,
    "blocks_per_plane": 2048,
    "pages_per_block": 256,
    "page_size_bytes": 4096,
    "oob_bytes_per_page": 224,
    "media_lanes_per_plane": 1,
    "page_buffer_banks_per_plane": 2,
    "t_read_page_ns": 4000.0,
    "t_program_page_ns": 75000.0,
    "t_erase_block_ns": 2_000_000.0,
    "ecc_decode_latency_ns": 500.0,
    "ecc_encode_latency_ns": 500.0,
    "ecc_decode_raw_bandwidth_GBps_per_die": 101.25,
    "ecc_encode_raw_bandwidth_GBps_per_die": 101.25,
    "channel_bandwidth_GBps": 101.25,
    "tsv_bandwidth_GBps": 1644.0,
    "media_lane_bandwidth_GBps": 2048.0,
    "logic_sram_bandwidth_GBps": 2048.0,
    "page_buffer_bandwidth_GBps": 2048.0,
    "command_address_bytes": 64,
    "logic_scheduler_issue_ns": 2.0,
    "address_generation_ns": 5.0,
    "ctrl_dram_latency_ns": 100.0,
    "ctrl_dram_issue_ns": 1.0,
    "mapping_update_ns": 25.0,
    # Host mapping control work charged per lookup/update on the shared host
    # compute-worker pool (one worker per stack). Zero charges no stage.
    "mapping_control_compute_ns": 5.0,
    "free_page_allocation_ns": 10.0,
    "flash_tsu_issue_ns": 10.0,
    "ctrl_dram_bytes": 0,
    "mapping_entries_per_page": 512,
    "page_read_queue_depth_per_stack": 4096,
    "write_coalescing_enabled": False,
    "write_buffer_completion_requires_flush": False,
    "write_buffer_pages": 1024,
    "write_buffer_flush_threshold_pages": 0,
    "auto_gc_enabled": True,
    "gc_low_watermark_pages": 0,
    "gc_hard_watermark_pages": 0,
    "gc_reserved_free_blocks_per_plane": 2,
    "gc_wear_leveling_weight": 0.0,
}
HBF_U32_FIELDS = {
    "speed_grade",
    "stacks",
    "channels_per_stack",
    "dies_per_channel",
    "planes_per_die",
    "blocks_per_plane",
    "pages_per_block",
    "media_lanes_per_plane",
    "page_buffer_banks_per_plane",
}
HBF_U64_FIELDS = {
    key
    for key, value in HBF_DEFAULTS.items()
    if isinstance(value, int) and not isinstance(value, bool)
} - HBF_U32_FIELDS
HBF_BOOL_FIELDS = {
    "write_coalescing_enabled",
    "write_buffer_completion_requires_flush",
    "auto_gc_enabled",
}
HBF_ZERO_ALLOWED_INTEGER_FIELDS = {
    "oob_bytes_per_page",
    "ctrl_dram_bytes",
    "write_buffer_pages",
    "write_buffer_flush_threshold_pages",
    "gc_low_watermark_pages",
    "gc_hard_watermark_pages",
    "gc_reserved_free_blocks_per_plane",
}
HBF_NONNEGATIVE_FLOAT_FIELDS = {
    "gc_wear_leveling_weight",
    "mapping_control_compute_ns",
}
EXTERNAL_DEFAULTS: dict[str, int | float | str] = {
    "kind": "cxl-memory",
    "capacity_bytes": 256 * 1024 * 1024 * 1024,
    "page_size_bytes": 4096,
    "request_segment_bytes": 4096,
    "media_channels": 8,
    "media_read_queues": 1,
    "media_write_queues": 1,
    "max_outstanding_requests": 512,
    "controller_issue_ns": 2.0,
    "controller_processing_ns": 20.0,
    "media_read_latency_ns": 90.0,
    "media_write_latency_ns": 90.0,
    "media_read_bandwidth_GBps": 204.8,
    "media_write_bandwidth_GBps": 204.8,
    "m2s_bandwidth_GBps": 36.0,
    "s2m_bandwidth_GBps": 36.0,
    "one_way_propagation_ns": 75.0,
    "command_bytes": 64,
    "completion_bytes": 16,
}
EXTERNAL_U64_FIELDS = {
    "capacity_bytes",
    "page_size_bytes",
    "request_segment_bytes",
}
EXTERNAL_U32_FIELDS = {
    "media_channels",
    "media_read_queues",
    "media_write_queues",
    "max_outstanding_requests",
    "command_bytes",
    "completion_bytes",
}
EXTERNAL_NONNEGATIVE_FLOAT_FIELDS = {
    "controller_issue_ns",
    "controller_processing_ns",
    "media_read_latency_ns",
    "media_write_latency_ns",
    "one_way_propagation_ns",
}
HYBRID_KNOB_DEFAULTS = {
    "max_outstanding_requests": 0,
}


class ContractError(ValueError):
    """A validation artifact violates its versioned data contract."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json_strict(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{path}: {error}") from error
    except ContractError as error:
        raise ContractError(f"{path}: {error}") from error


def _require_exact_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    where: str,
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ContractError(f"{where}: missing keys: {sorted(missing)}")
    if unknown:
        raise ContractError(f"{where}: unknown keys: {sorted(unknown)}")


def _require_uint(
    value: Any,
    where: str,
    *,
    positive: bool = False,
    maximum: int = UINT64_MAX,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{where}: expected integer")
    lower = 1 if positive else 0
    if not lower <= value <= maximum:
        qualifier = "positive " if positive else ""
        bits = 32 if maximum == (1 << 32) - 1 else 64
        raise ContractError(f"{where}: expected {qualifier}uint{bits}")
    return value


def _require_finite(
    value: Any,
    where: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}: expected number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ContractError(f"{where}: expected finite number")
    if positive and not parsed > 0.0:
        raise ContractError(f"{where}: expected positive number")
    if nonnegative and parsed < 0.0:
        raise ContractError(f"{where}: expected nonnegative number")
    return parsed


def resolve_hbm_config(overrides: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise ContractError("config: expected object")
    unknown = overrides.keys() - HBM_DEFAULTS.keys()
    if unknown:
        raise ContractError(f"config: unknown HBM keys: {sorted(unknown)}")
    resolved = dict(HBM_DEFAULTS)
    for key, value in overrides.items():
        if key in HBM_BOOL_FIELDS:
            if not isinstance(value, bool):
                raise ContractError(f"config.{key}: expected boolean")
            resolved[key] = value
        elif key in HBM_INTEGER_FIELDS:
            resolved[key] = _require_uint(
                value,
                f"config.{key}",
                positive=True,
                maximum=(
                    UINT64_MAX
                    if key in HBM_U64_FIELDS
                    else (1 << 32) - 1
                ),
            )
        elif key in HBM_NONNEGATIVE_FLOAT_FIELDS:
            resolved[key] = _require_finite(
                value, f"config.{key}", nonnegative=True)
        else:
            resolved[key] = _require_finite(
                value, f"config.{key}", positive=True)
    return resolved


def resolve_hbf_config(overrides: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise ContractError("config: expected object")
    unknown = overrides.keys() - HBF_DEFAULTS.keys()
    if unknown:
        raise ContractError(f"config: unknown HBF keys: {sorted(unknown)}")
    resolved = dict(HBF_DEFAULTS)
    for key, value in overrides.items():
        if key in HBF_BOOL_FIELDS:
            if not isinstance(value, bool):
                raise ContractError(f"config.{key}: expected boolean")
            resolved[key] = value
        elif key in HBF_U32_FIELDS | HBF_U64_FIELDS:
            resolved[key] = _require_uint(
                value,
                f"config.{key}",
                positive=key not in HBF_ZERO_ALLOWED_INTEGER_FIELDS,
                maximum=(
                    (1 << 32) - 1
                    if key in HBF_U32_FIELDS
                    else UINT64_MAX
                ),
            )
        elif key in HBF_NONNEGATIVE_FLOAT_FIELDS:
            resolved[key] = _require_finite(
                value, f"config.{key}", nonnegative=True)
        else:
            resolved[key] = _require_finite(
                value, f"config.{key}", positive=True)

    if resolved["pages_per_block"] > 1024:
        raise ContractError("config.pages_per_block: maximum is 1024")
    if resolved["page_size_bytes"] != 4096:
        raise ContractError("config.page_size_bytes: OCP HBF requires 4096")
    if resolved["oob_bytes_per_page"] >= resolved["page_size_bytes"]:
        raise ContractError(
            "config.oob_bytes_per_page must be smaller than page_size_bytes")
    if (
        resolved["write_coalescing_enabled"]
        and resolved["write_buffer_pages"] == 0
    ):
        raise ContractError(
            "config.write_buffer_pages must be positive when coalescing")
    if (
        resolved["gc_reserved_free_blocks_per_plane"]
        >= resolved["blocks_per_plane"]
    ):
        raise ContractError(
            "config.gc_reserved_free_blocks_per_plane must be smaller "
            "than blocks_per_plane")
    page_wire_bytes = (
        resolved["page_size_bytes"] + resolved["oob_bytes_per_page"])
    if (
        resolved["ecc_decode_latency_ns"]
        < page_wire_bytes
        / resolved["ecc_decode_raw_bandwidth_GBps_per_die"]
    ):
        raise ContractError(
            "config.ecc_decode_latency_ns is shorter than one codeword issue")
    if (
        resolved["ecc_encode_latency_ns"]
        < page_wire_bytes
        / resolved["ecc_encode_raw_bandwidth_GBps_per_die"]
    ):
        raise ContractError(
            "config.ecc_encode_latency_ns is shorter than one codeword issue")

    pages_per_stack = (
        resolved["channels_per_stack"]
        * resolved["dies_per_channel"]
        * resolved["planes_per_die"]
        * resolved["blocks_per_plane"]
        * resolved["pages_per_block"]
    )
    total_pages = pages_per_stack * resolved["stacks"]
    if total_pages > UINT64_MAX:
        raise ContractError("config: HBF total page count overflows uint64")
    if total_pages * resolved["page_size_bytes"] > UINT64_MAX:
        raise ContractError("config: HBF capacity bytes overflow uint64")
    mapping_pages_per_stack = (
        pages_per_stack + resolved["mapping_entries_per_page"] - 1
    ) // resolved["mapping_entries_per_page"]
    mapping_bytes_per_stack = (
        mapping_pages_per_stack * resolved["page_size_bytes"])
    write_buffer_bytes_per_stack = (
        resolved["write_buffer_pages"] * resolved["page_size_bytes"]
        if resolved["write_coalescing_enabled"]
        else 0
    )
    required_ctrl_dram_per_stack = (
        mapping_bytes_per_stack + write_buffer_bytes_per_stack +
        (4096 if resolved["auto_gc_enabled"] else 0))
    required_ctrl_dram = (
        required_ctrl_dram_per_stack * resolved["stacks"])
    if resolved["ctrl_dram_bytes"] == 0:
        resolved["ctrl_dram_bytes"] = required_ctrl_dram
    elif (
        resolved["ctrl_dram_bytes"] // resolved["stacks"]
        < required_ctrl_dram_per_stack
    ):
        raise ContractError(
            "config.ctrl_dram_bytes cannot hold the resident mapping "
            "and configured write buffer")
    return resolved


def resolve_hybrid_config(overrides: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise ContractError("config: expected object")
    _require_exact_keys(
        overrides,
        {"hbm", "hbf", "policy"},
        {"knobs"},
        "config",
    )
    hbm = resolve_hbm_config(overrides["hbm"])
    hbf = resolve_hbf_config(overrides["hbf"])

    policy = overrides["policy"]
    if not isinstance(policy, dict):
        raise ContractError("config.policy: expected object")
    _require_exact_keys(
        policy,
        {"kind", "read_boundary"},
        set(),
        "config.policy",
    )
    if policy["kind"] != "flat":
        raise ContractError(
            "config.policy.kind: hybrid oracle v1 requires flat")
    read_boundary = _require_uint(
        policy["read_boundary"],
        "config.policy.read_boundary",
    )
    if read_boundary % hbf["page_size_bytes"] != 0:
        raise ContractError(
            "config.policy.read_boundary must be HBF-page aligned")
    if read_boundary > hbm["capacity_bytes"]:
        raise ContractError(
            "config.policy.read_boundary exceeds HBM capacity")
    if read_boundary > (
        hbf["stacks"]
        * hbf["channels_per_stack"]
        * hbf["dies_per_channel"]
        * hbf["planes_per_die"]
        * hbf["blocks_per_plane"]
        * hbf["pages_per_block"]
        * hbf["page_size_bytes"]
    ):
        raise ContractError(
            "config.policy.read_boundary exceeds HBF capacity")

    knob_overrides = overrides.get("knobs", {})
    if not isinstance(knob_overrides, dict):
        raise ContractError("config.knobs: expected object")
    unknown_knobs = knob_overrides.keys() - HYBRID_KNOB_DEFAULTS.keys()
    if unknown_knobs:
        raise ContractError(
            f"config.knobs: unknown keys: {sorted(unknown_knobs)}")
    knobs = dict(HYBRID_KNOB_DEFAULTS)
    for key, value in knob_overrides.items():
        knobs[key] = _require_uint(value, f"config.knobs.{key}")
    if knobs["max_outstanding_requests"] != 0:
        raise ContractError(
            "config.knobs.max_outstanding_requests: "
            "hybrid oracle v1 requires an unbounded window")

    return {
        "hbm": hbm,
        "hbf": hbf,
        "policy": {
            "kind": "flat",
            "read_boundary": read_boundary,
        },
        "knobs": knobs,
    }


def resolve_external_config(overrides: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise ContractError("config: expected object")
    unknown = overrides.keys() - EXTERNAL_DEFAULTS.keys()
    if unknown:
        raise ContractError(
            f"config: unknown external keys: {sorted(unknown)}"
        )
    resolved = dict(EXTERNAL_DEFAULTS)
    for key, value in overrides.items():
        if key == "kind":
            if value not in {"host-dram", "cxl-memory", "nvme-ssd", "cxl-ssd"}:
                raise ContractError(
                    "config.kind: expected host-dram, cxl-memory, "
                    "nvme-ssd, or cxl-ssd"
                )
            resolved[key] = value
        elif key in EXTERNAL_U64_FIELDS:
            resolved[key] = _require_uint(
                value,
                f"config.{key}",
                positive=True,
            )
        elif key in EXTERNAL_U32_FIELDS:
            resolved[key] = _require_uint(
                value,
                f"config.{key}",
                positive=True,
                maximum=(1 << 32) - 1,
            )
        else:
            resolved[key] = _require_finite(
                value,
                f"config.{key}",
                nonnegative=key in EXTERNAL_NONNEGATIVE_FLOAT_FIELDS,
                positive=key not in EXTERNAL_NONNEGATIVE_FLOAT_FIELDS,
            )
    if resolved["capacity_bytes"] % resolved["page_size_bytes"] != 0:
        raise ContractError(
            "config.capacity_bytes must be page aligned"
        )
    if (
        resolved["request_segment_bytes"] < resolved["page_size_bytes"]
        or resolved["request_segment_bytes"]
        % resolved["page_size_bytes"] != 0
    ):
        raise ContractError(
            "config.request_segment_bytes must be a positive multiple of "
            "config.page_size_bytes"
        )
    return resolved


def validate_case(data: Any, *, source: str = "<memory>") -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ContractError(f"{source}: case root must be an object")
    _require_exact_keys(
        data,
        {"schema", "case_id", "model", "comparison", "config", "requests"},
        {"inspect_addresses", "initial_state", "notes"},
        source,
    )
    if data["schema"] != CASE_SCHEMA:
        raise ContractError(f"{source}: unsupported case schema")
    notes = data.get("notes", [])
    if not isinstance(notes, list) or not all(
        isinstance(note, str) and note.strip() for note in notes
    ):
        raise ContractError(
            f"{source}.notes: expected an array of non-empty strings")
    case_id = data["case_id"]
    if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
        raise ContractError(f"{source}: invalid case_id")
    if data["model"] not in {"hbm", "hbf", "hybrid", "external"}:
        raise ContractError(f"{source}: unsupported model {data['model']!r}")
    model = data["model"]

    comparison = data["comparison"]
    if not isinstance(comparison, dict):
        raise ContractError(f"{source}.comparison: expected object")
    _require_exact_keys(
        comparison,
        {"time_abs_tolerance_ns", "time_rel_tolerance"},
        set(),
        f"{source}.comparison",
    )
    abs_tolerance = _require_finite(
        comparison["time_abs_tolerance_ns"],
        f"{source}.comparison.time_abs_tolerance_ns",
        nonnegative=True,
    )
    rel_tolerance = _require_finite(
        comparison["time_rel_tolerance"],
        f"{source}.comparison.time_rel_tolerance",
        nonnegative=True,
    )

    if model == "hbm":
        resolved_config = resolve_hbm_config(data["config"])
    elif model == "hbf":
        resolved_config = resolve_hbf_config(data["config"])
    elif model == "external":
        resolved_config = resolve_external_config(data["config"])
    else:
        resolved_config = resolve_hybrid_config(data["config"])
    initial_state = data.get("initial_state", {})
    if not isinstance(initial_state, dict):
        raise ContractError(f"{source}.initial_state: expected object")
    if model in {"hbm", "hybrid", "external"}:
        _require_exact_keys(
            initial_state, set(), set(), f"{source}.initial_state")
        normalized_initial_state: dict[str, Any] = {}
    else:
        _require_exact_keys(
            initial_state,
            set(),
            {"prepopulate_lpns"},
            f"{source}.initial_state",
        )
        prepopulate = initial_state.get("prepopulate_lpns", [])
        if not isinstance(prepopulate, list):
            raise ContractError(
                f"{source}.initial_state.prepopulate_lpns: expected array")
        normalized_prepopulate = [
            _require_uint(
                lpn,
                f"{source}.initial_state.prepopulate_lpns[{index}]",
            )
            for index, lpn in enumerate(prepopulate)
        ]
        if len(normalized_prepopulate) != len(set(normalized_prepopulate)):
            raise ContractError(
                f"{source}.initial_state.prepopulate_lpns: duplicates")
        normalized_initial_state = {
            "prepopulate_lpns": normalized_prepopulate}
    inspect_addresses = data.get("inspect_addresses", [])
    if not isinstance(inspect_addresses, list):
        raise ContractError(f"{source}.inspect_addresses: expected array")
    inspected = [
        _require_uint(value, f"{source}.inspect_addresses[{index}]")
        for index, value in enumerate(inspect_addresses)
    ]
    if model == "hybrid" and inspected:
        raise ContractError(
            f"{source}.inspect_addresses: hybrid oracle v1 requires none")

    requests = data["requests"]
    if not isinstance(requests, list):
        raise ContractError(f"{source}.requests: expected array")
    if model in {"hybrid", "external"} and not requests:
        raise ContractError(
            f"{source}.requests: {model} oracle v1 requires at least one")
    normalized_requests: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_arrival = 0.0
    for index, request in enumerate(requests):
        where = f"{source}.requests[{index}]"
        if not isinstance(request, dict):
            raise ContractError(f"{where}: expected object")
        _require_exact_keys(
            request,
            {"id", "arrival_ns", "op", "address_space", "addr", "bytes"},
            set(),
            where,
        )
        request_id = request["id"]
        if (
            not isinstance(request_id, str)
            or not REQUEST_ID_RE.fullmatch(request_id)
        ):
            raise ContractError(f"{where}.id: invalid request ID")
        if request_id in seen_ids:
            raise ContractError(f"{where}.id: duplicate request ID")
        seen_ids.add(request_id)
        arrival = _require_finite(
            request["arrival_ns"], f"{where}.arrival_ns", nonnegative=True)
        if index and arrival < previous_arrival:
            raise ContractError(
                f"{where}.arrival_ns: requests must be nondecreasing")
        previous_arrival = arrival
        allowed_ops = {
            "hbm": {"read", "write"},
            "hbf": {"read", "write", "drain"},
            "hybrid": {"read"},
            "external": {"read", "write"},
        }[model]
        if request["op"] not in allowed_ops:
            expected_ops = {
                "hbm": "read or write",
                "hbf": "read, write, or drain",
                "hybrid": "read",
                "external": "read or write",
            }[model]
            raise ContractError(f"{where}.op: expected {expected_ops}")
        is_drain = model == "hbf" and request["op"] == "drain"
        allowed_address_spaces = (
            {"logical"}
            if model in {"hybrid", "external"}
            else ({"internal"} if is_drain else {"logical", "physical"})
        )
        if request["address_space"] not in allowed_address_spaces:
            expected_address_space = (
                "logical"
                if model in {"hybrid", "external"}
                else (
                    "internal"
                    if is_drain
                    else "logical or physical"
                )
            )
            raise ContractError(
                f"{where}.address_space: expected {expected_address_space}")
        addr = _require_uint(request["addr"], f"{where}.addr")
        byte_count = _require_uint(
            request["bytes"],
            f"{where}.bytes",
            positive=not is_drain,
        )
        if is_drain and (addr != 0 or byte_count != 0):
            raise ContractError(
                f"{where}: drain requires addr=0 and bytes=0")
        if model == "hbm":
            capacity_bytes = resolved_config["capacity_bytes"]
        elif model == "hbf":
            capacity_bytes = (
                resolved_config["stacks"]
                * resolved_config["channels_per_stack"]
                * resolved_config["dies_per_channel"]
                * resolved_config["planes_per_die"]
                * resolved_config["blocks_per_plane"]
                * resolved_config["pages_per_block"]
                * resolved_config["page_size_bytes"]
            )
        elif model == "external":
            capacity_bytes = resolved_config["capacity_bytes"]
        else:
            capacity_bytes = 0
            cursor = addr
            remaining = byte_count
            page_size = resolved_config["hbf"]["page_size_bytes"]
            boundary = resolved_config["policy"]["read_boundary"]
            while remaining:
                segment_bytes = min(
                    remaining, page_size - cursor % page_size)
                segment_capacity = (
                    resolved_config["hbm"]["capacity_bytes"]
                    if cursor < boundary
                    else (
                        resolved_config["hbf"]["stacks"]
                        * resolved_config["hbf"]["channels_per_stack"]
                        * resolved_config["hbf"]["dies_per_channel"]
                        * resolved_config["hbf"]["planes_per_die"]
                        * resolved_config["hbf"]["blocks_per_plane"]
                        * resolved_config["hbf"]["pages_per_block"]
                        * page_size
                    )
                )
                if (
                    cursor >= segment_capacity
                    or segment_bytes > segment_capacity - cursor
                ):
                    raise ContractError(
                        f"{where}: hybrid segment exceeds routed capacity")
                cursor += segment_bytes
                remaining -= segment_bytes
        if (
            model != "hybrid"
            and not is_drain
            and (
                addr >= capacity_bytes
                or byte_count > capacity_bytes - addr
            )
        ):
            raise ContractError(
                f"{where}: request exceeds {model.upper()} capacity")
        normalized_requests.append({
            "id": request_id,
            "arrival_ns": arrival,
            "op": request["op"],
            "address_space": request["address_space"],
            "addr": addr,
            "bytes": byte_count,
        })

    if model == "hbm":
        inspection_capacity = resolved_config["capacity_bytes"]
    elif model == "hbf":
        inspection_capacity = (
            resolved_config["stacks"]
            * resolved_config["channels_per_stack"]
            * resolved_config["dies_per_channel"]
            * resolved_config["planes_per_die"]
            * resolved_config["blocks_per_plane"]
            * resolved_config["pages_per_block"]
            * resolved_config["page_size_bytes"]
        )
    elif model == "external":
        inspection_capacity = resolved_config["capacity_bytes"]
    else:
        inspection_capacity = 0
    for index, address in enumerate(inspected):
        if address >= inspection_capacity:
            raise ContractError(
                f"{source}.inspect_addresses[{index}]: out of capacity")

    if model == "hbf":
        total_pages = inspection_capacity // resolved_config["page_size_bytes"]
        for index, lpn in enumerate(
            normalized_initial_state["prepopulate_lpns"]
        ):
            if lpn >= total_pages:
                raise ContractError(
                    f"{source}.initial_state.prepopulate_lpns[{index}]: "
                    "out of logical capacity")

    return {
        "schema": CASE_SCHEMA,
        "case_id": case_id,
        "model": model,
        "comparison": {
            "time_abs_tolerance_ns": abs_tolerance,
            "time_rel_tolerance": rel_tolerance,
        },
        "config_overrides": dict(data["config"]),
        "config": resolved_config,
        "initial_state": normalized_initial_state,
        "inspect_addresses": inspected,
        "requests": normalized_requests,
        "notes": list(notes),
    }


def load_case(path: Path) -> dict[str, Any]:
    return validate_case(load_json_strict(path), source=str(path))


def dump_json_line(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
