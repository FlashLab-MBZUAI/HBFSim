#!/usr/bin/env python3
"""Independent canonical Frontier hybrid-residency capacity arithmetic."""

from __future__ import annotations

from typing import Any


PLAN_SCHEMA_VERSION = 1
POLICY = "hybrid_residency_v1"
PRESSURE_BASIS = (
    "unique_resident_footprint_bytes/physical_hbm_capacity_bytes"
)
ACTIVE_WEIGHT_BUFFER_COUNT = 2
BLOCK_TABLE_ENTRY_BYTES = 4
RUNTIME_OVERHEAD_SOURCE = "configured_unprofiled_sensitivity"
CANONICAL_HBM_CAPACITY_BYTES = {
    96 * 1024**3,
    192 * 1024**3,
    288 * 1024**3,
    384 * 1024**3,
}
CANONICAL_PRESSURES = {
    0.75: ("0.75", 3, 4),
    1.0: ("1.0", 1, 1),
    1.25: ("1.25", 5, 4),
    1.5: ("1.5", 3, 2),
    2.0: ("2.0", 2, 1),
}


class HybridResidencyError(ValueError):
    """A capacity point cannot satisfy the canonical residency contract."""


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HybridResidencyError(f"{name} must be an integer > 0")
    return value


def _pressure_fraction(value: Any) -> tuple[str, int, int]:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) not in CANONICAL_PRESSURES
    ):
        raise HybridResidencyError(
            "target_pressure must be one of "
            f"{sorted(CANONICAL_PRESSURES)}"
        )
    return CANONICAL_PRESSURES[float(value)]


def build_hybrid_residency_plan(
    *,
    physical_hbm_capacity_bytes: int,
    target_pressure: float,
    immutable_weight_backing_bytes: int,
    runtime_overhead_bytes: int,
    active_weight_buffer_bytes_per_slot: int,
    kv_block_size_tokens: int,
    kv_page_bytes_per_layer: int,
    num_layers: int,
) -> dict[str, Any]:
    """Recompute Frontier's plan without importing Frontier implementation."""
    capacity = _positive_integer(
        physical_hbm_capacity_bytes,
        "physical_hbm_capacity_bytes",
    )
    if capacity not in CANONICAL_HBM_CAPACITY_BYTES:
        raise HybridResidencyError(
            "physical_hbm_capacity_bytes must be exactly "
            "96, 192, 288, or 384 GiB"
        )
    weights = _positive_integer(
        immutable_weight_backing_bytes,
        "immutable_weight_backing_bytes",
    )
    overhead = _positive_integer(
        runtime_overhead_bytes,
        "runtime_overhead_bytes",
    )
    buffer_per_slot = _positive_integer(
        active_weight_buffer_bytes_per_slot,
        "active_weight_buffer_bytes_per_slot",
    )
    block_tokens = _positive_integer(
        kv_block_size_tokens,
        "kv_block_size_tokens",
    )
    page_per_layer = _positive_integer(
        kv_page_bytes_per_layer,
        "kv_page_bytes_per_layer",
    )
    layers = _positive_integer(num_layers, "num_layers")
    pressure_text, numerator, denominator = _pressure_fraction(
        target_pressure
    )

    target_footprint = capacity * numerator // denominator
    kv_stride = page_per_layer * layers
    bytes_per_logical_block = kv_stride + BLOCK_TABLE_ENTRY_BYTES
    logical_budget = target_footprint - weights - overhead
    if logical_budget < bytes_per_logical_block:
        raise HybridResidencyError(
            "target footprint cannot contain immutable weights, runtime "
            "overhead, and one KV block"
        )
    logical_blocks = logical_budget // bytes_per_logical_block
    block_table = logical_blocks * BLOCK_TABLE_ENTRY_BYTES
    logical_kv = logical_blocks * kv_stride
    unique_footprint = weights + overhead + block_table + logical_kv
    slack = target_footprint - unique_footprint
    if slack < 0 or slack >= bytes_per_logical_block:
        raise HybridResidencyError(
            "unique-footprint floor division is inconsistent"
        )

    active_buffers = ACTIVE_WEIGHT_BUFFER_COUNT * buffer_per_slot
    fixed_hbm = overhead + block_table + active_buffers
    if fixed_hbm >= capacity:
        raise HybridResidencyError(
            "runtime overhead, block table, and active buffers exhaust HBM"
        )
    hot_blocks = min(
        logical_blocks,
        (capacity - fixed_hbm) // kv_stride,
    )
    if hot_blocks <= 0:
        raise HybridResidencyError(
            "fixed HBM allocations leave no complete hot KV block"
        )
    cold_blocks = logical_blocks - hot_blocks
    hot_kv = hot_blocks * kv_stride
    cold_kv = cold_blocks * kv_stride
    unused_hbm = capacity - fixed_hbm - hot_kv

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "policy": POLICY,
        "pressure_basis": PRESSURE_BASIS,
        "target_pressure": pressure_text,
        "target_pressure_numerator": numerator,
        "target_pressure_denominator": denominator,
        "physical_hbm_capacity_bytes": capacity,
        "target_footprint_bytes": target_footprint,
        "unique_resident_footprint_bytes": unique_footprint,
        "target_rounding_slack_bytes": slack,
        "immutable_weight_backing_bytes": weights,
        "runtime_overhead_bytes": overhead,
        "runtime_overhead_source": RUNTIME_OVERHEAD_SOURCE,
        "hardware_capacity_calibrated": False,
        "active_weight_buffer_count": ACTIVE_WEIGHT_BUFFER_COUNT,
        "active_weight_buffer_bytes_per_slot": buffer_per_slot,
        "active_weight_buffer_bytes": active_buffers,
        "block_table_entry_bytes": BLOCK_TABLE_ENTRY_BYTES,
        "block_table_bytes": block_table,
        "kv_block_size_tokens": block_tokens,
        "kv_page_bytes_per_layer": page_per_layer,
        "num_layers": layers,
        "kv_block_stride_bytes": kv_stride,
        "num_logical_kv_blocks": logical_blocks,
        "logical_kv_bytes": logical_kv,
        "hot_kv_blocks": hot_blocks,
        "hot_kv_bytes": hot_kv,
        "cold_kv_blocks": cold_blocks,
        "cold_kv_bytes": cold_kv,
        "fixed_hbm_bytes": fixed_hbm,
        "unused_hbm_bytes": unused_hbm,
        "backing_bytes": weights + cold_kv,
        "weight_mutability": "read_only",
        "kv_mutability": "mutable",
    }


def validate_hybrid_residency_plan(
    plan: Any,
    *,
    physical_hbm_capacity_bytes: int,
    target_pressure: float,
    immutable_weight_backing_bytes: int,
    runtime_overhead_bytes: int,
    active_weight_buffer_bytes_per_slot: int,
    kv_block_size_tokens: int,
    kv_page_bytes_per_layer: int,
    num_layers: int,
) -> dict[str, Any]:
    """Require byte-for-byte semantic equality with independent arithmetic."""
    if not isinstance(plan, dict):
        raise HybridResidencyError("residency_plan must be an object")
    expected = build_hybrid_residency_plan(
        physical_hbm_capacity_bytes=physical_hbm_capacity_bytes,
        target_pressure=target_pressure,
        immutable_weight_backing_bytes=immutable_weight_backing_bytes,
        runtime_overhead_bytes=runtime_overhead_bytes,
        active_weight_buffer_bytes_per_slot=(
            active_weight_buffer_bytes_per_slot
        ),
        kv_block_size_tokens=kv_block_size_tokens,
        kv_page_bytes_per_layer=kv_page_bytes_per_layer,
        num_layers=num_layers,
    )
    if plan != expected:
        differing = sorted(
            key
            for key in set(plan) | set(expected)
            if plan.get(key) != expected.get(key)
        )
        raise HybridResidencyError(
            "residency_plan differs from independent arithmetic at: "
            + ", ".join(differing)
        )
    return expected
