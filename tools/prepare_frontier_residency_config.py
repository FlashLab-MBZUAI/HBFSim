#!/usr/bin/env python3
"""Compile a Frontier memory object map into HBFSim hybrid residency.

The selected trace window determines traffic, never allocation. This tool
independently validates the digest-bound object population and Frontier plan,
then fits that unchanged population to an explicitly selected physical HBM
capacity. It emits the only supported production layer-streaming config
overlay plus a machine-readable placement receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


MANIFEST_SCHEMA = {
    "name": "hbfsim.frontier_memory_trace",
    "version": 4,
}
OBJECT_MAP_SCHEMA = {
    "name": "hbfsim.memory_object_map",
    "version": 4,
}
RECEIPT_SCHEMA = {
    "name": "hbfsim.frontier_residency_binding",
    "version": 3,
}
PLAN_POLICY = "hybrid_residency_v1"
PLAN_SCHEMA_VERSION = 1
ACTIVE_BUFFER_COUNT = 2
PLACEMENT_POLICY = "capacity_aware_static_weight_prefix_v2"


class ResidencyBindingError(ValueError):
    """The exported population cannot be bound without weakening evidence."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ResidencyBindingError(
            f"cannot read {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ResidencyBindingError(f"{description} must be a JSON object")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResidencyBindingError(f"{name} must be an object")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResidencyBindingError(f"{name} must be an integer")
    if value < minimum:
        raise ResidencyBindingError(f"{name} must be >= {minimum}")
    return value


def _expect(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ResidencyBindingError(
            f"{name}: expected {expected!r}, got {actual!r}"
        )


def _checked_add(lhs: int, rhs: int, name: str) -> int:
    value = lhs + rhs
    if value > (1 << 64) - 1:
        raise ResidencyBindingError(f"{name} exceeds uint64")
    return value


def _checked_mul(lhs: int, rhs: int, name: str) -> int:
    value = lhs * rhs
    if value > (1 << 64) - 1:
        raise ResidencyBindingError(f"{name} exceeds uint64")
    return value


def _ceil_div(value: int, divisor: int, name: str) -> int:
    if divisor <= 0:
        raise ResidencyBindingError(f"{name} has non-positive divisor")
    return value // divisor + int(value % divisor != 0)


def _compile_kv_partition(
    *,
    physical_hbm_capacity_bytes: int,
    page_size: int,
    metadata_pages: int,
    active_buffer_pages: int,
    logical_blocks: int,
    kv_stride: int,
) -> dict[str, int]:
    if physical_hbm_capacity_bytes % page_size:
        raise ResidencyBindingError("HBM capacity must be page aligned")
    fixed_pages = _checked_add(
        metadata_pages,
        _checked_mul(
            active_buffer_pages,
            ACTIVE_BUFFER_COUNT,
            "active buffer pages",
        ),
        "fixed runtime HBM pages",
    )
    capacity_pages = physical_hbm_capacity_bytes // page_size
    if fixed_pages >= capacity_pages:
        raise ResidencyBindingError(
            "physical HBM cannot hold runtime metadata and active buffers"
        )
    available_kv_bytes = _checked_mul(
        capacity_pages - fixed_pages,
        page_size,
        "available hot KV bytes",
    )
    maximum_hot_blocks = min(
        logical_blocks,
        available_kv_bytes // kv_stride,
    )
    if maximum_hot_blocks == logical_blocks:
        hot_blocks = logical_blocks
        hot_kv_pages = _ceil_div(
            _checked_mul(
                hot_blocks,
                kv_stride,
                "all-hot KV bytes",
            ),
            page_size,
            "all-hot KV pages",
        )
    else:
        alignment_period_blocks = page_size // math.gcd(
            kv_stride,
            page_size,
        )
        hot_blocks = (
            maximum_hot_blocks
            - maximum_hot_blocks % alignment_period_blocks
        )
        hot_kv_pages = (
            _checked_mul(
                hot_blocks,
                kv_stride,
                "page-aligned hot KV bytes",
            )
            // page_size
        )
    if hot_blocks < 1:
        raise ResidencyBindingError(
            "physical HBM cannot hold one hot KV block after fixed residency"
        )
    runtime_hbm_pages = _checked_add(
        fixed_pages,
        hot_kv_pages,
        "runtime HBM pages",
    )
    return {
        "hot_blocks": hot_blocks,
        "cold_blocks": logical_blocks - hot_blocks,
        "hot_kv_pages": hot_kv_pages,
        "runtime_hbm_pages": runtime_hbm_pages,
        "unused_hbm_pages": capacity_pages - runtime_hbm_pages,
    }


def _merged_interval_pages(
    intervals: list[tuple[int, int]],
) -> int:
    if not intervals:
        return 0
    intervals.sort()
    begin, end = intervals[0]
    pages = 0
    for next_begin, next_end in intervals[1:]:
        if next_begin <= end:
            end = max(end, next_end)
            continue
        pages = _checked_add(pages, end - begin, "layer backing pages")
        begin, end = next_begin, next_end
    return _checked_add(pages, end - begin, "layer backing pages")


def _trace_layer_backing_census(
    *,
    trace_path: Path,
    page_size: int,
    kv_region_begin: int,
    logical_kv_pages: int,
    hot_kv_pages: int,
    static_weight_resident_pages: int,
) -> dict[str, int]:
    """Count the maximum backing pages simultaneously needed by one layer."""
    semantic_kinds = {
        "model_weights",
        "shared_context",
        "generated_context",
        "scratch",
        "metadata",
    }
    kv_begin_page = kv_region_begin // page_size
    hot_kv_end_page = _checked_add(
        kv_begin_page,
        hot_kv_pages,
        "hot KV end page",
    )
    kv_end_page = _checked_add(
        kv_begin_page,
        logical_kv_pages,
        "logical KV end page",
    )
    current_layer: int | None = None
    intervals: list[tuple[int, int]] = []
    maximum_pages = 0
    maximum_layer = 0
    operations = 0

    def finish_layer() -> None:
        nonlocal intervals, maximum_pages, maximum_layer
        pages = _merged_interval_pages(intervals)
        if pages > maximum_pages:
            maximum_pages = pages
            maximum_layer = 0 if current_layer is None else current_layer
        intervals = []

    try:
        handle = trace_path.open("r", encoding="utf-8")
    except OSError as error:
        raise ResidencyBindingError(
            f"cannot inspect exported trace {trace_path}: {error}"
        ) from error
    with handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 3 or tokens[1].upper() not in {"R", "W"}:
                raise ResidencyBindingError(
                    f"trace line {line_number} is not canonical "
                    "<ADDR> <R|W> <BYTES>"
                )
            try:
                address = int(tokens[0], 0)
                request_bytes = int(tokens[2], 0)
            except ValueError as error:
                raise ResidencyBindingError(
                    f"trace line {line_number} has invalid address or bytes"
                ) from error
            if (
                address < 0
                or address > (1 << 64) - 1
                or request_bytes <= 0
                or request_bytes > (1 << 64) - 1
                or address + request_bytes > 1 << 64
            ):
                raise ResidencyBindingError(
                    f"trace line {line_number} exceeds the uint64 address "
                    "contract"
                )
            kind: str | None = None
            layer: int | None = None
            for token in tokens[3:]:
                key, separator, value = token.partition("=")
                if separator and key == "kind":
                    if kind is not None:
                        raise ResidencyBindingError(
                            f"trace line {line_number} duplicates kind"
                        )
                    kind = value
                elif not separator and token in semantic_kinds:
                    if kind is not None:
                        raise ResidencyBindingError(
                            f"trace line {line_number} duplicates kind"
                        )
                    kind = token
                elif separator and key == "layer":
                    if layer is not None:
                        raise ResidencyBindingError(
                            f"trace line {line_number} duplicates layer"
                        )
                    try:
                        layer = int(value, 10)
                    except ValueError as error:
                        raise ResidencyBindingError(
                            f"trace line {line_number} has invalid layer"
                        ) from error
                    if layer < 0 or layer > (1 << 64) - 1:
                        raise ResidencyBindingError(
                            f"trace line {line_number} has invalid layer"
                        )
            if kind not in semantic_kinds or layer is None:
                raise ResidencyBindingError(
                    f"trace line {line_number} must declare canonical kind "
                    "and layer"
                )
            if current_layer is None:
                current_layer = layer
            elif layer < current_layer:
                raise ResidencyBindingError(
                    f"trace line {line_number} decreases the layer id"
                )
            elif layer != current_layer:
                finish_layer()
                current_layer = layer

            first_page = address // page_size
            last_page_exclusive = (
                (address + request_bytes - 1) // page_size + 1
            )
            if kind == "model_weights":
                if last_page_exclusive > kv_begin_page:
                    raise ResidencyBindingError(
                        f"trace line {line_number} places model weights "
                        "outside the immutable prefix"
                    )
                backing_begin = max(
                    first_page,
                    static_weight_resident_pages,
                )
                if backing_begin < last_page_exclusive:
                    intervals.append((backing_begin, last_page_exclusive))
            elif kind in {"shared_context", "generated_context"}:
                if (
                    first_page < kv_begin_page
                    or last_page_exclusive > kv_end_page
                ):
                    raise ResidencyBindingError(
                        f"trace line {line_number} places KV outside its arena"
                    )
                backing_begin = max(first_page, hot_kv_end_page)
                if backing_begin < last_page_exclusive:
                    intervals.append(
                        (backing_begin, last_page_exclusive)
                    )
            operations += 1
    if operations == 0:
        raise ResidencyBindingError("exported trace has no operations")
    finish_layer()
    return {
        "operations": operations,
        "maximum_backing_pages": maximum_pages,
        "maximum_backing_layer": maximum_layer,
    }


def _compile_trace_fitted_partition(
    *,
    trace_path: Path,
    physical_hbm_capacity_bytes: int,
    page_size: int,
    metadata_pages: int,
    source_active_buffer_pages: int,
    kv_region_begin: int,
    logical_kv_pages: int,
    logical_blocks: int,
    kv_stride: int,
) -> dict[str, Any]:
    active_buffer_pages = source_active_buffer_pages
    for iteration in range(1, 65):
        partition = _compile_kv_partition(
            physical_hbm_capacity_bytes=physical_hbm_capacity_bytes,
            page_size=page_size,
            metadata_pages=metadata_pages,
            active_buffer_pages=active_buffer_pages,
            logical_blocks=logical_blocks,
            kv_stride=kv_stride,
        )
        census = _trace_layer_backing_census(
            trace_path=trace_path,
            page_size=page_size,
            kv_region_begin=kv_region_begin,
            logical_kv_pages=logical_kv_pages,
            hot_kv_pages=partition["hot_kv_pages"],
            static_weight_resident_pages=0,
        )
        required_pages = max(
            source_active_buffer_pages,
            census["maximum_backing_pages"],
        )
        if required_pages <= active_buffer_pages:
            return {
                "partition": partition,
                "active_buffer_pages": active_buffer_pages,
                "trace_census": census,
                "iterations": iteration,
            }
        active_buffer_pages = required_pages
    raise ResidencyBindingError(
        "trace-aware active-buffer/KV placement did not converge"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_path(value: Any, *, owner: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ResidencyBindingError(f"{name}.path must be non-empty")
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _write_atomic(path: Path, payload: bytes) -> None:
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


def _validate_regions(
    object_map: dict[str, Any],
    *,
    page_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_regions = object_map.get("regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ResidencyBindingError("object_map.regions must be non-empty")
    regions: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    previous_end = 0
    for index, raw_region in enumerate(raw_regions):
        region = _mapping(raw_region, f"object_map.regions[{index}]")
        name = region.get("name")
        if not isinstance(name, str) or not name:
            raise ResidencyBindingError(
                f"object_map.regions[{index}].name must be non-empty"
            )
        if name in by_name:
            raise ResidencyBindingError(f"duplicate object region {name}")
        begin = _integer(region.get("begin"), f"{name}.begin")
        end = _integer(region.get("end"), f"{name}.end", minimum=1)
        size = _integer(region.get("bytes"), f"{name}.bytes", minimum=1)
        alignment = _integer(
            region.get("alignment_bytes"),
            f"{name}.alignment_bytes",
            minimum=1,
        )
        _expect(alignment, page_size, f"{name}.alignment_bytes")
        if begin % page_size != 0:
            raise ResidencyBindingError(f"{name}.begin is not page aligned")
        _expect(end, _checked_add(begin, size, f"{name}.end"), f"{name}.end")
        if begin < previous_end:
            raise ResidencyBindingError(f"object region {name} overlaps")
        previous_end = end
        regions.append(region)
        by_name[name] = region
    address_space = _mapping(
        object_map.get("address_space"),
        "object_map.address_space",
    )
    _expect(address_space.get("begin"), 0, "address_space.begin")
    address_end = _integer(
        address_space.get("end"),
        "address_space.end",
        minimum=1,
    )
    _expect(
        address_space.get("bytes"),
        address_end,
        "address_space.bytes",
    )
    _expect(
        address_space.get("object_alignment_bytes"),
        page_size,
        "address_space.object_alignment_bytes",
    )
    if address_end % page_size != 0 or previous_end > address_end:
        raise ResidencyBindingError(
            "object population exceeds its aligned address space"
        )
    return regions, by_name


def prepare_residency_config(
    *,
    manifest_path: Path,
    object_map_path: Path,
    output_config_path: Path,
    receipt_path: Path,
    physical_hbm_capacity_bytes: int,
) -> dict[str, Any]:
    resolved_paths = [
        manifest_path.resolve(),
        object_map_path.resolve(),
        output_config_path.resolve(),
        receipt_path.resolve(),
    ]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ResidencyBindingError("input and output paths must be distinct")
    manifest_path, object_map_path, output_config_path, receipt_path = (
        resolved_paths
    )
    manifest = _load_object(manifest_path, "Frontier trace manifest")
    object_map = _load_object(object_map_path, "Frontier object map")
    _expect(manifest.get("schema"), MANIFEST_SCHEMA, "manifest.schema")
    _expect(object_map.get("schema"), OBJECT_MAP_SCHEMA, "object_map.schema")

    outputs = _mapping(manifest.get("outputs"), "manifest.outputs")
    recorded_object_map = _mapping(
        outputs.get("object_map"),
        "manifest.outputs.object_map",
    )
    object_map_bytes = object_map_path.stat().st_size
    object_map_sha256 = _sha256(object_map_path)
    _expect(
        recorded_object_map.get("bytes"),
        object_map_bytes,
        "object map recorded bytes",
    )
    _expect(
        recorded_object_map.get("sha256"),
        object_map_sha256,
        "object map recorded SHA-256",
    )
    trace_output = _mapping(outputs.get("trace"), "manifest.outputs.trace")
    trace_path = _artifact_path(
        trace_output.get("path"),
        owner=manifest_path,
        name="manifest.outputs.trace",
    )
    if trace_path in set(resolved_paths):
        raise ResidencyBindingError(
            "trace, manifest, object map, config, and receipt paths must be "
            "distinct"
        )
    try:
        trace_bytes = trace_path.stat().st_size
        trace_sha256 = _sha256(trace_path)
    except OSError as error:
        raise ResidencyBindingError(
            f"cannot read exported trace {trace_path}: {error}"
        ) from error
    _expect(
        _integer(
            trace_output.get("bytes"),
            "manifest.outputs.trace.bytes",
            minimum=1,
        ),
        trace_bytes,
        "trace recorded bytes",
    )
    recorded_trace_sha256 = trace_output.get("sha256")
    if (
        not isinstance(recorded_trace_sha256, str)
        or len(recorded_trace_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in recorded_trace_sha256
        )
    ):
        raise ResidencyBindingError("manifest trace SHA-256 is invalid")
    _expect(
        recorded_trace_sha256,
        trace_sha256,
        "trace recorded SHA-256",
    )

    semantics = _mapping(manifest.get("semantics"), "manifest.semantics")
    for field in (
        "source_requests_are_production",
        "memory_traffic_is_model_conditioned",
    ):
        _expect(semantics.get(field), True, f"manifest.semantics.{field}")
    _expect(
        semantics.get("memory_traffic_is_measured_gpu_traffic"),
        False,
        "manifest measured GPU traffic",
    )
    _expect(
        semantics.get("eligible_claim_scope"),
        "memory_system_service_only",
        "manifest eligible claim scope",
    )
    _expect(
        semantics.get("ttft_tpot_slo_claims_eligible"),
        False,
        "manifest TTFT/TPOT/SLO eligibility",
    )
    _expect(
        semantics.get("time_based_throughput_claims_eligible"),
        False,
        "manifest time-throughput eligibility",
    )
    _expect(
        semantics.get("model_weight_traffic"),
        "read_only",
        "manifest model weight traffic",
    )
    _expect(
        semantics.get("mutable_write_traffic"),
        "kv_append_and_update_only",
        "manifest mutable write traffic",
    )
    _expect(
        semantics.get("capacity_pressure_basis"),
        "unique_resident_footprint_bytes/physical_hbm_capacity_bytes",
        "manifest capacity pressure basis",
    )

    capacity = _mapping(
        manifest.get("capacity_accounting"),
        "manifest.capacity_accounting",
    )
    _expect(
        capacity.get("planner_mode"),
        "hybrid_residency",
        "capacity planner mode",
    )
    plan = _mapping(capacity.get("residency_plan"), "residency_plan")
    _expect(
        plan.get("schema_version"),
        PLAN_SCHEMA_VERSION,
        "residency plan schema",
    )
    _expect(plan.get("policy"), PLAN_POLICY, "residency plan policy")
    _expect(
        plan.get("active_weight_buffer_count"),
        ACTIVE_BUFFER_COUNT,
        "active buffer count",
    )
    _expect(plan.get("weight_mutability"), "read_only", "weight mutability")
    _expect(plan.get("kv_mutability"), "mutable", "KV mutability")

    page_size = _integer(
        _mapping(
            object_map.get("address_space"),
            "object_map.address_space",
        ).get("object_alignment_bytes"),
        "object map page size",
        minimum=1,
    )
    if page_size != 4096:
        raise ResidencyBindingError(
            "canonical Frontier residency currently requires 4096-byte pages"
        )
    regions, by_name = _validate_regions(
        object_map,
        page_size=page_size,
    )
    weight_regions = [
        region for region in regions
        if region.get("kind") == "model_weights"
    ]
    if not weight_regions:
        raise ResidencyBindingError("object map has no immutable weights")
    for region in weight_regions:
        _expect(
            region.get("placement_policy"),
            "model_weight",
            f"{region['name']} placement",
        )
    immutable_weight_bytes = sum(
        _integer(region.get("bytes"), f"{region['name']}.bytes", minimum=1)
        for region in weight_regions
    )
    immutable_weight_pages = sum(
        _ceil_div(
            _integer(region.get("bytes"), f"{region['name']}.bytes", minimum=1),
            page_size,
            f"{region['name']} pages",
        )
        for region in weight_regions
    )

    kv_region = _mapping(
        by_name.get("kv.physical_block_slots"),
        "kv.physical_block_slots",
    )
    block_table = _mapping(
        by_name.get("metadata.kv_allocator_blocks"),
        "metadata.kv_allocator_blocks",
    )
    runtime = _mapping(
        by_name.get("metadata.runtime_overhead"),
        "metadata.runtime_overhead",
    )
    _expect(kv_region.get("kind"), "generated_context", "KV region kind")
    _expect(
        kv_region.get("placement_policy"),
        "frontier_physical_block_id",
        "KV placement",
    )
    for region, name in (
        (block_table, "block table"),
        (runtime, "runtime overhead"),
    ):
        _expect(region.get("kind"), "metadata", f"{name} kind")
        _expect(
            region.get("placement_policy"),
            "hbm_only",
            f"{name} placement",
        )

    source_capacity_bytes = _integer(
        plan.get("physical_hbm_capacity_bytes"),
        "Frontier source-plan physical HBM capacity",
        minimum=1,
    )
    capacity_bytes = _integer(
        physical_hbm_capacity_bytes,
        "target physical HBM capacity",
        minimum=1,
    )
    unique_footprint = _integer(
        plan.get("unique_resident_footprint_bytes"),
        "unique resident footprint",
        minimum=1,
    )
    planned_weight_bytes = _integer(
        plan.get("immutable_weight_backing_bytes"),
        "planned immutable weight bytes",
        minimum=1,
    )
    runtime_bytes = _integer(
        plan.get("runtime_overhead_bytes"),
        "runtime overhead bytes",
        minimum=1,
    )
    block_table_bytes = _integer(
        plan.get("block_table_bytes"),
        "block table bytes",
        minimum=1,
    )
    source_active_buffer_bytes = _integer(
        plan.get("active_weight_buffer_bytes_per_slot"),
        "active buffer bytes per slot",
        minimum=1,
    )
    kv_stride = _integer(
        plan.get("kv_block_stride_bytes"),
        "KV block stride",
        minimum=1,
    )
    logical_blocks = _integer(
        plan.get("num_logical_kv_blocks"),
        "logical KV blocks",
        minimum=1,
    )
    source_hot_blocks = _integer(
        plan.get("hot_kv_blocks"),
        "Frontier source-plan hot KV blocks",
        minimum=1,
    )
    source_cold_blocks = _integer(
        plan.get("cold_kv_blocks"),
        "Frontier source-plan cold KV blocks",
    )
    if source_hot_blocks > logical_blocks:
        raise ResidencyBindingError("hot KV blocks exceed logical blocks")
    _expect(
        logical_blocks,
        _checked_add(
            source_hot_blocks,
            source_cold_blocks,
            "source-plan hot/cold KV blocks",
        ),
        "source-plan logical KV block partition",
    )
    logical_kv_bytes = _checked_mul(
        logical_blocks,
        kv_stride,
        "logical KV bytes",
    )
    _expect(
        plan.get("logical_kv_bytes"),
        logical_kv_bytes,
        "planned logical KV bytes",
    )
    exact_population = _checked_add(
        planned_weight_bytes,
        _checked_add(
            runtime_bytes,
            _checked_add(
                block_table_bytes,
                logical_kv_bytes,
                "block table and logical KV bytes",
            ),
            "overhead, block table, and logical KV bytes",
        ),
        "unique resident population",
    )
    _expect(unique_footprint, exact_population, "unique resident footprint")
    _expect(
        immutable_weight_bytes,
        planned_weight_bytes,
        "object-map immutable weight bytes",
    )
    kv_region_begin = _integer(
        kv_region.get("begin"),
        "KV region begin",
    )
    _expect(
        kv_region_begin // page_size,
        immutable_weight_pages,
        "immutable weight page prefix",
    )
    _expect(
        capacity.get("immutable_weight_backing_bytes"),
        planned_weight_bytes,
        "capacity immutable weight bytes",
    )
    _expect(
        _integer(kv_region.get("bytes"), "KV region bytes", minimum=1),
        logical_kv_bytes,
        "KV region bytes",
    )
    _expect(kv_region.get("num_slots"), logical_blocks, "KV region slots")
    _expect(kv_region.get("bytes_per_slot"), kv_stride, "KV slot stride")
    _expect(
        _integer(block_table.get("bytes"), "block table region bytes", minimum=1),
        block_table_bytes,
        "block table region bytes",
    )
    _expect(
        block_table.get("num_logical_kv_blocks"),
        logical_blocks,
        "block table logical blocks",
    )
    _expect(
        _integer(runtime.get("bytes"), "runtime region bytes", minimum=1),
        runtime_bytes,
        "runtime region bytes",
    )

    metadata_pages = _checked_add(
        _ceil_div(runtime_bytes, page_size, "runtime pages"),
        _ceil_div(block_table_bytes, page_size, "block table pages"),
        "metadata pages",
    )
    logical_kv_pages = _ceil_div(
        logical_kv_bytes,
        page_size,
        "logical KV pages",
    )
    source_active_buffer_pages = _ceil_div(
        source_active_buffer_bytes,
        page_size,
        "active buffer pages",
    )
    source_partition = _compile_kv_partition(
        physical_hbm_capacity_bytes=source_capacity_bytes,
        page_size=page_size,
        metadata_pages=metadata_pages,
        active_buffer_pages=source_active_buffer_pages,
        logical_blocks=logical_blocks,
        kv_stride=kv_stride,
    )
    _expect(
        source_hot_blocks,
        source_partition["hot_blocks"],
        "Frontier source-plan hot KV blocks",
    )
    _expect(
        source_cold_blocks,
        source_partition["cold_blocks"],
        "Frontier source-plan cold KV blocks",
    )
    fitted = _compile_trace_fitted_partition(
        trace_path=trace_path,
        physical_hbm_capacity_bytes=capacity_bytes,
        page_size=page_size,
        metadata_pages=metadata_pages,
        source_active_buffer_pages=source_active_buffer_pages,
        kv_region_begin=kv_region_begin,
        logical_kv_pages=logical_kv_pages,
        logical_blocks=logical_blocks,
        kv_stride=kv_stride,
    )
    compiled_partition = fitted["partition"]
    active_buffer_pages = fitted["active_buffer_pages"]
    active_buffer_bytes = (
        source_active_buffer_bytes
        if active_buffer_pages == source_active_buffer_pages
        else _checked_mul(
            active_buffer_pages,
            page_size,
            "trace-fitted active buffer bytes",
        )
    )
    hot_blocks = compiled_partition["hot_blocks"]
    cold_blocks = compiled_partition["cold_blocks"]
    hot_kv_pages = compiled_partition["hot_kv_pages"]
    cold_kv_pages = logical_kv_pages - hot_kv_pages
    static_weight_resident_pages = min(
        immutable_weight_pages,
        compiled_partition["unused_hbm_pages"],
    )
    model_weight_backing_pages = (
        immutable_weight_pages - static_weight_resident_pages
    )
    runtime_hbm_pages = _checked_add(
        compiled_partition["runtime_hbm_pages"],
        static_weight_resident_pages,
        "runtime HBM pages with static weights",
    )
    unused_hbm_pages = (
        compiled_partition["unused_hbm_pages"]
        - static_weight_resident_pages
    )
    trace_census = _trace_layer_backing_census(
        trace_path=trace_path,
        page_size=page_size,
        kv_region_begin=kv_region_begin,
        logical_kv_pages=logical_kv_pages,
        hot_kv_pages=hot_kv_pages,
        static_weight_resident_pages=static_weight_resident_pages,
    )
    if cold_kv_pages != _ceil_div(
        _checked_mul(
            cold_blocks,
            kv_stride,
            "cold KV bytes",
        ),
        page_size,
        "cold KV pages",
    ):
        raise ResidencyBindingError(
            "compiled hot/cold KV page partition does not conserve"
        )
    unique_population_pages = _checked_add(
        immutable_weight_pages,
        _checked_add(
            metadata_pages,
            logical_kv_pages,
            "metadata and logical KV pages",
        ),
        "page-allocated unique population",
    )
    unique_population_bytes = _checked_mul(
        unique_population_pages,
        page_size,
        "page-allocated unique population bytes",
    )
    if unique_population_bytes < unique_footprint:
        raise ResidencyBindingError(
            "page allocation is smaller than exact unique population"
        )
    object_address_space = _mapping(
        object_map.get("address_space"),
        "object_map.address_space",
    )
    object_address_space_bytes = _integer(
        object_address_space.get("bytes"),
        "object_map.address_space.bytes",
        minimum=1,
    )
    _expect(
        object_address_space_bytes,
        unique_population_bytes,
        "canonical object address-space allocation",
    )

    config_values = {
        "trace": str(trace_path),
        "expected-trace-sha256": trace_sha256,
        "expected-trace-bytes": trace_bytes,
        "hbm-capacity-bytes": capacity_bytes,
        "layer-buffer-bytes": _checked_mul(
            active_buffer_pages,
            page_size,
            "page-rounded active buffer bytes",
        ),
        "explicit-residency-contract": "true",
        "residency-page-size-bytes": page_size,
        "residency-unique-footprint-bytes": unique_footprint,
        "residency-immutable-weight-bytes": planned_weight_bytes,
        "residency-immutable-weight-pages": immutable_weight_pages,
        "residency-static-weight-pages": static_weight_resident_pages,
        "residency-runtime-overhead-bytes": runtime_bytes,
        "residency-block-table-bytes": block_table_bytes,
        "residency-active-buffer-bytes": active_buffer_bytes,
        "residency-kv-region-begin": kv_region_begin,
        "residency-kv-block-stride-bytes": kv_stride,
        "residency-logical-kv-blocks": logical_blocks,
        "residency-hot-kv-blocks": hot_blocks,
    }
    manifest_sha256 = _sha256(manifest_path)
    config_lines = [
        "# hbfsim.frontier_residency_binding schema=3",
        f"# manifest-sha256={manifest_sha256}",
        f"# object-map-sha256={object_map_sha256}",
        f"# trace-sha256={trace_sha256}",
        f"# placement-policy={PLACEMENT_POLICY}",
        *[f"{key}={value}" for key, value in config_values.items()],
        "",
    ]
    config_payload = "\n".join(config_lines).encode("utf-8")
    config_sha256 = _sha256_bytes(config_payload)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "result": "pass",
        "claim_scope": {
            "eligible": "memory_system_service_only",
            "ttft_tpot_slo": False,
            "time_based_throughput": False,
        },
        "inputs": {
            "manifest": {
                "path": str(manifest_path),
                "bytes": manifest_path.stat().st_size,
                "sha256": manifest_sha256,
            },
            "object_map": {
                "path": str(object_map_path),
                "bytes": object_map_bytes,
                "sha256": object_map_sha256,
            },
            "trace": {
                "path": str(trace_path),
                "bytes": trace_bytes,
                "sha256": trace_sha256,
            },
        },
        "contract": {
            "placement_policy": PLACEMENT_POLICY,
            "page_size_bytes": page_size,
            "physical_hbm_capacity_bytes": capacity_bytes,
            "source_frontier_plan": {
                "physical_hbm_capacity_bytes": source_capacity_bytes,
                "hot_kv_blocks": source_hot_blocks,
                "cold_kv_blocks": source_cold_blocks,
            },
            "population_preserved_across_placement": True,
            "unique_resident_footprint_bytes": unique_footprint,
            "capacity_pressure": unique_footprint / capacity_bytes,
            "immutable_weight_bytes": planned_weight_bytes,
            "immutable_weight_pages": immutable_weight_pages,
            "static_weight_resident_pages": static_weight_resident_pages,
            "model_weight_backing_pages": model_weight_backing_pages,
            "runtime_overhead_bytes": runtime_bytes,
            "block_table_bytes": block_table_bytes,
            "active_buffer_bytes_per_slot": active_buffer_bytes,
            "active_buffer_pages_per_slot": active_buffer_pages,
            "source_active_weight_buffer_bytes_per_slot": (
                source_active_buffer_bytes
            ),
            "source_active_weight_buffer_pages_per_slot": (
                source_active_buffer_pages
            ),
            "trace_visible_max_backing_pages_per_layer": (
                trace_census["maximum_backing_pages"]
            ),
            "trace_visible_max_backing_layer": (
                trace_census["maximum_backing_layer"]
            ),
            "trace_fit_iterations": fitted["iterations"],
            "kv_region_begin": config_values["residency-kv-region-begin"],
            "kv_block_stride_bytes": kv_stride,
            "logical_kv_blocks": logical_blocks,
            "hot_kv_blocks": hot_blocks,
            "cold_kv_blocks": cold_blocks,
            "page_allocated_unique_footprint_pages": unique_population_pages,
            "page_allocated_unique_footprint_bytes": unique_population_bytes,
            "object_address_space_bytes": object_address_space_bytes,
            "footprint_page_rounding_bytes": (
                unique_population_bytes - unique_footprint
            ),
            "metadata_resident_pages": metadata_pages,
            "hot_kv_resident_pages": hot_kv_pages,
            "cold_kv_backing_pages": cold_kv_pages,
            "runtime_hbm_pages": runtime_hbm_pages,
            "unused_hbm_pages": unused_hbm_pages,
            "backing_unique_pages": (
                model_weight_backing_pages + cold_kv_pages
            ),
        },
        "output": {
            "config": {
                "path": str(output_config_path),
                "bytes": len(config_payload),
                "sha256": config_sha256,
            },
            "values": config_values,
        },
    }
    receipt_payload = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        _write_atomic(output_config_path, config_payload)
        _write_atomic(receipt_path, receipt_payload)
    except BaseException:
        output_config_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        raise
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object-map", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--physical-hbm-capacity-bytes",
        type=int,
        required=True,
        help=(
            "explicit target topology capacity; the logical object "
            "population is never resized"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = prepare_residency_config(
            manifest_path=args.manifest,
            object_map_path=args.object_map,
            output_config_path=args.output_config,
            receipt_path=args.receipt,
            physical_hbm_capacity_bytes=(
                args.physical_hbm_capacity_bytes
            ),
        )
    except ResidencyBindingError as error:
        raise SystemExit(f"error: {error}") from error
    contract = receipt["contract"]
    print(
        "Frontier residency binding: PASS "
        f"pressure={contract['capacity_pressure']:.6f} "
        f"logical_kv_blocks={contract['logical_kv_blocks']} "
        f"hot_kv_blocks={contract['hot_kv_blocks']} "
        f"cold_kv_blocks={contract['cold_kv_blocks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
