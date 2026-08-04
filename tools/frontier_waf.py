#!/usr/bin/env python3
"""Canonical Frontier/HBF write-amplification audit.

There is one WAF definition:

    physical NAND payload program bytes / logical bytes submitted to HBF

Initial-image population is pre-existing media state. OOB/ECC and transport
bytes are outside both counters. A read-only run has no denominator and
therefore reports ``null``, never zero.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


WAF_DEFINITION = "physical_write_bytes/logical_write_bytes"
MUTABLE_KV_KINDS = frozenset({"shared_context", "generated_context"})


class FrontierWafError(ValueError):
    """Frontier/HBF write accounting is not canonical or does not conserve."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrontierWafError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrontierWafError(f"{name} must be an integer")
    if value < minimum:
        raise FrontierWafError(f"{name} must be >= {minimum}")
    return value


def _trace_write_census(trace_path: Path) -> dict[str, int]:
    operations = 0
    byte_count = 0
    try:
        handle = trace_path.open(encoding="utf-8")
    except OSError as error:
        raise FrontierWafError(
            f"cannot read Frontier memory trace {trace_path}: {error}"
        ) from error
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 4 or fields[1] not in {"R", "W"}:
                raise FrontierWafError(
                    f"{trace_path}:{line_number}: malformed memory operation"
                )
            try:
                operation_bytes = int(fields[2], 10)
            except ValueError as error:
                raise FrontierWafError(
                    f"{trace_path}:{line_number}: byte count is not decimal"
                ) from error
            if operation_bytes <= 0:
                raise FrontierWafError(
                    f"{trace_path}:{line_number}: byte count must be positive"
                )
            metadata: dict[str, str] = {}
            for token in fields[3:]:
                if token.count("=") != 1:
                    raise FrontierWafError(
                        f"{trace_path}:{line_number}: malformed metadata"
                    )
                key, value = token.split("=", 1)
                if not key or not value or key in metadata:
                    raise FrontierWafError(
                        f"{trace_path}:{line_number}: duplicate/empty metadata"
                    )
                metadata[key] = value
            if fields[1] != "W":
                continue
            if (
                metadata.get("kind") not in MUTABLE_KV_KINDS
                or not metadata.get("label", "").startswith("kv.w.")
            ):
                raise FrontierWafError(
                    f"{trace_path}:{line_number}: non-KV write entered the "
                    "Frontier memory trace"
                )
            operations += 1
            byte_count += operation_bytes
    return {"operations": operations, "bytes": byte_count}


def audit_frontier_hbf_waf(
    *,
    summary: dict[str, Any],
    scenario: dict[str, Any],
    manifest: dict[str, Any],
    trace_path: Path,
) -> dict[str, Any]:
    """Validate and return the sole HBF WAF record for one Frontier replay."""
    if scenario.get("name") != "HBM+HBF-layer-streaming":
        raise FrontierWafError(
            "canonical Frontier WAF requires the HBM+HBF layer-streaming case"
        )
    semantics = _mapping(manifest.get("semantics"), "manifest.semantics")
    if (
        semantics.get("model_weight_traffic") != "read_only"
        or semantics.get("mutable_write_traffic")
        != "kv_append_and_update_only"
    ):
        raise FrontierWafError(
            "manifest does not separate read-only weights from mutable KV"
        )
    traffic = _mapping(
        manifest.get("traffic_census"),
        "manifest.traffic_census",
    )
    manifest_writes = _mapping(
        traffic.get("writes"),
        "manifest.traffic_census.writes",
    )
    trace_writes = _trace_write_census(trace_path)
    if trace_writes != {
        "operations": _integer(
            manifest_writes.get("operations"),
            "manifest write operations",
        ),
        "bytes": _integer(
            manifest_writes.get("bytes"),
            "manifest write bytes",
        ),
    }:
        raise FrontierWafError(
            "trace KV-write census differs from the digest-bound manifest"
        )

    config = _mapping(summary.get("config"), "summary.config")
    hbf_config = _mapping(config.get("hbf"), "summary.config.hbf")
    stats = _mapping(scenario.get("hbf_stats"), "scenario.hbf_stats")
    streaming = _mapping(
        scenario.get("layer_streaming"),
        "scenario.layer_streaming",
    )
    hybrid = _mapping(
        scenario.get("hybrid_path"),
        "scenario.hybrid_path",
    )
    logical_bytes = _integer(
        stats.get("logical_write_bytes"),
        "HBF logical write bytes",
    )
    physical_bytes = _integer(
        stats.get("physical_write_bytes"),
        "HBF physical write bytes",
    )
    writeback_bytes = _integer(
        streaming.get("writeback_bytes"),
        "layer-streaming writeback bytes",
    )
    hybrid_writeback_bytes = _integer(
        hybrid.get("hbf_backing_write_bytes"),
        "hybrid HBF writeback bytes",
    )
    if (
        logical_bytes != writeback_bytes
        or logical_bytes != hybrid_writeback_bytes
        or logical_bytes > trace_writes["bytes"]
    ):
        raise FrontierWafError(
            "HBF logical writes do not equal cold-KV writeback traffic"
        )

    page_size = _integer(
        hbf_config.get("page_size"),
        "HBF page size",
        minimum=1,
    )
    usable_capacity_bytes = _integer(
        hbf_config.get("capacity_bytes"),
        "usable HBF capacity",
        minimum=1,
    )
    total_pages = _integer(
        stats.get("total_pages"),
        "HBF total pages",
        minimum=1,
    )
    if total_pages * page_size != usable_capacity_bytes:
        raise FrontierWafError(
            "HBF total pages do not equal usable payload capacity"
        )

    data_programs = _integer(
        stats.get("data_programs"),
        "HBF foreground data programs",
    )
    mapping_programs = _integer(
        stats.get("mapping_page_programs"),
        "HBF mapping programs",
    )
    gc_relocations = _integer(
        stats.get("gc_relocations"),
        "HBF GC relocations",
    )
    page_programs = _integer(
        stats.get("page_programs"),
        "HBF page programs",
    )
    data_payload = _integer(
        stats.get("data_program_payload_bytes"),
        "HBF data-program payload bytes",
    )
    mapping_payload = _integer(
        stats.get("mapping_program_payload_bytes"),
        "HBF mapping-program payload bytes",
    )
    gc_payload = _integer(
        stats.get("gc_relocation_payload_bytes"),
        "HBF GC-relocation payload bytes",
    )
    if (
        page_programs != data_programs + mapping_programs + gc_relocations
        or data_payload != data_programs * page_size
        or mapping_payload != mapping_programs * page_size
        or gc_payload != gc_relocations * page_size
        or physical_bytes != data_payload + mapping_payload + gc_payload
        or physical_bytes != page_programs * page_size
    ):
        raise FrontierWafError(
            "HBF physical-write payload/program conservation failed"
        )
    if stats.get("waf_definition") != WAF_DEFINITION:
        raise FrontierWafError("HBF summary changed the canonical WAF definition")

    reported_waf = stats.get("waf")
    if logical_bytes == 0:
        if reported_waf is not None or physical_bytes != 0:
            raise FrontierWafError(
                "zero logical HBF writes require null WAF and zero programs"
            )
        waf: float | None = None
    else:
        expected_waf = physical_bytes / logical_bytes
        if (
            isinstance(reported_waf, bool)
            or not isinstance(reported_waf, (int, float))
            or not math.isfinite(float(reported_waf))
            or not math.isclose(
                float(reported_waf),
                expected_waf,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise FrontierWafError(
                "reported WAF differs from physical/logical write bytes"
            )
        waf = expected_waf

    return {
        "definition": WAF_DEFINITION,
        "logical_write_bytes": logical_bytes,
        "physical_write_bytes": physical_bytes,
        "waf": waf,
        "write_source": "mutable_kv_only",
        "read_only_weight_write_bytes": 0,
        "initial_image_included_in_workload_writes": False,
        "usable_hbf_capacity_bytes": usable_capacity_bytes,
        "physical_write_bytes_over_usable_hbf_capacity": (
            physical_bytes / usable_capacity_bytes
        ),
        "conservation_audit": {
            "data_program_payload_bytes": data_payload,
            "mapping_program_payload_bytes": mapping_payload,
            "gc_relocation_payload_bytes": gc_payload,
            "data_programs": data_programs,
            "mapping_page_programs": mapping_programs,
            "gc_relocations": gc_relocations,
            "page_programs": page_programs,
            "trace_kv_write_operations": trace_writes["operations"],
            "trace_kv_write_bytes": trace_writes["bytes"],
        },
    }
