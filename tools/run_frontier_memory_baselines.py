#!/usr/bin/env python3
"""Run one Frontier memory trace through the canonical four-baseline set.

The trace, placement contract, HBM geometry, and request-credit value are
shared.  The three backed cases differ only in backing technology.  The
all-HBM case is an explicitly labelled resident upper bound: it preserves the
HBM timing geometry and expands capacity only when the exported object address
space does not fit the point's physical HBM capacity.

This runner produces memory-system service evidence only.  It never promotes
dummy Frontier time into TTFT, TPOT, SLO, or time-throughput claims.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from frontier_waf import (
    FrontierWafError,
    audit_frontier_hbf_waf,
)
from prepare_frontier_residency_config import (
    ResidencyBindingError,
    prepare_residency_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CXL_OVERLAY = (
    REPOSITORY_ROOT
    / "configs/scenario_compare/external-cxl-memory.overlay"
)
DEFAULT_NVME_OVERLAY = (
    REPOSITORY_ROOT
    / "configs/scenario_compare/external-nvme-ssd.overlay"
)
SCHEMA = {
    "name": "hbfsim.frontier_memory_baseline_set",
    "version": 4,
}
SUMMARY_SCHEMA = {
    "name": "hbfsim.scenario_compare.summary",
    "version": 16,
}
BASELINE_SPECS = (
    ("all_hbm_upper_bound", "all-HBM"),
    ("hbm_hbf", "HBM+HBF-layer-streaming"),
    ("hbm_cxl_memory", "HBM+External-layer-streaming"),
    ("hbm_nvme_ssd", "HBM+External-layer-streaming"),
)
PLACEMENT_FIELDS = (
    "residency_policy",
    "explicit_residency_contract",
    "semantic_inputs_consumed",
    "unique_resident_footprint_pages",
    "unique_resident_footprint_bytes",
    "capacity_pressure_basis_bytes",
    "footprint_page_rounding_bytes",
    "hbm_capacity_bytes",
    "hbm_capacity_pressure",
    "hbm_only_resident_pages",
    "hot_kv_candidate_pages",
    "hot_kv_resident_pages",
    "data_pages",
    "model_weight_resident_pages",
    "model_weight_backing_pages",
    "cold_kv_backing_pages",
    "unknown_backing_pages",
    "backing_unique_pages",
    "resident_physical_pages",
    "effective_layer_buffer_pages",
    "unused_hbm_pages",
    "streamed_pages",
    "streamed_bytes",
    "foreground_resident_page_accesses",
    "foreground_buffer_page_accesses",
    "dirty_pages_written_back",
    "writeback_bytes",
    "max_layer_data_pages",
    "max_layer_data_bytes",
    "immutable_weight_logical_bytes",
    "runtime_overhead_logical_bytes",
    "block_table_logical_bytes",
    "active_buffer_logical_bytes_per_slot",
    "residency_page_size_bytes",
    "kv_block_stride_bytes",
    "logical_kv_blocks",
    "hot_kv_blocks",
    "cold_kv_blocks",
    "backing_request_credit_limit",
)


class BaselineSetError(ValueError):
    """The four-baseline set is incomplete or not byte-comparable."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise BaselineSetError(
            f"cannot read {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise BaselineSetError(f"{description} must be a JSON object")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaselineSetError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BaselineSetError(f"{name} must be an integer")
    if value < minimum:
        raise BaselineSetError(f"{name} must be >= {minimum}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path: Path, description: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise BaselineSetError(f"{description} is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _verify_snapshot(snapshot: dict[str, Any], description: str) -> None:
    path = Path(str(snapshot["path"]))
    actual = _snapshot(path, description)
    if actual != snapshot:
        raise BaselineSetError(
            f"{description} changed while the baseline set was running"
        )


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _config_u64(path: Path, key: str) -> int:
    found: int | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BaselineSetError(f"cannot read config {path}: {error}") from error
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise BaselineSetError(
                f"{path}:{line_number}: expected key=value"
            )
        name, value = (part.strip() for part in line.split("=", 1))
        if name != key:
            continue
        if found is not None:
            raise BaselineSetError(f"{path}: duplicate {key}")
        try:
            parsed = int(value, 10)
        except ValueError as error:
            raise BaselineSetError(
                f"{path}:{line_number}: {key} must be decimal"
            ) from error
        if parsed <= 0:
            raise BaselineSetError(f"{path}: {key} must be positive")
        found = parsed
    if found is None:
        raise BaselineSetError(f"{path}: missing {key}")
    return found


def _external_address_capacity(
    *,
    profile_capacity_bytes: int,
    backing_population_bytes: int,
    object_address_space_bytes: int,
) -> int:
    """Cover occupied backing pages and their global logical addresses."""
    return max(
        profile_capacity_bytes,
        backing_population_bytes,
        object_address_space_bytes,
    )


def _write_all_hbm_overlay(
    *,
    path: Path,
    trace: dict[str, Any],
    capacity_bytes: int,
) -> None:
    payload = "\n".join(
        (
            "# hbfsim.frontier_all_hbm_upper_bound schema=1",
            f"trace={trace['path']}",
            f"expected-trace-sha256={trace['sha256']}",
            f"expected-trace-bytes={trace['bytes']}",
            f"hbm-capacity-bytes={capacity_bytes}",
            f"flat-hbm-bytes={capacity_bytes}",
            f"static-direct-hbm-bytes={capacity_bytes}",
            "hbf-hbm-write-buffer-bytes=0",
            "",
        )
    )
    path.write_text(payload, encoding="utf-8")


def _run_case(
    *,
    binary: Path,
    stage: Path,
    baseline: str,
    scenario: str,
    configs: list[str],
    credit_limit: int,
    external_capacity_bytes: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[list[str], Path, Path, Path]:
    summary = stage / "summaries" / f"{baseline}.json"
    resolved_config = stage / "configs" / f"{baseline}.resolved.cfg"
    log = stage / "logs" / f"{baseline}.log"
    command = [str(binary)]
    for config in configs:
        command.extend(("--config", config))
    command.extend(
        (
            "--scenarios",
            scenario,
            "--max-outstanding-requests",
            str(credit_limit),
        )
    )
    if external_capacity_bytes is not None:
        command.extend(
            (
                "--external-backing-capacity-bytes",
                str(external_capacity_bytes),
            )
        )
    command.extend(
        (
            "--summary-json",
            str(summary.relative_to(stage)),
            "--config-out",
            str(resolved_config.relative_to(stage)),
        )
    )
    try:
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=stage,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BaselineSetError(
            f"{baseline} could not execute: {error}"
        ) from error
    if completed.returncode != 0:
        try:
            tail = log.read_text(encoding="utf-8")[-12000:]
        except OSError:
            tail = ""
        raise BaselineSetError(
            f"{baseline} failed with exit {completed.returncode}:\n{tail}"
        )
    for output, description in (
        (summary, "summary"),
        (resolved_config, "resolved config"),
        (log, "log"),
    ):
        if not output.is_file():
            raise BaselineSetError(
                f"{baseline} did not produce its {description}"
            )
    return command, summary, resolved_config, log


def _traffic_contract(manifest: dict[str, Any]) -> dict[str, int]:
    traffic = _mapping(
        manifest.get("traffic_census"),
        "manifest.traffic_census",
    )
    reads = _mapping(traffic.get("reads"), "traffic_census.reads")
    writes = _mapping(traffic.get("writes"), "traffic_census.writes")
    return {
        "ops": _integer(
            traffic.get("operations"),
            "traffic_census.operations",
            minimum=1,
        ),
        "logical_bytes": _integer(
            traffic.get("bytes"),
            "traffic_census.bytes",
            minimum=1,
        ),
        "reads": _integer(
            reads.get("operations"),
            "traffic_census.reads.operations",
        ),
        "writes": _integer(
            writes.get("operations"),
            "traffic_census.writes.operations",
        ),
    }


def _validate_manifest_claim_scope(manifest: dict[str, Any]) -> None:
    semantics = _mapping(manifest.get("semantics"), "manifest.semantics")
    expected = {
        "frontier_timing_is_hardware_calibrated": False,
        "eligible_claim_scope": "memory_system_service_only",
        "ttft_tpot_slo_claims_eligible": False,
        "time_based_throughput_claims_eligible": False,
        "export_mode": "dependency_barrier_batch_ready",
        "phase_dependency": "complete_before_next",
        "layer_scope": "global_phase_streaming_window",
        "model_weight_traffic": "read_only",
        "mutable_write_traffic": "kv_append_and_update_only",
        "capacity_pressure_basis": (
            "unique_resident_footprint_bytes/"
            "physical_hbm_capacity_bytes"
        ),
    }
    for field, value in expected.items():
        if semantics.get(field) != value:
            raise BaselineSetError(
                f"manifest.semantics.{field} is outside the canonical "
                "memory-service claim contract"
            )


def _validate_summary(
    *,
    path: Path,
    baseline: str,
    expected_scenario: str,
    expected_trace: dict[str, Any],
    expected_traffic: dict[str, int],
    credit_limit: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _load_object(path, f"{baseline} summary")
    if document.get("schema") != SUMMARY_SCHEMA:
        raise BaselineSetError(f"{baseline} summary schema drifted")
    if document.get("sanity") != "PASS":
        raise BaselineSetError(f"{baseline} summary did not pass sanity")
    workload = _mapping(document.get("workload"), f"{baseline}.workload")
    digest = _mapping(
        workload.get("trace_digest"),
        f"{baseline}.workload.trace_digest",
    )
    if (
        workload.get("trace_file_bytes") != expected_trace["bytes"]
        or digest
        != {
            "algorithm": "sha256",
            "value": expected_trace["sha256"],
        }
    ):
        raise BaselineSetError(f"{baseline} consumed a different trace")
    config = _mapping(document.get("config"), f"{baseline}.config")
    if (
        config.get("expected_trace_sha256") != expected_trace["sha256"]
        or config.get("expected_trace_bytes") != expected_trace["bytes"]
        or config.get("max_ops") != 0
        or config.get("max_outstanding_requests") != credit_limit
        or config.get("requested_scenarios") != [expected_scenario]
    ):
        raise BaselineSetError(
            f"{baseline} resolved trace/credit/scenario contract drifted"
        )
    scenarios = document.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or len(scenarios) != 1
        or not isinstance(scenarios[0], dict)
    ):
        raise BaselineSetError(
            f"{baseline} must contain exactly one scenario"
        )
    scenario = scenarios[0]
    if scenario.get("name") != expected_scenario:
        raise BaselineSetError(f"{baseline} scenario name drifted")
    if scenario.get("warnings") != []:
        raise BaselineSetError(f"{baseline} produced sanity warnings")
    for field, expected in expected_traffic.items():
        if scenario.get(field) != expected:
            raise BaselineSetError(
                f"{baseline}.{field}: expected {expected}, "
                f"got {scenario.get(field)!r}"
            )
    latency = _mapping(
        _mapping(
            scenario.get("time_breakdown"),
            f"{baseline}.time_breakdown",
        ).get("latency_work"),
        f"{baseline}.latency_work",
    )
    if latency.get("basis") != "offered_to_user_completion":
        raise BaselineSetError(f"{baseline} latency basis drifted")
    return document, scenario


def _service_metrics(
    scenario: dict[str, Any],
) -> dict[str, int | float | None]:
    time_breakdown = scenario["time_breakdown"]
    latency = time_breakdown["latency_work"]
    wall_clock = time_breakdown["wall_clock_ns"]
    hybrid = scenario["hybrid_path"]
    streaming = scenario.get("layer_streaming")
    if scenario["name"] == "HBM+HBF-layer-streaming":
        backing_read_bytes = (
            hybrid["hbf_static_read_bytes"]
            + hybrid["hbf_logical_read_bytes"]
        )
        backing_write_bytes = hybrid["hbf_backing_write_bytes"]
    elif scenario["name"] == "HBM+External-layer-streaming":
        backing_read_bytes = hybrid["external_backing_read_bytes"]
        backing_write_bytes = hybrid["external_backing_write_bytes"]
    else:
        backing_read_bytes = 0
        backing_write_bytes = 0
    return {
        "logical_bytes": scenario["logical_bytes"],
        "user_operations": latency["user_count"],
        "makespan_ns": wall_clock["makespan_ns"],
        "user_completion_span_ns": wall_clock["user_completion_span_ns"],
        "post_offer_user_completion_tail_ns": wall_clock[
            "post_offer_user_completion_tail_ns"
        ],
        "memory_system_logical_throughput_GBps": scenario[
            "makespan_throughput_GBps"
        ],
        "offered_average_ns": latency["average_ns"],
        "offered_p50_ns": latency["p50_ns"],
        "offered_p95_ns": latency["p95_ns"],
        "offered_max_ns": latency["max_ns"],
        "source_average_ns": latency["source_average_ns"],
        "source_p50_ns": latency["source_p50_ns"],
        "source_p95_ns": latency["source_p95_ns"],
        "source_max_ns": latency["source_max_ns"],
        "service_average_ns": latency["service_average_ns"],
        "service_p50_ns": latency["service_p50_ns"],
        "service_p95_ns": latency["service_p95_ns"],
        "service_max_ns": latency["service_max_ns"],
        "service_sum_work_ns": (
            latency["service_to_user_completion_sum_work_ns"]
        ),
        "hbm_user_accesses": scenario["hbm_user_accesses"],
        "hbm_background_accesses": scenario["hbm_background_accesses"],
        "backing_read_bytes": backing_read_bytes,
        "backing_write_bytes": backing_write_bytes,
        "capacity_pressure": (
            streaming["hbm_capacity_pressure"]
            if isinstance(streaming, dict)
            else None
        ),
        "backing_admission_max_wait_ns": (
            streaming["backing_admission_max_wait_ns"]
            if isinstance(streaming, dict)
            else None
        ),
        "exposed_prefetch_ns": (
            streaming["exposed_prefetch_ns"]
            if isinstance(streaming, dict)
            else None
        ),
        "hidden_prefetch_ns": (
            streaming["hidden_prefetch_ns"]
            if isinstance(streaming, dict)
            else None
        ),
    }


def _validate_cross_baseline_contract(
    *,
    summaries: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    contract: dict[str, Any],
    all_hbm_capacity_bytes: int,
    credit_limit: int,
) -> None:
    all_hbm_document, all_hbm = summaries["all_hbm_upper_bound"]
    if all_hbm_document["config"]["explicit_residency_contract"]:
        raise BaselineSetError("all-HBM upper bound retained backed placement")
    if (
        all_hbm_document["config"]["hbm_capacity_bytes"]
        != all_hbm_capacity_bytes
        or all_hbm["hbf_accesses"] != 0
        or all_hbm["external_accesses"] != 0
    ):
        raise BaselineSetError("all-HBM resident upper bound is not isolated")

    backed_names = (
        "hbm_hbf",
        "hbm_cxl_memory",
        "hbm_nvme_ssd",
    )
    backed = [summaries[name] for name in backed_names]
    expected_contract = {
        "page_size_bytes": contract["page_size_bytes"],
        "unique_resident_footprint_bytes": (
            contract["unique_resident_footprint_bytes"]
        ),
        "immutable_weight_bytes": (
            contract["immutable_weight_bytes"]
        ),
        "immutable_weight_pages": (
            contract["immutable_weight_pages"]
        ),
        "static_weight_resident_pages": (
            contract["static_weight_resident_pages"]
        ),
        "runtime_overhead_bytes": contract["runtime_overhead_bytes"],
        "block_table_bytes": contract["block_table_bytes"],
        "active_buffer_bytes_per_slot": (
            contract["active_buffer_bytes_per_slot"]
        ),
        "kv_region_begin": contract["kv_region_begin"],
        "kv_block_stride_bytes": contract["kv_block_stride_bytes"],
        "logical_kv_blocks": contract["logical_kv_blocks"],
        "hot_kv_blocks": contract["hot_kv_blocks"],
    }
    reference_plan: dict[str, Any] | None = None
    reference_hbm: dict[str, Any] | None = None
    reference_hybrid: tuple[int, int] | None = None
    for baseline, (document, scenario) in zip(
        backed_names,
        backed,
        strict=True,
    ):
        config = document["config"]
        if (
            not config["explicit_residency_contract"]
            or config["residency_contract"] != expected_contract
            or config["hbm_capacity_bytes"]
            != contract["physical_hbm_capacity_bytes"]
            or config["max_outstanding_requests"] != credit_limit
        ):
            raise BaselineSetError(
                f"{baseline} did not consume the canonical residency contract"
            )
        streaming = _mapping(
            scenario.get("layer_streaming"),
            f"{baseline}.layer_streaming",
        )
        plan = {field: streaming.get(field) for field in PLACEMENT_FIELDS}
        if reference_plan is None:
            reference_plan = plan
        elif plan != reference_plan:
            raise BaselineSetError(
                f"{baseline} placement differs from the HBF baseline"
            )
        hbm = dict(_mapping(config.get("hbm"), f"{baseline}.config.hbm"))
        hbm.pop("capacity_bytes", None)
        if reference_hbm is None:
            reference_hbm = hbm
        elif hbm != reference_hbm:
            raise BaselineSetError(
                f"{baseline} changed the HBM timing geometry"
            )
        hybrid = scenario["hybrid_path"]
        hbm_traffic = (
            hybrid["hbm_foreground_bytes"],
            hybrid["hbm_streaming_write_bytes"],
        )
        if reference_hybrid is None:
            reference_hybrid = hbm_traffic
        elif hbm_traffic != reference_hybrid:
            raise BaselineSetError(
                f"{baseline} changed foreground/streaming HBM traffic"
            )

    all_hbm_geometry = dict(all_hbm_document["config"]["hbm"])
    all_hbm_geometry.pop("capacity_bytes", None)
    if all_hbm_geometry != reference_hbm:
        raise BaselineSetError(
            "all-HBM upper bound changed HBM timing geometry"
        )

    hbf_scenario = summaries["hbm_hbf"][1]
    cxl_scenario = summaries["hbm_cxl_memory"][1]
    nvme_scenario = summaries["hbm_nvme_ssd"][1]
    hbf_hybrid = hbf_scenario["hybrid_path"]
    hbf_stats = _mapping(
        hbf_scenario.get("hbf_stats"),
        "hbm_hbf.hbf_stats",
    )
    hbf_streaming = _mapping(
        hbf_scenario.get("layer_streaming"),
        "hbm_hbf.layer_streaming",
    )
    cold_kv_pages = _integer(
        hbf_streaming.get("cold_kv_backing_pages"),
        "hbm_hbf cold-KV backing pages",
    )
    initial_data_pages = _integer(
        hbf_stats.get("initial_logical_data_pages"),
        "hbm_hbf initial logical data pages",
    )
    compact_initial_data_pages = _integer(
        hbf_stats.get("compact_initial_logical_data_pages"),
        "hbm_hbf compact initial logical data pages",
    )
    compact_live_data_pages = _integer(
        hbf_stats.get("compact_live_logical_data_pages"),
        "hbm_hbf compact live logical data pages",
    )
    compact_retired_data_pages = _integer(
        hbf_stats.get("compact_retired_logical_data_pages"),
        "hbm_hbf compact retired logical data pages",
    )
    initial_mapping_pages = _integer(
        hbf_stats.get("initial_mapping_pages"),
        "hbm_hbf initial mapping pages",
    )
    compact_initial_mapping_pages = _integer(
        hbf_stats.get("compact_initial_mapping_pages"),
        "hbm_hbf compact initial mapping pages",
    )
    if (
        hbf_stats.get("accounting_verified") is not True
        or initial_data_pages != cold_kv_pages
        or compact_initial_data_pages != cold_kv_pages
        or compact_live_data_pages + compact_retired_data_pages
        != cold_kv_pages
        or initial_mapping_pages != compact_initial_mapping_pages
        or (cold_kv_pages != 0 and initial_mapping_pages == 0)
    ):
        raise BaselineSetError(
            "HBF did not physically instantiate the complete mutable "
            "cold-KV initial population"
        )
    hbf_read_bytes = (
        hbf_hybrid["hbf_static_read_bytes"]
        + hbf_hybrid["hbf_logical_read_bytes"]
    )
    hbf_write_bytes = hbf_hybrid["hbf_backing_write_bytes"]
    for baseline, scenario, expected_kind in (
        ("hbm_cxl_memory", cxl_scenario, "cxl-memory"),
        ("hbm_nvme_ssd", nvme_scenario, "nvme-ssd"),
    ):
        stats = _mapping(
            scenario.get("external_backing_stats"),
            f"{baseline}.external_backing_stats",
        )
        hybrid = scenario["hybrid_path"]
        if (
            stats.get("kind") != expected_kind
            or hybrid["external_backing_read_bytes"] != hbf_read_bytes
            or hybrid["external_backing_write_bytes"] != hbf_write_bytes
        ):
            raise BaselineSetError(
                f"{baseline} changed backed traffic rather than media"
            )


def run_baselines(
    *,
    scenario_compare: Path,
    hardware_config: Path,
    manifest_path: Path,
    object_map_path: Path,
    output_dir: Path,
    cxl_overlay: Path = DEFAULT_CXL_OVERLAY,
    nvme_overlay: Path = DEFAULT_NVME_OVERLAY,
    credit_limit: int = 64,
    case_parallelism: int = 1,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if credit_limit <= 0:
        raise BaselineSetError("credit_limit must be positive")
    if case_parallelism < 1 or case_parallelism > len(BASELINE_SPECS):
        raise BaselineSetError(
            f"case_parallelism must be between 1 and {len(BASELINE_SPECS)}"
        )
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise BaselineSetError("timeout_seconds must be positive or None")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise BaselineSetError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    inputs = {
        "runner": _snapshot(
            Path(__file__).resolve(),
            "baseline runner",
        ),
        "scenario_compare": _snapshot(
            scenario_compare, "scenario_compare binary"
        ),
        "hardware_config": _snapshot(hardware_config, "hardware config"),
        "manifest": _snapshot(manifest_path, "Frontier manifest"),
        "object_map": _snapshot(object_map_path, "Frontier object map"),
        "cxl_overlay": _snapshot(cxl_overlay, "CXL overlay"),
        "nvme_overlay": _snapshot(nvme_overlay, "NVMe overlay"),
    }
    binary = Path(inputs["scenario_compare"]["path"])
    hardware = Path(inputs["hardware_config"]["path"])
    manifest_file = Path(inputs["manifest"]["path"])
    object_map_file = Path(inputs["object_map"]["path"])
    cxl = Path(inputs["cxl_overlay"]["path"])
    nvme = Path(inputs["nvme_overlay"]["path"])
    manifest = _load_object(manifest_file, "Frontier manifest")
    model = _mapping(manifest.get("model"), "manifest.model")
    model_name = model.get("name")
    profile_id = model.get("profile_id")
    if model_name == "llama31_70b" and profile_id in {
        "w8a16-kv-bf16",
        "bf16-kv-bf16",
    }:
        workload_role = "paper_candidate"
    elif (
        isinstance(model_name, str)
        and model_name
        and isinstance(profile_id, str)
        and profile_id.endswith("-ci-smoke")
    ):
        workload_role = "ci_smoke"
    else:
        raise BaselineSetError(
            "canonical baseline runner accepts Llama 3.1 70B "
            "W8A16/KV-BF16 or BF16/KV-BF16 paper candidates and "
            "explicit *-ci-smoke structural fixtures only"
        )
    _validate_manifest_claim_scope(manifest)
    expected_traffic = _traffic_contract(manifest)
    selection = _mapping(manifest.get("selection"), "manifest.selection")
    full_replay = selection.get("is_full_replay")
    if not isinstance(full_replay, bool):
        raise BaselineSetError("manifest.selection.is_full_replay is invalid")

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    try:
        for directory in ("configs", "summaries", "logs"):
            (stage / directory).mkdir()
        hardware_hbm_capacity = _config_u64(
            hardware,
            "hbm-capacity-bytes",
        )
        hardware_hbm_stacks = _config_u64(hardware, "hbm-stacks")
        if (
            hardware_hbm_capacity
            != hardware_hbm_stacks * 48 * 1024**3
        ):
            raise BaselineSetError(
                "hardware config is not a canonical 48 GiB-per-stack "
                "96/192/288/384 GiB topology"
            )
        residency_config = stage / "configs/residency.cfg"
        residency_receipt_path = stage / "configs/residency.receipt.json"
        try:
            residency_receipt = prepare_residency_config(
                manifest_path=manifest_file,
                object_map_path=object_map_file,
                output_config_path=residency_config,
                receipt_path=residency_receipt_path,
                physical_hbm_capacity_bytes=hardware_hbm_capacity,
            )
        except ResidencyBindingError as error:
            raise BaselineSetError(
                f"cannot bind Frontier residency: {error}"
            ) from error
        contract = _mapping(
            residency_receipt.get("contract"),
            "residency receipt contract",
        )
        trace = _mapping(
            _mapping(
                residency_receipt.get("inputs"),
                "residency receipt inputs",
            ).get("trace"),
            "residency receipt trace",
        )
        trace_snapshot = _snapshot(
            Path(str(trace.get("path"))),
            "Frontier trace",
        )
        if (
            trace_snapshot["bytes"] != trace.get("bytes")
            or trace_snapshot["sha256"] != trace.get("sha256")
        ):
            raise BaselineSetError(
                "residency receipt trace identity is not current"
            )
        inputs["trace"] = trace_snapshot
        physical_hbm_capacity = _integer(
            contract.get("physical_hbm_capacity_bytes"),
            "physical HBM capacity",
            minimum=1,
        )
        if hardware_hbm_capacity != physical_hbm_capacity:
            raise BaselineSetError(
                "placement compiler did not target the selected hardware "
                "HBM capacity"
            )
        object_address_space = _integer(
            contract.get("object_address_space_bytes"),
            "object address-space bytes",
            minimum=1,
        )
        all_hbm_capacity = max(
            physical_hbm_capacity,
            object_address_space,
        )
        all_hbm_overlay = stage / "configs/all-hbm-upper-bound.cfg"
        _write_all_hbm_overlay(
            path=all_hbm_overlay,
            trace=trace,
            capacity_bytes=all_hbm_capacity,
        )
        generated_snapshots = {
            "residency config": _snapshot(
                residency_config,
                "generated residency config",
            ),
            "residency receipt": _snapshot(
                residency_receipt_path,
                "generated residency receipt",
            ),
            "all-HBM overlay": _snapshot(
                all_hbm_overlay,
                "generated all-HBM overlay",
            ),
        }
        backing_capacity = _integer(
            contract.get("backing_unique_pages"),
            "backing unique pages",
        ) * _integer(
            contract.get("page_size_bytes"),
            "residency page size",
            minimum=1,
        )
        cxl_capacity = _external_address_capacity(
            profile_capacity_bytes=_config_u64(
                cxl,
                "external-backing-capacity-bytes",
            ),
            backing_population_bytes=backing_capacity,
            object_address_space_bytes=object_address_space,
        )
        nvme_capacity = _external_address_capacity(
            profile_capacity_bytes=_config_u64(
                nvme,
                "external-backing-capacity-bytes",
            ),
            backing_population_bytes=backing_capacity,
            object_address_space_bytes=object_address_space,
        )

        case_arguments = {
            "all_hbm_upper_bound": {
                "configs": [
                    str(hardware),
                    str(all_hbm_overlay.relative_to(stage)),
                ],
            },
            "hbm_hbf": {
                "configs": [
                    str(hardware),
                    str(residency_config.relative_to(stage)),
                ],
            },
            "hbm_cxl_memory": {
                "configs": [
                    str(hardware),
                    str(residency_config.relative_to(stage)),
                    str(cxl),
                ],
                "external_capacity_bytes": cxl_capacity,
            },
            "hbm_nvme_ssd": {
                "configs": [
                    str(hardware),
                    str(residency_config.relative_to(stage)),
                    str(nvme),
                ],
                "external_capacity_bytes": nvme_capacity,
            },
        }
        summaries: dict[
            str,
            tuple[dict[str, Any], dict[str, Any]],
        ] = {}
        baseline_receipts: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=case_parallelism) as executor:
            futures = {
                baseline: executor.submit(
                    _run_case,
                    binary=binary,
                    stage=stage,
                    baseline=baseline,
                    scenario=scenario,
                    credit_limit=credit_limit,
                    timeout_seconds=timeout_seconds,
                    **case_arguments[baseline],
                )
                for baseline, scenario in BASELINE_SPECS
            }
            for baseline, scenario in BASELINE_SPECS:
                (
                    command,
                    summary_path,
                    resolved_path,
                    log_path,
                ) = futures[baseline].result()
                document, scenario_result = _validate_summary(
                    path=summary_path,
                    baseline=baseline,
                    expected_scenario=scenario,
                    expected_trace=trace,
                    expected_traffic=expected_traffic,
                    credit_limit=credit_limit,
                )
                summaries[baseline] = (document, scenario_result)
                baseline_receipts[baseline] = {
                    "scenario": scenario,
                    "command": command,
                    "summary": _artifact(summary_path, stage),
                    "resolved_config": _artifact(resolved_path, stage),
                    "log": _artifact(log_path, stage),
                    "memory_service_metrics": _service_metrics(
                        scenario_result
                    ),
                }

        _validate_cross_baseline_contract(
            summaries=summaries,
            contract=contract,
            all_hbm_capacity_bytes=all_hbm_capacity,
            credit_limit=credit_limit,
        )
        try:
            canonical_waf = audit_frontier_hbf_waf(
                summary=summaries["hbm_hbf"][0],
                scenario=summaries["hbm_hbf"][1],
                manifest=manifest,
                trace_path=Path(inputs["trace"]["path"]),
            )
        except FrontierWafError as error:
            raise BaselineSetError(
                f"canonical HBF WAF audit failed: {error}"
            ) from error
        for name, snapshot in inputs.items():
            _verify_snapshot(snapshot, name)
        for name, snapshot in generated_snapshots.items():
            _verify_snapshot(snapshot, name)

        paper_blockers = [
            "suite-level certificate and complete capacity grid not attached",
            "external CXL/NVMe timing parameters are uncalibrated sensitivity values",
            "Frontier compute/communication timing is not calibrated",
        ]
        if workload_role != "paper_candidate":
            paper_blockers.insert(
                0,
                "workload is an explicit CI smoke fixture, not Llama 3.1 70B",
            )
        if not full_replay:
            paper_blockers.insert(
                0,
                "selected trace is not the complete Frontier replay",
            )
        payload = {
            "schema": SCHEMA,
            "result": "pass",
            "claim_scope": {
                "eligible": "memory_system_service_only",
                "ttft_tpot_slo": False,
                "time_based_throughput": False,
            },
            "source": {
                **inputs,
                "model": {
                    "name": model["name"],
                    "profile_id": model["profile_id"],
                    "workload_role": workload_role,
                },
                "selection": {
                    "is_full_replay": full_replay,
                    "batches": selection.get("batches"),
                    "total_frontier_batches": selection.get(
                        "total_frontier_batches"
                    ),
                },
                "generated_configs": {
                    "residency": _artifact(
                        residency_config,
                        stage,
                    ),
                    "residency_receipt": _artifact(
                        residency_receipt_path,
                        stage,
                    ),
                    "all_hbm_upper_bound": _artifact(
                        all_hbm_overlay,
                        stage,
                    ),
                },
            },
            "contract": {
                "traffic": expected_traffic,
                "request_credit_limit_per_active_pool": credit_limit,
                "host_case_parallelism": case_parallelism,
                "hardware_topology": {
                    "hbm_stacks": hardware_hbm_stacks,
                    "hbm_capacity_bytes": hardware_hbm_capacity,
                    "hbf_stacks": _config_u64(
                        hardware,
                        "hbf-stacks",
                    ),
                },
                "backed_placement_is_byte_identical": True,
                "backed_scheduler_is_identical": True,
                "all_hbm_upper_bound": {
                    "placement": "fully_resident",
                    "same_hbm_timing_geometry": True,
                    "physical_point_capacity_bytes": physical_hbm_capacity,
                    "provisioned_capacity_bytes": all_hbm_capacity,
                    "capacity_expanded": (
                        all_hbm_capacity > physical_hbm_capacity
                    ),
                    "provisioning_rule": (
                        "max(physical_hbm_capacity_bytes,"
                        "object_address_space_bytes)"
                    ),
                },
                "external_capacity": {
                    "rule": (
                        "max(profile_capacity_bytes,"
                        "page_allocated_backing_population_bytes,"
                        "global_logical_address_span_bytes)"
                    ),
                    "required_backing_bytes": backing_capacity,
                    "required_logical_address_span_bytes": (
                        object_address_space
                    ),
                    "cxl_memory_bytes": cxl_capacity,
                    "nvme_ssd_bytes": nvme_capacity,
                },
                "residency": contract,
                "canonical_hbf_waf": canonical_waf,
            },
            "baselines": baseline_receipts,
            "eligibility": {
                "four_baseline_contract_valid": True,
                "canonical_waf_valid": True,
                "full_frontier_replay": full_replay,
                "paper_result_eligible": False,
                "paper_blockers": paper_blockers,
            },
        }
        receipt_path = stage / "baseline-set.receipt.json"
        receipt_path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            raise BaselineSetError(
                f"output directory appeared during run: {output_dir}"
            )
        os.replace(stage, output_dir)
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path, required=True)
    parser.add_argument("--hardware-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cxl-overlay",
        type=Path,
        default=DEFAULT_CXL_OVERLAY,
    )
    parser.add_argument(
        "--nvme-overlay",
        type=Path,
        default=DEFAULT_NVME_OVERLAY,
    )
    parser.add_argument("--credit-limit", type=int, default=64)
    parser.add_argument(
        "--case-parallelism",
        type=int,
        default=1,
        help=(
            "number of independent baseline processes to run concurrently; "
            "simulation inputs and atomic publication are unchanged"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0,
        help="0 disables the per-baseline timeout",
    )
    arguments = parser.parse_args()
    timeout = (
        None
        if arguments.timeout_seconds == 0
        else arguments.timeout_seconds
    )
    try:
        payload = run_baselines(
            scenario_compare=arguments.scenario_compare,
            hardware_config=arguments.hardware_config,
            manifest_path=arguments.manifest,
            object_map_path=arguments.object_map,
            output_dir=arguments.output_dir,
            cxl_overlay=arguments.cxl_overlay,
            nvme_overlay=arguments.nvme_overlay,
            credit_limit=arguments.credit_limit,
            case_parallelism=arguments.case_parallelism,
            timeout_seconds=timeout,
        )
    except (BaselineSetError, ResidencyBindingError) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        "Frontier four-baseline set: PASS "
        f"model={payload['source']['model']['profile_id']} "
        f"full_replay={str(payload['eligibility']['full_frontier_replay']).lower()}"
    )
    print(f"receipt={arguments.output_dir.resolve() / 'baseline-set.receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
