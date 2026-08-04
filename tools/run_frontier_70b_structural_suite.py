#!/usr/bin/env python3
"""Run the one canonical 70B Frontier structural suite and audit every window.

This entry point deliberately produces memory-system structural evidence only.
It always runs the verified steady, burst, and long-context/decode-tail
windows with the primary W8A16/KV-BF16 profile. Dummy Frontier timing is
required and is never promoted to TTFT, TPOT, SLO, or time-throughput evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, TextIO

from audit_frontier_replay import audit_replay
from prepare_frontier_integration import (
    BUNDLE_MANIFEST,
    prepare as prepare_frontier_integration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
FRONTIER_REPOSITORY = "https://github.com/NetX-lab/Frontier"
FRONTIER_REVISION = "a4b22df8211864bf229258ecdfbe680f048f2d77"
INTEGRATION_ID = "hbfsim.frontier-hybrid-residency.v8"
SUITE_ID = "qwen-thinking-llama31-70b-production-v1"
MODEL_NAME = "llama31_70b"
PRIMARY_PRECISION_PROFILE = "w8a16-kv-bf16"
STRUCTURAL_HBM_CAPACITY_BYTES = 96 * 1024**3
STRUCTURAL_CAPACITY_PRESSURE = 1.0
STRUCTURAL_RUNTIME_OVERHEAD_BYTES = 8 * 1024**3
WINDOWS = (
    "steady",
    "burst",
    "long_context_decode_tail",
)
REQUIRED_RUNTIME_BEHAVIORS = (
    "multi_request_batching",
    "prefix_kv_lifecycle",
    "steady_burst_long_context_decode_tail",
)


class StructuralSuiteError(ValueError):
    """The canonical structural suite cannot be run or qualified."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructuralSuiteError(
            f"cannot load {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise StructuralSuiteError(f"{description} must be a JSON object")
    return value


def _expect_equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise StructuralSuiteError(
            f"{path}: expected {expected!r}, got {actual!r}"
        )


def _git(*arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise StructuralSuiteError(
            f"git {' '.join(arguments)} failed: {process.stderr.strip()}"
        )
    return process.stdout


def _require_clean_hbfsim_commit() -> str:
    tracked_status = _git(
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise StructuralSuiteError(
            "HBFSim tracked files must be clean before a suite run:\n"
            + tracked_status.rstrip()
        )
    revision = _git("rev-parse", "HEAD").strip()
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise StructuralSuiteError("HBFSim HEAD is not a full lowercase SHA")
    return revision


def validate_request_suite_verification(
    receipt_path: Path,
) -> tuple[dict[str, Any], Path]:
    receipt_path = receipt_path.resolve()
    receipt = _load_object(
        receipt_path,
        "request-suite verification receipt",
    )
    _expect_equal(
        receipt.get("schema"),
        {
            "name": "hbfsim.qwen_bailian_frontier_suite_verification",
            "version": 1,
        },
        "request-suite verification schema",
    )
    _expect_equal(receipt.get("result"), "pass", "request-suite result")
    _expect_equal(receipt.get("suite_id"), SUITE_ID, "request-suite ID")
    _expect_equal(
        receipt.get("frontier_workload"),
        {
            "model": {
                "identity": "meta-llama/Llama-3.1-70B",
                "frontier_name": MODEL_NAME,
            },
            "primary_precision_profile": PRIMARY_PRECISION_PROFILE,
            "sensitivity_precision_profiles": ["bf16-kv-bf16"],
            "required_runtime_behaviors": list(
                REQUIRED_RUNTIME_BEHAVIORS
            ),
        },
        "request-suite Frontier workload",
    )
    windows = receipt.get("windows")
    if not isinstance(windows, dict) or set(windows) != set(WINDOWS):
        raise StructuralSuiteError(
            "request-suite verification must cover exactly all three windows"
        )
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise StructuralSuiteError(
            "request-suite verification artifacts must be an object"
        )
    suite_manifest = artifacts.get("suite_manifest")
    if not isinstance(suite_manifest, dict):
        raise StructuralSuiteError(
            "request-suite suite_manifest artifact must be an object"
        )
    suite_manifest_path_value = suite_manifest.get("path")
    if (
        not isinstance(suite_manifest_path_value, str)
        or not suite_manifest_path_value
    ):
        raise StructuralSuiteError(
            "request-suite suite_manifest.path must be non-empty"
        )
    suite_manifest_path = Path(suite_manifest_path_value).resolve()
    if not suite_manifest_path.is_file():
        raise StructuralSuiteError(
            f"request-suite manifest is missing: {suite_manifest_path}"
        )
    _expect_equal(
        suite_manifest.get("sha256"),
        _sha256_file(suite_manifest_path),
        "request-suite manifest digest",
    )
    _expect_equal(
        suite_manifest.get("bytes"),
        suite_manifest_path.stat().st_size,
        "request-suite manifest bytes",
    )
    return receipt, suite_manifest_path.parent


def _build_frontier_command(
    *,
    frontier_python: Path,
    request_csv: Path,
    output_dir: Path,
    run_id: str,
    precision_profile: str = PRIMARY_PRECISION_PROFILE,
    physical_hbm_capacity_bytes: int = STRUCTURAL_HBM_CAPACITY_BYTES,
    capacity_pressure: str = str(STRUCTURAL_CAPACITY_PRESSURE),
    runtime_overhead_bytes: int = STRUCTURAL_RUNTIME_OVERHEAD_BYTES,
) -> list[str]:
    return [
        str(frontier_python),
        "-m",
        "frontier.main",
        "--simulation_mode",
        "offline",
        "--sys_arch",
        "co-location",
        "--cc_backend_config_type",
        "analytical",
        "--seed",
        "42",
        "--log_level",
        "warning",
        "--cluster_config_num_replicas",
        "1",
        "--replica_config_device",
        "h20",
        "--replica_config_model_name",
        MODEL_NAME,
        "--replica_config_memory_precision_profile",
        precision_profile,
        "--replica_config_memory_margin_fraction",
        "0",
        "--replica_config_attn_tensor_parallel_size",
        "1",
        "--replica_config_moe_tensor_parallel_size",
        "1",
        "--replica_config_moe_expert_parallel_size",
        "1",
        "--replica_config_num_pipeline_stages",
        "1",
        "--replica_config_attn_data_parallel_size",
        "1",
        "--replica_scheduler_config_type",
        "vllm_v1",
        "--decode_cuda_graph_mode",
        "none",
        "--vllm_v1_scheduler_config_batch_size_cap",
        "128",
        "--vllm_v1_scheduler_config_max_tokens_in_batch",
        "8192",
        "--vllm_v1_scheduler_config_enable_chunked_prefill",
        "--vllm_v1_scheduler_config_long_prefill_token_threshold",
        "8192",
        "--no-vllm_v1_scheduler_config_enable_preemption",
        "--vllm_v1_scheduler_config_enable_prefix_caching",
        "--vllm_v1_scheduler_config_prefix_caching_hash_algo",
        "builtin",
        "--vllm_v1_scheduler_config_num_preallocate_tokens",
        "0",
        "--vllm_v1_scheduler_config_num_blocks_mode",
        "hybrid_residency",
        "--vllm_v1_scheduler_config_hybrid_physical_hbm_capacity_bytes",
        str(physical_hbm_capacity_bytes),
        "--vllm_v1_scheduler_config_hybrid_capacity_pressure_target",
        capacity_pressure,
        "--vllm_v1_scheduler_config_non_kv_cache_overhead_bytes",
        str(runtime_overhead_bytes),
        "--vllm_v1_scheduler_config_runtime_weights_memory_source",
        "param_counter",
        "--request_generator_config_type",
        "trace_replay",
        "--trace_request_generator_config_trace_file",
        str(request_csv),
        "--trace_request_generator_config_max_tokens",
        "131072",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
        "1",
        "--metrics_config_output_dir",
        str(output_dir),
        "--metrics_config_run_id",
        run_id,
        "--metrics_config_write_metrics",
        "--metrics_config_store_request_metrics",
        "--metrics_config_store_batch_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_utilization_metrics",
        "--metrics_config_store_frontier_stage_batch_ledger",
        "--no-metrics_config_store_frontier_stage_batch_ledger_summary",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_write_json_trace",
    ]


def _window_summary(
    audit: dict[str, Any],
    window: str,
    *,
    precision_profile: str = PRIMARY_PRECISION_PROFILE,
    physical_hbm_capacity_bytes: int = STRUCTURAL_HBM_CAPACITY_BYTES,
    capacity_pressure: str = str(STRUCTURAL_CAPACITY_PRESSURE),
    runtime_overhead_bytes: int = STRUCTURAL_RUNTIME_OVERHEAD_BYTES,
) -> dict[str, Any]:
    _expect_equal(audit.get("result"), "pass", f"{window} audit result")
    request_suite = audit.get("request_suite")
    if not isinstance(request_suite, dict):
        raise StructuralSuiteError(f"{window} request_suite must be an object")
    _expect_equal(
        request_suite.get("window_id"),
        window,
        f"{window} audit window",
    )
    requests = audit.get("requests")
    ledger = audit.get("ledger")
    lifecycle = audit.get("kv_lifecycle")
    eligibility = audit.get("eligibility")
    model_memory = audit.get("model_memory")
    residency_plan = audit.get("residency_plan")
    for name, value in (
        ("requests", requests),
        ("ledger", ledger),
        ("kv_lifecycle", lifecycle),
        ("eligibility", eligibility),
        ("model_memory", model_memory),
        ("residency_plan", residency_plan),
    ):
        if not isinstance(value, dict):
            raise StructuralSuiteError(
                f"{window} audit {name} must be an object"
            )
    _expect_equal(requests.get("count"), 256, f"{window} request count")
    batch_size_counts = ledger.get("batch_size_counts")
    if not isinstance(batch_size_counts, dict) or not batch_size_counts:
        raise StructuralSuiteError(
            f"{window} batch_size_counts must be non-empty"
        )
    try:
        max_batch_size = max(
            int(size)
            for size, count in batch_size_counts.items()
            if int(count) > 0
        )
    except (TypeError, ValueError) as error:
        raise StructuralSuiteError(
            f"{window} batch_size_counts are malformed"
        ) from error
    if max_batch_size <= 1:
        raise StructuralSuiteError(
            f"{window} did not execute a multi-request batch"
        )
    event_counts = lifecycle.get("event_counts")
    if not isinstance(event_counts, dict):
        raise StructuralSuiteError(
            f"{window} lifecycle event_counts must be an object"
        )
    _expect_equal(
        event_counts.get("prefix_lookup"),
        256,
        f"{window} prefix lookup count",
    )
    _expect_equal(
        event_counts.get("prefix_admission"),
        256,
        f"{window} prefix admission count",
    )
    reservation = lifecycle.get("no_preemption_admission")
    if not isinstance(reservation, dict):
        raise StructuralSuiteError(
            f"{window} reservation audit must be an object"
        )
    _expect_equal(
        reservation.get("policy"),
        "full_request_kv_reservation_v1",
        f"{window} reservation policy",
    )
    max_reserved = reservation.get("max_reserved_blocks")
    capacity = reservation.get("capacity_blocks")
    if (
        not isinstance(max_reserved, int)
        or not isinstance(capacity, int)
        or max_reserved > capacity
    ):
        raise StructuralSuiteError(
            f"{window} full-request reservation exceeds capacity"
        )
    _expect_equal(
        model_memory.get("profile_id"),
        precision_profile,
        f"{window} memory precision",
    )
    _expect_equal(
        residency_plan.get("policy"),
        "hybrid_residency_v1",
        f"{window} residency policy",
    )
    _expect_equal(
        residency_plan.get("physical_hbm_capacity_bytes"),
        physical_hbm_capacity_bytes,
        f"{window} physical HBM capacity",
    )
    _expect_equal(
        residency_plan.get("target_pressure"),
        capacity_pressure,
        f"{window} capacity pressure target",
    )
    _expect_equal(
        residency_plan.get("runtime_overhead_bytes"),
        runtime_overhead_bytes,
        f"{window} runtime overhead",
    )
    _expect_equal(
        capacity,
        residency_plan.get("num_logical_kv_blocks"),
        f"{window} logical KV capacity",
    )
    _expect_equal(
        eligibility.get("eligible_for_memory_system_service_claims"),
        True,
        f"{window} memory-system claim eligibility",
    )
    _expect_equal(
        eligibility.get("eligible_for_ttft_tpot_slo_claims"),
        False,
        f"{window} TTFT/TPOT/SLO eligibility",
    )
    _expect_equal(
        eligibility.get("eligible_for_time_based_throughput_claims"),
        False,
        f"{window} time-throughput eligibility",
    )
    _expect_equal(
        eligibility.get("frontier_timing_kind"),
        "dummy",
        f"{window} timing kind",
    )
    return {
        "requests": 256,
        "prefill_tokens": requests.get("prefill_tokens"),
        "decode_tokens": requests.get("decode_tokens"),
        "total_tokens": requests.get("total_tokens"),
        "batches": ledger.get("rows"),
        "max_batch_size": max_batch_size,
        "prefix_hit_ratio": audit["accounting"]["prefix_hit_ratio"],
        "kv_lifecycle_events": lifecycle.get("events"),
        "kv_capacity_blocks": capacity,
        "max_reserved_blocks": max_reserved,
        "max_referenced_physical_blocks": reservation.get(
            "max_referenced_physical_blocks"
        ),
        "residency_plan": residency_plan,
        "timing_kind": "dummy",
        "claim_scope": "memory_system_service_only",
    }


def _open_atomic_text(path: Path) -> tuple[TextIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    return handle, Path(handle.name)


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    handle, temporary = _open_atomic_text(path)
    try:
        with handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_suite(
    *,
    frontier_dir: Path,
    frontier_python: Path,
    request_suite_verification: Path,
    frontier_integration_receipt: Path,
    output_dir: Path,
) -> dict[str, Any]:
    hbfsim_revision = _require_clean_hbfsim_commit()
    frontier_dir = frontier_dir.resolve()
    frontier_python = Path(os.path.abspath(frontier_python))
    request_suite_verification = request_suite_verification.resolve()
    frontier_integration_receipt = frontier_integration_receipt.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise StructuralSuiteError(
            f"output directory already exists: {output_dir}"
        )
    if not frontier_python.is_file():
        raise StructuralSuiteError(
            f"Frontier Python executable is missing: {frontier_python}"
        )
    verified_integration = prepare_frontier_integration(
        action="verify",
        frontier_dir=frontier_dir,
    )
    _expect_equal(
        verified_integration.get("integration_id"),
        INTEGRATION_ID,
        "Frontier integration ID",
    )
    receipt = _load_object(
        frontier_integration_receipt,
        "Frontier integration receipt",
    )
    _expect_equal(
        receipt,
        verified_integration,
        "Frontier integration receipt",
    )
    _, suite_dir = validate_request_suite_verification(
        request_suite_verification
    )

    output_dir.mkdir(parents=True)
    logs_dir = output_dir / "logs"
    audits_dir = output_dir / "audits"
    logs_dir.mkdir()
    audits_dir.mkdir()
    window_receipts: dict[str, Any] = {}
    for window in WINDOWS:
        request_csv = suite_dir / f"{window}.frontier.csv"
        request_manifest = suite_dir / f"{window}.adapter-manifest.json"
        if not request_csv.is_file() or not request_manifest.is_file():
            raise StructuralSuiteError(
                f"verified suite is missing {window} request artifacts"
            )
        run_id = f"{window}_w8"
        command = _build_frontier_command(
            frontier_python=frontier_python,
            request_csv=request_csv,
            output_dir=output_dir,
            run_id=run_id,
        )
        log_path = logs_dir / f"{window}.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(
                command,
                cwd=frontier_dir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0:
            raise StructuralSuiteError(
                f"Frontier {window} run failed with exit "
                f"{process.returncode}; see {log_path}"
            )
        frontier_output = (
            output_dir
            / MODEL_NAME
            / "offline_batch"
            / run_id
        )
        audit_path = audits_dir / f"{window}.audit.json"
        audit = audit_replay(
            request_csv=request_csv,
            request_manifest=request_manifest,
            request_suite_verification=request_suite_verification,
            frontier_output_dir=frontier_output,
            frontier_revision=FRONTIER_REVISION,
            frontier_repository=FRONTIER_REPOSITORY,
            frontier_integration_manifest=BUNDLE_MANIFEST,
            frontier_integration_receipt=frontier_integration_receipt,
            output=audit_path,
        )
        window_receipts[window] = {
            "summary": _window_summary(audit, window),
            "audit": {
                "path": str(audit_path),
                "bytes": audit_path.stat().st_size,
                "sha256": _sha256_file(audit_path),
            },
            "log": {
                "path": str(log_path),
                "bytes": log_path.stat().st_size,
                "sha256": _sha256_file(log_path),
            },
            "command": command,
        }

    payload = {
        "schema": {
            "name": "hbfsim.frontier_70b_structural_suite",
            "version": 2,
        },
        "result": "pass",
        "suite_id": SUITE_ID,
        "model": MODEL_NAME,
        "precision_profile": PRIMARY_PRECISION_PROFILE,
        "runtime_behaviors": {
            behavior: True for behavior in REQUIRED_RUNTIME_BEHAVIORS
        },
        "eligibility": {
            "structural_memory_suite_valid": True,
            "memory_system_service_claims": True,
            "ttft_tpot_slo_claims": False,
            "time_based_throughput_claims": False,
            "paper_result_eligible": False,
            "paper_blockers": [
                "complete resident-footprint/HBM-capacity grid not attached",
                "equal-workload HBM/HBF/CXL/NVMe baselines not attached",
                "canonical WAF audit not attached",
                "Frontier compute/communication timing is dummy",
            ],
        },
        "source": {
            "hbfsim_revision": hbfsim_revision,
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "frontier_repository": FRONTIER_REPOSITORY,
            "frontier_revision": FRONTIER_REVISION,
            "integration": verified_integration,
            "request_suite_verification": {
                "path": str(request_suite_verification),
                "bytes": request_suite_verification.stat().st_size,
                "sha256": _sha256_file(request_suite_verification),
            },
        },
        "windows": window_receipts,
    }
    _write_atomic_json(
        output_dir / "frontier-70b-structural-suite.json",
        payload,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--frontier-python", type=Path)
    parser.add_argument(
        "--request-suite-verification",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--frontier-integration-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    frontier_python = (
        arguments.frontier_python
        if arguments.frontier_python is not None
        else arguments.frontier_dir / ".venv/bin/python"
    )
    try:
        payload = run_suite(
            frontier_dir=arguments.frontier_dir,
            frontier_python=frontier_python,
            request_suite_verification=(
                arguments.request_suite_verification
            ),
            frontier_integration_receipt=(
                arguments.frontier_integration_receipt
            ),
            output_dir=arguments.output_dir,
        )
    except (
        OSError,
        StructuralSuiteError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        "qualified structural windows: "
        + ", ".join(payload["windows"])
    )
    print(
        "receipt="
        + str(
            arguments.output_dir.resolve()
            / "frontier-70b-structural-suite.json"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
