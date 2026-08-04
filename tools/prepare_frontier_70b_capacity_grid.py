#!/usr/bin/env python3
"""Prepare the one canonical Llama 3.1 70B capacity-pressure grid.

This is a deterministic preflight, not an experiment result.  It derives every
planner point from the tracked model descriptors, verifies the three production
request CSVs, and separates three states:

* planner_infeasible: the target footprint cannot hold the fixed objects;
* request_infeasible: a complete request can never fit in the logical KV pool;
* execution_candidate: static checks pass, but a full Frontier replay is still
  required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, TextIO

from export_frontier_memory_trace import (
    ExportError,
    derive_dense_model_capacity_inputs,
)
from frontier_hybrid_residency import (
    HybridResidencyError,
    build_hybrid_residency_plan,
)
from run_frontier_70b_structural_suite import (
    MODEL_NAME,
    PRIMARY_PRECISION_PROFILE,
    SUITE_ID,
    WINDOWS,
    StructuralSuiteError,
    validate_request_suite_verification,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_NAME = "hbfsim.frontier_70b_capacity_grid_preflight"
SCHEMA_VERSION = 1
RUNTIME_OVERHEAD_BYTES = 8 * 1024**3
PRESSURES = (0.75, 1.0, 1.25, 1.5, 2.0)
TOPOLOGIES = (
    {
        "topology": "2h6f",
        "hbm_stacks": 2,
        "hbf_stacks": 6,
        "physical_hbm_capacity_gib": 96,
    },
    {
        "topology": "4h4f",
        "hbm_stacks": 4,
        "hbf_stacks": 4,
        "physical_hbm_capacity_gib": 192,
    },
    {
        "topology": "6h2f",
        "hbm_stacks": 6,
        "hbf_stacks": 2,
        "physical_hbm_capacity_gib": 288,
    },
)
MODEL_DESCRIPTORS = (
    REPOSITORY_ROOT
    / "configs/workloads/frontier/llama31-70b-w8a16-kv-bf16.json",
    REPOSITORY_ROOT
    / "configs/workloads/frontier/llama31-70b-bf16-kv-bf16.json",
)
EXPECTED_PRECISION_PROFILES = (
    PRIMARY_PRECISION_PROFILE,
    "bf16-kv-bf16",
)
REQUIRED_CSV_FIELDS = {
    "arrived_at",
    "num_prefill_tokens",
    "num_decode_tokens",
    "source_chat_id",
}


class CapacityGridError(ValueError):
    """The canonical capacity grid cannot be derived without ambiguity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_text(value: Any, path: str, *, minimum: int = 0) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as error:
        raise CapacityGridError(f"{path} must be an integer") from error
    if str(result) != str(value).strip() or result < minimum:
        raise CapacityGridError(f"{path} must be an integer >= {minimum}")
    return result


def _artifact_path(
    *,
    suite_dir: Path,
    artifact: Any,
    path: str,
) -> Path:
    if not isinstance(artifact, dict):
        raise CapacityGridError(f"{path} must be an object")
    relative = artifact.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
    ):
        raise CapacityGridError(f"{path}.path must be a relative path")
    resolved = (suite_dir / relative).resolve()
    try:
        resolved.relative_to(suite_dir)
    except ValueError as error:
        raise CapacityGridError(f"{path}.path escapes the suite directory") from error
    if not resolved.is_file():
        raise CapacityGridError(f"{path} is missing: {resolved}")
    expected_bytes = artifact.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or resolved.stat().st_size != expected_bytes
    ):
        raise CapacityGridError(f"{path}.bytes does not match {resolved}")
    expected_sha256 = artifact.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or _sha256_file(resolved) != expected_sha256
    ):
        raise CapacityGridError(f"{path}.sha256 does not match {resolved}")
    return resolved


def _window_requirement(
    *,
    receipt: dict[str, Any],
    suite_dir: Path,
    window: str,
) -> dict[str, Any]:
    windows = receipt.get("windows")
    if not isinstance(windows, dict):
        raise CapacityGridError("request-suite windows must be an object")
    window_receipt = windows.get(window)
    if not isinstance(window_receipt, dict):
        raise CapacityGridError(f"request-suite window {window} must be an object")
    request_csv = _artifact_path(
        suite_dir=suite_dir,
        artifact=window_receipt.get("request_csv"),
        path=f"request-suite window {window}.request_csv",
    )
    statistics = window_receipt.get("statistics")
    if not isinstance(statistics, dict):
        raise CapacityGridError(
            f"request-suite window {window}.statistics must be an object"
        )

    rows = 0
    total_prefill = 0
    total_decode = 0
    max_prefill = 0
    max_decode = 0
    max_total = 0
    maximum: dict[str, Any] | None = None
    try:
        handle = request_csv.open(newline="", encoding="utf-8")
    except OSError as error:
        raise CapacityGridError(f"cannot read {request_csv}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_CSV_FIELDS.issubset(
            reader.fieldnames
        ):
            raise CapacityGridError(
                f"{window} request CSV is missing required fields"
            )
        for row_index, row in enumerate(reader):
            prefill = _integer_text(
                row.get("num_prefill_tokens"),
                f"{window} row {row_index}.num_prefill_tokens",
                minimum=1,
            )
            decode = _integer_text(
                row.get("num_decode_tokens"),
                f"{window} row {row_index}.num_decode_tokens",
                minimum=1,
            )
            source_chat_id = _integer_text(
                row.get("source_chat_id"),
                f"{window} row {row_index}.source_chat_id",
            )
            total = prefill + decode
            kv_tokens = total - 1
            required_blocks = (kv_tokens + 15) // 16
            candidate = {
                "row_index": row_index,
                "source_chat_id": source_chat_id,
                "prefill_tokens": prefill,
                "decode_tokens": decode,
                "total_tokens": total,
                "kv_tokens": kv_tokens,
                "required_blocks": required_blocks,
            }
            if (
                maximum is None
                or required_blocks > maximum["required_blocks"]
                or (
                    required_blocks == maximum["required_blocks"]
                    and total > maximum["total_tokens"]
                )
            ):
                maximum = candidate
            rows += 1
            total_prefill += prefill
            total_decode += decode
            max_prefill = max(max_prefill, prefill)
            max_decode = max(max_decode, decode)
            max_total = max(max_total, total)
    if rows != 256 or maximum is None:
        raise CapacityGridError(
            f"{window} must contain exactly 256 non-empty requests"
        )
    expected_statistics = {
        "records": rows,
        "prefill_tokens": total_prefill,
        "decode_tokens": total_decode,
        "total_tokens": total_prefill + total_decode,
        "max_prefill_tokens": max_prefill,
        "max_decode_tokens": max_decode,
        "max_total_tokens": max_total,
    }
    for name, expected in expected_statistics.items():
        if statistics.get(name) != expected:
            raise CapacityGridError(
                f"{window} statistics.{name}: expected {expected}, "
                f"got {statistics.get(name)!r}"
            )
    return {
        "request_csv": {
            "path": str(request_csv),
            "bytes": request_csv.stat().st_size,
            "sha256": _sha256_file(request_csv),
        },
        **expected_statistics,
        "maximum_full_request_reservation": maximum,
    }


def _model_inputs() -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for descriptor_path in MODEL_DESCRIPTORS:
        try:
            inputs = derive_dense_model_capacity_inputs(descriptor_path)
        except ExportError as error:
            raise CapacityGridError(
                f"model descriptor is invalid: {descriptor_path}: {error}"
            ) from error
        if inputs["model_name"] != MODEL_NAME:
            raise CapacityGridError(
                f"descriptor {descriptor_path} is not {MODEL_NAME}"
            )
        profile = str(inputs["precision_profile"])
        if profile in profiles:
            raise CapacityGridError(f"duplicate precision profile {profile}")
        profiles[profile] = inputs
    if tuple(profiles) != EXPECTED_PRECISION_PROFILES:
        raise CapacityGridError(
            "model descriptors must contain exactly the primary W8 profile "
            "followed by the BF16 sensitivity"
        )
    return profiles


def build_capacity_grid(
    *,
    request_suite_verification: Path,
    hbfsim_revision: str,
) -> dict[str, Any]:
    """Build the canonical preflight without running Frontier."""
    if (
        len(hbfsim_revision) != 40
        or hbfsim_revision != hbfsim_revision.lower()
        or any(
            character not in "0123456789abcdef"
            for character in hbfsim_revision
        )
    ):
        raise CapacityGridError("hbfsim_revision must be a full lowercase SHA")
    try:
        receipt, suite_dir = validate_request_suite_verification(
            request_suite_verification
        )
    except StructuralSuiteError as error:
        raise CapacityGridError(
            f"request-suite verification is invalid: {error}"
        ) from error
    suite_dir = suite_dir.resolve()
    window_requirements = {
        window: _window_requirement(
            receipt=receipt,
            suite_dir=suite_dir,
            window=window,
        )
        for window in WINDOWS
    }
    profiles = _model_inputs()

    points: list[dict[str, Any]] = []
    for profile_id, model in profiles.items():
        for topology in TOPOLOGIES:
            capacity_bytes = (
                int(topology["physical_hbm_capacity_gib"]) * 1024**3
            )
            for pressure in PRESSURES:
                identity = (
                    f"{profile_id}.{topology['topology']}."
                    f"p{str(pressure).replace('.', '_')}"
                )
                point: dict[str, Any] = {
                    "point_id": identity,
                    "precision_profile": profile_id,
                    **topology,
                    "physical_hbm_capacity_bytes": capacity_bytes,
                    "target_pressure": str(pressure),
                    "runtime_overhead_bytes": RUNTIME_OVERHEAD_BYTES,
                    "runtime_overhead_source": (
                        "configured_unprofiled_sensitivity"
                    ),
                    "hardware_capacity_calibrated": False,
                }
                try:
                    plan = build_hybrid_residency_plan(
                        physical_hbm_capacity_bytes=capacity_bytes,
                        target_pressure=pressure,
                        immutable_weight_backing_bytes=int(
                            model["immutable_weight_backing_bytes"]
                        ),
                        runtime_overhead_bytes=RUNTIME_OVERHEAD_BYTES,
                        active_weight_buffer_bytes_per_slot=int(
                            model["active_weight_buffer_bytes_per_slot"]
                        ),
                        kv_block_size_tokens=int(
                            model["kv_block_size_tokens"]
                        ),
                        kv_page_bytes_per_layer=int(
                            model["kv_page_bytes_per_layer"]
                        ),
                        num_layers=int(model["num_layers"]),
                    )
                except HybridResidencyError as error:
                    point.update(
                        {
                            "static_preflight_status": "planner_infeasible",
                            "planner_feasible": False,
                            "single_request_admission_feasible": None,
                            "execution_required": False,
                            "infeasibility": {
                                "kind": "planner",
                                "reason": str(error),
                            },
                            "residency_plan": None,
                        }
                    )
                    points.append(point)
                    continue

                blocking_windows = [
                    window
                    for window, requirement in window_requirements.items()
                    if requirement["maximum_full_request_reservation"][
                        "required_blocks"
                    ]
                    > plan["num_logical_kv_blocks"]
                ]
                if blocking_windows:
                    status = "request_infeasible"
                    infeasibility: dict[str, Any] | None = {
                        "kind": "single_request_admission",
                        "blocking_windows": blocking_windows,
                        "capacity_blocks": plan["num_logical_kv_blocks"],
                        "required_blocks": max(
                            window_requirements[window][
                                "maximum_full_request_reservation"
                            ]["required_blocks"]
                            for window in blocking_windows
                        ),
                    }
                else:
                    status = "execution_candidate"
                    infeasibility = None
                point.update(
                    {
                        "static_preflight_status": status,
                        "planner_feasible": True,
                        "single_request_admission_feasible": (
                            not blocking_windows
                        ),
                        "execution_required": not blocking_windows,
                        "infeasibility": infeasibility,
                        "residency_plan": plan,
                    }
                )
                points.append(point)

    expected_points = (
        len(EXPECTED_PRECISION_PROFILES)
        * len(TOPOLOGIES)
        * len(PRESSURES)
    )
    if len(points) != expected_points:
        raise CapacityGridError("capacity grid point census is incomplete")
    status_counts = {
        status: sum(
            point["static_preflight_status"] == status for point in points
        )
        for status in (
            "planner_infeasible",
            "request_infeasible",
            "execution_candidate",
        )
    }
    verification_path = request_suite_verification.resolve()
    return {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "result": "pass",
        "suite_id": SUITE_ID,
        "model": MODEL_NAME,
        "source": {
            "hbfsim_revision": hbfsim_revision,
            "request_suite_verification": {
                "path": str(verification_path),
                "bytes": verification_path.stat().st_size,
                "sha256": _sha256_file(verification_path),
            },
            "model_descriptors": {
                profile: model["model_descriptor"]
                for profile, model in profiles.items()
            },
        },
        "contract": {
            "pressure_basis": (
                "unique_resident_footprint_bytes/"
                "physical_hbm_capacity_bytes"
            ),
            "pressures": [str(value) for value in PRESSURES],
            "topologies": list(TOPOLOGIES),
            "precision_profiles": list(EXPECTED_PRECISION_PROFILES),
            "runtime_overhead_bytes": RUNTIME_OVERHEAD_BYTES,
            "runtime_overhead_source": (
                "configured_unprofiled_sensitivity"
            ),
            "hardware_capacity_calibrated": False,
            "no_preemption_admission_policy": (
                "full_request_kv_reservation_v1"
            ),
        },
        "model_capacity_inputs": profiles,
        "window_requirements": window_requirements,
        "points": points,
        "census": {
            "points": len(points),
            **status_counts,
        },
        "eligibility": {
            "static_grid_preflight_valid": True,
            "paper_result_eligible": False,
            "paper_blockers": [
                "execution-candidate Frontier replays not attached",
                "equal-workload HBM/HBF/CXL/NVMe baselines not attached",
                "canonical WAF audit not attached",
                "runtime overhead is configured and unprofiled",
                "Frontier compute/communication timing is not calibrated",
            ],
        },
    }


def _git(*arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise CapacityGridError(
            f"git {' '.join(arguments)} failed: {process.stderr.strip()}"
        )
    return process.stdout


def _require_clean_hbfsim_commit() -> str:
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        raise CapacityGridError(
            "HBFSim tracked files must be clean before publishing a "
            "capacity-grid preflight:\n"
            + status.rstrip()
        )
    return _git("rev-parse", "HEAD").strip()


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
    if path.exists():
        raise CapacityGridError(f"output already exists: {path}")
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the canonical 70B capacity-pressure grid without "
            "claiming that execution candidates have run."
        )
    )
    parser.add_argument(
        "--request-suite-verification",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    revision = _require_clean_hbfsim_commit()
    payload = build_capacity_grid(
        request_suite_verification=arguments.request_suite_verification,
        hbfsim_revision=revision,
    )
    output = arguments.output.resolve()
    _write_atomic_json(output, payload)
    print(
        "capacity-grid preflight: "
        f"{payload['census']['execution_candidate']} execution candidates, "
        f"{payload['census']['request_infeasible']} request-infeasible, "
        f"{payload['census']['planner_infeasible']} planner-infeasible"
    )
    print(f"receipt={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
