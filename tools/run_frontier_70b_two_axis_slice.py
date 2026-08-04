#!/usr/bin/env python3
"""Run the fixed-population/mid-topology 70B two-axis memory slice.

This runner consumes three memory traces exported from complete Frontier
replays of the same verified 256-request Qwen burst.  The p1.0 trace is reused
verbatim across 6H2F, 4H4F, and 2H6F for the HBM/HBF mix-ratio axis.  The
p0.75, p1.0, and p1.25 traces are compared only on 4H4F for the capacity
pressure axis.  There are exactly five non-duplicate cells and four baselines
per cell; no pure all-HBF case exists in this study.

Each invocation runs one predeclared temporal-quantile stratum.  A stratum is
partial memory-transaction evidence even though its source Frontier replay is
complete, so this tool never promotes it to a full-matrix paper result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from run_frontier_memory_baselines import (
    BASELINE_SPECS,
    BaselineSetError,
    SCHEMA as BASELINE_SCHEMA,
    run_baselines,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STUDY = (
    REPOSITORY_ROOT
    / "configs/studies/frontier-70b-two-axis-burst.json"
)
HARDWARE_CONFIGS = {
    "6h2f": REPOSITORY_ROOT / "configs/scenario_compare/usecase-6h2f.cfg",
    "4h4f": REPOSITORY_ROOT / "configs/scenario_compare/usecase-4h4f.cfg",
    "2h6f": REPOSITORY_ROOT / "configs/scenario_compare/usecase-2h6f.cfg",
}
EXPECTED_HBM_CAPACITY = {
    "6h2f": 6 * 48 * 1024**3,
    "4h4f": 4 * 48 * 1024**3,
    "2h6f": 2 * 48 * 1024**3,
}
EXPECTED_HBF_STACKS = {
    "6h2f": 2,
    "4h4f": 4,
    "2h6f": 6,
}
EXPECTED_POPULATIONS = {
    "p0_75": "0.75",
    "p1_0": "1.0",
    "p1_25": "1.25",
}
EXPECTED_CELLS = (
    {
        "cell_id": "mix_ratio.6h2f.common_p1_0",
        "axis": "mix_ratio",
        "topology": "6h2f",
        "population_source": "p1_0",
    },
    {
        "cell_id": "mix_ratio.4h4f.common_p1_0",
        "axis": "mix_ratio_and_pressure",
        "topology": "4h4f",
        "population_source": "p1_0",
    },
    {
        "cell_id": "mix_ratio.2h6f.common_p1_0",
        "axis": "mix_ratio",
        "topology": "2h6f",
        "population_source": "p1_0",
    },
    {
        "cell_id": "pressure.4h4f.p0_75",
        "axis": "pressure",
        "topology": "4h4f",
        "population_source": "p0_75",
    },
    {
        "cell_id": "pressure.4h4f.p1_25",
        "axis": "pressure",
        "topology": "4h4f",
        "population_source": "p1_25",
    },
)
EXPECTED_BASELINES = tuple(name for name, _ in BASELINE_SPECS)
MANIFEST_SCHEMA = {
    "name": "hbfsim.frontier_memory_trace",
    "version": 4,
}
STUDY_SCHEMA = {
    "name": "hbfsim.frontier_70b_two_axis_study",
    "version": 1,
}
INPUT_SCHEMA = {
    "name": "hbfsim.frontier_70b_two_axis_inputs",
    "version": 1,
}
RECEIPT_SCHEMA = {
    "name": "hbfsim.frontier_70b_two_axis_slice",
    "version": 1,
}


class TwoAxisSliceError(ValueError):
    """The two-axis slice is incomplete or not byte-comparable."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise TwoAxisSliceError(
            f"cannot read {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise TwoAxisSliceError(f"{description} must be a JSON object")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TwoAxisSliceError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TwoAxisSliceError(f"{name} must be an integer")
    if value < minimum:
        raise TwoAxisSliceError(f"{name} must be >= {minimum}")
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
        raise TwoAxisSliceError(f"{description} is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _verify_snapshot(snapshot: dict[str, Any], description: str) -> None:
    actual = _snapshot(Path(str(snapshot.get("path"))), description)
    if actual != snapshot:
        raise TwoAxisSliceError(f"{description} changed during the slice")


def _artifact_path(
    value: Any,
    *,
    owner: Path,
    description: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise TwoAxisSliceError(f"{description}.path must be non-empty")
    path = Path(value)
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _verified_artifact_snapshot(
    record: Any,
    *,
    owner: Path,
    description: str,
) -> dict[str, Any]:
    artifact = _mapping(record, description)
    path = _artifact_path(
        artifact.get("path"),
        owner=owner,
        description=description,
    )
    snapshot = _snapshot(path, description)
    if (
        artifact.get("bytes") != snapshot["bytes"]
        or artifact.get("sha256") != snapshot["sha256"]
    ):
        raise TwoAxisSliceError(f"{description} digest drifted")
    return snapshot


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
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


def _config_integer(path: Path, key: str) -> int:
    values: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if name == key:
            try:
                values.append(int(value, 10))
            except ValueError as error:
                raise TwoAxisSliceError(
                    f"{path}: {key} must be decimal"
                ) from error
    if len(values) != 1 or values[0] <= 0:
        raise TwoAxisSliceError(
            f"{path}: expected one positive {key}"
        )
    return values[0]


def _source_revision(binary_version: str) -> str:
    status = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "status",
            "--short",
            "--untracked-files=no",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise TwoAxisSliceError(
            "tracked source must be clean before a two-axis evidence run"
        )
    revision = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    source_revision = revision.stdout.strip()
    if (
        revision.returncode != 0
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
    ):
        raise TwoAxisSliceError("cannot resolve the HBFSim source revision")
    version_match = re.search(
        r"\(git ([0-9a-f]{40})\)$",
        binary_version,
    )
    if (
        version_match is None
        or version_match.group(1) != source_revision
        or "dirty-at-configure" in binary_version
    ):
        raise TwoAxisSliceError(
            "scenario_compare was not built from the clean current HEAD"
        )
    return source_revision


def _validate_study(path: Path) -> dict[str, Any]:
    study = _load_object(path, "two-axis study")
    if study.get("schema") != STUDY_SCHEMA:
        raise TwoAxisSliceError("two-axis study schema drifted")
    workload = _mapping(study.get("workload"), "study.workload")
    if workload != {
        "model": "meta-llama/Llama-3.1-70B",
        "precision_profile": "w8a16-kv-bf16",
        "request_suite": "qwen-bailian",
        "window": "burst",
        "requests": 256,
        "frontier_full_replay_required": True,
    }:
        raise TwoAxisSliceError("two-axis workload contract drifted")
    if study.get("population_sources") != {
        name: {
            "planner_topology": "4h4f",
            "target_pressure": pressure,
        }
        for name, pressure in EXPECTED_POPULATIONS.items()
    }:
        raise TwoAxisSliceError("two-axis population sources drifted")
    if study.get("cells") != list(EXPECTED_CELLS):
        raise TwoAxisSliceError("two-axis five-cell contract drifted")
    if study.get("baselines") != list(EXPECTED_BASELINES):
        raise TwoAxisSliceError("two-axis four-baseline contract drifted")
    baseline_semantics = _mapping(
        study.get("baseline_semantics"),
        "study.baseline_semantics",
    )
    if (
        baseline_semantics.get("hbm_hbf")
        != "HBM+HBF_hybrid_layer_streaming"
        or baseline_semantics.get("pure_all_hbf_included") is not False
    ):
        raise TwoAxisSliceError(
            "hbm_hbf must be hybrid and pure all-HBF must be excluded"
        )
    if study.get("execution") != {
        "default_cell_parallelism": 1,
        "maximum_cell_parallelism": len(EXPECTED_CELLS),
        "within_cell": "two_cases_concurrent_four_baselines_atomic",
    }:
        raise TwoAxisSliceError("two-axis execution contract drifted")
    sampling = _mapping(
        study.get("memory_sampling"),
        "study.memory_sampling",
    )
    if (
        sampling.get("scheme")
        != "frontier_batch_temporal_quantiles_v1"
        or sampling.get("quantiles") != ["0", "0.5", "1"]
        or sampling.get("execution_priority") != ["0.5", "0", "1"]
        or sampling.get("batches_per_quantile") != 1
        or sampling.get("aggregate_across_quantiles") is not False
    ):
        raise TwoAxisSliceError("two-axis temporal sampling contract drifted")
    return study


def _expected_batch_row(total_batches: int, quantile: Decimal) -> int:
    return int(
        (quantile * Decimal(total_batches - 1)).to_integral_value(
            rounding="ROUND_FLOOR"
        )
    )


def _manifest_input(
    *,
    population: str,
    manifest_path: Path,
    object_map_path: Path,
    quantile: Decimal,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    object_map_path = object_map_path.resolve()
    manifest = _load_object(manifest_path, f"{population} manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise TwoAxisSliceError(f"{population} manifest schema drifted")
    model = _mapping(manifest.get("model"), f"{population}.model")
    if (
        model.get("name") != "llama31_70b"
        or model.get("profile_id") != "w8a16-kv-bf16"
    ):
        raise TwoAxisSliceError(
            f"{population} is not Llama 3.1 70B W8A16/KV-BF16"
        )
    semantics = _mapping(
        manifest.get("semantics"),
        f"{population}.semantics",
    )
    if (
        semantics.get("eligible_claim_scope")
        != "memory_system_service_only"
        or semantics.get("ttft_tpot_slo_claims_eligible") is not False
        or semantics.get("time_based_throughput_claims_eligible") is not False
    ):
        raise TwoAxisSliceError(
            f"{population} upgraded dummy Frontier timing claims"
        )
    selection = _mapping(
        manifest.get("selection"),
        f"{population}.selection",
    )
    total_batches = _integer(
        selection.get("total_frontier_batches"),
        f"{population} total Frontier batches",
        minimum=1,
    )
    expected_row = _expected_batch_row(total_batches, quantile)
    batch_ids = selection.get("batch_ids")
    if (
        selection.get("batch_start") != expected_row
        or selection.get("batches") != 1
        or not isinstance(batch_ids, list)
        or len(batch_ids) != 1
        or selection.get("is_full_replay") is not False
        or selection.get("scaling_applied") is not False
        or selection.get("byte_sampling_applied") is not False
    ):
        raise TwoAxisSliceError(
            f"{population} is not the predeclared {quantile} "
            "one-batch temporal stratum"
        )
    capacity = _mapping(
        manifest.get("capacity_accounting"),
        f"{population}.capacity_accounting",
    )
    plan = _mapping(
        capacity.get("residency_plan"),
        f"{population}.residency_plan",
    )
    if (
        plan.get("physical_hbm_capacity_bytes")
        != EXPECTED_HBM_CAPACITY["4h4f"]
        or str(plan.get("target_pressure"))
        != EXPECTED_POPULATIONS[population]
    ):
        raise TwoAxisSliceError(
            f"{population} was not planned at its fixed 4H4F pressure"
        )
    outputs = _mapping(manifest.get("outputs"), f"{population}.outputs")
    trace_record = _mapping(
        outputs.get("trace"),
        f"{population}.outputs.trace",
    )
    trace_path = _artifact_path(
        trace_record.get("path"),
        owner=manifest_path,
        description=f"{population}.outputs.trace",
    )
    trace_snapshot = _snapshot(trace_path, f"{population} trace")
    if (
        trace_record.get("bytes") != trace_snapshot["bytes"]
        or trace_record.get("sha256") != trace_snapshot["sha256"]
    ):
        raise TwoAxisSliceError(f"{population} trace digest drifted")
    object_record = _mapping(
        outputs.get("object_map"),
        f"{population}.outputs.object_map",
    )
    object_snapshot = _snapshot(
        object_map_path,
        f"{population} object map",
    )
    if (
        object_record.get("bytes") != object_snapshot["bytes"]
        or object_record.get("sha256") != object_snapshot["sha256"]
    ):
        raise TwoAxisSliceError(f"{population} object-map digest drifted")
    sources = _mapping(
        manifest.get("sources"),
        f"{population}.sources",
    )
    transitive_artifacts = {
        "audit": _verified_artifact_snapshot(
            sources.get("audit"),
            owner=manifest_path,
            description=f"{population}.sources.audit",
        ),
        "model_descriptor": _verified_artifact_snapshot(
            sources.get("model_descriptor"),
            owner=manifest_path,
            description=f"{population}.sources.model_descriptor",
        ),
    }
    frontier_artifacts = _mapping(
        sources.get("frontier_artifacts"),
        f"{population}.sources.frontier_artifacts",
    )
    if not frontier_artifacts:
        raise TwoAxisSliceError(
            f"{population} has no transitive Frontier artifacts"
        )
    for name, record in sorted(frontier_artifacts.items()):
        transitive_artifacts[f"frontier.{name}"] = (
            _verified_artifact_snapshot(
                record,
                owner=manifest_path,
                description=f"{population}.frontier_artifacts.{name}",
            )
        )
    try:
        request_csv = transitive_artifacts["frontier.request_csv"]
        request_verification = transitive_artifacts[
            "frontier.request_suite_verification"
        ]
    except KeyError as error:
        raise TwoAxisSliceError(
            f"{population} is missing verified request-stream artifacts"
        ) from error
    return {
        "manifest": _snapshot(manifest_path, f"{population} manifest"),
        "object_map": object_snapshot,
        "trace": trace_snapshot,
        "transitive_artifacts": transitive_artifacts,
        "selection": {
            "quantile": str(quantile),
            "total_frontier_batches": total_batches,
            "batch_row": expected_row,
            "batch_ids": batch_ids,
        },
        "population": {
            "unique_resident_footprint_bytes": plan.get(
                "unique_resident_footprint_bytes"
            ),
            "logical_kv_blocks": plan.get("num_logical_kv_blocks"),
            "immutable_weight_backing_bytes": plan.get(
                "immutable_weight_backing_bytes"
            ),
            "runtime_overhead_bytes": plan.get("runtime_overhead_bytes"),
            "block_table_bytes": plan.get("block_table_bytes"),
        },
        "request_identity": {
            "request_csv_bytes": request_csv["bytes"],
            "request_csv_sha256": request_csv["sha256"],
            "request_suite_verification_bytes": request_verification["bytes"],
            "request_suite_verification_sha256": request_verification[
                "sha256"
            ],
        },
    }


def _validate_baseline_receipt(
    *,
    receipt: dict[str, Any],
    population: dict[str, Any],
    topology: str,
    scenario_compare: dict[str, Any],
) -> None:
    if (
        receipt.get("schema") != BASELINE_SCHEMA
        or receipt.get("result") != "pass"
        or set(_mapping(receipt.get("baselines"), "baselines"))
        != set(EXPECTED_BASELINES)
    ):
        raise TwoAxisSliceError("baseline-set receipt is incomplete")
    source = _mapping(receipt.get("source"), "baseline source")
    if source.get("scenario_compare") != scenario_compare:
        raise TwoAxisSliceError(
            f"{topology} baseline was produced by a different simulator binary"
        )
    for name in ("manifest", "object_map", "trace"):
        if source.get(name) != population[name]:
            raise TwoAxisSliceError(
                f"{topology} baseline consumed a different {name}"
            )
    contract = _mapping(receipt.get("contract"), "baseline contract")
    hardware = _mapping(
        contract.get("hardware_topology"),
        "baseline hardware topology",
    )
    if hardware != {
        "hbm_stacks": int(topology[0]),
        "hbm_capacity_bytes": EXPECTED_HBM_CAPACITY[topology],
        "hbf_stacks": EXPECTED_HBF_STACKS[topology],
    }:
        raise TwoAxisSliceError(f"{topology} hardware contract drifted")
    residency = _mapping(contract.get("residency"), "baseline residency")
    if (
        residency.get("placement_policy")
        != "capacity_aware_static_weight_prefix_v2"
        or residency.get("population_preserved_across_placement") is not True
        or residency.get("physical_hbm_capacity_bytes")
        != EXPECTED_HBM_CAPACITY[topology]
    ):
        raise TwoAxisSliceError(
            f"{topology} did not use capacity-aware fixed-population placement"
        )


def _resume_or_run_cell(
    *,
    scenario_compare: Path,
    output_dir: Path,
    cell: dict[str, str],
    population: dict[str, Any],
    credit_limit: int,
    timeout_seconds: float | None,
    scenario_snapshot: dict[str, Any],
) -> dict[str, Any]:
    cell_dir = output_dir / "cells" / cell["cell_id"]
    baseline_dir = cell_dir / "baselines"
    receipt_path = baseline_dir / "baseline-set.receipt.json"
    if receipt_path.is_file():
        receipt = _load_object(receipt_path, f"{cell['cell_id']} receipt")
    else:
        if cell_dir.exists():
            raise TwoAxisSliceError(
                f"incomplete cell directory must be inspected: {cell_dir}"
            )
        cell_dir.mkdir(parents=True)
        try:
            receipt = run_baselines(
                scenario_compare=scenario_compare,
                hardware_config=HARDWARE_CONFIGS[cell["topology"]],
                manifest_path=Path(population["manifest"]["path"]),
                object_map_path=Path(population["object_map"]["path"]),
                output_dir=baseline_dir,
                credit_limit=credit_limit,
                case_parallelism=2,
                timeout_seconds=timeout_seconds,
            )
        except BaselineSetError as error:
            raise TwoAxisSliceError(
                f"{cell['cell_id']} baseline set failed: {error}"
            ) from error
    _validate_baseline_receipt(
        receipt=receipt,
        population=population,
        topology=cell["topology"],
        scenario_compare=scenario_snapshot,
    )
    return receipt


def _result_rows(
    *,
    cell: dict[str, str],
    population_name: str,
    population: dict[str, Any],
    receipt: dict[str, Any],
) -> list[dict[str, Any]]:
    contract = receipt["contract"]
    residency = contract["residency"]
    waf = contract["canonical_hbf_waf"]
    rows = []
    for baseline in EXPECTED_BASELINES:
        baseline_record = receipt["baselines"][baseline]
        metrics = baseline_record["memory_service_metrics"]
        waf_fields = (
            {
                "logical_write_bytes": waf["logical_write_bytes"],
                "physical_write_bytes": waf["physical_write_bytes"],
                "waf": waf["waf"],
                "physical_write_over_hbf_capacity": (
                    waf[
                        "physical_write_bytes_over_usable_hbf_capacity"
                    ]
                ),
            }
            if baseline == "hbm_hbf"
            else {
                "logical_write_bytes": None,
                "physical_write_bytes": None,
                "waf": None,
                "physical_write_over_hbf_capacity": None,
            }
        )
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "axis": cell["axis"],
                "topology": cell["topology"],
                "population_source": population_name,
                "quantile": population["selection"]["quantile"],
                "frontier_total_batches": population["selection"][
                    "total_frontier_batches"
                ],
                "frontier_batch_row": population["selection"]["batch_row"],
                "frontier_batch_ids": json.dumps(
                    population["selection"]["batch_ids"],
                    separators=(",", ":"),
                ),
                "trace_sha256": population["trace"]["sha256"],
                "trace_ops": contract["traffic"]["ops"],
                "trace_logical_bytes": contract["traffic"]["logical_bytes"],
                "unique_resident_footprint_bytes": residency[
                    "unique_resident_footprint_bytes"
                ],
                "physical_hbm_capacity_bytes": residency[
                    "physical_hbm_capacity_bytes"
                ],
                "observed_capacity_pressure": residency[
                    "capacity_pressure"
                ],
                "logical_kv_blocks": residency["logical_kv_blocks"],
                "hot_kv_blocks": residency["hot_kv_blocks"],
                "cold_kv_blocks": residency["cold_kv_blocks"],
                "baseline": baseline,
                "scenario": baseline_record["scenario"],
                "memory_system_logical_throughput_GBps": metrics[
                    "memory_system_logical_throughput_GBps"
                ],
                "makespan_ns": metrics["makespan_ns"],
                "offered_average_ns": metrics["offered_average_ns"],
                "offered_p50_ns": metrics["offered_p50_ns"],
                "offered_p95_ns": metrics["offered_p95_ns"],
                "offered_max_ns": metrics["offered_max_ns"],
                "source_average_ns": metrics["source_average_ns"],
                "source_p50_ns": metrics["source_p50_ns"],
                "source_p95_ns": metrics["source_p95_ns"],
                "source_max_ns": metrics["source_max_ns"],
                "service_average_ns": metrics["service_average_ns"],
                "service_p50_ns": metrics["service_p50_ns"],
                "service_p95_ns": metrics["service_p95_ns"],
                "service_max_ns": metrics["service_max_ns"],
                "backing_admission_max_wait_ns": metrics[
                    "backing_admission_max_wait_ns"
                ],
                "exposed_prefetch_ns": metrics["exposed_prefetch_ns"],
                "hidden_prefetch_ns": metrics["hidden_prefetch_ns"],
                "backing_read_bytes": metrics["backing_read_bytes"],
                "backing_write_bytes": metrics["backing_write_bytes"],
                **waf_fields,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def run_slice(
    *,
    scenario_compare: Path,
    study_path: Path,
    population_paths: dict[str, tuple[Path, Path]],
    output_dir: Path,
    quantile: Decimal,
    credit_limit: int,
    cell_parallelism: int,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    if quantile not in {Decimal("0"), Decimal("0.5"), Decimal("1")}:
        raise TwoAxisSliceError("quantile must be 0, 0.5, or 1")
    if credit_limit <= 0:
        raise TwoAxisSliceError("credit_limit must be positive")
    if not 1 <= cell_parallelism <= len(EXPECTED_CELLS):
        raise TwoAxisSliceError(
            f"cell_parallelism must be between 1 and {len(EXPECTED_CELLS)}"
        )
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise TwoAxisSliceError("timeout_seconds must be positive or None")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    study = _validate_study(study_path)
    scenario_snapshot = _snapshot(
        scenario_compare,
        "scenario_compare binary",
    )
    version = subprocess.run(
        [scenario_snapshot["path"], "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if version.returncode != 0 or "git " not in version.stdout:
        raise TwoAxisSliceError("scenario_compare version is not auditable")
    source_revision = _source_revision(version.stdout.strip())
    hardware_snapshots = {
        topology: _snapshot(path, f"{topology} hardware config")
        for topology, path in HARDWARE_CONFIGS.items()
    }
    for topology, path in HARDWARE_CONFIGS.items():
        if (
            _config_integer(path, "hbm-capacity-bytes")
            != EXPECTED_HBM_CAPACITY[topology]
            or _config_integer(path, "hbf-stacks")
            != EXPECTED_HBF_STACKS[topology]
        ):
            raise TwoAxisSliceError(f"{topology} hardware config drifted")
    populations = {
        name: _manifest_input(
            population=name,
            manifest_path=population_paths[name][0],
            object_map_path=population_paths[name][1],
            quantile=quantile,
        )
        for name in EXPECTED_POPULATIONS
    }
    request_identities = {
        json.dumps(
            population["request_identity"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for population in populations.values()
    }
    if len(request_identities) != 1:
        raise TwoAxisSliceError(
            "pressure populations do not share one verified request stream"
        )
    inputs = {
        "schema": INPUT_SCHEMA,
        "result": "pass",
        "quantile": str(quantile),
        "runner": _snapshot(Path(__file__), "two-axis runner"),
        "study": _snapshot(study_path, "two-axis study"),
        "scenario_compare": scenario_snapshot,
        "scenario_compare_version": version.stdout.strip(),
        "hbfsim_revision": source_revision,
        "hardware_configs": hardware_snapshots,
        "populations": populations,
        "request_identity": populations["p1_0"]["request_identity"],
        "credit_limit": credit_limit,
        "cell_parallelism": cell_parallelism,
        "timeout_seconds": timeout_seconds,
    }
    inputs_path = output_dir / "slice.inputs.json"
    if inputs_path.is_file():
        if _load_object(inputs_path, "existing slice inputs") != inputs:
            raise TwoAxisSliceError("existing slice inputs differ")
    else:
        if any(output_dir.iterdir()):
            raise TwoAxisSliceError(
                "non-empty output directory has no matching input receipt"
            )
        _write_json_atomic(inputs_path, inputs)

    cells_root = output_dir / "cells"
    cells_root.mkdir(exist_ok=True)
    cell_receipts: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cell_parallelism) as executor:
        futures = {
            cell["cell_id"]: executor.submit(
                _resume_or_run_cell,
                scenario_compare=Path(scenario_snapshot["path"]),
                output_dir=output_dir,
                cell=cell,
                population=populations[cell["population_source"]],
                credit_limit=credit_limit,
                timeout_seconds=timeout_seconds,
                scenario_snapshot=scenario_snapshot,
            )
            for cell in EXPECTED_CELLS
        }
        for cell in EXPECTED_CELLS:
            population_name = cell["population_source"]
            receipt = futures[cell["cell_id"]].result()
            cell_receipts[cell["cell_id"]] = receipt
            rows.extend(
                _result_rows(
                    cell=cell,
                    population_name=population_name,
                    population=populations[population_name],
                    receipt=receipt,
                )
            )

    _verify_snapshot(inputs["runner"], "two-axis runner")
    _verify_snapshot(inputs["study"], "two-axis study")
    _verify_snapshot(
        inputs["scenario_compare"],
        "scenario_compare binary",
    )
    for topology, snapshot in inputs["hardware_configs"].items():
        _verify_snapshot(snapshot, f"{topology} hardware config")
    for population_name, population in populations.items():
        for name in ("manifest", "object_map", "trace"):
            _verify_snapshot(
                population[name],
                f"{population_name} {name}",
            )
        for name, snapshot in population["transitive_artifacts"].items():
            _verify_snapshot(
                snapshot,
                f"{population_name} {name}",
            )
    if _source_revision(version.stdout.strip()) != source_revision:
        raise TwoAxisSliceError("HBFSim source revision changed during slice")

    mix_receipts = [
        cell_receipts[cell["cell_id"]]
        for cell in EXPECTED_CELLS[:3]
    ]
    mix_source_contract = {
        "manifest": mix_receipts[0]["source"]["manifest"],
        "object_map": mix_receipts[0]["source"]["object_map"],
        "trace": mix_receipts[0]["source"]["trace"],
        "traffic": mix_receipts[0]["contract"]["traffic"],
        "unique_resident_footprint_bytes": mix_receipts[0]["contract"][
            "residency"
        ]["unique_resident_footprint_bytes"],
        "logical_kv_blocks": mix_receipts[0]["contract"]["residency"][
            "logical_kv_blocks"
        ],
    }
    for receipt in mix_receipts[1:]:
        candidate = {
            "manifest": receipt["source"]["manifest"],
            "object_map": receipt["source"]["object_map"],
            "trace": receipt["source"]["trace"],
            "traffic": receipt["contract"]["traffic"],
            "unique_resident_footprint_bytes": receipt["contract"][
                "residency"
            ]["unique_resident_footprint_bytes"],
            "logical_kv_blocks": receipt["contract"]["residency"][
                "logical_kv_blocks"
            ],
        }
        if candidate != mix_source_contract:
            raise TwoAxisSliceError(
                "mix-ratio cells are not one byte-identical population"
            )

    table_path = output_dir / "results.csv"
    _write_csv(table_path, rows)
    receipt_path = output_dir / "slice.receipt.json"
    payload = {
        "schema": RECEIPT_SCHEMA,
        "result": "pass",
        "study_id": study["study_id"],
        "quantile": str(quantile),
        "claim_scope": study["claim_scope"],
        "source": {
            "inputs": _artifact(inputs_path, output_dir),
        },
        "contract": {
            "cell_count": len(EXPECTED_CELLS),
            "baseline_runs": len(rows),
            "cell_parallelism": cell_parallelism,
            "mix_ratio_same_logical_trace": True,
            "mix_ratio_same_address_population": True,
            "mix_ratio_capacity_pressure_is_observed": True,
            "pressure_axis_fixed_topology": "4h4f",
            "hbm_hbf_semantics": "HBM+HBF_hybrid_layer_streaming",
            "pure_all_hbf_included": False,
            "temporal_stratum_is_partial_trace": True,
            "aggregate_across_quantiles": False,
        },
        "cells": {
            cell["cell_id"]: {
                "axis": cell["axis"],
                "topology": cell["topology"],
                "population_source": cell["population_source"],
                "baseline_receipt": _artifact(
                    output_dir
                    / "cells"
                    / cell["cell_id"]
                    / "baselines"
                    / "baseline-set.receipt.json",
                    output_dir,
                ),
            }
            for cell in EXPECTED_CELLS
        },
        "table": _artifact(table_path, output_dir),
        "eligibility": {
            "five_cell_contract_complete": True,
            "twenty_baseline_runs_complete": True,
            "mix_ratio_byte_identity_verified": True,
            "full_frontier_replays_required_upstream": True,
            "memory_trace_is_temporal_stratum": True,
            "paper_result_eligible": False,
        },
    }
    _write_json_atomic(receipt_path, payload)
    return payload


def _decimal_quantile(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except Exception as error:
        raise argparse.ArgumentTypeError(
            "quantile must be 0, 0.5, or 1"
        ) from error
    if parsed not in {Decimal("0"), Decimal("0.5"), Decimal("1")}:
        raise argparse.ArgumentTypeError(
            "quantile must be 0, 0.5, or 1"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path, required=True)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    for name in EXPECTED_POPULATIONS:
        option = name.replace("_", "-")
        parser.add_argument(
            f"--{option}-manifest",
            dest=f"{name}_manifest",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--{option}-object-map",
            dest=f"{name}_object_map",
            type=Path,
            required=True,
        )
    parser.add_argument("--quantile", type=_decimal_quantile, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--credit-limit", type=int, default=64)
    parser.add_argument(
        "--cell-parallelism",
        type=int,
        default=1,
        help=(
            "number of cells to execute concurrently (default: 1; each cell "
            "already runs two baseline cases concurrently)"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=0,
        help="0 disables the per-baseline timeout",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    population_paths = {
        name: (
            getattr(arguments, f"{name}_manifest"),
            getattr(arguments, f"{name}_object_map"),
        )
        for name in EXPECTED_POPULATIONS
    }
    try:
        payload = run_slice(
            scenario_compare=arguments.scenario_compare,
            study_path=arguments.study,
            population_paths=population_paths,
            output_dir=arguments.output_dir,
            quantile=arguments.quantile,
            credit_limit=arguments.credit_limit,
            cell_parallelism=arguments.cell_parallelism,
            timeout_seconds=(
                None
                if arguments.timeout_seconds == 0
                else arguments.timeout_seconds
            ),
        )
    except TwoAxisSliceError as error:
        raise SystemExit(f"error: {error}") from error
    print(
        "Frontier 70B two-axis slice: PASS "
        f"quantile={payload['quantile']} "
        f"cells={payload['contract']['cell_count']} "
        f"baselines={payload['contract']['baseline_runs']}"
    )
    print(
        f"receipt={arguments.output_dir.resolve() / 'slice.receipt.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
