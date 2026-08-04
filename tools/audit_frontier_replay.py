#!/usr/bin/env python3
"""Audit a Qwen-Bailian -> Frontier replay before memory-object export.

The Frontier stage/batch and KV lifecycle ledgers are serving artifacts, not a
GPU memory trace.  This tool verifies their provenance, token accounting, and
complete KV allocator state machine before publishing the digest-bound input
to the memory-object exporter.  The current contract intentionally accepts
only one dense co-located TP=PP=DP=1 replica; other architectures need
different conservation rules and must fail closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, TextIO

from frontier_memory_contract import (
    ALLOCATOR_STATE_SCHEMA_VERSION,
    MEMORY_CONTRACT_SCHEMA_VERSION,
    RESIDENCY_PLAN_SCHEMA_VERSION,
    allocator_state_sha256,
)
from frontier_hybrid_residency import (
    HybridResidencyError,
    validate_hybrid_residency_plan,
)


AUDIT_SCHEMA_NAME = "hbfsim.frontier_replay_audit"
AUDIT_SCHEMA_VERSION = 7
ADAPTER_SCHEMA_NAME = "hbfsim.qwen_bailian_frontier_adapter"
ADAPTER_SCHEMA_VERSION = 2
REQUEST_SUITE_SCHEMA_NAME = "hbfsim.qwen_bailian_frontier_suite"
REQUEST_SUITE_SCHEMA_VERSION = 1
REQUEST_SUITE_VERIFICATION_SCHEMA_NAME = (
    "hbfsim.qwen_bailian_frontier_suite_verification"
)
REQUEST_SUITE_VERIFICATION_SCHEMA_VERSION = 1
REQUIRED_REQUEST_WINDOWS = {
    "steady": "robust_center_mad_v1",
    "burst": "minimum_arrival_span_v1",
    "long_context_decode_tail": "balanced_input_output_tail_v1",
}
ADAPTER_OUTPUT_COLUMNS = [
    "arrived_at",
    "num_prefill_tokens",
    "num_decode_tokens",
    "session_id",
    "block_hash_ids",
    "source_chat_id",
    "parent_chat_id",
    "turn",
    "request_type",
]
FRONTIER_REPOSITORY = "https://github.com/NetX-lab/Frontier"
DEFAULT_FRONTIER_INTEGRATION_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "integrations/frontier/manifest.json"
)
REQUIRED_REQUEST_COLUMNS = {
    "arrived_at",
    "num_prefill_tokens",
    "num_decode_tokens",
    "session_id",
    "block_hash_ids",
}
NO_PREEMPTION_ADMISSION_POLICY = "full_request_kv_reservation_v1"
MEMORY_PRECISION_PROFILE_FIELDS = {
    "profile_id",
    "matrix_weight_dtype",
    "matrix_weight_bytes",
    "non_matrix_weight_dtype",
    "non_matrix_weight_bytes",
    "kv_dtype",
    "kv_bytes",
    "activation_dtype",
    "activation_bytes",
    "quantization_scheme",
    "scale_dtype",
    "scale_bytes",
    "zero_point_bytes",
    "claim_scope",
}
SUPPORTED_DTYPE_BYTES = {
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


class AuditError(ValueError):
    """A replay artifact violates the supported audited contract."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as error:
        raise AuditError(f"cannot read {description}: {path}: {error}") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise AuditError(
            f"invalid JSON in {description}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise AuditError(f"{description} must be a JSON object: {path}")
    return value


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{path} must be an object")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AuditError(f"{path} must be an integer >= {minimum}")
    return value


def _csv_integer(value: str | None, path: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value) if value is not None else None
    except ValueError:
        parsed = None
    if parsed is None or parsed < minimum or str(parsed) != value:
        raise AuditError(f"{path} must be an integer >= {minimum}")
    return parsed


def _finite_number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise AuditError(f"{path} must be finite and >= {minimum}")
    return result


def _csv_number(value: str | None, path: str, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value) if value is not None else math.nan
    except ValueError:
        parsed = math.nan
    if not math.isfinite(parsed) or parsed < minimum:
        raise AuditError(f"{path} must be finite and >= {minimum}")
    return parsed


def _csv_hash_ids(value: str | None, path: str, *, expected: int) -> list[int]:
    if value is None or not value:
        raise AuditError(f"{path} must contain {expected} block hash IDs")
    fields = value.split("|")
    if len(fields) != expected:
        raise AuditError(
            f"{path} must contain {expected} block hash IDs, got {len(fields)}"
        )
    hashes = [
        _csv_integer(field, f"{path}[{index}]")
        for index, field in enumerate(fields)
    ]
    return hashes


def _expect_equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise AuditError(f"{path}: expected {expected!r}, got {actual!r}")


def _sha256_string(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AuditError(f"{path} must be 64 lowercase hexadecimal characters")
    return value


def _validate_memory_precision_profile(
    value: Any,
    *,
    profile_id: str,
    path: str,
) -> dict[str, Any]:
    profile = _mapping(value, path)
    if set(profile) != MEMORY_PRECISION_PROFILE_FIELDS:
        raise AuditError(
            f"{path} fields mismatch: "
            f"missing={sorted(MEMORY_PRECISION_PROFILE_FIELDS - set(profile))}, "
            f"extra={sorted(set(profile) - MEMORY_PRECISION_PROFILE_FIELDS)}"
        )
    _expect_equal(profile.get("profile_id"), profile_id, f"{path}.profile_id")
    for prefix in ("matrix_weight", "non_matrix_weight", "kv", "activation"):
        dtype = profile.get(f"{prefix}_dtype")
        if dtype not in SUPPORTED_DTYPE_BYTES:
            raise AuditError(
                f"{path}.{prefix}_dtype is unsupported: {dtype!r}"
            )
        _expect_equal(
            _integer(
                profile.get(f"{prefix}_bytes"),
                f"{path}.{prefix}_bytes",
                minimum=1,
            ),
            SUPPORTED_DTYPE_BYTES[dtype],
            f"{path}.{prefix} dtype/bytes",
        )
    zero_point_bytes = _integer(
        profile.get("zero_point_bytes"),
        f"{path}.zero_point_bytes",
    )
    _expect_equal(
        zero_point_bytes,
        0,
        f"{path}.zero_point_bytes (symmetric storage)",
    )
    scheme = profile.get("quantization_scheme")
    scale_dtype = profile.get("scale_dtype")
    scale_bytes = _integer(
        profile.get("scale_bytes"),
        f"{path}.scale_bytes",
    )
    if scheme == "none":
        _expect_equal(scale_dtype, None, f"{path}.scale_dtype")
        _expect_equal(scale_bytes, 0, f"{path}.scale_bytes")
    elif scheme == "symmetric_per_output_channel":
        if scale_dtype not in SUPPORTED_DTYPE_BYTES:
            raise AuditError(
                f"{path}.scale_dtype is unsupported: {scale_dtype!r}"
            )
        _expect_equal(
            scale_bytes,
            SUPPORTED_DTYPE_BYTES[scale_dtype],
            f"{path}.scale dtype/bytes",
        )
    else:
        raise AuditError(
            f"{path}.quantization_scheme must be 'none' or "
            "'symmetric_per_output_channel'"
        )
    _expect_equal(
        profile.get("claim_scope"),
        "memory_storage_and_traffic_only",
        f"{path}.claim_scope",
    )
    return dict(profile)


def _open_atomic_text(path: Path) -> tuple[TextIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    return handle, Path(handle.name)


def _read_requests(path: Path) -> list[dict[str, Any]]:
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise AuditError(f"cannot read request CSV: {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AuditError("request CSV has no header")
        missing = REQUIRED_REQUEST_COLUMNS - set(reader.fieldnames)
        if missing:
            raise AuditError(
                "request CSV is missing columns: " + ", ".join(sorted(missing))
            )
        requests: list[dict[str, Any]] = []
        previous_arrival = -1.0
        for ordinal, row in enumerate(reader):
            arrival = _csv_number(
                row.get("arrived_at"), f"request row {ordinal}.arrived_at"
            )
            if arrival < previous_arrival:
                raise AuditError("request CSV arrivals must be nondecreasing")
            previous_arrival = arrival
            prefill = _csv_integer(
                row.get("num_prefill_tokens"),
                f"request row {ordinal}.num_prefill_tokens",
                minimum=1,
            )
            decode = _csv_integer(
                row.get("num_decode_tokens"),
                f"request row {ordinal}.num_decode_tokens",
                minimum=1,
            )
            block_hash_ids = _csv_hash_ids(
                row.get("block_hash_ids"),
                f"request row {ordinal}.block_hash_ids",
                expected=(prefill + 15) // 16,
            )
            requests.append(
                {
                    "id": str(ordinal),
                    "arrival_s": arrival,
                    "prefill_tokens": prefill,
                    "decode_tokens": decode,
                    "total_tokens": prefill + decode,
                    "block_hash_ids": block_hash_ids,
                }
            )
    if not requests:
        raise AuditError("request CSV contains no requests")
    return requests


def _validate_adapter_manifest(
    manifest: dict[str, Any], request_csv: Path, requests: list[dict[str, Any]]
) -> dict[str, Any]:
    schema = _mapping(manifest.get("schema"), "adapter manifest.schema")
    _expect_equal(schema.get("name"), ADAPTER_SCHEMA_NAME, "adapter schema.name")
    _expect_equal(
        schema.get("version"), ADAPTER_SCHEMA_VERSION, "adapter schema.version"
    )
    semantics = _mapping(
        manifest.get("adapter_semantics"), "adapter manifest.adapter_semantics"
    )
    _expect_equal(
        semantics.get("boundary"),
        "serving_request",
        "adapter boundary",
    )
    _expect_equal(
        semantics.get("output_is_hbfsim_memory_trace"),
        False,
        "adapter output_is_hbfsim_memory_trace",
    )
    suite_config = _mapping(
        manifest.get("suite_config"), "adapter manifest.suite_config"
    )
    suite_id = suite_config.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id:
        raise AuditError("adapter suite_config.suite_id must be non-empty")
    suite_config_sha256 = suite_config.get("sha256")
    if (
        not isinstance(suite_config_sha256, str)
        or len(suite_config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in suite_config_sha256
        )
    ):
        raise AuditError(
            "adapter suite_config.sha256 must be 64 lowercase hex digits"
        )
    _integer(
        suite_config.get("bytes"),
        "adapter suite_config.bytes",
        minimum=1,
    )

    source = _mapping(manifest.get("source"), "adapter manifest.source")
    for key in ("repository", "revision", "license", "artifact", "sha256"):
        value = source.get(key)
        if not isinstance(value, str) or not value:
            raise AuditError(f"adapter source.{key} must be non-empty")
    if (
        len(source["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in source["sha256"])
    ):
        raise AuditError("adapter source.sha256 must be 64 lowercase hex digits")
    _integer(source.get("bytes"), "adapter source.bytes", minimum=1)
    _integer(source.get("records"), "adapter source.records", minimum=1)
    _expect_equal(
        source.get("hash_block_tokens"),
        16,
        "adapter source.hash_block_tokens",
    )

    workload = _mapping(
        manifest.get("frontier_workload"),
        "adapter manifest.frontier_workload",
    )
    model = _mapping(workload.get("model"), "adapter frontier_workload.model")
    for key in ("identity", "frontier_name"):
        value = model.get(key)
        if not isinstance(value, str) or not value:
            raise AuditError(f"adapter frontier_workload.model.{key} must be non-empty")
    primary_precision = workload.get("primary_precision_profile")
    if not isinstance(primary_precision, str) or not primary_precision:
        raise AuditError(
            "adapter frontier_workload.primary_precision_profile must be non-empty"
        )
    sensitivities = workload.get("sensitivity_precision_profiles")
    if (
        not isinstance(sensitivities, list)
        or not sensitivities
        or any(not isinstance(value, str) or not value for value in sensitivities)
    ):
        raise AuditError(
            "adapter frontier_workload.sensitivity_precision_profiles "
            "must be a non-empty string list"
        )

    output = _mapping(manifest.get("output"), "adapter manifest.output")
    _expect_equal(
        output.get("format"),
        "frontier_trace_replay_csv",
        "adapter output.format",
    )
    output_path = output.get("path")
    if (
        not isinstance(output_path, str)
        or not output_path
        or Path(output_path).is_absolute()
        or Path(output_path).parent != Path(".")
        or Path(output_path).name != request_csv.name
    ):
        raise AuditError(
            "adapter output.path must be the request CSV basename"
        )
    _expect_equal(
        output.get("sha256"), _sha256_file(request_csv), "adapter output.sha256"
    )
    _expect_equal(
        output.get("bytes"), request_csv.stat().st_size, "adapter output.bytes"
    )
    _expect_equal(
        output.get("columns"),
        ADAPTER_OUTPUT_COLUMNS,
        "adapter output.columns",
    )
    selection = _mapping(manifest.get("selection"), "adapter manifest.selection")
    window_id = selection.get("window_id")
    if window_id not in REQUIRED_REQUEST_WINDOWS:
        raise AuditError(
            "adapter selection.window_id must be steady, burst, or "
            "long_context_decode_tail"
        )
    _expect_equal(
        selection.get("algorithm"),
        REQUIRED_REQUEST_WINDOWS[window_id],
        "adapter selection.algorithm",
    )
    _expect_equal(selection.get("records"), len(requests), "adapter selected records")
    start_index = _integer(
        selection.get("start_index"),
        "adapter selection.start_index",
    )
    end_index = _integer(
        selection.get("end_index_exclusive"),
        "adapter selection.end_index_exclusive",
        minimum=1,
    )
    _expect_equal(
        end_index - start_index,
        len(requests),
        "adapter selection index span",
    )
    statistics = _mapping(manifest.get("statistics"), "adapter manifest.statistics")
    _expect_equal(
        statistics.get("prefill_tokens"),
        sum(request["prefill_tokens"] for request in requests),
        "adapter statistics.prefill_tokens",
    )
    _expect_equal(
        statistics.get("decode_tokens"),
        sum(request["decode_tokens"] for request in requests),
        "adapter statistics.decode_tokens",
    )
    _expect_equal(
        statistics.get("total_tokens"),
        sum(request["total_tokens"] for request in requests),
        "adapter statistics.total_tokens",
    )
    _expect_equal(
        statistics.get("max_prefill_tokens"),
        max(request["prefill_tokens"] for request in requests),
        "adapter statistics.max_prefill_tokens",
    )
    _expect_equal(
        statistics.get("max_decode_tokens"),
        max(request["decode_tokens"] for request in requests),
        "adapter statistics.max_decode_tokens",
    )
    eligibility = _mapping(
        manifest.get("eligibility"), "adapter manifest.eligibility"
    )
    _expect_equal(
        eligibility.get("production_request_window_validated"),
        True,
        "adapter production request validation",
    )
    _expect_equal(
        eligibility.get("frontier_replay_attached"),
        False,
        "adapter pre-audit Frontier attachment",
    )
    _expect_equal(
        eligibility.get("paper_result_eligible"),
        False,
        "adapter pre-audit paper eligibility",
    )
    return {
        "suite_id": suite_id,
        "suite_config": suite_config,
        "source": source,
        "frontier_workload": workload,
        "window_id": window_id,
        "selection": selection,
    }


def _verified_artifact(
    metadata: Any,
    description: str,
) -> Path:
    artifact = _mapping(metadata, description)
    path_value = artifact.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise AuditError(f"{description}.path must be non-empty")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise AuditError(f"{description}.path must be absolute")
    path = path.resolve()
    if not path.is_file():
        raise AuditError(f"missing {description}: {path}")
    _expect_equal(
        artifact.get("sha256"),
        _sha256_file(path),
        f"{description}.sha256",
    )
    _expect_equal(
        artifact.get("bytes"),
        path.stat().st_size,
        f"{description}.bytes",
    )
    return path


def _suite_relative_artifact(
    suite_dir: Path,
    metadata: Any,
    description: str,
) -> Path:
    artifact = _mapping(metadata, description)
    relative = artifact.get("path")
    if not isinstance(relative, str) or not relative:
        raise AuditError(f"{description}.path must be non-empty")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise AuditError(f"{description}.path must be a safe relative path")
    path = (suite_dir / relative_path).resolve()
    try:
        path.relative_to(suite_dir)
    except ValueError as error:
        raise AuditError(f"{description}.path escapes suite directory") from error
    if not path.is_file():
        raise AuditError(f"missing {description}: {path}")
    _expect_equal(artifact.get("sha256"), _sha256_file(path), f"{description}.sha256")
    _expect_equal(artifact.get("bytes"), path.stat().st_size, f"{description}.bytes")
    return path


def _validate_request_suite_verification(
    *,
    verification_path: Path,
    request_csv: Path,
    request_manifest_path: Path,
    adapter: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    verification = _load_object(
        verification_path,
        "request-suite verification receipt",
    )
    _expect_equal(
        verification.get("schema"),
        {
            "name": REQUEST_SUITE_VERIFICATION_SCHEMA_NAME,
            "version": REQUEST_SUITE_VERIFICATION_SCHEMA_VERSION,
        },
        "request-suite verification schema",
    )
    _expect_equal(verification.get("result"), "pass", "request-suite verification result")
    _expect_equal(
        verification.get("suite_id"),
        adapter["suite_id"],
        "request-suite verification suite_id",
    )
    verification_eligibility = _mapping(
        verification.get("eligibility"),
        "request-suite verification eligibility",
    )
    _expect_equal(
        verification_eligibility.get("production_request_suite_verified"),
        True,
        "request-suite production verification",
    )
    _expect_equal(
        verification_eligibility.get("paper_result_eligible"),
        False,
        "request-suite standalone paper eligibility",
    )
    _expect_equal(
        verification.get("frontier_workload"),
        adapter["frontier_workload"],
        "request-suite Frontier workload",
    )

    artifacts = _mapping(
        verification.get("artifacts"),
        "request-suite verification artifacts",
    )
    if set(artifacts) != {"suite_config", "source", "suite_manifest"}:
        raise AuditError(
            "request-suite verification artifacts must contain exactly "
            "suite_config, source, and suite_manifest"
        )
    suite_config_path = _verified_artifact(
        artifacts["suite_config"],
        "request-suite config",
    )
    source_path = _verified_artifact(
        artifacts["source"],
        "request-suite source",
    )
    suite_manifest_path = _verified_artifact(
        artifacts["suite_manifest"],
        "request-suite manifest",
    )
    _expect_equal(
        artifacts["suite_config"].get("sha256"),
        adapter["suite_config"].get("sha256"),
        "request-suite config/adapter digest",
    )
    _expect_equal(
        artifacts["suite_config"].get("bytes"),
        adapter["suite_config"].get("bytes"),
        "request-suite config/adapter bytes",
    )
    _expect_equal(
        artifacts["source"].get("sha256"),
        adapter["source"].get("sha256"),
        "request-suite source/adapter digest",
    )
    _expect_equal(
        artifacts["source"].get("bytes"),
        adapter["source"].get("bytes"),
        "request-suite source/adapter bytes",
    )

    suite = _load_object(suite_manifest_path, "request-suite manifest")
    _expect_equal(
        suite.get("schema"),
        {
            "name": REQUEST_SUITE_SCHEMA_NAME,
            "version": REQUEST_SUITE_SCHEMA_VERSION,
        },
        "request-suite manifest schema",
    )
    _expect_equal(suite.get("suite_id"), adapter["suite_id"], "request-suite suite_id")
    _expect_equal(
        suite.get("suite_config"),
        adapter["suite_config"],
        "request-suite config identity",
    )
    _expect_equal(suite.get("source"), adapter["source"], "request-suite source identity")
    _expect_equal(
        suite.get("frontier_workload"),
        adapter["frontier_workload"],
        "request-suite workload identity",
    )
    windows = _mapping(suite.get("windows"), "request-suite windows")
    if set(windows) != set(REQUIRED_REQUEST_WINDOWS):
        raise AuditError(
            "request-suite manifest must contain steady, burst, and "
            "long_context_decode_tail"
        )
    window_id = adapter["window_id"]
    suite_window = _mapping(
        windows.get(window_id),
        f"request-suite windows.{window_id}",
    )
    _expect_equal(
        suite_window.get("selection"),
        adapter["selection"],
        "request-suite window selection",
    )
    suite_dir = suite_manifest_path.parent.resolve()
    suite_csv = _suite_relative_artifact(
        suite_dir,
        suite_window.get("request_csv"),
        "request-suite request_csv",
    )
    suite_request_manifest = _suite_relative_artifact(
        suite_dir,
        suite_window.get("request_manifest"),
        "request-suite request_manifest",
    )
    _expect_equal(suite_csv, request_csv.resolve(), "request-suite CSV path")
    _expect_equal(
        suite_request_manifest,
        request_manifest_path.resolve(),
        "request-suite adapter manifest path",
    )

    receipt_windows = _mapping(
        verification.get("windows"),
        "request-suite verification windows",
    )
    if set(receipt_windows) != set(REQUIRED_REQUEST_WINDOWS):
        raise AuditError(
            "request-suite verification must cover all three required windows"
        )
    _expect_equal(
        receipt_windows[window_id],
        suite_window,
        "request-suite verified window",
    )
    return {
        "suite_id": adapter["suite_id"],
        "window_id": window_id,
        "selection": adapter["selection"],
        "frontier_workload": adapter["frontier_workload"],
    }, {
        "request_suite_verification": verification_path.resolve(),
        "request_suite_manifest": suite_manifest_path,
        "request_suite_config": suite_config_path,
        "request_source": source_path,
    }


def _validate_frontier_integration(
    *,
    bundle_manifest_path: Path,
    receipt_path: Path,
    runtime_path: Path,
    frontier_repository: str,
    frontier_revision: str,
) -> tuple[dict[str, Any], Path]:
    bundle_manifest_path = bundle_manifest_path.resolve()
    bundle = _load_object(
        bundle_manifest_path,
        "Frontier integration bundle manifest",
    )
    _expect_equal(
        bundle.get("schema"),
        {"name": "hbfsim.frontier_integration_patch", "version": 1},
        "Frontier integration bundle schema",
    )
    integration_id = bundle.get("integration_id")
    if not isinstance(integration_id, str) or not integration_id:
        raise AuditError("Frontier integration_id must be non-empty")
    upstream = _mapping(bundle.get("upstream"), "Frontier integration upstream")
    _expect_equal(
        upstream.get("repository"),
        frontier_repository,
        "Frontier integration repository",
    )
    _expect_equal(
        upstream.get("revision"),
        frontier_revision,
        "Frontier integration revision",
    )
    if (
        not isinstance(frontier_revision, str)
        or not frontier_revision
        or len(frontier_revision) != 40
        or any(character not in "0123456789abcdef" for character in frontier_revision)
    ):
        raise AuditError("Frontier revision must be a full lowercase git SHA")
    contracts = _mapping(
        bundle.get("contracts"),
        "Frontier integration contracts",
    )
    _expect_equal(
        contracts,
        {
            "memory_contract_schema_version": MEMORY_CONTRACT_SCHEMA_VERSION,
            "allocator_state_schema_version": ALLOCATOR_STATE_SCHEMA_VERSION,
            "residency_plan_schema_version": (
                RESIDENCY_PLAN_SCHEMA_VERSION
            ),
            "kv_lifecycle_schema_version": 1,
        },
        "Frontier integration contract versions",
    )
    patch = _mapping(bundle.get("patch"), "Frontier integration patch")
    patch_relative = patch.get("path")
    if not isinstance(patch_relative, str) or not patch_relative:
        raise AuditError("Frontier integration patch path must be non-empty")
    if len(bundle_manifest_path.parents) < 3:
        raise AuditError("Frontier integration manifest path has no repository root")
    repository_root = bundle_manifest_path.parents[2]
    patch_path = (repository_root / patch_relative).resolve()
    try:
        patch_path.relative_to(repository_root)
    except ValueError as error:
        raise AuditError("Frontier integration patch escapes its repository") from error
    if not patch_path.is_file():
        raise AuditError(f"Frontier integration patch is missing: {patch_path}")
    _expect_equal(
        patch.get("bytes"),
        patch_path.stat().st_size,
        "Frontier integration patch bytes",
    )
    _expect_equal(
        patch.get("sha256"),
        _sha256_file(patch_path),
        "Frontier integration patch digest",
    )
    post_apply = _mapping(
        bundle.get("post_apply_sha256"),
        "Frontier integration post-apply hashes",
    )
    if not post_apply:
        raise AuditError("Frontier integration post-apply hashes are empty")
    for path_text, digest in post_apply.items():
        if (
            not isinstance(path_text, str)
            or not path_text
            or Path(path_text).is_absolute()
            or ".." in Path(path_text).parts
        ):
            raise AuditError(
                f"invalid Frontier integration post-apply path: {path_text!r}"
            )
        _sha256_string(
            digest,
            f"Frontier integration post-apply digest for {path_text}",
        )

    receipt = _load_object(receipt_path, "Frontier integration receipt")
    _expect_equal(
        receipt.get("schema"),
        {"name": "hbfsim.frontier_integration_receipt", "version": 1},
        "Frontier integration receipt schema",
    )
    _expect_equal(receipt.get("result"), "pass", "Frontier integration receipt")
    _expect_equal(
        receipt.get("integration_id"),
        integration_id,
        "Frontier integration receipt integration_id",
    )
    _expect_equal(
        receipt.get("upstream"),
        upstream,
        "Frontier integration receipt upstream",
    )
    _expect_equal(
        receipt.get("patch"),
        patch,
        "Frontier integration receipt patch",
    )
    _expect_equal(
        receipt.get("contracts"),
        contracts,
        "Frontier integration receipt contracts",
    )
    _expect_equal(
        receipt.get("post_apply_sha256"),
        post_apply,
        "Frontier integration receipt post-apply hashes",
    )
    _expect_equal(
        receipt.get("staged_diff_sha256"),
        patch.get("sha256"),
        "Frontier integration receipt staged diff digest",
    )

    runtime = _load_object(runtime_path, "Frontier runtime integration identity")
    _expect_equal(
        runtime.get("schema"),
        {"name": "hbfsim.frontier_integration_runtime", "version": 1},
        "Frontier runtime integration schema",
    )
    _expect_equal(
        runtime.get("integration_id"),
        integration_id,
        "Frontier runtime integration_id",
    )
    _expect_equal(
        runtime.get("upstream_revision"),
        frontier_revision,
        "Frontier runtime upstream revision",
    )
    _expect_equal(
        runtime.get("contracts"),
        contracts,
        "Frontier runtime contract versions",
    )
    return {
        "integration_id": integration_id,
        "bundle_manifest_sha256": _sha256_file(bundle_manifest_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "patch": {
            "sha256": patch["sha256"],
            "bytes": patch["bytes"],
        },
        "contracts": contracts,
        "post_apply_file_count": len(post_apply),
    }, patch_path


def _validate_frontier_config(
    config: dict[str, Any],
    request_csv: Path,
    expected_workload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    _expect_equal(config.get("sys_arch"), "co-location", "Frontier sys_arch")
    request_generator = _mapping(
        config.get("request_generator_config"),
        "Frontier request_generator_config",
    )
    _expect_equal(
        request_generator.get("name"), "trace_replay", "request generator name"
    )
    for field in ("prefill_scale_factor", "decode_scale_factor", "time_scale_factor"):
        _expect_equal(
            _finite_number(request_generator.get(field), f"request generator {field}"),
            1.0,
            f"request generator {field}",
        )
    configured_trace = request_generator.get("trace_file")
    if not isinstance(configured_trace, str) or not configured_trace:
        raise AuditError("request generator trace_file must be non-empty")
    configured_path = Path(configured_trace).expanduser()
    if not configured_path.is_absolute():
        raise AuditError("request generator trace_file must be an absolute path")
    if not configured_path.is_file():
        raise AuditError(f"configured Frontier trace_file no longer exists: {configured_path}")
    _expect_equal(
        _sha256_file(configured_path),
        _sha256_file(request_csv),
        "configured Frontier trace digest",
    )

    cluster = _mapping(config.get("cluster_config"), "Frontier cluster_config")
    _expect_equal(cluster.get("cluster_type"), 1, "Frontier cluster_type")
    _expect_equal(cluster.get("num_replicas"), 1, "Frontier num_replicas")
    replica = _mapping(cluster.get("replica_config"), "Frontier replica_config")
    _expect_equal(replica.get("num_pipeline_stages"), 1, "Frontier PP")
    _expect_equal(replica.get("data_parallel_size"), 1, "Frontier DP")
    _expect_equal(
        replica.get("attn_data_parallel_size", 1),
        1,
        "Frontier attention DP",
    )
    _expect_equal(
        replica.get("attn_tensor_parallel_size"),
        1,
        "Frontier attention TP",
    )
    _expect_equal(
        replica.get("moe_tensor_parallel_size"),
        1,
        "Frontier MoE TP",
    )
    _expect_equal(
        replica.get("moe_expert_parallel_size"),
        1,
        "Frontier MoE EP",
    )
    speculative = _mapping(
        replica.get("speculative_decoding_config"),
        "Frontier speculative_decoding_config",
    )
    _expect_equal(
        speculative.get("enabled"), False, "Frontier speculative decoding"
    )
    model_config = _mapping(
        replica.get("model_config"), "Frontier replica model_config"
    )
    _expect_equal(model_config.get("is_moe"), False, "Frontier dense model")
    selected_precision_id = replica.get("memory_precision_profile")
    if not isinstance(selected_precision_id, str) or not selected_precision_id:
        raise AuditError(
            "Frontier replica memory_precision_profile must be non-empty"
        )
    primary_precision_id = expected_workload["primary_precision_profile"]
    sensitivity_precision_ids = expected_workload[
        "sensitivity_precision_profiles"
    ]
    allowed_precision_ids = [
        primary_precision_id,
        *sensitivity_precision_ids,
    ]
    if selected_precision_id not in allowed_precision_ids:
        raise AuditError(
            "Frontier memory_precision_profile is outside the verified "
            f"request-suite contract: selected={selected_precision_id!r}, "
            f"allowed={allowed_precision_ids!r}"
        )
    raw_precision_profiles = _mapping(
        model_config.get("memory_precision_profiles"),
        "Frontier model_config.memory_precision_profiles",
    )
    if set(raw_precision_profiles) != set(allowed_precision_ids):
        raise AuditError(
            "Frontier memory precision profiles must exactly match the "
            "request-suite primary and sensitivity profiles"
        )
    precision_profiles = {
        profile_id: _validate_memory_precision_profile(
            raw_precision_profiles[profile_id],
            profile_id=profile_id,
            path=(
                "Frontier model_config.memory_precision_profiles."
                f"{profile_id}"
            ),
        )
        for profile_id in allowed_precision_ids
    }
    _expect_equal(
        model_config.get("default_memory_precision_profile"),
        primary_precision_id,
        "Frontier default_memory_precision_profile",
    )
    model_source_identity = _mapping(
        model_config.get("model_source_identity"),
        "Frontier model_config.model_source_identity",
    )
    for field in ("repository", "revision", "config_access"):
        field_value = model_source_identity.get(field)
        if not isinstance(field_value, str) or not field_value:
            raise AuditError(
                "Frontier model_config.model_source_identity."
                f"{field} must be non-empty"
            )
    model_source_revision = model_source_identity["revision"]
    if (
        len(model_source_revision) != 40
        or model_source_revision != model_source_revision.lower()
        or any(
            character not in "0123456789abcdef"
            for character in model_source_revision
        )
    ):
        raise AuditError(
            "Frontier model source revision must be a full lowercase git SHA"
        )
    if not isinstance(model_config.get("tie_word_embeddings"), bool):
        raise AuditError(
            "Frontier model_config.tie_word_embeddings must be an explicit boolean"
        )
    source_config_path = model_config.get("source_config_path")
    if not isinstance(source_config_path, str) or not source_config_path:
        raise AuditError(
            "Frontier model_config.source_config_path must be a non-empty string"
        )
    source_config_sha256 = model_config.get("source_config_sha256")
    if (
        not isinstance(source_config_sha256, str)
        or len(source_config_sha256) != 64
        or source_config_sha256 != source_config_sha256.lower()
        or any(
            character not in "0123456789abcdef"
            for character in source_config_sha256
        )
    ):
        raise AuditError(
            "Frontier model_config.source_config_sha256 must be 64 lowercase "
            "hexadecimal digits"
        )
    _integer(
        model_config.get("max_position_embeddings"),
        "Frontier max_position_embeddings",
        minimum=1,
    )
    scheduler = _mapping(
        cluster.get("replica_scheduler_config"),
        "Frontier replica_scheduler_config",
    )
    _expect_equal(scheduler.get("name"), "vllm_v1", "Frontier scheduler")
    _expect_equal(
        scheduler.get("num_blocks_mode"),
        "hybrid_residency",
        "Frontier scheduler num_blocks_mode",
    )
    _expect_equal(
        scheduler.get("num_blocks"),
        0,
        "Frontier configured num_blocks before hybrid derivation",
    )
    physical_hbm_capacity_bytes = _integer(
        scheduler.get("hybrid_physical_hbm_capacity_bytes"),
        "Frontier hybrid physical HBM capacity",
        minimum=1,
    )
    if physical_hbm_capacity_bytes not in {
        96 * 1024**3,
        192 * 1024**3,
        288 * 1024**3,
        384 * 1024**3,
    }:
        raise AuditError(
            "Frontier hybrid physical HBM capacity is outside the "
            "canonical 96/192/288/384 GiB capacity set"
        )
    capacity_pressure_target = _finite_number(
        scheduler.get("hybrid_capacity_pressure_target"),
        "Frontier hybrid capacity pressure target",
    )
    if capacity_pressure_target not in {0.75, 1.0, 1.25, 1.5, 2.0}:
        raise AuditError(
            "Frontier hybrid capacity pressure is outside the canonical grid"
        )
    runtime_overhead_bytes = _integer(
        scheduler.get("non_kv_cache_overhead_bytes"),
        "Frontier hybrid runtime overhead bytes",
        minimum=1,
    )
    _expect_equal(
        scheduler.get("gpu_memory_utilization"),
        None,
        "Frontier hybrid gpu_memory_utilization",
    )
    _expect_equal(
        scheduler.get(
            "enable_runtime_non_kv_cache_overhead_profiling"
        ),
        False,
        "Frontier hybrid runtime overhead profiling",
    )
    _expect_equal(
        scheduler.get("runtime_weights_memory_source"),
        "param_counter",
        "Frontier hybrid weight-memory source",
    )
    block_size = _integer(scheduler.get("block_size"), "Frontier block_size", minimum=1)
    _expect_equal(block_size, 16, "Frontier prefix block size")
    _expect_equal(
        scheduler.get("enable_prefix_caching"),
        True,
        "Frontier prefix caching",
    )
    _expect_equal(
        scheduler.get("enable_preemption"),
        False,
        "Frontier preemption configuration",
    )
    _expect_equal(
        scheduler.get("num_preallocate_tokens", 0),
        0,
        "Frontier KV preallocation",
    )
    caching_hash_algo = scheduler.get("prefix_caching_hash_algo")
    if caching_hash_algo not in {"builtin", "sha256"}:
        raise AuditError(
            "Frontier prefix_caching_hash_algo must be 'builtin' or 'sha256'"
        )

    metrics = _mapping(config.get("metrics_config"), "Frontier metrics_config")
    _expect_equal(
        metrics.get("store_frontier_stage_batch_ledger"),
        True,
        "Frontier stage ledger enabled",
    )
    predictor = _mapping(
        cluster.get("execution_time_predictor_config"),
        "Frontier execution_time_predictor_config",
    )
    dummy = predictor.get("enable_dummy_mode")
    if not isinstance(dummy, bool):
        raise AuditError("Frontier enable_dummy_mode must be boolean")
    return {
        "model": replica.get("model_name"),
        "model_config": model_config,
        "memory_precision": {
            "selected_profile_id": selected_precision_id,
            "selected_profile_role": (
                "primary"
                if selected_precision_id == primary_precision_id
                else "sensitivity"
            ),
            "selected_profile": precision_profiles[selected_precision_id],
            "primary_profile_id": primary_precision_id,
            "sensitivity_profile_ids": sensitivity_precision_ids,
            "configured_profiles": precision_profiles,
            "model_source_identity": model_source_identity,
        },
        "device": replica.get("device"),
        "parallelism": {
            "pipeline": replica.get("num_pipeline_stages"),
            "data": replica.get("data_parallel_size"),
            "attention_tensor": replica.get("attn_tensor_parallel_size"),
            "moe_tensor": replica.get("moe_tensor_parallel_size"),
            "moe_expert": replica.get("moe_expert_parallel_size"),
        },
        "scheduler": {
            "name": scheduler.get("name"),
            "num_blocks_mode": scheduler.get("num_blocks_mode"),
            "block_size_tokens": block_size,
            "physical_hbm_capacity_bytes": physical_hbm_capacity_bytes,
            "capacity_pressure_target": capacity_pressure_target,
            "runtime_overhead_bytes": runtime_overhead_bytes,
            "runtime_overhead_source": (
                "configured_unprofiled_sensitivity"
            ),
            "hardware_capacity_calibrated": False,
            "prefix_caching": scheduler.get("enable_prefix_caching"),
            "prefix_caching_hash_algo": caching_hash_algo,
            "chunked_prefill": scheduler.get("enable_chunked_prefill"),
            "max_tokens_in_batch": scheduler.get("max_tokens_in_batch"),
            "batch_size_cap": scheduler.get("batch_size_cap"),
        },
    }, dummy


def _validate_model_memory_ledger(
    system: dict[str, Any],
    frontend: dict[str, Any],
) -> dict[str, Any]:
    model_weight_memory = _mapping(
        system.get("model_weight_memory"),
        "system model_weight_memory",
    )
    monolithic = _mapping(
        model_weight_memory.get("MONOLITHIC"),
        "system model_weight_memory.MONOLITHIC",
    )
    ledger = _mapping(
        monolithic.get("memory_ledger"),
        "system model weight memory_ledger",
    )
    profile = frontend["memory_precision"]["selected_profile"]
    _expect_equal(
        monolithic.get("memory_precision_profile"),
        profile,
        "system selected memory precision profile",
    )
    _expect_equal(
        ledger.get("profile"),
        profile,
        "system memory ledger precision profile",
    )
    categories = _mapping(
        ledger.get("categories"),
        "system memory ledger categories",
    )
    expected_categories = {
        "attention",
        "ffn",
        "normalization",
        "pipeline_boundary_and_auxiliary",
    }
    _expect_equal(
        set(categories),
        expected_categories,
        "system memory ledger category set",
    )
    totals = {
        "parameters": 0,
        "matrix_payload_bytes": 0,
        "non_matrix_payload_bytes": 0,
        "quantization_metadata_bytes": 0,
        "memory_bytes": 0,
    }
    category_memory_bytes: dict[str, int] = {}
    for category_name in sorted(expected_categories):
        category = _mapping(
            categories[category_name],
            f"system memory ledger category {category_name}",
        )
        parameters = _integer(
            category.get("parameters"),
            f"system {category_name} parameters",
        )
        matrix_parameters = _integer(
            category.get("matrix_parameters"),
            f"system {category_name} matrix_parameters",
        )
        non_matrix_parameters = _integer(
            category.get("non_matrix_parameters"),
            f"system {category_name} non_matrix_parameters",
        )
        scale_elements = _integer(
            category.get("scale_elements"),
            f"system {category_name} scale_elements",
        )
        matrix_payload_bytes = _integer(
            category.get("matrix_payload_bytes"),
            f"system {category_name} matrix_payload_bytes",
        )
        non_matrix_payload_bytes = _integer(
            category.get("non_matrix_payload_bytes"),
            f"system {category_name} non_matrix_payload_bytes",
        )
        metadata_bytes = _integer(
            category.get("quantization_metadata_bytes"),
            f"system {category_name} quantization_metadata_bytes",
        )
        memory_bytes = _integer(
            category.get("memory_bytes"),
            f"system {category_name} memory_bytes",
        )
        _expect_equal(
            parameters,
            matrix_parameters + non_matrix_parameters,
            f"system {category_name} parameter partition",
        )
        _expect_equal(
            matrix_payload_bytes,
            matrix_parameters * profile["matrix_weight_bytes"],
            f"system {category_name} matrix payload",
        )
        _expect_equal(
            non_matrix_payload_bytes,
            non_matrix_parameters * profile["non_matrix_weight_bytes"],
            f"system {category_name} non-matrix payload",
        )
        _expect_equal(
            metadata_bytes,
            scale_elements
            * (profile["scale_bytes"] + profile["zero_point_bytes"]),
            f"system {category_name} quantization metadata",
        )
        _expect_equal(
            memory_bytes,
            matrix_payload_bytes
            + non_matrix_payload_bytes
            + metadata_bytes,
            f"system {category_name} memory conservation",
        )
        totals["parameters"] += parameters
        totals["matrix_payload_bytes"] += matrix_payload_bytes
        totals["non_matrix_payload_bytes"] += non_matrix_payload_bytes
        totals["quantization_metadata_bytes"] += metadata_bytes
        totals["memory_bytes"] += memory_bytes
        category_memory_bytes[category_name] = memory_bytes

    _expect_equal(
        ledger.get("total_parameters"),
        totals["parameters"],
        "system memory ledger total_parameters",
    )
    _expect_equal(
        ledger.get("matrix_payload_bytes"),
        totals["matrix_payload_bytes"],
        "system memory ledger matrix payload",
    )
    _expect_equal(
        ledger.get("non_matrix_payload_bytes"),
        totals["non_matrix_payload_bytes"],
        "system memory ledger non-matrix payload",
    )
    _expect_equal(
        ledger.get("quantization_metadata_bytes"),
        totals["quantization_metadata_bytes"],
        "system memory ledger quantization metadata",
    )
    _expect_equal(
        ledger.get("total_memory_bytes"),
        totals["memory_bytes"],
        "system memory ledger total bytes",
    )
    _expect_equal(
        monolithic.get("total_parameters"),
        totals["parameters"],
        "system model weight total parameters",
    )
    _expect_equal(
        monolithic.get("total_memory_bytes"),
        totals["memory_bytes"],
        "system model weight total bytes",
    )
    _expect_equal(
        monolithic.get("storage_breakdown"),
        {
            "matrix_payload_bytes": totals["matrix_payload_bytes"],
            "non_matrix_payload_bytes": totals[
                "non_matrix_payload_bytes"
            ],
            "quantization_metadata_bytes": totals[
                "quantization_metadata_bytes"
            ],
        },
        "system model weight storage breakdown",
    )
    breakdown = _mapping(
        monolithic.get("breakdown"),
        "system model weight breakdown",
    )
    breakdown_fields = {
        "attention": "attention_memory_bytes",
        "ffn": "ffn_memory_bytes",
        "normalization": "normalization_memory_bytes",
        "pipeline_boundary_and_auxiliary": (
            "pipeline_boundary_and_auxiliary_memory_bytes"
        ),
    }
    for category_name, field_name in breakdown_fields.items():
        _expect_equal(
            breakdown.get(field_name),
            category_memory_bytes[category_name],
            f"system model weight breakdown {category_name}",
        )
    streaming = _mapping(
        monolithic.get("weight_streaming_ledger"),
        "system model weight_streaming_ledger",
    )
    num_layers = _integer(
        streaming.get("num_transformer_layers"),
        "system streaming num_transformer_layers",
        minimum=1,
    )
    _expect_equal(
        num_layers,
        frontend["model_config"]["num_layers"],
        "system streaming/model layer count",
    )
    streaming_objects = _mapping(
        streaming.get("objects"),
        "system streaming objects",
    )
    _expect_equal(
        set(streaming_objects),
        {
            "embedding",
            "transformer_layer",
            "final_norm",
            "output_head",
        },
        "system streaming object set",
    )
    validated_objects: dict[str, dict[str, int]] = {}
    for object_name, object_value in streaming_objects.items():
        object_ledger = _mapping(
            object_value,
            f"system streaming object {object_name}",
        )
        parameters = _integer(
            object_ledger.get("parameters"),
            f"system streaming {object_name} parameters",
        )
        matrix_parameters = _integer(
            object_ledger.get("matrix_parameters"),
            f"system streaming {object_name} matrix parameters",
        )
        non_matrix_parameters = _integer(
            object_ledger.get("non_matrix_parameters"),
            f"system streaming {object_name} non-matrix parameters",
        )
        matrix_payload = _integer(
            object_ledger.get("matrix_payload_bytes"),
            f"system streaming {object_name} matrix payload",
        )
        non_matrix_payload = _integer(
            object_ledger.get("non_matrix_payload_bytes"),
            f"system streaming {object_name} non-matrix payload",
        )
        scale_elements = _integer(
            object_ledger.get("scale_elements"),
            f"system streaming {object_name} scale elements",
        )
        metadata = _integer(
            object_ledger.get("quantization_metadata_bytes"),
            f"system streaming {object_name} metadata",
        )
        memory = _integer(
            object_ledger.get("memory_bytes"),
            f"system streaming {object_name} memory",
        )
        _expect_equal(
            parameters,
            matrix_parameters + non_matrix_parameters,
            f"system streaming {object_name} parameter partition",
        )
        _expect_equal(
            matrix_payload,
            matrix_parameters * profile["matrix_weight_bytes"],
            f"system streaming {object_name} matrix bytes",
        )
        _expect_equal(
            non_matrix_payload,
            non_matrix_parameters * profile["non_matrix_weight_bytes"],
            f"system streaming {object_name} non-matrix bytes",
        )
        _expect_equal(
            metadata,
            scale_elements
            * (profile["scale_bytes"] + profile["zero_point_bytes"]),
            f"system streaming {object_name} metadata bytes",
        )
        _expect_equal(
            memory,
            matrix_payload + non_matrix_payload + metadata,
            f"system streaming {object_name} memory conservation",
        )
        validated_objects[object_name] = {
            "parameters": parameters,
            "memory_bytes": memory,
        }
    reconstructed_weight_bytes = (
        validated_objects["embedding"]["memory_bytes"]
        + num_layers
        * validated_objects["transformer_layer"]["memory_bytes"]
        + validated_objects["final_norm"]["memory_bytes"]
        + validated_objects["output_head"]["memory_bytes"]
    )
    reconstructed_parameters = (
        validated_objects["embedding"]["parameters"]
        + num_layers * validated_objects["transformer_layer"]["parameters"]
        + validated_objects["final_norm"]["parameters"]
        + validated_objects["output_head"]["parameters"]
    )
    _expect_equal(
        reconstructed_weight_bytes,
        totals["memory_bytes"],
        "system streaming immutable weight bytes",
    )
    _expect_equal(
        reconstructed_parameters,
        totals["parameters"],
        "system streaming immutable parameters",
    )
    active_buffer_bytes_per_slot = _integer(
        streaming.get("active_buffer_bytes_per_slot"),
        "system streaming active buffer bytes per slot",
        minimum=1,
    )
    _expect_equal(
        active_buffer_bytes_per_slot,
        max(
            object_ledger["memory_bytes"]
            for object_ledger in validated_objects.values()
        ),
        "system streaming active-buffer maximum",
    )
    _expect_equal(
        streaming.get("immutable_weight_backing_bytes"),
        totals["memory_bytes"],
        "system streaming immutable backing bytes",
    )
    _expect_equal(
        streaming.get("total_parameters"),
        totals["parameters"],
        "system streaming total parameters",
    )
    return {
        "profile_id": profile["profile_id"],
        "profile_role": frontend["memory_precision"][
            "selected_profile_role"
        ],
        "total_parameters": totals["parameters"],
        "total_memory_bytes": totals["memory_bytes"],
        "storage_breakdown": {
            key: totals[key]
            for key in (
                "matrix_payload_bytes",
                "non_matrix_payload_bytes",
                "quantization_metadata_bytes",
            )
        },
        "category_memory_bytes": category_memory_bytes,
        "weight_streaming": {
            "num_transformer_layers": num_layers,
            "active_buffer_bytes_per_slot": (
                active_buffer_bytes_per_slot
            ),
            "immutable_weight_backing_bytes": totals["memory_bytes"],
            "objects": validated_objects,
        },
    }


def _read_request_metrics(
    path: Path, requests: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise AuditError(f"cannot read Frontier request metrics: {path}: {error}") from error
    required = {
        "Request Id",
        "request_num_tokens",
        "request_num_prefill_tokens",
        "request_num_decode_tokens",
        "request_cached_prefill_tokens",
        "request_prefix_cache_query_blocks",
        "request_prefix_cache_hit_blocks",
    }
    metrics: dict[str, dict[str, int]] = {}
    totals = {"cached_tokens": 0, "query_blocks": 0, "hit_blocks": 0}
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AuditError("Frontier request metrics has no header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise AuditError(
                "Frontier request metrics is missing columns: "
                + ", ".join(sorted(missing))
            )
        for row_no, row in enumerate(reader, start=2):
            request_id = row.get("Request Id")
            if request_id is None or request_id in metrics:
                raise AuditError(
                    f"Frontier request metrics row {row_no} has a missing or duplicate ID"
                )
            values = {
                "total": _csv_integer(
                    row.get("request_num_tokens"), f"request metrics {request_id}.total"
                ),
                "prefill": _csv_integer(
                    row.get("request_num_prefill_tokens"),
                    f"request metrics {request_id}.prefill",
                ),
                "decode": _csv_integer(
                    row.get("request_num_decode_tokens"),
                    f"request metrics {request_id}.decode",
                ),
                "cached": _csv_integer(
                    row.get("request_cached_prefill_tokens"),
                    f"request metrics {request_id}.cached",
                ),
                "query_blocks": _csv_integer(
                    row.get("request_prefix_cache_query_blocks"),
                    f"request metrics {request_id}.query_blocks",
                ),
                "hit_blocks": _csv_integer(
                    row.get("request_prefix_cache_hit_blocks"),
                    f"request metrics {request_id}.hit_blocks",
                ),
            }
            if values["hit_blocks"] > values["query_blocks"]:
                raise AuditError(f"request {request_id} has more hit than query blocks")
            if values["cached"] > values["prefill"]:
                raise AuditError(f"request {request_id} caches more than its prefill")
            metrics[request_id] = values
            totals["cached_tokens"] += values["cached"]
            totals["query_blocks"] += values["query_blocks"]
            totals["hit_blocks"] += values["hit_blocks"]

    expected_ids = {request["id"] for request in requests}
    _expect_equal(set(metrics), expected_ids, "Frontier request metric IDs")
    for request in requests:
        values = metrics[request["id"]]
        _expect_equal(values["prefill"], request["prefill_tokens"], "request prefill")
        _expect_equal(values["decode"], request["decode_tokens"], "request decode")
        _expect_equal(values["total"], request["total_tokens"], "request total")
        cacheable_hashes = request["block_hash_ids"][
            : request["prefill_tokens"] // 16
        ]
        _expect_equal(
            values["query_blocks"],
            len(cacheable_hashes),
            f"request {request['id']} complete prefix-cache query blocks",
        )
    return metrics, totals


def _read_ledger(
    path: Path,
    requests: list[dict[str, Any]],
    request_metrics: dict[str, dict[str, int]],
    *,
    expected_caching_hash_algo: str,
) -> tuple[dict[str, int], dict[str, Any], list[dict[str, Any]]]:
    requests_by_id = {request["id"]: request for request in requests}
    expected_ids = set(requests_by_id)
    scheduled = {request_id: 0 for request_id in expected_ids}
    rows = 0
    batch_ids: set[int] = set()
    batch_size_counts: dict[int, int] = {}
    contracts: list[dict[str, Any]] = []
    next_context_by_request: dict[str, int] = {}
    final_frontier_by_request: dict[str, int] = {}
    final_kv_frontier_by_request: dict[str, int] = {}
    contract_stream_id: str | None = None
    contract_num_gpu_blocks: int | None = None
    contract_residency_plan: dict[str, Any] | None = None
    previous_lifecycle_cursor = 0
    first_start: float | None = None
    last_end = 0.0
    previous_start = -1.0
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise AuditError(f"cannot read Frontier stage ledger: {path}: {error}") from error
    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise AuditError(f"Frontier stage ledger line {line_no} is blank")
            try:
                row = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise AuditError(
                    f"invalid Frontier stage ledger JSON at line {line_no}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise AuditError(f"Frontier stage ledger line {line_no} is not an object")
            _expect_equal(row.get("cluster_type"), "MONOLITHIC", "ledger cluster")
            _expect_equal(row.get("replica_id"), 0, "ledger replica")
            _expect_equal(row.get("dp_id"), 0, "ledger data-parallel id")
            _expect_equal(row.get("stage_id"), 0, "ledger pipeline stage")
            batch_id = _integer(row.get("batch_id"), f"ledger line {line_no}.batch_id")
            _expect_equal(
                batch_id,
                rows,
                f"ledger line {line_no} contiguous batch order",
            )
            if batch_id in batch_ids:
                raise AuditError(f"duplicate ledger batch_id {batch_id}")
            batch_ids.add(batch_id)
            request_ids = row.get("request_ids")
            request_tokens = row.get("request_num_tokens")
            if not isinstance(request_ids, list) or not request_ids:
                raise AuditError(f"ledger line {line_no}.request_ids must be non-empty")
            if not isinstance(request_tokens, list) or len(request_tokens) != len(request_ids):
                raise AuditError(
                    f"ledger line {line_no} request IDs/tokens have different lengths"
                )
            if len(set(map(str, request_ids))) != len(request_ids):
                raise AuditError(f"ledger line {line_no} repeats a request ID")
            normalized_request_ids: list[str] = []
            normalized_request_tokens: list[int] = []
            for position, (request_id_value, token_count) in enumerate(
                zip(request_ids, request_tokens)
            ):
                request_id = str(request_id_value)
                if request_id not in expected_ids:
                    raise AuditError(
                        f"ledger line {line_no} references unknown request {request_id}"
                    )
                normalized_tokens = _integer(
                    token_count,
                    f"ledger line {line_no}.request_num_tokens[{position}]",
                    minimum=1,
                )
                normalized_request_ids.append(request_id)
                normalized_request_tokens.append(normalized_tokens)
                scheduled[request_id] += normalized_tokens
            start = _finite_number(
                row.get("stage_start_ts"), f"ledger line {line_no}.stage_start_ts"
            )
            end = _finite_number(
                row.get("stage_end_ts"), f"ledger line {line_no}.stage_end_ts"
            )
            if end < start:
                raise AuditError(f"ledger line {line_no} ends before it starts")
            if start < previous_start:
                raise AuditError("Frontier stage ledger starts must be nondecreasing")
            previous_start = start
            if first_start is None:
                first_start = start
            last_end = max(last_end, end)

            contract = _mapping(
                row.get("hbfsim_memory_contract"),
                f"ledger line {line_no}.hbfsim_memory_contract",
            )
            _expect_equal(
                contract.get("schema_version"),
                MEMORY_CONTRACT_SCHEMA_VERSION,
                f"ledger line {line_no} memory contract schema",
            )
            _expect_equal(
                contract.get("contract"),
                "frontier-hbfsim-memory-object",
                f"ledger line {line_no} memory contract name",
            )
            _expect_equal(
                contract.get("batch_id"),
                batch_id,
                f"ledger line {line_no} contract batch_id",
            )
            _expect_equal(
                contract.get("cluster_type"),
                "MONOLITHIC",
                f"ledger line {line_no} contract cluster",
            )
            _expect_equal(
                contract.get("replica_id"),
                0,
                f"ledger line {line_no} contract replica",
            )
            _expect_equal(
                contract.get("dp_id"),
                0,
                f"ledger line {line_no} contract dp",
            )
            _expect_equal(
                _finite_number(
                    contract.get("captured_at"),
                    f"ledger line {line_no} contract captured_at",
                ),
                start,
                f"ledger line {line_no} contract capture/start time",
            )
            runtime_filter = contract.get("runtime_live_filter")
            if runtime_filter is not None:
                raise AuditError(
                    f"ledger line {line_no} used runtime-live batch filtering"
                )
            if "lifecycle_events" in contract:
                raise AuditError(
                    f"ledger line {line_no} embeds lifecycle_events; "
                    "compact contract v4 requires one canonical lifecycle ledger"
                )
            allocator_state = _mapping(
                contract.get("allocator_state"),
                f"ledger line {line_no} contract allocator_state",
            )
            _expect_equal(
                allocator_state,
                {
                    "schema_version": ALLOCATOR_STATE_SCHEMA_VERSION,
                    "digest": "sha256",
                    "canonical_json": "sorted-keys-compact",
                    "block_order": "request-ownership-order",
                },
                f"ledger line {line_no} allocator-state contract",
            )

            kv_cache = _mapping(
                contract.get("kv_cache"),
                f"ledger line {line_no} contract kv_cache",
            )
            _expect_equal(
                kv_cache.get("block_size_tokens"),
                16,
                f"ledger line {line_no} KV block size",
            )
            _expect_equal(
                kv_cache.get("enable_prefix_caching"),
                True,
                f"ledger line {line_no} prefix caching",
            )
            _expect_equal(
                kv_cache.get("num_preallocate_tokens"),
                0,
                f"ledger line {line_no} KV preallocation",
            )
            _expect_equal(
                kv_cache.get("caching_hash_algo"),
                expected_caching_hash_algo,
                f"ledger line {line_no} KV caching hash algorithm",
            )
            _expect_equal(
                kv_cache.get("no_preemption_admission_policy"),
                NO_PREEMPTION_ADMISSION_POLICY,
                f"ledger line {line_no} no-preemption admission policy",
            )
            num_gpu_blocks = _integer(
                kv_cache.get("num_gpu_blocks"),
                f"ledger line {line_no} num_gpu_blocks",
                minimum=1,
            )
            residency_plan = _mapping(
                contract.get("residency_plan"),
                f"ledger line {line_no} residency_plan",
            )
            _expect_equal(
                residency_plan.get("schema_version"),
                RESIDENCY_PLAN_SCHEMA_VERSION,
                f"ledger line {line_no} residency-plan schema",
            )
            _expect_equal(
                residency_plan.get("policy"),
                "hybrid_residency_v1",
                f"ledger line {line_no} residency policy",
            )
            _expect_equal(
                residency_plan.get("num_logical_kv_blocks"),
                num_gpu_blocks,
                f"ledger line {line_no} logical KV capacity",
            )
            stream_id = kv_cache.get("lifecycle_stream_id")
            if not isinstance(stream_id, str) or not stream_id:
                raise AuditError(
                    f"ledger line {line_no} lifecycle_stream_id must be non-empty"
                )
            if contract_stream_id is None:
                contract_stream_id = stream_id
                contract_num_gpu_blocks = num_gpu_blocks
                contract_residency_plan = dict(residency_plan)
            else:
                _expect_equal(
                    stream_id,
                    contract_stream_id,
                    f"ledger line {line_no} lifecycle stream",
                )
                _expect_equal(
                    num_gpu_blocks,
                    contract_num_gpu_blocks,
                    f"ledger line {line_no} num_gpu_blocks",
                )
                _expect_equal(
                    residency_plan,
                    contract_residency_plan,
                    f"ledger line {line_no} residency plan",
                )

            lifecycle_cursor = _integer(
                contract.get("lifecycle_event_cursor"),
                f"ledger line {line_no} lifecycle_event_cursor",
            )
            if lifecycle_cursor < previous_lifecycle_cursor:
                raise AuditError(
                    "Frontier lifecycle cursors must be nondecreasing"
                )
            previous_lifecycle_cursor = lifecycle_cursor

            snapshots = contract.get("requests")
            if not isinstance(snapshots, list) or len(snapshots) != len(
                normalized_request_ids
            ):
                raise AuditError(
                    f"ledger line {line_no} contract request count differs from batch"
                )
            snapshot_ids = [
                str(_mapping(snapshot, "contract request").get("request_id"))
                for snapshot in snapshots
            ]
            _expect_equal(
                snapshot_ids,
                normalized_request_ids,
                f"ledger line {line_no} contract request order",
            )

            for position, (snapshot_value, request_id, scheduled_tokens) in enumerate(
                zip(snapshots, normalized_request_ids, normalized_request_tokens)
            ):
                snapshot = _mapping(
                    snapshot_value,
                    f"ledger line {line_no} contract requests[{position}]",
                )
                _expect_equal(
                    snapshot.get("runtime_epoch"),
                    0,
                    f"ledger line {line_no} request {request_id} runtime epoch",
                )
                _expect_equal(
                    snapshot.get("scheduled_tokens"),
                    scheduled_tokens,
                    f"ledger line {line_no} request {request_id} scheduled tokens",
                )
                request = requests_by_id[request_id]
                context_before = _integer(
                    snapshot.get("context_tokens_before"),
                    f"ledger line {line_no} request {request_id} context",
                )
                expected_context = next_context_by_request.get(
                    request_id,
                    request_metrics[request_id]["cached"],
                )
                _expect_equal(
                    context_before,
                    expected_context,
                    f"ledger line {line_no} request {request_id} context continuity",
                )
                expected_phase = (
                    "prefill"
                    if context_before < request["prefill_tokens"]
                    else "decode"
                )
                _expect_equal(
                    snapshot.get("phase"),
                    expected_phase,
                    f"ledger line {line_no} request {request_id} phase",
                )
                scheduler_frontier = context_before + scheduled_tokens
                _expect_equal(
                    snapshot.get("scheduler_token_frontier_after"),
                    scheduler_frontier,
                    f"ledger line {line_no} request {request_id} scheduler frontier",
                )
                if scheduler_frontier > request["total_tokens"]:
                    raise AuditError(
                        f"ledger line {line_no} request {request_id} exceeds total tokens"
                    )
                kv_before = (
                    context_before
                    if expected_phase == "prefill"
                    else context_before - 1
                )
                kv_after = kv_before + scheduled_tokens
                _expect_equal(
                    snapshot.get("kv_tokens_before"),
                    kv_before,
                    f"ledger line {line_no} request {request_id} KV frontier before",
                )
                _expect_equal(
                    snapshot.get("kv_tokens_after"),
                    kv_after,
                    f"ledger line {line_no} request {request_id} KV frontier after",
                )
                produces_output_token = snapshot.get("produces_output_token")
                if not isinstance(produces_output_token, bool):
                    raise AuditError(
                        f"ledger line {line_no} request {request_id} "
                        "produces_output_token must be boolean"
                    )
                expected_output = (
                    expected_phase == "decode"
                    or (
                        request["decode_tokens"] > 0
                        and scheduler_frontier == request["prefill_tokens"]
                    )
                )
                _expect_equal(
                    produces_output_token,
                    expected_output,
                    f"ledger line {line_no} request {request_id} output boundary",
                )
                _expect_equal(
                    snapshot.get("num_prefill_tokens"),
                    request["prefill_tokens"],
                    f"ledger line {line_no} request {request_id} prefill tokens",
                )
                _expect_equal(
                    snapshot.get("num_decode_tokens"),
                    request["decode_tokens"],
                    f"ledger line {line_no} request {request_id} decode tokens",
                )
                _expect_equal(
                    snapshot.get("cached_prefill_tokens"),
                    request_metrics[request_id]["cached"],
                    f"ledger line {line_no} request {request_id} cached tokens",
                )
                allocator_required_blocks = (
                    scheduler_frontier + 15
                ) // 16
                _expect_equal(
                    snapshot.get("allocator_required_blocks"),
                    allocator_required_blocks,
                    f"ledger line {line_no} request {request_id} allocator blocks",
                )
                _expect_equal(
                    _integer(
                        snapshot.get("allocated_block_count"),
                        f"ledger line {line_no} request {request_id} "
                        "allocated_block_count",
                    ),
                    allocator_required_blocks,
                    f"ledger line {line_no} request {request_id} allocated block count",
                )
                _sha256_string(
                    snapshot.get("allocator_state_sha256"),
                    f"ledger line {line_no} request {request_id} "
                    "allocator_state_sha256",
                )
                next_context = scheduler_frontier
                if (
                    expected_phase == "prefill"
                    and scheduler_frontier == request["prefill_tokens"]
                ):
                    next_context += 1
                next_context_by_request[request_id] = next_context
                final_frontier_by_request[request_id] = next_context
                final_kv_frontier_by_request[request_id] = kv_after

            contract["_audit_line_no"] = line_no
            contract["_audit_stage_start_s"] = start
            contract["_audit_stage_end_s"] = end
            contracts.append(contract)
            rows += 1
            batch_size_counts[len(request_ids)] = (
                batch_size_counts.get(len(request_ids), 0) + 1
            )
    if rows == 0:
        raise AuditError("Frontier stage ledger contains no rows")
    _expect_equal(batch_ids, set(range(rows)), "Frontier batch IDs")
    for request in requests:
        request_id = request["id"]
        _expect_equal(
            final_frontier_by_request.get(request_id),
            request["total_tokens"],
            f"request {request_id} final scheduler frontier",
        )
        _expect_equal(
            final_kv_frontier_by_request.get(request_id),
            request["total_tokens"] - 1,
            f"request {request_id} final KV frontier",
        )
    return scheduled, {
        "rows": rows,
        "first_stage_start_s": first_start,
        "last_stage_end_s": last_end,
        "stage_span_s": last_end - float(first_start),
        "batch_size_counts": {
            str(size): count for size, count in sorted(batch_size_counts.items())
        },
        "memory_contract_schema_version": MEMORY_CONTRACT_SCHEMA_VERSION,
        "kv_lifecycle_stream_id": contract_stream_id,
        "num_gpu_blocks": contract_num_gpu_blocks,
        "residency_plan": contract_residency_plan,
        "no_preemption_admission_policy": (
            NO_PREEMPTION_ADMISSION_POLICY
        ),
    }, contracts


def _read_kv_lifecycle(
    path: Path,
    contracts: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    request_metrics: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if not contracts:
        raise AuditError("cannot audit KV lifecycle without memory contracts")
    stream_id = contracts[0]["kv_cache"]["lifecycle_stream_id"]
    num_gpu_blocks = int(contracts[0]["kv_cache"]["num_gpu_blocks"])
    supported_events = {
        "prefix_lookup",
        "prefix_admission",
        "touch",
        "evict",
        "allocate",
        "cache_assign",
        "release",
    }
    events: list[dict[str, Any]] = []
    previous_time = -1.0
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise AuditError(
            f"cannot read Frontier KV lifecycle ledger: {path}: {error}"
        ) from error
    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise AuditError(
                    f"Frontier KV lifecycle ledger line {line_no} is blank"
                )
            try:
                event = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise AuditError(
                    f"invalid Frontier KV lifecycle JSON at line {line_no}: "
                    f"{error}"
                ) from error
            event = _mapping(event, f"KV lifecycle line {line_no}")
            _expect_equal(
                event.get("schema_version"),
                1,
                f"KV lifecycle line {line_no} schema",
            )
            _expect_equal(
                event.get("stream_id"),
                stream_id,
                f"KV lifecycle line {line_no} stream",
            )
            event_index = _integer(
                event.get("event_index"),
                f"KV lifecycle line {line_no} event_index",
            )
            _expect_equal(
                event_index,
                len(events),
                f"KV lifecycle line {line_no} contiguous event index",
            )
            _expect_equal(
                event.get("event_id"),
                f"{stream_id}:{event_index:012d}",
                f"KV lifecycle line {line_no} event_id",
            )
            event_time = _finite_number(
                event.get("event_time"),
                f"KV lifecycle line {line_no} event_time",
            )
            if event_time < previous_time:
                raise AuditError(
                    "Frontier KV lifecycle event times must be nondecreasing"
                )
            previous_time = event_time
            if event.get("event") not in supported_events:
                raise AuditError(
                    f"KV lifecycle line {line_no} has unsupported event "
                    f"{event.get('event')!r}"
                )
            _expect_equal(
                event.get("cluster_type"),
                "MONOLITHIC",
                f"KV lifecycle line {line_no} cluster",
            )
            _expect_equal(
                event.get("replica_id"),
                0,
                f"KV lifecycle line {line_no} replica",
            )
            _expect_equal(
                event.get("dp_id"),
                0,
                f"KV lifecycle line {line_no} dp",
            )
            observed_after_batch_id = _integer(
                event.get("observed_after_batch_id"),
                f"KV lifecycle line {line_no} observed_after_batch_id",
            )
            if observed_after_batch_id >= len(contracts):
                raise AuditError(
                    f"KV lifecycle line {line_no} observes unknown batch "
                    f"{observed_after_batch_id}"
                )
            events.append(event)
    if not events:
        raise AuditError("Frontier KV lifecycle ledger contains no events")

    previous_contract_cursor = 0
    for contract in contracts:
        contract_cursor = int(contract["lifecycle_event_cursor"])
        if contract_cursor > len(events):
            raise AuditError(
                f"batch {contract['batch_id']} lifecycle cursor exceeds ledger"
            )
        for event in events[previous_contract_cursor:contract_cursor]:
            if event["event"] == "release":
                if int(event["observed_after_batch_id"]) >= int(
                    contract["batch_id"]
                ):
                    raise AuditError(
                        f"batch {contract['batch_id']} cursor includes release "
                        f"{event['event_id']} that is not from an earlier batch"
                    )
                continue
            _expect_equal(
                event.get("observed_after_batch_id"),
                contract["batch_id"],
                f"batch {contract['batch_id']} lifecycle event "
                f"{event['event_id']} observation",
            )
            _expect_equal(
                event.get("event_time"),
                contract["captured_at"],
                f"batch {contract['batch_id']} lifecycle event "
                f"{event['event_id']} time",
            )
        previous_contract_cursor = contract_cursor

    if any(event["event"] != "release" for event in events[previous_contract_cursor:]):
        raise AuditError(
            "non-release lifecycle events remain beyond the final batch cursor"
        )
    for event in events:
        if event["event"] != "release":
            continue
        observed_after_batch_id = int(event["observed_after_batch_id"])
        _expect_equal(
            event["event_time"],
            contracts[observed_after_batch_id]["_audit_stage_end_s"],
            f"lifecycle event {event['event_id']} release time",
        )

    block_ref_counts = [0] * num_gpu_blocks
    block_hashes: list[int | None] = [None] * num_gpu_blocks
    request_blocks: dict[str, list[int]] = {}
    requests_by_id = {request["id"]: request for request in requests}
    expected_request_ids = set(requests_by_id)
    prefix_lookup_blocks: dict[str, list[int]] = {}
    prefix_admitted_blocks: dict[str, list[int]] = {}
    touched_requests: set[str] = set()
    released_requests: set[str] = set()
    event_counts = {event_type: 0 for event_type in sorted(supported_events)}
    max_reserved_blocks = 0
    # Verified ref-count transitions are the authoritative live-set source.
    # Maintaining their zero/nonzero crossings avoids rescanning the complete
    # physical block pool at every production batch checkpoint.
    referenced_physical_blocks = 0
    max_referenced_physical_blocks = 0
    reservation_checkpoints = 0

    def checked_block_id(value: Any, path_text: str) -> int:
        block_id = _integer(value, path_text)
        if block_id >= num_gpu_blocks:
            raise AuditError(
                f"{path_text}={block_id} exceeds num_gpu_blocks={num_gpu_blocks}"
            )
        return block_id

    def apply_event(event: dict[str, Any]) -> None:
        nonlocal referenced_physical_blocks
        event_type = str(event["event"])
        event_counts[event_type] += 1
        request_id_value = event.get("request_id")
        request_id = (
            None if request_id_value is None else str(request_id_value)
        )
        if request_id is not None and request_id not in expected_request_ids:
            raise AuditError(
                f"lifecycle event {event['event_id']} references unknown "
                f"request {request_id}"
            )
        if event_type == "prefix_lookup":
            if request_id is None:
                raise AuditError("prefix_lookup event has no request_id")
            if request_id in prefix_lookup_blocks:
                raise AuditError(
                    f"request {request_id} has multiple prefix_lookup events"
                )
            requested_hashes = event.get("requested_block_hashes")
            matched_blocks = event.get("matched_blocks")
            if not isinstance(requested_hashes, list) or not isinstance(
                matched_blocks, list
            ):
                raise AuditError("prefix_lookup event has malformed block lists")
            request = requests_by_id[request_id]
            cacheable_hashes = request["block_hash_ids"][
                : request["prefill_tokens"] // 16
            ]
            _expect_equal(
                requested_hashes,
                cacheable_hashes,
                f"lifecycle event {event['event_id']} requested complete-block hashes",
            )
            if len(matched_blocks) > len(requested_hashes):
                raise AuditError(
                    f"lifecycle event {event['event_id']} matches more blocks "
                    "than requested"
                )
            expected_hit_tokens = len(matched_blocks) * 16
            _expect_equal(
                event.get("hit_tokens"),
                expected_hit_tokens,
                f"lifecycle event {event['event_id']} hit_tokens",
            )
            for position, matched_value in enumerate(matched_blocks):
                matched = _mapping(
                    matched_value,
                    f"lifecycle event {event['event_id']} matched block {position}",
                )
                block_id = checked_block_id(
                    matched.get("block_id"),
                    f"lifecycle event {event['event_id']} block_id",
                )
                _expect_equal(
                    matched.get("block_hash"),
                    requested_hashes[position],
                    f"lifecycle event {event['event_id']} prefix hash order",
                )
                _expect_equal(
                    matched.get("block_hash"),
                    block_hashes[block_id],
                    f"lifecycle event {event['event_id']} matched hash",
                )
                _expect_equal(
                    matched.get("ref_count"),
                    block_ref_counts[block_id],
                    f"lifecycle event {event['event_id']} matched ref_count",
                )
            matched_ids = [
                int(_mapping(value, "matched block")["block_id"])
                for value in matched_blocks
            ]
            if len(matched_ids) != len(set(matched_ids)):
                raise AuditError(
                    f"lifecycle event {event['event_id']} repeats a matched block"
                )
            prefix_lookup_blocks[request_id] = matched_ids
            return
        if event_type == "prefix_admission":
            if request_id is None:
                raise AuditError("prefix_admission event has no request_id")
            if request_id not in prefix_lookup_blocks:
                raise AuditError(
                    f"request {request_id} prefix_admission precedes lookup"
                )
            if request_id in prefix_admitted_blocks:
                raise AuditError(
                    f"request {request_id} has multiple prefix_admission events"
                )
            admitted_blocks = event.get("admitted_blocks")
            if not isinstance(admitted_blocks, list):
                raise AuditError("prefix_admission has malformed admitted_blocks")
            _expect_equal(
                event.get("admitted_tokens"),
                len(admitted_blocks) * 16,
                f"lifecycle event {event['event_id']} admitted_tokens",
            )
            for admitted_value in admitted_blocks:
                admitted = _mapping(admitted_value, "prefix admitted block")
                block_id = checked_block_id(
                    admitted.get("block_id"), "prefix admitted block_id"
                )
                _expect_equal(
                    admitted.get("block_hash"),
                    block_hashes[block_id],
                    f"lifecycle event {event['event_id']} admitted hash",
                )
                _expect_equal(
                    admitted.get("ref_count"),
                    block_ref_counts[block_id],
                    f"lifecycle event {event['event_id']} admitted ref_count",
                )
            admitted_ids = [
                int(_mapping(value, "prefix admitted block")["block_id"])
                for value in admitted_blocks
            ]
            matched_ids = prefix_lookup_blocks[request_id]
            expected_admitted_ids = list(matched_ids)
            if (
                len(matched_ids) * 16
                == requests_by_id[request_id]["prefill_tokens"]
                and matched_ids
            ):
                expected_admitted_ids.pop()
            _expect_equal(
                admitted_ids,
                expected_admitted_ids,
                f"lifecycle event {event['event_id']} admitted block sequence",
            )
            _expect_equal(
                event.get("admitted_tokens"),
                request_metrics[request_id]["cached"],
                f"lifecycle event {event['event_id']} cached prefill tokens",
            )
            prefix_admitted_blocks[request_id] = admitted_ids
            return
        if event_type == "evict":
            block_id = checked_block_id(
                event.get("block_id"),
                f"lifecycle event {event['event_id']} block_id",
            )
            _expect_equal(
                block_ref_counts[block_id],
                0,
                f"lifecycle event {event['event_id']} evicted ref_count",
            )
            _expect_equal(
                event.get("evicted_block_hash"),
                block_hashes[block_id],
                f"lifecycle event {event['event_id']} evicted hash",
            )
            if block_hashes[block_id] is None:
                raise AuditError(
                    f"lifecycle event {event['event_id']} evicts an uncached block"
                )
            block_hashes[block_id] = None
            return
        if event_type == "allocate":
            if request_id is None:
                raise AuditError("allocate event has no request_id")
            if request_id not in prefix_admitted_blocks:
                raise AuditError(
                    f"request {request_id} allocation precedes prefix admission"
                )
            block_id = checked_block_id(
                event.get("block_id"),
                f"lifecycle event {event['event_id']} block_id",
            )
            _expect_equal(
                event.get("ref_count_before"),
                block_ref_counts[block_id],
                f"lifecycle event {event['event_id']} ref_count_before",
            )
            _expect_equal(
                block_ref_counts[block_id],
                0,
                f"lifecycle event {event['event_id']} allocate free block",
            )
            _expect_equal(
                block_hashes[block_id],
                None,
                f"lifecycle event {event['event_id']} allocate uncached block",
            )
            _expect_equal(
                event.get("ref_count_after"),
                1,
                f"lifecycle event {event['event_id']} ref_count_after",
            )
            if any(
                block_id in owned_blocks
                for owned_blocks in request_blocks.values()
            ):
                raise AuditError(
                    f"lifecycle event {event['event_id']} allocates an owned block"
                )
            block_ref_counts[block_id] = 1
            referenced_physical_blocks += 1
            request_blocks.setdefault(request_id, []).append(block_id)
            return
        if event_type == "cache_assign":
            if request_id is None:
                raise AuditError("cache_assign event has no request_id")
            block_id = checked_block_id(
                event.get("block_id"),
                f"lifecycle event {event['event_id']} block_id",
            )
            if block_ref_counts[block_id] <= 0 or block_hashes[block_id] is not None:
                raise AuditError(
                    f"lifecycle event {event['event_id']} assigns an invalid block"
                )
            owned_blocks = request_blocks.get(request_id, [])
            if block_id not in owned_blocks:
                raise AuditError(
                    f"lifecycle event {event['event_id']} assigns a block not "
                    "owned by the request"
                )
            block_ordinal = owned_blocks.index(block_id)
            request = requests_by_id[request_id]
            expected_hashes = request["block_hash_ids"][
                : request["prefill_tokens"] // 16
            ]
            if block_ordinal >= len(expected_hashes):
                raise AuditError(
                    f"lifecycle event {event['event_id']} assigns a block "
                    "outside the request prefix"
                )
            assigned_hash = _integer(
                event.get("block_hash"),
                f"lifecycle event {event['event_id']} block_hash",
            )
            _expect_equal(
                assigned_hash,
                expected_hashes[block_ordinal],
                f"lifecycle event {event['event_id']} request prefix hash",
            )
            block_hashes[block_id] = assigned_hash
            return
        transitions = event.get("blocks")
        if not isinstance(transitions, list) or not transitions:
            raise AuditError(
                f"lifecycle event {event['event_id']} has no block transitions"
            )
        if request_id is None:
            raise AuditError(
                f"lifecycle event {event['event_id']} has no request_id"
            )
        transition_ids: list[int] = []
        for transition_value in transitions:
            transition = _mapping(transition_value, "block ref transition")
            block_id = checked_block_id(
                transition.get("block_id"),
                f"lifecycle event {event['event_id']} block_id",
            )
            transition_ids.append(block_id)
            _expect_equal(
                transition.get("block_hash"),
                block_hashes[block_id],
                f"lifecycle event {event['event_id']} block_hash",
            )
            ref_count_before = block_ref_counts[block_id]
            _expect_equal(
                transition.get("ref_count_before"),
                ref_count_before,
                f"lifecycle event {event['event_id']} ref_count_before",
            )
            expected_after = (
                ref_count_before + 1
                if event_type == "touch"
                else ref_count_before - 1
            )
            if expected_after < 0:
                raise AuditError(
                    f"lifecycle event {event['event_id']} underflows ref_count"
                )
            _expect_equal(
                transition.get("ref_count_after"),
                expected_after,
                f"lifecycle event {event['event_id']} ref_count_after",
            )
            if event_type == "release":
                _expect_equal(
                    transition.get("became_free"),
                    expected_after == 0,
                    f"lifecycle event {event['event_id']} became_free",
                )
            block_ref_counts[block_id] = expected_after
            if ref_count_before == 0 and expected_after > 0:
                referenced_physical_blocks += 1
            elif ref_count_before > 0 and expected_after == 0:
                referenced_physical_blocks -= 1
            if referenced_physical_blocks < 0:
                raise AuditError(
                    "incremental referenced-block count underflowed"
                )
        if len(transition_ids) != len(set(transition_ids)):
            raise AuditError(
                f"lifecycle event {event['event_id']} repeats a block"
            )
        if event_type == "touch":
            if request_id not in prefix_admitted_blocks:
                raise AuditError(
                    f"request {request_id} touch precedes prefix admission"
                )
            if request_id in touched_requests:
                raise AuditError(
                    f"request {request_id} has multiple touch events"
                )
            _expect_equal(
                transition_ids,
                prefix_admitted_blocks[request_id],
                f"lifecycle event {event['event_id']} admitted touch sequence",
            )
            existing = request_blocks.setdefault(request_id, [])
            if any(block_id in existing for block_id in transition_ids):
                raise AuditError(
                    f"lifecycle event {event['event_id']} touches a block "
                    "already owned by the request"
                )
            existing.extend(transition_ids)
            touched_requests.add(request_id)
            return
        if request_id in released_requests:
            raise AuditError(f"request {request_id} has multiple release events")
        current = request_blocks.get(request_id)
        if current is None or set(current) != set(transition_ids):
            raise AuditError(
                f"lifecycle event {event['event_id']} release set does not "
                "match request ownership"
            )
        del request_blocks[request_id]
        released_requests.add(request_id)

    next_event_index = 0
    for contract in contracts:
        cursor = int(contract["lifecycle_event_cursor"])
        if cursor > len(events):
            raise AuditError(
                f"batch {contract['batch_id']} lifecycle cursor exceeds ledger"
            )
        while next_event_index < cursor:
            apply_event(events[next_event_index])
            next_event_index += 1
        reserved_blocks = sum(
            (
                requests_by_id[request_id]["total_tokens"]
                - 1
                + 15
            )
            // 16
            for request_id in request_blocks
        )
        if reserved_blocks > num_gpu_blocks:
            raise AuditError(
                f"batch {contract['batch_id']} violates "
                f"{NO_PREEMPTION_ADMISSION_POLICY}: "
                f"reserved_blocks={reserved_blocks}, "
                f"num_gpu_blocks={num_gpu_blocks}"
            )
        if referenced_physical_blocks > num_gpu_blocks:
            raise AuditError(
                f"batch {contract['batch_id']} references more physical KV "
                "blocks than the block pool contains"
            )
        max_reserved_blocks = max(max_reserved_blocks, reserved_blocks)
        max_referenced_physical_blocks = max(
            max_referenced_physical_blocks,
            referenced_physical_blocks,
        )
        reservation_checkpoints += 1
        for snapshot_value in contract["requests"]:
            snapshot = _mapping(snapshot_value, "contract request snapshot")
            request_id = str(snapshot["request_id"])
            owned = request_blocks.get(request_id, [])
            _expect_equal(
                snapshot.get("allocated_block_count"),
                len(owned),
                f"batch {contract['batch_id']} request {request_id} "
                "replayed block count",
            )
            replayed_blocks = [
                {
                    "block_id": block_id,
                    "block_hash": block_hashes[block_id],
                    "ref_count": block_ref_counts[block_id],
                }
                for block_id in owned
            ]
            try:
                replayed_digest = allocator_state_sha256(
                    request_id,
                    replayed_blocks,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise AuditError(
                    f"batch {contract['batch_id']} request {request_id} "
                    f"cannot canonicalize replayed allocator state: {error}"
                ) from error
            _expect_equal(
                snapshot.get("allocator_state_sha256"),
                replayed_digest,
                f"batch {contract['batch_id']} request {request_id} "
                "allocator-state digest",
            )
    while next_event_index < len(events):
        apply_event(events[next_event_index])
        next_event_index += 1
    _expect_equal(
        set(prefix_lookup_blocks),
        expected_request_ids,
        "requests with prefix_lookup lifecycle events",
    )
    _expect_equal(
        set(prefix_admitted_blocks),
        expected_request_ids,
        "requests with prefix_admission lifecycle events",
    )
    _expect_equal(
        touched_requests,
        {
            request_id
            for request_id, admitted_blocks in prefix_admitted_blocks.items()
            if admitted_blocks
        },
        "requests with prefix-hit touch lifecycle events",
    )
    _expect_equal(
        released_requests,
        expected_request_ids,
        "requests with release lifecycle events",
    )
    if request_blocks:
        raise AuditError(
            "Frontier KV lifecycle ends with live request ownership: "
            + ", ".join(sorted(request_blocks))
        )
    if any(ref_count != 0 for ref_count in block_ref_counts):
        raise AuditError("Frontier KV lifecycle ends with nonzero block references")
    _expect_equal(
        referenced_physical_blocks,
        0,
        "incremental referenced physical block count",
    )

    return {
        "schema_version": 1,
        "events": len(events),
        "event_counts": event_counts,
        "stream_id": stream_id,
        "num_gpu_blocks": num_gpu_blocks,
        "final_cached_blocks": sum(
            block_hash is not None for block_hash in block_hashes
        ),
        "final_referenced_blocks": 0,
        "cursor_checkpoints": len(contracts),
        "no_preemption_admission": {
            "policy": NO_PREEMPTION_ADMISSION_POLICY,
            "verified_checkpoints": reservation_checkpoints,
            "max_reserved_blocks": max_reserved_blocks,
            "max_referenced_physical_blocks": (
                max_referenced_physical_blocks
            ),
            "capacity_blocks": num_gpu_blocks,
        },
    }


def audit_replay(
    *,
    request_csv: Path,
    request_manifest: Path,
    request_suite_verification: Path,
    frontier_output_dir: Path,
    frontier_revision: str,
    output: Path,
    frontier_repository: str = FRONTIER_REPOSITORY,
    frontier_integration_manifest: Path = DEFAULT_FRONTIER_INTEGRATION_MANIFEST,
    frontier_integration_receipt: Path,
) -> dict[str, Any]:
    request_csv = request_csv.resolve()
    request_manifest = request_manifest.resolve()
    request_suite_verification = request_suite_verification.resolve()
    frontier_output_dir = frontier_output_dir.resolve()
    frontier_integration_manifest = frontier_integration_manifest.resolve()
    frontier_integration_receipt = frontier_integration_receipt.resolve()
    output = output.resolve()
    if not request_csv.is_file():
        raise AuditError(f"request CSV does not exist: {request_csv}")
    if not request_manifest.is_file():
        raise AuditError(f"request manifest does not exist: {request_manifest}")
    if not request_suite_verification.is_file():
        raise AuditError(
            "request-suite verification receipt does not exist: "
            f"{request_suite_verification}"
        )
    if not frontier_output_dir.is_dir():
        raise AuditError(f"Frontier output directory does not exist: {frontier_output_dir}")
    if not frontier_repository.strip():
        raise AuditError("frontier_repository must be non-empty")
    protected_inputs = {
        request_csv,
        request_manifest,
        request_suite_verification,
        frontier_integration_manifest,
        frontier_integration_receipt,
    }
    if output in protected_inputs or output.parent == frontier_output_dir and output.name in {
        "config.json",
        "system_metrics.json",
        "request_metrics.csv",
        "frontier_stage_batch_ledger.jsonl",
        "frontier_kv_block_lifecycle.jsonl",
        "frontier_hbfsim_integration.json",
    }:
        raise AuditError("audit output must not overwrite an input artifact")

    paths = {
        "config": frontier_output_dir / "config.json",
        "system_metrics": frontier_output_dir / "system_metrics.json",
        "request_metrics": frontier_output_dir / "request_metrics.csv",
        "stage_batch_ledger": frontier_output_dir / "frontier_stage_batch_ledger.jsonl",
        "kv_block_lifecycle": (
            frontier_output_dir / "frontier_kv_block_lifecycle.jsonl"
        ),
        "runtime_integration": (
            frontier_output_dir / "frontier_hbfsim_integration.json"
        ),
    }
    for description, path in paths.items():
        if not path.is_file():
            raise AuditError(f"missing Frontier {description}: {path}")
    if not frontier_integration_manifest.is_file():
        raise AuditError(
            "missing Frontier integration bundle manifest: "
            f"{frontier_integration_manifest}"
        )
    if not frontier_integration_receipt.is_file():
        raise AuditError(
            "missing Frontier integration receipt: "
            f"{frontier_integration_receipt}"
        )
    integration, integration_patch_path = _validate_frontier_integration(
        bundle_manifest_path=frontier_integration_manifest,
        receipt_path=frontier_integration_receipt,
        runtime_path=paths["runtime_integration"],
        frontier_repository=frontier_repository,
        frontier_revision=frontier_revision,
    )
    if output == integration_patch_path:
        raise AuditError("audit output must not overwrite the integration patch")

    requests = _read_requests(request_csv)
    adapter_manifest = _load_object(request_manifest, "request manifest")
    adapter = _validate_adapter_manifest(adapter_manifest, request_csv, requests)
    request_suite, request_suite_artifacts = _validate_request_suite_verification(
        verification_path=request_suite_verification,
        request_csv=request_csv,
        request_manifest_path=request_manifest,
        adapter=adapter,
    )
    protected_inputs.update(request_suite_artifacts.values())
    if output in protected_inputs:
        raise AuditError("audit output must not overwrite an input artifact")
    config = _load_object(paths["config"], "Frontier config")
    frontend, dummy_timing = _validate_frontier_config(
        config,
        request_csv,
        adapter["frontier_workload"],
    )
    _expect_equal(
        frontend["model"],
        adapter["frontier_workload"]["model"]["frontier_name"],
        "Frontier/request-suite model",
    )
    max_position_embeddings = int(
        frontend["model_config"]["max_position_embeddings"]
    )
    for request in requests:
        if request["total_tokens"] > max_position_embeddings:
            raise AuditError(
                f"request {request['id']} has {request['total_tokens']} tokens, "
                "which exceeds Frontier max_position_embeddings="
                f"{max_position_embeddings}"
            )
    request_metrics, metric_totals = _read_request_metrics(
        paths["request_metrics"], requests
    )
    scheduled, ledger, contracts = _read_ledger(
        paths["stage_batch_ledger"],
        requests,
        request_metrics,
        expected_caching_hash_algo=frontend["scheduler"][
            "prefix_caching_hash_algo"
        ],
    )
    kv_lifecycle = _read_kv_lifecycle(
        paths["kv_block_lifecycle"],
        contracts,
        requests,
        request_metrics,
    )

    for request in requests:
        request_id = request["id"]
        cached = request_metrics[request_id]["cached"]
        first_decode_token = 1 if request["decode_tokens"] > 0 else 0
        conserved = scheduled[request_id] + cached + first_decode_token
        _expect_equal(conserved, request["total_tokens"], f"request {request_id} token conservation")

    total_prefill = sum(request["prefill_tokens"] for request in requests)
    total_decode = sum(request["decode_tokens"] for request in requests)
    total_tokens = total_prefill + total_decode
    total_scheduled = sum(scheduled.values())
    system = _load_object(paths["system_metrics"], "Frontier system metrics")
    model_memory = _validate_model_memory_ledger(system, frontend)
    model_config = frontend["model_config"]
    num_kv_heads = _integer(
        model_config.get("num_kv_heads"),
        "Frontier model num_kv_heads",
        minimum=1,
    )
    head_dim = _integer(
        model_config.get("head_dim"),
        "Frontier model head_dim",
        minimum=1,
    )
    num_layers = _integer(
        model_config.get("num_layers"),
        "Frontier model num_layers",
        minimum=1,
    )
    block_size_tokens = int(
        frontend["scheduler"]["block_size_tokens"]
    )
    kv_bytes = int(
        frontend["memory_precision"]["selected_profile"]["kv_bytes"]
    )
    kv_page_bytes_per_layer = (
        block_size_tokens
        * 2
        * num_kv_heads
        * head_dim
        * kv_bytes
    )
    try:
        residency_plan = validate_hybrid_residency_plan(
            ledger.get("residency_plan"),
            physical_hbm_capacity_bytes=int(
                frontend["scheduler"][
                    "physical_hbm_capacity_bytes"
                ]
            ),
            target_pressure=float(
                frontend["scheduler"]["capacity_pressure_target"]
            ),
            immutable_weight_backing_bytes=int(
                model_memory["total_memory_bytes"]
            ),
            runtime_overhead_bytes=int(
                frontend["scheduler"]["runtime_overhead_bytes"]
            ),
            active_weight_buffer_bytes_per_slot=int(
                model_memory["weight_streaming"][
                    "active_buffer_bytes_per_slot"
                ]
            ),
            kv_block_size_tokens=block_size_tokens,
            kv_page_bytes_per_layer=kv_page_bytes_per_layer,
            num_layers=num_layers,
        )
    except HybridResidencyError as error:
        raise AuditError(
            f"Frontier hybrid-residency plan is invalid: {error}"
        ) from error
    _expect_equal(
        ledger.get("num_gpu_blocks"),
        residency_plan["num_logical_kv_blocks"],
        "Frontier allocator/hybrid logical block capacity",
    )
    metadata = _mapping(system.get("simulation_metadata"), "system simulation_metadata")
    _expect_equal(metadata.get("total_requests"), len(requests), "system total_requests")
    _expect_equal(
        metadata.get("completed_requests"), len(requests), "system completed_requests"
    )
    throughput = _mapping(system.get("throughput_metrics"), "system throughput_metrics")
    _expect_equal(
        throughput.get("total_tokens_processed"), total_tokens, "system total tokens"
    )
    _expect_equal(
        throughput.get("total_decode_tokens_generated"),
        total_decode,
        "system decode tokens",
    )
    cache = _mapping(system.get("prefix_cache_statistics"), "system prefix cache")
    _expect_equal(cache.get("requests"), len(requests), "cache requests")
    _expect_equal(
        cache.get("total_cached_prefill_tokens"),
        metric_totals["cached_tokens"],
        "cache cached tokens",
    )
    _expect_equal(
        cache.get("total_query_blocks"),
        metric_totals["query_blocks"],
        "cache query blocks",
    )
    _expect_equal(
        cache.get("total_hit_blocks"),
        metric_totals["hit_blocks"],
        "cache hit blocks",
    )
    preemption = _mapping(
        system.get("preemption_statistics"), "system preemption statistics"
    )
    _expect_equal(
        preemption.get("total_preemption_events"), 0, "preemption events"
    )
    _expect_equal(
        preemption.get("total_preempted_requests"), 0, "preempted requests"
    )
    speculative = _mapping(
        system.get("spec_decode_statistics"), "system speculative decode statistics"
    )
    _expect_equal(
        speculative.get("total_iterations"), 0, "speculative decode iterations"
    )

    artifact_paths = {
        "request_csv": request_csv,
        "request_manifest": request_manifest,
        **request_suite_artifacts,
        "integration_bundle_manifest": frontier_integration_manifest,
        "integration_receipt": frontier_integration_receipt,
        "integration_patch": integration_patch_path,
        **paths,
    }
    payload = {
        "schema": {"name": AUDIT_SCHEMA_NAME, "version": AUDIT_SCHEMA_VERSION},
        "result": "pass",
        "boundary": {
            "input": "qwen_bailian_serving_requests",
            "output": "frontier_scheduler_batch_ledger",
            "output_is_hbfsim_memory_trace": False,
        },
        "request_suite": request_suite,
        "eligibility": {
            "production_request_window_verified": True,
            "scheduler_ledger_valid": True,
            "memory_object_contract_valid": True,
            "kv_block_lifecycle_valid": True,
            "hybrid_residency_plan_valid": True,
            "capacity_pressure_basis_valid": True,
            "hardware_capacity_calibrated": False,
            "frontier_timing_kind": (
                "dummy" if dummy_timing else "profiled_unvalidated"
            ),
            # A profile-backed run is not automatically calibrated. Frontier's
            # output does not carry independent prediction-error evidence, so
            # this audit must not promote it to a performance baseline merely
            # because dummy mode is off.
            "frontier_timing_usable_for_performance": False,
            "eligible_for_structural_memory_object_export": True,
            "eligible_for_timed_memory_object_export": False,
            "eligible_for_memory_system_service_claims": True,
            "eligible_for_ttft_tpot_slo_claims": False,
            "eligible_for_time_based_throughput_claims": False,
            "timing_blocker": (
                "dummy execution-time predictor"
                if dummy_timing
                else "no independent profile-calibration artifact"
            ),
        },
        "frontier": {
            "repository": frontier_repository,
            "revision": frontier_revision,
            "integration": integration,
            **frontend,
        },
        "model_memory": model_memory,
        "residency_plan": residency_plan,
        "requests": {
            "count": len(requests),
            "prefill_tokens": total_prefill,
            "decode_tokens": total_decode,
            "total_tokens": total_tokens,
        },
        "accounting": {
            "scheduled_tokens": total_scheduled,
            "cached_prefill_tokens": metric_totals["cached_tokens"],
            "prefill_generated_first_tokens": len(requests),
            "identity": (
                "total_tokens = scheduled_tokens + cached_prefill_tokens + "
                "one prefill-generated first token per request"
            ),
            "prefix_query_blocks": metric_totals["query_blocks"],
            "prefix_hit_blocks": metric_totals["hit_blocks"],
            "prefix_hit_ratio": (
                metric_totals["hit_blocks"] / metric_totals["query_blocks"]
                if metric_totals["query_blocks"]
                else 0.0
            ),
        },
        "ledger": ledger,
        "kv_lifecycle": kv_lifecycle,
        "artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for name, path in artifact_paths.items()
        },
    }

    handle, temporary = _open_atomic_text(output)
    try:
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
        handle.close()
        os.replace(temporary, output)
    except Exception:
        if not handle.closed:
            handle.close()
        temporary.unlink(missing_ok=True)
        raise
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-csv", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument(
        "--request-suite-verification",
        type=Path,
        required=True,
    )
    parser.add_argument("--frontier-output-dir", type=Path, required=True)
    parser.add_argument("--frontier-revision", required=True)
    parser.add_argument("--frontier-repository", default=FRONTIER_REPOSITORY)
    parser.add_argument(
        "--frontier-integration-manifest",
        type=Path,
        default=DEFAULT_FRONTIER_INTEGRATION_MANIFEST,
    )
    parser.add_argument(
        "--frontier-integration-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = audit_replay(
            request_csv=args.request_csv,
            request_manifest=args.request_manifest,
            request_suite_verification=args.request_suite_verification,
            frontier_output_dir=args.frontier_output_dir,
            frontier_revision=args.frontier_revision,
            frontier_repository=args.frontier_repository,
            frontier_integration_manifest=args.frontier_integration_manifest,
            frontier_integration_receipt=args.frontier_integration_receipt,
            output=args.output,
        )
    except (AuditError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    eligibility = result["eligibility"]
    print(
        f"audited {result['requests']['count']} requests: "
        f"scheduled_tokens={result['accounting']['scheduled_tokens']} "
        f"prefix_hit_ratio={result['accounting']['prefix_hit_ratio']:.6f} "
        f"timing={eligibility['frontier_timing_kind']}"
    )
    print(f"audit={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
