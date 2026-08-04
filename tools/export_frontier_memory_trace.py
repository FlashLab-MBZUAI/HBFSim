#!/usr/bin/env python3
"""Export an audited Frontier replay as a deterministic HBFSim memory trace.

This exporter is intentionally strict.  Version 4 supports one co-located
dense transformer replica with PP=DP=TP=1, prefix caching, no preemption, and
no speculative decoding.  Frontier's physical KV block IDs become finite,
reusable logical slots; the exporter never invents a second KV allocator.

The generated traffic is model-conditioned, not measured GPU traffic.  Weight,
KV, embedding, output-head, metadata, and intentionally excluded scratch
traffic are all documented in the digest-bound manifest.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, TextIO

from frontier_memory_contract import (
    MEMORY_CONTRACT_SCHEMA_VERSION,
    allocator_state_sha256,
)
from frontier_hybrid_residency import (
    HybridResidencyError,
    validate_hybrid_residency_plan,
)


AUDIT_SCHEMA_NAME = "hbfsim.frontier_replay_audit"
AUDIT_SCHEMA_VERSION = 7
MODEL_SCHEMA_NAME = "hbfsim.frontier_dense_model_memory"
MODEL_SCHEMA_VERSION = 2
TRACE_SCHEMA_NAME = "hbfsim.frontier_memory_trace"
TRACE_SCHEMA_VERSION = 4
OBJECT_MAP_SCHEMA_NAME = "hbfsim.memory_object_map"
PHASE_MAP_SCHEMA_NAME = "hbfsim.frontier_phase_map"
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
UINT64_MAX = (1 << 64) - 1


class ExportError(ValueError):
    """An input cannot be exported without violating the current contract."""


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
        raise ExportError(f"cannot read {description}: {path}: {error}") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise ExportError(
            f"invalid JSON in {description}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ExportError(f"{description} must be a JSON object: {path}")
    return value


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExportError(f"{path} must be an object")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExportError(f"{path} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ExportError(f"{path} must be boolean")
    return value


def _finite_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ExportError(f"{path} must be a finite number")
    return float(value)


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"{path} must be a non-empty string")
    return value.strip()


def _sha256_string(value: Any, path: str) -> str:
    result = _nonempty_string(value, path)
    if (
        len(result) != 64
        or result != result.lower()
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ExportError(f"{path} must be 64 lowercase hex digits")
    return result


def _revision_string(value: Any, path: str) -> str:
    result = _nonempty_string(value, path)
    if (
        len(result) != 40
        or result != result.lower()
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ExportError(f"{path} must be a full lowercase git SHA")
    return result


def _validate_precision_profile(
    value: Any,
    path: str,
) -> dict[str, Any]:
    profile = _mapping(value, path)
    if set(profile) != MEMORY_PRECISION_PROFILE_FIELDS:
        raise ExportError(
            f"{path} fields mismatch: "
            f"missing={sorted(MEMORY_PRECISION_PROFILE_FIELDS - set(profile))}, "
            f"extra={sorted(set(profile) - MEMORY_PRECISION_PROFILE_FIELDS)}"
        )
    _nonempty_string(profile.get("profile_id"), f"{path}.profile_id")
    for prefix in ("matrix_weight", "non_matrix_weight", "kv", "activation"):
        dtype = profile.get(f"{prefix}_dtype")
        if dtype not in SUPPORTED_DTYPE_BYTES:
            raise ExportError(
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
        f"{path}.zero_point_bytes",
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
            raise ExportError(
                f"{path}.scale_dtype is unsupported: {scale_dtype!r}"
            )
        _expect_equal(
            scale_bytes,
            SUPPORTED_DTYPE_BYTES[scale_dtype],
            f"{path}.scale dtype/bytes",
        )
    else:
        raise ExportError(
            f"{path}.quantization_scheme must be 'none' or "
            "'symmetric_per_output_channel'"
        )
    _expect_equal(
        profile.get("claim_scope"),
        "memory_storage_and_traffic_only",
        f"{path}.claim_scope",
    )
    return dict(profile)


def _expect_equal(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise ExportError(f"{path}: expected {expected!r}, got {actual!r}")


def _checked_add(lhs: int, rhs: int, path: str) -> int:
    result = lhs + rhs
    if lhs < 0 or rhs < 0 or result > UINT64_MAX:
        raise ExportError(f"{path} overflows uint64")
    return result


def _checked_mul(lhs: int, rhs: int, path: str) -> int:
    if lhs < 0 or rhs < 0 or (lhs and rhs > UINT64_MAX // lhs):
        raise ExportError(f"{path} overflows uint64")
    return lhs * rhs


def _align_up(value: int, alignment: int, path: str) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ExportError(f"{path} alignment must be a positive power of two")
    return _checked_mul(
        (value + alignment - 1) // alignment,
        alignment,
        path,
    )


def _open_temporary(path: Path) -> tuple[TextIO, Path]:
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


def _write_json_temporary(path: Path, value: Any) -> Path:
    handle, temporary = _open_temporary(path)
    try:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        return temporary
    except BaseException:
        if not handle.closed:
            handle.close()
        temporary.unlink(missing_ok=True)
        raise


def _artifact_from_audit(
    audit: dict[str, Any], name: str
) -> tuple[Path, dict[str, Any]]:
    artifacts = _mapping(audit.get("artifacts"), "audit.artifacts")
    artifact = _mapping(artifacts.get(name), f"audit.artifacts.{name}")
    path = Path(_nonempty_string(artifact.get("path"), f"artifact {name}.path"))
    if not path.is_absolute() or not path.is_file():
        raise ExportError(f"artifact {name} is not an existing absolute file: {path}")
    _expect_equal(
        path.stat().st_size,
        _integer(artifact.get("bytes"), f"artifact {name}.bytes"),
        f"artifact {name} byte size",
    )
    _expect_equal(
        _sha256_file(path),
        _nonempty_string(artifact.get("sha256"), f"artifact {name}.sha256"),
        f"artifact {name} digest",
    )
    return path, artifact


def _validate_audit(
    audit: dict[str, Any],
) -> dict[str, Path]:
    schema = _mapping(audit.get("schema"), "audit.schema")
    _expect_equal(schema.get("name"), AUDIT_SCHEMA_NAME, "audit schema.name")
    _expect_equal(
        schema.get("version"), AUDIT_SCHEMA_VERSION, "audit schema.version"
    )
    _expect_equal(audit.get("result"), "pass", "audit result")
    eligibility = _mapping(audit.get("eligibility"), "audit.eligibility")
    _expect_equal(
        eligibility.get("scheduler_ledger_valid"),
        True,
        "audit scheduler ledger",
    )
    _expect_equal(
        eligibility.get("memory_object_contract_valid"),
        True,
        "audit memory contract",
    )
    _expect_equal(
        eligibility.get("kv_block_lifecycle_valid"),
        True,
        "audit KV lifecycle",
    )
    _expect_equal(
        eligibility.get("production_request_window_verified"),
        True,
        "audit production request window",
    )
    _expect_equal(
        eligibility.get("eligible_for_structural_memory_object_export"),
        True,
        "audit structural export eligibility",
    )
    _expect_equal(
        eligibility.get("eligible_for_memory_system_service_claims"),
        True,
        "audit memory-system service eligibility",
    )
    _expect_equal(
        eligibility.get("eligible_for_ttft_tpot_slo_claims"),
        False,
        "audit TTFT/TPOT/SLO ineligibility",
    )
    paths: dict[str, Path] = {}
    for name in (
        "request_csv",
        "request_manifest",
        "request_suite_verification",
        "request_suite_manifest",
        "request_suite_config",
        "request_source",
        "integration_bundle_manifest",
        "integration_patch",
        "integration_receipt",
        "config",
        "system_metrics",
        "request_metrics",
        "stage_batch_ledger",
        "kv_block_lifecycle",
        "runtime_integration",
    ):
        paths[name] = _artifact_from_audit(audit, name)[0]
    return paths


def _validate_model_descriptor(
    descriptor: dict[str, Any],
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    schema = _mapping(descriptor.get("schema"), "model descriptor.schema")
    _expect_equal(schema.get("name"), MODEL_SCHEMA_NAME, "model schema.name")
    _expect_equal(
        schema.get("version"), MODEL_SCHEMA_VERSION, "model schema.version"
    )
    model = _mapping(descriptor.get("model"), "model descriptor.model")
    _expect_equal(
        model.get("architecture"),
        "dense_decoder_transformer",
        "model.architecture",
    )
    source = _mapping(model.get("source"), "model.source")
    architecture = _mapping(
        descriptor.get("architecture"), "model descriptor.architecture"
    )
    precision = _mapping(
        descriptor.get("precision"), "model descriptor.precision"
    )
    traffic = _mapping(descriptor.get("traffic_model"), "traffic_model")
    addressing = _mapping(descriptor.get("addressing"), "addressing")
    frontier_repository = _nonempty_string(
        source.get("frontier_repository"),
        "model.source.frontier_repository",
    )
    frontier_revision = _revision_string(
        source.get("frontier_revision"),
        "model.source.frontier_revision",
    )
    frontier_config_path = _nonempty_string(
        source.get("frontier_config_path"),
        "model.source.frontier_config_path",
    )
    frontier_config_sha256 = _sha256_string(
        source.get("frontier_config_sha256"),
        "model.source.frontier_config_sha256",
    )
    model_repository = _nonempty_string(
        source.get("model_repository"),
        "model.source.model_repository",
    )
    model_revision = _revision_string(
        source.get("model_revision"),
        "model.source.model_revision",
    )
    config_access = _nonempty_string(
        source.get("config_access"),
        "model.source.config_access",
    )
    precision_profile = _validate_precision_profile(
        precision,
        "model descriptor.precision",
    )

    values = {
        "name": _nonempty_string(model.get("name"), "model.name"),
        "source": {
            "frontier_repository": frontier_repository,
            "frontier_revision": frontier_revision,
            "frontier_config_path": frontier_config_path,
            "frontier_config_sha256": frontier_config_sha256,
            "model_repository": model_repository,
            "model_revision": model_revision,
            "config_access": config_access,
        },
        "num_layers": _integer(
            architecture.get("num_layers"), "architecture.num_layers", minimum=1
        ),
        "hidden_size": _integer(
            architecture.get("hidden_size"), "architecture.hidden_size", minimum=1
        ),
        "intermediate_size": _integer(
            architecture.get("intermediate_size"),
            "architecture.intermediate_size",
            minimum=1,
        ),
        "num_attention_heads": _integer(
            architecture.get("num_attention_heads"),
            "architecture.num_attention_heads",
            minimum=1,
        ),
        "num_key_value_heads": _integer(
            architecture.get("num_key_value_heads"),
            "architecture.num_key_value_heads",
            minimum=1,
        ),
        "vocab_size": _integer(
            architecture.get("vocab_size"),
            "architecture.vocab_size",
            minimum=1,
        ),
        "gated_mlp": _boolean(
            architecture.get("gated_mlp"), "architecture.gated_mlp"
        ),
        "attention_bias": _boolean(
            architecture.get("attention_bias"), "architecture.attention_bias"
        ),
        "mlp_bias": _boolean(
            architecture.get("mlp_bias"), "architecture.mlp_bias"
        ),
        "tie_word_embeddings": _boolean(
            architecture.get("tie_word_embeddings"),
            "architecture.tie_word_embeddings",
        ),
        "memory_precision": precision_profile,
        **precision_profile,
        "alignment": _integer(
            addressing.get("object_alignment_bytes"),
            "addressing.object_alignment_bytes",
            minimum=1,
        ),
        "kv_slot_layout": _nonempty_string(
            addressing.get("kv_slot_layout"), "addressing.kv_slot_layout"
        ),
        "kv_slot_layout_status": _nonempty_string(
            addressing.get("kv_slot_layout_status"),
            "addressing.kv_slot_layout_status",
        ),
        "block_table_entry_bytes": _integer(
            traffic.get("block_table_entry_bytes"),
            "traffic_model.block_table_entry_bytes",
            minimum=1,
        ),
        "scratch_read_bytes_per_token_per_layer": _integer(
            traffic.get("scratch_read_bytes_per_token_per_layer"),
            "traffic_model.scratch_read_bytes_per_token_per_layer",
        ),
        "scratch_write_bytes_per_token_per_layer": _integer(
            traffic.get("scratch_write_bytes_per_token_per_layer"),
            "traffic_model.scratch_write_bytes_per_token_per_layer",
        ),
    }
    if values["hidden_size"] % values["num_attention_heads"] != 0:
        raise ExportError("hidden_size must be divisible by num_attention_heads")
    values["head_dim"] = (
        values["hidden_size"] // values["num_attention_heads"]
    )
    if values["num_key_value_heads"] > values["num_attention_heads"]:
        raise ExportError("num_key_value_heads cannot exceed num_attention_heads")
    if values["attention_bias"] or values["mlp_bias"]:
        raise ExportError("v2 supports bias-free dense transformer models only")
    if not values["gated_mlp"]:
        raise ExportError("v2 requires a gated MLP")
    _expect_equal(
        traffic.get("weight_reads"),
        "once_per_batch_per_layer",
        "traffic_model.weight_reads",
    )
    _expect_equal(
        traffic.get("attention_kv_reads"),
        "context_once_per_request_per_layer",
        "traffic_model.attention_kv_reads",
    )
    _expect_equal(
        traffic.get("kv_writes"),
        "scheduled_input_tokens_once_per_layer",
        "traffic_model.kv_writes",
    )
    _expect_equal(
        traffic.get("output_head_reads"),
        "once_per_batch_with_output_token",
        "traffic_model.output_head_reads",
    )
    _expect_equal(
        traffic.get("embedding_address_policy"),
        "sha256_per_scheduled_token_surrogate_rows",
        "traffic_model.embedding_address_policy",
    )
    if (
        values["scratch_read_bytes_per_token_per_layer"] != 0
        or values["scratch_write_bytes_per_token_per_layer"] != 0
    ):
        raise ExportError(
            "v2 keeps scratch traffic explicitly unmodeled; scratch byte "
            "coefficients must both be zero"
        )
    _expect_equal(
        addressing.get("kv_slot_identity"),
        "frontier_physical_block_id",
        "addressing.kv_slot_identity",
    )
    _expect_equal(
        values["kv_slot_layout"],
        "block_major_layer_major_token_major",
        "addressing.kv_slot_layout",
    )
    _expect_equal(
        values["kv_slot_layout_status"],
        "canonical_object_mapping_not_backend_measured",
        "addressing.kv_slot_layout_status",
    )
    if audit is None:
        return values

    frontier = _mapping(audit.get("frontier"), "audit.frontier")
    _expect_equal(
        frontier_repository,
        frontier.get("repository"),
        "model/Frontier source repository",
    )
    _expect_equal(
        frontier.get("revision"),
        frontier_revision,
        "model/Frontier source revision",
    )
    _expect_equal(frontier.get("model"), values["name"], "Frontier/model name")
    frontier_model = _mapping(
        frontier.get("model_config"), "audit.frontier.model_config"
    )
    _expect_equal(
        frontier_model.get("source_config_path"),
        frontier_config_path,
        "Frontier model_config.source_config_path",
    )
    _expect_equal(
        frontier_model.get("source_config_sha256"),
        frontier_config_sha256,
        "Frontier model_config.source_config_sha256",
    )
    frontier_precision = _mapping(
        frontier.get("memory_precision"),
        "audit.frontier.memory_precision",
    )
    _expect_equal(
        frontier_precision.get("selected_profile"),
        precision_profile,
        "Frontier/model precision profile",
    )
    model_source_identity = _mapping(
        frontier_precision.get("model_source_identity"),
        "audit Frontier model source identity",
    )
    _expect_equal(
        model_source_identity.get("repository"),
        model_repository,
        "Frontier/model repository identity",
    )
    _expect_equal(
        model_source_identity.get("revision"),
        model_revision,
        "Frontier/model revision identity",
    )
    _expect_equal(
        model_source_identity.get("config_access"),
        config_access,
        "Frontier/model config access",
    )
    equivalence = {
        "num_layers": "num_layers",
        "num_q_heads": "num_attention_heads",
        "num_kv_heads": "num_key_value_heads",
        "embedding_dim": "hidden_size",
        "mlp_hidden_dim": "intermediate_size",
        "vocab_size": "vocab_size",
        "use_gated_mlp": "gated_mlp",
        "use_bias": "mlp_bias",
        "use_qkv_bias": "attention_bias",
        "tie_word_embeddings": "tie_word_embeddings",
    }
    for frontier_key, descriptor_key in equivalence.items():
        _expect_equal(
            frontier_model.get(frontier_key),
            values[descriptor_key],
            f"Frontier model_config.{frontier_key}",
        )
    _expect_equal(
        frontier_model.get("norm"),
        1,
        "Frontier model_config.norm (RMSNorm)",
    )
    _expect_equal(
        frontier_model.get("post_attn_norm"),
        True,
        "Frontier model_config.post_attn_norm",
    )
    _expect_equal(
        frontier_model.get("use_qk_norm"),
        False,
        "Frontier model_config.use_qk_norm",
    )
    _expect_equal(
        frontier_model.get("attn_output_gate"),
        False,
        "Frontier model_config.attn_output_gate",
    )
    explicit_head_dim = frontier_model.get("head_dim")
    if explicit_head_dim is not None:
        _expect_equal(
            explicit_head_dim,
            values["head_dim"],
            "Frontier model_config.head_dim",
        )
    parallelism = _mapping(frontier.get("parallelism"), "audit parallelism")
    for name, value in parallelism.items():
        _expect_equal(value, 1, f"Frontier parallelism.{name}")
    return values


class AddressAllocator:
    def __init__(self, alignment: int):
        self.alignment = alignment
        self.cursor = 0
        self.regions: list[dict[str, Any]] = []

    def allocate(
        self,
        *,
        name: str,
        kind: str,
        size: int,
        policy: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if size <= 0:
            raise ExportError(f"object {name} size must be positive")
        begin = _align_up(self.cursor, self.alignment, f"object {name} base")
        end = _checked_add(begin, size, f"object {name} end")
        region = {
            "name": name,
            "kind": kind,
            "begin": begin,
            "end": end,
            "bytes": size,
            "alignment_bytes": self.alignment,
            "placement_policy": policy,
        }
        if metadata:
            region.update(metadata)
        self.regions.append(region)
        self.cursor = end
        return region


def _weight_layout(
    values: dict[str, Any],
    allocator: AddressAllocator,
) -> dict[str, Any]:
    hidden = values["hidden_size"]
    intermediate = values["intermediate_size"]
    q_heads = values["num_attention_heads"]
    kv_heads = values["num_key_value_heads"]
    head_dim = values["head_dim"]
    matrix_bytes = values["matrix_weight_bytes"]
    non_matrix_bytes = values["non_matrix_weight_bytes"]
    metadata_bytes_per_output = (
        values["scale_bytes"] + values["zero_point_bytes"]
    )

    def matrix_storage(
        *,
        name: str,
        elements: int,
        output_channels: int,
    ) -> dict[str, Any]:
        if elements % output_channels:
            raise ExportError(
                f"{name} elements are not divisible by output channels"
            )
        elements_per_output = elements // output_channels
        row_payload_bytes = _checked_mul(
            elements_per_output,
            matrix_bytes,
            f"{name} row payload bytes",
        )
        row_stride_bytes = _checked_add(
            row_payload_bytes,
            metadata_bytes_per_output,
            f"{name} row stride",
        )
        payload_bytes = _checked_mul(
            elements,
            matrix_bytes,
            f"{name} matrix payload bytes",
        )
        scale_elements = (
            output_channels
            if values["quantization_scheme"]
            != "none"
            else 0
        )
        quantization_metadata_bytes = _checked_mul(
            scale_elements,
            metadata_bytes_per_output,
            f"{name} quantization metadata bytes",
        )
        return {
            "name": name,
            "storage_kind": "matrix",
            "elements": elements,
            "matrix_parameters": elements,
            "non_matrix_parameters": 0,
            "output_channels": output_channels,
            "elements_per_output": elements_per_output,
            "row_payload_bytes": row_payload_bytes,
            "row_stride_bytes": row_stride_bytes,
            "matrix_payload_bytes": payload_bytes,
            "non_matrix_payload_bytes": 0,
            "scale_elements": scale_elements,
            "quantization_metadata_bytes": (
                quantization_metadata_bytes
            ),
            "bytes": _checked_mul(
                output_channels,
                row_stride_bytes,
                f"{name} stored bytes",
            ),
        }

    def vector_storage(*, name: str, elements: int) -> dict[str, Any]:
        payload_bytes = _checked_mul(
            elements,
            non_matrix_bytes,
            f"{name} non-matrix payload bytes",
        )
        return {
            "name": name,
            "storage_kind": "non_matrix",
            "elements": elements,
            "matrix_parameters": 0,
            "non_matrix_parameters": elements,
            "matrix_payload_bytes": 0,
            "non_matrix_payload_bytes": payload_bytes,
            "scale_elements": 0,
            "quantization_metadata_bytes": 0,
            "bytes": payload_bytes,
        }

    def category_ledger(
        components: list[dict[str, Any]],
        *,
        multiplier: int,
    ) -> dict[str, int]:
        ledger = {
            "parameters": 0,
            "matrix_parameters": 0,
            "non_matrix_parameters": 0,
            "matrix_payload_bytes": 0,
            "non_matrix_payload_bytes": 0,
            "scale_elements": 0,
            "quantization_metadata_bytes": 0,
            "memory_bytes": 0,
        }
        for component in components:
            ledger["parameters"] += component["elements"] * multiplier
            for field in (
                "matrix_parameters",
                "non_matrix_parameters",
                "matrix_payload_bytes",
                "non_matrix_payload_bytes",
                "scale_elements",
                "quantization_metadata_bytes",
            ):
                ledger[field] += component[field] * multiplier
            ledger["memory_bytes"] += component["bytes"] * multiplier
        return ledger

    embedding_elements = _checked_mul(
        values["vocab_size"], hidden, "embedding elements"
    )
    embedding_storage = matrix_storage(
        name="token_embedding",
        elements=embedding_elements,
        output_channels=values["vocab_size"],
    )
    embedding = allocator.allocate(
        name="model.token_embedding",
        kind="model_weights",
        size=embedding_storage["bytes"],
        policy="model_weight",
        metadata={
            key: value
            for key, value in embedding_storage.items()
            if key != "name"
        },
    )

    component_storage = {
        "input_layernorm": vector_storage(
            name="input_layernorm",
            elements=hidden,
        ),
        "q_proj": matrix_storage(
            name="q_proj",
            elements=_checked_mul(
                hidden,
                q_heads * head_dim,
                "q_proj elements",
            ),
            output_channels=q_heads * head_dim,
        ),
        "k_proj": matrix_storage(
            name="k_proj",
            elements=_checked_mul(
                hidden,
                kv_heads * head_dim,
                "k_proj elements",
            ),
            output_channels=kv_heads * head_dim,
        ),
        "v_proj": matrix_storage(
            name="v_proj",
            elements=_checked_mul(
                hidden,
                kv_heads * head_dim,
                "v_proj elements",
            ),
            output_channels=kv_heads * head_dim,
        ),
        "o_proj": matrix_storage(
            name="o_proj",
            elements=_checked_mul(
                q_heads * head_dim,
                hidden,
                "o_proj elements",
            ),
            output_channels=hidden,
        ),
        "post_attention_layernorm": vector_storage(
            name="post_attention_layernorm",
            elements=hidden,
        ),
        "gate_proj": matrix_storage(
            name="gate_proj",
            elements=_checked_mul(
                hidden,
                intermediate,
                "gate_proj elements",
            ),
            output_channels=intermediate,
        ),
        "up_proj": matrix_storage(
            name="up_proj",
            elements=_checked_mul(
                hidden,
                intermediate,
                "up_proj elements",
            ),
            output_channels=intermediate,
        ),
        "down_proj": matrix_storage(
            name="down_proj",
            elements=_checked_mul(
                intermediate,
                hidden,
                "down_proj elements",
            ),
            output_channels=hidden,
        ),
    }
    per_layer_elements = sum(
        component["elements"]
        for component in component_storage.values()
    )
    per_layer_bytes = sum(
        component["bytes"] for component in component_storage.values()
    )
    layers: list[dict[str, Any]] = []
    for layer_id in range(values["num_layers"]):
        offset = 0
        subobjects: list[dict[str, Any]] = []
        for component, storage in component_storage.items():
            subobjects.append(
                {
                    **storage,
                    "offset": offset,
                }
            )
            offset += storage["bytes"]
        _expect_equal(
            offset,
            per_layer_bytes,
            f"model.layer.{layer_id} subobject bytes",
        )
        layer = allocator.allocate(
            name=f"model.layer.{layer_id}",
            kind="model_weights",
            size=per_layer_bytes,
            policy="model_weight",
            metadata={"model_layer": layer_id, "subobjects": subobjects},
        )
        layers.append(layer)

    final_norm_storage = vector_storage(
        name="final_norm",
        elements=hidden,
    )
    final_norm = allocator.allocate(
        name="model.final_norm",
        kind="model_weights",
        size=final_norm_storage["bytes"],
        policy="model_weight",
        metadata={
            key: value
            for key, value in final_norm_storage.items()
            if key != "name"
        },
    )
    if values["tie_word_embeddings"]:
        lm_head_storage = embedding_storage
        lm_head = {
            **embedding,
            "name": "model.lm_head",
            "alias_of": embedding["name"],
        }
    else:
        lm_head_storage = matrix_storage(
            name="lm_head",
            elements=embedding_elements,
            output_channels=values["vocab_size"],
        )
        lm_head = allocator.allocate(
            name="model.lm_head",
            kind="model_weights",
            size=lm_head_storage["bytes"],
            policy="model_weight",
            metadata={
                key: value
                for key, value in lm_head_storage.items()
                if key != "name"
            },
        )
    categories = {
        "attention": category_ledger(
            [
                component_storage[name]
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            ],
            multiplier=values["num_layers"],
        ),
        "ffn": category_ledger(
            [
                component_storage[name]
                for name in ("gate_proj", "up_proj", "down_proj")
            ],
            multiplier=values["num_layers"],
        ),
        "normalization": category_ledger(
            [
                component_storage["input_layernorm"],
                component_storage["post_attention_layernorm"],
            ],
            multiplier=values["num_layers"],
        ),
        "pipeline_boundary_and_auxiliary": category_ledger(
            [
                embedding_storage,
                final_norm_storage,
                *(
                    []
                    if values["tie_word_embeddings"]
                    else [lm_head_storage]
                ),
            ],
            multiplier=1,
        ),
    }
    breakdown_bytes = {
        name: category["memory_bytes"]
        for name, category in categories.items()
    }
    storage_breakdown = {
        field: sum(category[field] for category in categories.values())
        for field in (
            "matrix_payload_bytes",
            "non_matrix_payload_bytes",
            "quantization_metadata_bytes",
        )
    }
    return {
        "embedding": embedding,
        "layers": layers,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "per_layer_elements": per_layer_elements,
        "per_layer_bytes": per_layer_bytes,
        "component_elements": {
            name: component["elements"]
            for name, component in component_storage.items()
        },
        "component_storage": component_storage,
        "resident_parameters": sum(
            category["parameters"] for category in categories.values()
        ),
        "resident_bytes": sum(breakdown_bytes.values()),
        "breakdown_bytes": breakdown_bytes,
        "storage_breakdown": storage_breakdown,
        "categories": categories,
    }


def derive_dense_model_capacity_inputs(
    model_descriptor_path: Path,
) -> dict[str, Any]:
    """Derive the exact planner inputs from one canonical model descriptor."""
    model_descriptor_path = model_descriptor_path.resolve()
    descriptor = _load_object(
        model_descriptor_path,
        "model descriptor",
    )
    values = _validate_model_descriptor(descriptor, None)
    _expect_equal(
        values["block_table_entry_bytes"],
        4,
        "hybrid-residency block-table entry bytes",
    )
    allocator = AddressAllocator(values["alignment"])
    weights = _weight_layout(values, allocator)
    objects = {
        "embedding": {
            "count": 1,
            "bytes_per_object": int(weights["embedding"]["bytes"]),
        },
        "transformer_layer": {
            "count": int(values["num_layers"]),
            "bytes_per_object": int(weights["per_layer_bytes"]),
        },
        "final_norm": {
            "count": 1,
            "bytes_per_object": int(weights["final_norm"]["bytes"]),
        },
        "output_head": {
            "count": 1,
            "bytes_per_object": int(weights["lm_head"]["bytes"]),
        },
    }
    active_buffer_bytes_per_slot = max(
        entry["bytes_per_object"] for entry in objects.values()
    )
    kv_block_size_tokens = 16
    kv_page_bytes_per_layer = _checked_mul(
        kv_block_size_tokens,
        _checked_mul(
            2 * values["num_key_value_heads"] * values["head_dim"],
            values["kv_bytes"],
            "KV bytes per token per layer",
        ),
        "KV page bytes per layer",
    )
    return {
        "model_name": values["name"],
        "precision_profile": values["profile_id"],
        "model_descriptor": {
            "path": str(model_descriptor_path),
            "bytes": model_descriptor_path.stat().st_size,
            "sha256": _sha256_file(model_descriptor_path),
        },
        "total_parameters": int(weights["resident_parameters"]),
        "immutable_weight_backing_bytes": int(weights["resident_bytes"]),
        "active_weight_buffer_bytes_per_slot": (
            active_buffer_bytes_per_slot
        ),
        "weight_streaming_objects": objects,
        "kv_block_size_tokens": kv_block_size_tokens,
        "kv_page_bytes_per_layer": kv_page_bytes_per_layer,
        "num_layers": int(values["num_layers"]),
        "block_table_entry_bytes": int(
            values["block_table_entry_bytes"]
        ),
    }


def _validate_frontier_weight_and_kv_capacity(
    *,
    audit: dict[str, Any],
    artifacts: dict[str, Path],
    values: dict[str, Any],
    weights: dict[str, Any],
    kv_block_stride: int,
    num_gpu_blocks: int,
) -> dict[str, Any]:
    """Cross-check planner, metrics, and exported object-layout accounting."""
    expected_weight_bytes = _integer(
        weights.get("resident_bytes"),
        "exported resident weight bytes",
        minimum=1,
    )
    expected_parameters = _integer(
        weights.get("resident_parameters"),
        "exported resident parameters",
        minimum=1,
    )

    system = _load_object(
        artifacts["system_metrics"], "Frontier system metrics"
    )
    model_weight_memory = _mapping(
        system.get("model_weight_memory"),
        "Frontier system_metrics.model_weight_memory",
    )
    monolithic = _mapping(
        model_weight_memory.get("MONOLITHIC"),
        "Frontier system_metrics.model_weight_memory.MONOLITHIC",
    )
    _expect_equal(
        monolithic.get("total_parameters"),
        expected_parameters,
        "Frontier total resident parameters",
    )
    _expect_equal(
        monolithic.get("total_memory_bytes"),
        expected_weight_bytes,
        "Frontier total resident weight bytes",
    )
    _expect_equal(
        monolithic.get("memory_precision_profile"),
        values["memory_precision"],
        "Frontier selected memory precision profile",
    )
    _expect_equal(
        monolithic.get("storage_breakdown"),
        weights["storage_breakdown"],
        "Frontier model weight storage breakdown",
    )
    expected_memory_ledger = {
        "profile": values["memory_precision"],
        "total_parameters": expected_parameters,
        "total_memory_bytes": expected_weight_bytes,
        **weights["storage_breakdown"],
        "categories": weights["categories"],
    }
    _expect_equal(
        monolithic.get("memory_ledger"),
        expected_memory_ledger,
        "Frontier exact model memory ledger",
    )
    breakdown = _mapping(
        monolithic.get("breakdown"),
        "Frontier model weight breakdown",
    )
    breakdown_field_names = {
        "attention": "attention_memory_bytes",
        "ffn": "ffn_memory_bytes",
        "normalization": "normalization_memory_bytes",
        "pipeline_boundary_and_auxiliary": (
            "pipeline_boundary_and_auxiliary_memory_bytes"
        ),
    }
    for name, field_name in breakdown_field_names.items():
        _expect_equal(
            breakdown.get(field_name),
            weights["breakdown_bytes"][name],
            f"Frontier model weight breakdown.{field_name}",
        )
    _expect_equal(
        sum(
            _integer(
                breakdown.get(field_name),
                f"Frontier model weight breakdown.{field_name}",
            )
            for field_name in breakdown_field_names.values()
        ),
        expected_weight_bytes,
        "Frontier model weight breakdown sum",
    )

    config = _load_object(artifacts["config"], "Frontier config")
    cluster = _mapping(config.get("cluster_config"), "Frontier cluster_config")
    scheduler = _mapping(
        cluster.get("replica_scheduler_config"),
        "Frontier replica_scheduler_config",
    )
    planner_mode = _nonempty_string(
        scheduler.get("num_blocks_mode"),
        "Frontier scheduler num_blocks_mode",
    )
    _expect_equal(
        planner_mode,
        "hybrid_residency",
        "Frontier scheduler num_blocks_mode",
    )
    _expect_equal(
        scheduler.get("num_blocks"),
        0,
        "Frontier configured num_blocks before hybrid derivation",
    )
    weight_memory_source = _nonempty_string(
        scheduler.get("runtime_weights_memory_source"),
        "Frontier scheduler runtime_weights_memory_source",
    )
    _expect_equal(
        weight_memory_source,
        "param_counter",
        "Frontier scheduler runtime_weights_memory_source",
    )
    runtime_overhead_profiled = _boolean(
        scheduler.get(
            "enable_runtime_non_kv_cache_overhead_profiling",
            False,
        ),
        (
            "Frontier scheduler "
            "enable_runtime_non_kv_cache_overhead_profiling"
        ),
    )
    _expect_equal(
        runtime_overhead_profiled,
        False,
        "Frontier hybrid runtime overhead profiling",
    )
    non_kv_overhead = _integer(
        scheduler.get("non_kv_cache_overhead_bytes"),
        "Frontier scheduler non_kv_cache_overhead_bytes",
        minimum=1,
    )
    physical_hbm_capacity_bytes = _integer(
        scheduler.get("hybrid_physical_hbm_capacity_bytes"),
        "Frontier hybrid physical HBM capacity",
        minimum=1,
    )
    capacity_pressure_target = _finite_number(
        scheduler.get("hybrid_capacity_pressure_target"),
        "Frontier hybrid capacity pressure target",
    )
    _expect_equal(
        scheduler.get("gpu_memory_utilization"),
        None,
        "Frontier hybrid gpu_memory_utilization",
    )

    active_weight_buffer_bytes_per_slot = max(
        int(weights["embedding"]["bytes"]),
        int(weights["per_layer_bytes"]),
        int(weights["final_norm"]["bytes"]),
        int(weights["lm_head"]["bytes"]),
    )
    audit_model_memory = _mapping(
        audit.get("model_memory"),
        "audit.model_memory",
    )
    audit_streaming = _mapping(
        audit_model_memory.get("weight_streaming"),
        "audit.model_memory.weight_streaming",
    )
    _expect_equal(
        audit_streaming.get("active_buffer_bytes_per_slot"),
        active_weight_buffer_bytes_per_slot,
        "independent/audited active weight buffer bytes",
    )
    _expect_equal(
        audit_streaming.get("immutable_weight_backing_bytes"),
        expected_weight_bytes,
        "independent/audited immutable weight backing bytes",
    )
    if kv_block_stride % values["num_layers"] != 0:
        raise ExportError(
            "KV block stride is not divisible by model layer count"
        )
    try:
        residency_plan = validate_hybrid_residency_plan(
            audit.get("residency_plan"),
            physical_hbm_capacity_bytes=physical_hbm_capacity_bytes,
            target_pressure=capacity_pressure_target,
            immutable_weight_backing_bytes=expected_weight_bytes,
            runtime_overhead_bytes=non_kv_overhead,
            active_weight_buffer_bytes_per_slot=(
                active_weight_buffer_bytes_per_slot
            ),
            kv_block_size_tokens=16,
            kv_page_bytes_per_layer=(
                kv_block_stride // values["num_layers"]
            ),
            num_layers=values["num_layers"],
        )
    except HybridResidencyError as error:
        raise ExportError(
            f"Frontier hybrid-residency plan is invalid: {error}"
        ) from error
    _expect_equal(
        num_gpu_blocks,
        residency_plan["num_logical_kv_blocks"],
        "Frontier logical KV capacity from hybrid residency",
    )
    return {
        "planner_mode": planner_mode,
        "weight_memory_source": weight_memory_source,
        "runtime_overhead_source": (
            "configured_unprofiled_sensitivity"
        ),
        "non_kv_cache_overhead_runtime_profiled": runtime_overhead_profiled,
        "hardware_capacity_calibrated": False,
        "capacity_interpretation": (
            "unique_footprint_hybrid_residency_sensitivity"
        ),
        "immutable_weight_backing_bytes": expected_weight_bytes,
        "model_parameters": expected_parameters,
        "memory_precision_profile": values["memory_precision"],
        "immutable_weight_storage_breakdown": dict(
            weights["storage_breakdown"]
        ),
        "immutable_weight_breakdown_bytes": dict(
            weights["breakdown_bytes"]
        ),
        "residency_plan": residency_plan,
    }


def _read_ledger_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise ExportError(f"stage ledger line {line_no} is blank")
            try:
                row = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ExportError(
                    f"invalid stage ledger JSON at line {line_no}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ExportError(f"stage ledger line {line_no} is not an object")
            rows.append(row)
    if not rows:
        raise ExportError("stage ledger has no rows")
    return rows


def _read_lifecycle_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise ExportError(f"KV lifecycle line {line_no} is blank")
            try:
                event = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ExportError(
                    f"invalid KV lifecycle JSON at line {line_no}: {error}"
                ) from error
            if not isinstance(event, dict):
                raise ExportError(
                    f"KV lifecycle line {line_no} is not an object"
                )
            events.append(event)
    if not events:
        raise ExportError("KV lifecycle ledger has no events")
    return events


def _materialize_selected_allocator_states(
    *,
    ledger_rows: list[dict[str, Any]],
    lifecycle_path: Path,
    batch_start: int,
    stop: int,
) -> None:
    """Replay the audited lifecycle and materialize only selected KV snapshots."""

    events = _read_lifecycle_events(lifecycle_path)
    first_contract = _mapping(
        ledger_rows[0].get("hbfsim_memory_contract"),
        "first memory contract",
    )
    first_kv_cache = _mapping(
        first_contract.get("kv_cache"),
        "first memory contract.kv_cache",
    )
    num_gpu_blocks = _integer(
        first_kv_cache.get("num_gpu_blocks"),
        "first memory contract num_gpu_blocks",
        minimum=1,
    )
    block_hashes: list[int | None] = [None] * num_gpu_blocks
    block_ref_counts = [0] * num_gpu_blocks
    request_blocks: dict[str, list[int]] = {}
    next_event_index = 0

    def block_id(value: Any, path: str) -> int:
        result = _integer(value, path)
        if result >= num_gpu_blocks:
            raise ExportError(
                f"{path}={result} exceeds num_gpu_blocks={num_gpu_blocks}"
            )
        return result

    def apply_event(event: dict[str, Any]) -> None:
        event_type = str(event.get("event"))
        request_id_value = event.get("request_id")
        request_id = (
            None if request_id_value is None else str(request_id_value)
        )
        if event_type in {"prefix_lookup", "prefix_admission"}:
            return
        if event_type == "evict":
            current = block_id(event.get("block_id"), "evict block_id")
            block_hashes[current] = None
            return
        if event_type == "allocate":
            if request_id is None:
                raise ExportError("allocate lifecycle event has no request_id")
            current = block_id(event.get("block_id"), "allocate block_id")
            block_ref_counts[current] = _integer(
                event.get("ref_count_after"),
                "allocate ref_count_after",
                minimum=1,
            )
            request_blocks.setdefault(request_id, []).append(current)
            return
        if event_type == "cache_assign":
            current = block_id(event.get("block_id"), "cache_assign block_id")
            block_hashes[current] = _integer(
                event.get("block_hash"),
                "cache_assign block_hash",
            )
            return
        if event_type not in {"touch", "release"}:
            raise ExportError(
                f"unsupported KV lifecycle event during materialization: "
                f"{event_type!r}"
            )
        if request_id is None:
            raise ExportError(f"{event_type} lifecycle event has no request_id")
        transitions = event.get("blocks")
        if not isinstance(transitions, list) or not transitions:
            raise ExportError(
                f"{event_type} lifecycle event has no block transitions"
            )
        transition_ids: list[int] = []
        for position, transition_value in enumerate(transitions):
            transition = _mapping(
                transition_value,
                f"{event_type} transition[{position}]",
            )
            current = block_id(
                transition.get("block_id"),
                f"{event_type} transition[{position}].block_id",
            )
            transition_ids.append(current)
            block_ref_counts[current] = _integer(
                transition.get("ref_count_after"),
                f"{event_type} transition[{position}].ref_count_after",
            )
        if event_type == "touch":
            request_blocks.setdefault(request_id, []).extend(transition_ids)
        else:
            request_blocks.pop(request_id, None)

    previous_cursor = 0
    for row_index, row in enumerate(ledger_rows[:stop]):
        contract = _mapping(
            row.get("hbfsim_memory_contract"),
            f"batch {row_index} memory contract",
        )
        _expect_equal(
            contract.get("schema_version"),
            MEMORY_CONTRACT_SCHEMA_VERSION,
            f"batch {row_index} memory contract schema",
        )
        if "lifecycle_events" in contract:
            raise ExportError(
                f"batch {row_index} embeds obsolete lifecycle_events"
            )
        cursor = _integer(
            contract.get("lifecycle_event_cursor"),
            f"batch {row_index} lifecycle_event_cursor",
        )
        if cursor < previous_cursor or cursor > len(events):
            raise ExportError(
                f"batch {row_index} has an invalid lifecycle cursor {cursor}"
            )
        while next_event_index < cursor:
            apply_event(events[next_event_index])
            next_event_index += 1
        previous_cursor = cursor
        if row_index < batch_start:
            continue
        snapshots = contract.get("requests")
        if not isinstance(snapshots, list) or not snapshots:
            raise ExportError(f"batch {row_index} has no request snapshots")
        for snapshot_value in snapshots:
            snapshot = _mapping(
                snapshot_value,
                f"batch {row_index} request snapshot",
            )
            request_id = _nonempty_string(
                snapshot.get("request_id"),
                f"batch {row_index} request_id",
            )
            owned = request_blocks.get(request_id, [])
            blocks = [
                {
                    "block_id": current,
                    "block_hash": block_hashes[current],
                    "ref_count": block_ref_counts[current],
                }
                for current in owned
            ]
            _expect_equal(
                snapshot.get("allocated_block_count"),
                len(blocks),
                f"batch {row_index} request {request_id} allocated block count",
            )
            _expect_equal(
                snapshot.get("allocator_state_sha256"),
                allocator_state_sha256(request_id, blocks),
                f"batch {row_index} request {request_id} allocator-state digest",
            )
            snapshot["allocated_blocks"] = blocks


def _surrogate_embedding_row(
    *,
    request_id: str,
    token_position: int,
    model_name: str,
    vocab_size: int,
) -> int:
    key = f"{model_name}\0{request_id}\0{token_position}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % vocab_size


def _batch_census(row: dict[str, Any]) -> dict[str, Any]:
    """Record the selected scheduler batch without implying batch size."""

    batch_id = _integer(row.get("batch_id"), "stage ledger batch_id")
    contract = _mapping(
        row.get("hbfsim_memory_contract"),
        f"batch {batch_id} memory contract",
    )
    snapshots = contract.get("requests")
    if not isinstance(snapshots, list) or not snapshots:
        raise ExportError(f"batch {batch_id} must contain at least one request")

    request_ids = [
        _nonempty_string(
            _mapping(snapshot, f"batch {batch_id} request[{index}]").get(
                "request_id"
            ),
            f"batch {batch_id} request[{index}].request_id",
        )
        for index, snapshot in enumerate(snapshots)
    ]
    if len(set(request_ids)) != len(request_ids):
        raise ExportError(f"batch {batch_id} request IDs must be unique")
    row_request_ids = row.get("request_ids")
    if not isinstance(row_request_ids, list):
        raise ExportError(f"batch {batch_id} request_ids must be an array")
    _expect_equal(row_request_ids, request_ids, f"batch {batch_id} request IDs")

    scheduled_by_request = [
        _integer(
            _mapping(snapshot, f"batch {batch_id} request[{index}]").get(
                "scheduled_tokens"
            ),
            f"batch {batch_id} request[{index}].scheduled_tokens",
            minimum=1,
        )
        for index, snapshot in enumerate(snapshots)
    ]
    row_request_tokens = row.get("request_num_tokens")
    if not isinstance(row_request_tokens, list):
        raise ExportError(
            f"batch {batch_id} request_num_tokens must be an array"
        )
    _expect_equal(
        row_request_tokens,
        scheduled_by_request,
        f"batch {batch_id} scheduled tokens",
    )

    phases = sorted(
        {
            _nonempty_string(
                _mapping(snapshot, f"batch {batch_id} request[{index}]").get(
                    "phase"
                ),
                f"batch {batch_id} request[{index}].phase",
            )
            for index, snapshot in enumerate(snapshots)
        }
    )
    if not set(phases).issubset({"prefill", "decode"}):
        raise ExportError(f"batch {batch_id} has unsupported request phase")

    stage_start_s = _finite_number(
        row.get("stage_start_ts"), f"batch {batch_id} stage_start_ts"
    )
    stage_end_s = _finite_number(
        row.get("stage_end_ts"), f"batch {batch_id} stage_end_ts"
    )
    if stage_start_s < 0 or stage_end_s < stage_start_s:
        raise ExportError(f"batch {batch_id} has an invalid stage interval")

    return {
        "batch_id": batch_id,
        "request_count": len(snapshots),
        "request_ids": request_ids,
        "scheduled_tokens": sum(scheduled_by_request),
        "request_phases": phases,
        "produces_output_token": any(
            _boolean(
                _mapping(snapshot, f"batch {batch_id} request[{index}]").get(
                    "produces_output_token"
                ),
                f"batch {batch_id} request[{index}].produces_output_token",
            )
            for index, snapshot in enumerate(snapshots)
        ),
        "frontier_stage_start_s": stage_start_s,
        "frontier_stage_end_s": stage_end_s,
    }


def export_memory_trace(
    *,
    audit_path: Path,
    model_descriptor_path: Path,
    output_trace: Path,
    object_map_path: Path,
    phase_map_path: Path,
    manifest_path: Path,
    batch_start: int = 0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    paths = [
        audit_path,
        model_descriptor_path,
        output_trace,
        object_map_path,
        phase_map_path,
        manifest_path,
    ]
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ExportError("all input and output paths must be distinct")
    if batch_start < 0:
        raise ExportError("batch_start must be non-negative")
    if max_batches is not None and max_batches <= 0:
        raise ExportError("max_batches must be positive when provided")

    audit_path = audit_path.resolve()
    model_descriptor_path = model_descriptor_path.resolve()
    output_trace = output_trace.resolve()
    object_map_path = object_map_path.resolve()
    phase_map_path = phase_map_path.resolve()
    manifest_path = manifest_path.resolve()
    audit = _load_object(audit_path, "Frontier replay audit")
    artifacts = _validate_audit(audit)
    descriptor = _load_object(model_descriptor_path, "model descriptor")
    values = _validate_model_descriptor(descriptor, audit)
    ledger_rows = _read_ledger_rows(artifacts["stage_batch_ledger"])
    if batch_start >= len(ledger_rows):
        raise ExportError(
            f"batch_start={batch_start} is outside {len(ledger_rows)} ledger rows"
        )
    stop = (
        len(ledger_rows)
        if max_batches is None
        else min(len(ledger_rows), batch_start + max_batches)
    )
    selected_rows = ledger_rows[batch_start:stop]
    expected_batch_ids = list(range(batch_start, stop))
    actual_batch_ids = [
        _integer(row.get("batch_id"), "stage ledger batch_id")
        for row in selected_rows
    ]
    _expect_equal(actual_batch_ids, expected_batch_ids, "selected batch IDs")
    _materialize_selected_allocator_states(
        ledger_rows=ledger_rows,
        lifecycle_path=artifacts["kv_block_lifecycle"],
        batch_start=batch_start,
        stop=stop,
    )
    selected_batch_census = [_batch_census(row) for row in selected_rows]

    first_contract = _mapping(
        selected_rows[0].get("hbfsim_memory_contract"),
        "first memory contract",
    )
    kv_cache = _mapping(first_contract.get("kv_cache"), "first contract kv_cache")
    block_size = _integer(
        kv_cache.get("block_size_tokens"), "KV block size", minimum=1
    )
    num_gpu_blocks = _integer(
        kv_cache.get("num_gpu_blocks"), "KV num_gpu_blocks", minimum=1
    )
    _expect_equal(block_size, 16, "KV block size")

    allocator = AddressAllocator(values["alignment"])
    weights = _weight_layout(values, allocator)
    kv_bytes_per_token_per_layer = _checked_mul(
        2 * values["num_key_value_heads"] * values["head_dim"],
        values["kv_bytes"],
        "KV bytes per token per layer",
    )
    kv_layer_stride = _checked_mul(
        block_size,
        kv_bytes_per_token_per_layer,
        "KV layer stride",
    )
    kv_block_stride = _checked_mul(
        values["num_layers"], kv_layer_stride, "KV block stride"
    )
    capacity_accounting = _validate_frontier_weight_and_kv_capacity(
        audit=audit,
        artifacts=artifacts,
        values=values,
        weights=weights,
        kv_block_stride=kv_block_stride,
        num_gpu_blocks=num_gpu_blocks,
    )
    kv_region = allocator.allocate(
        name="kv.physical_block_slots",
        kind="generated_context",
        size=_checked_mul(
            num_gpu_blocks, kv_block_stride, "KV slot arena bytes"
        ),
        policy="frontier_physical_block_id",
        metadata={
            "num_slots": num_gpu_blocks,
            "block_size_tokens": block_size,
            "bytes_per_token_per_layer": kv_bytes_per_token_per_layer,
            "bytes_per_layer_per_slot": kv_layer_stride,
            "bytes_per_slot": kv_block_stride,
        },
    )
    residency_plan = capacity_accounting["residency_plan"]
    allocator.allocate(
        name="metadata.kv_allocator_blocks",
        kind="metadata",
        size=int(residency_plan["block_table_bytes"]),
        policy="hbm_only",
        metadata={
            "num_logical_kv_blocks": num_gpu_blocks,
            "block_table_entry_bytes": values["block_table_entry_bytes"],
            "accounting_role": "hybrid_residency_block_table",
        },
    )

    request_count = _integer(
        _mapping(audit.get("requests"), "audit.requests").get("count"),
        "audit request count",
        minimum=1,
    )
    max_blocks_per_request = max(
        _integer(
            snapshot.get("allocated_block_count"),
            "memory contract allocated_block_count",
            minimum=1,
        )
        for row in ledger_rows
        for snapshot in row["hbfsim_memory_contract"]["requests"]
    )
    metadata_stride = _align_up(
        _checked_mul(
            max_blocks_per_request,
            values["block_table_entry_bytes"],
            "metadata bytes per request",
        ),
        64,
        "metadata request stride",
    )
    request_metadata_bytes = _checked_mul(
        request_count,
        metadata_stride,
        "request metadata arena bytes",
    )
    runtime_overhead_bytes = int(
        residency_plan["runtime_overhead_bytes"]
    )
    if request_metadata_bytes > runtime_overhead_bytes:
        raise ExportError(
            "request block-table metadata exceeds the configured runtime "
            "overhead allocation"
        )
    metadata_region = allocator.allocate(
        name="metadata.runtime_overhead",
        kind="metadata",
        size=runtime_overhead_bytes,
        policy="hbm_only",
        metadata={
            "request_slots": request_count,
            "request_stride_bytes": metadata_stride,
            "request_metadata_bytes": request_metadata_bytes,
            "block_table_entry_bytes": values["block_table_entry_bytes"],
            "accounting_role": (
                "configured_runtime_activation_and_framework_overhead"
            ),
            "unmodeled_remainder_bytes": (
                runtime_overhead_bytes - request_metadata_bytes
            ),
        },
    )
    address_space_bytes = _align_up(
        allocator.cursor, values["alignment"], "address space bytes"
    )

    region_by_name = {region["name"]: region for region in allocator.regions}
    census = {
        "operations": 0,
        "bytes": 0,
        "reads": {"operations": 0, "bytes": 0},
        "writes": {"operations": 0, "bytes": 0},
        "by_kind": defaultdict(lambda: {"operations": 0, "bytes": 0}),
        "by_object": defaultdict(lambda: {"operations": 0, "bytes": 0}),
    }
    phases: list[dict[str, Any]] = []
    trace_handle, trace_temporary = _open_temporary(output_trace)
    temporary_paths: list[Path] = [trace_temporary]
    phase_id = 0

    def emit(
        *,
        address: int,
        operation: str,
        byte_count: int,
        kind: str,
        label: str,
        object_name: str,
        phase: int,
        at_ns: float,
    ) -> None:
        if operation not in {"R", "W"} or byte_count <= 0:
            raise ExportError("trace emitter received an invalid operation")
        region = region_by_name[object_name]
        end = _checked_add(address, byte_count, "trace operation end")
        if address < region["begin"] or end > region["end"]:
            raise ExportError(
                f"trace operation {label} escapes object {object_name}"
            )
        trace_handle.write(
            f"0x{address:x} {operation} {byte_count} "
            f"kind={kind} label={label} phase={phase} layer={phase} "
            f"at={at_ns:.9f}\n"
        )
        census["operations"] += 1
        census["bytes"] += byte_count
        direction = "reads" if operation == "R" else "writes"
        census[direction]["operations"] += 1
        census[direction]["bytes"] += byte_count
        census["by_kind"][kind]["operations"] += 1
        census["by_kind"][kind]["bytes"] += byte_count
        census["by_object"][object_name]["operations"] += 1
        census["by_object"][object_name]["bytes"] += byte_count
        phases[-1]["operations"] += 1
        phases[-1]["bytes"] += byte_count
        phases[-1][direction]["operations"] += 1
        phases[-1][direction]["bytes"] += byte_count

    def begin_phase(
        *,
        row: dict[str, Any],
        component: str,
        model_layer: int | None,
        at_ns: float,
    ) -> int:
        nonlocal phase_id
        current = phase_id
        phase_id += 1
        phases.append(
            {
                "phase_id": current,
                "batch_id": int(row["batch_id"]),
                "component": component,
                "model_layer": model_layer,
                "frontier_stage_start_s": float(row["stage_start_ts"]),
                "frontier_stage_end_s": float(row["stage_end_ts"]),
                "offered_at_ns": at_ns,
                "operations": 0,
                "bytes": 0,
                "reads": {"operations": 0, "bytes": 0},
                "writes": {"operations": 0, "bytes": 0},
            }
        )
        trace_handle.write(
            f"# phase={current} batch={row['batch_id']} "
            f"component={component}"
            + (
                ""
                if model_layer is None
                else f" model_layer={model_layer}"
            )
            + "\n"
        )
        return current

    def emit_kv_segments(
        *,
        snapshot: dict[str, Any],
        model_layer: int,
        token_begin: int,
        token_count: int,
        operation: str,
        phase: int,
        at_ns: float,
    ) -> None:
        remaining = token_count
        cursor = token_begin
        blocks = snapshot["allocated_blocks"]
        while remaining:
            block_ordinal = cursor // block_size
            token_offset = cursor % block_size
            if block_ordinal >= len(blocks):
                raise ExportError(
                    f"request {snapshot['request_id']} KV frontier exceeds "
                    "its Frontier block snapshot"
                )
            block = blocks[block_ordinal]
            block_id = _integer(block.get("block_id"), "snapshot block_id")
            block_tokens = min(remaining, block_size - token_offset)
            offset = (
                block_id * kv_block_stride
                + model_layer * kv_layer_stride
                + token_offset * kv_bytes_per_token_per_layer
            )
            address = kv_region["begin"] + offset
            kind = (
                "shared_context"
                if block.get("block_hash") is not None
                else "generated_context"
            )
            emit(
                address=address,
                operation=operation,
                byte_count=block_tokens * kv_bytes_per_token_per_layer,
                kind=kind,
                label=(
                    f"kv.{operation.lower()}.request.{snapshot['request_id']}."
                    f"block.{block_id}.model_layer.{model_layer}"
                ),
                object_name=kv_region["name"],
                phase=phase,
                at_ns=at_ns,
            )
            cursor += block_tokens
            remaining -= block_tokens

    def emit_embedding_rows(
        *,
        snapshot: dict[str, Any],
        phase: int,
        at_ns: float,
    ) -> None:
        """Emit one deterministic surrogate vocabulary row per input token.

        The serving trace does not contain token IDs.  Hashing each request
        position independently avoids inventing sequential vocabulary
        locality.  Adjacent surrogate rows are coalesced only when their
        physical addresses are genuinely contiguous.
        """

        request_id = str(snapshot["request_id"])
        token_begin = int(snapshot["kv_tokens_before"])
        token_count = int(snapshot["scheduled_tokens"])
        run_token_begin = token_begin
        run_row = _surrogate_embedding_row(
            request_id=request_id,
            token_position=token_begin,
            model_name=values["name"],
            vocab_size=values["vocab_size"],
        )
        run_rows = 1
        previous_row = run_row

        def flush_run() -> None:
            emit(
                address=(
                    weights["embedding"]["begin"]
                    + run_row * embedding_row_bytes
                ),
                operation="R",
                byte_count=run_rows * embedding_row_bytes,
                kind="model_weights",
                label=(
                    f"embedding.request.{request_id}.tokens."
                    f"{run_token_begin}-{run_token_begin + run_rows}"
                ),
                object_name=weights["embedding"]["name"],
                phase=phase,
                at_ns=at_ns,
            )

        for token_position in range(token_begin + 1, token_begin + token_count):
            row = _surrogate_embedding_row(
                request_id=request_id,
                token_position=token_position,
                model_name=values["name"],
                vocab_size=values["vocab_size"],
            )
            if row == previous_row + 1:
                run_rows += 1
                previous_row = row
                continue
            flush_run()
            run_token_begin = token_position
            run_row = row
            run_rows = 1
            previous_row = row
        flush_run()

    try:
        trace_handle.write(
            f"# HBFSim Frontier memory-object trace "
            f"schema={TRACE_SCHEMA_VERSION}\n"
        )
        trace_handle.write(f"# audit_sha256={_sha256_file(audit_path)}\n")
        trace_handle.write(
            f"# model_descriptor_sha256={_sha256_file(model_descriptor_path)}\n"
        )
        trace_handle.write(
            "# timing=dependency_barrier_batch_ready "
            "phase_contract=complete-before-next "
            "layer_contract=global-phase-streaming-window\n"
        )

        embedding_row_bytes = weights["embedding"]["row_stride_bytes"]
        output_producing_batches = 0
        for row in selected_rows:
            contract = _mapping(
                row.get("hbfsim_memory_contract"), "stage memory contract"
            )
            _expect_equal(
                contract.get("batch_id"), row.get("batch_id"), "contract batch ID"
            )
            batch_ready_ns = float(row["stage_start_ts"]) * 1e9

            at_ns = batch_ready_ns
            current_phase = begin_phase(
                row=row,
                component="embedding",
                model_layer=None,
                at_ns=at_ns,
            )
            for snapshot in contract["requests"]:
                emit_embedding_rows(
                    snapshot=snapshot,
                    phase=current_phase,
                    at_ns=at_ns,
                )

            for model_layer, layer_region in enumerate(weights["layers"]):
                at_ns = batch_ready_ns
                current_phase = begin_phase(
                    row=row,
                    component="transformer_layer",
                    model_layer=model_layer,
                    at_ns=at_ns,
                )
                emit(
                    address=layer_region["begin"],
                    operation="R",
                    byte_count=layer_region["bytes"],
                    kind="model_weights",
                    label=f"weights.model_layer.{model_layer}",
                    object_name=layer_region["name"],
                    phase=current_phase,
                    at_ns=at_ns,
                )
                for snapshot in contract["requests"]:
                    emit_kv_segments(
                        snapshot=snapshot,
                        model_layer=model_layer,
                        token_begin=int(snapshot["kv_tokens_before"]),
                        token_count=int(snapshot["scheduled_tokens"]),
                        operation="W",
                        phase=current_phase,
                        at_ns=at_ns,
                    )
                    emit_kv_segments(
                        snapshot=snapshot,
                        model_layer=model_layer,
                        token_begin=0,
                        token_count=int(snapshot["kv_tokens_after"]),
                        operation="R",
                        phase=current_phase,
                        at_ns=at_ns,
                    )
                    request_id = _integer(
                        int(snapshot["request_id"]), "request ID"
                    )
                    metadata_bytes = (
                        int(snapshot["allocator_required_blocks"])
                        * values["block_table_entry_bytes"]
                    )
                    emit(
                        address=(
                            metadata_region["begin"]
                            + request_id * metadata_stride
                        ),
                        operation="R",
                        byte_count=metadata_bytes,
                        kind="metadata",
                        label=(
                            f"block_table.request.{snapshot['request_id']}."
                            f"model_layer.{model_layer}"
                        ),
                        object_name=metadata_region["name"],
                        phase=current_phase,
                        at_ns=at_ns,
                    )

            at_ns = batch_ready_ns
            current_phase = begin_phase(
                row=row,
                component="final_norm",
                model_layer=None,
                at_ns=at_ns,
            )
            emit(
                address=weights["final_norm"]["begin"],
                operation="R",
                byte_count=weights["final_norm"]["bytes"],
                kind="model_weights",
                label="weights.final_norm",
                object_name=weights["final_norm"]["name"],
                phase=current_phase,
                at_ns=at_ns,
            )
            if any(
                snapshot["produces_output_token"]
                for snapshot in contract["requests"]
            ):
                output_producing_batches += 1
                current_phase = begin_phase(
                    row=row,
                    component="lm_head",
                    model_layer=None,
                    at_ns=at_ns,
                )
                lm_head_object = (
                    weights["embedding"]
                    if "alias_of" in weights["lm_head"]
                    else weights["lm_head"]
                )
                emit(
                    address=lm_head_object["begin"],
                    operation="R",
                    byte_count=lm_head_object["bytes"],
                    kind="model_weights",
                    label="weights.lm_head",
                    object_name=lm_head_object["name"],
                    phase=current_phase,
                    at_ns=at_ns,
                )

        trace_handle.flush()
        os.fsync(trace_handle.fileno())
        trace_handle.close()

        normalized_census = {
            **{
                key: value
                for key, value in census.items()
                if key not in {"by_kind", "by_object"}
            },
            "by_kind": {
                key: value for key, value in sorted(census["by_kind"].items())
            },
            "by_object": {
                key: value
                for key, value in sorted(census["by_object"].items())
            },
        }
        object_map = {
            "schema": {
                "name": OBJECT_MAP_SCHEMA_NAME,
                "version": TRACE_SCHEMA_VERSION,
            },
            "address_space": {
                "begin": 0,
                "end": address_space_bytes,
                "bytes": address_space_bytes,
                "object_alignment_bytes": values["alignment"],
            },
            "regions": allocator.regions,
            "excluded_objects": [
                {
                    "name": "scratch.activations",
                    "reason": (
                        "no calibrated kernel-level off-chip activation model; "
                        "the declared coefficients are explicitly zero"
                    ),
                    "read_bytes_per_token_per_layer": 0,
                    "write_bytes_per_token_per_layer": 0,
                }
            ],
        }
        phase_map = {
            "schema": {
                "name": PHASE_MAP_SCHEMA_NAME,
                "version": TRACE_SCHEMA_VERSION,
            },
            "timing_semantics": {
                "mode": "dependency_barrier_batch_ready",
                "frontier_stage_start_preserved": True,
                "frontier_execution_time_used_as_hardware_latency": False,
                "phase_dependency": "complete_before_next",
                "phase_scope": "global_trace_order",
                "layer_scope": "global_phase_streaming_window",
            },
            "phases": phases,
        }
        object_temporary = _write_json_temporary(object_map_path, object_map)
        phase_temporary = _write_json_temporary(phase_map_path, phase_map)
        temporary_paths.extend([object_temporary, phase_temporary])

        trace_digest = _sha256_file(trace_temporary)
        object_digest = _sha256_file(object_temporary)
        phase_digest = _sha256_file(phase_temporary)
        manifest = {
            "schema": {
                "name": TRACE_SCHEMA_NAME,
                "version": TRACE_SCHEMA_VERSION,
            },
            "semantics": {
                "source_requests_are_production": True,
                "memory_traffic_is_model_conditioned": True,
                "memory_traffic_is_measured_gpu_traffic": False,
                "frontier_timing_is_hardware_calibrated": False,
                "eligible_claim_scope": "memory_system_service_only",
                "ttft_tpot_slo_claims_eligible": False,
                "time_based_throughput_claims_eligible": False,
                "export_mode": "dependency_barrier_batch_ready",
                "phase_dependency": "complete_before_next",
                "layer_scope": "global_phase_streaming_window",
                "embedding_rows_are_true_token_ids": False,
                "embedding_surrogate": (
                    "SHA-256(model_name, request_id, token_position), "
                    "independently per scheduled token"
                ),
                "semantic_kind_definitions": {
                    "shared_context": (
                        "full KV block admitted to Frontier prefix cache; "
                        "does not by itself assert concurrent ref_count > 1"
                    ),
                    "generated_context": (
                        "partial or otherwise non-cache-addressable KV block"
                    ),
                },
                "kv_slot_layout_is_backend_measured": False,
                "model_weight_traffic": "read_only",
                "mutable_write_traffic": (
                    "kv_append_and_update_only"
                ),
                "kv_capacity_is_hardware_calibrated": capacity_accounting[
                    "hardware_capacity_calibrated"
                ],
                "capacity_pressure_basis": (
                    "unique_resident_footprint_bytes/"
                    "physical_hbm_capacity_bytes"
                ),
            },
            "selection": {
                "total_frontier_batches": len(ledger_rows),
                "batch_start": batch_start,
                "batches": len(selected_rows),
                "output_producing_batches": output_producing_batches,
                "batch_ids": actual_batch_ids,
                "selected_batch_census": selected_batch_census,
                "is_full_replay": (
                    batch_start == 0 and len(selected_rows) == len(ledger_rows)
                ),
                "scaling_applied": False,
                "byte_sampling_applied": False,
            },
            "model": values,
            "equations": {
                "head_dim": "hidden_size / num_attention_heads",
                "kv_bytes_per_token_per_layer": (
                    "2 * num_key_value_heads * head_dim * kv_bytes"
                ),
                "kv_layer_stride": (
                    "block_size_tokens * kv_bytes_per_token_per_layer"
                ),
                "kv_block_stride": "num_layers * kv_layer_stride",
                "dense_layer_weight_elements": (
                    "2*hidden_size + q_proj + k_proj + v_proj + o_proj + "
                    "3*hidden_size*intermediate_size"
                ),
                "matrix_storage_bytes": (
                    "matrix_elements * matrix_weight_bytes + "
                    "output_channels * (scale_bytes + zero_point_bytes)"
                ),
                "non_matrix_storage_bytes": (
                    "non_matrix_elements * non_matrix_weight_bytes"
                ),
                "attention_kv_read_bytes": (
                    "kv_tokens_after * kv_bytes_per_token_per_layer, once per "
                    "request and transformer layer"
                ),
                "embedding_row": (
                    "uint64_be(SHA-256(model_name || NUL || request_id || NUL "
                    "|| token_position)[0:8]) mod vocab_size"
                ),
                "kv_write_bytes": (
                    "scheduled_tokens * kv_bytes_per_token_per_layer, once per "
                    "transformer layer"
                ),
                "evaluated": {
                    "kv_bytes_per_token_per_layer": (
                        kv_bytes_per_token_per_layer
                    ),
                    "kv_layer_stride": kv_layer_stride,
                    "kv_block_stride": kv_block_stride,
                    "dense_layer_weight_elements": (
                        weights["per_layer_elements"]
                    ),
                    "dense_layer_weight_bytes": weights["per_layer_bytes"],
                    "immutable_weight_backing_bytes": (
                        weights["resident_bytes"]
                    ),
                },
            },
            "capacity_accounting": capacity_accounting,
            "traffic_census": normalized_census,
            "sources": {
                "frontier": audit["frontier"],
                "audit": {
                    "path": str(audit_path),
                    "bytes": audit_path.stat().st_size,
                    "sha256": _sha256_file(audit_path),
                },
                "model_descriptor": {
                    "path": str(model_descriptor_path),
                    "bytes": model_descriptor_path.stat().st_size,
                    "sha256": _sha256_file(model_descriptor_path),
                },
                "frontier_artifacts": audit["artifacts"],
            },
            "outputs": {
                "trace": {
                    "path": str(output_trace),
                    "bytes": trace_temporary.stat().st_size,
                    "sha256": trace_digest,
                },
                "object_map": {
                    "path": str(object_map_path),
                    "bytes": object_temporary.stat().st_size,
                    "sha256": object_digest,
                },
                "phase_map": {
                    "path": str(phase_map_path),
                    "bytes": phase_temporary.stat().st_size,
                    "sha256": phase_digest,
                },
            },
        }
        manifest_temporary = _write_json_temporary(manifest_path, manifest)
        temporary_paths.append(manifest_temporary)

        os.replace(trace_temporary, output_trace)
        os.replace(object_temporary, object_map_path)
        os.replace(phase_temporary, phase_map_path)
        os.replace(manifest_temporary, manifest_path)
        temporary_paths.clear()
        return manifest
    except BaseException:
        if not trace_handle.closed:
            trace_handle.close()
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--model-descriptor", type=Path, required=True)
    parser.add_argument("--output-trace", type=Path, required=True)
    parser.add_argument("--object-map", type=Path, required=True)
    parser.add_argument("--phase-map", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = export_memory_trace(
            audit_path=args.audit,
            model_descriptor_path=args.model_descriptor,
            output_trace=args.output_trace,
            object_map_path=args.object_map,
            phase_map_path=args.phase_map,
            manifest_path=args.manifest,
            batch_start=args.batch_start,
            max_batches=args.max_batches,
        )
    except (ExportError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    census = manifest["traffic_census"]
    print(
        f"exported {manifest['selection']['batches']} batches: "
        f"ops={census['operations']} bytes={census['bytes']}"
    )
    print(f"trace={args.output_trace.resolve()}")
    print(f"manifest={args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
