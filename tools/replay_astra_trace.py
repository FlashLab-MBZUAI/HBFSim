#!/usr/bin/env python3
"""Replay an ASTRA-sim HBF trace across the EXPERIMENT cases (ec0-ec6).

Taxonomy: USE cases (uc*, run_use_cases.py)
keep the simple synthetic inputs — their job is checking model invariants and
contracts. EXPERIMENT cases (ec*, this tool) carry model-derived ASTRA
workloads over the same composition axis — their job is answering the
research questions. Same hardware points, different inputs, different
purpose; never mix the two vocabularies.

One derived LLM trace (with `at=` arrivals and semantic kinds) is replayed over
the composition axis of the comparison table. Pass the generator's
`--workload-manifest` so the FLAT/static boundary follows the selected model's
actual KV base: weights remain below it (HBM side), while
KV/activations/communication remain above it (HBF side). The historical 8 GiB
boundary is no longer guessed: every replay must carry a digest- and
workload-ID-bound generator manifest. EC6 layer
streaming still keeps scratch/metadata in HBM; an exporter without `layer=`
metadata produces one streaming window and therefore no next-layer overlap.
Composition rows then differ by STACK COUNTS serving each side, which is
the decode question: how many flash stacks does the KV sweep want, and
what does shrinking HBM cost the weights stream. Capacities never bind
(footprint ~ GiBs), so this axis is pure bandwidth/latency composition.

Replay defaults to a 512-request closed-loop window on top of `at=` arrivals;
pass `--window 0` only for an intentional open-loop sensitivity run.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from plot_time_breakdown import write_visualization
from time_breakdown_report import (
    SummaryInput,
    build_time_breakdown_report,
    write_time_breakdown_report,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.certificate import (  # noqa: E402
    EXPLORATORY_VALIDATION,
    CertificateError,
    VerifiedCertificate,
    attach_certificate_to_summary_file,
    ensure_exploratory_summary,
    verify_certificate,
)
from validation.contracts import load_json_strict  # noqa: E402

WORKLOAD_ID_RE = re.compile(r"^[0-9a-fA-F]{64}$")
WORKLOAD_ID_PARAMETERS = (
    "model_profile", "model_profile_sha256", "mode",
    "layers", "hidden", "heads", "kv_heads", "ffn", "vocab_size",
    "tie_word_embeddings", "weight_dtype", "weight_bytes",
    "kv_dtype", "kv_bytes", "activation_dtype", "activation_bytes",
    "batch", "prompt_len", "decode_tokens", "max_seq", "tp",
    "wg_per_kernel", "wavefronts",
)
WORKLOAD_ID_SCHEMA = "astra-sim.llm-fine-grained-workload-contract-v2"
WORKLOAD_MANIFEST_SCHEMA = {
    "name": "astra-sim.llm-fine-grained-workload",
    "version": 3,
}
MODEL_PROFILE_SCHEMA = {"name": "astra-sim.llm-model-profile", "version": 1}
PROFILE_ARCHITECTURE_FIELDS = (
    "layers", "hidden", "heads", "kv_heads", "ffn", "vocab_size",
    "tie_word_embeddings", "max_seq",
)
PROFILE_PRECISION_FIELDS = (
    "weight_dtype", "weight_bytes", "kv_dtype", "kv_bytes",
    "activation_dtype", "activation_bytes",
)

CASES = (
    ("ec0", "8×HBM",                      "usecase-baseline.cfg", "all-HBM"),
    ("ec1", "8×HBF",                      "usecase-baseline.cfg", "all-HBF"),
    ("ec2", "6×HBM + 2×HBF (FLAT)",       "usecase-6h2f.cfg", "HBM-HBF-Flat"),
    ("ec3", "4×HBM + 4×HBF (FLAT)",       "usecase-4h4f.cfg", "HBM-HBF-Flat"),
    ("ec4", "2×HBM + 6×HBF (FLAT)",       "usecase-2h6f.cfg", "HBM-HBF-Flat"),
    ("ec5", "2×HBM + 6×HBF (Static-direct, bypass FTL)",
     "usecase-2h6f.cfg", "HBF-static-direct-read"),
    ("ec6", "6×HBM + 2×HBF (Hybrid residency + dual HBM buffers)",
     "usecase-6h2f.cfg",
     "HBM+HBF-layer-streaming"),
)


@dataclass(frozen=True)
class ConfigRun:
    """One isolated simulator invocation and its unpublished summary."""

    config: str
    scenarios: tuple[str, ...]
    canonical_summary: Path
    temporary_summary: Path


@dataclass(frozen=True)
class ConfigResult:
    run: ConfigRun
    summary: dict
    elapsed_seconds: float


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def load_workload_manifest(path: Path) -> tuple[dict, int]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read workload manifest {path}: {error}") from error
    if manifest.get("schema") != WORKLOAD_MANIFEST_SCHEMA:
        raise ValueError("unsupported ASTRA workload manifest schema")
    workload_id = manifest.get("workload_id")
    if not isinstance(workload_id, str) or not WORKLOAD_ID_RE.fullmatch(workload_id):
        raise ValueError("ASTRA workload manifest lacks a valid workload_id")

    def validate_file_record(record: object, label: str) -> Path:
        if not isinstance(record, dict):
            raise ValueError(f"ASTRA workload manifest lacks {label} provenance")
        artifact_path = record.get("path")
        digest = record.get("sha256")
        if (not isinstance(artifact_path, str) or not artifact_path or
                not isinstance(digest, str) or
                not WORKLOAD_ID_RE.fullmatch(digest)):
            raise ValueError(
                f"ASTRA workload manifest has invalid {label} provenance")
        artifact = Path(artifact_path)
        if not artifact.is_file() or sha256_file(artifact) != digest.lower():
            raise ValueError(f"ASTRA workload manifest {label} digest mismatch")
        expected_bytes = record.get("bytes")
        if (expected_bytes is not None and
                (type(expected_bytes) is not int or expected_bytes < 0 or
                 artifact.stat().st_size != expected_bytes)):
            raise ValueError(f"ASTRA workload manifest {label} size mismatch")
        return artifact

    generator = manifest.get("generator")
    validate_file_record(generator, "generator")
    gpu_configuration = manifest.get("gpu_configuration")
    validate_file_record(gpu_configuration, "GPU configuration")
    parameters = manifest.get("parameters")
    if (not isinstance(parameters, dict) or
            set(parameters) != set(WORKLOAD_ID_PARAMETERS)):
        raise ValueError("ASTRA workload manifest lacks normalized parameters")
    for field in (
            "layers", "hidden", "heads", "kv_heads", "ffn", "vocab_size",
            "weight_bytes", "kv_bytes", "activation_bytes", "batch",
            "prompt_len", "max_seq", "tp", "wg_per_kernel", "wavefronts"):
        if type(parameters[field]) is not int or parameters[field] <= 0:
            raise ValueError(
                f"ASTRA workload manifest has invalid parameter {field}")
    if (type(parameters["decode_tokens"]) is not int or
            parameters["decode_tokens"] < 0):
        raise ValueError(
            "ASTRA workload manifest has invalid parameter decode_tokens")
    if type(parameters["tie_word_embeddings"]) is not bool:
        raise ValueError(
            "ASTRA workload manifest has invalid parameter tie_word_embeddings")
    for field in (
            "model_profile", "weight_dtype", "kv_dtype",
            "activation_dtype"):
        if not isinstance(parameters[field], str) or not parameters[field]:
            raise ValueError(
                f"ASTRA workload manifest has invalid parameter {field}")
    profile_digest = parameters["model_profile_sha256"]
    if (not isinstance(profile_digest, str) or
            not WORKLOAD_ID_RE.fullmatch(profile_digest)):
        raise ValueError(
            "ASTRA workload manifest has invalid model profile digest")
    mode = parameters["mode"]
    if mode not in ("prefill", "decode", "prefill+decode", "train"):
        raise ValueError("ASTRA workload manifest has invalid mode")
    if mode in ("decode", "prefill+decode") and parameters["decode_tokens"] == 0:
        raise ValueError(
            "ASTRA decode workload must contain at least one decode token")
    required_tokens = parameters["prompt_len"] + (
        parameters["decode_tokens"]
        if mode in ("decode", "prefill+decode") else 0)
    if required_tokens > parameters["max_seq"]:
        raise ValueError("ASTRA workload exceeds its resolved context length")
    if (parameters["hidden"] % parameters["heads"] != 0 or
            parameters["heads"] % parameters["kv_heads"] != 0):
        raise ValueError("ASTRA workload has invalid attention geometry")
    tp = parameters.get("tp")
    if any(parameters[field] % tp != 0 for field in (
            "hidden", "heads", "kv_heads", "ffn", "vocab_size")):
        raise ValueError("ASTRA workload has invalid TP sharding geometry")

    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError("ASTRA workload manifest lacks model contract")
    profile_record = model.get("profile")
    if (not isinstance(profile_record, dict) or
            set(profile_record) != {
                "name", "path", "bytes", "sha256", "source"}):
        raise ValueError(
            "ASTRA workload manifest has invalid model profile provenance")
    profile_path = validate_file_record(profile_record, "model profile")
    if (profile_record["name"] != parameters["model_profile"] or
            profile_record["sha256"].lower() != profile_digest.lower()):
        raise ValueError(
            "ASTRA workload manifest model profile identity is inconsistent")
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ASTRA model profile: {error}") from error
    if (not isinstance(profile, dict) or
            set(profile) != {
                "schema", "name", "architecture", "precision", "source"} or
            profile.get("schema") != MODEL_PROFILE_SCHEMA or
            profile.get("name") != parameters["model_profile"]):
        raise ValueError("ASTRA workload model profile schema/name is invalid")
    architecture = profile.get("architecture")
    precision = profile.get("precision")
    if (not isinstance(architecture, dict) or
            set(architecture) != set(PROFILE_ARCHITECTURE_FIELDS) or
            not isinstance(precision, dict) or
            set(precision) != set(PROFILE_PRECISION_FIELDS)):
        raise ValueError("ASTRA workload model profile fields are invalid")
    for field in PROFILE_ARCHITECTURE_FIELDS:
        value = architecture[field]
        if field == "tie_word_embeddings":
            valid = type(value) is bool
        else:
            valid = type(value) is int and value > 0
        if not valid:
            raise ValueError(
                f"ASTRA workload model profile architecture.{field} is invalid")
    for field in ("weight_bytes", "kv_bytes", "activation_bytes"):
        if type(precision[field]) is not int or precision[field] <= 0:
            raise ValueError(
                f"ASTRA workload model profile precision.{field} is invalid")
    for field in ("weight_dtype", "kv_dtype", "activation_dtype"):
        if not isinstance(precision[field], str) or not precision[field]:
            raise ValueError(
                f"ASTRA workload model profile precision.{field} is invalid")
    source = profile.get("source")
    if (not isinstance(source, dict) or
            set(source) != {
                "repository", "path", "revision", "upstream_sha256",
                "verification"}):
        raise ValueError("ASTRA workload model profile source is invalid")
    if any(not isinstance(source[field], str) or not source[field]
           for field in ("repository", "path", "revision", "verification")):
        raise ValueError("ASTRA workload model profile source is incomplete")
    upstream_digest = source["upstream_sha256"]
    if (upstream_digest is not None and
            (not isinstance(upstream_digest, str) or
             not WORKLOAD_ID_RE.fullmatch(upstream_digest))):
        raise ValueError(
            "ASTRA workload model profile upstream digest is invalid")
    if profile_record["source"] != source:
        raise ValueError(
            "ASTRA workload manifest/profile source provenance differs")

    resolved_architecture = {
        field: parameters[field] for field in PROFILE_ARCHITECTURE_FIELDS}
    resolved_precision = {
        field: parameters[field] for field in PROFILE_PRECISION_FIELDS}
    if (model.get("resolved_architecture") != resolved_architecture or
            model.get("resolved_precision") != resolved_precision):
        raise ValueError(
            "ASTRA workload resolved model fields differ from its identity")
    profile_defaults = {**architecture, **precision}
    expected_overrides = {
        field: parameters[field]
        for field in (*PROFILE_ARCHITECTURE_FIELDS,
                      *PROFILE_PRECISION_FIELDS)
        if parameters[field] != profile_defaults[field]
    }
    if (model.get("profile_overrides") != expected_overrides or
            model.get("profile_exact") is not (not expected_overrides)):
        raise ValueError(
            "ASTRA workload model profile override accounting is invalid")
    if (model.get("architecture") != "dense_decoder_transformer" or
            model.get("weight_scope") != "full_resident_model"):
        raise ValueError("ASTRA workload has an unsupported model weight scope")

    hidden = parameters["hidden"]
    heads = parameters["heads"]
    kv_heads = parameters["kv_heads"]
    layers = parameters["layers"]
    ffn = parameters["ffn"]
    vocab = parameters["vocab_size"]
    weight_bytes = parameters["weight_bytes"]
    head_dim = hidden // heads
    kv_dim = head_dim * kv_heads
    embedding_elements = vocab * hidden
    qkv_elements = hidden * hidden + 2 * hidden * kv_dim
    projection_elements = hidden * hidden
    mlp_elements = 3 * hidden * ffn
    decoder_parameters = layers * (
        2 * hidden + qkv_elements + projection_elements + mlp_elements)
    model_parameters = (
        embedding_elements + decoder_parameters + hidden +
        (0 if parameters["tie_word_embeddings"] else embedding_elements)
    )
    logical_model_weight_bytes = model_parameters * weight_bytes
    per_rank_embedding = embedding_elements * weight_bytes // tp
    per_rank_input_norms = layers * hidden * weight_bytes
    per_rank_attention = (
        layers * (qkv_elements + projection_elements) * weight_bytes // tp)
    per_rank_post_norms = layers * hidden * weight_bytes
    per_rank_mlp = layers * mlp_elements * weight_bytes // tp
    per_rank_final_norm = hidden * weight_bytes
    per_rank_lm_head = (
        0 if parameters["tie_word_embeddings"] else per_rank_embedding)
    expected_breakdown = {
        "token_embedding": per_rank_embedding,
        "decoder_input_norms": per_rank_input_norms,
        "decoder_attention_matrices": per_rank_attention,
        "decoder_post_attention_norms": per_rank_post_norms,
        "decoder_mlp_matrices": per_rank_mlp,
        "final_norm": per_rank_final_norm,
        "lm_head_resident": per_rank_lm_head,
    }
    per_rank_decoder_weights = (
        per_rank_input_norms + per_rank_attention + per_rank_post_norms +
        per_rank_mlp)
    per_rank_resident_weights = sum(expected_breakdown.values())
    expected_model_scalars = {
        "parameter_count": model_parameters,
        "logical_model_weight_bytes": logical_model_weight_bytes,
        "decoder_block_parameters": decoder_parameters,
        "per_rank_resident_weight_bytes": per_rank_resident_weights,
        "per_rank_decoder_block_weight_bytes": per_rank_decoder_weights,
    }
    if any(model.get(field) != value
           for field, value in expected_model_scalars.items()):
        raise ValueError("ASTRA workload model parameter accounting is invalid")
    if model.get("per_rank_weight_breakdown") != expected_breakdown:
        raise ValueError("ASTRA workload model weight breakdown is invalid")
    if manifest.get("per_rank_weight_bytes") != per_rank_resident_weights:
        raise ValueError("ASTRA workload per-rank weight total is invalid")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != tp:
        raise ValueError("ASTRA workload manifest has incomplete rank artifacts")
    seen_ranks = set()
    for artifact in artifacts:
        validate_file_record(artifact, "ET artifact")
        rank = artifact.get("rank") if isinstance(artifact, dict) else None
        if type(rank) is not int or rank < 0 or rank >= tp or rank in seen_ranks:
            raise ValueError("ASTRA workload manifest has invalid ET ranks")
        for field in (
                "compute_logical_read_bytes",
                "compute_logical_write_bytes",
                "compute_logical_cache_line_requests",
                "stream_layers"):
            value = artifact.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"ASTRA workload manifest has invalid ET {field}")
        seen_ranks.add(rank)
    if seen_ranks != set(range(tp)):
        raise ValueError("ASTRA workload manifest ET ranks are not contiguous")

    collective = manifest.get("mscclpp")
    collective_plan_id = None
    if tp > 1:
        validate_file_record(collective, "MSCCL++ plan")
        collective_plan_id = collective.get("plan_id")
        if (not isinstance(collective_plan_id, str) or
                not WORKLOAD_ID_RE.fullmatch(collective_plan_id)):
            raise ValueError("ASTRA workload manifest has invalid collective plan id")
        try:
            plan = json.loads(Path(collective["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read MSCCL++ plan: {error}") from error
        if plan.get("plan_id") != collective_plan_id:
            raise ValueError("ASTRA workload manifest/plan identities differ")
        if plan.get("schema") != {
                "name": "astra-sim.mscclpp-subset", "version": 1}:
            raise ValueError("unsupported ASTRA MSCCL++ subset plan schema")
        identity_plan = dict(plan)
        identity_plan.pop("plan_id", None)
        expected_plan_id = hashlib.sha256(json.dumps(
            identity_plan, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        if collective_plan_id.lower() != expected_plan_id:
            raise ValueError(
                "ASTRA MSCCL++ plan identity does not match its content")
    elif collective is not None:
        raise ValueError("single-rank ASTRA workload unexpectedly names a collective plan")

    identity_payload = {
        "schema": WORKLOAD_ID_SCHEMA,
        "generator_sha256": generator["sha256"].lower(),
        "gpu_configuration_sha256": gpu_configuration["sha256"].lower(),
        "collective_plan_id": collective_plan_id,
        "parameters": {name: parameters[name]
                       for name in WORKLOAD_ID_PARAMETERS},
    }
    expected_workload_id = hashlib.sha256(json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    if workload_id.lower() != expected_workload_id:
        raise ValueError("ASTRA workload manifest identity does not match its inputs")
    address_space = manifest.get("address_space")
    if not isinstance(address_space, dict):
        raise ValueError("ASTRA workload manifest lacks address_space")
    names = ("weights", "kv", "activations", "communication")
    regions = []
    for name in names:
        region = address_space.get(name)
        if not isinstance(region, dict):
            raise ValueError(f"ASTRA workload manifest lacks {name} region")
        base = region.get("base")
        end = region.get("end")
        if (not isinstance(base, int) or isinstance(base, bool) or
                not isinstance(end, int) or isinstance(end, bool) or
                base < 0 or end <= base or end > (1 << 64)):
            raise ValueError(f"ASTRA workload manifest has invalid {name} bounds")
        regions.append((name, base, end))
    for left, right in zip(regions, regions[1:]):
        if left[2] > right[1]:
            raise ValueError(
                f"ASTRA workload regions overlap: {left[0]} and {right[0]}")
    boundary = address_space["kv"]["base"]
    if address_space["weights"]["end"] > boundary:
        raise ValueError("ASTRA workload weights cross the KV placement boundary")

    weights_region = address_space["weights"]
    if (weights_region.get("logical_bytes") != per_rank_resident_weights or
            weights_region.get("logical_model_bytes") !=
            logical_model_weight_bytes or
            weights_region.get("allocated_bytes") !=
            weights_region["end"] - weights_region["base"]):
        raise ValueError("ASTRA workload weight address accounting is invalid")
    weight_objects = weights_region.get("objects")
    if (not isinstance(weight_objects, dict) or
            set(weight_objects) != {
                "token_embedding", "decoder_layers", "final_norm", "lm_head"}):
        raise ValueError("ASTRA workload lacks complete weight object addresses")

    def validate_weight_object(name: str) -> dict:
        record = weight_objects[name]
        if not isinstance(record, dict):
            raise ValueError(f"ASTRA workload has invalid {name} weight object")
        base = record.get("base")
        end = record.get("end")
        if (type(base) is not int or type(end) is not int or
                base < weights_region["base"] or end <= base or
                end > weights_region["end"]):
            raise ValueError(f"ASTRA workload has invalid {name} weight bounds")
        return record

    token_embedding = validate_weight_object("token_embedding")
    decoder_layers = validate_weight_object("decoder_layers")
    final_norm = validate_weight_object("final_norm")
    lm_head = validate_weight_object("lm_head")
    per_layer_decoder_weights = per_rank_decoder_weights // layers
    if (token_embedding.get("bytes") != per_rank_embedding or
            token_embedding["end"] - token_embedding["base"] !=
            per_rank_embedding or
            token_embedding["base"] != weights_region["base"] or
            token_embedding["end"] > decoder_layers["base"]):
        raise ValueError("ASTRA workload token-embedding address is invalid")
    layer_stride = decoder_layers.get("layer_stride")
    if (type(layer_stride) is not int or
            layer_stride < per_layer_decoder_weights or
            decoder_layers.get("logical_bytes_per_layer") !=
            per_layer_decoder_weights or
            decoder_layers.get("layers") != layers or
            decoder_layers["end"] !=
            decoder_layers["base"] + layers * layer_stride or
            weights_region.get("layer_stride") != layer_stride):
        raise ValueError("ASTRA workload decoder-layer addresses are invalid")
    if (final_norm.get("bytes") != per_rank_final_norm or
            final_norm["end"] - final_norm["base"] != per_rank_final_norm or
            final_norm["base"] != decoder_layers["end"]):
        raise ValueError("ASTRA workload final-norm address is invalid")
    expected_lm_head_bytes = per_rank_embedding
    if (lm_head.get("bytes") != expected_lm_head_bytes or
            lm_head["end"] - lm_head["base"] != expected_lm_head_bytes):
        raise ValueError("ASTRA workload LM-head address is invalid")
    if parameters["tie_word_embeddings"]:
        if (lm_head.get("alias_of") != "token_embedding" or
                lm_head["base"] != token_embedding["base"] or
                lm_head["end"] != token_embedding["end"] or
                weights_region["end"] != final_norm["end"]):
            raise ValueError("ASTRA workload tied LM-head alias is invalid")
    elif (lm_head.get("alias_of") is not None or
          not (token_embedding["end"] <= decoder_layers["base"] and
               decoder_layers["end"] <= final_norm["base"] and
               final_norm["end"] <= lm_head["base"]) or
          weights_region["end"] != lm_head["end"]):
        raise ValueError("ASTRA workload untied weight objects overlap")

    kv_step = (
        parameters["batch"] * 2 * kv_dim * parameters["kv_bytes"] // tp)
    kv_capacity = layers * parameters["max_seq"] * kv_step
    kv_region = address_space["kv"]
    kv_layer_stride = kv_region.get("layer_stride")
    if (kv_region.get("bytes_per_batch_position_per_layer") != kv_step or
            kv_region.get("batch") != parameters["batch"] or
            kv_region.get("logical_capacity_bytes") != kv_capacity or
            manifest.get("per_rank_kv_capacity_bytes") != kv_capacity or
            type(kv_layer_stride) is not int or
            kv_layer_stride < parameters["max_seq"] * kv_step or
            kv_region["end"] != kv_region["base"] + layers * kv_layer_stride or
            kv_region.get("allocated_bytes") !=
            kv_region["end"] - kv_region["base"]):
        raise ValueError("ASTRA workload KV address accounting is invalid")

    initial_state = manifest.get("initial_state")
    initial_kv = (
        initial_state.get("kv") if isinstance(initial_state, dict) else None)
    if not isinstance(initial_kv, dict):
        raise ValueError("ASTRA workload lacks its initial KV state")
    decode_only = mode == "decode"
    initial_prompt = parameters["prompt_len"] if decode_only else 0
    initial_per_layer = initial_prompt * kv_step
    expected_initial_kv = {
        "state_at_trace_start":
            "preexisting_prompt" if decode_only else "empty",
        "required_at_trace_start": decode_only,
        "kind": "shared_context",
        "prompt_positions_per_sequence": initial_prompt,
        "batch": parameters["batch"],
        "bytes_per_layer": initial_per_layer,
        "total_bytes_per_rank": initial_per_layer * layers,
        "base": kv_region["base"],
        "layer_stride": kv_layer_stride,
    }
    if initial_kv != expected_initial_kv:
        raise ValueError("ASTRA workload initial KV state is invalid")
    return manifest, boundary


def inspect_trace(path: Path) -> dict:
    digest = hashlib.sha256()
    header = None
    requests = 0
    byte_count = 0
    try:
        with path.open("rb") as handle:
            for line_number, raw_bytes in enumerate(handle, start=1):
                digest.update(raw_bytes)
                try:
                    raw = raw_bytes.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"ASTRA trace line {line_number} is not UTF-8") from error
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if "schema=astra-sim.hbfsim-memory-trace" not in line:
                        continue
                    if header is not None:
                        raise ValueError("ASTRA HBF trace has duplicate schema headers")
                    fields = {}
                    for token in line[1:].split():
                        if "=" in token:
                            key, value = token.split("=", 1)
                            if key in fields:
                                raise ValueError(
                                    f"duplicate {key} in ASTRA trace header")
                            fields[key] = value
                    if fields.get("schema") != "astra-sim.hbfsim-memory-trace":
                        continue
                    if fields.get("version") != "2":
                        raise ValueError("unsupported ASTRA HBF trace schema version")
                    workload_id = fields.get("workload_id", "")
                    if not WORKLOAD_ID_RE.fullmatch(workload_id):
                        raise ValueError(
                            "ASTRA HBF trace header lacks a valid workload_id")
                    try:
                        expected_requests = int(fields.get("requests", ""), 10)
                        expected_bytes = int(fields.get("bytes", ""), 10)
                    except ValueError as error:
                        raise ValueError(
                            "ASTRA HBF trace header has invalid census fields") from error
                    if expected_requests < 0 or expected_bytes < 0:
                        raise ValueError(
                            "ASTRA HBF trace header has negative census fields")
                    header = {
                        "workload_id": workload_id.lower(),
                        "requests": expected_requests,
                        "bytes": expected_bytes,
                    }
                    continue
                fields = line.split("#", 1)[0].split()
                if not fields:
                    continue
                if header is None:
                    raise ValueError(
                        "ASTRA HBF trace schema header must precede its body")
                if len(fields) < 3 or fields[1] not in {"R", "W"}:
                    raise ValueError(
                        f"ASTRA trace line {line_number} is not normalized")
                try:
                    request_bytes = int(fields[2], 0)
                except ValueError as error:
                    raise ValueError(
                        f"ASTRA trace line {line_number} has invalid bytes") from error
                if request_bytes <= 0:
                    raise ValueError(
                        f"ASTRA trace line {line_number} has nonpositive bytes")
                requests += 1
                byte_count += request_bytes
    except OSError as error:
        raise ValueError(f"cannot read ASTRA trace {path}: {error}") from error
    if header is None:
        raise ValueError("ASTRA HBF trace lacks its schema-v2 provenance header")
    if requests != header["requests"] or byte_count != header["bytes"]:
        raise ValueError(
            "ASTRA HBF trace body does not match its request/byte census")
    header["sha256"] = digest.hexdigest()
    return header


def run_config(
        run: ConfigRun, scenario_compare: Path, config_dir: Path,
        trace: Path, window: int, boundary: int,
        validation_certificate: VerifiedCertificate | None = None,
) -> ConfigResult:
    """Run and validate one config without mutating shared aggregation state."""
    cmd = [str(scenario_compare.resolve()),
           "--config", str((config_dir / run.config).resolve()),
           "--trace", str(trace.resolve()),
           "--scenarios", ",".join(run.scenarios),
           "--flat-hbm-bytes", str(boundary),
           "--static-direct-hbm-bytes", str(boundary),
           "--max-outstanding-requests", str(window),
           "--summary-json", str(run.temporary_summary)]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed_seconds = time.monotonic() - started
    if proc.returncode != 0:
        detail = (f"stderr:\n{proc.stderr[-2000:].strip()}\n"
                  f"stdout tail:\n{proc.stdout[-2000:].strip()}")
        raise RuntimeError(
            f"scenario_compare exited {proc.returncode}:\n{detail}")
    if not run.temporary_summary.exists():
        raise RuntimeError(f"no summary written: {run.temporary_summary}")
    if validation_certificate is None:
        data = load_json_strict(run.temporary_summary)
        ensure_exploratory_summary(data)
    else:
        data = attach_certificate_to_summary_file(
            run.temporary_summary, validation_certificate)
    if data.get("schema") != {
            "name": "hbfsim.scenario_compare.summary", "version": 16}:
        raise RuntimeError("unsupported or missing scenario summary schema")
    if data.get("sanity") != "PASS":
        raise RuntimeError(
            f"summary reports SANITY={data.get('sanity')!r}")
    written_scenarios = {
        block.get("name") for block in data.get("scenarios", [])}
    if written_scenarios != set(run.scenarios):
        raise RuntimeError(
            f"summary scenarios {sorted(written_scenarios)} do not match "
            f"requested {sorted(run.scenarios)}")
    return ConfigResult(run, data, elapsed_seconds)


def select_cases(spec: str) -> tuple[tuple[str, str, str, str], ...]:
    """Return requested cases in canonical order, rejecting typos/duplicates."""
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    if not requested:
        raise argparse.ArgumentTypeError("--cases must name at least one case")
    if len(set(requested)) != len(requested):
        raise argparse.ArgumentTypeError("--cases contains a duplicate case")
    known = {case[0] for case in CASES}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown case(s): {','.join(unknown)}; choose from {','.join(sorted(known))}")
    selected = set(requested)
    return tuple(case for case in CASES if case[0] in selected)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--workload-manifest", type=Path, required=True,
        help=("ASTRA generator manifest; binds workload provenance and derives "
              "the KV placement boundary"))
    parser.add_argument(
        "--boundary", type=nonnegative_int,
        help="explicit FLAT/static boundary (decimal or 0x...); overrides manifest")
    parser.add_argument("--scenario-compare", type=Path,
                        default=Path("build/scenario_compare"))
    parser.add_argument("--config-dir", type=Path,
                        default=Path("configs/scenario_compare"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/astra-replay-w512"))
    parser.add_argument("--window", type=int, default=512,
                        help="closed-loop 4 KiB page transactions "
                             "(0 = intentional open loop)")
    parser.add_argument(
        "--jobs", type=positive_int, default=1,
        help=("independent config processes to run concurrently (default: 1; "
              "choose conservatively because each process owns simulator and "
              "trace state)"),
    )
    parser.add_argument(
        "--cases",
        default=",".join(case[0] for case in CASES),
        help="comma-separated experiment cases to run (default: ec0,...,ec6)",
    )
    parser.add_argument(
        "--validation-certificate",
        type=Path,
        help=(
            "attach a verified foundational certificate; omitted runs remain "
            "explicitly exploratory"
        ),
    )
    args = parser.parse_args()
    if args.window < 0:
        parser.error("--window must be non-negative")
    validation_certificate: VerifiedCertificate | None = None
    if args.validation_certificate is not None:
        try:
            validation_certificate = verify_certificate(
                args.validation_certificate,
                repository=ROOT,
                scenario_compare=args.scenario_compare,
            )
        except CertificateError as error:
            parser.error(f"invalid validation certificate: {error}")

    # Serialize the entire replay transaction, including provenance and trace
    # preflight.  A schema-v2 trace may be hundreds of GiB, so acquiring this
    # lock after inspection would let two writers spend hours scanning the
    # same input before one of them is rejected at publication time.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.out_dir / ".astra-replay.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"another replay is already using {args.out_dir}")
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"requested_cases={args.cases}\n")
    lock_handle.flush()

    try:
        workload_manifest, manifest_boundary = load_workload_manifest(
            args.workload_manifest)
    except ValueError as error:
        parser.error(str(error))
    try:
        trace_contract = inspect_trace(args.trace)
    except ValueError as error:
        parser.error(str(error))
    if trace_contract["workload_id"] != workload_manifest["workload_id"].lower():
        parser.error(
            "ASTRA trace workload_id does not match --workload-manifest")
    if args.boundary is not None:
        boundary = args.boundary
        boundary_source = "explicit"
    elif manifest_boundary is not None:
        boundary = manifest_boundary
        boundary_source = "workload_manifest.kv.base"
    try:
        cases = select_cases(args.cases)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    case_suffix = "" if len(cases) == len(CASES) else "-" + "-".join(
        case[0] for case in cases)
    artifact_stem = f"astra-replay{case_suffix}"
    out_csv = args.out_dir / f"{artifact_stem}.csv"
    time_csv = args.out_dir / f"{artifact_stem}.time-breakdown.csv"
    time_markdown = args.out_dir / f"{artifact_stem}.time-breakdown.md"
    time_html = args.out_dir / f"{artifact_stem}.time-breakdown.html"
    manifest_path = args.out_dir / f"{artifact_stem}.manifest.json"
    # The manifest is the selected case-set commit record. Withdraw it and
    # its derived tables while holding the directory lock so a failed rerun
    # cannot leave an old generation looking current. Canonical summaries are
    # replaced only after every simulator invocation and timing contract pass.
    for path in (manifest_path, out_csv, time_csv, time_markdown, time_html):
        path.unlink(missing_ok=True)

    by_config: dict[str, list[str]] = {}
    for _uc, _name, config, scenario in cases:
        by_config.setdefault(config, [])
        if scenario not in by_config[config]:
            by_config[config].append(scenario)

    blocks: dict[tuple[str, str], dict] = {}
    summary_inputs: list[SummaryInput] = []
    pending_summaries: dict[Path, Path] = {}
    config_runs: list[ConfigRun] = []
    for config, scenarios in by_config.items():
        summary = (args.out_dir / f"{Path(config).stem}.summary.json").resolve()
        temporary_summary = summary.with_name(
            f"{summary.name}.{uuid.uuid4().hex}.tmp")
        pending_summaries[summary] = temporary_summary
        config_runs.append(ConfigRun(
            config=config,
            scenarios=tuple(scenarios),
            canonical_summary=summary,
            temporary_summary=temporary_summary,
        ))

    try:
        results: dict[str, ConfigResult] = {}
        failures: list[tuple[str, Exception]] = []
        worker_count = min(args.jobs, len(config_runs))
        print(
            f"[parallel] {len(config_runs)} config(s), jobs={worker_count}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Future[ConfigResult], ConfigRun] = {}
            for run in config_runs:
                print(
                    f"[queued] {run.config}: {','.join(run.scenarios)}",
                    flush=True,
                )
                future = executor.submit(
                    run_config, run, args.scenario_compare, args.config_dir,
                    args.trace, args.window, boundary,
                    validation_certificate)
                futures[future] = run
            for future in as_completed(futures):
                run = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # Report all config failures together.
                    failures.append((run.config, error))
                    print(f"[failed] {run.config}: {error}", flush=True)
                else:
                    results[run.config] = result
                    print(
                        f"[done] {run.config}: "
                        f"{result.elapsed_seconds:.2f}s",
                        flush=True,
                    )
        if failures:
            details = "\n".join(
                f"{config}: {error}" for config, error in failures)
            raise RuntimeError(
                "one or more config replays failed; no summaries published:\n"
                f"{details}")

        # Aggregate in canonical config order. Workers only own their temp file;
        # all shared dictionaries and report inputs are built on this thread.
        for run in config_runs:
            result = results[run.config]
            data = result.summary
            for block in data["scenarios"]:
                blocks[(run.config, block["name"])] = block

            scenario_metadata = {
                scenario: {"case": case_id, "config": run.config}
                for case_id, _name, case_config, scenario in cases
                if case_config == run.config
            }
            summary_inputs.append(SummaryInput(
                label=Path(run.config).stem,
                summary=data,
                source=str(run.canonical_summary),
                scenario_metadata=scenario_metadata,
            ))

        # The shared validator is stricter than the small console table below:
        # reject an incomplete or internally inconsistent timing contract
        # before replacing any previously published canonical summary.
        build_time_breakdown_report(summary_inputs)

        # Publish only after every selected config completed and validated, so
        # a later failure cannot leave a mixed old/new experiment generation.
        for summary, temporary_summary in pending_summaries.items():
            temporary_summary.replace(summary)
        pending_summaries.clear()
    finally:
        for temporary_summary in pending_summaries.values():
            temporary_summary.unlink(missing_ok=True)

    # Publish one WAF only: HBF media payload writes divided by logical writes
    # accepted by HBF. Raw counters remain in the summary for diagnosis, but
    # alternate denominators must not appear as competing WAF definitions.
    # A missing denominator is rendered as "-", never as zero.
    print(f"\n{'case':5s} {'组成':26s} {'前端完成[GB/s]':>12s} {'含排空[GB/s]':>12s} {'排空尾[ms]':>10s} "
          f"{'平均延迟[µs]':>11s} {'p95[µs]':>9s} {'HBM次':>9s} {'HBF次':>9s} "
          f"{'ECC等[ms]':>10s} {'ECC util[%]':>11s} {'ECC par':>8s} {'ECC飞行':>7s} "
          f"{'WAF':>10s} "
          f"{'programs':>9s} {'data':>9s} {'sanity'}")
    csv_rows = [["case", "composition", "user_completion_throughput_GBps",
                 "makespan_throughput_GBps", "drain_tail_ms", "avg_us",
                 "p95_us", "hbm_accesses", "hbf_accesses",
                 "hbf_ecc_queue_wait_work_ns", "hbf_ecc_issue_utilization",
                 "hbf_ecc_issue_parallelism", "hbf_ecc_max_inflight_per_die",
                 "waf",
                 "cooperative_user_write_bytes",
                 "cooperative_destaged_bytes",
                 "page_programs",
                 "data_programs",
                 "layer_streaming_layers",
                 "layer_streaming_streamed_pages",
                 "layer_streaming_streamed_bytes",
                 "layer_streaming_dirty_pages_written_back",
                 "layer_streaming_writeback_bytes",
                 "layer_streaming_backing_request_credit_limit",
                 "layer_streaming_backing_max_inflight_requests",
                 "layer_streaming_backing_admission_waited_requests",
                 "layer_streaming_backing_admission_wait_work_ns",
                 "layer_streaming_user_waited_ops",
                 "layer_streaming_user_wait_work_ns",
                 "layer_streaming_exposed_prefetch_ns",
                 "layer_streaming_hidden_prefetch_ns",
                 "layer_streaming_buffer_reuse_wait_work_ns",
                 "warnings"]]
    for uc, name, config, scenario in cases:
        s = blocks[(config, scenario)]
        time = s["time_breakdown"]
        wall = time["wall_clock_ns"]
        lat = time["latency_work"]
        hbf_work = time["stage_work"].get("hbf") or {}
        streaming_work = time["stage_work"]["layer_streaming_controller"]
        ecc_resource = time["resource_busy"]["hbf_ecc_issue"]
        hbf = s.get("hbf_stats") or {}
        hybrid = s.get("hybrid_path") or {}
        streaming = s.get("layer_streaming") or {}
        warn = len(s.get("warnings", []))
        waf = hbf.get("waf")
        cooperative_user_bytes = hybrid.get("hbm_write_buffer_user_write_bytes", 0)
        cooperative_destaged_bytes = hybrid.get(
            "hbm_write_buffer_destaged_bytes", 0)
        programs = hbf.get("page_programs", 0)
        data_programs = hbf.get("data_programs", 0)
        ecc_wait_ns = hbf_work.get("ecc_queue_wait_work_ns", 0.0)
        ecc_util = ecc_resource["utilization"]
        ecc_parallelism = (
            ecc_resource["busy_ns"] / ecc_resource["active_span_ns"]
            if ecc_resource["active_span_ns"] > 0.0 else 0.0)
        ecc_max_inflight = hbf.get("ecc_max_inflight_per_die", 0)
        waf_text = f"{waf:.2f}" if waf is not None else "-"
        print(f"{uc:5s} {name:26s} {s['user_completion_throughput_GBps']:12.2f} "
              f"{s['makespan_throughput_GBps']:12.2f} "
              f"{wall['drain_tail_ns']/1e6:10.3f} "
              f"{lat['average_ns']/1e3:11.2f} "
              f"{lat['p95_ns']/1e3:9.2f} {s['hbm_accesses']:9d} "
              f"{s['hbf_accesses']:9d} {ecc_wait_ns/1e6:10.3f} "
              f"{ecc_util * 100.0:11.3f} {ecc_parallelism:8.2f} {ecc_max_inflight:7d} "
              f"{waf_text:>10s} "
              f"{programs:9d} {data_programs:9d} {warn}")
        csv_rows.append([
            uc, name, f"{s['user_completion_throughput_GBps']:.3f}",
            f"{s['makespan_throughput_GBps']:.3f}",
            f"{wall['drain_tail_ns']/1e6:.4f}",
            f"{lat['average_ns']/1e3:.2f}",
            f"{lat['p95_ns']/1e3:.2f}", s["hbm_accesses"],
            s["hbf_accesses"], f"{ecc_wait_ns:.6f}", f"{ecc_util:.9f}",
            f"{ecc_parallelism:.6f}", ecc_max_inflight,
            waf_text, cooperative_user_bytes,
            cooperative_destaged_bytes, programs, data_programs,
            streaming.get("layers", 0),
            streaming.get("streamed_pages", 0),
            streaming.get("streamed_bytes", 0),
            streaming.get("dirty_pages_written_back", 0),
            streaming.get("writeback_bytes", 0),
            streaming.get("backing_request_credit_limit", 0),
            streaming.get("backing_max_inflight_requests", 0),
            streaming.get("backing_admission_waited_requests", 0),
            f"{streaming_work['backing_admission_wait_work_ns']:.6f}",
            streaming.get("user_waited_ops", 0),
            f"{streaming_work['user_wait_work_ns']:.6f}",
            f"{streaming_work['exposed_prefetch_ns']:.6f}",
            f"{streaming_work['hidden_prefetch_ns']:.6f}",
            f"{streaming_work['buffer_reuse_wait_work_ns']:.6f}",
            warn,
        ])
    temporary = out_csv.with_name(f"{out_csv.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(csv_rows)
    temporary.replace(out_csv)
    write_time_breakdown_report(summary_inputs, time_csv, time_markdown)
    write_visualization(
        summary_inputs,
        time_html,
        title="HBFSim ASTRA replay timing",
    )
    summary_paths = [
        (args.out_dir / f"{Path(config).stem}.summary.json").resolve()
        for config in by_config
    ]
    artifacts = summary_paths + [
        out_csv.resolve(), time_csv.resolve(), time_markdown.resolve(),
        time_html.resolve()]
    manifest = {
        "schema": {"name": "hbfsim.astra_replay.manifest", "version": 3},
        "validation": (
            validation_certificate.summary_block()
            if validation_certificate is not None
            else EXPLORATORY_VALIDATION
        ),
        "run_id": uuid.uuid4().hex,
        "cases": [case[0] for case in cases],
        "window": args.window,
        "jobs": worker_count,
        "boundary": boundary,
        "boundary_source": boundary_source,
        "workload_manifest": {
            "path": str(args.workload_manifest.resolve()),
            "sha256": sha256_file(args.workload_manifest),
            "workload_id": workload_manifest["workload_id"].lower(),
            "kv_base": manifest_boundary,
        },
        "trace": {
            "path": str(args.trace.resolve()),
            "sha256": trace_contract["sha256"],
            "workload_id": trace_contract["workload_id"],
            "requests": trace_contract["requests"],
            "bytes": trace_contract["bytes"],
        },
        "artifacts": {
            path.name: {"sha256": sha256_file(path)} for path in artifacts
        },
    }
    temporary_manifest = manifest_path.with_name(
        f"{manifest_path.name}.{uuid.uuid4().hex}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary_manifest.replace(manifest_path)
    print(f"wrote: {out_csv}")
    print(f"wrote: {time_csv}")
    print(f"wrote: {time_markdown}")
    print(f"wrote: {time_html}")
    print(f"wrote: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
