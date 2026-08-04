#!/usr/bin/env python3
"""Prepare the fail-closed Kimi K3 TP8 capacity-only preflight.

This tool reproduces the published per-rank loader census, then compares a
physical all-HBM layout with a conservative hybrid layout that keeps all
non-routed weights, known 1M runtime state, explicit runtime overhead, and two
full routed-layer TP8 shard buffers in HBM.  All routed-expert weights remain
on the backing tier.  The output answers only whether the bytes fit; dynamic
memory service remains deliberately undetermined.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DESCRIPTOR = (
    REPOSITORY_ROOT / "configs/workloads/kimi-k3-tp8-capacity.json"
)
DEFAULT_STUDY = (
    REPOSITORY_ROOT / "configs/studies/kimi-k3-hbf-questions.json"
)
DESCRIPTOR_SCHEMA = {
    "name": "hbfsim.kimi_k3_tp8_capacity",
    "version": 1,
}
STUDY_SCHEMA = {
    "name": "hbfsim.kimi_k3_hbf_research_questions",
    "version": 1,
}
RECEIPT_SCHEMA = {
    "name": "hbfsim.kimi_k3_capacity_preflight",
    "version": 1,
}
GIB = 1024**3
EXPECTED_HBM_GIB = (96, 192, 288)
EXPECTED_OVERHEAD_GIB = (8, 16, 32)
EXPECTED_QUESTION_IDS = ("Q1_scale_in", "Q2_expert_delivery", "Q3_mutable_state")


class KimiK3PreflightError(ValueError):
    """The Kimi K3 capacity contract or its arithmetic is invalid."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise KimiK3PreflightError(
            f"cannot read {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise KimiK3PreflightError(f"{description} must be a JSON object")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise KimiK3PreflightError(f"{name} must be an object")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KimiK3PreflightError(f"{name} must be a positive integer")
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
        raise KimiK3PreflightError(f"{description} is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise KimiK3PreflightError("ceil_div requires non-negative/positive inputs")
    return (numerator + denominator - 1) // denominator


def _validate_descriptor(path: Path) -> dict[str, Any]:
    descriptor = _load_object(path, "Kimi K3 capacity descriptor")
    if descriptor.get("schema") != DESCRIPTOR_SCHEMA:
        raise KimiK3PreflightError("Kimi K3 descriptor schema drifted")
    intended = _mapping(descriptor.get("intended_use"), "intended_use")
    if (
        intended.get("paper_claim_eligible_by_itself") is not False
        or intended.get("allowed_claim")
        != "vendor_accounting_reproduced_and_capacity_feasibility_sensitivity"
        or set(intended.get("forbidden_claims", []))
        != {
            "memory_system_service_feasible",
            "single_node_deployable",
            "ttft",
            "tpot",
            "slo",
            "token_throughput",
        }
    ):
        raise KimiK3PreflightError("Kimi K3 descriptor claim boundary drifted")
    model = _mapping(descriptor.get("model"), "model")
    architecture = _mapping(model.get("architecture"), "model.architecture")
    expected_architecture = {
        "total_parameters_reported": 2_780_000_000_000,
        "active_parameters_per_token_reported": 104_200_000_000,
        "decoder_layers": 93,
        "dense_layers": 1,
        "moe_layers": 92,
        "kda_layers": 69,
        "mla_layers": 24,
        "hidden_size": 7168,
        "latent_moe_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "routed_experts": 896,
        "selected_experts_per_token": 16,
        "shared_experts": 2,
        "max_context_tokens": 1_048_576,
    }
    if architecture != expected_architecture:
        raise KimiK3PreflightError("Kimi K3 architecture contract drifted")
    if (
        model.get("revision") != "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
        or model.get("config_sha256")
        != "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
    ):
        raise KimiK3PreflightError("Kimi K3 source identity drifted")
    accounting = _mapping(
        descriptor.get("vendor_loader_accounting"),
        "vendor_loader_accounting",
    )
    categories = _mapping(
        accounting.get("tp8_per_rank_weight_categories_bytes"),
        "tp8_per_rank_weight_categories_bytes",
    )
    expected_categories = {
        "dense_mlp": 182_000_000,
        "routed_expert_packed_values_and_scales": 180_807_000_000,
        "shared_expert": 3_039_000_000,
        "kda_attention_gemm": 7_763_000_000,
        "mla_attention_gemm": 2_029_000_000,
        "norm": 6_000_000,
        "other_weights": 11_231_000_000,
    }
    if categories != expected_categories:
        raise KimiK3PreflightError("Kimi K3 TP8 weight categories drifted")
    total = _positive_integer(
        accounting.get("tp8_per_rank_weight_total_bytes"),
        "TP8 per-rank weight total",
    )
    if sum(categories.values()) != total or total != 205_057_000_000:
        raise KimiK3PreflightError("Kimi K3 TP8 weight category sum does not conserve")
    if (
        accounting.get("full_checkpoint_physical_storage_bytes")
        != 1_560_860_000_000
        or accounting.get("full_checkpoint_routed_expert_bytes")
        != 1_446_456_000_000
        or accounting.get("tp8_per_rank_known_1m_runtime_state_bytes")
        != 15_491_000_000
    ):
        raise KimiK3PreflightError("Kimi K3 vendor loader census drifted")
    sensitivity = _mapping(
        descriptor.get("capacity_sensitivity"),
        "capacity_sensitivity",
    )
    if (
        sensitivity.get("tensor_parallel_degree") != 8
        or sensitivity.get("accelerators_per_single_node_domain") != 8
        or sensitivity.get("hbm_capacities_gib_per_rank")
        != list(EXPECTED_HBM_GIB)
        or sensitivity.get("runtime_overhead_gib_per_rank")
        != list(EXPECTED_OVERHEAD_GIB)
    ):
        raise KimiK3PreflightError("Kimi K3 capacity axes drifted")
    policy = _mapping(
        sensitivity.get("hybrid_capacity_policy"),
        "hybrid_capacity_policy",
    )
    if (
        policy.get("active_layer_buffer_count") != 2
        or policy.get("routed_layer_bytes_derivation")
        != "tp8_routed_expert_bytes/92"
        or policy.get("capacity_interpretation")
        != "conservative_route_independent_fit_test"
        or policy.get("service_interpretation")
        != "undetermined_without_router_trace_and_memory_replay"
    ):
        raise KimiK3PreflightError("Kimi K3 hybrid capacity policy drifted")
    return descriptor


def _validate_study(path: Path, descriptor_path: Path) -> dict[str, Any]:
    study = _load_object(path, "Kimi K3 research study")
    if study.get("schema") != STUDY_SCHEMA:
        raise KimiK3PreflightError("Kimi K3 study schema drifted")
    raw_descriptor = study.get("model_descriptor")
    if not isinstance(raw_descriptor, str) or not raw_descriptor:
        raise KimiK3PreflightError("Kimi K3 study descriptor path is missing")
    linked = (path.parent / raw_descriptor).resolve()
    if linked != descriptor_path.resolve():
        raise KimiK3PreflightError("Kimi K3 study binds a different descriptor")
    questions = study.get("questions")
    if not isinstance(questions, list) or tuple(
        question.get("id") if isinstance(question, dict) else None
        for question in questions
    ) != EXPECTED_QUESTION_IDS:
        raise KimiK3PreflightError("Kimi K3 study must contain exactly Q1/Q2/Q3")
    common = _mapping(study.get("common_contract"), "common_contract")
    if (
        common.get("canonical_transaction_fields")
        != ["address", "op", "bytes"]
        or common.get("same_logical_trace_different_placement") is not True
        or common.get("hbm_hbf_means_hybrid_not_all_hbf") is not True
        or common.get("pure_all_hbf_isolated_bound_only") is not True
    ):
        raise KimiK3PreflightError("Kimi K3 common transaction contract drifted")
    gates = _mapping(study.get("evidence_gates"), "evidence_gates")
    if (
        gates.get("static_capacity_preflight_before_dynamic_replay") is not True
        or gates.get("long_span_fast_path_requires_event_equivalence") is not True
        or gates.get("real_router_trace_required_for_k3_expert_claim") is not True
        or gates.get("dummy_compute_timing_claim_scope")
        != "memory_system_service_only"
        or gates.get("ttft_tpot_slo_require_calibrated_compute_communication")
        is not True
        or gates.get("canonical_waf")
        != "physical_write_bytes/logical_write_bytes"
        or gates.get("zero_logical_write_waf") is not None
    ):
        raise KimiK3PreflightError("Kimi K3 evidence gates drifted")
    return study


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


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
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


def prepare_preflight(
    *,
    descriptor_path: Path,
    study_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    descriptor_path = descriptor_path.resolve()
    study_path = study_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise KimiK3PreflightError("Kimi K3 preflight output directory must be empty")
    descriptor = _validate_descriptor(descriptor_path)
    _validate_study(study_path, descriptor_path)
    accounting = descriptor["vendor_loader_accounting"]
    categories = accounting["tp8_per_rank_weight_categories_bytes"]
    weights = accounting["tp8_per_rank_weight_total_bytes"]
    routed = categories["routed_expert_packed_values_and_scales"]
    non_routed = weights - routed
    known_state = accounting["tp8_per_rank_known_1m_runtime_state_bytes"]
    moe_layers = descriptor["model"]["architecture"]["moe_layers"]
    active_buffer_count = descriptor["capacity_sensitivity"][
        "hybrid_capacity_policy"
    ]["active_layer_buffer_count"]
    routed_layer_buffer_bytes = _ceil_div(routed, moe_layers)
    active_buffers_bytes = routed_layer_buffer_bytes * active_buffer_count
    hybrid_base_bytes = non_routed + known_state + active_buffers_bytes
    all_hbm_known_bytes = weights + known_state
    rows: list[dict[str, Any]] = []
    for hbm_gib in EXPECTED_HBM_GIB:
        hbm_bytes = hbm_gib * GIB
        for overhead_gib in EXPECTED_OVERHEAD_GIB:
            overhead_bytes = overhead_gib * GIB
            all_hbm_required = all_hbm_known_bytes + overhead_bytes
            hybrid_required = hybrid_base_bytes + overhead_bytes
            rows.append(
                {
                    "hbm_capacity_gib_per_rank": hbm_gib,
                    "runtime_overhead_gib_per_rank": overhead_gib,
                    "all_hbm_known_bytes": all_hbm_known_bytes,
                    "all_hbm_required_bytes": all_hbm_required,
                    "all_hbm_headroom_bytes": hbm_bytes - all_hbm_required,
                    "all_hbm_capacity_feasible": hbm_bytes >= all_hbm_required,
                    "hybrid_hbm_base_bytes": hybrid_base_bytes,
                    "hybrid_hbm_required_bytes": hybrid_required,
                    "hybrid_hbm_headroom_bytes": hbm_bytes - hybrid_required,
                    "hybrid_capacity_feasible": hbm_bytes >= hybrid_required,
                    "backing_required_bytes": routed,
                    "backing_required_gib_ceil": _ceil_div(routed, GIB),
                    "memory_service_feasible": "undetermined",
                }
            )
    table_path = output_dir / "capacity.csv"
    _write_csv_atomic(table_path, rows)
    receipt_path = output_dir / "preflight.receipt.json"
    payload = {
        "schema": RECEIPT_SCHEMA,
        "result": "pass",
        "source": {
            "descriptor": _snapshot(descriptor_path, "Kimi K3 descriptor"),
            "study": _snapshot(study_path, "Kimi K3 study"),
            "tool": _snapshot(Path(__file__), "Kimi K3 preflight tool"),
        },
        "reproduced_vendor_accounting": {
            "tp8_per_rank_weight_bytes": weights,
            "tp8_per_rank_known_1m_runtime_state_bytes": known_state,
            "tp8_per_rank_known_total_bytes": all_hbm_known_bytes,
            "reported_known_total_gib": accounting["reported_tp8_gib"][
                "known_total"
            ],
        },
        "hybrid_capacity_policy": {
            "resident_non_routed_weight_bytes": non_routed,
            "routed_layer_buffer_bytes_per_slot": routed_layer_buffer_bytes,
            "active_layer_buffer_count": active_buffer_count,
            "active_layer_buffers_bytes": active_buffers_bytes,
            "known_1m_runtime_state_bytes": known_state,
            "hybrid_hbm_base_bytes_before_runtime_overhead": hybrid_base_bytes,
            "routed_expert_backing_bytes": routed,
            "route_trace_required_for_capacity": False,
            "route_trace_required_for_memory_service": True,
        },
        "census": {
            "hbm_capacity_points": len(EXPECTED_HBM_GIB),
            "runtime_overhead_points": len(EXPECTED_OVERHEAD_GIB),
            "rows": len(rows),
            "all_hbm_feasible_rows": sum(
                bool(row["all_hbm_capacity_feasible"]) for row in rows
            ),
            "hybrid_feasible_rows": sum(
                bool(row["hybrid_capacity_feasible"]) for row in rows
            ),
        },
        "table": _artifact(table_path, output_dir),
        "eligibility": {
            "vendor_byte_accounting_reproduced": True,
            "capacity_arithmetic_exact_over_rounded_vendor_inputs": True,
            "single_node_means_fixed_tp8_domain": True,
            "capacity_feasible_is_not_memory_service_feasible": True,
            "real_router_trace_present": False,
            "dynamic_memory_replay_present": False,
            "compute_communication_timing_present": False,
            "claim_scope": "static_capacity_sensitivity_only",
            "paper_result_eligible": False,
        },
    }
    _write_json_atomic(receipt_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        payload = prepare_preflight(
            descriptor_path=arguments.descriptor,
            study_path=arguments.study,
            output_dir=arguments.output_dir,
        )
    except KimiK3PreflightError as error:
        raise SystemExit(f"error: {error}") from error
    census = payload["census"]
    print(
        "prepared Kimi K3 TP8 capacity sensitivity: "
        f"all-HBM {census['all_hbm_feasible_rows']}/{census['rows']} fit, "
        f"hybrid {census['hybrid_feasible_rows']}/{census['rows']} fit; "
        "memory service remains undetermined"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
