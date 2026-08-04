#!/usr/bin/env python3
"""Fit and evaluate the public H100 + Llama 2 70B timing candidate.

Only Vidur profile rows enter the predictor.  MLPerf artifacts are opened after
the operator evaluation is frozen and are used solely to decide whether an
end-to-end held-out comparison is scientifically admissible.  If the declared
MLPerf configuration lies outside the fitted domain, the tool records the
blocker and emits no performance prediction.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_EVEN
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any, Iterable


PROTOCOL_SCHEMA = {
    "name": "hbfsim.validation.timing-calibration-protocol",
    "version": 1,
}
CANDIDATE_SCHEMA = {
    "name": "hbfsim.validation.timing-calibration-candidate",
    "version": 1,
}
OFFICIAL_CALIBRATION_ID = "h100-llama2-70b-vidur-interpolation-v1"
OFFICIAL_PROTOCOL_SHA256 = (
    "65243dd780f67aa7ac7b697ce537f8fb123a942756d095036a2dc83dc7a1b6fe"
)
DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parent.parent
    / "validation/calibration/h100-llama2-70b-calibration-protocol.json"
)
HEX_RE = re.compile(r"^[0-9a-f]+$")
U32_MAX = 2**32 - 1


class CalibrationError(ValueError):
    """An input or declared calibration contract is malformed or inconsistent."""


class _DuplicateJsonKey(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _load_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=Decimal,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except OSError as error:
        raise CalibrationError(f"cannot read {description} {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise CalibrationError(f"{description} is not UTF-8: {path}") from error
    except _DuplicateJsonKey as error:
        raise CalibrationError(
            f"{description} contains duplicate key {error.args[0]!r}"
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise CalibrationError(
            f"invalid JSON in {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CalibrationError(f"{description} must be a JSON object")
    return value, payload


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationError(f"{path} must be an object")
    return value


def _array(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        suffix = " and non-empty" if nonempty else ""
        raise CalibrationError(f"{path} must be an array{suffix}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibrationError(f"{path} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise CalibrationError(f"{path} must be boolean")
    return value


def _hex(value: Any, path: str, width: int) -> str:
    text = _string(value, path)
    if len(text) != width or not HEX_RE.fullmatch(text):
        raise CalibrationError(f"{path} must be {width} lowercase hexadecimal digits")
    return text


def _decimal(value: Any, path: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise CalibrationError(f"{path} must be a finite decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise CalibrationError(f"{path} must be a finite decimal") from error
    if not result.is_finite() or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        raise CalibrationError(f"{path} must be a finite {qualifier}decimal")
    return result


def _safe_relative(value: Any, path: str) -> Path:
    relative = Path(_string(value, path))
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise CalibrationError(f"{path} must be a safe relative path")
    return relative


def _decimal_text(value: Decimal, places: int = 12) -> str:
    quantum = Decimal(1).scaleb(-places)
    normalized = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    text = format(normalized, "f").rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def _median(values: Iterable[Decimal]) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise CalibrationError("cannot take the median of an empty observation set")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _nearest_rank(values: list[Decimal], percentile: int) -> Decimal:
    if not values:
        raise CalibrationError("cannot compute a percentile of an empty set")
    ordered = sorted(values)
    rank = (Decimal(len(ordered)) * Decimal(percentile) / Decimal(100)).to_integral_value(
        rounding=ROUND_CEILING
    )
    return ordered[max(0, int(rank) - 1)]


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise CalibrationError(f"cannot read {description} {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise CalibrationError(f"{description} is not UTF-8: {path}") from error


def _artifact_map(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = _array(bundle.get("artifacts"), "anchor bundle.artifacts", nonempty=True)
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(artifacts):
        artifact = _mapping(raw, f"anchor bundle.artifacts[{index}]")
        artifact_id = _string(
            artifact.get("artifact_id"), f"anchor bundle.artifacts[{index}].artifact_id"
        )
        if artifact_id in result:
            raise CalibrationError(f"duplicate anchor bundle artifact {artifact_id}")
        result[artifact_id] = artifact
    return result


def _verified_artifact_path(
    bundle_dir: Path,
    artifacts: dict[str, dict[str, Any]],
    artifact_id: str,
) -> Path:
    if artifact_id not in artifacts:
        raise CalibrationError(f"anchor bundle is missing artifact {artifact_id}")
    artifact = artifacts[artifact_id]
    relative = _safe_relative(
        artifact.get("bundle_path"), f"anchor artifact {artifact_id}.bundle_path"
    )
    path = bundle_dir / relative
    if not path.is_file() or path.is_symlink():
        raise CalibrationError(f"anchor artifact is not a regular file: {path}")
    if path.resolve() != path:
        raise CalibrationError(f"anchor artifact traverses a symlink: {artifact_id}")
    expected_size = _integer(
        artifact.get("bytes"), f"anchor artifact {artifact_id}.bytes", minimum=1
    )
    expected_sha = _hex(
        artifact.get("sha256"), f"anchor artifact {artifact_id}.sha256", 64
    )
    if path.stat().st_size != expected_size or _sha256_file(path) != expected_sha:
        raise CalibrationError(f"anchor artifact digest mismatch: {artifact_id}")
    return path


def _validate_anchor_inputs(
    protocol: dict[str, Any],
    bundle_dir: Path,
    verification_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Path],
    dict[str, Any],
]:
    anchor_contract = _mapping(protocol.get("anchor"), "protocol.anchor")
    bundle_path = bundle_dir / "bundle-manifest.json"
    copied_manifest_path = bundle_dir / "anchor-manifest.json"
    bundle, bundle_bytes = _load_object(bundle_path, "anchor bundle manifest")
    copied_manifest, copied_manifest_bytes = _load_object(
        copied_manifest_path, "copied anchor manifest"
    )
    verification, _ = _load_object(verification_path, "anchor verification receipt")

    if bundle.get("schema") != anchor_contract.get("bundle_schema"):
        raise CalibrationError("anchor bundle schema mismatch")
    if verification.get("schema") != anchor_contract.get("verification_schema"):
        raise CalibrationError("anchor verification schema mismatch")
    anchor_id = _string(anchor_contract.get("anchor_id"), "protocol.anchor.anchor_id")
    bundle_anchor = _mapping(bundle.get("anchor"), "bundle.anchor")
    if bundle_anchor.get("anchor_id") != anchor_id:
        raise CalibrationError("anchor bundle id mismatch")
    if copied_manifest.get("anchor_id") != anchor_id:
        raise CalibrationError("copied anchor manifest id mismatch")
    if verification.get("anchor_id") != anchor_id or verification.get("result") != "pass":
        raise CalibrationError("anchor verification receipt is not a pass for this anchor")
    expected_manifest_sha = _hex(
        anchor_contract.get("manifest_sha256"), "protocol.anchor.manifest_sha256", 64
    )
    if _sha256_bytes(copied_manifest_bytes) != expected_manifest_sha:
        raise CalibrationError("copied anchor manifest does not match the protocol pin")
    manifest_record = _mapping(bundle_anchor.get("manifest"), "bundle.anchor.manifest")
    if manifest_record.get("sha256") != expected_manifest_sha:
        raise CalibrationError("anchor bundle does not bind the pinned manifest")
    if bundle.get("result") != "pass":
        raise CalibrationError("anchor bundle result is not pass")
    required_state = _string(
        anchor_contract.get("required_timing_claim_state"),
        "protocol.anchor.required_timing_claim_state",
    )
    bundle_eligibility = _mapping(bundle.get("eligibility"), "bundle.eligibility")
    verification_eligibility = _mapping(
        verification.get("eligibility"), "verification.eligibility"
    )
    if (
        bundle_eligibility.get("timing_claim_state") != required_state
        or verification_eligibility.get("timing_claim_state") != required_state
    ):
        raise CalibrationError("anchor timing claim state was unexpectedly promoted")
    for key in (
        "predictor_calibration_valid",
        "end_to_end_external_validation_valid",
        "frontier_timing_usable_for_performance",
        "paper_result_eligible",
    ):
        if bundle_eligibility.get(key) is not False or verification_eligibility.get(key) is not False:
            raise CalibrationError(f"anchor receipt illegally promotes {key}")
    checks = _mapping(verification.get("checks"), "verification.checks")
    if not checks or any(value is not True for value in checks.values()):
        raise CalibrationError("anchor verification receipt contains a failed check")
    inputs = _mapping(verification.get("inputs"), "verification.inputs")
    verified_bundle = _mapping(inputs.get("bundle_manifest"), "verification.inputs.bundle_manifest")
    if verified_bundle.get("sha256") != _sha256_bytes(bundle_bytes):
        raise CalibrationError("anchor verification is for a different bundle manifest")
    verified_anchor = _mapping(inputs.get("anchor_manifest"), "verification.inputs.anchor_manifest")
    if verified_anchor.get("sha256") != expected_manifest_sha:
        raise CalibrationError("anchor verification is for a different anchor manifest")

    artifacts = _artifact_map(bundle)
    manifest_sources = _mapping(copied_manifest.get("sources"), "anchor manifest.sources")
    declared_artifacts: dict[str, dict[str, Any]] = {}
    for source_id, source_raw in manifest_sources.items():
        source = _mapping(source_raw, f"anchor manifest.sources.{source_id}")
        for raw in _array(source.get("artifacts"), f"anchor manifest.sources.{source_id}.artifacts"):
            artifact = _mapping(raw, f"anchor manifest.sources.{source_id}.artifact")
            artifact_id = _string(artifact.get("id"), "anchor manifest artifact.id")
            if artifact_id in declared_artifacts:
                raise CalibrationError(f"duplicate manifest artifact id {artifact_id}")
            declared_artifacts[artifact_id] = {**artifact, "source_id": source_id}

    paths: dict[str, Path] = {}
    for artifact_id in declared_artifacts:
        path = _verified_artifact_path(bundle_dir, artifacts, artifact_id)
        record = artifacts[artifact_id]
        declaration = declared_artifacts[artifact_id]
        if (
            record.get("source_id") != declaration["source_id"]
            or record.get("kind") != declaration.get("kind")
            or record.get("sha256") != declaration.get("sha256")
            or record.get("bytes") != declaration.get("bytes")
        ):
            raise CalibrationError(f"bundle/manifest mismatch for artifact {artifact_id}")
        paths[artifact_id] = path
    return bundle, declared_artifacts, paths, verification


def _validate_workload_evidence(
    protocol: dict[str, Any], evidence_dir: Path
) -> tuple[dict[str, Any], Path, list[tuple[int, int]]]:
    contract = _mapping(protocol.get("workload_evidence"), "protocol.workload_evidence")
    receipt_path = evidence_dir / "workload-evidence.json"
    receipt, _ = _load_object(receipt_path, "workload evidence receipt")
    if receipt.get("schema") != contract.get("schema") or receipt.get("result") != "pass":
        raise CalibrationError("workload evidence receipt is not a compatible pass")
    if receipt.get("calibration_id") != protocol.get("calibration_id"):
        raise CalibrationError("workload evidence calibration id mismatch")
    dataset_contract = _mapping(contract.get("dataset"), "protocol.workload_evidence.dataset")
    dataset_receipt = _mapping(receipt.get("dataset"), "workload receipt.dataset")
    for key in ("bytes", "md5", "sha256", "source_url"):
        if dataset_receipt.get(key) != dataset_contract.get(key):
            raise CalibrationError(f"workload dataset {key} mismatch")
    if dataset_receipt.get("digest_checked_before_pickle_load") is not True:
        raise CalibrationError("workload evidence did not enforce pre-unpickle digests")

    lengths_contract = _mapping(contract.get("lengths_csv"), "protocol.lengths_csv")
    lengths_receipt = _mapping(receipt.get("lengths"), "workload receipt.lengths")
    relative = _safe_relative(lengths_receipt.get("path"), "workload receipt.lengths.path")
    lengths_path = evidence_dir / relative
    if not lengths_path.is_file() or lengths_path.is_symlink():
        raise CalibrationError("workload lengths CSV is missing or a symlink")
    if lengths_path.resolve() != lengths_path:
        raise CalibrationError("workload lengths CSV traverses a symlink")
    for key in ("columns", "rows", "bytes", "sha256", "pairs_u32le_sha256"):
        if lengths_receipt.get(key) != lengths_contract.get(key):
            raise CalibrationError(f"workload lengths receipt {key} mismatch")
    if lengths_path.stat().st_size != lengths_contract["bytes"]:
        raise CalibrationError("workload lengths byte count mismatch")
    if _sha256_file(lengths_path) != lengths_contract["sha256"]:
        raise CalibrationError("workload lengths SHA-256 mismatch")

    pairs: list[tuple[int, int]] = []
    with lengths_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != lengths_contract["columns"]:
            raise CalibrationError("workload lengths CSV header mismatch")
        for expected_index, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise CalibrationError("workload lengths CSV has a malformed row")
            try:
                qsl_idx = int(row["qsl_idx"])
                input_tokens = int(row["input_tokens"])
                output_tokens = int(row["reference_output_tokens"])
            except (TypeError, ValueError) as error:
                raise CalibrationError("workload lengths CSV contains a non-integer") from error
            if str(qsl_idx) != row["qsl_idx"] or qsl_idx != expected_index:
                raise CalibrationError("workload QSL indices are not canonical and contiguous")
            if input_tokens <= 0 or output_tokens <= 0:
                raise CalibrationError("workload token lengths must be positive")
            if input_tokens > U32_MAX or output_tokens > U32_MAX:
                raise CalibrationError("workload token length exceeds uint32")
            pairs.append((input_tokens, output_tokens))
    if len(pairs) != lengths_contract["rows"]:
        raise CalibrationError("workload lengths row count mismatch")
    pair_sha = hashlib.sha256(
        b"".join(struct.pack("<II", left, right) for left, right in pairs)
    ).hexdigest()
    if pair_sha != lengths_contract["pairs_u32le_sha256"]:
        raise CalibrationError("workload length-pair digest mismatch")
    observed = {
        "input_tokens": {
            "min": min(left for left, _ in pairs),
            "max": max(left for left, _ in pairs),
            "sum": sum(left for left, _ in pairs),
        },
        "reference_output_tokens": {
            "min": min(right for _, right in pairs),
            "max": max(right for _, right in pairs),
            "sum": sum(right for _, right in pairs),
        },
    }
    for key, value in observed.items():
        if value != lengths_contract.get(key) or value != lengths_receipt.get(key):
            raise CalibrationError(f"workload {key} statistics mismatch")
    scope = _mapping(receipt.get("scope"), "workload receipt.scope")
    if scope != {
        "contains_prompts": False,
        "contains_token_ids": False,
        "contains_reference_text": False,
        "establishes_timing_calibration": False,
    }:
        raise CalibrationError("workload evidence scope was unexpectedly broadened")
    return receipt, lengths_path, pairs


def _coordinate_value(row: dict[str, str], specification: dict[str, Any]) -> Decimal:
    column = _string(specification.get("column"), "coordinate.column")
    raw = row.get(column)
    if raw is None or not re.fullmatch(r"0|[1-9][0-9]*", raw):
        raise CalibrationError(f"coordinate {column} must be a canonical nonnegative integer")
    value = Decimal(raw)
    transform = specification.get("transform")
    if transform == "identity":
        return value
    if transform == "square":
        return value * value
    raise CalibrationError(f"unsupported coordinate transform {transform!r}")


def _model_id(family_id: str, target_id: str, group: dict[str, str]) -> str:
    suffix = ",".join(f"{key}={group[key]}" for key in sorted(group))
    return f"{family_id}.{target_id}.{suffix}"


def _interpolate(
    key: tuple[Decimal, ...],
    indices: tuple[int, ...],
    axes: list[list[Decimal]],
    table: dict[tuple[Decimal, ...], Decimal],
) -> Decimal:
    weights: list[Decimal] = []
    for dimension, (coordinate, index) in enumerate(zip(key, indices)):
        lower = axes[dimension][index - 1]
        upper = axes[dimension][index + 1]
        if upper <= lower:
            raise CalibrationError("non-increasing interpolation axis")
        weights.append((coordinate - lower) / (upper - lower))
    prediction = Decimal(0)
    corner_count = 1 << len(key)
    for mask in range(corner_count):
        corner: list[Decimal] = []
        weight = Decimal(1)
        for dimension, index in enumerate(indices):
            upper_corner = bool(mask & (1 << dimension))
            corner.append(axes[dimension][index + (1 if upper_corner else -1)])
            weight *= weights[dimension] if upper_corner else Decimal(1) - weights[dimension]
        corner_key = tuple(corner)
        if corner_key not in table:
            raise CalibrationError("evaluation point has an incomplete corner set")
        prediction += weight * table[corner_key]
    if prediction < 0:
        raise CalibrationError("interpolator produced a negative timing")
    return prediction


def _fit_models(
    protocol: dict[str, Any],
    declarations: dict[str, dict[str, Any]],
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[list[str]], list[list[str]], dict[str, Any]]:
    fit = _mapping(protocol.get("fit_protocol"), "protocol.fit_protocol")
    if fit.get("time_unit") != "milliseconds":
        raise CalibrationError("only millisecond Vidur profiles are supported")
    if fit.get("target_statistic") != "published_median":
        raise CalibrationError("fit target must remain the published median")
    if fit.get("duplicate_key_policy") != "group_then_decimal_median":
        raise CalibrationError("unsupported duplicate-key policy")
    split = _mapping(fit.get("split"), "protocol.fit_protocol.split")
    if split != {
        "algorithm": "coordinate_interleaved_v1",
        "evaluation_coordinate_index_parity": 1,
        "axis_boundaries_are_training": True,
        "evaluation_requires_complete_adjacent_corner_set": True,
        "all_rows_with_the_same_profile_key_share_one_partition": True,
    }:
        raise CalibrationError("unsupported or weakened fit split")
    interpolator = _mapping(fit.get("interpolator"), "protocol.fit_protocol.interpolator")
    if interpolator != {
        "algorithm": "multilinear_adjacent_corners_v1",
        "coordinate_arithmetic": "decimal",
        "extrapolation_allowed": False,
        "negative_prediction_allowed": False,
    }:
        raise CalibrationError("unsupported or weakened interpolator contract")
    acceptance = _mapping(fit.get("acceptance"), "protocol.fit_protocol.acceptance")
    max_wape = _decimal(acceptance.get("maximum_wape_pct"), "acceptance.maximum_wape_pct", positive=True)
    max_p90 = _decimal(acceptance.get("maximum_p90_ape_pct"), "acceptance.maximum_p90_ape_pct", positive=True)
    max_bias = _decimal(
        acceptance.get("maximum_absolute_signed_bias_pct"),
        "acceptance.maximum_absolute_signed_bias_pct",
        positive=True,
    )

    profile_cache: dict[str, list[dict[str, str]]] = {}
    model_results: list[dict[str, Any]] = []
    predictor_rows: list[list[str]] = []
    evaluation_rows: list[list[str]] = []
    overall_truth: list[Decimal] = []
    overall_predictions: list[Decimal] = []
    fit_artifact_ids: set[str] = set()

    families = _array(fit.get("families"), "protocol.fit_protocol.families", nonempty=True)
    for family_index, family_raw in enumerate(families):
        family = _mapping(family_raw, f"fit family[{family_index}]")
        family_id = _string(family.get("id"), f"fit family[{family_index}].id")
        artifact_id = _string(
            family.get("artifact_id"), f"fit family[{family_index}].artifact_id"
        )
        declaration = declarations.get(artifact_id)
        if declaration is None or declaration.get("source_id") != "vidur":
            raise CalibrationError(f"fit family {family_id} is not bound to a Vidur artifact")
        if declaration.get("kind") != "csv_profile":
            raise CalibrationError(f"fit family {family_id} does not use a CSV profile")
        fit_artifact_ids.add(artifact_id)
        if artifact_id not in profile_cache:
            with paths[artifact_id].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != declaration.get("columns"):
                    raise CalibrationError(f"profile header mismatch for {artifact_id}")
                rows = list(reader)
            if len(rows) != declaration.get("rows"):
                raise CalibrationError(f"profile row count mismatch for {artifact_id}")
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise CalibrationError(f"malformed CSV row in {artifact_id}")
            profile_cache[artifact_id] = rows
        rows = profile_cache[artifact_id]

        filters = _mapping(family.get("filters"), f"fit family {family_id}.filters")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in filters.items()):
            raise CalibrationError(f"fit family {family_id} filters must map strings")
        group_values = _array(
            family.get("group_values"), f"fit family {family_id}.group_values", nonempty=True
        )
        coordinates = [
            _mapping(item, f"fit family {family_id}.coordinate")
            for item in _array(
                family.get("coordinates"), f"fit family {family_id}.coordinates", nonempty=True
            )
        ]
        if len(coordinates) not in (1, 2):
            raise CalibrationError(f"fit family {family_id} must have one or two coordinates")
        targets = [
            _mapping(item, f"fit family {family_id}.target")
            for item in _array(
                family.get("targets"), f"fit family {family_id}.targets", nonempty=True
            )
        ]
        minimum_fraction = _decimal(
            family.get("minimum_evaluation_fraction"),
            f"fit family {family_id}.minimum_evaluation_fraction",
            positive=True,
        )
        if minimum_fraction > 1:
            raise CalibrationError(f"fit family {family_id} evaluation fraction exceeds one")

        seen_groups: set[tuple[tuple[str, str], ...]] = set()
        for group_index, group_raw in enumerate(group_values):
            group = _mapping(group_raw, f"fit family {family_id}.group_values[{group_index}]")
            if not group or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in group.items()
            ):
                raise CalibrationError(f"fit family {family_id} group must map strings")
            group_key = tuple(sorted(group.items()))
            if group_key in seen_groups:
                raise CalibrationError(f"fit family {family_id} has a duplicate group")
            seen_groups.add(group_key)
            selected = [
                row
                for row in rows
                if all(row.get(key) == value for key, value in {**filters, **group}.items())
            ]
            if not selected:
                raise CalibrationError(f"fit family {family_id} group {group} is empty")

            grouped_rows: dict[tuple[Decimal, ...], list[dict[str, str]]] = defaultdict(list)
            for row in selected:
                key = tuple(_coordinate_value(row, specification) for specification in coordinates)
                grouped_rows[key].append(row)
            axes = [sorted({key[index] for key in grouped_rows}) for index in range(len(coordinates))]
            axis_indices = [
                {value: index for index, value in enumerate(axis)} for axis in axes
            ]
            evaluation_keys: list[tuple[Decimal, ...]] = []
            for key in sorted(grouped_rows):
                indices = tuple(
                    axis_indices[dimension][coordinate]
                    for dimension, coordinate in enumerate(key)
                )
                if not all(
                    index % 2 == 1 and 0 < index < len(axes[dimension]) - 1
                    for dimension, index in enumerate(indices)
                ):
                    continue
                corners_complete = True
                for mask in range(1 << len(key)):
                    corner = tuple(
                        axes[dimension][
                            index + (1 if mask & (1 << dimension) else -1)
                        ]
                        for dimension, index in enumerate(indices)
                    )
                    if corner not in grouped_rows:
                        corners_complete = False
                        break
                if corners_complete:
                    evaluation_keys.append(key)
            evaluation_set = set(evaluation_keys)
            evaluation_fraction = Decimal(len(evaluation_keys)) / Decimal(len(grouped_rows))
            if evaluation_fraction < minimum_fraction:
                raise CalibrationError(
                    f"fit family {family_id} group {group} evaluation coverage "
                    f"{evaluation_fraction} is below {minimum_fraction}"
                )

            for target_index, target_raw in enumerate(targets):
                target_id = _string(
                    target_raw.get("id"), f"fit family {family_id}.targets[{target_index}].id"
                )
                target_column = _string(
                    target_raw.get("column"),
                    f"fit family {family_id}.targets[{target_index}].column",
                )
                table: dict[tuple[Decimal, ...], Decimal] = {}
                raw_observations = 0
                for key, key_rows in grouped_rows.items():
                    values: list[Decimal] = []
                    for row in key_rows:
                        raw = row.get(target_column)
                        if raw is None or raw == "":
                            raise CalibrationError(
                                f"target {target_column} is empty in selected profile rows"
                            )
                        values.append(
                            _decimal(raw, f"{artifact_id}.{target_column}", positive=True)
                        )
                    raw_observations += len(values)
                    table[key] = _median(values)

                model_id = _model_id(family_id, target_id, group)
                table_hasher = hashlib.sha256()
                split_hasher = hashlib.sha256()
                for key in sorted(table):
                    coordinates_text = [_decimal_text(value, places=0) for value in key]
                    timing_text = _decimal_text(table[key])
                    canonical = "\t".join([model_id, *coordinates_text, timing_text]) + "\n"
                    table_hasher.update(canonical.encode("utf-8"))
                    partition = "evaluation" if key in evaluation_set else "training"
                    split_hasher.update(
                        ("\t".join([model_id, *coordinates_text, partition]) + "\n").encode(
                            "utf-8"
                        )
                    )
                    predictor_rows.append(
                        [
                            model_id,
                            coordinates_text[0],
                            coordinates_text[1] if len(coordinates_text) == 2 else "",
                            timing_text,
                        ]
                    )

                truths: list[Decimal] = []
                predictions: list[Decimal] = []
                apes: list[Decimal] = []
                absolute_errors: list[Decimal] = []
                for key in evaluation_keys:
                    indices = tuple(
                        axis_indices[dimension][coordinate]
                        for dimension, coordinate in enumerate(key)
                    )
                    truth = table[key]
                    prediction = _interpolate(key, indices, axes, table)
                    error = abs(prediction - truth)
                    ape = error / truth * Decimal(100)
                    truths.append(truth)
                    predictions.append(prediction)
                    absolute_errors.append(error)
                    apes.append(ape)
                    coordinates_text = [_decimal_text(value, places=0) for value in key]
                    evaluation_rows.append(
                        [
                            model_id,
                            coordinates_text[0],
                            coordinates_text[1] if len(coordinates_text) == 2 else "",
                            _decimal_text(truth),
                            _decimal_text(prediction),
                            _decimal_text(error),
                            _decimal_text(ape),
                        ]
                    )
                if not truths:
                    raise CalibrationError(f"model {model_id} has no evaluation points")
                wape = sum(absolute_errors) / sum(truths) * Decimal(100)
                mape = sum(apes) / Decimal(len(apes))
                p90 = _nearest_rank(apes, 90)
                median_ape = _median(apes)
                signed_bias = (
                    (sum(predictions) - sum(truths)) / sum(truths) * Decimal(100)
                )
                passed = (
                    wape <= max_wape
                    and p90 <= max_p90
                    and abs(signed_bias) <= max_bias
                )
                model_results.append(
                    {
                        "model_id": model_id,
                        "family_id": family_id,
                        "target_id": target_id,
                        "target_column": target_column,
                        "artifact_id": artifact_id,
                        "group": dict(sorted(group.items())),
                        "coordinates": coordinates,
                        "raw_observations": raw_observations,
                        "unique_profile_keys": len(table),
                        "training_keys": len(table) - len(evaluation_keys),
                        "evaluation_keys": len(evaluation_keys),
                        "evaluation_fraction": _decimal_text(evaluation_fraction),
                        "domain": [
                            {
                                "min": _decimal_text(axis[0], places=0),
                                "max": _decimal_text(axis[-1], places=0),
                                "values": len(axis),
                            }
                            for axis in axes
                        ],
                        "deployment_table_sha256": table_hasher.hexdigest(),
                        "split_sha256": split_hasher.hexdigest(),
                        "metrics": {
                            "mape_pct": _decimal_text(mape),
                            "median_ape_pct": _decimal_text(median_ape),
                            "p90_ape_pct": _decimal_text(p90),
                            "wape_pct": _decimal_text(wape),
                            "signed_bias_pct": _decimal_text(signed_bias),
                        },
                        "thresholds": {
                            "maximum_wape_pct": _decimal_text(max_wape),
                            "maximum_p90_ape_pct": _decimal_text(max_p90),
                            "maximum_absolute_signed_bias_pct": _decimal_text(max_bias),
                        },
                        "result": "pass" if passed else "fail",
                    }
                )
                overall_truth.extend(truths)
                overall_predictions.extend(predictions)

    model_ids = [model["model_id"] for model in model_results]
    if len(model_ids) != len(set(model_ids)):
        raise CalibrationError("fit protocol produces duplicate model ids")
    predictor_rows.sort(key=lambda row: (row[0], Decimal(row[1]), Decimal(row[2] or 0)))
    evaluation_rows.sort(key=lambda row: (row[0], Decimal(row[1]), Decimal(row[2] or 0)))
    overall_abs = [abs(prediction - truth) for truth, prediction in zip(overall_truth, overall_predictions)]
    overall = {
        "models": len(model_results),
        "passing_models": sum(model["result"] == "pass" for model in model_results),
        "evaluation_keys": len(overall_truth),
        "wape_pct": _decimal_text(sum(overall_abs) / sum(overall_truth) * Decimal(100)),
        "signed_bias_pct": _decimal_text(
            (sum(overall_predictions) - sum(overall_truth))
            / sum(overall_truth)
            * Decimal(100)
        ),
        "fit_artifact_ids": sorted(fit_artifact_ids),
        "mlperf_artifacts_opened_during_fit": [],
    }
    return sorted(model_results, key=lambda item: item["model_id"]), predictor_rows, evaluation_rows, overall


def _class_assignments(source: str, class_name: str, description: str) -> dict[str, ast.AST]:
    try:
        tree = ast.parse(source, filename=description)
    except SyntaxError as error:
        raise CalibrationError(f"invalid Python syntax in {description}: {error}") from error
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(matches) != 1:
        raise CalibrationError(f"{description} must define class {class_name} exactly once")
    result: dict[str, ast.AST] = {}
    for statement in matches[0].body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            if target.id in result:
                raise CalibrationError(f"duplicate assignment {class_name}.{target.id}")
            result[target.id] = statement.value
    return result


def _literal(assignments: dict[str, ast.AST], key: str, path: str) -> Any:
    if key not in assignments:
        raise CalibrationError(f"{path} is missing")
    return _safe_config_literal(assignments[key], path)


def _safe_config_literal(node: ast.AST, path: str) -> Any:
    """Evaluate data-only config syntax, including exact integer arithmetic.

    ``ast.literal_eval`` intentionally rejects the ``1024 + 1024`` expression
    used by NVIDIA's pinned base configuration.  Supporting that expression by
    importing the configuration would execute repository code, so this parser
    instead accepts a deliberately small AST subset and nothing executable.
    """

    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise CalibrationError(f"{path} contains an unsupported constant")
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [
            _safe_config_literal(item, f"{path}[{index}]")
            for index, item in enumerate(node.elts)
        ]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for index, (key_node, value_node) in enumerate(zip(node.keys, node.values)):
            if key_node is None:
                raise CalibrationError(f"{path} must not contain dictionary unpacking")
            literal_key = _safe_config_literal(key_node, f"{path}.key[{index}]")
            try:
                duplicate = literal_key in result
            except TypeError as error:
                raise CalibrationError(f"{path} contains an unhashable key") from error
            if duplicate:
                raise CalibrationError(f"{path} contains duplicate key {literal_key!r}")
            result[literal_key] = _safe_config_literal(
                value_node, f"{path}[{literal_key!r}]"
            )
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _safe_config_literal(node.operand, f"{path}.operand")
        if isinstance(operand, bool) or not isinstance(operand, int):
            raise CalibrationError(f"{path} unary arithmetic must use an integer")
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        left = _safe_config_literal(node.left, f"{path}.left")
        right = _safe_config_literal(node.right, f"{path}.right")
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, int)
            or not isinstance(right, int)
        ):
            raise CalibrationError(f"{path} arithmetic must use integers")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    raise CalibrationError(f"{path} is not a safe data-only literal")


def _attribute_text(assignments: dict[str, ast.AST], key: str, path: str) -> str:
    if key not in assignments:
        raise CalibrationError(f"{path} is missing")
    return ast.unparse(assignments[key])


def _mllog_events(path: Path, description: str) -> dict[str, list[Any]]:
    events: dict[str, list[Any]] = defaultdict(list)
    for line_number, line in enumerate(_read_text(path, description).splitlines(), start=1):
        marker = ":::MLLOG "
        if marker not in line:
            continue
        try:
            payload = json.loads(
                line.split(marker, 1)[1],
                parse_float=Decimal,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_object,
            )
        except (_DuplicateJsonKey, json.JSONDecodeError, ValueError) as error:
            raise CalibrationError(
                f"invalid MLLOG JSON in {description} line {line_number}: {error}"
            ) from error
        if not isinstance(payload, dict) or not isinstance(payload.get("key"), str):
            raise CalibrationError(f"malformed MLLOG event in {description} line {line_number}")
        events[payload["key"]].append(payload.get("value"))
    return events


def _single_event(events: dict[str, list[Any]], key: str, description: str) -> Any:
    values = events.get(key, [])
    if len(values) != 1:
        raise CalibrationError(f"{description} must contain exactly one {key} event")
    return values[0]


def _compare_event(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, int):
        if isinstance(actual, bool) or actual != expected:
            raise CalibrationError(f"{path} mismatch: expected={expected}, actual={actual}")
    else:
        if _decimal(actual, path) != _decimal(expected, f"{path}.expected"):
            raise CalibrationError(f"{path} mismatch: expected={expected}, actual={actual}")


def _assess_end_to_end_gate(
    protocol: dict[str, Any], paths: dict[str, Path], pairs: list[tuple[int, int]]
) -> dict[str, Any]:
    gate = _mapping(protocol.get("end_to_end_gate"), "protocol.end_to_end_gate")
    roles = _mapping(gate.get("artifact_roles"), "end_to_end_gate.artifact_roles")
    evidence_ids = [
        _string(value, f"end_to_end_gate.artifact_roles.{key}")
        for key, value in sorted(roles.items())
    ]
    declared_ids = [
        _string(value, "end_to_end_gate.evidence_artifacts[]")
        for value in _array(gate.get("evidence_artifacts"), "end_to_end_gate.evidence_artifacts")
    ]
    if sorted(evidence_ids) != sorted(declared_ids):
        raise CalibrationError("end-to-end artifact role/list census mismatch")
    for artifact_id in evidence_ids:
        if artifact_id not in paths:
            raise CalibrationError(f"end-to-end evidence artifact is missing: {artifact_id}")

    sut = _mapping(gate.get("declared_mlperf_sut"), "end_to_end_gate.declared_mlperf_sut")
    scope = _mapping(gate.get("vidur_fit_scope"), "end_to_end_gate.vidur_fit_scope")
    for key in (
        "accelerators",
        "replicas",
        "gpus_per_replica",
        "tensor_parallelism",
        "pipeline_parallelism",
        "gpu_batch_size",
        "max_num_tokens",
        "max_input_len",
        "max_output_len",
    ):
        _integer(sut.get(key), f"declared_mlperf_sut.{key}", minimum=1)
    for key in (
        "tensor_parallelism",
        "decode_batch_size_min",
        "decode_batch_size_max",
        "total_tokens_min",
        "total_tokens_max",
        "max_model_len",
    ):
        _integer(scope.get(key), f"vidur_fit_scope.{key}", minimum=1)
    for key in ("cpu_runtime_overhead_profile_present", "scheduler_trace_profile_present"):
        _boolean(scope.get(key), f"vidur_fit_scope.{key}")
    for key in (
        "weight_precision",
        "kv_cache_precision",
        "framework",
        "batch_scheduler_policy",
        "context_chunking_policy",
    ):
        _string(sut.get(key), f"declared_mlperf_sut.{key}")
    _string(scope.get("precision"), "vidur_fit_scope.precision")
    base_source = _read_text(paths[roles["base_config"]], roles["base_config"])
    offline_source = _read_text(paths[roles["offline_config"]], roles["offline_config"])
    server_source = _read_text(paths[roles["server_config"]], roles["server_config"])
    preprocessor_source = _read_text(paths[roles["preprocessor"]], roles["preprocessor"])

    base = _class_assignments(base_source, "GPUBaseConfig", roles["base_config"])
    base_build = _mapping(_literal(base, "trtllm_build_flags", "GPUBaseConfig.trtllm_build_flags"), "base build flags")
    base_runtime = _mapping(
        _literal(base, "trtllm_runtime_flags", "GPUBaseConfig.trtllm_runtime_flags"),
        "base runtime flags",
    )
    if base_build.get("max_input_len") != sut.get("max_input_len"):
        raise CalibrationError("NVIDIA base config max_input_len mismatch")
    if base_build.get("max_seq_len") != sut.get("max_input_len") + sut.get("max_output_len"):
        raise CalibrationError("NVIDIA base config max_seq_len mismatch")
    if base_runtime.get("batch_scheduler_policy") != sut.get("batch_scheduler_policy"):
        raise CalibrationError("NVIDIA batch scheduler policy mismatch")
    if base_runtime.get("context_chunking_policy") != sut.get("context_chunking_policy"):
        raise CalibrationError("NVIDIA context chunking policy mismatch")

    for source, prefix in ((offline_source, "Offline"), (server_source, "Server")):
        hopper = _class_assignments(source, f"Hopper{prefix}GPUBaseConfig", roles[f"{prefix.lower()}_config"])
        if _literal(hopper, "precision", f"Hopper{prefix}GPUBaseConfig.precision") != sut.get("weight_precision"):
            raise CalibrationError(f"NVIDIA {prefix} precision mismatch")
        checkpoint = _mapping(
            _literal(hopper, "trtllm_checkpoint_flags", f"Hopper{prefix}GPUBaseConfig.trtllm_checkpoint_flags"),
            f"{prefix} checkpoint flags",
        )
        if checkpoint.get("kv_cache_dtype") != sut.get("kv_cache_precision"):
            raise CalibrationError(f"NVIDIA {prefix} KV precision mismatch")
        pp2 = _class_assignments(source, "H100_SXM_80GB_PP2x1", roles[f"{prefix.lower()}_config"])
        batch_size = _mapping(
            _literal(pp2, "gpu_batch_size", f"{prefix}.H100_PP2x1.gpu_batch_size"),
            f"{prefix} gpu batch size",
        )
        if batch_size.get("llama2-70b") != sut.get("gpu_batch_size"):
            raise CalibrationError(f"NVIDIA {prefix} gpu_batch_size mismatch")
        build = _mapping(
            _literal(pp2, "trtllm_build_flags", f"{prefix}.H100_PP2x1.trtllm_build_flags"),
            f"{prefix} H100 build flags",
        )
        for key, expected in (
            ("tensor_parallelism", sut.get("tensor_parallelism")),
            ("pipeline_parallelism", sut.get("pipeline_parallelism")),
            ("max_num_tokens", sut.get("max_num_tokens")),
        ):
            if build.get(key) != expected:
                raise CalibrationError(f"NVIDIA {prefix} {key} mismatch")
        x4 = _class_assignments(source, "H100_SXM_80GB_PP2x4", roles[f"{prefix.lower()}_config"])
        if not _attribute_text(x4, "system", f"{prefix}.H100_PP2x4.system").endswith(
            "H100_SXM_80GBx8"
        ):
            raise CalibrationError(f"NVIDIA {prefix} x4 system mismatch")

    try:
        preprocessor_tree = ast.parse(preprocessor_source, filename=roles["preprocessor"])
    except SyntaxError as error:
        raise CalibrationError("invalid NVIDIA preprocessor source") from error
    preprocessor_constants: dict[str, Any] = {}
    for node in preprocessor_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                preprocessor_constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    if preprocessor_constants.get("G_MAX_INPUT_TOK_LEN", 0) < max(left for left, _ in pairs):
        raise CalibrationError("NVIDIA preprocessor cannot represent the workload input lengths")

    generation, _ = _load_object(paths[roles["generation_config"]], "NVIDIA generation config")
    generation_config = _mapping(generation.get("generation_config"), "generation_config")
    if generation_config.get("max_output_len") != sut.get("max_output_len"):
        raise CalibrationError("NVIDIA generation max_output_len mismatch")
    if generation_config.get("streaming") is not True:
        raise CalibrationError("NVIDIA generation config is not streaming")

    detail_expectations = _mapping(
        gate.get("detail_log_expectations"), "end_to_end_gate.detail_log_expectations"
    )
    qsl_digests: list[str] = []
    observed_details: dict[str, Any] = {}
    for scenario in ("offline", "server"):
        events = _mllog_events(paths[roles[f"{scenario}_detail"]], f"{scenario} detail log")
        qsl = _single_event(events, "loaded_qsl_set", f"{scenario} detail log")
        if (
            not isinstance(qsl, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in qsl)
            or len(qsl) != detail_expectations.get("loaded_qsl_set_count")
            or sorted(qsl) != list(range(len(qsl)))
        ):
            raise CalibrationError(f"{scenario} loaded QSL set is not an exact permutation")
        qsl_sha = hashlib.sha256(
            b"".join(struct.pack("<I", value) for value in qsl)
        ).hexdigest()
        if qsl_sha != detail_expectations.get("loaded_qsl_set_u32le_sha256"):
            raise CalibrationError(f"{scenario} loaded QSL set digest mismatch")
        qsl_digests.append(qsl_sha)
        expected = _mapping(detail_expectations.get(scenario), f"detail expectations.{scenario}")
        observed: dict[str, Any] = {}
        for key, expected_value in expected.items():
            actual = _single_event(events, key, f"{scenario} detail log")
            _compare_event(actual, expected_value, f"{scenario}.{key}")
            observed[key] = str(actual) if isinstance(actual, Decimal) else actual
        observed_details[scenario] = observed
    if len(set(qsl_digests)) != 1:
        raise CalibrationError("Offline and Server QSL permutations differ")

    checks = {
        "exact_openorca_input_lengths": True,
        "exact_mlperf_qsl_permutation": True,
        "exact_mlperf_query_counts": True,
        "tensor_parallelism_match": (
            sut["tensor_parallelism"] == scope["tensor_parallelism"]
        ),
        "total_token_budget_covered": sut["max_num_tokens"] <= scope["total_tokens_max"],
        "maximum_sequence_length_covered": (
            sut["max_input_len"] + sut["max_output_len"] <= scope["max_model_len"]
        ),
        "weight_precision_match": sut["weight_precision"] == scope["precision"],
        "kv_cache_precision_match": sut["kv_cache_precision"] == scope["precision"],
        "runtime_kernel_match_or_validated_bridge": False,
        "declared_batch_domain_covered": (
            sut["gpu_batch_size"] <= scope["decode_batch_size_max"]
        ),
        "cpu_runtime_overhead_covered": scope["cpu_runtime_overhead_profile_present"],
        "scheduler_and_arrival_process_covered": scope["scheduler_trace_profile_present"],
        "per_request_generated_length_distribution_known": False,
    }
    requirements = _mapping(
        gate.get("external_validation_requirements"),
        "end_to_end_gate.external_validation_requirements",
    )
    required_boolean_keys = sorted(set(requirements) - {"maximum_external_wape_pct"})
    if set(required_boolean_keys) != {
        "tensor_parallelism_match",
        "total_token_budget_covered",
        "maximum_sequence_length_covered",
        "weight_precision_match",
        "kv_cache_precision_match",
        "runtime_kernel_match_or_validated_bridge",
        "declared_batch_domain_covered",
        "cpu_runtime_overhead_covered",
        "scheduler_and_arrival_process_covered",
        "per_request_generated_length_distribution_known",
    }:
        raise CalibrationError("unexpected external-validation requirement set")
    if any(requirements[key] is not True for key in required_boolean_keys):
        raise CalibrationError("external-validation boolean requirements must remain true")
    _decimal(
        requirements.get("maximum_external_wape_pct"),
        "external_validation_requirements.maximum_external_wape_pct",
        positive=True,
    )
    all_external_requirements_met = all(checks[key] for key in required_boolean_keys)
    if gate.get("out_of_domain_action") != "do_not_emit_performance_prediction":
        raise CalibrationError("out-of-domain action must remain fail-closed")

    blocker_messages = {
        "tensor_parallelism_match": "the MLPerf and Vidur tensor-parallel domains differ",
        "total_token_budget_covered": "the MLPerf token budget exceeds the Vidur profile domain",
        "maximum_sequence_length_covered": "the MLPerf maximum sequence exceeds the Vidur model-length domain",
        "weight_precision_match": "Vidur fit is FP16 while the MLPerf submission uses FP8 weights",
        "kv_cache_precision_match": "Vidur fit is FP16 while the MLPerf submission uses an FP8 KV cache",
        "runtime_kernel_match_or_validated_bridge": "no validated bridge exists from Vidur profiling kernels to TensorRT 10.8",
        "declared_batch_domain_covered": (
            f"MLPerf gpu_batch_size={sut['gpu_batch_size']} exceeds the Vidur decode profile maximum "
            f"of {scope['decode_batch_size_max']}"
        ),
        "cpu_runtime_overhead_covered": "the public Vidur anchor contains no CPU/runtime overhead profile",
        "scheduler_and_arrival_process_covered": "the public anchor contains no TensorRT scheduler/arrival timing trace",
        "per_request_generated_length_distribution_known": "MLPerf publishes only aggregate generated-token counts for this result",
    }
    blockers = [
        blocker_messages[key] for key in required_boolean_keys if not checks[key]
    ]
    if all_external_requirements_met:
        raise CalibrationError(
            "protocol unexpectedly permits external validation without an implemented simulator"
        )
    return {
        "status": "not_run_out_of_domain",
        "evidence_artifact_ids": sorted(evidence_ids),
        "declared_mlperf_sut": sut,
        "vidur_fit_scope": scope,
        "detail_log_observations": observed_details,
        "qsl_permutation_u32le_sha256": qsl_digests[0],
        "checks": checks,
        "requirements": requirements,
        "all_external_requirements_met": False,
        "held_out_targets": gate.get("held_out_targets"),
        "predictions": None,
        "errors": None,
        "blockers": blockers,
    }


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def fit_calibration(
    *,
    protocol_path: Path,
    anchor_bundle_dir: Path,
    anchor_verification_path: Path,
    workload_evidence_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for path, description in (
        (protocol_path, "calibration protocol"),
        (anchor_verification_path, "anchor verification receipt"),
    ):
        if path.is_symlink():
            raise CalibrationError(f"{description} must not be a symlink")
    for path, description in (
        (anchor_bundle_dir, "anchor bundle"),
        (workload_evidence_dir, "workload evidence"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise CalibrationError(f"{description} must be a non-symlink directory")
    if output_dir.is_symlink():
        raise CalibrationError(f"output directory must not be a symlink: {output_dir}")
    protocol_path = protocol_path.resolve()
    anchor_bundle_dir = anchor_bundle_dir.resolve()
    anchor_verification_path = anchor_verification_path.resolve()
    workload_evidence_dir = workload_evidence_dir.resolve()
    output_dir = output_dir.resolve()
    protocol, protocol_bytes = _load_object(protocol_path, "calibration protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise CalibrationError("unsupported calibration protocol schema")
    calibration_id = _string(protocol.get("calibration_id"), "protocol.calibration_id")
    if (
        calibration_id == OFFICIAL_CALIBRATION_ID
        and _sha256_bytes(protocol_bytes) != OFFICIAL_PROTOCOL_SHA256
    ):
        raise CalibrationError("official calibration protocol digest mismatch")
    if output_dir.exists() or output_dir.is_symlink():
        raise CalibrationError(f"output directory already exists: {output_dir}")
    for root, description in (
        (anchor_bundle_dir, "anchor bundle"),
        (workload_evidence_dir, "workload evidence"),
    ):
        try:
            output_dir.relative_to(root)
        except ValueError:
            pass
        else:
            raise CalibrationError(f"output directory must be outside the {description}")

    bundle, declarations, paths, verification = _validate_anchor_inputs(
        protocol, anchor_bundle_dir, anchor_verification_path
    )
    workload_receipt, lengths_path, pairs = _validate_workload_evidence(
        protocol, workload_evidence_dir
    )

    # MLPerf files are not opened until this fit and internal evaluation return.
    models, predictor_rows, evaluation_rows, overall = _fit_models(
        protocol, declarations, paths
    )
    operator_valid = all(model["result"] == "pass" for model in models)
    end_to_end = _assess_end_to_end_gate(protocol, paths, pairs)

    claim_policy = _mapping(protocol.get("claim_policy"), "protocol.claim_policy")
    if claim_policy.get("frontier_timing_usable_for_performance") is not False:
        raise CalibrationError("claim policy must keep Frontier performance timing disabled")
    if claim_policy.get("paper_result_eligible") is not False:
        raise CalibrationError("claim policy must keep paper-result eligibility disabled")
    if claim_policy.get("llama31_w8a16_inheritance_allowed") is not False:
        raise CalibrationError("claim policy must forbid Llama 3.1 W8A16 inheritance")
    state = (
        claim_policy.get("operator_calibration_pass_state")
        if operator_valid
        else claim_policy.get("operator_calibration_fail_state")
    )
    _string(state, "claim timing state")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        protocol_copy = temporary / "calibration-protocol.json"
        protocol_copy.write_bytes(protocol_bytes)
        inputs_dir = temporary / "inputs"
        inputs_dir.mkdir()
        workload_receipt_copy = inputs_dir / "workload-evidence.json"
        workload_receipt_bytes = (
            workload_evidence_dir / "workload-evidence.json"
        ).read_bytes()
        workload_receipt_copy.write_bytes(workload_receipt_bytes)
        lengths_copy = inputs_dir / "lengths.csv"
        shutil.copyfile(lengths_path, lengths_copy)

        predictor_path = temporary / "predictor-tables.csv"
        _write_csv(
            predictor_path,
            ["model_id", "coordinate_0", "coordinate_1", "time_ms"],
            predictor_rows,
        )
        evaluation_path = temporary / "operator-evaluation.csv"
        _write_csv(
            evaluation_path,
            [
                "model_id",
                "coordinate_0",
                "coordinate_1",
                "observed_ms",
                "predicted_ms",
                "absolute_error_ms",
                "absolute_percentage_error_pct",
            ],
            evaluation_rows,
        )

        artifacts = []
        for artifact_id, relative, kind in (
            ("calibration.protocol", Path("calibration-protocol.json"), "calibration_protocol"),
            (
                "workload.evidence",
                Path("inputs/workload-evidence.json"),
                "workload_evidence_receipt",
            ),
            ("workload.lengths", Path("inputs/lengths.csv"), "workload_lengths"),
            ("predictor.tables", Path("predictor-tables.csv"), "predictor_tables"),
            (
                "predictor.operator_evaluation",
                Path("operator-evaluation.csv"),
                "operator_evaluation",
            ),
        ):
            path = temporary / relative
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "kind": kind,
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )

        manifest = {
            "schema": CANDIDATE_SCHEMA,
            "result": "pass" if operator_valid else "fail",
            "calibration_id": calibration_id,
            "inputs": {
                "anchor": {
                    "anchor_id": bundle["anchor"]["anchor_id"],
                    "anchor_manifest_sha256": bundle["anchor"]["manifest"]["sha256"],
                    "bundle_manifest_sha256": _sha256_file(
                        anchor_bundle_dir / "bundle-manifest.json"
                    ),
                    "verification_receipt_sha256": _sha256_file(
                        anchor_verification_path
                    ),
                    "timing_claim_state": bundle["eligibility"]["timing_claim_state"],
                },
                "fit_artifact_ids": overall["fit_artifact_ids"],
                "forbidden_fit_source": "mlperf",
                "mlperf_artifacts_opened_during_fit": overall[
                    "mlperf_artifacts_opened_during_fit"
                ],
                "workload_evidence": {
                    "dataset_sha256": workload_receipt["dataset"]["sha256"],
                    "lengths_sha256": workload_receipt["lengths"]["sha256"],
                    "pairs_u32le_sha256": workload_receipt["lengths"][
                        "pairs_u32le_sha256"
                    ],
                },
            },
            "predictor": {
                "algorithm": protocol["fit_protocol"]["interpolator"]["algorithm"],
                "split": protocol["fit_protocol"]["split"],
                "deployment_refit": protocol["fit_protocol"]["deployment_refit"],
                "models": models,
                "overall": overall,
            },
            "end_to_end_external_validation": end_to_end,
            "artifacts": artifacts,
            "eligibility": {
                "anchor_integrity_valid": True,
                "workload_shape_integrity_valid": True,
                "mlperf_excluded_from_fit": True,
                "operator_predictor_calibration_valid": operator_valid,
                "end_to_end_external_validation_valid": False,
                "frontier_timing_usable_for_performance": False,
                "paper_result_eligible": False,
                "llama31_w8a16_inheritance_allowed": False,
                "timing_claim_state": state,
                "blockers": end_to_end["blockers"] if operator_valid else [
                    "one or more operator/collective predictors failed the declared thresholds"
                ],
            },
        }
        manifest_path = temporary / "candidate-manifest.json"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--anchor-bundle-dir", type=Path, required=True)
    parser.add_argument("--anchor-verification", type=Path, required=True)
    parser.add_argument("--workload-evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = fit_calibration(
            protocol_path=args.protocol,
            anchor_bundle_dir=args.anchor_bundle_dir,
            anchor_verification_path=args.anchor_verification,
            workload_evidence_dir=args.workload_evidence_dir,
            output_dir=args.output_dir,
        )
    except (CalibrationError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"calibrated {result['calibration_id']}: "
        f"models={result['predictor']['overall']['models']} "
        f"evaluation_keys={result['predictor']['overall']['evaluation_keys']} "
        f"state={result['eligibility']['timing_claim_state']}"
    )
    print(
        "external_validation="
        f"{result['end_to_end_external_validation']['status']}"
    )
    print(f"candidate={args.output_dir.resolve() / 'candidate-manifest.json'}")
    return 0 if result["result"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
