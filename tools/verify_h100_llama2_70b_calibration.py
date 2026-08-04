#!/usr/bin/env python3
"""Independently reproduce and verify the public H100 timing candidate.

The verifier intentionally does not import the fitter.  It reconstructs the
Vidur split and interpolation results from the pinned profile bytes, rebuilds
both CSV artifacts byte-for-byte, and independently re-evaluates the MLPerf
applicability gate.  A passing receipt validates the operator-level candidate;
it does not turn the out-of-domain MLPerf result into end-to-end validation.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_EVEN
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
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
VERIFICATION_SCHEMA = {
    "name": "hbfsim.validation.timing-calibration-verification",
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
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_INTEGER_RE = re.compile(r"^(0|[1-9][0-9]*)$")
U32_MAX = 2**32 - 1
EXPECTED_CANDIDATE_ARTIFACTS = {
    "calibration.protocol": ("calibration_protocol", "calibration-protocol.json"),
    "workload.evidence": (
        "workload_evidence_receipt",
        "inputs/workload-evidence.json",
    ),
    "workload.lengths": ("workload_lengths", "inputs/lengths.csv"),
    "predictor.tables": ("predictor_tables", "predictor-tables.csv"),
    "predictor.operator_evaluation": (
        "operator_evaluation",
        "operator-evaluation.csv",
    ),
}


class CalibrationVerificationError(ValueError):
    """The candidate cannot be independently reproduced from its pinned inputs."""


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKey(key)
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except OSError as error:
        raise CalibrationVerificationError(
            f"cannot read {label} {path}: {error}"
        ) from error
    except UnicodeDecodeError as error:
        raise CalibrationVerificationError(f"{label} is not UTF-8: {path}") from error
    except _DuplicateKey as error:
        raise CalibrationVerificationError(
            f"{label} contains duplicate key {error.args[0]!r}"
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise CalibrationVerificationError(
            f"invalid JSON in {label} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CalibrationVerificationError(f"{label} must be a JSON object")
    return value, raw


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationVerificationError(f"{path} must be an object")
    return value


def _list(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = " non-empty" if nonempty else ""
        raise CalibrationVerificationError(f"{path} must be a{qualifier} array")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationVerificationError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CalibrationVerificationError(f"{path} must be an integer >= {minimum}")
    return value


def _decimal(value: Any, path: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise CalibrationVerificationError(f"{path} must be a finite decimal")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise CalibrationVerificationError(f"{path} must be a finite decimal") from error
    if not number.is_finite() or (positive and number <= 0):
        qualifier = "positive " if positive else ""
        raise CalibrationVerificationError(f"{path} must be a finite {qualifier}decimal")
    return number


def _hex64(value: Any, path: str) -> str:
    digest = _text(value, path)
    if not HEX64_RE.fullmatch(digest):
        raise CalibrationVerificationError(
            f"{path} must be 64 lowercase hexadecimal digits"
        )
    return digest


def _relative(value: Any, path: str) -> Path:
    relative = Path(_text(value, path))
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise CalibrationVerificationError(f"{path} must be a safe relative path")
    return relative


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_decimal(value: Decimal, places: int = 12) -> str:
    rounded = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
    text = format(rounded, "f").rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def _median(values: Iterable[Decimal]) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise CalibrationVerificationError("empty median input")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _nearest_rank(values: list[Decimal], percentile: int) -> Decimal:
    if not values:
        raise CalibrationVerificationError("empty percentile input")
    ordered = sorted(values)
    rank = (Decimal(len(ordered)) * Decimal(percentile) / Decimal(100)).to_integral_value(
        rounding=ROUND_CEILING
    )
    return ordered[int(rank) - 1]


def _regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise CalibrationVerificationError(f"cannot stat {label} {path}: {error}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise CalibrationVerificationError(f"{label} is not a regular non-symlink file")


def _candidate_artifacts(
    candidate_dir: Path, manifest: dict[str, Any]
) -> dict[str, Path]:
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise CalibrationVerificationError("candidate root must be a non-symlink directory")
    records = _list(manifest.get("artifacts"), "candidate.artifacts", nonempty=True)
    if len(records) != len(EXPECTED_CANDIDATE_ARTIFACTS):
        raise CalibrationVerificationError("candidate artifact count mismatch")
    resolved: dict[str, Path] = {}
    declared_files = {Path("candidate-manifest.json")}
    for index, raw_record in enumerate(records):
        record = _object(raw_record, f"candidate.artifacts[{index}]")
        artifact_id = _text(
            record.get("artifact_id"), f"candidate.artifacts[{index}].artifact_id"
        )
        if artifact_id in resolved or artifact_id not in EXPECTED_CANDIDATE_ARTIFACTS:
            raise CalibrationVerificationError(
                f"unexpected or duplicate candidate artifact {artifact_id}"
            )
        expected_kind, expected_path = EXPECTED_CANDIDATE_ARTIFACTS[artifact_id]
        if record.get("kind") != expected_kind or record.get("path") != expected_path:
            raise CalibrationVerificationError(
                f"candidate artifact contract mismatch for {artifact_id}"
            )
        relative = _relative(record.get("path"), f"candidate artifact {artifact_id}.path")
        path = candidate_dir / relative
        _regular_file(path, f"candidate artifact {artifact_id}")
        expected_bytes = _integer(
            record.get("bytes"), f"candidate artifact {artifact_id}.bytes", minimum=1
        )
        expected_sha = _hex64(
            record.get("sha256"), f"candidate artifact {artifact_id}.sha256"
        )
        if path.stat().st_size != expected_bytes or _sha_file(path) != expected_sha:
            raise CalibrationVerificationError(
                f"candidate artifact digest mismatch for {artifact_id}"
            )
        resolved[artifact_id] = path
        declared_files.add(relative)
    if set(resolved) != set(EXPECTED_CANDIDATE_ARTIFACTS):
        raise CalibrationVerificationError("candidate artifact id census mismatch")

    observed_files: set[Path] = set()
    observed_dirs: set[Path] = set()
    for current, directories, files in os.walk(candidate_dir, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise CalibrationVerificationError("candidate contains an unsafe directory entry")
            observed_dirs.add(path.relative_to(candidate_dir))
        for name in files:
            path = current_path / name
            _regular_file(path, "candidate file")
            observed_files.add(path.relative_to(candidate_dir))
    if observed_dirs != {Path("inputs")} or observed_files != declared_files:
        raise CalibrationVerificationError(
            "candidate directory contains missing, unattached, or unexpected entries"
        )
    return resolved


def _anchor_chain(
    protocol: dict[str, Any],
    candidate: dict[str, Any],
    bundle_dir: Path,
    verification_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path], dict[str, Any]]:
    contract = _object(protocol.get("anchor"), "protocol.anchor")
    bundle_path = bundle_dir / "bundle-manifest.json"
    anchor_copy_path = bundle_dir / "anchor-manifest.json"
    _regular_file(bundle_path, "anchor bundle manifest")
    _regular_file(anchor_copy_path, "anchor manifest copy")
    _regular_file(verification_path, "anchor verification receipt")
    bundle, bundle_bytes = _read_json(bundle_path, "anchor bundle manifest")
    anchor, anchor_bytes = _read_json(anchor_copy_path, "anchor manifest copy")
    verification, verification_bytes = _read_json(
        verification_path, "anchor verification receipt"
    )
    if bundle.get("schema") != contract.get("bundle_schema") or bundle.get("result") != "pass":
        raise CalibrationVerificationError("anchor bundle schema/result mismatch")
    if (
        verification.get("schema") != contract.get("verification_schema")
        or verification.get("result") != "pass"
    ):
        raise CalibrationVerificationError("anchor verification schema/result mismatch")
    anchor_id = _text(contract.get("anchor_id"), "protocol.anchor.anchor_id")
    expected_anchor_sha = _hex64(
        contract.get("manifest_sha256"), "protocol.anchor.manifest_sha256"
    )
    bundle_anchor = _object(bundle.get("anchor"), "anchor bundle.anchor")
    if bundle_anchor.get("anchor_id") != anchor_id or anchor.get("anchor_id") != anchor_id:
        raise CalibrationVerificationError("anchor id mismatch")
    if verification.get("anchor_id") != anchor_id:
        raise CalibrationVerificationError("anchor verification id mismatch")
    if _sha_bytes(anchor_bytes) != expected_anchor_sha:
        raise CalibrationVerificationError("anchor manifest protocol pin mismatch")
    bundle_anchor_manifest = _object(
        bundle_anchor.get("manifest"), "anchor bundle.anchor.manifest"
    )
    if bundle_anchor_manifest.get("sha256") != expected_anchor_sha:
        raise CalibrationVerificationError("anchor bundle manifest pin mismatch")
    verification_inputs = _object(verification.get("inputs"), "anchor verification.inputs")
    if _object(verification_inputs.get("bundle_manifest"), "verified bundle").get(
        "sha256"
    ) != _sha_bytes(bundle_bytes):
        raise CalibrationVerificationError("anchor verification binds another bundle")
    if _object(verification_inputs.get("anchor_manifest"), "verified anchor").get(
        "sha256"
    ) != expected_anchor_sha:
        raise CalibrationVerificationError("anchor verification binds another anchor")
    verification_checks = _object(verification.get("checks"), "anchor verification.checks")
    if not verification_checks or any(value is not True for value in verification_checks.values()):
        raise CalibrationVerificationError("anchor verification has a failed check")

    required_state = _text(
        contract.get("required_timing_claim_state"),
        "protocol.anchor.required_timing_claim_state",
    )
    for label, source in (
        ("bundle", _object(bundle.get("eligibility"), "anchor bundle.eligibility")),
        (
            "verification",
            _object(verification.get("eligibility"), "anchor verification.eligibility"),
        ),
    ):
        if source.get("timing_claim_state") != required_state:
            raise CalibrationVerificationError(f"anchor {label} state was promoted")
        for key in (
            "predictor_calibration_valid",
            "end_to_end_external_validation_valid",
            "frontier_timing_usable_for_performance",
            "paper_result_eligible",
        ):
            if source.get(key) is not False:
                raise CalibrationVerificationError(f"anchor {label} promotes {key}")

    candidate_inputs = _object(candidate.get("inputs"), "candidate.inputs")
    candidate_anchor = _object(candidate_inputs.get("anchor"), "candidate.inputs.anchor")
    expected_candidate_anchor = {
        "anchor_id": anchor_id,
        "anchor_manifest_sha256": expected_anchor_sha,
        "bundle_manifest_sha256": _sha_bytes(bundle_bytes),
        "verification_receipt_sha256": _sha_bytes(verification_bytes),
        "timing_claim_state": required_state,
    }
    if candidate_anchor != expected_candidate_anchor:
        raise CalibrationVerificationError("candidate anchor input record mismatch")

    declarations: dict[str, dict[str, Any]] = {}
    sources = _object(anchor.get("sources"), "anchor manifest.sources")
    if set(sources) != {"vidur", "mlperf"}:
        raise CalibrationVerificationError("anchor sources must be exactly Vidur and MLPerf")
    intended = _object(anchor.get("intended_use"), "anchor intended_use")
    if (
        intended.get("vidur_role") != "calibration_fit_input"
        or intended.get("mlperf_role") != "held_out_external_validation_only"
        or intended.get("mlperf_fit_allowed") is not False
    ):
        raise CalibrationVerificationError("anchor source-role boundary mismatch")
    for source_id, raw_source in sources.items():
        source = _object(raw_source, f"anchor.sources.{source_id}")
        for index, raw_artifact in enumerate(
            _list(source.get("artifacts"), f"anchor.sources.{source_id}.artifacts")
        ):
            artifact = _object(
                raw_artifact, f"anchor.sources.{source_id}.artifacts[{index}]"
            )
            artifact_id = _text(artifact.get("id"), "anchor artifact id")
            if artifact_id in declarations:
                raise CalibrationVerificationError(
                    f"duplicate anchor artifact declaration {artifact_id}"
                )
            declarations[artifact_id] = {**artifact, "source_id": source_id}

    bundle_records: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(
        _list(bundle.get("artifacts"), "anchor bundle.artifacts", nonempty=True)
    ):
        record = _object(raw_record, f"anchor bundle.artifacts[{index}]")
        artifact_id = _text(record.get("artifact_id"), "bundle artifact id")
        if artifact_id in bundle_records:
            raise CalibrationVerificationError(f"duplicate bundle artifact {artifact_id}")
        bundle_records[artifact_id] = record
    if set(bundle_records) != {"anchor.manifest", *declarations}:
        raise CalibrationVerificationError("anchor bundle artifact census mismatch")

    paths: dict[str, Path] = {}
    for artifact_id, declaration in declarations.items():
        record = bundle_records[artifact_id]
        if (
            record.get("source_id") != declaration["source_id"]
            or record.get("kind") != declaration.get("kind")
            or record.get("bytes") != declaration.get("bytes")
            or record.get("sha256") != declaration.get("sha256")
        ):
            raise CalibrationVerificationError(
                f"anchor declaration/bundle mismatch for {artifact_id}"
            )
        path = bundle_dir / _relative(record.get("bundle_path"), "bundle artifact path")
        _regular_file(path, f"anchor artifact {artifact_id}")
        if path.resolve() != path:
            raise CalibrationVerificationError(
                f"anchor artifact traverses a symlink: {artifact_id}"
            )
        if (
            path.stat().st_size != _integer(record.get("bytes"), "anchor bytes", minimum=1)
            or _sha_file(path) != _hex64(record.get("sha256"), "anchor sha256")
        ):
            raise CalibrationVerificationError(
                f"anchor artifact digest mismatch for {artifact_id}"
            )
        paths[artifact_id] = path
    return declarations, paths, bundle


def _workload_pairs(
    protocol: dict[str, Any], receipt_path: Path, lengths_path: Path
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    contract = _object(protocol.get("workload_evidence"), "protocol.workload_evidence")
    receipt, _ = _read_json(receipt_path, "copied workload receipt")
    if (
        receipt.get("schema") != contract.get("schema")
        or receipt.get("result") != "pass"
        or receipt.get("calibration_id") != protocol.get("calibration_id")
    ):
        raise CalibrationVerificationError("workload receipt identity mismatch")
    dataset_contract = _object(contract.get("dataset"), "workload dataset contract")
    dataset_receipt = _object(receipt.get("dataset"), "workload receipt.dataset")
    for key in ("source_url", "bytes", "md5", "sha256"):
        if dataset_receipt.get(key) != dataset_contract.get(key):
            raise CalibrationVerificationError(f"workload dataset {key} mismatch")
    if dataset_receipt.get("digest_checked_before_pickle_load") is not True:
        raise CalibrationVerificationError("workload receipt lacks pre-unpickle digest check")
    length_contract = _object(contract.get("lengths_csv"), "workload lengths contract")
    length_receipt = _object(receipt.get("lengths"), "workload receipt.lengths")
    if length_receipt.get("path") != "lengths.csv":
        raise CalibrationVerificationError("copied workload lengths path mismatch")
    for key in (
        "columns",
        "rows",
        "bytes",
        "sha256",
        "pairs_u32le_sha256",
        "input_tokens",
        "reference_output_tokens",
    ):
        if length_receipt.get(key) != length_contract.get(key):
            raise CalibrationVerificationError(f"workload length contract mismatch: {key}")
    if (
        lengths_path.stat().st_size != length_contract.get("bytes")
        or _sha_file(lengths_path) != length_contract.get("sha256")
    ):
        raise CalibrationVerificationError("workload lengths file digest mismatch")
    pairs: list[tuple[int, int]] = []
    with lengths_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != length_contract.get("columns"):
            raise CalibrationVerificationError("workload lengths header mismatch")
        for expected_index, row in enumerate(reader):
            if None in row or any(item is None for item in row.values()):
                raise CalibrationVerificationError("malformed workload length row")
            values = [row["qsl_idx"], row["input_tokens"], row["reference_output_tokens"]]
            if any(not CANONICAL_INTEGER_RE.fullmatch(item) for item in values):
                raise CalibrationVerificationError("non-canonical workload integer")
            qsl_idx, input_tokens, output_tokens = map(int, values)
            if qsl_idx != expected_index or not (0 < input_tokens <= U32_MAX) or not (
                0 < output_tokens <= U32_MAX
            ):
                raise CalibrationVerificationError("invalid workload length row")
            pairs.append((input_tokens, output_tokens))
    if len(pairs) != length_contract.get("rows"):
        raise CalibrationVerificationError("workload row count mismatch")
    pair_digest = hashlib.sha256(
        b"".join(struct.pack("<II", left, right) for left, right in pairs)
    ).hexdigest()
    if pair_digest != length_contract.get("pairs_u32le_sha256"):
        raise CalibrationVerificationError("workload pair digest mismatch")
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
    if any(observed[key] != length_contract.get(key) for key in observed):
        raise CalibrationVerificationError("workload length statistics mismatch")
    if receipt.get("scope") != {
        "contains_prompts": False,
        "contains_token_ids": False,
        "contains_reference_text": False,
        "establishes_timing_calibration": False,
    }:
        raise CalibrationVerificationError("workload evidence scope mismatch")
    return receipt, pairs


def _profile_coordinate(row: dict[str, str], spec: dict[str, Any]) -> Decimal:
    column = _text(spec.get("column"), "coordinate.column")
    raw = row.get(column)
    if raw is None or not CANONICAL_INTEGER_RE.fullmatch(raw):
        raise CalibrationVerificationError(f"invalid profile coordinate {column}")
    value = Decimal(raw)
    if spec.get("transform") == "identity":
        return value
    if spec.get("transform") == "square":
        return value**2
    raise CalibrationVerificationError("unsupported coordinate transform")


def _model_name(family: str, target: str, group: dict[str, str]) -> str:
    suffix = ",".join(f"{key}={group[key]}" for key in sorted(group))
    return f"{family}.{target}.{suffix}"


def _independent_prediction(
    point: tuple[Decimal, ...],
    positions: tuple[int, ...],
    axes: list[list[Decimal]],
    table: dict[tuple[Decimal, ...], Decimal],
) -> Decimal:
    """Evaluate linear/bilinear interpolation without the fitter's corner loop."""

    if len(point) == 1:
        x = point[0]
        lower_x = axes[0][positions[0] - 1]
        upper_x = axes[0][positions[0] + 1]
        lower_y = table[(lower_x,)]
        upper_y = table[(upper_x,)]
        return lower_y + (upper_y - lower_y) * (x - lower_x) / (upper_x - lower_x)
    if len(point) == 2:
        x, y = point
        x0 = axes[0][positions[0] - 1]
        x1 = axes[0][positions[0] + 1]
        y0 = axes[1][positions[1] - 1]
        y1 = axes[1][positions[1] + 1]
        along_y0 = table[(x0, y0)] + (table[(x1, y0)] - table[(x0, y0)]) * (
            x - x0
        ) / (x1 - x0)
        along_y1 = table[(x0, y1)] + (table[(x1, y1)] - table[(x0, y1)]) * (
            x - x0
        ) / (x1 - x0)
        return along_y0 + (along_y1 - along_y0) * (y - y0) / (y1 - y0)
    raise CalibrationVerificationError("only one- and two-dimensional models are supported")


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _reproduce_operator_candidate(
    protocol: dict[str, Any],
    declarations: dict[str, dict[str, Any]],
    paths: dict[str, Path],
) -> tuple[dict[str, Any], bytes, bytes, bool]:
    fit = _object(protocol.get("fit_protocol"), "protocol.fit_protocol")
    if (
        fit.get("time_unit") != "milliseconds"
        or fit.get("target_statistic") != "published_median"
        or fit.get("duplicate_key_policy") != "group_then_decimal_median"
        or fit.get("deployment_refit") != "all_unique_profile_keys_after_evaluation"
    ):
        raise CalibrationVerificationError("unsupported fit contract")
    expected_split = {
        "algorithm": "coordinate_interleaved_v1",
        "evaluation_coordinate_index_parity": 1,
        "axis_boundaries_are_training": True,
        "evaluation_requires_complete_adjacent_corner_set": True,
        "all_rows_with_the_same_profile_key_share_one_partition": True,
    }
    expected_interpolator = {
        "algorithm": "multilinear_adjacent_corners_v1",
        "coordinate_arithmetic": "decimal",
        "extrapolation_allowed": False,
        "negative_prediction_allowed": False,
    }
    if fit.get("split") != expected_split or fit.get("interpolator") != expected_interpolator:
        raise CalibrationVerificationError("split/interpolator contract was weakened")
    acceptance = _object(fit.get("acceptance"), "fit acceptance")
    maximum_wape = _decimal(acceptance.get("maximum_wape_pct"), "maximum WAPE", positive=True)
    maximum_p90 = _decimal(
        acceptance.get("maximum_p90_ape_pct"), "maximum P90 APE", positive=True
    )
    maximum_bias = _decimal(
        acceptance.get("maximum_absolute_signed_bias_pct"),
        "maximum signed bias",
        positive=True,
    )

    profile_rows: dict[str, list[dict[str, str]]] = {}
    predictor_rows: list[list[str]] = []
    evaluation_rows: list[list[str]] = []
    models: list[dict[str, Any]] = []
    all_truth: list[Decimal] = []
    all_prediction: list[Decimal] = []
    fit_ids: set[str] = set()
    for family_index, raw_family in enumerate(
        _list(fit.get("families"), "fit families", nonempty=True)
    ):
        family = _object(raw_family, f"fit family[{family_index}]")
        family_id = _text(family.get("id"), "fit family id")
        artifact_id = _text(family.get("artifact_id"), "fit artifact id")
        declaration = declarations.get(artifact_id)
        if declaration is None or declaration.get("source_id") != "vidur":
            raise CalibrationVerificationError("a fit family does not use Vidur")
        if declaration.get("kind") != "csv_profile":
            raise CalibrationVerificationError("a fit family does not use a CSV profile")
        fit_ids.add(artifact_id)
        if artifact_id not in profile_rows:
            with paths[artifact_id].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != declaration.get("columns"):
                    raise CalibrationVerificationError("Vidur profile header mismatch")
                imported = list(reader)
            if len(imported) != declaration.get("rows") or any(
                None in row or any(value is None for value in row.values())
                for row in imported
            ):
                raise CalibrationVerificationError("Vidur profile row contract mismatch")
            profile_rows[artifact_id] = imported
        filters = _object(family.get("filters"), f"{family_id}.filters")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in filters.items()):
            raise CalibrationVerificationError("fit filters must map strings")
        coordinates = [
            _object(item, f"{family_id}.coordinate")
            for item in _list(family.get("coordinates"), f"{family_id}.coordinates", nonempty=True)
        ]
        if len(coordinates) not in (1, 2):
            raise CalibrationVerificationError("fit coordinates must be one- or two-dimensional")
        targets = [
            _object(item, f"{family_id}.target")
            for item in _list(family.get("targets"), f"{family_id}.targets", nonempty=True)
        ]
        minimum_fraction = _decimal(
            family.get("minimum_evaluation_fraction"),
            f"{family_id}.minimum_evaluation_fraction",
            positive=True,
        )
        if minimum_fraction > 1:
            raise CalibrationVerificationError("evaluation fraction exceeds one")
        seen_groups: set[tuple[tuple[str, str], ...]] = set()
        for group_index, raw_group in enumerate(
            _list(family.get("group_values"), f"{family_id}.groups", nonempty=True)
        ):
            group = _object(raw_group, f"{family_id}.group[{group_index}]")
            if not group or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in group.items()
            ):
                raise CalibrationVerificationError("fit group must map strings")
            group_key = tuple(sorted(group.items()))
            if group_key in seen_groups:
                raise CalibrationVerificationError("duplicate fit group")
            seen_groups.add(group_key)
            selected = [
                row
                for row in profile_rows[artifact_id]
                if all(row.get(key) == value for key, value in {**filters, **group}.items())
            ]
            if not selected:
                raise CalibrationVerificationError("empty fit group")
            observations: dict[tuple[Decimal, ...], list[dict[str, str]]] = defaultdict(list)
            for row in selected:
                key = tuple(_profile_coordinate(row, spec) for spec in coordinates)
                observations[key].append(row)
            axes = [
                sorted({key[dimension] for key in observations})
                for dimension in range(len(coordinates))
            ]
            locations = [
                {value: index for index, value in enumerate(axis)} for axis in axes
            ]
            evaluation_points: list[tuple[Decimal, ...]] = []
            for point in sorted(observations):
                positions = tuple(
                    locations[dimension][value] for dimension, value in enumerate(point)
                )
                if not all(
                    position % 2 == 1 and 0 < position < len(axes[dimension]) - 1
                    for dimension, position in enumerate(positions)
                ):
                    continue
                if len(point) == 1:
                    needed = {
                        (axes[0][positions[0] - 1],),
                        (axes[0][positions[0] + 1],),
                    }
                else:
                    needed = {
                        (axes[0][positions[0] - 1], axes[1][positions[1] - 1]),
                        (axes[0][positions[0] - 1], axes[1][positions[1] + 1]),
                        (axes[0][positions[0] + 1], axes[1][positions[1] - 1]),
                        (axes[0][positions[0] + 1], axes[1][positions[1] + 1]),
                    }
                if needed.issubset(observations):
                    evaluation_points.append(point)
            evaluation_set = set(evaluation_points)
            fraction = Decimal(len(evaluation_points)) / Decimal(len(observations))
            if fraction < minimum_fraction:
                raise CalibrationVerificationError("evaluation fraction below protocol minimum")

            for target_index, raw_target in enumerate(targets):
                target_id = _text(raw_target.get("id"), "fit target id")
                column = _text(raw_target.get("column"), "fit target column")
                table: dict[tuple[Decimal, ...], Decimal] = {}
                raw_count = 0
                for point, rows in observations.items():
                    samples: list[Decimal] = []
                    for row in rows:
                        raw_value = row.get(column)
                        if raw_value is None or raw_value == "":
                            raise CalibrationVerificationError("empty timing observation")
                        samples.append(_decimal(raw_value, column, positive=True))
                    raw_count += len(samples)
                    table[point] = _median(samples)
                model_id = _model_name(family_id, target_id, group)
                table_digest = hashlib.sha256()
                split_digest = hashlib.sha256()
                for point in sorted(table):
                    coordinate_text = [_format_decimal(value, places=0) for value in point]
                    timing_text = _format_decimal(table[point])
                    table_digest.update(
                        ("\t".join([model_id, *coordinate_text, timing_text]) + "\n").encode()
                    )
                    partition = "evaluation" if point in evaluation_set else "training"
                    split_digest.update(
                        ("\t".join([model_id, *coordinate_text, partition]) + "\n").encode()
                    )
                    predictor_rows.append(
                        [
                            model_id,
                            coordinate_text[0],
                            coordinate_text[1] if len(coordinate_text) == 2 else "",
                            timing_text,
                        ]
                    )
                truths: list[Decimal] = []
                predictions: list[Decimal] = []
                absolute_errors: list[Decimal] = []
                percentage_errors: list[Decimal] = []
                for point in evaluation_points:
                    positions = tuple(
                        locations[dimension][value]
                        for dimension, value in enumerate(point)
                    )
                    truth = table[point]
                    prediction = _independent_prediction(point, positions, axes, table)
                    if prediction < 0:
                        raise CalibrationVerificationError("negative interpolation result")
                    absolute_error = abs(prediction - truth)
                    percentage_error = absolute_error / truth * Decimal(100)
                    truths.append(truth)
                    predictions.append(prediction)
                    absolute_errors.append(absolute_error)
                    percentage_errors.append(percentage_error)
                    coordinate_text = [_format_decimal(value, places=0) for value in point]
                    evaluation_rows.append(
                        [
                            model_id,
                            coordinate_text[0],
                            coordinate_text[1] if len(coordinate_text) == 2 else "",
                            _format_decimal(truth),
                            _format_decimal(prediction),
                            _format_decimal(absolute_error),
                            _format_decimal(percentage_error),
                        ]
                    )
                if not truths:
                    raise CalibrationVerificationError("model has no evaluation points")
                wape = sum(absolute_errors) / sum(truths) * Decimal(100)
                mape = sum(percentage_errors) / Decimal(len(percentage_errors))
                p90 = _nearest_rank(percentage_errors, 90)
                bias = (sum(predictions) - sum(truths)) / sum(truths) * Decimal(100)
                passed = wape <= maximum_wape and p90 <= maximum_p90 and abs(bias) <= maximum_bias
                models.append(
                    {
                        "model_id": model_id,
                        "family_id": family_id,
                        "target_id": target_id,
                        "target_column": column,
                        "artifact_id": artifact_id,
                        "group": dict(sorted(group.items())),
                        "coordinates": coordinates,
                        "raw_observations": raw_count,
                        "unique_profile_keys": len(table),
                        "training_keys": len(table) - len(evaluation_points),
                        "evaluation_keys": len(evaluation_points),
                        "evaluation_fraction": _format_decimal(fraction),
                        "domain": [
                            {
                                "min": _format_decimal(axis[0], places=0),
                                "max": _format_decimal(axis[-1], places=0),
                                "values": len(axis),
                            }
                            for axis in axes
                        ],
                        "deployment_table_sha256": table_digest.hexdigest(),
                        "split_sha256": split_digest.hexdigest(),
                        "metrics": {
                            "mape_pct": _format_decimal(mape),
                            "median_ape_pct": _format_decimal(_median(percentage_errors)),
                            "p90_ape_pct": _format_decimal(p90),
                            "wape_pct": _format_decimal(wape),
                            "signed_bias_pct": _format_decimal(bias),
                        },
                        "thresholds": {
                            "maximum_wape_pct": _format_decimal(maximum_wape),
                            "maximum_p90_ape_pct": _format_decimal(maximum_p90),
                            "maximum_absolute_signed_bias_pct": _format_decimal(maximum_bias),
                        },
                        "result": "pass" if passed else "fail",
                    }
                )
                all_truth.extend(truths)
                all_prediction.extend(predictions)
    model_ids = [model["model_id"] for model in models]
    if len(model_ids) != len(set(model_ids)):
        raise CalibrationVerificationError("duplicate reconstructed model id")
    predictor_rows.sort(key=lambda row: (row[0], Decimal(row[1]), Decimal(row[2] or 0)))
    evaluation_rows.sort(key=lambda row: (row[0], Decimal(row[1]), Decimal(row[2] or 0)))
    all_absolute = [
        abs(prediction - truth) for truth, prediction in zip(all_truth, all_prediction)
    ]
    ordered_models = sorted(models, key=lambda item: item["model_id"])
    overall = {
        "models": len(ordered_models),
        "passing_models": sum(model["result"] == "pass" for model in ordered_models),
        "evaluation_keys": len(all_truth),
        "wape_pct": _format_decimal(sum(all_absolute) / sum(all_truth) * Decimal(100)),
        "signed_bias_pct": _format_decimal(
            (sum(all_prediction) - sum(all_truth)) / sum(all_truth) * Decimal(100)
        ),
        "fit_artifact_ids": sorted(fit_ids),
        "mlperf_artifacts_opened_during_fit": [],
    }
    predictor = {
        "algorithm": expected_interpolator["algorithm"],
        "split": expected_split,
        "deployment_refit": fit["deployment_refit"],
        "models": ordered_models,
        "overall": overall,
    }
    predictor_bytes = _csv_bytes(
        ["model_id", "coordinate_0", "coordinate_1", "time_ms"], predictor_rows
    )
    evaluation_bytes = _csv_bytes(
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
    return predictor, predictor_bytes, evaluation_bytes, all(
        model["result"] == "pass" for model in ordered_models
    )


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise CalibrationVerificationError(f"cannot read {label}: {error}") from error
    except UnicodeDecodeError as error:
        raise CalibrationVerificationError(f"{label} is not UTF-8") from error


def _class_body(source: str, class_name: str, label: str) -> dict[str, ast.AST]:
    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as error:
        raise CalibrationVerificationError(f"invalid Python in {label}: {error}") from error
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise CalibrationVerificationError(f"{label} must define {class_name} once")
    assignments: dict[str, ast.AST] = {}
    for statement in classes[0].body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            key = statement.targets[0].id
            if key in assignments:
                raise CalibrationVerificationError(f"duplicate {class_name}.{key}")
            assignments[key] = statement.value
    return assignments


def _data_literal(node: ast.AST, path: str) -> Any:
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise CalibrationVerificationError(f"unsupported constant in {path}")
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_data_literal(item, f"{path}[]") for item in node.elts]
        return items if isinstance(node, ast.List) else tuple(items)
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise CalibrationVerificationError(f"dictionary unpacking in {path}")
            key = _data_literal(key_node, f"{path}.key")
            try:
                duplicate = key in result
            except TypeError as error:
                raise CalibrationVerificationError(f"unhashable key in {path}") from error
            if duplicate:
                raise CalibrationVerificationError(f"duplicate dictionary key in {path}")
            result[key] = _data_literal(value_node, f"{path}[{key!r}]")
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _data_literal(node.operand, f"{path}.operand")
        if isinstance(value, bool) or not isinstance(value, int):
            raise CalibrationVerificationError(f"non-integer unary expression in {path}")
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left = _data_literal(node.left, f"{path}.left")
        right = _data_literal(node.right, f"{path}.right")
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, int)
            or not isinstance(right, int)
        ):
            raise CalibrationVerificationError(f"non-integer arithmetic in {path}")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    raise CalibrationVerificationError(f"{path} is not data-only syntax")


def _assignment_value(assignments: dict[str, ast.AST], key: str, path: str) -> Any:
    if key not in assignments:
        raise CalibrationVerificationError(f"{path} is missing")
    return _data_literal(assignments[key], path)


def _assignment_expression(assignments: dict[str, ast.AST], key: str, path: str) -> str:
    if key not in assignments:
        raise CalibrationVerificationError(f"{path} is missing")
    return ast.unparse(assignments[key])


def _mllog(path: Path, label: str) -> dict[str, list[Any]]:
    events: dict[str, list[Any]] = defaultdict(list)
    for line_number, line in enumerate(_read_text(path, label).splitlines(), start=1):
        if ":::MLLOG " not in line:
            continue
        try:
            event = json.loads(
                line.split(":::MLLOG ", 1)[1],
                parse_float=Decimal,
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except (_DuplicateKey, json.JSONDecodeError, ValueError) as error:
            raise CalibrationVerificationError(
                f"invalid MLLOG event in {label} line {line_number}"
            ) from error
        if not isinstance(event, dict) or not isinstance(event.get("key"), str):
            raise CalibrationVerificationError(f"malformed MLLOG event in {label}")
        events[event["key"]].append(event.get("value"))
    return events


def _one_event(events: dict[str, list[Any]], key: str, label: str) -> Any:
    values = events.get(key, [])
    if len(values) != 1:
        raise CalibrationVerificationError(f"{label} must contain one {key} event")
    return values[0]


def _event_matches(actual: Any, expected: Any, path: str) -> None:
    if isinstance(expected, int):
        if isinstance(actual, bool) or actual != expected:
            raise CalibrationVerificationError(f"{path} event mismatch")
        return
    if _decimal(actual, path) != _decimal(expected, f"{path}.expected"):
        raise CalibrationVerificationError(f"{path} event mismatch")


def _reproduce_external_gate(
    protocol: dict[str, Any], paths: dict[str, Path], pairs: list[tuple[int, int]]
) -> dict[str, Any]:
    gate = _object(protocol.get("end_to_end_gate"), "protocol.end_to_end_gate")
    roles = _object(gate.get("artifact_roles"), "end_to_end artifact roles")
    role_ids = [_text(value, f"artifact role {key}") for key, value in sorted(roles.items())]
    listed_ids = [
        _text(value, "end_to_end evidence id")
        for value in _list(gate.get("evidence_artifacts"), "end_to_end evidence artifacts")
    ]
    if sorted(role_ids) != sorted(listed_ids) or any(item not in paths for item in role_ids):
        raise CalibrationVerificationError("external evidence artifact census mismatch")
    sut = _object(gate.get("declared_mlperf_sut"), "declared MLPerf SUT")
    scope = _object(gate.get("vidur_fit_scope"), "Vidur fit scope")
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
        _integer(sut.get(key), f"declared MLPerf SUT.{key}", minimum=1)
    for key in (
        "tensor_parallelism",
        "decode_batch_size_min",
        "decode_batch_size_max",
        "total_tokens_min",
        "total_tokens_max",
        "max_model_len",
    ):
        _integer(scope.get(key), f"Vidur fit scope.{key}", minimum=1)
    for key in ("cpu_runtime_overhead_profile_present", "scheduler_trace_profile_present"):
        if not isinstance(scope.get(key), bool):
            raise CalibrationVerificationError(f"Vidur fit scope.{key} must be boolean")
    for key in (
        "weight_precision",
        "kv_cache_precision",
        "framework",
        "batch_scheduler_policy",
        "context_chunking_policy",
    ):
        _text(sut.get(key), f"declared MLPerf SUT.{key}")
    _text(scope.get("precision"), "Vidur fit scope.precision")

    base_source = _read_text(paths[roles["base_config"]], "NVIDIA base config")
    offline_source = _read_text(paths[roles["offline_config"]], "NVIDIA Offline config")
    server_source = _read_text(paths[roles["server_config"]], "NVIDIA Server config")
    base = _class_body(base_source, "GPUBaseConfig", "NVIDIA base config")
    build = _object(
        _assignment_value(base, "trtllm_build_flags", "GPUBaseConfig.trtllm_build_flags"),
        "NVIDIA build flags",
    )
    runtime = _object(
        _assignment_value(base, "trtllm_runtime_flags", "GPUBaseConfig.trtllm_runtime_flags"),
        "NVIDIA runtime flags",
    )
    if (
        build.get("max_input_len") != sut["max_input_len"]
        or build.get("max_seq_len") != sut["max_input_len"] + sut["max_output_len"]
        or runtime.get("batch_scheduler_policy") != sut.get("batch_scheduler_policy")
        or runtime.get("context_chunking_policy") != sut.get("context_chunking_policy")
    ):
        raise CalibrationVerificationError("NVIDIA base configuration mismatch")
    for label, source in (("Offline", offline_source), ("Server", server_source)):
        hopper = _class_body(source, f"Hopper{label}GPUBaseConfig", f"NVIDIA {label} config")
        if _assignment_value(hopper, "precision", f"{label}.precision") != sut.get(
            "weight_precision"
        ):
            raise CalibrationVerificationError(f"NVIDIA {label} precision mismatch")
        checkpoint = _object(
            _assignment_value(hopper, "trtllm_checkpoint_flags", f"{label}.checkpoint"),
            f"NVIDIA {label} checkpoint flags",
        )
        if checkpoint.get("kv_cache_dtype") != sut.get("kv_cache_precision"):
            raise CalibrationVerificationError(f"NVIDIA {label} KV precision mismatch")
        pp2 = _class_body(source, "H100_SXM_80GB_PP2x1", f"NVIDIA {label} config")
        batches = _object(
            _assignment_value(pp2, "gpu_batch_size", f"{label}.gpu_batch_size"),
            f"NVIDIA {label} batch sizes",
        )
        pp2_build = _object(
            _assignment_value(pp2, "trtllm_build_flags", f"{label}.build_flags"),
            f"NVIDIA {label} build flags",
        )
        if batches.get("llama2-70b") != sut["gpu_batch_size"]:
            raise CalibrationVerificationError(f"NVIDIA {label} batch size mismatch")
        for key in ("tensor_parallelism", "pipeline_parallelism", "max_num_tokens"):
            if pp2_build.get(key) != sut.get(key):
                raise CalibrationVerificationError(f"NVIDIA {label} {key} mismatch")
        x4 = _class_body(source, "H100_SXM_80GB_PP2x4", f"NVIDIA {label} config")
        if not _assignment_expression(x4, "system", f"{label}.system").endswith(
            "H100_SXM_80GBx8"
        ):
            raise CalibrationVerificationError(f"NVIDIA {label} system mismatch")

    preprocessor_source = _read_text(
        paths[roles["preprocessor"]], "NVIDIA preprocessor"
    )
    try:
        preprocessor_tree = ast.parse(preprocessor_source)
    except SyntaxError as error:
        raise CalibrationVerificationError("invalid NVIDIA preprocessor") from error
    constants: dict[str, Any] = {}
    for statement in preprocessor_tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            try:
                constants[statement.targets[0].id] = ast.literal_eval(statement.value)
            except (TypeError, ValueError):
                pass
    if constants.get("G_MAX_INPUT_TOK_LEN", 0) < max(left for left, _ in pairs):
        raise CalibrationVerificationError("NVIDIA preprocessor input limit mismatch")
    generation, _ = _read_json(paths[roles["generation_config"]], "generation config")
    generation_config = _object(generation.get("generation_config"), "generation_config")
    if (
        generation_config.get("max_output_len") != sut["max_output_len"]
        or generation_config.get("streaming") is not True
    ):
        raise CalibrationVerificationError("NVIDIA generation configuration mismatch")

    expectations = _object(gate.get("detail_log_expectations"), "detail log expectations")
    qsl_digests: list[str] = []
    observations: dict[str, Any] = {}
    for scenario in ("offline", "server"):
        events = _mllog(paths[roles[f"{scenario}_detail"]], f"{scenario} detail log")
        qsl = _one_event(events, "loaded_qsl_set", f"{scenario} detail log")
        if (
            not isinstance(qsl, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in qsl)
            or len(qsl) != expectations.get("loaded_qsl_set_count")
            or sorted(qsl) != list(range(len(qsl)))
        ):
            raise CalibrationVerificationError(f"{scenario} QSL set is not a permutation")
        digest = hashlib.sha256(
            b"".join(struct.pack("<I", item) for item in qsl)
        ).hexdigest()
        if digest != expectations.get("loaded_qsl_set_u32le_sha256"):
            raise CalibrationVerificationError(f"{scenario} QSL digest mismatch")
        qsl_digests.append(digest)
        scenario_expectations = _object(expectations.get(scenario), f"{scenario} expectations")
        observed: dict[str, Any] = {}
        for key, expected in scenario_expectations.items():
            actual = _one_event(events, key, f"{scenario} detail log")
            _event_matches(actual, expected, f"{scenario}.{key}")
            observed[key] = str(actual) if isinstance(actual, Decimal) else actual
        observations[scenario] = observed
    if len(set(qsl_digests)) != 1:
        raise CalibrationVerificationError("Offline and Server QSL permutations differ")

    checks = {
        "exact_openorca_input_lengths": True,
        "exact_mlperf_qsl_permutation": True,
        "exact_mlperf_query_counts": True,
        "tensor_parallelism_match": (
            sut["tensor_parallelism"] == scope.get("tensor_parallelism")
        ),
        "total_token_budget_covered": sut["max_num_tokens"] <= scope["total_tokens_max"],
        "maximum_sequence_length_covered": (
            sut["max_input_len"] + sut["max_output_len"] <= scope["max_model_len"]
        ),
        "weight_precision_match": sut["weight_precision"] == scope.get("precision"),
        "kv_cache_precision_match": sut["kv_cache_precision"] == scope.get("precision"),
        "runtime_kernel_match_or_validated_bridge": False,
        "declared_batch_domain_covered": (
            sut["gpu_batch_size"] <= scope["decode_batch_size_max"]
        ),
        "cpu_runtime_overhead_covered": scope.get("cpu_runtime_overhead_profile_present"),
        "scheduler_and_arrival_process_covered": scope.get("scheduler_trace_profile_present"),
        "per_request_generated_length_distribution_known": False,
    }
    requirements = _object(
        gate.get("external_validation_requirements"), "external validation requirements"
    )
    required_keys = sorted(set(requirements) - {"maximum_external_wape_pct"})
    expected_required_keys = {
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
    }
    if set(required_keys) != expected_required_keys or any(
        requirements[key] is not True for key in required_keys
    ):
        raise CalibrationVerificationError("external validation requirement set mismatch")
    _decimal(
        requirements.get("maximum_external_wape_pct"),
        "maximum external WAPE",
        positive=True,
    )
    if gate.get("out_of_domain_action") != "do_not_emit_performance_prediction":
        raise CalibrationVerificationError("external gate is not fail-closed")
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
    blockers = [blocker_messages[key] for key in required_keys if not checks[key]]
    if all(checks[key] for key in required_keys):
        raise CalibrationVerificationError(
            "external comparison became eligible without an implemented simulator"
        )
    return {
        "status": "not_run_out_of_domain",
        "evidence_artifact_ids": sorted(role_ids),
        "declared_mlperf_sut": sut,
        "vidur_fit_scope": scope,
        "detail_log_observations": observations,
        "qsl_permutation_u32le_sha256": qsl_digests[0],
        "checks": checks,
        "requirements": requirements,
        "all_external_requirements_met": False,
        "held_out_targets": gate.get("held_out_targets"),
        "predictions": None,
        "errors": None,
        "blockers": blockers,
    }


def verify_candidate(
    *,
    protocol_path: Path,
    anchor_bundle_dir: Path,
    anchor_verification_path: Path,
    candidate_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    for path, label in (
        (protocol_path, "protocol"),
        (anchor_verification_path, "anchor verification"),
    ):
        if path.is_symlink():
            raise CalibrationVerificationError(f"{label} must not be a symlink")
    for path, label in (
        (anchor_bundle_dir, "anchor bundle"),
        (candidate_dir, "candidate"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise CalibrationVerificationError(f"{label} must be a non-symlink directory")
    if output_path.is_symlink():
        raise CalibrationVerificationError("output must not be a symlink")
    protocol_path = protocol_path.resolve()
    anchor_bundle_dir = anchor_bundle_dir.resolve()
    anchor_verification_path = anchor_verification_path.resolve()
    candidate_dir = candidate_dir.resolve()
    output_path = output_path.resolve()
    if output_path.exists() or output_path.is_symlink():
        raise CalibrationVerificationError(f"output already exists: {output_path}")
    for root, label in (
        (anchor_bundle_dir, "anchor bundle"),
        (candidate_dir, "candidate"),
    ):
        try:
            output_path.relative_to(root)
        except ValueError:
            pass
        else:
            raise CalibrationVerificationError(f"output must be outside the {label}")

    protocol, protocol_bytes = _read_json(protocol_path, "calibration protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise CalibrationVerificationError("unsupported calibration protocol schema")
    calibration_id = _text(protocol.get("calibration_id"), "protocol.calibration_id")
    if (
        calibration_id == OFFICIAL_CALIBRATION_ID
        and _sha_bytes(protocol_bytes) != OFFICIAL_PROTOCOL_SHA256
    ):
        raise CalibrationVerificationError("official calibration protocol digest mismatch")
    verification_contract = _object(protocol.get("verification"), "protocol.verification")
    verification_schema = _object(
        verification_contract.get("schema"), "protocol.verification.schema"
    )
    if verification_schema != VERIFICATION_SCHEMA:
        raise CalibrationVerificationError("unsupported calibration verification schema")
    required_checks = [
        _text(item, "protocol.verification.required_checks[]")
        for item in _list(
            verification_contract.get("required_checks"),
            "protocol.verification.required_checks",
            nonempty=True,
        )
    ]
    if len(required_checks) != len(set(required_checks)):
        raise CalibrationVerificationError("duplicate required verification check")

    candidate_manifest_path = candidate_dir / "candidate-manifest.json"
    _regular_file(candidate_manifest_path, "candidate manifest")
    candidate, candidate_bytes = _read_json(candidate_manifest_path, "candidate manifest")
    if (
        candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("calibration_id") != calibration_id
    ):
        raise CalibrationVerificationError("candidate identity mismatch")
    artifacts = _candidate_artifacts(candidate_dir, candidate)
    if artifacts["calibration.protocol"].read_bytes() != protocol_bytes:
        raise CalibrationVerificationError("candidate protocol copy differs from input")

    declarations, source_paths, anchor_bundle = _anchor_chain(
        protocol,
        candidate,
        anchor_bundle_dir,
        anchor_verification_path,
    )
    workload_receipt, pairs = _workload_pairs(
        protocol, artifacts["workload.evidence"], artifacts["workload.lengths"]
    )
    workload_input = _object(
        _object(candidate.get("inputs"), "candidate.inputs").get("workload_evidence"),
        "candidate.inputs.workload_evidence",
    )
    expected_workload_input = {
        "dataset_sha256": workload_receipt["dataset"]["sha256"],
        "lengths_sha256": workload_receipt["lengths"]["sha256"],
        "pairs_u32le_sha256": workload_receipt["lengths"]["pairs_u32le_sha256"],
    }
    if workload_input != expected_workload_input:
        raise CalibrationVerificationError("candidate workload input record mismatch")

    predictor, predictor_bytes, evaluation_bytes, operator_valid = (
        _reproduce_operator_candidate(protocol, declarations, source_paths)
    )
    if artifacts["predictor.tables"].read_bytes() != predictor_bytes:
        raise CalibrationVerificationError("predictor table bytes were not reproduced")
    if artifacts["predictor.operator_evaluation"].read_bytes() != evaluation_bytes:
        raise CalibrationVerificationError("operator evaluation bytes were not reproduced")
    if candidate.get("predictor") != predictor:
        raise CalibrationVerificationError("candidate predictor summary was not reproduced")
    candidate_inputs = _object(candidate.get("inputs"), "candidate.inputs")
    if (
        candidate_inputs.get("fit_artifact_ids")
        != predictor["overall"]["fit_artifact_ids"]
        or candidate_inputs.get("forbidden_fit_source") != "mlperf"
        or candidate_inputs.get("mlperf_artifacts_opened_during_fit") != []
    ):
        raise CalibrationVerificationError("candidate source-role separation mismatch")

    external_gate = _reproduce_external_gate(protocol, source_paths, pairs)
    if candidate.get("end_to_end_external_validation") != external_gate:
        raise CalibrationVerificationError("external validation gate was not reproduced")
    claim_policy = _object(protocol.get("claim_policy"), "protocol.claim_policy")
    if (
        claim_policy.get("frontier_timing_usable_for_performance") is not False
        or claim_policy.get("paper_result_eligible") is not False
        or claim_policy.get("llama31_w8a16_inheritance_allowed") is not False
    ):
        raise CalibrationVerificationError("claim policy was broadened")
    expected_state = claim_policy.get(
        "operator_calibration_pass_state" if operator_valid else "operator_calibration_fail_state"
    )
    expected_eligibility = {
        "anchor_integrity_valid": True,
        "workload_shape_integrity_valid": True,
        "mlperf_excluded_from_fit": True,
        "operator_predictor_calibration_valid": operator_valid,
        "end_to_end_external_validation_valid": False,
        "frontier_timing_usable_for_performance": False,
        "paper_result_eligible": False,
        "llama31_w8a16_inheritance_allowed": False,
        "timing_claim_state": expected_state,
        "blockers": external_gate["blockers"]
        if operator_valid
        else ["one or more operator/collective predictors failed the declared thresholds"],
    }
    if candidate.get("eligibility") != expected_eligibility:
        raise CalibrationVerificationError("candidate eligibility was not reproduced")
    expected_result = "pass" if operator_valid else "fail"
    if candidate.get("result") != expected_result:
        raise CalibrationVerificationError("candidate result mismatch")

    checks = {key: True for key in required_checks}
    expected_check_set = {
        "protocol_integrity",
        "anchor_chain_integrity",
        "workload_evidence_integrity",
        "artifact_census_integrity",
        "source_role_separation",
        "predictor_tables_reproduced",
        "operator_holdout_reproduced",
        "metric_thresholds_reproduced",
        "mlperf_evidence_reproduced",
        "out_of_domain_gate_reproduced",
        "eligibility_reproduced",
    }
    if set(checks) != expected_check_set:
        raise CalibrationVerificationError("verification check contract mismatch")
    receipt = {
        "schema": verification_schema,
        "result": "pass",
        "calibration_id": calibration_id,
        "inputs": {
            "protocol": {
                "path": str(protocol_path),
                "bytes": len(protocol_bytes),
                "sha256": _sha_bytes(protocol_bytes),
            },
            "anchor_bundle_manifest": {
                "path": str(anchor_bundle_dir / "bundle-manifest.json"),
                "sha256": _sha_file(anchor_bundle_dir / "bundle-manifest.json"),
            },
            "anchor_verification": {
                "path": str(anchor_verification_path),
                "sha256": _sha_file(anchor_verification_path),
            },
            "candidate_manifest": {
                "path": str(candidate_manifest_path),
                "bytes": len(candidate_bytes),
                "sha256": _sha_bytes(candidate_bytes),
            },
        },
        "checks": checks,
        "reproduced": {
            "models": predictor["overall"]["models"],
            "passing_models": predictor["overall"]["passing_models"],
            "evaluation_keys": predictor["overall"]["evaluation_keys"],
            "operator_wape_pct": predictor["overall"]["wape_pct"],
            "operator_signed_bias_pct": predictor["overall"]["signed_bias_pct"],
            "predictor_tables_sha256": _sha_bytes(predictor_bytes),
            "operator_evaluation_sha256": _sha_bytes(evaluation_bytes),
            "external_validation_status": external_gate["status"],
            "external_performance_prediction_emitted": False,
        },
        "eligibility": expected_eligibility,
        "anchor_timing_claim_state": anchor_bundle["eligibility"]["timing_claim_state"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--anchor-bundle-dir", type=Path, required=True)
    parser.add_argument("--anchor-verification", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        receipt = verify_candidate(
            protocol_path=arguments.protocol,
            anchor_bundle_dir=arguments.anchor_bundle_dir,
            anchor_verification_path=arguments.anchor_verification,
            candidate_dir=arguments.candidate_dir,
            output_path=arguments.output,
        )
    except (CalibrationVerificationError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"verified {receipt['calibration_id']}: "
        f"models={receipt['reproduced']['models']} "
        f"evaluation_keys={receipt['reproduced']['evaluation_keys']} "
        f"external_validation={receipt['reproduced']['external_validation_status']}"
    )
    print(f"verification={arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
