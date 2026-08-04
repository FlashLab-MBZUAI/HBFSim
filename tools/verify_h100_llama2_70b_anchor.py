#!/usr/bin/env python3
"""Independently verify a public H100 + Llama 2 70B anchor bundle.

This verifier deliberately does not import the anchor importer.  It rebuilds
the artifact census, CSV/profile summaries, and MLPerf records from the copied
bytes, then emits a separate fail-closed receipt.  A passing receipt validates
the public anchor inputs only; it does not validate a timing predictor.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, TextIO


ANCHOR_SCHEMA = {
    "name": "hbfsim.validation.public-timing-anchor",
    "version": 1,
}
BUNDLE_SCHEMA = {
    "name": "hbfsim.validation.public-timing-anchor-bundle",
    "version": 1,
}
VERIFICATION_SCHEMA = {
    "name": "hbfsim.validation.public-timing-anchor-verification",
    "version": 1,
}
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "validation/calibration/h100-llama2-70b-public-anchor.json"
)
STAT_NAMES = ("min", "max", "mean", "median", "std")
ROUNDING_EPSILON = Decimal("0.000000000001")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_ID_RE = re.compile(r"^[0-9a-f]{40}$")
INTEGER_RE = re.compile(r"^(0|[1-9][0-9]*)$")


class AnchorVerificationError(ValueError):
    """The bundle does not independently reproduce its pinned contract."""


class _RepeatedKey(ValueError):
    pass


def _object_without_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _RepeatedKey(key)
        result[key] = value
    return result


def _forbid_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        decoded = json.loads(
            raw.decode("utf-8"),
            parse_constant=_forbid_json_constant,
            object_pairs_hook=_object_without_repeated_keys,
        )
    except OSError as error:
        raise AnchorVerificationError(f"cannot read {label} {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise AnchorVerificationError(f"{label} is not UTF-8: {path}") from error
    except _RepeatedKey as error:
        raise AnchorVerificationError(
            f"{label} contains duplicate key {error.args[0]!r}"
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise AnchorVerificationError(f"invalid JSON in {label} {path}: {error}") from error
    if not isinstance(decoded, dict):
        raise AnchorVerificationError(f"{label} must be a JSON object")
    return decoded, raw


def _expect_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnchorVerificationError(f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        raise AnchorVerificationError(
            f"{path} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _nonempty_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnchorVerificationError(f"{path} must be a non-empty string")
    return value.strip()


def _false(value: Any, path: str) -> None:
    if value is not False:
        raise AnchorVerificationError(f"{path} must be false")


def _natural(value: Any, path: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnchorVerificationError(f"{path} must be an integer >= {minimum}")
    return value


def _text_list(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise AnchorVerificationError(f"{path} must be an array")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise AnchorVerificationError(f"{path} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise AnchorVerificationError(f"{path} contains duplicates")
    return list(value)


def _columns(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise AnchorVerificationError(f"{path} must be an array")
    if not all(isinstance(item, str) for item in value):
        raise AnchorVerificationError(f"{path} must contain strings")
    if len(value) != len(set(value)):
        raise AnchorVerificationError(f"{path} contains duplicate columns")
    return list(value)


def _relative_path(value: Any, path: str) -> Path:
    text = _nonempty_text(value, path)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise AnchorVerificationError(f"{path} must be a safe relative path")
    return relative


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_rule_shape(rules_value: Any, columns: list[str], path: str) -> None:
    rules = _expect_keys(
        rules_value,
        {
            "timing_prefixes",
            "nullable_timing_prefixes",
            "conditional_timing",
            "integer_columns",
            "constant_columns",
            "allowed_values",
            "integer_ranges",
            "exact_value_counts",
            "profile_key",
            "expected_unique_keys",
            "expected_duplicate_rows",
        },
        path,
    )
    prefixes = _text_list(rules["timing_prefixes"], f"{path}.timing_prefixes", nonempty=True)
    nullable = _text_list(rules["nullable_timing_prefixes"], f"{path}.nullable_timing_prefixes")
    if not set(nullable).issubset(prefixes):
        raise AnchorVerificationError(f"{path}.nullable_timing_prefixes is not a subset")
    for prefix in prefixes:
        for stat_name in STAT_NAMES:
            if f"{prefix}.{stat_name}" not in columns:
                raise AnchorVerificationError(f"{path} references a missing timing column")
    referenced = set(_columns(rules["integer_columns"], f"{path}.integer_columns"))
    referenced.update(_columns(rules["profile_key"], f"{path}.profile_key", nonempty=True))
    for name in ("constant_columns", "allowed_values", "integer_ranges", "exact_value_counts"):
        mapping = rules[name]
        if not isinstance(mapping, dict):
            raise AnchorVerificationError(f"{path}.{name} must be an object")
        referenced.update(mapping)
    if not referenced.issubset(columns):
        raise AnchorVerificationError(
            f"{path} references unknown columns: {sorted(referenced - set(columns))}"
        )
    _natural(rules["expected_unique_keys"], f"{path}.expected_unique_keys", positive=True)
    _natural(rules["expected_duplicate_rows"], f"{path}.expected_duplicate_rows")
    if not isinstance(rules["conditional_timing"], list):
        raise AnchorVerificationError(f"{path}.conditional_timing must be an array")
    for index, condition_value in enumerate(rules["conditional_timing"]):
        condition_path = f"{path}.conditional_timing[{index}]"
        condition = _expect_keys(condition_value, {"when", "present", "missing"}, condition_path)
        when = _expect_keys(condition["when"], {"column", "equals"}, f"{condition_path}.when")
        if when["column"] not in columns or not isinstance(when["equals"], str):
            raise AnchorVerificationError(f"{condition_path}.when is invalid")
        for name in ("present", "missing"):
            selected = _text_list(condition[name], f"{condition_path}.{name}", nonempty=True)
            if not set(selected).issubset(prefixes):
                raise AnchorVerificationError(f"{condition_path}.{name} references an unknown prefix")


def _manifest_artifacts(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    _expect_keys(
        manifest,
        {"schema", "anchor_id", "intended_use", "target", "sources", "claim_boundary"},
        "anchor manifest",
    )
    if manifest["schema"] != ANCHOR_SCHEMA:
        raise AnchorVerificationError("unsupported anchor manifest schema")
    _nonempty_text(manifest["anchor_id"], "anchor_id")
    intended = _expect_keys(
        manifest["intended_use"],
        {
            "vidur_role",
            "mlperf_role",
            "mlperf_fit_allowed",
            "bundle_alone_establishes_calibration",
            "bundle_alone_establishes_external_validation",
            "paper_result_eligible_by_itself",
        },
        "intended_use",
    )
    if intended["vidur_role"] != "calibration_fit_input":
        raise AnchorVerificationError("Vidur fit role drifted")
    if intended["mlperf_role"] != "held_out_external_validation_only":
        raise AnchorVerificationError("MLPerf held-out role drifted")
    for name in (
        "mlperf_fit_allowed",
        "bundle_alone_establishes_calibration",
        "bundle_alone_establishes_external_validation",
        "paper_result_eligible_by_itself",
    ):
        _false(intended[name], f"intended_use.{name}")
    if not isinstance(manifest["target"], dict):
        raise AnchorVerificationError("target must be an object")
    sources = manifest["sources"]
    if not isinstance(sources, dict) or set(sources) != {"vidur", "mlperf"}:
        raise AnchorVerificationError("sources must contain exactly vidur and mlperf")
    result: list[tuple[str, dict[str, Any]]] = []
    ids: set[str] = set()
    for source_id in ("vidur", "mlperf"):
        source = sources[source_id]
        expected_source_keys = (
            {
                "role",
                "fit_allowed",
                "repository",
                "commit",
                "source_tree",
                "license",
                "license_status",
                "measurement_semantics",
                "artifacts",
            }
            if source_id == "vidur"
            else {
                "role",
                "fit_allowed",
                "repository",
                "commit",
                "source_tree",
                "license",
                "license_status",
                "public_result_id",
                "artifacts",
            }
        )
        source = _expect_keys(source, expected_source_keys, f"sources.{source_id}")
        if not GIT_ID_RE.fullmatch(str(source["commit"])) or not GIT_ID_RE.fullmatch(
            str(source["source_tree"])
        ):
            raise AnchorVerificationError(f"sources.{source_id} has an invalid git pin")
        expected_role = (
            "calibration_fit_input"
            if source_id == "vidur"
            else "held_out_external_validation_only"
        )
        if source["role"] != expected_role or source["fit_allowed"] is not (source_id == "vidur"):
            raise AnchorVerificationError(f"sources.{source_id} role/fit separation drifted")
        if not isinstance(source["artifacts"], list) or not source["artifacts"]:
            raise AnchorVerificationError(f"sources.{source_id}.artifacts must be non-empty")
        paths: set[str] = set()
        for index, artifact_value in enumerate(source["artifacts"]):
            artifact_path = f"sources.{source_id}.artifacts[{index}]"
            if not isinstance(artifact_value, dict):
                raise AnchorVerificationError(f"{artifact_path} must be an object")
            kind = artifact_value.get("kind")
            base = {"id", "kind", "path", "sha256", "bytes", "validation"}
            expected = base | ({"rows", "columns"} if kind == "csv_profile" else set())
            artifact = _expect_keys(artifact_value, expected, artifact_path)
            artifact_id = _nonempty_text(artifact["id"], f"{artifact_path}.id")
            if artifact_id in ids:
                raise AnchorVerificationError(f"duplicate artifact id {artifact_id!r}")
            ids.add(artifact_id)
            source_path = _relative_path(artifact["path"], f"{artifact_path}.path").as_posix()
            if source_path in paths:
                raise AnchorVerificationError(f"duplicate source path {source_path!r}")
            paths.add(source_path)
            if not SHA256_RE.fullmatch(str(artifact["sha256"])):
                raise AnchorVerificationError(f"{artifact_path}.sha256 is invalid")
            _natural(artifact["bytes"], f"{artifact_path}.bytes", positive=True)
            if kind == "csv_profile":
                _natural(artifact["rows"], f"{artifact_path}.rows", positive=True)
                csv_columns = _columns(artifact["columns"], f"{artifact_path}.columns", nonempty=True)
                _verify_rule_shape(artifact["validation"], csv_columns, f"{artifact_path}.validation")
            elif kind == "source_file":
                if artifact["validation"] != {}:
                    raise AnchorVerificationError(f"{artifact_path}.validation must be empty")
            elif kind == "json_record":
                validation = _expect_keys(artifact["validation"], {"expected_fields"}, f"{artifact_path}.validation")
                if not isinstance(validation["expected_fields"], dict) or not validation["expected_fields"]:
                    raise AnchorVerificationError(f"{artifact_path}.expected_fields must be non-empty")
            elif kind == "mlperf_performance_summary":
                _expect_keys(
                    artifact["validation"],
                    {"scenario", "mode", "tokens_per_second", "samples_per_second", "result"},
                    f"{artifact_path}.validation",
                )
            elif kind == "mlperf_accuracy_summary":
                _expect_keys(artifact["validation"], {"expected_metrics"}, f"{artifact_path}.validation")
            elif kind == "mlperf_user_conf":
                _expect_keys(artifact["validation"], {"expected_entries"}, f"{artifact_path}.validation")
            else:
                raise AnchorVerificationError(f"unsupported artifact kind {kind!r}")
            result.append((source_id, artifact))
    boundary = _expect_keys(
        manifest["claim_boundary"],
        {
            "timing_claim_state_after_import",
            "frontier_timing_usable_for_performance",
            "paper_result_eligible",
            "importer_can_establish",
            "importer_cannot_establish",
            "required_next_evidence",
        },
        "claim_boundary",
    )
    if boundary["timing_claim_state_after_import"] != "anchor_ready_not_calibrated":
        raise AnchorVerificationError("claim boundary attempts to promote imported timing")
    _false(boundary["frontier_timing_usable_for_performance"], "claim_boundary.frontier_timing_usable_for_performance")
    _false(boundary["paper_result_eligible"], "claim_boundary.paper_result_eligible")
    for name in ("importer_can_establish", "importer_cannot_establish", "required_next_evidence"):
        _text_list(boundary[name], f"claim_boundary.{name}", nonempty=True)
    return result


def _bundle_file(root: Path, relative_text: str, label: str) -> Path:
    relative = _relative_path(relative_text, f"{label}.bundle_path")
    unresolved = root / relative
    try:
        mode = unresolved.lstat().st_mode
    except OSError as error:
        raise AnchorVerificationError(f"missing {label}: {unresolved}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise AnchorVerificationError(f"{label} is not a regular non-symlink file")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AnchorVerificationError(f"{label} escapes bundle directory") from error
    return resolved


def _nonnegative_decimal(text: str, path: str) -> Decimal:
    try:
        value = Decimal(text)
    except (InvalidOperation, TypeError) as error:
        raise AnchorVerificationError(f"{path} is not a decimal") from error
    if not value.is_finite() or value < 0:
        raise AnchorVerificationError(f"{path} must be finite and non-negative")
    return value


def _audit_stat_group(
    row: dict[str, str],
    prefix: str,
    row_path: str,
) -> bool:
    raw_values = [row[f"{prefix}.{name}"].strip() for name in STAT_NAMES]
    blanks = sum(value == "" for value in raw_values)
    if blanks:
        if blanks != len(STAT_NAMES):
            raise AnchorVerificationError(f"{row_path}.{prefix} is partially empty")
        return False
    parsed = {
        name: _nonnegative_decimal(raw_values[index], f"{row_path}.{prefix}.{name}")
        for index, name in enumerate(STAT_NAMES)
    }
    if parsed["min"] > parsed["max"]:
        raise AnchorVerificationError(f"{row_path}.{prefix} has min > max")
    low = parsed["min"] - ROUNDING_EPSILON
    high = parsed["max"] + ROUNDING_EPSILON
    if not low <= parsed["mean"] <= high or not low <= parsed["median"] <= high:
        raise AnchorVerificationError(f"{row_path}.{prefix} center is outside [min, max]")
    return True


def _rebuild_profile(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    rules = artifact["validation"]
    value_census = {column: Counter() for column in rules["exact_value_counts"]}
    range_values: dict[str, list[int]] = {column: [] for column in rules["integer_ranges"]}
    profile_keys: Counter[tuple[str, ...]] = Counter()
    timing_counts: Counter[str] = Counter()
    total_rows = 0
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise AnchorVerificationError(f"cannot read profile {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != artifact["columns"]:
            raise AnchorVerificationError(f"{artifact['id']} header does not match the anchor")
        for line_number, row in enumerate(reader, start=2):
            total_rows += 1
            row_path = f"{artifact['id']} row {line_number}"
            if None in row or any(value is None for value in row.values()):
                raise AnchorVerificationError(f"{row_path} has a malformed field count")
            for column in rules["integer_columns"]:
                if not INTEGER_RE.fullmatch(row[column].strip()):
                    raise AnchorVerificationError(f"{row_path}.{column} is not a non-negative integer")
            for column, expected in rules["constant_columns"].items():
                if row[column] != expected:
                    raise AnchorVerificationError(f"{row_path}.{column} constant drifted")
            for column, allowed in rules["allowed_values"].items():
                if row[column] not in allowed:
                    raise AnchorVerificationError(f"{row_path}.{column} is outside its domain")
            for column in value_census:
                value_census[column][row[column]] += 1
            for column in range_values:
                range_values[column].append(int(row[column]))
            profile_keys[tuple(row[column] for column in rules["profile_key"])] += 1
            presence: dict[str, bool] = {}
            for prefix in rules["timing_prefixes"]:
                observed = _audit_stat_group(row, prefix, row_path)
                presence[prefix] = observed
                if observed:
                    timing_counts[prefix] += 1
                elif prefix not in rules["nullable_timing_prefixes"]:
                    raise AnchorVerificationError(f"{row_path}.{prefix} may not be empty")
            for condition in rules["conditional_timing"]:
                when = condition["when"]
                if row[when["column"]] != when["equals"]:
                    continue
                if any(not presence[prefix] for prefix in condition["present"]):
                    raise AnchorVerificationError(f"{row_path} is missing phase timing")
                if any(presence[prefix] for prefix in condition["missing"]):
                    raise AnchorVerificationError(f"{row_path} carries cross-phase timing")
    if total_rows != artifact["rows"]:
        raise AnchorVerificationError(f"{artifact['id']} row census drifted")
    for column, expected in rules["exact_value_counts"].items():
        if dict(sorted(value_census[column].items())) != expected:
            raise AnchorVerificationError(f"{artifact['id']} value census drifted for {column}")
    observed_ranges: dict[str, dict[str, int]] = {}
    for column, expected in rules["integer_ranges"].items():
        observed = {"min": min(range_values[column]), "max": max(range_values[column])}
        if observed != expected:
            raise AnchorVerificationError(f"{artifact['id']} range drifted for {column}")
        observed_ranges[column] = observed
    unique_keys = len(profile_keys)
    duplicate_rows = sum(count - 1 for count in profile_keys.values())
    if unique_keys != rules["expected_unique_keys"] or duplicate_rows != rules["expected_duplicate_rows"]:
        raise AnchorVerificationError(f"{artifact['id']} profile-key census drifted")
    return {
        "artifact_id": artifact["id"],
        "rows": total_rows,
        "columns": list(artifact["columns"]),
        "timing_observations": dict(sorted(timing_counts.items())),
        "exact_value_counts": {
            column: dict(sorted(counter.items()))
            for column, counter in sorted(value_census.items())
        },
        "integer_ranges": observed_ranges,
        "unique_profile_keys": unique_keys,
        "duplicate_profile_rows": duplicate_rows,
    }


def _utf8_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise AnchorVerificationError(f"cannot decode {label}: {error}") from error


def _matched_value(lines: list[str], expression: str, label: str) -> str:
    matcher = re.compile(expression)
    matches = [match.group(1).strip() for line in lines if (match := matcher.match(line))]
    if not matches or len(set(matches)) != 1:
        raise AnchorVerificationError(f"{label} has a missing or conflicting summary field")
    return matches[0]


def _rebuild_performance(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    lines = _utf8_text(path, artifact["id"]).splitlines()
    scenario = _matched_value(lines, r"^Scenario\s*:\s*(.+)$", artifact["id"])
    mode = _matched_value(lines, r"^Mode\s*:\s*(.+)$", artifact["id"])
    result = _matched_value(lines, r"^Result is\s*:\s*(.+)$", artifact["id"])
    if scenario == "Offline":
        token_re = r"^Tokens per second:\s*(.+)$"
        sample_re = r"^Samples per second:\s*(.+)$"
    elif scenario == "Server":
        token_re = r"^Completed tokens per second\s*:\s*(.+)$"
        sample_re = r"^Completed samples per second\s*:\s*(.+)$"
    else:
        raise AnchorVerificationError(f"{artifact['id']} has an unknown scenario")
    tokens = _matched_value(lines, token_re, artifact["id"])
    samples = _matched_value(lines, sample_re, artifact["id"])
    expected = artifact["validation"]
    if (scenario, mode, result) != (expected["scenario"], expected["mode"], expected["result"]):
        raise AnchorVerificationError(f"{artifact['id']} scenario/mode/result drifted")
    if _nonnegative_decimal(tokens, f"{artifact['id']}.tokens") != Decimal(expected["tokens_per_second"]):
        raise AnchorVerificationError(f"{artifact['id']} token score drifted")
    if _nonnegative_decimal(samples, f"{artifact['id']}.samples") != Decimal(expected["samples_per_second"]):
        raise AnchorVerificationError(f"{artifact['id']} sample score drifted")
    return {
        "artifact_id": artifact["id"],
        "scenario": scenario,
        "mode": mode,
        "tokens_per_second": expected["tokens_per_second"],
        "samples_per_second": expected["samples_per_second"],
        "result": result,
    }


def _rebuild_accuracy(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    lines = _utf8_text(path, artifact["id"]).splitlines()
    mappings = [line for line in lines if line.strip().startswith("{")]
    if len(mappings) != 1:
        raise AnchorVerificationError(f"{artifact['id']} has an invalid metrics census")
    try:
        metrics = ast.literal_eval(mappings[0])
    except (SyntaxError, ValueError) as error:
        raise AnchorVerificationError(f"{artifact['id']} metrics cannot be parsed") from error
    if not isinstance(metrics, dict):
        raise AnchorVerificationError(f"{artifact['id']} metrics are not a mapping")
    expected = artifact["validation"]["expected_metrics"]
    normalized: dict[str, Any] = {}
    for key, expected_value in expected.items():
        if key not in metrics:
            raise AnchorVerificationError(f"{artifact['id']} lacks {key}")
        actual = metrics[key]
        if isinstance(expected_value, int):
            if isinstance(actual, bool) or actual != expected_value:
                raise AnchorVerificationError(f"{artifact['id']} metric {key} drifted")
            normalized[key] = actual
        else:
            if Decimal(str(actual)) != Decimal(expected_value):
                raise AnchorVerificationError(f"{artifact['id']} metric {key} drifted")
            normalized[key] = expected_value
    hashes = [line[5:] for line in lines if line.startswith("hash=")]
    if len(hashes) != 1 or not SHA256_RE.fullmatch(hashes[0]):
        raise AnchorVerificationError(f"{artifact['id']} accuracy hash is invalid")
    return {
        "artifact_id": artifact["id"],
        "metrics": normalized,
        "accuracy_hash": hashes[0],
    }


def _rebuild_json_record(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    record, _ = _read_json(path, artifact["id"])
    expected = artifact["validation"]["expected_fields"]
    for key, expected_value in expected.items():
        if record.get(key) != expected_value:
            raise AnchorVerificationError(f"{artifact['id']} field {key} drifted")
    return {"artifact_id": artifact["id"], "expected_fields": expected}


def _rebuild_user_conf(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(_utf8_text(path, artifact["id"]).splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise AnchorVerificationError(f"{artifact['id']} line {line_number} is malformed")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value or key in entries:
            raise AnchorVerificationError(f"{artifact['id']} line {line_number} is duplicated")
        entries[key] = value
    if entries != artifact["validation"]["expected_entries"]:
        raise AnchorVerificationError(f"{artifact['id']} configuration drifted")
    return {"artifact_id": artifact["id"], "entries": entries}


def _rebuild_semantic_record(path: Path, artifact: dict[str, Any]) -> dict[str, Any] | None:
    kind = artifact["kind"]
    if kind == "csv_profile":
        return _rebuild_profile(path, artifact)
    if kind == "json_record":
        return _rebuild_json_record(path, artifact)
    if kind == "mlperf_performance_summary":
        return _rebuild_performance(path, artifact)
    if kind == "mlperf_accuracy_summary":
        return _rebuild_accuracy(path, artifact)
    if kind == "mlperf_user_conf":
        return _rebuild_user_conf(path, artifact)
    return None


def _canonical_repository(value: str) -> str:
    text = value.strip().rstrip("/").lower()
    return text[:-4] if text.endswith(".git") else text


def _expected_eligibility(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_integrity_valid": True,
        "calibration_input_integrity_valid": True,
        "held_out_reference_integrity_valid": True,
        "predictor_calibration_valid": False,
        "end_to_end_external_validation_valid": False,
        "frontier_timing_usable_for_performance": False,
        "paper_result_eligible": False,
        "timing_claim_state": "anchor_ready_not_calibrated",
        "blockers": list(manifest["claim_boundary"]["required_next_evidence"]),
    }


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle: TextIO = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, path)
    except Exception:
        if not handle.closed:
            handle.close()
        temporary.unlink(missing_ok=True)
        raise


def verify_anchor(
    *,
    manifest_path: Path,
    bundle_dir: Path,
    output: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    bundle_dir = bundle_dir.resolve()
    output = output.resolve()
    if not manifest_path.is_file():
        raise AnchorVerificationError(f"anchor manifest does not exist: {manifest_path}")
    if not bundle_dir.is_dir() or bundle_dir.is_symlink():
        raise AnchorVerificationError(f"bundle directory does not exist or is a symlink: {bundle_dir}")
    if output.exists() or output.is_symlink():
        raise AnchorVerificationError(f"verification output already exists: {output}")
    try:
        output.relative_to(bundle_dir)
    except ValueError:
        pass
    else:
        raise AnchorVerificationError("verification output must be outside the bundle")

    manifest, manifest_raw = _read_json(manifest_path, "anchor manifest")
    declared_artifacts = _manifest_artifacts(manifest)
    copied_manifest_path = _bundle_file(bundle_dir, "anchor-manifest.json", "copied anchor manifest")
    copied_manifest_raw = copied_manifest_path.read_bytes()
    if copied_manifest_raw != manifest_raw:
        raise AnchorVerificationError("copied anchor manifest is not byte-identical to the trusted manifest")
    bundle_manifest_path = _bundle_file(bundle_dir, "bundle-manifest.json", "bundle manifest")
    bundle, bundle_raw = _read_json(bundle_manifest_path, "bundle manifest")
    _expect_keys(
        bundle,
        {"schema", "result", "anchor", "sources", "profiles", "held_out_records", "artifacts", "eligibility"},
        "bundle manifest",
    )
    if bundle["schema"] != BUNDLE_SCHEMA or bundle["result"] != "pass":
        raise AnchorVerificationError("bundle schema/result is not passing")
    anchor = _expect_keys(
        bundle["anchor"],
        {"anchor_id", "manifest", "intended_use", "target", "claim_boundary"},
        "bundle.anchor",
    )
    if anchor["anchor_id"] != manifest["anchor_id"]:
        raise AnchorVerificationError("bundle anchor_id drifted")
    if anchor["intended_use"] != manifest["intended_use"] or anchor["target"] != manifest["target"]:
        raise AnchorVerificationError("bundle target/intended use drifted")
    if anchor["claim_boundary"] != manifest["claim_boundary"]:
        raise AnchorVerificationError("bundle claim boundary drifted")
    expected_manifest_identity = {
        "bundle_path": "anchor-manifest.json",
        "bytes": len(manifest_raw),
        "sha256": _sha256(manifest_raw),
    }
    if anchor["manifest"] != expected_manifest_identity:
        raise AnchorVerificationError("bundle anchor-manifest identity drifted")

    expected_sources: dict[str, Any] = {}
    for source_id in ("vidur", "mlperf"):
        declared = manifest["sources"][source_id]
        actual = _expect_keys(
            bundle["sources"].get(source_id) if isinstance(bundle["sources"], dict) else None,
            {
                "repository",
                "origin",
                "commit",
                "source_tree",
                "tracked_tree_clean",
                "role",
                "fit_allowed",
                "license",
                "license_status",
            },
            f"bundle.sources.{source_id}",
        )
        expected = {
            "repository": declared["repository"],
            "commit": declared["commit"],
            "source_tree": declared["source_tree"],
            "tracked_tree_clean": True,
            "role": declared["role"],
            "fit_allowed": declared["fit_allowed"],
            "license": declared["license"],
            "license_status": declared["license_status"],
        }
        for key, value in expected.items():
            if actual.get(key) != value:
                raise AnchorVerificationError(f"bundle.sources.{source_id}.{key} drifted")
        if _canonical_repository(actual["origin"]) != _canonical_repository(declared["repository"]):
            raise AnchorVerificationError(f"bundle.sources.{source_id}.origin drifted")
        expected_sources[source_id] = actual
    if set(bundle["sources"]) != {"vidur", "mlperf"}:
        raise AnchorVerificationError("bundle source census drifted")

    if not isinstance(bundle["artifacts"], list):
        raise AnchorVerificationError("bundle.artifacts must be an array")
    expected_artifact_records: list[dict[str, Any]] = [
        {
            "artifact_id": "anchor.manifest",
            "source_id": "anchor",
            "source_path": manifest_path.name,
            "bundle_path": "anchor-manifest.json",
            "kind": "anchor_manifest",
            "bytes": len(manifest_raw),
            "sha256": _sha256(manifest_raw),
        }
    ]
    profile_records: list[dict[str, Any]] = []
    held_out_records: list[dict[str, Any]] = []
    expected_file_paths = {"anchor-manifest.json", "bundle-manifest.json"}
    for source_id, artifact in declared_artifacts:
        relative = (Path("sources") / source_id / Path(artifact["path"])).as_posix()
        copied_path = _bundle_file(bundle_dir, relative, artifact["id"])
        actual_bytes = copied_path.stat().st_size
        actual_digest = _file_sha256(copied_path)
        if actual_bytes != artifact["bytes"] or actual_digest != artifact["sha256"]:
            raise AnchorVerificationError(f"{artifact['id']} copied bytes do not match the anchor")
        expected_file_paths.add(relative)
        expected_artifact_records.append(
            {
                "artifact_id": artifact["id"],
                "source_id": source_id,
                "source_path": artifact["path"],
                "bundle_path": relative,
                "kind": artifact["kind"],
                "bytes": actual_bytes,
                "sha256": actual_digest,
            }
        )
        semantic = _rebuild_semantic_record(copied_path, artifact)
        if semantic is not None:
            if artifact["kind"] == "csv_profile":
                profile_records.append(semantic)
            elif source_id == "mlperf":
                held_out_records.append(semantic)
    if bundle["artifacts"] != expected_artifact_records:
        raise AnchorVerificationError("bundle artifact ledger drifted")

    actual_file_paths: set[str] = set()
    for path in bundle_dir.rglob("*"):
        relative = path.relative_to(bundle_dir).as_posix()
        if path.is_symlink():
            raise AnchorVerificationError(f"bundle contains symlink {relative}")
        if path.is_file():
            actual_file_paths.add(relative)
        elif not path.is_dir():
            raise AnchorVerificationError(f"bundle contains non-file artifact {relative}")
    if actual_file_paths != expected_file_paths:
        raise AnchorVerificationError(
            f"bundle file census mismatch: missing={sorted(expected_file_paths - actual_file_paths)}, "
            f"extra={sorted(actual_file_paths - expected_file_paths)}"
        )

    profile_records.sort(key=lambda item: item["artifact_id"])
    held_out_records.sort(key=lambda item: item["artifact_id"])
    if bundle["profiles"] != profile_records:
        raise AnchorVerificationError("bundle profile summaries do not independently reproduce")
    if bundle["held_out_records"] != held_out_records:
        raise AnchorVerificationError("bundle held-out summaries do not independently reproduce")
    eligibility = _expected_eligibility(manifest)
    if bundle["eligibility"] != eligibility:
        raise AnchorVerificationError("bundle eligibility attempts to change the claim boundary")

    receipt = {
        "schema": VERIFICATION_SCHEMA,
        "result": "pass",
        "anchor_id": manifest["anchor_id"],
        "inputs": {
            "anchor_manifest": {
                "path": str(manifest_path),
                "bytes": len(manifest_raw),
                "sha256": _sha256(manifest_raw),
            },
            "bundle_manifest": {
                "path": str(bundle_manifest_path),
                "bytes": len(bundle_raw),
                "sha256": _sha256(bundle_raw),
            },
        },
        "checks": {
            "copied_manifest_byte_identical": True,
            "source_pin_records_valid": True,
            "artifact_census_exact": True,
            "artifact_digests_valid": True,
            "profile_contracts_independently_rebuilt": True,
            "held_out_records_independently_rebuilt": True,
            "mlperf_excluded_from_fit": True,
            "claim_boundary_fail_closed": True,
        },
        "sources": expected_sources,
        "profiles": profile_records,
        "held_out_records": held_out_records,
        "eligibility": eligibility,
    }
    _write_atomic(output, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify_anchor(
            manifest_path=args.manifest,
            bundle_dir=args.bundle_dir,
            output=args.output,
        )
    except (AnchorVerificationError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"verified {result['anchor_id']}: profiles={len(result['profiles'])} "
        f"held_out_records={len(result['held_out_records'])} "
        f"timing={result['eligibility']['timing_claim_state']}"
    )
    print(f"verification={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
