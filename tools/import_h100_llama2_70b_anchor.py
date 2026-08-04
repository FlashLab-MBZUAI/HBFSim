#!/usr/bin/env python3
"""Import the pinned public H100 + Llama 2 70B timing anchor.

The importer never downloads data and never treats a successful import as a
calibrated simulator.  It verifies two clean, pinned source checkouts, checks
every declared byte and table contract, and publishes a self-contained bundle
transactionally.  MLPerf is retained as held-out evidence and is never a fit
input.
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
import shutil
import stat
import subprocess
import tempfile
from typing import Any


ANCHOR_SCHEMA = {
    "name": "hbfsim.validation.public-timing-anchor",
    "version": 1,
}
BUNDLE_SCHEMA = {
    "name": "hbfsim.validation.public-timing-anchor-bundle",
    "version": 1,
}
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "validation/calibration/h100-llama2-70b-public-anchor.json"
)
TIMING_SUFFIXES = ("min", "max", "mean", "median", "std")
TIMING_ROUNDING_TOLERANCE = Decimal("1e-12")
HEX_RE = re.compile(r"^[0-9a-f]+$")
NONNEGATIVE_INTEGER_RE = re.compile(r"^(0|[1-9][0-9]*)$")


class AnchorImportError(ValueError):
    """The source checkouts or anchor contract are not exact."""


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
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except OSError as error:
        raise AnchorImportError(f"cannot read {description} {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise AnchorImportError(f"{description} is not UTF-8: {path}") from error
    except _DuplicateJsonKey as error:
        raise AnchorImportError(
            f"{description} contains duplicate key {error.args[0]!r}"
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise AnchorImportError(f"invalid JSON in {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnchorImportError(f"{description} must be a JSON object: {path}")
    return value, payload


def _exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise AnchorImportError(
            f"{path} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnchorImportError(f"{path} must be an object")
    return value


def _list(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        suffix = " and non-empty" if nonempty else ""
        raise AnchorImportError(f"{path} must be an array{suffix}")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnchorImportError(f"{path} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise AnchorImportError(f"{path} must be boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AnchorImportError(f"{path} must be an integer >= {minimum}")
    return value


def _hex(value: Any, path: str, length: int) -> str:
    text = _string(value, path)
    if len(text) != length or text != text.lower() or not HEX_RE.fullmatch(text):
        raise AnchorImportError(
            f"{path} must be {length} lowercase hexadecimal digits"
        )
    return text


def _safe_relative(value: Any, path: str) -> Path:
    text = _string(value, path)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise AnchorImportError(f"{path} must be a safe relative path")
    return relative


def _strings(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    values = _list(value, path, nonempty=nonempty)
    result = [_string(item, f"{path}[{index}]") for index, item in enumerate(values)]
    if len(result) != len(set(result)):
        raise AnchorImportError(f"{path} contains duplicates")
    return result


def _column_names(value: Any, path: str, *, nonempty: bool = False) -> list[str]:
    values = _list(value, path, nonempty=nonempty)
    if not all(isinstance(item, str) for item in values):
        raise AnchorImportError(f"{path} must contain column-name strings")
    if len(values) != len(set(values)):
        raise AnchorImportError(f"{path} contains duplicates")
    return list(values)


def _validate_target(target_value: Any) -> None:
    target = _mapping(target_value, "target")
    _exact_keys(target, {"model", "vidur_profile", "mlperf_held_out"}, "target")
    model = _mapping(target["model"], "target.model")
    _exact_keys(
        model,
        {
            "id",
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "tokenizer_vocab_size",
            "profiled_padded_vocab_size",
            "max_model_len",
        },
        "target.model",
    )
    _string(model["id"], "target.model.id")
    for key in set(model) - {"id"}:
        _integer(model[key], f"target.model.{key}", minimum=1)

    profile = _mapping(target["vidur_profile"], "target.vidur_profile")
    _exact_keys(
        profile,
        {"device_label", "network_label", "precision", "tensor_parallel_workers"},
        "target.vidur_profile",
    )
    for key in ("device_label", "network_label", "precision"):
        _string(profile[key], f"target.vidur_profile.{key}")
    workers = _list(
        profile["tensor_parallel_workers"],
        "target.vidur_profile.tensor_parallel_workers",
        nonempty=True,
    )
    normalized_workers = [
        _integer(value, f"target.vidur_profile.tensor_parallel_workers[{index}]", minimum=1)
        for index, value in enumerate(workers)
    ]
    if normalized_workers != sorted(set(normalized_workers)):
        raise AnchorImportError(
            "target.vidur_profile.tensor_parallel_workers must be sorted and unique"
        )

    held_out = _mapping(target["mlperf_held_out"], "target.mlperf_held_out")
    _exact_keys(
        held_out,
        {
            "system",
            "accelerators",
            "accelerator_memory",
            "weight_precision",
            "framework",
            "runtime_and_precision_match_vidur_fit",
        },
        "target.mlperf_held_out",
    )
    for key in ("system", "accelerator_memory", "weight_precision", "framework"):
        _string(held_out[key], f"target.mlperf_held_out.{key}")
    _integer(held_out["accelerators"], "target.mlperf_held_out.accelerators", minimum=1)
    if _boolean(
        held_out["runtime_and_precision_match_vidur_fit"],
        "target.mlperf_held_out.runtime_and_precision_match_vidur_fit",
    ):
        raise AnchorImportError(
            "the held-out MLPerf runtime/precision mismatch must remain explicit"
        )


def _validate_csv_rules(
    rules_value: Any,
    columns: list[str],
    path: str,
) -> None:
    rules = _mapping(rules_value, path)
    _exact_keys(
        rules,
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
    timing_prefixes = _strings(rules["timing_prefixes"], f"{path}.timing_prefixes", nonempty=True)
    nullable = _strings(rules["nullable_timing_prefixes"], f"{path}.nullable_timing_prefixes")
    if not set(nullable).issubset(timing_prefixes):
        raise AnchorImportError(f"{path}.nullable_timing_prefixes is not a subset")
    for prefix in timing_prefixes:
        required = {f"{prefix}.{suffix}" for suffix in TIMING_SUFFIXES}
        if not required.issubset(columns):
            raise AnchorImportError(
                f"{path}.timing_prefixes references missing columns for {prefix}"
            )

    referenced_columns: set[str] = set()
    for key in ("integer_columns", "profile_key"):
        referenced_columns.update(
            _column_names(
                rules[key],
                f"{path}.{key}",
                nonempty=(key == "profile_key"),
            )
        )
    for key in ("constant_columns", "allowed_values", "integer_ranges", "exact_value_counts"):
        mapping = _mapping(rules[key], f"{path}.{key}")
        referenced_columns.update(mapping)
        for column, value in mapping.items():
            if not isinstance(column, str):
                raise AnchorImportError(f"{path}.{key} keys must be strings")
            if key == "constant_columns":
                if not isinstance(value, str):
                    raise AnchorImportError(f"{path}.{key}.{column} must be a string")
            elif key == "allowed_values":
                _strings(value, f"{path}.{key}.{column}", nonempty=True)
            elif key == "integer_ranges":
                bounds = _mapping(value, f"{path}.{key}.{column}")
                _exact_keys(bounds, {"min", "max"}, f"{path}.{key}.{column}")
                lower = _integer(bounds["min"], f"{path}.{key}.{column}.min")
                upper = _integer(bounds["max"], f"{path}.{key}.{column}.max")
                if lower > upper:
                    raise AnchorImportError(f"{path}.{key}.{column} has min > max")
            else:
                counts = _mapping(value, f"{path}.{key}.{column}")
                if not counts:
                    raise AnchorImportError(f"{path}.{key}.{column} must be non-empty")
                for count_key, count in counts.items():
                    if not isinstance(count_key, str):
                        raise AnchorImportError(f"{path}.{key}.{column} keys must be strings")
                    _integer(count, f"{path}.{key}.{column}.{count_key}")
    if not referenced_columns.issubset(columns):
        raise AnchorImportError(
            f"{path} references unknown columns: {sorted(referenced_columns - set(columns))}"
        )
    unique_keys = _integer(
        rules["expected_unique_keys"],
        f"{path}.expected_unique_keys",
        minimum=1,
    )
    duplicate_rows = _integer(
        rules["expected_duplicate_rows"],
        f"{path}.expected_duplicate_rows",
    )
    if unique_keys + duplicate_rows <= 0:
        raise AnchorImportError(f"{path} has an empty profile-key census")

    conditionals = _list(rules["conditional_timing"], f"{path}.conditional_timing")
    for index, conditional_value in enumerate(conditionals):
        conditional_path = f"{path}.conditional_timing[{index}]"
        conditional = _mapping(conditional_value, conditional_path)
        _exact_keys(conditional, {"when", "present", "missing"}, conditional_path)
        when = _mapping(conditional["when"], f"{conditional_path}.when")
        _exact_keys(when, {"column", "equals"}, f"{conditional_path}.when")
        column = _string(when["column"], f"{conditional_path}.when.column")
        if column not in columns:
            raise AnchorImportError(f"{conditional_path}.when references unknown column")
        if not isinstance(when["equals"], str):
            raise AnchorImportError(f"{conditional_path}.when.equals must be a string")
        for key in ("present", "missing"):
            prefixes = _strings(conditional[key], f"{conditional_path}.{key}", nonempty=True)
            if not set(prefixes).issubset(timing_prefixes):
                raise AnchorImportError(f"{conditional_path}.{key} references unknown timing prefix")


def _validate_artifact(value: Any, source_id: str, index: int) -> dict[str, Any]:
    path = f"sources.{source_id}.artifacts[{index}]"
    artifact = _mapping(value, path)
    kind = _string(artifact.get("kind"), f"{path}.kind")
    base_keys = {"id", "kind", "path", "sha256", "bytes", "validation"}
    expected_keys = base_keys | ({"rows", "columns"} if kind == "csv_profile" else set())
    _exact_keys(artifact, expected_keys, path)
    _string(artifact["id"], f"{path}.id")
    _safe_relative(artifact["path"], f"{path}.path")
    _hex(artifact["sha256"], f"{path}.sha256", 64)
    _integer(artifact["bytes"], f"{path}.bytes", minimum=1)
    if kind == "csv_profile":
        _integer(artifact["rows"], f"{path}.rows", minimum=1)
        columns = _list(artifact["columns"], f"{path}.columns", nonempty=True)
        if not all(isinstance(column, str) for column in columns):
            raise AnchorImportError(f"{path}.columns must contain strings")
        if len(columns) != len(set(columns)):
            raise AnchorImportError(f"{path}.columns contains duplicates")
        _validate_csv_rules(artifact["validation"], columns, f"{path}.validation")
    elif kind == "source_file":
        if artifact["validation"] != {}:
            raise AnchorImportError(f"{path}.validation must be empty for source_file")
    elif kind == "json_record":
        validation = _mapping(artifact["validation"], f"{path}.validation")
        _exact_keys(validation, {"expected_fields"}, f"{path}.validation")
        if not _mapping(validation["expected_fields"], f"{path}.validation.expected_fields"):
            raise AnchorImportError(f"{path}.validation.expected_fields must be non-empty")
    elif kind == "mlperf_performance_summary":
        validation = _mapping(artifact["validation"], f"{path}.validation")
        _exact_keys(
            validation,
            {"scenario", "mode", "tokens_per_second", "samples_per_second", "result"},
            f"{path}.validation",
        )
        for key in validation:
            _string(validation[key], f"{path}.validation.{key}")
    elif kind == "mlperf_accuracy_summary":
        validation = _mapping(artifact["validation"], f"{path}.validation")
        _exact_keys(validation, {"expected_metrics"}, f"{path}.validation")
        if not _mapping(validation["expected_metrics"], f"{path}.validation.expected_metrics"):
            raise AnchorImportError(f"{path}.validation.expected_metrics must be non-empty")
    elif kind == "mlperf_user_conf":
        validation = _mapping(artifact["validation"], f"{path}.validation")
        _exact_keys(validation, {"expected_entries"}, f"{path}.validation")
        entries = _mapping(validation["expected_entries"], f"{path}.validation.expected_entries")
        if not entries or not all(isinstance(key, str) and isinstance(item, str) for key, item in entries.items()):
            raise AnchorImportError(f"{path}.validation.expected_entries must map strings to strings")
    else:
        raise AnchorImportError(f"{path}.kind is unsupported: {kind!r}")
    return artifact


def _validate_manifest(manifest: dict[str, Any]) -> None:
    _exact_keys(
        manifest,
        {"schema", "anchor_id", "intended_use", "target", "sources", "claim_boundary"},
        "anchor manifest",
    )
    if manifest["schema"] != ANCHOR_SCHEMA:
        raise AnchorImportError("unsupported anchor manifest schema")
    _string(manifest["anchor_id"], "anchor_id")

    intended = _mapping(manifest["intended_use"], "intended_use")
    _exact_keys(
        intended,
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
        raise AnchorImportError("Vidur must remain the calibration fit input")
    if intended["mlperf_role"] != "held_out_external_validation_only":
        raise AnchorImportError("MLPerf must remain held-out-only")
    for key in (
        "mlperf_fit_allowed",
        "bundle_alone_establishes_calibration",
        "bundle_alone_establishes_external_validation",
        "paper_result_eligible_by_itself",
    ):
        if _boolean(intended[key], f"intended_use.{key}"):
            raise AnchorImportError(f"intended_use.{key} must remain false")
    _validate_target(manifest["target"])

    sources = _mapping(manifest["sources"], "sources")
    if set(sources) != {"vidur", "mlperf"}:
        raise AnchorImportError("sources must contain exactly vidur and mlperf")
    artifact_ids: set[str] = set()
    for source_id in ("vidur", "mlperf"):
        source = _mapping(sources[source_id], f"sources.{source_id}")
        if source_id == "vidur":
            expected = {
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
        else:
            expected = {
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
        _exact_keys(source, expected, f"sources.{source_id}")
        expected_role = (
            "calibration_fit_input"
            if source_id == "vidur"
            else "held_out_external_validation_only"
        )
        if source["role"] != expected_role:
            raise AnchorImportError(f"sources.{source_id}.role must be {expected_role}")
        fit_allowed = _boolean(source["fit_allowed"], f"sources.{source_id}.fit_allowed")
        if fit_allowed != (source_id == "vidur"):
            raise AnchorImportError(f"sources.{source_id}.fit_allowed violates role separation")
        for key in ("repository", "license", "license_status"):
            _string(source[key], f"sources.{source_id}.{key}")
        _hex(source["commit"], f"sources.{source_id}.commit", 40)
        _hex(source["source_tree"], f"sources.{source_id}.source_tree", 40)
        if source_id == "vidur":
            semantics = _mapping(source["measurement_semantics"], "sources.vidur.measurement_semantics")
            if set(semantics) != {"attention", "mlp", "collectives"}:
                raise AnchorImportError("Vidur measurement semantics must name attention, mlp, and collectives")
            for key, value in semantics.items():
                _string(value, f"sources.vidur.measurement_semantics.{key}")
        else:
            _string(source["public_result_id"], "sources.mlperf.public_result_id")
        source_paths: set[str] = set()
        for index, value in enumerate(_list(source["artifacts"], f"sources.{source_id}.artifacts", nonempty=True)):
            artifact = _validate_artifact(value, source_id, index)
            artifact_id = artifact["id"]
            if artifact_id in artifact_ids:
                raise AnchorImportError(f"duplicate artifact id {artifact_id!r}")
            artifact_ids.add(artifact_id)
            if artifact["path"] in source_paths:
                raise AnchorImportError(f"duplicate source path {artifact['path']!r}")
            source_paths.add(artifact["path"])

    boundary = _mapping(manifest["claim_boundary"], "claim_boundary")
    _exact_keys(
        boundary,
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
        raise AnchorImportError("import cannot claim a calibrated timing state")
    for key in ("frontier_timing_usable_for_performance", "paper_result_eligible"):
        if _boolean(boundary[key], f"claim_boundary.{key}"):
            raise AnchorImportError(f"claim_boundary.{key} must remain false")
    for key in ("importer_can_establish", "importer_cannot_establish", "required_next_evidence"):
        _strings(boundary[key], f"claim_boundary.{key}", nonempty=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(root: Path, *args: str) -> str:
    command = ["git", "-C", str(root), *args]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as error:
        raise AnchorImportError("git is required to verify source provenance") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "git command failed"
        raise AnchorImportError(f"cannot verify git checkout {root}: {detail}") from error
    return completed.stdout.strip()


def _normalized_repository(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized.lower()


def _verify_checkout(root: Path, source: dict[str, Any], source_id: str) -> dict[str, Any]:
    if not root.is_dir():
        raise AnchorImportError(f"{source_id} checkout does not exist: {root}")
    if _run_git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise AnchorImportError(f"{source_id} source is not a git worktree: {root}")
    head = _run_git(root, "rev-parse", "HEAD")
    tree = _run_git(root, "rev-parse", "HEAD^{tree}")
    if head != source["commit"]:
        raise AnchorImportError(
            f"{source_id} HEAD mismatch: expected {source['commit']}, got {head}"
        )
    if tree != source["source_tree"]:
        raise AnchorImportError(
            f"{source_id} source tree mismatch: expected {source['source_tree']}, got {tree}"
        )
    status_text = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    if status_text:
        raise AnchorImportError(f"{source_id} tracked source tree is dirty:\n{status_text}")
    remote = _run_git(root, "remote", "get-url", "origin")
    if _normalized_repository(remote) != _normalized_repository(source["repository"]):
        raise AnchorImportError(
            f"{source_id} origin mismatch: expected {source['repository']!r}, got {remote!r}"
        )
    return {
        "repository": source["repository"],
        "origin": remote,
        "commit": head,
        "source_tree": tree,
        "tracked_tree_clean": True,
        "role": source["role"],
        "fit_allowed": source["fit_allowed"],
    }


def _artifact_path(root: Path, relative_text: str, description: str) -> Path:
    root = root.resolve()
    relative = _safe_relative(relative_text, f"{description}.path")
    unresolved = root / relative
    try:
        mode = unresolved.lstat().st_mode
    except OSError as error:
        raise AnchorImportError(f"missing {description}: {unresolved}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise AnchorImportError(f"{description} must be a regular non-symlink file: {unresolved}")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise AnchorImportError(f"{description} escapes source checkout") from error
    return path


def _decimal(value: str, path: str) -> Decimal:
    if value == "":
        raise AnchorImportError(f"{path} is unexpectedly empty")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise AnchorImportError(f"{path} is not a decimal: {value!r}") from error
    if not result.is_finite() or result < 0:
        raise AnchorImportError(f"{path} must be finite and non-negative")
    return result


def _timing_group(row: dict[str, str], prefix: str, path: str) -> tuple[bool, dict[str, Decimal]]:
    raw = {suffix: row[f"{prefix}.{suffix}"].strip() for suffix in TIMING_SUFFIXES}
    empty = [suffix for suffix, value in raw.items() if value == ""]
    if empty:
        if len(empty) != len(TIMING_SUFFIXES):
            raise AnchorImportError(f"{path}.{prefix} is only partially populated")
        return False, {}
    values = {
        suffix: _decimal(value, f"{path}.{prefix}.{suffix}")
        for suffix, value in raw.items()
    }
    if values["min"] > values["max"]:
        raise AnchorImportError(f"{path}.{prefix} has min > max")
    for suffix in ("mean", "median"):
        if not (
            values["min"] - TIMING_ROUNDING_TOLERANCE
            <= values[suffix]
            <= values["max"] + TIMING_ROUNDING_TOLERANCE
        ):
            raise AnchorImportError(f"{path}.{prefix}.{suffix} is outside [min, max]")
    return True, values


def _inspect_csv(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise AnchorImportError(f"cannot read CSV profile {path}: {error}") from error
    rules = artifact["validation"]
    timing_prefixes = list(rules["timing_prefixes"])
    nullable = set(rules["nullable_timing_prefixes"])
    counts = {column: Counter() for column in rules["exact_value_counts"]}
    integer_values: dict[str, list[int]] = {
        column: [] for column in rules["integer_ranges"]
    }
    profile_keys: Counter[tuple[str, ...]] = Counter()
    timing_observations = Counter()
    rows = 0
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != artifact["columns"]:
            raise AnchorImportError(
                f"{artifact['id']} CSV header mismatch: expected={artifact['columns']!r}, "
                f"actual={reader.fieldnames!r}"
            )
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            row_path = f"{artifact['id']} row {row_number}"
            if None in row or any(value is None for value in row.values()):
                raise AnchorImportError(f"{row_path} has a malformed column count")
            for column in rules["integer_columns"]:
                value = row[column].strip()
                if not NONNEGATIVE_INTEGER_RE.fullmatch(value):
                    raise AnchorImportError(f"{row_path}.{column} must be a non-negative integer")
            for column, expected in rules["constant_columns"].items():
                if row[column] != expected:
                    raise AnchorImportError(
                        f"{row_path}.{column} mismatch: expected {expected!r}, got {row[column]!r}"
                    )
            for column, allowed in rules["allowed_values"].items():
                if row[column] not in allowed:
                    raise AnchorImportError(
                        f"{row_path}.{column} is outside {allowed!r}: {row[column]!r}"
                    )
            for column in counts:
                counts[column][row[column]] += 1
            for column in integer_values:
                integer_values[column].append(int(row[column]))
            key = tuple(row[column] for column in rules["profile_key"])
            profile_keys[key] += 1

            present: dict[str, bool] = {}
            for prefix in timing_prefixes:
                has_values, _ = _timing_group(row, prefix, row_path)
                present[prefix] = has_values
                if has_values:
                    timing_observations[prefix] += 1
                elif prefix not in nullable:
                    raise AnchorImportError(f"{row_path}.{prefix} may not be empty")
            for conditional in rules["conditional_timing"]:
                when = conditional["when"]
                if row[when["column"]] != when["equals"]:
                    continue
                for prefix in conditional["present"]:
                    if not present[prefix]:
                        raise AnchorImportError(f"{row_path}.{prefix} must be present")
                for prefix in conditional["missing"]:
                    if present[prefix]:
                        raise AnchorImportError(f"{row_path}.{prefix} must be empty")

    if rows != artifact["rows"]:
        raise AnchorImportError(
            f"{artifact['id']} row count mismatch: expected {artifact['rows']}, got {rows}"
        )
    unique_profile_keys = len(profile_keys)
    duplicate_profile_rows = sum(count - 1 for count in profile_keys.values())
    if unique_profile_keys != rules["expected_unique_keys"]:
        raise AnchorImportError(
            f"{artifact['id']} unique profile-key count mismatch: "
            f"expected={rules['expected_unique_keys']}, actual={unique_profile_keys}"
        )
    if duplicate_profile_rows != rules["expected_duplicate_rows"]:
        raise AnchorImportError(
            f"{artifact['id']} duplicate profile-row count mismatch: "
            f"expected={rules['expected_duplicate_rows']}, actual={duplicate_profile_rows}"
        )
    for column, expected_counts in rules["exact_value_counts"].items():
        actual = dict(sorted(counts[column].items()))
        if actual != expected_counts:
            raise AnchorImportError(
                f"{artifact['id']} value census mismatch for {column}: "
                f"expected={expected_counts!r}, actual={actual!r}"
            )
    observed_ranges: dict[str, dict[str, int]] = {}
    for column, bounds in rules["integer_ranges"].items():
        observed = {"min": min(integer_values[column]), "max": max(integer_values[column])}
        if observed != bounds:
            raise AnchorImportError(
                f"{artifact['id']} range mismatch for {column}: expected={bounds}, actual={observed}"
            )
        observed_ranges[column] = observed
    return {
        "artifact_id": artifact["id"],
        "rows": rows,
        "columns": list(artifact["columns"]),
        "timing_observations": dict(sorted(timing_observations.items())),
        "exact_value_counts": {
            column: dict(sorted(counter.items())) for column, counter in sorted(counts.items())
        },
        "integer_ranges": observed_ranges,
        "unique_profile_keys": unique_profile_keys,
        "duplicate_profile_rows": duplicate_profile_rows,
    }


def _decode_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise AnchorImportError(f"cannot read {description} {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise AnchorImportError(f"{description} is not UTF-8: {path}") from error


def _single_summary_value(lines: list[str], pattern: str, path: str) -> str:
    regex = re.compile(pattern)
    values = [match.group(1).strip() for line in lines if (match := regex.match(line))]
    if not values:
        raise AnchorImportError(f"{path} is missing a required MLPerf field")
    if len(set(values)) != 1:
        raise AnchorImportError(f"{path} has conflicting MLPerf field values: {values}")
    return values[0]


def _inspect_performance(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    lines = _decode_text(path, artifact["id"]).splitlines()
    scenario = _single_summary_value(lines, r"^Scenario\s*:\s*(.+)$", artifact["id"])
    mode = _single_summary_value(lines, r"^Mode\s*:\s*(.+)$", artifact["id"])
    result = _single_summary_value(lines, r"^Result is\s*:\s*(.+)$", artifact["id"])
    if scenario == "Offline":
        token_pattern = r"^Tokens per second:\s*(.+)$"
        sample_pattern = r"^Samples per second:\s*(.+)$"
    elif scenario == "Server":
        token_pattern = r"^Completed tokens per second\s*:\s*(.+)$"
        sample_pattern = r"^Completed samples per second\s*:\s*(.+)$"
    else:
        raise AnchorImportError(f"{artifact['id']} has unsupported scenario {scenario!r}")
    tokens = _single_summary_value(lines, token_pattern, artifact["id"])
    samples = _single_summary_value(lines, sample_pattern, artifact["id"])
    observed = {
        "scenario": scenario,
        "mode": mode,
        "tokens_per_second": tokens,
        "samples_per_second": samples,
        "result": result,
    }
    expected = artifact["validation"]
    for key in ("scenario", "mode", "result"):
        if observed[key] != expected[key]:
            raise AnchorImportError(
                f"{artifact['id']} {key} mismatch: expected={expected[key]!r}, actual={observed[key]!r}"
            )
    for key in ("tokens_per_second", "samples_per_second"):
        if _decimal(observed[key], f"{artifact['id']}.{key}") != _decimal(
            expected[key], f"{artifact['id']}.expected.{key}"
        ):
            raise AnchorImportError(f"{artifact['id']} {key} mismatch")
        observed[key] = expected[key]
    return {"artifact_id": artifact["id"], **observed}


def _inspect_accuracy(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    lines = _decode_text(path, artifact["id"]).splitlines()
    metric_lines = [line for line in lines if line.strip().startswith("{")]
    if len(metric_lines) != 1:
        raise AnchorImportError(f"{artifact['id']} must contain exactly one metrics mapping")
    try:
        metrics = ast.literal_eval(metric_lines[0])
    except (SyntaxError, ValueError) as error:
        raise AnchorImportError(f"{artifact['id']} contains invalid metrics") from error
    if not isinstance(metrics, dict):
        raise AnchorImportError(f"{artifact['id']} metrics must be a mapping")
    expected = artifact["validation"]["expected_metrics"]
    observed: dict[str, Any] = {}
    for key, expected_value in expected.items():
        if key not in metrics:
            raise AnchorImportError(f"{artifact['id']} is missing metric {key}")
        actual = metrics[key]
        if isinstance(expected_value, int):
            if isinstance(actual, bool) or actual != expected_value:
                raise AnchorImportError(f"{artifact['id']} metric {key} mismatch")
            observed[key] = actual
        else:
            if Decimal(str(actual)) != Decimal(expected_value):
                raise AnchorImportError(f"{artifact['id']} metric {key} mismatch")
            observed[key] = expected_value
    hash_lines = [line for line in lines if line.startswith("hash=")]
    if len(hash_lines) != 1 or not re.fullmatch(r"hash=[0-9a-f]{64}", hash_lines[0]):
        raise AnchorImportError(f"{artifact['id']} has no unique accuracy hash")
    return {
        "artifact_id": artifact["id"],
        "metrics": observed,
        "accuracy_hash": hash_lines[0][5:],
    }


def _inspect_json_record(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    value, _ = _load_object(path, artifact["id"])
    expected = artifact["validation"]["expected_fields"]
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise AnchorImportError(
                f"{artifact['id']} field {key} mismatch: expected={expected_value!r}, actual={value.get(key)!r}"
            )
    return {"artifact_id": artifact["id"], "expected_fields": expected}


def _inspect_user_conf(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(_decode_text(path, artifact["id"]).splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise AnchorImportError(f"{artifact['id']} line {line_number} has no '='")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value or key in entries:
            raise AnchorImportError(f"{artifact['id']} line {line_number} is malformed or duplicated")
        entries[key] = value
    expected = artifact["validation"]["expected_entries"]
    if entries != expected:
        raise AnchorImportError(
            f"{artifact['id']} entries mismatch: expected={expected!r}, actual={entries!r}"
        )
    return {"artifact_id": artifact["id"], "entries": entries}


def _inspect_artifact(path: Path, artifact: dict[str, Any]) -> dict[str, Any] | None:
    size = path.stat().st_size
    digest = _sha256_file(path)
    if size != artifact["bytes"]:
        raise AnchorImportError(
            f"{artifact['id']} byte count mismatch: expected {artifact['bytes']}, got {size}"
        )
    if digest != artifact["sha256"]:
        raise AnchorImportError(
            f"{artifact['id']} SHA-256 mismatch: expected {artifact['sha256']}, got {digest}"
        )
    kind = artifact["kind"]
    if kind == "csv_profile":
        return _inspect_csv(path, artifact)
    if kind == "json_record":
        return _inspect_json_record(path, artifact)
    if kind == "mlperf_performance_summary":
        return _inspect_performance(path, artifact)
    if kind == "mlperf_accuracy_summary":
        return _inspect_accuracy(path, artifact)
    if kind == "mlperf_user_conf":
        return _inspect_user_conf(path, artifact)
    return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def import_anchor(
    *,
    manifest_path: Path,
    vidur_root: Path,
    mlperf_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    vidur_root = vidur_root.resolve()
    mlperf_root = mlperf_root.resolve()
    output_dir = output_dir.resolve()
    manifest, manifest_bytes = _load_object(manifest_path, "anchor manifest")
    _validate_manifest(manifest)
    if output_dir.exists() or output_dir.is_symlink():
        raise AnchorImportError(f"output directory already exists: {output_dir}")
    if _is_within(output_dir, vidur_root) or _is_within(output_dir, mlperf_root):
        raise AnchorImportError("output directory must be outside both source checkouts")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    roots = {"vidur": vidur_root, "mlperf": mlperf_root}
    source_receipts: dict[str, Any] = {}
    source_paths: dict[tuple[str, str], Path] = {}
    profile_summaries: list[dict[str, Any]] = []
    held_out_summaries: list[dict[str, Any]] = []
    for source_id in ("vidur", "mlperf"):
        source = manifest["sources"][source_id]
        source_receipts[source_id] = _verify_checkout(roots[source_id], source, source_id)
        source_receipts[source_id]["license"] = source["license"]
        source_receipts[source_id]["license_status"] = source["license_status"]
        for artifact in source["artifacts"]:
            path = _artifact_path(roots[source_id], artifact["path"], artifact["id"])
            summary = _inspect_artifact(path, artifact)
            source_paths[(source_id, artifact["id"])] = path
            if summary is not None:
                if artifact["kind"] == "csv_profile":
                    profile_summaries.append(summary)
                elif source_id == "mlperf":
                    held_out_summaries.append(summary)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        manifest_copy = temporary / "anchor-manifest.json"
        manifest_copy.write_bytes(manifest_bytes)
        artifacts: list[dict[str, Any]] = [
            {
                "artifact_id": "anchor.manifest",
                "source_id": "anchor",
                "source_path": manifest_path.name,
                "bundle_path": "anchor-manifest.json",
                "kind": "anchor_manifest",
                "bytes": len(manifest_bytes),
                "sha256": _sha256_bytes(manifest_bytes),
            }
        ]
        for source_id in ("vidur", "mlperf"):
            for artifact in manifest["sources"][source_id]["artifacts"]:
                source_path = source_paths[(source_id, artifact["id"])]
                bundle_relative = Path("sources") / source_id / Path(artifact["path"])
                target = temporary / bundle_relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target)
                copied = {
                    "artifact_id": artifact["id"],
                    "source_id": source_id,
                    "source_path": artifact["path"],
                    "bundle_path": bundle_relative.as_posix(),
                    "kind": artifact["kind"],
                    "bytes": target.stat().st_size,
                    "sha256": _sha256_file(target),
                }
                if copied["bytes"] != artifact["bytes"] or copied["sha256"] != artifact["sha256"]:
                    raise AnchorImportError(f"copy verification failed for {artifact['id']}")
                artifacts.append(copied)

        bundle = {
            "schema": BUNDLE_SCHEMA,
            "result": "pass",
            "anchor": {
                "anchor_id": manifest["anchor_id"],
                "manifest": {
                    "bundle_path": "anchor-manifest.json",
                    "bytes": len(manifest_bytes),
                    "sha256": _sha256_bytes(manifest_bytes),
                },
                "intended_use": manifest["intended_use"],
                "target": manifest["target"],
                "claim_boundary": manifest["claim_boundary"],
            },
            "sources": source_receipts,
            "profiles": sorted(profile_summaries, key=lambda item: item["artifact_id"]),
            "held_out_records": sorted(held_out_summaries, key=lambda item: item["artifact_id"]),
            "artifacts": artifacts,
            "eligibility": {
                "source_integrity_valid": True,
                "calibration_input_integrity_valid": True,
                "held_out_reference_integrity_valid": True,
                "predictor_calibration_valid": False,
                "end_to_end_external_validation_valid": False,
                "frontier_timing_usable_for_performance": False,
                "paper_result_eligible": False,
                "timing_claim_state": "anchor_ready_not_calibrated",
                "blockers": list(manifest["claim_boundary"]["required_next_evidence"]),
            },
        }
        bundle_path = temporary / "bundle-manifest.json"
        with bundle_path.open("w", encoding="utf-8", newline="") as handle:
            json.dump(bundle, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_dir)
        return bundle
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--vidur-root", type=Path, required=True)
    parser.add_argument("--mlperf-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = import_anchor(
            manifest_path=args.manifest,
            vidur_root=args.vidur_root,
            mlperf_root=args.mlperf_root,
            output_dir=args.output_dir,
        )
    except (AnchorImportError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"imported {result['anchor']['anchor_id']}: "
        f"profiles={len(result['profiles'])} "
        f"held_out_records={len(result['held_out_records'])} "
        f"timing={result['eligibility']['timing_claim_state']}"
    )
    print(f"bundle={args.output_dir.resolve() / 'bundle-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
