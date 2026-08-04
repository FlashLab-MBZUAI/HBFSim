#!/usr/bin/env python3
"""Validate facet-scoped, digest-bound external-simulator evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.compare_ledgers import (  # noqa: E402
    first_difference,
    load_ledger,
)
from validation.contracts import (  # noqa: E402
    ContractError,
    load_case,
    load_json_strict,
)
from validation.hbf_oracle import build_ledger as build_hbf_ledger  # noqa: E402
from validation.hbm_oracle import build_ledger as build_hbm_ledger  # noqa: E402
from validation.ledger_reducer import reduce_ledger  # noqa: E402


TOOLS_SCHEMA = {
    "name": "hbfsim.validation.external-tools",
    "version": 2,
}
WRAPPER_SCHEMA = {
    "name": "hbfsim.validation.external-wrapper-output",
    "version": 2,
}
NORMALIZED_SCHEMA = {
    "name": "hbfsim.validation.external-normalized",
    "version": 2,
}
SOURCE_SCHEMA = {
    "name": "hbfsim.validation.external-source-provenance",
    "version": 1,
}
EVIDENCE_SCHEMA = {
    "name": "hbfsim.validation.external-evidence",
    "version": 2,
}
REPORT_SCHEMA = {
    "name": "hbfsim.validation.external-report",
    "version": 2,
}
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")


def _exact_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    where: str,
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ContractError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise ContractError(f"{where}: unknown keys {sorted(unknown)}")


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where}: expected object")
    return value


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{where}: expected nonnegative integer")
    return value


def _positive_int(value: Any, where: str) -> int:
    parsed = _nonnegative_int(value, where)
    if parsed == 0:
        raise ContractError(f"{where}: expected positive integer")
    return parsed


def _positive_number(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ContractError(f"{where}: expected positive finite number")
    return float(value)


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where}: expected nonempty string")
    return value


def _identifier(value: Any, where: str) -> str:
    parsed = _nonempty_string(value, where)
    if not IDENTIFIER_RE.fullmatch(parsed):
        raise ContractError(f"{where}: invalid identifier")
    return parsed


def _string_list(
    value: Any,
    where: str,
    *,
    nonempty: bool = True,
) -> list[str]:
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        qualifier = "nonempty " if nonempty else ""
        raise ContractError(
            f"{where}: expected {qualifier}list of unique strings")
    return value


def _argv(value: Any, where: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(argument, str) and argument for argument in value)
    ):
        raise ContractError(f"{where}: expected nonempty argv")
    return value


def _argv_list(value: Any, where: str) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{where}: expected nonempty command list")
    return [
        _argv(command, f"{where}[{index}]")
        for index, command in enumerate(value)
    ]


def _relative_path(value: Any, where: str) -> Path:
    parsed = _nonempty_string(value, where)
    path = Path(parsed)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError(f"{where}: expected contained relative path")
    return path


def _normalize_ramulator2(raw: dict[str, Any]) -> dict[str, int | float]:
    _exact_keys(
        raw,
        {
            "clock_period_ns",
            "request_epoch_cycle",
            "completion_cycle",
            "transaction_bytes",
            "read_requests",
            "write_requests",
            "row_hits",
            "row_misses",
            "row_conflicts",
        },
        set(),
        "ramulator2.raw",
    )
    clock_period = _positive_number(
        raw["clock_period_ns"], "ramulator2.raw.clock_period_ns")
    epoch = _nonnegative_int(
        raw["request_epoch_cycle"],
        "ramulator2.raw.request_epoch_cycle",
    )
    completion = _nonnegative_int(
        raw["completion_cycle"], "ramulator2.raw.completion_cycle")
    if completion < epoch:
        raise ContractError(
            "ramulator2.raw.completion_cycle precedes request epoch")
    transaction_bytes = _positive_int(
        raw["transaction_bytes"], "ramulator2.raw.transaction_bytes")
    reads = _nonnegative_int(
        raw["read_requests"], "ramulator2.raw.read_requests")
    writes = _nonnegative_int(
        raw["write_requests"], "ramulator2.raw.write_requests")
    row_hits = _nonnegative_int(
        raw["row_hits"], "ramulator2.raw.row_hits")
    row_misses = _nonnegative_int(
        raw["row_misses"], "ramulator2.raw.row_misses")
    row_conflicts = _nonnegative_int(
        raw["row_conflicts"], "ramulator2.raw.row_conflicts")
    if row_hits + row_misses + row_conflicts != reads + writes:
        raise ContractError(
            "ramulator2.raw row classifications do not cover all requests")
    return {
        "read_bytes": reads * transaction_bytes,
        "write_bytes": writes * transaction_bytes,
        "row_hits": row_hits,
        "row_misses": row_misses,
        "row_conflicts": row_conflicts,
        "service_span_ns": (completion - epoch) * clock_period,
    }


def _normalize_mqsim(raw: dict[str, Any]) -> dict[str, int]:
    _exact_keys(
        raw,
        {
            "page_size_bytes",
            "issued_flash_reads",
            "issued_flash_programs",
            "mapping_flash_reads",
            "mapping_flash_programs",
            "issued_flash_erases",
            "gc_executions",
        },
        set(),
        "mqsim.raw",
    )
    page_size = _positive_int(
        raw["page_size_bytes"], "mqsim.raw.page_size_bytes")
    reads = _nonnegative_int(
        raw["issued_flash_reads"], "mqsim.raw.issued_flash_reads")
    programs = _nonnegative_int(
        raw["issued_flash_programs"], "mqsim.raw.issued_flash_programs")
    mapping_reads = _nonnegative_int(
        raw["mapping_flash_reads"], "mqsim.raw.mapping_flash_reads")
    mapping_programs = _nonnegative_int(
        raw["mapping_flash_programs"],
        "mqsim.raw.mapping_flash_programs",
    )
    erases = _nonnegative_int(
        raw["issued_flash_erases"], "mqsim.raw.issued_flash_erases")
    gc_executions = _nonnegative_int(
        raw["gc_executions"], "mqsim.raw.gc_executions")
    if mapping_reads > reads or mapping_programs > programs:
        raise ContractError(
            "mqsim.raw mapping commands exceed total flash commands")
    if erases != 0 or gc_executions != 0:
        raise ContractError(
            "mqsim-stats-wrapper-v2 covers direct page I/O only; "
            "erase or GC work was observed")
    data_reads = reads - mapping_reads
    data_programs = programs - mapping_programs
    return {
        "data_read_bytes": data_reads * page_size,
        "data_write_bytes": data_programs * page_size,
        "data_page_reads": data_reads,
        "data_page_programs": data_programs,
    }


ADAPTERS: dict[str, Callable[[dict[str, Any]], dict[str, int | float]]] = {
    "ramulator2-stats-wrapper-v2": _normalize_ramulator2,
    "mqsim-stats-wrapper-v2": _normalize_mqsim,
}


def _validate_dependency(value: Any, where: str) -> dict[str, str]:
    dependency = _object(value, where)
    _exact_keys(
        dependency,
        {"id", "repository", "commit"},
        set(),
        where,
    )
    _identifier(dependency["id"], f"{where}.id")
    _nonempty_string(dependency["repository"], f"{where}.repository")
    commit = _nonempty_string(dependency["commit"], f"{where}.commit")
    if not GIT_OBJECT_RE.fullmatch(commit):
        raise ContractError(f"{where}.commit: expected full Git SHA")
    return dependency


def load_tools(path: Path) -> list[dict[str, Any]]:
    manifest = load_json_strict(path)
    if not isinstance(manifest, dict):
        raise ContractError(f"{path}: manifest must be an object")
    _exact_keys(manifest, {"schema", "tools"}, set(), str(path))
    if manifest["schema"] != TOOLS_SCHEMA:
        raise ContractError(f"{path}: unsupported tools schema")
    if not isinstance(manifest["tools"], list) or not manifest["tools"]:
        raise ContractError(f"{path}: tools must be a nonempty list")

    tool_ids: set[str] = set()
    tools: list[dict[str, Any]] = []
    for index, raw_tool in enumerate(manifest["tools"]):
        where = f"{path}:tools[{index}]"
        tool = _object(raw_tool, where)
        _exact_keys(
            tool,
            {
                "id",
                "domain",
                "repository",
                "commit",
                "source_tree",
                "pin_checked_at",
                "adapter",
                "required_build_artifacts",
                "required_case_artifacts",
                "dependencies",
                "fixture",
                "facets",
                "cases",
            },
            set(),
            where,
        )
        tool_id = _identifier(tool["id"], f"{where}.id")
        if tool_id in tool_ids:
            raise ContractError(f"{where}.id: duplicate tool ID")
        tool_ids.add(tool_id)
        if tool["domain"] not in {"dram", "flash"}:
            raise ContractError(f"{where}.domain: expected dram or flash")
        _nonempty_string(tool["repository"], f"{where}.repository")
        for key in ("commit", "source_tree"):
            value = _nonempty_string(tool[key], f"{where}.{key}")
            if not GIT_OBJECT_RE.fullmatch(value):
                raise ContractError(
                    f"{where}.{key}: expected full Git SHA")
        _nonempty_string(tool["pin_checked_at"], f"{where}.pin_checked_at")
        if tool["adapter"] not in ADAPTERS:
            raise ContractError(f"{where}.adapter: unknown adapter")
        _string_list(
            tool["required_build_artifacts"],
            f"{where}.required_build_artifacts",
        )
        required_case_artifacts = _string_list(
            tool["required_case_artifacts"],
            f"{where}.required_case_artifacts",
        )
        for required in ("raw_output", "hbfsim_case", "hbfsim_ledger"):
            if required not in required_case_artifacts:
                raise ContractError(
                    f"{where}.required_case_artifacts omits {required}")

        dependencies = tool["dependencies"]
        if not isinstance(dependencies, list):
            raise ContractError(f"{where}.dependencies: expected list")
        dependency_ids: set[str] = set()
        for dependency_index, dependency in enumerate(dependencies):
            validated = _validate_dependency(
                dependency,
                f"{where}.dependencies[{dependency_index}]",
            )
            if validated["id"] in dependency_ids:
                raise ContractError(
                    f"{where}.dependencies: duplicate dependency ID")
            dependency_ids.add(validated["id"])

        fixture = _object(tool["fixture"], f"{where}.fixture")
        _exact_keys(
            fixture, {"raw", "expected"}, set(), f"{where}.fixture")
        _relative_path(fixture["raw"], f"{where}.fixture.raw")
        _relative_path(fixture["expected"], f"{where}.fixture.expected")

        facets = tool["facets"]
        if not isinstance(facets, list) or not facets:
            raise ContractError(f"{where}.facets: expected nonempty list")
        facet_ids: set[str] = set()
        facet_case_ids: dict[str, list[str]] = {}
        for facet_index, raw_facet in enumerate(facets):
            facet_where = f"{where}.facets[{facet_index}]"
            facet = _object(raw_facet, facet_where)
            _exact_keys(
                facet,
                {
                    "id",
                    "claim",
                    "shared_boundary",
                    "excluded",
                    "cases",
                },
                set(),
                facet_where,
            )
            facet_id = _identifier(facet["id"], f"{facet_where}.id")
            if facet_id in facet_ids:
                raise ContractError(f"{facet_where}.id: duplicate facet ID")
            facet_ids.add(facet_id)
            _nonempty_string(facet["claim"], f"{facet_where}.claim")
            _string_list(
                facet["shared_boundary"],
                f"{facet_where}.shared_boundary",
            )
            _string_list(facet["excluded"], f"{facet_where}.excluded")
            facet_case_ids[facet_id] = _string_list(
                facet["cases"], f"{facet_where}.cases")

        cases = tool["cases"]
        if not isinstance(cases, list) or not cases:
            raise ContractError(f"{where}.cases: expected nonempty list")
        case_ids: set[str] = set()
        cases_by_facet = {facet_id: [] for facet_id in facet_ids}
        for case_index, raw_case in enumerate(cases):
            case_where = f"{where}.cases[{case_index}]"
            case = _object(raw_case, case_where)
            _exact_keys(
                case,
                {
                    "case_id",
                    "facet_id",
                    "hbfsim_case",
                    "external_inputs",
                    "shared_metrics",
                    "hbfsim_metrics",
                },
                set(),
                case_where,
            )
            case_id = _identifier(case["case_id"], f"{case_where}.case_id")
            if case_id in case_ids:
                raise ContractError(
                    f"{case_where}.case_id: duplicate case ID")
            case_ids.add(case_id)
            facet_id = _identifier(
                case["facet_id"], f"{case_where}.facet_id")
            if facet_id not in facet_ids:
                raise ContractError(
                    f"{case_where}.facet_id: unknown facet")
            cases_by_facet[facet_id].append(case_id)
            _relative_path(
                case["hbfsim_case"], f"{case_where}.hbfsim_case")
            external_inputs = _object(
                case["external_inputs"], f"{case_where}.external_inputs")
            expected_input_names = {
                name
                for name in required_case_artifacts
                if name in {"config", "workload", "trace"}
            }
            if set(external_inputs) != expected_input_names:
                raise ContractError(
                    f"{case_where}.external_inputs: expected keys "
                    f"{sorted(expected_input_names)}")
            for name, input_path in external_inputs.items():
                _relative_path(
                    input_path,
                    f"{case_where}.external_inputs.{name}",
                )
            shared_metrics = _string_list(
                case["shared_metrics"], f"{case_where}.shared_metrics")
            metric_map = _object(
                case["hbfsim_metrics"], f"{case_where}.hbfsim_metrics")
            if set(metric_map) != set(shared_metrics):
                raise ContractError(
                    f"{case_where}.hbfsim_metrics keys differ from "
                    "shared_metrics")
            for metric, ledger_metric in metric_map.items():
                _nonempty_string(
                    ledger_metric,
                    f"{case_where}.hbfsim_metrics.{metric}",
                )
        for facet_id in facet_ids:
            if facet_case_ids[facet_id] != cases_by_facet[facet_id]:
                raise ContractError(
                    f"{where}: facet {facet_id} case census/order differs "
                    "from tool cases")
        tools.append(tool)
    return tools


def normalize_wrapper(
    wrapper: dict[str, Any],
    tool: dict[str, Any],
    *,
    require_provenance: str,
    source: str,
) -> dict[str, Any]:
    if not isinstance(wrapper, dict):
        raise ContractError(f"{source}: wrapper output must be an object")
    _exact_keys(
        wrapper,
        {
            "schema",
            "adapter",
            "tool_id",
            "tool_commit",
            "provenance",
            "case_id",
            "raw",
        },
        set(),
        source,
    )
    if wrapper["schema"] != WRAPPER_SCHEMA:
        raise ContractError(f"{source}: unsupported wrapper schema")
    expected_identity = {
        "adapter": tool["adapter"],
        "tool_id": tool["id"],
        "tool_commit": tool["commit"],
    }
    for key, expected in expected_identity.items():
        if wrapper[key] != expected:
            raise ContractError(
                f"{source}.{key}: expected {expected!r}")
    if wrapper["provenance"] != require_provenance:
        raise ContractError(
            f"{source}.provenance: expected {require_provenance!r}, "
            f"observed {wrapper['provenance']!r}")
    _identifier(wrapper["case_id"], f"{source}.case_id")
    raw = _object(wrapper["raw"], f"{source}.raw")
    metrics = ADAPTERS[tool["adapter"]](raw)
    return {
        "schema": NORMALIZED_SCHEMA,
        "tool_id": tool["id"],
        "domain": tool["domain"],
        "case_id": wrapper["case_id"],
        "metrics": metrics,
    }


def validate_fixture(
    tool: dict[str, Any],
    fixture_dir: Path,
) -> dict[str, Any]:
    raw_path = fixture_dir / tool["fixture"]["raw"]
    expected_path = fixture_dir / tool["fixture"]["expected"]
    wrapper = load_json_strict(raw_path)
    normalized = normalize_wrapper(
        wrapper,
        tool,
        require_provenance="fixture",
        source=str(raw_path),
    )
    expected = load_json_strict(expected_path)
    if normalized != expected:
        raise ContractError(
            f"{tool['id']}: normalized fixture does not match "
            f"{expected_path}")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _resolve_artifact(
    evidence_dir: Path,
    artifact: Any,
    where: str,
) -> Path:
    value = _object(artifact, where)
    _exact_keys(value, {"path", "sha256", "bytes"}, set(), where)
    relative = _relative_path(value["path"], f"{where}.path")
    if not isinstance(value["sha256"], str) or not SHA256_RE.fullmatch(
        value["sha256"]
    ):
        raise ContractError(f"{where}.sha256: invalid digest")
    expected_bytes = _nonnegative_int(value["bytes"], f"{where}.bytes")
    path = (evidence_dir / relative).resolve()
    try:
        path.relative_to(evidence_dir.resolve())
    except ValueError as error:
        raise ContractError(f"{where}.path escapes evidence directory") from error
    if not path.is_file():
        raise ContractError(f"{where}.path does not exist: {path}")
    if path.stat().st_size != expected_bytes:
        raise ContractError(
            f"{where}.bytes: expected {expected_bytes}, "
            f"observed {path.stat().st_size}")
    observed = _sha256_file(path)
    if observed != value["sha256"]:
        raise ContractError(
            f"{where}.sha256: expected {value['sha256']}, "
            f"observed {observed}")
    return path


def _resolve_artifact_map(
    evidence_dir: Path,
    value: Any,
    *,
    required: list[str],
    where: str,
) -> dict[str, Path]:
    artifacts = _object(value, where)
    if set(artifacts) != set(required):
        raise ContractError(
            f"{where}: expected artifact keys {sorted(required)}, "
            f"observed {sorted(artifacts)}")
    return {
        name: _resolve_artifact(
            evidence_dir,
            artifacts[name],
            f"{where}.{name}",
        )
        for name in required
    }


def _validate_source_provenance(
    tool: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    source = _object(load_json_strict(path), str(path))
    _exact_keys(
        source,
        {
            "schema",
            "repository",
            "commit",
            "tree",
            "tracked_clean",
            "dependencies",
        },
        set(),
        str(path),
    )
    if source["schema"] != SOURCE_SCHEMA:
        raise ContractError(f"{path}: unsupported source provenance schema")
    expected = {
        "repository": tool["repository"],
        "commit": tool["commit"],
        "tree": tool["source_tree"],
        "tracked_clean": True,
        "dependencies": tool["dependencies"],
    }
    for key, value in expected.items():
        if source[key] != value:
            raise ContractError(
                f"{path}.{key}: expected {value!r}, "
                f"observed {source[key]!r}")
    return source


def _compare_metrics(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    source: str,
) -> None:
    if expected.keys() != actual.keys():
        raise ContractError(
            f"{source}: metric keys differ: expected "
            f"{sorted(expected)}, observed {sorted(actual)}")
    for key in expected:
        left = expected[key]
        right = actual[key]
        if isinstance(left, bool) or isinstance(right, bool):
            raise ContractError(f"{source}.{key}: booleans are not metrics")
        if isinstance(left, int) and isinstance(right, int):
            equal = left == right
        else:
            equal = (
                isinstance(left, (int, float))
                and isinstance(right, (int, float))
                and math.isfinite(float(left))
                and math.isfinite(float(right))
                and math.isclose(
                    float(left),
                    float(right),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
            )
        if not equal:
            raise ContractError(
                f"{source}.{key}: expected {left!r}, observed {right!r}")


def _bind_ledger_to_case(
    tool: dict[str, Any],
    case_spec: dict[str, Any],
    case_path: Path,
    ledger_path: Path,
) -> dict[str, int | float]:
    case = load_case(case_path)
    if case["case_id"] != case_spec["case_id"]:
        raise ContractError(
            f"{case_path}: case_id does not match manifest")
    expected_model = "hbm" if tool["domain"] == "dram" else "hbf"
    if case["model"] != expected_model:
        raise ContractError(
            f"{case_path}: expected {expected_model} case")
    actual_ledger = load_ledger(ledger_path)
    expected_ledger = (
        build_hbm_ledger(case)
        if expected_model == "hbm"
        else build_hbf_ledger(case)
    )
    difference = first_difference(
        expected_ledger,
        actual_ledger,
        abs_tolerance=case["comparison"]["time_abs_tolerance_ns"],
        rel_tolerance=case["comparison"]["time_rel_tolerance"],
    )
    if difference is not None:
        raise ContractError(
            f"{ledger_path}: HBFSim ledger does not match the "
            "manifest-bound independent case:\n"
            + difference.format(
                case_id=case["case_id"],
                replay="digest-bound evidence ledger",
            )
        )
    reduced = reduce_ledger(actual_ledger, source=str(ledger_path))
    metrics = reduced["metrics"]
    result: dict[str, int | float] = {}
    for external_name, ledger_name in case_spec["hbfsim_metrics"].items():
        if ledger_name not in metrics:
            raise ContractError(
                f"{ledger_path}: missing reduced metric {ledger_name}")
        value = metrics[ledger_name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ContractError(
                f"{ledger_path}: reduced metric {ledger_name} is not numeric")
        result[external_name] = value
    return result


def _validate_current_file_copy(
    repository_path: str,
    evidence_path: Path,
    repository: Path,
    *,
    label: str,
) -> None:
    current = (repository / repository_path).resolve()
    try:
        current.relative_to(repository.resolve())
    except ValueError as error:
        raise ContractError(
            f"manifest {label} escapes repository: {current}") from error
    if not current.is_file():
        raise ContractError(f"manifest {label} does not exist: {current}")
    if _artifact_record(current) != _artifact_record(evidence_path):
        raise ContractError(
            f"{evidence_path}: evidence {label} differs from current "
            f"manifest input {current}")


def validate_actual_evidence(
    tool: dict[str, Any],
    evidence_path: Path,
    *,
    repository: Path,
    expected_hbfsim_probe: Path | None = None,
) -> dict[str, Any]:
    evidence = _object(load_json_strict(evidence_path), str(evidence_path))
    _exact_keys(
        evidence,
        {
            "schema",
            "tool_id",
            "tool_commit",
            "adapter",
            "provenance",
            "source",
            "build",
            "hbfsim_probe",
            "cases",
        },
        set(),
        str(evidence_path),
    )
    if evidence["schema"] != EVIDENCE_SCHEMA:
        raise ContractError(f"{evidence_path}: unsupported evidence schema")
    if evidence["provenance"] != "actual":
        raise ContractError(
            f"{evidence_path}: fixture evidence cannot become actual PASS")
    expected_identity = {
        "tool_id": tool["id"],
        "tool_commit": tool["commit"],
        "adapter": tool["adapter"],
    }
    for key, expected in expected_identity.items():
        if evidence[key] != expected:
            raise ContractError(
                f"{evidence_path}.{key}: expected {expected!r}")

    evidence_dir = evidence_path.parent.resolve()
    source_path = _resolve_artifact(
        evidence_dir,
        evidence["source"],
        f"{evidence_path}.source",
    )
    _validate_source_provenance(tool, source_path)

    build = _object(evidence["build"], f"{evidence_path}.build")
    _exact_keys(
        build,
        {"compiler", "commands", "artifacts"},
        set(),
        f"{evidence_path}.build",
    )
    compiler = _object(
        build["compiler"], f"{evidence_path}.build.compiler")
    _exact_keys(
        compiler,
        {"path", "version"},
        set(),
        f"{evidence_path}.build.compiler",
    )
    _nonempty_string(
        compiler["path"], f"{evidence_path}.build.compiler.path")
    _nonempty_string(
        compiler["version"], f"{evidence_path}.build.compiler.version")
    _argv_list(build["commands"], f"{evidence_path}.build.commands")
    build_paths = _resolve_artifact_map(
        evidence_dir,
        build["artifacts"],
        required=tool["required_build_artifacts"],
        where=f"{evidence_path}.build.artifacts",
    )

    probe_path = _resolve_artifact(
        evidence_dir,
        evidence["hbfsim_probe"],
        f"{evidence_path}.hbfsim_probe",
    )
    if expected_hbfsim_probe is not None:
        if not expected_hbfsim_probe.is_file():
            raise ContractError(
                f"expected HBFSim probe does not exist: "
                f"{expected_hbfsim_probe}")
        if _artifact_record(probe_path) != _artifact_record(
            expected_hbfsim_probe
        ):
            raise ContractError(
                f"{evidence_path}: HBFSim probe digest differs from the "
                "certificate probe")

    case_entries = evidence["cases"]
    if not isinstance(case_entries, list):
        raise ContractError(f"{evidence_path}.cases: expected list")
    expected_case_ids = [case["case_id"] for case in tool["cases"]]
    observed_case_ids = [
        entry.get("case_id") if isinstance(entry, dict) else None
        for entry in case_entries
    ]
    if observed_case_ids != expected_case_ids:
        raise ContractError(
            f"{evidence_path}.cases: expected exact ordered census "
            f"{expected_case_ids}, observed {observed_case_ids}")

    case_results: list[dict[str, Any]] = []
    for case_spec, raw_entry in zip(tool["cases"], case_entries):
        case_id = case_spec["case_id"]
        where = f"{evidence_path}.cases[{case_id}]"
        entry = _object(raw_entry, where)
        _exact_keys(
            entry,
            {"case_id", "commands", "artifacts"},
            set(),
            where,
        )
        commands = _object(entry["commands"], f"{where}.commands")
        _exact_keys(
            commands,
            {"external", "hbfsim"},
            set(),
            f"{where}.commands",
        )
        _argv(commands["external"], f"{where}.commands.external")
        _argv(commands["hbfsim"], f"{where}.commands.hbfsim")
        paths = _resolve_artifact_map(
            evidence_dir,
            entry["artifacts"],
            required=tool["required_case_artifacts"],
            where=f"{where}.artifacts",
        )
        _validate_current_file_copy(
            case_spec["hbfsim_case"],
            paths["hbfsim_case"],
            repository,
            label="HBFSim case",
        )
        for input_name, repository_path in case_spec[
            "external_inputs"
        ].items():
            _validate_current_file_copy(
                repository_path,
                paths[input_name],
                repository,
                label=f"external {input_name}",
            )
        wrapper = load_json_strict(paths["raw_output"])
        normalized = normalize_wrapper(
            wrapper,
            tool,
            require_provenance="actual",
            source=str(paths["raw_output"]),
        )
        if normalized["case_id"] != case_id:
            raise ContractError(
                f"{paths['raw_output']}: case_id does not match "
                f"manifest case {case_id}")
        if set(normalized["metrics"]) != set(case_spec["shared_metrics"]):
            raise ContractError(
                f"{paths['raw_output']}: adapter metrics differ from "
                "manifest shared_metrics")
        hbfsim_metrics = _bind_ledger_to_case(
            tool,
            case_spec,
            paths["hbfsim_case"],
            paths["hbfsim_ledger"],
        )
        _compare_metrics(
            hbfsim_metrics,
            normalized["metrics"],
            source=f"{evidence_path}:{case_id}:shared_metrics",
        )
        case_results.append({
            "case_id": case_id,
            "facet_id": case_spec["facet_id"],
            "status": "pass",
            "metrics": normalized["metrics"],
            "artifact_digests": {
                name: entry["artifacts"][name]["sha256"]
                for name in sorted(entry["artifacts"])
            },
        })

    facet_results = []
    for facet in tool["facets"]:
        case_ids = facet["cases"]
        facet_results.append({
            "id": facet["id"],
            "status": "pass",
            "claim": facet["claim"],
            "cases": case_ids,
            "shared_boundary": facet["shared_boundary"],
            "excluded": facet["excluded"],
        })
    return {
        "status": "pass",
        "source_digest": evidence["source"]["sha256"],
        "hbfsim_probe_digest": evidence["hbfsim_probe"]["sha256"],
        "build_artifact_digests": {
            name: build["artifacts"][name]["sha256"]
            for name in sorted(build_paths)
        },
        "cases": case_results,
        "facets": facet_results,
    }


def build_report(
    manifest_path: Path,
    fixture_dir: Path,
    evidence_dir: Path | None,
    *,
    repository: Path | None = None,
    expected_hbfsim_probe: Path | None = None,
) -> dict[str, Any]:
    tools = load_tools(manifest_path)
    if repository is None:
        repository = manifest_path.resolve().parents[2]
    repository = repository.resolve()
    entries = []
    validated_facets: list[dict[str, Any]] = []
    for tool in tools:
        normalized = validate_fixture(tool, fixture_dir)
        actual: dict[str, Any] = {
            "status": "not_run",
            "reason": "no actual evidence bundle supplied",
        }
        if evidence_dir is not None:
            evidence_path = evidence_dir / f"{tool['id']}.evidence.json"
            if evidence_path.exists():
                actual = validate_actual_evidence(
                    tool,
                    evidence_path,
                    repository=repository,
                    expected_hbfsim_probe=expected_hbfsim_probe,
                )
                for facet in actual["facets"]:
                    validated_facets.append({
                        "tool_id": tool["id"],
                        "domain": tool["domain"],
                        **facet,
                    })
            else:
                actual["reason"] = (
                    f"missing actual evidence bundle: {evidence_path}")
        entries.append({
            "tool_id": tool["id"],
            "domain": tool["domain"],
            "repository": tool["repository"],
            "commit": tool["commit"],
            "source_tree": tool["source_tree"],
            "adapter": tool["adapter"],
            "declared_facets": tool["facets"],
            "fixture": {
                "status": "pass",
                "case_id": normalized["case_id"],
                "metrics": normalized["metrics"],
            },
            "actual": actual,
        })
    statuses = [entry["actual"]["status"] for entry in entries]
    l3_status = (
        "pass"
        if statuses and all(status == "pass" for status in statuses)
        else "not_run"
        if all(status == "not_run" for status in statuses)
        else "partial"
    )
    return {
        "schema": REPORT_SCHEMA,
        "manifest_digest": _sha256_file(manifest_path),
        "tools": entries,
        "validated_facets": validated_facets,
        "l3_status": l3_status,
        "claim": (
            "L3 pass applies only to the explicitly listed facets, cases, "
            "shared boundaries, and shared metrics. Adapter fixtures alone "
            "never count as external agreement."
        ),
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--hbfsim-probe", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.evidence_dir is not None and args.hbfsim_probe is None:
        parser.error("--hbfsim-probe is required with --evidence-dir")
    try:
        report = build_report(
            args.manifest.resolve(),
            args.fixture_dir.resolve(),
            (
                args.evidence_dir.resolve()
                if args.evidence_dir is not None
                else None
            ),
            repository=(
                args.repository.resolve()
                if args.repository is not None
                else None
            ),
            expected_hbfsim_probe=(
                args.hbfsim_probe.resolve()
                if args.hbfsim_probe is not None
                else None
            ),
        )
        _atomic_write_json(args.report.resolve(), report)
    except (ContractError, OSError, ValueError) as error:
        print(
            f"external differential validation failed: {error}",
            file=sys.stderr,
        )
        return 1
    fixture_count = sum(
        entry["fixture"]["status"] == "pass"
        for entry in report["tools"]
    )
    case_count = sum(
        len(entry["actual"].get("cases", []))
        for entry in report["tools"]
    )
    print(
        "PASS external adapter contracts: "
        f"{fixture_count}/{len(report['tools'])} fixtures; "
        f"actual cases={case_count}; "
        f"L3={report['l3_status']}; "
        f"report={args.report.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
