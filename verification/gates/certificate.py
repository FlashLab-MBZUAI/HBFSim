#!/usr/bin/env python3
"""Run foundational gates and issue a commit- and executable-bound certificate."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verification.core.certificate import (  # noqa: E402
    CertificateError,
    assemble_certificate,
    collection_digest,
    file_digest,
    repository_state,
    write_json_atomic,
)
from verification.core.contracts import ContractError, load_json_strict  # noqa: E402
from evidence.external.verify import (  # noqa: E402
    REPORT_SCHEMA as EXTERNAL_REPORT_SCHEMA,
    load_tools as load_external_tools,
    validate_fixture as validate_external_fixture,
)
from verification.gates.placement import (  # noqa: E402
    COMPARE_FIELDS as BEHAVIORAL_COMPARE_FIELDS,
    SCHEMA as BEHAVIORAL_REPORT_SCHEMA,
)
from verification.core.analytical import (  # noqa: E402
    ABS_TOLERANCE_NS as ANALYTICAL_ABS_TOLERANCE_NS,
    CLAIMS as ANALYTICAL_CLAIMS,
    CONTROLLED_BOUNDARY as ANALYTICAL_CONTROLLED_BOUNDARY,
    HBM_FORMULA as ANALYTICAL_HBM_FORMULA,
    HBM_READ_BYTES as ANALYTICAL_HBM_READ_BYTES,
    HBIO_GBPS_AXIS as ANALYTICAL_HBIO_GBPS_AXIS,
    LARGE_PAGES as ANALYTICAL_LARGE_PAGES,
    LIMITATIONS as ANALYTICAL_LIMITATIONS,
    OVERLAP_BYTES_PER_TIER as ANALYTICAL_OVERLAP_BYTES_PER_TIER,
    OVERLAP_FORMULA as ANALYTICAL_OVERLAP_FORMULA,
    PAGE_BYTES as ANALYTICAL_PAGE_BYTES,
    PHASE_FORMULA as ANALYTICAL_PHASE_FORMULA,
    READ_NS_AXIS as ANALYTICAL_READ_NS_AXIS,
    REL_TOLERANCE as ANALYTICAL_REL_TOLERANCE,
    REPORT_SCHEMA as ANALYTICAL_REPORT_SCHEMA,
    SMALL_PAGES as ANALYTICAL_SMALL_PAGES,
)
from verification.gates.mutation import (  # noqa: E402
    MUTATIONS,
    REPORT_SCHEMA as MUTATION_REPORT_SCHEMA,
)


def _run_gate(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    log_dir: Path,
    timeout: int,
    environment: dict[str, str] | None = None,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    started = time.monotonic()
    process_environment = None
    if environment is not None:
        process_environment = os.environ.copy()
        process_environment.update(environment)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=process_environment,
    )
    duration = time.monotonic() - started
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record = {
        "status": "pass" if completed.returncode == 0 else "fail",
        "exit_code": completed.returncode,
        "duration_seconds": round(duration, 6),
        "command": command,
        "stdout_digest": file_digest(stdout_path),
        "stderr_digest": file_digest(stderr_path),
    }
    if completed.returncode:
        raise CertificateError(
            f"{name} gate failed with exit code {completed.returncode}; "
            f"see {stdout_path} and {stderr_path}"
        )
    return record, completed


def _validate_mutation_report(
    path: Path,
    *,
    commit: str,
    tree: str,
) -> dict[str, Any]:
    report = load_json_strict(path)
    if not isinstance(report, dict):
        raise CertificateError("mutation report must be an object")
    if report.get("schema") != MUTATION_REPORT_SCHEMA:
        raise CertificateError("mutation report schema is unsupported")
    if report.get("source_commit") != commit or report.get("source_tree") != tree:
        raise CertificateError(
            "mutation report is stale for the current source tree")
    entries = report.get("mutations")
    if not isinstance(entries, list):
        raise CertificateError("mutation report entries must be a list")
    expected = {mutation.mutation_id: mutation for mutation in MUTATIONS}
    observed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise CertificateError("malformed mutation report entry")
        mutation_id = entry["id"]
        if mutation_id in observed:
            raise CertificateError(f"duplicate mutation entry: {mutation_id}")
        observed[mutation_id] = entry
    if set(observed) != set(expected):
        raise CertificateError(
            "mutation report does not cover the current mutation set")
    for mutation_id, mutation in expected.items():
        entry = observed[mutation_id]
        if (
            entry.get("patch_digest") != mutation.patch_digest()
            or entry.get("critical") is not mutation.critical
            or entry.get("cases") != list(mutation.cases)
            or entry.get("status") != "killed"
            or entry.get("killed_by") not in mutation.cases
        ):
            raise CertificateError(
                f"mutation report did not validly kill {mutation_id}")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise CertificateError("mutation report summary must be an object")
    total = len(entries)
    critical_total = sum(mutation.critical for mutation in MUTATIONS)
    if (
        summary.get("total") != total
        or summary.get("killed") != total
        or summary.get("survived") != 0
        or summary.get("invalid") != 0
        or summary.get("critical_total") != critical_total
        or summary.get("critical_killed") != critical_total
        or summary.get("passed") is not True
    ):
        raise CertificateError("mutation report summary did not fully pass")
    return {
        "status": "pass",
        "total": total,
        "killed": total,
        "critical_total": critical_total,
        "critical_killed": critical_total,
        "report_digest": file_digest(path),
        "mutations": [
            {
                "id": entry["id"],
                "critical": entry["critical"],
                "patch_digest": entry["patch_digest"],
                "cases": entry["cases"],
                "status": entry["status"],
                "killed_by": entry["killed_by"],
            }
            for entry in entries
        ],
    }


def _exact_report_keys(
    value: dict[str, Any],
    required: set[str],
    where: str,
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required
    if missing:
        raise CertificateError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise CertificateError(f"{where}: unknown keys {sorted(unknown)}")


def _tagged_sha256(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        raise CertificateError(f"{where}: expected tagged SHA-256")
    return value


def _validate_external_report(
    path: Path,
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    report = load_json_strict(path)
    if not isinstance(report, dict):
        raise CertificateError("external report must be an object")
    _exact_report_keys(
        report,
        {
            "schema",
            "manifest_digest",
            "tools",
            "validated_facets",
            "l3_status",
            "claim",
        },
        "external report",
    )
    if report.get("schema") != EXTERNAL_REPORT_SCHEMA:
        raise CertificateError("external report schema is unsupported")
    expected_manifest_digest = (
        "sha256:" + file_digest(manifest_path)["value"])
    if report["manifest_digest"] != expected_manifest_digest:
        raise CertificateError(
            "external report does not bind the current manifest")
    tool_specs = load_external_tools(manifest_path)
    tools = report["tools"]
    if (
        not isinstance(tools, list)
        or [entry.get("tool_id") if isinstance(entry, dict) else None
            for entry in tools]
        != [tool["id"] for tool in tool_specs]
    ):
        raise CertificateError("external report has no tool entries")
    fixture_passed = 0
    actual_statuses: list[str] = []
    tool_records: list[dict[str, Any]] = []
    validated_facets: list[dict[str, Any]] = []
    cases_required = 0
    cases_passed = 0
    for tool, entry in zip(tool_specs, tools):
        if not isinstance(entry, dict):
            raise CertificateError("malformed external report entry")
        where = f"external report tool {tool['id']}"
        _exact_report_keys(
            entry,
            {
                "tool_id",
                "domain",
                "repository",
                "commit",
                "source_tree",
                "adapter",
                "declared_facets",
                "fixture",
                "actual",
            },
            where,
        )
        identity = {
            "tool_id": tool["id"],
            "domain": tool["domain"],
            "repository": tool["repository"],
            "commit": tool["commit"],
            "source_tree": tool["source_tree"],
            "adapter": tool["adapter"],
            "declared_facets": tool["facets"],
        }
        for key, expected in identity.items():
            if entry[key] != expected:
                raise CertificateError(
                    f"{where}.{key} differs from the current manifest")
        fixture = entry.get("fixture")
        actual = entry.get("actual")
        if not isinstance(fixture, dict):
            raise CertificateError("external adapter fixture is malformed")
        _exact_report_keys(
            fixture, {"status", "case_id", "metrics"},
            f"{where}.fixture",
        )
        normalized_fixture = validate_external_fixture(
            tool, manifest_path.parent / "fixtures")
        if fixture != {
            "status": "pass",
            "case_id": normalized_fixture["case_id"],
            "metrics": normalized_fixture["metrics"],
        }:
            raise CertificateError(
                "external adapter fixture differs from the current "
                "adapter contract")
        if not isinstance(actual, dict):
            raise CertificateError("external actual evidence is malformed")
        status = actual.get("status")
        if status not in {"pass", "not_run"}:
            raise CertificateError(
                "external actual evidence must pass or explicitly be not_run")
        declared_facets = tool["facets"]
        cases_required += sum(
            len(facet["cases"]) for facet in declared_facets)
        recorded_cases: list[dict[str, Any]] = []
        recorded_facets: list[dict[str, Any]] = []
        if status == "not_run":
            _exact_report_keys(
                actual, {"status", "reason"}, f"{where}.actual")
            if not isinstance(actual["reason"], str) or not actual["reason"]:
                raise CertificateError(
                    "external not_run evidence requires a reason")
            recorded_facets = [
                {**facet, "status": "not_run"}
                for facet in declared_facets
            ]
        else:
            _exact_report_keys(
                actual,
                {
                    "status",
                    "source_digest",
                    "hbfsim_probe_digest",
                    "build_artifact_digests",
                    "cases",
                    "facets",
                },
                f"{where}.actual",
            )
            _tagged_sha256(
                actual["source_digest"], f"{where}.actual.source_digest")
            _tagged_sha256(
                actual["hbfsim_probe_digest"],
                f"{where}.actual.hbfsim_probe_digest",
            )
            build_digests = actual["build_artifact_digests"]
            if (
                not isinstance(build_digests, dict)
                or set(build_digests)
                != set(tool["required_build_artifacts"])
            ):
                raise CertificateError(
                    f"{where}.actual build-artifact census differs from "
                    "the manifest")
            for name, digest in build_digests.items():
                _tagged_sha256(
                    digest, f"{where}.actual.build_artifact_digests.{name}")

            expected_case_ids = [
                case["case_id"] for case in tool["cases"]]
            cases = actual["cases"]
            if (
                not isinstance(cases, list)
                or [
                    case.get("case_id") if isinstance(case, dict) else None
                    for case in cases
                ] != expected_case_ids
            ):
                raise CertificateError(
                    f"{where}.actual case census/order differs from "
                    "the manifest")
            for case_spec, case in zip(tool["cases"], cases):
                case_where = (
                    f"{where}.actual.cases[{case_spec['case_id']}]")
                if not isinstance(case, dict):
                    raise CertificateError(
                        f"{case_where}: expected object")
                _exact_report_keys(
                    case,
                    {
                        "case_id",
                        "facet_id",
                        "status",
                        "metrics",
                        "artifact_digests",
                    },
                    case_where,
                )
                if (
                    case["case_id"] != case_spec["case_id"]
                    or case["facet_id"] != case_spec["facet_id"]
                    or case["status"] != "pass"
                ):
                    raise CertificateError(
                        f"{case_where}: identity or status mismatch")
                metrics = case["metrics"]
                if (
                    not isinstance(metrics, dict)
                    or set(metrics) != set(case_spec["shared_metrics"])
                ):
                    raise CertificateError(
                        f"{case_where}.metrics differ from shared_metrics")
                for metric_name, metric_value in metrics.items():
                    if (
                        isinstance(metric_value, bool)
                        or not isinstance(metric_value, (int, float))
                        or not math.isfinite(float(metric_value))
                    ):
                        raise CertificateError(
                            f"{case_where}.metrics.{metric_name}: "
                            "expected finite number")
                artifact_digests = case["artifact_digests"]
                if (
                    not isinstance(artifact_digests, dict)
                    or set(artifact_digests)
                    != set(tool["required_case_artifacts"])
                ):
                    raise CertificateError(
                        f"{case_where}: artifact census differs from "
                        "the manifest")
                for name, digest in artifact_digests.items():
                    _tagged_sha256(
                        digest,
                        f"{case_where}.artifact_digests.{name}",
                    )
                recorded_cases.append({
                    "case_id": case["case_id"],
                    "facet_id": case["facet_id"],
                    "status": "pass",
                    "metrics": case["metrics"],
                })

            expected_facets = [
                {**facet, "status": "pass"}
                for facet in declared_facets
            ]
            if actual["facets"] != expected_facets:
                raise CertificateError(
                    f"{where}.actual facets differ from the manifest")
            recorded_facets = expected_facets
            validated_facets.extend({
                "tool_id": tool["id"],
                "domain": tool["domain"],
                **facet,
            } for facet in expected_facets)
            cases_passed += len(recorded_cases)
        fixture_passed += 1
        actual_statuses.append(status)
        tool_records.append({
            "id": tool["id"],
            "domain": tool["domain"],
            "commit": tool["commit"],
            "source_tree": tool["source_tree"],
            "fixture_status": fixture["status"],
            "actual_status": status,
            "facets": recorded_facets,
            "cases": recorded_cases,
        })
    expected_l3 = (
        "pass"
        if all(status == "pass" for status in actual_statuses)
        else "not_run"
        if all(status == "not_run" for status in actual_statuses)
        else "partial"
    )
    if report.get("l3_status") != expected_l3:
        raise CertificateError("external report has inconsistent L3 status")
    if report["validated_facets"] != validated_facets:
        raise CertificateError(
            "external report validated-facet census is inconsistent")
    if not isinstance(report["claim"], str) or not report["claim"]:
        raise CertificateError("external report claim is missing")
    return {
        "status": "pass",
        "fixtures_passed": fixture_passed,
        "fixtures_total": len(tools),
        "cases_required": cases_required,
        "cases_passed": cases_passed,
        "l3_status": expected_l3,
        "manifest_digest": file_digest(manifest_path),
        "report_digest": file_digest(path),
        "validated_facets": validated_facets,
        "tools": tool_records,
    }


def _validate_behavioral_report(path: Path) -> dict[str, Any]:
    report = load_json_strict(path)
    if not isinstance(report, dict):
        raise CertificateError("behavioral differential report must be an object")
    if report.get("schema") != BEHAVIORAL_REPORT_SCHEMA:
        raise CertificateError(
            "behavioral differential report schema is unsupported")
    if report.get("status") != "PASS":
        raise CertificateError("behavioral differential did not pass")
    if (
        report.get("seeds") != 32
        or report.get("operations_per_seed") != 48
        or report.get("policies")
        != ["always-admit", "reuse-filtered"]
        or report.get("case_count") != 32
        or report.get("production_executions") != 64
        or report.get("compared_fields") != list(BEHAVIORAL_COMPARE_FIELDS)
    ):
        raise CertificateError(
            "behavioral differential report corpus is incomplete")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != 32:
        raise CertificateError(
            "behavioral differential report cases are incomplete")
    observed_seeds = {
        case.get("seed")
        for case in cases
        if isinstance(case, dict)
    }
    if observed_seeds != set(range(32)):
        raise CertificateError(
            "behavioral differential report seed census is incomplete")
    return {
        "status": "pass",
        "seeds": 32,
        "operations_per_seed": 48,
        "policies": report["policies"],
        "case_count": 32,
        "production_executions": 64,
        "compared_fields": report["compared_fields"],
        "report_digest": file_digest(path),
    }


def _resolve_analytical_artifact(
    artifact_dir: Path,
    value: Any,
    where: str,
) -> Path:
    if not isinstance(value, dict):
        raise CertificateError(f"{where}: expected artifact object")
    _exact_report_keys(
        value,
        {"path", "algorithm", "value", "bytes"},
        where,
    )
    raw_path = value["path"]
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or Path(raw_path).is_absolute()
        or ".." in Path(raw_path).parts
    ):
        raise CertificateError(f"{where}.path: invalid relative path")
    if (
        value["algorithm"] != "sha256"
        or not isinstance(value["value"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["value"]) is None
        or type(value["bytes"]) is not int
        or value["bytes"] < 0
    ):
        raise CertificateError(f"{where}: invalid digest record")
    path = (artifact_dir / raw_path).resolve()
    try:
        path.relative_to(artifact_dir.resolve())
    except ValueError as error:
        raise CertificateError(f"{where}.path escapes artifact dir") from error
    if not path.is_file() or file_digest(path) != {
        "algorithm": value["algorithm"],
        "value": value["value"],
        "bytes": value["bytes"],
    }:
        raise CertificateError(f"{where}: artifact digest mismatch")
    return path


def _analytical_number(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CertificateError(f"{where}: expected finite number")
    return float(value)


def _validate_analytical_command(value: Any, where: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise CertificateError(f"{where}: expected nonempty argv strings")


def _analytical_close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=ANALYTICAL_REL_TOLERANCE,
        abs_tol=ANALYTICAL_ABS_TOLERANCE_NS,
    )


def _analytical_bottleneck(read_ns: float, hbio_gbps: float) -> str:
    media = read_ns
    hbio = ANALYTICAL_PAGE_BYTES / hbio_gbps
    if _analytical_close(media, hbio):
        return "co-bottleneck"
    return "media" if media > hbio else "hbio-data"


def _validate_analytical_report(
    path: Path,
    *,
    artifact_dir: Path,
    source: dict[str, Any],
    simulator: Path,
    config: Path,
    repository: Path,
) -> dict[str, Any]:
    report = load_json_strict(path)
    if not isinstance(report, dict):
        raise CertificateError("analytical report must be an object")
    _exact_report_keys(
        report,
        {
            "schema",
            "status",
            "source",
            "simulator",
            "baseline_config",
            "tolerances",
            "hbm_data_bus_work",
            "hbf_media_hbio_phase_diagram",
            "independent_tier_overlap",
            "artifacts",
            "claims",
            "limitations",
        },
        "analytical report",
    )
    if report["schema"] != ANALYTICAL_REPORT_SCHEMA:
        raise CertificateError("analytical report schema is unsupported")
    if report["status"] != "pass":
        raise CertificateError("analytical report did not pass")
    report_source = report["source"]
    if not isinstance(report_source, dict):
        raise CertificateError("analytical report source is malformed")
    _exact_report_keys(
        report_source,
        {
            "commit",
            "tree",
            "tracked_clean",
            "untracked_files_observed",
            "untracked_files_affect_certificate",
        },
        "analytical report source",
    )
    if (
        report_source.get("commit") != source["commit"]
        or report_source.get("tree") != source["tree"]
        or report_source.get("tracked_clean") is not True
        or type(report_source.get("untracked_files_observed")) is not int
        or report_source["untracked_files_observed"] < 0
        or report_source.get("untracked_files_affect_certificate") is not False
    ):
        raise CertificateError(
            "analytical report source differs from the certificate tree")
    if report["simulator"] != file_digest(simulator):
        raise CertificateError(
            "analytical report simulator digest is stale")
    expected_config = {
        "path": config.resolve().relative_to(repository.resolve()).as_posix(),
        **file_digest(config),
    }
    if report["baseline_config"] != expected_config:
        raise CertificateError(
            "analytical report baseline config digest is stale")
    if report["tolerances"] != {
        "absolute_ns": ANALYTICAL_ABS_TOLERANCE_NS,
        "relative": ANALYTICAL_REL_TOLERANCE,
    }:
        raise CertificateError("analytical report tolerances changed")

    hbm = report["hbm_data_bus_work"]
    if not isinstance(hbm, dict):
        raise CertificateError("analytical HBM result is malformed")
    _exact_report_keys(
        hbm,
        {
            "case_id",
            "status",
            "formula",
            "read_bytes",
            "pseudo_channel_bandwidth_GBps",
            "resource_count",
            "predicted_busy_ns",
            "observed_busy_ns",
            "predicted_max_per_resource_busy_ns",
            "observed_max_per_resource_busy_ns",
            "command",
            "summary",
        },
        "analytical HBM result",
    )
    if (
        hbm["case_id"] != "analytical.hbm-data-bus-work"
        or hbm["status"] != "pass"
        or hbm["formula"] != ANALYTICAL_HBM_FORMULA
        or type(hbm["read_bytes"]) is not int
        or hbm["read_bytes"] != ANALYTICAL_HBM_READ_BYTES
        or type(hbm["resource_count"]) is not int
        or hbm["resource_count"] <= 0
    ):
        raise CertificateError("analytical HBM result is incomplete")
    _validate_analytical_command(
        hbm["command"], "analytical HBM command")
    bandwidth = _analytical_number(
        hbm["pseudo_channel_bandwidth_GBps"],
        "analytical HBM bandwidth",
    )
    if bandwidth <= 0:
        raise CertificateError("analytical HBM bandwidth must be positive")
    predicted_busy = hbm["read_bytes"] / bandwidth
    predicted_per_resource = predicted_busy / hbm["resource_count"]
    for observed, expected, where in (
        (hbm["predicted_busy_ns"], predicted_busy, "HBM predicted busy"),
        (hbm["observed_busy_ns"], predicted_busy, "HBM observed busy"),
        (
            hbm["predicted_max_per_resource_busy_ns"],
            predicted_per_resource,
            "HBM predicted max busy",
        ),
        (
            hbm["observed_max_per_resource_busy_ns"],
            predicted_per_resource,
            "HBM observed max busy",
        ),
    ):
        if not _analytical_close(
            _analytical_number(observed, where), expected
        ):
            raise CertificateError(f"{where} violates the closed form")
    _resolve_analytical_artifact(
        artifact_dir, hbm["summary"], "analytical HBM summary")

    phase = report["hbf_media_hbio_phase_diagram"]
    if not isinstance(phase, dict):
        raise CertificateError("analytical phase diagram is malformed")
    _exact_report_keys(
        phase,
        {
            "status",
            "formula",
            "controlled_boundary",
            "axes",
            "point_count",
            "region_counts",
            "max_absolute_error_ns",
            "max_relative_error",
            "points",
        },
        "analytical phase diagram",
    )
    expected_point_count = (
        len(ANALYTICAL_READ_NS_AXIS) * len(ANALYTICAL_HBIO_GBPS_AXIS))
    if (
        phase["status"] != "pass"
        or phase["formula"] != ANALYTICAL_PHASE_FORMULA
        or phase["controlled_boundary"] != ANALYTICAL_CONTROLLED_BOUNDARY
        or phase["axes"] != {
            "read_ns": list(ANALYTICAL_READ_NS_AXIS),
            "hbio_GBps": list(ANALYTICAL_HBIO_GBPS_AXIS),
        }
        or phase["point_count"] != expected_point_count
        or not isinstance(phase["points"], list)
        or len(phase["points"]) != expected_point_count
    ):
        raise CertificateError("analytical phase-grid census is incomplete")
    expected_coordinates = [
        (read_ns, hbio)
        for read_ns in ANALYTICAL_READ_NS_AXIS
        for hbio in ANALYTICAL_HBIO_GBPS_AXIS
    ]
    region_counts = {
        "media": 0,
        "hbio-data": 0,
        "co-bottleneck": 0,
    }
    reduced_points: list[dict[str, Any]] = []
    max_absolute_error = 0.0
    max_relative_error = 0.0
    for index, (point, coordinates) in enumerate(zip(
        phase["points"], expected_coordinates
    )):
        where = f"analytical phase point {index}"
        if not isinstance(point, dict):
            raise CertificateError(f"{where}: expected object")
        _exact_report_keys(
            point,
            {
                "read_ns",
                "hbio_GBps",
                "media_ii_ns",
                "hbio_data_ii_ns",
                "predicted_steady_state_ii_ns",
                "observed_steady_state_ii_ns",
                "absolute_error_ns",
                "relative_error",
                "predicted_bottleneck",
                "observed_bottleneck",
                "large_run_media_global_occupancy",
                "large_run_hbio_data_global_occupancy",
                "status",
                "runs",
            },
            where,
        )
        read_ns = _analytical_number(point["read_ns"], f"{where}.read_ns")
        hbio = _analytical_number(point["hbio_GBps"], f"{where}.hbio")
        if (read_ns, hbio) != coordinates:
            raise CertificateError(f"{where}: coordinate/order mismatch")
        media_ii = read_ns
        hbio_ii = ANALYTICAL_PAGE_BYTES / hbio
        predicted_ii = max(media_ii, hbio_ii)
        predicted_bottleneck = _analytical_bottleneck(read_ns, hbio)
        observed_ii = _analytical_number(
            point["observed_steady_state_ii_ns"],
            f"{where}.observed II",
        )
        media_global_occupancy = _analytical_number(
            point["large_run_media_global_occupancy"],
            f"{where}.media global occupancy",
        )
        hbio_global_occupancy = _analytical_number(
            point["large_run_hbio_data_global_occupancy"],
            f"{where}.HBIO global occupancy",
        )
        observed_from_occupancy = (
            "co-bottleneck"
            if _analytical_close(
                media_global_occupancy, hbio_global_occupancy)
            else "media"
            if media_global_occupancy > hbio_global_occupancy
            else "hbio-data"
        )
        if (
            point["status"] != "pass"
            or point["predicted_bottleneck"] != predicted_bottleneck
            or point["observed_bottleneck"] != predicted_bottleneck
            or observed_from_occupancy != predicted_bottleneck
            or not 0.0 <= media_global_occupancy <= 1.0
            or not 0.0 <= hbio_global_occupancy <= 1.0
        ):
            raise CertificateError(f"{where}: bottleneck result mismatch")
        for value, expected, label in (
            (point["media_ii_ns"], media_ii, "media II"),
            (point["hbio_data_ii_ns"], hbio_ii, "HBIO II"),
            (
                point["predicted_steady_state_ii_ns"],
                predicted_ii,
                "predicted II",
            ),
            (observed_ii, predicted_ii, "observed II"),
        ):
            if not _analytical_close(
                _analytical_number(value, f"{where}.{label}"), expected
            ):
                raise CertificateError(
                    f"{where}: {label} violates the closed form")
        runs = point["runs"]
        if not isinstance(runs, dict) or set(runs) != {
            str(ANALYTICAL_SMALL_PAGES),
            str(ANALYTICAL_LARGE_PAGES),
        }:
            raise CertificateError(f"{where}: run census is incomplete")
        makespans: dict[int, float] = {}
        for pages in (
            ANALYTICAL_SMALL_PAGES,
            ANALYTICAL_LARGE_PAGES,
        ):
            run = runs[str(pages)]
            if (
                not isinstance(run, dict)
                or set(run) != {
                    "pages", "makespan_ns", "command", "summary"
                }
                or run["pages"] != pages
            ):
                raise CertificateError(f"{where}: malformed n={pages} run")
            _validate_analytical_command(
                run["command"], f"{where}.n{pages}.command")
            makespans[pages] = _analytical_number(
                run["makespan_ns"], f"{where}.n{pages}.makespan")
            _resolve_analytical_artifact(
                artifact_dir,
                run["summary"],
                f"{where}.n{pages}.summary",
            )
        derived_ii = (
            makespans[ANALYTICAL_LARGE_PAGES]
            - makespans[ANALYTICAL_SMALL_PAGES]
        ) / (ANALYTICAL_LARGE_PAGES - ANALYTICAL_SMALL_PAGES)
        if not _analytical_close(derived_ii, observed_ii):
            raise CertificateError(
                f"{where}: run pair does not derive reported II")
        absolute_error = abs(observed_ii - predicted_ii)
        relative_error = absolute_error / predicted_ii
        if (
            not _analytical_close(
                _analytical_number(
                    point["absolute_error_ns"],
                    f"{where}.absolute error",
                ),
                absolute_error,
            )
            or not _analytical_close(
                _analytical_number(
                    point["relative_error"],
                    f"{where}.relative error",
                ),
                relative_error,
            )
        ):
            raise CertificateError(f"{where}: error fields are inconsistent")
        max_absolute_error = max(max_absolute_error, absolute_error)
        max_relative_error = max(max_relative_error, relative_error)
        region_counts[predicted_bottleneck] += 1
        reduced_points.append({
            "read_ns": read_ns,
            "hbio_GBps": hbio,
            "predicted_steady_state_ii_ns": predicted_ii,
            "observed_steady_state_ii_ns": observed_ii,
            "bottleneck": predicted_bottleneck,
            "status": "pass",
        })
    if phase["region_counts"] != region_counts:
        raise CertificateError("analytical phase region census is inconsistent")
    if (
        not _analytical_close(
            _analytical_number(
                phase["max_absolute_error_ns"],
                "analytical max absolute error",
            ),
            max_absolute_error,
        )
        or not _analytical_close(
            _analytical_number(
                phase["max_relative_error"],
                "analytical max relative error",
            ),
            max_relative_error,
        )
    ):
        raise CertificateError("analytical phase max error is inconsistent")

    overlap = report["independent_tier_overlap"]
    if not isinstance(overlap, dict):
        raise CertificateError("analytical overlap result is malformed")
    _exact_report_keys(
        overlap,
        {
            "case_id",
            "status",
            "formula",
            "bytes_per_tier",
            "hbm_only_makespan_ns",
            "hbf_only_makespan_ns",
            "predicted_mixed_makespan_ns",
            "observed_mixed_makespan_ns",
            "serialized_sum_ns",
            "hbm_resource_busy_preserved",
            "hbf_resource_busy_preserved",
            "runs",
        },
        "analytical overlap result",
    )
    hbm_ns = _analytical_number(
        overlap["hbm_only_makespan_ns"], "overlap HBM makespan")
    hbf_ns = _analytical_number(
        overlap["hbf_only_makespan_ns"], "overlap HBF makespan")
    observed_mixed = _analytical_number(
        overlap["observed_mixed_makespan_ns"], "overlap mixed makespan")
    predicted_mixed = max(hbm_ns, hbf_ns)
    if (
        overlap["case_id"] != "analytical.independent-tier-overlap"
        or overlap["status"] != "pass"
        or overlap["formula"] != ANALYTICAL_OVERLAP_FORMULA
        or overlap["bytes_per_tier"]
        != ANALYTICAL_OVERLAP_BYTES_PER_TIER
        or overlap["hbm_resource_busy_preserved"] is not True
        or overlap["hbf_resource_busy_preserved"] is not True
        or not _analytical_close(
            _analytical_number(
                overlap["predicted_mixed_makespan_ns"],
                "overlap predicted mixed",
            ),
            predicted_mixed,
        )
        or not _analytical_close(observed_mixed, predicted_mixed)
        or not _analytical_close(
            _analytical_number(
                overlap["serialized_sum_ns"], "overlap serialized sum"),
            hbm_ns + hbf_ns,
        )
        or not observed_mixed < hbm_ns + hbf_ns
    ):
        raise CertificateError(
            "analytical independent-tier overlap identity failed")
    runs = overlap["runs"]
    if (
        not isinstance(runs, dict)
        or set(runs) != {"hbm-only", "hbf-only", "mixed"}
    ):
        raise CertificateError("analytical overlap run census is incomplete")
    run_makespans: dict[str, float] = {}
    for run_id, run in runs.items():
        if (
            not isinstance(run, dict)
            or set(run) != {"makespan_ns", "command", "summary"}
        ):
            raise CertificateError(
                f"analytical overlap {run_id} is malformed")
        _validate_analytical_command(
            run["command"], f"analytical overlap {run_id} command")
        run_makespans[run_id] = _analytical_number(
            run["makespan_ns"],
            f"analytical overlap {run_id} makespan",
        )
        _resolve_analytical_artifact(
            artifact_dir,
            run["summary"],
            f"analytical overlap {run_id} summary",
        )
    if (
        not _analytical_close(run_makespans["hbm-only"], hbm_ns)
        or not _analytical_close(run_makespans["hbf-only"], hbf_ns)
        or not _analytical_close(run_makespans["mixed"], observed_mixed)
    ):
        raise CertificateError(
            "analytical overlap run makespans disagree with the result")

    artifacts = report["artifacts"]
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != {"phase_csv", "phase_markdown"}
    ):
        raise CertificateError("analytical report artifact census changed")
    for name, artifact in artifacts.items():
        _resolve_analytical_artifact(
            artifact_dir, artifact, f"analytical artifact {name}")
    claims = report["claims"]
    limitations = report["limitations"]
    if claims != list(ANALYTICAL_CLAIMS) or limitations != list(
        ANALYTICAL_LIMITATIONS
    ):
        raise CertificateError(
            "analytical claims/limitations census changed")

    return {
        "status": "pass",
        "report_digest": file_digest(path),
        "hbm": {
            "status": "pass",
            "read_bytes": hbm["read_bytes"],
            "predicted_busy_ns": predicted_busy,
            "observed_busy_ns": _analytical_number(
                hbm["observed_busy_ns"], "HBM observed busy"),
        },
        "phase_diagram": {
            "status": "pass",
            "point_count": expected_point_count,
            "region_counts": region_counts,
            "max_absolute_error_ns": max_absolute_error,
            "max_relative_error": max_relative_error,
            "points": reduced_points,
        },
        "independent_tier_overlap": {
            "status": "pass",
            "hbm_only_makespan_ns": hbm_ns,
            "hbf_only_makespan_ns": hbf_ns,
            "predicted_mixed_makespan_ns": predicted_mixed,
            "observed_mixed_makespan_ns": observed_mixed,
            "serialized_sum_ns": hbm_ns + hbf_ns,
            "hbm_resource_busy_preserved": True,
            "hbf_resource_busy_preserved": True,
        },
        "claims": claims,
        "limitations": limitations,
    }


def _validate_build_summary(
    path: Path,
    *,
    source: dict[str, Any],
    simulator: Path,
) -> dict[str, Any]:
    summary = load_json_strict(path)
    if not isinstance(summary, dict):
        raise CertificateError("build-provenance summary must be an object")
    if summary.get("sanity") != "PASS":
        raise CertificateError("build-provenance scenario did not pass")
    simulator = summary.get("simulator")
    build = summary.get("build")
    validation = summary.get("validation")
    if (
        not isinstance(simulator, dict)
        or simulator.get("git_commit") != source["commit"]
        or simulator.get("git_dirty") is not False
    ):
        raise CertificateError(
            "simulator was not configured from the clean current commit")
    expected_digest = file_digest(simulator)["value"]
    if (
        not isinstance(build, dict)
        or not isinstance(build.get("executable_digest"), dict)
        or build["executable_digest"].get("algorithm") != "sha256"
        or build["executable_digest"].get("value") != expected_digest
    ):
        raise CertificateError(
            "simulator self-reported executable digest is stale")
    if validation != {
        "status": "exploratory_unattached",
        "certificate": None,
    }:
        raise CertificateError(
            "direct scenario output is not explicitly exploratory")
    return {
        "summary_digest": file_digest(path),
        "simulator_version": simulator.get("version"),
        "build_type": build.get("type"),
        "compiler_id": build.get("compiler_id"),
        "compiler_version": build.get("compiler_version"),
    }


def _parse_ctest_output(output: str, gate: str) -> tuple[int, int]:
    match = re.search(
        r"(\d+)% tests passed, (\d+) tests failed out of (\d+)",
        output,
    )
    if match is None or int(match.group(2)) != 0:
        raise CertificateError(f"could not verify complete {gate} result")
    return int(match.group(3)), int(match.group(2))


def _count_passing_rows(path: Path, label: str) -> int:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get("verdict") != "PASS" for row in rows):
        raise CertificateError(f"{label} table is empty or contains failures")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--physical-probe", type=Path, required=True)
    parser.add_argument("--mutation-report", type=Path, required=True)
    parser.add_argument("--external-evidence-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()

    repository = args.repository.resolve()
    build_dir = args.build_dir.resolve()
    simulator = args.simulator.resolve()
    probe = args.probe.resolve()
    physical_probe = args.physical_probe.resolve()
    artifact_dir = args.artifact_dir.resolve()
    log_dir = artifact_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = repository_state(repository, require_clean=True)
        for path, label in (
            (simulator, "simulator"),
            (probe, "ledger_probe"),
            (physical_probe, "physical_probe"),
            (args.mutation_report.resolve(), "mutation report"),
        ):
            if not path.is_file():
                raise CertificateError(f"{label} not found: {path}")

        gates: dict[str, Any] = {}
        provenance_summary = artifact_dir / "build-provenance.summary.json"
        record, _ = _run_gate(
            "build-provenance",
            [
                str(simulator),
                "--synthetic-sequential-read-bytes", "4096",
                "--line-size", "4096",
                "--scenarios", "all-hbm",
                "--summary-json", str(provenance_summary),
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        provenance = _validate_build_summary(
            provenance_summary,
            source=source,
            simulator=simulator,
        )
        record.update(provenance)
        gates["build_provenance"] = record

        record, completed = _run_gate(
            "regression-ctest",
            [
                "ctest",
                "--test-dir", str(build_dir),
                "--output-on-failure",
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        tests_passed, tests_failed = _parse_ctest_output(
            completed.stdout, "CTest")
        record.update({
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
        })
        gates["regression_ctest"] = record

        sanitizer_build = artifact_dir / "sanitize-build"
        sanitizer_flags = (
            "-fsanitize=address,undefined -fno-omit-frame-pointer")
        configure_record, _ = _run_gate(
            "sanitizer-configure",
            [
                "cmake",
                "-S", str(repository),
                "-B", str(sanitizer_build),
                "-DCMAKE_BUILD_TYPE=Debug",
                f"-DCMAKE_CXX_FLAGS={sanitizer_flags}",
                "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address,undefined",
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        build_record, _ = _run_gate(
            "sanitizer-build",
            [
                "cmake",
                "--build", str(sanitizer_build),
                "--parallel", str(args.parallel),
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        sanitizer_platform = {
            "system": platform.system(),
            "machine": platform.machine(),
        }
        leak_detection = (
            "unsupported_on_darwin"
            if sanitizer_platform["system"] == "Darwin"
            else "enabled"
        )
        sanitizer_environment = {
            "ASAN_OPTIONS": (
                "detect_leaks=0:halt_on_error=1"
                if leak_detection == "unsupported_on_darwin"
                else "detect_leaks=1:halt_on_error=1"
            ),
            "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
        }
        sanitizer_record, completed = _run_gate(
            "sanitizer-ctest",
            [
                "ctest",
                "--test-dir", str(sanitizer_build),
                "--output-on-failure",
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
            environment=sanitizer_environment,
        )
        tests_passed, tests_failed = _parse_ctest_output(
            completed.stdout, "sanitizer CTest")
        sanitizer_record.update({
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "configure": configure_record,
            "build": build_record,
            "sanitizers": ["address", "undefined"],
            "platform": sanitizer_platform,
            "runtime_environment": sanitizer_environment,
            "leak_detection": leak_detection,
        })
        gates["sanitizer_ctest"] = sanitizer_record

        canonical_dir = artifact_dir / "canonical"
        record, _ = _run_gate(
            "canonical-oracle",
            [
                sys.executable, "-B",
                str(repository / "verification/gates/cases.py"),
                "--probe", str(probe),
                "--case-dir", str(repository / "verification/cases"),
                "--artifact-dir", str(canonical_dir),
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        actual_ledgers = sorted(canonical_dir.glob("*.actual.jsonl"))
        case_count = len(list(
            (repository / "verification/cases").glob("*.json")))
        if len(actual_ledgers) != case_count:
            raise CertificateError(
                "canonical oracle did not emit one actual ledger per case")
        record.update({
            "case_count": case_count,
            "actual_ledgers": collection_digest(
                canonical_dir,
                [path.relative_to(canonical_dir) for path in actual_ledgers],
            ),
            "cases": [
                {
                    "case_id": path.name.removesuffix(".actual.jsonl"),
                    "status": "pass",
                    "actual_ledger": file_digest(path),
                    "expected_ledger": file_digest(
                        path.with_name(
                            path.name.replace(
                                ".actual.jsonl", ".expected.jsonl"))),
                }
                for path in actual_ledgers
            ],
        })
        gates["canonical_oracle"] = record

        record, completed = _run_gate(
            "property-fuzz",
            [
                sys.executable, "-B",
                str(repository / "verification/gates/fuzz.py"),
                "--probe", str(probe),
                "--seed-start", "0",
                "--seeds", "32",
                "--max-requests", "50",
                "--artifact-dir", str(artifact_dir / "property-fuzz"),
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        match = re.search(
            r"(\d+) generated case\(s\), "
            r"(\d+) production execution\(s\)",
            completed.stdout,
        )
        if match is None:
            raise CertificateError("property-fuzz result was not parseable")
        record.update({
            "models": ["external", "hbf", "hbm", "hybrid"],
            "seed_start": 0,
            "seeds_per_model": 32,
            "max_requests": 50,
            "generated_cases": int(match.group(1)),
            "production_executions": int(match.group(2)),
            "metamorphic_properties": [
                "time_translation",
                "request_id_renaming",
            ],
        })
        gates["property_fuzz"] = record

        behavioral_report = (
            artifact_dir / "behavioral-differential-report.json"
        )
        _run_gate(
            "behavioral-differential",
            [
                sys.executable,
                "-B",
                str(
                    repository
                    / "verification/gates/placement.py"
                ),
                "--simulator",
                str(simulator),
                "--config",
                str(
                    repository
                    / "configs/systems/server-hbm128-hbf512.cfg"
                ),
                "--seeds",
                "32",
                "--operations",
                "48",
                "--report",
                str(behavioral_report),
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        gates["behavioral_differential"] = (
            _validate_behavioral_report(behavioral_report)
        )

        analytical_dir = artifact_dir / "analytical-microbench"
        analytical_report = (
            artifact_dir / "analytical-microbench-report.json"
        )
        _run_gate(
            "analytical-microbench",
            [
                sys.executable,
                "-B",
                str(
                    repository
                    / "verification/gates/analytical.py"
                ),
                "--simulator",
                str(simulator),
                "--config",
                str(
                    repository
                    / "configs/systems/eight-stack-baseline.cfg"
                ),
                "--repository",
                str(repository),
                "--artifact-dir",
                str(analytical_dir),
                "--report",
                str(analytical_report),
                "--timeout",
                str(min(args.timeout, 600)),
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        gates["analytical_microbench"] = _validate_analytical_report(
            analytical_report,
            artifact_dir=analytical_dir,
            source=source,
            simulator=simulator,
            config=(
                repository
                / "configs/systems/eight-stack-baseline.cfg"
            ),
            repository=repository,
        )

        physical_dir = artifact_dir / "physical-guards"
        record, _ = _run_gate(
            "physical-guards",
            [
                "/bin/bash",
                str(repository / "verification/gates/physical.sh"),
                str(physical_probe),
                str(physical_dir),
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        guard_files = sorted(physical_dir.glob("*.txt"))
        if not guard_files:
            raise CertificateError("physical guard emitted no check artifacts")
        record["checks_passed"] = len(guard_files)
        gates["physical_checks"] = record

        component_dir = artifact_dir / "component-checks"
        record, _ = _run_gate(
            "component-checks",
            [
                sys.executable, "-B",
                str(repository / "verification/gates/components.py"),
                "--simulator", str(simulator),
                "--out-dir", str(component_dir),
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        record["checks_passed"] = _count_passing_rows(
            component_dir / "component-checks.csv", "component-check")
        gates["component_checks"] = record

        amplification_dir = artifact_dir / "write-amplification"
        record, _ = _run_gate(
            "write-amplification",
            [
                sys.executable, "-B",
                str(repository / "verification/gates/waf.py"),
                "--simulator", str(simulator),
                "--out-dir", str(amplification_dir),
            ],
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        record["checks_passed"] = _count_passing_rows(
            amplification_dir / "write-amplification-verification.csv",
            "write-amplification",
        )
        gates["write_amplification"] = record

        external_report = artifact_dir / "external-report.json"
        external_command = [
            sys.executable, "-B",
            str(repository / "evidence/external/verify.py"),
            "--manifest",
            str(repository / "evidence/external/assets/tools.json"),
            "--fixture-dir",
            str(repository / "evidence/external/assets/fixtures"),
            "--repository",
            str(repository),
            "--hbfsim-probe",
            str(probe),
            "--report", str(external_report),
        ]
        if args.external_evidence_dir is not None:
            external_command.extend([
                "--evidence-dir",
                str(args.external_evidence_dir.resolve()),
            ])
        _run_gate(
            "external-differential",
            external_command,
            cwd=repository,
            log_dir=log_dir,
            timeout=args.timeout,
        )
        gates["external_differential"] = _validate_external_report(
            external_report,
            manifest_path=repository / "evidence/external/assets/tools.json",
        )
        gates["mutation"] = _validate_mutation_report(
            args.mutation_report.resolve(),
            commit=source["commit"],
            tree=source["tree"],
        )

        issued_at = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        certificate = assemble_certificate(
            repository=repository,
            simulator=simulator,
            ledger_probe=probe,
            physical_probe=physical_probe,
            issued_at_utc=issued_at,
            gates=gates,
        )
        write_json_atomic(args.output.resolve(), certificate)
    except (
        CertificateError,
        ContractError,
        OSError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        print(f"foundational certificate failed: {error}", file=sys.stderr)
        return 1

    print(
        "PASS foundational certificate: "
        f"L0-L2=pass, "
        f"L3={certificate['confidence']['l3_external_reference']}, "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
