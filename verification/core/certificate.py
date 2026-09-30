"""Strict foundational-certificate and summary-attachment contracts.

The certificate is evidence that one committed source tree and one exact
set of publication executables passed the repository's foundational gates.
It is deliberately not a signature and does not claim agreement with hardware.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from verification.core.analytical import (
    ABS_TOLERANCE_NS as ANALYTICAL_ABS_TOLERANCE_NS,
    CERTIFICATE_CLAIM as ANALYTICAL_CERTIFICATE_CLAIM,
    CLAIMS as ANALYTICAL_CLAIMS,
    HBM_READ_BYTES as ANALYTICAL_HBM_READ_BYTES,
    HBIO_GBPS_AXIS as ANALYTICAL_HBIO_GBPS_AXIS,
    LIMITATIONS as ANALYTICAL_LIMITATIONS,
    PAGE_BYTES as ANALYTICAL_PAGE_BYTES,
    READ_NS_AXIS as ANALYTICAL_READ_NS_AXIS,
    REL_TOLERANCE as ANALYTICAL_REL_TOLERANCE,
)
from verification.core.contracts import ContractError, load_json_strict


CERTIFICATE_SCHEMA = {
    "name": "hbfsim.verification.certificate",
    "version": 8,
}
SUMMARY_SCHEMA = {
    "name": "hbfsim.simulation.summary",
    "version": 19,
}
EXPLORATORY_VALIDATION = {
    "status": "exploratory_unattached",
    "certificate": None,
}
CONFIDENCE_KEYS = {
    "l0_runtime_and_regression",
    "l1_internal_contracts",
    "l2_independent_implementation",
    "l3_external_reference",
    "l4_calibration",
    "l5_external_hardware",
}
GATE_KEYS = {
    "build_provenance",
    "regression_ctest",
    "sanitizer_ctest",
    "canonical_oracle",
    "property_fuzz",
    "behavioral_differential",
    "analytical_microbench",
    "physical_checks",
    "component_checks",
    "write_amplification",
    "mutation",
    "external_differential",
}
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
DARWIN_LEAK_LIMITATION = (
    "LeakSanitizer is unavailable in the Darwin AddressSanitizer runtime; "
    "this certificate ran ASan+UBSan with leak detection disabled. Linux CI "
    "continues to require leak detection."
)


class CertificateError(ContractError):
    """A certificate or attachment violates the fail-closed contract."""


@dataclass(frozen=True)
class VerifiedCertificate:
    path: Path
    digest_sha256: str
    source_commit: str
    source_tree: str
    simulator_sha256: str
    issued_at_utc: str
    confidence: dict[str, str]
    claim: str

    @property
    def status(self) -> str:
        if self.confidence["l3_external_reference"] == "pass":
            return "external_l3_pass"
        return "foundational_l2_pass"

    def summary_block(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "certificate": {
                "schema": CERTIFICATE_SCHEMA,
                "digest": {
                    "algorithm": "sha256",
                    "value": self.digest_sha256,
                },
                "source_commit": self.source_commit,
                "source_tree": self.source_tree,
                "issued_at_utc": self.issued_at_utc,
                "confidence": copy.deepcopy(self.confidence),
                "claim": self.claim,
            },
        }


def _exact_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    where: str,
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise CertificateError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise CertificateError(f"{where}: unknown keys {sorted(unknown)}")


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CertificateError(f"{where}: expected object")
    return value


def _require_nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CertificateError(f"{where}: expected nonnegative integer")
    return value


def _require_finite_number(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CertificateError(f"{where}: expected finite number")
    return float(value)


def _analytical_close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=ANALYTICAL_REL_TOLERANCE,
        abs_tol=ANALYTICAL_ABS_TOLERANCE_NS,
    )


def _analytical_error_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-18)


def _require_string_list(value: Any, where: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise CertificateError(
            f"{where}: expected nonempty list of unique strings")
    return value


def _require_digest(value: Any, where: str) -> dict[str, Any]:
    digest = _require_object(value, where)
    _exact_keys(
        digest,
        {"algorithm", "value", "bytes"},
        set(),
        where,
    )
    if digest["algorithm"] != "sha256":
        raise CertificateError(f"{where}.algorithm: expected sha256")
    if not isinstance(digest["value"], str) or not DIGEST_RE.fullmatch(
        digest["value"]
    ):
        raise CertificateError(f"{where}.value: expected lowercase SHA-256")
    _require_nonnegative_int(digest["bytes"], f"{where}.bytes")
    return digest


def _require_collection_digest(
    value: Any,
    where: str,
) -> dict[str, Any]:
    digest = _require_object(value, where)
    _exact_keys(
        digest,
        {"algorithm", "value", "file_count", "total_bytes"},
        set(),
        where,
    )
    if digest["algorithm"] != "sha256":
        raise CertificateError(f"{where}.algorithm: expected sha256")
    if not isinstance(digest["value"], str) or not DIGEST_RE.fullmatch(
        digest["value"]
    ):
        raise CertificateError(f"{where}.value: expected lowercase SHA-256")
    if _require_nonnegative_int(
        digest["file_count"], f"{where}.file_count"
    ) == 0:
        raise CertificateError(f"{where}.file_count: expected nonzero")
    _require_nonnegative_int(digest["total_bytes"], f"{where}.total_bytes")
    return digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_digest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CertificateError(f"not a file: {path}")
    return {
        "algorithm": "sha256",
        "value": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def collection_digest(
    base: Path,
    paths: Iterable[Path],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for raw_path in sorted(set(paths), key=lambda item: item.as_posix()):
        path = raw_path if raw_path.is_absolute() else base / raw_path
        if not path.is_file():
            raise CertificateError(f"collection member is not a file: {path}")
        try:
            relative = path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError as error:
            raise CertificateError(
                f"collection member escapes base {base}: {path}"
            ) from error
        entries.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not entries:
        raise CertificateError(f"empty digest collection under {base}")
    payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(payload).hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }


def _run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise CertificateError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def repository_state(
    repository: Path,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    repository = repository.resolve()
    top = Path(_run_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if top != repository:
        raise CertificateError(
            f"repository must be its Git root: expected {top}, got {repository}"
        )
    tracked_status = _run_git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if require_clean and tracked_status:
        raise CertificateError(
            "publishable evidence requires a clean tracked worktree"
        )
    untracked = _run_git(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
    )
    return {
        "commit": _run_git(repository, "rev-parse", "HEAD"),
        "tree": _run_git(repository, "rev-parse", "HEAD^{tree}"),
        "tracked_clean": not bool(tracked_status),
        "untracked_files_observed": (
            0 if not untracked else len(untracked.splitlines())
        ),
        "untracked_files_affect_certificate": False,
    }


def _tracked_paths(repository: Path, prefix: str) -> list[Path]:
    output = _run_git(repository, "ls-files", "--", prefix)
    return [
        Path(line)
        for line in output.splitlines()
        if line
    ]


def input_digests(repository: Path) -> dict[str, Any]:
    case_paths = [
        path
        for path in _tracked_paths(repository, "verification/cases")
        if path.suffix == ".json"
    ]
    verification_paths = [
        path
        for path in _tracked_paths(repository, "verification")
        if path.suffix in {".py", ".cpp", ".sh"}
    ]
    external_paths = _tracked_paths(repository, "evidence/external")
    result = {
        "case_corpus": collection_digest(repository, case_paths),
        "verification_sources": collection_digest(
            repository, verification_paths),
        "external_evidence_inputs": collection_digest(
            repository, external_paths),
        "parameter_registry": file_digest(
            repository / "configs/parameter-provenance.json"),
    }
    return result


def _validate_gate_status(
    gates: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    gate = _require_object(gates.get(name), f"certificate.gates.{name}")
    if gate.get("status") != "pass":
        raise CertificateError(
            f"certificate.gates.{name}.status: expected pass")
    return gate


def validate_certificate_document(document: Any) -> dict[str, Any]:
    certificate = _require_object(document, "certificate")
    _exact_keys(
        certificate,
        {
            "schema",
            "issued_at_utc",
            "source",
            "build",
            "inputs",
            "gates",
            "confidence",
            "claim",
            "limitations",
            "publishable",
        },
        set(),
        "certificate",
    )
    if certificate["schema"] != CERTIFICATE_SCHEMA:
        raise CertificateError("certificate.schema: unsupported schema")
    if (
        not isinstance(certificate["issued_at_utc"], str)
        or not certificate["issued_at_utc"].endswith("Z")
    ):
        raise CertificateError(
            "certificate.issued_at_utc: expected UTC timestamp")

    source = _require_object(certificate["source"], "certificate.source")
    _exact_keys(
        source,
        {
            "commit",
            "tree",
            "tracked_clean",
            "untracked_files_observed",
            "untracked_files_affect_certificate",
        },
        set(),
        "certificate.source",
    )
    for name in ("commit", "tree"):
        if (
            not isinstance(source[name], str)
            or not GIT_OBJECT_RE.fullmatch(source[name])
        ):
            raise CertificateError(
                f"certificate.source.{name}: expected Git object ID")
    if source["tracked_clean"] is not True:
        raise CertificateError(
            "certificate.source.tracked_clean: expected true")
    _require_nonnegative_int(
        source["untracked_files_observed"],
        "certificate.source.untracked_files_observed",
    )
    if source["untracked_files_affect_certificate"] is not False:
        raise CertificateError(
            "certificate.source.untracked_files_affect_certificate: "
            "expected false")

    build = _require_object(certificate["build"], "certificate.build")
    _exact_keys(
        build,
        {
            "simulator",
            "ledger_probe",
            "physical_probe",
        },
        set(),
        "certificate.build",
    )
    for name in build:
        _require_digest(build[name], f"certificate.build.{name}")

    inputs = _require_object(certificate["inputs"], "certificate.inputs")
    _exact_keys(
        inputs,
        {
            "case_corpus",
            "verification_sources",
            "external_evidence_inputs",
            "parameter_registry",
        },
        set(),
        "certificate.inputs",
    )
    _require_collection_digest(
        inputs["case_corpus"], "certificate.inputs.case_corpus")
    _require_collection_digest(
        inputs["verification_sources"],
        "certificate.inputs.verification_sources",
    )
    _require_collection_digest(
        inputs["external_evidence_inputs"],
        "certificate.inputs.external_evidence_inputs",
    )
    _require_digest(
        inputs["parameter_registry"],
        "certificate.inputs.parameter_registry",
    )

    gates = _require_object(certificate["gates"], "certificate.gates")
    _exact_keys(gates, GATE_KEYS, set(), "certificate.gates")
    for name in GATE_KEYS:
        _validate_gate_status(gates, name)

    provenance = gates["build_provenance"]
    _require_digest(
        provenance.get("summary_digest"),
        "certificate.gates.build_provenance.summary_digest",
    )
    for key in (
        "simulator_version",
        "build_type",
        "compiler_id",
        "compiler_version",
    ):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise CertificateError(
                f"certificate.gates.build_provenance.{key}: "
                "expected nonempty string")

    canonical = gates["canonical_oracle"]
    case_count = _require_nonnegative_int(
        canonical.get("case_count"),
        "certificate.gates.canonical_oracle.case_count",
    )
    if case_count != inputs["case_corpus"]["file_count"]:
        raise CertificateError(
            "canonical case count does not match case-corpus digest")
    _require_collection_digest(
        canonical.get("actual_ledgers"),
        "certificate.gates.canonical_oracle.actual_ledgers",
    )
    cases = canonical.get("cases")
    if not isinstance(cases, list) or len(cases) != case_count:
        raise CertificateError(
            "certificate canonical case evidence is incomplete")
    case_ids: set[str] = set()
    for index, case in enumerate(cases):
        where = f"certificate.gates.canonical_oracle.cases[{index}]"
        case = _require_object(case, where)
        _exact_keys(
            case,
            {"case_id", "status", "actual_ledger", "expected_ledger"},
            set(),
            where,
        )
        case_id = case["case_id"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
        ):
            raise CertificateError(f"{where}.case_id: expected unique ID")
        case_ids.add(case_id)
        if case["status"] != "pass":
            raise CertificateError(f"{where}.status: expected pass")
        _require_digest(case["actual_ledger"], f"{where}.actual_ledger")
        _require_digest(case["expected_ledger"], f"{where}.expected_ledger")

    fuzz = gates["property_fuzz"]
    if fuzz.get("models") != ["external", "hbf", "hbm", "hybrid"]:
        raise CertificateError(
            "certificate.gates.property_fuzz.models: unexpected corpus")
    if (
        fuzz.get("seed_start") != 0
        or fuzz.get("seeds_per_model") != 32
        or fuzz.get("max_requests") != 50
        or fuzz.get("generated_cases") != 128
    ):
        raise CertificateError(
            "certificate.gates.property_fuzz: fast corpus is incomplete")
    executions = _require_nonnegative_int(
        fuzz.get("production_executions"),
        "certificate.gates.property_fuzz.production_executions",
    )
    if executions < 136:
        raise CertificateError(
            "certificate.gates.property_fuzz.production_executions: "
            "metamorphic executions are incomplete")

    behavioral = gates["behavioral_differential"]
    if (
        behavioral.get("policies")
        != ["always-admit", "reuse-filtered"]
        or behavioral.get("seeds") != 32
        or behavioral.get("operations_per_seed") != 48
        or behavioral.get("case_count") != 32
        or behavioral.get("production_executions") != 64
    ):
        raise CertificateError(
            "certificate.gates.behavioral_differential: "
            "independent corpus is incomplete")
    compared_fields = behavioral.get("compared_fields")
    required_behavioral_fields = {
        "page_observations",
        "hbm_hits",
        "hbf_bypasses",
        "promotions",
        "clean_evictions",
        "dirty_evictions",
        "dirty_writeback_bytes",
        "decision_fingerprint",
    }
    if (
        not isinstance(compared_fields, list)
        or not required_behavioral_fields.issubset(compared_fields)
    ):
        raise CertificateError(
            "certificate.gates.behavioral_differential: "
            "state/traffic comparison coverage is incomplete")
    _require_digest(
        behavioral.get("report_digest"),
        "certificate.gates.behavioral_differential.report_digest",
    )

    analytical = gates["analytical_microbench"]
    _exact_keys(
        analytical,
        {
            "status",
            "report_digest",
            "hbm",
            "phase_diagram",
            "independent_tier_overlap",
            "claims",
            "limitations",
        },
        set(),
        "certificate.gates.analytical_microbench",
    )
    _require_digest(
        analytical["report_digest"],
        "certificate.gates.analytical_microbench.report_digest",
    )

    analytical_hbm = _require_object(
        analytical["hbm"],
        "certificate.gates.analytical_microbench.hbm",
    )
    _exact_keys(
        analytical_hbm,
        {
            "status",
            "read_bytes",
            "predicted_busy_ns",
            "observed_busy_ns",
        },
        set(),
        "certificate.gates.analytical_microbench.hbm",
    )
    hbm_predicted = _require_finite_number(
        analytical_hbm["predicted_busy_ns"],
        "certificate.gates.analytical_microbench.hbm.predicted_busy_ns",
    )
    hbm_observed = _require_finite_number(
        analytical_hbm["observed_busy_ns"],
        "certificate.gates.analytical_microbench.hbm.observed_busy_ns",
    )
    if (
        analytical_hbm["status"] != "pass"
        or analytical_hbm["read_bytes"] != ANALYTICAL_HBM_READ_BYTES
        or hbm_predicted <= 0.0
        or not _analytical_close(hbm_observed, hbm_predicted)
    ):
        raise CertificateError(
            "certificate analytical HBM work identity failed")

    phase = _require_object(
        analytical["phase_diagram"],
        "certificate.gates.analytical_microbench.phase_diagram",
    )
    _exact_keys(
        phase,
        {
            "status",
            "point_count",
            "region_counts",
            "max_absolute_error_ns",
            "max_relative_error",
            "points",
        },
        set(),
        "certificate.gates.analytical_microbench.phase_diagram",
    )
    expected_coordinates = [
        (read_ns, hbio_gbps)
        for read_ns in ANALYTICAL_READ_NS_AXIS
        for hbio_gbps in ANALYTICAL_HBIO_GBPS_AXIS
    ]
    if (
        phase["status"] != "pass"
        or phase["point_count"] != len(expected_coordinates)
        or not isinstance(phase["points"], list)
        or len(phase["points"]) != len(expected_coordinates)
    ):
        raise CertificateError(
            "certificate analytical phase-grid census is incomplete")
    observed_regions = {
        "media": 0,
        "hbio-data": 0,
        "co-bottleneck": 0,
    }
    max_absolute_error = 0.0
    max_relative_error = 0.0
    for index, (point_value, coordinates) in enumerate(zip(
        phase["points"], expected_coordinates
    )):
        where = (
            "certificate.gates.analytical_microbench."
            f"phase_diagram.points[{index}]"
        )
        point = _require_object(point_value, where)
        _exact_keys(
            point,
            {
                "read_ns",
                "hbio_GBps",
                "predicted_steady_state_ii_ns",
                "observed_steady_state_ii_ns",
                "bottleneck",
                "status",
            },
            set(),
            where,
        )
        read_ns = _require_finite_number(
            point["read_ns"], f"{where}.read_ns")
        hbio_gbps = _require_finite_number(
            point["hbio_GBps"], f"{where}.hbio_GBps")
        if (read_ns, hbio_gbps) != coordinates:
            raise CertificateError(
                f"{where}: analytical coordinate/order mismatch")
        hbio_ii = ANALYTICAL_PAGE_BYTES / hbio_gbps
        predicted_ii = max(read_ns, hbio_ii)
        predicted_label = (
            "co-bottleneck"
            if _analytical_close(read_ns, hbio_ii)
            else "media"
            if read_ns > hbio_ii
            else "hbio-data"
        )
        recorded_predicted = _require_finite_number(
            point["predicted_steady_state_ii_ns"],
            f"{where}.predicted_steady_state_ii_ns",
        )
        recorded_observed = _require_finite_number(
            point["observed_steady_state_ii_ns"],
            f"{where}.observed_steady_state_ii_ns",
        )
        if (
            point["status"] != "pass"
            or point["bottleneck"] != predicted_label
            or not _analytical_close(recorded_predicted, predicted_ii)
            or not _analytical_close(recorded_observed, predicted_ii)
        ):
            raise CertificateError(
                f"{where}: analytical phase identity failed")
        absolute_error = abs(recorded_observed - predicted_ii)
        relative_error = absolute_error / predicted_ii
        max_absolute_error = max(max_absolute_error, absolute_error)
        max_relative_error = max(max_relative_error, relative_error)
        observed_regions[predicted_label] += 1

    region_counts = _require_object(
        phase["region_counts"],
        "certificate.gates.analytical_microbench."
        "phase_diagram.region_counts",
    )
    _exact_keys(
        region_counts,
        {"media", "hbio-data", "co-bottleneck"},
        set(),
        "certificate.gates.analytical_microbench."
        "phase_diagram.region_counts",
    )
    if (
        region_counts != observed_regions
        or observed_regions != {
            "media": 9,
            "hbio-data": 3,
            "co-bottleneck": 3,
        }
    ):
        raise CertificateError(
            "certificate analytical phase region census is inconsistent")
    recorded_max_absolute = _require_finite_number(
        phase["max_absolute_error_ns"],
        "certificate.gates.analytical_microbench."
        "phase_diagram.max_absolute_error_ns",
    )
    recorded_max_relative = _require_finite_number(
        phase["max_relative_error"],
        "certificate.gates.analytical_microbench."
        "phase_diagram.max_relative_error",
    )
    if (
        recorded_max_absolute < 0.0
        or recorded_max_relative < 0.0
        or not _analytical_error_close(
            recorded_max_absolute, max_absolute_error)
        or not _analytical_error_close(
            recorded_max_relative, max_relative_error)
        or recorded_max_absolute > ANALYTICAL_ABS_TOLERANCE_NS
        or recorded_max_relative > ANALYTICAL_REL_TOLERANCE
    ):
        raise CertificateError(
            "certificate analytical phase error bound is inconsistent")

    overlap = _require_object(
        analytical["independent_tier_overlap"],
        "certificate.gates.analytical_microbench."
        "independent_tier_overlap",
    )
    _exact_keys(
        overlap,
        {
            "status",
            "hbm_only_makespan_ns",
            "hbf_only_makespan_ns",
            "predicted_mixed_makespan_ns",
            "observed_mixed_makespan_ns",
            "serialized_sum_ns",
            "hbm_resource_busy_preserved",
            "hbf_resource_busy_preserved",
        },
        set(),
        "certificate.gates.analytical_microbench."
        "independent_tier_overlap",
    )
    hbm_only = _require_finite_number(
        overlap["hbm_only_makespan_ns"],
        "certificate analytical overlap HBM makespan",
    )
    hbf_only = _require_finite_number(
        overlap["hbf_only_makespan_ns"],
        "certificate analytical overlap HBF makespan",
    )
    predicted_mixed = _require_finite_number(
        overlap["predicted_mixed_makespan_ns"],
        "certificate analytical overlap predicted makespan",
    )
    observed_mixed = _require_finite_number(
        overlap["observed_mixed_makespan_ns"],
        "certificate analytical overlap observed makespan",
    )
    serialized_sum = _require_finite_number(
        overlap["serialized_sum_ns"],
        "certificate analytical overlap serialized sum",
    )
    if (
        overlap["status"] != "pass"
        or hbm_only <= 0.0
        or hbf_only <= 0.0
        or not _analytical_close(predicted_mixed, max(hbm_only, hbf_only))
        or not _analytical_close(observed_mixed, predicted_mixed)
        or not _analytical_close(serialized_sum, hbm_only + hbf_only)
        or not observed_mixed < serialized_sum
        or overlap["hbm_resource_busy_preserved"] is not True
        or overlap["hbf_resource_busy_preserved"] is not True
    ):
        raise CertificateError(
            "certificate analytical independent-tier overlap failed")
    if analytical["claims"] != list(ANALYTICAL_CLAIMS):
        raise CertificateError(
            "certificate analytical claim census changed")
    if analytical["limitations"] != list(ANALYTICAL_LIMITATIONS):
        raise CertificateError(
            "certificate analytical limitation census changed")

    for name in ("regression_ctest", "sanitizer_ctest"):
        ctest = gates[name]
        if _require_nonnegative_int(
            ctest.get("tests_passed"),
            f"certificate.gates.{name}.tests_passed",
        ) == 0:
            raise CertificateError(f"certificate {name} suite is empty")
        if ctest.get("tests_failed") != 0:
            raise CertificateError(f"certificate {name} suite has failures")

    sanitizer = gates["sanitizer_ctest"]
    if sanitizer.get("sanitizers") != ["address", "undefined"]:
        raise CertificateError(
            "certificate sanitizer set is incomplete")
    sanitizer_platform = _require_object(
        sanitizer.get("platform"),
        "certificate.gates.sanitizer_ctest.platform",
    )
    _exact_keys(
        sanitizer_platform,
        {"system", "machine"},
        set(),
        "certificate.gates.sanitizer_ctest.platform",
    )
    if any(
        not isinstance(sanitizer_platform[key], str)
        or not sanitizer_platform[key]
        for key in ("system", "machine")
    ):
        raise CertificateError(
            "certificate sanitizer platform is incomplete")
    sanitizer_environment = _require_object(
        sanitizer.get("runtime_environment"),
        "certificate.gates.sanitizer_ctest.runtime_environment",
    )
    _exact_keys(
        sanitizer_environment,
        {"ASAN_OPTIONS", "UBSAN_OPTIONS"},
        set(),
        "certificate.gates.sanitizer_ctest.runtime_environment",
    )
    expected_leak_status = (
        "unsupported_on_darwin"
        if sanitizer_platform["system"] == "Darwin"
        else "enabled"
    )
    expected_asan_options = (
        "detect_leaks=0:halt_on_error=1"
        if expected_leak_status == "unsupported_on_darwin"
        else "detect_leaks=1:halt_on_error=1"
    )
    if (
        sanitizer.get("leak_detection") != expected_leak_status
        or sanitizer_environment != {
            "ASAN_OPTIONS": expected_asan_options,
            "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
        }
    ):
        raise CertificateError(
            "certificate sanitizer runtime policy is inconsistent")

    for name, count_key in (
        ("physical_checks", "checks_passed"),
        ("component_checks", "checks_passed"),
        ("write_amplification", "checks_passed"),
    ):
        if _require_nonnegative_int(
            gates[name].get(count_key),
            f"certificate.gates.{name}.{count_key}",
        ) == 0:
            raise CertificateError(f"certificate {name} gate is empty")

    mutation = gates["mutation"]
    for key in ("total", "killed", "critical_total", "critical_killed"):
        _require_nonnegative_int(
            mutation.get(key), f"certificate.gates.mutation.{key}")
    if (
        mutation["total"] == 0
        or mutation["killed"] != mutation["total"]
        or mutation["critical_killed"] != mutation["critical_total"]
    ):
        raise CertificateError(
            "certificate mutation gate did not kill every selected mutation")
    _require_digest(
        mutation.get("report_digest"),
        "certificate.gates.mutation.report_digest",
    )
    mutation_entries = mutation.get("mutations")
    if (
        not isinstance(mutation_entries, list)
        or len(mutation_entries) != mutation["total"]
    ):
        raise CertificateError(
            "certificate mutation matrix is incomplete")
    mutation_ids: set[str] = set()
    for index, entry in enumerate(mutation_entries):
        where = f"certificate.gates.mutation.mutations[{index}]"
        entry = _require_object(entry, where)
        _exact_keys(
            entry,
            {
                "id",
                "critical",
                "patch_digest",
                "cases",
                "status",
                "killed_by",
            },
            set(),
            where,
        )
        mutation_id = entry["id"]
        if (
            not isinstance(mutation_id, str)
            or not mutation_id
            or mutation_id in mutation_ids
        ):
            raise CertificateError(f"{where}.id: expected unique ID")
        mutation_ids.add(mutation_id)
        if entry["status"] != "killed":
            raise CertificateError(f"{where}.status: expected killed")
        if not isinstance(entry["critical"], bool):
            raise CertificateError(f"{where}.critical: expected boolean")
        if (
            not isinstance(entry["patch_digest"], str)
            or not entry["patch_digest"].startswith("sha256:")
            or not DIGEST_RE.fullmatch(entry["patch_digest"][7:])
        ):
            raise CertificateError(
                f"{where}.patch_digest: expected tagged SHA-256")
        if (
            not isinstance(entry["cases"], list)
            or not entry["cases"]
            or entry["killed_by"] not in entry["cases"]
        ):
            raise CertificateError(f"{where}: invalid kill attribution")

    external = gates["external_differential"]
    _exact_keys(
        external,
        {
            "status",
            "fixtures_passed",
            "fixtures_total",
            "cases_required",
            "cases_passed",
            "l3_status",
            "manifest_digest",
            "report_digest",
            "validated_facets",
            "tools",
        },
        set(),
        "certificate.gates.external_differential",
    )
    fixtures = _require_nonnegative_int(
        external.get("fixtures_passed"),
        "certificate.gates.external_differential.fixtures_passed",
    )
    if fixtures == 0 or fixtures != external.get("fixtures_total"):
        raise CertificateError(
            "certificate external adapter fixtures are incomplete")
    if external.get("l3_status") not in {"pass", "partial", "not_run"}:
        raise CertificateError(
            "certificate external L3 status is invalid")
    _require_digest(
        external.get("manifest_digest"),
        "certificate.gates.external_differential.manifest_digest",
    )
    _require_digest(
        external.get("report_digest"),
        "certificate.gates.external_differential.report_digest",
    )
    external_tools = external.get("tools")
    if not isinstance(external_tools, list) or len(external_tools) != fixtures:
        raise CertificateError(
            "certificate external tool evidence is incomplete")
    tool_ids: set[str] = set()
    actual_statuses: list[str] = []
    derived_validated_facets: list[dict[str, Any]] = []
    derived_cases_required = 0
    derived_cases_passed = 0
    for index, tool in enumerate(external_tools):
        where = f"certificate.gates.external_differential.tools[{index}]"
        tool = _require_object(tool, where)
        _exact_keys(
            tool,
            {
                "id",
                "domain",
                "commit",
                "source_tree",
                "fixture_status",
                "actual_status",
                "facets",
                "cases",
            },
            set(),
            where,
        )
        tool_id = tool["id"]
        if (
            not isinstance(tool_id, str)
            or not tool_id
            or tool_id in tool_ids
        ):
            raise CertificateError(f"{where}.id: expected unique ID")
        tool_ids.add(tool_id)
        if tool["domain"] not in {"dram", "flash"}:
            raise CertificateError(f"{where}.domain: invalid domain")
        for name in ("commit", "source_tree"):
            if (
                not isinstance(tool[name], str)
                or not GIT_OBJECT_RE.fullmatch(tool[name])
            ):
                raise CertificateError(
                    f"{where}.{name}: invalid Git object")
        if tool["fixture_status"] != "pass":
            raise CertificateError(f"{where}.fixture_status: expected pass")
        if tool["actual_status"] not in {"pass", "not_run"}:
            raise CertificateError(f"{where}.actual_status: invalid status")
        actual_statuses.append(tool["actual_status"])

        facets = tool["facets"]
        if not isinstance(facets, list) or not facets:
            raise CertificateError(f"{where}.facets: expected nonempty list")
        facet_ids: set[str] = set()
        declared_case_ids: list[str] = []
        for facet_index, facet in enumerate(facets):
            facet_where = f"{where}.facets[{facet_index}]"
            facet = _require_object(facet, facet_where)
            _exact_keys(
                facet,
                {
                    "id",
                    "status",
                    "claim",
                    "cases",
                    "shared_boundary",
                    "excluded",
                },
                set(),
                facet_where,
            )
            facet_id = facet["id"]
            if (
                not isinstance(facet_id, str)
                or not facet_id
                or facet_id in facet_ids
            ):
                raise CertificateError(
                    f"{facet_where}.id: expected unique ID")
            facet_ids.add(facet_id)
            expected_facet_status = (
                "pass" if tool["actual_status"] == "pass" else "not_run")
            if facet["status"] != expected_facet_status:
                raise CertificateError(
                    f"{facet_where}.status: expected "
                    f"{expected_facet_status}")
            if not isinstance(facet["claim"], str) or not facet["claim"]:
                raise CertificateError(
                    f"{facet_where}.claim: expected nonempty string")
            cases = _require_string_list(
                facet["cases"], f"{facet_where}.cases")
            overlap = set(declared_case_ids).intersection(cases)
            if overlap:
                raise CertificateError(
                    f"{facet_where}.cases: duplicate case IDs "
                    f"{sorted(overlap)}")
            declared_case_ids.extend(cases)
            _require_string_list(
                facet["shared_boundary"],
                f"{facet_where}.shared_boundary",
            )
            _require_string_list(
                facet["excluded"], f"{facet_where}.excluded")
            if facet["status"] == "pass":
                derived_validated_facets.append({
                    "tool_id": tool_id,
                    "domain": tool["domain"],
                    **copy.deepcopy(facet),
                })

        cases = tool["cases"]
        if not isinstance(cases, list):
            raise CertificateError(f"{where}.cases: expected list")
        if tool["actual_status"] == "not_run" and cases:
            raise CertificateError(
                f"{where}.cases: not_run tool must have no actual cases")
        observed_case_ids: list[str] = []
        observed_case_facets: dict[str, str] = {}
        for case_index, case in enumerate(cases):
            case_where = f"{where}.cases[{case_index}]"
            case = _require_object(case, case_where)
            _exact_keys(
                case,
                {"case_id", "facet_id", "status", "metrics"},
                set(),
                case_where,
            )
            case_id = case["case_id"]
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in observed_case_ids
            ):
                raise CertificateError(
                    f"{case_where}.case_id: expected unique ID")
            if case["status"] != "pass":
                raise CertificateError(
                    f"{case_where}.status: expected pass")
            facet_id = case["facet_id"]
            if facet_id not in facet_ids:
                raise CertificateError(
                    f"{case_where}.facet_id: unknown facet")
            metrics = _require_object(
                case["metrics"], f"{case_where}.metrics")
            if not metrics:
                raise CertificateError(
                    f"{case_where}.metrics: expected nonempty object")
            for metric_name, metric_value in metrics.items():
                if not isinstance(metric_name, str) or not metric_name:
                    raise CertificateError(
                        f"{case_where}.metrics: invalid metric name")
                if (
                    isinstance(metric_value, bool)
                    or not isinstance(metric_value, (int, float))
                    or (
                        isinstance(metric_value, float)
                        and not math.isfinite(metric_value)
                    )
                ):
                    raise CertificateError(
                        f"{case_where}.metrics.{metric_name}: "
                        "expected finite number")
            observed_case_ids.append(case_id)
            observed_case_facets[case_id] = facet_id

        derived_cases_required += len(declared_case_ids)
        if tool["actual_status"] == "pass":
            if observed_case_ids != declared_case_ids:
                raise CertificateError(
                    f"{where}.cases: actual census/order differs from "
                    "declared facets")
            for facet in facets:
                for case_id in facet["cases"]:
                    if observed_case_facets[case_id] != facet["id"]:
                        raise CertificateError(
                            f"{where}.cases: {case_id} is assigned to the "
                            "wrong facet")
            derived_cases_passed += len(cases)
    derived_l3 = (
        "pass"
        if all(status == "pass" for status in actual_statuses)
        else "not_run"
        if all(status == "not_run" for status in actual_statuses)
        else "partial"
    )
    if derived_l3 != external["l3_status"]:
        raise CertificateError(
            "certificate external tool statuses disagree with L3")
    if (
        external["cases_required"] != derived_cases_required
        or external["cases_passed"] != derived_cases_passed
    ):
        raise CertificateError(
            "certificate external case census is inconsistent")
    if external["validated_facets"] != derived_validated_facets:
        raise CertificateError(
            "certificate external validated-facet census is inconsistent")

    confidence = _require_object(
        certificate["confidence"], "certificate.confidence")
    _exact_keys(
        confidence, CONFIDENCE_KEYS, set(), "certificate.confidence")
    for name in (
        "l0_runtime_and_regression",
        "l1_internal_contracts",
        "l2_independent_implementation",
    ):
        if confidence[name] != "pass":
            raise CertificateError(f"certificate.confidence.{name}: expected pass")
    if confidence["l3_external_reference"] != external["l3_status"]:
        raise CertificateError(
            "certificate confidence and external report disagree")
    for name in ("l4_calibration", "l5_external_hardware"):
        if confidence[name] != "not_run":
            raise CertificateError(
                f"certificate.confidence.{name}: expected not_run")

    if not isinstance(certificate["claim"], str) or not certificate["claim"]:
        raise CertificateError("certificate.claim: expected nonempty string")
    if ANALYTICAL_CERTIFICATE_CLAIM not in certificate["claim"]:
        raise CertificateError(
            "certificate.claim omits the analytical gate scope")
    limitations = certificate["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or not all(isinstance(item, str) and item for item in limitations)
    ):
        raise CertificateError(
            "certificate.limitations: expected nonempty strings")
    for facet in derived_validated_facets:
        if facet["id"] not in certificate["claim"]:
            raise CertificateError(
                "certificate.claim omits a validated L3 facet")
        for excluded in facet["excluded"]:
            if not any(excluded in limitation for limitation in limitations):
                raise CertificateError(
                    "certificate.limitations omit a validated facet "
                    f"exclusion: {excluded}")
    for analytical_limitation in ANALYTICAL_LIMITATIONS:
        if analytical_limitation not in limitations:
            raise CertificateError(
                "certificate.limitations omit an analytical boundary")
    sanitizer = certificate["gates"]["sanitizer_ctest"]
    if (
        sanitizer["leak_detection"] == "unsupported_on_darwin"
        and DARWIN_LEAK_LIMITATION not in limitations
    ):
        raise CertificateError(
            "certificate.limitations omit Darwin leak-detection boundary")
    if certificate["publishable"] is not True:
        raise CertificateError("certificate.publishable: expected true")
    return certificate


def assemble_certificate(
    *,
    repository: Path,
    simulator: Path,
    ledger_probe: Path,
    physical_probe: Path,
    issued_at_utc: str,
    gates: dict[str, Any],
) -> dict[str, Any]:
    source = repository_state(repository, require_clean=True)
    external_status = _require_object(
        gates.get("external_differential"),
        "gates.external_differential",
    ).get("l3_status")
    analytical = _require_object(
        gates.get("analytical_microbench"),
        "gates.analytical_microbench",
    )
    validated_facets = gates["external_differential"].get(
        "validated_facets", [])
    l3_scope = ", ".join(
        f"{facet['id']} ({len(facet['cases'])} cases)"
        for facet in validated_facets
    )
    claim = (
        "The exact committed tree and digest-bound publication binaries "
        "passed HBFSim's L0-L2 foundational gates for the declared model "
        f"contracts. {ANALYTICAL_CERTIFICATE_CLAIM}"
    )
    if l3_scope:
        claim += (
            " Facet-scoped L3 external differential evidence additionally "
            f"passed only for: {l3_scope}."
        )
    limitations = [
        (
            "This is model-relative implementation evidence, not a "
            "claim that HBFSim predicts unreleased HBF hardware."
        ),
        *copy.deepcopy(analytical.get("limitations", [])),
    ]
    if (
        gates["sanitizer_ctest"].get("leak_detection")
        == "unsupported_on_darwin"
    ):
        limitations.append(DARWIN_LEAK_LIMITATION)
    if validated_facets:
        limitations.extend(
            (
                f"L3 facet {facet['id']} excludes: "
                + "; ".join(facet["excluded"])
            )
            for facet in validated_facets
        )
    else:
        limitations.append(
            "No actual external-simulator facet passed in this certificate."
        )
    if external_status == "partial":
        limitations.append(
            "The aggregate L3 status is partial because not every declared "
            "external tool supplied passing actual evidence."
        )
    limitations.extend([
        "Calibration (L4) and held-out hardware validation (L5) remain not_run.",
        (
            "The certificate is digest-bound but is not a cryptographic "
            "signature or identity attestation."
        ),
    ])
    document = {
        "schema": CERTIFICATE_SCHEMA,
        "issued_at_utc": issued_at_utc,
        "source": source,
        "build": {
            "simulator": file_digest(simulator),
            "ledger_probe": file_digest(ledger_probe),
            "physical_probe": file_digest(physical_probe),
        },
        "inputs": input_digests(repository),
        "gates": copy.deepcopy(gates),
        "confidence": {
            "l0_runtime_and_regression": "pass",
            "l1_internal_contracts": "pass",
            "l2_independent_implementation": "pass",
            "l3_external_reference": external_status,
            "l4_calibration": "not_run",
            "l5_external_hardware": "not_run",
        },
        "claim": claim,
        "limitations": limitations,
        "publishable": True,
    }
    return validate_certificate_document(document)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_certificate(
    path: Path,
    *,
    repository: Path,
    simulator: Path,
) -> VerifiedCertificate:
    path = path.resolve()
    certificate = validate_certificate_document(load_json_strict(path))
    current_source = repository_state(repository, require_clean=True)
    for key in ("commit", "tree"):
        if certificate["source"][key] != current_source[key]:
            raise CertificateError(
                f"certificate source {key} does not match current repository")
    current_inputs = input_digests(repository)
    if certificate["inputs"] != current_inputs:
        raise CertificateError(
            "certificate validation inputs do not match current repository")
    current_binary = file_digest(simulator)
    if certificate["build"]["simulator"] != current_binary:
        raise CertificateError(
            "certificate simulator digest does not match executable")
    return VerifiedCertificate(
        path=path,
        digest_sha256=sha256_file(path),
        source_commit=certificate["source"]["commit"],
        source_tree=certificate["source"]["tree"],
        simulator_sha256=current_binary["value"],
        issued_at_utc=certificate["issued_at_utc"],
        confidence=copy.deepcopy(certificate["confidence"]),
        claim=certificate["claim"],
    )


def attach_certificate_to_summary(
    summary: dict[str, Any],
    certificate: VerifiedCertificate,
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise CertificateError("scenario summary must be an object")
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise CertificateError("scenario summary schema is unsupported")
    if summary.get("sanity") != "PASS":
        raise CertificateError("cannot certify a non-passing scenario summary")
    simulator = _require_object(
        summary.get("simulator"), "scenario summary.simulator")
    if simulator.get("git_commit") != certificate.source_commit:
        raise CertificateError(
            "scenario summary commit does not match certificate")
    if simulator.get("git_dirty") is not False:
        raise CertificateError(
            "scenario summary binary was configured from a dirty tree")
    build = _require_object(summary.get("build"), "scenario summary.build")
    executable = _require_object(
        build.get("executable_digest"),
        "scenario summary.build.executable_digest",
    )
    if (
        executable.get("algorithm") != "sha256"
        or executable.get("value") != certificate.simulator_sha256
    ):
        raise CertificateError(
            "scenario summary executable does not match certificate")
    target = certificate.summary_block()
    current = summary.get("validation")
    if current == target:
        return summary
    if current != EXPLORATORY_VALIDATION:
        raise CertificateError(
            "scenario summary has an unknown or conflicting validation block")
    result = copy.deepcopy(summary)
    result["validation"] = target
    return result


def attach_certificate_to_summary_file(
    summary_path: Path,
    certificate: VerifiedCertificate,
) -> dict[str, Any]:
    summary = load_json_strict(summary_path)
    attached = attach_certificate_to_summary(summary, certificate)
    write_json_atomic(summary_path, attached)
    return attached


def ensure_exploratory_summary(summary: dict[str, Any]) -> None:
    if summary.get("validation") != EXPLORATORY_VALIDATION:
        raise CertificateError(
            "uncertified scenario summary must be explicitly exploratory")
