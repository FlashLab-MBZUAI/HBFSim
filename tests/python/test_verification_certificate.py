#!/usr/bin/env python3
"""Contract tests for foundational certificates and summary attachment."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verification.core.analytical import (  # noqa: E402
    CLAIMS as ANALYTICAL_CLAIMS,
    HBIO_GBPS_AXIS as ANALYTICAL_HBIO_GBPS_AXIS,
    LIMITATIONS as ANALYTICAL_LIMITATIONS,
    PAGE_BYTES as ANALYTICAL_PAGE_BYTES,
    READ_NS_AXIS as ANALYTICAL_READ_NS_AXIS,
)
from verification.core.certificate import (  # noqa: E402
    CERTIFICATE_SCHEMA,
    DARWIN_LEAK_LIMITATION,
    CertificateError,
    assemble_certificate,
    attach_certificate_to_summary,
    collection_digest,
    file_digest,
    validate_certificate_document,
    verify_certificate,
    write_json_atomic,
)


class CertificateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="hbfsim-certificate-test-")
        self.root = Path(self.temporary.name)
        for directory in (
            "verification/cases",
            "verification/oracles",
            "verification/probes",
            "evidence/external/assets",
            "configs",
            "bin",
            "artifacts",
        ):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / "verification/cases/tiny.json").write_text(
            '{"schema":{"name":"test","version":1}}\n')
        (self.root / "verification/oracles/example.py").write_text(
            "def expected():\n    return 1\n")
        (self.root / "evidence/external/assets/tools.json").write_text(
            '{"schema":{"name":"tools","version":1},"tools":[]}\n')
        (self.root / "verification/probes/ledger.cpp").write_text(
            "int main() { return 0; }\n")
        (self.root / "configs/parameter-provenance.json").write_text(
            '{"schema":{"name":"parameters","version":1}}\n')
        self._git("init", "-q")
        self._git("config", "user.name", "Certificate Test")
        self._git("config", "user.email", "certificate@example.invalid")
        self._git("add", ".")
        self._git("commit", "-qm", "fixture")

        self.simulator = self.root / "bin/hbfsim"
        self.ledger_probe = self.root / "bin/ledger_probe"
        self.physical_probe = self.root / "bin/physical_probe"
        for path, payload in (
            (self.simulator, b"scenario-binary"),
            (self.ledger_probe, b"ledger-probe"),
            (self.physical_probe, b"physical-probe"),
        ):
            path.write_bytes(payload)
        actual = self.root / "artifacts/tiny.actual.jsonl"
        actual.write_text('{"kind":"header"}\n')
        expected = self.root / "artifacts/tiny.expected.jsonl"
        expected.write_text('{"kind":"header"}\n')
        mutation_report = self.root / "artifacts/mutation.json"
        mutation_report.write_text('{"passed":true}\n')
        external_report = self.root / "artifacts/external.json"
        external_report.write_text('{"l3_status":"not_run"}\n')
        behavioral_report = self.root / "artifacts/behavioral.json"
        behavioral_report.write_text('{"status":"PASS"}\n')
        analytical_report = self.root / "artifacts/analytical.json"
        analytical_report.write_text('{"status":"pass"}\n')
        analytical_points = []
        for read_ns in ANALYTICAL_READ_NS_AXIS:
            for hbio_gbps in ANALYTICAL_HBIO_GBPS_AXIS:
                hbio_ii = ANALYTICAL_PAGE_BYTES / hbio_gbps
                predicted_ii = max(read_ns, hbio_ii)
                bottleneck = (
                    "co-bottleneck"
                    if read_ns == hbio_ii
                    else "media"
                    if read_ns > hbio_ii
                    else "hbio-data"
                )
                analytical_points.append({
                    "read_ns": read_ns,
                    "hbio_GBps": hbio_gbps,
                    "predicted_steady_state_ii_ns": predicted_ii,
                    "observed_steady_state_ii_ns": predicted_ii,
                    "bottleneck": bottleneck,
                    "status": "pass",
                })
        self.gates = {
            "build_provenance": {
                "status": "pass",
                "summary_digest": file_digest(actual),
                "simulator_version": "0.1.0-dev",
                "build_type": "Release",
                "compiler_id": "TestCompiler",
                "compiler_version": "1.0",
            },
            "regression_ctest": {
                "status": "pass",
                "tests_passed": 3,
                "tests_failed": 0,
            },
            "sanitizer_ctest": {
                "status": "pass",
                "tests_passed": 3,
                "tests_failed": 0,
                "sanitizers": ["address", "undefined"],
                "platform": {
                    "system": "Linux",
                    "machine": "x86_64",
                },
                "runtime_environment": {
                    "ASAN_OPTIONS": (
                        "detect_leaks=1:halt_on_error=1"
                    ),
                    "UBSAN_OPTIONS": (
                        "halt_on_error=1:print_stacktrace=1"
                    ),
                },
                "leak_detection": "enabled",
            },
            "canonical_oracle": {
                "status": "pass",
                "case_count": 1,
                "actual_ledgers": collection_digest(
                    self.root / "artifacts",
                    [Path("tiny.actual.jsonl")],
                ),
                "cases": [{
                    "case_id": "tiny",
                    "status": "pass",
                    "actual_ledger": file_digest(actual),
                    "expected_ledger": file_digest(expected),
                }],
            },
            "property_fuzz": {
                "status": "pass",
                "models": ["external", "hbf", "hbm", "hybrid"],
                "seed_start": 0,
                "seeds_per_model": 32,
                "max_requests": 50,
                "generated_cases": 128,
                "production_executions": 136,
            },
            "behavioral_differential": {
                "status": "pass",
                "seeds": 32,
                "operations_per_seed": 48,
                "policies": ["always-admit", "reuse-filtered"],
                "case_count": 32,
                "production_executions": 64,
                "compared_fields": [
                    "page_observations",
                    "hbm_hits",
                    "hbf_bypasses",
                    "promotions",
                    "clean_evictions",
                    "dirty_evictions",
                    "dirty_writeback_bytes",
                    "decision_fingerprint",
                ],
                "report_digest": file_digest(behavioral_report),
            },
            "analytical_microbench": {
                "status": "pass",
                "report_digest": file_digest(analytical_report),
                "hbm": {
                    "status": "pass",
                    "read_bytes": 1024**2,
                    "predicted_busy_ns": 40960.0,
                    "observed_busy_ns": 40960.0,
                },
                "phase_diagram": {
                    "status": "pass",
                    "point_count": len(analytical_points),
                    "region_counts": {
                        "media": 9,
                        "hbio-data": 3,
                        "co-bottleneck": 3,
                    },
                    "max_absolute_error_ns": 0.0,
                    "max_relative_error": 0.0,
                    "points": analytical_points,
                },
                "independent_tier_overlap": {
                    "status": "pass",
                    "hbm_only_makespan_ns": 100.0,
                    "hbf_only_makespan_ns": 500.0,
                    "predicted_mixed_makespan_ns": 500.0,
                    "observed_mixed_makespan_ns": 500.0,
                    "serialized_sum_ns": 600.0,
                    "hbm_resource_busy_preserved": True,
                    "hbf_resource_busy_preserved": True,
                },
                "claims": list(ANALYTICAL_CLAIMS),
                "limitations": list(ANALYTICAL_LIMITATIONS),
            },
            "physical_checks": {
                "status": "pass",
                "checks_passed": 2,
            },
            "component_checks": {
                "status": "pass",
                "checks_passed": 2,
            },
            "write_amplification": {
                "status": "pass",
                "checks_passed": 2,
            },
            "mutation": {
                "status": "pass",
                "total": 2,
                "killed": 2,
                "critical_total": 2,
                "critical_killed": 2,
                "report_digest": file_digest(mutation_report),
                "mutations": [
                    {
                        "id": "MUT-1",
                        "critical": True,
                        "patch_digest": "sha256:" + "1" * 64,
                        "cases": ["tiny"],
                        "status": "killed",
                        "killed_by": "tiny",
                    },
                    {
                        "id": "MUT-2",
                        "critical": True,
                        "patch_digest": "sha256:" + "2" * 64,
                        "cases": ["tiny"],
                        "status": "killed",
                        "killed_by": "tiny",
                    },
                ],
            },
            "external_differential": {
                "status": "pass",
                "fixtures_passed": 2,
                "fixtures_total": 2,
                "cases_required": 6,
                "cases_passed": 0,
                "l3_status": "not_run",
                "manifest_digest": file_digest(
                    self.root / "evidence/external/assets/tools.json"),
                "report_digest": file_digest(external_report),
                "validated_facets": [],
                "tools": [
                    {
                        "id": "ramulator2",
                        "domain": "dram",
                        "commit": "1" * 40,
                        "source_tree": "3" * 40,
                        "fixture_status": "pass",
                        "actual_status": "not_run",
                        "facets": [{
                            "id": "dram.read-byte-accounting",
                            "status": "not_run",
                            "claim": "Matched DRAM read service.",
                            "cases": [
                                "dram-miss",
                                "dram-hit",
                                "dram-conflict",
                            ],
                            "shared_boundary": ["one DRAM bank"],
                            "excluded": ["DRAM writes"],
                        }],
                        "cases": [],
                    },
                    {
                        "id": "mqsim",
                        "domain": "flash",
                        "commit": "2" * 40,
                        "source_tree": "4" * 40,
                        "fixture_status": "pass",
                        "actual_status": "not_run",
                        "facets": [{
                            "id": "flash.direct-data-page-io",
                            "status": "not_run",
                            "claim": "Matched direct flash page I/O.",
                            "cases": [
                                "flash-write",
                                "flash-write-read",
                                "flash-overwrite",
                            ],
                            "shared_boundary": ["one SLC plane"],
                            "excluded": ["flash timing"],
                        }],
                        "cases": [],
                    },
                ],
            },
        }
        self.document = assemble_certificate(
            repository=self.root,
            simulator=self.simulator,
            ledger_probe=self.ledger_probe,
            physical_probe=self.physical_probe,
            issued_at_utc="2026-07-28T00:00:00Z",
            gates=self.gates,
        )
        self.certificate_path = self.root / "certificate.json"
        write_json_atomic(self.certificate_path, self.document)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _git(self, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=self.root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    def _verify(self):
        return verify_certificate(
            self.certificate_path,
            repository=self.root,
            simulator=self.simulator,
        )

    def _summary(self) -> dict:
        return {
            "schema": {
                "name": "hbfsim.simulation.summary",
                "version": 19,
            },
            "simulator": {
                "git_commit": self._git("rev-parse", "HEAD"),
                "git_dirty": False,
            },
            "build": {
                "executable_digest": {
                    "algorithm": "sha256",
                    "value": file_digest(self.simulator)["value"],
                },
            },
            "validation": {
                "status": "exploratory_unattached",
                "certificate": None,
            },
            "sanity": "PASS",
            "scenarios": [],
        }

    def test_clean_bound_certificate_verifies_and_attaches(self) -> None:
        verified = self._verify()
        self.assertNotIn("external_tools", self.document["inputs"])
        self.assertEqual(
            self.document["inputs"]["external_evidence_inputs"][
                "file_count"
            ],
            1,
        )
        attached = attach_certificate_to_summary(self._summary(), verified)
        self.assertEqual(
            attached["validation"]["status"],
            "foundational_l2_pass",
        )
        reference = attached["validation"]["certificate"]
        self.assertEqual(reference["schema"], CERTIFICATE_SCHEMA)
        self.assertEqual(reference["source_commit"], verified.source_commit)
        self.assertEqual(
            attach_certificate_to_summary(attached, verified),
            attached,
        )

    def test_untracked_material_does_not_invalidate_certificate(self) -> None:
        (self.root / "notes.bin").write_bytes(b"not model input")
        self.assertEqual(self._verify().source_commit,
                         self._git("rev-parse", "HEAD"))

    def test_tracked_source_drift_is_rejected(self) -> None:
        (self.root / "verification/oracles/example.py").write_text(
            "def expected():\n    return 2\n")
        with self.assertRaisesRegex(CertificateError, "clean tracked"):
            self._verify()

    def test_binary_drift_is_rejected(self) -> None:
        self.simulator.write_bytes(b"different-binary")
        with self.assertRaisesRegex(CertificateError, "digest"):
            self._verify()

    def test_nonpublishable_document_is_rejected(self) -> None:
        document = json.loads(json.dumps(self.document))
        document["publishable"] = False
        with self.assertRaisesRegex(CertificateError, "publishable"):
            validate_certificate_document(document)

    def test_l3_claim_is_facet_scoped_and_carries_exclusions(self) -> None:
        gates = json.loads(json.dumps(self.gates))
        external = gates["external_differential"]
        external["l3_status"] = "pass"
        external["cases_passed"] = 6
        for tool in external["tools"]:
            tool["actual_status"] = "pass"
            facet = tool["facets"][0]
            facet["status"] = "pass"
            tool["cases"] = [
                {
                    "case_id": case_id,
                    "facet_id": facet["id"],
                    "status": "pass",
                    "metrics": {"matched_work": index + 1},
                }
                for index, case_id in enumerate(facet["cases"])
            ]
            external["validated_facets"].append({
                "tool_id": tool["id"],
                "domain": tool["domain"],
                **facet,
            })
        document = assemble_certificate(
            repository=self.root,
            simulator=self.simulator,
            ledger_probe=self.ledger_probe,
            physical_probe=self.physical_probe,
            issued_at_utc="2026-07-28T00:00:00Z",
            gates=gates,
        )
        self.assertEqual(
            document["confidence"]["l3_external_reference"], "pass")
        for facet in external["validated_facets"]:
            self.assertIn(facet["id"], document["claim"])
            for excluded in facet["excluded"]:
                self.assertTrue(any(
                    excluded in limitation
                    for limitation in document["limitations"]
                ))

    def test_external_case_census_tampering_is_rejected(self) -> None:
        document = json.loads(json.dumps(self.document))
        document["gates"]["external_differential"][
            "cases_required"
        ] = 5
        with self.assertRaisesRegex(CertificateError, "case census"):
            validate_certificate_document(document)

    def test_missing_required_gate_is_rejected(self) -> None:
        document = json.loads(json.dumps(self.document))
        del document["gates"]["mutation"]
        with self.assertRaisesRegex(CertificateError, "missing"):
            validate_certificate_document(document)

    def test_incomplete_behavioral_differential_is_rejected(self) -> None:
        document = json.loads(json.dumps(self.document))
        document["gates"]["behavioral_differential"]["seeds"] = 31
        with self.assertRaisesRegex(
            CertificateError, "behavioral_differential"
        ):
            validate_certificate_document(document)

    def test_inconsistent_sanitizer_runtime_policy_is_rejected(self) -> None:
        document = json.loads(json.dumps(self.document))
        document["gates"]["sanitizer_ctest"]["runtime_environment"][
            "ASAN_OPTIONS"
        ] = "detect_leaks=0:halt_on_error=1"
        with self.assertRaisesRegex(
            CertificateError, "sanitizer runtime policy"
        ):
            validate_certificate_document(document)

    def test_darwin_leak_boundary_is_recorded_as_a_limitation(self) -> None:
        gates = json.loads(json.dumps(self.gates))
        sanitizer = gates["sanitizer_ctest"]
        sanitizer["platform"] = {
            "system": "Darwin",
            "machine": "arm64",
        }
        sanitizer["runtime_environment"]["ASAN_OPTIONS"] = (
            "detect_leaks=0:halt_on_error=1"
        )
        sanitizer["leak_detection"] = "unsupported_on_darwin"
        document = assemble_certificate(
            repository=self.root,
            simulator=self.simulator,
            ledger_probe=self.ledger_probe,
            physical_probe=self.physical_probe,
            issued_at_utc="2026-07-28T00:00:00Z",
            gates=gates,
        )
        self.assertIn(DARWIN_LEAK_LIMITATION, document["limitations"])

    def test_analytical_phase_point_tampering_is_rejected(self) -> None:
        document = json.loads(json.dumps(self.document))
        point = document["gates"]["analytical_microbench"][
            "phase_diagram"
        ]["points"][0]
        point["observed_steady_state_ii_ns"] = 1000.0
        with self.assertRaisesRegex(
            CertificateError, "analytical phase identity"
        ):
            validate_certificate_document(document)

    def test_analytical_phase_census_tampering_is_rejected(self) -> None:
        document = json.loads(json.dumps(self.document))
        document["gates"]["analytical_microbench"][
            "phase_diagram"
        ]["region_counts"]["media"] = 11
        with self.assertRaisesRegex(
            CertificateError, "phase region census"
        ):
            validate_certificate_document(document)

    def test_analytical_overlap_serialization_is_rejected(self) -> None:
        document = json.loads(json.dumps(self.document))
        overlap = document["gates"]["analytical_microbench"][
            "independent_tier_overlap"
        ]
        overlap["predicted_mixed_makespan_ns"] = 600.0
        overlap["observed_mixed_makespan_ns"] = 600.0
        with self.assertRaisesRegex(
            CertificateError, "independent-tier overlap"
        ):
            validate_certificate_document(document)

    def test_analytical_limitation_omission_is_rejected(self) -> None:
        document = json.loads(json.dumps(self.document))
        document["limitations"].remove(ANALYTICAL_LIMITATIONS[0])
        with self.assertRaisesRegex(
            CertificateError, "analytical boundary"
        ):
            validate_certificate_document(document)

    def test_summary_without_explicit_exploratory_state_is_rejected(
        self,
    ) -> None:
        summary = self._summary()
        del summary["validation"]
        with self.assertRaisesRegex(CertificateError, "unknown or conflicting"):
            attach_certificate_to_summary(summary, self._verify())

    def test_stale_summary_commit_is_rejected(self) -> None:
        summary = self._summary()
        summary["simulator"]["git_commit"] = "0" * 40
        with self.assertRaisesRegex(CertificateError, "commit"):
            attach_certificate_to_summary(summary, self._verify())


if __name__ == "__main__":
    unittest.main()
