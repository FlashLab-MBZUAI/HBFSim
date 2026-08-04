#!/usr/bin/env python3
"""Contract tests for fail-closed external differential evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.compare_ledgers import load_ledger  # noqa: E402
from validation.contracts import (  # noqa: E402
    ContractError,
    dump_json_line,
    load_case,
)
from validation.external_differential import (  # noqa: E402
    EVIDENCE_SCHEMA,
    SOURCE_SCHEMA,
    WRAPPER_SCHEMA,
    build_report,
    load_tools,
    normalize_wrapper,
    validate_actual_evidence,
    validate_fixture,
)
from validation.hbf_oracle import build_ledger as build_hbf_ledger  # noqa: E402
from validation.hbm_oracle import build_ledger as build_hbm_ledger  # noqa: E402
from validation.ledger_reducer import reduce_ledger  # noqa: E402
from validation.generate_certificate import (  # noqa: E402
    _validate_external_report,
)
from validation.run_actual_external_differential import (  # noqa: E402
    _ramulator_configure_command,
)


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "validation/external/tools.json"
FIXTURES = ROOT / "validation/external/fixtures"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


class ExternalDifferentialContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = load_tools(MANIFEST)

    def test_actual_ramulator_build_is_network_independent(self) -> None:
        source = Path("/qualified/ramulator2")
        command = _ramulator_configure_command(
            build_source=source,
            cmake_build=Path("/isolated/build"),
            cxx="/qualified/c++",
        )
        self.assertIn(
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            command,
        )
        self.assertIn(
            "-DFETCHCONTENT_SOURCE_DIR_FMT="
            "/qualified/ramulator2/ext/fmt",
            command,
        )
        self.assertIn(
            "-DFETCHCONTENT_SOURCE_DIR_YAML_CPP="
            "/qualified/ramulator2/ext/yaml-cpp",
            command,
        )

    def _ramulator_raw(
        self,
        metrics: dict[str, int | float],
    ) -> dict[str, int | float]:
        clock_period_ns = 0.625
        service_span_ns = float(metrics["service_span_ns"])
        cycles = round(service_span_ns / clock_period_ns)
        self.assertAlmostEqual(
            cycles * clock_period_ns, service_span_ns, places=12)
        return {
            "clock_period_ns": clock_period_ns,
            "request_epoch_cycle": 1,
            "completion_cycle": 1 + cycles,
            "transaction_bytes": 64,
            "read_requests": int(metrics["read_bytes"]) // 64,
            "write_requests": int(metrics["write_bytes"]) // 64,
            "row_hits": int(metrics["row_hits"]),
            "row_misses": int(metrics["row_misses"]),
            "row_conflicts": int(metrics["row_conflicts"]),
        }

    def _make_ramulator_bundle(
        self,
        root: Path,
    ) -> tuple[dict[str, Any], Path, Path, Path]:
        """Build synthetic evidence that exercises plumbing, never L3 claims."""
        tool = next(
            tool for tool in self.tools if tool["id"] == "ramulator2")
        evidence_root = root / "evidence"
        evidence_root.mkdir()
        reference_probe = root / "current-validation-probe"
        reference_probe.write_bytes(b"current validation probe")
        evidence_probe = evidence_root / "hbfsim-validation-probe"
        shutil.copy2(reference_probe, evidence_probe)

        source_path = evidence_root / "ramulator2/source-provenance.json"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(json.dumps({
            "schema": SOURCE_SCHEMA,
            "repository": tool["repository"],
            "commit": tool["commit"],
            "tree": tool["source_tree"],
            "tracked_clean": True,
            "dependencies": tool["dependencies"],
        }))

        build_dir = evidence_root / "ramulator2/build"
        build_dir.mkdir()
        build_artifacts: dict[str, Any] = {}
        for name in tool["required_build_artifacts"]:
            path = build_dir / name
            path.write_bytes(f"synthetic {name}".encode())
            build_artifacts[name] = _artifact(evidence_root, path)

        case_entries = []
        for case_spec in tool["cases"]:
            case_id = case_spec["case_id"]
            case_dir = evidence_root / "ramulator2/cases" / case_id
            case_dir.mkdir(parents=True)
            artifact_paths: dict[str, Path] = {}

            case_source = ROOT / case_spec["hbfsim_case"]
            case_copy = case_dir / "hbfsim-case.json"
            shutil.copy2(case_source, case_copy)
            artifact_paths["hbfsim_case"] = case_copy
            for name, repository_path in case_spec[
                "external_inputs"
            ].items():
                source = ROOT / repository_path
                destination = case_dir / f"{name}{source.suffix}"
                shutil.copy2(source, destination)
                artifact_paths[name] = destination

            case = load_case(case_copy)
            ledger = (
                build_hbm_ledger(case)
                if case["model"] == "hbm"
                else build_hbf_ledger(case)
            )
            ledger_path = case_dir / "hbfsim-ledger.jsonl"
            ledger_path.write_text("".join(
                dump_json_line(record) + "\n" for record in ledger
            ))
            self.assertEqual(load_ledger(ledger_path), ledger)
            artifact_paths["hbfsim_ledger"] = ledger_path
            reduced = reduce_ledger(ledger)["metrics"]
            shared_metrics = {
                external_name: reduced[ledger_name]
                for external_name, ledger_name
                in case_spec["hbfsim_metrics"].items()
            }
            wrapper_path = case_dir / "raw-wrapper.json"
            wrapper_path.write_text(json.dumps({
                "schema": WRAPPER_SCHEMA,
                "adapter": tool["adapter"],
                "tool_id": tool["id"],
                "tool_commit": tool["commit"],
                "provenance": "actual",
                "case_id": case_id,
                "raw": self._ramulator_raw(shared_metrics),
            }))
            artifact_paths["raw_output"] = wrapper_path
            run_log = case_dir / "run.log"
            run_log.write_text("synthetic plumbing test only\n")
            artifact_paths["run_log"] = run_log

            case_entries.append({
                "case_id": case_id,
                "commands": {
                    "external": ["./synthetic-ramulator", case_id],
                    "hbfsim": ["./synthetic-probe", case_id],
                },
                "artifacts": {
                    name: _artifact(evidence_root, artifact_paths[name])
                    for name in tool["required_case_artifacts"]
                },
            })

        evidence = {
            "schema": EVIDENCE_SCHEMA,
            "tool_id": tool["id"],
            "tool_commit": tool["commit"],
            "adapter": tool["adapter"],
            "provenance": "actual",
            "source": _artifact(evidence_root, source_path),
            "build": {
                "compiler": {
                    "path": "/synthetic/c++",
                    "version": "synthetic compiler 1",
                },
                "commands": [["synthetic-build", tool["commit"]]],
                "artifacts": build_artifacts,
            },
            "hbfsim_probe": _artifact(evidence_root, evidence_probe),
            "cases": case_entries,
        }
        evidence_path = evidence_root / "ramulator2.evidence.json"
        evidence_path.write_text(json.dumps(evidence))
        return tool, evidence_path, reference_probe, evidence_root

    def test_pinned_adapter_fixtures_normalize_exactly(self) -> None:
        normalized = {
            tool["id"]: validate_fixture(tool, FIXTURES)
            for tool in self.tools
        }
        self.assertEqual(
            normalized["ramulator2"]["metrics"]["service_span_ns"], 62.5)
        self.assertEqual(
            normalized["mqsim"]["metrics"]["data_write_bytes"], 2560)

    def test_no_evidence_is_explicit_not_run(self) -> None:
        report = build_report(MANIFEST, FIXTURES, None)
        self.assertEqual(report["l3_status"], "not_run")
        self.assertEqual(report["validated_facets"], [])
        self.assertTrue(all(
            entry["fixture"]["status"] == "pass"
            and entry["actual"]["status"] == "not_run"
            for entry in report["tools"]
        ))

    def test_fixture_cannot_be_relabeled_actual(self) -> None:
        tool = self.tools[0]
        wrapper = json.loads(
            (FIXTURES / tool["fixture"]["raw"]).read_text())
        with self.assertRaisesRegex(
            ContractError, "provenance.*expected 'actual'"
        ):
            normalize_wrapper(
                wrapper,
                tool,
                require_provenance="actual",
                source="<fixture>",
            )

    def test_commit_drift_is_fail_closed(self) -> None:
        tool = self.tools[0]
        wrapper = json.loads(
            (FIXTURES / tool["fixture"]["raw"]).read_text())
        wrapper["tool_commit"] = "0" * 40
        with self.assertRaisesRegex(ContractError, "tool_commit.*expected"):
            normalize_wrapper(
                wrapper,
                tool,
                require_provenance="fixture",
                source="<drift>",
            )

    def test_unknown_raw_metric_is_rejected(self) -> None:
        tool = self.tools[0]
        wrapper = json.loads(
            (FIXTURES / tool["fixture"]["raw"]).read_text())
        wrapper["raw"]["looks_close_enough"] = True
        with self.assertRaisesRegex(ContractError, "unknown keys"):
            normalize_wrapper(
                wrapper,
                tool,
                require_provenance="fixture",
                source="<unknown>",
            )

    def test_mqsim_gc_is_outside_the_validated_facet(self) -> None:
        tool = next(tool for tool in self.tools if tool["id"] == "mqsim")
        wrapper = json.loads(
            (FIXTURES / tool["fixture"]["raw"]).read_text())
        wrapper["raw"]["gc_executions"] = 1
        with self.assertRaisesRegex(ContractError, "direct page I/O only"):
            normalize_wrapper(
                wrapper,
                tool,
                require_provenance="fixture",
                source="<gc-out-of-scope>",
            )

    def test_malformed_supplied_evidence_is_failure_not_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence_dir = Path(directory)
            (evidence_dir / "ramulator2.evidence.json").write_text("{}")
            with self.assertRaises(ContractError):
                build_report(MANIFEST, FIXTURES, evidence_dir)

    def test_manifest_requires_full_pins_and_exact_case_census(self) -> None:
        for tool in self.tools:
            self.assertRegex(tool["commit"], r"^[0-9a-f]{40,64}$")
            self.assertRegex(tool["source_tree"], r"^[0-9a-f]{40,64}$")
            declared = [
                case_id
                for facet in tool["facets"]
                for case_id in facet["cases"]
            ]
            self.assertEqual(
                declared, [case["case_id"] for case in tool["cases"]])
            for case in tool["cases"]:
                self.assertTrue((ROOT / case["hbfsim_case"]).is_file())
                for path in case["external_inputs"].values():
                    self.assertTrue((ROOT / path).is_file())

        data = json.loads(MANIFEST.read_text())
        data["tools"][0]["commit"] = "master"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tools.json"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ContractError, "full Git SHA"):
                load_tools(path)

    def test_actual_bundle_binds_all_cases_inputs_and_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool, evidence_path, reference_probe, _ = (
                self._make_ramulator_bundle(Path(directory)))
            result = validate_actual_evidence(
                tool,
                evidence_path,
                repository=ROOT,
                expected_hbfsim_probe=reference_probe,
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                [case["case_id"] for case in result["cases"]],
                [case["case_id"] for case in tool["cases"]],
            )

    def test_missing_actual_case_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool, evidence_path, reference_probe, _ = (
                self._make_ramulator_bundle(Path(directory)))
            evidence = json.loads(evidence_path.read_text())
            evidence["cases"].pop()
            evidence_path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ContractError, "ordered census"):
                validate_actual_evidence(
                    tool,
                    evidence_path,
                    repository=ROOT,
                    expected_hbfsim_probe=reference_probe,
                )

    def test_artifact_digest_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool, evidence_path, reference_probe, evidence_root = (
                self._make_ramulator_bundle(Path(directory)))
            evidence = json.loads(evidence_path.read_text())
            relative = evidence["cases"][0]["artifacts"][
                "raw_output"]["path"]
            (evidence_root / relative).write_text("{}")
            with self.assertRaisesRegex(ContractError, "sha256|bytes"):
                validate_actual_evidence(
                    tool,
                    evidence_path,
                    repository=ROOT,
                    expected_hbfsim_probe=reference_probe,
                )

    def test_current_external_input_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool, evidence_path, reference_probe, evidence_root = (
                self._make_ramulator_bundle(Path(directory)))
            evidence = json.loads(evidence_path.read_text())
            record = evidence["cases"][0]["artifacts"]["config"]
            copied_config = evidence_root / record["path"]
            copied_config.write_text(copied_config.read_text() + "\n# drift\n")
            evidence["cases"][0]["artifacts"]["config"] = _artifact(
                evidence_root, copied_config)
            evidence_path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(
                ContractError, "differs from current manifest input"
            ):
                validate_actual_evidence(
                    tool,
                    evidence_path,
                    repository=ROOT,
                    expected_hbfsim_probe=reference_probe,
                )

    def test_probe_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool, evidence_path, reference_probe, evidence_root = (
                self._make_ramulator_bundle(Path(directory)))
            evidence = json.loads(evidence_path.read_text())
            probe_path = evidence_root / evidence["hbfsim_probe"]["path"]
            probe_path.write_bytes(b"different probe")
            evidence["hbfsim_probe"] = _artifact(evidence_root, probe_path)
            evidence_path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ContractError, "probe digest differs"):
                validate_actual_evidence(
                    tool,
                    evidence_path,
                    repository=ROOT,
                    expected_hbfsim_probe=reference_probe,
                )

    def test_source_tree_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool, evidence_path, reference_probe, evidence_root = (
                self._make_ramulator_bundle(Path(directory)))
            evidence = json.loads(evidence_path.read_text())
            source_path = evidence_root / evidence["source"]["path"]
            source = json.loads(source_path.read_text())
            source["tree"] = "0" * 40
            source_path.write_text(json.dumps(source))
            evidence["source"] = _artifact(evidence_root, source_path)
            evidence_path.write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ContractError, "tree.*expected"):
                validate_actual_evidence(
                    tool,
                    evidence_path,
                    repository=ROOT,
                    expected_hbfsim_probe=reference_probe,
                )

    def test_certificate_import_preserves_partial_facet_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, reference_probe, evidence_root = (
                self._make_ramulator_bundle(root))
            report = build_report(
                MANIFEST,
                FIXTURES,
                evidence_root,
                repository=ROOT,
                expected_hbfsim_probe=reference_probe,
            )
            report_path = root / "external-report.json"
            report_path.write_text(json.dumps(report))
            gate = _validate_external_report(
                report_path, manifest_path=MANIFEST)
            self.assertEqual(gate["l3_status"], "partial")
            self.assertEqual(gate["cases_required"], 6)
            self.assertEqual(gate["cases_passed"], 3)
            self.assertEqual(
                [facet["id"] for facet in gate["validated_facets"]],
                ["dram.read-row-state-service"],
            )

            report["tools"][0]["source_tree"] = "0" * 40
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(
                ContractError, "source_tree.*current manifest"
            ):
                _validate_external_report(
                    report_path, manifest_path=MANIFEST)


if __name__ == "__main__":
    unittest.main()
