#!/usr/bin/env python3
"""Contract tests for the three-stratum Frontier 70B evidence verifier."""

from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from run_frontier_70b_two_axis_slice import (  # noqa: E402
    EXPECTED_BASELINES,
    EXPECTED_CELLS,
    INPUT_SCHEMA,
    RECEIPT_SCHEMA as SLICE_RECEIPT_SCHEMA,
    _expected_batch_row,
    _result_rows,
)
from run_frontier_memory_baselines import SCHEMA as BASELINE_SCHEMA  # noqa: E402
from verify_frontier_70b_two_axis_evidence import (  # noqa: E402
    CERTIFICATE_SCHEMA,
    METRIC_FIELDS,
    EvidenceError,
    verify_evidence,
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _artifact(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class EvidenceFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        shared = root / "shared"
        shared.mkdir()
        self.runner = shared / "runner.py"
        self.runner.write_text("# runner\n", encoding="utf-8")
        self.study = shared / "study.json"
        _write_json(self.study, {"study_id": "fixture"})
        self.binary = shared / "scenario_compare"
        self.binary.write_bytes(b"fixture-binary\n")
        self.hardware = {}
        for topology in ("6h2f", "4h4f", "2h6f"):
            path = shared / f"{topology}.cfg"
            path.write_text(f"# {topology}\n", encoding="utf-8")
            self.hardware[topology] = _artifact(path)
        request_csv = shared / "requests.csv"
        request_csv.write_text("request_id\n0\n", encoding="utf-8")
        request_verification = shared / "request-verification.json"
        _write_json(request_verification, {"result": "pass"})
        self.request_identity = {
            "request_csv_bytes": request_csv.stat().st_size,
            "request_csv_sha256": _artifact(request_csv)["sha256"],
            "request_suite_verification_bytes": request_verification.stat().st_size,
            "request_suite_verification_sha256": _artifact(request_verification)["sha256"],
        }
        self.shared_transitive = {
            "frontier.request_csv": _artifact(request_csv),
            "frontier.request_suite_verification": _artifact(request_verification),
        }

    def _population(self, directory: Path, name: str, quantile: str) -> dict[str, object]:
        population_dir = directory / name
        population_dir.mkdir(parents=True)
        manifest = population_dir / "manifest.json"
        _write_json(manifest, {"population": name, "quantile": quantile})
        object_map = population_dir / "object-map.json"
        _write_json(object_map, {"population": name})
        trace = population_dir / "memory.trace"
        trace.write_text(f"0x0 R 4096 population={name} q={quantile}\n", encoding="utf-8")
        total = 101
        return {
            "manifest": _artifact(manifest),
            "object_map": _artifact(object_map),
            "trace": _artifact(trace),
            "transitive_artifacts": dict(self.shared_transitive),
            "selection": {
                "quantile": quantile,
                "total_frontier_batches": total,
                "batch_row": _expected_batch_row(total, Decimal(quantile)),
                "batch_ids": [int(Decimal(quantile) * 100)],
            },
            "population": {
                "unique_resident_footprint_bytes": 100,
                "logical_kv_blocks": 4,
                "immutable_weight_backing_bytes": 40,
                "runtime_overhead_bytes": 20,
                "block_table_bytes": 4,
            },
            "request_identity": dict(self.request_identity),
        }

    @staticmethod
    def _metrics(seed: int) -> dict[str, object]:
        return {name: seed + index for index, name in enumerate(METRIC_FIELDS)}

    def make_slice(self, quantile: str) -> Path:
        directory = self.root / f"q{quantile.replace('.', '_')}"
        directory.mkdir()
        populations = {
            name: self._population(directory, name, quantile)
            for name in ("p0_75", "p1_0", "p1_25")
        }
        inputs = {
            "schema": INPUT_SCHEMA,
            "result": "pass",
            "quantile": quantile,
            "runner": _artifact(self.runner),
            "study": _artifact(self.study),
            "scenario_compare": _artifact(self.binary),
            "scenario_compare_version": "scenario_compare fixture (git " + "a" * 40 + ")",
            "hbfsim_revision": "a" * 40,
            "hardware_configs": self.hardware,
            "populations": populations,
            "request_identity": self.request_identity,
            "credit_limit": 64,
            "timeout_seconds": None,
        }
        inputs_path = directory / "slice.inputs.json"
        _write_json(inputs_path, inputs)
        cells: dict[str, object] = {}
        result_rows: list[dict[str, object]] = []
        for cell_index, cell in enumerate(EXPECTED_CELLS):
            population = populations[cell["population_source"]]
            baseline_path = (
                directory
                / "cells"
                / cell["cell_id"]
                / "baselines"
                / "baseline-set.receipt.json"
            )
            baseline_receipt = {
                "schema": BASELINE_SCHEMA,
                "result": "pass",
                "source": {
                    "manifest": population["manifest"],
                    "object_map": population["object_map"],
                    "trace": population["trace"],
                },
                "contract": {
                    "traffic": {"ops": 10, "logical_bytes": 40960},
                    "residency": {
                        "unique_resident_footprint_bytes": 100,
                        "physical_hbm_capacity_bytes": 200,
                        "capacity_pressure": 0.5,
                        "logical_kv_blocks": 4,
                        "hot_kv_blocks": 3,
                        "cold_kv_blocks": 1,
                    },
                    "canonical_hbf_waf": {
                        "logical_write_bytes": 0,
                        "physical_write_bytes": 0,
                        "waf": None,
                        "physical_write_bytes_over_usable_hbf_capacity": 0,
                    },
                },
                "baselines": {
                    baseline: {
                        "scenario": baseline,
                        "memory_service_metrics": self._metrics(
                            cell_index * 100 + baseline_index * 20
                        ),
                    }
                    for baseline_index, baseline in enumerate(EXPECTED_BASELINES)
                },
            }
            _write_json(baseline_path, baseline_receipt)
            cells[cell["cell_id"]] = {
                "axis": cell["axis"],
                "topology": cell["topology"],
                "population_source": cell["population_source"],
                "baseline_receipt": _artifact(baseline_path),
            }
            result_rows.extend(
                _result_rows(
                    cell=cell,
                    population_name=cell["population_source"],
                    population=population,
                    receipt=baseline_receipt,
                )
            )
        table_path = directory / "results.csv"
        _write_table(table_path, result_rows)
        receipt = {
            "schema": SLICE_RECEIPT_SCHEMA,
            "result": "pass",
            "study_id": "fixture",
            "quantile": quantile,
            "claim_scope": {"eligible": "memory_system_service_only"},
            "source": {"inputs": _artifact(inputs_path)},
            "contract": {
                "cell_count": 5,
                "baseline_runs": 20,
                "mix_ratio_same_logical_trace": True,
                "mix_ratio_same_address_population": True,
                "mix_ratio_capacity_pressure_is_observed": True,
                "pressure_axis_fixed_topology": "4h4f",
                "hbm_hbf_semantics": "HBM+HBF_hybrid_layer_streaming",
                "pure_all_hbf_included": False,
                "temporal_stratum_is_partial_trace": True,
                "aggregate_across_quantiles": False,
            },
            "cells": cells,
            "table": _artifact(table_path),
            "eligibility": {
                "five_cell_contract_complete": True,
                "twenty_baseline_runs_complete": True,
                "mix_ratio_byte_identity_verified": True,
                "full_frontier_replays_required_upstream": True,
                "memory_trace_is_temporal_stratum": True,
                "paper_result_eligible": False,
            },
        }
        receipt_path = directory / "slice.receipt.json"
        _write_json(receipt_path, receipt)
        return receipt_path


class Frontier70BTwoAxisEvidenceTest(unittest.TestCase):
    def test_exact_three_strata_produce_stratified_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            receipts = {q: fixture.make_slice(q) for q in ("0", "0.5", "1")}
            output = root / "certificate"
            payload = verify_evidence(receipt_paths=receipts, output_dir=output)
            self.assertEqual(payload["schema"], CERTIFICATE_SCHEMA)
            self.assertEqual(payload["contract"]["baseline_rows"], 60)
            self.assertEqual(
                payload["contract"]["aggregation"],
                "concatenation_only_no_cross_quantile_statistics",
            )
            self.assertFalse(payload["eligibility"]["paper_result_eligible"])
            with (output / "stratified-results.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 60)

    def test_claim_promotion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            receipts = {q: fixture.make_slice(q) for q in ("0", "0.5", "1")}
            promoted = json.loads(receipts["0.5"].read_text(encoding="utf-8"))
            promoted["eligibility"]["paper_result_eligible"] = True
            _write_json(receipts["0.5"], promoted)
            with self.assertRaisesRegex(EvidenceError, "eligibility"):
                verify_evidence(
                    receipt_paths=receipts,
                    output_dir=root / "certificate",
                )

    def test_non_mechanical_quantile_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = EvidenceFixture(root)
            receipts = {q: fixture.make_slice(q) for q in ("0", "0.5", "1")}
            receipt = json.loads(receipts["0.5"].read_text(encoding="utf-8"))
            inputs_path = Path(receipt["source"]["inputs"]["path"])
            inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
            inputs["populations"]["p1_0"]["selection"]["batch_row"] = 49
            _write_json(inputs_path, inputs)
            receipt["source"]["inputs"] = _artifact(inputs_path)
            _write_json(receipts["0.5"], receipt)
            with self.assertRaisesRegex(EvidenceError, "not mechanical"):
                verify_evidence(
                    receipt_paths=receipts,
                    output_dir=root / "certificate",
                )


if __name__ == "__main__":
    unittest.main()
