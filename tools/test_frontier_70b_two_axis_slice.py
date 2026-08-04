#!/usr/bin/env python3
"""Contract tests for the fixed 70B two-axis representative slice."""

from __future__ import annotations

import copy
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
    BASELINE_SCHEMA,
    DEFAULT_STUDY,
    EXPECTED_BASELINES,
    EXPECTED_CELLS,
    EXPECTED_HBM_CAPACITY,
    TwoAxisSliceError,
    _expected_batch_row,
    _manifest_input,
    _validate_baseline_receipt,
    _validate_study,
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _artifact(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class Frontier70BTwoAxisSliceTest(unittest.TestCase):
    def test_canonical_study_has_five_orthogonal_cells(self) -> None:
        study = _validate_study(DEFAULT_STUDY)
        self.assertEqual(study["cells"], list(EXPECTED_CELLS))
        self.assertEqual(len(study["cells"]), 5)
        self.assertEqual(
            {
                cell["topology"]
                for cell in study["cells"][:3]
            },
            {"6h2f", "4h4f", "2h6f"},
        )
        self.assertEqual(
            {
                cell["population_source"]
                for cell in study["cells"][:3]
            },
            {"p1_0"},
        )
        self.assertEqual(
            study["baseline_semantics"]["hbm_hbf"],
            "HBM+HBF_hybrid_layer_streaming",
        )
        self.assertFalse(
            study["baseline_semantics"]["pure_all_hbf_included"]
        )
        self.assertEqual(
            study["execution"],
            {
                "default_cell_parallelism": 1,
                "maximum_cell_parallelism": 5,
                "within_cell": (
                    "two_cases_concurrent_four_baselines_atomic"
                ),
            },
        )

    def test_temporal_quantile_rows_are_mechanical(self) -> None:
        total = 189_903
        self.assertEqual(_expected_batch_row(total, Decimal("0")), 0)
        self.assertEqual(
            _expected_batch_row(total, Decimal("0.5")),
            94_951,
        )
        self.assertEqual(
            _expected_batch_row(total, Decimal("1")),
            total - 1,
        )

    def test_resume_rejects_a_different_simulator_binary(self) -> None:
        receipt = {
            "schema": BASELINE_SCHEMA,
            "result": "pass",
            "baselines": {name: {} for name in EXPECTED_BASELINES},
            "source": {
                "scenario_compare": {
                    "path": "/old/scenario_compare",
                    "bytes": 1,
                    "sha256": "0" * 64,
                },
            },
        }
        with self.assertRaisesRegex(
            TwoAxisSliceError,
            "different simulator binary",
        ):
            _validate_baseline_receipt(
                receipt=receipt,
                population={},
                topology="4h4f",
                scenario_compare={
                    "path": "/new/scenario_compare",
                    "bytes": 1,
                    "sha256": "1" * 64,
                },
            )

    def test_manifest_input_binds_selection_population_and_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "memory.trace"
            trace.write_text(
                "0x0 R 4096 kind=model_weights phase=0 layer=0\n",
                encoding="utf-8",
            )
            object_map = root / "object-map.json"
            object_map.write_bytes(_json_bytes({"regions": []}))
            audit = root / "audit.json"
            audit.write_bytes(_json_bytes({"result": "pass"}))
            descriptor = root / "model.json"
            descriptor.write_bytes(_json_bytes({"model": "llama31_70b"}))
            request_csv = root / "burst.frontier.csv"
            request_csv.write_text("request_id\n0\n", encoding="utf-8")
            verification = root / "suite.verification.json"
            verification.write_bytes(_json_bytes({"result": "pass"}))
            request_manifest = root / "request.manifest.json"
            request_manifest.write_bytes(_json_bytes({"result": "pass"}))
            frontier_artifacts = {
                "request_csv": _artifact(request_csv),
                "request_suite_verification": _artifact(verification),
                "request_manifest": _artifact(request_manifest),
            }
            manifest = {
                "schema": {
                    "name": "hbfsim.frontier_memory_trace",
                    "version": 4,
                },
                "model": {
                    "name": "llama31_70b",
                    "profile_id": "w8a16-kv-bf16",
                },
                "semantics": {
                    "eligible_claim_scope": "memory_system_service_only",
                    "ttft_tpot_slo_claims_eligible": False,
                    "time_based_throughput_claims_eligible": False,
                },
                "selection": {
                    "total_frontier_batches": 101,
                    "batch_start": 50,
                    "batches": 1,
                    "batch_ids": [77],
                    "is_full_replay": False,
                    "scaling_applied": False,
                    "byte_sampling_applied": False,
                },
                "capacity_accounting": {
                    "residency_plan": {
                        "physical_hbm_capacity_bytes": (
                            EXPECTED_HBM_CAPACITY["4h4f"]
                        ),
                        "target_pressure": "1.0",
                        "unique_resident_footprint_bytes": 123,
                        "num_logical_kv_blocks": 7,
                        "immutable_weight_backing_bytes": 80,
                        "runtime_overhead_bytes": 32,
                        "block_table_bytes": 11,
                    }
                },
                "outputs": {
                    "trace": _artifact(trace),
                    "object_map": _artifact(object_map),
                },
                "sources": {
                    "audit": _artifact(audit),
                    "model_descriptor": _artifact(descriptor),
                    "frontier_artifacts": frontier_artifacts,
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_bytes(_json_bytes(manifest))
            parsed = _manifest_input(
                population="p1_0",
                manifest_path=manifest_path,
                object_map_path=object_map,
                quantile=Decimal("0.5"),
            )
            self.assertEqual(parsed["selection"]["batch_row"], 50)
            self.assertEqual(parsed["selection"]["batch_ids"], [77])
            self.assertEqual(
                parsed["population"]["logical_kv_blocks"],
                7,
            )
            self.assertEqual(
                parsed["request_identity"]["request_csv_sha256"],
                _artifact(request_csv)["sha256"],
            )
            self.assertIn(
                "frontier.request_manifest",
                parsed["transitive_artifacts"],
            )

            wrong_selection = copy.deepcopy(manifest)
            wrong_selection["selection"]["batch_start"] = 49
            manifest_path.write_bytes(_json_bytes(wrong_selection))
            with self.assertRaisesRegex(
                TwoAxisSliceError,
                "predeclared",
            ):
                _manifest_input(
                    population="p1_0",
                    manifest_path=manifest_path,
                    object_map_path=object_map,
                    quantile=Decimal("0.5"),
                )

    def test_study_mutation_is_rejected(self) -> None:
        study = json.loads(DEFAULT_STUDY.read_text(encoding="utf-8"))
        study["baseline_semantics"]["hbm_hbf"] = "all-HBF"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "study.json"
            path.write_bytes(_json_bytes(study))
            with self.assertRaisesRegex(
                TwoAxisSliceError,
                "hbm_hbf",
            ):
                _validate_study(path)


if __name__ == "__main__":
    unittest.main()
