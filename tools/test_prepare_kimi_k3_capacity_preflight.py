#!/usr/bin/env python3
"""Contract tests for the Kimi K3 TP8 capacity-only preflight."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from prepare_kimi_k3_capacity_preflight import (  # noqa: E402
    DEFAULT_DESCRIPTOR,
    DEFAULT_STUDY,
    RECEIPT_SCHEMA,
    KimiK3PreflightError,
    _validate_descriptor,
    _validate_study,
    prepare_preflight,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class KimiK3CapacityPreflightTest(unittest.TestCase):
    def test_canonical_preflight_reproduces_capacity_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            payload = prepare_preflight(
                descriptor_path=DEFAULT_DESCRIPTOR,
                study_path=DEFAULT_STUDY,
                output_dir=output,
            )
            self.assertEqual(payload["schema"], RECEIPT_SCHEMA)
            self.assertEqual(payload["census"]["rows"], 9)
            self.assertEqual(payload["census"]["all_hbm_feasible_rows"], 3)
            self.assertEqual(payload["census"]["hybrid_feasible_rows"], 9)
            self.assertEqual(
                payload["hybrid_capacity_policy"][
                    "hybrid_hbm_base_bytes_before_runtime_overhead"
                ],
                43_671_586_958,
            )
            self.assertEqual(
                payload["hybrid_capacity_policy"]["routed_expert_backing_bytes"],
                180_807_000_000,
            )
            self.assertFalse(payload["eligibility"]["paper_result_eligible"])
            with (output / "capacity.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            row_96_8 = next(
                row
                for row in rows
                if row["hbm_capacity_gib_per_rank"] == "96"
                and row["runtime_overhead_gib_per_rank"] == "8"
            )
            self.assertEqual(row_96_8["all_hbm_capacity_feasible"], "False")
            self.assertEqual(row_96_8["hybrid_capacity_feasible"], "True")
            self.assertEqual(row_96_8["memory_service_feasible"], "undetermined")

    def test_architecture_mutation_is_rejected(self) -> None:
        descriptor = json.loads(DEFAULT_DESCRIPTOR.read_text(encoding="utf-8"))
        descriptor["model"]["architecture"]["selected_experts_per_token"] = 15
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "descriptor.json"
            _write_json(path, descriptor)
            with self.assertRaisesRegex(KimiK3PreflightError, "architecture"):
                _validate_descriptor(path)

    def test_missing_research_question_is_rejected(self) -> None:
        study = json.loads(DEFAULT_STUDY.read_text(encoding="utf-8"))
        study["questions"] = copy.deepcopy(study["questions"][:2])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "study.json"
            study["model_descriptor"] = str(DEFAULT_DESCRIPTOR.resolve())
            _write_json(path, study)
            with self.assertRaisesRegex(KimiK3PreflightError, "Q1/Q2/Q3"):
                _validate_study(path, DEFAULT_DESCRIPTOR)


if __name__ == "__main__":
    unittest.main()
