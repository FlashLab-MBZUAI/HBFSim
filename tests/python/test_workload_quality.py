#!/usr/bin/env python3
"""Regression tests for fail-closed behavioral workload quality gates."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from workloads.trace_analysis import analyze_trace  # noqa: E402
from workloads.quality import (  # noqa: E402
    QualityContract,
    assess,
)


class WorkloadQualityTests(unittest.TestCase):
    def analyze(self, pages: list[int], *, hbm_pages: int = 2) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.txt"
            trace.write_text(
                "".join(
                    f"0x{page * 4096:x} R 4096 at={index}\n"
                    for index, page in enumerate(pages)),
                encoding="utf-8",
            )
            return analyze_trace(
                trace,
                hbm_capacity_bytes=hbm_pages * 4096,
            )

    def test_reuse_positive_capacity_pressure_passes(self) -> None:
        locality = self.analyze([0, 1, 0, 1, 2, 0, 1, 2])
        result = assess(
            locality,
            QualityContract(
                min_operations=8,
                min_unique_pages=3,
                min_page_touches_per_unique_page=2.0,
                max_address_span_to_occupied_page_ratio=1.0,
                reuse_expectation="present",
                min_reuse_page_touches=4,
                working_set_relation="exceeds-hbm",
                min_reuse_within_hbm_ratio=0.4,
            ),
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_small_sparse_trace_fails_with_specific_checks(self) -> None:
        locality = self.analyze([0, 1000], hbm_pages=4)
        result = assess(
            locality,
            QualityContract(
                min_operations=8,
                min_unique_pages=4,
                reuse_expectation="present",
                min_reuse_page_touches=2,
                working_set_relation="exceeds-hbm",
            ),
        )
        self.assertEqual(result["status"], "FAIL")
        failed = {
            check["name"] for check in result["checks"]
            if not check["passed"]
        }
        self.assertTrue({
            "operation_count",
            "unique_occupied_pages",
            "address_span_to_occupied_page_ratio",
            "reuse_page_touches",
            "working_set_relation",
        }.issubset(failed))

    def test_scan_negative_control_can_require_zero_reuse(self) -> None:
        locality = self.analyze(list(range(8)), hbm_pages=2)
        result = assess(
            locality,
            QualityContract(
                min_operations=8,
                min_unique_pages=8,
                reuse_expectation="none",
                working_set_relation="exceeds-hbm",
                max_address_span_to_occupied_page_ratio=1.0,
            ),
        )
        self.assertEqual(result["status"], "PASS")

    def test_invalid_contract_fails_closed(self) -> None:
        locality = self.analyze([0])
        with self.assertRaisesRegex(ValueError, "at least one"):
            assess(
                locality,
                QualityContract(
                    min_page_touches_per_unique_page=0.5),
            )


if __name__ == "__main__":
    unittest.main()
