#!/usr/bin/env python3
"""Contracts for the qualified behavioral-placement experiment suite."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_trace_locality import analyze_trace  # noqa: E402
from assess_workload_quality import assess  # noqa: E402
from run_behavioral_placement_experiments import (  # noqa: E402
    PAGE_SIZE,
    _write_trace,
    build_cases,
)


class BehavioralExperimentPlanTests(unittest.TestCase):
    def test_case_keys_and_roles_are_stable_and_orthogonal(self) -> None:
        cases = build_cases("smoke", 32)
        self.assertEqual(
            [case.key for case in cases],
            [
                "sequential_scan",
                "random_scan",
                "hotset_fits",
                "capacity_edge",
                "cyclic_over_capacity",
                "hot_cold_pollution",
                "phase_shift",
                "skewed_read",
                "dirty_hotset_fits",
                "dirty_over_capacity",
            ],
        )
        self.assertEqual(len({case.key for case in cases}), len(cases))
        self.assertTrue(all(case.signature for case in cases))

    def test_every_smoke_workload_satisfies_its_own_quality_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in build_cases("smoke", 32):
                with self.subTest(case=case.key):
                    trace = root / f"{case.key}.trace"
                    _write_trace(trace, case.accesses)
                    locality = analyze_trace(
                        trace,
                        hbm_capacity_bytes=32 * PAGE_SIZE,
                    )
                    result = assess(locality, case.quality_contract)
                    self.assertEqual(
                        result["status"],
                        "PASS",
                        [item for item in result["checks"]
                         if not item["passed"]],
                    )

    def test_negative_and_positive_controls_have_expected_reuse(self) -> None:
        cases = {case.key: case for case in build_cases("smoke", 32)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = {}
            for key in (
                "sequential_scan",
                "random_scan",
                "hotset_fits",
                "cyclic_over_capacity",
                "hot_cold_pollution",
            ):
                trace = root / f"{key}.trace"
                _write_trace(trace, cases[key].accesses)
                reports[key] = analyze_trace(
                    trace,
                    hbm_capacity_bytes=32 * PAGE_SIZE,
                )
        self.assertEqual(
            reports["sequential_scan"]["temporal_page_locality"][
                "reuse_page_touches"
            ],
            0,
        )
        self.assertEqual(
            reports["random_scan"]["temporal_page_locality"][
                "reuse_page_touches"
            ],
            0,
        )
        self.assertEqual(
            reports["hotset_fits"]["temporal_page_locality"][
                "reuse_within_hbm_ratio"
            ],
            1.0,
        )
        self.assertEqual(
            reports["cyclic_over_capacity"]["temporal_page_locality"][
                "reuse_within_hbm_ratio"
            ],
            0.0,
        )
        self.assertEqual(
            reports["hot_cold_pollution"]["temporal_page_locality"][
                "reuse_within_hbm_ratio"
            ],
            0.0,
        )

    def test_core_profile_scales_operations_without_changing_roles(self) -> None:
        smoke = build_cases("smoke", 32)
        core = build_cases("core", 32)
        self.assertEqual(
            [(case.key, case.role) for case in smoke],
            [(case.key, case.role) for case in core],
        )
        for small, large in zip(smoke, core, strict=True):
            self.assertGreaterEqual(len(large.accesses), len(small.accesses))

    def test_invalid_tier_size_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least eight"):
            build_cases("smoke", 4)


if __name__ == "__main__":
    unittest.main()
