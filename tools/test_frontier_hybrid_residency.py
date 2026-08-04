#!/usr/bin/env python3
"""Independent contract tests for Frontier hybrid-residency arithmetic."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from frontier_hybrid_residency import (
    HybridResidencyError,
    build_hybrid_residency_plan,
    validate_hybrid_residency_plan,
)


W8_WEIGHT_BYTES = 70_568_973_312
W8_ACTIVE_BUFFER_BYTES = 1_050_929_664
RUNTIME_OVERHEAD_BYTES = 8 * 1024**3


def plan(capacity_gib: int, pressure: float) -> dict[str, object]:
    return build_hybrid_residency_plan(
        physical_hbm_capacity_bytes=capacity_gib * 1024**3,
        target_pressure=pressure,
        immutable_weight_backing_bytes=W8_WEIGHT_BYTES,
        runtime_overhead_bytes=RUNTIME_OVERHEAD_BYTES,
        active_weight_buffer_bytes_per_slot=W8_ACTIVE_BUFFER_BYTES,
        kv_block_size_tokens=16,
        kv_page_bytes_per_layer=65_536,
        num_layers=80,
    )


class FrontierHybridResidencyTest(unittest.TestCase):
    def test_grid_conserves_three_disjoint_ledgers(self) -> None:
        for capacity_gib in (96, 192, 288, 384):
            for pressure in (0.75, 1.0, 1.25, 1.5, 2.0):
                if capacity_gib == 96 and pressure == 0.75:
                    with self.assertRaisesRegex(
                        HybridResidencyError,
                        "cannot contain immutable weights",
                    ):
                        plan(capacity_gib, pressure)
                    continue
                value = plan(capacity_gib, pressure)
                self.assertEqual(
                    value["target_footprint_bytes"],
                    value["unique_resident_footprint_bytes"]
                    + value["target_rounding_slack_bytes"],
                )
                self.assertLess(
                    value["target_rounding_slack_bytes"],
                    value["kv_block_stride_bytes"]
                    + value["block_table_entry_bytes"],
                )
                self.assertEqual(
                    value["unique_resident_footprint_bytes"],
                    value["immutable_weight_backing_bytes"]
                    + value["runtime_overhead_bytes"]
                    + value["block_table_bytes"]
                    + value["logical_kv_bytes"],
                )
                self.assertEqual(
                    value["num_logical_kv_blocks"],
                    value["hot_kv_blocks"] + value["cold_kv_blocks"],
                )
                self.assertEqual(
                    value["logical_kv_bytes"],
                    value["hot_kv_bytes"] + value["cold_kv_bytes"],
                )
                self.assertEqual(
                    value["physical_hbm_capacity_bytes"],
                    value["fixed_hbm_bytes"]
                    + value["hot_kv_bytes"]
                    + value["unused_hbm_bytes"],
                )
                self.assertEqual(
                    value["backing_bytes"],
                    value["immutable_weight_backing_bytes"]
                    + value["cold_kv_bytes"],
                )

    def test_exact_96_gib_pressure_one_plan(self) -> None:
        value = plan(96, 1.0)
        self.assertEqual(value["num_logical_kv_blocks"], 4_562)
        self.assertEqual(value["hot_kv_blocks"], 4_562)
        self.assertEqual(value["cold_kv_blocks"], 0)
        self.assertEqual(
            value["unique_resident_footprint_bytes"],
            103_076_944_712,
        )
        self.assertEqual(value["target_rounding_slack_bytes"], 2_270_392)

    def test_validator_rejects_every_plan_mutation(self) -> None:
        canonical = plan(192, 1.5)
        arguments = {
            "physical_hbm_capacity_bytes": 192 * 1024**3,
            "target_pressure": 1.5,
            "immutable_weight_backing_bytes": W8_WEIGHT_BYTES,
            "runtime_overhead_bytes": RUNTIME_OVERHEAD_BYTES,
            "active_weight_buffer_bytes_per_slot": W8_ACTIVE_BUFFER_BYTES,
            "kv_block_size_tokens": 16,
            "kv_page_bytes_per_layer": 65_536,
            "num_layers": 80,
        }
        self.assertEqual(
            validate_hybrid_residency_plan(canonical, **arguments),
            canonical,
        )
        for field in canonical:
            mutated = copy.deepcopy(canonical)
            value = mutated[field]
            if isinstance(value, bool):
                mutated[field] = not value
            elif isinstance(value, int):
                mutated[field] = value + 1
            else:
                mutated[field] = f"{value}-mutated"
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    HybridResidencyError,
                    "differs from independent arithmetic",
                ):
                    validate_hybrid_residency_plan(
                        mutated,
                        **arguments,
                    )


if __name__ == "__main__":
    unittest.main()
