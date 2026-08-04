#!/usr/bin/env python3
"""Fail-closed tests for the canonical 70B Frontier structural runner."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from frontier_hybrid_residency import (  # noqa: E402
    build_hybrid_residency_plan,
)
from run_frontier_70b_structural_suite import (  # noqa: E402
    MODEL_NAME,
    PRIMARY_PRECISION_PROFILE,
    STRUCTURAL_CAPACITY_PRESSURE,
    STRUCTURAL_HBM_CAPACITY_BYTES,
    STRUCTURAL_RUNTIME_OVERHEAD_BYTES,
    StructuralSuiteError,
    WINDOWS,
    _build_frontier_command,
    _window_summary,
)


W8_WEIGHT_BYTES = 70_568_973_312
W8_ACTIVE_BUFFER_BYTES = 1_050_929_664


def _structural_plan() -> dict:
    return build_hybrid_residency_plan(
        physical_hbm_capacity_bytes=STRUCTURAL_HBM_CAPACITY_BYTES,
        target_pressure=STRUCTURAL_CAPACITY_PRESSURE,
        immutable_weight_backing_bytes=W8_WEIGHT_BYTES,
        runtime_overhead_bytes=STRUCTURAL_RUNTIME_OVERHEAD_BYTES,
        active_weight_buffer_bytes_per_slot=W8_ACTIVE_BUFFER_BYTES,
        kv_block_size_tokens=16,
        kv_page_bytes_per_layer=65_536,
        num_layers=80,
    )


class Frontier70BStructuralSuiteTest(unittest.TestCase):
    def test_command_is_the_single_pinned_multi_request_path(self) -> None:
        command = _build_frontier_command(
            frontier_python=Path("/frontier/.venv/bin/python"),
            request_csv=Path("/suite/steady.frontier.csv"),
            output_dir=Path("/evidence"),
            run_id="steady_w8",
        )
        command_text = " ".join(command)
        self.assertIn(
            "--replica_config_model_name " + MODEL_NAME,
            command_text,
        )
        self.assertIn(
            "--replica_config_memory_precision_profile "
            + PRIMARY_PRECISION_PROFILE,
            command_text,
        )
        self.assertIn(
            "--vllm_v1_scheduler_config_batch_size_cap 128",
            command_text,
        )
        self.assertIn(
            "--no-vllm_v1_scheduler_config_enable_preemption",
            command,
        )
        self.assertIn("--log_level warning", command_text)
        self.assertIn("--seed 42", command_text)
        self.assertNotIn("7b", command_text.lower())
        self.assertNotIn("batch299", command_text.lower())
        self.assertEqual(
            WINDOWS,
            ("steady", "burst", "long_context_decode_tail"),
        )

    @staticmethod
    def _valid_audit(window: str) -> dict:
        residency_plan = _structural_plan()
        capacity_blocks = residency_plan["num_logical_kv_blocks"]
        return {
            "result": "pass",
            "request_suite": {"window_id": window},
            "requests": {
                "count": 256,
                "prefill_tokens": 1000,
                "decode_tokens": 2000,
                "total_tokens": 3000,
            },
            "accounting": {"prefix_hit_ratio": 0.25},
            "ledger": {
                "rows": 10,
                "batch_size_counts": {"1": 2, "8": 8},
            },
            "kv_lifecycle": {
                "events": 100,
                "event_counts": {
                    "prefix_lookup": 256,
                    "prefix_admission": 256,
                },
                "no_preemption_admission": {
                    "policy": "full_request_kv_reservation_v1",
                    "capacity_blocks": capacity_blocks,
                    "max_reserved_blocks": capacity_blocks - 8,
                    "max_referenced_physical_blocks": capacity_blocks - 64,
                },
            },
            "eligibility": {
                "eligible_for_memory_system_service_claims": True,
                "eligible_for_ttft_tpot_slo_claims": False,
                "eligible_for_time_based_throughput_claims": False,
                "frontier_timing_kind": "dummy",
            },
            "model_memory": {
                "profile_id": PRIMARY_PRECISION_PROFILE,
            },
            "residency_plan": residency_plan,
        }

    def test_summary_requires_real_batching_and_dummy_claim_boundary(
        self,
    ) -> None:
        summary = _window_summary(
            self._valid_audit("steady"),
            "steady",
        )
        self.assertEqual(summary["max_batch_size"], 8)
        self.assertEqual(
            summary["claim_scope"],
            "memory_system_service_only",
        )
        self.assertEqual(
            summary["max_reserved_blocks"],
            _structural_plan()["num_logical_kv_blocks"] - 8,
        )

        single = self._valid_audit("steady")
        single["ledger"]["batch_size_counts"] = {"1": 10}
        with self.assertRaisesRegex(
            StructuralSuiteError,
            "multi-request batch",
        ):
            _window_summary(single, "steady")

        promoted = self._valid_audit("steady")
        promoted["eligibility"][
            "eligible_for_ttft_tpot_slo_claims"
        ] = True
        with self.assertRaisesRegex(
            StructuralSuiteError,
            "TTFT/TPOT/SLO",
        ):
            _window_summary(promoted, "steady")

    def test_summary_rejects_reservation_overcommit(self) -> None:
        audit = self._valid_audit("burst")
        audit["kv_lifecycle"]["no_preemption_admission"][
            "max_reserved_blocks"
        ] = _structural_plan()["num_logical_kv_blocks"] + 1
        with self.assertRaisesRegex(
            StructuralSuiteError,
            "reservation exceeds capacity",
        ):
            _window_summary(audit, "burst")


if __name__ == "__main__":
    unittest.main()
