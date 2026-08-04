#!/usr/bin/env python3
"""Fail-closed tests for the canonical 70B capacity-grid preflight."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from prepare_frontier_70b_capacity_grid import (  # noqa: E402
    CapacityGridError,
    build_capacity_grid,
)
from run_frontier_70b_structural_suite import (  # noqa: E402
    MODEL_NAME,
    PRIMARY_PRECISION_PROFILE,
    REQUIRED_RUNTIME_BEHAVIORS,
    SUITE_ID,
)


WINDOW_MAXIMA = {
    "steady": (85, 56_742),
    "burst": (18_644, 3_194),
    "long_context_decode_tail": (122, 68_858),
}


class Frontier70BCapacityGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.suite = self.root / "suite"
        self.suite.mkdir()
        self.verification = self.root / "verification.json"
        self._write_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _artifact(path: Path, *, relative: bool = False) -> dict:
        return {
            "path": path.name if relative else str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _write_fixture(self) -> None:
        windows: dict[str, dict] = {}
        fieldnames = [
            "arrived_at",
            "num_prefill_tokens",
            "num_decode_tokens",
            "session_id",
            "block_hash_ids",
            "source_chat_id",
            "parent_chat_id",
            "turn",
            "request_type",
        ]
        for window_index, (window, maximum) in enumerate(
            WINDOW_MAXIMA.items()
        ):
            rows = [
                {
                    "arrived_at": str(row_index),
                    "num_prefill_tokens": "1",
                    "num_decode_tokens": "1",
                    "session_id": str(1000 * window_index + row_index),
                    "block_hash_ids": "",
                    "source_chat_id": str(
                        1000 * window_index + row_index
                    ),
                    "parent_chat_id": "-1",
                    "turn": "1",
                    "request_type": "thinking",
                }
                for row_index in range(255)
            ]
            rows.append(
                {
                    "arrived_at": "255",
                    "num_prefill_tokens": str(maximum[0]),
                    "num_decode_tokens": str(maximum[1]),
                    "session_id": str(1000 * window_index + 255),
                    "block_hash_ids": "",
                    "source_chat_id": str(1000 * window_index + 255),
                    "parent_chat_id": "-1",
                    "turn": "1",
                    "request_type": "thinking",
                }
            )
            request_csv = self.suite / f"{window}.frontier.csv"
            with request_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            prefill = sum(int(row["num_prefill_tokens"]) for row in rows)
            decode = sum(int(row["num_decode_tokens"]) for row in rows)
            windows[window] = {
                "request_csv": self._artifact(
                    request_csv,
                    relative=True,
                ),
                "statistics": {
                    "records": 256,
                    "prefill_tokens": prefill,
                    "decode_tokens": decode,
                    "total_tokens": prefill + decode,
                    "max_prefill_tokens": max(
                        int(row["num_prefill_tokens"]) for row in rows
                    ),
                    "max_decode_tokens": max(
                        int(row["num_decode_tokens"]) for row in rows
                    ),
                    "max_total_tokens": max(
                        int(row["num_prefill_tokens"])
                        + int(row["num_decode_tokens"])
                        for row in rows
                    ),
                },
            }
        suite_manifest = self.suite / "suite-manifest.json"
        suite_manifest.write_text("{}\n", encoding="utf-8")
        receipt = {
            "schema": {
                "name": (
                    "hbfsim.qwen_bailian_frontier_suite_verification"
                ),
                "version": 1,
            },
            "result": "pass",
            "suite_id": SUITE_ID,
            "frontier_workload": {
                "model": {
                    "identity": "meta-llama/Llama-3.1-70B",
                    "frontier_name": MODEL_NAME,
                },
                "primary_precision_profile": PRIMARY_PRECISION_PROFILE,
                "sensitivity_precision_profiles": ["bf16-kv-bf16"],
                "required_runtime_behaviors": list(
                    REQUIRED_RUNTIME_BEHAVIORS
                ),
            },
            "windows": windows,
            "artifacts": {
                "suite_manifest": self._artifact(suite_manifest),
            },
        }
        self.verification.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def build(self) -> dict:
        return build_capacity_grid(
            request_suite_verification=self.verification,
            hbfsim_revision="a" * 40,
        )

    def test_exact_grid_separates_three_static_states(self) -> None:
        result = self.build()
        self.assertEqual(result["result"], "pass")
        self.assertEqual(
            result["census"],
            {
                "points": 30,
                "planner_infeasible": 4,
                "request_infeasible": 2,
                "execution_candidate": 24,
            },
        )
        points = {
            point["point_id"]: point for point in result["points"]
        }
        self.assertEqual(len(points), 30)
        self.assertEqual(
            {
                point_id
                for point_id, point in points.items()
                if point["static_preflight_status"]
                == "planner_infeasible"
            },
            {
                "w8a16-kv-bf16.2h6f.p0_75",
                "bf16-kv-bf16.2h6f.p0_75",
                "bf16-kv-bf16.2h6f.p1_0",
                "bf16-kv-bf16.2h6f.p1_25",
            },
        )
        self.assertEqual(
            {
                point_id
                for point_id, point in points.items()
                if point["static_preflight_status"]
                == "request_infeasible"
            },
            {
                "bf16-kv-bf16.2h6f.p1_5",
                "bf16-kv-bf16.4h4f.p0_75",
            },
        )
        self.assertEqual(
            points["w8a16-kv-bf16.2h6f.p1_0"][
                "residency_plan"
            ]["num_logical_kv_blocks"],
            4562,
        )
        self.assertEqual(
            result["window_requirements"][
                "long_context_decode_tail"
            ]["maximum_full_request_reservation"]["required_blocks"],
            4312,
        )
        self.assertFalse(
            result["eligibility"]["paper_result_eligible"]
        )

    def test_model_inputs_are_exact_for_both_precision_profiles(self) -> None:
        models = self.build()["model_capacity_inputs"]
        self.assertEqual(
            models["w8a16-kv-bf16"][
                "immutable_weight_backing_bytes"
            ],
            70_568_973_312,
        )
        self.assertEqual(
            models["w8a16-kv-bf16"][
                "active_weight_buffer_bytes_per_slot"
            ],
            1_050_929_664,
        )
        self.assertEqual(
            models["bf16-kv-bf16"][
                "immutable_weight_backing_bytes"
            ],
            141_107_412_992,
        )
        self.assertEqual(
            models["bf16-kv-bf16"][
                "active_weight_buffer_bytes_per_slot"
            ],
            2_101_346_304,
        )

    def test_rejects_request_csv_digest_drift(self) -> None:
        path = self.suite / "steady.frontier.csv"
        path.write_text(
            path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CapacityGridError,
            "request_csv.bytes",
        ):
            self.build()

    def test_rejects_verified_statistics_drift(self) -> None:
        receipt = json.loads(self.verification.read_text(encoding="utf-8"))
        receipt["windows"]["burst"]["statistics"][
            "max_total_tokens"
        ] += 1
        self.verification.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            CapacityGridError,
            "burst statistics.max_total_tokens",
        ):
            self.build()


if __name__ == "__main__":
    unittest.main()
