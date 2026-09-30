#!/usr/bin/env python3
"""End-to-end proof that 1/1000 HBF DRAM drives mapping and data buffering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class HbfControllerDramRatioTest(unittest.TestCase):
    # Keep the test directly runnable under unittest while allowing CTest to
    # override the binary through the explicit --reference argument below.
    reference: Path = ROOT / "build/hbfsim-reference"

    def test_ratio_reaches_runtime_mapping_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            completed = subprocess.run(
                [
                    str(self.reference),
                    "--config",
                    str(ROOT / "configs/systems/6hbm-2hbf.cfg"),
                    "--config",
                    str(
                        ROOT
                        / "configs/overlays/hbf/cached-l2p-1-over-1000.cfg"
                    ),
                    "--synthetic-sequential-read-bytes",
                    "4096",
                    "--scenarios",
                    "all-hbf",
                    "--summary-json",
                    str(summary),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(summary.read_text(encoding="utf-8"))
        config = result["config"]["hbf"]
        self.assertEqual(config["mapping_mode"], "cached")
        self.assertEqual(config["ctrl_dram_capacity_denominator"], 1000)
        self.assertEqual(config["ctrl_dram_bytes"], 1_099_505_664)
        stats = result["scenarios"][0]["hbf_stats"]
        self.assertEqual(stats["mapping_table_bytes"], 2 * 1024**3)
        self.assertEqual(stats["resident_mapping_table_bytes"], 0)
        self.assertEqual(stats["mapping_directory_bytes"], 4_194_304)
        self.assertEqual(stats["controller_dram_budget_bytes"], 1_099_505_664)
        self.assertEqual(stats["write_buffer_capacity_bytes"], 8_388_608)
        self.assertEqual(stats["mapping_cache_capacity_bytes"], (1_086_922_752 - 2 * 4096))
        self.assertEqual(stats["mapping_cache_peak_entries"], 1)
        self.assertEqual(stats["mapping_cache_misses"], 1)
        self.assertEqual(stats["mapping_cache_hits"], 63)
        self.assertEqual(stats["mapping_media_reads"], 1)
        stage = result["scenarios"][0]["time_breakdown"]["stage_work"][
            "hbf"
        ]["mapping_cache"]
        self.assertEqual(stage["capacity_bytes"], (1_086_922_752 - 2 * 4096))
        self.assertEqual(stage["misses"], 1)
        self.assertEqual(stage["hits"], 63)

    def test_ratio_budgeted_write_buffer_is_used_at_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "write-read.trace"
            trace.write_text("0x0 W 1024\n0x0 R 1024\n", encoding="utf-8")
            summary = root / "summary.json"
            completed = subprocess.run(
                [
                    str(self.reference),
                    "--config",
                    str(ROOT / "configs/systems/6hbm-2hbf.cfg"),
                    "--config",
                    str(
                        ROOT
                        / "configs/overlays/hbf/cached-l2p-1-over-1000.cfg"
                    ),
                    "--trace",
                    str(trace),
                    "--max-outstanding-requests",
                    "1",
                    "--scenarios",
                    "all-hbf",
                    "--summary-json",
                    str(summary),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(summary.read_text(encoding="utf-8"))
        stats = result["scenarios"][0]["hbf_stats"]
        self.assertEqual(stats["controller_dram_budget_bytes"], 1_099_505_664)
        self.assertEqual(stats["write_buffer_capacity_bytes"], 8_388_608)
        self.assertEqual(stats["mapping_cache_capacity_bytes"], (1_086_922_752 - 2 * 4096))
        self.assertGreaterEqual(stats["write_buffer_dram_write_ops"], 1)
        self.assertGreaterEqual(stats["write_buffer_dram_read_ops"], 1)
        self.assertEqual(stats["write_buffer_dram_write_bytes"], 1024)
        self.assertGreaterEqual(stats["write_buffer_dram_read_bytes"], 1024)
        stage = result["scenarios"][0]["time_breakdown"]["stage_work"][
            "hbf"
        ]
        self.assertGreater(stage["write_buffer_dram_work_ns"], 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    HbfControllerDramRatioTest.reference = args.reference.resolve()
    unittest.main(argv=[__file__, *remaining])
