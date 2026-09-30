#!/usr/bin/env python3
"""Address heatmaps are opt-in: off by default, exact when requested."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReferenceHeatmapDisableTest(unittest.TestCase):
    reference: Path = ROOT / "build/hbfsim-reference"

    def run_all_hbm(self, root: Path, name: str, *extra: str) -> tuple[dict, str]:
        trace = root / "large-read.trace"
        trace.write_text("0x0 R 4194304\n", encoding="utf-8")
        summary = root / f"{name}.json"
        completed = subprocess.run(
            [
                str(self.reference),
                "--config",
                str(ROOT / "configs/systems/6hbm-2hbf.cfg"),
                "--trace",
                str(trace),
                "--scenarios",
                "all-hbm",
                "--max-outstanding-requests",
                "1",
                "--trace-mode",
                "off",
                "--summary-json",
                str(summary),
                *extra,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        return json.loads(summary.read_text(encoding="utf-8")), completed.stdout

    def test_heatmap_is_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, stdout = self.run_all_hbm(Path(directory), "default")
        self.assertEqual(result["sanity"], "PASS")
        self.assertEqual(result["config"]["address_heatmap_bins"], 0)
        scenario = result["scenarios"][0]
        self.assertEqual(scenario["name"], "all-hbm")
        self.assertIsNone(scenario["address_heatmap"])
        self.assertEqual(scenario["logical_bytes"], 4 * 1024**2)
        self.assertIn("Address heatmap: disabled", stdout)
        self.assertNotIn("model error", stdout)

    def test_requested_bins_produce_an_exact_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.run_all_hbm(
                Path(directory), "bins", "--address-heatmap-bins", "16")
        self.assertEqual(result["sanity"], "PASS")
        self.assertEqual(result["config"]["address_heatmap_bins"], 16)
        heatmap = result["scenarios"][0]["address_heatmap"]
        self.assertEqual(heatmap["bin_count"], 16)
        workload = next(
            domain for domain in heatmap["domains"]
            if domain["domain"] == "workload_logical")
        self.assertEqual(workload["totals"]["read_bytes"], 4 * 1024**2)
        self.assertEqual(len(workload["bins"]), 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    ReferenceHeatmapDisableTest.reference = args.reference.resolve()
    unittest.main(argv=[__file__, *remaining])
