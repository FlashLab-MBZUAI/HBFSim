#!/usr/bin/env python3
"""FLAT routed entirely to HBM must equal all-hbm bit for bit.

Every direct composition issues its HBM page transactions through one
synchronous discipline, so `flat` with a boundary above the whole footprint
sees the exact op stream and controller path that `all-hbm` sees. The runner
enforces this as its policy-identity sanity check; this test pins it on the
README quick-start trace, where an all-hbm-only batch path once produced a
1.5x shorter makespan than every other scenario for identical traffic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_CONFIG = ROOT / "configs/systems/server-hbm128-hbf512.cfg"
POLICY_CONFIG = ROOT / "configs/policies/reference/server-hbm128-hbf512.cfg"
HBM_CAPACITY_BYTES = 128 * 1024**3
IDENTICAL_FIELDS = (
    "ops",
    "reads",
    "writes",
    "logical_bytes",
    "hbm_accesses",
    "hbm_user_accesses",
    "hbm_background_accesses",
    "hbm_issue_mode",
    "time_breakdown",
    "user_completion_throughput_GBps",
    "makespan_throughput_GBps",
    "offered_load",
    "hbm_stats",
)
CSV_COLUMNS_THAT_MAY_DIFFER = {"scenario", "warnings"}
# `flat` instantiates an HBF device that `all-hbm` never builds. Its
# presence shows up as per-tier peak rates, HBF stage-work blocks, and HBF
# resource counts; none of it carries timing, so it is masked before the
# comparison rather than compared.
DEVICE_PRESENCE_KEYS = {"hbf", "external", "external_backing", "external_read",
                        "external_write"}
DEVICE_PRESENCE_PREFIXES = ("hbf_", "external_")


def without_device_presence(value):
    if isinstance(value, dict):
        return {
            key: without_device_presence(item)
            for key, item in value.items()
            if key not in DEVICE_PRESENCE_KEYS
            and not key.startswith(DEVICE_PRESENCE_PREFIXES)
        }
    if isinstance(value, list):
        return [without_device_presence(item) for item in value]
    return value


class ReferencePolicyIdentityTest(unittest.TestCase):
    reference: Path

    def test_flat_routed_entirely_to_hbm_equals_all_hbm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.txt"
            summary = root / "summary.json"
            table = root / "summary.csv"
            subprocess.run(
                [
                    str(self.reference),
                    "--generate-semantic-llm", str(trace),
                    "--llm-tokens", "4",
                    "--llm-layers", "2",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            completed = subprocess.run(
                [
                    str(self.reference),
                    "--config", str(SYSTEM_CONFIG),
                    "--config", str(POLICY_CONFIG),
                    "--trace", str(trace),
                    "--scenarios", "all-hbm,flat",
                    "--flat-hbm-bytes", str(HBM_CAPACITY_BYTES),
                    "--summary-json", str(summary),
                    "--summary-csv", str(table),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            self.assertNotIn("policy identity violated", completed.stdout)
            # FLAT with nothing routed to HBF is a configuration warning, not
            # a conservation failure: the run stays valid and exits 0.
            self.assertIn("SANITY: PASS_WITH_WARNINGS", completed.stdout)
            self.assertIn("did not exercise both HBM and HBF", completed.stderr)
            document = json.loads(summary.read_text(encoding="utf-8"))
            with table.open(encoding="utf-8", newline="") as handle:
                rows = {row["scenario"]: row for row in csv.DictReader(handle)}

        self.assertEqual(document["sanity"], "PASS_WITH_WARNINGS")
        by_name = {row["name"]: row for row in document["scenarios"]}
        self.assertEqual(set(by_name), {"all-hbm", "flat"})
        all_hbm = by_name["all-hbm"]
        flat = by_name["flat"]
        self.assertEqual(flat["hbf_accesses"], 0)
        self.assertEqual(all_hbm["hbm_issue_mode"], "synchronous-per-transaction")
        for field in IDENTICAL_FIELDS:
            self.assertEqual(
                without_device_presence(all_hbm[field]),
                without_device_presence(flat[field]),
                field,
            )
        # The HBF blocks exist only on the flat side and carry no traffic.
        self.assertIsNone(all_hbm["time_breakdown"]["stage_work"]["hbf"])
        self.assertIsNotNone(flat["time_breakdown"]["stage_work"]["hbf"])
        self.assertEqual(flat["hbf_user_accesses"], 0)
        self.assertEqual(flat["hbf_background_accesses"], 0)
        self.assertEqual(all_hbm["warnings"], [])
        self.assertTrue(
            any("did not exercise both" in warning for warning in flat["warnings"]),
            flat["warnings"],
        )
        self.assertEqual(set(rows), {"all-hbm", "flat"})
        for column, value in rows["all-hbm"].items():
            if column in CSV_COLUMNS_THAT_MAY_DIFFER or column.startswith("hbf_"):
                continue
            self.assertEqual(value, rows["flat"][column], column)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    ReferencePolicyIdentityTest.reference = args.reference.resolve()
    unittest.main(argv=[__file__, *remaining])
