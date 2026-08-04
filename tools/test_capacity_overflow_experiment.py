#!/usr/bin/env python3
"""End-to-end contract for the capacity-overflow comparison artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


# The exact replay is intentionally identical in Release and sanitizer builds.
# A Debug ASan+UBSan binary can take more than two minutes for this workload;
# keep a finite hang guard without treating instrumentation overhead as a model
# failure.
SUBPROCESS_TIMEOUT_SECONDS = 300


class CapacityOverflowExperimentTest(unittest.TestCase):
    experiment: Path
    analyzer = Path(__file__).resolve().with_name(
        "analyze_overflow_offload_experiment.py"
    )

    def test_artifact_is_conserved_renderable_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hbm_data_pages = 8192
            read_buffer_pages = 8
            offload_pages = 512
            hbm_total_pages = hbm_data_pages + read_buffer_pages
            total_write_pages = hbm_data_pages + offload_pages
            summary = root / "summary.json"
            csv = root / "summary.csv"
            report = root / "report.md"
            subprocess.run(
                [
                    str(self.experiment),
                    "--hbm-capacity-bytes",
                    str(hbm_total_pages * 4096),
                    "--total-write-bytes",
                    str(total_write_pages * 4096),
                    "--read-buffer-bytes",
                    str(read_buffer_pages * 4096),
                    "--layer-bytes",
                    str(offload_pages * 4096),
                    "--batch-pages",
                    "8",
                    "--sample-batches",
                    "32",
                    "--json",
                    str(summary),
                    "--csv",
                    str(csv),
                ],
                text=True,
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
                check=True,
            )
            analyzed = subprocess.run(
                [
                    sys.executable,
                    str(self.analyzer),
                    "--input",
                    str(summary),
                    "--markdown",
                    str(report),
                    "--experiment",
                    str(self.experiment),
                ],
                text=True,
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
                check=True,
            )
            self.assertIn(
                "real-capacity offloading experiment validation: PASS",
                analyzed.stdout,
            )

            document = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(
                document["schema"],
                {
                    "name": "hbfsim.capacity-overflow-experiment",
                    "version": 7,
                },
            )
            self.assertIn("git_commit", document["generator"])
            self.assertIsInstance(document["generator"]["git_dirty"], bool)
            self.assertEqual(
                document["generator"]["executable"]["value"],
                hashlib.sha256(self.experiment.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                document["validation"],
                {
                    "status": "exploratory_unattached",
                    "certificate": None,
                },
            )
            self.assertNotIn("spill", json.dumps(document).lower())
            self.assertEqual(
                document["workload"]["offload_pages"],
                offload_pages,
            )
            self.assertEqual(
                document["layer_round_trip"]["transfer_bytes"],
                offload_pages * 4096,
            )
            self.assertEqual(
                document["layer_round_trip"]["destination"],
                "original-hbm-slots",
            )
            self.assertEqual(
                document["layer_round_trip"]["default_reference"]["model"],
                "Llama-3.1-405B",
            )
            self.assertFalse(
                document["profile"]["base_die_link"][
                    "sandisk_public_d2d_measurement_available"
                ]
            )
            self.assertEqual(
                [case["name"] for case in document["cases"]],
                ["hbf", "cxl-memory", "nvme-ssd"],
            )
            for case in document["cases"]:
                self.assertEqual(case["conservation"], "PASS")
                self.assertEqual(
                    case["capacity"]["hbm_total_bytes"],
                    hbm_total_pages * 4096,
                )
                self.assertEqual(
                    case["capacity"]["hbm_data_bytes"],
                    hbm_data_pages * 4096,
                )
                self.assertEqual(
                    case["traffic"]["offload_bytes"],
                    offload_pages * 4096,
                )
                self.assertEqual(
                    case["traffic"]["readback_bytes"],
                    offload_pages * 4096,
                )
                self.assertEqual(case["sampling"]["convergence"], "PASS")
                self.assertLessEqual(
                    case["sampling"]["offload_relative_drift"],
                    document["method"]["convergence_threshold"],
                )
                self.assertLessEqual(
                    case["sampling"]["read_relative_drift"],
                    document["method"]["convergence_threshold"],
                )
                self.assertLessEqual(
                    case["sampling"]["fill_relative_drift"],
                    document["method"]["convergence_threshold"],
                )
                layer = case["layer_round_trip"]
                self.assertAlmostEqual(
                    layer["e2e_elapsed_ns"],
                    layer["offload_elapsed_ns"]
                    + layer["restore_to_hbm_elapsed_ns"],
                )
                self.assertEqual(layer["convergence"], "PASS")
                self.assertLessEqual(
                    layer["offload_relative_drift"],
                    document["method"]["convergence_threshold"],
                )
                self.assertLessEqual(
                    layer["restore_relative_drift"],
                    document["method"]["convergence_threshold"],
                )
                self.assertEqual(
                    [
                        point["window_pages"]
                        for point in case["service_curve"]["points"]
                    ],
                    [1, 4, 16, 64, 256],
                )
                self.assertEqual(
                    case["service_curve"]["provenance"],
                    "measured_sample",
                )
                for point in case["service_curve"]["points"]:
                    self.assertGreater(point["offload_payload_GBps"], 0)
                    self.assertGreater(point["readback_payload_GBps"], 0)
                    for phase in (
                        "offload_latency_ns",
                        "readback_latency_ns",
                    ):
                        for rank in ("p50", "p95", "p99"):
                            self.assertGreaterEqual(
                                point[phase]["offered"][rank],
                                point[phase]["service"][rank],
                            )
                for sample in ("base", "doubled"):
                    self.assertEqual(
                        case["raw_samples"][sample]["digest"]["algorithm"],
                        "sha256",
                    )
                    record = case["raw_samples"][sample]["digest"][
                        "canonical_record"
                    ]
                    self.assertEqual(
                        case["raw_samples"][sample]["digest"]["value"],
                        hashlib.sha256(record.encode()).hexdigest(),
                    )

            lifecycle = document["layer_lifecycle"]
            self.assertEqual(
                lifecycle["restore_counts"],
                [1, 2, 4, 8, 16, 32],
            )
            self.assertEqual(
                [point["restore_count"] for point in lifecycle["points"]],
                [1, 2, 4, 8, 16, 32],
            )

            lines = csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            self.assertTrue(
                lines[0].startswith(
                    "case,backing,hbm_capacity_bytes,total_write_bytes,"
                )
            )
            self.assertIn("fill_relative_drift", lines[0])
            self.assertIn("offload_bytes", lines[0])
            self.assertIn("layer_e2e_elapsed_ns", lines[0])
            self.assertNotIn("spill", lines[0].lower())
            rendered = report.read_text(encoding="utf-8")
            self.assertIn("not read-only weights", rendered)
            self.assertIn("crash-consistency comparison", rendered)
            self.assertIn("Capacity and traffic are exact", rendered)
            self.assertIn("## Parameter summary", rendered)
            self.assertIn("## D2D evidence boundary", rendered)
            self.assertIn("## Layer round-trip result", rendered)
            self.assertIn("full layer is resident and usable in HBM", rendered)
            self.assertIn("offloading GB/s", rendered)
            self.assertIn("HBM fill ns/byte", rendered)
            self.assertIn("physical-ceiling checks passed", rendered)
            self.assertIn("## Controller-window service curve", rendered)
            self.assertIn("## Layer reuse lifecycle", rendered)

            def assert_rejected(
                corrupted_document: dict[str, object],
                expected_message: str,
                suffix: str,
            ) -> None:
                corrupted = root / f"corrupted-{suffix}.json"
                corrupted.write_text(
                    json.dumps(corrupted_document),
                    encoding="utf-8",
                )
                rejected_report = root / f"rejected-{suffix}.md"
                rejected = subprocess.run(
                    [
                        sys.executable,
                        str(self.analyzer),
                        "--input",
                        str(corrupted),
                        "--markdown",
                        str(rejected_report),
                        "--experiment",
                        str(self.experiment),
                    ],
                    text=True,
                    capture_output=True,
                    timeout=SUBPROCESS_TIMEOUT_SECONDS,
                    check=False,
                )
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn(expected_message, rejected.stderr)
                self.assertFalse(rejected_report.exists())

            corrupted = copy.deepcopy(document)
            corrupted["cases"][0]["traffic"]["offload_bytes"] += 1
            assert_rejected(
                corrupted,
                "target traffic does not conserve",
                "traffic",
            )

            corrupted = copy.deepcopy(document)
            corrupted["cases"][1]["external"]["m2s_wire_bytes"] += 1
            assert_rejected(
                corrupted,
                "payload/protocol/wire accounting does not conserve",
                "wire",
            )

            corrupted = copy.deepcopy(document)
            corrupted["cases"][0]["service_curve"]["points"][1][
                "window_pages"
            ] = 5
            assert_rejected(
                corrupted,
                "exact sample throughput does not conserve",
                "service-curve",
            )

            corrupted = copy.deepcopy(document)
            corrupted["layer_lifecycle"]["points"][0]["winner"] = "hbf"
            assert_rejected(
                corrupted,
                "lifecycle winner is not derived",
                "lifecycle",
            )

            corrupted = copy.deepcopy(document)
            corrupted["cases"][0]["raw_samples"]["base"]["digest"][
                "value"
            ] = "0" * 64
            assert_rejected(
                corrupted,
                "sample summary digest is invalid",
                "sample-digest",
            )

            corrupted = copy.deepcopy(document)
            corrupted["validation"]["status"] = "validated"
            assert_rejected(
                corrupted,
                "must be explicitly exploratory",
                "validation-state",
            )

    def test_rejects_non_overflow_and_undersized_replay_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_cases = (
                (
                    8200,
                    8200,
                    "total writes must directly exceed physical HBM capacity",
                ),
                (
                    40,
                    48,
                    (
                        "target HBM, offloading volume, and layer must "
                        "contain the doubled replay windows"
                    ),
                ),
            )
            for index, (hbm_pages, write_pages, expected_error) in enumerate(
                invalid_cases
            ):
                with self.subTest(expected_error=expected_error):
                    result = subprocess.run(
                        [
                            str(self.experiment),
                            "--hbm-capacity-bytes",
                            str(hbm_pages * 4096),
                            "--total-write-bytes",
                            str(write_pages * 4096),
                            "--read-buffer-bytes",
                            str(8 * 4096),
                            "--layer-bytes",
                            str(max(8, write_pages - (hbm_pages - 8)) * 4096),
                            "--batch-pages",
                            "8",
                            "--sample-batches",
                            "32",
                            "--json",
                            str(root / f"invalid-{index}.json"),
                            "--csv",
                            str(root / f"invalid-{index}.csv"),
                        ],
                        text=True,
                        capture_output=True,
                        timeout=SUBPROCESS_TIMEOUT_SECONDS,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected_error, result.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    CapacityOverflowExperimentTest.experiment = arguments.experiment.resolve()
    unittest.main(argv=["test_capacity_overflow_experiment.py"])
