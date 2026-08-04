#!/usr/bin/env python3
"""Regression contract for canonical HBF stage-work observability."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path


STAGE_FIELDS = (
    ("hbf_ingress_queue_wait_work_ns", "ingress_queue_wait_work_ns"),
    ("hbf_scheduler_queue_wait_work_ns", "scheduler_queue_wait_work_ns"),
    ("hbf_ecc_queue_wait_work_ns", "ecc_queue_wait_work_ns"),
)

HBF_STATS_FIELDS = (
    ("hbf_ecc_decode_ops", "ecc_decode_ops"),
    ("hbf_ecc_encode_ops", "ecc_encode_ops"),
    ("hbf_ecc_codeword_bytes", "ecc_codeword_bytes"),
    ("hbf_ecc_issue_parallelism", "ecc_issue_parallelism"),
    ("hbf_ecc_max_inflight_per_die", "ecc_max_inflight_per_die"),
    ("hbf_write_buffer_slot_wait_work_ns", "write_buffer_slot_wait_work_ns"),
)

RESOURCE_FIELDS = (
    ("hbf_ecc_issue_busy_ns", "hbf_ecc_issue", "busy_ns"),
    ("hbf_hbio_command_utilization", "hbf_hbio_command", "utilization"),
    ("hbf_hbio_data_utilization", "hbf_hbio_data", "utilization"),
)

FIELDS = tuple(field[0] for field in
               STAGE_FIELDS + HBF_STATS_FIELDS + RESOURCE_FIELDS)


def close_enough(left: float, right: float) -> bool:
    # The comparison table intentionally renders nanoseconds to two decimal
    # places, while JSON retains six decimal places.
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=0.005001)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trace = root / "raw-wait.trace"
        summary_json = root / "summary.json"
        summary_csv = root / "summary.csv"

        # One HBF plane and simultaneous requests intentionally create queueing.
        # Spacing each LPN by a mapping page also avoids turning the workload
        # into a resident-mapping-only microbenchmark.
        lines = []
        for index in range(96):
            address = index * 512 * 4096
            lines.append(f"0x{address:x} W 4096 at=0\n")
        trace.write_text("".join(lines))

        command = [
            str(args.scenario_compare),
            "--trace", str(trace),
            "--scenarios", "all-HBF",
            "--line-size", "4096",
            "--hbf-page-size", "4096",
            "--hbf-oob-bytes", "224",
            "--hbf-stacks", "1",
            "--hbf-channels", "1",
            "--hbf-dies-per-channel", "1",
            "--hbf-ecc-decode-raw-bw", "135",
            "--hbf-ecc-encode-raw-bw", "135",
            "--hbf-channel-bw", "135",
            "--hbf-hbio-bw", "128",
            "--hbf-tsv-bw", "137",
            "--hbf-planes-per-die", "1",
            "--hbf-blocks-per-plane", "16",
            "--hbf-pages-per-block", "64",
            "--hbf-gc-low-watermark-pages", "128",
            "--hbf-gc-reserved-free-blocks-per-plane", "1",
            "--summary-json", str(summary_json),
            "--summary-csv", str(summary_csv),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "scenario_compare failed:\n" + completed.stdout + completed.stderr)

        summary = json.loads(summary_json.read_text())
        if summary.get("schema") != {
                "name": "hbfsim.scenario_compare.summary", "version": 16}:
            raise AssertionError("unexpected summary schema")
        if (summary.get("config", {}).get("hbm", {}).get(
                "address_mapping_scheme") != "burst-pch-bank-swizzle-v1"):
            raise AssertionError("summary is missing the versioned HBM address map")
        scenarios = summary.get("scenarios") or []
        if len(scenarios) != 1 or scenarios[0].get("name") != "all-HBF":
            raise AssertionError("raw-wait regression did not run exactly all-HBF")
        scenario = scenarios[0]
        stats = scenario.get("hbf_stats") or {}
        time = scenario["time_breakdown"]
        stage = time["stage_work"]["hbf"]
        resources = time["resource_busy"]

        with summary_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1 or rows[0].get("scenario") != "all-HBF":
            raise AssertionError("summary CSV did not contain exactly all-HBF")

        json_fields = [
            (csv_field, stage, json_field)
            for csv_field, json_field in STAGE_FIELDS
        ] + [
            (csv_field, stats, json_field)
            for csv_field, json_field in HBF_STATS_FIELDS
        ] + [
            (csv_field, resources[resource], json_field)
            for csv_field, resource, json_field in RESOURCE_FIELDS
        ]
        for csv_field, source, json_field in json_fields:
            if json_field not in source:
                raise AssertionError(
                    f"summary JSON source is missing {json_field} for {csv_field}")
            if csv_field not in rows[0]:
                raise AssertionError(f"summary CSV is missing {csv_field}")
            json_value = float(source[json_field])
            csv_value = float(rows[0][csv_field])
            if not math.isfinite(json_value) or json_value < 0.0:
                raise AssertionError(f"invalid raw wait value for {json_field}")
            if not close_enough(json_value, csv_value):
                raise AssertionError(
                    f"JSON/CSV mismatch for {json_field}: {json_value} != {csv_value}")

        if float(stage["ingress_queue_wait_work_ns"]) <= 0.0:
            raise AssertionError("simultaneous requests produced no ingress wait work")
        if float(stage["scheduler_queue_wait_work_ns"]) <= 0.0:
            raise AssertionError("single-plane pressure produced no scheduler queue wait")
        if float(stats["ecc_encode_ops"]) <= 0.0:
            raise AssertionError("write pressure produced no ECC encode operations")
        if float(resources["hbf_ecc_issue"]["busy_ns"]) <= 0.0:
            raise AssertionError("ECC operations produced no issue-port busy time")

        for field in FIELDS:
            if field not in completed.stdout:
                raise AssertionError(f"console comparison table is missing {field}")
        for token in (
                "ingress_queue_wait_work_ns=",
                "scheduler_queue_wait_work_ns=",
                "ecc_queue_wait_work_ns=",
                "write_buffer_slot_wait_work_ns="):
            if token not in completed.stdout:
                raise AssertionError(f"device summary is missing {token}")

        artifacts = completed.stdout + summary_json.read_text() + summary_csv.read_text()
        for legacy in (
                "hbf_logic_die_queue_wait_ns", "hbf_scheduler_queue_wait_ns",
                "hbf_ecc_queue_wait_ns", "hbf_write_buffer_slot_wait_ns"):
            if legacy in artifacts:
                raise AssertionError(f"legacy ambiguous wait field survived: {legacy}")

    print("canonical HBF work fields are consistent across JSON/CSV/console")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
