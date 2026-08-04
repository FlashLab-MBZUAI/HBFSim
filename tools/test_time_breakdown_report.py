#!/usr/bin/env python3
"""Unit regressions for the shared suite-level time-breakdown renderer."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from time_breakdown_report import (  # noqa: E402
    BASE_DIE_LINK_COMPONENTS,
    DEVICE_STAGE_COMPONENTS,
    DEVICE_STAGE_TOTAL,
    RESOURCE_NAMES,
    LAYER_STREAMING_CONTROLLER_FIELDS,
    SUMMARY_SCHEMA,
    TIME_SEMANTICS,
    SummaryInput,
    TimeBreakdownError,
    build_time_breakdown_report,
    write_time_breakdown_report,
)


def device_stage(device: str) -> dict:
    work = {key: 0.0 for key in DEVICE_STAGE_COMPONENTS}
    if device == "hbm":
        work["scheduler_queue_wait_work_ns"] = 20.0
        work["channel_transfer_work_ns"] = 10.0
    else:
        work["scheduler_queue_wait_work_ns"] = 40.0
        work["array_read_work_ns"] = 100.0
        work["ecc_queue_wait_work_ns"] = 4.0
        work["ecc_response_latency_work_ns"] = 6.0
    work[DEVICE_STAGE_TOTAL] = sum(work.values())
    if device == "hbf":
        work["write_buffer"] = {
            "slot_waited_ops": 1,
            "slot_wait_work_ns": 5.0,
        }
        work["resident_mapping"] = {
            "table_bytes": 32768,
            "table_bytes_per_stack": 16384,
            "pages_per_stack": 4,
            "lookup_ops": 12,
            "user_lookup_ops": 10,
            "gc_lookup_ops": 2,
            "update_ops": 3,
            "user_update_ops": 2,
            "gc_update_ops": 1,
            "dram_waited_ops": 1,
            "dram_wait_work_ns": 10.0,
            "dram_wait_max_ns": 10.0,
            "dram_issue_busy_ns": 15.0,
            "dram_resources": 2,
        }
        work["ecc_directional"] = {
            "decode": {
                "queue_wait_work_ns": 3.0,
                "response_latency_work_ns": 4.0,
                "issue_busy_ns": 3.0,
                "ops": 2,
                "codeword_bytes": 8448,
            },
            "encode": {
                "queue_wait_work_ns": 1.0,
                "response_latency_work_ns": 2.0,
                "issue_busy_ns": 2.0,
                "ops": 1,
                "codeword_bytes": 4224,
            },
        }
    return work


def resource_metric(
    *, busy: float = 0.0, count: int = 0, span: float = 0.0
) -> dict:
    capacity = count * span
    return {
        "busy_ns": busy,
        "resource_count": count,
        "active_span_ns": span,
        "capacity_time_ns": capacity,
        "utilization": busy / capacity if capacity else 0.0,
    }


def scenario(*, hbm: bool = True, hbf: bool = True) -> dict:
    resources = {name: resource_metric() for name in RESOURCE_NAMES}
    if hbm:
        resources["hbm_data_bus"] = resource_metric(
            busy=50.0, count=2, span=100.0
        )
    if hbf:
        resources["hbf_plane_media"] = resource_metric(
            busy=100.0, count=2, span=100.0
        )
        resources["hbf_ecc_issue"] = resource_metric(
            busy=5.0, count=2, span=100.0
        )
    streaming = {key: 0.0 for key in LAYER_STREAMING_CONTROLLER_FIELDS}
    return {
        "name": "all-HBF" if hbf and not hbm else "mixed",
        "ops": 2,
        "time_breakdown": {
            "contract_version": 2,
            "semantics": dict(TIME_SEMANTICS),
            "wall_clock_ns": {
                "trace_origin_ns": 100.0,
                "last_offered_arrival_ns": 150.0,
                "last_user_completion_ns": 250.0,
                "quiescent_finish_ns": 300.0,
                "offered_arrival_span_ns": 50.0,
                "post_offer_user_completion_tail_ns": 100.0,
                "user_completion_span_ns": 150.0,
                "drain_tail_ns": 50.0,
                "makespan_ns": 200.0,
            },
            "latency_work": {
                "basis": "offered_to_user_completion",
                "user_count": 2,
                "service_to_user_completion_sum_work_ns": 80.0,
                "front_end_admission_waited_ops": 1,
                "front_end_admission_wait_work_ns": 20.0,
                "front_end_admission_max_wait_ns": 20.0,
                "offered_to_user_completion_sum_work_ns": 100.0,
                "phase_barriers": 1,
                "phase_dependency_waited_ops": 1,
                "phase_dependency_wait_work_ns": 30.0,
                "phase_dependency_max_wait_ns": 30.0,
                "source_to_user_completion_sum_work_ns": 130.0,
                "average_ns": 50.0,
                "p50_ns": 30.0,
                "p95_ns": 70.0,
                "max_ns": 70.0,
                "service_average_ns": 40.0,
                "service_p50_ns": 30.0,
                "service_p95_ns": 50.0,
                "service_max_ns": 50.0,
                "source_average_ns": 65.0,
                "source_p50_ns": 30.0,
                "source_p95_ns": 100.0,
                "source_max_ns": 100.0,
            },
            "stage_work": {
                "scope": "all_device_work_including_background_and_drain",
                "hbm": device_stage("hbm") if hbm else None,
                "hbf": device_stage("hbf") if hbf else None,
                "base_die_link": {
                    key: 0.0 for key in BASE_DIE_LINK_COMPONENTS
                },
                "layer_streaming_controller": {
                    "present": False,
                    **streaming,
                },
                "cooperative_write_controller": {
                    "present": False,
                    "full_waited_ops": 0,
                    "full_wait_work_ns": 0.0,
                },
            },
            "resource_busy": resources,
        },
    }


def summary(*, hbm: bool = True, hbf: bool = True) -> dict:
    return {
        "schema": dict(SUMMARY_SCHEMA),
        "sanity": "PASS",
        "scenarios": [scenario(hbm=hbm, hbf=hbf)],
    }


def archived_summary(*, hbm: bool = True, hbf: bool = True) -> dict:
    data = summary(hbm=hbm, hbf=hbf)
    data["schema"]["version"] = 9
    time = data["scenarios"][0]["time_breakdown"]
    for device in ("hbm", "hbf"):
        stage = time["stage_work"][device]
        if stage is None:
            continue
        stage["metadata_cache_work_ns"] = stage.pop("mapping_dram_work_ns")
        if device == "hbf":
            del stage["resident_mapping"]
    del time["resource_busy"]["hbf_mapping_dram_issue"]
    return data


class TimeBreakdownReportTests(unittest.TestCase):
    def make_input(
        self, data: dict | None = None, *, label: str = "run-a"
    ) -> SummaryInput:
        return SummaryInput(
            label=label,
            summary=data or summary(),
            source=f"/tmp/{label}.summary.json",
            scenario_metadata={"mixed": {"case": "ec2", "config": "cfg-a"}},
        )

    def test_report_separates_additive_wall_work_and_resource_capacity(self) -> None:
        report = build_time_breakdown_report([self.make_input()])
        self.assertEqual(len(report.overviews), 1)
        overview = report.overviews[0]
        self.assertEqual(overview.largest_stage, "hbf.array_read_work_ns")
        self.assertEqual(overview.largest_stage_work_ns, 100.0)
        self.assertEqual(
            overview.highest_utilization_resource, "hbf_plane_media"
        )
        self.assertEqual(overview.highest_resource_utilization, 0.5)

        wall_segments = [
            row for row in report.rows
            if row["domain"] == "wall_clock"
            and row["role"] == "additive_segment"
        ]
        self.assertEqual(sum(float(row["value_ns"]) for row in wall_segments), 200.0)
        self.assertTrue(all(row["share_basis"] == "makespan_ns"
                            for row in wall_segments))
        stage_rows = [row for row in report.rows if row["domain"] == "stage_work"]
        self.assertTrue(stage_rows)
        self.assertTrue(all(row["share_basis"] != "makespan_ns"
                            for row in stage_rows))
        source_total = next(
            row for row in report.rows
            if row["component"] == "source_to_user_completion_sum_work_ns"
        )
        self.assertEqual(source_total["value_ns"], 130.0)
        self.assertEqual(source_total["role"], "source_work_total")

    def test_null_hbf_subsystem_has_no_fake_hbf_rows(self) -> None:
        data = summary(hbm=True, hbf=False)
        data["scenarios"][0]["name"] = "all-HBM"
        source = SummaryInput(label="hbm", summary=data)
        report = build_time_breakdown_report([source])
        self.assertFalse(any(
            row["domain"] == "stage_work" and str(row["scope"]).startswith("hbf")
            for row in report.rows
        ))
        self.assertFalse(any(
            row["domain"] == "resource_busy" and str(row["scope"]).startswith("hbf")
            for row in report.rows
        ))
        self.assertEqual(
            report.overviews[0].largest_stage,
            "hbm.scheduler_queue_wait_work_ns",
        )

    def test_hierarchy_roles_prevent_children_from_being_summed_twice(self) -> None:
        report = build_time_breakdown_report([self.make_input()])
        primary = [
            row for row in report.rows
            if row["domain"] == "stage_work"
            and row["scope"] == "hbf"
            and row["role"] == "overlapping_work_component"
        ]
        total = next(
            row for row in report.rows
            if row["domain"] == "stage_work"
            and row["scope"] == "hbf"
            and row["role"] == "scope_work_total"
        )
        children = [
            row for row in report.rows
            if row["domain"] == "stage_work"
            and row["role"] in {"attribution_child", "attribution_subset"}
        ]
        self.assertEqual(
            sum(float(row["value_ns"]) for row in primary),
            float(total["value_ns"]),
        )
        self.assertTrue(children)
        self.assertGreater(
            sum(float(row["value_ns"]) for row in primary + children),
            float(total["value_ns"]),
        )
        self.assertTrue(all("already included" in str(row["note"])
                            for row in children))

    def test_missing_resident_mapping_block_is_rejected(self) -> None:
        stored = summary()
        del stored["scenarios"][0]["time_breakdown"]["stage_work"]["hbf"][
            "resident_mapping"
        ]
        with self.assertRaisesRegex(
                TimeBreakdownError, "resident_mapping"):
            build_time_breakdown_report([self.make_input(stored)])

    def test_schema_v9_contract_is_validated_without_v16_metrics(self) -> None:
        stored = archived_summary()
        hbf = stored["scenarios"][0]["time_breakdown"]["stage_work"]["hbf"]
        hbf["metadata_cache_work_ns"] = 7.0
        hbf[DEVICE_STAGE_TOTAL] += 7.0
        report = build_time_breakdown_report([self.make_input(stored)])
        self.assertTrue(any(
            row["scope"] == "hbf"
            and row["component"] == "metadata_cache_work_ns"
            and row["value_ns"] == 7.0
            for row in report.rows
        ))
        self.assertFalse(any(
            row["scope"] == "hbf.resident_mapping"
            or row["scope"] == "hbf_mapping_dram_issue"
            for row in report.rows
        ))

    def test_bad_contracts_fail_closed(self) -> None:
        valid = summary()
        corruptions = (
            (
                ("scenarios", 0, "time_breakdown", "contract_version"),
                1,
                "unsupported version",
            ),
            (
                ("scenarios", 0, "time_breakdown", "semantics", "stage_work"),
                "ambiguous",
                "does not match contract",
            ),
            (
                (
                    "scenarios", 0, "time_breakdown", "wall_clock_ns",
                    "makespan_ns",
                ),
                201.0,
                "must equal makespan",
            ),
            (
                (
                    "scenarios", 0, "time_breakdown", "latency_work",
                    "source_to_user_completion_sum_work_ns",
                ),
                131.0,
                "offered work plus phase dependency wait",
            ),
            (
                (
                    "scenarios", 0, "time_breakdown", "stage_work", "hbf",
                    "array_read_work_ns",
                ),
                -1.0,
                "finite and non-negative",
            ),
            (
                (
                    "scenarios", 0, "time_breakdown", "stage_work", "hbf",
                    "total_overlapping_work_ns",
                ),
                999.0,
                "must sum",
            ),
            (
                (
                    "scenarios", 0, "time_breakdown", "resource_busy",
                    "hbf_plane_media", "busy_ns",
                ),
                201.0,
                "exceeds capacity-time",
            ),
        )
        for path, value, message in corruptions:
            with self.subTest(message=message):
                broken = copy.deepcopy(valid)
                cursor = broken
                for component in path[:-1]:
                    cursor = cursor[component]
                cursor[path[-1]] = value
                with self.assertRaisesRegex(TimeBreakdownError, message):
                    build_time_breakdown_report([self.make_input(broken)])

    def test_large_ecc_reduction_roundoff_is_tolerated_but_drift_is_not(
        self,
    ) -> None:
        directional_sum = 447_526_995.75637656
        independently_accumulated_parent = 447_526_995.75214976
        data = summary()
        hbf = data["scenarios"][0]["time_breakdown"]["stage_work"]["hbf"]
        hbf["ecc_directional"]["decode"]["issue_busy_ns"] = directional_sum
        hbf["ecc_directional"]["encode"]["issue_busy_ns"] = 0.0
        resources = data["scenarios"][0]["time_breakdown"]["resource_busy"]
        resources["hbf_ecc_issue"] = resource_metric(
            busy=independently_accumulated_parent,
            count=2,
            span=300_000_000.0,
        )

        report = build_time_breakdown_report([self.make_input(data)])
        self.assertEqual(len(report.overviews), 1)

        broken = copy.deepcopy(data)
        broken_resource = broken["scenarios"][0]["time_breakdown"][
            "resource_busy"
        ]["hbf_ecc_issue"]
        broken_resource["busy_ns"] = directional_sum - 1.0
        broken_resource["utilization"] = (
            broken_resource["busy_ns"] / broken_resource["capacity_time_ns"]
        )
        with self.assertRaisesRegex(
            TimeBreakdownError,
            "directional ECC issue busy must sum",
        ):
            build_time_breakdown_report([self.make_input(broken)])

    def test_long_trace_average_roundoff_is_tolerated_but_drift_is_not(
        self,
    ) -> None:
        data = summary()
        scenario_data = data["scenarios"][0]
        scenario_data["ops"] = 75_497_472
        latency = scenario_data["time_breakdown"]["latency_work"]
        latency["user_count"] = 75_497_472
        latency["service_to_user_completion_sum_work_ns"] = (
            86_124_898_634.94797
        )
        latency["offered_to_user_completion_sum_work_ns"] = (
            latency["service_to_user_completion_sum_work_ns"]
            + latency["front_end_admission_wait_work_ns"]
        )
        latency["source_to_user_completion_sum_work_ns"] = (
            latency["offered_to_user_completion_sum_work_ns"]
            + latency["phase_dependency_wait_work_ns"]
        )
        expected = (
            latency["offered_to_user_completion_sum_work_ns"]
            / latency["user_count"]
        )
        expected_service = (
            latency["service_to_user_completion_sum_work_ns"]
            / latency["user_count"]
        )
        expected_source = (
            latency["source_to_user_completion_sum_work_ns"]
            / latency["user_count"]
        )
        # Reproduce the scale of an independent online-mean reduction seen in
        # a long run (about 0.3 parts per billion).
        latency["average_ns"] = expected - 3.5e-7
        latency["service_average_ns"] = expected_service - 3.5e-7
        latency["source_average_ns"] = expected_source - 3.5e-7
        latency["max_ns"] = max(latency["max_ns"], expected)
        latency["service_max_ns"] = max(
            latency["service_max_ns"], expected_service)
        latency["source_max_ns"] = max(
            latency["source_max_ns"], expected_source)

        report = build_time_breakdown_report([self.make_input(data)])
        self.assertEqual(len(report.overviews), 1)

        broken = copy.deepcopy(data)
        broken["scenarios"][0]["time_breakdown"]["latency_work"][
            "average_ns"
        ] = expected + 0.001
        with self.assertRaisesRegex(
            TimeBreakdownError,
            "average must equal offered work divided by user_count",
        ):
            build_time_breakdown_report([self.make_input(broken)])

    def test_write_is_atomic_with_respect_to_validation(self) -> None:
        broken = summary()
        broken["scenarios"][0]["time_breakdown"]["wall_clock_ns"][
            "drain_tail_ns"
        ] = 51.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "time-breakdown.csv"
            markdown_path = root / "time-breakdown.md"
            csv_path.write_text("old csv\n")
            markdown_path.write_text("old markdown\n")
            with self.assertRaises(TimeBreakdownError):
                write_time_breakdown_report(
                    [self.make_input(broken)], csv_path, markdown_path
                )
            self.assertEqual(csv_path.read_text(), "old csv\n")
            self.assertEqual(markdown_path.read_text(), "old markdown\n")

    def test_write_and_multi_summary_cli(self) -> None:
        first = summary(hbm=True, hbf=False)
        first["scenarios"][0]["name"] = "all-HBM"
        second = summary(hbm=False, hbf=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.json"
            second_path = root / "second.json"
            first_path.write_text(json.dumps(first))
            second_path.write_text(json.dumps(second))
            csv_path = root / "time-breakdown.csv"
            markdown_path = root / "time-breakdown.md"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "tools" / "time_breakdown_report.py"),
                    "--summary",
                    f"baseline={first_path}",
                    "--summary",
                    f"flash={second_path}",
                    "--csv",
                    str(csv_path),
                    "--markdown",
                    str(markdown_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["run_label"] for row in rows}, {"baseline", "flash"})
            markdown = markdown_path.read_text()
            self.assertIn("Only the three wall-clock segments are additive", markdown)
            self.assertIn("baseline", markdown)
            self.assertIn("flash", markdown)


if __name__ == "__main__":
    unittest.main()
