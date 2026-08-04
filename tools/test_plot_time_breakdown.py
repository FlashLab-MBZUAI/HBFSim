#!/usr/bin/env python3
"""Regression tests for plot_time_breakdown.py."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
TOOL = TOOLS / "plot_time_breakdown.py"
sys.path.insert(0, str(TOOLS))

from test_time_breakdown_report import (  # noqa: E402
    archived_summary as archived_fixture_summary,
    summary as fixture_summary,
)
from time_breakdown_report import SummaryInput  # noqa: E402

SPEC = importlib.util.spec_from_file_location("plot_time_breakdown", TOOL)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {TOOL}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def named_summary(name: str, *, hbm: bool, hbf: bool) -> dict:
    data = copy.deepcopy(fixture_summary(hbm=hbm, hbf=hbf))
    scenario = data["scenarios"][0]
    scenario["name"] = name
    scenario["reads"] = scenario["ops"]
    scenario["writes"] = 0
    scenario["hbm_accesses"] = scenario["ops"] if hbm and not hbf else 0
    scenario["hbf_accesses"] = scenario["ops"] if hbf and not hbm else 0
    scenario["user_completion_throughput_GBps"] = 123.5
    return data


def named_archived_summary(name: str, *, hbm: bool, hbf: bool) -> dict:
    data = copy.deepcopy(archived_fixture_summary(hbm=hbm, hbf=hbf))
    scenario = data["scenarios"][0]
    scenario["name"] = name
    scenario["reads"] = scenario["ops"]
    scenario["writes"] = 0
    scenario["hbm_accesses"] = scenario["ops"] if hbm and not hbf else 0
    scenario["hbf_accesses"] = scenario["ops"] if hbf and not hbm else 0
    scenario["user_completion_throughput_GBps"] = 123.5
    return data


class TimeBreakdownVisualizationUnitTests(unittest.TestCase):
    def inputs(self) -> list[SummaryInput]:
        return [
            SummaryInput(
                label="baseline",
                summary={
                    "schema": {"name": "hbfsim.scenario_compare.summary", "version": 16},
                    "sanity": "PASS",
                    "scenarios": [
                        named_summary("all-HBM", hbm=True, hbf=False)["scenarios"][0],
                        named_summary("all-HBF", hbm=False, hbf=True)["scenarios"][0],
                    ],
                },
                source="baseline.json",
            ),
            SummaryInput(
                label="ec2",
                summary=named_summary("HBM-HBF-Flat", hbm=True, hbf=True),
                source="ec2.json",
            ),
        ]

    def test_model_preserves_input_order_and_excludes_attribution_children(self) -> None:
        model = MODULE.build_visualization_model(
            self.inputs(),
            aliases={
                "baseline/all-HBM": "EC0",
                "baseline/all-HBF": "EC1",
            },
        )
        self.assertEqual(
            [scenario.label for scenario in model.scenarios],
            ["EC0", "EC1", "ec2"],
        )
        self.assertIn("hbm.scheduler_queue_wait_work_ns", model.stage_keys)
        self.assertIn("hbf.array_read_work_ns", model.stage_keys)
        self.assertFalse(any("hbf.ecc.decode" in key for key in model.stage_keys))
        self.assertNotIn(
            "hbf.resident_mapping.dram_wait_work_ns",
            model.stage_keys,
        )
        self.assertIn(
            "hbf.resident_mapping.dram_wait_work_ns",
            model.attribution_keys,
        )
        self.assertIn(
            "hbf.write_buffer.slot_wait_work_ns",
            model.attribution_keys,
        )
        self.assertEqual(model.scenarios[0].hbm_accesses, 2)
        self.assertEqual(model.scenarios[1].hbf_accesses, 2)

    def test_html_separates_timing_semantics_and_labels_every_dimension(self) -> None:
        model = MODULE.build_visualization_model(
            self.inputs(),
            aliases={
                "baseline/all-HBM": "EC0",
                "baseline/all-HBF": "EC1",
            },
        )
        rendered = MODULE.render_html(model, title="Timing test")
        self.assertIn("Additive wall clock", rendered)
        self.assertIn("Per-operation source-to-completion work", rendered)
        self.assertIn("Overlapping stage work", rendered)
        self.assertIn("Stage-cost detail", rendered)
        self.assertIn("Nested wait / direction attribution", rendered)
        self.assertIn("Resource utilization and average parallelism", rendered)
        self.assertIn("Offered-arrival span:", rendered)
        self.assertIn("Post-offer completion tail:", rendered)
        self.assertIn("Phase dependency wait:", rendered)
        self.assertIn("Front-end admission wait:", rendered)
        self.assertIn("Service (first credit to completion):", rendered)
        self.assertIn("µs/op</strong>", rendered)
        self.assertIn("HBM · Scheduler queue wait", rendered)
        self.assertIn("HBF · Array read", rendered)
        self.assertIn("HBF resident mapping · DRAM issue wait", rendered)
        self.assertIn("Included in HBF scheduler queue wait", rendered)
        self.assertIn("never add them to the primary-stage total", rendered)
        self.assertIn("role=\"img\"", rendered)
        self.assertNotIn("http://", rendered)
        self.assertNotIn("https://", rendered)
        self.assertNotIn("<script", rendered)

    def test_unknown_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(
                MODULE.TimeBreakdownVisualizationError,
                "matched no scenario"):
            MODULE.build_visualization_model(
                self.inputs(), aliases={"missing/scenario": "EC9"})

    def test_schema_v9_metadata_stage_renders_without_resident_mapping(self) -> None:
        data = named_archived_summary("all-HBF", hbm=False, hbf=True)
        hbf = data["scenarios"][0]["time_breakdown"]["stage_work"]["hbf"]
        hbf["metadata_cache_work_ns"] = 7.0
        hbf["total_overlapping_work_ns"] += 7.0
        model = MODULE.build_visualization_model([
            SummaryInput(label="EC1", summary=data, source="ec1-v9.json")
        ])
        rendered = MODULE.render_html(model, title="Archived timing")
        self.assertIn("HBF · Metadata cache", rendered)
        self.assertNotIn("HBF resident mapping · DRAM issue wait", rendered)
        self.assertIn("time-breakdown contract v2", rendered)


class TimeBreakdownVisualizationCliTests(unittest.TestCase):
    def test_cli_writes_self_contained_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            output = root / "timing.html"
            summary_path.write_text(
                json.dumps(named_summary("all-HBM", hbm=True, hbf=False)),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--summary", f"ec0={summary_path}",
                    "--output", str(output),
                    "--title", "CLI timing",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("1 scenarios", completed.stdout)
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", rendered)
            self.assertIn("CLI timing", rendered)
            self.assertIn("ec0", rendered)
            self.assertLess(len(rendered.encode("utf-8")), 2 * 1024 * 1024)

    def test_failure_does_not_clobber_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            output = root / "timing.html"
            summary_path.write_text(
                json.dumps(named_summary("all-HBM", hbm=True, hbf=False)),
                encoding="utf-8",
            )
            output.write_text("keep-me", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--summary", f"ec0={summary_path}",
                    "--alias", "missing/scenario=EC9",
                    "--output", str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("matched no scenario", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep-me")


if __name__ == "__main__":
    unittest.main()
