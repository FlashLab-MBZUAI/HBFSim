#!/usr/bin/env python3
"""Cross the audited Frontier -> exporter -> HBFSim structural boundary.

This test proves byte/accounting integration only.  It intentionally uses a
dummy-timed multi-request fixture and must never be cited as performance
eligibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from export_frontier_memory_trace import export_memory_trace  # noqa: E402
from run_frontier_memory_baselines import (  # noqa: E402
    _external_address_capacity,
    run_baselines,
)
import test_audit_frontier_replay as audit_fixture_module  # noqa: E402
import test_export_frontier_memory_trace as export_fixture_module  # noqa: E402


class FrontierMemoryEndToEndTest(unittest.TestCase):
    scenario_compare: Path

    def test_external_capacity_covers_sparse_global_address_span(self) -> None:
        self.assertEqual(
            _external_address_capacity(
                profile_capacity_bytes=256,
                backing_population_bytes=128,
                object_address_space_bytes=480,
            ),
            480,
        )
        self.assertEqual(
            _external_address_capacity(
                profile_capacity_bytes=512,
                backing_population_bytes=128,
                object_address_space_bytes=480,
            ),
            512,
        )

    def test_audited_object_population_replays_on_canonical_baselines(
        self,
    ) -> None:
        audit_fixture = audit_fixture_module.AuditFrontierReplayTest(
            methodName="audit"
        )
        audit_fixture.setUp()
        try:
            audit_fixture.audit()
            root = Path(audit_fixture.root)
            model = root / "model.json"
            model_writer = (
                export_fixture_module.ExportFrontierMemoryTraceTest.__new__(
                    export_fixture_module.ExportFrontierMemoryTraceTest
                )
            )
            model_writer.model = model
            model_writer._write_model()
            trace = root / "memory.trace"
            object_map = root / "objects.json"
            phase_map = root / "phases.json"
            manifest_path = root / "memory.manifest.json"
            manifest = export_memory_trace(
                audit_path=audit_fixture.output,
                model_descriptor_path=model,
                output_trace=trace,
                object_map_path=object_map,
                phase_map_path=phase_map,
                manifest_path=manifest_path,
                max_batches=1,
            )
            output_dir = root / "memory-baselines"
            receipt = run_baselines(
                scenario_compare=self.scenario_compare,
                hardware_config=(
                    TOOLS.parent
                    / "configs/scenario_compare/usecase-2h6f.cfg"
                ),
                manifest_path=manifest_path,
                object_map_path=object_map,
                output_dir=output_dir,
                credit_limit=64,
                timeout_seconds=30,
            )
            self.assertEqual(
                set(receipt["baselines"]),
                {
                    "all_hbm_upper_bound",
                    "hbm_hbf",
                    "hbm_cxl_memory",
                    "hbm_nvme_ssd",
                },
            )
            self.assertEqual(
                receipt["source"]["model"]["workload_role"],
                "ci_smoke",
            )
            self.assertTrue(
                receipt["contract"]["backed_placement_is_byte_identical"]
            )
            self.assertTrue(
                receipt["contract"]["backed_scheduler_is_identical"]
            )
            self.assertTrue(
                receipt["eligibility"]["canonical_waf_valid"]
            )
            self.assertFalse(
                receipt["eligibility"]["paper_result_eligible"]
            )
            residency_contract = receipt["contract"]["residency"]
            phase_count = len(
                json.loads(phase_map.read_text(encoding="utf-8"))["phases"]
            )
            summaries = {}
            for baseline, baseline_receipt in receipt["baselines"].items():
                summary_path = (
                    output_dir / baseline_receipt["summary"]["path"]
                )
                summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                summaries[baseline] = summary
                self.assertEqual(
                    summary["validation"],
                    {
                        "status": "exploratory_unattached",
                        "certificate": None,
                    },
                )
                self.assertEqual(
                    summary["workload"]["trace_digest"]["value"],
                    manifest["outputs"]["trace"]["sha256"],
                )
                self.assertEqual(
                    summary["config"]["expected_trace_sha256"],
                    manifest["outputs"]["trace"]["sha256"],
                )
                self.assertEqual(
                    summary["config"]["expected_trace_bytes"],
                    manifest["outputs"]["trace"]["bytes"],
                )
                self.assertEqual(len(summary["scenarios"]), 1)
                scenario = summary["scenarios"][0]
                self.assertEqual(
                    scenario["name"],
                    baseline_receipt["scenario"],
                )
                self.assertEqual(
                    scenario["ops"],
                    manifest["traffic_census"]["operations"],
                )
                self.assertEqual(
                    scenario["logical_bytes"],
                    manifest["traffic_census"]["bytes"],
                )
                self.assertEqual(
                    scenario["reads"],
                    manifest["traffic_census"]["reads"]["operations"],
                )
                self.assertEqual(
                    scenario["writes"],
                    manifest["traffic_census"]["writes"]["operations"],
                )
                latency = scenario["time_breakdown"]["latency_work"]
                wall = scenario["time_breakdown"]["wall_clock_ns"]
                self.assertEqual(latency["phase_barriers"], phase_count - 1)
                self.assertGreater(
                    latency["phase_dependency_waited_ops"],
                    0,
                )
                self.assertGreater(
                    latency["phase_dependency_wait_work_ns"],
                    0,
                )
                self.assertGreater(
                    wall["last_offered_arrival_ns"],
                    wall["trace_origin_ns"],
                )
                self.assertAlmostEqual(
                    latency["source_to_user_completion_sum_work_ns"],
                    latency["phase_dependency_wait_work_ns"]
                    + latency["offered_to_user_completion_sum_work_ns"],
                    delta=max(
                        1e-6,
                        latency["source_to_user_completion_sum_work_ns"]
                        * 1e-12,
                    ),
                )
                self.assertFalse(scenario["warnings"])
            backed_names = (
                "hbm_hbf",
                "hbm_cxl_memory",
                "hbm_nvme_ssd",
            )
            backed_scenarios = [
                summaries[name]["scenarios"][0] for name in backed_names
            ]
            for name, scenario in zip(
                backed_names,
                backed_scenarios,
                strict=True,
            ):
                summary = summaries[name]
                self.assertTrue(
                    summary["config"]["explicit_residency_contract"]
                )
                streaming = scenario["layer_streaming"]
                self.assertTrue(
                    streaming["explicit_residency_contract"]
                )
                self.assertEqual(
                    streaming["capacity_pressure_basis_bytes"],
                    residency_contract[
                        "unique_resident_footprint_bytes"
                    ],
                )
                self.assertEqual(
                    streaming["unique_resident_footprint_bytes"],
                    residency_contract[
                        "page_allocated_unique_footprint_bytes"
                    ],
                )
                self.assertEqual(
                    streaming["footprint_page_rounding_bytes"],
                    residency_contract[
                        "footprint_page_rounding_bytes"
                    ],
                )
                self.assertEqual(scenario["hbf_user_accesses"], 0)
                self.assertEqual(
                    scenario["hbm_user_accesses"],
                    streaming["foreground_resident_page_accesses"]
                    + streaming["foreground_buffer_page_accesses"],
                )
            hbf = backed_scenarios[0]["layer_streaming"]
            for field in (
                "capacity_pressure_basis_bytes",
                "unique_resident_footprint_bytes",
                "hbm_only_resident_pages",
                "hot_kv_resident_pages",
                "model_weight_backing_pages",
                "cold_kv_backing_pages",
                "resident_physical_pages",
                "effective_layer_buffer_pages",
                "unused_hbm_pages",
            ):
                for external in backed_scenarios[1:]:
                    self.assertEqual(
                        hbf[field],
                        external["layer_streaming"][field],
                    )
            self.assertEqual(
                backed_scenarios[1]["external_backing_stats"]["kind"],
                "cxl-memory",
            )
            self.assertEqual(
                backed_scenarios[2]["external_backing_stats"]["kind"],
                "nvme-ssd",
            )
            all_hbm_summary = summaries["all_hbm_upper_bound"]
            self.assertFalse(
                all_hbm_summary["config"]["explicit_residency_contract"]
            )
            all_hbm = all_hbm_summary["scenarios"][0]
            self.assertEqual(all_hbm["hbf_accesses"], 0)
            self.assertEqual(all_hbm["external_accesses"], 0)
            self.assertFalse(
                receipt["contract"]["all_hbm_upper_bound"][
                    "capacity_expanded"
                ]
            )
            canonical_waf = receipt["contract"]["canonical_hbf_waf"]
            hbf_scenario = backed_scenarios[0]
            hbf_stats = hbf_scenario["hbf_stats"]
            self.assertEqual(
                canonical_waf["definition"],
                "physical_write_bytes/logical_write_bytes",
            )
            self.assertEqual(
                canonical_waf["logical_write_bytes"],
                hbf_stats["logical_write_bytes"],
            )
            self.assertEqual(
                canonical_waf["physical_write_bytes"],
                hbf_stats["physical_write_bytes"],
            )
            self.assertEqual(
                canonical_waf["write_source"],
                "mutable_kv_only",
            )
            self.assertEqual(
                canonical_waf["read_only_weight_write_bytes"],
                0,
            )
            self.assertFalse(
                canonical_waf[
                    "initial_image_included_in_workload_writes"
                ]
            )
            self.assertAlmostEqual(
                canonical_waf[
                    "physical_write_bytes_over_usable_hbf_capacity"
                ],
                canonical_waf["physical_write_bytes"]
                / canonical_waf["usable_hbf_capacity_bytes"],
            )
            if canonical_waf["logical_write_bytes"] == 0:
                self.assertIsNone(canonical_waf["waf"])
            else:
                self.assertAlmostEqual(
                    canonical_waf["waf"],
                    canonical_waf["physical_write_bytes"]
                    / canonical_waf["logical_write_bytes"],
                )
            self.assertNotIn(
                "canonical HBF WAF audit not attached",
                receipt["eligibility"]["paper_blockers"],
            )

            trace.write_bytes(trace.read_bytes() + b"# post-binding drift\n")
            command = receipt["baselines"]["hbm_hbf"]["command"]
            rejected = subprocess.run(
                command,
                cwd=output_dir,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "executed trace identity differs from the digest-bound "
                "residency config",
                rejected.stdout + rejected.stderr,
            )
        finally:
            audit_fixture.tearDown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-compare", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    FrontierMemoryEndToEndTest.scenario_compare = (
        arguments.scenario_compare.resolve()
    )
    unittest.main(argv=["test_frontier_memory_structural_integration.py"])
