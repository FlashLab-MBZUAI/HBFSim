#!/usr/bin/env python3
"""Native CLI coverage for every mapping organization and checkpoint restart."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hbfsim_client.simulation_session import ResolvedSystemConfig, SimulationSession, SimulationSessionError
from tests.python.mapping_organizations import PROFILES, VARIANTS, mapping_observations, run, submit

SIMULATOR = ROOT / "build/hbfsim"


class MappingOrganizations(unittest.TestCase):
    def test_disabled_hbf_has_no_mapping_observation(self):
        config = ResolvedSystemConfig.load((ROOT / "configs/systems/eight-stack-baseline.cfg",))
        with SimulationSession(simulator_path=SIMULATOR, system_config=config,
                enable_hbm=True, enable_hbf=False) as session:
            pass
        final = session.source_receipt()["final_measurement"]
        self.assertEqual(final["hbf_mapping_observations"], dict(foreground=None, post_drain=None))
        self.assertIsNone(mapping_observations(None, final))

    def test_serving_and_terminal_drain_keep_their_own_mapping_decomposition(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            override = directory / "pressure.cfg"
            override.write_text("hbf-blocks-per-plane=64\nhbf-pages-per-block=8\n"
                "hbf-logical-capacity-bytes=1638400\nhbf-ctrl-dram-bytes=0\n"
                "hbf-write-coalescing=true\nhbf-write-buffer-pages=2\n"
                "hbf-write-buffer-completion-requires-flush=false\n")
            for variant in ("resident", "block", "extent", "object-segment"):
                with self.subTest(variant=variant):
                    config = ResolvedSystemConfig.load((ROOT / "configs/systems/eight-stack-baseline.cfg",
                        PROFILES / "base.cfg", PROFILES / f"{variant}.cfg", override)).resolve(SIMULATOR)
                    with SimulationSession(simulator_path=SIMULATOR, system_config=config,
                            enable_hbm=True, enable_hbf=True, initial_hbf_logical_pages=400) as session:
                        initial = session.hbf_mapping_snapshot("initial")
                        rng = random.Random(19)
                        submit(session, "updates", [("W", rng.randrange(400) * 4096, 4096) for _ in range(800)])
                        foreground = session.hbf_mapping_snapshot("foreground")
                    final = session.source_receipt()["final_measurement"]
                    report = mapping_observations(initial, final)
                    self.assertEqual(report["foreground"], foreground)
                    self.assertFalse(report["post_drain"]["dirty"])
                    windows = report["windows"]
                    self.assertGreater(windows["terminal_drain"]["programs"]["data"], 0)
                    self.assertGreater(windows["terminal_drain"]["programs"]["metadata"], 0)
                    self.assertEqual(windows["total"]["physical_write_bytes"],
                        windows["foreground"]["physical_write_bytes"] + windows["terminal_drain"]["physical_write_bytes"])
                    for window in windows.values():
                        self.assertEqual(sum(window["program_payload_bytes"].values()), window["physical_write_bytes"])
                    maintenance = windows["foreground"]["maintenance"]
                    if variant == "resident":
                        self.assertGreater(maintenance["events"]["gc_runs"], 0)
                    else:
                        self.assertIsNone(report["post_drain"]["page_gc_runs"])
                        self.assertNotIn("gc_runs", maintenance["events"])
                        event = "block_replacements" if variant == "block" else "cleaned_segments"
                        self.assertGreater(maintenance["events"][event], 0)
                        self.assertGreater(windows["foreground"]["programs"]["copy"], 0)
                        self.assertEqual(final["device_workload_totals"]["hbf"]["gc_runs"], 0)
                    # A caller must not attach foreground categories to final media totals.
                    mismatched = deepcopy(final)
                    mismatched["hbf_mapping_observations"]["post_drain"] = foreground
                    with self.assertRaises(ValueError):
                        mapping_observations(initial, mismatched)

    def test_same_trace_all_policies_and_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = run(SIMULATOR, Path(temporary), operations=96, logical_pages=128,
                         blocks=64, pages_per_block=8, buffer_pages=2)
            self.assertEqual(tuple(row["variant"] for row in report["results"]), VARIANTS)
            for row in report["results"]:
                with self.subTest(variant=row["variant"]):
                    stats = row["saved"]
                    self.assertEqual(stats["logical_capacity_pages"], 128)
                    self.assertEqual(stats["page_programs"], sum(stats[k] for k in
                        ("data_programs", "copy_programs", "padding_programs", "metadata_programs")))
                    self.assertGreater(stats["metadata_programs"], 0)
                    self.assertFalse(stats["dirty"])
                    self.assertFalse(row["recovery_final"]["dirty"])
                    self.assertEqual(row["recovery_final"]["logical_write_bytes"], 64)
                    # Sixty-four cache-line writes become a single page program
                    # before checkpoint (mapping pages are counted separately).
                    fragment = next(phase for phase in row["phases"] if phase["name"]=="fragments")
                    self.assertEqual(fragment["after"]["data_programs"] - fragment["before"]["data_programs"], 1)

    def test_dense_initial_images_and_object_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            override = directory / "small.cfg"
            override.write_text("hbf-blocks-per-plane=64\nhbf-pages-per-block=8\n"
                                "hbf-logical-capacity-bytes=524288\n")
            for kind in ("block", "block-log", "extent", "object-segment"):
                config = ResolvedSystemConfig.load((ROOT / "configs/systems/eight-stack-baseline.cfg",
                    PROFILES / "base.cfg", PROFILES / f"{kind}.cfg", override))
                with SimulationSession(simulator_path=SIMULATOR, system_config=config,
                        enable_hbm=True, enable_hbf=True, initial_hbf_logical_first_lpn=3,
                        initial_hbf_logical_pages=61, hbf_wear_output_prefix=directory / kind) as session:
                    snapshot = session.hbf_mapping_snapshot("initial")
                    self.assertEqual(snapshot["page_programs"], 0)
                    submit(session, "mutable", [("R", 3 * 4096, 64), ("W", 3 * 4096, 64)])
                    if kind == "object-segment":
                        with self.assertRaises(SimulationSessionError):
                            session.hbf_object_command("CREATE", "overlap", object_id=1, first_lpn=3, page_count=1)
                        session.hbf_object_command("SEAL", "seal", object_id=0)
                        session.hbf_object_command("DELETE", "delete", object_id=0)
                    session.checkpoint("finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path, default=SIMULATOR)
    args, remaining = parser.parse_known_args()
    SIMULATOR = args.simulator.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
