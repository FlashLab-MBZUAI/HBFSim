#!/usr/bin/env python3
"""The ``python3 -m hbfsim`` front door stays truthful to the simulator.

Pure checks cover name resolution, scenario and metric catalogs, sweep grids
and table rendering. End-to-end checks run the compiled programs from
``--build-dir`` and prove that the front door passes configurations through
unchanged, reports what the summary JSON says, and fails closed with a hint.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hbfsim  # noqa: E402
from hbfsim import catalog, cli, results, runner  # noqa: E402
from hbfsim.workspace import HbfsimError  # noqa: E402


SCENARIO_CONSTANT = re.compile(
    r'constexpr\s+const\s+char\*\s+k\w+Scenario\s*=\s*"([^"]+)"\s*;'
)


class CatalogTest(unittest.TestCase):
    def test_scenarios_match_the_reference_runner_source(self) -> None:
        source = (ROOT / "src/app/reference_runner.cpp").read_text(encoding="utf-8")
        declared = set(SCENARIO_CONSTANT.findall(source))
        self.assertEqual(set(catalog.SCENARIOS), declared)
        self.assertLessEqual(set(catalog.CORE_SCENARIOS), declared)

    def test_every_shipped_profile_resolves_by_name_and_stem(self) -> None:
        for kind in catalog.KIND_DIRECTORIES:
            entries = catalog.profiles(kind)
            self.assertTrue(entries, kind)
            for entry in entries:
                self.assertEqual(catalog.resolve(kind, entry.name), entry.path)
                self.assertEqual(catalog.resolve(kind, entry.path), entry.path)
                self.assertTrue(entry.description, entry.name)

    def test_short_forms_and_suggestions(self) -> None:
        overlays = ROOT / "configs/overlays"
        self.assertEqual(catalog.resolve("overlay", "ocp-v070-grade3"),
                         (overlays / "hbf/ocp-v070-grade3.cfg").resolve())
        self.assertEqual(catalog.resolve("overlay", "mapping/block"),
                         (overlays / "hbf/mapping/block.cfg").resolve())
        # An exact name wins over a same-stem profile in a subdirectory.
        self.assertEqual(catalog.resolve("system", "4hbm-4hbf"),
                         (ROOT / "configs/systems/4hbm-4hbf.cfg").resolve())
        with self.assertRaisesRegex(HbfsimError, "hbf/ocp-v070-grade3"):
            catalog.resolve("overlay", "grade3")
        with self.assertRaisesRegex(HbfsimError, "4hbm-4hbf"):
            catalog.resolve("system", "4hbm4hbf")

    def test_ambiguous_stems_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CMakeLists.txt").write_text("", encoding="utf-8")
            for sub in ("a", "b"):
                path = root / "configs/overlays" / sub / "same.cfg"
                path.parent.mkdir(parents=True)
                path.write_text("# One overlay.\nhbm-stacks=1\n", encoding="utf-8")
            with mock_env(HBFSIM_ROOT=str(root)):
                with self.assertRaisesRegex(HbfsimError, "ambiguous"):
                    catalog.resolve("overlay", "same")
                self.assertEqual(catalog.resolve("overlay", "a/same"),
                                 (root / "configs/overlays/a/same.cfg").resolve())

    def test_policy_pairing_is_by_file_name_only(self) -> None:
        server = catalog.resolve("system", "server-hbm128-hbf512")
        self.assertEqual(
            catalog.default_policy(server),
            (ROOT / "configs/policies/reference/server-hbm128-hbf512.cfg").resolve(),
        )
        self.assertIsNone(catalog.default_policy(catalog.resolve("system", "sglang-small")))

    def test_scenario_selection(self) -> None:
        self.assertEqual(catalog.scenario_list(None), catalog.CORE_SCENARIOS)
        self.assertEqual(catalog.scenario_list("all"), tuple(catalog.SCENARIOS))
        self.assertEqual(catalog.scenario_list("flat,all-hbm,flat"), ("flat", "all-hbm"))
        with self.assertRaisesRegex(HbfsimError, "all-hbm"):
            catalog.scenario_list("al-hbm")
        with self.assertRaises(HbfsimError):
            catalog.scenario_list("")

    def test_config_parser_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.cfg"
            path.write_text("hbm-stacks 4\n", encoding="utf-8")
            with self.assertRaisesRegex(HbfsimError, "key=value"):
                catalog.parse_config(path)


class GridAndTableTest(unittest.TestCase):
    def test_assignments_and_axes(self) -> None:
        self.assertEqual(runner.parse_assignments(["--hbf-read-ns=8000", "a-b=c"]),
                         {"hbf-read-ns": "8000", "a-b": "c"})
        for bad in ("novalue=", "=1", "no-separator", "Bad_Key=1"):
            with self.assertRaises(HbfsimError, msg=bad):
                runner.parse_assignments([bad])
        axes = runner.parse_axes(["hbf-read-ns=2000,4000", "system=a,b,c"])
        grid = runner.sweep_grid(axes)
        self.assertEqual(len(grid), 6)
        self.assertEqual(grid[0], {"hbf-read-ns": "2000", "system": "a"})
        self.assertEqual(grid[-1], {"hbf-read-ns": "4000", "system": "c"})
        with self.assertRaises(HbfsimError):
            runner.parse_axes(["x=1", "x=2"])
        with self.assertRaises(HbfsimError):
            runner.sweep_grid({})

    def test_tables(self) -> None:
        rows = [{"scenario": "all-hbm", "makespan_us": 3.383, "hbf_waf": None, "hbm_accesses": 1920},
                {"scenario": "all-hbf", "makespan_us": 5177.1, "hbf_waf": 3.6667, "hbm_accesses": 0}]
        metrics = ("makespan_us", "hbm_accesses", "hbf_waf")
        text = results.format_table(rows, metrics=metrics)
        self.assertIn("makespan (us)", text)
        self.assertIn("1,920", text)
        self.assertIn("5,177", text)
        self.assertIn("-", text.splitlines()[2])
        markdown = results.format_table(rows, metrics=metrics, style="markdown")
        self.assertTrue(markdown.startswith("| scenario |"))
        csv_text = results.format_table(rows, metrics=metrics, style="csv")
        self.assertEqual(csv_text.splitlines()[0], "scenario,makespan_us,hbm_accesses,hbf_waf")
        self.assertEqual(json.loads(results.format_table(rows, metrics=metrics, style="json"))[1]["hbf_waf"],
                         3.6667)
        self.assertEqual(results.format_value(0.0000123), "1.230e-05")
        self.assertEqual(results.format_value(True), "yes")

    def test_schema_version_follows_the_published_contract(self) -> None:
        from reports.address_heatmap import SUMMARY_SCHEMA_VERSION
        self.assertEqual(results.SUMMARY_SCHEMA["version"], SUMMARY_SCHEMA_VERSION)


class EndToEndTest(unittest.TestCase):
    build_dir: Path

    def setUp(self) -> None:
        self.environment = mock_env(HBFSIM_BUILD_DIR=str(self.build_dir))
        self.environment.__enter__()
        self.addCleanup(self.environment.__exit__, None, None, None)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_run_matches_a_direct_reference_invocation(self) -> None:
        result = hbfsim.run(out_dir=self.directory / "run")
        self.assertTrue(result.ok, [s.warnings for s in result.scenarios])
        self.assertEqual([s.name for s in result.scenarios], list(catalog.CORE_SCENARIOS))
        for name in ("command.txt", "resolved.cfg", "stdout.txt", "trace.txt"):
            self.assertTrue((self.directory / "run" / name).is_file(), name)
        # The front door adds nothing: the recorded command reproduces the summary.
        replay = self.directory / "replay.json"
        command = [part if not part.endswith("summary.json") else str(replay)
                   for part in result.command]
        completed = subprocess.run(command, capture_output=True, text=True, check=False,
                                   cwd=self.directory)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        original = json.loads(result.summary_path.read_text(encoding="utf-8"))
        replayed = json.loads(replay.read_text(encoding="utf-8"))
        self.assertEqual(original["scenarios"], replayed["scenarios"])
        all_hbm = result.scenario("all-hbm")
        raw = all_hbm.raw["time_breakdown"]["wall_clock_ns"]["makespan_ns"]
        self.assertAlmostEqual(all_hbm["makespan_us"], raw / 1000)
        self.assertEqual(all_hbm["hbf_accesses"], 0)
        self.assertGreater(result.scenario("all-hbf")["hbf_accesses"], 0)

    def test_overrides_reach_the_simulator(self) -> None:
        slow = hbfsim.run(scenarios="all-hbf", options={"hbf-read-ns": 8000},
                          out_dir=self.directory / "slow")
        fast = hbfsim.run(scenarios="all-hbf", options={"hbf-read-ns": 2000},
                          out_dir=self.directory / "fast")
        self.assertIn("hbf-read-ns=8000", (self.directory / "slow/resolved.cfg").read_text())
        self.assertGreater(slow.scenario("all-hbf")["mean_latency_us"],
                           fast.scenario("all-hbf")["mean_latency_us"])

    def test_output_directories_are_never_mixed(self) -> None:
        hbfsim.run(scenarios="all-hbm", out_dir=self.directory / "once")
        with self.assertRaisesRegex(HbfsimError, "already contains"):
            hbfsim.run(scenarios="all-hbm", out_dir=self.directory / "once")

    def test_sweep_writes_one_row_per_point_and_scenario(self) -> None:
        points = hbfsim.sweep(vary={"hbf-read-ns": [2000, 4000], "overlay": ["none", "ocp-v070-grade1"]},
                              scenarios="all-hbm,all-hbf", out_dir=self.directory / "sweep", jobs=2)
        self.assertEqual(len(points), 4)
        lines = (self.directory / "sweep/sweep.csv").read_text().strip().splitlines()
        self.assertEqual(len(lines), 1 + 4 * 2)
        self.assertTrue(lines[0].startswith("hbf-read-ns,overlay,scenario,"))
        manifest = json.loads((self.directory / "sweep/sweep.json").read_text())
        self.assertEqual(len(manifest["points"]), 4)
        grade1 = self.directory / "sweep" / manifest["points"][1]["directory"]
        self.assertIn("ocp-v070-grade1.cfg", (grade1 / "command.txt").read_text())

    def test_sweep_keeps_going_past_a_rejected_point(self) -> None:
        # The second boundary sits inside the cooperative write region (see
        # test_engine_errors_carry_a_hint); the engine rejects that point only.
        capacity = catalog.parse_config(catalog.resolve("system", "4hbm-4hbf"))["hbm-capacity-bytes"]
        points = hbfsim.sweep(vary={"flat-hbm-bytes": ["536870912", capacity]},
                              system="4hbm-4hbf", scenarios="flat",
                              out_dir=self.directory / "partial")
        self.assertIsNotNone(points[0].result)
        self.assertIsNone(points[1].result)
        self.assertIn("flat-hbm-bytes", points[1].error)
        manifest = json.loads((self.directory / "partial/sweep.json").read_text())
        self.assertIsNotNone(manifest["points"][1]["error"])

    def test_engine_errors_carry_a_hint(self) -> None:
        # A flat boundary at the full HBM capacity lands inside the cooperative
        # write region; the engine rejects it and the front door explains why.
        capacity = catalog.parse_config(catalog.resolve("system", "4hbm-4hbf"))["hbm-capacity-bytes"]
        with self.assertRaisesRegex(
                HbfsimError, r"hint: .*cooperative_write_region_base=") as caught:
            hbfsim.run(system="4hbm-4hbf", scenarios="flat",
                       options={"flat-hbm-bytes": capacity},
                       out_dir=self.directory / "flat")
        self.assertIn("overlaps the cooperative write region", str(caught.exception))
        self.assertIn("hbf-hbm-write-buffer-bytes", str(caught.exception))

    def test_cli_commands(self) -> None:
        for argv in (["list"], ["doctor"],
                     ["quickstart", "--no-build", "--out", str(self.directory / "qs")],
                     ["show", str(self.directory / "qs"), "--format", "csv"]):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = cli.main(argv)
            self.assertEqual(status, 0, (argv, output.getvalue()))
        self.assertIn("Sanity check: PASS", self.cli_output(
            ["quickstart", "--no-build", "--out", str(self.directory / "qs2")]))
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            self.assertEqual(cli.main(["run", "--system", "no-such-system"]), 2)
        self.assertIn("unknown system", errors.getvalue())

    def cli_output(self, argv: list[str]) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli.main(argv)
        return output.getvalue()

    def test_open_session_resolves_ratio_sized_profiles(self) -> None:
        from hbfsim_client import Transaction

        with hbfsim.open_session("server-hbm128-hbf512", options={"hbf-read-ns": 2000},
                                 out_dir=self.directory / "session",
                                 initial_hbf_logical_pages=64) as session:
            batch = [Transaction(id=f"r{index}", target="HBF_LOGICAL", op="R",
                                 addr=index * 4096, bytes=4096, issue_ns=0.0)
                     for index in range(64)]
            result = session.run(batch)
        self.assertGreater(result.elapsed_ns, 2000)
        self.assertIn("hbf-read-ns=2000", (self.directory / "session/overrides.cfg").read_text())
        self.assertIn("hbf-blocks-per-plane=",
                      (self.directory / "session/resolved-geometry.cfg").read_text())


@contextlib.contextmanager
def mock_env(**values: str):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="CMake build directory; without it only pure checks run")
    args, remaining = parser.parse_known_args()
    if args.build_dir is None:
        del EndToEndTest
    else:
        EndToEndTest.build_dir = args.build_dir.resolve()
    unittest.main(argv=[__file__, *remaining])
