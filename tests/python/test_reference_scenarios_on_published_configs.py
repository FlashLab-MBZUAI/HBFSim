#!/usr/bin/env python3
"""Every reference scenario accepts every published system/policy pair.

The published system profiles ship with a same-named reference policy, and
CTest replays each pair with ``--scenarios all-hbm --max-ops 1``. That never
exercised the other scenarios, so a policy boundary that the engine rejects
(the ``flat``/``direct-read`` boundary landing inside the cooperative HBM
write region) shipped unnoticed. This test runs every scenario declared in
``src/app/reference_runner.cpp`` on every pair that ``cmake/tests/
Simulation.cmake`` publishes, with the smoke trace.

Contract per (pair, scenario): the runner must not reject the configuration.
It either exits 0, or exits 1 with ``SANITY: FAIL`` caused only by sanity
warnings: the summary JSON exists and contains the scenario row, and no
stdout/stderr line starts with ``error:``. The core scenarios that every
profile is documented to support must additionally exit 0.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from test_system_configs import (  # noqa: E402
    cmake_published_profiles,
    published_system_profiles,
)

SYSTEM_CONFIGS = ROOT / "configs/systems"
POLICY_CONFIGS = ROOT / "configs/policies/reference"
RUNNER_SOURCE = ROOT / "src/app/reference_runner.cpp"

SCENARIO_CONSTANT = re.compile(
    r'constexpr\s+const\s+char\*\s+k\w+Scenario\s*=\s*"([^"]+)"\s*;')
# Scenarios every published pair must run cleanly (exit 0, sanity PASS).
CLEAN_SCENARIOS = ("all-hbm", "all-hbf", "flat", "direct-read", "hbf-streaming")


def published_pairs() -> list[str]:
    """The pair names CTest publishes that also have a same-named policy.

    ``test_system_configs`` owns the drift assertion that the CTest list
    (``cmake_published_profiles``) equals the policy-paired system set
    (``published_system_profiles``); this matrix runs their intersection so
    a drift fails that one contract rather than every cell here.
    """

    published = {name.removesuffix(".cfg") for name in cmake_published_profiles()}
    paired = {name.removesuffix(".cfg") for name in published_system_profiles()}
    return sorted(published & paired)


def reference_scenarios() -> list[str]:
    """Every scenario the reference runner declares by name."""

    return SCENARIO_CONSTANT.findall(RUNNER_SOURCE.read_text(encoding="utf-8"))


class PublishedScenarioMatrixTest(unittest.TestCase):
    reference: Path

    def test_scenarios_are_declared(self) -> None:
        scenarios = reference_scenarios()
        self.assertTrue(scenarios)
        self.assertEqual(len(scenarios), len(set(scenarios)))
        self.assertLessEqual(set(CLEAN_SCENARIOS), set(scenarios))

    def test_every_scenario_runs_on_every_published_pair(self) -> None:
        pairs = published_pairs()
        self.assertTrue(pairs)
        scenarios = reference_scenarios()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.txt"
            generated = self.run_reference(
                "--generate-semantic-llm", str(trace),
                "--llm-tokens", "4",
                "--llm-layers", "2",
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            cells = [(pair, scenario) for pair in pairs for scenario in scenarios]
            workers = max(1, min(4, os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                verdicts = list(pool.map(
                    lambda cell: self.run_cell(root, trace, *cell), cells))
        for (pair, scenario), verdict in zip(cells, verdicts):
            with self.subTest(pair=pair, scenario=scenario):
                self.assertIsNone(verdict.rejection, verdict.rejection)
                if scenario in CLEAN_SCENARIOS:
                    self.assertEqual(
                        verdict.returncode, 0,
                        f"{pair}/{scenario} must run cleanly:\n{verdict.log}")

    class Verdict:
        def __init__(self, returncode: int, rejection: str | None, log: str) -> None:
            self.returncode = returncode
            self.rejection = rejection
            self.log = log

    def run_cell(self, root: Path, trace: Path, pair: str, scenario: str) -> "Verdict":
        summary = root / f"{pair}-{scenario}.json"
        completed = self.run_reference(
            "--config", str(SYSTEM_CONFIGS / f"{pair}.cfg"),
            "--config", str(POLICY_CONFIGS / f"{pair}.cfg"),
            "--trace", str(trace),
            "--scenarios", scenario,
            "--summary-json", str(summary),
        )
        log = f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        rejection = self.rejection(completed, summary, scenario, log)
        return self.Verdict(completed.returncode, rejection, log)

    @staticmethod
    def rejection(completed: subprocess.CompletedProcess[str], summary: Path,
                  scenario: str, log: str) -> str | None:
        """Why the runner rejected the configuration, or None if it ran."""

        errors = [line for line in (completed.stdout + completed.stderr).splitlines()
                  if line.startswith("error:")]
        if errors:
            return f"runner error {errors}:\n{log}"
        if completed.returncode not in (0, 1):
            return f"exit {completed.returncode}:\n{log}"
        if completed.returncode == 1 and "SANITY: FAIL" not in completed.stdout:
            return f"exit 1 without a sanity verdict:\n{log}"
        if not summary.is_file():
            return f"no summary JSON written:\n{log}"
        document = json.loads(summary.read_text(encoding="utf-8"))
        names = [row.get("name") for row in document.get("scenarios", [])]
        if scenario not in names:
            return f"summary JSON lacks the {scenario} row (has {names}):\n{log}"
        return None

    def run_reference(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.reference), *arguments],
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    PublishedScenarioMatrixTest.reference = args.reference.resolve()
    unittest.main(argv=[__file__, *remaining])
