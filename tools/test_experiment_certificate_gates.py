#!/usr/bin/env python3
"""Check paper/debug runner certificate publication boundaries."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_RUNNERS = {
    "run_paper_capacity_overflow.py",
}


class ExperimentCertificateGateTests(unittest.TestCase):
    def test_every_paper_runner_requires_certificate(self) -> None:
        runners = sorted((ROOT / "tools").glob("run_paper_*.py"))
        self.assertEqual(
            {script.name for script in runners},
            PAPER_RUNNERS,
            "publication entry points require an explicit reviewed allowlist",
        )
        for script in runners:
            with self.subTest(script=script.name):
                completed = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=ROOT,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    "--validation-certificate",
                    completed.stderr,
                )

    def test_debug_runners_expose_optional_certificate_flag(self) -> None:
        for script in (
            "replay_astra_trace.py",
            "run_synthetic_experiments.py",
            "run_waf_cases.py",
        ):
            with self.subTest(script=script):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools" / script),
                        "--help",
                    ],
                    cwd=ROOT,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(
                    "--validation-certificate",
                    completed.stdout,
                )


if __name__ == "__main__":
    unittest.main()
