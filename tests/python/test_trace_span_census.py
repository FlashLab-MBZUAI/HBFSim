#!/usr/bin/env python3
"""Verify that huge span traces are counted by interval, not cache line."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class TraceSpanCensusTest(unittest.TestCase):
    simulator: Path

    def test_comment_only_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "empty.trace"
            trace.write_text("# intentionally empty\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    str(self.simulator),
                    "--trace",
                    str(trace),
                    "--trace-census-only",
                    "true",
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("trace contains no memory operations", completed.stderr)

    def test_terabyte_overlapping_spans_finish_without_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "huge.trace"
            trace.write_text(
                "0x0 R 1099511627776 kind=model_weights phase=0 at=0\n"
                "0x8000000000 W 549755813888 kind=generated_context phase=1 at=1\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    str(self.simulator),
                    "--trace",
                    str(trace),
                    "--trace-census-only",
                    "true",
                ],
                text=True,
                capture_output=True,
                timeout=30,
                check=True,
            )
            census = json.loads(completed.stdout)
            self.assertEqual(census["operations"], 2)
            self.assertEqual(census["reads"], 1)
            self.assertEqual(census["writes"], 1)
            self.assertEqual(census["logical_bytes"], 1649267441664)
            self.assertEqual(census["unique_line_bytes"], 1099511627776)
            self.assertEqual(census["explicit_phases"], 2)
            self.assertEqual(
                census["trace_sha256"],
                hashlib.sha256(trace.read_bytes()).hexdigest(),
            )

    def test_census_rejects_malformed_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "bad.trace"
            trace.write_text(
                "0x0 R 64 kind=model_weights phase=bad\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    str(self.simulator),
                    "--trace",
                    str(trace),
                    "--trace-census-only",
                    "true",
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("invalid phase id", completed.stderr)

    def test_census_keeps_layer_identity_separate_from_phase_dependency(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "layer-and-phase.trace"
            trace.write_text(
                "0x0 R 64 kind=model_weights layer=1 phase=2\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    str(self.simulator),
                    "--trace",
                    str(trace),
                    "--trace-census-only",
                    "true",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            census = json.loads(completed.stdout)
            self.assertEqual(census["explicit_phases"], 1)

    def test_census_rejects_duplicate_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "duplicate-phase.trace"
            trace.write_text(
                "0x0 R 64 kind=model_weights phase=1 phase=2\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    str(self.simulator),
                    "--trace",
                    str(trace),
                    "--trace-census-only",
                    "true",
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate phase field", completed.stderr)

    def census(self, line: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "line.trace"
            trace.write_text(line + "\n", encoding="utf-8")
            return subprocess.run(
                [
                    str(self.simulator),
                    "--trace",
                    str(trace),
                    "--trace-census-only",
                    "true",
                ],
                text=True,
                capture_output=True,
            )

    def test_malformed_byte_counts_are_hard_errors(self) -> None:
        # A count that does not parse must not silently demote to a label
        # and run the op at --line-size.
        for line, message in (
            ("0x0 R bytes=abc", "invalid byte count"),
            ("0x0 R size=x", "invalid byte count"),
            ("0x0 R -64", "invalid byte count"),
            ("0x0 R +", "invalid byte count"),
            ("0x1000 R 64 128", "more than one byte count"),
            ("0x1000 R 64 bytes=128", "more than one byte count"),
        ):
            with self.subTest(line=line):
                completed = self.census(line)
                self.assertNotEqual(completed.returncode, 0, completed.stdout)
                self.assertIn(message, completed.stderr)

    def test_bare_count_and_free_form_label_still_parse(self) -> None:
        completed = self.census("0x1000 R 128 my_region kind=model_weights")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        census = json.loads(completed.stdout)
        self.assertEqual(census["logical_bytes"], 128)
        self.assertEqual(census["read_bytes"], 128)
        hex_count = self.census("0x1000 W bytes=0x40")
        self.assertEqual(hex_count.returncode, 0, hex_count.stderr)
        self.assertEqual(json.loads(hex_count.stdout)["write_bytes"], 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    TraceSpanCensusTest.simulator = arguments.simulator.resolve()
    unittest.main(argv=["test_trace_span_census.py"])
