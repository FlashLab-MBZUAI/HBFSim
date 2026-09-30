#!/usr/bin/env python3
"""Regression tests for analyze_trace_locality.py."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / "workloads" / "trace_analysis.py"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from workloads.trace_analysis import (  # noqa: E402
    TraceInputError,
    analyze_trace,
    parse_trace_line,
)


class TraceLocalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_trace(self, contents: str) -> Path:
        path = self.root / "trace.txt"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_exact_global_and_per_kind_metrics(self) -> None:
        trace = self.write_trace(
            "# address order and op-first order are both legal\n"
            "0x0 R 4096 model_weights at=0 # inline comment\n"
            "READ 0x1000 bytes=4096 kind=weights arrival=1\n"
            "0x0 W size=4096 generated_context\n"
            "0x1ff0 R 32 kind=shared_context at=2.5\n")
        report = analyze_trace(trace)
        overall = report["overall"]

        self.assertEqual(report["schema"], {
            "name": "hbfsim.trace_locality", "version": 2})
        self.assertEqual(
            report["trace_sha256"], hashlib.sha256(trace.read_bytes()).hexdigest())
        self.assertEqual(overall["ops"], 4)
        self.assertEqual(overall["reads"], 3)
        self.assertEqual(overall["writes"], 1)
        self.assertEqual(overall["bytes"], {
            "total": 12_320, "read": 8_224, "write": 4_096})
        self.assertEqual(overall["address"], {
            "min": 0,
            "max_inclusive": 8_207,
            "max_exclusive": 8_208,
            "span_bytes": 8_208,
        })
        self.assertEqual(overall["unique_occupied_bytes"], 8_208)
        self.assertEqual(overall["unique_occupied_pages"], 3)
        self.assertEqual(overall["page_touches"], 5)
        self.assertAlmostEqual(overall["reuse_ratio"], 0.4)
        self.assertAlmostEqual(
            overall["byte_coverage_reuse_ratio"], (12_320 - 8_208) / 12_320)
        self.assertEqual(overall["exact_address_signature_repeat_ops"], 1)
        self.assertAlmostEqual(
            overall["exact_address_signature_repeat_share"], 0.25)
        self.assertEqual(overall["top_1_percent_pages"], 1)
        self.assertAlmostEqual(
            overall["top_1_percent_page_traffic_share"], 8_192 / 12_320)
        self.assertAlmostEqual(overall["sequential_adjacency_fraction"], 1 / 3)
        self.assertEqual(overall["stride_abs_delta_bytes_p50"], 4_096)
        self.assertEqual(overall["stride_abs_delta_bytes_p95"], 8_176)
        temporal = report["temporal_page_locality"]
        self.assertEqual(temporal["cold_page_touches"], 3)
        self.assertEqual(temporal["reuse_page_touches"], 2)
        self.assertEqual(temporal["reuse_distance_pages_p50"], 1)
        self.assertEqual(temporal["reuse_distance_pages_p95"], 1)
        self.assertIsNone(temporal["reuse_within_hbm_ratio"])

        weights = report["per_kind"]["model_weights"]
        self.assertEqual(weights["ops"], 2)
        self.assertEqual(weights["unique_occupied_bytes"], 8_192)
        self.assertEqual(weights["sequential_adjacency_fraction"], 1.0)
        self.assertEqual(report["per_kind"]["metadata"]["ops"], 0)
        self.assertIsNone(
            report["per_kind"]["metadata"]["sequential_adjacency_fraction"])

    def test_default_size_aliases_decimal_and_free_form_label(self) -> None:
        trace = self.write_trace(
            "00016 load activation at=0\n"
            "store 80 32 label_without_semantics\n"
            "96 R label=custom bytes=16\n")
        report = analyze_trace(trace, line_size=128, page_size=64)
        overall = report["overall"]
        self.assertEqual(overall["ops"], 3)
        self.assertEqual(overall["bytes"]["total"], 176)
        self.assertEqual(report["per_kind"]["scratch"]["bytes"]["total"], 128)
        self.assertEqual(report["per_kind"]["unknown"]["ops"], 2)

    def test_multi_page_request_is_byte_exact(self) -> None:
        trace = self.write_trace("0xff0 R 0x2020 metadata\n")
        report = analyze_trace(trace)
        overall = report["overall"]
        self.assertEqual(overall["page_touches"], 4)
        self.assertEqual(overall["unique_occupied_pages"], 4)
        self.assertEqual(overall["unique_occupied_bytes"], 0x2020)
        self.assertEqual(overall["top_1_percent_pages"], 1)
        self.assertAlmostEqual(
            overall["top_1_percent_page_traffic_share"], 4096 / 0x2020)

    def test_exact_lru_reuse_distance_and_hbm_fit(self) -> None:
        trace = self.write_trace(
            "0x0 R 4096\n"
            "0x1000 R 4096\n"
            "0x0 R 4096\n"
            "0x2000 R 4096\n"
            "0x1000 R 4096\n"
            "0x0 R 4096\n")
        report = analyze_trace(
            trace,
            hbm_capacity_bytes=2 * 4096,
        )
        temporal = report["temporal_page_locality"]
        self.assertEqual(temporal["cold_page_touches"], 3)
        self.assertEqual(temporal["reuse_page_touches"], 3)
        self.assertEqual(temporal["reuse_distance_pages_p50"], 2)
        self.assertEqual(temporal["reuse_distance_pages_p95"], 2)
        self.assertEqual(temporal["reuse_distance_pages_max"], 2)
        self.assertEqual(temporal["reuse_within_hbm_pages"], 1)
        self.assertAlmostEqual(temporal["reuse_within_hbm_ratio"], 1 / 3)

    def test_hbm_capacity_must_be_page_aligned(self) -> None:
        trace = self.write_trace("0x0 R 4096\n")
        with self.assertRaisesRegex(TraceInputError, "page-size aligned"):
            analyze_trace(trace, hbm_capacity_bytes=4097)

    def test_malformed_and_overflow_fail_closed(self) -> None:
        bad_lines = (
            "0x0 R 0\n",
            "0xffffffffffffffff R 2\n",
            "0x0 X 64\n",
            "0x0 R bytes=nope\n",
            "0x0 R 64 bytes=64\n",
            "0x0 R 64 at=nan\n",
            "0x0 R 64 at=-1\n",
            "0x0 R 64 kind=typo\n",
            "0x0 R 64 bytse=32\n",
            "0x0 R 1_024\n",
            "0x0 R 64 at=1_0\n",
            "-1 R 64\n",
        )
        for contents in bad_lines:
            with self.subTest(contents=contents):
                trace = self.write_trace(contents)
                with self.assertRaises(TraceInputError):
                    analyze_trace(trace)

    def test_empty_and_invalid_utf8_fail_closed(self) -> None:
        with self.assertRaises(TraceInputError):
            analyze_trace(self.write_trace("# no requests\n\n"))
        trace = self.root / "invalid.trace"
        trace.write_bytes(b"0x0 R 64\n\xff")
        with self.assertRaises(TraceInputError):
            analyze_trace(trace)

    def test_analysis_limits_stop_large_inputs_before_page_expansion(self) -> None:
        trace = self.write_trace("0x0 R 0x100000000\n")
        with self.assertRaisesRegex(TraceInputError, "max-page-fragments"):
            analyze_trace(trace, max_page_fragments=10)

        trace = self.write_trace("0x0 R 64\n0x40 R 64\n")
        with self.assertRaisesRegex(TraceInputError, "max-ops"):
            analyze_trace(trace, max_ops=1)

    def test_cli_output_is_deterministic_and_not_replaced_on_error(self) -> None:
        trace = self.write_trace("0x100 R 64 scratch at=0\n")
        output = self.root / "nested" / "locality.json"
        command = [
            sys.executable,
            "-B",
            str(ANALYZER),
            "--trace",
            str(trace),
            "--output",
            str(output),
        ]
        first = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(first.stdout, "")
        first_bytes = output.read_bytes()
        parsed = json.loads(first_bytes)
        self.assertEqual(parsed["overall"]["ops"], 1)
        subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(output.read_bytes(), first_bytes)

        trace.write_text("0xffffffffffffffff R 2\n", encoding="utf-8")
        failed = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("overflows uint64", failed.stderr)
        self.assertEqual(output.read_bytes(), first_bytes)

    def test_cli_rejects_normalized_same_input_output_without_clobber(self) -> None:
        trace = self.write_trace("0x100 R 64 scratch at=0\n")
        original = trace.read_bytes()
        # The spelling differs, but normalization names the input itself.
        output = self.root / "unused" / ".." / trace.name
        completed = subprocess.run([
            sys.executable, "-B",
            str(ANALYZER),
            "--trace", str(trace), "--output", str(output),
        ], check=False, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("trace and output paths must be different", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(trace.read_bytes(), original)

    def test_cli_rejects_existing_hardlink_alias_without_clobber(self) -> None:
        trace = self.write_trace("0x200 W 64 generated_context at=0\n")
        alias = self.root / "report.json"
        os.link(trace, alias)
        original = trace.read_bytes()
        self.assertTrue(trace.samefile(alias))
        completed = subprocess.run([
            sys.executable, "-B",
            str(ANALYZER),
            "--trace", str(trace), "--output", str(alias),
        ], check=False, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("trace and output paths must be different", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(trace.read_bytes(), original)
        self.assertEqual(alias.read_bytes(), original)

    def test_parse_line_rejects_duplicate_kind_and_arrival(self) -> None:
        with self.assertRaises(TraceInputError):
            parse_trace_line(
                "0x0 R 64 scratch kind=metadata",
                line_no=1,
                line_size=64,
            )
        with self.assertRaises(TraceInputError):
            parse_trace_line(
                "0x0 R 64 at=0 arrival=1",
                line_no=1,
                line_size=64,
            )

    def test_phase_layer_and_compute_are_independent(self) -> None:
        operation = parse_trace_line(
            "0x0 R 64 phase=3 layer=7 compute_ns=12.5",
            line_no=1,
            line_size=64,
        )
        self.assertIsNotNone(operation)
        assert operation is not None
        self.assertEqual(operation.arrival_ns, 0.0)

        arrived = parse_trace_line(
            "0x0 R 64 at=12.5",
            line_no=1,
            line_size=64,
        )
        self.assertIsNotNone(arrived)
        assert arrived is not None
        self.assertEqual(arrived.arrival_ns, 12.5)

        bad_lines = (
            "0x0 R 64 phase=1 phase=2",
            "0x0 R 64 layer=1 layer=2",
            "0x0 R 64 compute_ns=1 compute=2",
            "0x0 R 64 compute_ns=-1",
            "0x0 R 64 compute_ns=nan",
        )
        for contents in bad_lines:
            with self.subTest(contents=contents):
                with self.assertRaises(TraceInputError):
                    parse_trace_line(
                        contents,
                        line_no=1,
                        line_size=64,
                    )


if __name__ == "__main__":
    unittest.main()
