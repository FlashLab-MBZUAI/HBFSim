#!/usr/bin/env python3
"""Regression tests for plot_trace_heatmap.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "plot_trace_heatmap.py"
SPEC = importlib.util.spec_from_file_location("plot_trace_heatmap", TOOL)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {TOOL}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TraceHeatmapUnitTests(unittest.TestCase):
    def test_cross_bin_overlap_is_byte_exact_in_both_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "cross.trace"
            trace.write_text(
                "0x8 R 10 model_weights\n"
                "0xe W 8 shared_context\n",
                encoding="utf-8",
            )
            analysis = MODULE.analyze_trace(trace, 32, 4)

            weights = MODULE.KIND_INDEX["model_weights"]
            shared = MODULE.KIND_INDEX["shared_context"]
            self.assertEqual(analysis.full.read[weights], [0, 8, 2, 0])
            self.assertEqual(analysis.full.write[shared], [0, 2, 6, 0])
            self.assertEqual(analysis.zoom.read[weights], [4, 3, 3, 0])
            self.assertEqual(analysis.zoom.write[shared], [0, 1, 4, 3])
            self.assertEqual(sum(map(sum, analysis.full.read)), 10)
            self.assertEqual(sum(map(sum, analysis.zoom.write)), 8)
            self.assertEqual(analysis.full.occupied_bins(), 2)
            self.assertEqual(analysis.zoom.occupied_bins(), 4)

    def test_semantic_alias_filter_reverse_grammar_and_default_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "aliases.trace"
            trace.write_text(
                "READ 0x100 bytes=16 kind=weight at=1.5\n"
                "0x200 STORE shared-kv\n"
                "0x300 R 8 label=untyped\n",
                encoding="utf-8",
            )
            analysis = MODULE.analyze_trace(
                trace, 1024, 8, line_size=32,
                selected_kinds=("model_weights", "shared_context"))
            self.assertEqual(analysis.summary.parsed_operations, 3)
            self.assertEqual(analysis.summary.selected_operations, 2)
            self.assertEqual(analysis.summary.totals.read_bytes, 16)
            self.assertEqual(analysis.summary.totals.write_bytes, 32)
            self.assertEqual(analysis.summary.observed_begin, 0x100)
            self.assertEqual(analysis.summary.observed_end, 0x220)

    def test_more_bins_than_bytes_remains_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "tiny.trace"
            trace.write_text("0 R 1\n", encoding="utf-8")
            analysis = MODULE.analyze_trace(trace, 1, 16, line_size=1)
            self.assertEqual(analysis.full.read[0], [1] + [0] * 15)
            self.assertEqual(analysis.zoom.occupied_bins(), 1)

    def test_capacity_and_uint64_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "top.trace"
            trace.write_text(
                "0xffffffffffffffff R 1 metadata\n", encoding="utf-8")
            analysis = MODULE.analyze_trace(
                trace, 1 << 64, 8, selected_kinds=("metadata",))
            self.assertEqual(analysis.summary.observed_end, 1 << 64)
            self.assertEqual(sum(map(sum, analysis.full.read)), 1)

            with self.assertRaisesRegex(
                    MODULE.TraceHeatmapError, "exceeds capacity"):
                MODULE.analyze_trace(trace, (1 << 64) - 1, 8)

    def test_full_view_distribution_reports_uniform_read_and_zero_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "uniform.trace"
            trace.write_text(
                "0 R 4\n"
                "4 R 4\n"
                "8 R 4\n"
                "12 R 4\n",
                encoding="utf-8",
            )
            analysis = MODULE.analyze_trace(trace, 16, 4)
            rendered = MODULE.render_html(analysis, trace)

            uniform = MODULE._bin_traffic_stats([4, 4, 4, 4])
            self.assertEqual(uniform.occupied_bins, 4)
            self.assertEqual((uniform.minimum_bytes, uniform.maximum_bytes),
                             (4, 4))
            self.assertEqual(uniform.population_cv, 0.0)
            self.assertIn(
                '<th scope="row">Read traffic</th><td>4/4</td>'
                '<td>4 B</td><td>4 B</td><td>4 B</td>'
                '<td>0.000000</td>',
                rendered,
            )
            self.assertIn(
                '<th scope="row">Write traffic</th><td>0/4</td>'
                '<td>0 B</td><td>0 B</td><td>0 B</td><td>N/A</td>',
                rendered,
            )
            self.assertIn("Min bytes/bin (all bins)", rendered)
            self.assertIn("exact request overlap", rendered)

    def test_full_view_distribution_quantifies_skew_and_empty_bins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "skew.trace"
            trace.write_text("0 R 4\n0 R 4\n0 R 4\n", encoding="utf-8")
            analysis = MODULE.analyze_trace(trace, 16, 4)
            rendered = MODULE.render_html(analysis, trace)

            skewed = MODULE._bin_traffic_stats([12, 0, 0, 0])
            self.assertEqual(skewed.occupied_bins, 1)
            self.assertEqual((skewed.minimum_bytes, skewed.total_bytes,
                              skewed.maximum_bytes), (0, 12, 12))
            self.assertAlmostEqual(skewed.population_cv, 3 ** 0.5)
            self.assertIn(
                '<th scope="row">All traffic</th><td>1/4</td>'
                '<td>0 B</td><td>3 B</td><td>12 B</td>'
                '<td>1.732051</td>',
                rendered,
            )


class TraceHeatmapCliTests(unittest.TestCase):
    def _run(self, trace: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(TOOL), "--trace", str(trace),
             "--output", str(output), "--capacity-bytes", "0x1000",
             "--bins", "8", *extra],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_html_is_self_contained_and_has_fixed_full_and_zoom_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "valid.trace"
            output = root / "map.html"
            trace.write_text(
                "0x100 R 128 model_weights\n"
                "0x300 W 64 generated_context\n",
                encoding="utf-8",
            )
            completed = self._run(trace, output)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("full-capacity view", rendered)
            self.assertIn("observed-working-set zoom", rendered)
            self.assertIn("8 fixed bins", rendered)
            self.assertIn("model_weights", rendered)
            self.assertIn("generated_context", rendered)
            self.assertNotIn("http://", rendered)
            self.assertNotIn("https://", rendered)
            self.assertNotIn("<script", rendered)

    def test_kind_filter_changes_span_and_legend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "filter.trace"
            output = root / "map.html"
            trace.write_text(
                "0x10 R 16 weights\n"
                "0x800 W 16 shared_context\n",
                encoding="utf-8",
            )
            completed = self._run(trace, output, "--kind", "weights")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("[0x10, 0x20)", rendered)
            self.assertIn("model_weights", rendered)
            self.assertNotIn("<th scope=\"row\">shared_context</th>", rendered)

    def test_malformed_overflow_and_bad_kind_fail_without_clobbering(self) -> None:
        bad_lines = (
            ("0xffffffffffffffff R 2\n", "overflows uint64"),
            ("0x10 R bytes=0\n", "zero-byte"),
            ("0x10 R kind=typo\n", "unknown semantic kind"),
            ("0x10 R at=nan\n", "invalid arrival"),
            ("no operation here\n", "no R/W operation"),
        )
        for line, expected in bad_lines:
            with self.subTest(line=line), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace = root / "bad.trace"
                output = root / "map.html"
                trace.write_text(line, encoding="utf-8")
                output.write_text("keep-me", encoding="utf-8")
                completed = self._run(trace, output)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(expected, completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)
                self.assertEqual(output.read_text(encoding="utf-8"), "keep-me")

    def test_strict_parser_failures_do_not_clobber_existing_output(self) -> None:
        bad_lines = (
            ("0x10 R 64 bytes=32\n", "duplicate request-size field"),
            ("0x10 R scratch kind=metadata\n",
             "duplicate semantic-kind field"),
            ("0x10 R at=0 arrival=1\n", "duplicate arrival-time field"),
            ("0x10 R label=first region=second\n",
             "invalid or duplicate label field"),
            ("0x10 R 64oops\n", "invalid non-negative integer"),
            ("0x10 R bytse=64\n", "unsupported field"),
            ("0x10 R at=1_0\n", "invalid arrival time"),
        )
        for line, expected in bad_lines:
            with self.subTest(line=line), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace = root / "bad.trace"
                output = root / "map.html"
                trace.write_text(line, encoding="utf-8")
                output.write_text("keep-existing-output", encoding="utf-8")
                completed = self._run(trace, output)
                self.assertEqual(completed.returncode, 2)
                self.assertIn(expected, completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)
                self.assertEqual(
                    output.read_text(encoding="utf-8"),
                    "keep-existing-output")

    def test_empty_selection_missing_file_and_same_path_are_clean_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "only-read.trace"
            output = root / "map.html"
            trace.write_text("0x10 R 8 scratch\n", encoding="utf-8")
            selected = self._run(trace, output, "--kind", "metadata")
            self.assertEqual(selected.returncode, 2)
            self.assertIn("no operations matching", selected.stderr)
            self.assertFalse(output.exists())

            missing = self._run(root / "missing.trace", output)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("cannot stat trace", missing.stderr)
            self.assertNotIn("Traceback", missing.stderr)

            same = subprocess.run(
                [sys.executable, str(TOOL), "--trace", str(trace),
                 "--output", str(trace), "--capacity-bytes", "4096"],
                capture_output=True, text=True, check=False)
            self.assertEqual(same.returncode, 2)
            self.assertIn("trace and output paths must be different", same.stderr)
            self.assertEqual(trace.read_text(encoding="utf-8"), "0x10 R 8 scratch\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
