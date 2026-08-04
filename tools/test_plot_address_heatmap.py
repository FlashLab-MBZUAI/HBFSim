#!/usr/bin/env python3
"""Standalone regression tests for plot_address_heatmap.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_address_heatmap as heatmap


TOOL = Path(__file__).with_name("plot_address_heatmap.py")


def _empty_source_metrics() -> dict[str, list[int]]:
    return {
        f"source_{metric}": [0] * len(heatmap.TRAFFIC_SOURCES)
        for metric in heatmap.METRICS
    }


def _make_bin(begin: int, end: int) -> dict[str, object]:
    result: dict[str, object] = {
        "begin": begin,
        "end": end,
        **{metric: 0 for metric in heatmap.METRICS},
        **_empty_source_metrics(),
    }
    return result


def _add_traffic(
        item: dict[str, object], source: str, direction: str,
        byte_count: int, access_count: int) -> None:
    source_index = heatmap.TRAFFIC_SOURCES.index(source)
    for suffix, value in (("bytes", byte_count), ("accesses", access_count)):
        metric = f"{direction}_{suffix}"
        source_field = f"source_{metric}"
        item[metric] = int(item[metric]) + value
        source_values = item[source_field]
        assert isinstance(source_values, list)
        source_values[source_index] += value


def _make_domain(name: str, domain_index: int) -> dict[str, object]:
    bins = [_make_bin(0, 4096), _make_bin(4096, 8192)]
    _add_traffic(
        bins[0], "workload", "read", 4096 * (domain_index + 1),
        domain_index + 1,
    )
    _add_traffic(
        bins[1], "direct", "write", 1024 * (domain_index + 1),
        2 * (domain_index + 1),
    )
    if name == "hbf_physical":
        _add_traffic(
            bins[1], "garbage_collection", "erase", 8192, 1)

    totals = {
        metric: sum(int(item[metric]) for item in bins)
        for metric in heatmap.METRICS
    }
    source_totals = []
    for source_index, source in enumerate(heatmap.TRAFFIC_SOURCES):
        entry: dict[str, object] = {"source": source}
        for metric in heatmap.METRICS:
            entry[metric] = sum(
                int(item[f"source_{metric}"][source_index])  # type: ignore[index]
                for item in bins
            )
        source_totals.append(entry)

    regions = [{
        "name": "weights <hot>" if domain_index == 0 else f"region-{domain_index}",
        "kind": heatmap.REGION_KINDS[domain_index],
        "begin": 0,
        "end": 4096,
    }]
    return {
        "domain": name,
        "size_bytes": 8192,
        "totals": totals,
        "source_totals": source_totals,
        "regions": regions,
        "bins": bins,
    }


def valid_artifact() -> dict[str, object]:
    return {
        "schema": heatmap.SCHEMA,
        "bin_count": 2,
        "traffic_sources": list(heatmap.TRAFFIC_SOURCES),
        "domains": [
            _make_domain(name, index)
            for index, name in enumerate(heatmap.DOMAIN_ORDER)
        ],
    }


def summary_artifact(*names: str) -> dict[str, object]:
    scenarios = []
    for name in names:
        artifact = valid_artifact()
        domains = artifact["domains"]
        assert isinstance(domains, list)
        regions = domains[0]["regions"]
        assert isinstance(regions, list)
        regions[0]["name"] = f"selected-{name}"
        scenarios.append({
            "name": name,
            "address_heatmap": artifact,
            # Selection must not over-constrain unrelated scenario fields.
            "time_breakdown": {"future_field": 1},
        })
    return {
        "schema": {
            "name": heatmap.SUMMARY_SCHEMA,
            "version": heatmap.SUMMARY_SCHEMA_VERSION,
        },
        "metadata": {"producer": "test"},
        "scenarios": scenarios,
    }


class ValidationTests(unittest.TestCase):
    def test_valid_contract_and_accounting(self) -> None:
        parsed = heatmap.validate_heatmap(valid_artifact())
        self.assertEqual(parsed.bin_count, 2)
        self.assertEqual(
            tuple(domain.name for domain in parsed.domains),
            heatmap.DOMAIN_ORDER,
        )
        by_name = {domain.name: domain for domain in parsed.domains}
        self.assertEqual(
            by_name["hbf_physical"].totals["erase_bytes"], 8192)

    def test_rejects_domain_order_and_unknown_region_kind(self) -> None:
        wrong_order = valid_artifact()
        domains = wrong_order["domains"]
        assert isinstance(domains, list)
        domains[0], domains[1] = domains[1], domains[0]
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"root\.domains\[0\]\.domain: expected 'workload_logical'"):
            heatmap.validate_heatmap(wrong_order)

        wrong_kind = valid_artifact()
        domains = wrong_kind["domains"]
        assert isinstance(domains, list)
        regions = domains[0]["regions"]
        assert isinstance(regions, list)
        regions[0]["kind"] = "cache-ish"
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError, r"unknown kind 'cache-ish'"):
            heatmap.validate_heatmap(wrong_kind)

    def test_rejects_nonconserving_bin_and_domain_totals(self) -> None:
        broken_bin = valid_artifact()
        domains = broken_bin["domains"]
        assert isinstance(domains, list)
        bins = domains[0]["bins"]
        assert isinstance(bins, list)
        bins[0]["source_read_bytes"][0] += 1
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"source_read_bytes: sum .* does not match read_bytes"):
            heatmap.validate_heatmap(broken_bin)

        broken_total = valid_artifact()
        domains = broken_total["domains"]
        assert isinstance(domains, list)
        domains[0]["totals"]["write_accesses"] += 1
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"totals\.write_accesses: value .* does not match "
                r"source_totals sum"):
            heatmap.validate_heatmap(broken_total)

    def test_cross_bin_access_fanout_is_valid(self) -> None:
        artifact = valid_artifact()
        domains = artifact["domains"]
        assert isinstance(domains, list)
        domain = domains[0]
        bins = domain["bins"]
        assert isinstance(bins, list)

        # Model one 8 KiB workload read: domain/source scope counts the record
        # once, while both touched bins count one spatial access each.
        bins[1]["read_bytes"] += 4096
        bins[1]["read_accesses"] += 1
        bins[1]["source_read_bytes"][0] += 4096
        bins[1]["source_read_accesses"][0] += 1
        domain["totals"]["read_bytes"] += 4096
        domain["source_totals"][0]["read_bytes"] += 4096

        parsed = heatmap.validate_heatmap(artifact)
        self.assertEqual(parsed.domains[0].totals["read_accesses"], 1)
        self.assertEqual(
            sum(item.metrics["read_accesses"] for item in parsed.domains[0].bins),
            2,
        )

    def test_rejects_gaps_and_noncanonical_sources(self) -> None:
        gap = valid_artifact()
        domains = gap["domains"]
        assert isinstance(domains, list)
        bins = domains[0]["bins"]
        assert isinstance(bins, list)
        bins[1]["begin"] = 4097
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"bins\[1\]\.begin: expected canonical begin 4096"):
            heatmap.validate_heatmap(gap)

        source_order = valid_artifact()
        sources = source_order["traffic_sources"]
        assert isinstance(sources, list)
        sources[0], sources[1] = sources[1], sources[0]
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError, r"must exactly match canonical order"):
            heatmap.validate_heatmap(source_order)

    def test_rejects_bin_count_above_core_limit_before_domain_work(self) -> None:
        artifact = valid_artifact()
        artifact["bin_count"] = heatmap.MAX_BIN_COUNT + 1
        # Deliberately leave two-bin domains in place. The boundedness check
        # must fire before any domain/bin traversal or allocation assumptions.
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"root\.bin_count: must not exceed contract maximum 8192"):
            heatmap.validate_heatmap(artifact)

    def test_rejects_noncanonical_bins_and_values_above_contract_bounds(self) -> None:
        noncanonical = valid_artifact()
        domains = noncanonical["domains"]
        assert isinstance(domains, list)
        bins = domains[0]["bins"]
        assert isinstance(bins, list)
        bins[0]["end"] = 1
        bins[1]["begin"] = 1
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"bins\[0\]\.end: expected canonical end 4096, got 1"):
            heatmap.validate_heatmap(noncanonical)

        oversized_extent = valid_artifact()
        domains = oversized_extent["domains"]
        assert isinstance(domains, list)
        domains[0]["size_bytes"] = heatmap.ADDRESS_SPACE_SIZE + 1
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"size_bytes: must not exceed uint64 address-space end"):
            heatmap.validate_heatmap(oversized_extent)

        oversized_counter = valid_artifact()
        domains = oversized_counter["domains"]
        assert isinstance(domains, list)
        domains[0]["totals"]["read_bytes"] = heatmap.UINT64_MAX + 1
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"totals\.read_bytes: must not exceed uint64 maximum"):
            heatmap.validate_heatmap(oversized_counter)

    def test_full_uint64_address_space_extent_is_valid(self) -> None:
        artifact = valid_artifact()
        domains = artifact["domains"]
        assert isinstance(domains, list)
        midpoint = heatmap.ADDRESS_SPACE_SIZE // 2
        for domain in domains:
            domain["size_bytes"] = heatmap.ADDRESS_SPACE_SIZE
            bins = domain["bins"]
            assert isinstance(bins, list)
            bins[0]["begin"], bins[0]["end"] = 0, midpoint
            bins[1]["begin"], bins[1]["end"] = (
                midpoint, heatmap.ADDRESS_SPACE_SIZE)

        parsed = heatmap.validate_heatmap(artifact)
        self.assertEqual(
            parsed.domains[0].bins[-1].end, heatmap.ADDRESS_SPACE_SIZE)
        rendered = heatmap.render_html(parsed)
        self.assertIn("0x10000000000000000", rendered)


class SummaryInputTests(unittest.TestCase):
    def test_single_scenario_is_selected_implicitly_or_explicitly(self) -> None:
        summary = summary_artifact("only")
        implicit = heatmap.select_heatmap(summary)
        explicit = heatmap.select_heatmap(summary, "only")
        self.assertEqual(
            implicit.domains[0].regions[0].name, "selected-only")
        self.assertEqual(implicit, explicit)

    def test_multiple_scenarios_require_exact_selection(self) -> None:
        summary = summary_artifact("alpha", "beta")
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"--scenario: is required .*'alpha', 'beta'"):
            heatmap.select_heatmap(summary)

        selected = heatmap.select_heatmap(summary, "beta")
        self.assertEqual(
            selected.domains[0].regions[0].name, "selected-beta")
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"unknown scenario 'Beta'.*'alpha', 'beta'"):
            heatmap.select_heatmap(summary, "Beta")

    def test_duplicate_names_and_missing_heatmap_are_rejected(self) -> None:
        duplicate = summary_artifact("same", "same")
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"root\.scenarios\[1\]\.name: duplicate scenario name 'same'"):
            heatmap.select_heatmap(duplicate, "same")

        missing = summary_artifact("only")
        scenarios = missing["scenarios"]
        assert isinstance(scenarios, list)
        del scenarios[0]["address_heatmap"]
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"root\.scenarios\[0\]: scenario 'only' is missing "
                r"'address_heatmap'"):
            heatmap.select_heatmap(missing)

    def test_summary_schema_and_nested_heatmap_remain_strict(self) -> None:
        wrong_version = summary_artifact("only")
        schema = wrong_version["schema"]
        assert isinstance(schema, dict)
        schema["version"] = 6
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"root\.schema\.version: expected 16, got 6"):
            heatmap.select_heatmap(wrong_version)

        malformed = summary_artifact("only")
        scenarios = malformed["scenarios"]
        assert isinstance(scenarios, list)
        scenarios[0]["address_heatmap"]["bin_count"] = 0
        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"root\.scenarios\[0\]\.address_heatmap\.bin_count"):
            heatmap.select_heatmap(malformed)

        with self.assertRaisesRegex(
                heatmap.HeatmapInputError,
                r"--scenario: is only valid for an "
                r"hbfsim\.scenario_compare\.summary input"):
            heatmap.select_heatmap(valid_artifact(), "only")


class RenderingTests(unittest.TestCase):
    def test_offline_accessible_domain_panel_rendering(self) -> None:
        rendered = heatmap.render_html(
            heatmap.validate_heatmap(valid_artifact()))

        self.assertTrue(rendered.startswith("<!doctype html>"))
        self.assertEqual(
            rendered.count('class="domain-panel"'),
            len(heatmap.DOMAIN_ORDER),
        )
        for domain in heatmap.DOMAIN_ORDER:
            self.assertIn(f'data-domain="{domain}"', rendered)
        self.assertIn('class="traffic-mark traffic-read"', rendered)
        self.assertIn('id="write-hatch"', rendered)
        self.assertIn('id="erase-crosshatch"', rendered)
        self.assertIn('data-direction="erase"', rendered)
        self.assertIn('class="region-overlay"', rendered)
        self.assertIn('pointer-events="stroke"', rendered)
        self.assertIn("@media (prefers-color-scheme: dark)", rendered)
        self.assertIn("fill=\"var(--read-color)\"", rendered)
        self.assertIn("fill=\"var(--panel-even)\"", rendered)
        self.assertIn("Exact domain totals", rendered)
        self.assertIn("Traffic-source accounting", rendered)
        self.assertIn("weights &lt;hot&gt;", rendered)
        self.assertNotIn("weights <hot>", rendered)

        lower = rendered.lower()
        self.assertNotIn("<script", lower)
        self.assertNotIn("<link", lower)
        self.assertNotIn("http://", lower)
        self.assertNotIn("https://", lower)

    def test_byte_formatter_keeps_integer_trailing_zeroes(self) -> None:
        self.assertEqual(heatmap._format_bytes(100 * 1024), "100 KiB")
        self.assertEqual(heatmap._format_bytes(10 * 1024), "10 KiB")


class CommandLineTests(unittest.TestCase):
    def _run(
            self, input_path: Path, output_path: Path,
            scenario: str | None = None) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable, str(TOOL), "--input", str(input_path),
            "--output", str(output_path),
        ]
        if scenario is not None:
            command.extend(("--scenario", scenario))
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "heatmap.json"
            first = root / "first.html"
            second = root / "second.html"
            source.write_text(json.dumps(valid_artifact()), encoding="utf-8")

            first_run = self._run(source, first)
            second_run = self._run(source, second)
            self.assertEqual(first_run.returncode, 0, first_run.stderr)
            self.assertEqual(second_run.returncode, 0, second_run.stderr)
            self.assertEqual(first_run.stdout, "")
            self.assertEqual(first_run.stderr, "")
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_cli_selects_summary_scenario_without_silent_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            multiple = root / "multiple.json"
            selected_output = root / "selected.html"
            ambiguous_output = root / "ambiguous.html"
            multiple.write_text(
                json.dumps(summary_artifact("alpha", "beta")),
                encoding="utf-8",
            )
            ambiguous_output.write_text("keep-me", encoding="utf-8")

            ambiguous = self._run(multiple, ambiguous_output)
            self.assertEqual(ambiguous.returncode, 2)
            self.assertIn(
                "--scenario: is required for a summary with multiple scenarios",
                ambiguous.stderr,
            )
            self.assertEqual(
                ambiguous_output.read_text(encoding="utf-8"), "keep-me")

            selected = self._run(multiple, selected_output, "beta")
            self.assertEqual(selected.returncode, 0, selected.stderr)
            self.assertIn(
                "selected-beta",
                selected_output.read_text(encoding="utf-8"),
            )

            single = root / "single.json"
            single_output = root / "single.html"
            single.write_text(
                json.dumps(summary_artifact("only")), encoding="utf-8")
            implicit = self._run(single, single_output)
            self.assertEqual(implicit.returncode, 0, implicit.stderr)
            self.assertIn(
                "selected-only", single_output.read_text(encoding="utf-8"))

    def test_cli_rejects_scenario_for_direct_heatmap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "heatmap.json"
            output = root / "output.html"
            source.write_text(json.dumps(valid_artifact()), encoding="utf-8")
            completed = self._run(source, output, "not-applicable")
            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "--scenario: is only valid for an "
                "hbfsim.scenario_compare.summary input",
                completed.stderr,
            )
            self.assertFalse(output.exists())

    def test_bad_input_reports_path_and_does_not_clobber_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "broken.json"
            output = root / "existing.html"
            artifact = valid_artifact()
            domains = artifact["domains"]
            assert isinstance(domains, list)
            bins = domains[2]["bins"]
            assert isinstance(bins, list)
            bins[0]["source_read_accesses"][0] += 7
            source.write_text(json.dumps(artifact), encoding="utf-8")
            output.write_text("keep-me", encoding="utf-8")

            completed = self._run(source, output)
            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "root.domains[2].bins[0].source_read_accesses", completed.stderr)
            self.assertIn("does not match read_accesses", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep-me")

    def test_malformed_json_has_source_location_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "broken.json"
            output = root / "output.html"
            source.write_text('{"schema":\n', encoding="utf-8")

            completed = self._run(source, output)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("invalid JSON at line 2, column 1", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertFalse(output.exists())

    def test_duplicate_key_and_same_path_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "duplicate.json"
            output = root / "output.html"
            source.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
            duplicate = self._run(source, output)
            self.assertEqual(duplicate.returncode, 2)
            self.assertIn("duplicate key 'schema'", duplicate.stderr)

            same_path = subprocess.run(
                [sys.executable, str(TOOL), "--input", str(source),
                 "-o", str(source)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(same_path.returncode, 2)
            self.assertIn("input and output paths must be different", same_path.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
