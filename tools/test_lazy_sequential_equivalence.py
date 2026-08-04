#!/usr/bin/env python3
"""Regression contract for lazy sequential equivalence and scoped EC6 knobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


FLAT = "HBM-HBF-Flat"
STATIC_DIRECT = "HBF-static-direct-read"
LAYER_STREAMING = "HBM+HBF-layer-streaming"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=120)


def load_single_scenario(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 1:
        raise AssertionError("expected exactly one scenario result")
    return scenarios[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        # The HBF half begins and ends in the middle of different 512-entry
        # mapping pages. The compact lazy image must still reproduce ordinary
        # materialized prepopulation exactly, without allocating one L2P entry
        # per page.
        request_bytes = 4096
        total_bytes = 3 * 1024 * 1024
        flat_hbm_bytes = 1024 * 1024
        request_count = total_bytes // request_bytes
        trace = root / "materialized.trace"
        trace.write_text(
            "".join(
                f"0x{index * request_bytes:x} R {request_bytes}\n"
                for index in range(request_count)
            ),
            encoding="utf-8",
        )
        materialized_summary = root / "materialized.json"
        lazy_summary = root / "lazy.json"
        common = [
            "--scenarios", FLAT,
            "--line-size", str(request_bytes),
            "--interarrival-ns", "0",
            "--flat-hbm-bytes", str(flat_hbm_bytes),
            "--hbf-page-size", str(request_bytes),
            "--hbf-oob-bytes", "224",
            "--hbf-stacks", "2",
            "--hbf-channels", "1",
            "--hbf-dies-per-channel", "1",
            "--hbf-planes-per-die", "4",
            "--hbf-blocks-per-plane", "32",
            "--hbf-pages-per-block", "64",
        ]
        materialized = run([
            str(args.scenario_compare),
            "--trace", str(trace),
            "--summary-json", str(materialized_summary),
            *common,
        ])
        if materialized.returncode != 0:
            raise RuntimeError(materialized.stdout + materialized.stderr)
        lazy = run([
            str(args.scenario_compare),
            "--synthetic-sequential-read-bytes", str(total_bytes),
            "--summary-json", str(lazy_summary),
            *common,
        ])
        if lazy.returncode != 0:
            raise RuntimeError(lazy.stdout + lazy.stderr)

        materialized_result = load_single_scenario(materialized_summary)
        lazy_result = load_single_scenario(lazy_summary)
        original_materialized_stats = dict(materialized_result["hbf_stats"])
        original_lazy_stats = dict(lazy_result["hbf_stats"])
        materialized_hbf_stats = dict(original_materialized_stats)
        lazy_hbf_stats = dict(original_lazy_stats)
        representation_counters = (
            "compact_initial_logical_data_pages",
            "compact_initial_mapping_pages",
            "compact_live_logical_data_pages",
            "compact_live_mapping_pages",
            "compact_retired_logical_data_pages",
            "compact_retired_mapping_pages",
        )
        for key in representation_counters:
            materialized_hbf_stats.pop(key)
            lazy_hbf_stats.pop(key)
        materialized_result["hbf_stats"] = materialized_hbf_stats
        lazy_result["hbf_stats"] = lazy_hbf_stats
        if lazy_result != materialized_result:
            raise AssertionError(
                "lazy sequential result differs from the equivalent materialized trace")
        expected_hbf_bytes = total_bytes - flat_hbm_bytes
        expected_hbf_pages = expected_hbf_bytes // request_bytes
        if (
            original_materialized_stats[
                "compact_initial_logical_data_pages"],
            original_materialized_stats["compact_initial_mapping_pages"],
            original_lazy_stats["compact_initial_logical_data_pages"],
            original_lazy_stats["compact_live_logical_data_pages"],
            original_lazy_stats["compact_retired_logical_data_pages"],
        ) != (
            0,
            0,
            expected_hbf_pages,
            expected_hbf_pages,
            0,
        ):
            raise AssertionError(
                "lazy/materialized initial-image representation counters "
                "lost their distinct contracts")
        if (
            original_lazy_stats["compact_initial_mapping_pages"] == 0
            or original_lazy_stats["compact_initial_mapping_pages"]
            != original_lazy_stats["compact_live_mapping_pages"]
            or original_lazy_stats["compact_retired_mapping_pages"] != 0
            or original_lazy_stats["initial_logical_data_pages"]
            != expected_hbf_pages
            or original_materialized_stats["initial_logical_data_pages"]
            != expected_hbf_pages
            or original_lazy_stats["initial_mapping_pages"]
            != original_materialized_stats["initial_mapping_pages"]
        ):
            raise AssertionError(
                "lazy/materialized initial-image population did not conserve "
                "data and mapping pages")
        if lazy_result["hybrid_path"]["hbf_logical_read_bytes"] != \
                expected_hbf_bytes:
            raise AssertionError(
                "flat logical-HBF path bytes do not match the routed suffix")

        # Static-direct uses the same block-local page bijection in both
        # forms. The lazy form reserves dense immutable block extents rather
        # than materializing one programmed-page entry per source page.
        for scenario in (STATIC_DIRECT,):
            materialized_static_summary = root / "materialized-static.json"
            lazy_static_summary = root / "lazy-static.json"
            static_common = [
                "--scenarios", scenario,
                "--line-size", str(request_bytes),
                "--interarrival-ns", "0",
                "--static-direct-hbm-bytes", str(flat_hbm_bytes),
                "--hbf-page-size", str(request_bytes),
                "--hbf-stacks", "2",
                "--hbf-channels", "1",
                "--hbf-dies-per-channel", "1",
                "--hbf-planes-per-die", "4",
                "--hbf-blocks-per-plane", "32",
                "--hbf-pages-per-block", "64",
                "--max-outstanding-requests", "32",
            ]
            materialized_static = run([
                str(args.scenario_compare),
                "--trace", str(trace),
                "--summary-json", str(materialized_static_summary),
                *static_common,
            ])
            if materialized_static.returncode != 0:
                raise RuntimeError(
                    materialized_static.stdout + materialized_static.stderr)
            lazy_static = run([
                str(args.scenario_compare),
                "--synthetic-sequential-read-bytes", str(total_bytes),
                "--summary-json", str(lazy_static_summary),
                *static_common,
            ])
            if lazy_static.returncode != 0:
                raise RuntimeError(lazy_static.stdout + lazy_static.stderr)
            lazy_static_result = load_single_scenario(lazy_static_summary)
            materialized_static_result = load_single_scenario(
                materialized_static_summary)
            lazy_hbf_stats = dict(lazy_static_result["hbf_stats"])
            materialized_hbf_stats = dict(
                materialized_static_result["hbf_stats"])
            static_accounting_keys = {
                "free_pages",
                "valid_pages",
                "static_reserved_pages",
                "static_unmaterialized_pages",
            }
            for key in static_accounting_keys:
                lazy_hbf_stats.pop(key, None)
                materialized_hbf_stats.pop(key, None)
            lazy_static_result["hbf_stats"] = lazy_hbf_stats
            materialized_static_result["hbf_stats"] = materialized_hbf_stats
            if lazy_static_result != materialized_static_result:
                differing_keys = [
                    key for key in sorted(
                        lazy_static_result.keys() |
                        materialized_static_result.keys())
                    if lazy_static_result.get(key) !=
                    materialized_static_result.get(key)
                ]
                raise AssertionError(
                    "lazy static-direct result differs from materialized trace: "
                    + ", ".join(differing_keys))
            if lazy_static_result["hybrid_path"][
                    "hbf_logical_read_bytes"] != 0:
                raise AssertionError(
                    "static-direct HBF bytes were mislabeled as logical")

        # EC6-only geometry is inactive when EC6 is not selected. Neither zero
        # nor an otherwise overflowing layer-buffer value may block all-HBM.
        direct_trace = root / "direct-only.trace"
        direct_trace.write_text("0x0 R 4096\n", encoding="utf-8")
        for label, layer_buffer in (
                ("zero", "0"),
                ("oversized", str(2**64 - 1))):
            summary_path = root / f"direct-{label}.json"
            direct = run([
                str(args.scenario_compare),
                "--trace", str(direct_trace),
                "--scenarios", "all-HBM",
                "--line-size", "4096",
                "--layer-buffer-bytes", layer_buffer,
                "--summary-json", str(summary_path),
            ])
            if direct.returncode != 0:
                raise AssertionError(
                    f"inactive {label} EC6 buffer rejected direct-only run:\n"
                    + direct.stdout + direct.stderr)
            config = json.loads(summary_path.read_text(encoding="utf-8"))["config"]
            for obsolete in (
                    "layer_streaming_two_buffer_limit_bytes",
                    "layer_streaming_capacity_bytes",
                    "hbm_layer_streaming_region_base_addr",
                    "hbm_foreground_region_bytes"):
                if obsolete in config:
                    raise AssertionError(
                        f"inactive EC6 retained static-reservation field {obsolete}")

        for layer_buffer in ("0", str(2**64 - 1)):
            rejected = run([
                str(args.scenario_compare),
                "--trace", str(direct_trace),
                "--scenarios", LAYER_STREAMING,
                "--line-size", "4096",
                "--layer-buffer-bytes", layer_buffer,
            ])
            if rejected.returncode == 0:
                raise AssertionError("active EC6 accepted invalid layer-buffer geometry")

    print("lazy sequential equivalence and EC6 option scoping: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
