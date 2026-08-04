#!/usr/bin/env python3
"""Focused contracts for the controlled synthetic experiment runner."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_trace_locality import analyze_trace  # noqa: E402
from generate_synthetic_trace import GeneratorConfig, generate, publish  # noqa: E402
from run_synthetic_experiments import (  # noqa: E402
    BALANCED_SPAN,
    BASELINE_HBF_CAPACITY,
    BASELINE_HBM_CAPACITY,
    OUTPUT_CONFIG_EXPECTATIONS,
    OUTPUT_TARGETS,
    OUTPUT_TOPOLOGY_LEAVES,
    PAGE,
    SCHEMA_VERSION,
    _rows,
    build_plan,
    invalidate_suite_outputs,
    validate_capacity_coverage,
    validate_generator_manifest,
    validate_locality_report,
    validate_matched_output_traces,
    validate_native_physical_coverage,
    validate_output_invariants,
    validate_summary,
)


def _materialize_case(
    case, root: Path, seed: int, placement_seed: int = 1,
) -> tuple[Path, Path]:
    trace = root / f"{case.key}.trace"
    manifest_path = root / f"{case.key}.manifest.json"
    config = GeneratorConfig(
        pattern=case.pattern,
        pages=case.pages,
        passes=case.passes,
        request_bytes=PAGE,
        base_address=0,
        address_span_bytes=case.address_span,
        placement=case.placement,
        placement_seed=placement_seed,
        placement_modulus=case.placement_modulus,
        read_percent=case.read_percent,
        seed=seed,
        kind=case.kind,
        at_ns=0,
        stride_pages=(case.stride_pages
                      if case.pattern == "modular-stride" else None),
        hot_page_percent=case.hot_page_percent,
        hot_access_percent=case.hot_access_percent,
        phase_comments=True,
    )
    lines, manifest = generate(config)
    publish(trace, manifest_path, lines, manifest)
    return trace, manifest_path


def _coverage_summary(case, bins: int = 256) -> dict:
    scenarios = []
    for name in case.targets[0].scenarios:
        if name == "all-HBM":
            domain_name = "hbm_physical"
            capacity = BASELINE_HBM_CAPACITY
        elif name in {"all-HBF", "HBF-static-direct-read"}:
            domain_name = "hbf_physical"
            capacity = BASELINE_HBF_CAPACITY
        else:
            raise AssertionError(f"unsupported test scenario {name}")
        scenarios.append({
            "name": name,
            "address_heatmap": {
                "bin_count": bins,
                "domains": [{
                    "domain": domain_name,
                    "size_bytes": capacity,
                    "bins": [{
                        "read_bytes": PAGE,
                        "write_bytes": 0,
                        "erase_bytes": 0,
                    } for _ in range(bins)],
                }],
            },
        })
    return {"scenarios": scenarios}


def _output_summary(case, target) -> dict:
    expectation = OUTPUT_CONFIG_EXPECTATIONS[target.config]
    operations = case.pages * case.passes
    topology_rounds = case.pages // OUTPUT_TOPOLOGY_LEAVES
    logical_bytes = operations * PAGE
    zero_hybrid = {
        "hbf_static_read_bytes": logical_bytes,
        "hbf_logical_read_bytes": 0,
        "hbf_backing_write_bytes": 0,
        "background_hbf_writes": 0,
        "hbm_foreground_bytes": 0,
        "hbm_streaming_write_bytes": 0,
        "base_die_link_read_bytes": 0,
        "base_die_link_write_bytes": 0,
    }
    hbf_stats = {
        "logical_read_bytes": 0,
        "physical_read_bytes": logical_bytes,
        "logical_write_bytes": 0,
        "physical_write_bytes": 0,
        "page_reads": operations,
        "read_buffer_hits": 0,
        "read_buffer_misses": operations,
        "mapping_entries": 0,
        "mapping_lookup_ops": 0,
        "mapping_update_ops": 0,
        "mapping_page_programs": 0,
        "data_programs": 0,
        "page_programs": 0,
        "block_erases": 0,
        "gc_runs": 0,
        "gc_relocations": 0,
        "gc_user_blocked_runs": 0,
        "channels": 32,
        "active_channels": 32,
        "dies": 128,
        "active_dies": 128,
        "planes": 512,
        "active_planes": 512,
        "subarrays": OUTPUT_TOPOLOGY_LEAVES,
        "active_subarrays": OUTPUT_TOPOLOGY_LEAVES,
        "max_subarray_reads": topology_rounds,
        "avg_active_subarray_reads": topology_rounds,
        "subarray_read_skew": 1,
    }
    return {
        "config": {
            "static_direct_hbm_bytes": 0,
            "hbf": {
                "capacity_bytes": BASELINE_HBF_CAPACITY,
                "stacks": 8,
                "channels": 4,
                "dies_per_channel": 4,
                "planes_per_die": 4,
                "subarrays_per_plane": 32,
                "media_lanes_per_plane": expectation["local_ports_per_plane"],
                "page_buffer_banks_per_plane": expectation[
                    "local_ports_per_plane"],
                "read_ns": 1000,
                "flash_tsu_issue_ns": expectation["flash_tsu_issue_ns"],
                "media_lane_bw_GBps": 2048,
                "page_buffer_bw_GBps": 2048,
                "logic_scheduler_issue_ns": expectation[
                    "logic_scheduler_issue_ns"],
                "command_address_bytes": 64,
                "batch_activation": True,
                "ecc_decode_raw_bw_GBps_per_die": expectation[
                    "ecc_decode_raw_bw_GBps_per_die"],
                "ecc_encode_raw_bw_GBps_per_die": expectation[
                    "ecc_encode_raw_bw_GBps_per_die"],
                "channel_bw_GBps": expectation["channel_bw_GBps"],
                "hbio_bw_GBps": expectation["hbio_bw_GBps"],
                "tsv_bw_GBps": expectation["tsv_bw_GBps"],
                "logic_sram_bw_GBps": expectation["logic_sram_bw_GBps"],
                "ecc_issue_topology": "shared-per-die",
                "bandwidth_semantics": {
                    "channel": "raw-codeword-per-channel",
                    "tsv": "shared-command-and-raw-codeword-per-stack",
                    "hbio": "decoded-payload-per-stack",
                    "media_lane": "raw-codeword-per-lane",
                    "page_buffer": "raw-codeword-per-bank",
                    "logic_sram": "decoded-payload-per-stack",
                },
            },
        },
        "scenarios": [{
            "name": "HBF-static-direct-read",
            "hbm_accesses": 0,
            "hbm_user_accesses": 0,
            "hbm_background_accesses": 0,
            "hbf_accesses": operations,
            "hbf_user_accesses": operations,
            "hbf_background_accesses": 0,
            "hybrid_path": zero_hybrid,
            "hbm_stats": {"read_bytes": 0, "write_bytes": 0},
            "hbf_stats": hbf_stats,
        }],
    }


class SyntheticPlanTests(unittest.TestCase):
    def test_smoke_plan_is_small_and_orthogonal(self) -> None:
        plan = build_plan("smoke", ("raw", "balanced"))
        self.assertEqual(len(plan), 7)
        self.assertEqual(len({case.key for case in plan}), len(plan))
        self.assertEqual(
            {case.pattern for case in plan if case.group == "raw"},
            {"sequential", "random-permutation", "modular-stride", "hotspot"},
        )
        balanced = [case for case in plan if case.group == "balanced"]
        self.assertEqual({case.read_percent for case in balanced}, {70, 100})
        self.assertTrue(all(case.address_span == BALANCED_SPAN for case in balanced))

    def test_packed_and_spread_random_are_matched_except_span_and_key(self) -> None:
        raw = {case.key: case for case in build_plan("core", ("raw",))}
        packed = raw["raw-random-permutation"]
        spread = raw["raw-random-permutation-spread"]
        self.assertEqual(packed.address_span, packed.pages * PAGE)
        self.assertEqual(spread.address_span, BALANCED_SPAN)
        self.assertEqual(packed.seed_offset, spread.seed_offset)
        for field in (
            "group", "pattern", "pages", "passes", "read_percent", "kind",
            "targets", "stride_pages", "hot_page_percent", "hot_access_percent",
            "window", "placement", "placement_seed_offset",
            "placement_modulus", "heatmap_capacity",
            "static_direct_hbm_bytes",
        ):
            self.assertEqual(getattr(packed, field), getattr(spread, field), field)

    def test_balanced_read_mix_is_a_matched_pair(self) -> None:
        balanced = sorted(
            build_plan("core", ("balanced",)), key=lambda case: case.read_percent)
        self.assertEqual([case.read_percent for case in balanced], [70, 100])
        self.assertEqual(balanced[0].seed_offset, balanced[1].seed_offset)
        for field in (
            "group", "pattern", "pages", "passes", "address_span", "kind",
            "targets", "stride_pages", "hot_page_percent", "hot_access_percent",
            "window", "placement", "placement_seed_offset",
            "placement_modulus", "heatmap_capacity",
            "static_direct_hbm_bytes",
        ):
            self.assertEqual(
                getattr(balanced[0], field), getattr(balanced[1], field), field)

    def test_group_selection_and_rejection(self) -> None:
        raw = build_plan("smoke", ("raw",))
        self.assertTrue(raw)
        self.assertTrue(all(case.group == "raw" for case in raw))
        with self.assertRaises(ValueError):
            build_plan("smoke", ())
        with self.assertRaises(ValueError):
            build_plan("smoke", ("not-a-group",))
        with self.assertRaises(ValueError):
            build_plan("huge", ("raw",))

    def test_native_capacity_coverage_is_sparse_uniform_and_matched(self) -> None:
        cases = {case.key: case
                 for case in build_plan("smoke", ("coverage",))}
        self.assertEqual(set(cases), {
            "coverage-hbm-native-384g", "coverage-hbf-native-4t"})
        hbm = cases["coverage-hbm-native-384g"]
        hbf = cases["coverage-hbf-native-4t"]
        self.assertEqual((hbm.address_span, hbm.heatmap_capacity),
                         (BASELINE_HBM_CAPACITY, BASELINE_HBM_CAPACITY))
        self.assertEqual((hbf.address_span, hbf.heatmap_capacity),
                         (BASELINE_HBF_CAPACITY, BASELINE_HBF_CAPACITY))
        self.assertEqual(hbm.pages, 4096)
        self.assertEqual(hbm.pages % 256, 0)
        self.assertEqual(hbm.placement, "stratified-random")
        self.assertEqual(hbm.seed_offset, hbf.seed_offset)
        self.assertEqual(
            hbm.placement_seed_offset, hbf.placement_seed_offset)
        self.assertEqual(hbm.targets[0].scenarios, ("all-HBM", "all-HBF"))
        self.assertEqual(hbf.targets[0].scenarios, ("all-HBF",))
        for field in (
            "group", "pattern", "pages", "passes", "read_percent", "kind",
            "stride_pages", "hot_page_percent", "hot_access_percent",
            "window", "placement",
        ):
            self.assertEqual(getattr(hbm, field), getattr(hbf, field), field)

    def test_output_plan_is_matched_direct_hbf_at_two_windows(self) -> None:
        self.assertEqual(SCHEMA_VERSION, 5)
        expected_configs = tuple(target.config for target in OUTPUT_TARGETS)
        for profile, pages, high_window in (
            ("smoke", 16384, 16384),
            ("core", 65536, 32768),
        ):
            with self.subTest(profile=profile):
                cases = build_plan(profile, ("output",))
                self.assertEqual(len(cases), 2)
                self.assertEqual([case.window for case in cases],
                                 [512, high_window])
                self.assertTrue(all(case.pages == pages for case in cases))
                self.assertTrue(all(case.pages % 256 == 0 for case in cases))
                self.assertTrue(all(case.passes == 1 for case in cases))
                self.assertTrue(all(case.read_percent == 100 for case in cases))
                self.assertTrue(all(
                    case.address_span == BASELINE_HBF_CAPACITY and
                    case.heatmap_capacity == BASELINE_HBF_CAPACITY
                    for case in cases))
                self.assertTrue(all(
                    case.static_direct_hbm_bytes == 0 for case in cases))
                self.assertTrue(all(
                    case.placement == "stratified-random" for case in cases))
                self.assertTrue(all(
                    case.placement_modulus == OUTPUT_TOPOLOGY_LEAVES
                    for case in cases))
                self.assertTrue(all(
                    tuple(target.config for target in case.targets) ==
                    expected_configs for case in cases))
                self.assertTrue(all(
                    target.scenarios == ("HBF-static-direct-read",)
                    for case in cases for target in case.targets))
                left, right = cases
                for field in (
                    "group", "pattern", "pages", "passes", "address_span",
                    "read_percent", "kind", "targets", "stride_pages",
                    "hot_page_percent", "hot_access_percent",
                    "seed_offset", "placement",
                    "placement_seed_offset", "placement_modulus",
                    "heatmap_capacity",
                    "static_direct_hbm_bytes",
                ):
                    self.assertEqual(
                        getattr(left, field), getattr(right, field), field)


class ArtifactValidatorTests(unittest.TestCase):
    def test_generator_manifest_binds_case_counts_and_trace_digest(self) -> None:
        case = build_plan("smoke", ("raw",))[0]
        seed = 41 + case.seed_offset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, manifest_path = _materialize_case(case, root, seed)
            validated = validate_generator_manifest(
                case, trace, manifest_path, seed, 1)
            self.assertEqual(validated["ops"], case.pages * case.passes)

            original = json.loads(manifest_path.read_text(encoding="utf-8"))
            corruptions = (
                ("schema", {"name": "hbfsim.synthetic_trace.manifest",
                            "version": 4}),
                ("ops", original["ops"] + 1),
                ("read_ops", True),
                ("working_set_pages", original["working_set_pages"] + 1),
                ("placement", "stratified-random"),
                ("placement_seed", original["placement_seed"] + 1),
                ("placement_modulus", 17),
            )
            for key, value in corruptions:
                with self.subTest(key=key):
                    changed = copy.deepcopy(original)
                    changed[key] = value
                    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        validate_generator_manifest(
                            case, trace, manifest_path, seed, 1)
            manifest_path.write_text(json.dumps(original), encoding="utf-8")
            trace.write_bytes(trace.read_bytes() + b"# changed\n")
            with self.assertRaisesRegex(RuntimeError, "trace_file_bytes|trace_digest"):
                validate_generator_manifest(
                    case, trace, manifest_path, seed, 1)

    def test_capacity_coverage_validates_strata_and_every_bin(self) -> None:
        for case in build_plan("smoke", ("coverage",)):
            with self.subTest(case=case.key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace, _ = _materialize_case(
                    case, root, 10, placement_seed=11)
                validate_capacity_coverage(case, trace, 256)

                original_lines = trace.read_text(encoding="utf-8").splitlines()
                records = [index for index, line in enumerate(original_lines)
                           if line and not line.startswith("#")]
                first_address = original_lines[records[0]].split()[0]
                second_address = original_lines[records[1]].split()[0]

                lines = original_lines.copy()
                other_phase_index = next(
                    index for index in records[case.pages:]
                    if lines[index].split()[0] != first_address)
                first_fields = lines[records[0]].split()
                other_fields = lines[other_phase_index].split()
                first_fields[0], other_fields[0] = (
                    other_fields[0], first_fields[0])
                lines[records[0]] = " ".join(first_fields)
                lines[other_phase_index] = " ".join(other_fields)
                trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "phase 0"):
                    validate_capacity_coverage(case, trace, 256)

                lines = original_lines.copy()
                fields = lines[records[0]].split()
                fields[0] = second_address
                lines[records[0]] = " ".join(fields)
                trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "per-page access"):
                    validate_capacity_coverage(case, trace, 256)

                lines = original_lines.copy()
                for index in records:
                    fields = lines[index].split()
                    if fields[0] == first_address:
                        fields[0] = second_address
                        lines[index] = " ".join(fields)
                trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "unique pages"):
                    validate_capacity_coverage(case, trace, 256)

    def test_output_trace_is_read_only_uniform_and_byte_matched(self) -> None:
        cases = build_plan("smoke", ("output",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = {}
            for case in cases:
                trace, _ = _materialize_case(
                    case, root, seed=12, placement_seed=13)
                traces[case.key] = trace
                validate_capacity_coverage(case, trace, 256)
            validate_matched_output_traces(cases, traces)

            trace = traces[cases[0].key]
            original_lines = trace.read_text(
                encoding="utf-8").splitlines()
            record_indexes = [
                index for index, line in enumerate(original_lines)
                if line and not line.startswith("#")]
            span_slots = cases[0].address_span // PAGE
            stratum_slots = span_slots // cases[0].pages
            by_stratum = {}
            for index in record_indexes:
                slot = int(original_lines[index].split()[0], 0) // PAGE
                by_stratum[slot // stratum_slots] = index
            first_slot = (
                int(original_lines[by_stratum[0]].split()[0], 0) // PAGE)
            duplicate_residue = first_slot % OUTPUT_TOPOLOGY_LEAVES
            second_begin = stratum_slots
            replacement_slot = second_begin + (
                (duplicate_residue - second_begin) % OUTPUT_TOPOLOGY_LEAVES)
            broken_topology = original_lines.copy()
            fields = broken_topology[by_stratum[1]].split()
            fields[0] = hex(replacement_slot * PAGE)
            broken_topology[by_stratum[1]] = " ".join(fields)
            trace.write_text(
                "\n".join(broken_topology) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "placement round 0"):
                validate_capacity_coverage(cases[0], trace, 256)

            lines = original_lines.copy()
            record = next(index for index, line in enumerate(lines)
                          if line and not line.startswith("#"))
            fields = lines[record].split()
            fields[1] = "W"
            lines[record] = " ".join(fields)
            trace.write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not a read"):
                validate_capacity_coverage(cases[0], trace, 256)
            with self.assertRaisesRegex(RuntimeError, "byte-identical"):
                validate_matched_output_traces(cases, traces)

    def test_locality_report_digest_and_counts_are_cross_checked(self) -> None:
        case = build_plan("smoke", ("raw",))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, _ = _materialize_case(case, root, 7)
            report_path = root / "locality.json"
            report = analyze_trace(trace, line_size=PAGE, page_size=PAGE)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            validate_locality_report(case, trace, report_path)
            report["trace_sha256"] = "0" * 64
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "trace_sha256"):
                validate_locality_report(case, trace, report_path)

    def test_stale_suite_commit_records_are_withdrawn_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "synthetic-manifest.json"
            csv_path = root / "synthetic-results.csv"
            breakdown_csv = root / "time-breakdown.csv"
            breakdown_markdown = root / "time-breakdown.md"
            breakdown_html = root / "time-breakdown.html"
            manifest.write_text("old manifest", encoding="utf-8")
            csv_path.write_text("old csv", encoding="utf-8")
            breakdown_csv.write_text("old breakdown", encoding="utf-8")
            breakdown_markdown.write_text("old breakdown", encoding="utf-8")
            breakdown_html.write_text("old visualization", encoding="utf-8")
            returned_manifest, returned_csv = invalidate_suite_outputs(root)
            self.assertEqual((returned_manifest, returned_csv),
                             (manifest, csv_path))
            self.assertFalse(manifest.exists())
            self.assertFalse(csv_path.exists())
            self.assertFalse(breakdown_csv.exists())
            self.assertFalse(breakdown_markdown.exists())
            self.assertFalse(breakdown_html.exists())


class SummaryValidatorTests(unittest.TestCase):
    def test_native_physical_coverage_requires_every_device_bin(self) -> None:
        cases = (
            *build_plan("smoke", ("coverage",)),
            build_plan("smoke", ("output",))[0],
        )
        for case in cases:
            with self.subTest(case=case.key):
                valid = _coverage_summary(case)
                validate_native_physical_coverage(case, valid, 256)
                broken = copy.deepcopy(valid)
                broken["scenarios"][0]["address_heatmap"]["domains"][0][
                    "bins"][17]["read_bytes"] = 0
                with self.assertRaisesRegex(RuntimeError, "occupied 255/256"):
                    validate_native_physical_coverage(case, broken, 256)
                if case.group == "output":
                    wrong_direction = copy.deepcopy(valid)
                    target_bin = wrong_direction["scenarios"][0][
                        "address_heatmap"]["domains"][0]["bins"][17]
                    target_bin["read_bytes"] = 0
                    target_bin["write_bytes"] = PAGE
                    with self.assertRaisesRegex(RuntimeError, "read-only"):
                        validate_native_physical_coverage(
                            case, wrong_direction, 256)

    def test_output_invariants_bind_direct_path_and_all_three_configs(self) -> None:
        expected_ports = [16, 32, 32]
        for profile, expected_reads in (("smoke", 1), ("core", 4)):
            case = build_plan(profile, ("output",))[0]
            for target, ports in zip(case.targets, expected_ports, strict=True):
                with self.subTest(profile=profile, config=target.config):
                    valid = _output_summary(case, target)
                    validate_output_invariants(case, target, valid)
                    self.assertEqual(
                        valid["config"]["hbf"]["media_lanes_per_plane"],
                        ports)
                    self.assertEqual(
                        valid["config"]["hbf"][
                            "page_buffer_banks_per_plane"], ports)
                    self.assertEqual(
                        valid["scenarios"][0]["hbf_stats"][
                            "max_subarray_reads"], expected_reads)

    def test_output_invariants_reject_path_ftl_and_fabric_drift(self) -> None:
        case = build_plan("smoke", ("output",))[0]
        target = case.targets[-1]
        valid = _output_summary(case, target)
        corruptions = (
            (("config", "static_direct_hbm_bytes"), 1,
             "static_direct_hbm_bytes"),
            (("config", "hbf", "subarrays_per_plane"), 31,
             "subarrays_per_plane"),
            (("config", "hbf", "media_lanes_per_plane"), 16,
             "media_lanes_per_plane"),
            (("config", "hbf", "channel_bw_GBps"), 421.875,
             "channel_bw_GBps"),
            (("config", "hbf", "hbio_bw_GBps"), 1600,
             "hbio_bw_GBps"),
            (("config", "hbf", "tsv_bw_GBps"), 1712.5,
             "tsv_bw_GBps"),
            (("config", "hbf", "read_ns"), 3999,
             "read_ns"),
            (("config", "hbf", "flash_tsu_issue_ns"), 31.25,
             "flash_tsu_issue_ns"),
            (("config", "hbf", "media_lane_bw_GBps"), 2047,
             "media_lane_bw_GBps"),
            (("config", "hbf", "page_buffer_bw_GBps"), 2047,
             "page_buffer_bw_GBps"),
            (("config", "hbf", "logic_scheduler_issue_ns"), 2,
             "logic_scheduler_issue_ns"),
            (("config", "hbf", "command_address_bytes"), 63,
             "command_address_bytes"),
            (("config", "hbf", "batch_activation"), False,
             "batch_activation"),
            (("scenarios", 0, "hbm_user_accesses"), 1,
             "hbm_user_accesses"),
            (("scenarios", 0, "hbf_user_accesses"), case.pages - 1,
             "hbf_user_accesses"),
            (("scenarios", 0, "hbm_stats", "read_bytes"), PAGE,
             "hbm_stats.read_bytes"),
            (("scenarios", 0, "hybrid_path", "hbf_logical_read_bytes"),
             PAGE, "hbf_logical_read_bytes"),
            (("scenarios", 0, "hbf_stats", "physical_read_bytes"),
             (case.pages - 1) * PAGE, "physical_read_bytes"),
            (("scenarios", 0, "hbf_stats", "mapping_lookup_ops"), 1,
             "mapping_lookup_ops"),
            (("scenarios", 0, "hbf_stats", "page_programs"), 1,
             "page_programs"),
            (("scenarios", 0, "hbf_stats", "gc_runs"), 1,
             "gc_runs"),
            (("scenarios", 0, "hbf_stats", "active_subarrays"),
             OUTPUT_TOPOLOGY_LEAVES - 1, "active_subarrays"),
            (("scenarios", 0, "hbf_stats", "max_subarray_reads"), 2,
             "max_subarray_reads"),
            (("scenarios", 0, "hbf_stats", "subarray_read_skew"), 1.1,
             "subarray_read_skew"),
        )
        for path, value, message in corruptions:
            with self.subTest(field=message):
                broken = copy.deepcopy(valid)
                cursor = broken
                for component in path[:-1]:
                    cursor = cursor[component]
                cursor[path[-1]] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    validate_output_invariants(case, target, broken)

    def test_duplicate_scenario_names_are_rejected_before_set_comparison(self) -> None:
        case = build_plan("smoke", ("raw",))[0]
        target = case.targets[0]
        data = {
            "schema": {"name": "hbfsim.scenario_compare.summary", "version": 16},
            "sanity": "PASS",
            "scenarios": [{"name": "all-HBM"}, {"name": "all-HBM"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace"
            trace.write_text("0 R 4096\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate scenario"):
                validate_summary(
                    case, target, data, trace,
                    {"sha256": "0" * 64, "bytes": 1, "path": "ignored"},
                    512, 256)

    def test_csv_extractor_zeros_null_devices_but_rejects_missing_fields(self) -> None:
        scenario = {
            "name": "all-HBM",
            "user_completion_throughput_GBps": 1,
            "makespan_throughput_GBps": 1,
            "time_breakdown": {
                "wall_clock_ns": {
                    "user_completion_span_ns": 1,
                    "drain_tail_ns": 0,
                    "makespan_ns": 1,
                },
                "latency_work": {
                    "average_ns": 1,
                    "p50_ns": 1,
                    "p95_ns": 1,
                    "max_ns": 1,
                },
            },
            "hbm_stats": None,
            "hbf_stats": None,
            "layer_streaming": None,
            "hybrid_path": {"hbm_streaming_write_bytes": 0},
        }
        case = build_plan("smoke", ("raw",))[0]
        result = {
            "case": case,
            "target": case.targets[0],
            "data": {
                "config": {
                    "hbf": {
                        "subarrays_per_plane": 32,
                        "media_lanes_per_plane": 16,
                        "page_buffer_banks_per_plane": 16,
                        "read_ns": 1000,
                        "flash_tsu_issue_ns": 10,
                        "media_lane_bw_GBps": 2048,
                        "page_buffer_bw_GBps": 2048,
                        "logic_scheduler_issue_ns": 2,
                        "command_address_bytes": 64,
                        "batch_activation": True,
                        "ecc_decode_raw_bw_GBps_per_die": 105.46875,
                        "channel_bw_GBps": 421.875,
                        "tsv_bw_GBps": 1712.5,
                        "logic_sram_bw_GBps": 2048,
                        "hbio_bw_GBps": 1600,
                    },
                },
                "scenarios": [scenario],
            },
            "window": 512,
            "host_seconds": 1.0,
        }
        row = _rows(result)[0]
        self.assertEqual(row["config"], "usecase-baseline.cfg")
        self.assertIsNone(row["placement_modulus"])
        self.assertEqual(row["hbf_media_lanes_per_plane"], 16)
        self.assertEqual(row["hbf_flash_tsu_issue_ns"], 10)
        self.assertEqual(row["hbm_read_bytes"], 0)
        self.assertEqual(row["hbf_page_reads"], 0)
        self.assertEqual(row["streaming_pages"], 0)

        output_case = build_plan("smoke", ("output",))[0]
        result["case"] = output_case
        result["target"] = output_case.targets[0]
        output_row = _rows(result)[0]
        self.assertEqual(
            output_row["placement_modulus"], OUTPUT_TOPOLOGY_LEAVES)

        result["data"]["scenarios"][0]["hbm_stats"] = {"write_bytes": 0}
        with self.assertRaisesRegex(RuntimeError, "read_bytes"):
            _rows(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
