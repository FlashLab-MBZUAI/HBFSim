#!/usr/bin/env python3
"""Focused byte-range census regressions for check_ec_sanity.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from check_ec_sanity import (
    BOUNDARY,
    PAGE,
    census_lines,
    check_ec6_fill_conservation,
    select_cases,
    sha256_file,
    validate_manifest,
    validate_summary,
)


def request(addr: int, op: str, size: int) -> str:
    return f"0x{addr:x} {op} {size}\n"


class CensusTests(unittest.TestCase):
    def ecc_summary_fixture(self) -> dict:
        return {
            "schema": {
                "name": "hbfsim.scenario_compare.summary",
                "version": 16,
            },
            "sanity": "PASS",
            "workload": {
                "trace_digest": {"algorithm": "sha256", "value": "a" * 64},
            },
            "config": {
                "ops": 1,
                "max_outstanding_requests": 512,
                "layer_buffer_bytes": 32 * 1024**3,
                "flat_hbm_bytes": BOUNDARY,
                "static_direct_hbm_bytes": BOUNDARY,
                "line_size": PAGE,
                "hbf_hbm_write_buffer_destage": "deferred",
                "hbm": {"stacks": 6},
                "hbf": {
                    "stacks": 2,
                    "placement_mapping_scheme":
                        "page-striped-stack-local-mapping-v2",
                    "page_size": PAGE,
                    "read_ns": 1000.0,
                    "program_ns": 95000.0,
                    "program_verify_ns": 5000.0,
                    "ecc_decode_latency_ns": 250.0,
                    "ecc_encode_latency_ns": 250.0,
                    "ecc_decode_raw_bw_GBps_per_die": 105.46875,
                    "ecc_encode_raw_bw_GBps_per_die": 105.46875,
                    "ecc_codeword_size_bytes": 4320,
                    "ecc_decode_initiation_ns": 40.96,
                    "ecc_encode_initiation_ns": 40.96,
                    "ecc_issue_topology": "shared-per-die",
                },
            },
            "simulator": {"git_commit": "abc", "git_dirty": False},
            "build": {
                "executable_digest": {"algorithm": "sha256", "value": "b" * 64},
            },
        }

    def test_summary_binds_ecc_contract(self) -> None:
        fixture = self.ecc_summary_fixture()
        validate_summary(
            fixture, "a" * 64, 512, 1, (6, 2), Path("summary.json"))
        for field, invalid in (
                ("ecc_decode_raw_bw_GBps_per_die", 100.0),
                ("ecc_decode_initiation_ns", 41.0),
                ("ecc_issue_topology", "split-per-die")):
            mutated = json.loads(json.dumps(fixture))
            mutated["config"]["hbf"][field] = invalid
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_summary(
                    mutated, "a" * 64, 512, 1, (6, 2), Path("summary.json"))
        mutated = json.loads(json.dumps(fixture))
        mutated["config"]["hbf"]["placement_mapping_scheme"] = "obsolete"
        with self.assertRaises(RuntimeError):
            validate_summary(
                mutated, "a" * 64, 512, 1, (6, 2), Path("summary.json"))

    def test_partial_write_does_not_park_untouched_bytes(self) -> None:
        result = census_lines([
            request(BOUNDARY + 100, "W", 128),
            request(BOUNDARY, "R", PAGE),
        ])
        self.assertEqual(result["rbytes_above_parked"], 128)
        self.assertEqual(result["rbytes_above_hbf"], PAGE - 128)
        self.assertEqual(result["dirty_union_bytes"], 128)

    def test_cross_page_read_accounts_each_page(self) -> None:
        result = census_lines([
            request(BOUNDARY + PAGE, "W", 128),
            request(BOUNDARY + PAGE - 128, "R", 256),
        ])
        self.assertEqual(result["rbytes_above_parked"], 128)
        self.assertEqual(result["rbytes_above_hbf"], 128)
        self.assertEqual(result["above_pages"], 2)

    def test_static_direct_census_is_page_and_placement_aware(self) -> None:
        result = census_lines([
            request(BOUNDARY, "W", PAGE),
            request(BOUNDARY, "R", PAGE),
            f"0x{BOUNDARY + PAGE:x} R {PAGE} metadata\n",
            f"0x{BOUNDARY + 2 * PAGE:x} R {PAGE} model_weights\n",
        ])
        self.assertEqual(result["static_direct_read_page_fragments"], 1)

    def test_boundary_crossing_is_clipped_byte_exactly(self) -> None:
        result = census_lines([
            request(BOUNDARY - 64, "W", 128),
            request(BOUNDARY - 64, "R", 128),
        ])
        self.assertEqual(result["wbytes_above"], 64)
        self.assertEqual(result["rbytes_above"], 64)
        self.assertEqual(result["rbytes_above_parked"], 64)
        self.assertEqual(result["rbytes_above_hbf"], 0)

    def test_overlapping_writes_count_union_once(self) -> None:
        result = census_lines([
            request(BOUNDARY + 100, "W", 100),
            request(BOUNDARY + 150, "W", 100),
        ])
        self.assertEqual(result["wbytes_above"], 200)
        self.assertEqual(result["dirty_union_bytes"], 150)
        self.assertEqual(result["wpages_above"], 1)
        self.assertEqual(result["dirty_range_count_by_kind"]["unknown"], 1)

    def test_inline_comments_and_ast_kind_are_accepted(self) -> None:
        result = census_lines([
            request(BOUNDARY, "R", 128).rstrip() +
            " generated_context at=1.25 # decode KV\n",
        ])
        self.assertEqual(result["reads"], 1)
        self.assertEqual(result["per_kind"]["generated_context"], 128)

    def test_layer_census_supports_plain_and_explicit_streams(self) -> None:
        plain = census_lines([f"0x1000 R {PAGE} model_weights\n"])
        self.assertEqual(plain["streaming_layers"], 1)
        self.assertEqual(plain["explicit_layer_requests"], 0)

        explicit = census_lines([
            f"0x1000 R {PAGE} model_weights layer=0\n",
            f"0x2000 W {PAGE} generated_context layer=1\n",
        ])
        self.assertEqual(explicit["streaming_layers"], 2)
        self.assertEqual(explicit["explicit_layer_requests"], 2)
        with self.assertRaises(ValueError):
            census_lines([
                f"0x1000 R {PAGE} model_weights layer=1\n",
                f"0x2000 R {PAGE} model_weights layer=0\n",
            ])

    def test_ec6_layer_dma_and_writeback_conserve(self) -> None:
        trace = census_lines([
            f"0x1000 R {PAGE} model_weights layer=0\n",
            f"0x2000 W {PAGE} generated_context layer=1\n",
        ])
        scenario = {
            "ops": 2,
            "hbm_user_accesses": 2,
            "hbf_user_accesses": 0,
            "layer_streaming": {
                "mode": "layer_streaming",
                "residency_policy": "trace-derived-first-touch",
                "compact_resident_mapping": False,
                "layers": 2,
                "explicit_layer_requests": 2,
                "data_pages": 2,
                "resident_physical_pages": 0,
                "hbm_only_resident_pages": 0,
                "hot_kv_resident_pages": 0,
                "model_weight_resident_pages": 0,
                "backing_unique_pages": 2,
                "effective_layer_buffer_pages": 1,
                "max_layer_data_pages": 1,
                "foreground_resident_page_accesses": 0,
                "foreground_buffer_page_accesses": 2,
                "streamed_pages": 1,
                "streamed_bytes": PAGE,
                "dirty_pages_written_back": 1,
                "writeback_bytes": PAGE,
            },
            "hybrid_path": {
                "base_die_link_read_bytes": PAGE,
                "base_die_link_write_bytes": PAGE,
                "hbm_streaming_write_bytes": PAGE,
                "hbf_static_read_bytes": PAGE,
                "hbf_logical_read_bytes": 0,
                "hbf_backing_write_bytes": PAGE,
            },
        }
        detail, ok = check_ec6_fill_conservation(trace, scenario)
        self.assertTrue(ok, detail)

        scenario["hybrid_path"]["hbm_streaming_write_bytes"] = 0
        detail, ok = check_ec6_fill_conservation(trace, scenario)
        self.assertFalse(ok, detail)

    def test_ec6_layer_check_rejects_boolean_counters(self) -> None:
        trace = census_lines([f"0x1000 R {PAGE} model_weights\n"])
        scenario = {
            "ops": 1,
            "hbm_user_accesses": 1,
            "hbf_user_accesses": 0,
            "layer_streaming": {
                "mode": "layer_streaming",
                "residency_policy": "trace-derived-first-touch",
                "compact_resident_mapping": False,
                "layers": 1,
                "explicit_layer_requests": 0,
                "data_pages": 1,
                "resident_physical_pages": 0,
                "hbm_only_resident_pages": 0,
                "hot_kv_resident_pages": 0,
                "model_weight_resident_pages": 0,
                "backing_unique_pages": 1,
                "effective_layer_buffer_pages": 1,
                "max_layer_data_pages": 1,
                "foreground_resident_page_accesses": 0,
                "foreground_buffer_page_accesses": 1,
                "streamed_pages": True,
                "streamed_bytes": PAGE,
                "dirty_pages_written_back": 0,
                "writeback_bytes": 0,
            },
            "hybrid_path": {
                "base_die_link_read_bytes": PAGE,
                "base_die_link_write_bytes": 0,
                "hbm_streaming_write_bytes": PAGE,
                "hbf_static_read_bytes": PAGE,
                "hbf_logical_read_bytes": 0,
                "hbf_backing_write_bytes": 0,
            },
        }
        detail, ok = check_ec6_fill_conservation(trace, scenario)
        self.assertFalse(ok, detail)

    def test_non_normalized_or_invalid_ast_input_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            census_lines([f"R 0x{BOUNDARY:x} 128\n"])
        with self.assertRaises(ValueError):
            census_lines([request(BOUNDARY, "X", 128)])
        with self.assertRaises(ValueError):
            census_lines([request((1 << 64) - 64, "W", 128)])
        with self.assertRaises(ValueError):
            census_lines([f"0x{BOUNDARY:x} W 0\n"])

    def test_manifest_binds_case_artifacts(self) -> None:
        cases = select_cases("ec2,ec3")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = [
                root / "usecase-6h2f.summary.json",
                root / "usecase-4h4f.summary.json",
                root / "astra-replay-ec2-ec3.csv",
                root / "astra-replay-ec2-ec3.time-breakdown.csv",
                root / "astra-replay-ec2-ec3.time-breakdown.md",
                root / "astra-replay-ec2-ec3.time-breakdown.html",
            ]
            for index, artifact in enumerate(artifacts):
                artifact.write_text(f"artifact-{index}\n")
            trace_digest = "a" * 64
            workload_id = "b" * 64
            workload_manifest = root / "llm.manifest.json"
            workload_manifest.write_text("workload manifest\n")
            manifest = {
                "schema": {
                    "name": "hbfsim.astra_replay.manifest",
                    "version": 2,
                },
                "cases": ["ec2", "ec3"],
                "window": 512,
                "boundary": BOUNDARY,
                "boundary_source": "workload_manifest.kv.base",
                "workload_manifest": {
                    "path": str(workload_manifest),
                    "sha256": sha256_file(workload_manifest),
                    "workload_id": workload_id,
                    "kv_base": BOUNDARY,
                },
                "trace": {
                    "sha256": trace_digest,
                    "workload_id": workload_id,
                },
                "artifacts": {
                    artifact.name: {"sha256": sha256_file(artifact)}
                    for artifact in artifacts
                },
            }
            (root / "astra-replay-ec2-ec3.manifest.json").write_text(
                json.dumps(manifest))
            self.assertEqual(
                validate_manifest(root, cases, trace_digest, 512), BOUNDARY)

            missing = json.loads(json.dumps(manifest))
            missing["artifacts"].pop(
                "astra-replay-ec2-ec3.time-breakdown.md")
            (root / "astra-replay-ec2-ec3.manifest.json").write_text(
                json.dumps(missing))
            with self.assertRaisesRegex(RuntimeError, "artifact set mismatch"):
                validate_manifest(root, cases, trace_digest, 512)

            extra = json.loads(json.dumps(manifest))
            extra["artifacts"]["unexpected.txt"] = {"sha256": "0" * 64}
            (root / "astra-replay-ec2-ec3.manifest.json").write_text(
                json.dumps(extra))
            with self.assertRaisesRegex(RuntimeError, "artifact set mismatch"):
                validate_manifest(root, cases, trace_digest, 512)

            (root / "astra-replay-ec2-ec3.manifest.json").write_text(
                json.dumps(manifest))
            artifacts[-1].unlink()
            with self.assertRaisesRegex(RuntimeError, "artifact digest mismatch"):
                validate_manifest(root, cases, trace_digest, 512)
            artifacts[-1].write_text(f"artifact-{len(artifacts) - 1}\n")

            artifacts[0].write_text("mutated\n")
            with self.assertRaisesRegex(RuntimeError, "artifact digest mismatch"):
                validate_manifest(root, cases, trace_digest, 512)

    def test_dynamic_boundary_is_used_by_census_and_summary(self) -> None:
        boundary = 0x8000
        trace = census_lines([
            request(0x7000, "R", PAGE),
            request(0x8000, "R", PAGE),
        ], boundary)
        self.assertEqual(trace["below"], 1)
        self.assertEqual(trace["above"], 1)

        fixture = self.ecc_summary_fixture()
        fixture["config"]["flat_hbm_bytes"] = boundary
        fixture["config"]["static_direct_hbm_bytes"] = boundary
        validate_summary(
            fixture, "a" * 64, 512, 1, (6, 2), Path("summary.json"),
            boundary)


if __name__ == "__main__":
    unittest.main()
