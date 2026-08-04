#!/usr/bin/env python3
"""Contract and mutation tests for the Frontier 70B HBF wear study."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from run_frontier_70b_hbf_wear_study import (  # noqa: E402
    PAGE_SIZE,
    SUMMARY_SCHEMA,
    WAF_DEFINITION,
    WearStudyError,
    _ReplayState,
    _aggregate,
    _generate_base_traces,
    _load_study_config,
    _sha256,
    _static_weight_reserved_blocks,
    _verify_media_summary,
)
from validation.certificate import EXPLORATORY_VALIDATION  # noqa: E402


class FrontierWearStudyTests(unittest.TestCase):
    def test_tracked_config_predeclares_three_residues_and_topologies(self) -> None:
        config = _load_study_config(
            ROOT / "configs" / "studies" / "frontier-70b-hbf-wear.json"
        )
        self.assertEqual(config["sampling"]["block_stride"], 64)
        self.assertEqual(config["sampling"]["residues"], [0, 21, 42])
        self.assertEqual(config["epoch_counts"], [1, 8])
        self.assertEqual(
            [value["id"] for value in config["topologies"]],
            ["6H2F", "4H4F", "2H6F"],
        )
        self.assertEqual(
            [
                value["expected_normalized_mutable_blocks_per_plane"]
                for value in config["topologies"]
            ],
            [120, 124, 125],
        )

    def test_static_weight_block_count_matches_complete_and_edge_extents(
        self,
    ) -> None:
        common = {
            "stacks": 1,
            "planes_per_stack": 2,
            "blocks_per_plane": 4,
            "pages_per_block": 4,
        }
        self.assertEqual(_static_weight_reserved_blocks(
            weight_begin_page=0,
            weight_end_page=8,
            **common,
        ), 2)
        self.assertEqual(_static_weight_reserved_blocks(
            weight_begin_page=1,
            weight_end_page=9,
            **common,
        ), 3)
        self.assertEqual(_static_weight_reserved_blocks(
            weight_begin_page=0,
            weight_end_page=32,
            **common,
        ), 8)

    def test_lifecycle_state_rejects_unknown_release_and_tracks_order(self) -> None:
        state = _ReplayState(4)
        state.apply({
            "event_index": 0,
            "event": "allocate",
            "request_id": "r",
            "block_id": 2,
        })
        state.apply({
            "event_index": 1,
            "event": "touch",
            "request_id": "r",
            "blocks": [{"block_id": 3}],
        })
        self.assertEqual(state.request_blocks["r"], [2, 3])
        state.apply({
            "event_index": 2,
            "event": "release",
            "request_id": "r",
            "blocks": [{"block_id": 2}, {"block_id": 3}],
        })
        self.assertEqual(state.request_blocks, {})
        with self.assertRaisesRegex(WearStudyError, "unknown request"):
            state.apply({
                "event_index": 3,
                "event": "release",
                "request_id": "r",
                "blocks": [{"block_id": 2}],
            })

    def test_tiny_replay_preserves_batch_layer_request_token_order(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbfsim-frontier-wear-replay-test-"
        ) as directory:
            root = Path(directory)
            lifecycle = root / "lifecycle.jsonl"
            ledger = root / "ledger.jsonl"
            lifecycle.write_text(
                "\n".join(json.dumps(value) for value in ({
                    "event_index": 0,
                    "event": "allocate",
                    "request_id": "7",
                    "block_id": 0,
                }, {
                    "event_index": 1,
                    "event": "allocate",
                    "request_id": "7",
                    "block_id": 2,
                }, {
                    "event_index": 2,
                    "event": "release",
                    "request_id": "7",
                    "blocks": [{"block_id": 0}, {"block_id": 2}],
                })) + "\n",
                encoding="utf-8",
            )
            ledger.write_text(json.dumps({
                "batch_id": 0,
                "hbfsim_memory_contract": {
                    "schema_version": 4,
                    "lifecycle_event_cursor": 2,
                    "requests": [{
                        "request_id": "7",
                        "allocated_block_count": 2,
                        "kv_tokens_before": 16,
                        "scheduled_tokens": 1,
                        "phase": "decode",
                    }],
                },
            }) + "\n", encoding="utf-8")
            source = {
                "logical_kv_blocks": 4,
                "lifecycle_path": lifecycle,
                "ledger_path": ledger,
                "ledger_rows": 1,
                "lifecycle_events": 3,
            }
            placements = [{
                "id": "T",
                "stacks": 1,
                "hot_kv_blocks": 2,
                "cold_kv_blocks": 2,
                "usable_hbf_payload_capacity_bytes": 1024 * PAGE_SIZE,
            }, {
                "id": "U",
                "stacks": 2,
                "hot_kv_blocks": 2,
                "cold_kv_blocks": 2,
                "usable_hbf_payload_capacity_bytes": 1024 * PAGE_SIZE,
            }]
            demand, cells = _generate_base_traces(
                source=source,
                placements=placements,
                sampling={"block_stride": 2, "residues": [0]},
                workload={
                    "num_layers": 2,
                    "kv_block_tokens": 16,
                    "kv_bytes_per_token_per_layer": PAGE_SIZE,
                },
                out_dir=root / "out",
            )
            self.assertEqual(demand["scheduled_kv_append_tokens"], 1)
            exact = demand["full_population"]["T"]
            self.assertEqual(exact["cold_kv_token_writes"], 1)
            self.assertEqual(exact["logical_hbf_write_bytes"], 2 * PAGE_SIZE)
            cell = cells["T.r0"]
            self.assertEqual(cell["sampled_token_writes_per_epoch"], 1)
            self.assertEqual(cell["expected_mapping_checkpoint_pages"], 1)
            self.assertEqual(
                cells["U.r0"]["expected_mapping_checkpoint_pages"],
                1,
            )
            records = [
                line.split()
                for line in Path(cell["base_trace_path"]).read_text().splitlines()
                if line and not line.startswith("#")
            ]
            self.assertEqual(records, [
                ["0x0", "W", "4096"],
                ["0x10000", "W", "4096"],
            ])

    @staticmethod
    def _valid_summary(trace: Path) -> dict:
        logical = 2 * PAGE_SIZE
        programs = 3
        return {
            "schema": SUMMARY_SCHEMA,
            "sanity": "PASS",
            "validation": EXPLORATORY_VALIDATION,
            "workload": {
                "trace_file_bytes": trace.stat().st_size,
                "trace_digest": {
                    "algorithm": "sha256",
                    "value": _sha256(trace),
                },
            },
            "config": {
                "hbf": {
                    "capacity_bytes": 100 * PAGE_SIZE,
                    "stacks": 1,
                    "pages_per_block": 10,
                    "gc_reserved_free_blocks_per_plane": 0,
                    "gc_low_watermark_pages": 0,
                    "gc_hard_watermark_pages": 0,
                },
            },
            "scenarios": [{
                "name": "all-HBF",
                "ops": 2,
                "reads": 0,
                "writes": 2,
                "logical_bytes": logical,
                "hbf_stats": {
                    "logical_read_bytes": 0,
                    "logical_write_bytes": logical,
                    "physical_write_bytes": programs * PAGE_SIZE,
                    "data_program_payload_bytes": 2 * PAGE_SIZE,
                    "mapping_program_payload_bytes": PAGE_SIZE,
                    "gc_relocation_payload_bytes": 0,
                    "page_programs": programs,
                    "data_programs": 2,
                    "mapping_page_programs": 1,
                    "gc_relocations": 0,
                    "gc_runs": 0,
                    "block_erases": 0,
                    "gc_data_relocations": 0,
                    "gc_mapping_relocations": 0,
                    "gc_reclaimed_invalid_pages": 0,
                    "initial_logical_data_pages": 2,
                    "initial_mapping_pages": 1,
                    "page_reads": 0,
                    "physical_read_bytes": 0,
                    "flash_scheduler_enqueues": 3,
                    "flash_scheduler_issues": 3,
                    "ecc_decode_ops": 0,
                    "ecc_encode_ops": 3,
                    "static_unmaterialized_pages": 0,
                    "free_pages": 94,
                    "valid_pages": 3,
                    "invalid_pages": 3,
                    "mapping_entries": 2,
                    "pending_program_pages": 0,
                    "pending_mapping_publications": 0,
                    "accounting_verified": True,
                    "total_pages": 100,
                    "waf": 1.5,
                    "waf_definition": WAF_DEFINITION,
                },
            }],
        }

    def test_media_summary_oracle_rejects_physical_write_mutation(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="hbfsim-frontier-wear-summary-test-"
        ) as directory:
            root = Path(directory)
            trace = root / "trace"
            trace.write_text("0x0 W 4096\n0x1000 W 4096\n", encoding="utf-8")
            summary_path = root / "summary.json"
            valid = self._valid_summary(trace)
            summary_path.write_text(json.dumps(valid), encoding="utf-8")
            _, result = _verify_media_summary(
                summary_path=summary_path,
                trace_record={
                    "bytes": trace.stat().st_size,
                    "sha256": _sha256(trace),
                },
                expected_records=2,
                expected_initial_pages=2,
                expected_initial_mapping_pages=1,
                expected_mapping_programs=1,
                validation_certificate=None,
            )
            self.assertEqual(result["waf"], 1.5)

            broken = copy.deepcopy(valid)
            broken["scenarios"][0]["hbf_stats"]["physical_write_bytes"] -= 1
            summary_path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(WearStudyError, "physical write bytes"):
                _verify_media_summary(
                    summary_path=summary_path,
                    trace_record={
                        "bytes": trace.stat().st_size,
                        "sha256": _sha256(trace),
                    },
                    expected_records=2,
                    expected_initial_pages=2,
                    expected_initial_mapping_pages=1,
                    expected_mapping_programs=1,
                    validation_certificate=None,
                )

    def test_lifetime_projection_is_explicit_uniform_wear_sensitivity(self) -> None:
        study = {
            "epoch_counts": [1, 8],
            "lifetime_sensitivity": {
                "kv_append_tokens_per_second": [100],
                "assumed_media_cycles": [1000],
            },
        }
        demand = {
            "full_population": {
                "T": {
                    "hot_kv_blocks": 1,
                    "cold_kv_blocks": 1,
                    "cold_kv_token_writes": 50,
                    "cold_kv_token_fraction": 0.5,
                    "logical_hbf_write_bytes": 1000,
                    "logical_hbf_write_bytes_per_kv_append_token": 10.0,
                },
            },
        }
        placements = [{
            "id": "T",
            "stacks": 1,
            "usable_hbf_payload_capacity_bytes": 1_000_000,
        }]
        results = [{
            "cell_id": "T.r0",
            "topology": "T",
            "epochs": 8,
            "waf": 1.2,
            "logical_write_bytes": 100,
            "physical_write_bytes": 120,
            "gc_runs": 1,
            "gc_relocations": 2,
        }, {
            "cell_id": "T.r1",
            "topology": "T",
            "epochs": 8,
            "waf": 1.4,
            "logical_write_bytes": 100,
            "physical_write_bytes": 140,
            "gc_runs": 1,
            "gc_relocations": 3,
        }]
        topology, lifetime = _aggregate(
            study=study,
            demand=demand,
            placements=placements,
            results=results,
        )
        self.assertAlmostEqual(topology[0]["measured_waf_mean"], 1.3)
        self.assertAlmostEqual(
            topology[0]["projected_physical_write_bytes"],
            1300,
        )
        self.assertEqual(
            lifetime[0]["status"],
            "uncalibrated_first_order_sensitivity",
        )
        expected_days = 13.0 * 100 * 86400 / 1_000_000
        self.assertAlmostEqual(
            lifetime[0]["usable_capacity_writes_per_day"],
            expected_days,
        )
        missing_gc = copy.deepcopy(results)
        missing_gc[0]["gc_runs"] = 0
        with self.assertRaisesRegex(WearStudyError, "did not exercise GC"):
            _aggregate(
                study=study,
                demand=demand,
                placements=placements,
                results=missing_gc,
            )


if __name__ == "__main__":
    unittest.main()
