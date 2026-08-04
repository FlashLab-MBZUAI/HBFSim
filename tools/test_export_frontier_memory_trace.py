#!/usr/bin/env python3
"""Regression tests for export_frontier_memory_trace.py."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from analyze_trace_locality import parse_trace_line  # noqa: E402
from export_frontier_memory_trace import (  # noqa: E402
    AddressAllocator,
    ExportError,
    _surrogate_embedding_row,
    _validate_model_descriptor,
    _weight_layout,
    export_memory_trace,
)
import test_audit_frontier_replay as audit_fixture_module  # noqa: E402


class ExportFrontierMemoryTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.audit_fixture = audit_fixture_module.AuditFrontierReplayTest(
            methodName="audit"
        )
        self.audit_fixture.setUp()
        self.root = Path(self.audit_fixture.root)
        self.audit = self.audit_fixture.output
        self.audit_fixture.audit()
        self.model = self.root / "model.json"
        self.trace = self.root / "memory.trace"
        self.object_map = self.root / "objects.json"
        self.phase_map = self.root / "phases.json"
        self.manifest = self.root / "memory.manifest.json"
        self._write_model()

    def tearDown(self) -> None:
        self.audit_fixture.tearDown()

    def _write_model(self) -> None:
        self.model.write_text(
            json.dumps(
                {
                    "schema": {
                        "name": "hbfsim.frontier_dense_model_memory",
                        "version": 2,
                    },
                    "model": {
                        "name": "fixture-model",
                        "architecture": "dense_decoder_transformer",
                        "source": {
                            "frontier_repository": (
                                "https://github.com/NetX-lab/Frontier"
                            ),
                            "frontier_config_path": (
                                "data/config/models/fixture-model.json"
                            ),
                            "frontier_revision": (
                                "a4b22df8211864bf229258ecdfbe680f048f2d77"
                            ),
                            "frontier_config_sha256": "a" * 64,
                            "model_repository": (
                                "https://example.invalid/model"
                            ),
                            "model_revision": "0" * 40,
                            "config_access": "unit-test",
                        },
                    },
                    "architecture": {
                        "num_layers": 2,
                        "hidden_size": 8,
                        "intermediate_size": 16,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "vocab_size": 32,
                        "gated_mlp": True,
                        "attention_bias": False,
                        "mlp_bias": False,
                        "tie_word_embeddings": False,
                    },
                    "precision": {
                        "profile_id": "fp16-ci-smoke",
                        "matrix_weight_dtype": "float16",
                        "matrix_weight_bytes": 2,
                        "non_matrix_weight_dtype": "float16",
                        "non_matrix_weight_bytes": 2,
                        "kv_dtype": "float16",
                        "kv_bytes": 2,
                        "activation_dtype": "float16",
                        "activation_bytes": 2,
                        "quantization_scheme": "none",
                        "scale_dtype": None,
                        "scale_bytes": 0,
                        "zero_point_bytes": 0,
                        "claim_scope": (
                            "memory_storage_and_traffic_only"
                        ),
                    },
                    "traffic_model": {
                        "weight_reads": "once_per_batch_per_layer",
                        "attention_kv_reads": (
                            "context_once_per_request_per_layer"
                        ),
                        "kv_writes": (
                            "scheduled_input_tokens_once_per_layer"
                        ),
                        "output_head_reads": (
                            "once_per_batch_with_output_token"
                        ),
                        "embedding_address_policy": (
                            "sha256_per_scheduled_token_surrogate_rows"
                        ),
                        "block_table_entry_bytes": 4,
                        "scratch_read_bytes_per_token_per_layer": 0,
                        "scratch_write_bytes_per_token_per_layer": 0,
                    },
                    "addressing": {
                        "object_alignment_bytes": 4096,
                        "kv_slot_identity": "frontier_physical_block_id",
                        "kv_slot_layout": (
                            "block_major_layer_major_token_major"
                        ),
                        "kv_slot_layout_status": (
                            "canonical_object_mapping_not_backend_measured"
                        ),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def export(self, *, batch_start: int = 0, max_batches: int | None = 1) -> dict:
        return export_memory_trace(
            audit_path=self.audit,
            model_descriptor_path=self.model,
            output_trace=self.trace,
            object_map_path=self.object_map,
            phase_map_path=self.phase_map,
            manifest_path=self.manifest,
            batch_start=batch_start,
            max_batches=max_batches,
        )

    def test_export_is_byte_exact_collision_free_and_parseable(self) -> None:
        result = self.export()
        self.assertEqual(
            result["schema"],
            {"name": "hbfsim.frontier_memory_trace", "version": 4},
        )
        self.assertEqual(result["selection"]["batch_ids"], [0])
        self.assertEqual(
            result["selection"]["selected_batch_census"],
            [
                {
                    "batch_id": 0,
                    "request_count": 2,
                    "request_ids": ["0", "1"],
                    "scheduled_tokens": 32,
                    "request_phases": ["prefill"],
                    "produces_output_token": True,
                    "frontier_stage_start_s": 0.0,
                    "frontier_stage_end_s": 1.0,
                }
            ],
        )
        self.assertFalse(result["selection"]["is_full_replay"])
        self.assertFalse(result["selection"]["scaling_applied"])
        self.assertEqual(
            result["equations"]["evaluated"]["kv_bytes_per_token_per_layer"],
            16,
        )
        self.assertEqual(
            result["equations"]["evaluated"]["dense_layer_weight_bytes"],
            1184,
        )
        capacity = result["capacity_accounting"]
        self.assertEqual(capacity["planner_mode"], "hybrid_residency")
        self.assertEqual(capacity["weight_memory_source"], "param_counter")
        self.assertEqual(
            capacity["runtime_overhead_source"],
            "configured_unprofiled_sensitivity",
        )
        self.assertFalse(
            capacity["non_kv_cache_overhead_runtime_profiled"]
        )
        self.assertFalse(capacity["hardware_capacity_calibrated"])
        self.assertEqual(
            capacity["capacity_interpretation"],
            "unique_footprint_hybrid_residency_sensitivity",
        )
        self.assertEqual(capacity["immutable_weight_backing_bytes"], 3408)
        self.assertEqual(capacity["model_parameters"], 1704)
        self.assertEqual(
            capacity["residency_plan"],
            json.loads(self.audit.read_text(encoding="utf-8"))[
                "residency_plan"
            ],
        )
        self.assertFalse(
            result["semantics"]["kv_capacity_is_hardware_calibrated"]
        )
        census = result["traffic_census"]
        self.assertEqual(census["bytes"], 5992)
        self.assertEqual(census["writes"]["bytes"], 1024)
        self.assertEqual(census["reads"]["bytes"], 4968)

        object_map = json.loads(self.object_map.read_text(encoding="utf-8"))
        regions = object_map["regions"]
        for previous, current in zip(regions, regions[1:]):
            self.assertLessEqual(previous["end"], current["begin"])
        self.assertEqual(
            object_map["address_space"]["bytes"] % 4096,
            0,
        )
        self.assertEqual(
            object_map["excluded_objects"][0]["name"],
            "scratch.activations",
        )
        regions_by_name = {
            region["name"]: region for region in object_map["regions"]
        }
        self.assertEqual(
            regions_by_name["metadata.kv_allocator_blocks"]["bytes"],
            20,
        )
        self.assertEqual(
            regions_by_name["metadata.runtime_overhead"]["bytes"],
            self.audit_fixture.RUNTIME_OVERHEAD_BYTES,
        )

        parsed = []
        for line_no, line in enumerate(
            self.trace.read_text(encoding="utf-8").splitlines(), start=1
        ):
            operation = parse_trace_line(line, line_no=line_no, line_size=64)
            if operation is not None:
                fields = {
                    key: value
                    for token in line.split()
                    if "=" in token
                    for key, value in [token.split("=", 1)]
                }
                self.assertEqual(fields.get("layer"), fields.get("phase"))
                parsed.append(operation)
        self.assertEqual(len(parsed), census["operations"])
        self.assertEqual(sum(operation.bytes for operation in parsed), 5992)

        phase_map = json.loads(self.phase_map.read_text(encoding="utf-8"))
        self.assertEqual(
            phase_map["timing_semantics"],
            {
                "mode": "dependency_barrier_batch_ready",
                "frontier_stage_start_preserved": True,
                "frontier_execution_time_used_as_hardware_latency": False,
                "phase_dependency": "complete_before_next",
                "phase_scope": "global_trace_order",
                "layer_scope": "global_phase_streaming_window",
            },
        )
        self.assertEqual(
            [phase["component"] for phase in phase_map["phases"]],
            [
                "embedding",
                "transformer_layer",
                "transformer_layer",
                "final_norm",
                "lm_head",
            ],
        )
        self.assertEqual(
            sum(phase["bytes"] for phase in phase_map["phases"]),
            census["bytes"],
        )
        self.assertEqual(
            len({phase["offered_at_ns"] for phase in phase_map["phases"]}),
            1,
        )
        self.assertEqual(
            result["semantics"]["export_mode"],
            "dependency_barrier_batch_ready",
        )
        self.assertEqual(
            result["semantics"]["phase_dependency"],
            "complete_before_next",
        )
        self.assertIn(
            "# timing=dependency_barrier_batch_ready "
            "phase_contract=complete-before-next "
            "layer_contract=global-phase-streaming-window",
            self.trace.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            result["semantics"]["layer_scope"],
            "global_phase_streaming_window",
        )

    def test_llama31_70b_precision_descriptors_have_exact_ledgers(self) -> None:
        expected = {
            "llama31-70b-w8a16-kv-bf16.json": {
                "resident_bytes": 70_568_973_312,
                "storage_breakdown": {
                    "matrix_payload_bytes": 70_552_387_584,
                    "non_matrix_payload_bytes": 2_637_824,
                    "quantization_metadata_bytes": 13_947_904,
                },
                "breakdown_bytes": {
                    "attention": 12_082_544_640,
                    "ffn": 56_381_931_520,
                    "normalization": 2_621_440,
                    "pipeline_boundary_and_auxiliary": 2_101_875_712,
                },
            },
            "llama31-70b-bf16-kv-bf16.json": {
                "resident_bytes": 141_107_412_992,
                "storage_breakdown": {
                    "matrix_payload_bytes": 141_104_775_168,
                    "non_matrix_payload_bytes": 2_637_824,
                    "quantization_metadata_bytes": 0,
                },
                "breakdown_bytes": {
                    "attention": 24_159_191_040,
                    "ffn": 112_742_891_520,
                    "normalization": 2_621_440,
                    "pipeline_boundary_and_auxiliary": 4_202_708_992,
                },
            },
        }
        descriptor_dir = (
            TOOLS.parent / "configs/workloads/frontier"
        )
        for file_name, expected_ledger in expected.items():
            descriptor = json.loads(
                (descriptor_dir / file_name).read_text(encoding="utf-8")
            )
            model = descriptor["model"]
            source = model["source"]
            architecture = descriptor["architecture"]
            precision = descriptor["precision"]
            audit = {
                "frontier": {
                    "repository": source["frontier_repository"],
                    "revision": source["frontier_revision"],
                    "model": model["name"],
                    "model_config": {
                        "source_config_path": source[
                            "frontier_config_path"
                        ],
                        "source_config_sha256": source[
                            "frontier_config_sha256"
                        ],
                        "num_layers": architecture["num_layers"],
                        "num_q_heads": architecture[
                            "num_attention_heads"
                        ],
                        "num_kv_heads": architecture[
                            "num_key_value_heads"
                        ],
                        "embedding_dim": architecture["hidden_size"],
                        "mlp_hidden_dim": architecture[
                            "intermediate_size"
                        ],
                        "vocab_size": architecture["vocab_size"],
                        "use_gated_mlp": architecture["gated_mlp"],
                        "use_bias": architecture["mlp_bias"],
                        "use_qkv_bias": architecture["attention_bias"],
                        "tie_word_embeddings": architecture[
                            "tie_word_embeddings"
                        ],
                        "norm": 1,
                        "post_attn_norm": True,
                        "use_qk_norm": False,
                        "attn_output_gate": False,
                        "head_dim": 128,
                    },
                    "memory_precision": {
                        "selected_profile": precision,
                        "model_source_identity": {
                            "repository": source["model_repository"],
                            "revision": source["model_revision"],
                            "config_access": source["config_access"],
                        },
                    },
                    "parallelism": {
                        "pipeline": 1,
                        "data": 1,
                        "attention_tensor": 1,
                        "moe_tensor": 1,
                        "moe_expert": 1,
                    },
                }
            }
            values = _validate_model_descriptor(descriptor, audit)
            layout = _weight_layout(values, AddressAllocator(4096))
            self.assertEqual(
                layout["resident_parameters"],
                70_553_706_496,
            )
            self.assertEqual(
                layout["resident_bytes"],
                expected_ledger["resident_bytes"],
            )
            self.assertEqual(
                layout["storage_breakdown"],
                expected_ledger["storage_breakdown"],
            )
            self.assertEqual(
                layout["breakdown_bytes"],
                expected_ledger["breakdown_bytes"],
            )

    def test_single_request_batch_is_named_by_id_and_censused_separately(
        self,
    ) -> None:
        result = self.export(batch_start=2)
        self.assertEqual(result["selection"]["batch_ids"], [2])
        self.assertEqual(
            result["selection"]["selected_batch_census"],
            [
                {
                    "batch_id": 2,
                    "request_count": 1,
                    "request_ids": ["0"],
                    "scheduled_tokens": 1,
                    "request_phases": ["decode"],
                    "produces_output_token": True,
                    "frontier_stage_start_s": 2.0,
                    "frontier_stage_end_s": 3.0,
                }
            ],
        )

    def test_embedding_rows_are_hashed_independently_per_scheduled_token(
        self,
    ) -> None:
        self.export()
        object_map = json.loads(self.object_map.read_text(encoding="utf-8"))
        embedding = next(
            region
            for region in object_map["regions"]
            if region["name"] == "model.token_embedding"
        )
        row_bytes = 8 * 2
        actual_rows: list[int] = []
        for line in self.trace.read_text(encoding="utf-8").splitlines():
            if "label=embedding.request." not in line:
                continue
            fields = line.split()
            address = int(fields[0], 16)
            byte_count = int(fields[2])
            self.assertEqual(byte_count % row_bytes, 0)
            first_row = (address - embedding["begin"]) // row_bytes
            actual_rows.extend(
                range(first_row, first_row + byte_count // row_bytes)
            )

        ledger = self.audit_fixture.frontier / "frontier_stage_batch_ledger.jsonl"
        first_row = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
        expected_rows: list[int] = []
        for snapshot in first_row["hbfsim_memory_contract"]["requests"]:
            token_begin = snapshot["kv_tokens_before"]
            for token_position in range(
                token_begin,
                token_begin + snapshot["scheduled_tokens"],
            ):
                expected_rows.append(
                    _surrogate_embedding_row(
                        request_id=str(snapshot["request_id"]),
                        token_position=token_position,
                        model_name="fixture-model",
                        vocab_size=32,
                    )
                )
        self.assertEqual(actual_rows, expected_rows)
        self.assertGreater(len(set(actual_rows)), 1)

    def test_export_is_deterministic_when_republished_to_same_paths(self) -> None:
        first = self.export(max_batches=None)
        first_bytes = {
            path.name: path.read_bytes()
            for path in (
                self.trace,
                self.object_map,
                self.phase_map,
                self.manifest,
            )
        }
        second = self.export(max_batches=None)
        second_bytes = {
            path.name: path.read_bytes()
            for path in (
                self.trace,
                self.object_map,
                self.phase_map,
                self.manifest,
            )
        }
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(second["selection"]["is_full_replay"])

    def test_physical_block_id_reuses_the_same_kv_address(self) -> None:
        self.export(max_batches=None)
        block_zero_addresses = []
        for line in self.trace.read_text(encoding="utf-8").splitlines():
            if "kv.r.request.0.block.0.model_layer.0" not in line:
                continue
            block_zero_addresses.append(int(line.split()[0], 16))
        self.assertGreaterEqual(len(block_zero_addresses), 2)
        self.assertEqual(len(set(block_zero_addresses)), 1)

    def test_kv_layer_address_advances_by_exactly_one_layer_stride(self) -> None:
        self.export()
        addresses: dict[int, int] = {}
        for line in self.trace.read_text(encoding="utf-8").splitlines():
            if "label=kv.w.request.0.block.0.model_layer." not in line:
                continue
            fields = line.split()
            label = next(
                field for field in fields if field.startswith("label=")
            )
            layer = int(label.rsplit(".", 1)[1])
            addresses.setdefault(layer, int(fields[0], 16))

        self.assertEqual(set(addresses), {0, 1})
        self.assertEqual(addresses[1] - addresses[0], 16 * 16)

    def test_rejects_model_mismatch_without_publishing_outputs(self) -> None:
        model = json.loads(self.model.read_text(encoding="utf-8"))
        model["architecture"]["num_layers"] = 3
        self.model.write_text(json.dumps(model), encoding="utf-8")
        with self.assertRaisesRegex(ExportError, "model_config.num_layers"):
            self.export()
        for path in (self.trace, self.object_map, self.phase_map, self.manifest):
            self.assertFalse(path.exists())

    def test_rejects_precision_that_disagrees_with_frontier_model(self) -> None:
        model = json.loads(self.model.read_text(encoding="utf-8"))
        model["precision"]["kv_dtype"] = "float8_e4m3fn"
        model["precision"]["kv_bytes"] = 1
        self.model.write_text(json.dumps(model), encoding="utf-8")
        with self.assertRaisesRegex(ExportError, "precision.kv_dtype"):
            self.export()
        for path in (self.trace, self.object_map, self.phase_map, self.manifest):
            self.assertFalse(path.exists())

    def test_rejects_embedding_sharing_or_source_digest_mismatch(self) -> None:
        model = json.loads(self.model.read_text(encoding="utf-8"))
        model["architecture"]["tie_word_embeddings"] = True
        self.model.write_text(json.dumps(model), encoding="utf-8")
        with self.assertRaisesRegex(ExportError, "tie_word_embeddings"):
            self.export()
        for path in (self.trace, self.object_map, self.phase_map, self.manifest):
            self.assertFalse(path.exists())

        model["architecture"]["tie_word_embeddings"] = False
        model["model"]["source"]["frontier_config_sha256"] = "b" * 64
        self.model.write_text(json.dumps(model), encoding="utf-8")
        with self.assertRaisesRegex(ExportError, "source_config_sha256"):
            self.export()
        for path in (self.trace, self.object_map, self.phase_map, self.manifest):
            self.assertFalse(path.exists())

    def test_rejects_mutated_audited_artifact(self) -> None:
        ledger = self.audit_fixture.frontier / "frontier_stage_batch_ledger.jsonl"
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ExportError, "byte size|digest"):
            self.export()
        self.assertFalse(self.manifest.exists())

    def test_rejects_mutated_hybrid_residency_plan(
        self,
    ) -> None:
        audit = json.loads(self.audit.read_text(encoding="utf-8"))
        audit["residency_plan"]["hot_kv_blocks"] += 1
        self.audit.write_text(json.dumps(audit), encoding="utf-8")
        with self.assertRaisesRegex(
            ExportError,
            "hybrid-residency plan is invalid",
        ):
            self.export()
        for output in (
            self.trace,
            self.object_map,
            self.phase_map,
            self.manifest,
        ):
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
