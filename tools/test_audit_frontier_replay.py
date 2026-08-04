#!/usr/bin/env python3
"""Regression tests for audit_frontier_replay.py."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from audit_frontier_replay import AuditError, audit_replay  # noqa: E402
from frontier_memory_contract import (  # noqa: E402
    ALLOCATOR_STATE_SCHEMA_VERSION,
    MEMORY_CONTRACT_SCHEMA_VERSION,
    RESIDENCY_PLAN_SCHEMA_VERSION,
    allocator_state_sha256,
)
from frontier_hybrid_residency import (  # noqa: E402
    build_hybrid_residency_plan,
)
from prepare_qwen_bailian_workload import (  # noqa: E402
    REQUIRED_RUNTIME_BEHAVIORS,
    REQUIRED_WINDOWS,
    SELECTOR_ALGORITHMS,
    prepare_suite,
)
from verify_qwen_bailian_workload import verify_suite  # noqa: E402


class AuditFrontierReplayTest(unittest.TestCase):
    PHYSICAL_HBM_CAPACITY_BYTES = 96 * 1024**3
    RUNTIME_OVERHEAD_BYTES = 77_309_405_340

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.request_source = self.root / "source.jsonl"
        self.request_suite_config = self.root / "suite-config.json"
        self.request_suite_dir = self.root / "request-suite"
        self.request_suite_verification = self.root / "suite-verification.json"
        self._write_request_suite()
        self.request_csv = self.request_suite_dir / "steady.frontier.csv"
        self.request_manifest = (
            self.request_suite_dir / "steady.adapter-manifest.json"
        )
        self.frontier = self.root / "frontier"
        self.frontier.mkdir()
        self.integration_receipt = self.root / "frontier-integration-receipt.json"
        self.output = self.root / "audit.json"
        self._write_valid_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_json(self, path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_request_suite(self) -> None:
        rows = [
            {
                "chat_id": 10,
                "parent_chat_id": -1,
                "timestamp": 0,
                "input_length": 17,
                "output_length": 3,
                "type": "thinking",
                "turn": 1,
                "hash_ids": [100, 101],
            },
            {
                "chat_id": 11,
                "parent_chat_id": 10,
                "timestamp": 0.5,
                "input_length": 32,
                "output_length": 2,
                "type": "thinking",
                "turn": 2,
                "hash_ids": [100, 102],
            },
        ]
        self.request_source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        source_bytes = self.request_source.read_bytes()
        expected = {
            "start_index": 0,
            "end_index_exclusive": 2,
            "source_start_timestamp_s": "0",
            "source_end_timestamp_s": "0.5",
            "duration_s": "0.5",
            "prefill_tokens": 49,
            "decode_tokens": 5,
            "total_tokens": 54,
            "max_prefill_tokens": 32,
            "max_decode_tokens": 3,
            "max_total_tokens": 34,
        }
        self._write_json(
            self.request_suite_config,
            {
                "schema": {
                    "name": "hbfsim.qwen_bailian_frontier_suite_config",
                    "version": 1,
                },
                "suite_id": "fixture-suite-v1",
                "intended_use": {
                    "role": "unit_test",
                    "paper_result_eligible_by_itself": False,
                    "claims_boundary": "unit test only",
                },
                "source": {
                    "repository": "https://example.invalid/qwen",
                    "revision": "0" * 40,
                    "license": "test-only",
                    "artifact": "source.jsonl",
                    "sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "bytes": len(source_bytes),
                    "records": 2,
                    "hash_block_tokens": 16,
                    "statistics": {
                        "duration_s": "0.5",
                        "prefill_tokens": 49,
                        "decode_tokens": 5,
                        "max_prefill_tokens": 32,
                        "max_decode_tokens": 3,
                    },
                },
                "frontier_workload": {
                    "model": {
                        "identity": "example/fixture-model",
                        "frontier_name": "fixture-model",
                    },
                    "primary_precision_profile": "fp16-ci-smoke",
                    "sensitivity_precision_profiles": ["bf16-ci-smoke"],
                    "required_runtime_behaviors": list(
                        REQUIRED_RUNTIME_BEHAVIORS
                    ),
                },
                "window_policy": {
                    "request_count": 2,
                    "required_windows": list(REQUIRED_WINDOWS),
                    "selectors": {
                        window_id: {
                            "algorithm": SELECTOR_ALGORITHMS[window_id],
                            "expected": expected,
                        }
                        for window_id in REQUIRED_WINDOWS
                    },
                },
            },
        )
        prepare_suite(
            source=self.request_source,
            suite_config=self.request_suite_config,
            output_dir=self.request_suite_dir,
        )
        verify_suite(
            source=self.request_source,
            suite_config=self.request_suite_config,
            suite_dir=self.request_suite_dir,
            output=self.request_suite_verification,
        )

    def _write_valid_fixture(self) -> None:
        self._write_json(
            self.frontier / "config.json",
            {
                "sys_arch": "co-location",
                "request_generator_config": {
                    "name": "trace_replay",
                    "trace_file": str(self.request_csv.resolve()),
                    "prefill_scale_factor": 1.0,
                    "decode_scale_factor": 1.0,
                    "time_scale_factor": 1.0,
                },
                "metrics_config": {"store_frontier_stage_batch_ledger": True},
                "cluster_config": {
                    "cluster_type": 1,
                    "num_replicas": 1,
                    "replica_scheduler_config": {
                        "name": "vllm_v1",
                        "block_size": 16,
                        "enable_prefix_caching": True,
                        "prefix_caching_hash_algo": "builtin",
                        "enable_preemption": False,
                        "num_preallocate_tokens": 0,
                        "enable_chunked_prefill": True,
                        "max_tokens_in_batch": 32,
                        "batch_size_cap": 2,
                        "num_blocks": 0,
                        "num_blocks_mode": "hybrid_residency",
                        "gpu_memory_utilization": None,
                        "non_kv_cache_overhead_bytes": (
                            self.RUNTIME_OVERHEAD_BYTES
                        ),
                        "hybrid_physical_hbm_capacity_bytes": (
                            self.PHYSICAL_HBM_CAPACITY_BYTES
                        ),
                        "hybrid_capacity_pressure_target": 0.75,
                        "runtime_weights_memory_source": "param_counter",
                        "enable_runtime_non_kv_cache_overhead_profiling": False,
                    },
                    "execution_time_predictor_config": {
                        "enable_dummy_mode": True
                    },
                    "replica_config": {
                        "memory_precision_profile": "fp16-ci-smoke",
                        "memory_margin_fraction": 0.0,
                        "num_pipeline_stages": 1,
                        "data_parallel_size": 1,
                        "attn_tensor_parallel_size": 1,
                        "moe_tensor_parallel_size": 1,
                        "moe_expert_parallel_size": 1,
                        "model_name": "fixture-model",
                        "device": "fixture-device",
                        "device_config": {
                            "total_memory_gb": 5968 / (1024**3),
                        },
                        "model_config": {
                            "num_layers": 2,
                            "num_q_heads": 2,
                            "num_kv_heads": 1,
                            "head_dim": 4,
                            "embedding_dim": 8,
                            "mlp_hidden_dim": 16,
                            "vocab_size": 32,
                            "use_gated_mlp": True,
                            "use_bias": False,
                            "use_qkv_bias": False,
                            "norm": 1,
                            "post_attn_norm": True,
                            "use_qk_norm": False,
                            "attn_output_gate": False,
                            "tie_word_embeddings": False,
                            "source_config_path": (
                                "data/config/models/fixture-model.json"
                            ),
                            "source_config_sha256": "a" * 64,
                            "is_moe": False,
                            "max_position_embeddings": 4096,
                            "torch_dtype": "float16",
                            "quantization_config": {"quant_method": None},
                            "memory_precision_profiles": {
                                "fp16-ci-smoke": {
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
                                "bf16-ci-smoke": {
                                    "profile_id": "bf16-ci-smoke",
                                    "matrix_weight_dtype": "bfloat16",
                                    "matrix_weight_bytes": 2,
                                    "non_matrix_weight_dtype": "bfloat16",
                                    "non_matrix_weight_bytes": 2,
                                    "kv_dtype": "bfloat16",
                                    "kv_bytes": 2,
                                    "activation_dtype": "bfloat16",
                                    "activation_bytes": 2,
                                    "quantization_scheme": "none",
                                    "scale_dtype": None,
                                    "scale_bytes": 0,
                                    "zero_point_bytes": 0,
                                    "claim_scope": (
                                        "memory_storage_and_traffic_only"
                                    ),
                                },
                            },
                            "default_memory_precision_profile": (
                                "fp16-ci-smoke"
                            ),
                            "model_source_identity": {
                                "repository": "https://example.invalid/model",
                                "revision": "0" * 40,
                                "config_access": "unit-test",
                            },
                        },
                        "speculative_decoding_config": {"enabled": False},
                    },
                },
            },
        )
        columns = [
            "Request Id",
            "request_num_tokens",
            "request_num_prefill_tokens",
            "request_num_decode_tokens",
            "request_cached_prefill_tokens",
            "request_prefix_cache_query_blocks",
            "request_prefix_cache_hit_blocks",
        ]
        with (self.frontier / "request_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerow(
                {
                    "Request Id": "0",
                    "request_num_tokens": "20",
                    "request_num_prefill_tokens": "17",
                    "request_num_decode_tokens": "3",
                    "request_cached_prefill_tokens": "0",
                    "request_prefix_cache_query_blocks": "1",
                    "request_prefix_cache_hit_blocks": "0",
                }
            )
            writer.writerow(
                {
                    "Request Id": "1",
                    "request_num_tokens": "34",
                    "request_num_prefill_tokens": "32",
                    "request_num_decode_tokens": "2",
                    "request_cached_prefill_tokens": "16",
                    "request_prefix_cache_query_blocks": "2",
                    "request_prefix_cache_hit_blocks": "1",
                }
            )
        stream_id = "monolithic:replica-0:dp-0"
        lifecycle = [
            {
                "event": "prefix_lookup",
                "request_id": "0",
                "requested_block_hashes": [100],
                "matched_blocks": [],
                "hit_tokens": 0,
            },
            {
                "event": "prefix_admission",
                "request_id": "0",
                "admitted_blocks": [],
                "admitted_tokens": 0,
            },
            {
                "event": "allocate",
                "request_id": "0",
                "block_id": 0,
                "ref_count_before": 0,
                "ref_count_after": 1,
            },
            {
                "event": "cache_assign",
                "request_id": "0",
                "block_id": 0,
                "block_hash": 100,
            },
            {
                "event": "prefix_lookup",
                "request_id": "1",
                "requested_block_hashes": [100, 102],
                "matched_blocks": [
                    {"block_id": 0, "block_hash": 100, "ref_count": 1}
                ],
                "hit_tokens": 16,
            },
            {
                "event": "prefix_admission",
                "request_id": "1",
                "admitted_blocks": [
                    {"block_id": 0, "block_hash": 100, "ref_count": 1}
                ],
                "admitted_tokens": 16,
            },
            {
                "event": "touch",
                "request_id": "1",
                "blocks": [
                    {
                        "block_id": 0,
                        "block_hash": 100,
                        "ref_count_before": 1,
                        "ref_count_after": 2,
                    }
                ],
            },
            {
                "event": "allocate",
                "request_id": "1",
                "block_id": 1,
                "ref_count_before": 0,
                "ref_count_after": 1,
            },
            {
                "event": "cache_assign",
                "request_id": "1",
                "block_id": 1,
                "block_hash": 102,
            },
            {
                "event": "allocate",
                "request_id": "0",
                "block_id": 2,
                "ref_count_before": 0,
                "ref_count_after": 1,
            },
            {
                "event": "allocate",
                "request_id": "1",
                "block_id": 3,
                "ref_count_before": 0,
                "ref_count_after": 1,
            },
            {
                "event": "release",
                "request_id": "1",
                "blocks": [
                    {
                        "block_id": 3,
                        "block_hash": None,
                        "ref_count_before": 1,
                        "ref_count_after": 0,
                        "became_free": True,
                    },
                    {
                        "block_id": 1,
                        "block_hash": 102,
                        "ref_count_before": 1,
                        "ref_count_after": 0,
                        "became_free": True,
                    },
                    {
                        "block_id": 0,
                        "block_hash": 100,
                        "ref_count_before": 2,
                        "ref_count_after": 1,
                        "became_free": False,
                    },
                ],
            },
            {
                "event": "release",
                "request_id": "0",
                "blocks": [
                    {
                        "block_id": 2,
                        "block_hash": None,
                        "ref_count_before": 1,
                        "ref_count_after": 0,
                        "became_free": True,
                    },
                    {
                        "block_id": 0,
                        "block_hash": 100,
                        "ref_count_before": 1,
                        "ref_count_after": 0,
                        "became_free": True,
                    },
                ],
            },
        ]
        event_times = [0.0] * 9 + [1.0] * 2 + [2.0, 4.0]
        event_batches = [0] * 9 + [1] * 3 + [3]
        for index, event in enumerate(lifecycle):
            event.update(
                {
                    "schema_version": 1,
                    "stream_id": stream_id,
                    "event_index": index,
                    "event_id": f"{stream_id}:{index:012d}",
                    "event_time": event_times[index],
                    "observed_after_batch_id": event_batches[index],
                    "cluster_type": "MONOLITHIC",
                    "replica_id": 0,
                    "dp_id": 0,
                }
            )
        (self.frontier / "frontier_kv_block_lifecycle.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in lifecycle),
            encoding="utf-8",
        )
        self._write_json(
            self.frontier / "frontier_hbfsim_integration.json",
            {
                "schema": {
                    "name": "hbfsim.frontier_integration_runtime",
                    "version": 1,
                },
                "integration_id": "hbfsim.frontier-hybrid-residency.v8",
                "upstream_revision": (
                    "a4b22df8211864bf229258ecdfbe680f048f2d77"
                ),
                "contracts": {
                    "memory_contract_schema_version": 4,
                    "allocator_state_schema_version": 1,
                    "residency_plan_schema_version": (
                        RESIDENCY_PLAN_SCHEMA_VERSION
                    ),
                    "kv_lifecycle_schema_version": 1,
                },
            },
        )
        bundle = json.loads(
            (
                TOOLS.parent / "integrations/frontier/manifest.json"
            ).read_text(encoding="utf-8")
        )
        self._write_json(
            self.integration_receipt,
            {
                "schema": {
                    "name": "hbfsim.frontier_integration_receipt",
                    "version": 1,
                },
                "result": "pass",
                "integration_id": bundle["integration_id"],
                "upstream": bundle["upstream"],
                "patch": bundle["patch"],
                "contracts": bundle["contracts"],
                "staged_diff_sha256": bundle["patch"]["sha256"],
                "post_apply_sha256": bundle["post_apply_sha256"],
            },
        )

        def contract(
            *,
            batch_id: int,
            captured_at: float,
            cursor: int,
            snapshots: list[dict],
        ) -> dict:
            residency_plan = build_hybrid_residency_plan(
                physical_hbm_capacity_bytes=(
                    self.PHYSICAL_HBM_CAPACITY_BYTES
                ),
                target_pressure=0.75,
                immutable_weight_backing_bytes=3408,
                runtime_overhead_bytes=self.RUNTIME_OVERHEAD_BYTES,
                active_weight_buffer_bytes_per_slot=1184,
                kv_block_size_tokens=16,
                kv_page_bytes_per_layer=256,
                num_layers=2,
            )
            return {
                "schema_version": MEMORY_CONTRACT_SCHEMA_VERSION,
                "contract": "frontier-hbfsim-memory-object",
                "captured_at": captured_at,
                "batch_id": batch_id,
                "schedule_iteration_id": batch_id,
                "cluster_type": "MONOLITHIC",
                "replica_id": 0,
                "dp_id": 0,
                "kv_cache": {
                    "block_size_tokens": 16,
                    "num_gpu_blocks": 5,
                    "enable_prefix_caching": True,
                    "caching_hash_algo": "builtin",
                    "num_preallocate_tokens": 0,
                    "lifecycle_stream_id": stream_id,
                    "no_preemption_admission_policy": (
                        "full_request_kv_reservation_v1"
                    ),
                },
                "residency_plan": residency_plan,
                "allocator_state": {
                    "schema_version": ALLOCATOR_STATE_SCHEMA_VERSION,
                    "digest": "sha256",
                    "canonical_json": "sorted-keys-compact",
                    "block_order": "request-ownership-order",
                },
                "requests": snapshots,
                "lifecycle_event_cursor": cursor,
            }

        def snapshot(
            *,
            request_id: str,
            phase: str,
            context: int,
            scheduled: int,
            prefill: int,
            decode: int,
            cached: int,
            blocks: list[dict],
        ) -> dict:
            kv_before = context if phase == "prefill" else context - 1
            return {
                "request_id": request_id,
                "runtime_epoch": 0,
                "phase": phase,
                "context_tokens_before": context,
                "scheduled_tokens": scheduled,
                "scheduler_token_frontier_after": context + scheduled,
                "kv_tokens_before": kv_before,
                "kv_tokens_after": kv_before + scheduled,
                "produces_output_token": (
                    phase == "decode"
                    or (
                        decode > 0
                        and context + scheduled == prefill
                    )
                ),
                "num_prefill_tokens": prefill,
                "num_decode_tokens": decode,
                "cached_prefill_tokens": cached,
                "allocator_required_blocks": (
                    context + scheduled + 15
                )
                // 16,
                "allocated_block_count": len(blocks),
                "allocator_state_sha256": allocator_state_sha256(
                    request_id,
                    blocks,
                ),
            }

        block0_ref2 = {"block_id": 0, "block_hash": 100, "ref_count": 2}
        block0_ref1 = {"block_id": 0, "block_hash": 100, "ref_count": 1}
        block1 = {"block_id": 1, "block_hash": 102, "ref_count": 1}
        block2 = {"block_id": 2, "block_hash": None, "ref_count": 1}
        block3 = {"block_id": 3, "block_hash": None, "ref_count": 1}
        ledger = [
            {
                "batch_id": 0,
                "cluster_type": "MONOLITHIC",
                "replica_id": 0,
                "dp_id": 0,
                "stage_id": 0,
                "request_ids": ["0", "1"],
                "request_num_tokens": [16, 16],
                "stage_start_ts": 0.0,
                "stage_end_ts": 1.0,
                "hbfsim_memory_contract": contract(
                    batch_id=0,
                    captured_at=0.0,
                    cursor=9,
                    snapshots=[
                        snapshot(
                            request_id="0",
                            phase="prefill",
                            context=0,
                            scheduled=16,
                            prefill=17,
                            decode=3,
                            cached=0,
                            blocks=[block0_ref2],
                        ),
                        snapshot(
                            request_id="1",
                            phase="prefill",
                            context=16,
                            scheduled=16,
                            prefill=32,
                            decode=2,
                            cached=16,
                            blocks=[block0_ref2, block1],
                        ),
                    ],
                ),
            },
            {
                "batch_id": 1,
                "cluster_type": "MONOLITHIC",
                "replica_id": 0,
                "dp_id": 0,
                "stage_id": 0,
                "request_ids": ["0", "1"],
                "request_num_tokens": [1, 1],
                "stage_start_ts": 1.0,
                "stage_end_ts": 2.0,
                "hbfsim_memory_contract": contract(
                    batch_id=1,
                    captured_at=1.0,
                    cursor=11,
                    snapshots=[
                        snapshot(
                            request_id="0",
                            phase="prefill",
                            context=16,
                            scheduled=1,
                            prefill=17,
                            decode=3,
                            cached=0,
                            blocks=[block0_ref2, block2],
                        ),
                        snapshot(
                            request_id="1",
                            phase="decode",
                            context=33,
                            scheduled=1,
                            prefill=32,
                            decode=2,
                            cached=16,
                            blocks=[block0_ref2, block1, block3],
                        ),
                    ],
                ),
            },
            {
                "batch_id": 2,
                "cluster_type": "MONOLITHIC",
                "replica_id": 0,
                "dp_id": 0,
                "stage_id": 0,
                "request_ids": ["0"],
                "request_num_tokens": [1],
                "stage_start_ts": 2.0,
                "stage_end_ts": 3.0,
                "hbfsim_memory_contract": contract(
                    batch_id=2,
                    captured_at=2.0,
                    cursor=12,
                    snapshots=[
                        snapshot(
                            request_id="0",
                            phase="decode",
                            context=18,
                            scheduled=1,
                            prefill=17,
                            decode=3,
                            cached=0,
                            blocks=[block0_ref1, block2],
                        )
                    ],
                ),
            },
            {
                "batch_id": 3,
                "cluster_type": "MONOLITHIC",
                "replica_id": 0,
                "dp_id": 0,
                "stage_id": 0,
                "request_ids": ["0"],
                "request_num_tokens": [1],
                "stage_start_ts": 3.0,
                "stage_end_ts": 4.0,
                "hbfsim_memory_contract": contract(
                    batch_id=3,
                    captured_at=3.0,
                    cursor=12,
                    snapshots=[
                        snapshot(
                            request_id="0",
                            phase="decode",
                            context=19,
                            scheduled=1,
                            prefill=17,
                            decode=3,
                            cached=0,
                            blocks=[block0_ref1, block2],
                        )
                    ],
                ),
            },
        ]
        (self.frontier / "frontier_stage_batch_ledger.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8"
        )
        self._write_json(
            self.frontier / "system_metrics.json",
            {
                "simulation_metadata": {
                    "total_requests": 2,
                    "completed_requests": 2,
                },
                "throughput_metrics": {
                    "total_tokens_processed": 54,
                    "total_decode_tokens_generated": 5,
                },
                "prefix_cache_statistics": {
                    "requests": 2,
                    "total_cached_prefill_tokens": 16,
                    "total_query_blocks": 3,
                    "total_hit_blocks": 1,
                },
                "preemption_statistics": {
                    "total_preemption_events": 0,
                    "total_preempted_requests": 0,
                },
                "spec_decode_statistics": {"total_iterations": 0},
                "model_weight_memory": {
                    "MONOLITHIC": {
                        "total_parameters": 1704,
                        "total_memory_bytes": 3408,
                        "memory_precision_profile": {
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
                        "storage_breakdown": {
                            "matrix_payload_bytes": 3328,
                            "non_matrix_payload_bytes": 80,
                            "quantization_metadata_bytes": 0,
                        },
                        "breakdown": {
                            "attention_memory_bytes": 768,
                            "ffn_memory_bytes": 1536,
                            "normalization_memory_bytes": 64,
                            "pipeline_boundary_and_auxiliary_memory_bytes": 1040,
                        },
                        "memory_ledger": {
                            "profile": {
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
                            "total_parameters": 1704,
                            "total_memory_bytes": 3408,
                            "matrix_payload_bytes": 3328,
                            "non_matrix_payload_bytes": 80,
                            "quantization_metadata_bytes": 0,
                            "categories": {
                                "attention": {
                                    "parameters": 384,
                                    "matrix_parameters": 384,
                                    "non_matrix_parameters": 0,
                                    "matrix_payload_bytes": 768,
                                    "non_matrix_payload_bytes": 0,
                                    "scale_elements": 0,
                                    "quantization_metadata_bytes": 0,
                                    "memory_bytes": 768,
                                },
                                "ffn": {
                                    "parameters": 768,
                                    "matrix_parameters": 768,
                                    "non_matrix_parameters": 0,
                                    "matrix_payload_bytes": 1536,
                                    "non_matrix_payload_bytes": 0,
                                    "scale_elements": 0,
                                    "quantization_metadata_bytes": 0,
                                    "memory_bytes": 1536,
                                },
                                "normalization": {
                                    "parameters": 32,
                                    "matrix_parameters": 0,
                                    "non_matrix_parameters": 32,
                                    "matrix_payload_bytes": 0,
                                    "non_matrix_payload_bytes": 64,
                                    "scale_elements": 0,
                                    "quantization_metadata_bytes": 0,
                                    "memory_bytes": 64,
                                },
                                "pipeline_boundary_and_auxiliary": {
                                    "parameters": 520,
                                    "matrix_parameters": 512,
                                    "non_matrix_parameters": 8,
                                    "matrix_payload_bytes": 1024,
                                    "non_matrix_payload_bytes": 16,
                                    "scale_elements": 0,
                                    "quantization_metadata_bytes": 0,
                                    "memory_bytes": 1040,
                                },
                            },
                        },
                        "weight_streaming_ledger": {
                            "num_transformer_layers": 2,
                            "objects": {
                                "embedding": {
                                    "parameters": 256,
                                    "matrix_parameters": 256,
                                    "non_matrix_parameters": 0,
                                    "matrix_payload_bytes": 512,
                                    "non_matrix_payload_bytes": 0,
                                    "scale_elements": 0,
                                    "quantization_metadata_bytes": 0,
                                    "memory_bytes": 512,
                                },
                                "transformer_layer": {
                                    "parameters": 592,
                                    "matrix_parameters": 576,
                                    "non_matrix_parameters": 16,
                                    "matrix_payload_bytes": 1152,
                                    "non_matrix_payload_bytes": 32,
                                    "scale_elements": 0,
                                    "quantization_metadata_bytes": 0,
                                    "memory_bytes": 1184,
                                },
                                "final_norm": {
                                    "parameters": 8,
                                    "matrix_parameters": 0,
                                    "non_matrix_parameters": 8,
                                    "matrix_payload_bytes": 0,
                                    "non_matrix_payload_bytes": 16,
                                    "scale_elements": 0,
                                    "quantization_metadata_bytes": 0,
                                    "memory_bytes": 16,
                                },
                                "output_head": {
                                    "parameters": 256,
                                    "matrix_parameters": 256,
                                    "non_matrix_parameters": 0,
                                    "matrix_payload_bytes": 512,
                                    "non_matrix_payload_bytes": 0,
                                    "scale_elements": 0,
                                    "quantization_metadata_bytes": 0,
                                    "memory_bytes": 512,
                                },
                            },
                            "active_buffer_bytes_per_slot": 1184,
                            "immutable_weight_backing_bytes": 3408,
                            "total_parameters": 1704,
                        },
                    }
                },
            },
        )

    def audit(self) -> dict:
        return audit_replay(
            request_csv=self.request_csv,
            request_manifest=self.request_manifest,
            request_suite_verification=self.request_suite_verification,
            frontier_output_dir=self.frontier,
            frontier_revision="a4b22df8211864bf229258ecdfbe680f048f2d77",
            frontier_integration_receipt=self.integration_receipt,
            output=self.output,
        )

    def test_valid_replay_conserves_tokens_and_marks_dummy_timing(self) -> None:
        result = self.audit()
        self.assertEqual(result["result"], "pass")
        self.assertEqual(
            result["schema"],
            {"name": "hbfsim.frontier_replay_audit", "version": 7},
        )
        self.assertEqual(result["accounting"]["scheduled_tokens"], 36)
        self.assertEqual(result["accounting"]["cached_prefill_tokens"], 16)
        self.assertEqual(result["accounting"]["prefill_generated_first_tokens"], 2)
        self.assertEqual(result["requests"]["total_tokens"], 54)
        self.assertEqual(
            result["accounting"]["prefix_hit_ratio"],
            1 / 3,
        )
        self.assertEqual(
            result["frontier"]["integration"]["integration_id"],
            "hbfsim.frontier-hybrid-residency.v8",
        )
        self.assertFalse(
            result["eligibility"]["frontier_timing_usable_for_performance"]
        )
        self.assertEqual(
            result["eligibility"]["timing_blocker"],
            "dummy execution-time predictor",
        )
        self.assertFalse(
            result["eligibility"]["eligible_for_ttft_tpot_slo_claims"]
        )
        self.assertEqual(
            result["kv_lifecycle"]["no_preemption_admission"],
            {
                "policy": "full_request_kv_reservation_v1",
                "verified_checkpoints": 4,
                "max_reserved_blocks": 5,
                "max_referenced_physical_blocks": 4,
                "capacity_blocks": 5,
            },
        )
        self.assertEqual(result["model_memory"]["total_memory_bytes"], 3408)
        self.assertEqual(
            result["residency_plan"]["num_logical_kv_blocks"],
            5,
        )
        self.assertEqual(json.loads(self.output.read_text()), result)

    def test_rejects_runtime_integration_identity_mismatch(self) -> None:
        path = self.frontier / "frontier_hbfsim_integration.json"
        identity = json.loads(path.read_text(encoding="utf-8"))
        identity["integration_id"] = "untracked-local-patch"
        self._write_json(path, identity)

        with self.assertRaisesRegex(AuditError, "runtime integration_id"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_integration_receipt_with_different_staged_diff(self) -> None:
        receipt = json.loads(
            self.integration_receipt.read_text(encoding="utf-8")
        )
        receipt["staged_diff_sha256"] = "0" * 64
        self._write_json(self.integration_receipt, receipt)

        with self.assertRaisesRegex(AuditError, "staged diff digest"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_token_conservation_failure_without_publishing(self) -> None:
        ledger_path = self.frontier / "frontier_stage_batch_ledger.jsonl"
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        rows[3]["request_num_tokens"][0] = 2
        ledger_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            AuditError, "scheduled tokens|scheduler frontier|token conservation"
        ):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_scaled_frontier_input(self) -> None:
        path = self.frontier / "config.json"
        config = json.loads(path.read_text())
        config["request_generator_config"]["decode_scale_factor"] = 0.5
        self._write_json(path, config)
        with self.assertRaisesRegex(AuditError, "decode_scale_factor"):
            self.audit()

    def test_rejects_nonfinite_json_number(self) -> None:
        path = self.frontier / "config.json"
        config = json.loads(path.read_text())
        config["cluster_config"]["replica_config"]["model_config"][
            "max_position_embeddings"
        ] = float("nan")
        self._write_json(path, config)
        with self.assertRaisesRegex(AuditError, "non-finite JSON number"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_request_digest_mismatch(self) -> None:
        with self.request_csv.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(AuditError, "output.sha256"):
            self.audit()

    def test_rejects_retired_adapter_schema_v1(self) -> None:
        manifest = json.loads(self.request_manifest.read_text(encoding="utf-8"))
        manifest["schema"]["version"] = 1
        self._write_json(self.request_manifest, manifest)
        with self.assertRaisesRegex(AuditError, "adapter schema.version"):
            self.audit()

    def test_rejects_request_suite_source_drift(self) -> None:
        with self.request_source.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(AuditError, "request-suite source.sha256"):
            self.audit()

    def test_rejects_incomplete_request_suite_verification(self) -> None:
        verification = json.loads(
            self.request_suite_verification.read_text(encoding="utf-8")
        )
        del verification["windows"]["burst"]
        self._write_json(self.request_suite_verification, verification)
        with self.assertRaisesRegex(AuditError, "cover all three required windows"):
            self.audit()

    def test_rejects_frontier_model_different_from_request_suite(self) -> None:
        path = self.frontier / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["cluster_config"]["replica_config"]["model_name"] = "other-model"
        self._write_json(path, config)
        with self.assertRaisesRegex(AuditError, "Frontier/request-suite model"):
            self.audit()

    def test_rejects_unsupported_pipeline_parallel_replay(self) -> None:
        path = self.frontier / "config.json"
        config = json.loads(path.read_text())
        config["cluster_config"]["replica_config"]["num_pipeline_stages"] = 2
        self._write_json(path, config)
        with self.assertRaisesRegex(AuditError, "Frontier PP"):
            self.audit()

    def test_rejects_implicit_embedding_sharing_semantics(self) -> None:
        path = self.frontier / "config.json"
        config = json.loads(path.read_text())
        del config["cluster_config"]["replica_config"]["model_config"][
            "tie_word_embeddings"
        ]
        self._write_json(path, config)
        with self.assertRaisesRegex(AuditError, "tie_word_embeddings"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_incorrect_output_boundary(self) -> None:
        path = self.frontier / "frontier_stage_batch_ledger.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["hbfsim_memory_contract"]["requests"][0][
            "produces_output_token"
        ] = True
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(AuditError, "output boundary"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_preemption_until_recomputation_is_accounted(self) -> None:
        path = self.frontier / "system_metrics.json"
        metrics = json.loads(path.read_text())
        metrics["preemption_statistics"]["total_preemption_events"] = 1
        metrics["preemption_statistics"]["total_preempted_requests"] = 1
        self._write_json(path, metrics)
        with self.assertRaisesRegex(AuditError, "preemption events"):
            self.audit()

    def test_rejects_model_memory_ledger_drift(self) -> None:
        path = self.frontier / "system_metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        monolithic = metrics["model_weight_memory"]["MONOLITHIC"]
        monolithic["total_parameters"] -= 520
        monolithic["total_memory_bytes"] -= 1040
        monolithic["breakdown"][
            "pipeline_boundary_and_auxiliary_memory_bytes"
        ] = 0
        self._write_json(path, metrics)
        with self.assertRaisesRegex(
            AuditError,
            "model weight total parameters",
        ):
            self.audit()

    def test_rejects_missing_full_request_reservation_policy(self) -> None:
        path = self.frontier / "frontier_stage_batch_ledger.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        del rows[0]["hbfsim_memory_contract"]["kv_cache"][
            "no_preemption_admission_policy"
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            AuditError,
            "no-preemption admission policy",
        ):
            self.audit()

    def test_rejects_full_request_reservation_overcommit(self) -> None:
        config_path = self.frontier / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        scheduler = config["cluster_config"]["replica_scheduler_config"]
        scheduler["non_kv_cache_overhead_bytes"] = (
            self.RUNTIME_OVERHEAD_BYTES + 516
        )
        self._write_json(config_path, config)

        four_block_plan = build_hybrid_residency_plan(
            physical_hbm_capacity_bytes=self.PHYSICAL_HBM_CAPACITY_BYTES,
            target_pressure=0.75,
            immutable_weight_backing_bytes=3408,
            runtime_overhead_bytes=self.RUNTIME_OVERHEAD_BYTES + 516,
            active_weight_buffer_bytes_per_slot=1184,
            kv_block_size_tokens=16,
            kv_page_bytes_per_layer=256,
            num_layers=2,
        )
        self.assertEqual(four_block_plan["num_logical_kv_blocks"], 4)

        path = self.frontier / "frontier_stage_batch_ledger.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            contract = row["hbfsim_memory_contract"]
            contract["kv_cache"]["num_gpu_blocks"] = 4
            contract["residency_plan"] = four_block_plan
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            AuditError,
            "full_request_kv_reservation_v1",
        ):
            self.audit()

    def test_rejects_mutated_lifecycle_refcount(self) -> None:
        lifecycle_path = self.frontier / "frontier_kv_block_lifecycle.jsonl"
        events = [
            json.loads(line) for line in lifecycle_path.read_text().splitlines()
        ]
        events[6]["blocks"][0]["ref_count_after"] = 3
        lifecycle_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(AuditError, "ref_count_after"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_lifecycle_hashes_not_bound_to_request_trace(self) -> None:
        lifecycle_path = self.frontier / "frontier_kv_block_lifecycle.jsonl"
        events = [
            json.loads(line) for line in lifecycle_path.read_text().splitlines()
        ]
        events[0]["requested_block_hashes"][0] = 999
        lifecycle_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(AuditError, "requested .*hashes"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_partial_prompt_block_as_prefix_cache_query(self) -> None:
        lifecycle_path = self.frontier / "frontier_kv_block_lifecycle.jsonl"
        events = [
            json.loads(line) for line in lifecycle_path.read_text().splitlines()
        ]
        events[0]["requested_block_hashes"].append(101)
        lifecycle_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(AuditError, "complete-block hashes"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_allocator_digest_not_matching_replayed_state(self) -> None:
        ledger_path = self.frontier / "frontier_stage_batch_ledger.jsonl"
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        rows[0]["hbfsim_memory_contract"]["requests"][0][
            "allocator_state_sha256"
        ] = "0" * 64
        ledger_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(AuditError, "allocator-state digest"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_legacy_embedded_lifecycle_copy(self) -> None:
        ledger_path = self.frontier / "frontier_stage_batch_ledger.jsonl"
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        rows[0]["hbfsim_memory_contract"]["lifecycle_events"] = []
        ledger_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(AuditError, "embeds lifecycle_events"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_rejects_partial_release_that_would_leave_ownership_live(self) -> None:
        lifecycle_path = self.frontier / "frontier_kv_block_lifecycle.jsonl"
        events = [
            json.loads(line) for line in lifecycle_path.read_text().splitlines()
        ]
        events[-1]["blocks"] = [
            block
            for block in events[-1]["blocks"]
            if block["block_id"] != 0
        ]
        lifecycle_path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(AuditError, "release set"):
            self.audit()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
