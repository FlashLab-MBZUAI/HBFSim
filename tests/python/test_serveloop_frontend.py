#!/usr/bin/env python3
"""Exact contracts for ServeLoop: mixed iterations, paged KV, placement, timing."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import hbserve
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SERVE = Path(hbserve.__file__).resolve().parents[1]
MODEL_ROOT = SERVE / "models"

from hbfsim_client.transaction_protocol import HbfGeometry  # noqa: E402
from hbfsim_client.simulation_session import ResolvedSystemConfig  # noqa: E402
from hbserve.catalog import convert_descriptor  # noqa: E402
from hbserve.__main__ import main as frontend_main
from hbserve.capabilities import current_capabilities
from hbserve.compiler import HBServeCompiler  # noqa: E402
from hbserve.contracts import (  # noqa: E402
    BatchSlice,
    LayerSpec,
    LinearTimingProvider,
    MemoryOnlyTimingProvider,
    ModelSpec,
    RequestSpec,
    RequestTrace,
    RooflineTimingProvider,
    RouterDecision,
    RouterTrace,
    ScheduledBatch,
    SchedulerPolicy,
    HBServeError,
    TraceProvenance,
    canonical_sha256,
)
from hbserve.engine import (  # noqa: E402
    BatchExecution,
    HBServeEngine,
)
from hbserve.hbfsim import HbfSimExecutor  # noqa: E402
from hbserve.io import (  # noqa: E402
    load_json_object,
    load_models,
    load_placement,
    load_request_trace,
    load_router,
    load_run_config,
    load_synthetic_request_config,
    create_run_directory,
    write_json_atomic,
)
from hbserve.placement import (  # noqa: E402
    KvPlacement,
    PlacementSpec,
    HBServePlacement,
)
from hbserve.run import (  # noqa: E402
    default_run_config,
    derive_placement,
    run_experiment,
)
from hbserve.synthetic import (  # noqa: E402
    HotsetZipfRouter,
    SyntheticRequestConfig,
    generate_requests,
)


def _dense(model_id: str = "dense", layers: int = 2) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        provenance={
            "kind": "ci_fixture",
            "source": "hand_accounted",
            "sha256": None,
        },
        vocab_size=8,
        embedding_bytes=64,
        final_norm_bytes=16,
        lm_head_bytes=64,
        tie_word_embeddings=False,
        layers=tuple(
            LayerSpec(
                attention_weight_bytes=32,
                ffn_weight_bytes=48,
                router_weight_bytes=0,
                shared_expert_weight_bytes=0,
                expert_weight_bytes=(),
                top_k=0,
                kv_bytes_per_token=4,
                flops_per_token=160,
                attention_flops_per_context_token=8,
            )
            for _ in range(layers)
        ),
        lm_head_flops_per_token=128,
    )


def _moe(model_id: str = "moe") -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        provenance={
            "kind": "ci_fixture",
            "source": "hand_accounted",
            "sha256": None,
        },
        vocab_size=8,
        embedding_bytes=64,
        final_norm_bytes=16,
        lm_head_bytes=64,
        tie_word_embeddings=False,
        layers=(
            LayerSpec(
                attention_weight_bytes=32,
                ffn_weight_bytes=0,
                router_weight_bytes=8,
                shared_expert_weight_bytes=24,
                expert_weight_bytes=(64, 64, 64, 64),
                top_k=2,
                kv_bytes_per_token=4,
                flops_per_token=384,
                pre_routing_flops_per_token=80,
                attention_flops_per_context_token=8,
            ),
        ),
        lm_head_flops_per_token=128,
    )


def _trace(*requests: RequestSpec) -> RequestTrace:
    return RequestTrace(
        provenance=TraceProvenance(
            kind="ci_fixture", source="hand-authored request trace"
        ),
        requests=tuple(requests),
    )


def _slice(
    request_id: str,
    begin: int,
    count: int,
    *,
    phase: str,
    emits: bool | None = None,
) -> BatchSlice:
    if emits is None:
        emits = phase == "decode"
    return BatchSlice(
        request_id=request_id,
        token_begin=begin,
        token_count=count,
        context_tokens_before=begin,
        emits_output=emits,
        phase=phase,
    )


def _spec(
    models: Mapping[str, ModelSpec],
    *,
    hbm_capacity_bytes: int = 1024 * 1024,
    hbf_capacity_bytes: int = 0,
    external_capacity_bytes: int = 0,
    weight_tier: str = "hbm",
    cold: str | None = None,
    kv_block_tokens: int = 16,
    hbm_runtime_reserve_bytes: int = 4096,
) -> PlacementSpec:
    return PlacementSpec(
        hbm_capacity_bytes=hbm_capacity_bytes,
        hbf_capacity_bytes=hbf_capacity_bytes,
        external_capacity_bytes=external_capacity_bytes,
        hbm_runtime_reserve_bytes=hbm_runtime_reserve_bytes,
        hbm_model_cache_bytes=0,
        model_weight_tiers={model_id: weight_tier for model_id in models},
        object_tier_overrides={},
        kv_block_tokens=kv_block_tokens,
        kv_placement=KvPlacement(hot="hbm", cold=cold),
    )


def _pool_spec(model: ModelSpec, blocks: int, **overrides: Any) -> PlacementSpec:
    """A placement whose HBM KV pool holds exactly ``blocks`` 4 KiB blocks."""

    weight_tier = overrides.get("weight_tier", "hbm")
    static = len(model.memory_objects) * 4096 if weight_tier == "hbm" else 0
    return _spec(
        {model.model_id: model},
        hbm_capacity_bytes=static + 4096 + blocks * 4096,
        hbm_runtime_reserve_bytes=4096,
        **overrides,
    )


_GEOMETRY = HbfGeometry(
    stacks=2,
    channels_per_stack=1,
    dies_per_channel=1,
    planes_per_die=1,
    blocks_per_plane=8,
    pages_per_block=64,
    page_size_bytes=4096,
)


class PlacementExecutor:
    """Placement-backed executor with a fixed per-batch latency (no engine)."""

    def __init__(self, placement: HBServePlacement, latency_ns: float) -> None:
        self.placement = placement
        self.latency_ns = latency_ns
        self._frontier = 0.0
        self.mapped: list[Any] = []

    @property
    def frontier_ns(self) -> float:
        return self._frontier

    @property
    def execution_identity(self) -> dict[str, Any]:
        return {
            "kind": "test_placement_executor",
            "physical_memory_timing": False,
            "placement": self.placement.frontier_ns_independent_state,
        }

    def admit_request(self, request: RequestSpec) -> int:
        return self.placement.admit_request(request)

    def can_reserve(self, slices: Sequence[BatchSlice]) -> bool:
        return self.placement.can_reserve(slices)

    def reserve(self, batch_id: int, slices: Sequence[BatchSlice]) -> dict[str, Any]:
        return self.placement.reserve(batch_id, slices)

    def preempt_request(self, request_id: str) -> dict[str, Any]:
        return self.placement.preempt_request(request_id)

    def release_request(self, request_id: str) -> None:
        self.placement.release_request(request_id)

    def execute(self, batch: Any) -> BatchExecution:
        origin = self._frontier
        mapped = self.placement.map_batch(batch, session_frontier_ns=origin)
        self.mapped.append(mapped)
        self._frontier = max(origin, batch.schedule.not_before_ns) + self.latency_ns
        return BatchExecution(
            batch_id=batch.schedule.batch_id,
            batch_origin_ns=origin,
            finish_ns=self._frontier,
            canonical_sha256=batch.digest,
            mapped_sha256=mapped.transaction_trace_sha256,
            remap_receipt=mapped.receipt,
            physical_completion={"finish_ns": self._frontier},
        )


def _engine(
    models: Mapping[str, ModelSpec],
    trace: RequestTrace,
    executor: Any,
    policy: SchedulerPolicy,
    *,
    timing: Any = None,
    router: Any = None,
) -> HBServeEngine:
    return HBServeEngine(
        models=dict(models),
        request_trace=trace,
        compiler=HBServeCompiler(
            models=dict(models), request_trace=trace, timing=timing, router=router
        ),
        executor=executor,
        policy=policy,
    )


class ContractAndCompilerTests(unittest.TestCase):
    def test_identifier_preserves_ascii_alphabet_and_full_string_contract(self):
        from hbserve.contracts import _identifier

        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.:/'
        for character in map(chr, range(256)):
            value = 'head/' + character + '/tail'
            if character in alphabet:
                self.assertEqual(_identifier(value, 'test'), value)
            else:
                with self.assertRaises(HBServeError):
                    _identifier(value, 'test')
        for value in ('', 'valid\n', '\nvalid', '汉字', 'K', '１２', None, 12, b'valid'):
            with self.assertRaises(HBServeError):
                _identifier(value, 'test')
        self.assertEqual(_identifier(alphabet, 'test'), alphabet)

    def test_strict_json_inputs_and_ci_example_are_loadable(self) -> None:
        example_root = SERVE / "examples"
        models = load_models(
            (
                example_root / "ci-dense-model.json",
                example_root / "ci-moe-model.json",
            )
        )
        self.assertEqual(set(models), {"dense-ci", "moe-ci"})
        placement = load_placement(example_root / "ci-placement.json")
        self.assertEqual(set(placement.model_weight_tiers), set(models))
        self.assertEqual(placement.kv_placement.cold, "external")
        policy, timing, prefetch_depth = load_run_config(example_root / "ci-run.json")
        self.assertEqual(policy.prefill_chunk_tokens, 4)
        self.assertEqual(timing.timing_model, "memory_only")
        self.assertEqual(prefetch_depth, 1)
        router = load_router(example_root / "ci-synthetic-router.json")
        self.assertEqual(router.provenance.kind, "synthetic_sensitivity")
        quickstart = load_synthetic_request_config(
            example_root / "quickstart-requests.json"
        )
        self.assertEqual(set(quickstart.model_probabilities), {"llama31_8b"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = _trace(RequestSpec("r0", 0.0, "dense-ci", 1, 1))
            trace_path = root / "trace.json"
            write_json_atomic(trace_path, trace.canonical())
            self.assertEqual(load_request_trace(trace_path).digest, trace.digest)
            with self.assertRaisesRegex(HBServeError, "already exists"):
                write_json_atomic(trace_path, trace.canonical())
            duplicate_path = root / "duplicate.json"
            duplicate_path.write_text(
                '{"schema": 1, "schema": 2}', encoding="utf-8"
            )
            with self.assertRaisesRegex(HBServeError, "duplicate JSON"):
                load_json_object(duplicate_path, "duplicate fixture")

    def test_model_descriptor_round_trips_without_an_implicit_size_rule(self) -> None:
        for model in (_dense(), _moe()):
            loaded = ModelSpec.from_dict(model.canonical())
            self.assertEqual(loaded, model)
            self.assertEqual(loaded.digest, model.digest)
        self.assertEqual(
            _moe().object_by_id[_moe().object_id("router", 0)].bytes,
            8,
        )
        self.assertEqual(_dense().kv_bytes_per_token, 8)

    def test_mixed_iteration_byte_accounting(self) -> None:
        model = _dense()
        requests = (
            RequestSpec("r0", 0.0, model.model_id, 2, 4),
            RequestSpec("r1", 1.0, model.model_id, 4, 1),
            RequestSpec("r2", 2.0, model.model_id, 4, 1),
        )
        trace = _trace(*requests)
        compiler = HBServeCompiler(
            models={model.model_id: model}, request_trace=trace
        )
        batch = compiler.compile(
            ScheduledBatch(
                batch_id=3,
                model_id=model.model_id,
                slices=(
                    _slice("r0", 3, 1, phase="decode"),
                    _slice("r1", 0, 2, phase="prefill", emits=False),
                    _slice("r2", 2, 2, phase="prefill", emits=True),
                ),
                not_before_ns=0.0,
            )
        )
        self.assertEqual(batch.schedule.kind, "mixed")
        # embedding 5 tokens x 8; per layer: weights 32 + 48 once, KV reads
        # r0 3x4 + r2 2x4 = 20, KV writes 5 tokens x 4 = 20; 2 layers; final
        # norm 16; LM head 64 once although two requests emit.
        self.assertEqual(batch.logical_bytes, 40 + 2 * (80 + 20 + 20) + 16 + 64)
        roles = batch.audit_summary()["roles"]
        self.assertEqual(roles["attention/weights"]["read_bytes"], 64)
        self.assertEqual(roles["ffn/weights"]["read_bytes"], 96)
        self.assertEqual(roles["attention/kv_read"]["read_bytes"], 40)
        self.assertEqual(roles["attention/kv_write"]["write_bytes"], 40)
        self.assertEqual(roles["lm_head/read"]["operations"], 1)
        self.assertEqual(roles["embedding/read"]["operations"], 5)

    def test_prefetch_depth_shapes_the_layer_dag(self) -> None:
        model = _dense(layers=3)
        request = RequestSpec("r0", 0.0, model.model_id, 2, 1)
        trace = _trace(request)
        schedule = ScheduledBatch(
            batch_id=0,
            model_id=model.model_id,
            slices=(_slice("r0", 0, 2, phase="prefill", emits=True),),
            not_before_ns=0.0,
        )
        for depth, expected_root in ((1, "embedding/complete"), (0, "layer/0/compute")):
            compiler = HBServeCompiler(
                models={model.model_id: model},
                request_trace=trace,
                prefetch_depth=depth,
            )
            batch = compiler.compile(schedule)
            by_id = {operation.id: operation for operation in batch.operations}
            layer_one_weights = next(
                operation
                for operation in batch.memory_operations
                if operation.role == "attention/weights"
                and batch.audit[operation.id]["layer"] == 1
            )
            self.assertEqual(
                [by_id[item].role for item in layer_one_weights.dependencies],
                [expected_root],
                f"prefetch depth {depth}",
            )
            layer_two_weights = next(
                operation
                for operation in batch.memory_operations
                if operation.role == "attention/weights"
                and batch.audit[operation.id]["layer"] == 2
            )
            self.assertEqual(
                [by_id[item].role for item in layer_two_weights.dependencies],
                ["layer/0/compute" if depth == 1 else "layer/1/compute"],
            )
            compute_two = next(
                operation
                for operation in batch.operations
                if operation.role == "layer/2/compute"
            )
            self.assertEqual(
                sorted(by_id[item].role for item in compute_two.dependencies),
                ["layer/1/compute", "layer/2/memory_complete"],
            )
            kv_write = next(
                operation
                for operation in batch.memory_operations
                if operation.role == "attention/kv_write"
                and batch.audit[operation.id]["layer"] == 2
            )
            self.assertEqual(
                [by_id[item].role for item in kv_write.dependencies],
                ["layer/1/compute"],
            )
            complete = batch.operations[-1]
            self.assertEqual(complete.role, "batch/complete")
            self.assertIn(kv_write.id, complete.dependencies)

    def test_roofline_timing_follows_the_flop_ledger(self) -> None:
        model = _dense()
        timing = RooflineTimingProvider(peak_tflops=1.0, efficiency=0.5)
        # 1 TFLOPS x 0.5 = 5e11 FLOP/s = 500 FLOP/ns.
        batch = ScheduledBatch(
            batch_id=0,
            model_id=model.model_id,
            slices=(
                _slice("r0", 3, 1, phase="decode"),
                _slice("r1", 0, 4, phase="prefill", emits=True),
            ),
            not_before_ns=0.0,
        )
        result = timing.timing_for(model=model, batch=batch)
        # decode: 160 + 8 x (3 + 1); prefill: 4 x 160 + 8 x (1+2+3+4).
        expected_layer = (160 + 32 + 640 + 80) / 500.0
        self.assertEqual(result.layer_ns, (expected_layer, expected_layer))
        self.assertEqual(result.tail_ns, 2 * 128 / 500.0)
        self.assertEqual(result.timing_model, "roofline")
        self.assertTrue(timing.includes_compute)
        self.assertFalse(MemoryOnlyTimingProvider().includes_compute)
        with self.assertRaises(HBServeError):
            RooflineTimingProvider(peak_tflops=1.0, efficiency=1.5)

    def test_catalog_converter_reproduces_the_public_derivation(self) -> None:
        from hbserve.public_model import (
            derive_public_model_capacity_inputs,
        )

        descriptor = MODEL_ROOT / "llama31-8b-w8-kv-bf16.json"
        model = convert_descriptor(descriptor)
        ledger = derive_public_model_capacity_inputs(descriptor)
        self.assertEqual(model.model_id, "llama31_8b")
        self.assertEqual(model.num_layers, 32)
        self.assertEqual(model.kv_bytes_per_token, 32 * 4096)
        self.assertLessEqual(
            abs(ledger["immutable_weight_backing_bytes"] - model.weight_footprint_bytes),
            (3 * 32 + 3) * 4096,
        )
        # Every weight object is a whole number of 4 KiB pages (the catalog
        # alignment), so an HBF-resident whole-object read is page-granular;
        # the padding per object is below one page.
        components = ledger["components"]
        for memory_object in model.memory_objects:
            if memory_object.kind != "embedding":
                self.assertEqual(memory_object.bytes % 4096, 0, memory_object.id)
        exact_attention = (
            components["attention_bytes_per_layer"] + components["norm_bytes_per_layer"]
        )
        self.assertLess(
            model.layers[0].attention_weight_bytes - exact_attention, 4096
        )
        self.assertGreaterEqual(
            model.layers[0].attention_weight_bytes, exact_attention
        )
        self.assertEqual(model.embedding_bytes, components["embedding_bytes"])
        # 2 x matrix parameters per token: attention (q, k, v, o) + gated FFN.
        attention = 4096 * 4096 + 2 * 4096 * 1024 + 4096 * 4096
        ffn = 3 * 4096 * 14336
        self.assertEqual(model.layers[0].flops_per_token, 2 * (attention + ffn))
        self.assertEqual(model.layers[0].attention_flops_per_context_token, 4 * 32 * 128)
        self.assertEqual(model.lm_head_flops_per_token, 2 * 128256 * 4096)
        self.assertEqual(model.provenance["kind"], "published_descriptor")
        moe = convert_descriptor(
            MODEL_ROOT / "qwen3-235b-a22b-fp8-kv-bf16.json"
        )
        self.assertEqual(len(moe.layers[-1].expert_weight_bytes), 128)
        self.assertEqual(moe.layers[-1].top_k, 8)
        self.assertEqual(
            ModelSpec.from_dict(json.loads(json.dumps(model.canonical()))).digest,
            model.digest,
        )

    def test_dense_catalog_preserves_tensor_precision_and_scales(self) -> None:
        from hbserve.public_model import derive_public_model_capacity_inputs

        for name, layers, hidden_size, intermediate_size, heads, matrix_bytes in (
            ("llama31-70b-w8a16", 80, 8192, 28672, 64, 1),
            ("llama31-405b-w8a16", 126, 16384, 53248, 128, 1),
            ("llama31-70b-bf16", 80, 8192, 28672, 64, 2),
        ):
            with self.subTest(descriptor=name), tempfile.TemporaryDirectory() as directory:
                descriptor = load_json_object(
                    MODEL_ROOT / "llama31-70b-w8a16-kv-bf16.json", "model template"
                )
                descriptor["architecture"].update({
                    "num_layers": layers, "hidden_size": hidden_size,
                    "num_attention_heads": heads,
                    "ffn": {"dense_intermediate_size": intermediate_size},
                })
                descriptor["precision"].update({
                    "matrix_weight_bytes": matrix_bytes,
                    "matrix_weight_dtype": "int8" if matrix_bytes == 1 else "bfloat16",
                    "quantization_scheme": "symmetric_per_output_channel" if matrix_bytes == 1 else "none",
                    "scale_bytes": 2 if matrix_bytes == 1 else 0,
                })
                path = Path(directory) / (name + ".json")
                write_json_atomic(path, descriptor)
                model = convert_descriptor(path)
                ledger = derive_public_model_capacity_inputs(path)
                architecture = descriptor["architecture"]
                precision = descriptor["precision"]
                hidden = architecture["hidden_size"]
                intermediate = architecture["ffn"]["dense_intermediate_size"]
                head_dim = architecture["attention"]["head_dim"]
                kv_width = architecture["attention"]["num_key_value_heads"] * head_dim
                matrix_bytes = precision["matrix_weight_bytes"]
                metadata_bytes = precision["scale_bytes"] + precision["zero_point_bytes"]
                non_matrix_bytes = precision["non_matrix_weight_bytes"]
                alignment = descriptor["addressing"]["object_alignment_bytes"]

                def aligned(value: int) -> int:
                    return (value + alignment - 1) // alignment * alignment

                attention_parameters = 2 * hidden * (hidden + kv_width)
                attention_bytes = (
                    attention_parameters * matrix_bytes
                    + 2 * (hidden + kv_width) * metadata_bytes
                    + 2 * hidden * non_matrix_bytes
                )
                ffn_parameters = 3 * hidden * intermediate
                ffn_bytes = (
                    ffn_parameters * matrix_bytes
                    + (2 * intermediate + hidden) * metadata_bytes
                )
                self.assertEqual(model.embedding_row_bytes, hidden * matrix_bytes + metadata_bytes)
                self.assertEqual(model.layers[0].attention_weight_bytes, aligned(attention_bytes))
                self.assertEqual(model.layers[0].ffn_weight_bytes, aligned(ffn_bytes))
                self.assertEqual(
                    model.layers[0].flops_per_token,
                    2 * (attention_parameters + ffn_parameters),
                )
                self.assertEqual(model.lm_head_bytes, aligned(model.embedding_bytes))
                self.assertEqual(model.final_norm_bytes, aligned(hidden * non_matrix_bytes))
                self.assertEqual(
                    model.kv_bytes_per_token,
                    architecture["num_layers"] * 2 * kv_width * precision["kv_bytes"],
                )
                padding = model.weight_footprint_bytes - ledger["immutable_weight_backing_bytes"]
                self.assertGreaterEqual(padding, 0)
                self.assertLess(padding, (2 * model.num_layers + 2) * alignment)
                self.assertEqual(
                    ModelSpec.from_dict(model.canonical()).digest, model.digest
                )

    def test_dense_catalog_rejects_unsupported_quantization(self) -> None:
        source = MODEL_ROOT / "llama31-70b-w8a16-kv-bf16.json"
        descriptor = json.loads(source.read_text(encoding="utf-8"))
        descriptor["precision"]["quantization_scheme"] = "unrecognized_scheme"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(descriptor), encoding="utf-8")
            with self.assertRaisesRegex(HBServeError, "quantization scheme"):
                convert_descriptor(path)

    def test_explicit_model_preserves_tied_embeddings_and_catalog_fails_closed(self) -> None:
        source = MODEL_ROOT / "llama31-70b-w8a16-kv-bf16.json"
        descriptor = json.loads(source.read_text(encoding="utf-8"))
        descriptor["architecture"]["tie_word_embeddings"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(descriptor), encoding="utf-8")
            with self.assertRaisesRegex(HBServeError, "tied word embeddings are not supported"):
                convert_descriptor(path)
        untied = convert_descriptor(source)
        tied = ModelSpec.from_dict(replace(
            untied, tie_word_embeddings=True, lm_head_bytes=untied.embedding_bytes
        ).canonical())
        self.assertTrue(tied.tie_word_embeddings)
        self.assertEqual(tied.lm_head_bytes, tied.embedding_bytes)
        self.assertEqual(
            tied.weight_footprint_bytes,
            untied.weight_footprint_bytes - untied.lm_head_bytes,
        )
        self.assertEqual(tied.lm_head_object_id(), tied.object_id("embedding"))

    def test_moe_reads_unique_expert_union_once(self) -> None:
        model = _moe()
        request = RequestSpec("r0", 0.0, model.model_id, 2, 1)
        trace = _trace(request)
        router = RouterTrace(
            provenance=TraceProvenance(
                kind="ci_fixture", source="hand-authored router trace"
            ),
            decisions=(
                RouterDecision("r0", 0, 0, (0, 1)),
                RouterDecision("r0", 1, 0, (1, 2)),
            ),
        )
        compiler = HBServeCompiler(
            models={model.model_id: model},
            request_trace=trace,
            router=router,
        )
        batch = compiler.compile(
            ScheduledBatch(
                batch_id=0,
                model_id=model.model_id,
                slices=(_slice("r0", 0, 2, phase="prefill", emits=True),),
                not_before_ns=0.0,
            )
        )
        experts = [
            operation
            for operation in batch.memory_operations
            if operation.role == "moe/routed_expert_weights"
        ]
        self.assertEqual(len(experts), 3)
        self.assertEqual(sum(operation.bytes for operation in experts), 192)
        counts = {
            int(batch.audit[operation.id]["expert"]): int(
                batch.audit[operation.id]["routed_tokens"]
            )
            for operation in experts
        }
        self.assertEqual(counts, {0: 1, 1: 2, 2: 1})
        summary = batch.audit_summary()
        self.assertEqual(
            summary["roles"]["moe/routed_expert_weights"]["read_bytes"],
            192,
        )
        self.assertEqual(
            sum(row["routed_tokens"] for row in summary["moe_expert_routes"]),
            4,
        )

    def test_selected_experts_wait_for_their_own_routing_phase(self) -> None:
        model = _moe()
        model = replace(model, layers=model.layers * 3)
        request = RequestSpec("r0", 0.0, model.model_id, 2, 1)
        trace = _trace(request)
        router = RouterTrace(
            provenance=TraceProvenance(kind="ci_fixture", source="causal router fixture"),
            decisions=tuple(
                RouterDecision("r0", token, layer, (0, 1))
                for token in range(2) for layer in range(3)
            ),
        )
        schedule = ScheduledBatch(
            batch_id=0, model_id=model.model_id,
            slices=(_slice("r0", 0, 2, phase="prefill", emits=True),),
            not_before_ns=0.0,
        )
        providers = (
            MemoryOnlyTimingProvider(),
            LinearTimingProvider(100, 10, 0, 0, moe_routing_fraction=0.25),
            RooflineTimingProvider(peak_tflops=1.0, efficiency=0.5),
        )
        for provider in providers:
            for depth in (0, 1, 3):
                with self.subTest(timing=provider.timing_model, prefetch_depth=depth):
                    batch = HBServeCompiler(
                        models={model.model_id: model}, request_trace=trace,
                        router=router, timing=provider, prefetch_depth=depth,
                    ).compile(schedule)
                    by_id = {operation.id: operation for operation in batch.operations}
                    by_role = {operation.role: operation for operation in batch.operations}

                    def ancestors(identifier: str) -> set[str]:
                        pending = list(by_id[identifier].dependencies)
                        found: set[str] = set()
                        while pending:
                            dependency = pending.pop()
                            if dependency not in found:
                                found.add(dependency)
                                pending.extend(by_id[dependency].dependencies)
                        return found

                    timing = provider.timing_for(model=model, batch=schedule)
                    for layer in range(3):
                        routing = by_role[f"layer/{layer}/routing_ready"]
                        compute = by_role[f"layer/{layer}/compute"]
                        self.assertAlmostEqual(
                            routing.duration_ns + compute.duration_ns,
                            timing.layer_ns[layer],
                        )
                        if provider.includes_compute:
                            self.assertGreater(routing.duration_ns, 0.0)
                        ready_dependencies = ancestors(routing.id)
                        if layer:
                            self.assertIn(by_role[f"layer/{layer - 1}/compute"].id,
                                          ready_dependencies)
                        for operation in batch.memory_operations:
                            if batch.audit[operation.id].get("layer") != layer:
                                continue
                            if operation.role in {"attention/weights", "moe/router_weights"}:
                                self.assertIn(operation.id, ready_dependencies)
                            if operation.role == "moe/routed_expert_weights":
                                self.assertEqual(operation.dependencies, (routing.id,))
                    if provider.timing_model == "roofline":
                        self.assertEqual(timing.routing_ns, ((2 * 80 + 3 * 8) / 500,) * 3)

        with self.assertRaisesRegex(HBServeError, "moe_routing_fraction"):
            LinearTimingProvider(1, 1, 0, 0).timing_for(model=model, batch=schedule)
        missing_ledger = replace(
            model, layers=tuple(replace(layer, pre_routing_flops_per_token=None)
                                for layer in model.layers),
        )
        with self.assertRaisesRegex(HBServeError, "pre_routing_flops_per_token"):
            providers[-1].timing_for(model=missing_ledger, batch=schedule)

    def test_router_trace_must_cover_every_processed_token(self) -> None:
        model = _moe()
        request = RequestSpec("r0", 0.0, model.model_id, 2, 1)
        with self.assertRaisesRegex(HBServeError, "coverage differs"):
            HBServeCompiler(
                models={model.model_id: model},
                request_trace=_trace(request),
                router=RouterTrace(
                    provenance=TraceProvenance(
                        kind="ci_fixture", source="incomplete"
                    ),
                    decisions=(RouterDecision("r0", 0, 0, (0, 1)),),
                ),
            )

    def test_synthetic_generation_and_router_are_deterministic(self) -> None:
        config = SyntheticRequestConfig(
            request_count=16,
            arrival_rate_per_second=4.0,
            prompt_lognormal_mean_tokens=32.0,
            prompt_lognormal_sigma=0.5,
            output_lognormal_mean_tokens=8.0,
            output_lognormal_sigma=0.4,
            model_probabilities={"a": 0.8, "b": 0.2},
            seed=7,
        )
        first = generate_requests(config)
        second = generate_requests(config)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.requests, second.requests)

        model = _moe()
        request = RequestSpec("r0", 0.0, model.model_id, 1, 1)
        router = HotsetZipfRouter(
            seed=11, hot_experts=2, hot_mass=0.9, alpha=1.2
        )
        selected = router.experts_for(
            request=request, token_index=0, layer=0, model=model
        )
        self.assertEqual(
            selected,
            router.experts_for(
                request=request, token_index=0, layer=0, model=model
            ),
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(set(selected)), 2)


class PlacementTests(unittest.TestCase):
    simulator: Path | None = None

    def test_block_table_grows_incrementally_per_iteration(self) -> None:
        model = _dense()
        request = RequestSpec("r0", 0.0, model.model_id, 5, 3)
        placement = HBServePlacement(
            models={model.model_id: model},
            spec=_spec({model.model_id: model}, kv_block_tokens=2),
        )
        placement.admit_request(request)
        hot = placement.frontier_ns_independent_state["kv"]["hot"]
        self.assertEqual(hot["block_bytes"], 4096)
        free_before = hot["capacity_blocks"]
        expected_blocks = []
        for batch_id, (begin, count, phase) in enumerate(
            ((0, 3, "prefill"), (3, 2, "prefill"), (5, 1, "decode"), (6, 1, "decode"))
        ):
            slices = (_slice("r0", begin, count, phase=phase, emits=phase == "decode" or begin + count == 5),)
            self.assertTrue(placement.can_reserve(slices))
            receipt = placement.reserve(batch_id, slices)
            live = placement._state_receipt()["live_kv"][0]
            expected_blocks.append(live["blocks_per_layer"])
            self.assertEqual(
                receipt["allocated_blocks"],
                (expected_blocks[-1] - (expected_blocks[-2] if batch_id else 0)) * 2,
            )
            placement._pending_batch_id = None
        # ceil(3/2)=2, ceil(5/2)=3, ceil(6/2)=3, ceil(7/2)=4 blocks per layer.
        self.assertEqual(expected_blocks, [2, 3, 3, 4])
        self.assertEqual(
            placement._state_receipt()["kv_hot_free_blocks"], free_before - 8
        )
        placement.release_request("r0")
        self.assertTrue(all(placement.receipt()["final_invariants"].values()))

    def test_kv_reads_split_at_block_boundaries_and_merge_when_contiguous(self) -> None:
        for kv_bytes, expected_pieces in ((4, 3), (256, 1)):
            model = ModelSpec(
                model_id="m",
                provenance=_dense().provenance,
                vocab_size=8,
                embedding_bytes=64,
                final_norm_bytes=16,
                lm_head_bytes=64,
                tie_word_embeddings=False,
                layers=(
                    LayerSpec(32, 48, 0, 0, (), 0, kv_bytes, 160, 8),
                ),
                lm_head_flops_per_token=128,
            )
            block_tokens = 2 if kv_bytes == 4 else 16
            request = RequestSpec("r0", 0.0, "m", 40, 2)
            trace = _trace(request)
            placement = HBServePlacement(
                models={"m": model},
                spec=_spec({"m": model}, kv_block_tokens=block_tokens),
            )
            placement.admit_request(request)
            decode = _slice("r0", 40, 1, phase="decode")
            placement.reserve(0, (_slice("r0", 0, 40, phase="prefill", emits=True),))
            placement._pending_batch_id = None
            placement.reserve(1, (decode,))
            placement._pending_batch_id = None
            compiler = HBServeCompiler(models={"m": model}, request_trace=trace)
            batch = compiler.compile(
                ScheduledBatch(batch_id=1, model_id="m", slices=(decode,), not_before_ns=0.0)
            )
            mapped = placement.map_batch(batch, session_frontier_ns=0.0)
            kv_read = next(
                operation
                for operation in batch.memory_operations
                if operation.role == "attention/kv_read"
            )
            pieces = [
                transaction
                for transaction in mapped.transactions
                if transaction.target == "HBM"
                and transaction.op == "R"
                and transaction.bytes <= kv_read.bytes
                and transaction.addr >= placement.hbm_kv_begin
            ]
            self.assertEqual(len(pieces), 40 * kv_bytes // (block_tokens * kv_bytes) if kv_bytes == 4 else expected_pieces)
            self.assertEqual(sum(piece.bytes for piece in pieces), kv_read.bytes)
            self.assertEqual(
                mapped.receipt["invariants"]["projected_logical_bytes"],
                batch.logical_bytes,
            )
            placement.release_request("r0")

    def test_preemption_without_cold_tier_recomputes_and_keeps_first_token(self) -> None:
        model = _dense()
        # Pool of exactly 6 blocks (2 layers): 16 tokens cost one block per
        # layer, so r0 growing past 32 tokens and r1 cannot coexist.
        spec = _pool_spec(model, 6)
        requests = (
            RequestSpec("r0", 0.0, model.model_id, 16, 20),
            RequestSpec("r1", 5.0, model.model_id, 16, 4),
        )
        trace = _trace(*requests)
        placement = HBServePlacement(models={model.model_id: model}, spec=spec)
        executor = PlacementExecutor(placement, 10.0)
        result = _engine(
            {model.model_id: model},
            trace,
            executor,
            SchedulerPolicy(max_batch_requests=4, max_batch_tokens=64, prefill_chunk_tokens=64),
        ).run()
        self.assertGreater(result["summary"]["preemptions"], 0)
        modes = {row["mode"] for row in result["scheduler"]["preemptions"]}
        self.assertEqual(modes, {"recompute"})
        rows = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(rows["r0"]["output_tokens"], 20)
        # The youngest request (r1) is preempted; the older one keeps its
        # blocks and its first-token time.
        self.assertGreater(rows["r1"]["preemptions"], 0)
        self.assertEqual(rows["r0"]["preemptions"], 0)
        self.assertEqual(rows["r0"]["first_token_ns"], 10.0)
        self.assertEqual(rows["r1"]["first_token_ns"], 20.0)
        self.assertTrue(all(placement.receipt()["final_invariants"].values()))
        # A recompute re-feeds the discarded context as one prefill chunk
        # whose end lies past the prompt.
        recompute = [
            item
            for batch in result["batches"]
            for item in batch["schedule"]["slices"]
            if item["phase"] == "prefill" and item["token_begin"] == 0
            and item["token_count"] > 16
        ]
        self.assertTrue(recompute)

    def test_hbf_cold_kv_reads_directly_without_swap_in(self) -> None:
        model = _dense()
        spec = _pool_spec(
            model,
            6,
            hbf_capacity_bytes=_GEOMETRY.capacity_bytes,
            weight_tier="hbf",
            cold="hbf",
        )
        requests = (
            RequestSpec("r0", 0.0, model.model_id, 16, 20),
            RequestSpec("r1", 5.0, model.model_id, 16, 4),
        )
        trace = _trace(*requests)
        placement = HBServePlacement(
            models={model.model_id: model}, spec=spec, hbf_geometry=_GEOMETRY
        )
        executor = PlacementExecutor(placement, 10.0)
        result = _engine(
            {model.model_id: model},
            trace,
            executor,
            SchedulerPolicy(max_batch_requests=4, max_batch_tokens=64, prefill_chunk_tokens=64),
        ).run()
        preemptions = result["scheduler"]["preemptions"]
        self.assertTrue(preemptions)
        self.assertEqual({row["mode"] for row in preemptions}, {"swap_out"})
        migration = result["scheduler"]["kv_migration_bytes"]
        self.assertGreater(migration["swap_out"], 0)
        self.assertEqual(migration["swap_in"], 0)
        cumulative = placement.receipt()["cumulative"]
        self.assertEqual(cumulative["kv_swap_in_bytes"], 0)
        self.assertTrue(all(placement.receipt()["final_invariants"].values()))
        # Swap-outs preserve data through real HBM-to-HBF copies. Subsequent
        # execution reads HBF directly, including when new blocks are needed.
        links = [
            transaction
            for mapped in executor.mapped
            for transaction in mapped.transactions
            if transaction.target.startswith("D2D_")
        ]
        self.assertTrue(links)
        self.assertTrue(all(transaction.stack in (0, 1) for transaction in links))
        self.assertEqual(
            sum(t.bytes for t in links if t.target == "D2D_HBM_TO_HBF"),
            migration["swap_out"],
        )
        self.assertEqual(
            sum(t.bytes for t in links if t.target == "D2D_HBF_TO_HBM"),
            migration["swap_in"],
        )
        hbf_writes = sum(
            transaction.bytes
            for mapped in executor.mapped
            for transaction in mapped.transactions
            if transaction.target == "HBF_LOGICAL" and transaction.op == "W"
        )
        self.assertGreater(hbf_writes, migration["swap_out"])
        table = {tx.id: tx for mapped in executor.mapped for tx in mapped.transactions}
        cold_reads = [tx for tx in table.values() if tx.target == "HBF_LOGICAL"
                      and tx.op == "R" and tx.addr >= placement._cold.begin]
        self.assertTrue(cold_reads, "resumed attention must consume its HBF KV home")
        first = cold_reads[0]
        seen, pending = set(), list(first.dependencies)
        while pending:
            name = pending.pop()
            if name not in seen:
                seen.add(name)
                pending.extend(table[name].dependencies)
        self.assertTrue(any(table[name].target == "HBF_LOGICAL" and table[name].op == "W"
                            and table[name].addr <= first.addr < table[name].addr + table[name].bytes
                            for name in seen), "direct reads must wait for the actual asynchronous copy")
        rows = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(rows["r0"]["first_token_ns"], 10.0)

    def test_waiting_requests_migrate_to_the_external_cold_tier_first(self) -> None:
        model = _dense()
        spec = _pool_spec(
            model, 6, external_capacity_bytes=64 * 4096, cold="external"
        )
        placement = HBServePlacement(models={model.model_id: model}, spec=spec)
        requests = (
            RequestSpec("r0", 0.0, model.model_id, 32, 2),
            RequestSpec("r1", 1.0, model.model_id, 32, 2),
        )
        for request in requests:
            placement.admit_request(request)
        placement.reserve(0, (_slice("r0", 0, 32, phase="prefill", emits=True),))
        placement._pending_migrations.clear()
        placement._pending_batch_id = None
        second = (_slice("r1", 0, 32, phase="prefill", emits=True),)
        self.assertTrue(placement.can_reserve(second))
        receipt = placement.reserve(1, second)
        self.assertEqual(receipt["swap_out_requests"], ["r0"])
        self.assertEqual(receipt["swap_out_blocks"], 4)
        placement._pending_migrations.clear()
        placement._pending_batch_id = None
        # Resuming r0 needs its 4 cold blocks back plus 2 for token 32; the
        # only way to fit is to evict the now-waiting r1 first.
        back = (_slice("r0", 32, 1, phase="decode"),)
        self.assertTrue(placement.can_reserve(back))
        receipt = placement.reserve(2, back)
        self.assertEqual(receipt["swap_out_requests"], ["r1"])
        self.assertEqual(receipt["swap_in_requests"], ["r0"])
        self.assertEqual(receipt["swap_in_blocks"], 4)
        self.assertEqual(receipt["allocated_blocks"], 2)
        placement.release_request("r0")
        placement.release_request("r1")
        self.assertTrue(placement.receipt()["final_invariants"]["kv_cold_pool_fully_free"])

    def test_hot_expert_override_splits_one_moe_batch_across_hbm_and_hbf(
        self,
    ) -> None:
        model = _moe()
        request = RequestSpec("r0", 0.0, model.model_id, 1, 1)
        trace = _trace(request)
        compiler = HBServeCompiler(
            models={model.model_id: model},
            request_trace=trace,
            router=RouterTrace(
                provenance=TraceProvenance(
                    kind="ci_fixture", source="hot/cold expert placement"
                ),
                decisions=(RouterDecision("r0", 0, 0, (0, 1)),),
            ),
        )
        placement = HBServePlacement(
            models={model.model_id: model},
            spec=PlacementSpec(
                hbm_capacity_bytes=1024 * 1024,
                hbf_capacity_bytes=1024 * 1024,
                external_capacity_bytes=0,
                hbm_runtime_reserve_bytes=4096,
                hbm_model_cache_bytes=0,
                model_weight_tiers={model.model_id: "hbf"},
                object_tier_overrides={
                    model.expert_object_id(0, 0): "hbm"
                },
            ),
        )
        placement.admit_request(request)
        slices = (_slice("r0", 0, 1, phase="prefill", emits=True),)
        placement.reserve(0, slices)
        batch = compiler.compile(
            ScheduledBatch(
                batch_id=0,
                model_id=model.model_id,
                slices=slices,
                not_before_ns=0.0,
            )
        )
        mapped = placement.map_batch(batch, session_frontier_ns=0.0)
        traffic = mapped.receipt["logical_traffic_by_target"]
        self.assertGreater(traffic["HBM"]["read_bytes"], 0)
        self.assertGreater(traffic["HBF_LOGICAL"]["read_bytes"], 0)
        self.assertEqual(
            sum(
                row["read_bytes"] + row["write_bytes"]
                for row in traffic.values()
            ),
            batch.logical_bytes,
        )
        placement.release_request(request.request_id)

    def test_multi_model_cache_activation_hit_and_lru_eviction(self) -> None:
        model_a = _dense("a")
        model_b = _dense("b")
        requests = tuple(
            RequestSpec(request_id, float(index), model_id, 1, 1)
            for index, (request_id, model_id) in enumerate(
                (("a0", "a"), ("a1", "a"), ("b0", "b"), ("a2", "a"))
            )
        )
        trace = _trace(*requests)
        compiler = HBServeCompiler(
            models={"a": model_a, "b": model_b}, request_trace=trace
        )
        one_model_extent = len(model_a.memory_objects) * 4096
        placement = HBServePlacement(
            models={"a": model_a, "b": model_b},
            spec=PlacementSpec(
                hbm_capacity_bytes=2 * 1024 * 1024,
                hbf_capacity_bytes=0,
                external_capacity_bytes=2 * 1024 * 1024,
                hbm_runtime_reserve_bytes=4096,
                hbm_model_cache_bytes=one_model_extent,
                model_weight_tiers={
                    "a": "external_cached_hbm",
                    "b": "external_cached_hbm",
                },
                object_tier_overrides={},
            ),
        )
        activations: list[dict[str, object]] = []
        for batch_id, request in enumerate(requests):
            placement.admit_request(request)
            slices = (_slice(request.request_id, 0, 1, phase="prefill", emits=True),)
            placement.reserve(batch_id, slices)
            canonical = compiler.compile(
                ScheduledBatch(
                    batch_id=batch_id,
                    model_id=request.model_id,
                    slices=slices,
                    not_before_ns=request.arrival_ns,
                )
            )
            mapped = placement.map_batch(canonical, session_frontier_ns=0.0)
            activations.append(dict(mapped.receipt["activation"]))
            placement.release_request(request.request_id)
        self.assertEqual(
            [row["cache_hit"] for row in activations],
            [False, True, False, False],
        )
        self.assertEqual(
            activations[0]["external_read_bytes"],
            model_a.weight_footprint_bytes,
        )
        self.assertEqual(
            activations[0]["hbm_install_write_bytes"],
            model_a.weight_footprint_bytes,
        )
        self.assertEqual(
            activations[0]["chunks"], len(model_a.memory_objects)
        )
        self.assertEqual(activations[2]["evicted_models"], ["a"])
        self.assertEqual(activations[3]["evicted_models"], ["b"])
        cumulative = placement.receipt()["cumulative"]
        self.assertEqual(cumulative["model_activations"], 3)
        self.assertEqual(cumulative["model_evictions"], 2)

    def test_hbm_objects_are_stripe_aligned_and_pieces_split_into_replicable_heads(
        self,
    ) -> None:
        # The HBM engine replicates a request across pseudo-channels only
        # when it starts on an address-map stripe and covers whole stripes;
        # placement therefore stripe-aligns HBM objects and splits every HBM
        # piece into a whole-stripe head plus scalar remainders, conserving
        # bytes exactly.
        model = ModelSpec(
            model_id="m",
            provenance=_dense().provenance,
            vocab_size=8,
            embedding_bytes=64,
            final_norm_bytes=16,
            lm_head_bytes=64,
            tie_word_embeddings=False,
            layers=(LayerSpec(208, 48, 0, 0, (), 0, 4, 160, 8),),
            lm_head_flops_per_token=128,
        )
        spec = PlacementSpec(
            hbm_capacity_bytes=1024 * 1024,
            hbf_capacity_bytes=0,
            external_capacity_bytes=0,
            hbm_runtime_reserve_bytes=16,
            hbm_model_cache_bytes=0,
            model_weight_tiers={"m": "hbm"},
            object_tier_overrides={},
            hbm_alignment_bytes=16,
            kv_block_tokens=16,
        )
        placement = HBServePlacement(
            models={"m": model}, spec=spec, hbm_stripe_bytes=64
        )
        layout = placement.frontier_ns_independent_state
        self.assertEqual(layout["hbm_stripe_bytes"], 64)
        self.assertEqual(layout["hbm_object_alignment_bytes"], 64)
        # Objects at least one stripe long start on a stripe; shorter ones
        # keep the allocation granularity (they can never replicate).
        for placed in placement._static.values():
            if placed.bytes >= 64:
                self.assertEqual(placed.addr % 64, 0, placed.object_id)
            else:
                self.assertEqual(placed.addr % 16, 0, placed.object_id)
        self.assertEqual(placement.hbm_kv_begin % 64, 0)
        self.assertEqual(
            placement._hbm_stripe_pieces(0, 208), [(0, 192), (192, 16)]
        )
        self.assertEqual(
            placement._hbm_stripe_pieces(48, 200), [(48, 16), (64, 128), (192, 56)]
        )
        self.assertEqual(placement._hbm_stripe_pieces(0, 48), [(0, 48)])
        self.assertEqual(placement._hbm_stripe_pieces(32, 80), [(32, 80)])
        request = RequestSpec("r0", 0.0, "m", 40, 2)
        trace = _trace(request)
        placement.admit_request(request)
        placement.reserve(0, (_slice("r0", 0, 40, phase="prefill", emits=True),))
        placement._pending_batch_id = None
        decode = _slice("r0", 40, 1, phase="decode")
        placement.reserve(1, (decode,))
        placement._pending_batch_id = None
        batch = HBServeCompiler(models={"m": model}, request_trace=trace).compile(
            ScheduledBatch(batch_id=1, model_id="m", slices=(decode,), not_before_ns=0.0)
        )
        mapped = placement.map_batch(batch, session_frontier_ns=0.0)
        kv_read = next(
            operation
            for operation in batch.memory_operations
            if operation.role == "attention/kv_read"
        )

        def hbm_reads_within(begin: int, end: int) -> list[tuple[int, int]]:
            return [
                (transaction.addr, transaction.bytes)
                for transaction in mapped.transactions
                if transaction.target == "HBM"
                and transaction.op == "R"
                and begin <= transaction.addr < end
            ]

        # 208 B object at a stripe boundary -> 192 B whole-stripe head + 16 B.
        attention_addr = placement._static[model.object_id("attention", 0)].addr
        attention_pieces = hbm_reads_within(attention_addr, attention_addr + 208)
        self.assertEqual(
            attention_pieces,
            [(attention_addr, 192), (attention_addr + 192, 16)],
        )
        # 40 tokens x 4 B = 160 B of contiguous 64 B blocks -> 128 B + 32 B.
        kv_pieces = hbm_reads_within(placement.hbm_kv_begin, placement._hot.end)
        self.assertEqual(
            kv_pieces,
            [(placement.hbm_kv_begin, 128), (placement.hbm_kv_begin + 128, 32)],
        )
        self.assertEqual(sum(count for _, count in kv_pieces), kv_read.bytes)
        self.assertEqual(
            mapped.receipt["invariants"]["projected_logical_bytes"],
            batch.logical_bytes,
        )
        self.assertTrue(mapped.receipt["invariants"]["canonical_bytes_conserved"])
        # Without a stripe the placement is unchanged: one piece per extent.
        plain = HBServePlacement(models={"m": model}, spec=spec)
        self.assertIsNone(plain.frontier_ns_independent_state["hbm_stripe_bytes"])
        self.assertEqual(plain._hbm_stripe_pieces(0, 208), [(0, 208)])

    def test_placement_presets_derive_capacities_from_the_system_config(self) -> None:
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        from hbfsim_client.simulation_session import ResolvedSystemConfig

        system = ResolvedSystemConfig.load(
            [ROOT / "configs/systems/miniquick/4hbm-4hbf.cfg"]
        ).resolve(self.simulator)
        model = _dense()
        spec = derive_placement(
            system=system, models={model.model_id: model}, preset="weights-hbf-kv-hbm"
        )
        self.assertEqual(spec.hbm_capacity_bytes, system.hbm_capacity_bytes)
        self.assertEqual(spec.model_weight_tiers, {model.model_id: "hbf"})
        self.assertIsNone(spec.kv_placement.cold)
        self.assertEqual(spec.hbf_capacity_bytes, system.logical_hbf_capacity_bytes)
        self.assertEqual(spec.hbf_capacity_bytes % 4096, 0)
        cold = derive_placement(
            system=system,
            models={model.model_id: model},
            preset="weights-hbf-kv-hbf-cold",
        )
        self.assertEqual(cold.kv_placement.cold, "hbf")
        all_hbm = derive_placement(
            system=system, models={model.model_id: model}, preset="all-hbm"
        )
        self.assertEqual(all_hbm.hbf_capacity_bytes, 0)
        with self.assertRaisesRegex(HBServeError, "unknown placement preset"):
            derive_placement(system=system, models={model.model_id: model}, preset="x")


class EngineTests(unittest.TestCase):
    def _placement_executor(
        self, model: ModelSpec, latency_ns: float = 100.0
    ) -> PlacementExecutor:
        placement = HBServePlacement(
            models={model.model_id: model}, spec=_spec({model.model_id: model})
        )
        return PlacementExecutor(placement, latency_ns)

    def test_arrivals_join_the_next_iteration_instead_of_the_next_cohort(self) -> None:
        # The CI request set: arrivals at 0, 455, 550, and 3501 ns.  With
        # mixed iterations the 455/550 ns arrivals are prefilled in the
        # iteration right after they arrive, alongside req0's decode.
        model = _dense()
        requests = (
            RequestSpec("req0", 0.0, model.model_id, 2, 2),
            RequestSpec("req1", 455.0, model.model_id, 2, 2),
            RequestSpec("req2", 550.0, model.model_id, 2, 2),
            RequestSpec("req3", 3501.0, model.model_id, 2, 2),
        )
        trace = _trace(*requests)
        result = _engine(
            {model.model_id: model},
            trace,
            self._placement_executor(model, 1000.0),
            SchedulerPolicy(max_batch_requests=8, max_batch_tokens=8, prefill_chunk_tokens=4),
        ).run()
        batches = result["batches"]
        second = batches[1]["schedule"]
        self.assertEqual(second["kind"], "mixed")
        self.assertEqual(
            [(item["request_id"], item["phase"]) for item in second["slices"]],
            [("req0", "decode"), ("req1", "prefill"), ("req2", "prefill")],
        )
        self.assertEqual(second["not_before_ns"], 1000.0)
        rows = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(rows["req1"]["first_token_ns"], 2000.0)
        self.assertEqual(rows["req2"]["first_token_ns"], 2000.0)
        self.assertEqual(
            [batch["kind"] for batch in batches],
            ["prefill", "mixed", "decode", "prefill", "decode"],
        )
        self.assertEqual(result["scheduler"]["iterations_by_kind"]["mixed"], 1)
        self.assertEqual(result["scheduler"]["policy"]["batch_phase_policy"], "mixed_iteration_v1")

    def test_chunked_prefill_emits_exactly_one_first_token(self) -> None:
        model = _dense()
        request = RequestSpec("r0", 0.0, model.model_id, 3, 1)
        trace = _trace(request)
        result = _engine(
            {model.model_id: model},
            trace,
            self._placement_executor(model),
            SchedulerPolicy(max_batch_requests=1, max_batch_tokens=2, prefill_chunk_tokens=2),
        ).run()
        self.assertEqual(result["scheduler"]["batches"], 2)
        self.assertEqual(result["requests"][0]["first_token_ns"], 200.0)
        self.assertEqual(result["requests"][0]["completion_ns"], 200.0)
        first_roles = result["batches"][0]["canonical_workload"]["roles"]
        second_roles = result["batches"][1]["canonical_workload"]["roles"]
        self.assertNotIn("lm_head/read", first_roles)
        self.assertIn("lm_head/read", second_roles)

    def test_token_budget_bounds_every_iteration(self) -> None:
        model = _dense()
        requests = (
            RequestSpec("r0", 0.0, model.model_id, 6, 2),
            RequestSpec("r1", 0.0, model.model_id, 6, 2),
        )
        trace = _trace(*requests)
        result = _engine(
            {model.model_id: model},
            trace,
            self._placement_executor(model),
            SchedulerPolicy(max_batch_requests=4, max_batch_tokens=4, prefill_chunk_tokens=4),
        ).run()
        for batch in result["batches"]:
            self.assertLessEqual(
                sum(item["token_count"] for item in batch["schedule"]["slices"]), 4
            )
        kinds = [batch["kind"] for batch in result["batches"]]
        self.assertEqual(kinds[0], "prefill")
        self.assertIn("mixed", kinds)

    def test_timing_model_selects_ttft_or_memory_critical_path_fields(self) -> None:
        model = _dense()
        request = RequestSpec("r0", 50.0, model.model_id, 2, 2)
        trace = _trace(request)
        policy = SchedulerPolicy(max_batch_requests=4, max_batch_tokens=16, prefill_chunk_tokens=16)
        linear = _engine(
            {model.model_id: model},
            trace,
            self._placement_executor(model),
            policy,
            timing=LinearTimingProvider(1, 1, 1, 1),
        ).run()
        row = linear["requests"][0]
        self.assertEqual(row["first_token_ns"], 150.0)
        self.assertEqual(row["completion_ns"], 250.0)
        self.assertEqual(row["ttft_ns"], 100.0)
        self.assertEqual(row["tpot_ns"], 100.0)
        self.assertEqual(linear["summary"]["timing_model"], "linear")
        self.assertIn("mean_ttft_ns", linear["summary"])
        self.assertTrue(linear["eligibility"]["ttft_tpot_reported"])
        self.assertFalse(linear["eligibility"]["ttft_tpot_slo_claim"])
        self.assertEqual(
            linear["workload_accounting"]["token_id_sources"],
            {"sha256_surrogate_token_id": 3},
        )
        memory_only = _engine(
            {model.model_id: model},
            trace,
            self._placement_executor(model),
            policy,
        ).run()
        row = memory_only["requests"][0]
        self.assertNotIn("ttft_ns", row)
        self.assertEqual(row["memory_critical_path_first_token_ns"], 100.0)
        self.assertNotIn("mean_ttft_ns", memory_only["summary"])
        self.assertEqual(
            memory_only["summary"]["mean_memory_critical_path_per_token_ns"], 100.0
        )
        self.assertEqual(memory_only["summary"]["timing_model"], "memory_only")
        self.assertFalse(memory_only["eligibility"]["ttft_tpot_reported"])


class FrontendTests(unittest.TestCase):
    def test_help_and_subcommands_share_one_entrypoint(self) -> None:
        with redirect_stdout(StringIO()) as output:
            self.assertEqual(frontend_main(["--help"]), 0)
        self.assertIn("hbserve", output.getvalue())
        self.assertIn("fixed memory windows", output.getvalue())
        for command in ("run", "model", "generate"):
            with self.subTest(command=command), redirect_stdout(StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    frontend_main([command, "--help"])
                self.assertEqual(raised.exception.code, 0)

    def test_incompatible_inputs_are_rejected_before_execution(self) -> None:
        cases = (
            (["--experiment", "window.json", "--requests", "requests.json"], "not allowed"),
            (["--experiment", "window.json", "--timing", "memory_only"], "closed-loop options"),
            (["--experiment", "window.json", "--model", "model.json"], "closed-loop options"),
            (["--experiment", "window.json", "--system", "system.cfg"], "closed-loop options"),
            (["--experiment", "window.json", "--placement", "all-hbm"], "closed-loop options"),
            (["--experiment", "window.json", "--preflight-only", "--topologies", "all-hbm"], "execution-only"),
            (["--experiment", "window.json", "--simulator", "unused-simulator", "--topologies", "all-hbm,all-hbm"], "unique IDs"),
            (["--experiment", "window.json", "--simulator", "unused-simulator", "--topologies", "all-hbm,"], "non-empty"),
            (["--experiment", "window.json", "--simulator", "unused-simulator", "--topologies", ""], "non-empty"),
            (["--experiment", "window.json"], "--simulator is required"),
            (["--requests", "requests.json", "--preflight-only"], "require --experiment"),
            (["--requests", "requests.json"], "--requests requires"),
            (["--model", "model.json"], "required"),
            (["--requests", "requests.json", "--model", "model.json", "--system", "system.cfg", "--placement", "all-hbm", "--run-config", "run.json", "--prefetch-depth", "0"], "cannot be combined"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments), redirect_stderr(StringIO()) as errors:
                with self.assertRaises(SystemExit) as raised:
                    frontend_main(["run", *arguments, "--out", "unused"])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, errors.getvalue())

    def test_run_directories_do_not_collide_within_one_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = create_run_directory(Path(directory))
            second = create_run_directory(Path(directory))
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_generate_and_model_remain_under_the_same_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            requests = Path(directory) / "requests.json"
            model = Path(directory) / "model.json"
            self.assertEqual(frontend_main([
                "generate", "--config", str(SERVE / "examples/ci-synthetic-requests.json"),
                "--output", str(requests),
            ]), 0)
            self.assertEqual(len(load_request_trace(requests).requests), 4)
            self.assertEqual(frontend_main([
                "model", str(MODEL_ROOT / "llama31-8b-w8-kv-bf16.json"),
                "--output", str(model),
            ]), 0)
            self.assertEqual(load_json_object(model, "model")["schema"]["name"], "hbserve.model")

    def test_capabilities_distinguish_execution_semantics(self) -> None:
        capabilities = current_capabilities()
        modes = capabilities["execution_modes"]
        self.assertTrue(modes["closed_loop"]["request_scheduling_feedback"])
        self.assertFalse(modes["fixed_window"]["request_scheduling_feedback"])
        self.assertEqual(modes["fixed_window"]["ttft_tpot"], "not_reported")
        self.assertTrue(capabilities["prefix_cache_lifecycle"])
        self.assertTrue(capabilities["prefix_block_hash_identity"])


class PhysicalIntegrationTests(unittest.TestCase):
    simulator: Path | None = None

    def test_capacity_preflight_matches_the_engine_and_resolves_aliases(self) -> None:
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        from hbfsim_client.simulation_session import ResolvedSystemConfig

        with tempfile.TemporaryDirectory(prefix="serveloop-capacity-") as directory:
            overlay = Path(directory) / "small.cfg"
            overrides = "\n".join((
                "hbf-stacks=1", "hbf-channels=1", "hbf-dies-per-channel=1",
                "hbf-planes-per-die=1", "hbf-blocks-per-plane=8",
                "hbf-pages-per-block=8", "hbf-page-size=4096",
                "hbf-gc-reserved-free-blocks-per-plane=2",
                "hbf-gc-low-watermark-pages=0", "hbf-gc-hard-watermark-pages=0",
                "hbf-write-coalescing=false", "hbf-thermal-enable=false", "",
            ))
            overlay.write_text(overrides, encoding="utf-8")
            paths = [ROOT / "configs/systems/eight-stack-baseline.cfg", overlay]
            system = ResolvedSystemConfig.load(paths).resolve(self.simulator)
            self.assertEqual(system.logical_hbf_capacity_bytes, 22 * 4096)
            model = _dense(layers=1)
            model = replace(model, layers=(replace(model.layers[0], attention_weight_bytes=24 * 4096),))
            models = {model.model_id: model}
            spec = derive_placement(system=system, models=models, preset="weights-hbf-kv-hbm")
            with self.assertRaisesRegex(HBServeError, "HBF payload capacity"):
                HBServePlacement(models=models, spec=spec)
            oversized = HBServePlacement(
                models=models,
                spec=replace(spec, hbf_capacity_bytes=system.hbf_geometry.capacity_bytes),
            )
            with self.assertRaisesRegex(HBServeError, "engine's logical capacity"):
                HbfSimExecutor(simulator_path=self.simulator, system_config_paths=paths,
                               placement=oversized)
            overlay.write_text(overrides + "hbf-capacity-bytes=524288\n", encoding="utf-8")
            expanded = ResolvedSystemConfig.load(paths).resolve(self.simulator)
            self.assertEqual(expanded.hbf_geometry.blocks_per_plane, 16)
            self.assertEqual(expanded.logical_hbf_capacity_bytes, 86 * 4096)
            overlay.write_text(
                overrides + "hbf-ctrl-dram-capacity-denominator=8\n", encoding="utf-8"
            )
            ratio = ResolvedSystemConfig.load(paths).resolve(self.simulator)
            self.assertEqual(ratio.hbf_ctrl_dram_bytes, 64 * 4096 // 8)

    def test_detached_cold_migrations_keep_extent_dependencies(self) -> None:
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        from hbfsim_client.simulation_session import (
            ResolvedSystemConfig,
            SimulationSession,
        )

        model = _dense()
        models = {model.model_id: model}
        system = ResolvedSystemConfig.load((
            ROOT / "configs/systems/4hbm-4hbf.cfg",
            ROOT / "configs/overlays/backing/nvme-ssd.cfg",
        ))
        for reuse_extent in (False, True):
            with self.subTest(reuse_extent=reuse_extent):
                spec = _pool_spec(
                    model, 6, external_capacity_bytes=12 * 4096, cold="external"
                )
                placement = HBServePlacement(models=models, spec=spec)
                requests = tuple(
                    RequestSpec(f"r{index}", float(index), model.model_id, 32, 8)
                    for index in range(3)
                )
                compiler = HBServeCompiler(
                    models=models, request_trace=_trace(*requests)
                )
                for request in requests:
                    placement.admit_request(request)
                plans = [
                    _slice("r0", 0, 32, phase="prefill", emits=True),
                    _slice("r1", 0, 32, phase="prefill", emits=True),
                    _slice("r1", 32, 1, phase="decode"),
                    _slice("r1", 33, 1, phase="decode"),
                    _slice("r1", 34, 1, phase="decode"),
                    _slice("r2", 0, 32, phase="prefill", emits=True)
                    if reuse_extent else _slice("r0", 32, 1, phase="decode"),
                ]
                receipts = []
                mapped_batches = []
                with SimulationSession(
                    simulator_path=self.simulator,
                    system_config=system,
                    enable_hbm=True,
                    enable_hbf=False,
                    enable_external=True,
                    hbm_capacity_bytes=spec.hbm_capacity_bytes,
                ) as session:
                    for batch_id, batch_slice in enumerate(plans):
                        if batch_id == 5 and reuse_extent:
                            placement.release_request("r0")
                        placement.reserve(batch_id, (batch_slice,))
                        canonical = compiler.compile(ScheduledBatch(
                            batch_id=batch_id,
                            model_id=model.model_id,
                            slices=(batch_slice,),
                            not_before_ns=0.0,
                        ))
                        mapped = placement.map_batch(
                            canonical,
                            session_frontier_ns=session.completed_frontier_ns,
                        )
                        mapped = replace(
                            mapped, frontier=(mapped.transactions[-1].id,),
                            completions=True,
                        )
                        mapped_batches.append(mapped)
                        receipts.append(session.submit(mapped))
                offloads = [
                    transaction for transaction in mapped_batches[1].transactions
                    if transaction.target == "EXTERNAL" and transaction.op == "W"
                ]
                original_completions = {
                    row["id"]: row for row in receipts[1]["transaction_completions"]
                }
                final_completions = {
                    row["id"]: row for row in receipts[-1]["transaction_completions"]
                }
                checked = 0
                for offload in offloads:
                    finish = original_completions[offload.id]["finish_ns"]
                    self.assertGreater(finish, receipts[-2]["blocking_finish_ns"])
                    self.assertIn(offload.id, mapped_batches[-2].retain)
                    for transaction in mapped_batches[-1].transactions:
                        if transaction.target != "EXTERNAL" or not (
                            transaction.addr < offload.addr + offload.bytes
                            and offload.addr < transaction.addr + transaction.bytes
                        ):
                            continue
                        checked += 1
                        self.assertIn(offload.id, transaction.dependencies)
                        self.assertGreaterEqual(
                            final_completions[transaction.id]["start_ns"], finish
                        )
                self.assertGreater(checked, 0)

    def _system_configs(self) -> tuple[Path, ...]:
        return (
            ROOT / "configs/systems/eight-stack-baseline.cfg",
            ROOT / "tests/fixtures/simulation-session-mini.cfg",
            ROOT / "configs/overlays/backing/cxl-memory.cfg",
        )

    def test_checked_in_cli_example(self) -> None:
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        example_root = SERVE / "examples"
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            command = [
                sys.executable,
                "-B",
                "-m",
                "hbserve",
                "run",
                "--simulator",
                str(self.simulator),
                "--system",
                str(ROOT / "configs/systems/eight-stack-baseline.cfg"),
                "--overlay",
                str(ROOT / "tests/fixtures/simulation-session-mini.cfg"),
                "--overlay",
                str(ROOT / "configs/overlays/backing/cxl-memory.cfg"),
            ]
            for model in (
                example_root / "ci-dense-model.json",
                example_root / "ci-moe-model.json",
            ):
                command.extend(("--model", str(model)))
            command.extend(
                (
                    "--requests",
                    str(example_root / "ci-synthetic-requests.json"),
                    "--router",
                    str(example_root / "ci-synthetic-router.json"),
                    "--placement",
                    str(example_root / "ci-placement.json"),
                    "--run-config",
                    str(example_root / "ci-run.json"),
                    "--out",
                    str(output_root / "ci"),
                )
            )
            executed = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(executed.returncode, 0, executed.stderr)
            headline = executed.stdout.strip().splitlines()
            self.assertLessEqual(len(headline), 10)
            self.assertIn("timing model: memory_only", executed.stdout)
            results = list((output_root / "ci").glob("*/result.json"))
            self.assertEqual(len(results), 1)
            result = json.loads(results[0].read_text(encoding="utf-8"))
            self.assertEqual(result["execution_mode"], "closed_loop")
            self.assertEqual(result["hbserve"]["summary"]["requests"], 4)
            self.assertEqual(
                result["hbserve"]["summary"]["timing_model"], "memory_only"
            )
            self.assertNotIn("mean_ttft_ns", result["hbserve"]["summary"])
            devices = result["final_execution"]["session"][
                "final_measurement"
            ]["device_workload_totals"]
            self.assertGreater(devices["hbf"]["logical_read_bytes"], 0)
            self.assertGreater(devices["external"]["read_bytes"], 0)
            self.assertGreater(
                devices["hbm"]["read_bytes"]
                + devices["hbm"]["write_bytes"],
                0,
            )
            # Running again must not collide with the first output directory.
            executed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False
            )
            self.assertEqual(executed.returncode, 0, executed.stderr)
            self.assertEqual(len(list((output_root / "ci").glob("*/result.json"))), 2)

    def test_hbf_and_external_coexist_in_closed_loop_serving(self) -> None:
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        hbf_model = _dense("hbf_model")
        cold_model = _dense("cold_model")
        requests = (
            RequestSpec("h0", 0.0, "hbf_model", 1, 1, token_ids=(1,)),
            RequestSpec("c0", 1.0, "cold_model", 1, 1, token_ids=(2,)),
        )
        trace = _trace(*requests)
        models = {"hbf_model": hbf_model, "cold_model": cold_model}
        compiler = HBServeCompiler(
            models=models,
            request_trace=trace,
            timing=MemoryOnlyTimingProvider(),
        )
        placement = HBServePlacement(
            models=models,
            spec=PlacementSpec(
                hbm_capacity_bytes=2 * 1024**3,
                hbf_capacity_bytes=ResolvedSystemConfig.load(
                    self._system_configs()
                ).resolve(self.simulator).logical_hbf_capacity_bytes,
                external_capacity_bytes=256 * 1024**3,
                hbm_runtime_reserve_bytes=4096,
                hbm_model_cache_bytes=len(cold_model.memory_objects) * 4096,
                model_weight_tiers={
                    "hbf_model": "hbf",
                    "cold_model": "external_cached_hbm",
                },
                object_tier_overrides={},
            ),
        )
        with HbfSimExecutor(
            simulator_path=self.simulator,
            system_config_paths=self._system_configs(),
            placement=placement,
        ) as executor:
            result = HBServeEngine(
                models=models,
                request_trace=trace,
                compiler=compiler,
                executor=executor,
                policy=SchedulerPolicy(
                    max_batch_requests=1,
                    max_batch_tokens=8,
                    prefill_chunk_tokens=8,
                ),
            ).run()
        final = executor.final_receipt()
        self.assertEqual(result["summary"]["requests"], 2)
        self.assertTrue(all(final["placement"]["final_invariants"].values()))
        self.assertTrue(
            result["eligibility"]["modeled_physical_memory_service"]
        )
        self.assertFalse(
            result["eligibility"]["absolute_memory_latency_claim"]
        )
        self.assertEqual(
            result["batches"][0]["remap"]["activation"]["cache_hit"],
            True,
        )
        self.assertEqual(
            result["batches"][1]["remap"]["activation"]["cache_hit"],
            False,
        )
        totals = final["session"]["final_measurement"][
            "device_workload_totals"
        ]
        self.assertGreater(totals["hbf"]["logical_read_bytes"], 0)
        self.assertGreater(totals["external"]["read_bytes"], 0)
        self.assertGreater(
            totals["hbm"]["read_bytes"] + totals["hbm"]["write_bytes"], 0
        )

    def test_roofline_compute_limit_and_native_memory_floor(self) -> None:
        if self.simulator is None:
            self.skipTest("no simulator supplied")
        # Hand-accounted small traffic keeps integration tests bounded. A full
        # 70B model cannot be asserted to attain the HBIO bandwidth ceiling:
        # bank sensing, internal transfer and Host work can set a lower rate.
        model = _dense()
        models = {model.model_id: model}
        paths = self._system_configs()
        system = ResolvedSystemConfig.load(paths).resolve(self.simulator)
        placement = derive_placement(
            system=system, models=models, preset="weights-hbf-kv-hbm")
        prompt = 32
        trace = _trace(RequestSpec("r0", 0.0, model.model_id, prompt, 2))
        times = []
        for peak_tflops in (0.000001, 1000.0):
            policy, timing, depth = __import__(
                "hbserve.io", fromlist=["x"]).run_config_from_dict({
                    "schema": {"name": "hbserve.run_config", "version": 2},
                    "scheduler": {"max_batch_requests": 1,
                                  "max_batch_tokens": prompt,
                                  "prefill_chunk_tokens": prompt},
                    "timing": {"type": "roofline", "prefetch_depth": 1},
                    "compute": {"peak_tflops": peak_tflops, "efficiency": 0.6},
                })
            experiment = run_experiment(
                simulator=self.simulator, system_config_paths=paths,
                models=models, request_trace=trace, router=None,
                placement_spec=placement, policy=policy, timing=timing,
                prefetch_depth=depth, input_artifacts={
                    "simulator": __import__("hbserve.run", fromlist=["x"])._artifact(self.simulator),
                    "system_configs": [__import__("hbserve.run", fromlist=["x"])._artifact(path)
                                       for path in paths]})
            batches = experiment["hbserve"]["batches"]
            self.assertEqual([batch["kind"] for batch in batches],
                             ["prefill", "decode"])
            elapsed = batches[0]["finish_ns"] - batches[0]["batch_origin_ns"]
            self.assertGreater(elapsed, 0)
            times.append(elapsed)
            totals = experiment["final_execution"]["session"][
                "final_measurement"]["device_workload_totals"]
            self.assertGreater(totals["hbf"]["logical_read_bytes"], 0)
            self.assertEqual(experiment["hbserve"]["summary"]["timing_model"],
                             "roofline")
            self.assertIn("mean_ttft_ns", experiment["hbserve"]["summary"])
        flops = sum(prompt * layer.flops_per_token +
                    layer.attention_flops_per_context_token * prompt *
                    (prompt + 1) // 2 for layer in model.layers)
        flops += model.lm_head_flops_per_token
        compute_ns = flops / (0.000001 * 1e12 * 0.6) * 1e9
        self.assertAlmostEqual(times[0] / compute_ns, 1.0, delta=0.10)
        self.assertGreater(times[0], times[1] * 10)


if __name__ == "__main__":
    # unittest.main's return object is not exposed through the helper above;
    # use a conventional program to preserve CTest's exit status.
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path)
    args, remaining = parser.parse_known_args()
    PhysicalIntegrationTests.simulator = (
        None if args.simulator is None else args.simulator.resolve()
    )
    PlacementTests.simulator = PhysicalIntegrationTests.simulator
    program = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
