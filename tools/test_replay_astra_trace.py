#!/usr/bin/env python3
"""Artifact-transaction regressions for replay_astra_trace.py."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "tools" / "replay_astra_trace.py"


FAKE_SCENARIO_COMPARE = """#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]

config = Path(value('--config')).name
barrier_dir = os.environ.get('PARALLEL_BARRIER_DIR')
if barrier_dir:
    barrier = Path(barrier_dir)
    barrier.mkdir(parents=True, exist_ok=True)
    (barrier / (config + '.ready')).write_text('ready\\n')
    deadline = time.monotonic() + 3.0
    while len(list(barrier.glob('*.ready'))) < 2:
        if time.monotonic() >= deadline:
            print('parallel barrier timed out', file=sys.stderr)
            raise SystemExit(8)
        time.sleep(0.01)
if os.environ.get('FAIL_CONFIG') == config:
    raise SystemExit(7)
scenarios = value('--scenarios').split(',')
summary = Path(value('--summary-json'))
resources = (
    'hbm_data_bus',
    'hbf_logic_ingress',
    'hbf_plane_media',
    'hbf_media_lane',
    'hbf_subarray',
    'hbf_page_buffer_bank',
    'hbf_flash_source_queue',
    'hbf_channel_command',
    'hbf_channel_data',
    'hbf_tsv',
    'hbf_sram',
    'hbf_hbio_command',
    'hbf_hbio_data',
    'hbf_sequencer',
    'hbf_ecc_issue',
    'hbf_mapping_dram_issue',
    'base_die_link_read',
    'base_die_link_write',
)
blocks = []
for scenario in scenarios:
    finish = 2.0 if os.environ.get('BAD_TIME_CONFIG') == config else 1.0
    blocks.append({
        'name': scenario,
        'ops': 1,
        'user_completion_throughput_GBps': 1.0,
        'makespan_throughput_GBps': 1.0,
        'time_breakdown': {
            'contract_version': 2,
            'semantics': {
                'wall_clock': 'additive_non_overlapping_spans',
                'latency_work': 'sum_across_user_operations_may_overlap',
                'stage_work': ('hierarchical_device_work_may_overlap_'
                               'do_not_sum_parent_and_children'),
                'controller_work': ('parent_totals_and_attribution_views_'
                                    'may_overlap'),
                'resource_busy': 'exclusive_per_resource_then_aggregated',
            },
            'wall_clock_ns': {
                'trace_origin_ns': 0.0,
                'last_offered_arrival_ns': 0.0,
                'last_user_completion_ns': 1.0,
                'quiescent_finish_ns': 1.0,
                'offered_arrival_span_ns': 0.0,
                'post_offer_user_completion_tail_ns': 1.0,
                'user_completion_span_ns': 1.0,
                'drain_tail_ns': 0.0,
                'makespan_ns': finish,
            },
            'latency_work': {
                'basis': 'offered_to_user_completion',
                'user_count': 1,
                'service_to_user_completion_sum_work_ns': 1.0,
                'front_end_admission_waited_ops': 0,
                'front_end_admission_wait_work_ns': 0.0,
                'front_end_admission_max_wait_ns': 0.0,
                'offered_to_user_completion_sum_work_ns': 1.0,
                'phase_barriers': 0,
                'phase_dependency_waited_ops': 0,
                'phase_dependency_wait_work_ns': 0.0,
                'phase_dependency_max_wait_ns': 0.0,
                'source_to_user_completion_sum_work_ns': 1.0,
                'average_ns': 1.0,
                'p50_ns': 1.0,
                'p95_ns': 1.0,
                'max_ns': 1.0,
                'service_average_ns': 1.0,
                'service_p50_ns': 1.0,
                'service_p95_ns': 1.0,
                'service_max_ns': 1.0,
                'source_average_ns': 1.0,
                'source_p50_ns': 1.0,
                'source_p95_ns': 1.0,
                'source_max_ns': 1.0,
            },
            'stage_work': {
                'scope': 'all_device_work_including_background_and_drain',
                'hbm': None,
                'hbf': None,
                'base_die_link': {
                    'read_queue_wait_work_ns': 0.0,
                    'write_queue_wait_work_ns': 0.0,
                    'read_serialization_work_ns': 0.0,
                    'write_serialization_work_ns': 0.0,
                    'read_fixed_latency_work_ns': 0.0,
                    'write_fixed_latency_work_ns': 0.0,
                },
                'layer_streaming_controller': {
                    'present': False,
                    'backing_admission_wait_work_ns': 0.0,
                    'backing_admission_max_wait_ns': 0.0,
                    'user_wait_work_ns': 0.0,
                    'user_max_wait_ns': 0.0,
                    'exposed_prefetch_ns': 0.0,
                    'hidden_prefetch_ns': 0.0,
                    'buffer_reuse_wait_work_ns': 0.0,
                },
                'cooperative_write_controller': {
                    'present': False,
                    'full_waited_ops': 0,
                    'full_wait_work_ns': 0.0,
                },
            },
            'resource_busy': {
                resource: {
                    'busy_ns': 0.0,
                    'resource_count': 0,
                    'active_span_ns': 0.0,
                    'capacity_time_ns': 0.0,
                    'utilization': 0.0,
                } for resource in resources
            },
        },
        'hbm_accesses': 0,
        'hbf_accesses': 0,
        'warnings': [],
        'hbf_stats': None,
    })
schema_version = int(os.environ.get('SUMMARY_SCHEMA_VERSION', '16'))
summary.write_text(json.dumps({
    'schema': {'name': 'hbfsim.scenario_compare.summary',
               'version': schema_version},
    'validation': {
        'status': 'exploratory_unattached',
        'certificate': None,
    },
    'sanity': 'PASS',
    'scenarios': blocks,
}))
"""


class ReplayTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binary = self.root / "fake_scenario_compare.py"
        self.binary.write_text(FAKE_SCENARIO_COMPARE)
        self.binary.chmod(0o755)
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        for name in ("usecase-6h2f.cfg", "usecase-4h4f.cfg"):
            (self.config_dir / name).write_text("# fake\n")
        self.generator = self.root / "generate.py"
        self.generator.write_text("# generator\n")
        self.gpu_configuration = self.root / "gpu.json"
        self.gpu_configuration.write_text("{}\n")
        self.et_artifact = self.root / "llm.0.et"
        self.et_artifact.write_bytes(b"et\n")
        self.model_profile = self.root / "fixture-small-bf16.json"
        profile_source = {
            "repository": "https://example.invalid/fixture-model",
            "path": "config.json",
            "revision": "fixture-revision",
            "upstream_sha256": None,
            "verification": "test_fixture",
        }
        profile = {
            "schema": {
                "name": "astra-sim.llm-model-profile",
                "version": 1,
            },
            "name": "fixture-small-bf16",
            "architecture": {
                "layers": 1,
                "hidden": 128,
                "heads": 8,
                "kv_heads": 2,
                "ffn": 256,
                "vocab_size": 256,
                "tie_word_embeddings": False,
                "max_seq": 64,
            },
            "precision": {
                "weight_dtype": "bfloat16",
                "weight_bytes": 2,
                "kv_dtype": "bfloat16",
                "kv_bytes": 2,
                "activation_dtype": "bfloat16",
                "activation_bytes": 2,
            },
            "source": profile_source,
        }
        self.model_profile.write_text(json.dumps(profile, sort_keys=True))
        profile_digest = hashlib.sha256(
            self.model_profile.read_bytes()).hexdigest()
        parameters = {
            "model_profile": "fixture-small-bf16",
            "model_profile_sha256": profile_digest,
            "mode": "decode",
            "layers": 1,
            "hidden": 128,
            "heads": 8,
            "kv_heads": 2,
            "ffn": 256,
            "vocab_size": 256,
            "tie_word_embeddings": False,
            "weight_dtype": "bfloat16",
            "weight_bytes": 2,
            "kv_dtype": "bfloat16",
            "kv_bytes": 2,
            "activation_dtype": "bfloat16",
            "activation_bytes": 2,
            "batch": 1,
            "prompt_len": 16,
            "decode_tokens": 2,
            "max_seq": 64,
            "tp": 1,
            "wg_per_kernel": 8,
            "wavefronts": 4,
        }
        identity = {
            "schema": "astra-sim.llm-fine-grained-workload-contract-v2",
            "generator_sha256": hashlib.sha256(
                self.generator.read_bytes()).hexdigest(),
            "gpu_configuration_sha256": hashlib.sha256(
                self.gpu_configuration.read_bytes()).hexdigest(),
            "collective_plan_id": None,
            "parameters": parameters,
        }
        self.workload_id = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        self.workload_manifest = self.root / "llm.manifest.json"
        self.workload_manifest.write_text(json.dumps({
            "schema": {
                "name": "astra-sim.llm-fine-grained-workload",
                "version": 3,
            },
            "workload_id": self.workload_id,
            "generator": {
                "path": str(self.generator),
                "bytes": self.generator.stat().st_size,
                "sha256": identity["generator_sha256"],
            },
            "model": {
                "profile": {
                    "name": "fixture-small-bf16",
                    "path": str(self.model_profile),
                    "bytes": self.model_profile.stat().st_size,
                    "sha256": profile_digest,
                    "source": profile_source,
                },
                "profile_exact": True,
                "profile_overrides": {},
                "architecture": "dense_decoder_transformer",
                "resolved_architecture": profile["architecture"],
                "resolved_precision": profile["precision"],
                "weight_scope": "full_resident_model",
                "parameter_count": 205_184,
                "logical_model_weight_bytes": 410_368,
                "decoder_block_parameters": 139_520,
                "per_rank_resident_weight_bytes": 410_368,
                "per_rank_decoder_block_weight_bytes": 279_040,
                "per_rank_weight_breakdown": {
                    "token_embedding": 65_536,
                    "decoder_input_norms": 256,
                    "decoder_attention_matrices": 81_920,
                    "decoder_post_attention_norms": 256,
                    "decoder_mlp_matrices": 196_608,
                    "final_norm": 256,
                    "lm_head_resident": 65_536,
                },
            },
            "parameters": parameters,
            "gpu_configuration": {
                "path": str(self.gpu_configuration),
                "bytes": self.gpu_configuration.stat().st_size,
                "sha256": identity["gpu_configuration_sha256"],
            },
            "address_space": {
                "weights": {
                    "base": 0x1000,
                    "end": 0x81000,
                    "layer_stride": 0x50000,
                    "logical_bytes": 410_368,
                    "logical_model_bytes": 410_368,
                    "allocated_bytes": 0x80000,
                    "objects": {
                        "token_embedding": {
                            "base": 0x1000,
                            "end": 0x11000,
                            "bytes": 65_536,
                        },
                        "decoder_layers": {
                            "base": 0x20000,
                            "end": 0x70000,
                            "layer_stride": 0x50000,
                            "logical_bytes_per_layer": 279_040,
                            "layers": 1,
                        },
                        "final_norm": {
                            "base": 0x70000,
                            "end": 0x70100,
                            "bytes": 256,
                        },
                        "lm_head": {
                            "base": 0x71000,
                            "end": 0x81000,
                            "bytes": 65_536,
                            "alias_of": None,
                        },
                    },
                },
                "kv": {
                    "base": 0x100000,
                    "end": 0x110000,
                    "layer_stride": 0x10000,
                    "bytes_per_batch_position_per_layer": 128,
                    "batch": 1,
                    "logical_capacity_bytes": 8_192,
                    "allocated_bytes": 0x10000,
                },
                "activations": {"base": 0x200000, "end": 0x210000},
                "communication": {"base": 0x300000, "end": 0x310000},
            },
            "initial_state": {
                "kv": {
                    "state_at_trace_start": "preexisting_prompt",
                    "required_at_trace_start": True,
                    "kind": "shared_context",
                    "prompt_positions_per_sequence": 16,
                    "batch": 1,
                    "bytes_per_layer": 2_048,
                    "total_bytes_per_rank": 2_048,
                    "base": 0x100000,
                    "layer_stride": 0x10000,
                },
            },
            "per_rank_weight_bytes": 410_368,
            "per_rank_kv_capacity_bytes": 8_192,
            "artifacts": [{
                "rank": 0,
                "path": str(self.et_artifact),
                "bytes": self.et_artifact.stat().st_size,
                "sha256": hashlib.sha256(
                    self.et_artifact.read_bytes()).hexdigest(),
                "compute_logical_read_bytes": 64,
                "compute_logical_write_bytes": 0,
                "compute_logical_cache_line_requests": 1,
                "stream_layers": 1,
            }],
            "mscclpp": None,
        }))
        self.trace = self.root / "trace.txt"
        self.trace.write_text(
            "# ASTRA-sim paper-aligned lossless memory-channel trace\n"
            "# schema=astra-sim.hbfsim-memory-trace version=2 "
            f"workload_id={self.workload_id} requests=1 bytes=64\n"
            "0x0 R 64\n")
        self.out = self.root / "out"
        self.out.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def command(self, jobs: int | None = None) -> list[str]:
        command = [
            sys.executable,
            "-B",
            str(REPLAY),
            "--trace",
            str(self.trace),
            "--workload-manifest",
            str(self.workload_manifest),
            "--scenario-compare",
            str(self.binary),
            "--config-dir",
            str(self.config_dir),
            "--out-dir",
            str(self.out),
            "--cases",
            "ec2,ec3",
        ]
        if jobs is not None:
            command.extend(["--jobs", str(jobs)])
        return command

    def test_later_failure_does_not_publish_earlier_summary(self) -> None:
        summaries = [
            self.out / "usecase-6h2f.summary.json",
            self.out / "usecase-4h4f.summary.json",
        ]
        for summary in summaries:
            summary.write_text("old-generation\n")
        environment = dict(os.environ, FAIL_CONFIG="usecase-4h4f.cfg")
        result = subprocess.run(
            self.command(jobs=2), capture_output=True, text=True, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([path.read_text() for path in summaries],
                         ["old-generation\n", "old-generation\n"])
        self.assertFalse((self.out / "astra-replay-ec2-ec3.csv").exists())
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.time-breakdown.csv").exists())
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.time-breakdown.md").exists())
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.time-breakdown.html").exists())
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.manifest.json").exists())
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_success_publishes_manifest_last(self) -> None:
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.out / "usecase-6h2f.summary.json").is_file())
        self.assertTrue((self.out / "usecase-4h4f.summary.json").is_file())
        self.assertTrue((self.out / "astra-replay-ec2-ec3.csv").is_file())
        time_csv = self.out / "astra-replay-ec2-ec3.time-breakdown.csv"
        time_markdown = self.out / "astra-replay-ec2-ec3.time-breakdown.md"
        time_html = self.out / "astra-replay-ec2-ec3.time-breakdown.html"
        self.assertTrue(time_csv.is_file())
        self.assertTrue(time_markdown.is_file())
        self.assertTrue(time_html.is_file())
        self.assertTrue(
            (self.out / "astra-replay-ec2-ec3.manifest.json").is_file())
        with (self.out / "astra-replay-ec2-ec3.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertIn("user_completion_throughput_GBps", rows[0])
        self.assertIn("hbf_ecc_queue_wait_work_ns", rows[0])
        self.assertNotIn("user_ack_throughput_GBps", rows[0])
        self.assertNotIn("hbf_ecc_queue_wait_ns", rows[0])
        for field in (
                "layer_streaming_layers",
                "layer_streaming_streamed_bytes",
                "layer_streaming_writeback_bytes",
                "layer_streaming_backing_request_credit_limit",
                "layer_streaming_backing_max_inflight_requests",
                "layer_streaming_backing_admission_waited_requests",
                "layer_streaming_backing_admission_wait_work_ns",
                "layer_streaming_exposed_prefetch_ns",
                "layer_streaming_hidden_prefetch_ns"):
            self.assertIn(field, rows[0])
        with time_csv.open(newline="") as handle:
            time_rows = list(csv.DictReader(handle))
        self.assertEqual({row["case"] for row in time_rows}, {"ec2", "ec3"})
        self.assertEqual(
            {(row["case"], row["config"]) for row in time_rows},
            {
                ("ec2", "usecase-6h2f.cfg"),
                ("ec3", "usecase-4h4f.cfg"),
            },
        )
        self.assertTrue(all(
            row["scenario"] == "HBM-HBF-Flat" for row in time_rows
        ))
        markdown = time_markdown.read_text()
        self.assertIn("Only the three wall-clock segments are additive", markdown)
        self.assertIn("usecase-6h2f", markdown)
        self.assertIn("usecase-4h4f", markdown)
        rendered = time_html.read_text()
        self.assertIn("HBFSim ASTRA replay timing", rendered)
        self.assertIn("Resource utilization and average parallelism", rendered)
        self.assertNotIn("<script", rendered)

        manifest_path = self.out / "astra-replay-ec2-ec3.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["schema"]["version"], 3)
        self.assertEqual(manifest["validation"], {
            "status": "exploratory_unattached",
            "certificate": None,
        })
        for summary_name in (
            "usecase-6h2f.summary.json",
            "usecase-4h4f.summary.json",
        ):
            summary = json.loads((self.out / summary_name).read_text())
            self.assertEqual(summary["validation"], manifest["validation"])
        expected_artifacts = {
            "usecase-6h2f.summary.json",
            "usecase-4h4f.summary.json",
            "astra-replay-ec2-ec3.csv",
            "astra-replay-ec2-ec3.time-breakdown.csv",
            "astra-replay-ec2-ec3.time-breakdown.md",
            "astra-replay-ec2-ec3.time-breakdown.html",
        }
        self.assertEqual(set(manifest["artifacts"]), expected_artifacts)
        for name, record in manifest["artifacts"].items():
            digest = hashlib.sha256((self.out / name).read_bytes()).hexdigest()
            self.assertEqual(record["sha256"], digest)

    def test_parallel_success_starts_configs_concurrently(self) -> None:
        barrier = self.root / "parallel-barrier"
        environment = dict(os.environ, PARALLEL_BARRIER_DIR=str(barrier))
        result = subprocess.run(
            self.command(jobs=2), capture_output=True, text=True,
            env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(barrier.glob("*.ready"))), 2)
        self.assertIn("[parallel] 2 config(s), jobs=2", result.stdout)
        self.assertIn("[done] usecase-6h2f.cfg", result.stdout)
        self.assertIn("[done] usecase-4h4f.cfg", result.stdout)
        self.assertTrue(
            (self.out / "astra-replay-ec2-ec3.manifest.json").is_file())

    def test_workload_manifest_derives_dynamic_kv_boundary(self) -> None:
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(
            (self.out / "astra-replay-ec2-ec3.manifest.json").read_text())
        self.assertEqual(manifest["schema"]["version"], 3)
        self.assertEqual(manifest["boundary"], 0x100000)
        self.assertEqual(manifest["boundary_source"],
                         "workload_manifest.kv.base")
        self.assertEqual(manifest["workload_manifest"]["sha256"],
                         hashlib.sha256(
                             self.workload_manifest.read_bytes()).hexdigest())
        self.assertEqual(manifest["workload_manifest"]["workload_id"],
                         self.workload_id)
        self.assertEqual(manifest["trace"]["workload_id"], self.workload_id)

    def test_trace_and_manifest_workload_ids_must_match(self) -> None:
        self.trace.write_text(
            "# schema=astra-sim.hbfsim-memory-trace version=2 "
            f"workload_id={'b' * 64} requests=1 bytes=64\n"
            "0x0 R 64\n")
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("workload_id does not match", result.stderr)
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.manifest.json").exists())

    def test_trace_census_and_generator_artifacts_fail_closed(self) -> None:
        self.trace.write_text(
            "# schema=astra-sim.hbfsim-memory-trace version=2 "
            f"workload_id={self.workload_id} requests=2 bytes=128\n"
            "0x0 R 64\n")
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not match its request/byte census", result.stderr)

        self.trace.write_text(
            "# schema=astra-sim.hbfsim-memory-trace version=2 "
            f"workload_id={self.workload_id} requests=1 bytes=64\n"
            "0x0 R 64\n")
        self.et_artifact.write_bytes(b"mutated\n")
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("ET artifact digest mismatch", result.stderr)
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.manifest.json").exists())

    def test_model_profile_and_weight_accounting_fail_closed(self) -> None:
        manifest = json.loads(self.workload_manifest.read_text())
        manifest["model"]["parameter_count"] += 1
        self.workload_manifest.write_text(json.dumps(manifest))
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("model parameter accounting is invalid", result.stderr)
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.manifest.json").exists())

        manifest["model"]["parameter_count"] -= 1
        self.workload_manifest.write_text(json.dumps(manifest))
        self.model_profile.write_text(
            self.model_profile.read_text() + "\n")
        result = subprocess.run(self.command(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("model profile digest mismatch", result.stderr)
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.manifest.json").exists())

    def test_jobs_must_be_positive(self) -> None:
        result = subprocess.run(
            self.command(jobs=0), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--jobs: must be at least 1", result.stderr)

    def test_bad_time_contract_preserves_summaries_and_withholds_manifest(self) -> None:
        summaries = [
            self.out / "usecase-6h2f.summary.json",
            self.out / "usecase-4h4f.summary.json",
        ]
        for summary_path in summaries:
            summary_path.write_text("old-generation\n")
        manifest = self.out / "astra-replay-ec2-ec3.manifest.json"
        manifest.write_text("old manifest\n")
        environment = dict(os.environ, BAD_TIME_CONFIG="usecase-4h4f.cfg")
        result = subprocess.run(
            self.command(), capture_output=True, text=True, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wall_clock_ns", result.stderr)
        self.assertEqual(
            [path.read_text() for path in summaries],
            ["old-generation\n", "old-generation\n"],
        )
        self.assertFalse(manifest.exists())
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.time-breakdown.csv").exists())
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.time-breakdown.md").exists())
        self.assertFalse(
            (self.out / "astra-replay-ec2-ec3.time-breakdown.html").exists())
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_out_dir_lock_rejects_concurrent_writer(self) -> None:
        lock_path = self.out / ".astra-replay.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(
                self.command(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("another replay is already using", result.stderr)

    def test_old_summary_schema_is_not_published(self) -> None:
        environment = dict(os.environ, SUMMARY_SCHEMA_VERSION="5")
        result = subprocess.run(
            self.command(), capture_output=True, text=True, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported or missing scenario summary schema", result.stderr)
        self.assertFalse((self.out / "usecase-6h2f.summary.json").exists())
        self.assertFalse((self.out / "astra-replay-ec2-ec3.csv").exists())


if __name__ == "__main__":
    unittest.main()
