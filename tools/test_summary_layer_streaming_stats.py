#!/usr/bin/env python3
"""Regression contract for hybrid-residency layer streaming."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path


SCENARIO = "HBM+HBF-layer-streaming"
EXTERNAL_SCENARIO = "HBM+External-layer-streaming"


def trace_identity_options(path: Path) -> list[str]:
    payload = path.read_bytes()
    return [
        "--expected-trace-sha256",
        hashlib.sha256(payload).hexdigest(),
        "--expected-trace-bytes",
        str(len(payload)),
    ]


COUNT_FIELDS = (
    "address_footprint_bytes",
    "unique_resident_footprint_pages",
    "unique_resident_footprint_bytes",
    "capacity_pressure_basis_bytes",
    "footprint_page_rounding_bytes",
    "hbm_capacity_bytes",
    "layers",
    "explicit_layer_requests",
    "explicit_compute_layers",
    "hbm_only_resident_pages",
    "hbm_only_resident_bytes",
    "hot_kv_candidate_pages",
    "hot_kv_candidate_bytes",
    "hot_kv_resident_pages",
    "hot_kv_resident_bytes",
    "data_pages",
    "data_bytes",
    "model_weight_resident_pages",
    "model_weight_resident_bytes",
    "model_weight_backing_pages",
    "model_weight_backing_bytes",
    "cold_kv_backing_pages",
    "cold_kv_backing_bytes",
    "unknown_backing_pages",
    "unknown_backing_bytes",
    "backing_unique_pages",
    "backing_unique_bytes",
    "resident_physical_pages",
    "resident_physical_bytes",
    "effective_layer_buffer_pages",
    "effective_layer_buffer_bytes",
    "unused_hbm_pages",
    "unused_hbm_bytes",
    "streamed_pages",
    "streamed_bytes",
    "foreground_resident_page_accesses",
    "foreground_buffer_page_accesses",
    "dirty_pages_written_back",
    "writeback_bytes",
    "max_layer_data_pages",
    "max_layer_data_bytes",
    "immutable_weight_logical_bytes",
    "runtime_overhead_logical_bytes",
    "block_table_logical_bytes",
    "active_buffer_logical_bytes_per_slot",
    "residency_page_size_bytes",
    "kv_block_stride_bytes",
    "logical_kv_blocks",
    "hot_kv_blocks",
    "cold_kv_blocks",
    "backing_request_credit_limit",
    "backing_max_inflight_requests",
    "backing_admission_waited_requests",
    "user_waited_ops",
)
TIME_FIELDS = (
    "compute_work_ns",
    "backing_admission_wait_work_ns",
    "backing_admission_max_wait_ns",
    "user_wait_work_ns",
    "user_max_wait_ns",
    "exposed_prefetch_ns",
    "hidden_prefetch_ns",
    "buffer_reuse_wait_work_ns",
)
CONTROLLER_TIME_FIELDS = tuple(
    field for field in TIME_FIELDS if field != "compute_work_ns"
)
REMOVED_TERMS = (
    "staging_mshr",
    "ready_fill_buffer",
    "install_queue",
    "semantic_staging",
    "prefetch_lookahead",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trace = root / "layer-streaming.trace"
        summary_json = root / "summary.json"
        summary_csv = root / "summary.csv"
        resolved_config = root / "resolved.cfg"
        trace.write_text(
            "0x100000 R 4096 shared_context layer=0 at=0\n"
            "0x101000 R 4096 shared_context layer=0 at=0\n"
            "0x102000 R 4096 shared_context layer=1 at=0\n"
            "0x103000 R 4096 shared_context layer=1 at=0\n"
            "0x104000 R 4096 shared_context layer=2 at=0\n"
            "0x105000 W 4096 generated_context layer=2 at=0\n"
            "0x106000 R 4096 shared_context layer=3 at=0\n"
            "0x107000 R 4096 shared_context layer=3 at=0\n"
            "0x108000 R 4096 shared_context layer=4 at=0\n"
            "0x109000 R 4096 shared_context layer=4 at=0\n"
            "0x10a000 R 8192 shared_context layer=5 at=0\n"
            "0x0 R 64 scratch layer=5 at=0\n",
            encoding="utf-8",
        )

        command = [
            str(args.scenario_compare),
            "--trace", str(trace),
            "--scenarios", SCENARIO,
            "--line-size", "64",
            "--hbf-page-size", "4096",
            "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "36864",
            "--layer-buffer-bytes", "8192",
            "--summary-json", str(summary_json),
            "--summary-csv", str(summary_csv),
            "--config-out", str(resolved_config),
        ]
        completed = run(command)
        if completed.returncode != 0:
            raise RuntimeError(
                "scenario_compare failed:\n" + completed.stdout + completed.stderr)

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        expected_schema = {
            "name": "hbfsim.scenario_compare.summary", "version": 16}
        if summary.get("schema") != expected_schema:
            raise AssertionError("unexpected summary schema")
        if summary.get("sanity") != "PASS":
            raise AssertionError("layer-streaming regression failed scenario sanity")
        config = summary.get("config") or {}
        if config.get("layer_buffer_bytes") != 8192:
            raise AssertionError("layer-buffer capacity did not round-trip")

        scenarios = {scenario["name"]: scenario
                     for scenario in summary.get("scenarios", [])}
        if set(scenarios) != {SCENARIO}:
            raise AssertionError("unexpected scenario set")
        streamed = scenarios[SCENARIO]
        stats = streamed.get("layer_streaming")
        if not isinstance(stats, dict):
            raise AssertionError("EC6 lacks layer_streaming metrics")
        for field in COUNT_FIELDS:
            if not isinstance(stats.get(field), int) or stats[field] < 0:
                raise AssertionError(f"invalid layer-streaming count: {field}")
        for field in TIME_FIELDS:
            value = stats.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise AssertionError(f"invalid layer-streaming time: {field}")
        if stats["layers"] != 6 or stats["explicit_layer_requests"] != 12:
            raise AssertionError("explicit layer grouping was not preserved")
        if stats.get("mode") != "layer_streaming":
            raise AssertionError("EC6 did not select layer streaming")
        if stats.get("residency_policy") != "trace-derived-first-touch":
            raise AssertionError("trace-derived residency policy is not explicit")
        if not stats.get("compact_resident_mapping"):
            raise AssertionError(
                "hybrid residency did not compact permanent HBM pages")
        if stats.get("semantic_inputs_consumed") is not True:
            raise AssertionError("hybrid planner ignored semantic inputs")
        if stats.get("explicit_residency_contract") is not False:
            raise AssertionError(
                "generic microbenchmark invented an explicit object contract")
        expected_pressure = 13 / 9
        if not math.isclose(
                stats.get("hbm_capacity_pressure", -1),
                expected_pressure,
                rel_tol=1e-12,
                abs_tol=0):
            raise AssertionError("capacity pressure is not unique-footprint/HBM")
        residency = (
            stats["hbm_only_resident_pages"],
            stats["hot_kv_candidate_pages"],
            stats["hot_kv_resident_pages"],
            stats["data_pages"],
            stats["backing_unique_pages"],
            stats["effective_layer_buffer_pages"],
        )
        if residency != (1, 11, 4, 12, 8, 2):
            raise AssertionError(f"unexpected hybrid residency plan: {residency}")
        if (stats["streamed_pages"], stats["dirty_pages_written_back"]) != (7, 1):
            raise AssertionError(
                "full-page write did not skip input or dirty writeback drifted")
        if stats["streamed_bytes"] != stats["streamed_pages"] * 4096:
            raise AssertionError("streamed page bytes do not conserve")
        if stats["writeback_bytes"] != stats["dirty_pages_written_back"] * 4096:
            raise AssertionError("writeback page bytes do not conserve")
        if stats["max_layer_data_bytes"] != 8192:
            raise AssertionError("a layer exceeded one ping-pong buffer")
        foreground_pages = (
            stats["foreground_resident_page_accesses"],
            stats["foreground_buffer_page_accesses"],
        )
        if foreground_pages != (5, 8):
            raise AssertionError(
                f"resident/buffer foreground split drifted: {foreground_pages}")
        if stats["exposed_prefetch_ns"] <= 0 or stats["hidden_prefetch_ns"] <= 0:
            raise AssertionError("two-layer trace did not exercise exposed and hidden DMA")
        if streamed["hbm_user_accesses"] != sum(foreground_pages):
            raise AssertionError(
                "EC6 foreground page transactions did not all execute from HBM")
        if streamed["hbf_user_accesses"] != 0:
            raise AssertionError("EC6 exposed HBF as a foreground user path")
        hybrid = streamed.get("hybrid_path") or {}
        backing_read_bytes = (
            hybrid.get("hbf_static_read_bytes", 0) +
            hybrid.get("hbf_logical_read_bytes", 0)
        )
        if not (
            backing_read_bytes == stats["streamed_bytes"]
            == hybrid.get("base_die_link_read_bytes")
            == hybrid.get("hbm_streaming_write_bytes")
        ):
            raise AssertionError("HBF-to-D2D-to-HBM bytes do not conserve")
        if (hybrid.get("hbf_static_read_bytes"),
                hybrid.get("hbf_logical_read_bytes")) != (0, 28672):
            raise AssertionError(
                "mutable KV did not use logical HBF from its initial version")
        if not (
            stats["writeback_bytes"] == hybrid.get("base_die_link_write_bytes")
            == hybrid.get("hbf_backing_write_bytes")
        ):
            raise AssertionError("HBM-to-D2D-to-HBF bytes do not conserve")
        regions = {
            (region["name"], region["kind"])
            for domain in streamed["address_heatmap"]["domains"]
            if domain["domain"] == "hbm_physical"
            for region in domain["regions"]
        }
        if not {
            ("layer_buffer_0", "layer_buffer"),
            ("layer_buffer_1", "layer_buffer"),
        } <= regions:
            raise AssertionError("both HBM layer-buffer regions are not explicit")

        controller = streamed["time_breakdown"]["stage_work"].get(
            "layer_streaming_controller")
        if not isinstance(controller, dict) or not controller.get("present"):
            raise AssertionError("layer-streaming controller work is absent")
        for field in CONTROLLER_TIME_FIELDS:
            if field not in controller:
                raise AssertionError(f"controller work lacks {field}")

        serialized = json.dumps(summary, sort_keys=True)
        for term in REMOVED_TERMS:
            if term in serialized:
                raise AssertionError(f"removed page-cache concept leaked: {term}")

        with summary_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if any(row.get("summary_schema_version") != "16" for row in rows):
            raise AssertionError("CSV schema version is not v16")
        row = {entry["scenario"]: entry for entry in rows}[SCENARIO]
        if row.get("layer_streaming_layers") != "6":
            raise AssertionError("CSV layer count disagrees with JSON")
        if row.get("layer_streaming_residency_policy") != "trace-derived-first-touch":
            raise AssertionError("CSV omitted the hybrid residency policy")

        # A layer larger than the declared completion-order window must queue
        # backing DMA pages and report the exact enforced concurrency bound.
        bounded = root / "bounded-backing.trace"
        bounded.write_text(
            "".join(
                f"0x{0x400000 + page * 4096:x} R 4096 "
                "model_weights layer=0 at=0\n"
                for page in range(8)
            ),
            encoding="utf-8",
        )
        bounded_summary = root / "bounded-backing.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(bounded),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", str(16 * 4096),
            "--layer-buffer-bytes", str(8 * 4096),
            "--max-outstanding-requests", "2",
            "--summary-json", str(bounded_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        bounded_stats = json.loads(
            bounded_summary.read_text(encoding="utf-8")
        )["scenarios"][0]["layer_streaming"]
        bounded_credit = (
            bounded_stats["backing_request_credit_limit"],
            bounded_stats["backing_max_inflight_requests"],
            bounded_stats["backing_admission_waited_requests"],
        )
        if bounded_credit != (2, 2, 6):
            raise AssertionError(
                f"backing DMA escaped its declared credit pool: "
                f"{bounded_credit}")
        if (
            bounded_stats["backing_admission_wait_work_ns"] <= 0.0
            or bounded_stats["backing_admission_max_wait_ns"] <= 0.0
            or bounded_stats["backing_admission_max_wait_ns"]
            > bounded_stats["backing_admission_wait_work_ns"]
        ):
            raise AssertionError(
                "bounded backing DMA did not expose conserved admission wait")

        config_text = resolved_config.read_text(encoding="utf-8")
        if "layer-buffer-bytes=8192" not in config_text:
            raise AssertionError("resolved config omitted layer-buffer-bytes")
        for term in (
            "prefetch-buffer-lines", "prefetch-lookahead-ops",
            "staging-mshr-limit", "staging-ready-fill-buffer-limit",
            "staging-install-queue-limit"):
            if term in config_text:
                raise AssertionError(f"resolved config retained obsolete key {term}")

        # Weights and cold/generated KV remain backing-tier data even when the
        # complete footprint could fit in HBM; only prefix-qualified hot KV is
        # eligible for permanent residency.
        # Independent phase accounting must remain identical to all-HBM.
        hbm_fit = root / "hbm-fit.trace"
        hbm_fit.write_text(
            "0x0 R 64 model_weights phase=0 layer=0 at=0\n"
            "0x1000 W 64 generated_context phase=1 layer=0 at=0\n",
            encoding="utf-8",
        )
        hbm_fit_summary = root / "hbm-fit.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(hbm_fit),
            "--scenarios", f"all-HBM,{SCENARIO}",
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "65536",
            "--layer-buffer-bytes", "8192",
            "--summary-json", str(hbm_fit_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        hbm_fit_scenarios = {
            scenario["name"]: scenario
            for scenario in json.loads(
                hbm_fit_summary.read_text(encoding="utf-8"))["scenarios"]
        }
        hbm_fit_ec6 = hbm_fit_scenarios[SCENARIO]
        hbm_fit_stats = hbm_fit_ec6["layer_streaming"]
        if hbm_fit_stats["mode"] != "layer_streaming":
            raise AssertionError("HBM-fit data bypassed EC6 layer streaming")
        if hbm_fit_stats["layers"] != 1:
            raise AssertionError("EC6 conflated phase identity with layer identity")
        if (
            hbm_fit_stats["data_pages"],
            hbm_fit_stats["effective_layer_buffer_pages"],
        ) != (2, 2):
            raise AssertionError("HBM-fit data did not use the two-buffer plan")
        if not all((
            hbm_fit_ec6["hbf_accesses"],
            hbm_fit_ec6["hbm_background_accesses"],
            hbm_fit_ec6["hybrid_path"]["base_die_link_read_bytes"],
        )):
            raise AssertionError("HBM-fit data bypassed HBF-to-HBM staging")
        hbm_fit_latency = hbm_fit_ec6["time_breakdown"]["latency_work"]
        all_hbm_latency = hbm_fit_scenarios["all-HBM"][
            "time_breakdown"]["latency_work"]
        for field in ("phase_barriers", "phase_dependency_waited_ops"):
            if hbm_fit_latency[field] != all_hbm_latency[field]:
                raise AssertionError(
                    f"EC6 phase accounting diverged from all-HBM: {field}")
        for field in ("phase_dependency_wait_work_ns",
                      "phase_dependency_max_wait_ns"):
            if hbm_fit_latency[field] < all_hbm_latency[field]:
                raise AssertionError(
                    f"EC6 tiering made causal phase wait smaller: {field}")
        if hbm_fit_latency["phase_barriers"] != 1 or \
                hbm_fit_latency["phase_dependency_waited_ops"] != 1:
            raise AssertionError("EC6 did not enforce the independent phase barrier")
        hbm_fit_regions = {
            region["name"]
            for domain in hbm_fit_ec6["address_heatmap"]["domains"]
            if domain["domain"] == "hbm_physical"
            for region in domain["regions"]
        }
        if "layer_buffer_0" not in hbm_fit_regions or \
                "layer_buffer_1" not in hbm_fit_regions:
            raise AssertionError(
                "HBM-fit data did not expose both active layer buffers")

        # A page first observed as generated/cold KV cannot become a pinned
        # resident merely because a future access is prefix-cache eligible.
        # This rejects a future-looking hotness oracle.
        causal = root / "first-touch-causal.trace"
        causal.write_text(
            "0x100000 W 4096 generated_context layer=0 at=0\n"
            "0x100000 R 4096 shared_context layer=1 at=0\n"
            "0x200000 R 4096 shared_context layer=1 at=0\n",
            encoding="utf-8",
        )
        causal_summary = root / "first-touch-causal.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(causal),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", str(5 * 4096),
            "--layer-buffer-bytes", str(2 * 4096),
            "--summary-json", str(causal_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        causal_stats = json.loads(
            causal_summary.read_text(encoding="utf-8")
        )["scenarios"][0]["layer_streaming"]
        causal_plan = (
            causal_stats["unique_resident_footprint_pages"],
            causal_stats["hot_kv_candidate_pages"],
            causal_stats["hot_kv_resident_pages"],
            causal_stats["cold_kv_backing_pages"],
            causal_stats["backing_unique_pages"],
        )
        if causal_plan != (2, 1, 1, 1, 1):
            raise AssertionError(
                f"first-touch hot-KV policy used future information: "
                f"{causal_plan}")
        if not math.isclose(
                causal_stats["hbm_capacity_pressure"],
                2 / 5,
                rel_tol=1e-12,
                abs_tol=0):
            raise AssertionError("repeated traffic inflated unique footprint")

        # Sparse high-address scratch/metadata is compacted into HBM, while the
        # data page still uses HBF-backed layer streaming.
        high_metadata = root / "high-metadata.trace"
        high_metadata.write_text(
            "0x200000 R 4096 model_weights layer=0 at=0\n"
            "0x90000000 R 64 metadata layer=0 at=0\n",
            encoding="utf-8",
        )
        high_metadata_summary = root / "high-metadata.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(high_metadata),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "65536",
            # This is an upper bound, not a reservation.
            "--layer-buffer-bytes", "65536",
            "--summary-json", str(high_metadata_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        high_metadata_result = json.loads(
            high_metadata_summary.read_text(encoding="utf-8"))["scenarios"][0]
        high_metadata_stats = high_metadata_result["layer_streaming"]
        if (
            high_metadata_stats["mode"],
            high_metadata_stats["compact_resident_mapping"],
            high_metadata_stats["hbm_only_resident_pages"],
            high_metadata_stats["data_pages"],
        ) != ("layer_streaming", True, 1, 1):
            raise AssertionError("HBM-only compact residency is incorrect")
        if not all((
            high_metadata_result["hbf_accesses"],
            high_metadata_result["hbm_background_accesses"],
            high_metadata_result["hybrid_path"]["base_die_link_read_bytes"],
        )):
            raise AssertionError(
                "sparse HBM-fit data bypassed layer streaming")

        # Explicit compute overlaps next-layer DMA and extends the execution
        # frontier. Missing compute remains a separately reported memory-only
        # run rather than inheriting an invented duration.
        compute_trace = root / "compute.trace"
        compute_trace.write_text(
            "0x1000 R 4096 model_weights layer=0 at=0\n"
            "0x2000 R 4096 model_weights layer=1 compute_ns=1000000 at=0\n"
            "0xa00000 R 4096 model_weights layer=2 at=0\n"
            "0xb00000 R 4096 model_weights layer=3 at=0\n"
            "0xc00000 R 4096 model_weights layer=4 at=0\n"
            "0xd00000 R 4096 model_weights layer=5 at=0\n",
            encoding="utf-8",
        )
        compute_summary = root / "compute.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(compute_trace),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "16384",
            "--layer-buffer-bytes", "4096",
            "--summary-json", str(compute_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        compute_result = json.loads(
            compute_summary.read_text(encoding="utf-8"))["scenarios"][0]
        compute_stats = compute_result["layer_streaming"]
        if (
            compute_stats["explicit_compute_layers"],
            compute_stats["compute_work_ns"],
        ) != (1, 1000000):
            raise AssertionError("explicit EC6 compute interval was not preserved")
        if compute_result["time_breakdown"]["wall_clock_ns"][
                "user_completion_span_ns"] < 1000000:
            raise AssertionError(
                "explicit compute did not extend EC6 completion")
        if compute_stats["hidden_prefetch_ns"] <= 0:
            raise AssertionError(
                "explicit compute did not overlap next-layer prefetch")
        if (
            compute_stats["data_pages"],
            compute_stats["effective_layer_buffer_pages"],
        ) != (6, 1):
            raise AssertionError(
                "compute-overlap trace did not exercise backed weights")

        # Plain R/W remains a first-class one-window input.
        plain = root / "plain.trace"
        plain.write_text("0x400000 R 4096 at=0\n", encoding="utf-8")
        plain_summary = root / "plain.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(plain),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "65536",
            "--layer-buffer-bytes", "4096",
            "--summary-json", str(plain_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        plain_stats = json.loads(plain_summary.read_text(encoding="utf-8"))[
            "scenarios"][0]["layer_streaming"]
        if plain_stats["layers"] != 1 or plain_stats["explicit_layer_requests"] != 0:
            raise AssertionError("plain trace fallback is not one streaming window")

        # User access census is physical page transactions. A no-op HBF drain
        # is not an access.
        scratch_only = root / "scratch-only.trace"
        scratch_only.write_text(
            "0x0 R 64 scratch layer=0 at=0\n", encoding="utf-8")
        scratch_summary = root / "scratch-only.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(scratch_only),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "65536",
            "--layer-buffer-bytes", "4096",
            "--summary-json", str(scratch_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        scratch_result = json.loads(
            scratch_summary.read_text(encoding="utf-8"))["scenarios"][0]
        scratch_split = (
            scratch_result["hbm_user_accesses"],
            scratch_result["hbm_background_accesses"],
            scratch_result["hbf_user_accesses"],
            scratch_result["hbf_background_accesses"],
        )
        if scratch_split != (1, 0, 0, 0):
            raise AssertionError(
                f"no-op HBF drain polluted access census: {scratch_split}")

        # Reusing parity 0 for layer 4 must wait for layer 2's complete HBM ->
        # D2D -> HBF writeback. Every data layer occupies a ping-pong buffer.
        reuse = root / "reuse-fence.trace"
        reuse.write_text(
            "0x1000 R 4096 model_weights layer=0 at=0\n"
            "0x2000 R 4096 model_weights layer=1 at=0\n"
            "0x800000 W 4096 generated_context layer=2 at=0\n"
            "0x940000 R 4096 model_weights layer=3 at=0\n"
            "0xa00000 R 4096 model_weights layer=4 at=0\n"
            "0xb00000 R 4096 model_weights layer=5 at=0\n",
            encoding="utf-8",
        )
        reuse_summary = root / "reuse-fence.json"
        reuse_chrome = root / "reuse-fence.chrome.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(reuse),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "16384",
            "--layer-buffer-bytes", "4096",
            "--base-die-link-write-bw", "0.01",
            "--hbf-write-buffer-completion-requires-flush", "true",
            "--summary-json", str(reuse_summary),
            "--chrome-trace", str(reuse_chrome),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        reuse_result = json.loads(
            reuse_summary.read_text(encoding="utf-8"))["scenarios"][0]
        reuse_wait_ns = reuse_result[
            "layer_streaming"]["buffer_reuse_wait_work_ns"]
        if reuse_wait_ns <= 0:
            raise AssertionError(
                "same-parity layer did not wait for dirty writeback completion; "
                f"stats={reuse_result['layer_streaming']}")

        events = [
            event for event in json.loads(
                reuse_chrome.read_text(encoding="utf-8"))["traceEvents"]
            if event.get("ph") == "X"
        ]

        def request_bounds(prefix: str, suffix: str) -> tuple[float, float]:
            matches = [
                event for event in events
                if str(event.get("args", {}).get("request", "")).startswith(prefix)
                and str(event.get("args", {}).get("request", "")).endswith(suffix)
            ]
            if not matches:
                raise AssertionError(
                    f"Chrome trace lacks request {prefix}*{suffix}")
            return (
                min(float(event["ts"]) for event in matches),
                max(float(event["ts"]) + float(event["dur"])
                    for event in matches),
            )

        _, layer2_writeback_end = request_bounds(
            "layer2/", "/hbf-writeback")
        layer4_install_start, _ = request_bounds("layer4/", "/hbm-install")
        if layer4_install_start + 1e-6 < layer2_writeback_end:
            raise AssertionError(
                "parity buffer was overwritten before prior HBF writeback completed")

        # Read-before-overwrite must fetch the old page, while a first-touch
        # full-page overwrite must not. Same-page operations are dependency
        # chained inside a layer.
        ordering = root / "ordering.trace"
        ordering.write_text(
            "0x1000 R 4096 model_weights layer=0 at=0\n"
            "0x2000 R 4096 model_weights layer=1 at=0\n"
            "0x500000 R 4096 generated_context layer=2 at=0\n"
            "0x500000 W 4096 generated_context layer=2 at=0\n"
            "0x600000 W 4096 generated_context layer=3 at=0\n"
            "0x600000 W 4096 generated_context layer=4 at=0\n"
            "0x600000 R 4096 generated_context layer=5 at=0\n"
            "0x700000 R 4096 model_weights layer=6 at=0\n"
            "0x800000 R 4096 model_weights layer=7 at=0\n"
            "0x900000 R 4096 model_weights layer=8 at=0\n"
            "0xa00000 R 4096 model_weights layer=9 at=0\n",
            encoding="utf-8",
        )
        ordering_summary = root / "ordering.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(ordering),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "16384",
            "--layer-buffer-bytes", "4096",
            "--summary-json", str(ordering_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        ordering_stats = json.loads(ordering_summary.read_text(encoding="utf-8"))[
            "scenarios"][0]["layer_streaming"]
        if ordering_stats["streamed_pages"] != 8:
            raise AssertionError("first-touch overwrite/read prefetch rule is incorrect")
        if ordering_stats["dirty_pages_written_back"] != 3:
            raise AssertionError("versioned same-page writebacks were lost")
        ordering_hybrid = json.loads(
            ordering_summary.read_text(encoding="utf-8"))["scenarios"][0][
                "hybrid_path"]
        if (ordering_hybrid["hbf_static_read_bytes"],
                ordering_hybrid["hbf_logical_read_bytes"]) != (24576, 8192):
            raise AssertionError(
                "immutable weights and mutable KV used the wrong HBF surfaces")

        # A production object contract reserves the complete population even
        # when this trace window touches only one weight page, one of three KV
        # slots, and a small portion of runtime metadata. Capacity pressure is
        # byte-exact; physical object/page rounding is reported separately.
        contracted = root / "explicit-residency.trace"
        contracted.write_text(
            "0x0 R 4096 model_weights layer=0 at=0\n"
            "0x2000 R 4096 shared_context layer=0 at=0\n"
            "0x4000 W 4096 generated_context layer=1 at=0\n"
            "0x200000 R 64 metadata layer=1 at=0\n",
            encoding="utf-8",
        )
        contracted_summary = root / "explicit-residency.json"
        contract_options = [
            "--explicit-residency-contract", "true",
            "--residency-page-size-bytes", "4096",
            "--residency-unique-footprint-bytes", "34584",
            "--residency-immutable-weight-bytes", "5000",
            "--residency-immutable-weight-pages", "2",
            "--residency-static-weight-pages", "1",
            "--residency-runtime-overhead-bytes", "5000",
            "--residency-block-table-bytes", "8",
            "--residency-active-buffer-bytes", "4096",
            "--residency-kv-region-begin", str(0x2000),
            "--residency-kv-block-stride-bytes", "8192",
            "--residency-logical-kv-blocks", "3",
            "--residency-hot-kv-blocks", "1",
        ]
        completed = run([
            str(args.scenario_compare), "--trace", str(contracted),
            "--scenarios", f"{SCENARIO},{EXTERNAL_SCENARIO}",
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--external-backing-page-size", "4096",
            "--hbm-capacity-bytes", "32768",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(contracted),
            *contract_options,
            "--summary-json", str(contracted_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        contracted_document = json.loads(
            contracted_summary.read_text(encoding="utf-8"))
        if contracted_document["sanity"] != "PASS":
            raise AssertionError("explicit residency contract failed sanity")
        if contracted_document["config"]["residency_contract"] != {
            "page_size_bytes": 4096,
            "unique_resident_footprint_bytes": 34584,
            "immutable_weight_bytes": 5000,
            "immutable_weight_pages": 2,
            "static_weight_resident_pages": 1,
            "runtime_overhead_bytes": 5000,
            "block_table_bytes": 8,
            "active_buffer_bytes_per_slot": 4096,
            "kv_region_begin": 0x2000,
            "kv_block_stride_bytes": 8192,
            "logical_kv_blocks": 3,
            "hot_kv_blocks": 1,
        }:
            raise AssertionError(
                "resolved explicit residency contract did not round-trip")
        contracted_scenarios = {
            scenario["name"]: scenario
            for scenario in contracted_document["scenarios"]
        }
        if set(contracted_scenarios) != {SCENARIO, EXTERNAL_SCENARIO}:
            raise AssertionError("explicit contract scenario set drifted")
        contract_fields = (
            "explicit_residency_contract",
            "unique_resident_footprint_pages",
            "unique_resident_footprint_bytes",
            "capacity_pressure_basis_bytes",
            "footprint_page_rounding_bytes",
            "hbm_only_resident_pages",
            "hot_kv_candidate_pages",
            "hot_kv_resident_pages",
            "model_weight_resident_pages",
            "model_weight_backing_pages",
            "cold_kv_backing_pages",
            "backing_unique_pages",
            "resident_physical_pages",
            "effective_layer_buffer_pages",
            "unused_hbm_pages",
            "immutable_weight_logical_bytes",
            "runtime_overhead_logical_bytes",
            "block_table_logical_bytes",
            "active_buffer_logical_bytes_per_slot",
            "residency_page_size_bytes",
            "kv_block_stride_bytes",
            "logical_kv_blocks",
            "hot_kv_blocks",
            "cold_kv_blocks",
        )
        hbf_contract = contracted_scenarios[SCENARIO]["layer_streaming"]
        external_contract = contracted_scenarios[
            EXTERNAL_SCENARIO]["layer_streaming"]
        if any(
                hbf_contract[field] != external_contract[field]
                for field in contract_fields):
            raise AssertionError(
                "HBF and external backing consumed different object contracts")
        expected_contract = (
            True,
            11, 45056, 34584, 10472,
            3, 6, 2, 1, 1, 4, 5, 6, 1, 0,
            5000, 5000, 8, 4096, 4096, 8192, 3, 1, 2,
        )
        actual_contract = tuple(
            hbf_contract[field] for field in contract_fields)
        if actual_contract != expected_contract:
            raise AssertionError(
                f"explicit residency accounting drifted: {actual_contract}")
        if not math.isclose(
                hbf_contract["hbm_capacity_pressure"],
                34584 / 32768,
                rel_tol=1e-12,
                abs_tol=0):
            raise AssertionError(
                "capacity pressure used trace pages instead of exact objects")

        all_weight_resident_options = list(contract_options)
        static_weight_index = all_weight_resident_options.index(
            "--residency-static-weight-pages"
        ) + 1
        all_weight_resident_options[static_weight_index] = "2"
        all_weight_resident_summary = root / "all-weight-resident.json"
        completed = run([
            str(args.scenario_compare), "--trace", str(contracted),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "36864",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(contracted),
            *all_weight_resident_options,
            "--summary-json", str(all_weight_resident_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        all_weight_resident = json.loads(
            all_weight_resident_summary.read_text(encoding="utf-8")
        )["scenarios"][0]
        if (
            all_weight_resident["layer_streaming"][
                "model_weight_resident_pages"
            ],
            all_weight_resident["layer_streaming"][
                "model_weight_backing_pages"
            ],
            all_weight_resident["hybrid_path"]["hbf_static_read_bytes"],
            all_weight_resident["hbf_stats"]["static_reserved_pages"],
        ) != (2, 0, 0, 0):
            raise AssertionError(
                "fully resident immutable weights still consumed HBF")

        missing_contract_field = run([
            str(args.scenario_compare), "--trace", str(contracted),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "32768",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(contracted),
            *contract_options[:-2],
        ])
        if (
            missing_contract_field.returncode == 0
            or "requires every residency-* field"
            not in missing_contract_field.stderr
        ):
            raise AssertionError(
                "partial explicit residency contract did not fail closed")

        wrong_population = run([
            str(args.scenario_compare), "--trace", str(contracted),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "32768",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(contracted),
            *contract_options,
            "--residency-unique-footprint-bytes", "34585",
        ])
        if (
            wrong_population.returncode == 0
            or "unique footprint is not byte-conservative"
            not in wrong_population.stderr
        ):
            raise AssertionError(
                "non-conservative explicit footprint did not fail closed")

        outside_kv = root / "outside-contract-kv.trace"
        outside_kv.write_text(
            "0x0 R 4096 model_weights layer=0 at=0\n"
            "0x300000 R 4096 shared_context layer=0 at=0\n",
            encoding="utf-8",
        )
        rejected = run([
            str(args.scenario_compare), "--trace", str(outside_kv),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "32768",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(outside_kv),
            *contract_options,
        ])
        if (
            rejected.returncode == 0
            or "KV trace page lies outside" not in rejected.stderr
        ):
            raise AssertionError(
                "KV access outside the explicit arena did not fail closed")

        outside_weight = root / "outside-contract-weight.trace"
        outside_weight.write_text(
            "0x300000 R 4096 model_weights layer=0 at=0\n",
            encoding="utf-8",
        )
        rejected = run([
            str(args.scenario_compare), "--trace", str(outside_weight),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "32768",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(outside_weight),
            *contract_options,
        ])
        if (
            rejected.returncode == 0
            or "outside the explicit immutable-weight prefix"
            not in rejected.stderr
        ):
            raise AssertionError(
                "model weight outside the immutable prefix did not fail closed")

        metadata_alias = root / "metadata-aliases-weight.trace"
        metadata_alias.write_text(
            "0x0 R 64 metadata layer=0 at=0\n",
            encoding="utf-8",
        )
        rejected = run([
            str(args.scenario_compare), "--trace", str(metadata_alias),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", "32768",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(metadata_alias),
            *contract_options,
        ])
        if (
            rejected.returncode == 0
            or "aliases the explicit immutable weight or KV population"
            not in rejected.stderr
        ):
            raise AssertionError(
                "HBM-only metadata aliasing the object population did not "
                "fail closed")

        # Untouched immutable pages must fence their complete NAND blocks
        # before mutable cold KV can allocate through the FTL.
        full_backing = root / "complete-backing-population.trace"
        full_backing.write_text(
            "0x0 R 4096 model_weights layer=0 at=0\n"
            "0xa000 R 4096 generated_context layer=1 at=0\n"
            "0xa000 W 4096 generated_context layer=1 at=0\n",
            encoding="utf-8",
        )
        full_backing_summary = root / "complete-backing-population.json"
        full_backing_contract = [
            "--explicit-residency-contract", "true",
            "--residency-page-size-bytes", "4096",
            "--residency-unique-footprint-bytes", "61444",
            "--residency-immutable-weight-bytes", str(9 * 4096),
            "--residency-immutable-weight-pages", "9",
            "--residency-static-weight-pages", "0",
            "--residency-runtime-overhead-bytes", "4096",
            "--residency-block-table-bytes", "4",
            "--residency-active-buffer-bytes", "4096",
            "--residency-kv-region-begin", str(9 * 4096),
            "--residency-kv-block-stride-bytes", "4096",
            "--residency-logical-kv-blocks", "5",
            "--residency-hot-kv-blocks", "1",
        ]
        small_hbf_geometry = [
            "--hbf-stacks", "1",
            "--hbf-channels", "1",
            "--hbf-dies-per-channel", "1",
            "--hbf-planes-per-die", "1",
            "--hbf-pages-per-block", "4",
            "--hbf-blocks-per-plane", "64",
            "--hbf-page-size", "4096",
            "--hbf-oob-bytes", "224",
            "--hbf-gc-low-watermark-pages", "0",
            "--hbf-gc-hard-watermark-pages", "0",
        ]
        completed = run([
            str(args.scenario_compare), "--trace", str(full_backing),
            "--scenarios", SCENARIO,
            "--hbm-capacity-bytes", "32768",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(full_backing),
            *small_hbf_geometry,
            *full_backing_contract,
            "--summary-json", str(full_backing_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        full_backing_result = json.loads(
            full_backing_summary.read_text(encoding="utf-8")
        )["scenarios"][0]
        if (
            full_backing_result["hbf_stats"]["static_reserved_pages"],
            full_backing_result["hbf_stats"]["static_unmaterialized_pages"],
        ) != (12, 11):
            raise AssertionError(
                "untouched immutable weight blocks were not physically fenced")
        hbf_initial = full_backing_result["hbf_stats"]
        if (
            hbf_initial["initial_logical_data_pages"],
            hbf_initial["initial_mapping_pages"],
            hbf_initial["compact_initial_logical_data_pages"],
            hbf_initial["compact_initial_mapping_pages"],
            hbf_initial["compact_live_logical_data_pages"],
            hbf_initial["compact_live_mapping_pages"],
            hbf_initial["compact_retired_logical_data_pages"],
            hbf_initial["compact_retired_mapping_pages"],
        ) != (4, 1, 4, 1, 3, 0, 1, 1):
            raise AssertionError(
                "complete cold-KV initial population was not physically "
                "occupied and retired through the mutable FTL")
        if (
            hbf_initial["compact_live_logical_data_pages"]
            + hbf_initial["compact_retired_logical_data_pages"]
            != full_backing_result["layer_streaming"][
                "cold_kv_backing_pages"
            ]
        ):
            raise AssertionError(
                "cold-KV contract and compact FTL population diverged")
        if (
            full_backing_result["hybrid_path"]["hbf_static_read_bytes"],
            full_backing_result["hybrid_path"]["hbf_logical_read_bytes"],
        ) != (4096, 4096):
            raise AssertionError(
                "immutable weights and mutable KV did not use separate HBF "
                "surfaces")

        # A capacity-aware resident prefix may begin or end in the middle of
        # a global HBF block coordinate. Both edge blocks still need complete
        # physical fencing; otherwise the mutable FTL can allocate over an
        # untouched immutable page in the trailing partial block.
        partial_prefix = root / "partial-weight-prefix.trace"
        partial_prefix.write_text(
            "0x2000 R 4096 model_weights layer=0 at=0\n"
            "0x7000 R 4096 generated_context layer=1 at=0\n"
            "0x7000 W 4096 generated_context layer=1 at=0\n",
            encoding="utf-8",
        )
        partial_prefix_summary = root / "partial-weight-prefix.json"
        partial_prefix_contract = [
            "--explicit-residency-contract", "true",
            "--residency-page-size-bytes", "4096",
            "--residency-unique-footprint-bytes", "49156",
            "--residency-immutable-weight-bytes", str(6 * 4096),
            "--residency-immutable-weight-pages", "6",
            "--residency-static-weight-pages", "2",
            "--residency-runtime-overhead-bytes", "4096",
            "--residency-block-table-bytes", "4",
            "--residency-active-buffer-bytes", "4096",
            "--residency-kv-region-begin", str(6 * 4096),
            "--residency-kv-block-stride-bytes", "4096",
            "--residency-logical-kv-blocks", "5",
            "--residency-hot-kv-blocks", "1",
        ]
        completed = run([
            str(args.scenario_compare), "--trace", str(partial_prefix),
            "--scenarios", SCENARIO,
            "--hbm-capacity-bytes", "32768",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(partial_prefix),
            *small_hbf_geometry,
            *partial_prefix_contract,
            "--summary-json", str(partial_prefix_summary),
        ])
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + completed.stderr)
        partial_prefix_result = json.loads(
            partial_prefix_summary.read_text(encoding="utf-8")
        )["scenarios"][0]
        if (
            partial_prefix_result["hbf_stats"]["static_reserved_pages"],
            partial_prefix_result["hbf_stats"][
                "static_unmaterialized_pages"
            ],
        ) != (8, 7):
            raise AssertionError(
                "partial resident prefix did not fence both HBF edge blocks")
        if (
            partial_prefix_result["layer_streaming"][
                "model_weight_resident_pages"
            ],
            partial_prefix_result["layer_streaming"][
                "model_weight_backing_pages"
            ],
            partial_prefix_result["hybrid_path"]["hbf_static_read_bytes"],
        ) != (2, 4, 4096):
            raise AssertionError(
                "partial resident prefix did not partition immutable weights")

        undersized_hbf_geometry = list(small_hbf_geometry)
        blocks_index = undersized_hbf_geometry.index(
            "--hbf-blocks-per-plane") + 1
        undersized_hbf_geometry[blocks_index] = "3"
        rejected = run([
            str(args.scenario_compare), "--trace", str(full_backing),
            "--scenarios", SCENARIO,
            "--hbm-capacity-bytes", "32768",
            "--layer-buffer-bytes", "4096",
            *trace_identity_options(full_backing),
            *undersized_hbf_geometry,
            *full_backing_contract,
        ])
        if (
            rejected.returncode == 0
            or "complete immutable-weight and cold-KV backing population "
            "exceeds physical HBF capacity" not in rejected.stderr
        ):
            raise AssertionError(
                "complete backing population escaped physical HBF capacity")

        # Layer metadata and finite-buffer geometry fail closed.
        invalid_traces = {
            "mixed": (
                "0x0 R 4096 layer=0\n"
                "0x1000 R 4096\n",
                "specify layer= on every request"),
            "backward": (
                "0x0 R 4096 layer=1\n"
                "0x1000 R 4096 layer=0\n",
                "nondecreasing"),
            "oversized": (
                "".join(
                    f"0x{0x100000 + page * 4096:x} R 4096 layer=0\n"
                    for page in range(17)),
                "fixed HBM residency plus two unfiltered active-layer buffers "
                "exceed configured HBM capacity"),
            "conflicting-compute": (
                "0x100000 R 4096 layer=0 compute_ns=1\n"
                "0x100000 R 4096 layer=0 compute_ns=2\n",
                "conflicting compute_ns"),
            "compute-without-layer": (
                "0x100000 R 4096 compute_ns=1\n",
                "compute_ns requires an explicit layer"),
            "conflicting-placement": (
                "0x100000 R 4096 model_weights layer=0\n"
                "0x100000 R 64 metadata layer=0\n",
                "conflicting HBM-only and tiered-data semantics"),
            "conflicting-data-semantics": (
                "0x100000 R 4096 model_weights layer=0\n"
                "0x100000 R 4096 shared_context layer=0\n",
                "conflicting model-weight and KV semantics"),
            "model-weight-write": (
                "0x100000 W 4096 model_weights layer=0\n",
                "model_weights must be read-only"),
            "model-weight-write-alias": (
                "0x100000 W 4096 layer=0\n"
                "0x100000 R 4096 model_weights layer=0\n",
                "aliases a write with read-only model weights"),
        }
        for name, (payload, diagnostic) in invalid_traces.items():
            invalid = root / f"{name}.trace"
            invalid.write_text(payload, encoding="utf-8")
            rejected = run([
                str(args.scenario_compare), "--trace", str(invalid),
                "--scenarios", SCENARIO,
                "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
                "--hbm-capacity-bytes", "65536",
                "--layer-buffer-bytes", "4096",
            ])
            if rejected.returncode == 0 or diagnostic not in rejected.stderr:
                raise AssertionError(f"invalid {name} layer trace did not fail closed")

        geometry = root / "two-buffer-geometry.trace"
        geometry.write_text(
            "".join(
                f"0x{0x200000 + page * 4096:x} R 4096 layer=0\n"
                for page in range(9)),
            encoding="utf-8",
        )
        rejected = run([
            str(args.scenario_compare), "--trace", str(geometry),
            "--scenarios", SCENARIO,
            "--hbf-page-size", "4096", "--hbf-oob-bytes", "224",
            "--hbm-capacity-bytes", str(16 * 4096),
            "--layer-buffer-bytes", str(9 * 4096),
        ])
        if rejected.returncode == 0 or (
                "fixed HBM residency plus two unfiltered active-layer buffers "
                "exceed configured HBM capacity") not in rejected.stderr:
            raise AssertionError(
                "EC6 accepted two runtime buffers larger than HBM")

        # Old page-cache controls are deleted, not silently accepted.
        rejected = run([
            str(args.scenario_compare), "--trace", str(plain),
            "--scenarios", SCENARIO, "--staging-mshr-limit", "4"])
        if rejected.returncode == 0:
            raise AssertionError("obsolete staging controller option was accepted")

    print("hybrid-residency layer-streaming regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
