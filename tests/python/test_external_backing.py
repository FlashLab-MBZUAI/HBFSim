#!/usr/bin/env python3
"""End-to-end contract for the no-HBF external-backing comparison path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCENARIO = "external-streaming"
HBF_SCENARIO = "hbf-streaming"
PAGE = 4096


def run_case(
    binary: Path,
    root: Path,
    trace: Path,
    kind: str,
    *,
    label: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> dict[str, object]:
    summary = root / f"{label or kind}.json"
    command = [
        str(binary),
        "--trace",
        str(trace),
        "--scenarios",
        SCENARIO,
        "--hbm-capacity-bytes",
        str(64 * 1024 * 1024),
        "--layer-buffer-bytes",
        str(1024 * 1024),
        "--external-backing-kind",
        kind,
        "--external-backing-capacity-bytes",
        str(64 * 1024 * 1024),
        "--external-backing-media-channels",
        "2",
        "--external-backing-max-outstanding-requests",
        "4",
        "--address-heatmap-bins",
        "64",
        *extra_args,
        "--summary-json",
        str(summary),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    document = json.loads(summary.read_text(encoding="utf-8"))
    if document.get("sanity") != "PASS":
        raise AssertionError(f"{kind}: simulator sanity failed")
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 1:
        raise AssertionError(f"{kind}: expected one scenario")
    scenario = scenarios[0]
    if scenario.get("name") != SCENARIO:
        raise AssertionError(f"{kind}: wrong scenario name")
    return scenario


def validate_case(scenario: dict[str, object], kind: str) -> None:
    streaming = scenario["layer_streaming"]
    hybrid = scenario["hybrid_path"]
    stats = scenario["external_backing_stats"]
    if streaming["backing"] != "external":
        raise AssertionError(f"{kind}: wrong layer backing")
    if streaming["residency_policy"] != "trace-derived-first-touch":
        raise AssertionError(f"{kind}: wrong residency policy")
    if stats["kind"] != kind:
        raise AssertionError(f"{kind}: resolved profile was not reported")
    if scenario["hbf_accesses"] != 0 or scenario["hbf_stats"] is not None:
        raise AssertionError(f"{kind}: no-HBF path exercised HBF")
    if scenario["hbm_user_accesses"] != 4:
        raise AssertionError(f"{kind}: foreground did not execute from HBM")
    if (
        streaming["streamed_bytes"],
        streaming["writeback_bytes"],
        hybrid["external_backing_read_bytes"],
        hybrid["external_backing_write_bytes"],
    ) != (2 * PAGE, PAGE, 2 * PAGE, PAGE):
        raise AssertionError(f"{kind}: backing traffic did not conserve")
    if (
        stats["read_requests"],
        stats["write_requests"],
        stats["read_bytes"],
        stats["write_bytes"],
    ) != (2, 1, 2 * PAGE, PAGE):
        raise AssertionError(f"{kind}: device request census is inconsistent")
    if (
        stats["m2s_payload_bytes"] != stats["write_bytes"]
        or stats["s2m_payload_bytes"] != stats["read_bytes"]
        or stats["m2s_wire_bytes"]
        != stats["m2s_payload_bytes"] + stats["m2s_protocol_bytes"]
        or stats["s2m_wire_bytes"]
        != stats["s2m_payload_bytes"] + stats["s2m_protocol_bytes"]
    ):
        raise AssertionError(f"{kind}: directional wire traffic does not conserve")

    time = scenario["time_breakdown"]
    external_work = time["stage_work"]["external_backing"]
    if external_work is None or external_work["total_overlapping_work_ns"] <= 0:
        raise AssertionError(f"{kind}: external stage work is absent")
    resources = time["resource_busy"]
    if resources["external_media"]["resource_count"] != 2:
        raise AssertionError(f"{kind}: external media topology is absent")
    if resources["external_controller"]["busy_ns"] <= 0:
        raise AssertionError(f"{kind}: external controller was not exercised")
    if resources["external_link_m2s"]["busy_ns"] <= 0:
        raise AssertionError(f"{kind}: M2S link was not exercised")
    if resources["external_link_s2m"]["busy_ns"] <= 0:
        raise AssertionError(f"{kind}: S2M link was not exercised")
    if (
        resources["external_link_m2s"]["active_span_ns"],
        resources["external_link_s2m"]["active_span_ns"],
    ) != (
        stats["m2s_active_span_ns"],
        stats["s2m_active_span_ns"],
    ):
        raise AssertionError(f"{kind}: directional link spans are inconsistent")

    domains = {
        item["domain"]: item for item in scenario["address_heatmap"]["domains"]
    }
    external = domains.get("external_physical")
    if external is None:
        raise AssertionError(f"{kind}: external heatmap domain is absent")
    totals = external["totals"]
    if (totals["read_bytes"], totals["write_bytes"]) != (2 * PAGE, PAGE):
        raise AssertionError(f"{kind}: external heatmap does not conserve")


def validate_plan_identity(binary: Path, root: Path, trace: Path) -> None:
    summary = root / "backing-plan-identity.json"
    subprocess.run(
        [
            str(binary),
            "--trace",
            str(trace),
            "--scenarios",
            f"{HBF_SCENARIO},{SCENARIO}",
            "--hbm-capacity-bytes",
            str(64 * 1024 * 1024),
            "--hbf-stacks", "1", "--hbf-channels", "1",
            "--hbf-dies-per-channel", "1", "--hbf-planes-per-die", "1",
            "--hbf-blocks-per-plane", "512", "--hbf-pages-per-block", "64",
            "--hbf-ctrl-dram-bytes", str(8 * 1024 * 1024),
            "--layer-buffer-bytes",
            str(1024 * 1024),
            "--hbf-page-size",
            str(PAGE),
            "--external-backing-kind",
            "cxl-memory",
            "--address-heatmap-bins",
            "64",
            "--summary-json",
            str(summary),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    document = json.loads(summary.read_text(encoding="utf-8"))
    if document.get("sanity") != "PASS":
        raise AssertionError(
            "HBF/external layer plan identity sanity gate failed")
    scenarios = {
        scenario["name"]: scenario
        for scenario in document["scenarios"]
    }
    hbf = scenarios[HBF_SCENARIO]
    external = scenarios[SCENARIO]
    plan_fields = (
        "compact_resident_mapping",
        "semantic_inputs_consumed",
        "address_footprint_bytes",
        "unique_resident_footprint_pages",
        "hbm_capacity_pressure",
        "layers",
        "explicit_layer_requests",
        "explicit_compute_layers",
        "compute_work_ns",
        "hbm_only_resident_pages",
        "hot_kv_candidate_pages",
        "hot_kv_resident_pages",
        "data_pages",
        "model_weight_backing_pages",
        "cold_kv_backing_pages",
        "unknown_backing_pages",
        "backing_unique_pages",
        "resident_physical_pages",
        "effective_layer_buffer_pages",
        "streamed_pages",
        "foreground_resident_page_accesses",
        "foreground_buffer_page_accesses",
        "dirty_pages_written_back",
        "max_layer_data_pages",
    )
    for field in plan_fields:
        if hbf["layer_streaming"][field] != external["layer_streaming"][field]:
            raise AssertionError(
                f"HBF/external layer plan diverged at {field}")
    if (hbf["layer_streaming"]["unused_hbm_pages"]
            + hbf["hbf_stats"]["host_hbm_reserved_bytes"] // PAGE
            != external["layer_streaming"]["unused_hbm_pages"]):
        raise AssertionError("HBF controller reservation does not conserve HBM capacity")
    for field in (
        "hbm_user_accesses",
        "hbm_background_accesses",
    ):
        if hbf[field] != external[field]:
            raise AssertionError(
                f"HBF/external HBM traffic diverged at {field}")
    for field in (
        "hbm_foreground_bytes",
        "hbm_streaming_write_bytes",
    ):
        if hbf["hybrid_path"][field] != external["hybrid_path"][field]:
            raise AssertionError(
                f"HBF/external HBM byte traffic diverged at {field}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="hbfsim-external-backing-") as tmp:
        root = Path(tmp)
        trace = root / "external.trace"
        trace.write_text(
            "\n".join(
                [
                    "0x100000 R 4096 model_weights phase=0 layer=0",
                    "0x200000 W 4096 generated_context phase=0 layer=0",
                    "0x200000 R 4096 generated_context phase=1 layer=1",
                    "0x300000 R 4096 scratch phase=1 layer=1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        lpddr = run_case(
            args.simulator, root, trace, "on-package-lpddr")
        host_dram = run_case(args.simulator, root, trace, "host-dram")
        cxl = run_case(args.simulator, root, trace, "cxl-memory")
        ssd = run_case(args.simulator, root, trace, "nvme-ssd")
        cxl_ssd = run_case(args.simulator, root, trace, "cxl-ssd")
        validate_case(lpddr, "on-package-lpddr")
        validate_case(host_dram, "host-dram")
        validate_case(cxl, "cxl-memory")
        validate_case(ssd, "nvme-ssd")
        validate_case(cxl_ssd, "cxl-ssd")
        validate_plan_identity(args.simulator, root, trace)
        lpddr_makespan = lpddr["time_breakdown"]["wall_clock_ns"]["makespan_ns"]
        dram_makespan = host_dram["time_breakdown"]["wall_clock_ns"]["makespan_ns"]
        ssd_makespan = ssd["time_breakdown"]["wall_clock_ns"]["makespan_ns"]
        if not lpddr_makespan < dram_makespan:
            raise AssertionError(
                "on-package LPDDR sensitivity must be faster than host DRAM "
                "for this trace"
            )
        if not ssd_makespan > dram_makespan:
            raise AssertionError(
                "NVMe SSD profile must be slower than host DRAM for this trace"
            )
        cxl_ssd_makespan = (
            cxl_ssd["time_breakdown"]["wall_clock_ns"]["makespan_ns"]
        )
        if not cxl_ssd_makespan > dram_makespan:
            raise AssertionError(
                "CXL-SSD profile must be slower than host DRAM for this trace"
            )

        # The same trace with the Cylon-style device cache enabled: the
        # generated-context writeback absorbs into the buffer DRAM and its
        # later stream-in hits, so the makespan must beat the uncached run.
        cxl_ssd_cached = run_case(
            args.simulator,
            root,
            trace,
            "cxl-ssd",
            label="cxl-ssd-cached",
            extra_args=(
                "--external-backing-cache-enabled", "true",
                "--external-backing-cache-capacity-bytes", str(64 * PAGE),
                "--external-backing-cache-ways", "0",
                "--external-backing-cache-policy", "s3fifo",
                "--external-backing-cache-prefetch-degree", "0",
                "--external-backing-cache-prefetch-stride", "1",
                "--external-backing-cache-hit-latency-ns", "90",
                "--external-backing-cache-hit-bw", "204.8",
            ),
        )
        validate_case(cxl_ssd_cached, "cxl-ssd")
        cache_census = cxl_ssd_cached["external_backing_stats"].get(
            "device_cache"
        )
        if not cache_census:
            raise AssertionError(
                "cached cxl-ssd run did not report its device-cache census"
            )
        if cache_census["read_hits"] < 1:
            raise AssertionError(
                "generated-context re-read did not hit the device cache"
            )
        if cache_census["write_hits"] + cache_census["write_misses"] != 1:
            raise AssertionError("cached run write census mismatch")
        cached_makespan = (
            cxl_ssd_cached["time_breakdown"]["wall_clock_ns"]["makespan_ns"]
        )
        if not cached_makespan < cxl_ssd_makespan:
            raise AssertionError(
                "device cache did not improve the cxl-ssd makespan"
            )
        tool_dir = (
            Path(__file__).resolve().parents[2] / "reports"
        )
        subprocess.run(
            [
                sys.executable,
                str(tool_dir / "time_breakdown.py"),
                "--summary", f"dram={root / 'host-dram.json'}",
                "--summary", f"lpddr={root / 'on-package-lpddr.json'}",
                "--summary", f"ssd={root / 'nvme-ssd.json'}",
                "--csv", str(root / "time-breakdown.csv"),
                "--markdown", str(root / "time-breakdown.md"),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(tool_dir / "time_dashboard.py"),
                "--summary", f"dram={root / 'host-dram.json'}",
                "--summary", f"ssd={root / 'nvme-ssd.json'}",
                "--output", str(root / "time-breakdown.html"),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(tool_dir / "address_heatmap.py"),
                "--input", str(root / "host-dram.json"),
                "--output", str(root / "external-address-heatmap.html"),
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        # The parity-buffer fence may clear before a new external backing read
        # finishes. The install must wait for that read, not reject it as a
        # future-ready waiter or charge buffer-reuse wait.
        future_ready = root / "future-ready.trace"
        future_ready.write_text(
            "\n".join(
                [
                    "0x100000 W 4096 generated_context layer=0",
                    "0x300000 R 4096 scratch layer=1",
                    "0x201000 R 4096 model_weights layer=2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        future = run_case(
            args.simulator,
            root,
            future_ready,
            "cxl-memory",
            label="future-ready",
            extra_args=(
                "--external-backing-media-read-latency-ns", "100000",
                "--external-backing-media-write-latency-ns", "0",
                "--external-backing-media-read-bw", "1000",
                "--external-backing-media-write-bw", "1000",
                "--external-backing-m2s-bw", "1000",
                "--external-backing-s2m-bw", "1000",
                "--external-backing-one-way-propagation-ns", "0",
            ),
        )
        if (
            future["layer_streaming"]["streamed_bytes"],
            future["layer_streaming"]["writeback_bytes"],
        ) != (PAGE, PAGE):
            raise AssertionError("future-ready parity case lost backing traffic")
        if future["layer_streaming"]["buffer_reuse_wait_work_ns"] != 0:
            raise AssertionError(
                "future-ready backing read was misattributed as buffer-reuse wait"
            )
    print("external-backing end-to-end contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
