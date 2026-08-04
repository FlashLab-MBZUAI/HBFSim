#!/usr/bin/env python3
"""USE cases: component-level input/output checks (user decision 2026-07-09).

A use case feeds ONE component a constructed input and checks its output
against a hand-computed expectation. End-to-end throughput belongs to the
EXPERIMENT cases (ec*, tools/replay_astra_trace.py); mechanism-level C++
assertions stay in physical_probe. This table is the human-review surface
between the two: every row shows 构造输入 -> 期望(公式) -> 实测 -> 判定.

Constants used in the expectation formulas are the model defaults, passed
explicitly on every run so the formula and the simulation share literals:
  tR=4000ns  tPROG=75000ns  tVERIFY=5000ns  (whole-plane exclusive)
  issue=2ns  mapping_dram_response=100ns  mapping_dram_issue=1ns
Exact-count rows (routing, coalescing, mapping, conservation) tolerate
zero; timing rows carry a stated tolerance for micro-terms (ECC, lane and
channel transfers) that are second-order against tR/tPROG.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from plot_time_breakdown import write_visualization
from time_breakdown_report import SummaryInput, write_time_breakdown_report

PAGE = 4096
T_READ = 4000.0
T_PROG = 75000.0
T_VERIFY = 5000.0
PLANES = 2

HBF_GEOM = (
    "--hbf-stacks", "1", "--hbf-channels", "1", "--hbf-dies-per-channel", "1",
    "--hbf-planes-per-die", str(PLANES), "--hbf-blocks-per-plane", "64",
    "--hbf-pages-per-block", "64", "--hbf-page-size", str(PAGE),
    "--hbf-oob-bytes", "224",
    "--hbf-media-lanes-per-plane", "16", "--hbf-subarrays-per-plane", "32",
    "--hbf-page-buffer-banks-per-plane", "16",
    "--hbf-ecc-decode-raw-bw", "135",
    "--hbf-ecc-encode-raw-bw", "135",
    "--hbf-channel-bw", "135", "--hbf-hbio-bw", "128",
    "--hbf-tsv-bw", "137",
    "--hbf-read-ns", str(int(T_READ)), "--hbf-program-ns", str(int(T_PROG)),
    "--hbf-program-verify-ns", str(int(T_VERIFY)),
    "--hbf-write-coalescing", "true",
    "--hbf-gc-low-watermark-pages", "16",
    "--hbf-gc-reserved-free-blocks-per-plane", "1",
    "--hbf-ctrl-dram-bytes", "1048576",
)
FLAT_GEOM = HBF_GEOM + (
    "--hbm-stacks", "1", "--hbm-channels", "4",
    "--flat-hbm-bytes", str(1 << 30),          # 1 GiB boundary
    "--static-direct-hbm-bytes", str(1 << 30),
    "--hbf-hbm-write-buffer-bytes", str(1 << 20),   # 1 MiB coop region
)
ABOVE = 1 << 30                                 # first byte above the boundary


def _run_scenario(binary: Path, out_dir: Path, tag: str, scenario: str,
                  lines: list[str], extra: tuple[str, ...] = ()) -> tuple[dict, SummaryInput]:
    trace = out_dir / f"{tag}.trace"
    trace.write_text("\n".join(lines) + "\n")
    summary = (out_dir / f"{tag}.summary.json").resolve()
    summary.unlink(missing_ok=True)
    geom = FLAT_GEOM if scenario != "all-HBF" else HBF_GEOM
    cmd = [str(binary.resolve()), "--trace", str(trace.resolve()),
           "--scenarios", scenario, *geom, *extra,
           "--summary-json", str(summary)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        detail = (f"stderr:\n{proc.stderr[-2000:].strip()}\n"
                  f"stdout tail:\n{proc.stdout[-2000:].strip()}")
        raise RuntimeError(
            f"{tag}: scenario_compare exited {proc.returncode}:\n{detail}")
    if not summary.exists():
        raise RuntimeError(f"{tag}: scenario_compare succeeded without writing {summary}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        obj: dict = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    data = json.loads(summary.read_text(), object_pairs_hook=reject_duplicate_keys)
    if data.get("schema") != {
            "name": "hbfsim.scenario_compare.summary", "version": 16}:
        raise RuntimeError(f"{tag}: unsupported or missing summary schema")
    if data.get("sanity") != "PASS":
        raise RuntimeError(f"{tag}: summary reports SANITY={data.get('sanity')!r}")
    matches = [s for s in data.get("scenarios", []) if s.get("name") == scenario]
    if len(matches) != 1:
        raise RuntimeError(f"{tag}: expected one {scenario!r} result, got {len(matches)}")
    return matches[0], SummaryInput(
        label=tag,
        summary=data,
        source=str(summary),
        scenario_metadata={
            scenario: {"case": tag, "config": "inline-use-case"},
        },
    )


def w(addr: int, nbytes: int = PAGE, at: float | None = None) -> str:
    tail = f" at={at}" if at is not None else ""
    return f"0x{addr:x} W {nbytes}{tail}"


def r(addr: int, nbytes: int = PAGE, at: float | None = None) -> str:
    tail = f" at={at}" if at is not None else ""
    return f"0x{addr:x} R {nbytes}{tail}"


def hbf(s: dict) -> dict:
    return s.get("hbf_stats") or {}


def data_programs(s: dict) -> int:
    return hbf(s).get("data_programs", 0)


def wall_clock(s: dict) -> dict:
    return s["time_breakdown"]["wall_clock_ns"]


def latency_work(s: dict) -> dict:
    return s["time_breakdown"]["latency_work"]


def resource_busy(s: dict, name: str) -> dict:
    return s["time_breakdown"]["resource_busy"][name]


def average_parallelism(metric: dict) -> float:
    span = float(metric["active_span_ns"])
    return float(metric["busy_ns"]) / span if span > 0.0 else 0.0


def user_active_span(s: dict) -> float:
    return wall_clock(s)["user_completion_span_ns"]


def rows(binary: Path, out_dir: Path,
         skip_deep_waf: bool = False) -> tuple[list[tuple], list[SummaryInput]]:
    out: list[tuple] = []
    summary_inputs: list[SummaryInput] = []

    def run_scenario(binary: Path, out_dir: Path, tag: str, scenario: str,
                     lines: list[str], extra: tuple[str, ...] = ()) -> dict:
        result, summary_input = _run_scenario(
            binary, out_dir, tag, scenario, lines, extra)
        summary_inputs.append(summary_input)
        return result

    def add(uc, component, stimulus, expect_text, expected, measured, tol):
        if isinstance(expected, str):
            ok = expected == measured
        elif tol == 0:
            ok = expected == measured
        else:
            lo, hi = (expected * (1 - tol[0]), expected * (1 + tol[1]))
            ok = lo <= measured <= hi
        out.append((uc, component, stimulus, expect_text, measured,
                    "PASS" if ok else "FAIL"))

    # ---- write path ----
    s = run_scenario(binary, out_dir, "uc-w1", "all-HBF", [w(0)] * 8)
    add("uc-w1", "写缓冲合并", "同一页连写 8 次",
        "data_programs=1, hits=7", (1, 7),
        (data_programs(s), hbf(s).get("write_buffer_hits", -1)), 0)

    s = run_scenario(binary, out_dir, "uc-w2", "all-HBF",
                     [w(i * PAGE) for i in range(16)])
    add("uc-w2", "写路径守恒", "16 个相异页各写 1 次(同 vpn)",
        "data=16, physical=(16数据+1映射)x4096=69632", (16, 17 * PAGE),
        (data_programs(s), hbf(s).get("physical_write_bytes", -1)), 0)

    s = run_scenario(binary, out_dir, "uc-w3", "all-HBF",
                     [w(v * 512 * PAGE) for v in range(3)])
    add("uc-w3", "mapping 持久化", "3 页落在 3 个相异 vpn",
        "mapping_page_programs=3", 3,
        hbf(s).get("mapping_page_programs", -1), 0)

    s = run_scenario(binary, out_dir, "uc-w4", "all-HBF",
                     [w(0, PAGE, 0.0), w(PAGE, PAGE, 20000.0),
                      w(2 * PAGE, PAGE, 40000.0)])
    add("uc-w4", "写完成语义", "3 次错峰 full-page 写",
        "resident mapping lookup 后写缓冲收纳，p50 < 1µs", 1000.0,
        latency_work(s)["p50_ns"], (1.0, 0.0))

    n, slots = 12, 4
    s = run_scenario(binary, out_dir, "uc-w5", "all-HBF",
                     [w(i * PAGE) for i in range(n)],
                     ("--hbf-write-buffer-pages", str(slots),
                      "--hbf-write-buffer-flush-threshold-pages", "2"))
    floor = (n - slots) / PLANES * (T_PROG + T_VERIFY)
    add("uc-w5", "slot 生命周期背压", f"{n} 页写 / {slots} slots / {PLANES} planes",
        f"用户段 ≥ (12-4)/2 x 80.005µs = {floor/1e3:.0f}µs", floor,
        user_active_span(s), (0.0, 0.25))

    s = run_scenario(binary, out_dir, "uc-w6", "all-HBF",
                     [w(i * PAGE) for i in range(4)])
    # 5 programs (4 data + 1 mapping); the mapping page pins to plane 0,
    # so the loaded plane serializes 3 exclusive programs.
    waves = 3 * (T_PROG + T_VERIFY)
    add("uc-w6", "program 整 plane 排他", "4 页写(+1 映射页),2 planes,排空",
        f"finish ≈ 满载 plane 3 program x 80.005µs = {waves/1e3:.0f}µs", waves,
        wall_clock(s)["makespan_ns"], (0.02, 0.10))

    # ---- read path ----
    lines = [r(0, PAGE, 0.0)] + [r(2 * i * PAGE, PAGE, float(50000 * i))
                                 for i in range(1, 4)]
    s = run_scenario(binary, out_dir, "uc-r1", "all-HBF", lines)
    r1_hbf = hbf(s)
    add("uc-r1", "全量 L2P 常驻读", "同 mapping group 的 4 个相异页错峰读",
        "lookup=4,page_reads=4,physical=logical=4x4096,mapping_programs=0",
        (4, 4, 4 * PAGE, 4 * PAGE, 0),
        (
            r1_hbf.get("mapping_lookup_ops"),
            r1_hbf.get("page_reads"),
            r1_hbf.get("logical_read_bytes"),
            r1_hbf.get("physical_read_bytes"),
            r1_hbf.get("mapping_page_programs"),
        ),
        0)

    lines = [r(2 * i * PAGE, PAGE, 0.0) for i in range(16)]
    batched = run_scenario(binary, out_dir, "uc-r2-batched", "all-HBF", lines)
    independent = run_scenario(
        binary, out_dir, "uc-r2-independent", "all-HBF", lines,
        ("--hbf-batch-activation", "false"))
    serialized = run_scenario(
        binary, out_dir, "uc-r2-single-subarray", "all-HBF", lines,
        ("--hbf-subarrays-per-plane", "1"))
    batched_h = hbf(batched)
    independent_h = hbf(independent)
    serialized_h = hbf(serialized)
    add("uc-r2", "sense round 同步约束/批内并行",
        "16 个相异页同时到达(2 planes 各 8)",
        "ideal independent ≤ batch ≤ single-subarray/1.5,且守恒相同",
        (True, True, True, True),
        (wall_clock(batched)["makespan_ns"] >=
         wall_clock(independent)["makespan_ns"],
         wall_clock(serialized)["makespan_ns"] >=
         1.5 * wall_clock(batched)["makespan_ns"],
         independent_h.get("page_reads") == batched_h.get("page_reads") ==
         serialized_h.get("page_reads"),
         independent_h.get("physical_read_bytes") ==
         batched_h.get("physical_read_bytes") ==
         serialized_h.get("physical_read_bytes")), 0)

    lines = [w(0, PAGE, 0.0), r(0, PAGE, 1000.0)]
    s = run_scenario(binary, out_dir, "uc-w7", "all-HBF", lines)
    # For two operations, the lower latency is 2 x average - max. The buffer
    # hit itself serves from SRAM and avoids NAND data media.
    read_lat = (2 * latency_work(s)["average_ns"] -
                latency_work(s)["max_ns"])
    add("uc-w7", "写缓冲读命中", "写入一页,flush 前读同页",
        "read 从缓冲 SRAM 服务:read_hits=1, 读延迟 < 1µs", (1, True),
        (hbf(s).get("write_buffer_read_hits", -1), read_lat < 1000.0), 0)

    lines = [r(2 * i * PAGE, PAGE, float(50000 * i)) for i in range(4)]
    s = run_scenario(binary, out_dir, "uc-r4", "all-HBF", lines)
    add("uc-r4", "读路径守恒", "4 相异页读(错峰,同 mapping group)",
        "page_reads=4 数据页，physical=logical=4x4096",
        (4, 4 * PAGE),
        (hbf(s).get("page_reads", -1), hbf(s).get("physical_read_bytes", -1)), 0)

    # GC quick conservation: 8 blocks/plane, of which 1 is the GC reserve
    # and 1 the mapping active -> data pool 6/plane; footprint 5/plane
    # (u_eff 83%, a 6/plane footprint is GENUINELY infeasible: the pool
    # packs with valid data and GC has nothing to reclaim — the lesson the
    # write-amplification suite geometry taught). Seq overwrite x3 -> GC must
    # fire, victims fully invalid (relocations exactly 0). Deep regimes use
    # run_waf_cases.py.
    # footprint 4/plane (u_eff 67%): 2 spare blocks/plane of AGING ROOM so
    # a full invalidation window (128 seq writes) completes before GC must
    # harvest — 5/plane leaves 1 spare, GC fires mid-window and relocates
    # half-valid victims (measured: 638 relocations; the aging-room lesson).
    fp = 4 * PLANES * 64
    initial = [r(i * PAGE) for i in range(fp)]
    seq = [w((i % fp) * PAGE) for i in range(fp * 3)]
    s = run_scenario(binary, out_dir, "uc-g1", "all-HBF", initial + seq,
                     ("--hbf-blocks-per-plane", "8",
                      "--hbf-gc-low-watermark-pages", "32",
                      "--hbf-write-buffer-completion-requires-flush", "true",
                      "--max-outstanding-requests", "1"))
    h = hbf(s)
    expected_data_programs = fp * 3
    expected_mapping_programs = 1
    expected_page_programs = expected_data_programs + expected_mapping_programs
    expected_gc_runs = 20
    expected_page_reads = fp
    expected_physical_read_bytes = expected_page_reads * PAGE
    expected_physical_write_bytes = expected_page_programs * PAGE
    expected_flash_transactions = (
        expected_page_reads + expected_page_programs + expected_gc_runs)
    expected = {
        "audit": True,
        "total/free/valid/invalid": (1024, 254, 513, 257),
        "pending publications/program/static": (0, 0, 0),
        "requests R/W/E": (fp, fp * 3, 0),
        "logical R/W B": (fp * PAGE, expected_data_programs * PAGE),
        "mapping entries/lookup/update": (
            fp, fp + expected_data_programs, expected_data_programs),
        "program D/M/all": (
            expected_data_programs,
            expected_mapping_programs,
            expected_page_programs,
        ),
        "GC runs/erase/reloc D/M/all": (
            expected_gc_runs, expected_gc_runs, 0, 0, 0),
        "GC reclaimed/invalidations": (1280, 1537),
        "page/physical reads": (
            expected_page_reads, expected_physical_read_bytes),
        "physical writes B": expected_physical_write_bytes,
        "scheduler enq/issue": (
            expected_flash_transactions, expected_flash_transactions),
        "ECC decode/encode": (expected_page_reads, expected_page_programs),
    }
    measured = {
        "audit": h.get("accounting_verified"),
        "total/free/valid/invalid": (
            h.get("total_pages"), h.get("free_pages"),
            h.get("valid_pages"), h.get("invalid_pages")),
        "pending publications/program/static": (
            h.get("pending_mapping_publications"),
            h.get("pending_program_pages"),
            h.get("static_unmaterialized_pages")),
        "requests R/W/E": (
            h.get("read_requests"), h.get("program_requests"),
            h.get("erase_requests")),
        "logical R/W B": (
            h.get("logical_read_bytes"), h.get("logical_write_bytes")),
        "mapping entries/lookup/update": (
            h.get("mapping_entries"),
            h.get("mapping_lookup_ops"),
            h.get("mapping_update_ops")),
        "program D/M/all": (
            h.get("data_programs"), h.get("mapping_page_programs"),
            h.get("page_programs")),
        "GC runs/erase/reloc D/M/all": (
            h.get("gc_runs"), h.get("block_erases"),
            h.get("gc_data_relocations"), h.get("gc_mapping_relocations"),
            h.get("gc_relocations")),
        "GC reclaimed/invalidations": (
            h.get("gc_reclaimed_invalid_pages"), h.get("invalidations")),
        "page/physical reads": (
            h.get("page_reads"), h.get("physical_read_bytes")),
        "physical writes B": h.get("physical_write_bytes"),
        "scheduler enq/issue": (
            h.get("flash_scheduler_enqueues"),
            h.get("flash_scheduler_issues")),
        "ECC decode/encode": (
            h.get("ecc_decode_ops"), h.get("ecc_encode_ops")),
    }
    add(
        "uc-g1", "GC 精确跨层守恒",
        f"显式初态 {fp} 页;8 块/plane,u_eff 67%,顺序覆写 3 轮",
        "audit=1,total=1024,live=513,invalid=257,free=254; "
        "data/map/all=1536/1/1537; GC/erase/reloc=20/20/0; "
        "reads=512; scheduler=2069; ECC=512/1537",
        expected, measured, 0)

    # ---- composition ----
    lines = [r(i * PAGE) for i in range(4)] + \
            [r(ABOVE + i * PAGE, PAGE) for i in range(6)]
    s = run_scenario(binary, out_dir, "uc-c1", "HBM-HBF-Flat", lines)
    add("uc-c1", "FLAT 地址路由", "边界下 4 读 + 边界上 6 读",
        "hbm=4, hbf=6(精确)", (4, 6),
        (s["hbm_accesses"], s["hbf_accesses"]), 0)

    lines = [r(0)] + [w(ABOVE + i * PAGE) for i in range(5)]
    s = run_scenario(binary, out_dir, "uc-c2", "HBM-HBF-Flat", lines)
    hp = s.get("hybrid_path") or {}
    add("uc-c2", "协作写守恒", "1 个 HBM 读 + 5 个 HBF-bound 写(经 staging 区)",
        "destaged==5x4096B 且 D2D 写字节==20480", (20480, 20480),
        (hp.get("hbm_write_buffer_destaged_bytes", -1),
         hp.get("base_die_link_write_bytes", -1)), 0)

    lines = [r(0, PAGE, 0.0), w(ABOVE, PAGE, 0.0),
             r(ABOVE, PAGE, 1000.0)]
    s = run_scenario(binary, out_dir, "uc-c3", "HBM-HBF-Flat", lines)
    add("uc-c3", "read-after-write 一致性", "HBM 探针 + 写 staging 后立刻读同页",
        "读走 HBM(parked),延迟 < 1µs", 1000.0,
        latency_work(s)["max_ns"], (1.0, 0.0))

    lines = [r(0, PAGE, 0.0), r(PAGE, PAGE, 50000.0)]
    s = run_scenario(binary, out_dir, "uc-c4", "all-HBF", lines)
    add("uc-c4", "at= 到达语义", "两读 at=0 / at=50µs",
        "finish ≥ 50µs + 服务(到达不可提前)", 50000.0,
        wall_clock(s)["makespan_ns"], (0.0, 0.5))

    # Metrics must use an elapsed interval, not an absolute timestamp. A
    # constant shift of every arrival therefore cannot change throughput.
    base = run_scenario(binary, out_dir, "uc-q1-base", "all-HBF",
                        [r(0, PAGE, 0.0), r(2 * PAGE, PAGE, 50000.0)])
    shifted = run_scenario(binary, out_dir, "uc-q1-shift", "all-HBF",
                           [r(0, PAGE, 1_000_000.0),
                            r(2 * PAGE, PAGE, 1_050_000.0)])
    hbm_base = run_scenario(binary, out_dir, "uc-q1-hbm-base", "all-HBM",
                            [r(0, PAGE, 0.0), r(2 * PAGE, PAGE, 50000.0)])
    hbm_shifted = run_scenario(
        binary, out_dir, "uc-q1-hbm-shift", "all-HBM",
        [r(0, PAGE, 1_000_000.0), r(2 * PAGE, PAGE, 1_050_000.0)])
    link_base = run_scenario(binary, out_dir, "uc-q1-link-base", "HBM-HBF-Flat",
                             [w(ABOVE, PAGE, 0.0)])
    link_shifted = run_scenario(
        binary, out_dir, "uc-q1-link-shift", "HBM-HBF-Flat",
        [w(ABOVE, PAGE, 1_000_000.0)])
    base_media = resource_busy(base, "hbf_plane_media")
    shifted_media = resource_busy(shifted, "hbf_plane_media")
    base_hbio = resource_busy(base, "hbf_hbio_data")
    shifted_hbio = resource_busy(shifted, "hbf_hbio_data")
    base_hbm_bus = resource_busy(hbm_base, "hbm_data_bus")
    shifted_hbm_bus = resource_busy(hbm_shifted, "hbm_data_bus")
    base_link = resource_busy(link_base, "base_die_link_write")
    shifted_link = resource_busy(link_shifted, "base_die_link_write")
    origin_base = (
        wall_clock(base)["user_completion_span_ns"],
        base["user_completion_throughput_GBps"],
        base_media["utilization"], average_parallelism(base_media),
        base_hbio["utilization"], base_hbm_bus["utilization"],
        average_parallelism(base_hbm_bus), base_link["utilization"])
    origin_shifted = (
        wall_clock(shifted)["user_completion_span_ns"],
        shifted["user_completion_throughput_GBps"],
        shifted_media["utilization"], average_parallelism(shifted_media),
        shifted_hbio["utilization"], shifted_hbm_bus["utilization"],
        average_parallelism(shifted_hbm_bus), shifted_link["utilization"])
    ulp_errors = [
        abs(lhs - rhs) / max(math.ulp(lhs), math.ulp(rhs))
        for lhs, rhs in zip(origin_base, origin_shifted)
        if lhs is not None and rhs is not None
    ]
    max_ulp_error = max(ulp_errors, default=math.inf)
    max_relative_error = max(
        (abs(lhs - rhs) / abs(lhs) if lhs != 0.0 else abs(lhs - rhs)
         for lhs, rhs in zip(origin_base, origin_shifted)
         if lhs is not None and rhs is not None),
        default=math.inf)
    out.append((
        "uc-q1", "结果时间原点不变性", "同一 trace 的 arrival 全部平移 1ms",
        "吞吐、HBM/HBF 利用率/并行度、D2D 利用率差异 ≤ 4 ULP",
        f"max_ulp={max_ulp_error:.1f}, max_rel={max_relative_error:.3e}",
        "PASS" if max_ulp_error <= 4.0 else "FAIL"))

    # Fail-closed CLI contract: invalid timing must be rejected before a
    # simulator run can emit a plausible-looking PASS result.
    invalid_trace = out_dir / "uc-q2.trace"
    invalid_trace.write_text(r(0) + "\n")
    invalid_cases = (
        ("--hbf-ctrl-dram-latency-ns", "-100"),
        ("--hbf-logic-scheduler-issue-ns", "-2"),
        ("--hbm-trfc-ns", "-5000"),
        ("--hbm-trefi-ns", "nan"),
        ("--hbf-read-ns", "inf"),
        ("--hbf-program-verify-ns", "-1"),
        ("--hbf-ecc-decode-raw-bw", "0"),
        ("--hbf-ecc-encode-latency-ns", "nan"),
        ("--hbf-ecc-decode-latency-ns", "1"),
    )
    rejected = []
    for option, value in invalid_cases:
        proc = subprocess.run(
            [str(binary.resolve()), "--trace", str(invalid_trace.resolve()),
             "--scenarios", "all-HBF", *HBF_GEOM, option, value],
            capture_output=True, text=True, timeout=60)
        rejected.append(proc.returncode != 0)
    add("uc-q2", "CLI fail-closed", "负/零 timing、NaN/Inf 与 ECC latency<II",
        "9 个非法配置全部非零退出", [True] * len(invalid_cases), rejected, 0)

    # ---- gates(深度断言组,整组一票)----
    probe = binary.parent / "physical_probe"
    ok = subprocess.run([str(probe), "all"], capture_output=True,
                        text=True, timeout=1800).returncode == 0
    out.append(("uc-p*", "physical_probe 断言组",
                "全部注册微场景(HBM 时序/FR-FCFS/tFAW/批激活/背压/组成路由)",
                "全部 C++ 内断言通过", "physical_probe all",
                "PASS" if ok else "FAIL"))

    if not skip_deep_waf:
        waf_script = Path(__file__).resolve().with_name("run_waf_cases.py")
        amplification_gate = subprocess.run(
            [sys.executable, str(waf_script),
             "--scenario-compare", str(binary.resolve()),
             "--out-dir", str((out_dir / "write-amplification-cases").resolve())],
            capture_output=True, text=True, timeout=3600)
        out.append(("uc-gc*", "GC/写放大深度门(run_waf_cases.py)",
                    "5 工况:顺序/kv-fifo/两档随机搅拌/子页 RMW",
                    "恒等式+锐利 regime+随机 GC 结构/单调关系", "5 cases",
                    "PASS" if amplification_gate.returncode == 0 else "FAIL"))
    return out, summary_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path,
                        default=Path("build/scenario_compare"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/use-cases"))
    parser.add_argument(
        "--skip-deep-waf", action="store_true",
        help="run the quick UC contract table and physical probes without the separate WAF suite")
    args = parser.parse_args()
    args.scenario_compare = args.scenario_compare.resolve()
    if not args.scenario_compare.is_file():
        parser.error(f"scenario_compare not found: {args.scenario_compare}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / "uc验证表.csv"
    time_csv_path = args.out_dir / "time-breakdown.csv"
    time_markdown_path = args.out_dir / "time-breakdown.md"
    time_html_path = args.out_dir / "time-breakdown.html"
    for artifact in (
            table_path, time_csv_path, time_markdown_path, time_html_path):
        artifact.unlink(missing_ok=True)

    table, summary_inputs = rows(
        args.scenario_compare, args.out_dir, args.skip_deep_waf)
    print(f"{'uc':7s} {'组件':22s} {'构造输入':34s} {'期望(公式)':44s} "
          f"{'实测':>24s} {'判定'}")
    csv_rows = [["uc", "component", "stimulus", "expected", "measured", "verdict"]]
    failed = False
    for uc, comp, stim, expect, measured, verdict in table:
        failed = failed or verdict != "PASS"
        m = str(measured)
        print(f"{uc:7s} {comp:22s} {stim:34s} {expect:44s} {m:>24s} {verdict}")
        csv_rows.append([uc, comp, stim, expect, m, verdict])
    temporary = table_path.with_suffix(table_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.writer(handle).writerows(csv_rows)
    temporary.replace(table_path)
    print(f"wrote: {table_path}")
    write_time_breakdown_report(
        summary_inputs, time_csv_path, time_markdown_path)
    write_visualization(
        summary_inputs,
        time_html_path,
        title="HBFSim use-case timing",
    )
    print(f"wrote: {time_csv_path}")
    print(f"wrote: {time_markdown_path}")
    print(f"wrote: {time_html_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
