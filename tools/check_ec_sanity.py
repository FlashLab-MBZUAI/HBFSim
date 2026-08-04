#!/usr/bin/env python3
"""EC sanity checks: is the experiment-case data physically reasonable?

Cross-checks the ASTRA replay outputs against ground truths that do not
come from the simulator: the trace itself (byte/op census per kind and per
boundary side), Little's law under the closed-loop window, and internal
conservation (access splits, layer-streaming DMA, program components).

  check 1  trace census: read/write bytes and op counts (ground truth)
  check 2  FLAT boundary split, BYTE-exact and order-aware: HBF logical
           read bytes == above-boundary reads MINUS re-reads of byte ranges
           already written (those ranges stay parked in the HBM region for the
           whole user phase and route to HBM); data programs == UNIQUE
           above-boundary write pages. Raw access counts legitimately
           diverge (64 KiB line splits, streamed writes count as HBM).
  check 3  Little's law: throughput == W x avg_op_bytes / avg_latency on
           every window-bound row (3%); arrival-bound rows sit below it
  check 4  EC5 FTL-bypass and EC6 HBM-only/buffer foreground routing
  check 5  EC6 layer DMA and writeback conserve pages and bytes
  check 6  program components: programs == data + mapping + GC on every row
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

BOUNDARY = 0x2_0000_0000
PAGE = 4096
UINT64_MAX = (1 << 64) - 1
SEMANTIC_KINDS = {
    "unknown", "model_weights", "shared_context", "generated_context",
    "scratch", "metadata",
}

CASES = (
    ("ec0", "usecase-baseline", "all-HBM"),
    ("ec1", "usecase-baseline", "all-HBF"),
    ("ec2", "usecase-6h2f", "HBM-HBF-Flat"),
    ("ec3", "usecase-4h4f", "HBM-HBF-Flat"),
    ("ec4", "usecase-2h6f", "HBM-HBF-Flat"),
    ("ec5", "usecase-2h6f", "HBF-static-direct-read"),
    ("ec6", "usecase-6h2f", "HBM+HBF-layer-streaming"),
)
EXPECTED_STACKS = {
    "usecase-baseline": (8, 8),
    "usecase-6h2f": (6, 2),
    "usecase-4h4f": (4, 4),
    "usecase-2h6f": (2, 6),
}


def merge_interval(
        ranges: list[tuple[int, int]], begin: int, end: int
) -> list[tuple[int, int]]:
    """Insert one half-open byte range into a sorted, disjoint range set."""
    merged: list[tuple[int, int]] = []
    incoming = (begin, end)
    inserted = False
    for current in ranges:
        if current[1] < incoming[0]:
            merged.append(current)
        elif incoming[1] < current[0]:
            if not inserted:
                merged.append(incoming)
                inserted = True
            merged.append(current)
        else:
            incoming = (min(incoming[0], current[0]),
                        max(incoming[1], current[1]))
    if not inserted:
        merged.append(incoming)
    return merged


def covered_bytes(ranges: list[tuple[int, int]], begin: int, end: int) -> int:
    return sum(max(0, min(end, right) - max(begin, left))
               for left, right in ranges if left < end and right > begin)


def census_lines(lines: Iterable[str], boundary: int = BOUNDARY) -> dict:
    ops = reads = writes = 0
    rbytes = wbytes = 0
    below = above = 0
    rbytes_above = wbytes_above = 0
    rbytes_above_hbf = rbytes_above_parked = 0
    above_pages: set[int] = set()
    wpages_above: set[int] = set()
    written_ranges: dict[int, list[tuple[int, int]]] = {}
    per_kind: dict[str, int] = {}
    all_read_pages_by_kind: dict[str, set[int]] = {}
    write_ops_by_kind: dict[str, int] = {}
    write_bytes_by_kind: dict[str, int] = {}
    write_pages_by_kind: dict[str, set[int]] = {}
    written_ranges_by_kind: dict[str, dict[int, list[tuple[int, int]]]] = {}
    pages_by_kind: dict[str, set[int]] = {}
    hbm_only_pages: set[int] = set()
    total_page_fragments = 0
    static_direct_read_page_fragments = 0
    explicit_layer_requests = 0
    layer_ids: set[int] = set()
    previous_layer: int | None = None
    for line_no, line in enumerate(lines, start=1):
        f = line.split("#", 1)[0].split()
        if not f:
            continue
        if len(f) < 3 or f[1] not in {"R", "W"}:
            raise ValueError(
                f"trace line {line_no}: checker requires normalized ASTRA "
                "grammar <addr> <R|W> <bytes> [kind] [at=<ns>]")
        try:
            addr, op, nbytes = int(f[0], 0), f[1], int(f[2], 0)
        except ValueError as error:
            raise ValueError(
                f"trace line {line_no}: invalid address or byte count") from error
        if addr < 0 or addr > UINT64_MAX or nbytes <= 0 or nbytes - 1 > UINT64_MAX - addr:
            raise ValueError(
                f"trace line {line_no}: address/byte range is invalid or overflows uint64")
        end_addr = addr + nbytes
        kind = "unknown"
        request_layer: int | None = None
        for token in f[3:]:
            if token.startswith(("at=", "arrival=")):
                try:
                    arrival = float(token.split("=", 1)[1])
                except ValueError as error:
                    raise ValueError(
                        f"trace line {line_no}: invalid arrival") from error
                if not math.isfinite(arrival) or arrival < 0:
                    raise ValueError(f"trace line {line_no}: invalid arrival")
                continue
            if token.startswith(("layer=", "phase=")):
                try:
                    request_layer = int(token.split("=", 1)[1], 0)
                except ValueError as error:
                    raise ValueError(
                        f"trace line {line_no}: invalid layer") from error
                if request_layer < 0 or request_layer > UINT64_MAX:
                    raise ValueError(f"trace line {line_no}: invalid layer")
                continue
            value = token
            if token.startswith(("kind=", "semantic=", "type=")):
                value = token.split("=", 1)[1]
            if value not in SEMANTIC_KINDS:
                raise ValueError(
                    f"trace line {line_no}: unsupported ASTRA token {token!r}")
            kind = value
        if request_layer is not None:
            explicit_layer_requests += 1
            layer_ids.add(request_layer)
            if previous_layer is not None and request_layer < previous_layer:
                raise ValueError(
                    f"trace line {line_no}: layer identifiers must be nondecreasing")
            previous_layer = request_layer
        elif explicit_layer_requests:
            raise ValueError(
                f"trace line {line_no}: layer metadata must cover every request")
        ops += 1
        per_kind[kind] = per_kind.get(kind, 0) + nbytes
        first_page = addr // PAGE
        last_page = (end_addr - 1) // PAGE
        pages_by_kind.setdefault(kind, set()).update(
            range(first_page, last_page + 1))
        if kind in {"scratch", "metadata"}:
            hbm_only_pages.update(range(first_page, last_page + 1))
        total_page_fragments += last_page - first_page + 1

        if op == "R":
            reads += 1
            rbytes += nbytes
            all_read_pages_by_kind.setdefault(kind, set()).update(
                range(first_page, last_page + 1))
        else:
            writes += 1
            wbytes += nbytes
        if addr < boundary:
            below += 1
        if end_addr > boundary:
            above += 1
        cursor = max(addr, boundary)
        if cursor >= end_addr:
            continue
        if op == "R":
            rbytes_above += end_addr - cursor
        else:
            wbytes_above += end_addr - cursor
            write_ops_by_kind[kind] = write_ops_by_kind.get(kind, 0) + 1
            write_bytes_by_kind[kind] = (
                write_bytes_by_kind.get(kind, 0) + end_addr - cursor)
        while cursor < end_addr:
            page = cursor // PAGE
            page_offset = cursor % PAGE
            segment_bytes = min(end_addr - cursor, PAGE - page_offset)
            segment_end = page_offset + segment_bytes
            above_pages.add(page)
            if op == "R":
                parked = covered_bytes(
                    written_ranges.get(page, []), page_offset, segment_end)
                rbytes_above_parked += parked
                rbytes_above_hbf += segment_bytes - parked
                if (kind in {
                        "unknown", "model_weights", "shared_context"} and (
                        segment_bytes > parked)):
                    static_direct_read_page_fragments += 1
            else:
                wpages_above.add(page)
                written_ranges[page] = merge_interval(
                    written_ranges.get(page, []), page_offset, segment_end)
                kind_pages = write_pages_by_kind.setdefault(kind, set())
                kind_pages.add(page)
                kind_ranges = written_ranges_by_kind.setdefault(kind, {})
                kind_ranges[page] = merge_interval(
                    kind_ranges.get(page, []), page_offset, segment_end)
            cursor += segment_bytes
    dirty_union_bytes = sum(
        end - begin for ranges in written_ranges.values() for begin, end in ranges)
    dirty_union_bytes_by_kind = {
        kind: sum(end - begin for ranges in pages.values()
                  for begin, end in ranges)
        for kind, pages in written_ranges_by_kind.items()
    }
    dirty_range_count_by_kind = {
        kind: sum(len(ranges) for ranges in pages.values())
        for kind, pages in written_ranges_by_kind.items()
    }
    if rbytes_above_hbf + rbytes_above_parked != rbytes_above:
        raise RuntimeError("byte-range census does not conserve above-boundary reads")
    if explicit_layer_requests not in (0, ops):
        raise ValueError("layer metadata must cover every request")
    return dict(ops=ops, reads=reads, writes=writes, rbytes=rbytes,
                wbytes=wbytes, below=below, above=above,
                above_pages=len(above_pages), rbytes_above=rbytes_above,
                wbytes_above=wbytes_above,
                rbytes_above_hbf=rbytes_above_hbf,
                rbytes_above_parked=rbytes_above_parked,
                wpages_above=len(wpages_above),
                dirty_union_bytes=dirty_union_bytes,
                per_kind=per_kind,
                write_ops_by_kind=write_ops_by_kind,
                write_bytes_by_kind=write_bytes_by_kind,
                write_pages_by_kind={
                    kind: len(pages) for kind, pages in write_pages_by_kind.items()},
                dirty_union_bytes_by_kind=dirty_union_bytes_by_kind,
                dirty_range_count_by_kind=dirty_range_count_by_kind,
                pages_by_kind={
                    kind: len(pages) for kind, pages in pages_by_kind.items()},
                hbm_only_pages=len(hbm_only_pages),
                total_page_fragments=total_page_fragments,
                static_direct_read_page_fragments=
                    static_direct_read_page_fragments,
                all_read_pages_by_kind={
                    kind: len(pages)
                    for kind, pages in all_read_pages_by_kind.items()},
                streaming_layers=(len(layer_ids) if explicit_layer_requests
                                  else (1 if ops else 0)),
                explicit_layer_requests=explicit_layer_requests)


def census(trace: Path, boundary: int = BOUNDARY) -> dict:
    with trace.open() as handle:
        return census_lines(handle, boundary)


def select_cases(spec: str) -> tuple[tuple[str, str, str], ...]:
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    known = {case[0] for case in CASES}
    if not requested:
        raise argparse.ArgumentTypeError("--cases must name at least one case")
    if len(set(requested)) != len(requested):
        raise argparse.ArgumentTypeError("--cases contains a duplicate case")
    unknown = sorted(set(requested) - known)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown case(s): {','.join(unknown)}; choose from {','.join(sorted(known))}")
    selected = set(requested)
    return tuple(case for case in CASES if case[0] in selected)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_summary(
        data: dict, trace_digest: str, window: int, expected_ops: int,
        expected_stacks: tuple[int, int], path: Path, boundary: int = BOUNDARY
) -> tuple[str, str, bool]:
    schema = data.get("schema") or {}
    if (schema.get("name"), schema.get("version")) != (
            "hbfsim.scenario_compare.summary", 16):
        raise RuntimeError(f"{path}: unsupported or missing summary schema")
    if data.get("sanity") != "PASS":
        raise RuntimeError(f"{path}: summary sanity is not PASS")
    recorded_digest = ((data.get("workload") or {}).get("trace_digest") or {})
    if (recorded_digest.get("algorithm"), recorded_digest.get("value")) != (
            "sha256", trace_digest):
        raise RuntimeError(f"{path}: trace digest does not match --trace")
    config = data.get("config") or {}
    if config.get("ops") != expected_ops:
        raise RuntimeError(f"{path}: summary op count does not match trace census")
    if config.get("max_outstanding_requests") != window:
        raise RuntimeError(f"{path}: closed-loop window does not match --window")
    layer_buffer_bytes = config.get("layer_buffer_bytes")
    if type(layer_buffer_bytes) is not int or layer_buffer_bytes <= 0:
        raise RuntimeError(f"{path}: invalid EC6 layer-buffer capacity")
    if (config.get("flat_hbm_bytes") != boundary or
            config.get("static_direct_hbm_bytes") != boundary):
        raise RuntimeError(f"{path}: ASTRA placement boundary mismatch")
    hbm = config.get("hbm") or {}
    hbf = config.get("hbf") or {}
    if (hbm.get("stacks"), hbf.get("stacks")) != expected_stacks:
        raise RuntimeError(f"{path}: HBM/HBF stack composition mismatch")
    if hbf.get("placement_mapping_scheme") != \
            "page-striped-stack-local-mapping-v2":
        raise RuntimeError(f"{path}: HBF placement-mapping scheme mismatch")
    if hbf.get("page_size") != PAGE or config.get("line_size") != PAGE:
        raise RuntimeError(f"{path}: EC page/line size must remain 4096 B")
    expected_media = {
        "read_ns": 1000.0,
        "program_ns": 95000.0,
        "program_verify_ns": 5000.0,
    }
    for key, expected in expected_media.items():
        actual = hbf.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(
                f"{path}: EC media timing mismatch for {key}: "
                f"expected {expected!r}, got {actual!r}")
    expected_ecc = {
        "ecc_decode_latency_ns": 250.0,
        "ecc_encode_latency_ns": 250.0,
        "ecc_decode_raw_bw_GBps_per_die": 105.46875,
        "ecc_encode_raw_bw_GBps_per_die": 105.46875,
        "ecc_codeword_size_bytes": 4320,
        "ecc_decode_initiation_ns": 40.96,
        "ecc_encode_initiation_ns": 40.96,
        "ecc_issue_topology": "shared-per-die",
    }
    for key, expected in expected_ecc.items():
        actual = hbf.get(key)
        if isinstance(expected, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected, rel_tol=0.0, abs_tol=1e-6)
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(
                f"{path}: EC ECC contract mismatch for {key}: "
                f"expected {expected!r}, got {actual!r}")
    if config.get("hbf_hbm_write_buffer_destage") != "deferred":
        raise RuntimeError(f"{path}: comparison-table destage must be deferred")
    simulator = data.get("simulator") or {}
    executable_digest = ((data.get("build") or {}).get("executable_digest") or {})
    if executable_digest.get("algorithm") != "sha256" or not executable_digest.get("value"):
        raise RuntimeError(f"{path}: executable digest is missing")
    commit = simulator.get("git_commit")
    dirty = simulator.get("git_dirty")
    if not isinstance(commit, str) or not isinstance(dirty, bool):
        raise RuntimeError(f"{path}: simulator revision provenance is missing")
    return executable_digest["value"], commit, dirty


def validate_manifest(
        replay_dir: Path, cases: tuple[tuple[str, str, str], ...],
        trace_digest: str, window: int
) -> int:
    suffix = "" if len(cases) == len(CASES) else "-" + "-".join(
        case[0] for case in cases)
    path = replay_dir / f"astra-replay{suffix}.manifest.json"
    data = json.loads(path.read_text())
    schema = data.get("schema") or {}
    if (schema.get("name"), schema.get("version")) != (
            "hbfsim.astra_replay.manifest", 2):
        raise RuntimeError(f"{path}: unsupported replay manifest schema")
    if data.get("cases") != [case[0] for case in cases]:
        raise RuntimeError(f"{path}: case set/order mismatch")
    boundary = data.get("boundary")
    if (type(boundary) is not int or boundary < 0 or boundary > UINT64_MAX):
        raise RuntimeError(f"{path}: invalid placement boundary")
    if data.get("window") != window:
        raise RuntimeError(f"{path}: closed-loop window mismatch")
    boundary_source = data.get("boundary_source")
    if boundary_source not in {"explicit", "workload_manifest.kv.base"}:
        raise RuntimeError(f"{path}: invalid placement-boundary provenance")
    trace = data.get("trace") or {}
    workload = data.get("workload_manifest") or {}
    workload_id = workload.get("workload_id")
    if (not isinstance(workload_id, str) or len(workload_id) != 64 or
            any(char not in "0123456789abcdef" for char in workload_id)):
        raise RuntimeError(f"{path}: invalid ASTRA workload identity")
    if trace.get("workload_id") != workload_id:
        raise RuntimeError(f"{path}: trace/workload identity mismatch")
    if boundary_source == "workload_manifest.kv.base" and (
            workload.get("kv_base") != boundary):
        raise RuntimeError(f"{path}: workload KV boundary mismatch")
    if trace.get("sha256") != trace_digest:
        raise RuntimeError(f"{path}: trace digest mismatch")
    workload_path = Path(workload.get("path", ""))
    workload_digest = workload.get("sha256")
    if (not workload_path.is_file() or not isinstance(workload_digest, str) or
            sha256_file(workload_path) != workload_digest):
        raise RuntimeError(f"{path}: workload manifest digest mismatch")
    configs = list(dict.fromkeys(case[1] for case in cases))
    expected_names = {
        *(f"{config}.summary.json" for config in configs),
        f"astra-replay{suffix}.csv",
        f"astra-replay{suffix}.time-breakdown.csv",
        f"astra-replay{suffix}.time-breakdown.md",
        f"astra-replay{suffix}.time-breakdown.html",
    }
    artifacts = data.get("artifacts") or {}
    if set(artifacts) != expected_names:
        raise RuntimeError(f"{path}: replay artifact set mismatch")
    for name in sorted(expected_names):
        artifact = replay_dir / name
        if not artifact.is_file() or artifacts[name].get("sha256") != sha256_file(artifact):
            raise RuntimeError(f"{path}: artifact digest mismatch for {name}")
    return boundary


def check_ec6_fill_conservation(trace: dict, scenario: dict) -> tuple[str, bool]:
    """Validate EC6 layer-DMA and writeback byte conservation."""
    streaming = scenario.get("layer_streaming") or {}
    hybrid = scenario.get("hybrid_path") or {}
    integer_fields = {
        "layers": streaming.get("layers"),
        "explicit_layer_requests": streaming.get("explicit_layer_requests"),
        "streamed_pages": streaming.get("streamed_pages"),
        "streamed_bytes": streaming.get("streamed_bytes"),
        "dirty_pages_written_back": streaming.get(
            "dirty_pages_written_back"),
        "writeback_bytes": streaming.get("writeback_bytes"),
        "data_pages": streaming.get("data_pages"),
        "resident_physical_pages": streaming.get("resident_physical_pages"),
        "hbm_only_resident_pages": streaming.get(
            "hbm_only_resident_pages"),
        "hot_kv_resident_pages": streaming.get("hot_kv_resident_pages"),
        "model_weight_resident_pages": streaming.get(
            "model_weight_resident_pages"),
        "backing_unique_pages": streaming.get("backing_unique_pages"),
        "effective_layer_buffer_pages": streaming.get(
            "effective_layer_buffer_pages"),
        "max_layer_data_pages": streaming.get("max_layer_data_pages"),
        "foreground_resident_page_accesses": streaming.get(
            "foreground_resident_page_accesses"),
        "foreground_buffer_page_accesses": streaming.get(
            "foreground_buffer_page_accesses"),
        "base_die_link_read_bytes": hybrid.get("base_die_link_read_bytes"),
        "base_die_link_write_bytes": hybrid.get("base_die_link_write_bytes"),
        "hbm_streaming_write_bytes": hybrid.get("hbm_streaming_write_bytes"),
        "hbf_static_read_bytes": hybrid.get("hbf_static_read_bytes"),
        "hbf_logical_read_bytes": hybrid.get("hbf_logical_read_bytes"),
        "hbf_backing_write_bytes": hybrid.get("hbf_backing_write_bytes"),
    }
    invalid_fields = [
        name for name, value in integer_fields.items()
        if type(value) is not int or value < 0
    ]
    if invalid_fields:
        invalid = ",".join(invalid_fields)
        return f"invalid or missing EC6 fill field(s): {invalid}", False
    read_bytes = integer_fields["streamed_bytes"]
    write_bytes = integer_fields["writeback_bytes"]
    ok = (
        integer_fields["layers"] > 0
        and integer_fields["layers"] == trace["streaming_layers"]
        and integer_fields["explicit_layer_requests"] ==
            trace["explicit_layer_requests"]
        and integer_fields["resident_physical_pages"] ==
            integer_fields["hbm_only_resident_pages"] +
            integer_fields["model_weight_resident_pages"] +
            integer_fields["hot_kv_resident_pages"]
        and integer_fields["data_pages"] ==
            integer_fields["model_weight_resident_pages"] +
            integer_fields["hot_kv_resident_pages"] +
            integer_fields["backing_unique_pages"]
        and integer_fields["effective_layer_buffer_pages"] ==
            integer_fields["max_layer_data_pages"]
        and integer_fields["foreground_resident_page_accesses"] +
            integer_fields["foreground_buffer_page_accesses"] ==
            trace["total_page_fragments"]
        and read_bytes == integer_fields["streamed_pages"] * PAGE
        and write_bytes == integer_fields["dirty_pages_written_back"] * PAGE
        and integer_fields["base_die_link_read_bytes"] == read_bytes
        and integer_fields["hbm_streaming_write_bytes"] == read_bytes
        and integer_fields["hbf_static_read_bytes"] +
            integer_fields["hbf_logical_read_bytes"] == read_bytes
        and integer_fields["base_die_link_write_bytes"] == write_bytes
        and integer_fields["hbf_backing_write_bytes"] == write_bytes
        and scenario.get("hbm_user_accesses") ==
            trace["total_page_fragments"]
        and scenario.get("hbf_user_accesses") == 0
    )
    expected_policy = (
        "capacity-aware-static-weight-prefix-v2"
        if streaming.get("explicit_residency_contract") is True
        else "trace-derived-first-touch"
    )
    ok = ok and (
        streaming.get("mode") == "layer_streaming"
        and streaming.get("residency_policy") == expected_policy
        and integer_fields["resident_physical_pages"] ==
            integer_fields["hbm_only_resident_pages"] +
            integer_fields["model_weight_resident_pages"] +
            integer_fields["hot_kv_resident_pages"]
        and streaming.get("compact_resident_mapping") ==
            (integer_fields["resident_physical_pages"] > 0)
    )
    detail = (
        f"layers {integer_fields['layers']:,}; HBF→D2D→HBM "
        f"{read_bytes:,} B; HBM→D2D→HBF {write_bytes:,} B")
    return detail, ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path,
                        default=Path("out/astra-replay-w512"))
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--cases", default=",".join(case[0] for case in CASES),
                        help="comma-separated experiment cases to check")
    args = parser.parse_args()
    try:
        cases = select_cases(args.cases)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    trace_digest = sha256_file(args.trace)
    boundary = validate_manifest(
        args.replay_dir, cases, trace_digest, args.window)
    t = census(args.trace, boundary)
    checks: list[tuple[str, str, str, bool]] = []

    checks.append(("check1", "trace 字节普查",
                   f"读 {t['rbytes']:,} + 写 {t['wbytes']:,} B, {t['ops']:,} ops",
                   True))

    blocks = {}
    summaries = {}
    provenance = set()
    for _ec, cfg, scen in cases:
        summary_path = args.replay_dir / f"{cfg}.summary.json"
        data = json.loads(summary_path.read_text())
        provenance.add(validate_summary(
            data, trace_digest, args.window, t["ops"], EXPECTED_STACKS[cfg],
            summary_path, boundary))
        summaries[cfg] = data
        blocks[(cfg, scen)] = next(s for s in data["scenarios"]
                                   if s["name"] == scen)
    if len(provenance) != 1:
        raise RuntimeError(
            "selected summaries were not produced by one executable/revision state")

    for ec, cfg, scen in cases:
        s = blocks[(cfg, scen)]
        h = s.get("hbf_stats") or {}

        if scen == "HBM-HBF-Flat":
            # Device-level truths (access counts legitimately diverge:
            # 64 KiB lines split, staged writes surface as HBM accesses):
            # HBF device logical READ bytes == above-boundary read bytes;
            # data programs == UNIQUE above-boundary write pages (raw write
            # bytes shrink by design: buffer/region coalescing).
            data_p = h.get("data_programs", 0)
            hp = s.get("hybrid_path") or {}
            region_bytes = summaries[cfg].get("config", {}).get(
                "hbf_hbm_write_buffer_bytes", 0)
            expected_resident_bytes = t["wpages_above"] * PAGE
            fits_without_pressure = region_bytes >= expected_resident_bytes
            ok = (fits_without_pressure
                  and hp.get("hbm_write_buffer_full_waits", -1) == 0
                  and hp.get("hbm_write_buffer_user_write_bytes", -1)
                      == t["wbytes_above"]
                  and h.get("logical_read_bytes", -1) == t["rbytes_above_hbf"]
                  and h.get("logical_write_bytes", -1) == t["dirty_union_bytes"]
                  and hp.get("hbm_write_buffer_peak_bytes", -1)
                      == expected_resident_bytes
                  and hp.get("hbm_write_buffer_destaged_bytes", -1)
                      == t["dirty_union_bytes"]
                  and data_p == t["wpages_above"])
            checks.append((f"check2/{ec}", "FLAT 边界拆分",
                           f"HBF逻辑读 {h.get('logical_read_bytes', -1):,}"
                           f"=={t['rbytes_above_hbf']:,} B(parked 重读 "
                           f"{t['rbytes_above_parked']:,} B 走 HBM), "
                           f"host写 {hp.get('hbm_write_buffer_user_write_bytes', -1):,} B, "
                           f"resident/destage {hp.get('hbm_write_buffer_peak_bytes', -1):,}/"
                           f"{hp.get('hbm_write_buffer_destaged_bytes', -1):,} B, "
                           f"dirty {t['dirty_union_bytes']:,} B, "
                           f"data页 {data_p}=={t['wpages_above']}", ok))

        if ec == "ec5":
            hp = s.get("hybrid_path") or {}
            expected_static = (
                t["static_direct_read_page_fragments"] * PAGE)
            ok = (
                hp.get("hbf_static_read_bytes", -1) == expected_static
                and h.get("logical_read_bytes", -1) == 0
                and h.get("logical_write_bytes", -1) == 0
                and h.get("data_programs", -1) == 0
                and hp.get("base_die_link_read_bytes", -1) == 0
                and hp.get("base_die_link_write_bytes", -1) == 0
            )
            checks.append(("check4/ec5", "静态直读绕FTL",
                           f"HBF physical {hp.get('hbf_static_read_bytes', -1):,}"
                           f"=={expected_static:,} B；FTL R/W="
                           f"{h.get('logical_read_bytes', -1)}/"
                           f"{h.get('logical_write_bytes', -1)} B", ok))

        if ec == "ec6":
            streaming = s.get("layer_streaming") or {}
            hbm_only_pages = t["hbm_only_pages"]
            foreground_pages = (
                streaming.get("foreground_resident_page_accesses", -1) +
                streaming.get("foreground_buffer_page_accesses", -1))
            ok = (
                streaming.get("hbm_only_resident_pages", -1) ==
                    hbm_only_pages
                and foreground_pages == t["total_page_fragments"]
                and s.get("hbm_user_accesses") ==
                    t["total_page_fragments"]
                and s.get("hbf_user_accesses") == 0
            )
            checks.append(("check4/ec6", "常驻/溢出分流",
                           f"mode={streaming.get('mode')}; HBM-only "
                           f"{streaming.get('hbm_only_resident_pages', -1)}"
                           f"=={hbm_only_pages}页；前台页访问 "
                           f"{foreground_pages}=={t['total_page_fragments']}", ok))

        lat = s["time_breakdown"]["latency_work"]["average_ns"]
        tp = s["user_completion_throughput_GBps"]
        latency_work = s["time_breakdown"]["latency_work"]
        waited = latency_work["front_end_admission_waited_ops"]
        wait_work = latency_work["front_end_admission_wait_work_ns"]
        wait_max = latency_work["front_end_admission_max_wait_ns"]
        causal = (
            math.isfinite(tp) and tp > 0
            and math.isfinite(lat) and lat >= 0
            and type(waited) is int and 0 <= waited <= t["ops"]
            and math.isfinite(wait_work) and wait_work >= 0
            and math.isfinite(wait_max) and 0 <= wait_max <= wait_work
        )
        checks.append((
            f"check3/{ec}",
            "页事务窗口/因果",
            f"W={args.window}; 吞吐 {tp:.0f} GB/s; "
            f"平均父请求延迟 {lat/1e3:.2f} µs; 等待父请求 {waited:,}",
            causal,
        ))

        programs = h.get("page_programs", 0)
        if programs:
            data_p = h.get("data_programs", 0)
            mapping_p = h.get("mapping_page_programs", 0)
            gc_p = h.get("gc_relocations", 0)
            ok = (h.get("physical_write_bytes", -1) == programs * PAGE
                  and programs == data_p + mapping_p + gc_p
                  and data_p > 0)
            checks.append((f"check6/{ec}", "program 组分守恒",
                           f"programs {programs} = data {data_p} + mapping "
                           f"{mapping_p} + gc {gc_p}", ok))

    if any(case[0] == "ec6" for case in cases):
        s6 = blocks[("usecase-6h2f", "HBM+HBF-layer-streaming")]
        detail, ok = check_ec6_fill_conservation(t, s6)
        checks.append(("check5/ec6", "级联填充放大",
                       detail, ok))

    failed = False
    print(f"{'check':12s} {'项':14s} {'数值':64s} 判定")
    for cid, name, detail, ok in checks:
        failed = failed or not ok
        print(f"{cid:12s} {name:14s} {detail:64s} {'PASS' if ok else 'FAIL'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
