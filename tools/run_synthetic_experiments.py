#!/usr/bin/env python3
"""Run a compact, controlled HBF/HBM characterization matrix.

This runner intentionally does not form the Cartesian product of every
workload parameter.  It covers four questions with small, orthogonal groups:

* ``raw``: sequential/random/stride/hotspot reads on all-HBM and all-HBF;
* ``balanced``: a working set spread across the 8 GiB FLAT boundary, including
  a read/write mix, on EC2--EC5;
* ``coverage``: sparse stratified-random traces spanning the native 384 GiB
  HBM and 4 TiB HBF address capacities;
* ``output``: matched 4 TiB direct-HBF reads comparing the current local
  outputs, one-output-per-subarray, and an array-supply upper bound at two
  outstanding page-transaction windows.

Every trace has a generator manifest, locality report, capacity/zoom logical
heatmap, simulator summary, and digest-bound suite manifest.  The runner first
withdraws stale suite commit records, writes individual artifacts atomically,
and publishes the manifest last as the success record only after every selected
run validates as a schema-v16 SANITY=PASS output.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, NoReturn

from analyze_trace_locality import TraceInputError, parse_trace_line
from plot_time_breakdown import write_visualization
from time_breakdown_report import SummaryInput, write_time_breakdown_report


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.certificate import (  # noqa: E402
    EXPLORATORY_VALIDATION,
    CertificateError,
    VerifiedCertificate,
    attach_certificate_to_summary,
    ensure_exploratory_summary,
    verify_certificate,
)

DEFAULT_OUT = Path("out/synthetic-characterization")
PAGE = 4096
FLAT_BOUNDARY = 8 * 1024**3
BALANCED_SPAN = 16 * 1024**3
BASELINE_HBM_CAPACITY = 384 * 1024**3
BASELINE_HBF_CAPACITY = 4 * 1024**4
COMMON_HEATMAP_CAPACITY = BASELINE_HBF_CAPACITY
OUTPUT_STACKS = 8
OUTPUT_CHANNELS_PER_STACK = 4
OUTPUT_DIES_PER_CHANNEL = 4
OUTPUT_PLANES_PER_DIE = 4
OUTPUT_SUBARRAYS_PER_PLANE = 32
# Static-direct consumes a source page, whose low mixed radices select
# plane/die/channel/stack (4*4*4*8=512) before the page coordinate selects
# subarray modulo 32.  Therefore 16,384 source-page residues, not PPN % 32,
# form one complete physical topology-leaf period.
OUTPUT_TOPOLOGY_LEAVES = (
    OUTPUT_STACKS * OUTPUT_CHANNELS_PER_STACK * OUTPUT_DIES_PER_CHANNEL *
    OUTPUT_PLANES_PER_DIE * OUTPUT_SUBARRAYS_PER_PLANE)

SCHEMA_NAME = "hbfsim.synthetic_experiment.manifest"
SCHEMA_VERSION = 5
SUMMARY_SCHEMA = {"name": "hbfsim.scenario_compare.summary", "version": 16}
GENERATOR_SCHEMA = {"name": "hbfsim.synthetic_trace.manifest", "version": 3}
LOCALITY_SCHEMA = {"name": "hbfsim.trace_locality", "version": 2}


@dataclass(frozen=True)
class Target:
    config: str
    scenarios: tuple[str, ...]


@dataclass(frozen=True)
class TraceCase:
    key: str
    group: str
    pattern: str
    pages: int
    passes: int
    address_span: int
    read_percent: int
    kind: str
    targets: tuple[Target, ...]
    stride_pages: int = 1
    hot_page_percent: int = 10
    hot_access_percent: int = 90
    window: int | None = None
    seed_offset: int = 0
    placement: str = "linear"
    placement_seed_offset: int = 0
    placement_modulus: int | None = None
    heatmap_capacity: int = COMMON_HEATMAP_CAPACITY
    static_direct_hbm_bytes: int = FLAT_BOUNDARY


RAW_TARGETS = (
    Target("usecase-baseline.cfg", ("all-HBM", "all-HBF")),
)
BALANCED_TARGETS = (
    Target("usecase-6h2f.cfg", ("HBM-HBF-Flat",)),
    Target("usecase-4h4f.cfg", ("HBM-HBF-Flat",)),
    Target(
        "usecase-2h6f.cfg",
        ("HBM-HBF-Flat", "HBF-static-direct-read"),
    ),
)
HBF_NATIVE_TARGETS = (
    Target("usecase-baseline.cfg", ("all-HBF",)),
)
OUTPUT_TARGETS = (
    Target("usecase-baseline.cfg", ("HBF-static-direct-read",)),
    Target(
        "usecase-baseline-local-output.cfg",
        ("HBF-static-direct-read",),
    ),
    Target(
        "usecase-baseline-output-upper-bound.cfg",
        ("HBF-static-direct-read",),
    ),
)

# All three output profiles retain 32 independently timed subarray sense
# domains per plane.  The local-output profiles remove the current 2:1
# subarray-to-lane/page-buffer sharing.  The upper bound does not invent more
# fabric ports: it raises the existing aggregate paths to the 1 microsecond
# bound, so these rates are part of the experiment contract.
OUTPUT_CONFIG_EXPECTATIONS: dict[str, dict[str, int | float]] = {
    "usecase-baseline.cfg": {
        "local_ports_per_plane": 16,
        "ecc_decode_raw_bw_GBps_per_die": 105.46875,
        "ecc_encode_raw_bw_GBps_per_die": 105.46875,
        "channel_bw_GBps": 421.875,
        "hbio_bw_GBps": 1600,
        "tsv_bw_GBps": 1712.5,
        "logic_sram_bw_GBps": 2048,
        "flash_tsu_issue_ns": 10,
        "logic_scheduler_issue_ns": 2,
    },
    "usecase-baseline-local-output.cfg": {
        "local_ports_per_plane": 32,
        "ecc_decode_raw_bw_GBps_per_die": 105.46875,
        "ecc_encode_raw_bw_GBps_per_die": 105.46875,
        "channel_bw_GBps": 421.875,
        "hbio_bw_GBps": 1600,
        "tsv_bw_GBps": 1712.5,
        "logic_sram_bw_GBps": 2048,
        "flash_tsu_issue_ns": 10,
        "logic_scheduler_issue_ns": 2,
    },
    "usecase-baseline-output-upper-bound.cfg": {
        "local_ports_per_plane": 32,
        "ecc_decode_raw_bw_GBps_per_die": 552.96,
        "ecc_encode_raw_bw_GBps_per_die": 552.96,
        "channel_bw_GBps": 2211.84,
        "hbio_bw_GBps": 8388.608,
        "tsv_bw_GBps": 8978.432,
        "logic_sram_bw_GBps": 8388.608,
        "flash_tsu_issue_ns": 7.8125,
        "logic_scheduler_issue_ns": 0.48828125,
    },
}


def build_plan(profile: str, groups: Iterable[str]) -> tuple[TraceCase, ...]:
    selected = set(groups)
    valid_groups = {"raw", "balanced", "coverage", "output"}
    if not selected or not selected <= valid_groups:
        raise ValueError(
            "groups must be a non-empty subset of "
            "raw,balanced,coverage,output")
    if profile == "smoke":
        # Multiples of ten make the 90/10 hotspot and 70/30 R/W contracts
        # exact rather than probabilistic approximations.
        raw_pages, coverage_pages = 1_000, 4_096
        output_pages, output_high_window = 16_384, 16_384
    elif profile == "core":
        raw_pages, coverage_pages = 16_000, 16_384
        output_pages, output_high_window = 65_536, 32_768
    else:
        raise ValueError("profile must be smoke or core")

    cases: list[TraceCase] = []
    if "raw" in selected:
        for key, pattern, address_span, seed_offset in (
            ("raw-sequential", "sequential", raw_pages * PAGE, 0),
            ("raw-random-permutation", "random-permutation",
             raw_pages * PAGE, 1),
            # Matched with packed random: only key/address span differ.  The
            # explicit shared seed offset prevents list insertion order from
            # silently changing the pair's random permutation.
            ("raw-random-permutation-spread", "random-permutation",
             BALANCED_SPAN, 1),
            ("raw-modular-stride", "modular-stride", raw_pages * PAGE, 2),
            ("raw-hotspot", "hotspot", raw_pages * PAGE, 3),
        ):
            # Ten hotspot passes guarantee that the 10% cold accesses touch
            # every page in the 90% cold tail at least once.
            passes = 10 if pattern == "hotspot" else 2
            cases.append(TraceCase(
                key=key,
                group="raw",
                pattern=pattern,
                pages=raw_pages,
                passes=passes,
                address_span=address_span,
                read_percent=100,
                kind="unknown",
                targets=RAW_TARGETS,
                stride_pages=17,
                seed_offset=seed_offset,
            ))
    if "balanced" in selected:
        for read_percent in (100, 70):
            cases.append(TraceCase(
                key=f"balanced-random-r{read_percent}",
                group="balanced",
                pattern="random-permutation",
                pages=raw_pages,
                passes=2,
                address_span=BALANCED_SPAN,
                read_percent=read_percent,
                kind="unknown",
                targets=BALANCED_TARGETS,
                # Matched R/W pair: identical page order and address placement.
                seed_offset=4,
            ))
    if "coverage" in selected:
        # Sixteen (smoke) or sixty-four (core) independently jittered pages per
        # 256-bin heatmap region.  Both cases share access and placement seeds;
        # the 384 GiB trace is a fair EC0/EC1 comparison, while the 4 TiB trace
        # is HBF-native because its addresses exceed the HBM device capacity.
        for key, capacity, targets in (
            ("coverage-hbm-native-384g", BASELINE_HBM_CAPACITY, RAW_TARGETS),
            ("coverage-hbf-native-4t", BASELINE_HBF_CAPACITY,
             HBF_NATIVE_TARGETS),
        ):
            cases.append(TraceCase(
                key=key,
                group="coverage",
                pattern="random-permutation",
                pages=coverage_pages,
                passes=2,
                address_span=capacity,
                read_percent=100,
                kind="unknown",
                targets=targets,
                seed_offset=9,
                placement="stratified-random",
                placement_seed_offset=10,
                heatmap_capacity=capacity,
            ))
    if "output" in selected:
        # These cases intentionally generate byte-identical traces.  Only the
        # outstanding window changes, so any per-target difference is caused
        # by concurrency rather than placement or access order.  A zero static
        # boundary sends every read through physical-direct HBF and bypasses
        # the mutable FTL completely.  One seeded permutation of the static
        # mixed-radix leaf period gives every subarray one smoke page; core has
        # four complete logical leaf periods. The simulator's block-local
        # reversible placement hash intentionally lets high address bits
        # influence the physical leaf, so runtime coverage is validated as a
        # balanced hash occupancy rather than assumed from raw address bits.
        for key, window in (
            ("output-static-direct-w512", 512),
            ("output-static-direct-high-concurrency", output_high_window),
        ):
            cases.append(TraceCase(
                key=key,
                group="output",
                pattern="random-permutation",
                pages=output_pages,
                passes=1,
                address_span=BASELINE_HBF_CAPACITY,
                read_percent=100,
                kind="unknown",
                targets=OUTPUT_TARGETS,
                window=window,
                seed_offset=11,
                placement="stratified-random",
                placement_seed_offset=12,
                placement_modulus=OUTPUT_TOPOLOGY_LEAVES,
                heatmap_capacity=BASELINE_HBF_CAPACITY,
                static_direct_hbm_bytes=0,
            ))
    return tuple(cases)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, role: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {role} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid {role} {path}: root must be an object")
    return value


def _required_object(value: dict, key: str, path: str) -> dict:
    child = value.get(key)
    if not isinstance(child, dict):
        raise RuntimeError(f"{path}.{key} must be an object")
    return child


def _required_int(value: dict, key: str, path: str) -> int:
    child = value.get(key)
    if type(child) is not int or child < 0:
        raise RuntimeError(f"{path}.{key} must be a non-negative integer")
    return child


def _required_number(value: dict, key: str, path: str) -> int | float:
    child = value.get(key)
    if (type(child) not in (int, float) or child < 0 or
            child == float("inf") or child == float("-inf") or
            child != child):
        raise RuntimeError(f"{path}.{key} must be a finite non-negative number")
    return child


def _required_bool(value: dict, key: str, path: str) -> bool:
    child = value.get(key)
    if type(child) is not bool:
        raise RuntimeError(f"{path}.{key} must be a boolean")
    return child


def _expect(value: dict, key: str, expected: object, path: str) -> None:
    actual = value.get(key)
    if type(expected) is int:
        valid = type(actual) is int and actual == expected
    elif expected is None:
        valid = actual is None and key in value
    else:
        valid = type(actual) is type(expected) and actual == expected
    if not valid:
        raise RuntimeError(
            f"{path}.{key}={actual!r}, expected {expected!r}")


def _snapshot(path: Path, role: str) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"{role} is not a file: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    identity_before = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"{role} changed while its digest was captured: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": after.st_size,
        "sha256": digest,
    }


def _verify_snapshot(path: Path, expected: dict[str, object], role: str) -> None:
    actual = _snapshot(path, role)
    if actual != expected:
        raise RuntimeError(
            f"{role} changed during the synthetic run: "
            f"expected {expected}, observed {actual}")


def invalidate_suite_outputs(out_dir: Path) -> tuple[Path, Path]:
    """Withdraw prior commit records before any work in a newly locked run."""
    manifest = out_dir / "synthetic-manifest.json"
    csv_path = out_dir / "synthetic-results.csv"
    for path in (
        manifest,
        csv_path,
        out_dir / "time-breakdown.csv",
        out_dir / "time-breakdown.md",
        out_dir / "time-breakdown.html",
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise RuntimeError(f"cannot withdraw stale artifact {path}: {error}") from error
    return manifest, csv_path


def validate_generator_manifest(
    case: TraceCase,
    trace: Path,
    manifest_path: Path,
    seed: int,
    placement_seed: int,
) -> dict:
    """Bind one generated trace and its v3 manifest to the planned case."""
    data = _load_json_object(manifest_path, "generator manifest")
    if data.get("schema") != GENERATOR_SCHEMA:
        raise RuntimeError(
            f"{manifest_path}: unsupported generator manifest schema "
            f"{data.get('schema')!r}")
    operations = case.pages * case.passes
    scaled_reads = operations * case.read_percent
    if scaled_reads % 100:
        raise RuntimeError(f"{case.key}: non-integral planned read count")
    read_ops = scaled_reads // 100
    write_ops = operations - read_ops
    is_hotspot = case.pattern == "hotspot"
    expected = {
        "pattern": case.pattern,
        "seed": seed,
        "placement": case.placement,
        "placement_seed": placement_seed,
        "placement_modulus": case.placement_modulus,
        "ops": operations,
        "read_ops": read_ops,
        "write_ops": write_ops,
        "request_bytes": PAGE,
        "bytes": operations * PAGE,
        "read_bytes": read_ops * PAGE,
        "write_bytes": write_ops * PAGE,
        "working_set_pages": case.pages,
        "unique_pages": case.pages,
        "base_address": 0,
        "address_span_bytes": case.address_span,
        "read_percent": str(case.read_percent),
        "kind": case.kind,
        "passes": case.passes,
        "stride_pages": case.stride_pages if case.pattern == "modular-stride" else None,
        "hot_page_percent": str(case.hot_page_percent) if is_hotspot else None,
        "hot_access_percent": str(case.hot_access_percent) if is_hotspot else None,
        "hot_pages": case.pages * case.hot_page_percent // 100 if is_hotspot else None,
        "hot_ops": operations * case.hot_access_percent // 100 if is_hotspot else None,
        "trace_file_bytes": trace.stat().st_size,
        "trace_path": str(trace.resolve()),
    }
    for key, expected_value in expected.items():
        _expect(data, key, expected_value, str(manifest_path))
    at_ns = data.get("at_ns")
    if type(at_ns) not in (int, float) or isinstance(at_ns, bool) or at_ns != 0:
        raise RuntimeError(f"{manifest_path}.at_ns={at_ns!r}, expected 0")
    digest = _required_object(data, "trace_digest", str(manifest_path))
    _expect(digest, "algorithm", "sha256", f"{manifest_path}.trace_digest")
    actual_sha = sha256_file(trace)
    _expect(digest, "value", actual_sha, f"{manifest_path}.trace_digest")
    return data


def validate_capacity_coverage(
    case: TraceCase,
    trace: Path,
    heatmap_bins: int,
) -> None:
    """Prove sparse native-capacity cases cover every bin uniformly.

    Sorting the unique addresses recovers logical-page stratum order because
    stratified placement uses non-overlapping ordered strata.  This checks the
    generated trace itself rather than trusting a visual impression or only
    checking its first and last address.
    """
    if case.group not in {"coverage", "output"}:
        return
    if case.placement != "stratified-random":
        raise RuntimeError(
            f"{case.key}: capacity-spanning case requires stratified-random")
    if case.address_span != case.heatmap_capacity:
        raise RuntimeError(
            f"{case.key}: address span must equal heatmap capacity")
    if case.pages % heatmap_bins:
        raise RuntimeError(
            f"{case.key}: {case.pages} pages must be divisible by "
            f"{heatmap_bins} heatmap bins")

    address_counts: dict[int, int] = {}
    address_sequence: list[int] = []
    operations = 0
    try:
        with trace.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                request = parse_trace_line(
                    line, line_no=line_no, line_size=PAGE)
                if request is None:
                    continue
                operations += 1
                if request.bytes != PAGE:
                    raise RuntimeError(
                        f"{case.key}: trace line {line_no} is not one page")
                if case.group == "output" and request.op != "R":
                    raise RuntimeError(
                        f"{case.key}: output trace line {line_no} is not a read")
                address_counts[request.address] = (
                    address_counts.get(request.address, 0) + 1)
                address_sequence.append(request.address)
    except (OSError, UnicodeError, TraceInputError) as error:
        raise RuntimeError(
            f"{case.key}: cannot validate coverage trace: {error}") from error

    expected_operations = case.pages * case.passes
    if operations != expected_operations or len(address_counts) != case.pages:
        raise RuntimeError(
            f"{case.key}: observed {operations} ops/"
            f"{len(address_counts)} unique "
            f"pages, expected {expected_operations}/{case.pages}")
    repeat_counts = set(address_counts.values())
    if repeat_counts != {case.passes}:
        raise RuntimeError(
            f"{case.key}: per-page access counts {sorted(repeat_counts)} do "
            f"not all equal passes={case.passes}")
    addresses = set(address_counts)
    for phase in range(case.passes):
        begin = phase * case.pages
        end = begin + case.pages
        if set(address_sequence[begin:end]) != addresses:
            raise RuntimeError(
                f"{case.key}: phase {phase} is not a complete working-set "
                "permutation")

    span_slots = case.address_span // PAGE
    ordered_addresses = sorted(addresses)
    ordered_slots: list[int] = []
    for page, address in enumerate(ordered_addresses):
        if address % PAGE:
            raise RuntimeError(f"{case.key}: unaligned address {address:#x}")
        slot = address // PAGE
        ordered_slots.append(slot)
        begin = page * span_slots // case.pages
        end = (page + 1) * span_slots // case.pages
        if not begin <= slot < end:
            raise RuntimeError(
                f"{case.key}: page {page} slot {slot} is outside "
                f"stratum [{begin}, {end})")

    if case.placement_modulus is not None:
        modulus = case.placement_modulus
        if case.pages % modulus:
            raise RuntimeError(
                f"{case.key}: pages must be divisible by placement modulus")
        expected_residues = set(range(modulus))
        for round_index, begin in enumerate(range(0, case.pages, modulus)):
            round_slots = ordered_slots[begin:begin + modulus]
            residues = {slot % modulus for slot in round_slots}
            if residues != expected_residues:
                raise RuntimeError(
                    f"{case.key}: placement round {round_index} does not "
                    f"cover all {modulus} slot residues")

    counts = [0] * heatmap_bins
    for address in addresses:
        if address + PAGE > case.heatmap_capacity:
            raise RuntimeError(
                f"{case.key}: request at {address:#x} exceeds capacity")
        counts[address * heatmap_bins // case.heatmap_capacity] += 1
    expected_per_bin = case.pages // heatmap_bins
    if any(count != expected_per_bin for count in counts):
        raise RuntimeError(
            f"{case.key}: unique-page bins are not uniform: "
            f"min={min(counts)}, max={max(counts)}, "
            f"expected={expected_per_bin}")


def validate_matched_output_traces(
    cases: Iterable[TraceCase],
    trace_by_key: dict[str, Path],
) -> None:
    """Require the two output-window traces to be byte-identical."""
    output_cases = [case for case in cases if case.group == "output"]
    if not output_cases:
        return
    if len(output_cases) != 2:
        raise RuntimeError(
            f"output group requires exactly two matched cases, got "
            f"{len(output_cases)}")
    comparable_fields = (
        "group", "pattern", "pages", "passes", "address_span",
        "read_percent", "kind", "targets", "stride_pages",
        "hot_page_percent", "hot_access_percent",
        "seed_offset", "placement", "placement_seed_offset",
        "placement_modulus", "heatmap_capacity", "static_direct_hbm_bytes",
    )
    left, right = output_cases
    for field in comparable_fields:
        if getattr(left, field) != getattr(right, field):
            raise RuntimeError(
                f"output cases are not matched at field {field}")
    try:
        left_digest = sha256_file(trace_by_key[left.key])
        right_digest = sha256_file(trace_by_key[right.key])
    except KeyError as error:
        raise RuntimeError(
            f"missing generated output trace for {error.args[0]}") from error
    if left_digest != right_digest:
        raise RuntimeError(
            "output window cases did not generate byte-identical traces")


def validate_locality_report(
    case: TraceCase,
    trace: Path,
    report_path: Path,
) -> dict:
    data = _load_json_object(report_path, "locality report")
    if data.get("schema") != LOCALITY_SCHEMA:
        raise RuntimeError(
            f"{report_path}: unsupported locality schema {data.get('schema')!r}")
    actual_sha = sha256_file(trace)
    _expect(data, "trace_sha256", actual_sha, str(report_path))
    overall = _required_object(data, "overall", str(report_path))
    operations = case.pages * case.passes
    reads = operations * case.read_percent // 100
    for key, expected in (
        ("ops", operations),
        ("reads", reads),
        ("writes", operations - reads),
        ("unique_occupied_pages", case.pages),
        ("unique_occupied_bytes", case.pages * PAGE),
    ):
        _expect(overall, key, expected, f"{report_path}.overall")
    traffic = _required_object(overall, "bytes", f"{report_path}.overall")
    for key, expected in (
        ("total", operations * PAGE),
        ("read", reads * PAGE),
        ("write", (operations - reads) * PAGE),
    ):
        _expect(traffic, key, expected, f"{report_path}.overall.bytes")
    return data


def validate_native_physical_coverage(
    case: TraceCase,
    data: dict,
    heatmap_bins: int,
) -> None:
    """Require every physical-device bin to receive capacity-spanning traffic."""
    if case.group not in {"coverage", "output"}:
        return
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list):
        raise RuntimeError(f"{case.key}: scenarios must be an array")
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise RuntimeError(f"{case.key}: scenario must be an object")
        name = scenario.get("name")
        if name == "all-HBM":
            domain_name = "hbm_physical"
            capacity = BASELINE_HBM_CAPACITY
        elif name in {"all-HBF", "HBF-static-direct-read"}:
            domain_name = "hbf_physical"
            capacity = BASELINE_HBF_CAPACITY
        else:
            raise RuntimeError(
                f"{case.key}: unsupported coverage scenario {name!r}")
        heatmap = _required_object(
            scenario, "address_heatmap", f"{case.key}.{name}")
        _expect(heatmap, "bin_count", heatmap_bins,
                f"{case.key}.{name}.address_heatmap")
        domains = heatmap.get("domains")
        if not isinstance(domains, list) or not all(
                isinstance(domain, dict) for domain in domains):
            raise RuntimeError(
                f"{case.key}.{name}.address_heatmap.domains must be an "
                "array of objects")
        matches = [domain for domain in domains
                   if domain.get("domain") == domain_name]
        if len(matches) != 1:
            raise RuntimeError(
                f"{case.key}.{name}: expected one {domain_name} domain")
        domain = matches[0]
        _expect(domain, "size_bytes", capacity,
                f"{case.key}.{name}.{domain_name}")
        bins = domain.get("bins")
        if (not isinstance(bins, list) or len(bins) != heatmap_bins or
                not all(isinstance(item, dict) for item in bins)):
            raise RuntimeError(
                f"{case.key}.{name}.{domain_name}.bins must contain exactly "
                f"{heatmap_bins} objects")
        occupied = 0
        for index, item in enumerate(bins):
            path = f"{case.key}.{name}.{domain_name}.bins[{index}]"
            read_bytes = _required_int(item, "read_bytes", path)
            write_bytes = _required_int(item, "write_bytes", path)
            erase_bytes = _required_int(item, "erase_bytes", path)
            if case.group == "output" and (write_bytes or erase_bytes):
                raise RuntimeError(
                    f"{path}: output physical coverage must be read-only")
            traffic = read_bytes if case.group == "output" else (
                read_bytes + write_bytes + erase_bytes)
            occupied += traffic != 0
        if occupied != heatmap_bins:
            raise RuntimeError(
                f"{case.key}.{name}.{domain_name}: occupied "
                f"{occupied}/{heatmap_bins} physical bins")


def validate_output_invariants(
    case: TraceCase,
    target: Target,
    data: dict,
) -> None:
    """Fail closed unless an output case is a pure physical-direct HBF read."""
    if case.group != "output":
        return
    if (case.pattern != "random-permutation" or
            case.placement != "stratified-random" or
            case.placement_modulus != OUTPUT_TOPOLOGY_LEAVES or
            case.passes != 1 or case.read_percent != 100 or
            case.address_span != BASELINE_HBF_CAPACITY or
            case.heatmap_capacity != BASELINE_HBF_CAPACITY or
            case.static_direct_hbm_bytes != 0):
        raise RuntimeError(f"{case.key}: unsupported output invariant case")

    expectation = OUTPUT_CONFIG_EXPECTATIONS.get(target.config)
    if expectation is None:
        raise RuntimeError(
            f"{case.key}: unsupported output target {target.config!r}")
    config = _required_object(data, "config", f"{case.key}.summary")
    _expect(config, "static_direct_hbm_bytes", 0, f"{case.key}.config")
    hbf_config = _required_object(config, "hbf", f"{case.key}.config")
    config_path = f"{case.key}.config.hbf"
    for key, expected in (
        ("capacity_bytes", BASELINE_HBF_CAPACITY),
        ("stacks", OUTPUT_STACKS),
        ("channels", OUTPUT_CHANNELS_PER_STACK),
        ("dies_per_channel", OUTPUT_DIES_PER_CHANNEL),
        ("planes_per_die", OUTPUT_PLANES_PER_DIE),
        ("subarrays_per_plane", OUTPUT_SUBARRAYS_PER_PLANE),
        ("media_lanes_per_plane", expectation["local_ports_per_plane"]),
        ("page_buffer_banks_per_plane",
         expectation["local_ports_per_plane"]),
        ("read_ns", 1000),
        ("flash_tsu_issue_ns", expectation["flash_tsu_issue_ns"]),
        ("media_lane_bw_GBps", 2048),
        ("page_buffer_bw_GBps", 2048),
        ("logic_scheduler_issue_ns",
         expectation["logic_scheduler_issue_ns"]),
        ("command_address_bytes", 64),
        ("batch_activation", True),
        ("ecc_decode_raw_bw_GBps_per_die",
         expectation["ecc_decode_raw_bw_GBps_per_die"]),
        ("ecc_encode_raw_bw_GBps_per_die",
         expectation["ecc_encode_raw_bw_GBps_per_die"]),
        ("channel_bw_GBps", expectation["channel_bw_GBps"]),
        ("hbio_bw_GBps", expectation["hbio_bw_GBps"]),
        ("tsv_bw_GBps", expectation["tsv_bw_GBps"]),
        ("logic_sram_bw_GBps", expectation["logic_sram_bw_GBps"]),
        ("ecc_issue_topology", "shared-per-die"),
    ):
        _expect(hbf_config, key, expected, config_path)
    expected_semantics = {
        "channel": "raw-codeword-per-channel",
        "tsv": "shared-command-and-raw-codeword-per-stack",
        "hbio": "decoded-payload-per-stack",
        "media_lane": "raw-codeword-per-lane",
        "page_buffer": "raw-codeword-per-bank",
        "logic_sram": "decoded-payload-per-stack",
    }
    _expect(
        hbf_config, "bandwidth_semantics", expected_semantics, config_path)

    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 1:
        raise RuntimeError(f"{case.key}: output summary requires one scenario")
    scenario = scenarios[0]
    if (not isinstance(scenario, dict) or
            scenario.get("name") != "HBF-static-direct-read"):
        raise RuntimeError(
            f"{case.key}: output summary requires HBF-static-direct-read")

    operations = case.pages
    logical_bytes = operations * PAGE
    scenario_path = f"{case.key}.HBF-static-direct-read"
    for key, expected in (
        ("hbm_accesses", 0),
        ("hbm_user_accesses", 0),
        ("hbm_background_accesses", 0),
        ("hbf_accesses", operations),
        ("hbf_user_accesses", operations),
        ("hbf_background_accesses", 0),
    ):
        _expect(scenario, key, expected, scenario_path)

    hybrid = _required_object(scenario, "hybrid_path", scenario_path)
    for key, expected in (
        ("hbf_static_read_bytes", logical_bytes),
        ("hbf_logical_read_bytes", 0),
        ("hbf_backing_write_bytes", 0),
        ("background_hbf_writes", 0),
        ("hbm_foreground_bytes", 0),
        ("hbm_streaming_write_bytes", 0),
        ("base_die_link_read_bytes", 0),
        ("base_die_link_write_bytes", 0),
    ):
        _expect(hybrid, key, expected, f"{scenario_path}.hybrid_path")

    hbm_stats = _required_object(scenario, "hbm_stats", scenario_path)
    for key in ("read_bytes", "write_bytes"):
        _expect(hbm_stats, key, 0, f"{scenario_path}.hbm_stats")

    hbf_stats = _required_object(scenario, "hbf_stats", scenario_path)
    for key, expected in (
        # Static direct reads address preplaced physical pages and deliberately
        # bypass the FTL. Keep them in hbf_static_read_bytes instead of
        # misclassifying them as logical/FTL traffic.
        ("logical_read_bytes", 0),
        ("physical_read_bytes", logical_bytes),
        ("logical_write_bytes", 0),
        ("physical_write_bytes", 0),
        ("page_reads", operations),
        ("read_buffer_hits", 0),
        ("read_buffer_misses", operations),
        ("mapping_entries", 0),
        ("mapping_lookup_ops", 0),
        ("mapping_update_ops", 0),
        ("mapping_page_programs", 0),
        ("data_programs", 0),
        ("page_programs", 0),
        ("block_erases", 0),
        ("gc_runs", 0),
        ("gc_relocations", 0),
        ("gc_user_blocked_runs", 0),
    ):
        _expect(hbf_stats, key, expected, f"{scenario_path}.hbf_stats")

    for key, expected in (
        ("channels", OUTPUT_STACKS * OUTPUT_CHANNELS_PER_STACK),
        ("active_channels", OUTPUT_STACKS * OUTPUT_CHANNELS_PER_STACK),
        ("dies", OUTPUT_STACKS * OUTPUT_CHANNELS_PER_STACK *
         OUTPUT_DIES_PER_CHANNEL),
        ("active_dies", OUTPUT_STACKS * OUTPUT_CHANNELS_PER_STACK *
         OUTPUT_DIES_PER_CHANNEL),
        ("planes", OUTPUT_STACKS * OUTPUT_CHANNELS_PER_STACK *
         OUTPUT_DIES_PER_CHANNEL * OUTPUT_PLANES_PER_DIE),
        ("active_planes", OUTPUT_STACKS * OUTPUT_CHANNELS_PER_STACK *
         OUTPUT_DIES_PER_CHANNEL * OUTPUT_PLANES_PER_DIE),
        ("subarrays", OUTPUT_TOPOLOGY_LEAVES),
    ):
        _expect(hbf_stats, key, expected, f"{scenario_path}.hbf_stats")
    active_subarrays = _required_int(
        hbf_stats, "active_subarrays", f"{scenario_path}.hbf_stats")
    placement_rounds = case.pages // OUTPUT_TOPOLOGY_LEAVES
    expected_fraction = 1.0 - math.exp(-placement_rounds)
    minimum_active = math.floor(
        OUTPUT_TOPOLOGY_LEAVES * expected_fraction * 0.9)
    if not minimum_active <= active_subarrays <= OUTPUT_TOPOLOGY_LEAVES:
        raise RuntimeError(
            f"{scenario_path}.hbf_stats.active_subarrays="
            f"{active_subarrays}, expected balanced hashed occupancy in "
            f"[{minimum_active},{OUTPUT_TOPOLOGY_LEAVES}]")
    max_subarray_reads = _required_int(
        hbf_stats, "max_subarray_reads", f"{scenario_path}.hbf_stats")
    average_reads = _required_number(
        hbf_stats, "avg_active_subarray_reads", f"{scenario_path}.hbf_stats")
    read_skew = _required_number(
        hbf_stats, "subarray_read_skew", f"{scenario_path}.hbf_stats")
    expected_average = operations / active_subarrays
    if not math.isclose(
            average_reads, expected_average, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(
            f"{scenario_path}.hbf_stats active_subarrays/"
            "avg_active_subarray_reads do not conserve page transactions")
    if not (
            math.ceil(expected_average) <= max_subarray_reads <= operations and
            math.isclose(
                read_skew,
                max_subarray_reads / average_reads,
                rel_tol=1e-12,
                abs_tol=1e-12)):
        raise RuntimeError(
            f"{scenario_path}.hbf_stats max_subarray_reads/"
            "subarray_read_skew are inconsistent")


def validate_summary(
    case: TraceCase,
    target: Target,
    data: dict,
    trace: Path,
    executable_snapshot: dict[str, object],
    effective_window: int,
    heatmap_bins: int,
) -> None:
    path = f"{case.key}/{target.config} summary"
    if data.get("schema") != SUMMARY_SCHEMA:
        raise RuntimeError(f"{path}: unsupported summary schema")
    if data.get("sanity") != "PASS":
        raise RuntimeError(f"{path}: SANITY={data.get('sanity')!r}")
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, list) or not all(
            isinstance(item, dict) for item in scenarios):
        raise RuntimeError(f"{path}.scenarios must be an array of objects")
    names = [scenario.get("name") for scenario in scenarios]
    if not all(isinstance(name, str) for name in names):
        raise RuntimeError(f"{path}: every scenario name must be a string")
    if len(names) != len(set(names)):
        raise RuntimeError(f"{path}: duplicate scenario names {names}")
    if set(names) != set(target.scenarios) or len(names) != len(target.scenarios):
        raise RuntimeError(
            f"{path}: scenario set {names} does not match {list(target.scenarios)}")

    workload = _required_object(data, "workload", path)
    digest = _required_object(workload, "trace_digest", f"{path}.workload")
    _expect(digest, "algorithm", "sha256", f"{path}.workload.trace_digest")
    _expect(digest, "value", sha256_file(trace), f"{path}.workload.trace_digest")
    _expect(workload, "trace_file_bytes", trace.stat().st_size, f"{path}.workload")
    build = _required_object(data, "build", path)
    executable_digest = _required_object(build, "executable_digest", f"{path}.build")
    _expect(executable_digest, "algorithm", "sha256",
            f"{path}.build.executable_digest")
    _expect(executable_digest, "value", executable_snapshot["sha256"],
            f"{path}.build.executable_digest")
    _expect(build, "executable_file_bytes", executable_snapshot["bytes"],
            f"{path}.build")
    config = _required_object(data, "config", path)
    operations = case.pages * case.passes
    for key, expected in (
        ("ops", operations),
        ("max_outstanding_requests", effective_window),
        ("address_heatmap_bins", heatmap_bins),
        ("static_direct_hbm_bytes", case.static_direct_hbm_bytes),
    ):
        _expect(config, key, expected, f"{path}.config")
    if case.group == "coverage":
        _expect(config, "hbm_capacity_bytes", BASELINE_HBM_CAPACITY,
                f"{path}.config")
        hbf = _required_object(config, "hbf", f"{path}.config")
        _expect(hbf, "capacity_bytes", BASELINE_HBF_CAPACITY,
                f"{path}.config.hbf")
    reads = operations * case.read_percent // 100
    for index, scenario in enumerate(scenarios):
        scenario_path = f"{path}.scenarios[{index}]"
        for key, expected in (
            ("ops", operations),
            ("reads", reads),
            ("writes", operations - reads),
            ("logical_bytes", operations * PAGE),
        ):
            _expect(scenario, key, expected, scenario_path)
    validate_native_physical_coverage(case, data, heatmap_bins)
    validate_output_invariants(case, target, data)


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed ({result.returncode}): {rendered}\n"
            f"stdout tail:\n{result.stdout[-2000:]}\n"
            f"stderr tail:\n{result.stderr[-2000:]}")
    return result


def _generator_command(
    case: TraceCase,
    trace: Path,
    manifest: Path,
    generator: Path,
    seed: int,
    placement_seed: int,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        str(generator),
        "--output",
        str(trace),
        "--manifest",
        str(manifest),
        "--pattern",
        case.pattern,
        "--pages",
        str(case.pages),
        "--passes",
        str(case.passes),
        "--bytes",
        str(PAGE),
        "--base",
        "0",
        "--address-span",
        str(case.address_span),
        "--read-percent",
        str(case.read_percent),
        "--seed",
        str(seed),
        "--placement",
        case.placement,
        "--placement-seed",
        str(placement_seed),
        "--kind",
        case.kind,
        "--at",
        "0",
        "--phase-comments",
    ]
    if case.pattern == "modular-stride":
        command.extend(("--stride-pages", str(case.stride_pages)))
    if case.placement_modulus is not None:
        command.extend((
            "--placement-modulus", str(case.placement_modulus)))
    if case.pattern == "hotspot":
        command.extend((
            "--hot-page-percent", str(case.hot_page_percent),
            "--hot-access-percent", str(case.hot_access_percent),
        ))
    return command


def _run_target(
    *,
    case: TraceCase,
    target: Target,
    trace: Path,
    scenario_compare: Path,
    config_dir: Path,
    runs_dir: Path,
    window: int,
    heatmap_bins: int,
    timeout: int,
    executable_snapshot: dict[str, object],
    config_snapshot: dict[str, object],
    validation_certificate: VerifiedCertificate | None,
) -> dict:
    case_dir = runs_dir / case.key
    case_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(target.config).stem
    summary = case_dir / f"{stem}.summary.json"
    temporary = summary.with_name(f".{summary.name}.{os.getpid()}.tmp")
    effective_window = case.window if case.window is not None else window
    config_path = config_dir / target.config
    command = [
        str(scenario_compare.resolve()),
        "--config",
        str(config_path.resolve()),
        "--trace",
        str(trace.resolve()),
        "--scenarios",
        ",".join(target.scenarios),
        "--flat-hbm-bytes",
        str(FLAT_BOUNDARY),
        "--static-direct-hbm-bytes",
        str(case.static_direct_hbm_bytes),
        "--max-outstanding-requests",
        str(effective_window),
        "--address-heatmap-bins",
        str(heatmap_bins),
        "--summary-json",
        str(temporary.resolve()),
    ]
    started = time.monotonic()
    try:
        _verify_snapshot(scenario_compare, executable_snapshot, "scenario_compare")
        _verify_snapshot(config_path, config_snapshot, f"config {target.config}")
        completed = _run(command, timeout=timeout)
        host_seconds = time.monotonic() - started
        _verify_snapshot(scenario_compare, executable_snapshot, "scenario_compare")
        _verify_snapshot(config_path, config_snapshot, f"config {target.config}")
        data = _load_json_object(temporary, "scenario summary")
        if validation_certificate is None:
            ensure_exploratory_summary(data)
        else:
            data = attach_certificate_to_summary(
                data, validation_certificate)
            _write_atomic(
                temporary,
                json.dumps(data, indent=2) + "\n",
            )
        validate_summary(
            case, target, data, trace, executable_snapshot,
            effective_window, heatmap_bins)
        os.replace(temporary, summary)
        log = case_dir / f"{stem}.log"
        _write_atomic(log, completed.stdout + completed.stderr)
        return {
            "case": case,
            "target": target,
            "summary": summary,
            "host_seconds": host_seconds,
            "window": effective_window,
            "data": data,
            "log": log,
        }
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


CSV_FIELDS = (
    "case",
    "group",
    "config",
    "scenario",
    "pattern",
    "placement",
    "placement_modulus",
    "address_span_bytes",
    "heatmap_capacity_bytes",
    "working_set_pages",
    "passes",
    "read_percent",
    "window",
    "static_direct_hbm_bytes",
    "hbf_subarrays_per_plane",
    "hbf_media_lanes_per_plane",
    "hbf_page_buffer_banks_per_plane",
    "hbf_read_ns",
    "hbf_flash_tsu_issue_ns",
    "hbf_media_lane_bw_GBps",
    "hbf_page_buffer_bw_GBps",
    "hbf_logic_scheduler_issue_ns",
    "hbf_command_address_bytes",
    "hbf_batch_activation",
    "hbf_ecc_decode_raw_bw_GBps_per_die",
    "hbf_channel_bw_GBps",
    "hbf_tsv_bw_GBps",
    "hbf_logic_sram_bw_GBps",
    "hbf_hbio_bw_GBps",
    "hbf_active_channels",
    "hbf_active_dies",
    "hbf_active_planes",
    "hbf_active_subarrays",
    "hbf_max_subarray_reads",
    "hbf_avg_active_subarray_reads",
    "hbf_subarray_read_skew",
    "host_seconds",
    "user_completion_GBps",
    "makespan_GBps",
    "user_completion_ms",
    "drain_ms",
    "makespan_ms",
    "avg_us",
    "p50_us",
    "p95_us",
    "max_us",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "hbf_physical_read_bytes",
    "hbf_physical_write_bytes",
    "hbf_page_reads",
    "hbf_page_programs",
    "streaming_layers",
    "streaming_pages",
    "streaming_bytes",
    "streaming_foreground_page_accesses",
    "streaming_dirty_pages_written_back",
    "streaming_writeback_bytes",
    "streaming_exposed_prefetch_ns",
    "streaming_hidden_prefetch_ns",
)


def _nullable_stats(
    scenario: dict,
    key: str,
    path: str,
    *,
    allow_null: bool = True,
) -> dict | None:
    if key not in scenario:
        raise RuntimeError(f"{path}.{key} is missing")
    value = scenario[key]
    if value is None and not allow_null:
        raise RuntimeError(f"{path}.{key} must be an object")
    if value is not None and not isinstance(value, dict):
        raise RuntimeError(f"{path}.{key} must be an object or null")
    return value


def _stats_counter(stats: dict | None, key: str, path: str) -> int:
    return 0 if stats is None else _required_int(stats, key, path)


def _stats_number(
    stats: dict | None,
    key: str,
    path: str,
) -> int | float:
    return 0 if stats is None else _required_number(stats, key, path)


def _rows(result: dict) -> list[dict[str, object]]:
    case: TraceCase = result["case"]
    target: Target = result["target"]
    summary_config = _required_object(
        result["data"], "config", f"{case.key}/{target.config}")
    hbf_config = _required_object(
        summary_config, "hbf", f"{case.key}/{target.config}.config")
    rows = []
    for index, scenario in enumerate(result["data"]["scenarios"]):
        scenario_path = (
            f"{case.key}/{target.config}.scenarios[{index}]({scenario.get('name')})")
        wall = scenario["time_breakdown"]["wall_clock_ns"]
        latency = scenario["time_breakdown"]["latency_work"]
        hbm = _nullable_stats(scenario, "hbm_stats", scenario_path)
        hbf = _nullable_stats(scenario, "hbf_stats", scenario_path)
        streaming = _nullable_stats(scenario, "layer_streaming", scenario_path)
        rows.append({
            "case": case.key,
            "group": case.group,
            "config": target.config,
            "scenario": scenario["name"],
            "pattern": case.pattern,
            "placement": case.placement,
            "placement_modulus": case.placement_modulus,
            "address_span_bytes": case.address_span,
            "heatmap_capacity_bytes": case.heatmap_capacity,
            "working_set_pages": case.pages,
            "passes": case.passes,
            "read_percent": case.read_percent,
            "window": result["window"],
            "static_direct_hbm_bytes": case.static_direct_hbm_bytes,
            "hbf_subarrays_per_plane": _required_int(
                hbf_config, "subarrays_per_plane", "summary.config.hbf"),
            "hbf_media_lanes_per_plane": _required_int(
                hbf_config, "media_lanes_per_plane", "summary.config.hbf"),
            "hbf_page_buffer_banks_per_plane": _required_int(
                hbf_config, "page_buffer_banks_per_plane",
                "summary.config.hbf"),
            "hbf_read_ns": _required_number(
                hbf_config, "read_ns", "summary.config.hbf"),
            "hbf_flash_tsu_issue_ns": _required_number(
                hbf_config, "flash_tsu_issue_ns", "summary.config.hbf"),
            "hbf_media_lane_bw_GBps": _required_number(
                hbf_config, "media_lane_bw_GBps", "summary.config.hbf"),
            "hbf_page_buffer_bw_GBps": _required_number(
                hbf_config, "page_buffer_bw_GBps", "summary.config.hbf"),
            "hbf_logic_scheduler_issue_ns": _required_number(
                hbf_config, "logic_scheduler_issue_ns",
                "summary.config.hbf"),
            "hbf_command_address_bytes": _required_int(
                hbf_config, "command_address_bytes", "summary.config.hbf"),
            "hbf_batch_activation": _required_bool(
                hbf_config, "batch_activation", "summary.config.hbf"),
            "hbf_ecc_decode_raw_bw_GBps_per_die": _required_number(
                hbf_config, "ecc_decode_raw_bw_GBps_per_die",
                "summary.config.hbf"),
            "hbf_channel_bw_GBps": _required_number(
                hbf_config, "channel_bw_GBps", "summary.config.hbf"),
            "hbf_tsv_bw_GBps": _required_number(
                hbf_config, "tsv_bw_GBps", "summary.config.hbf"),
            "hbf_logic_sram_bw_GBps": _required_number(
                hbf_config, "logic_sram_bw_GBps", "summary.config.hbf"),
            "hbf_hbio_bw_GBps": _required_number(
                hbf_config, "hbio_bw_GBps", "summary.config.hbf"),
            "hbf_active_channels": _stats_counter(
                hbf, "active_channels", f"{scenario_path}.hbf_stats"),
            "hbf_active_dies": _stats_counter(
                hbf, "active_dies", f"{scenario_path}.hbf_stats"),
            "hbf_active_planes": _stats_counter(
                hbf, "active_planes", f"{scenario_path}.hbf_stats"),
            "hbf_active_subarrays": _stats_counter(
                hbf, "active_subarrays", f"{scenario_path}.hbf_stats"),
            "hbf_max_subarray_reads": _stats_counter(
                hbf, "max_subarray_reads", f"{scenario_path}.hbf_stats"),
            "hbf_avg_active_subarray_reads": _stats_number(
                hbf, "avg_active_subarray_reads",
                f"{scenario_path}.hbf_stats"),
            "hbf_subarray_read_skew": _stats_number(
                hbf, "subarray_read_skew", f"{scenario_path}.hbf_stats"),
            "host_seconds": f"{result['host_seconds']:.6f}",
            "user_completion_GBps": scenario["user_completion_throughput_GBps"],
            "makespan_GBps": scenario["makespan_throughput_GBps"],
            "user_completion_ms": wall["user_completion_span_ns"] / 1e6,
            "drain_ms": wall["drain_tail_ns"] / 1e6,
            "makespan_ms": wall["makespan_ns"] / 1e6,
            "avg_us": latency["average_ns"] / 1e3,
            "p50_us": latency["p50_ns"] / 1e3,
            "p95_us": latency["p95_ns"] / 1e3,
            "max_us": latency["max_ns"] / 1e3,
            "hbm_read_bytes": _stats_counter(
                hbm, "read_bytes", f"{scenario_path}.hbm_stats"),
            "hbm_write_bytes": _stats_counter(
                hbm, "write_bytes", f"{scenario_path}.hbm_stats"),
            "hbf_physical_read_bytes": _stats_counter(
                hbf, "physical_read_bytes", f"{scenario_path}.hbf_stats"),
            "hbf_physical_write_bytes": _stats_counter(
                hbf, "physical_write_bytes", f"{scenario_path}.hbf_stats"),
            "hbf_page_reads": _stats_counter(
                hbf, "page_reads", f"{scenario_path}.hbf_stats"),
            "hbf_page_programs": _stats_counter(
                hbf, "page_programs", f"{scenario_path}.hbf_stats"),
            "streaming_layers": _stats_counter(
                streaming, "layers", f"{scenario_path}.layer_streaming"),
            "streaming_pages": _stats_counter(
                streaming, "streamed_pages",
                f"{scenario_path}.layer_streaming"),
            "streaming_bytes": _stats_counter(
                streaming, "streamed_bytes",
                f"{scenario_path}.layer_streaming"),
            "streaming_foreground_page_accesses": _stats_counter(
                streaming, "foreground_buffer_page_accesses",
                f"{scenario_path}.layer_streaming"),
            "streaming_dirty_pages_written_back": _stats_counter(
                streaming, "dirty_pages_written_back",
                f"{scenario_path}.layer_streaming"),
            "streaming_writeback_bytes": _stats_counter(
                streaming, "writeback_bytes",
                f"{scenario_path}.layer_streaming"),
            "streaming_exposed_prefetch_ns": _stats_number(
                streaming, "exposed_prefetch_ns",
                f"{scenario_path}.layer_streaming"),
            "streaming_hidden_prefetch_ns": _stats_number(
                streaming, "hidden_prefetch_ns",
                f"{scenario_path}.layer_streaming"),
        })
    return rows


def _csv_payload(rows: list[dict[str, object]]) -> str:
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _positive(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "core"), default="smoke")
    parser.add_argument(
        "--groups", default="raw,balanced,coverage,output")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scenario-compare", type=Path,
                        default=ROOT / "build" / "scenario_compare")
    parser.add_argument("--config-dir", type=Path,
                        default=ROOT / "configs" / "scenario_compare")
    parser.add_argument("--generator", type=Path,
                        default=ROOT / "tools" / "generate_synthetic_trace.py")
    parser.add_argument("--analyzer", type=Path,
                        default=ROOT / "tools" / "analyze_trace_locality.py")
    parser.add_argument("--heatmap", type=Path,
                        default=ROOT / "tools" / "plot_trace_heatmap.py")
    parser.add_argument("--window", type=_nonnegative, default=512)
    parser.add_argument("--jobs", type=_positive, default=2)
    parser.add_argument("--heatmap-bins", type=_positive, default=256)
    parser.add_argument("--seed", type=_nonnegative, default=1)
    parser.add_argument("--timeout", type=_positive, default=7200)
    parser.add_argument(
        "--validation-certificate",
        type=Path,
        help=(
            "attach a verified foundational certificate; omitted runs remain "
            "explicitly exploratory"
        ),
    )
    args = parser.parse_args()

    groups = tuple(item.strip() for item in args.groups.split(",") if item.strip())
    try:
        plan = build_plan(args.profile, groups)
    except ValueError as error:
        parser.error(str(error))
    for case in plan:
        if (case.group in {"coverage", "output"} and
                case.pages % args.heatmap_bins):
            parser.error(
                f"capacity-spanning case {case.key} has {case.pages} pages, "
                "which "
                f"must be divisible by --heatmap-bins={args.heatmap_bins}")
    for path, label in (
        (args.scenario_compare, "scenario_compare"),
        (args.generator, "generator"),
        (args.analyzer, "analyzer"),
        (args.heatmap, "heatmap"),
    ):
        if not path.is_file():
            parser.error(f"{label} does not exist: {path}")
    validation_certificate: VerifiedCertificate | None = None
    if args.validation_certificate is not None:
        try:
            validation_certificate = verify_certificate(
                args.validation_certificate,
                repository=ROOT,
                scenario_compare=args.scenario_compare,
            )
        except CertificateError as error:
            parser.error(f"invalid validation certificate: {error}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.out_dir / ".synthetic-experiments.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"another synthetic run is using {args.out_dir}")

        # The manifest is the suite commit record. Withdraw it and the old CSV
        # as soon as this process owns the lock, so any later failure cannot be
        # mistaken for a successful fresh run.
        manifest_path, csv_path = invalidate_suite_outputs(args.out_dir)
        tool_paths = {
            "runner": Path(__file__).resolve(),
            "scenario_compare": args.scenario_compare,
            "generator": args.generator,
            "analyzer": args.analyzer,
            "heatmap": args.heatmap,
            "time_breakdown_report":
                ROOT / "tools" / "time_breakdown_report.py",
            "time_breakdown_visualization":
                ROOT / "tools" / "plot_time_breakdown.py",
        }
        tool_snapshots = {
            name: _snapshot(path, name) for name, path in tool_paths.items()
        }
        target_configs = sorted({target.config for case in plan
                                 for target in case.targets})
        config_paths = {
            name: args.config_dir / name for name in target_configs
        }
        config_snapshots = {
            name: _snapshot(path, f"config {name}")
            for name, path in config_paths.items()
        }

        traces_dir = args.out_dir / "traces"
        reports_dir = args.out_dir / "reports"
        runs_dir = args.out_dir / "runs"
        traces_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        case_artifacts: dict[str, dict[str, Path]] = {}
        for case in plan:
            trace = traces_dir / f"{case.key}.trace"
            generator_manifest = traces_dir / f"{case.key}.manifest.json"
            locality = reports_dir / f"{case.key}.locality.json"
            heatmap = reports_dir / f"{case.key}.heatmap.html"
            effective_seed = args.seed + case.seed_offset
            effective_placement_seed = (
                args.seed + case.placement_seed_offset)
            _verify_snapshot(
                args.generator, tool_snapshots["generator"], "generator")
            _run(_generator_command(
                case, trace, generator_manifest, args.generator,
                effective_seed, effective_placement_seed), timeout=args.timeout)
            _verify_snapshot(
                args.generator, tool_snapshots["generator"], "generator")
            validate_generator_manifest(
                case, trace, generator_manifest, effective_seed,
                effective_placement_seed)
            validate_capacity_coverage(case, trace, args.heatmap_bins)
            _verify_snapshot(args.analyzer, tool_snapshots["analyzer"], "analyzer")
            _run([
                sys.executable, "-B", str(args.analyzer),
                "--trace", str(trace), "--output", str(locality),
                "--line-size", str(PAGE), "--page-size", str(PAGE),
            ], timeout=args.timeout)
            _verify_snapshot(args.analyzer, tool_snapshots["analyzer"], "analyzer")
            validate_locality_report(case, trace, locality)
            _verify_snapshot(args.heatmap, tool_snapshots["heatmap"], "heatmap")
            _run([
                sys.executable, "-B", str(args.heatmap),
                "--trace", str(trace), "--output", str(heatmap),
                "--capacity-bytes", str(case.heatmap_capacity),
                "--bins", str(args.heatmap_bins),
                "--line-size", str(PAGE),
            ], timeout=args.timeout)
            _verify_snapshot(args.heatmap, tool_snapshots["heatmap"], "heatmap")
            case_artifacts[case.key] = {
                "trace": trace,
                "generator_manifest": generator_manifest,
                "locality": locality,
                "heatmap": heatmap,
            }

        validate_matched_output_traces(
            plan,
            {key: artifacts["trace"]
             for key, artifacts in case_artifacts.items()},
        )

        futures = []
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for case in plan:
                for target in case.targets:
                    futures.append(executor.submit(
                        _run_target,
                        case=case,
                        target=target,
                        trace=case_artifacts[case.key]["trace"],
                        scenario_compare=args.scenario_compare,
                        config_dir=args.config_dir,
                        runs_dir=runs_dir,
                        window=args.window,
                        heatmap_bins=args.heatmap_bins,
                        timeout=args.timeout,
                        executable_snapshot=tool_snapshots["scenario_compare"],
                        config_snapshot=config_snapshots[target.config],
                        validation_certificate=validation_certificate,
                    ))
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda item: (
            item["case"].key, item["target"].config))
        rows = [row for result in results for row in _rows(result)]
        rows.sort(key=lambda row: (
            str(row["case"]), str(row["config"]), str(row["scenario"])))
        for name, path in tool_paths.items():
            _verify_snapshot(path, tool_snapshots[name], name)
        for name, path in config_paths.items():
            _verify_snapshot(path, config_snapshots[name], f"config {name}")
        _write_atomic(csv_path, _csv_payload(rows))
        time_breakdown_csv = args.out_dir / "time-breakdown.csv"
        time_breakdown_markdown = args.out_dir / "time-breakdown.md"
        time_breakdown_html = args.out_dir / "time-breakdown.html"
        time_inputs = [
            SummaryInput(
                label=(
                    f"{result['case'].key}/"
                    f"{Path(result['target'].config).stem}"),
                summary=result["data"],
                source=str(result["summary"].relative_to(args.out_dir)),
                scenario_metadata={
                    scenario: {
                        "case": result["case"].key,
                        "config": result["target"].config,
                    }
                    for scenario in result["target"].scenarios
                },
            )
            for result in results
        ]
        write_time_breakdown_report(
            time_inputs,
            time_breakdown_csv,
            time_breakdown_markdown,
        )
        write_visualization(
            time_inputs,
            time_breakdown_html,
            title="HBFSim synthetic experiment timing",
        )

        artifacts: dict[str, dict[str, object]] = {}
        for case in plan:
            for name, path in case_artifacts[case.key].items():
                relative = str(path.relative_to(args.out_dir))
                artifacts[relative] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "role": name,
                }
        for result in results:
            for role in ("summary", "log"):
                path = result[role]
                relative = str(path.relative_to(args.out_dir))
                artifacts[relative] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "role": role,
                }
        artifacts[csv_path.name] = {
            "bytes": csv_path.stat().st_size,
            "sha256": sha256_file(csv_path),
            "role": "comparison_csv",
        }
        for path, role in (
            (time_breakdown_csv, "time_breakdown_csv"),
            (time_breakdown_markdown, "time_breakdown_markdown"),
            (time_breakdown_html, "time_breakdown_html"),
        ):
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
            }
        manifest = {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "validation": (
                validation_certificate.summary_block()
                if validation_certificate is not None
                else EXPLORATORY_VALIDATION
            ),
            "profile": args.profile,
            "groups": list(groups),
            "window": args.window,
            "seed": args.seed,
            "flat_boundary_bytes": FLAT_BOUNDARY,
            "logical_heatmap_capacity_semantics": "per-case",
            "tools": {
                f"{name}_sha256": snapshot["sha256"]
                for name, snapshot in tool_snapshots.items()
            },
            "config_files": {
                name: snapshot for name, snapshot in config_snapshots.items()
            },
            "cases": [
                {
                    **{key: value for key, value in asdict(case).items()
                       if key != "targets"},
                    "targets": [asdict(target) for target in case.targets],
                }
                for case in plan
            ],
            "artifacts": artifacts,
        }
        _write_atomic(
            manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"completed {len(plan)} traces, {len(results)} simulator runs, "
          f"{len(rows)} scenario rows")
    print(f"results: {csv_path}")
    print(f"time breakdown: {time_breakdown_markdown}")
    print(f"time visualization: {time_breakdown_html}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
