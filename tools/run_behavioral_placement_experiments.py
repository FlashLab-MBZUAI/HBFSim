#!/usr/bin/env python3
"""Run qualified behavior-only HBM/HBF placement experiments.

This runner deliberately separates three questions that are otherwise easy to
conflate:

1. Is the trace large and shaped correctly for the mechanism it claims to test?
2. Does a serialized production run exactly match an independent placement
   oracle and the case's declared mechanism signature?
3. Under a shared stress window, what performance and resource behavior does
   the model predict?

Only the third layer is exploratory. A throughput value cannot make a malformed
workload or an incorrect placement state machine pass the first two layers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, NoReturn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyze_trace_locality import analyze_trace  # noqa: E402
from assess_workload_quality import QualityContract, assess  # noqa: E402
from validation.behavioral_oracle import (  # noqa: E402
    Access as OracleAccess,
    simulate,
)
from validation.certificate import (  # noqa: E402
    EXPLORATORY_VALIDATION,
    file_digest,
    verify_certificate,
)
from validation.run_behavioral_differential import (  # noqa: E402
    COMPARE_FIELDS,
)


SCHEMA = {
    "name": "hbfsim.behavioral-placement-experiments",
    "version": 1,
}
RESULTS_SCHEMA = {
    "name": "hbfsim.behavioral-placement-results",
    "version": 1,
}
PAGE_SIZE = 4096
DEMAND = "HBM+HBF-demand-fill"
BEHAVIORAL = "HBM+HBF-behavioral-placement"
ALL_HBM = "all-HBM"
ALL_HBF = "all-HBF"
FLAT = "HBM-HBF-Flat"
STRESS_SCENARIOS = (ALL_HBM, ALL_HBF, FLAT, DEMAND, BEHAVIORAL)
QUALIFICATION_SCENARIOS = (DEMAND, BEHAVIORAL)
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Access:
    page: int
    op: str = "R"


@dataclass(frozen=True)
class WorkloadCase:
    key: str
    role: str
    description: str
    accesses: tuple[Access, ...]
    quality_contract: QualityContract
    signature: str


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def _permutation(size: int, seed: int) -> list[int]:
    order = list(range(size))
    state = seed & MASK64
    for upper in range(size - 1, 0, -1):
        state = _splitmix64(state)
        other = state % (upper + 1)
        order[upper], order[other] = order[other], order[upper]
    return order


def _cyclic(
    pages: int,
    passes: int,
    *,
    write_passes: set[int] | None = None,
) -> tuple[Access, ...]:
    write_passes = write_passes or set()
    return tuple(
        Access(page, "W" if cycle in write_passes else "R")
        for cycle in range(passes)
        for page in range(pages)
    )


def _hot_cold_pollution(
    *,
    hot_pages: int,
    cycles: int,
    cold_per_hot: int,
) -> tuple[Access, ...]:
    accesses: list[Access] = []
    cold = hot_pages
    for _ in range(cycles):
        for hot in range(hot_pages):
            accesses.append(Access(hot))
            for _ in range(cold_per_hot):
                accesses.append(Access(cold))
                cold += 1
    return tuple(accesses)


def _phase_shift(
    *,
    pages_per_phase: int,
    passes_per_phase: int,
) -> tuple[Access, ...]:
    accesses: list[Access] = []
    for phase in range(2):
        base = phase * pages_per_phase
        for _ in range(passes_per_phase):
            accesses.extend(
                Access(base + page)
                for page in range(pages_per_phase)
            )
    return tuple(accesses)


def _skewed(
    *,
    pages: int,
    hot_pages: int,
    operations: int,
) -> tuple[Access, ...]:
    if operations < pages or hot_pages >= pages:
        raise ValueError("invalid skewed workload dimensions")
    accesses = [Access(page) for page in range(pages)]
    hot_cursor = 0
    cold_cursor = hot_pages
    while len(accesses) < operations:
        ordinal = len(accesses) - pages
        if ordinal % 5 == 4:
            accesses.append(Access(cold_cursor))
            cold_cursor += 1
            if cold_cursor == pages:
                cold_cursor = hot_pages
        else:
            accesses.append(Access(hot_cursor))
            hot_cursor = (hot_cursor + 5) % hot_pages
    return tuple(accesses)


def build_cases(profile: str, hbm_pages: int) -> tuple[WorkloadCase, ...]:
    if profile not in {"smoke", "core"}:
        raise ValueError(f"unsupported profile: {profile}")
    if hbm_pages < 8:
        raise ValueError("the experiment suite requires at least eight HBM pages")

    core = profile == "core"
    scan_pages = 1024 if core else 128
    hot_passes = 64 if core else 12
    edge_passes = 32 if core else 6
    over_pages = hbm_pages + hbm_pages // 2
    over_passes = 32 if core else 6
    pollution_cycles = 128 if core else 24
    phase_passes = 32 if core else 8
    skew_ops = 8192 if core else 768
    dirty_fit_passes = 32 if core else 8
    dirty_pressure_passes = 16 if core else 6

    sequential_scan = tuple(Access(page) for page in range(scan_pages))
    random_scan = tuple(
        Access(page) for page in _permutation(scan_pages, 0x5EED)
    )
    hot_pages = hbm_pages // 2
    phase_pages = hbm_pages // 2
    skew_pages = hbm_pages * 4
    skew_hot = max(4, hbm_pages // 2)
    dirty_fit = _cyclic(
        hot_pages,
        dirty_fit_passes,
        write_passes={1, max(2, dirty_fit_passes // 2)},
    )
    dirty_pressure = tuple(
        Access(page, "W" if cycle % 3 == 1 else "R")
        for cycle in range(dirty_pressure_passes)
        for page in range(over_pages)
    )

    def contract(
        accesses: tuple[Access, ...],
        *,
        unique_pages: int,
        reuse: str,
        relation: str,
        min_reads: int = 0,
        min_writes: int = 0,
        min_fit: float | None = None,
    ) -> QualityContract:
        return QualityContract(
            min_operations=len(accesses),
            min_unique_pages=unique_pages,
            min_reads=min_reads,
            min_writes=min_writes,
            min_page_touches_per_unique_page=(
                len(accesses) / unique_pages
            ),
            max_address_span_to_occupied_page_ratio=1.0,
            reuse_expectation=reuse,
            min_reuse_page_touches=(
                max(1, len(accesses) - unique_pages)
                if reuse == "present" else 0
            ),
            working_set_relation=relation,
            min_reuse_within_hbm_ratio=min_fit,
        )

    hot_fit = _cyclic(hot_pages, hot_passes)
    capacity_edge = _cyclic(hbm_pages, edge_passes)
    cyclic_over = _cyclic(over_pages, over_passes)
    pollution = _hot_cold_pollution(
        hot_pages=8,
        cycles=pollution_cycles,
        cold_per_hot=4,
    )
    pollution_unique = 8 + pollution_cycles * 8 * 4
    phase_shift = _phase_shift(
        pages_per_phase=phase_pages,
        passes_per_phase=phase_passes,
    )
    skewed = _skewed(
        pages=skew_pages,
        hot_pages=skew_hot,
        operations=skew_ops,
    )

    return (
        WorkloadCase(
            "sequential_scan",
            "negative-control",
            "One pass over packed pages; there is no temporal reuse to cache.",
            sequential_scan,
            contract(
                sequential_scan,
                unique_pages=scan_pages,
                reuse="none",
                relation="exceeds-hbm",
                min_reads=scan_pages,
            ),
            "reuse filtering must bypass every page and issue no migration",
        ),
        WorkloadCase(
            "random_scan",
            "negative-control",
            "The same no-reuse working set in a seeded random order.",
            random_scan,
            contract(
                random_scan,
                unique_pages=scan_pages,
                reuse="none",
                relation="exceeds-hbm",
                min_reads=scan_pages,
            ),
            "access order may change timing but cannot manufacture reuse",
        ),
        WorkloadCase(
            "hotset_fits",
            "positive-control",
            "A repeated hot set at half of the HBM tier capacity.",
            hot_fit,
            contract(
                hot_fit,
                unique_pages=hot_pages,
                reuse="present",
                relation="fits-hbm",
                min_reads=len(hot_fit),
                min_fit=1.0,
            ),
            "each page bypasses once, promotes once, then remains an HBM hit",
        ),
        WorkloadCase(
            "capacity_edge",
            "boundary-control",
            "A cyclic working set exactly equal to HBM tier capacity.",
            capacity_edge,
            contract(
                capacity_edge,
                unique_pages=hbm_pages,
                reuse="present",
                relation="fits-hbm",
                min_reads=len(capacity_edge),
                min_fit=1.0,
            ),
            "the exact-capacity set must stabilize without replacement",
        ),
        WorkloadCase(
            "cyclic_over_capacity",
            "negative-control",
            "A cyclic set 1.5x HBM capacity with reuse distance above capacity.",
            cyclic_over,
            contract(
                cyclic_over,
                unique_pages=over_pages,
                reuse="present",
                relation="exceeds-hbm",
                min_reads=len(cyclic_over),
            ),
            "reuse exists but LRU cannot retain it; zero HBM-fit reuse is expected",
        ),
        WorkloadCase(
            "hot_cold_pollution",
            "placement-opportunity",
            "Eight reused pages interleaved with four unique cold pages per hot access.",
            pollution,
            contract(
                pollution,
                unique_pages=pollution_unique,
                reuse="present",
                relation="exceeds-hbm",
                min_reads=len(pollution),
            ),
            "cold pages bypass HBM while the hot set remains resident",
        ),
        WorkloadCase(
            "phase_shift",
            "adaptation-control",
            "A fitting hot set is replaced by a disjoint fitting hot set.",
            phase_shift,
            contract(
                phase_shift,
                unique_pages=phase_pages * 2,
                reuse="present",
                relation="fits-hbm",
                min_reads=len(phase_shift),
                min_fit=1.0,
            ),
            "the controller must learn the second phase without future hints",
        ),
        WorkloadCase(
            "skewed_read",
            "distribution-sweep",
            "A 4x-capacity set with deterministic 80/20 hot/cold access skew.",
            skewed,
            contract(
                skewed,
                unique_pages=skew_pages,
                reuse="present",
                relation="exceeds-hbm",
                min_reads=len(skewed),
                min_fit=0.5,
            ),
            "frequent pages should create both HBM hits and HBF bypasses",
        ),
        WorkloadCase(
            "dirty_hotset_fits",
            "write-control",
            "A fitting hot set with explicit write passes and final dirty drain.",
            dirty_fit,
            contract(
                dirty_fit,
                unique_pages=hot_pages,
                reuse="present",
                relation="fits-hbm",
                min_reads=len(dirty_fit) - 2 * hot_pages,
                min_writes=2 * hot_pages,
                min_fit=1.0,
            ),
            "dirty pages stay resident and are written back exactly at drain",
        ),
        WorkloadCase(
            "dirty_over_capacity",
            "write-pressure",
            "A 1.5x-capacity cyclic set that repeatedly dirties displaced pages.",
            dirty_pressure,
            contract(
                dirty_pressure,
                unique_pages=over_pages,
                reuse="present",
                relation="exceeds-hbm",
                min_reads=sum(item.op == "R" for item in dirty_pressure),
                min_writes=sum(item.op == "W" for item in dirty_pressure),
            ),
            "dirty replacement must exercise HBM read, D2D write, and HBF program",
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_trace(path: Path, accesses: Iterable[Access]) -> None:
    path.write_text(
        "".join(
            f"0x{access.page * PAGE_SIZE:x} {access.op} {PAGE_SIZE}\n"
            for access in accesses
        ),
        encoding="utf-8",
    )


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _scenario_command(
    *,
    binary: Path,
    config: Path,
    trace: Path,
    scenarios: tuple[str, ...],
    hbm_pages: int,
    flat_pages: int,
    history_pages: int,
    outstanding: int,
    interarrival_ns: int,
    summary_json: Path,
    summary_csv: Path,
    config_out: Path,
) -> list[str]:
    return [
        str(binary),
        "--config",
        str(config),
        "--trace",
        str(trace),
        "--scenarios",
        ",".join(scenarios),
        "--behavioral-hbm-bytes",
        str(hbm_pages * PAGE_SIZE),
        "--behavioral-history-pages",
        str(history_pages),
        "--behavioral-promotion-threshold",
        "2",
        "--flat-hbm-bytes",
        str(flat_pages * PAGE_SIZE),
        "--max-outstanding-requests",
        str(outstanding),
        "--interarrival-ns",
        str(interarrival_ns),
        "--address-heatmap-bins",
        "128",
        "--trace-mode",
        "off",
        "--summary-json",
        str(summary_json),
        "--summary-csv",
        str(summary_csv),
        "--config-out",
        str(config_out),
    ]


def _execute_scenarios(
    *,
    binary: Path,
    config: Path,
    trace: Path,
    scenarios: tuple[str, ...],
    hbm_pages: int,
    flat_pages: int,
    history_pages: int,
    outstanding: int,
    interarrival_ns: int,
    directory: Path,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    summary_json = directory / "summary.json"
    summary_csv = directory / "summary.csv"
    config_out = directory / "resolved.cfg"
    command = _scenario_command(
        binary=binary,
        config=config,
        trace=trace,
        scenarios=scenarios,
        hbm_pages=hbm_pages,
        flat_pages=flat_pages,
        history_pages=history_pages,
        outstanding=outstanding,
        interarrival_ns=interarrival_ns,
        summary_json=summary_json,
        summary_csv=summary_csv,
        config_out=config_out,
    )
    completed = _run(command, cwd=ROOT)
    (directory / "stdout.log").write_text(
        completed.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(
        completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"scenario_compare failed for {trace.name} at W={outstanding}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}")
    document = json.loads(summary_json.read_text(encoding="utf-8"))
    if document.get("sanity") != "PASS":
        raise RuntimeError(
            f"{summary_json}: scenario_compare sanity did not pass")
    observed = tuple(item["name"] for item in document["scenarios"])
    if observed != scenarios:
        raise RuntimeError(
            f"{summary_json}: expected scenarios {scenarios}, got {observed}")
    return document


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    condition: bool,
    *,
    actual: Any,
    requirement: str,
    explanation: str,
) -> None:
    checks.append({
        "id": check_id,
        "status": "PASS" if condition else "FAIL",
        "actual": actual,
        "requirement": requirement,
        "explanation": explanation,
    })


def _by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {scenario["name"]: scenario for scenario in summary["scenarios"]}


def _qualify_against_oracle(
    case: WorkloadCase,
    summary: dict[str, Any],
    *,
    hbm_pages: int,
    history_pages: int,
) -> list[dict[str, Any]]:
    scenarios = _by_name(summary)
    oracle_accesses = [
        OracleAccess(access.page, access.op)
        for access in case.accesses
    ]
    checks: list[dict[str, Any]] = []
    for scenario_name, policy in (
        (DEMAND, "always-admit"),
        (BEHAVIORAL, "reuse-filtered"),
    ):
        expected = simulate(
            oracle_accesses,
            policy=policy,
            hbm_pages=hbm_pages,
            history_pages=history_pages,
        )
        actual = scenarios[scenario_name]["behavioral_tiering"]
        differences = {
            field: {
                "expected": expected[field],
                "actual": actual.get(field),
            }
            for field in COMPARE_FIELDS
            if actual.get(field) != expected[field]
        }
        _check(
            checks,
            f"oracle/{policy}",
            not differences,
            actual=differences,
            requirement="no field differs from the independent sequential oracle",
            explanation=(
                "W=1 reduces the production event model to the independent "
                "causal placement state machine."
            ),
        )
    return checks


def _stress_checks(
    case: WorkloadCase,
    quality: dict[str, Any],
    summary: dict[str, Any],
    *,
    hbm_pages: int,
) -> list[dict[str, Any]]:
    by_name = _by_name(summary)
    adaptive = by_name[BEHAVIORAL]
    demand = by_name[DEMAND]
    placement = adaptive["behavioral_tiering"]
    demand_placement = demand["behavioral_tiering"]
    checks: list[dict[str, Any]] = []
    ops = len(case.accesses)
    logical_bytes = ops * PAGE_SIZE

    _check(
        checks,
        "workload/quality-contract",
        quality["status"] == "PASS",
        actual=quality["status"],
        requirement="PASS",
        explanation="The workload must prove its declared scale/locality role.",
    )
    common_counts = {
        scenario["name"]: {
            "ops": scenario["ops"],
            "logical_bytes": scenario["logical_bytes"],
        }
        for scenario in summary["scenarios"]
    }
    _check(
        checks,
        "cross-scenario/logical-input",
        all(
            value == {"ops": ops, "logical_bytes": logical_bytes}
            for value in common_counts.values()
        ),
        actual=common_counts,
        requirement=f"every scenario has ops={ops}, bytes={logical_bytes}",
        explanation="All performance rows must consume the same logical trace.",
    )
    actions = (
        placement["hbm_hits"]
        + placement["hbf_bypasses"]
        + placement["promotions"]
    )
    _check(
        checks,
        "behavior/action-conservation",
        actions == placement["page_observations"] == ops,
        actual={
            "actions": actions,
            "observations": placement["page_observations"],
            "ops": ops,
        },
        requirement="hits+bypasses+promotions == observations == operations",
        explanation="Every full-page request must receive exactly one action.",
    )
    _check(
        checks,
        "behavior/no-semantic-input",
        placement["semantic_inputs_consumed"] is False,
        actual=placement["semantic_inputs_consumed"],
        requirement="false",
        explanation="This experiment evaluates behavior-only placement.",
    )
    _check(
        checks,
        "behavior/capacity-and-drain",
        (
            placement["peak_resident_pages"] <= hbm_pages
            and placement["final_resident_pages"] <= hbm_pages
            and placement["final_dirty_pages"] == 0
        ),
        actual={
            "peak": placement["peak_resident_pages"],
            "final": placement["final_resident_pages"],
            "dirty": placement["final_dirty_pages"],
        },
        requirement=f"peak/final <= {hbm_pages} and final_dirty == 0",
        explanation="Placement cannot invent HBM slots or leave dirty data behind.",
    )

    if case.key in {"sequential_scan", "random_scan"}:
        all_hbf = by_name[ALL_HBF]
        _check(
            checks,
            "signature/no-reuse-bypass",
            (
                placement["hbf_bypasses"] == ops
                and placement["promotions"] == 0
                and placement["hbm_hits"] == 0
                and placement["backing_fill_bytes"] == 0
            ),
            actual={
                key: placement[key]
                for key in (
                    "hbf_bypasses",
                    "promotions",
                    "hbm_hits",
                    "backing_fill_bytes",
                )
            },
            requirement="all requests bypass; no promotion, hit, or fill",
            explanation=case.signature,
        )
        _check(
            checks,
            "signature/demand-fill-pollutes",
            demand_placement["promotions"] == ops,
            actual=demand_placement["promotions"],
            requirement=f"{ops} promotions",
            explanation="The demand-fill baseline is the intentional foil.",
        )
        _check(
            checks,
            "cross-scenario/cold-bypass-degeneration",
            (
                adaptive["hbf_stats"] == all_hbf["hbf_stats"]
                and adaptive["time_breakdown"]["wall_clock_ns"]
                == all_hbf["time_breakdown"]["wall_clock_ns"]
                and adaptive["time_breakdown"]["latency_work"]
                == all_hbf["time_breakdown"]["latency_work"]
            ),
            actual={
                "hbf_stats_equal":
                    adaptive["hbf_stats"] == all_hbf["hbf_stats"],
                "wall_clock_equal":
                    adaptive["time_breakdown"]["wall_clock_ns"]
                    == all_hbf["time_breakdown"]["wall_clock_ns"],
                "latency_work_equal":
                    adaptive["time_breakdown"]["latency_work"]
                    == all_hbf["time_breakdown"]["latency_work"],
            },
            requirement=(
                "a pure cold bypass must reduce exactly to the direct "
                "all-HBF execution path"
            ),
            explanation=(
                "This equivalence catches front-end scheduling drift without "
                "using a performance-order heuristic."
            ),
        )
    elif case.key in {"hotset_fits", "capacity_edge", "phase_shift"}:
        unique = quality["measurements"]["unique_occupied_pages"]
        expected_promotions = unique
        expected_bypasses = unique
        expected_hits = ops - 2 * unique
        _check(
            checks,
            "signature/fitting-reuse",
            (
                placement["promotions"] == expected_promotions
                and placement["hbf_bypasses"] == expected_bypasses
                and placement["hbm_hits"] == expected_hits
                and placement["clean_evictions"] == 0
                and placement["dirty_evictions"] == 0
            ),
            actual={
                key: placement[key]
                for key in (
                    "promotions",
                    "hbf_bypasses",
                    "hbm_hits",
                    "clean_evictions",
                    "dirty_evictions",
                )
            },
            requirement=(
                f"promotions={unique}, bypasses={unique}, "
                f"hits={expected_hits}, evictions=0"
            ),
            explanation=case.signature,
        )
    elif case.key == "cyclic_over_capacity":
        fit_ratio = quality["measurements"]["reuse_within_hbm_ratio"]
        _check(
            checks,
            "signature/reuse-does-not-fit",
            fit_ratio == 0.0,
            actual=fit_ratio,
            requirement="0.0",
            explanation=case.signature,
        )
    elif case.key == "hot_cold_pollution":
        _check(
            checks,
            "signature/filter-prevents-pollution",
            (
                placement["backing_fill_bytes"]
                < demand_placement["backing_fill_bytes"]
                and placement["hbm_hits"] > 0
                and placement["hbf_bypasses"] > 0
            ),
            actual={
                "behavioral_hits": placement["hbm_hits"],
                "demand_hits": demand_placement["hbm_hits"],
                "behavioral_fill_bytes": placement["backing_fill_bytes"],
                "demand_fill_bytes": demand_placement["backing_fill_bytes"],
                "behavioral_bypasses": placement["hbf_bypasses"],
            },
            requirement=(
                "adaptive fills < demand fills, with positive HBM hits and "
                "HBF bypasses"
            ),
            explanation=(
                case.signature
                + "; concurrent same-page waiters may let demand-fill retain "
                "the same hit count, so hit dominance is not an invariant"
            ),
        )
    elif case.key == "skewed_read":
        _check(
            checks,
            "signature/skew-uses-both-tiers",
            placement["hbm_hits"] > 0 and placement["hbf_bypasses"] > 0,
            actual={
                "hbm_hits": placement["hbm_hits"],
                "hbf_bypasses": placement["hbf_bypasses"],
            },
            requirement="both values are positive",
            explanation=case.signature,
        )
    elif case.key == "dirty_hotset_fits":
        unique = quality["measurements"]["unique_occupied_pages"]
        _check(
            checks,
            "signature/dirty-drain",
            (
                placement["dirty_evictions"] == 0
                and placement["drain_writeback_pages"] == unique
                and placement["dirty_writeback_pages"] == unique
                and placement["dirty_writeback_bytes"] == unique * PAGE_SIZE
            ),
            actual={
                key: placement[key]
                for key in (
                    "dirty_evictions",
                    "drain_writeback_pages",
                    "dirty_writeback_pages",
                    "dirty_writeback_bytes",
                )
            },
            requirement=f"{unique} pages written back only at drain",
            explanation=case.signature,
        )
    elif case.key == "dirty_over_capacity":
        link_write = adaptive["hybrid_path"]["base_die_link_write_bytes"]
        _check(
            checks,
            "signature/dirty-replacement",
            (
                placement["dirty_evictions"] > 0
                and placement["dirty_writeback_bytes"] > 0
                and link_write == placement["dirty_writeback_bytes"]
                and adaptive["hbf_stats"]["logical_write_bytes"]
                >= placement["dirty_writeback_bytes"]
            ),
            actual={
                "dirty_evictions": placement["dirty_evictions"],
                "dirty_writeback_bytes": placement["dirty_writeback_bytes"],
                "d2d_write_bytes": link_write,
                "hbf_logical_write_bytes":
                    adaptive["hbf_stats"]["logical_write_bytes"],
            },
            requirement="dirty eviction and byte-exact D2D/HBF writeback",
            explanation=case.signature,
        )
    return checks


def _policy_invariance_checks(
    qualification: dict[str, Any],
    stress: dict[str, Any],
) -> list[dict[str, Any]]:
    qualified = _by_name(qualification)
    stressed = _by_name(stress)
    checks: list[dict[str, Any]] = []
    for scenario_name in QUALIFICATION_SCENARIOS:
        expected = qualified[scenario_name]["behavioral_tiering"]
        actual = stressed[scenario_name]["behavioral_tiering"]
        differences = {
            field: {
                "serialized": expected.get(field),
                "stress": actual.get(field),
            }
            for field in COMPARE_FIELDS
            if actual.get(field) != expected.get(field)
        }
        _check(
            checks,
            "policy-order-invariance/"
            + actual["policy"],
            not differences,
            actual=differences,
            requirement=(
                "placement state and traffic are invariant to arrival spacing "
                "and request window"
            ),
            explanation=(
                "The current policy consumes address/op order, not device "
                "completion order. Timing may change latency, never actions."
            ),
        )
    return checks


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _result_rows(
    case: WorkloadCase,
    locality: dict[str, Any],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    by_name = _by_name(summary)
    all_hbf_user = by_name[ALL_HBF]["user_completion_throughput_GBps"]
    all_hbf_make = by_name[ALL_HBF]["makespan_throughput_GBps"]
    demand_user = by_name[DEMAND]["user_completion_throughput_GBps"]
    demand_make = by_name[DEMAND]["makespan_throughput_GBps"]
    rows: list[dict[str, Any]] = []
    for scenario in summary["scenarios"]:
        placement = scenario["behavioral_tiering"]
        hbm_stats = scenario["hbm_stats"] or {}
        hbf_stats = scenario["hbf_stats"] or {}
        resources = scenario["time_breakdown"]["resource_busy"]
        wall = scenario["time_breakdown"]["wall_clock_ns"]
        row: dict[str, Any] = {
            "case": case.key,
            "role": case.role,
            "scenario": scenario["name"],
            "operations": scenario["ops"],
            "logical_bytes": scenario["logical_bytes"],
            "unique_pages":
                locality["overall"]["unique_occupied_pages"],
            "reuse_touch_ratio":
                locality["temporal_page_locality"]["reuse_touch_ratio"],
            "reuse_within_hbm_ratio":
                locality["temporal_page_locality"][
                    "reuse_within_hbm_ratio"
                ],
            "user_throughput_GBps":
                scenario["user_completion_throughput_GBps"],
            "makespan_throughput_GBps":
                scenario["makespan_throughput_GBps"],
            "user_speedup_vs_all_hbf": _safe_ratio(
                scenario["user_completion_throughput_GBps"],
                all_hbf_user,
            ),
            "makespan_speedup_vs_all_hbf": _safe_ratio(
                scenario["makespan_throughput_GBps"],
                all_hbf_make,
            ),
            "user_speedup_vs_demand_fill": _safe_ratio(
                scenario["user_completion_throughput_GBps"],
                demand_user,
            ),
            "makespan_speedup_vs_demand_fill": _safe_ratio(
                scenario["makespan_throughput_GBps"],
                demand_make,
            ),
            "user_completion_span_ns":
                wall["user_completion_span_ns"],
            "makespan_ns": wall["makespan_ns"],
            "drain_tail_ns": wall["drain_tail_ns"],
            "hbm_user_accesses": scenario["hbm_user_accesses"],
            "hbf_user_accesses": scenario["hbf_user_accesses"],
            "hbm_background_accesses":
                scenario["hbm_background_accesses"],
            "hbf_background_accesses":
                scenario["hbf_background_accesses"],
            "hbm_read_bytes": hbm_stats.get("read_bytes", 0),
            "hbm_write_bytes": hbm_stats.get("write_bytes", 0),
            "hbf_logical_read_bytes":
                hbf_stats.get("logical_read_bytes", 0),
            "hbf_logical_write_bytes":
                hbf_stats.get("logical_write_bytes", 0),
            "hbf_physical_read_bytes":
                hbf_stats.get("physical_read_bytes", 0),
            "hbf_physical_write_bytes":
                hbf_stats.get("physical_write_bytes", 0),
            "base_die_read_bytes":
                scenario["hybrid_path"]["base_die_link_read_bytes"],
            "base_die_write_bytes":
                scenario["hybrid_path"]["base_die_link_write_bytes"],
            "hbm_data_bus_utilization":
                resources["hbm_data_bus"]["utilization"],
            "hbf_plane_media_utilization":
                resources["hbf_plane_media"]["utilization"],
            "hbf_channel_data_utilization":
                resources["hbf_channel_data"]["utilization"],
            "base_die_read_utilization":
                resources["base_die_link_read"]["utilization"],
            "base_die_write_utilization":
                resources["base_die_link_write"]["utilization"],
            "dual_tier_user_paths": (
                scenario["hbm_user_accesses"] > 0
                and scenario["hbf_user_accesses"] > 0
            ),
        }
        for field in (
            "hbm_hits",
            "hbf_bypasses",
            "promotions",
            "clean_evictions",
            "dirty_evictions",
            "backing_fill_bytes",
            "hbm_install_bytes",
            "dirty_writeback_bytes",
            "decision_fingerprint",
        ):
            row[field] = None if placement is None else placement[field]
        row["migration_bytes"] = (
            None if placement is None else
            placement["backing_fill_bytes"]
            + placement["hbm_install_bytes"]
            + placement["dirty_writeback_bytes"]
        )
        row["migration_bytes_per_logical_byte"] = (
            None if row["migration_bytes"] is None else
            row["migration_bytes"] / scenario["logical_bytes"]
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(
    path: Path,
    *,
    profile: str,
    hbm_pages: int,
    outstanding: int,
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    overall_status: str,
    validation: dict[str, Any],
) -> None:
    lookup = {
        (row["case"], row["scenario"]): row
        for row in rows
    }
    lines = [
        "# Behavior-only HBM/HBF experiment report",
        "",
        f"- Overall qualification: **{overall_status}**",
        f"- Profile: `{profile}`",
        f"- HBM tier: `{hbm_pages}` pages ({hbm_pages * PAGE_SIZE} bytes)",
        f"- Stress window: `{outstanding}` requests",
        f"- Validation attachment: `{validation['status']}`",
        "",
        "Performance values below are model predictions, not acceptance "
        "criteria. Acceptance is based on workload contracts, independent "
        "state equivalence, conservation, and declared mechanism signatures.",
        "",
        "| workload | role | quality | oracle | mechanism | all-HBM | all-HBF | "
        "flat | demand-fill | behavioral | beh/all-HBF | beh/demand | "
        "HBM hits | HBF bypasses | migration/logical | dual user paths |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|",
    ]
    for case in cases:
        key = case["key"]
        adaptive = lookup[(key, BEHAVIORAL)]
        scenario_values = [
            _format(lookup[(key, name)]["user_throughput_GBps"])
            for name in (ALL_HBM, ALL_HBF, FLAT, DEMAND, BEHAVIORAL)
        ]
        lines.append(
            "| "
            + " | ".join([
                key,
                case["role"],
                case["quality_status"],
                case["oracle_status"],
                case["mechanism_status"],
                *scenario_values,
                _format(adaptive["user_speedup_vs_all_hbf"]),
                _format(adaptive["user_speedup_vs_demand_fill"]),
                _format(adaptive["hbm_hits"]),
                _format(adaptive["hbf_bypasses"]),
                _format(
                    adaptive["migration_bytes_per_logical_byte"]),
                _format(adaptive["dual_tier_user_paths"]),
            ])
            + " |"
        )
    lines.extend([
        "",
        "### Makespan throughput (GB/s)",
        "",
        "This table includes end-of-run drain; it is the required view for "
        "write-heavy cases.",
        "",
        "| workload | all-HBM | all-HBF | flat | demand-fill | behavioral | "
        "beh/all-HBF | beh/demand | drain tail (ns) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in cases:
        key = case["key"]
        adaptive = lookup[(key, BEHAVIORAL)]
        scenario_values = [
            _format(lookup[(key, name)]["makespan_throughput_GBps"])
            for name in (ALL_HBM, ALL_HBF, FLAT, DEMAND, BEHAVIORAL)
        ]
        lines.append(
            "| "
            + " | ".join([
                key,
                *scenario_values,
                _format(adaptive["makespan_speedup_vs_all_hbf"]),
                _format(adaptive["makespan_speedup_vs_demand_fill"]),
                _format(adaptive["drain_tail_ns"]),
            ])
            + " |"
        )
    lines.extend([
        "",
        "Interpretation rule: a surprising throughput value is investigated "
        "only after the row's quality, oracle, and mechanism columns all pass. "
        "A failed column invalidates the experiment rather than inviting a "
        "post-hoc performance story.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-compare",
        type=Path,
        default=ROOT / "build/scenario_compare",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/scenario_compare/server-4k-hbf4x.cfg",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "out/behavioral-placement-experiments",
    )
    parser.add_argument("--profile", choices=("smoke", "core"), default="core")
    parser.add_argument("--hbm-pages", type=int, default=32)
    parser.add_argument("--history-pages", type=int, default=8192)
    parser.add_argument("--max-outstanding-requests", type=int, default=64)
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument("--validation-certificate", type=Path)
    parser.add_argument("--overflow-experiment", type=Path)
    args = parser.parse_args()

    binary = args.scenario_compare.resolve()
    config = args.config.resolve()
    out_dir = args.out_dir.resolve()
    if not binary.is_file():
        parser.error(f"scenario_compare does not exist: {binary}")
    if not config.is_file():
        parser.error(f"config does not exist: {config}")
    if args.hbm_pages < 8:
        parser.error("--hbm-pages must be at least eight")
    if args.history_pages <= 0:
        parser.error("--history-pages must be positive")
    if args.max_outstanding_requests <= 1:
        parser.error("--max-outstanding-requests must exceed one")
    if out_dir.exists():
        parser.error(f"output directory already exists: {out_dir}")

    try:
        all_cases = build_cases(args.profile, args.hbm_pages)
    except ValueError as error:
        parser.error(str(error))
    known = {case.key for case in all_cases}
    selected = set(args.selected_cases or known)
    unknown = selected - known
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")
    cases = tuple(case for case in all_cases if case.key in selected)

    validation: dict[str, Any] = dict(EXPLORATORY_VALIDATION)
    if args.validation_certificate is not None:
        if args.overflow_experiment is None:
            parser.error(
                "--overflow-experiment is required with "
                "--validation-certificate")
        try:
            verified = verify_certificate(
                args.validation_certificate.resolve(),
                repository=ROOT,
                scenario_compare=binary,
                overflow_offload_experiment=
                    args.overflow_experiment.resolve(),
            )
        except Exception as error:
            parser.error(f"invalid validation certificate: {error}")
        validation = verified.summary_block()

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{out_dir.name}.tmp-",
        dir=out_dir.parent,
    ))
    case_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, start=1):
            print(
                f"[{index}/{len(cases)}] qualifying {case.key}",
                flush=True,
            )
            case_dir = staging / "cases" / case.key
            case_dir.mkdir(parents=True)
            trace = case_dir / "workload.trace"
            _write_trace(trace, case.accesses)
            locality = analyze_trace(
                trace,
                page_size=PAGE_SIZE,
                hbm_capacity_bytes=args.hbm_pages * PAGE_SIZE,
            )
            quality = assess(locality, case.quality_contract)
            _write_json(case_dir / "locality.json", locality)
            _write_json(case_dir / "quality.json", quality)
            unique_pages = locality["overall"]["unique_occupied_pages"]
            # A one-sided FLAT run is rejected by scenario_compare because it
            # says nothing about mixed-media placement. Bisect this case's
            # occupied-page count, capped by the common HBM budget, and record
            # the exact case-local boundary in the manifest.
            flat_pages = min(
                args.hbm_pages,
                max(1, unique_pages // 2),
            )

            qualification = _execute_scenarios(
                binary=binary,
                config=config,
                trace=trace,
                scenarios=QUALIFICATION_SCENARIOS,
                hbm_pages=args.hbm_pages,
                flat_pages=flat_pages,
                history_pages=args.history_pages,
                outstanding=1,
                interarrival_ns=1_000_000,
                directory=case_dir / "qualification-w1",
            )
            oracle_checks = _qualify_against_oracle(
                case,
                qualification,
                hbm_pages=args.hbm_pages,
                history_pages=args.history_pages,
            )

            print(
                f"[{index}/{len(cases)}] stressing {case.key}",
                flush=True,
            )
            stress = _execute_scenarios(
                binary=binary,
                config=config,
                trace=trace,
                scenarios=STRESS_SCENARIOS,
                hbm_pages=args.hbm_pages,
                flat_pages=flat_pages,
                history_pages=args.history_pages,
                outstanding=args.max_outstanding_requests,
                interarrival_ns=1,
                directory=case_dir / (
                    f"stress-w{args.max_outstanding_requests}"
                ),
            )
            mechanism_checks = _stress_checks(
                case,
                quality,
                stress,
                hbm_pages=args.hbm_pages,
            )
            mechanism_checks.extend(
                _policy_invariance_checks(qualification, stress)
            )
            all_checks = oracle_checks + mechanism_checks
            oracle_status = (
                "PASS"
                if all(item["status"] == "PASS" for item in oracle_checks)
                else "FAIL"
            )
            mechanism_status = (
                "PASS"
                if all(
                    item["status"] == "PASS"
                    for item in mechanism_checks
                )
                else "FAIL"
            )
            record = {
                "key": case.key,
                "role": case.role,
                "description": case.description,
                "signature": case.signature,
                "operations": len(case.accesses),
                "reads": sum(item.op == "R" for item in case.accesses),
                "writes": sum(item.op == "W" for item in case.accesses),
                "trace": {
                    "path": str(
                        Path("cases") / case.key / "workload.trace"
                    ),
                    "sha256": _sha256(trace),
                },
                "quality_contract": asdict(case.quality_contract),
                "flat_boundary_pages": flat_pages,
                "flat_boundary_bytes": flat_pages * PAGE_SIZE,
                "quality_status": quality["status"],
                "oracle_status": oracle_status,
                "mechanism_status": mechanism_status,
                "checks": all_checks,
                "artifacts": {
                    "locality": str(
                        Path("cases") / case.key / "locality.json"
                    ),
                    "quality": str(
                        Path("cases") / case.key / "quality.json"
                    ),
                    "qualification_summary": str(
                        Path("cases") / case.key
                        / "qualification-w1/summary.json"
                    ),
                    "stress_summary": str(
                        Path("cases") / case.key
                        / f"stress-w{args.max_outstanding_requests}"
                        / "summary.json"
                    ),
                },
            }
            case_records.append(record)
            rows.extend(_result_rows(case, locality, stress))

        overall_status = (
            "PASS"
            if all(
                case["quality_status"] == "PASS"
                and case["oracle_status"] == "PASS"
                and case["mechanism_status"] == "PASS"
                for case in case_records
            )
            else "FAIL"
        )
        _write_csv(staging / "results.csv", rows)
        _write_report(
            staging / "report.md",
            profile=args.profile,
            hbm_pages=args.hbm_pages,
            outstanding=args.max_outstanding_requests,
            cases=case_records,
            rows=rows,
            overall_status=overall_status,
            validation=validation,
        )
        document = {
            "schema": SCHEMA,
            "status": overall_status,
            "purpose": (
                "qualified mechanism exploration; performance values are "
                "model predictions, not workload acceptance criteria"
            ),
            "profile": args.profile,
            "source": {
                "scenario_compare": str(binary),
                "scenario_compare_digest": file_digest(binary),
                "config": str(config),
                "config_digest": file_digest(config),
            },
            "validation": validation,
            "experiment": {
                "page_size_bytes": PAGE_SIZE,
                "hbm_tier_pages": args.hbm_pages,
                "hbm_tier_bytes": args.hbm_pages * PAGE_SIZE,
                "history_pages": args.history_pages,
                "promotion_threshold": 2,
                "qualification_max_outstanding_requests": 1,
                "qualification_interarrival_ns": 1_000_000,
                "stress_max_outstanding_requests":
                    args.max_outstanding_requests,
                "stress_interarrival_ns": 1,
                "qualification_scenarios":
                    list(QUALIFICATION_SCENARIOS),
                "stress_scenarios": list(STRESS_SCENARIOS),
            },
            "cases": case_records,
            "results": {
                "schema": RESULTS_SCHEMA,
                "path": "results.csv",
                "rows": len(rows),
                "sha256": _sha256(staging / "results.csv"),
            },
            "report": {
                "path": "report.md",
                "sha256": _sha256(staging / "report.md"),
            },
        }
        _write_json(staging / "manifest.json", document)
        os.replace(staging, out_dir)
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        _fail(str(error))

    if overall_status != "PASS":
        print(
            f"FAIL behavioral placement experiments: {out_dir}",
            file=sys.stderr,
        )
        return 2
    print(
        f"PASS behavioral placement experiments: "
        f"{len(case_records)} workloads, {len(rows)} result rows; "
        f"report={out_dir / 'report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
