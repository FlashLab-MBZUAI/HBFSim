#!/usr/bin/env python3
"""Differentially validate behavioral placement against an independent oracle."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.behavioral_oracle import Access, simulate  # noqa: E402


SCHEMA = {
    "name": "hbfsim.validation.behavioral-differential",
    "version": 1,
}
SCENARIOS = {
    "HBM+HBF-demand-fill": "always-admit",
    "HBM+HBF-behavioral-placement": "reuse-filtered",
}
COMPARE_FIELDS = (
    "page_observations",
    "hbm_hits",
    "hbf_bypasses",
    "cold_misses",
    "history_hits",
    "promotions",
    "promotions_with_backing_fill",
    "promotions_without_backing_fill",
    "clean_evictions",
    "dirty_evictions",
    "history_evictions",
    "hbm_foreground_bytes",
    "hbf_bypass_bytes",
    "backing_fill_pages",
    "backing_fill_bytes",
    "hbm_install_pages",
    "hbm_install_bytes",
    "dirty_writeback_pages",
    "dirty_writeback_bytes",
    "drain_writeback_pages",
    "drain_writeback_bytes",
    "peak_resident_pages",
    "final_resident_pages",
    "final_dirty_pages",
    "peak_history_pages",
    "decision_fingerprint",
)


def generate(seed: int, operations: int) -> list[Access]:
    rng = random.Random(seed)
    accesses = [
        Access(0, "R"),
        Access(0, "R"),
        Access(0, "W"),
        Access(1, "R"),
        Access(1, "R"),
        Access(2, "W"),
    ]
    while len(accesses) < operations:
        # A changing hot pair plus a sparse cold tail exercises reuse,
        # displacement, phase shifts, and dirty state without future hints.
        phase = len(accesses) // max(4, operations // 4)
        if rng.random() < 0.72:
            page = (phase * 2 + rng.randrange(2)) % 8
        else:
            page = rng.randrange(8)
        op = "W" if rng.random() < 0.28 else "R"
        accesses.append(Access(page, op))
    return accesses[:operations]


def write_trace(path: Path, accesses: list[Access], seed: int) -> None:
    kinds = (
        "model_weights",
        "shared_context",
        "generated_context",
        "scratch",
        "metadata",
    )
    lines = []
    for index, access in enumerate(accesses):
        lines.append(
            f"0x{access.page * access.bytes:x} {access.op} {access.bytes} "
            f"{kinds[(index + seed) % len(kinds)]} "
            f"label=oracle-{seed}-{index} phase={index // 5} "
            # One millisecond separates requests, intentionally reducing the
            # production event system to the same sequential policy state
            # machine as the independent oracle.
            f"layer={index // 3} at={index * 1_000_000}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--operations", type=int, default=32)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.seeds <= 0 or args.operations < 6:
        parser.error("--seeds must be positive and --operations at least six")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "seeds": args.seeds,
        "operations_per_seed": args.operations,
        "policies": list(SCENARIOS.values()),
        "compared_fields": list(COMPARE_FIELDS),
        "production_executions": args.seeds * len(SCENARIOS),
        "cases": [],
        "status": "FAIL",
    }
    with tempfile.TemporaryDirectory(
            prefix="hbfsim-behavioral-differential-") as directory:
        root = Path(directory)
        for seed in range(args.seeds):
            hbm_pages = 1 + seed % 4
            history_pages = 3 + seed % 7
            trace = root / f"seed-{seed}.trace"
            summary = root / f"seed-{seed}.json"
            accesses = generate(seed, args.operations)
            write_trace(trace, accesses, seed)
            command = (
                str(args.scenario_compare),
                "--config", str(args.config),
                "--trace", str(trace),
                "--scenarios", ",".join(SCENARIOS),
                "--behavioral-hbm-bytes", str(hbm_pages * 4096),
                "--behavioral-history-pages", str(history_pages),
                "--behavioral-promotion-threshold", "2",
                "--max-outstanding-requests", "1",
                "--summary-json", str(summary),
                "--trace-mode", "off",
            )
            completed = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"seed {seed}: scenario_compare failed\n"
                    f"stdout:\n{completed.stdout[-4000:]}\n"
                    f"stderr:\n{completed.stderr[-4000:]}")
            document = json.loads(summary.read_text(encoding="utf-8"))
            if document.get("sanity") != "PASS":
                raise RuntimeError(
                    f"seed {seed}: production summary failed sanity")
            actual = {
                scenario["name"]: scenario["behavioral_tiering"]
                for scenario in document["scenarios"]
            }
            case: dict[str, Any] = {
                "seed": seed,
                "hbm_pages": hbm_pages,
                "history_pages": history_pages,
                "policies": {},
            }
            for scenario, policy in SCENARIOS.items():
                expected = simulate(
                    accesses,
                    policy=policy,
                    hbm_pages=hbm_pages,
                    history_pages=history_pages,
                )
                observed = actual[scenario]
                differences = {
                    field: {
                        "expected": expected[field],
                        "actual": observed[field],
                    }
                    for field in COMPARE_FIELDS
                    if observed.get(field) != expected[field]
                }
                if differences:
                    raise RuntimeError(
                        f"seed {seed} policy {policy}: "
                        f"oracle mismatch {json.dumps(differences, sort_keys=True)}")
                case["policies"][policy] = {
                    "decision_fingerprint": expected[
                        "decision_fingerprint"],
                    "hbm_hits": expected["hbm_hits"],
                    "hbf_bypasses": expected["hbf_bypasses"],
                    "promotions": expected["promotions"],
                    "dirty_writeback_pages": expected[
                        "dirty_writeback_pages"],
                }
            report["cases"].append(case)

    report["status"] = "PASS"
    report["case_count"] = len(report["cases"])
    if args.report is not None:
        atomic_json(args.report.resolve(), report)
    print(
        "PASS behavioral differential: "
        f"{args.seeds} seeds x {len(SCENARIOS)} policies, "
        f"{args.operations} operations/seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
