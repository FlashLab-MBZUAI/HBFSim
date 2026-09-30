#!/usr/bin/env python3
"""Example 1 — compare where data lives: HBM only, HBF only, and mixes.

Runs the five core reference scenarios on the default server profile
(4 HBM stacks + 4 HBF stacks) with the synthetic LLM-like smoke trace, then
reads individual numbers back out of the result. This is what
`python3 -m hbfsim quickstart` does, written as a script you can edit.

    python3 examples/01_first_comparison.py

Try next: change SYSTEM to "6hbm-2hbf", or add overlays=["ocp-v070-grade1"].
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a clone without installing

import hbfsim  # noqa: E402


SYSTEM = "server-hbm128-hbf512"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args()

    result = hbfsim.run(
        system=SYSTEM,
        scenarios=hbfsim.CORE_SCENARIOS,  # all-hbm, all-hbf, flat, direct-read, hbf-streaming
        out_dir=args.out,
    )
    print(result.table())
    print(f"\nsanity check: {result.sanity}; files in {result.summary_path.parent}")

    # Every headline metric is a plain number; everything else is in .raw.
    hbm = result.scenario("all-hbm")
    hbf = result.scenario("all-hbf")
    slowdown = hbf["makespan_us"] / hbm["makespan_us"]
    print(f"\nall-hbf takes {slowdown:,.0f}x longer than all-hbm on this trace.")
    stats = hbf.raw["hbf_stats"]
    print(f"all-hbf programmed {stats['page_programs']} NAND pages "
          f"(WAF {stats['waf']:.2f}) and erased {stats['block_erases']} blocks.")
    print("Hybrid placements (flat, direct-read, hbf-streaming) land in between;")
    print("which one wins depends on what the workload reads and writes.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
