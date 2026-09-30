#!/usr/bin/env python3
"""Example 2 — what does the flash translation layer's mapping cost?

HBF behaves like memory, but underneath it is NAND flash: the host keeps a
logical-to-physical (L2P) map, and how that map is organized and cached in
HBM changes latency, write amplification (WAF) and wear. This example keeps
the hardware and trace fixed, gives the mapping a budget of 1/1000 of HBF
capacity in HBM, and sweeps the mapping strategy.

    python3 examples/02_ftl_mapping_tradeoffs.py

The same sweep from the command line:

    python3 -m hbfsim sweep --overlay cached-l2p-1-over-1000 \\
        --vary overlay=mapping/page-cache,mapping/entry-cache,mapping/compressed-cache,mapping/block \\
        --scenarios all-hbf --demo-tokens 16 --demo-layers 8

Try next: add "hbf-read-ns" as a second axis, or a bigger trace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a clone without installing

import hbfsim  # noqa: E402


STRATEGIES = {
    "mapping/page-cache": "cache whole translation pages (the default cached L2P)",
    "mapping/entry-cache": "cache individual entries (DFTL-style)",
    "mapping/compressed-cache": "cache compressed translation pages",
    "mapping/block": "map whole blocks (tiny table, rewrites on small writes)",
}
METRICS = ("makespan_us", "mean_latency_us", "p95_latency_us", "hbf_waf", "hbf_block_erases")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--jobs", type=int, default=4, help="points to run in parallel")
    args = parser.parse_args()

    points = hbfsim.sweep(
        system="server-hbm128-hbf512",
        overlays=["cached-l2p-1-over-1000"],   # HBM budget for the mapping cache
        vary={"overlay": list(STRATEGIES)},     # appended after the budget overlay
        scenarios="all-hbf",                    # every access goes through the FTL
        demo_tokens=16, demo_layers=8,          # ~55k accesses: enough to exercise the cache
        out_dir=args.out, jobs=args.jobs,
    )
    failed = [point for point in points if point.error]
    for point in failed:
        print(f"rejected {dict(point.labels)}: {point.error}")
    rows = [row for point in points if point.result for row in point.result.rows()]
    print(hbfsim.format_table(rows, metrics=METRICS))

    print("\nstrategies:")
    for name, description in STRATEGIES.items():
        print(f"  {name:<26} {description}")
    by_name = {point.labels["overlay"]: point.result.scenario("all-hbf")
               for point in points if point.result}
    block, page = by_name["mapping/block"], by_name["mapping/page-cache"]
    print(f"\nBlock mapping writes {block['hbf_waf'] / page['hbf_waf']:.1f}x more flash per "
          "logical byte than page mapping here: small updates rewrite whole blocks.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
