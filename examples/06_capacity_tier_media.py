#!/usr/bin/env python3
"""Example 6 — HBF versus other capacity tiers behind HBM.

HBF is one answer to "HBM is too small". Others put the overflow in host
DRAM, CXL memory, on-package LPDDR, an NVMe SSD, or a CXL-attached SSD. The
backing overlays in configs/overlays/backing/ describe those devices; the
external-streaming scenario double-buffers each layer from the external
device into HBM exactly as hbf-streaming does from HBF.

    python3 examples/06_capacity_tier_media.py

Read the result for what it is: a small synthetic stream on exploratory
device envelopes (see each overlay's header and configs/parameter-provenance.json).
It shows how to set up the comparison, not which product wins.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a clone without installing

import hbfsim  # noqa: E402


MEDIA = ("on-package-lpddr", "host-dram", "cxl-memory", "nvme-ssd", "cxl-ssd")
METRICS = ("makespan_us", "throughput_GBps", "mean_latency_us", "hbf_accesses", "external_accesses")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--jobs", type=int, default=4, help="points to run in parallel")
    args = parser.parse_args()

    points = hbfsim.sweep(
        vary={"overlay": list(MEDIA)},
        scenarios="hbf-streaming,external-streaming",
        demo_tokens=16, demo_layers=8,
        out_dir=args.out, jobs=args.jobs,
    )
    rows, warnings = [], []
    for point in points:
        if point.error:
            print(f"rejected {dict(point.labels)}: {point.error}")
            continue
        for scenario in point.result.scenarios:
            # hbf-streaming never touches the backing device: report it once.
            if scenario.name == "hbf-streaming" and rows:
                continue
            tier = "HBF" if scenario.name == "hbf-streaming" else point.labels["overlay"]
            rows.append({"tier": tier, "scenario": scenario.name, **scenario.metrics})
            warnings += [f"{tier}: {warning}" for warning in scenario.warnings]
    print(hbfsim.format_table(rows, metrics=METRICS))
    print("\nEvery row streams the same layers into HBM; only the device behind HBM changes.")
    if warnings:
        print("\nThe simulator flags where the synthetic arrival rate exceeds a device's link;")
        print("latency there reflects queueing, not the device alone:")
        for warning in warnings:
            print(f"  {warning.split(';')[0]}")
    return 1 if any(point.error for point in points) else 0


if __name__ == "__main__":
    raise SystemExit(main())
