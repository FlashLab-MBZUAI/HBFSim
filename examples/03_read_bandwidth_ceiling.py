#!/usr/bin/env python3
"""Example 3 — what limits HBF read bandwidth: the interface or the NAND?

The OCP HBF speed grade sets each channel's interface rate (grade 1/2/3 =
48/96/192 GB/s). But a page must first be sensed from the NAND array, which
takes tR (hbf-read-ns). With B banks per stack, array bandwidth is roughly
B x page / tR. This example issues a burst of 4 KiB reads at t=0 through the
persistent engine and measures sustained bandwidth for each grade and tR.

    python3 examples/03_read_bandwidth_ceiling.py

It uses `hbfsim.open_session`, the same entry point you would use to write
your own policy (see example 5).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a clone without installing

import hbfsim  # noqa: E402
from hbfsim.runner import prepare_output_directory  # noqa: E402
from hbfsim_client import Transaction  # noqa: E402


SYSTEM = "4hbm-4hbf"
GRADES = ("ocp-v070-grade1", "ocp-v070-grade2", "ocp-v070-grade3")
READ_TIMES_NS = (4000, 2000, 1000)
PAGE = 4096


def burst_bandwidth(grade: str, read_ns: int, pages: int, out_dir: Path) -> float:
    """GB/s for `pages` independent 4 KiB reads issued at the same instant."""

    with hbfsim.open_session(SYSTEM, overlays=[grade], options={"hbf-read-ns": read_ns},
                             out_dir=out_dir,
                             initial_hbf_logical_pages=pages) as session:  # data already written
        reads = [Transaction(id=f"r{page}", target="HBF_LOGICAL", op="R",
                             addr=page * PAGE, bytes=PAGE, issue_ns=0.0)
                 for page in range(pages)]
        result = session.run(reads, completions=False)
    return pages * PAGE / result.elapsed_ns  # bytes per ns == GB/s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--pages", type=int, default=4096, help="reads per burst")
    args = parser.parse_args()
    out = prepare_output_directory(args.out, "example-bandwidth")

    rows = []
    for grade in GRADES:
        row = {"grade": grade.rsplit("-", 1)[-1]}
        for read_ns in READ_TIMES_NS:
            point = out / f"{grade}-tR{read_ns}"
            row[f"tR={read_ns / 1000:g}us"] = burst_bandwidth(grade, read_ns, args.pages, point)
        rows.append(row)

    columns = list(rows[0])
    print(f"HBF read bandwidth (GB/s), {SYSTEM}, {args.pages} x 4 KiB reads at t=0\n")
    print("  ".join(f"{column:>10}" for column in columns))
    for row in rows:
        print("  ".join(f"{row[columns[0]]:>10}" if index == 0 else f"{row[column]:>10.0f}"
                        for index, column in enumerate(columns)))

    grade2, grade3 = rows[1][columns[1]], rows[2][columns[1]]
    print(f"\nAt tR=4us, grade 3 reaches {grade3 / grade2:.2f}x grade 2: the NAND array, not the")
    print("interface, is the ceiling. Faster sensing helps every grade; a faster interface")
    print("only pays off once the array (and internal ECC/TSV paths) can keep up.")
    print(f"Per-run configs and wear maps: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
