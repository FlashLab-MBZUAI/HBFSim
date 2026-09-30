#!/usr/bin/env python3
"""Example 5 — prototype your own placement policy in Python.

The engine (`build/hbfsim`) knows nothing about workloads or policies: you
send it a DAG of memory transactions — target device, address, size, issue
time and dependencies — and it returns when each one physically completed.
Every placement, caching, migration or prefetch idea is therefore just code
that decides which transactions to issue.

This example serves a skewed (Zipf-like) read stream whose data lives in HBF
and compares three policies:

  all-hbf           every read goes to HBF
  promote-on-reuse  YOUR POLICY: a small LRU cache in HBM; a page read twice
                    is copied into HBM (an HBF read, then a dependent HBM
                    write) and later reads hit HBM
  all-hbm           every read goes to HBM (the unreachable upper bound)

    python3 examples/05_custom_policy.py
    python3 examples/05_custom_policy.py --cache-pages 64 --threshold 3

Edit `promote_on_reuse` to try your own idea.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import itertools
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a clone without installing

import hbfsim  # noqa: E402
from hbfsim.runner import prepare_output_directory  # noqa: E402
from hbfsim_client import Transaction  # noqa: E402


PAGE = 4096


def zipf_stream(pages: int, requests: int, skew: float, seed: int) -> list[int]:
    """Deterministic page numbers where page rank r is chosen with weight 1/r^skew."""

    weights = [1.0 / (rank ** skew) for rank in range(1, pages + 1)]
    cumulative = list(itertools.accumulate(weights))
    rng = random.Random(seed)
    order = list(range(pages))
    rng.shuffle(order)  # hot pages are scattered, not the lowest addresses
    return [order[index] for index in rng.choices(range(pages), cum_weights=cumulative, k=requests)]


def all_hbf(stream: list[int], gap_ns: float, **_: int) -> list[Transaction]:
    return [Transaction(id=f"q{i}", target="HBF_LOGICAL", op="R",
                        addr=page * PAGE, bytes=PAGE, issue_ns=i * gap_ns)
            for i, page in enumerate(stream)]


def all_hbm(stream: list[int], gap_ns: float, **_: int) -> list[Transaction]:
    return [Transaction(id=f"q{i}", target="HBM", op="R",
                        addr=page * PAGE, bytes=PAGE, issue_ns=i * gap_ns)
            for i, page in enumerate(stream)]


def promote_on_reuse(stream: list[int], gap_ns: float, *, cache_pages: int,
                     threshold: int) -> list[Transaction]:
    """Copy a page into an HBM cache once it has been read `threshold` times."""

    transactions: list[Transaction] = []
    cache: OrderedDict[int, tuple[int, str]] = OrderedDict()  # page -> (slot, fill id), LRU order
    free_slots = list(range(cache_pages))
    last_reader: dict[int, str] = {}  # slot -> last transaction that read it
    reads: dict[int, int] = {}
    for i, page in enumerate(stream):
        now, request = i * gap_ns, f"q{i}"
        if page in cache:  # hit: read HBM once the fill has landed
            slot, fill = cache[page]
            cache.move_to_end(page)
            transactions.append(Transaction(id=request, target="HBM", op="R",
                                            addr=slot * PAGE, bytes=PAGE, issue_ns=now,
                                            dependencies=(fill,)))
            last_reader[slot] = request
            continue
        transactions.append(Transaction(id=request, target="HBF_LOGICAL", op="R",
                                        addr=page * PAGE, bytes=PAGE, issue_ns=now))
        reads[page] = reads.get(page, 0) + 1
        if reads[page] < threshold:
            continue
        if free_slots:
            slot = free_slots.pop()
        else:  # evict the least recently used page; data is read-only, so no write-back
            _, (slot, _) = cache.popitem(last=False)
        # The fill writes the bytes this miss just read, and must not overwrite
        # the slot before its previous occupant's last read has finished.
        after = (request,) + ((last_reader[slot],) if slot in last_reader else ())
        fill = f"fill{i}"
        transactions.append(Transaction(id=fill, target="HBM", op="W", addr=slot * PAGE,
                                        bytes=PAGE, issue_ns=now, dependencies=after))
        cache[page] = (slot, fill)
        last_reader.pop(slot, None)
    return transactions


def measure(name: str, transactions: list[Transaction], args: argparse.Namespace,
            out: Path) -> dict[str, object]:
    with hbfsim.open_session(args.system, out_dir=out / name,
                             initial_hbf_logical_pages=args.pages) as session:  # data starts in HBF
        result = session.run(transactions)
    issue = {t.id: t.issue_ns for t in transactions}
    latency = sorted(record.finish_ns - issue[record.id]
                     for record in result.completions if record.id.startswith("q"))
    targets = [t.target for t in transactions]
    return {
        "policy": name,
        "makespan_us": result.elapsed_ns / 1000,
        "mean_latency_us": statistics.fmean(latency) / 1000,
        "p95_latency_us": latency[int(0.95 * (len(latency) - 1))] / 1000,
        "hbf_reads": targets.count("HBF_LOGICAL"),
        "hbm_fills": sum(t.id.startswith("fill") for t in transactions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--system", default="4hbm-4hbf")
    parser.add_argument("--pages", type=int, default=2048, help="data footprint in 4 KiB pages")
    parser.add_argument("--requests", type=int, default=6000)
    parser.add_argument("--gap-ns", type=float, default=20.0, help="time between requests")
    parser.add_argument("--skew", type=float, default=1.1, help="Zipf exponent")
    parser.add_argument("--cache-pages", type=int, default=128, help="HBM cache size in pages")
    parser.add_argument("--threshold", type=int, default=2, help="reads before promotion")
    args = parser.parse_args()
    out = prepare_output_directory(args.out, "example-policy")

    stream = zipf_stream(args.pages, args.requests, args.skew, seed=7)
    options = {"cache_pages": args.cache_pages, "threshold": args.threshold}
    rows = [measure(name, policy(stream, args.gap_ns, **options), args, out)
            for name, policy in (("all-hbf", all_hbf),
                                 ("promote-on-reuse", promote_on_reuse),
                                 ("all-hbm", all_hbm))]
    print(f"{args.requests:,} reads over {args.pages:,} pages (Zipf {args.skew}); "
          f"HBM cache {args.cache_pages} pages ({args.cache_pages / args.pages:.0%} of data)\n")
    print(hbfsim.format_table(rows, metrics=("makespan_us", "mean_latency_us", "p95_latency_us",
                                             "hbf_reads", "hbm_fills")))
    ours, flash = rows[1], rows[0]
    print(f"\npromote-on-reuse cut mean latency {flash['mean_latency_us'] / ours['mean_latency_us']:.1f}x "
          f"vs all-hbf while caching {args.cache_pages / args.pages:.0%} of the data in HBM.")
    print(f"Session configs and HBF wear maps: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
