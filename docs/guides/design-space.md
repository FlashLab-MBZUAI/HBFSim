# Design-space recipes

> Status: Current
> Last reviewed: 2026-09-29

Each recipe below is a common HBF design question with commands that were run
against this revision. Result tables are examples from the shipped profiles;
the point of a recipe is the method, and your numbers will move with the
workload and parameters. Commands run from the repository root after
`python3 -m hbfsim build`.

## Before you start: match the workload to the question

The built-in smoke trace (`--demo-tokens 4 --demo-layers 2`, 1,920 accesses;
`16`/`8` gives about 55,000) finishes in a second but touches a few megabytes.
It exposes **latency, FTL and write-path** effects well. It does **not**
stress capacity, link bandwidth or thermal limits:

| Question depends on… | Use |
| --- | --- |
| latency, mapping, write buffering, WAF | the smoke trace, possibly larger (`--demo-tokens 16 --demo-layers 8`) |
| sustained bandwidth | a burst through the engine ([example 3](../../examples/03_read_bandwidth_ceiling.py)) or a long trace |
| GC, wear and lifetime | a write-heavy trace on a small HBF (recipe below) |
| capacity and HBM:HBF ratio | real model footprints through [ServeLoop](serveloop.md) |
| thermal throttling | sustained load over seconds of simulated time |

Two habits keep sweeps honest: compare points that replay the **same trace**
(`sweep` does this for you), and read every **warning** — an open-loop arrival
rate above a device's peak turns latency into queueing time. Add
`--set max-outstanding-requests=64` for closed-loop admission when latency is
the claim.

## How far is HBF from HBM, and which placement closes the gap?

```bash
python3 -m hbfsim quickstart
```

On the smoke trace, `all-hbf` is three orders of magnitude slower than
`all-hbm`; `flat`, `direct-read` and `hbf-streaming` land in between depending
on which data they keep in flash. Change the placement boundary with
`--set flat-hbm-bytes=...` or the layer buffer with
`--set layer-buffer-bytes=...` and rerun. [Example 1](../../examples/01_first_comparison.py)
shows how to read individual numbers out of the result.

## Interface speed grade or NAND sensing: what limits read bandwidth?

```bash
python3 examples/03_read_bandwidth_ceiling.py
```

```text
     grade      tR=4us      tR=2us      tR=1us
    grade1         488         914        1300
    grade2         917        1628        2347
    grade3         918        1631        2355
```

Grade 1 → 2 doubles bandwidth, but grade 3 adds nothing at any tR: with 256
banks per stack and 4 KiB pages, array parallelism and internal paths cap each
stack before the 192 GB/s-per-channel interface does. The runner says so
directly — `run --overlay ocp-v070-grade3` warns that the ECC, raw channel and
TSV ceilings are below the interface bandwidth. Try raising
`hbf-ecc-decode-raw-bw`, `hbf-channel-bw` or `hbf-tsv-bw` together with a
shorter `hbf-read-ns` to find the next bottleneck.

## What does the FTL mapping strategy cost?

```bash
python3 -m hbfsim sweep --overlay cached-l2p-1-over-1000 \
  --vary overlay=mapping/page-cache,mapping/entry-cache,mapping/compressed-cache,mapping/block \
  --scenarios all-hbf --demo-tokens 16 --demo-layers 8 --jobs 4 \
  --metrics makespan_us,mean_latency_us,p95_latency_us,hbf_waf,hbf_block_erases
```

```text
overlay                   scenario  makespan (us)  mean lat (us)  p95 lat (us)  HBF WAF  erases
mapping/page-cache        all-hbf           5,441          63.16         594.4    1.792     172
mapping/entry-cache       all-hbf           9,198          24.32         45.81    1.792     172
mapping/compressed-cache  all-hbf           5,700          61.98           492    1.792     172
mapping/block             all-hbf          71,816          521.2         2,947    14.48      13
```

`cached-l2p-1-over-1000` gives the mapping cache an HBM budget of 1/1000 of HBF
capacity; the swept overlay then selects what the cache holds. Entry caching
trades makespan for much better request latency here; block mapping needs a
tiny table but rewrites whole blocks on small updates (8× the WAF). The fully
resident page table (`mapping/resident`) does not fit this budget, and the
sweep reports that point as rejected. Vary the budget with
`--set hbf-ctrl-dram-capacity-denominator=...`. The
[mapping organizations reference](../reference/hbf-mapping-organizations.md)
describes every strategy; [example 2](../../examples/02_ftl_mapping_tradeoffs.py)
is the same study as a script.

## Does NAND program time matter?

```bash
python3 -m hbfsim sweep --vary hbf-program-ns=37500,75000,150000 \
  --scenarios all-hbf --demo-tokens 16 --demo-layers 8 --jobs 3 \
  --metrics makespan_us,mean_latency_us,p95_latency_us
```

```text
hbf-program-ns  scenario  makespan (us)  mean lat (us)  p95 lat (us)
37500           all-hbf           4,849          25.12         81.03
75000           all-hbf           4,924          25.12         81.03
150000          all-hbf           5,074          25.12         81.03
```

Request latency does not change: the host write buffer acknowledges writes and
programs flash in the background, so tPROG only lengthens the drain at the end
(makespan). Program time reaches the critical path once writes outrun the
buffer — shrink `hbf-write-buffer-pages` or use a write-heavy trace to find
that point.

## Wear leveling versus write amplification

GC and wear need writes that fill the device. Generate a hotspot write stream
and replay it on a scaled-down HBF (`endurance-regime-scaled`: 4 MiB raw, so
GC runs within seconds):

```bash
mkdir -p out
python3 workloads/synthetic/generate.py --output out/hot-writes.trace \
  --pattern hotspot --pages 700 --ops 20000 --read-percent 0 \
  --hot-page-percent 10 --hot-access-percent 90 --seed 1
python3 -m hbfsim sweep --system 4hbm-4hbf --policy none \
  --overlay endurance-regime-scaled --trace out/hot-writes.trace \
  --vary hbf-gc-wear-leveling-weight=0,0.5 --scenarios all-hbf --jobs 2 \
  --metrics makespan_us,hbf_waf,hbf_block_erases,hbf_max_block_erases,hbf_gc_relocations
```

```text
hbf-gc-wear-leveling-weight  scenario  makespan (us)  HBF WAF  erases  max P/E  GC copies
0                            all-hbf       1.876e+06    1.005   2,520      118        106
0.5                          all-hbf       2.401e+06    1.137   2,847       44      2,732
```

Without wear leveling, GC keeps recycling the hot blocks: the most-worn block
reaches 118 erases while writes stay nearly unamplified. Weight 0.5 moves cold
data out of fresh blocks, cutting the worst block to 44 erases — about 2.7×
more lifetime — for 13% write amplification and a 28% longer run. Each run
directory's `summary-hbf-wear/all-hbf.html` maps the per-block wear. Lifetime
projection and zone management are covered by the
[host HBF management reference](../reference/host-hbf-management.md).

## Is HBF a better capacity tier than DRAM, CXL or SSDs?

```bash
python3 examples/06_capacity_tier_media.py
```

The backing overlays (`host-dram`, `cxl-memory`, `on-package-lpddr`,
`nvme-ssd`, `cxl-ssd`, plus `cxl-ssd-cached` for a device DRAM cache) swap the
device behind HBM, and `external-streaming` streams layers from it exactly as
`hbf-streaming` does from HBF. Most envelopes are exploratory; the
`backing/calibrated/` overlays carry measured A100/H200 host timings. Heed the
open-loop warnings, or add `--set max-outstanding-requests=64`.

## How much HBM versus how much HBF?

The topology profiles `2hbm-6hbf`, `4hbm-4hbf`, `6hbm-2hbf` and
`eight-stack-baseline` keep eight stack slots and vary the split (the
`miniquick/` variants are capacity-scaled). A ratio question is a capacity
question, so it needs a workload whose footprint approaches HBM: model
weights plus KV cache for a real model and batch. Run it through
[ServeLoop](serveloop.md) with `--system configs/systems/<topology>.cfg`. On the
smoke trace every topology fits in HBM and the results barely differ.

## Thermal limits

Profiles such as `2hbm-6hbf` enable an RC thermal model and throttling
governor, and the `thermal-boundary-55c`/`65c`/`70c` and
`thermal-steady-state-start` overlays set boundary conditions. Thermal time
constants are seconds, so these need sustained workloads; a millisecond smoke
trace never heats up. See the [model reference](../reference/model.md#thermal-reliability-and-evidence-limits).

## Your own idea

A policy that decides what to keep in HBM is a Python function over
transactions: [example 5](../../examples/05_custom_policy.py) prototypes a
promotion cache in about sixty lines. [Extending HBFSim](extending.md)
explains the next steps, and the "Share a study, result, or question" issue
template is the place to publish what you find.
