# Real-Capacity HBM Offloading Experiment

This experiment writes saturated mutable state past the physical capacity of
the repository's 4H4F HBM profile, compares HBF with two no-HBF backing tiers,
and then cold-reads the exact prefix offloaded in FIFO order. It also projects
one concrete layer-sized immediate round-trip from HBM offloading through full
restoration and verified availability in HBM.

```text
HBF offloading:      HBM victim read -> per-stack D2D write -> HBF page program
External offloading: HBM victim read
                   -> M2S command+payload -> propagation -> controller
                   -> media write -> S2M completion -> propagation

HBF readback:      HBF page read -> per-stack D2D read -> HBM DMA install
                  -> foreground HBM read
External readback: M2S command -> propagation -> controller -> media read
                 -> S2M payload+completion -> propagation -> HBM DMA install
                 -> foreground HBM read

Layer round-trip: HBM original slots -> selected backing -> original HBM slots
                  -> foreground HBM verification read
```

The workload is append-only KV or other mutable write-back state. Model weights
are normally read-only or preloaded, so calling this a continuous “weights
write” would mix placement and write-back semantics. The separate layer metric
uses a weight-layer-sized payload only as a concrete data-volume reference.

## Workload and Capacity Parameters

The default point now uses the actual capacity declared by
`configs/scenario_compare/usecase-4h4f.cfg`, not a MiB-scale substitute.

| Quantity | Default |
|---|---:|
| Physical HBM | 192 GiB |
| Reserved HBM DMA buffer | 1 MiB |
| Usable HBM KV window | 191.999023 GiB |
| Saturated source writes | 200 GiB |
| Amount beyond physical HBM | 8 GiB |
| FIFO offloading and cold readback | 8.000977 GiB |
| Source pages | 52,428,800 |
| Offloaded/readback pages | 2,097,408 |
| Transfer batch | 256 pages / 1 MiB |
| Target batches | 8,193 |
| Layer round-trip transfer footprint | 3,041 MiB / 3,041 batches |
| Target HBF | 2 TiB |

The offloading volume is 1 MiB larger than the amount beyond physical HBM
because that 1 MiB is explicitly reserved for the readback DMA buffer and is
unavailable to KV residency.

The default layer size is one
[Llama 3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct/blob/main/config.json)
transformer block at one byte per parameter. With hidden size 16,384,
intermediate size 53,248, eight KV heads, and 128 elements per head, its linear
weights plus two RMSNorm vectors contain 3,187,703,808 bytes. The experiment
rounds this to 3,041 MiB so the 1 MiB transfer batch divides it exactly; the
1,015,808-byte padding is reported explicitly. `--layer-bytes` can replace
this footprint but must remain page- and batch-aligned.

## Backing and Replay Parameters

| Component | Capacity / topology | Key latency | Bandwidth / replay |
|---|---|---|---|
| HBM | 192 GiB; 4 stacks × 32 channels × 2 pseudo-channels | 32 B burst | 6,553.6 GB/s aggregate interface ceiling |
| HBF | 2 TiB; 4 × 4 × 4 × 4 = 256 planes | 1 µs page read; 95 + 5 µs program/verify | Modeled per-stack D2D read/write: 2,048/512 GB/s |
| CXL memory | 256 GiB; 8 media channels; 512 global credits | 2/20 ns controller issue/process; 90 ns media; 75 ns one-way propagation | Media read/write 204.8/204.8 GB/s; M2S/S2M 36/36 GB/s |
| NVMe SSD | 4 TiB; 4 media channels; 512 global credits | 20/500 ns controller issue/process; 80/100 µs media; 250 ns one-way propagation | Media read/write 14/7 GB/s; M2S/S2M 16/16 GB/s |
| Timing replay | 512 MiB sampled HBF | 1% convergence threshold | HBM fill: 16/32 MiB; offloading: 32/64 batches |

## D2D Evidence Boundary

[Sandisk's public fact sheet](https://documents.sandisk.com/content/dam/asset-library/en_us/assets/public/sandisk/collateral/company/Sandisk-HBF-Fact-Sheet.pdf)
gives 1.6 TB/s as the first-generation **HBF stack read bandwidth**. It does
not publish a separate HBM↔HBF D2D read bandwidth, D2D write bandwidth, or
fixed D2D latency. Therefore 1.6 TB/s is used as the HBF HBIO read target and
is not relabeled as a D2D measurement.

The 2,048/512 GB/s per-stack D2D read/write point comes from the supplied TCAD
architecture design and is tagged `literature_derived`, not vendor-measured.
The source gives no separate fixed D2D latency, so the profile adds 0 ns; this
means “no sourced extra term,” not a claim of a physically zero-latency link.

At this modeled point, one 4 KiB page contributes 2 ns of D2D read
serialization or 8 ns of D2D write serialization on its owning stack. With
four independent stack links, the 3,041 MiB layer has ideal aggregate D2D-only
lower bounds of 0.389 ms from HBF to HBM and 1.557 ms from HBM to HBF. These
are link serialization bounds, not end-to-end transfer times; NAND
read/program, ECC, scheduling, and HBM service remain in the physical path.
The two bounds sum to only 0.62% of the reported HBF layer E2E, while the
read-direction bound is 6.62% of the restore-only phase. They are not an
additive stage breakdown because the simulator pipelines and overlaps
resources.

## Exact Accounting

Let `P` be page size, `H` physical HBM bytes, `R` reserved DMA bytes, and `W`
total source-write bytes:

```text
HBM resident data bytes = H - R
offload bytes           = W - (H - R)
readback bytes          = offload bytes
HBM write bytes         = W + offload bytes
HBM read bytes          = 2 * offload bytes
```

At the default point:

```text
HBM write bytes = 223,339,347,968
HBM read bytes  =  17,181,966,336
```

HBF and each external backing receive exactly 8,590,983,168 write bytes and
return exactly the same number of read bytes. HBF D2D traffic is identical in
both directions.

The layer round-trip is an independent timing projection over 3,188,719,616
bytes in each direction. It starts at the first HBM victim read, restores every
page to a distinct original HBM slot, and stops after the last foreground HBM
verification read. Thus the full layer is simultaneously HBM-resident at
completion rather than merely streamed through the 1 MiB DMA buffer.

## Why Timing Uses Converged Periodic Replay

Real capacity must not be confused with blindly materializing every simulator
event. A literal 200 GiB replay at 4 KiB produces 52,428,800 source pages. HBM
then splits this 4H4F profile at 32 B burst granularity, producing exactly
7,516,291,072 HBM burst children across source writes, eviction, DMA install,
and foreground readback. Enumerating those repeated steady-state events would
measure simulator runtime and memory consumption rather than the architecture.

The scalable method therefore separates exact quantities from extrapolated
ones:

- Capacity, page counts, bytes, requests, D2D traffic, mapping checkpoints,
  and phase ordering are exact at the 192/200 GiB target.
- The simulator executes the existing page-exact composition for 32 batches,
  then repeats with 64 batches.
- HBM fill is independently replayed over 16 MiB and 32 MiB resident windows.
  Its ns/byte rate must pass the same convergence check.
- The doubled window covers 64 MiB of real 4 KiB backing traffic and uses the
  same HBM/HBF topology, timings, schedulers, FTL, D2D links, and external
  devices.
- Per-batch offloading and readback timing must drift by at most 1% when the
  window doubles. The experiment fails closed otherwise.
- Only after convergence passes is doubled-window ns/batch multiplied by the
  exact 8,193 target batches.
- Each backing also receives a separate original-slot layer replay. Its
  32-batch and 64-batch samples keep the offloaded set no larger than the
  resident window, restore each page to its original HBM slot, and verify it
  with a foreground read. Its offloading and restore ns/batch rates must each
  pass the same 1% convergence gate before multiplication by the 3,041 layer
  batches.

The sampled HBF reduces only `blocks_per_plane` from 8,192 to 2, giving 512 MiB
of fresh media. Stack/channel/die/plane parallelism and all timing parameters
remain unchanged. The 64 MiB doubled sample fits without GC. The target writes
only about 0.4% of the 2 TiB HBF, so the full projection is also a fresh-media,
no-GC point.

This is a real-capacity simulation with converged periodic timing projection;
it is deliberately not presented as a page-by-page replay of all 200 GiB.

## Controller Policy

1. Saturated full-page writes fill the usable HBM resident window.
2. Each later write selects the oldest dirty page by FIFO.
3. Pages within a batch overlap on independent HBM, link, channel, die, and
   plane resources.
4. A victim slot is reused only after its backing copy is visible. HBF waits
   for payload page programming, not merely an SRAM write-buffer ack.
5. The next batch starts after the previous batch completes.
6. Readback fetches the exact prefix offloaded in FIFO order through the
   reserved HBM buffer and then performs a foreground HBM read.
7. The independent layer replay instead restricts the offloaded set to the
   resident-window size, restores every page to its original distinct HBM
   slot, and completes only after the last foreground verification read.

Every page records two different latency contracts:

- `offered`: workload-visible bulk demand arrival to completion;
- `service`: admission of that page's bounded transfer batch to completion.

Offered latency includes controller-window waiting; service latency does not.
The artifact never publishes an unqualified `offload_latency` or
`read_latency`, because the former implementation mixed these two origins.

HBF's L2P is resident, so data becomes readable before the dirty mapping
checkpoint is persisted. Mapping drain is reported separately; this is not a
power-loss or crash-consistency comparison.

Layer E2E is an immediate isolated data-movement metric: readback begins as
soon as offloading completes, and the original destination slots are assumed
reclaimable. It excludes an arbitrary residency/compute gap and does not add
the cost of offloading a different layer that might occupy those slots.

## Reproduce

```bash
cmake -S . -B build
cmake --build build --target overflow_offload_experiment
./build/overflow_offload_experiment
python3 tools/analyze_overflow_offload_experiment.py \
  --input out/capacity-overflow/summary.json \
  --markdown out/capacity-overflow/report.md \
  --experiment build/overflow_offload_experiment
```

The capacity point can be changed explicitly:

```bash
./build/overflow_offload_experiment \
  --hbm-capacity-bytes 206158430208 \
  --total-write-bytes 214748364800 \
  --read-buffer-bytes 1048576 \
  --layer-bytes 3188719616 \
  --batch-pages 256 \
  --sample-batches 32
```

The tool rejects a workload that does not exceed physical HBM, misaligned
capacities, a batch that does not fit both HBM regions, a batch that does not
exactly divide overflow, a target smaller than either doubled replay window,
insufficient sampled/target backing capacity, or a non-convergent doubled
replay.

The authoritative artifact schema is version 7. It contains exact capacity/
traffic, digest-bound raw base/doubled sample summaries, projected target
timing, measured service curves, derived layer lifecycle, generator executable
SHA-256, and an explicit validation block. Obsolete `spill_*`, ambiguous
latency fields, and older artifact schemas are intentionally not retained.

A direct binary run always says:

```json
{"status": "exploratory_unattached", "certificate": null}
```

For a paper artifact, use the certificate-aware runner:

```bash
python3 tools/run_paper_capacity_overflow.py \
  --experiment build/overflow_offload_experiment \
  --scenario-compare build/scenario_compare \
  --validation-certificate \
    build/foundational-validation-certificate.json \
  --output-dir out/paper-capacity-overflow
```

It rejects a dirty/stale source tree, a mismatched validation input, or any
overflow binary whose SHA-256 differs from certificate schema v6.

## Default Result

| Backing | Offloading ms | Offloading GB/s | Readback ms | Readback GB/s | Drain ms |
|---|---:|---:|---:|---:|---:|
| HBF | 828.937 | 10.364 | 15.797 | 543.824 | 1.707 |
| CXL memory | 246.103 | 34.908 | 244.612 | 35.121 | 0 |
| NVMe SSD | 2,063.746 | 4.163 | 1,287.463 | 6.673 | 0 |

At this point:

- CXL memory offloading is `3.37x` faster than HBF because HBF is
  program-latency limited.
- HBF offloading is `2.49x` faster than the SSD.
- HBF cold readback is `15.49x` faster than CXL memory and `81.50x` faster than
  the SSD.

For the 3,041 MiB layer-sized immediate round-trip:

| Backing | Layer offloading ms | Restore-to-HBM ms | E2E ms | Bidirectional effective GB/s |
|---|---:|---:|---:|---:|
| HBF | 307.676 | 5.876 | 313.551 | 20.339 |
| CXL memory | 91.345 | 90.794 | 182.138 | 35.014 |
| NVMe SSD | 766.000 | 477.869 | 1,243.869 | 5.127 |

CXL memory is `1.72x` faster than HBF for this write-then-read round-trip because
HBF's 100 µs NAND program/verify dominates the offloading half. HBF is `3.96x`
faster than the SSD because its 5.876 ms read-side restoration outweighs its
slower program side. This should not be read as an inference-time weight-write
pattern: Sandisk's published Llama comparison reads pretrained weights, while
this round-trip deliberately exercises both directions for a symmetric and
easy-to-understand data-movement comparison.

The exact 1,024-page service curve is:

| Backing | Window pages | Offload GB/s | Readback GB/s | Highest reported utilization |
|---|---:|---:|---:|---:|
| HBF | 1 / 4 / 16 / 64 / 256 | 0.041 / 0.163 / 0.651 / 2.601 / 10.364 | 2.872 / 11.476 / 42.132 / 153.290 / 538.465 | HBF media 0.4% → 79.1% |
| CXL memory | 1 / 4 / 16 / 64 / 256 | 7.222 / 17.882 / 28.455 / 33.397 / 34.910 | 7.146 / 17.879 / 28.589 / 33.582 / 35.121 | M2S 10.3% → 57.2% |
| NVMe SSD | 1 / 4 / 16 / 64 / 256 | 0.040 / 0.157 / 0.588 / 1.879 / 4.163 | 0.050 / 0.197 / 0.755 / 2.599 / 6.673 | media 0.5% → 58.6% |

This table is more informative than the old 256-page point alone. It shows
where concurrency hides fixed latency, where the curve bends, and which named
resource approaches saturation. The artifact additionally retains p50/p95/p99
offered and service latency, every queue-wait work counter, and M2S/S2M
payload-to-wire efficiency for every point.

Layer reuse changes the conclusion because one program-heavy offload can be
amortized across multiple fast HBF restores:

| Restores N | HBF total ms | CXL memory total ms | NVMe SSD total ms | Winner |
|---:|---:|---:|---:|---|
| 1 | 313.551 | 182.138 | 1,243.869 | CXL memory |
| 2 | 319.427 | 272.932 | 1,721.739 | CXL memory |
| 4 | 331.178 | 454.519 | 2,677.477 | HBF |
| 8 | 354.681 | 817.693 | 4,588.953 | HBF |
| 16 | 401.687 | 1,544.042 | 8,411.907 | HBF |
| 32 | 495.698 | 2,996.739 | 16,057.813 | HBF |

For this declared profile, the first discrete HBF≤CXL-memory point is `N=4`;
HBF≤NVMe-SSD already holds at `N=1`. These are derived from converged samples,
not independent simulations or hardware break-even claims.

The convergence audit is:

| Backing | Offloading/batch 32 | Offloading/batch 64 | Drift | Read/batch 32 | Read/batch 64 | Drift |
|---|---:|---:|---:|---:|---:|---:|
| HBF | 101.170 µs | 101.176 µs | 0.0062% | 1.929 µs | 1.928 µs | 0.0663% |
| CXL memory | 30.036 µs | 30.038 µs | 0.0066% | 29.856 µs | 29.856 µs | 0% |
| NVMe SSD | 251.889 µs | 251.891 µs | 0.0008% | 157.142 µs | 157.142 µs | 0% |

The original-slot layer replays independently converge as follows:

| Backing | Layer offloading/batch 32 | Layer offloading/batch 64 | Drift | Restore/batch 32 | Restore/batch 64 | Drift |
|---|---:|---:|---:|---:|---:|---:|
| HBF | 101.176 µs | 101.176 µs | 0.0005% | 1.932 µs | 1.932 µs | 0.0035% |
| CXL memory | 30.038 µs | 30.038 µs | 0.0016% | 29.856 µs | 29.856 µs | 0% |
| NVMe SSD | 251.891 µs | 251.891 µs | 0.0002% | 157.142 µs | 157.142 µs | 0% |

The 16-to-32 MiB HBM fill replay changes from 0.000168011 to
0.000168439 ns/byte, a 0.2543% drift. All fill, offloading, and readback checks
are below the 1% threshold.

## Independent Physical Bounds

- HBM fills the 191.999 GiB resident window in 34.725 ms, reaching 90.6% of
  its 6,553.6 GB/s derived interface ceiling.
- HBF's 256-plane program ceiling is
  `256 * 4096 B / 100 us = 10.486 GB/s`; projected offloading is
  10.364 GB/s.
- HBF's array-read ceiling is
  `256 * 4096 B / 1 us = 1048.576 GB/s`; full readback reaches 543.824 GB/s.
- CXL memory remains below the 36 GB/s directional M2S/S2M wire ceilings.
  Because those links serialize command/completion protocol bytes as well as
  payload, the corresponding payload ceilings are strictly below 36 GB/s;
  measured offload/readback payload rates are 34.908/35.121 GB/s.
- SSD remains below its 7/14 GB/s write/read media ceilings.
- The HBF target dirties 4,100 mapping pages. Total payload programs are
  2,101,508 pages, payload-byte WAF is `1.001955x`, and 17 checkpoint rounds
  project to a 1.707 ms drain tail.

No measured or projected route exceeds its independently derived resource
ceiling.

## Verification Boundary

The composition and artifact tests fail closed on:

- FIFO victim-order violations, including multiple wraps of the resident
  window;
- HBF early-ack slot reuse;
- unequal page granularity or invalid HBM/read-buffer geometry;
- page, byte, access, link, FTL, or mapping-checkpoint mismatch;
- non-causal fill, offloading, readback, or drain timing;
- failed 16-to-32 MiB HBM-fill or 32-to-64-batch convergence;
- a layer that is not batch-aligned, does not fit the offloaded set/HBM
  resident window, or cannot contain the doubled replay;
- a layer restore that aliases the DMA buffer instead of returning every page
  to a distinct original HBM slot;
- non-convergent layer offloading or restore timing, or an E2E total that does
  not equal its two causal phases;
- missing or ambiguous offered/service latency distributions;
- a missing 1/4/16/64/256-page exact service-curve point, invalid
  resource-local utilization, queue-work mismatch, or directional
  payload/protocol/wire mismatch;
- a missing 1/2/4/8/16/32-restore lifecycle point or a winner/break-even that
  cannot be recomputed from the converged layer sample;
- raw-sample digest drift, executable self-digest drift, or an uncertified
  artifact presented as a paper result;
- projected throughput above HBM, HBF, external-media, or host-link ceilings;
- malformed or corrupted JSON artifacts.

Passing these checks establishes internal consistency with the declared model.
It does not calibrate HBF, CXL memory, or SSD parameters against product
hardware. The SSD profile has no FTL/GC, wear, tail-latency distribution,
failure consistency, thermal behavior, or endurance model, so it cannot
establish steady-state or product-level storage behavior.
