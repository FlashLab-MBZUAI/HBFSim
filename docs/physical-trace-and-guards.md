# Physical Trace And Guards

`ctest`, `physical_guard`, and `use_cases` form the repository's internal
verification gate. None substitutes for parameter calibration or external
hardware validation.

After touching physical-model code, run the scenario guard:

```bash
cmake --build build --target physical_guard
```

This runs the phase-1 HBM and HBF guard scenarios and writes outputs under:

```text
out/physical-guards/
```

Run the cross-component and write-amplification checks as well after a model change:

```bash
cmake --build build --target use_cases
```

Normal `ctest` includes `use_cases_quick_contracts`, which runs the UC table
and physical probes with `--skip-deep-waf`. The explicit `use_cases` target
remains the release gate and also runs the longer write-amplification suite.

## Guard Scenarios

### HBM

`hbm-interface`

Checks the one-source-of-truth interface derivation: pin rate, DQ width,
pseudo-channel split, BL, command-clock ratio, burst bytes/time, and tCCD
cycles. It also checks burst/PC/bank-group-interleaved address round trips,
including non-power-of-two topology near the uint64 limit; exact and
one-ULP-after-edge alignment; refresh-vs-clock-wait attribution; exact
same/different-bank-group tCCD spacing,
that a sequential stream reaches at least 85% of the derived interface peak,
width/rate/BL metamorphic relations, and fail-closed invalid profiles.

`hbm-boundaries`

Checks derived-burst/address-mapping splitting, parent-completion aggregation,
physical-byte accounting, address round trips, and fail-closed capacity bounds.

`hbm-row`

Checks row-buffer behavior. The expected shape is:

```text
first-row-miss   -> ACT + RD + data burst
same-row-hit     -> no ACT/PRE, waits on tCCD_L if needed
new-row-conflict -> waits tRAS, then PRE + ACT + RD
```

`hbm-channels`

Checks channel-level independence. Four channels should finish clearly faster
than one channel for the same stream, and the output should show
`active_pch=4/4` with `bus_parallelism` greater than one.

`hbm-refresh`

Checks refresh pressure. With refresh enabled, finish time and `refresh_stall_ns` must increase. All-bank refresh can also close open rows, turning later accesses into row misses instead of row conflicts.

`hbm-turnaround`

Checks read/write direction switching. The second request is still a row hit, but it must wait on:

```text
read -> write: tRTW
write -> read: tWTR_L or tWTR_S, depending on bank group
```

The burst duration used by these gates is derived from width, pin rate, and BL;
there is no separately configurable tBL or burst-extension repair path.

### HBF

`hbf-rounding`

Checks that sub-page writes still consume real SLC page programs plus mapping
metadata. The 128 B case has one 2048 B data program plus one 2048 B mapping
checkpoint program, so the canonical WAF is `(2048+2048)/128 = 32`. A zero
logical-write denominator must render as `n/a`/`null`, not zero.

`hbf-resident-mapping`

Checks that the complete per-stack L2P table is resident in controller DRAM,
that every logical read performs one pipelined DRAM lookup, and that no
mapping-page flash read appears on the foreground path.

`hbf-gc`

Checks that free-space pressure triggers real relocation and block erase, not a
single synthetic amplification multiplier. It also separates foreground
Data/Mapping admission from GC-only relocation capacity: appends to an active
role block cannot consume a fictitious whole block, while unused pages in an
active GC block must advance the soft-watermark loop. Deterministic stress
cases lock the no-pressure result at zero GC and the stranded-capacity result
at one victim with each live LPN relocated exactly once. A two-plane guard
also requires the request that actually opens a new role block to carry the
watermark decision; a later append on another plane cannot inherit that GC.

`hbf-parallelism`

Checks that independent page programs are spread across planes. The 4-plane run
must show a clear speedup, `active_planes=4/4`, and `media_parallelism` greater
than one. This catches accidental single-plane or fixed-latency HBF behavior.

`hbf-media-lanes`

Checks CBA-like internal read data-path parallelism inside one plane. The
4-lane run keeps channel/die/plane count fixed, gives subarrays and page-buffer
banks enough width, changes only `media_lanes_per_plane`, and must show a clear
read-stream speedup, `active_media_lanes=4/4`, and `read_lane_parallelism`
greater than one. This catches accidental collapse back to one internal
array-to-page-buffer transfer path per plane.

`hbf-page-buffer-banks`

Checks banked page-buffer output inside one plane. The 4-bank run keeps
channel/die/plane count fixed, gives subarrays and media lanes enough width,
changes only `page_buffer_banks_per_plane`, and must show a clear read-stream
speedup, `active_page_buffer_banks=4/4`, and
`page_buffer_bank_parallelism` greater than one. This catches accidental
serialization through one plane-wide page buffer.

`hbf-subarrays`

Checks derived subarray read-sense conflicts inside one plane. Pages 0 and 2
map to the same subarray and serialize; pages 0 and 1 map to different
subarrays and overlap. The guard also requires subarray entities in the trace,
so the Chrome view can show where the array-read wait occurred.

`hbf-exact-calendar`

Creates more than 64 reusable serial-resource gaps and more than 64 joinable
sense rounds, then requires later-ready work to use all of them. This catches
lossy fixed-history schedulers that manufacture queueing while hardware is
idle. It also checks exact plane-window behavior: a read can backfill before a
future program, but its complete sense/lane/page-buffer path cannot straddle a
full-plane program; short programs/erases can also reverse-backfill a common
gap before a future read. Additional cases lock temporal cache behavior,
block-epoch retirement of delayed mapping/cache publications, block-local
erase isolation (including another plane in the same stack), GC ownership
pins, fail-closed static reservation, and `RawPhysical` separation from the
FTL/GC. It also verifies that issue/drain arrivals form one fail-closed causal
barrier.

`hbf-ecc-pipeline`

Separates ECC response latency from issue throughput. It checks that changing
latency moves the first completion without changing II; same-die codewords
start one II apart while latency windows overlap; different dies issue in
parallel; OOB bytes lengthen codewords; a saturated stream reaches the
configured payload ceiling; decode and encode share only the issue port; and
invalid/non-finite configurations fail closed. It also locks the raw
lane/page-buffer/channel/TSV -> ECC -> decoded SRAM/HBIO order, verifies that
mapping/RMW/buffer-destage bypass external HBIO, checks exact-byte partial-write
ingress and erased fill, checks that an unmapped read returns erased-value bytes
through SRAM/HBIO without media or ECC work, and proves rejected or duplicate
physical programs do not mutate later resource timing.

`hbf-tsu`

Checks that read, program, and erase operations pass through the flash
transaction scheduler and die sequencer.

`hbf-coalesce`

Checks write-buffer staging, dirty range merge, read-hit from SRAM, and one
eventual page program.

`hbf-mapping-batch`

Checks dirty mapping-page batching so repeated writes can share one mapping
page program.

## How To Read Chrome Trace

Open:

```text
chrome://tracing
```

Click `Load`, then select a file such as:

```text
out/physical-guards/hbm-row.trace.json
out/physical-guards/hbm-turnaround.trace.json
```

Chrome trace has three important concepts:

`lane`

The left-side row. In HBFSim this is the simulated entity from `TraceSpan.entity`, not an OS thread. Examples:

```text
stack0/ch0/pch0/bg0/bank0/row7/off0
stack0/ch0/pch0
stack0/logic
stack0/ch0/die0/plane0
stack0/ch0/die0/plane0/lane3
```

`bar`

A rectangle on a lane. It is one `TraceSpan`: a simulated interval with `start_ns`, `end_ns`, name, category, entity, and detail. The width is simulated time.

`args`

Click a bar and inspect the args panel. The most important fields are:

```text
request   request id
op        read/write/erase
entity    simulated resource
detail    timing or resource explanation
start_ns  exact simulated start time
end_ns    exact simulated end time
```

The top ruler is Chrome's own time display. For exact values, trust `start_ns` and `end_ns` in args.

## What The Categories Mean

`mapping`

Address mapping or address generation.

`queue`

Waiting. This is not useful work. The `detail` field should say why, for example:

```text
RD waits for bankgroup constraint tCCD_L (RD->RD)
PRE waits for bank constraint tRAS (ACT->PRE)
WR waits for pseudochannel constraint tRTW (RD->WR)
```

`hbm_command`

HBM command timing such as `ACT`, `PRE`, `RD`, `WR`.

`hbm_bus`

Data movement on the pseudochannel data bus.

`refresh`

Refresh interval such as `REFab` or `REFsb`.

`hbio`, `tsv`, `flash_channel`, `flash_array`, `sram`, `ecc_issue`,
`ecc_latency`, `sequencer`

HBF-side physical resources. `sequencer` is the TSU-like flash transaction
issue point on the die. `flash_array` spans may appear on
`.../planeN/laneM` for page reads; program and erase remain plane-level
barriers. `ecc_issue` occupies only the codeword initiation interval derived
from raw per-die bandwidth. `ecc_latency` begins at that issue instant and may
overlap later codewords; it preserves first-access response latency without
creating a false 500 ns serialization ceiling.

For data spans, `media_lane`, `page_buffer`, `flash_channel`, `tsv`, and
`ecc_*` details report raw page+OOB bytes. `sram` and `hbio` details report
decoded payload bytes. Internal mapping/GC/RMW spans have no `hbio` data span.

## Do Not Read Too Much Into Color

Chrome chooses colors automatically from event names/categories. Color is a visual aid, not the source of truth. Use the lane name, bar name, category, and args detail.

## Reading A Request

Start with the request id in the printed table, then find the same request in Chrome trace args.

For `hbm-row`, a healthy conflict request should look like:

```text
wait_PRE [tRAS] -> PRE -> ACT -> RD -> read_burst
```

For `hbm-turnaround`, a healthy read-after-write request should look like:

```text
wait_RD [tWTR_L] -> RD -> read_burst
```

For `hbf-coalesce`, a healthy buffered write/read sequence should expose:

```text
write_buffer_stage -> write_buffer_read_hit -> write_buffer_flush
```

For `hbf-gc`, a healthy pressure case should expose:

```text
gc_read_valid_page -> gc_program_relocated_page -> block_erase
```

If a future code change removes one of these waits, or turns a conflict into a hit without explanation, that is a model bug until proven otherwise.

## Scenario Compare Is A Different Tool

`physical_guard` is the regression guard. `scenario_compare` is for broader
interpretation across:

```text
all-HBM
all-HBF
HBM-HBF-Flat
HBF-static-direct-read
HBM+HBF-layer-streaming
```

Those scenario outputs are generated under `out/` and are not committed. Keep
the command, configuration, and interpretation in documentation instead of
versioning large transient output files.

For the layer-streaming scenario, Chrome traces include layer/page request ids
for the system-level path:

```text
layerN/pageP/hbf-read              HBF backing read
layerN/pageP/d2d-read              D2D transfer into HBM
layerN/pageP/hbm-install           HBM buffer write
request/buffer-pageP               user-visible HBM service
layerN/pageP/writeback-hbm-read    authoritative dirty-page read
layerN/pageP/d2d-write             D2D transfer toward HBF
layerN/pageP/hbf-writeback         logical HBF write
```

These spans sit above the HBM and HBF physical devices. They should be used to
check whether next-layer HBF latency is hidden by current-layer execution, D2D transfer, and HBM buffering
rather than interpreted as internal SSD/NVMe protocol.
