# Physical Model Status

This document records the implemented physical boundary. It describes what the
code simulates, not what future HBF hardware is guaranteed to do. Parameter
evidence and scientific claim limits are defined in
[`model-validation.md`](model-validation.md).

## Event and State Contract

HBFSim is a deterministic event-level simulator. A request carries an arrival
time, byte range, operation, and optional semantic kind. Devices reject invalid
or out-of-capacity ranges, non-finite configuration values, and nominal arrivals
that move backward.

Resource reservations can extend into the future, but architectural state is
not exposed early. HBF mapping installation, old-page invalidation, page
validity, read-buffer fills, and write-buffer release happen at
timestamped commit events. Later accesses to the same logical page wait for
earlier pending writes when those writes determine the visible value. End-of-run
drain includes buffered data, mapping persistence, GC work, and other scheduled
background commits.

All byte and time helpers reject overflow. Utilization is normalized by the
capacity of the relevant parallel resources; plane-media busy time is computed
as interval union so overlapping work cannot create utilization above one.

## HBM

The HBM model contains:

```text
request byte range
  -> derive tCK, DQ bandwidth, burst bytes/time from pin rate + width + BL
  -> split at derived burst boundaries; each burst maps to one PC/bank/row
  -> channel / pseudo-channel / bank-group / bank / row decode
  -> bounded FR-FCFS pending queue
  -> ACT / PRE / RD / WR timing gates
  -> refresh-free command and data-burst window
  -> pseudo-channel data bus
  -> one aggregated parent completion
```

Implemented timing concepts include `tRCDRD`, `tRCDWR`, `tCL`, `tCWL`,
derived `tBL`, `tRP`, `tRAS`, `tRC`, `tWR`, `tRTP`, cycle-based
`tCCD_S/L`, `tRRD_S/L`, `tFAW`,
`tWTR_S/L`, `tRTW`, `tREFI`, `tRFC`, `tRFCsb`, and `tRREFD`.

Important semantics:

- Per-channel bandwidth is `pin_rate_Gbps * channel_width_bits / 8`.
  Channel width and row bytes divide evenly across pseudo-channels. A
  pseudo-channel burst moves
  `(channel_width_bits / pseudo_channels / 8) * burst_length` bytes in
  `burst_length / pin_rate_Gbps` ns. Command tCK is
  `data_rate_per_command_clock / pin_rate_Gbps`; tCCD is specified in those
  command clocks. There is no independent aggregate-bandwidth or tBL input.
- Every command issue is rounded up to the derived command-clock grid. The
  remaining core/turnaround values are absolute-ns minima; their effective
  delays therefore round up causally rather than drifting off clock edges.
- The fixed address map interleaves at derived burst granularity in
  stack/channel/pseudo-channel order, then rotates bank groups before banks,
  columns, and rows. Reversible high-bit pseudo-channel and bank-selection
  swizzles avoid common power-of-two stride aliases. This is a versioned,
  optimistic controller mapping policy (`burst-pch-bank-swizzle-v1`), not a
  JEDEC topology fact; every summary records it. It keeps sequential traffic
  from being artificially trapped in one tCCD_L chain while preserving exact
  encode/decode round trips.
- An unaligned parent request accounts for every physical burst it touches and
  may route children to multiple pseudo-channels. Parent start is the earliest
  child start; parent finish is the latest child finish. The latency breakdown
  follows the critical child rather than summing parallel paths.
- Column timing and turnaround gates use the one derived burst duration.
  The obsolete `max(configured tBL, bytes/configured bandwidth)` and
  burst-extension reconciliation path no longer exists.
- A refresh cannot bisect an ACT/column/data interval. All-bank refresh closes
  open rows, including refreshes processed during a long idle gap; a request
  rechecks activation after such a closure.
- Address encode/decode and enqueue reject ranges outside `capacity_bytes`.
- FR-FCFS may bypass the oldest request only with a row hit that has actually
  arrived by the oldest request's eligible issue point. Time and
  queue-depth-derived bypass-count caps prevent unbounded starvation;
  `hbm_max_queue_occupancy` exposes the finite admission bound.

Current limitations:

- public product pages do not provide a complete HBM4 speed-bin timing table;
  the normalized 6.4 Gb/s/pin use-case point, 48 GiB capacity, and 14 ns-class
  core timing minima remain explicitly graded assumptions even though the
  width/BL/tCCD relations are source-anchored extrapolations and internally
  self-consistent;
- same-bank refresh is conservatively modeled as a pseudo-channel-wide stall;
  rotating bank choice and target-bank row closure are not represented;
- refresh follows a fixed cadence and does not implement JEDEC pull-in or
  postpone credits;
- no power-down, self-refresh, thermal, power, or signal-integrity model;
- no differential command-trace validation against an independent DRAM
  simulator has yet been integrated;
- the current hashed address map can reduce stride-induced conflicts relative
  to a simple linear controller map; mapping-policy sensitivity remains future
  work and absolute results must identify the recorded mapping scheme.

## HBF

HBF is modeled as flash-as-memory rather than an SSD/NVMe block device. The core
path has no PCIe transport, namespace, host submission queue, or SSD completion
queue.

```text
top-level user request
  -> external HBIO request command
  -> per-stack logic-die admission and pipelined resident-L2P lookup

external media read
  -> internal command: TSV -> channel -> die sequencer
  -> subarray sense -> raw media lane -> raw page-buffer bank
  -> raw channel -> raw TSV -> pipelined ECC decode
  -> full decoded page in logic-die SRAM
  -> exact requested payload bytes over external HBIO

external media write
  -> exact requested payload fragment over external HBIO -> SRAM
  -> old-page internal read or erased-value SRAM fill when a merge is needed
  -> internal program command: TSV -> channel
  -> full decoded page from SRAM -> pipelined ECC encode
  -> raw TSV -> raw channel -> die sequencer -> array program/verify

internal mapping / GC / RMW / buffer-destage transaction
  -> starts or ends in SRAM and uses the same internal command/raw-data path
  -> no additional external HBIO request or payload transfer
```

Read-buffer hits and unmapped erased-value reads are decoded-SRAM paths: they
return only the requested payload bytes over HBIO and do not fabricate a
flash-side raw transfer or ECC operation. All top-level user operations still
pay their one external HBIO request-command transfer.

Implemented mechanisms include:

- configurable SLC page/block/die/channel/stack geometry and OOB transfer bytes;
- complete stack-partitioned LPN-to-PPN mapping resident in controller DRAM,
  with independently configured response latency and issue interval; foreground
  translation performs no mapping-page media read, while dirty persistent
  checkpoints are batched and programmed at drain;
- page-granular data striping across HBF stacks with deterministic
  mapping-group lane rotation; mapping pages remain stack-local over each
  stack's consecutive local entries, and the static-direct path reuses the
  same data placement while bypassing FTL metadata work;
- per-stack write buffering, dirty-range coalescing, partial-page merge,
  read-your-write behavior, honest slot lifetime, and backpressure;
- PPN-keyed read buffer whose entry is usable only after its fill event;
- source queues for user, mapping, GC, and prepopulation transactions;
- page validity, invalidation, free-page allocation, block erase count, GC
  victim selection, relocation, reserved-free-block policy, foreground data
  program counts, and canonical WAF
  (`physical_write_bytes / logical_write_bytes`).
  Data-page relocation updates the resident L2P entry, marks the affected
  mapping VPN dirty, and persists it through the ordinary checkpoint path.
  GC headroom distinguishes same-role Data/Mapping active pages, whole blocks
  available beyond the foreground reserve, and relocation-only capacity in
  reserved/active-GC blocks. Opening a foreground block is charged at block
  granularity, and every reclaimed victim must increase relocation capacity by
  at least its invalid-page count. Data, Mapping, and GC use independent plane
  cursors; the soft projection previews the exact preferred/round-robin plane,
  so maintenance cannot move a block-opening charge to a later request;
- explicit `RawPhysical` block/page ownership for direct physical programs;
  those blocks are unavailable to FTL allocation and GC, while raw programs
  into controller-owned data/mapping/static blocks fail closed;
- per-stack HB IO/TSV timelines, per-die shared decode/encode ECC issue ports,
  per-channel links,
  per-plane media, per-subarray sense rounds, internal media lanes, and
  page-buffer banks;
- exact future-reservation calendars for serial resources and exact
  per-subarray availability for batch sense rounds; every idle interval or
  joinable round that is still reachable after the causal arrival watermark is
  retained rather than truncated by a fixed history limit;
- exact plane calendars spanning sense rounds, subarrays, media lanes, and
  page-buffer banks. Reads may backfill before a later-ready full-plane
  program/erase, but atomic path preview prevents any read from straddling its
  non-preemptible plane window;
- Chrome trace spans and a request breakdown for queueing, translation,
  commands, transfers, array work, ECC issue/latency, SRAM, mapping, and
  maintenance. ECC response latency is not treated as exclusive resource
  occupancy; raw codeword bandwidth determines its initiation interval.
- source-aware payload routing: only host request/response bytes use external
  HBIO. Mapping, GC, RMW, and buffered destage stay on the internal SRAM/ECC/
  TSV/channel path. New partial pages explicitly fill untouched bytes with the
  NAND erased value.
- stack-filtered commit/state advancement and block epochs. A physical erase
  retires the old block incarnation before later callbacks can publish it,
  purges decoded read-buffer state at commit, and gates only accesses to the
  affected block. Program completion also invalidates the exact PPN's cached
  erased value even though programming does not change the block epoch;
  independent planes in the same stack and unrelated stacks neither wait nor
  fast-forward.

ECC latency and ECC throughput are independent controls. For one raw codeword,
`II = (page + OOB) / raw ECC bandwidth`; the shared per-die decode/encode issue
port is occupied for that II. Completion remains `issue start + configured
latency`, so multiple latency windows may overlap. Flash-side transfer
bandwidths are raw page+OOB rates, while logic SRAM and HBIO are decoded-payload
rates.

Programs and erases are currently non-preemptible. An earlier suspend/resume
path was removed because it changed a plane's future frontier without a safe way
to revise completions already returned to callers. Suspend should return only
with a revisable event primitive and a defensible latency/behavior source.

Batch activation models synchronous sense rounds: up to one page per subarray
may share a round, while the next round waits for the current `tR`. The
independent-subarray mode is retained as an optimistic sensitivity bound. It is
not treated as an equally validated hardware mode.

GC may discover that invalidations are pending but not yet committed. A
hard-pressure allocation may wait for the earliest relevant commit and retry;
preventive soft-watermark cleaning returns without manufacturing a victim from
future state. Subsequent requests are causally gated only when hard GC actually
materializes that state transition. Active-GC leftovers remain valid relocation
headroom, so a cleaning loop converges on net reclaimed pages instead of moving
the same live set until an iteration guard is exhausted. Hard-pressure telemetry
is recorded only when the request actually waits for a commit or executes GC;
an unsatisfied threshold with no work does not inject a synthetic 1 ns stall.

Current limitations:

- no retention, read/program disturb, RBER, probabilistic ECC failure, bad-block
  growth, P/E wear-out, thermal coupling, power, or energy;
- SLC only and one physical page per program operation;
- exact program/verify/erase timings and much of the internal topology are
  exploratory parameters, not published HBF specifications;
- wear leveling is a victim-score policy, not a calibrated lifetime model;
- exact ECC maximum-in-flight telemetry retains one response-latency interval
  per codeword and sorts interval endpoints during stats refresh. It uses O(N)
  memory and O(N log N) refresh work on a trace with N ECC operations;
- no independent flash/FTL simulator differential suite or HBF silicon
  validation dataset.

## External Backing

The no-HBF backing tier is deliberately separate from HBF. CXL memory and
NVMe SSD use one `ExternalBackingDevice` implementation with different
resolved parameters, so a profile switch cannot silently change the
layer-streaming placement or controller policy.

Pages are striped by address across media channels. Each channel has a finite
serialized media calendar; configured media bandwidth is aggregate across
channels. A global outstanding credit covers the complete request lifetime.
Both reads and writes traverse requester-to-device wire transfer, one-way
propagation, controller issue/processing, media latency/serialization,
device-to-requester wire transfer, and return propagation. M2S and S2M are
independent full-duplex resources. Every serial resource uses earliest-gap
reservation, preventing API call order from creating false head-of-line
blocking.

The model reports outstanding/controller/media/link queue work, fixed latency
work, serialization busy work, resource-local spans/utilization, request and
payload counts, directional command/completion protocol bytes, exact wire
conservation, and `external_physical` heatmap traffic. Fixed processing,
media-response, and propagation latency does not consume a serial resource.

The built-in CXL-memory and NVMe-SSD profiles are exploratory envelopes.
They do not model CPU caches, NUMA coherence, CXL.cache traffic, OS paging, a
complete PCIe/NVMe protocol implementation, SSD FTL/GC, wear, tail latency,
failure consistency, or a specific commercial product. Parameter sweeps can
expose sensitivity to the stages that do exist, but omitted mechanisms require
dedicated models before product-level claims.

## Composition Layer

Six policies are available:

1. `all-HBM`
2. `all-HBF`
3. `HBM-HBF-Flat`
4. `HBF-static-direct-read`
5. `HBM+HBF-layer-streaming`
6. `HBM+External-layer-streaming`

Direct policies share one engine. Tier boundaries must be HBF-page aligned.
Requests are split at HBF page/placement boundaries so one parent cannot be
charged wholly to the first page's medium or stack. Read and write fragments
retain one user-level completion, and per-stack D2D byte accounting is split the
same way.

Static HBF placement uses a capacity-checked, block-local reversible
permutation. Deterministic fabric and wordline rotations mix high address bits
without modulo wrapping or scattering a source block across unrelated physical
blocks. Static pages reserve their whole physical NAND blocks, and those blocks
are removed from mutable FTL allocation.
`HBF-static-direct-read` is EC5: eligible reads use this physical path without
FTL/resident-mapping/checkpoint work, while writes stay in HBM.

When no explicit initial-image manifest is supplied, a page is inferred to
exist initially only when a read observes at least one byte not covered by an
earlier trace write. A full-page write-first page is therefore never
prepopulated using knowledge of a future read, while a partial write followed
by a read of untouched bytes correctly requires the old image. A partial update
of a statically placed semantic page likewise performs copy-on-write from that
initial page. The rule is shared by direct and layer-streaming policies. It remains a
trace-derived approximation; a separate image manifest is planned.

Production EC6 consumes a digest-bound Frontier object-map contract. The full
runtime-overhead and block-table allocations are resident even if a selected
window touches only a subset. The plan's lowest physical KV block IDs are hot;
the remaining KV blocks and all immutable weights use the selected backing
tier. The trace controls traffic, not allocation or hotness. A trace-derived
first-touch fallback exists only for generic microbenchmarks without an object
map. The HBF and external scenarios consume the same explicit contract.

A trace must label all requests with nondecreasing `layer=N` metadata or label
none; the unlabeled form is one streaming window. Independently,
`phase=N` expresses a complete-before-next request dependency and must likewise
label all requests or none. On HBF backing, immutable weights use a complete
block-fenced, pre-resolved, stack-striped physical extent. Mutable cold KV uses
logical HBF from its initial version onward. The complete cold-KV population,
including pages untouched by the selected window, consumes real Data and
Mapping pages before simulated time. A compact directory avoids one host
hash-table entry per cold page without creating virtual free space: overwrite,
mapping checkpoint, and GC lazily materialize the affected state, and a
differential oracle requires the same timing, relocation, write accounting,
and final FTL state as ordinary per-page prepopulation. Initial-image pages are
pre-existing state and therefore do not enter workload logical/physical write
bytes or WAF. Prefix state and later writebacks consequently share one
authoritative FTL path rather than falling back to stale static media. Both HBF
paths then perform an owning-stack D2D read. On external backing, a page read includes external
media and host-link service. Either route ends with a full-page HBM buffer
write. Fixed and hot-KV residents bypass this DMA, and every foreground
fragment accesses HBM.

When layer N starts execution, layer N+1 begins prefetch into the other buffer.
Execution of a layer waits for its complete input image and the previous
layer's completion. Reusing a parity buffer also waits for the earlier user's
dirty writeback. Full-page writes skip unnecessary input reads; partial writes
obtain the old page before modification. Dirty pages follow an HBM full-page
read, D2D write, and logical HBF write at layer completion. An optional
per-layer `compute_ns=N` interval overlaps next-layer prefetch and gates layer
completion; absent compute remains explicitly unmodeled.

`layer_buffer_bytes` is the upper bound per buffer. Production EC6 reserves
the plan's page-rounded active-buffer size and verifies that every
trace-visible layer fits while enforcing `fixed + hot KV + 2*effective buffer
+ unused == HBM capacity`. The authoritative pressure metric uses the
byte-exact exported unique population divided by physical HBM capacity.
Per-object page allocation and its rounding are separately reported. An
oversized layer, fixed-resident
exhaustion, mixed metadata, decreasing identifiers, or conflicting compute
declarations fail closed. Summary schema v16 reports the `layer_streaming`
mode, selected backing and hybrid-residency policy, unique-footprint pressure,
fixed/hot-KV/backing/unused page census, buffer size, layer/page/byte
conservation, explicit compute, foreground HBM
accesses, exposed/hidden prefetch, user wait, and buffer-reuse wait. A
layer-streaming composition object is single-use to prevent cross-run state
leakage.

`HBM+External-layer-streaming` is the no-HBF control. It retains all EC6
residency, prefetch, compute-overlap, dirty-page, and buffer-reuse decisions,
but replaces static/logical HBF plus per-stack D2D traffic with external-media
and host-link traffic. Foreground requests still execute only from HBM. This
paired construction makes the scenario suitable for isolating the benefit of
HBF placement and its base-die fabric from a conventional offload path.

Cooperative HBM write staging owns a disjoint upper HBM region made of fixed
HBF-page-sized slots. Each resident logical page owns at most one slot, so
overlapping and adjacent sub-page writes merge into byte-exact dirty ranges
instead of consuming request-sized aliases. Unaligned and cross-page requests
are split at page boundaries; slot occupancy is charged by resident pages while
foreground HBM traffic and destage payload are charged only for bytes actually
written. Reads of dirty ranges use the slot plus their page offset, while clean
ranges retain their coherent backing placement. A residency generation protects
placement commits and releases from stale destage events, and requests larger
than the region stream through finite slots with explicit backpressure.

For FLAT, clean data already has logical-HBF backing, so HBF can merge the
coalesced dirty payload through its normal partial-page RMW path. Destage
accounts for HBM read, per-stack D2D transfer, and logical HBF acceptance. User
completion and final makespan are reported separately so deferred work cannot
disappear from throughput accounting.

The cooperative metrics intentionally keep `hbm_write_buffer_user_write_bytes`
(all accepted host bytes, including overwrites) separate from
`hbm_write_buffer_destaged_bytes` (the dirty payload after in-HBM coalescing).
Consequently, HBF's device `waf` always uses bytes accepted by the logical HBF
interface as its denominator. Cooperative host bytes and dirty-union destage
bytes remain raw composition counters; neither is published as an alternative
WAF.

## Metrics and Reproducibility

- `user_completion_throughput_GBps` uses
  `time_breakdown.wall_clock_ns.user_completion_span_ns`, from the trace origin
  to the last user completion.
- `makespan_throughput_GBps` extends through final background drain.
- `time_breakdown.latency_work` uses offered arrival to user completion as its
  primary online latency. It also reports first-credit-to-completion service
  latency and source-ready-to-completion latency as separate full
  distributions. Shifting every trace timestamp by a constant must not change
  any elapsed-span throughput or resource utilization.
- Summary JSON schema v16 records revision/dirty state, build, invocation, trace
  digest, full resolved config, scenario metrics, warnings, and sanity verdict.
  Wall-clock spans are additive; latency and stage work may overlap and must
  never be added to reconstruct elapsed time.

Run these gates after changing the physical model:

```bash
ctest --test-dir build --output-on-failure
cmake --build build --target physical_guard
cmake --build build --target use_cases
cmake --build build --target waf_cases
```

These gates verify implementation invariants. They do not replace calibration
or external validation.
