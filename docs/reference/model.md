# Physical and host model

> Status: Current
> Last reviewed: 2026-09-29

HBFSim is a causal memory-transaction simulator. Its current reference contract
is OCP HBF v0.7.0 (3 August 2026, 130 pages) and the public organization of
JEDEC HBM4 JESD270-4 (April 2025). This is not electrical, flit-level or complete
protocol conformance. All parameter evidence and unresolved requirements are
recorded in [parameter provenance](../../configs/parameter-provenance.json).

## Configuration ownership

`host::HbfConfig` contains `device` and `host` settings.
`physical::hbf::HbfDeviceConfig` describes grade, physical geometry, NAND, ECC,
internal data paths and the optional thermal model. `HbfHostConfig` describes
mapping, allocation, Host DRAM, write buffering, GC and zone management.
`HbfController` composes host management with the physical transaction pipeline.
`physical::hbf::HbfDevice` owns independent channel transport, program-order
state, completion visibility and the decoded bank caches. It contains no L2P,
allocator, GC or copyback implementation.

`physical::hbm::HbmConfig` separates `device`, `timing` and `controller` settings.
The current default is HBM4. Small test geometries are scaled experiments and
must not be described as shipping devices. HBF Grade does not select an HBM
generation.

## HBF external interface

The authoritative bandwidth choice is Table 2/4 on page 16:

| Grade | Maximum channels / core dies | x64 lane rate | Payload per channel | Maximum payload per stack |
|---|---:|---:|---:|---:|
| 1 | 8 / 8 | 8 GT/s | 48 GB/s | 384 GB/s |
| 2 | 16 / 16 | 16 GT/s | 96 GB/s | 1536 GB/s |
| 3 | 16 / 16 | 32 GT/s | 192 GB/s | 3072 GB/s |

All bandwidths are decimal. `64 × rate / 8 × 0.75` applies AXI efficiency once.
Each channel has independent receive and transmit resources. The physical page
selects the channel; a stack-wide bandwidth pool cannot move a transaction to a
less busy channel. Busy work sums over physical resources, so total byte count
divided by *per-channel* bandwidth is the conservation equation.

Pages 32 and 35 instead state 94 and 188 GB/s per module. This conflicts with
page 16 and remains an explicit uncertainty; no erratum has been obtained.
Internal NAND/ECC/TSV bandwidth does not automatically increase with Grade.

## Read and program transactions

The device page is 4096 bytes. Host reads may have arbitrary byte ranges, but
the controller splits at page boundaries and covers both ends with aligned
64-byte device bursts. OOB bytes cross the internal raw channel and ECC only.
An erased raw page is an error; the host may synthesize an unmapped *logical*
page's erased value in its buffer without issuing a NAND read.

Each NAND bank has one ordered sense resource. Separate banks may overlap.
The raw page buffer, internal lane, channel, TSV, ECC and decoded SRAM retain
separate finite calendars. ECC initiation throughput is independent of response
latency. Program and erase are non-preemptible bank operations.

Each bank retains two decoded pages independently. A third page can evict a
prior page in that bank. Temporal fill/touch/invalidation records prevent a
future fill from appearing early and preserve already-issued responses during
backfilled scheduling. There is no configurable pooled stack cache or batch
sense-round mode. Concurrent reads that miss before a fill completes are not
silently converted into already-completed cache hits.

Physical programs append without skipped pages. Page zero automatically erases
its block, including its first program after an initial erased state. The one
exception is the opt-in snapshot precondition `hbf-initial-free-blocks-erased`
(default off): a freshly installed image declares its never-programmed blocks
already erased, as a controller's pre-erased free pool, so each such block's
first page-zero program skips the erase and its P/E cycle once; reused blocks
and blocks that saw a raw erase still autoerase, and persistent images refuse
unconsumed pre-erased blocks. Lifetime studies keep it off. A full
4096-byte payload crosses the device interface once at actual program issue;
the write response completes after the single NAND program interval. The current
array timings are 4 us for a read and 75 us for the complete program, including
internal verification. Host write buffering
can acknowledge earlier, but its completion is explicitly a host-buffer
acceptance and final drain accounts for destage.
The simulator also retains an explicit raw erase for low-level ordering and
wear diagnostics. It is not an additional OCP AXI transaction claim, and Host
GC and zone reset never use it to replace mandatory page-zero autoerase.

Ordinary host requests assemble complete pages. `ZONE_WRITE_BATCH` additionally
executes timed 64 B-aligned channel fragments through a device accumulator:
one 4 KiB buffer/credit per outstanding page, duplicate rejection (0x2), full
queue rejection (0x4), first-receipt timeout (0x5), and sequential-page checks
(0x6). Full pages share the ordinary ECC/TSV/NAND programming path; each accepted
command completes after program. Timeouts discard coverage without programming
or advancing the host allocation cursor. The batch starts and ends at a
completed IO frontier; it does not retain partial pages across batches.
Page-zero erase begins after complete assembly in this path. The timeout and
credit count are explicit configurable assumptions. The device exposes partial
read coverage/status, but interleaved reads within these write batches,
per-AXI-ID response queues, scratchpad SRAM writes, UCIe CRC/replay and packet
framing are not implemented.

## Host mapping, GC and wear

OCP section 11.4 excludes device GC and copyback. It does not prescribe a
particular host FTL. The simulator retains full-resident and bounded cached
mapping as host research policies. Cached mapping can use page, entry or affine
extent representations; those are alternative host algorithms for the same
physical medium, not compatibility modes.

Mapping metadata, write buffering, mapping scratch and one GC copy page per
active GC stack share a finite controller-storage capacity. Mapping and write
buffer accesses retain their per-stack latency and issue resources. This storage
budget is reserved at the top of the system's physical HBM address range and
subtracted from application capacity. A mapped HBF topology without HBM is
invalid; raw, unbuffered device-only traffic requires no host allocation.
There is no independent CPU DRAM device or configurable global payload port.
GC copying follows HBF read -> HBM write -> HBM read -> HBF program; the copy
slot remains occupied until destination ingress consumes its payload.

Controller-buffer transfers use a **channel-level HBM model**: burst-rounded
traffic is striped through the reserved HBM partitions and shares each HBM
pseudo-channel data calendar with application channel transfers. Interface
width, pin rate, bursts, bandwidth efficiency and active HBM channels determine
transfer service.
Future reservations leave earlier idle intervals usable. Buffer accesses retain
the explicit controller response-latency/issue assumptions; they do not expand
into DRAM row commands, model buffer row locality, or contribute row-hit counts.
This distinction is reported as `hbm-reserved-shared-data-channels`, not a claim
of command-level validation of controller traffic. Application accesses additionally
pay the aggregate access latency and direction-switch delay described below;
controller DMA retains its bus-service abstraction. Buffer bytes are included in physical HBM
totals and identified separately by `controller_buffer_*` / `host_hbm_*`.

Shared HBM calendars reclaim history only at a joint device frontier. The
session advances that frontier after servicing earlier HBM transfers and before
admitting the next ready transaction; reference event loops use their common
event time. Reclamation is lazy per channel, so a long batch does not retain
every expired gap or force every symmetric channel to materialize on each
admission. Speculative HBF completion times never advance this frontier, and
earlier idle intervals remain available until both paths have passed them.

A full page provided by a workload is not implicitly read from HBM again at
HBF ingress. A coalescing buffer is a separate, capacity-charged HBM copy: staging
writes it and flushing reads it once. Partial-page assembly and mapping/GC
payloads pay their own HBM transfers. Tiered application migration accesses
remain explicit and must not be relabeled as these controller copies.

Logical writes are copy-on-write. Mapping publications and invalidations obey
completion dependencies and block epochs. GC cannot reclaim a block containing
pending programs, unpublished mappings or outstanding reads. Reclaim releases
host allocation ownership; it does not erase the device. The next page-zero
program performs and counts the physical erase. Therefore GC runs, zone resets
and P/E cycles are different quantities. Free-page accounting denotes pages
available to the host allocator, including blocks that will autoerase on reuse.

Optional static wear leveling copies live pages through the same host/device
resources. Its destination selection includes the mandatory future page-zero
erase, while physical P/E counters advance only for an actual scheduled erase.
The raw physical path rejects implicit logical images and in-place logical
updates. Immutable static images remain useful as explicit no-mapping research
placements, with reserved physical ownership.

## Zones and persistent images

Host zones are equal-sized and block aligned. A zone swap is channel local,
requires both zones invalid and no outstanding I/O, and moves only the address
permutation held by the base die. Physical P/E histories never move with logical
zone names. The host may explicitly select the coldest-invalid-zone policy
before subsequent writes. RESET only releases host allocation state; it does
not implicitly remap, erase or copy data.

The persistent image format is **v7**. It stores physical geometry, mapping and
allocation state (including the host mapping organization's state), block wear,
ownership and the zone permutation. Restore starts
a new timing origin and distinguishes inherited wear from workload erases. Old
images are rejected; no legacy reader or compatibility switch is retained.
An installed initial image represents explicitly supplied completed data, not
free writes performed inside the measured window.

## HBM4

The reference stack has 32 channels, two pseudochannels per channel, 64 DQ bits
per channel, and 8 Gb/s per pin: 2048 GB/s of raw DQ bandwidth. The default
organization uses 16 bank groups with four banks per pseudochannel, following
primary HBM4 IP documentation. Capacity is a system choice (48 GiB per stack in
the full profile), with separately labeled scaled test profiles.

The sole timing implementation is `channel-aggregate-v2`. An application request
retains exact burst-rounded bytes per pseudochannel under the existing address
map. Consecutive groups of 4 pseudochannels share a finite arbitration queue
and data calendar. A group's duration is its busiest lane's demand: isolated
transfer bandwidth is preserved, while disjoint concurrent accesses inside a
group can serialize. `hbm-service-group-channels=1` is a sensitivity setting
of the same implementation. Default access latencies are 42 ns for reads and 38 ns for
writes; service uses 94% of raw interface bandwidth, with an 8 ns application
read/write direction-switch delay. These values are explicit modeling assumptions.

Transfers round-robin in at most 4096 B per-channel quanta. A lone pending
request can combine consecutive quanta up to a known external-event boundary
or reserved DMA interval. Parent completion stops advancement so dependent HBF
and compute work can become ready. This avoids expanding every 32 B burst while
retaining channel contention, backpressure, capacity and transaction dependencies.
The 4 KiB quantum is retained because larger quanta gave little extra speed
after grouping and substantially distorted short-request queueing. This is an
HBM arbitration approximation, not an approximation of HBF pages or commands.
On paired mixed serving batches the grouped model reproduced the ungrouped
channel model's makespan within a few percent while running several times
faster; the residual is queueing detail inside a service group.

ACT/PRE/RD/WR scheduling, row hits/conflicts, bank timing constraints and periodic
refresh events have been removed. Their average cost is approximated by access
latency and bandwidth efficiency; detailed row/refresh counters are no longer
reported. Bank coordinates remain only as address-map geometry. This model is
suited to HBF-focused system studies, not DRAM controller or row-locality claims.
Public organization evidence does not calibrate these timing parameters; the
aggregate latency and efficiency defaults are declared assumptions, listed with
their sensitivity in the [configuration catalog](../../configs/README.md#hbm).

## Thermal, reliability and evidence limits

The optional lumped RC model and admission pacing represent light throttling.
Its thermal boundaries, energy costs and initial temperature are experiment
assumptions. Full OCP Normal/Light/Severe/Shutdown state transitions, CATTRIP,
maintenance-only service and link shutdown are not implemented. Experiments must
stay within the supported light-throttle envelope; this model does not establish
thermal safety or compliance.

The hardware parameter study additionally couples measured normal/paced native
service rates to an exact, workload-averaged RC response over seconds. Its main
curves start at idle equilibrium and enter light throttling only when the
workload reaches the configured threshold. These are read-only thermal
projections, not long native transaction replays; they do not produce page-tail
or application latency. R, C, static power and access energy remain uncalibrated
assumptions. The 85/80 C settings are study choices, not universal OCP thresholds.
See thermal response, calibration and limits (`docs/studies/hbf-thermal-response-20260916.md`, no longer in the repository).

The standard specifies a 0–105 C junction operating range and 24-hour powered
retention at 85 C. Product-specific MAXPEC and read-disturb limits cannot be
inferred from those statements. Retention failures, host refresh, disturb
counters, spare/bad-block management, MMIO/IEEE1500 and link faults are outside
this transaction model. Persistence tests prove simulator state continuity,
not indefinite physical flash retention.

## Composition and measurements

`SimulationSession` executes generic memory requests and dependencies; ServeLoop
owns models, tokens, layout and placement. HBM/HBF slot replacement, explicit
D2D migration and external backing are system compositions. External storage
keeps its own directional transport, controller, media and cache resources.
The host GC buffer is not an unreported extra HBM slot.

Eight reference policies are available: `all-hbm`, `all-hbf`, `flat`, `direct-read`,
`hbf-streaming`, `external-streaming`, `demand-fill`, and `reuse-filtered`.
Their routing and capacity boundaries are specified in the
[scenario reference](scenarios.md); they share these physical device models.

Report host acknowledgement and drained completion separately, plus logical
bytes, physical reads/programs, mapping traffic, Host DRAM work and real P/E
counts. Summary schema **v19**, full resolved configuration, source and binary
hashes, and input trace hashes bind comparisons to the actual implementation.
A bounded memory trace is not a complete LLM latency or throughput result.
