# Model Validation and Evidence Policy

HBFSim exists to explore HBM/HBF architecture choices for AI workloads. Its
most important contract is not that every configured number is "real"; public
HBF information is not yet complete enough for that. The contract is that the
simulator makes the boundary between evidence, assumption, and result explicit
and fails closed when a request violates the modeled state or capacity.

## What a Result Means

A HBFSim result is conditional on four things:

1. the event and resource model implemented by the recorded source revision;
2. the fully resolved device and composition configuration;
3. the initial memory image and workload trace;
4. the limitations listed in this document.

It is valid to compare mechanisms under the same declared model. It is not yet
valid to present absolute latency, throughput, endurance, energy, or lifetime
as a prediction of a shipping HBF device.

Every summary JSON records a schema version, simulator revision and tracked
dirty state, compiler/build identity, invocation, working directory, trace
SHA-256, resolved configuration, and scenario results. It also contains an
explicit validation block. A direct or uncertified run is
`exploratory_unattached`; it must not be presented as a paper result merely
because `sanity` passed. Preserve the JSON with any reported figure.

Summary schema v16 records HBM pin rate, DQ widths, BL, command-clock ratio,
derived tCK/burst/bandwidth values, and cycle-derived tCCD alongside the
absolute core-timing minima. It retains the v4 separation of ECC response
latency from codeword initiation throughput and the v3 separation between
cooperative host-write bytes and coalesced dirty-union destage bytes.

V12 retains the canonical `time_breakdown` separation between additive
wall-clock spans, summed per-operation latency work, overlapping device and
controller work, and resource capacity-time. Its primary latency is
offered-to-completion; service and source-to-completion remain separate full
distributions. EC6 records its unique-page backing plan, selected HBF or
external backing, effective layer-buffer size, layer count,
streamed/writeback pages and bytes, foreground HBM accesses, exposed/hidden
prefetch, user wait, and parity-buffer reuse wait.

Every v16 scenario also embeds an `hbfsim.address_heatmap.v1`. Its five domains
separate workload-logical requests, burst-aligned HBM physical traffic,
mutable HBF logical traffic, page/block-granular HBF physical traffic, and
exact external-backing traffic.
Read, write, and erase bytes conserve exactly through their bins and fixed
traffic-source attribution. Access counts have different semantics: a domain
counts a record once, while every bin touched by that record counts the access,
so bin accesses intentionally fan out. Physical-direct HBF requests bypass the
logical FTL surface and therefore leave `hbf_logical` empty. The default 1024
bins per domain can be changed up to 8192; accumulator memory is bounded by
that resolution and the fixed source vocabulary rather than trace length.
When coalescing combines dirty ranges from several sources into one HBF page,
the indivisible physical merge read/program is attributed to the source with
the greatest dirty-byte coverage; equal-byte ties use canonical source order.
V6 and older artifacts must not be mixed with v7 experiment tables.

## Evidence Grades

Parameters use these grades in `configs/parameter-provenance.json`:

| Grade | Meaning | Permitted claim |
|---|---|---|
| `vendor_published` | A primary vendor source directly states the value. | The source reports this product target or property. |
| `literature_derived` | A paper or public design description supports the value or derivation, but it is not a measured HBF product specification. | This experiment follows the cited design interpretation. |
| `exploratory_assumption` | The value closes a missing specification or exposes a sensitivity axis. | This is a model input; conclusions must include a sweep or caveat. |
| `implementation_policy` | The value defines simulator behavior rather than hardware. | This is the policy evaluated by the experiment. |

Two primary public anchors are currently available:

- The [Sandisk HBF fact sheet](https://documents.sandisk.com/content/dam/asset-library/en_us/assets/public/sandisk/collateral/company/Sandisk-HBF-Fact-Sheet.pdf)
  states a first-generation target of 1.6 TB/s read bandwidth, 256 Gb per die,
  16 dies, and 512 GB total capacity.
- KIOXIA's [XL-FLASH product page](https://americas.kioxia.com/en-us/business/memory/xlflash.html)
  states a 4 kB page, SLC operation, a 16-plane architecture, and read latency
  below 5 microseconds. It does not publish the exact program, verify, or erase
  values used by HBFSim.

The normalized HBM comparison profile has a separately auditable boundary.
Cadence and Synopsys public HBM4 IP material support a 2048-bit organization,
32 64-bit channels, two pseudo-channels per channel, BL8, and 64 banks per
pseudo-channel. A published Micron patent application gives a related-art example with a
4:1 DQ-rate/command-clock ratio and tCCD_S/L of 2/4 command clocks. The
use-case profile applies those dimensionless relations at an intentionally
reduced 6.4 Gb/s/pin, yielding 1638.4 GB/s per stack. Current Micron HBM4
material reports more than 11 Gb/s and more than 2.8 TB/s, so the normalized
point is an exploratory control, not a shipping-product claim. Its 48 GiB
capacity and 14 ns-class core timing minima are also explicit assumptions.
The default `burst-pch-bank-swizzle-v1` address map is separately graded as an
implementation policy: it hashes high address bits into pseudo-channel and
bank selection to avoid common stride aliases and is not attributed to JEDEC.

The CXL-memory and NVMe-SSD backing profiles are also exploratory controls.
Both use the same explicit request path: global outstanding admission, M2S
command/payload, propagation, controller issue and processing, media
latency/serialization, S2M payload/completion, and return propagation. Their
capacity, page granularity, media channels, device credits, controller/media
timing, directional bandwidth, propagation, and command/completion sizes are
recorded in `configs/parameter-provenance.json`. CXL 4.0 and NVMe 2.1 support
the interface direction and command/completion structure; public Micron CZ120
and KIOXIA CM7-V data anchor only declared product-envelope values. Internal
channels, fixed latency, controller timing, and other unsourced fields remain
`exploratory_assumption`. These profiles are not calibrated claims about a
specific CPU memory, CXL fabric, PCIe link, or SSD.

The external model does not implement CXL.cache coherence, a complete PCIe/
NVMe protocol stack, SSD FTL/GC, wear, tail-latency distributions, or failure
consistency. Absolute external-backing conclusions require workload-relevant
measurements and sensitivity sweeps; none of those omitted mechanisms may be
treated as implicitly covered by a default multiplier.

The shipped use-case geometry computes 512 GiB per HBF stack, not the vendor's
512 GB decimal figure. That is about 7.4% larger and must be treated as an
exploratory geometry choice. Every EC topology profile uses the same TCAD
manuscript point: a 1 microsecond NAND read and a 100 microsecond cell write
split into 95 microseconds program plus the model's explicit 5 microsecond
verify. The former 4/75 microsecond values remain explicit sensitivities, not
topology-dependent defaults. These are literature assumptions, not measured
HBF product timings; the 2 millisecond erase remains exploratory.

The public HBF material does not disclose ECC latency, decoder/encoder count,
initiation interval, or whether read and write pipelines share hardware.
HBFSim therefore treats decode/encode response latency as an exploratory input
and models throughput separately from raw codeword bandwidth
(data + OOB) per die. The initiation interval is
`II = (page bytes + OOB bytes) / raw bandwidth`; it is the exclusive issue-port
occupancy, whereas response latency is measured from issue start to that
codeword's completion. The configured latency must cover at least one II, but
later codewords may be initiated before an earlier latency window completes.
Every EC topology profile uses 250 ns as a common calibration residual: with
explicit command/transfer work, one uncontended 4 KiB read completes in
1283.77 ns, close to the paper's aggregate 1.29 microsecond page-read value.
It is not an ECC measurement. The shipped
16-die use-case profiles use 105.46875 raw GB/s per die: a 4320 B
codeword starts every 40.96 ns, yielding 100 GB/s of 4 KiB payload per die and
1.6 TB/s per stack. This is a compatibility bound derived from the vendor's
aggregate read target, not a measured ECC parameter. Encode uses equal
provisioning only as a symmetric sensitivity baseline. Decode and encode
conservatively share one issue port per die.

The bandwidth boundary is explicit. Flash media lanes, page-buffer banks,
channels, TSV, and ECC carry the raw page+OOB codeword; the TSV shares that
same per-stack timeline with internal command bytes. Logic-die SRAM and
external HBIO carry decoded payload. A user read sends one request command over
HBIO, performs translation, sends the internal flash command over TSV/channel,
then follows `array -> raw media lane -> raw page buffer -> raw channel -> raw
TSV -> ECC decode -> decoded SRAM -> exact requested payload over HBIO`. A user
write sends one request command and the exact host fragment over HBIO, stages
and merges decoded bytes in SRAM, sends an internal program command over
TSV/channel, then follows `full-page SRAM -> ECC encode -> raw TSV -> raw
channel -> die sequencer -> array program/verify`. Mapping-page traffic, GC
relocation, read-modify-write, and write-buffer destage start from or terminate
in SRAM and do not consume another external HBIO payload transfer. The per-die
ECC topology and its placement at this boundary are implementation assumptions
because public material does not disclose them.

## Verification Is Not Calibration

HBFSim records six levels of confidence rather than collapsing them into one
“reasonable result” judgment:

1. **L0 runtime/regression:** ordinary and ASan+UBSan builds, CTest, and
   fail-closed runtime checks pass.
2. **L1 internal contracts:** conservation laws, exact address round trips,
   causal event ordering, boundary rejection, hand-computed timing budgets,
   physical guards, and use-case checks pass.
3. **L2 independent implementation:** independent tiny HBM, HBF/FTL, Hybrid,
   and external-backing event/state oracles agree with production code;
   deterministic
   property/metamorphic fuzz passes; an independent ledger reducer reproduces
   accounting; and every declared production mutation is killed.
4. **L3 external reference:** actual pinned, unmodified Ramulator2 and MQSim
   builds agree case-by-case on explicitly named facets. The current facets
   are only single-bank DDR4 read row-state/service behavior and direct
   full-page SLC data-I/O work. Adapter fixtures validate translation formats
   only and cannot produce L3.
5. **L4 calibration:** uncertain parameters are fitted to declared public or
   measured device data with held-out observations.
6. **L5 external hardware:** measurements not used for calibration are
   predicted within a declared domain.

The generated foundational certificate is bound to one clean tracked commit,
tree, exact `scenario_compare`, `validation_probe`, `physical_probe`, and
`overflow_offload_experiment` executables, validation implementation, case
corpus, parameter registry, every tracked external-validation input, per-case
ledgers, property corpus, and mutation matrix. With no supplied actual
external evidence bundle it states L3 `not_run`. A complete bundle may mark L3
`pass` only for the two named facets and their exact six-case census. It does
not validate MQSim end-to-end timing, flash GC/erase, DRAM writes/turnaround,
refresh, multi-bank scheduling, HBM topology, or any HBF-specific data path.
L4 and L5 remain `not_run` in either case.

Certificate schema v6 also makes three closed-form production microbenchmarks
mandatory. It checks that HBM data-bus busy work equals transferred bytes over
per-pseudo-channel bandwidth; that a 25-point isolated, serial HBF read grid
follows `max(media read time, HBIO payload serialization time)`; and that a
Direct FLAT run preserves isolated HBM/HBF resource work while mixed makespan
equals the slower tier rather than their sum. These are model-relative
implementation identities. The grid disables batching and buffering, makes
all other HBF stages non-binding, and is neither an application phase diagram
nor L4/L5 evidence.

The same schema records the sanitizer host platform and exact ASan/UBSan
runtime options. Linux certificate runs require LeakSanitizer. Darwin's ASan
runtime does not support leak detection, so that omission is recorded as an
explicit certificate limitation rather than silently reported as equivalent
coverage.

`replay_astra_trace.py`, `run_synthetic_experiments.py`, and
`run_waf_cases.py` accept `--validation-certificate`; without it their
summaries and suite manifests remain explicitly exploratory. Every
`run_paper_*.py` runner requires the flag and rejects a stale commit/tree,
changed tracked source, or different executable. Attaching a certificate does
not freeze an EC number or turn an assumption into calibration. It says only
that the exact implementation used for that result passed the recorded
foundational evidence.

Frontier-derived workloads require a separate qualification gate. A
structurally valid Frontier export is not automatically performance evidence.
There is currently no Frontier publication entry point: the former
single-request 7B phase study was retired because it did not exercise
production batching or capacity pressure. The retained 7B descriptor is CI
smoke only. The pinned 70B request suite now has independently reconstructed
steady/burst/long-context-decode-tail windows. The pinned Frontier v8
integration, schema-v7 replay audit, residency-plan v1, and schema-v2 model
descriptors now bind the exact 70B W8A16/BF16 storage ledgers, hybrid
weight/KV placement, full-request no-preemption admission, and dummy-timing
claim boundary. The canonical per-trace memory runner now executes
all-HBM/HBF/CXL/NVMe and fails closed on traffic, placement, scheduler/credit,
HBM-geometry, or backing-byte drift. This remains a structural prerequisite,
not a publication entry point. The former 72-cell full-replay matrix was never
executed, could not scale at exact physical granularity, and has been deleted.
The one current evidence path runs the byte-identical five-cell burst slice at
the mechanically selected q=0/q=0.5/q=1 Frontier batches. Its verifier
revalidates all 60 baseline rows but preserves them as separate temporal
strata and keeps `paper_result_eligible=false`. Frontier exports and direct
`scenario_compare` results remain exploratory.
Dummy or otherwise uncalibrated Frontier timing may support only modeled
memory-system service; it may never be reported as TTFT, TPOT, SLO attainment,
end-to-end latency, or token throughput.

Passing L0-L2 means the implementation agrees with its declared model through
independent executable checks. It does not prove that the model is a complete
description of future HBF hardware.

## State, Time, and Initial Conditions

- Device requests have nondecreasing nominal arrival times. Resource queues may
  reorder only according to their documented scheduler.
- Mapping changes, page validity, buffered-write visibility, and static
  reservations become visible at modeled completion events, not when a future
  operation is merely scheduled.
- State-dependent callbacks are filtered per stack, while a physical erase
  gates only accesses whose mapping/physical target is the erased block. An
  independent plane in the same stack may continue; neither it nor another
  stack is forced through the erase completion time. Targeted future
  dependencies apply only their program/mapping/erase callback chain and leave
  an object-scoped block/LPN/VPN/PPN watermark. They do not materialize every
  callback on the stack, so independent-plane timing is invariant to API call
  order while the affected object cannot use future state in its past.
- Every erasable block has an epoch. Scheduling a destructive physical erase
  retires the prior epoch immediately; delayed program, mapping, buffered-write,
  or read-buffer publication tagged with the old epoch cannot resurrect the
  erased contents when its callback later fires. Pending LPN and mapping-page
  publications retain an explicit destructive deadline, so filtering their
  retired epoch cannot accidentally discard the dependency on the in-flight
  erase itself.
- A later access to an LPN waits for earlier pending writes that determine its
  value. A staged foreground write supersedes any stale prefetch fill.
- An ordinary read of an unmapped LPN returns NAND's erased-value bytes through
  decoded SRAM and exact-byte HBIO egress. It does not invent an array read,
  ECC decode, mapping entry, or zero-filled backing page.
- Read-buffer replacement uses timestamped fill/access events, not host call
  order. A request may merge with a still-in-flight fill. The L2P table is
  complete and resident, so mapping lookup has no fill, miss, or eviction
  state. Program completion invalidates any
  cached erased-value line for the programmed PPN; block-epoch checks alone
  are insufficient because erased-to-programmed does not advance that epoch.
- A partial write to a previously unmapped page gives untouched bytes NAND's
  erased value and charges the corresponding SRAM fill. This is an explicit
  implementation policy, not an inferred zero-filled initial image. Reads
  before or after destage assemble dirty and erased bytes consistently.
- Plane scheduling uses exact reservation calendars. Reads, programs, and
  erases may backfill a real common idle interval before a future reservation, but
  a read's complete sense-to-media-lane-to-page-buffer path must fit on one side of
  the non-preemptible full-plane window; a read may not straddle that barrier.
  A destructive erase also waits for the raw page-buffer capture or program
  completion deadline of every read/program already issued to its target block,
  even when a long translation or ECC stage leaves an otherwise reusable plane gap.
- Direct physical programming claims a `RawPhysical` block and page owner.
  FTL allocation cannot enter that block, controller-owned data/mapping blocks
  reject raw programming, and raw blocks are excluded from GC victim selection.
  Physical access is therefore not an implicit alternate spelling of an LPN.
- GC uses separate capacity contracts. Foreground Data/Mapping admission counts
  only the matching role's active pages plus whole free blocks beyond the
  per-plane reserve. Relocation counts all whole free blocks plus unused pages
  in active GC blocks. Soft-watermark projection debits a complete block only
  when the pending allocation must open one; appending to an active role block
  does not consume free-block headroom. The projection follows the role's exact
  preferred/round-robin target plane, with independent Data/Mapping/GC cursors,
  rather than borrowing an active page from another plane. A soft-only miss never advances future
  commits to create a victim, while hard pressure may wait for committed state.
  Each completed victim must increase relocation headroom by at least its
  invalid-page count, providing a checked convergence invariant.
- Static direct placement is a capacity-checked, block-local reversible
  permutation. Deterministic fabric/wordline rotations mix structured high
  address bits without modulo wrapping or block scattering. Reserved physical
  pages fence complete NAND blocks out of the mutable FTL pool.
- In the absence of an explicit image manifest, a page is inferred to exist
  initially when a read observes bytes not covered by earlier writes. A
  full-page write-first page is never prepopulated using knowledge of a later
  read; partial writes still preserve untouched initial bytes. Static semantic
  copy-on-write uses the same rule. This inference is deterministic but
  workload-dependent; an explicit initial-image manifest is a release
  requirement for portable experiments.
- Initial-image prepopulation materializes state before simulated time; it
  does not schedule a flash read/program/erase, consume modeled time, or add
  address-heatmap traffic. The `prepopulate` heatmap source is reserved in the
  closed vocabulary but remains zero for this state-only initialization path.
- Heatmap regions named `semantic_observed_extent/<kind>` are min/max overlays
  of observed requests. They are visual bounds and may contain gaps; only bin
  counters claim exact traffic coverage.
- Production EC6 consumes a digest-bound complete object population rather
  than inferring capacity from touched pages or the maximum logical address.
  Runtime metadata, block tables, and plan-selected hot KV remain resident;
  immutable weights and cold KV are backed. Two buffers satisfy
  `resident + 2*effective buffer + unused == HBM capacity`, and every
  trace-visible layer must fit the planned effective buffer. Initial immutable
  data pages use a
  capacity-checked, stack-striped physical extent with translations resolved
  before movement. A page whose newest version came from an earlier writeback
  uses the logical FTL path. Both routes cross the owning per-stack D2D link
  and enter one of two HBM parity buffers. The next layer launches when
  current-layer execution starts.
- Full-page writes skip the initial HBF read; partial writes obtain the old
  page. Dirty pages causally read the authoritative HBM buffer, cross D2D, and
  enter logical HBF at layer completion.
- `layer-buffer-bytes` is the upper bound per parity buffer. An explicit
  contract reserves the plan size; generic microbenchmarks without object maps
  may size dynamically. EC6 rejects a workload when the layer or HBM geometry
  does not fit and prevents buffer reuse until earlier writeback completes.
- A `LayerStreamingComposition` instance is single-use so state cannot leak
  between scenario runs. Trace mode controls diagnostic retention but does not
  alter aggregate timing or traffic.


## Current Model Boundary

Implemented mechanisms include HBM bank/row timing, command constraints,
refresh interference, FR-FCFS admission, HBF resident-L2P/FTL/write buffering, physical
read/program/erase work, mapping persistence, GC relocation, per-resource
contention, static placement, cooperative destaging, and EC6 layer streaming.

Important omissions and approximations:

- no RBER distribution, ECC failure probability, retention, read disturb,
  program disturb, bad-block growth, P/E lifetime, thermal throttling, power, or
  energy model;
- SLC only; no MLC/TLC/QLC state or program-step behavior;
- program and erase are non-preemptible. Suspend/resume was removed because a
  synchronous implementation could not revise completions already returned;
- HBM same-bank refresh currently applies a conservative pseudo-channel-wide
  stall and does not model rotating bank selection or JEDEC pull-in/postpone
  credits;
- batch activation is a synchronous round constraint. The independent-subarray
  mode is an optimistic upper bound, not an alternative claim about hardware;
- active EC6 layer buffers and cooperative write regions are allocated
  disjointly from ordinary HBM foreground space. Other controller-side
  auxiliary buffers are individually checked, but a complete physical
  floorplan, shared SRAM budget, and area/power coupling are absent;
- the sibling ASTRA frontend is based on upstream `518bd51` and implements the
  published 3.0 paper subset documented in its
  `docs/astra-sim-3.0-implementation.md`; it is not vendored here, so each
  experiment must record its exact sibling commit. Replay remains one-way
  rather than a closed co-simulation; the configured outstanding window adds
  HBF-side backpressure but cannot revise ASTRA's earlier GPU issue schedule;
- exact maximum ECC in-flight depth currently retains one latency interval per
  codeword and sorts interval endpoints when stats are refreshed. This is
  exact, but its memory is linear and its refresh cost is O(N log N); very long
  AI traces should treat this diagnostic as a known scalability cost until an
  augmented interval structure replaces it.
- temporal read-buffer touch sets are pruned as causal time advances, but a
  very large open-loop batch with identical arrival timestamps retains O(N)
  pending touches until the next causal watermark. This is a known scalability
  cost for extreme same-time synthetic traces.

## How to Report an Experiment

At minimum, archive the summary JSON, input trace or its retrievable source,
initial-image description, source revision, and the exact parameter provenance
file. Preserve each scenario's `time_breakdown` and `address_heatmap`, plus the
rendered heatmap used in a report. Report user-completion throughput separately
from makespan throughput; the latter includes deferred background drain. Do
not present latency work or stage work as additive portions of makespan, and do
not infer logical HBF utilization from the physical-direct panel. For uncertain
parameters, show a sensitivity interval rather than only the default point.

Before publishing absolute claims, add an independent reference comparison and
an external validation dataset. Until then, describe HBFSim as an exploratory
simulator, and describe conclusions as model-relative.
