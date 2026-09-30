# Evidence policy

> Status: Current
> Last reviewed: 2026-09-29

HBFSim compares memory mechanisms under a declared model. Results depend on
the source and executable, the resolved system configuration, the initial
memory image, and the workload trace. They are not measurements of a shipping
HBF device. Archive all four inputs with the result.

## Parameter evidence

| Grade | Meaning |
|---|---|
| `standard_published` | A retrieved primary standard directly states the requirement or value. |
| `vendor_published` | A primary vendor source states a product property or target. |
| `literature_derived` | A cited paper or design supports an interpretation; it is not an HBF measurement. |
| `measured_calibration_with_exploratory_fallbacks` | A repository-local measurement fits the value for one named platform (L4 where a held-out run exists); the same keys keep exploratory values on other platforms. |
| `exploratory_assumption` | A value closes a missing specification or defines a sensitivity. |
| `implementation_policy` | An algorithm or representation chosen by the simulator or Host. |

These evidence grades are unrelated to HBF speed Grades 1/2/3. A speed grade
selects an interface rate and maximum organization; it is not a reliability,
quality, or confidence rating.

The [OCP HBF v0.7.0 specification](https://www.opencompute.org/documents/ocp-hbf-architecture-specification-v0-7-0-final-pdf)
was retrieved in full (130 pages, 3 August 2026). Its SHA-256 and page-specific
bindings are in [parameter provenance](../../configs/parameter-provenance.json).
Page 16 gives 48/96/192 GB/s per channel and 384/1536/3072 GB/s at the respective
maximum channel counts, using 75% AXI efficiency once. Pages 32/35 instead give
94/188 GB/s per module. We follow page 16 and retain that discrepancy explicitly;
no erratum has been obtained. The older announcement-only 400/3000 GB/s overlays
are removed.

Page 16's 512 GiB example is a geometry reference, not a capacity guaranteed for
every Grade. The full comparison profiles use 16 channels, one die per channel,
16 banks per die, and 4096-byte pages. Miniquick profiles preserve their declared
scaled capacity. Internal ECC, raw channels and TSV are separately provisioned
assumptions: changing Grade alone never increases them implicitly.

HBF NAND read/program/erase timing, OOB size, ECC latency and throughput,
controller buffer capacity/access timing, thermal RC, and the exact internal data path are
assumptions. All current profiles use a 4 microsecond NAND read and a
75 microsecond complete program, including internal verification. No separate
verification delay or work counter is added. These selected research inputs
are not OCP timing requirements or measured HBF latency.

OCP section 11.4 excludes device GC and copyback. Host mapping, collection,
coalescing and placement algorithms are implementation policies. Copies cross
the ordinary HBF interface and hold a finite controller copy slot. OCP's Host
system page buffer / xPU Buffer does not establish a CPU DDR attachment or a
shared 64 GB/s memory port. Our selected architecture reserves controller storage
from the system's physical HBM and charges buffer transfers to the same HBM data
channels as application traffic. This placement is a research choice, not an OCP
requirement. Buffer transfers use channel-level timing without per-row DRAM
commands; application HBM accesses retain the command model. Zone invalidation,
ownership reclamation and physical page-zero autoerase have separate counters.

The HBM comparison follows the public HBM4 JESD270-4 organization referenced by
OCP: 2048 DQ bits, 32 channels, two pseudochannels per channel and an 8 Gb/s
reference pin rate. The [JEDEC announcement](https://www.businesswire.com/news/home/20250416843598/en/)
and primary [Synopsys HBM verification documentation](https://www.synopsys.com/verification/verification-ip/memory/hbm-verificationip.html)
support the organization. The complete JESD270-4 timing tables were not obtained.
Absolute timing minima, command-clock ratio, burst and refresh timings remain
explicit model assumptions. The address swizzle and scheduler are implementation
policies; public organization evidence does not validate their product realism.

External DRAM, CXL and NVMe profiles retain separately declared directional
transport, media and controller assumptions. Their sources do not imply full
PCIe, CXL coherence, SSD FTL, or failure consistency coverage.

## State, causality and initial conditions

Resource reservations must conserve occupancy and may backfill only genuine
idle intervals. One HBF bank senses serially; program/erase are non-preemptible.
Responses and mapping publications cannot become visible before completion.
Block epochs prevent stale callbacks from restoring invalidated data. Device
cache records use modeled event time, not API call order, and remain local to
each bank. Independent resources must not inherit unrelated future barriers.

Raw erased-page reads are errors. An unmapped logical read is an explicit Host
policy that assembles erased-value bytes in a host buffer. Partial logical writes
preserve existing data or the declared erased value, then program full pages.
Every physical page-zero program pays an erase, including the first program
after an erased initial state, unless a snapshot declares
`hbf-initial-free-blocks-erased` (reported as `preconditioned_erased_blocks`;
never in lifetime evidence). GC reclamation alone does not pay another erase.

Initial images establish completed state before measured time and do not
invent program, erase or transfer traffic. Reference-runner inference from
read-before-write is workload-dependent; prefix comparisons must supply the
same explicit image or complete object population. Untouched immutable objects
still occupy reserved physical blocks.

Persistent image v6 restores quiescent mapping/allocation/zone state and physical
wear at a new time origin. Old images are rejected. Command-boundary crashes do
not simulate partially completed NAND operations or retain uncheckpointed wear.
Persistence round trips establish simulator continuity, not retention physics.

## Verification and claim scope

Verification levels describe evidence, not marketing quality:

1. L0: build, runtime and focused regression checks.
2. L1: causal state transitions, resource/byte conservation, exact address
   round trips, hand-calculated timing and sharp regimes.
3. L2: independently implemented oracles or differential execution, for the
   exact facets exercised by the current model and corpus.
4. L3: pinned unmodified external simulators, only within the named common
   facets; an adapter fixture alone is insufficient.
5. L4: calibration to declared data with held-out observations.
6. L5: predictions against independent hardware measurements.

An old certificate does not carry across this model change. A certificate must
bind a clean tracked tree, executable hashes, current schema/corpus and recorded
checks. Passing a bank-cache oracle or HBM scheduler differential does not imply
independent validation of all Host GC or application timing. Missing or obsolete
evidence is reported as such, never promoted to a pass by changing a numeric
baseline. See [coverage](../verification/coverage.md) for executable contracts.

## Reporting a result

Summary schema v19 records resolved HBF/HBM organization, resource work, mapping,
physical media traffic, controller buffer accesses and the standard identities. Preserve the
JSON, trace SHA-256, initial image/population digest, source/executable identity,
and parameter registry. Old summaries must be regenerated for current studies.

Report Host acknowledgement separately from completion after drain. Resource
busy time is summed across resources and can exceed makespan; it is not an
additive share of wall time. Physical read bytes, full-page programs, actual
erases, mapping checkpoints and Host relocation bytes have distinct meanings.
Wear extrapolation needs paired block histories and an explicit workload
window, not subtraction of unpaired marginal histograms.

The current model omits AXI fragments and response IDs, UCIe framing/retry,
MMIO/IEEE1500, retention/disturb faults, bad-block growth, severe thermal and
shutdown behavior. Light-throttle RC pacing is an assumption with a limited
operating envelope. It cannot establish OCP electrical, reliability or thermal
conformance. HBM refresh uses causal target-bank PRE/REF events, permits work
in unaffected banks, and does not implement JEDEC pull-in/postpone credits.

ServeLoop owns application workloads and placement. A two-layer excerpt measures
that excerpt's memory service only. Neither uncalibrated compute timing nor a
bounded memory run establishes full-request TTFT, TPOT, token throughput or SLOs.
Model-relative comparisons are valid within those stated limits.

The address-heatmap artifact reports five domains, with names fixed in the
[scenario contract](scenarios.md). Each domain retains its own traffic denominator.
