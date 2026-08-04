# Real workload frontends

HBFSim's physical simulator consumes memory requests, not user requests or
operator graphs. A valid request has an address, read/write direction, byte
count, earliest issue time, and optional semantic kind/layer. A source is not a
memory trace merely because it is called a trace.

## Selected architecture

The primary serving path is deliberately compositional:

```text
Qwen-Bailian production request trace
  -> Frontier request replay
  -> scheduler/batch/KV-cache event ledger
  -> strict replay and KV-lifecycle audit
  -> deterministic memory-object exporter
  -> HBFSim HBM/HBF physical replay
```

Qwen-Bailian supplies the real production distribution: request arrival time,
input/output length, conversation ancestry, request type, and anonymized
16-token prefix hashes. Frontier is the closest available frontend for turning
those requests into vLLM-style batches and KV-cache state because it models the
scheduler-batch-engine loop, chunked prefill, prefix caching, quantization,
speculative decoding, and prefill/decode disaggregation.

The boundary between Frontier and HBFSim remains explicit. Neither the Qwen
dataset nor unmodified Frontier emits GPU load/store addresses. The HBFSim
integration adds an immutable per-batch memory contract and an event-level KV
ledger to Frontier, audits both artifacts by replaying the allocator state
machine, and only then derives stable object addresses. The resulting requests
are production-request-conditioned, model-derived memory traffic. They are not
measured GPU load/store traffic.

ASTRA-sim/Chakra remains useful as a secondary training and distributed-
communication frontend. The sibling paper-aligned implementation converts a
strict, hardware-bound Chakra contract into lossless cache-line traffic with
dynamic non-overlapping tensor regions and a digest-bound workload identity.
It is based on official upstream commit `518bd51`, but it is our implementation
of the published ASTRA-sim 3.0 paper, not the ASTRA authors' unreleased source.

## Candidate assessment

| Source | Real information retained | Missing at the HBFSim boundary | Role |
| --- | --- | --- | --- |
| Qwen-Bailian | Production arrivals, token lengths, sessions, request type, prefix-block reuse | Batches, model/precision, memory objects, addresses, reads/writes | Primary serving demand source |
| Frontier | Batch/scheduler progression, exact KV block ownership and reuse, runtime optimizations, compute/communication/transfer timing | Measured address-level GPU memory requests and an HBF-aware feedback loop | Primary serving frontend |
| ASTRA-sim + Chakra | Operator dependencies, compute/communication graph, modeled cache-line addresses, distributed topology | Production serving arrivals, measured GPU memory traffic, and closed-loop HBF feedback | Secondary training/communication frontend |

This is therefore not a three-way winner-takes-all choice: Qwen-Bailian and
Frontier are complementary. Direct Qwen-to-HBFSim conversion is rejected
because it would omit batching and systematically over-count model-weight
reads. ASTRA-sim remains valuable for training workloads but is not the primary
source for long-context online inference.

The current upstream ASTRA-sim repository calls itself ASTRA-sim 2.0. No
official 3.0 branch, release, or tag was found at the pinned revision. Reports
must call the sibling path “ASTRA-sim 3.0 paper-aligned”, record its exact
revision, and identify its tensor addresses and traffic volumes as modeled,
not measured.

## Decision by workload question

| Research question | Primary input | Why |
| --- | --- | --- |
| Online inference demand, batching, prefix reuse, long context | Qwen-Bailian + Frontier | Retains production arrivals and request shapes, then applies serving policy |
| HBM/HBF placement, bandwidth, latency, GC/WAF | HBFSim | This is the only layer that models the target memory devices |
| Distributed training and collective/network sensitivity | Chakra + official ASTRA-sim | Retains operator dependencies and collective communication |
| Reproduce the original Bailian deployment's latency | None of these alone | Model, hardware, parallelism, scheduler configuration, and measured calibration are not published together |

“Real workload” therefore means that the offered request process comes from
production. Memory traffic remains model-conditioned and derived. Reports must
use those words and must never call the derived requests measured GPU traffic.

## Pinned sources reviewed

The initial evaluation used these exact upstream revisions:

| Project | Revision | License |
| --- | --- | --- |
| `alibaba-edu/qwen-bailian-usagetraces-anon` | `5f7439c51ec248a0c585f7d90a41a6f57773b912` | Apache-2.0 |
| `NetX-lab/Frontier` | `a4b22df8211864bf229258ecdfbe680f048f2d77` | MIT |
| `astra-sim/astra-sim` | `518bd513ae110428cd62eb60efc0f3993fd53c70` | MIT |

Do not vendor the datasets. Download a named upstream file into `out/`, retain
its Git LFS object digest, and record the exact upstream revision in every
derived manifest.

## Prepare the canonical production suite

There is one production entry point. It consumes the pinned Thinking source
and the tracked suite config; arbitrary `start_index` and `max_requests`
selection no longer exists.

```bash
python3 tools/prepare_qwen_bailian_workload.py \
  --input out/workloads/qwen-bailian/qwen_thinking_blksz_16.jsonl \
  --suite-config \
    configs/workloads/frontier/qwen-thinking-llama31-70b-production.json \
  --output-dir out/workloads/qwen-bailian/llama31-70b-production

python3 tools/verify_qwen_bailian_workload.py \
  --input out/workloads/qwen-bailian/qwen_thinking_blksz_16.jsonl \
  --suite-config \
    configs/workloads/frontier/qwen-thinking-llama31-70b-production.json \
  --suite-dir out/workloads/qwen-bailian/llama31-70b-production \
  --output \
    out/workloads/qwen-bailian/llama31-70b-production.verification.json
```

The config pins the full 27,901,454-byte source digest, revision, 10,812
records, source statistics, Llama 3.1 70B identity, primary
`w8a16-kv-bf16` profile, BF16 sensitivity profile, and all three 256-request
selectors. The generator validates every source record before publishing an
atomic directory. It rejects duplicate JSON keys, unordered ancestry or
timestamps, invalid turn progression, non-positive token lengths, and a
prefix-hash count different from `ceil(input_length / 16)`.

The independent verifier does not trust the generated CSVs. It reconstructs
all three configured source slices, compares each CSV byte for byte, rechecks
every digest and statistic, and rejects an extra unattached file or directory.
Its receipt is a required input to the Frontier replay auditor.

The output columns consumed by Frontier are `arrived_at`,
`num_prefill_tokens`, `num_decode_tokens`, `session_id`, and
`block_hash_ids`. Source IDs, parent IDs, turn number, and request type remain
as audit columns. The source's final hash identity is retained when the prompt
ends in a partial 16-token block, but only the leading
`floor(num_prefill_tokens / 16)` complete blocks are eligible for Frontier
lookup, cache assignment, and query/hit accounting. Hash repetition in an
input window is not a cache-hit rate; the audited Frontier prefix/KV lifecycle
supplies that result.

## Run and audit a Frontier replay

The current structural exporter deliberately supports one dense, co-located
replica with TP=PP=DP=1, 16-token KV blocks, prefix caching, no preemption, no
speculative decoding, and no KV preallocation. Its small 7B fixture exists
only for CI. It is not a paper workload.

Frontier must be built from the pinned revision plus the HBFSim memory-ledger
changes. The run must enable `frontier_stage_batch_ledger.jsonl`; the
integration additionally publishes `frontier_kv_block_lifecycle.jsonl` and
`frontier_hbfsim_integration.json`.

The exact patch, base revision, post-apply file hashes, and contract versions
are tracked under `integrations/frontier/`. Prepare only a clean checkout at
the pinned revision:

```bash
python3 tools/prepare_frontier_integration.py check \
  --frontier-dir FRONTIER_CHECKOUT
python3 tools/prepare_frontier_integration.py apply \
  --frontier-dir FRONTIER_CHECKOUT \
  --receipt FRONTIER_INTEGRATION_RECEIPT
python3 tools/prepare_frontier_integration.py verify \
  --frontier-dir FRONTIER_CHECKOUT
```

`apply` stages the exact patch. `verify` requires the staged diff to be
byte-identical to that patch, rejects any unstaged or untracked file, and
checks every post-apply digest. A hand-edited or merely self-reported Frontier
tree is not eligible.

The current integration also binds the exact model JSON path and SHA-256 into
Frontier's resolved config. Its parameter counter includes attention, FFN,
per-layer and final normalization, embeddings, an untied output head, and
pipeline-boundary/auxiliary allocations. `system_metrics.json` publishes those
categories separately. For Llama 3.1 70B it also binds the exact selected
memory profile: symmetric INT8 matrix payload plus one BF16 scale per output
channel for `w8a16-kv-bf16`, or all-BF16 storage for the capacity
sensitivity. The auditor and exporter independently reconstruct matrix
payload, non-matrix payload, quantization metadata, category totals, and total
parameters; a replay whose planner, model descriptor, and resident-byte
accounting disagree is rejected.

With preemption disabled, admission uses
`full_request_kv_reservation_v1`. Each admitted request reserves
`ceil((input_tokens + output_tokens - 1) / 16)` blocks, regardless of a
possible future prefix hit. The lifecycle audit reconstructs all simultaneously
active requests at every batch cursor and proves that the sum of those full
reservations never exceeds the physical block pool. Merely observing zero
preemption is not sufficient.

The execution-time predictor may be dummy for structural export because its
latencies are not consumed as hardware timings. Dummy and uncalibrated profiled
runs are both ineligible for timed export.

Every replay admitted to the memory-export stage must pass the accounting
auditor:

```bash
python3 tools/audit_frontier_replay.py \
  --request-csv \
    out/workloads/qwen-bailian/llama31-70b-production/steady.frontier.csv \
  --request-manifest \
    out/workloads/qwen-bailian/llama31-70b-production/steady.adapter-manifest.json \
  --request-suite-verification \
    out/workloads/qwen-bailian/llama31-70b-production.verification.json \
  --frontier-output-dir FRONTIER_OUTPUT_DIRECTORY \
  --frontier-revision a4b22df8211864bf229258ecdfbe680f048f2d77 \
  --frontier-integration-receipt FRONTIER_INTEGRATION_RECEIPT \
  --output FRONTIER_OUTPUT_DIRECTORY/hbfsim.frontier-audit.json
```

The audit checks every input and output digest, requires all Frontier
length/time scale factors to remain one, validates completed-request and cache
statistics, requires an explicit embedding-sharing decision plus the exact
model-config path and SHA-256, and proves for every request:

```text
input + output tokens
  = scheduled tokens + cached prefill tokens + prefill-generated first token
```

It also replays every `prefix_lookup`, `prefix_admission`, `touch`, `evict`,
`allocate`, `cache_assign`, and `release` event. At each batch cursor, the
reconstructed block ID, hash, reference count, and request ownership must match
the compact contract's block count and canonical allocator-state SHA-256.
Memory contract v4 keeps the lifecycle stream as the single physical-state
source, embeds one exact residency-plan v1 in every batch, and does not repeat
full block tables or events in every decode batch. Event IDs
and times must be monotonic, an
eviction may target only a free hashed block, a release may remove only an
owned block, a partial final prompt block may never be queried or assigned a
cache hash, and the final state must contain no live ownership or nonzero
reference count. Frontier also fails the run if its sequential event queue
drains before every generated request completes. The audit fails closed on any
mismatch.

The audit distinguishes a valid scheduler ledger from valid timing. Dummy
timing is never performance data. A non-dummy profile is still labelled
`profiled_unvalidated` until an independent calibration artifact is supplied;
merely loading a profiling CSV is not evidence of prediction accuracy. Audit
schema v7 therefore exposes separate booleans for structural export,
memory-system service claims, TTFT/TPOT/SLO claims, and time-based throughput
claims. Dummy runs may satisfy only the first two.

Do not assemble the three production runs with copied shell commands. The
canonical structural-suite entry point fixes the model, precision, scheduler,
batch policy, dummy-timing boundary, seed, and all three windows:

```bash
python3 tools/run_frontier_70b_structural_suite.py \
  --frontier-dir FRONTIER_CHECKOUT \
  --request-suite-verification \
    out/workloads/qwen-bailian/llama31-70b-production.verification.json \
  --frontier-integration-receipt FRONTIER_INTEGRATION_RECEIPT \
  --output-dir FRONTIER_STRUCTURAL_SUITE_OUTPUT
```

It requires a clean tracked HBFSim commit, verifies that the Frontier staged
diff is byte-identical to integration v8, refuses an existing output
directory, runs all three 256-request windows, audits each full ledger, and
publishes one aggregate receipt only after every window demonstrates actual
multi-request batching, prefix/KV lifecycle coverage, exact 70B memory
accounting, exact hybrid-residency conservation, and full-request reservation
conservation. The fixed structural point is 96 GiB HBM, pressure 1.0, and an
explicit 8 GiB unprofiled runtime-overhead sensitivity. The receipt
intentionally sets `paper_result_eligible=false`; it is a prerequisite for
capacity planning and the bounded four-baseline evidence slice, not a
publication runner.

Generate the canonical static capacity preflight independently of execution:

```bash
python3 tools/prepare_frontier_70b_capacity_grid.py \
  --request-suite-verification \
    out/workloads/qwen-bailian/llama31-70b-production.verification.json \
  --output FRONTIER_CAPACITY_GRID_PREFLIGHT
```

The preflight covers both W8A16/KV-BF16 and BF16/KV-BF16, all 96/192/288 GiB
HBM topologies, and exact pressure targets 0.75/1.0/1.25/1.5/2.0. It derives
the weight and active-buffer ledgers from the tracked descriptors and scans all
three byte-exact request CSVs. Every point is classified as
`planner_infeasible`, `request_infeasible`, or `execution_candidate`.
`request_infeasible` means the logical KV pool cannot hold even the largest
single full-request reservation in at least one window. An
`execution_candidate` is not a passing experiment. The former 72-cell
full-replay matrix tried to turn every candidate, every window, and every
baseline into one transaction, but one Frontier replay contains tens of
thousands of batches whose exact memory objects expand to tens or hundreds of
GiB per batch. The design was never operationally executable at the required
64-byte HBM/4-KiB HBF fidelity, produced no evidence, and has been deleted.
Do not reconstruct it with shell loops.

The one current execution path is the predeclared W8A16/KV-BF16 burst slice in
`configs/studies/frontier-70b-two-axis-burst.json`. It consumes three complete
upstream Frontier replays at 4H4F pressures 0.75, 1.0, and 1.25, then exports
the mechanically selected q=0, q=0.5, and q=1 batches. Each stratum executes
five cells and four baselines: a byte-identical p1.0 population across 6H2F,
4H4F, and 2H6F, plus fixed-4H4F p0.75/p1.0/p1.25. Run each quantile with
`tools/run_frontier_70b_two_axis_slice.py`, then certify all three without
averaging them:

```bash
python3 tools/verify_frontier_70b_two_axis_evidence.py \
  --q0-receipt TWO_AXIS_Q0/slice.receipt.json \
  --q0-5-receipt TWO_AXIS_Q0_5/slice.receipt.json \
  --q1-receipt TWO_AXIS_Q1/slice.receipt.json \
  --output-dir FRONTIER_70B_TWO_AXIS_EVIDENCE
```

The verifier reopens all 15 cell receipts and 60 baseline records, rechecks
their transitive digests, exact temporal rows, latency fields, byte identity,
and canonical WAF, and emits only a stratified concatenation. The certificate
remains `paper_result_eligible=false`: configured runtime overhead and external
CXL/NVMe timing are sensitivities, Frontier compute/communication timing is
dummy, and three single-batch memory traces are not a full replay. TTFT, TPOT,
SLO, end-to-end throughput, cross-quantile averages, and absolute external
device claims remain forbidden.

## Frontier 70B HBF WAF and wear-demand study

The canonical application-level wear path is
`tools/run_frontier_70b_hbf_wear_study.py` with the tracked
`configs/studies/frontier-70b-hbf-wear.json` contract. It consumes the complete
audited 256-request burst lifecycle rather than a q=0/q=0.5/q=1 batch sample.
For 6H2F, 4H4F, and 2H6F it first recomputes the full-lifecycle active-weight
buffer and hot/cold KV fixed point, then counts every cold-KV append in exact
Frontier batch, layer, request, and token order. Read-only streamed weights do
not enter the write denominator.

Full-capacity logical demand is exact. Physical media replay is deliberately
separate: it selects three predeclared block-ID residues at stride 64, compacts
their addresses without reordering requests, preserves mutable-data occupancy
after the read-only weight reservation, and runs one and eight repeated
lifecycle epochs. One epoch exposes fill/checkpoint overhead; eight epochs
must exercise steady GC. Every result must conserve data-page programs,
mapping checkpoints, GC relocations, erases, logical bytes, and physical
payload bytes. The only WAF is
`physical_write_bytes / logical_write_bytes`; OOB, ECC, and link bytes are
excluded. The topology aggregate is a logical-byte-weighted mean over the
three residues, never an average of already-rounded WAF values.

The study also reports physical bytes divided by usable HBF capacity and a
first-order sensitivity over explicitly assumed new-KV-token rates and media
cycle budgets. Those rows are not an HBF product lifetime, cost, or DWPD
prediction: timing/endurance parameters are not calibrated, uniform wear is
assumed, and retention, disturb, bad-block growth, P/E failure, thermal, and
power effects are absent. Frontier compute/communication timing remains dummy,
so TTFT, TPOT, SLO, and end-to-end throughput are forbidden.

For target-matched Llama 3.1 calibration, replay the same Qwen window against a
real serving backend with the companion
[`blitz-serving/trace-replayer`](https://github.com/blitz-serving/trace-replayer),
then compare request count,
prefix-hit blocks, batch composition, TTFT, TPOT, and throughput against
Frontier. The calibration manifest must name the model, weights/activation/KV
precision, accelerator, parallelism, vLLM revision and flags, warm-up policy,
and observed error. The replayer is a validation instrument; it still does not
produce an address-level HBFSim trace.

The first no-new-measurement public anchor is separately pinned at
`validation/calibration/h100-llama2-70b-public-anchor.json`. It imports
Microsoft Vidur's H100 Llama 2 70B FP16 operator/collective profiles as fit
inputs and MLPerf Inference v5.0 NVIDIA result `5.0-0057` as held-out-only
evidence. `tools/import_h100_llama2_70b_anchor.py` creates an immutable source
bundle, and `tools/verify_h100_llama2_70b_anchor.py` independently rebuilds
every profile and MLPerf record. A passing receipt is deliberately labelled
`anchor_ready_not_calibrated`: it contains no candidate-predictor error and no
held-out prediction, so it cannot enable Frontier TTFT, TPOT, SLO, throughput,
or timed export.

The companion fixed protocol and
`tools/fit_h100_llama2_70b_calibration.py` now build a Vidur-only multilinear
interpolation candidate. Its independent verifier reconstructs 49 model
instances and 20,952 evaluation keys without importing the fitter; all models
pass the declared operator gates on the pinned public profiles. That advances
only the narrow state to
`operator_calibrated_external_validation_not_established`. The MLPerf gate is
held out until after fitting and then stops without a prediction: the NVIDIA
submission is TensorRT FP8/FP8-KV at configured GPU batch size 1024, while the
Vidur profiles are FP16 with decode batch size at most 128 and omit the serving
runtime, host overhead, scheduler/arrival trace, and per-request generated
lengths. Therefore this candidate still cannot enable Frontier TTFT, TPOT,
SLO, throughput, or timed export. See
[`validation/calibration/README.md`](../validation/calibration/README.md) for
the exact scope and commands.

## Current structural exporter boundary

The schema-v2 model-memory descriptors have distinct exact ledgers for
Llama 3.1 70B `w8a16-kv-bf16` and `bf16-kv-bf16`; the Llama 2 7B descriptor is
still CI-only. This closes model geometry and byte accounting, not publication
qualification. Hybrid-residency plan v1 and the complete 30-point static grid
preflight are implemented and independently audited. The canonical per-trace
runner now executes all-HBM/HBF/CXL/NVMe with digest-bound traffic and
cross-baseline placement/scheduler/credit checks. Its HBF case now performs
the canonical WAF audit: every trace write must be mutable KV, logical bytes
must equal cold-KV writeback, physical bytes must conserve data-page,
mapping-checkpoint, and GC-relocation payload programs, and a zero denominator
must serialize as `null`. It also reports physical write bytes divided by
usable HBF payload capacity as wear context, not as a second WAF. The
five-cell runner is composed only by the three-stratum verifier described
above. That verifier accepts the exact q=0/q=0.5/q=1 census and preserves the
one WAF definition, but it does not turn sampled memory transactions into a
full replay or calibrated paper result. Runtime-overhead profiling, external
tier calibration, scalable exact execution, and a later publication
certificate remain required before a Frontier-derived performance result can
be published.

Trace schema v4 implements these rules:

1. Model embeddings, every transformer layer, final norm, output head, the
   finite KV-slot arena, and block-table metadata occupy collision-free,
   aligned regions. Metadata is tagged `hbm_only`.
2. A Frontier physical `block_id` is the address identity of one finite KV
   slot. Eviction and later allocation reuse that same address; the exporter
   does not invent a second allocator. The established HBFSim
   `shared_context` kind means a full block admitted to the prefix cache; the
   manifest states explicitly that this label alone does not prove a
   simultaneous reference count greater than one.
3. Dense layer weights are read once per scheduled batch and layer, not once
   per request. Final norm is read for every batch. The full output head is
   read only when at least one request snapshot explicitly produces an output
   token; an intermediate chunked-prefill-only batch does not pay that traffic.
   MoE is rejected by v2 rather than approximated.
4. `head_dim = hidden_size / num_attention_heads` and
   `K+V bytes/token/layer = 2 * num_key_value_heads * head_dim * kv_bytes`.
   The v2 canonical object map is block-major, layer-major, then token-major.
   This keeps every Frontier physical block as one finite cross-layer object.
   It is explicitly not claimed to be a measured or backend-specific vLLM
   tensor stride; address-layout sensitivity requires a separately named
   model version.
5. Scheduled input tokens produce KV writes. Each request/layer reads its
   modeled post-schedule context once. This is an explicit IO-optimal traffic
   model, not a kernel trace.
6. Qwen does not expose token IDs. Each scheduled embedding lookup therefore
   uses an independent deterministic SHA-256 surrogate of model name, request
   ID, and token position. The manifest explicitly says these are not true
   token IDs; sequential vocabulary locality is never invented.
7. Off-chip scratch traffic is excluded because no calibrated kernel-level
   trace exists. Both scratch coefficients must be zero or export fails.
   This is different from claiming scratch has zero hardware cost.
8. Frontier batch-ready times are preserved only as offered times. Ordered
   global phase IDs express complete-before-next dependencies, and matching
   global layer IDs identify EC6 streaming windows for embedding, every model
   layer, final norm, and (when emitted) LM head. Final norm and LM head are
   separate windows so one planned active buffer is sufficient. Frontier
   execution latency is not promoted to
   HBFSim hardware latency, so the exporter does not emit `compute_ns`.
9. The trace, object map, phase map, source artifacts, model descriptor, and
   audit are SHA-256 bound. Batch selection is contiguous; scaling and byte
   sampling are forbidden.
10. The exporter independently recomputes the complete hybrid plan. Unique
    footprint is immutable weights plus the logical KV arena, global block
    records, and configured runtime overhead. Physical HBM contains exactly two
    active-weight buffers, hot KV, block records, runtime overhead, and unused
    bytes. Backing contains read-only weights and cold KV. Frontier's logical
    block count, plan, system weight ledger, and exported object map must agree
    byte-for-byte before any output is published.
    `prepare_frontier_residency_config.py` then digest-checks the manifest and
    object map plus the trace bytes, recomputes page allocation and rounding,
    and emits a self-contained explicit HBFSim residency overlay/receipt.
    HBFSim rechecks the expected trace bytes/SHA-256 at execution and rejects
    truncation. Publication runs may not reconstruct capacity from the subset
    of pages touched by a selected trace window.
11. Capacity pressure is exactly
    `unique_resident_footprint_bytes / physical_hbm_capacity_bytes`; target
    points are represented as exact rational values, and whole-block rounding
    slack is explicit. HBF demand is never created by pretending the complete
    weight set is HBM-resident.
12. The current runtime-overhead value is explicitly
    `configured_unprofiled_sensitivity`, so
    `hardware_capacity_calibrated=false`. It supports structural and
    sensitivity claims only. A calibrated capacity claim must consume a
    digest-bound runtime-overhead profile under a new reviewed contract rather
    than silently reinterpreting this plan.

For a large structural trace, validate its exact population without expanding
multi-gigabyte spans into cache-line requests:

```bash
build/scenario_compare \
  --trace FRONTIER_OUTPUT_DIRECTORY/hbfsim-memory.trace \
  --trace-census-only true
```

The census uses interval union for unique cache-line coverage and reports
operation counts, read/write bytes, phase count, and trace digest. A full
device replay may be much more expensive because HBFSim must expand each
logical span into physical HBM bursts or HBF pages. Use a contiguous
`--max-batches` exporter window for detailed smoke tests; never scale its bytes
and report it as a full run.

## Required production suite

The canonical suite uses the pinned Qwen Thinking distribution and exactly
three contiguous 256-request windows:

| Window | Selector | Start | Arrival span | Prefill | Decode | Tail coverage |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| steady | closest to the sliding-window median duration/prefill/decode under MAD normalization | 3074 | 160.011 s | 1,185,372 | 885,429 | 39,341 input; 56,742 output |
| burst | minimum arrival span, earliest tie | 810 | 43.869 s | 707,190 | 1,244,249 | 21,201 input; 18,154 output |
| long_context_decode_tail | maximize the smaller of normalized max-input and max-output, then their sum | 4332 | 167.136 s | 1,099,024 | 914,852 | global-max 51,622 input; 68,858 output |

The long-tail selector is intentionally balanced. Merely choosing the global
longest decode would underrepresent long input context. The selected window
contains the source's largest input and a decode at 90.6% of the source
maximum.

Every EC and every all-HBM/HBF/CXL/NVMe baseline for a given row must consume
the same verified window and, after Frontier export, the same byte-identical
logical memory workload. Reports must distinguish source-request time,
frontend scheduling time, HBFSim offered time, user completion, and drain.

The Llama 2 7B FP16 descriptor remains only as a fast structural unit-test
fixture. There is no 7B paper runner, hand-selected `batch299`, or retained
7B result baseline. Llama 3.1 405B is outside the primary suite and may be
added only as a separately named scalability stress after the 70B path is
complete.
