# HBFSim

HBFSim is an early-stage, event-level simulator for exploring how
flash-as-memory HBF could complement HBM under AI workloads. Its goal is to
give researchers and hardware/software teams an inspectable place to study
placement, staging, bandwidth, latency, write amplification, and capacity
trade-offs before HBF hardware and complete public specifications exist.

The project is **pre-alpha**. It is suitable for mechanism exploration and
sensitivity studies, not yet for product prediction or reliability/endurance
claims. Published HBF facts, literature-derived assumptions, and exploratory
defaults are kept separate in
[`docs/model-validation.md`](docs/model-validation.md) and
[`configs/parameter-provenance.json`](configs/parameter-provenance.json).

## Current Scope

Implemented physical-model pieces:

- HBM stack model with pin-rate/DQ-width/BL-derived bandwidth and burst timing,
  command-clock-aligned issue, address mapping, bank state, row
  hit/miss/conflict, refresh interference, and pseudo-channel bus scheduling.
- HBF stack model with SLC page geometry, a complete per-stack L2P table
  resident in controller DRAM, write buffer/coalescing, mapping-page checkpoint
  batching, TSU-like flash transaction
  scheduling, read/program/erase media operations, pipelined ECC with separate
  response-latency and raw-codeword-throughput controls, an explicit raw
  page+OOB versus decoded-payload boundary, SRAM/HB IO breakdown, exact plane
  reservation calendars, epoch-protected state publication, and GC
  relocation/erase paths. Direct physical blocks are owned separately from
  FTL/GC blocks so the two interfaces cannot silently reinterpret each
  other's pages.
- External-backing model for no-HBF baselines, with shared CXL-memory and
  NVMe-SSD scheduling semantics, end-to-end device credits, explicit
  M2S/controller/media/S2M stages, address-striped media channels, asymmetric
  media bandwidth/latency, full-duplex links, and separate payload/protocol/
  wire accounting.
- Hybrid memory system layer with compact fixed and hot-KV HBM residency,
  dynamically sized active-layer buffers, selectable HBF or external backing,
  and separate foreground versus background access accounting.
- Scenario comparison across `all-HBM`, `all-HBF`, `HBM-HBF-Flat`,
  `HBF-static-direct-read`, `HBM+HBF-layer-streaming`, and
  `HBM+External-layer-streaming`.
- Per-scenario observability in summary schema v16: an auditable timing
  breakdown and a bounded, source-attributed address heatmap spanning workload
  logical, HBM physical, HBF logical, HBF physical, and external physical
  address domains.
- Static architecture editor under `web/architecture-editor/`, with node-level
  code references that open the corresponding C++ definition.

Generated outputs live under `out/` and are intentionally not versioned.

## Build And Verify

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
cmake --build build --target physical_guard
```

For changes to timing, mapping, composition, or accounting, also run the deep
component checks:

```bash
cmake --build build --target use_cases
```

`ctest` covers deterministic integration and replay checks. `physical_guard`
and `use_cases` exercise physical invariants, causal ordering, regime changes,
cross-path consistency, and canonical WAF accounting.
The default tests also run a fast exact resident-mapping/WAF oracle and mutation
tests that prove the verifier rejects corrupted GC, free-page, read, ECC,
erase, and scheduler counts. The longer statistical regimes remain available
through `cmake --build build --target waf_cases`.

To isolate capacity-driven dirty eviction from the proactive EC6 layer
streaming policy, run the dedicated HBM overflow comparison:

```bash
./build/overflow_offload_experiment
python3 tools/analyze_overflow_offload_experiment.py \
  --input out/capacity-overflow/summary.json \
  --markdown out/capacity-overflow/report.md \
  --experiment ./build/overflow_offload_experiment
```

It writes 200 GiB through the 192 GiB 4H4F HBM profile, compares FIFO
offloading to HBF, host-linked CXL memory, and an NVMe SSD, then cold-reads
the exact offloaded prefix back through HBM. Capacity and traffic are exact;
timing is projected only after 16-to-32 MiB HBM-fill and 32-to-64-batch
page-exact replays converge. In addition, exact samples sweep controller
windows of 1/4/16/64/256 pages and report offered versus service latency,
resource utilization, queue work, and wire efficiency. A converged layer
projection reports one offload plus 1/2/4/8/16/32 restores and derives the
winner and discrete break-even.

Direct output is explicitly `exploratory_unattached`. A paper artifact must
use `tools/run_paper_capacity_overflow.py`, which verifies a clean-tree
foundational certificate against the exact `scenario_compare` and overflow
binaries before attachment. The policy, equations, method, and default result
are documented in
[`docs/capacity-overflow-experiment.md`](docs/capacity-overflow-experiment.md).

The current LLM WAF/wear experiment consumes the complete audited Frontier
70B KV lifecycle:

```bash
python3 tools/run_frontier_70b_hbf_wear_study.py \
  --scenario-compare build/scenario_compare \
  --audit FRONTIER_BURST/burst.audit.json \
  --placement-manifest FRONTIER_SAMPLE/manifest.json \
  --object-map FRONTIER_SAMPLE/object-map.json \
  --study-config configs/studies/frontier-70b-hbf-wear.json \
  --out-dir out/frontier-70b/hbf-wear
```

It derives exact logical cold-KV writes for 6H2F, 4H4F, and 2H6F from the
256-request burst, then measures canonical media WAF with three predeclared
capacity-scaled block strata at one and eight lifecycle epochs. The report
separates logical bytes, physical payload bytes, GC relocation, and
physical-write/HBF-capacity. Its lifetime table is an explicitly uncalibrated
uniform-wear sensitivity, not a product lifetime or DWPD claim. The independent
`waf_cases` target remains the mechanism-level GC/WAF verification suite.
Guard artifacts are written under:

```text
out/physical-guards/
```

## Architecture Editor

Launch the local editor:

```bash
python3 -m http.server 5174 --bind 127.0.0.1 --directory web/architecture-editor
```

Open:

```text
http://127.0.0.1:5174/?repo=/absolute/path/to/HBFSim
```

Click a node's `code:` label, or select a node and click `Open`, to jump to the
current C++ or Python definition through a `vscode://file...` link. The explicit
`repo` query keeps the editor portable and avoids embedding a developer's
machine path.

The maintained templates cover the L0-L6 boundary from workload evidence and
the canonical trace contract through EC0-EC6 composition, the no-HBF external
control, HBM/HBF/external device and media resources, and schema-v16
observability. Entity color denotes hardware or policy type; the `L0`-`L6`
eyebrow denotes abstraction level.

Check topology, required entities, abstraction levels, edge endpoints, and
code-reference drift with:

```bash
node web/architecture-editor/tools/design-sync.mjs
```

## Scenario Compare

The scenario tool consumes Ramulator-compatible traces:

```text
0xADDR R
0xADDR W
```

It also accepts optional request size and semantic labels:

```text
0xADDR R 4096 model_weights
0xADDR W bytes=4096 kind=generated_context
```

Semantic labels let `HBF-static-direct-read` inspect the raw static HBF read
path without FTL translation. Production Frontier layer streaming consumes a
digest-bound explicit residency contract: complete runtime metadata and block
tables plus as many low-ID hot-KV blocks as fit the explicitly selected
topology are resident in HBM; read-only weights and cold/overflow KV use
backing and two HBM ping-pong buffers. The logical object population and trace
stay fixed while the placement compiler changes only the hot/cold partition.
The overlay binds the exact trace path, bytes, and SHA-256, so substitution,
post-binding mutation, and `max-ops` truncation fail before simulation. The
selected trace window determines traffic but cannot shrink the allocated
population. A trace-derived first-touch policy remains only for generic
microbenchmarks without an object map.
`HBM+External-layer-streaming` runs the byte-identical placement, prefetch,
foreground-HBM, dirty-writeback, and double-buffer policy with CXL memory or
an NVMe SSD as the backing tier. Logical address holes and repeated accesses
consume no additional HBM capacity.

Frontier exports must not assemble those cases with separate shell commands.
`tools/run_frontier_memory_baselines.py` binds one manifest/object map/trace
and runs the canonical `all_hbm_upper_bound`, `hbm_hbf`,
`hbm_cxl_memory`, and `hbm_nvme_ssd` set. It rejects traffic, placement,
credit, HBM-geometry, or backed-byte drift and emits one aggregate receipt.
The all-HBM case is labelled as a fully resident upper bound; CXL-memory and
NVMe-SSD numbers remain parameter sensitivities until calibrated.
`hbm_hbf` always means the HBM+HBF hybrid/layer-streaming system, never a
pure all-HBF result.
The canonical same-population mix-ratio plus fixed-4H4F pressure slice is
declared in `configs/studies/frontier-70b-two-axis-burst.json` and executed by
`tools/run_frontier_70b_two_axis_slice.py`. Run its exact q=0/q=0.5/q=1
strata through `tools/verify_frontier_70b_two_axis_evidence.py`; the verifier
publishes only a stratified 60-row memory-service table. The former
equal-pressure 72-cell full-replay matrix resized each topology's KV
population, was not operationally executable at exact physical granularity,
and has been deleted rather than retained as a second path.
The next-scale study is the fixed-TP8 Kimi K3 capacity/service boundary; its
three falsifiable questions, source accounting, and evidence gates are in
[`docs/kimi-k3-study.md`](docs/kimi-k3-study.md).

`phase=N` and `layer=N` have separate meanings. A phase is a complete-before-
next request dependency used by every scenario. A layer identifies an EC6
streaming window. Add nondecreasing values to every operation when both
contracts are known:

```text
0xADDR R 4096 model_weights phase=0 layer=0
0xADDR W 4096 generated_context phase=1 layer=1
```

In `layer_streaming` mode, EC6 streams each layer's nonresident pages from its
backing tier into HBM, executes every foreground access from HBM, overlaps the
next layer's transfer with the current layer when possible, and writes dirty
cold/overflow pages back at layer completion. The HBF route is HBF→D2D→HBM and
HBM→D2D→HBF; the external route includes the configured external-media and
host-link stages. A trace with no `layer=` metadata is deliberately treated as
one streaming window. Optional `compute_ns=N` on an explicitly labelled layer
supplies a nonnegative execution interval that overlaps next-layer prefetch;
repeated declarations in a layer must agree. Omitting it keeps the result
explicitly memory-only.

`--layer-buffer-bytes` is an upper bound for one buffer. Production runs use
the page-rounded buffer size in the explicit plan and fail if a trace-visible
layer exceeds it; generic microbenchmarks size dynamically.
`--max-outstanding-requests W` limits backing-page-sized physical
transactions. Direct compositions use one completion-order pool. Layer
streaming uses two independent same-sized pools: one for foreground HBM page
transactions and one shared by backing-tier DMA reads and writebacks. This
allows HBM and HBF/external media to overlap without giving the staged path
unbounded backing-tier concurrency. A large trace record or layer transfer is
split before admission and returns one credit per completed page transaction;
its user-visible latency is the maximum of its child completions. Parents
ready in the same phase receive one page transaction per round, while
overlapping read/write pages retain program order. `W=0` leaves both pools
open loop.
Read-only direct-composition studies may instead use
`--max-hbm-outstanding-requests` and `--max-hbf-outstanding-requests` as
independent completion-order pools; they remove false cross-tier head-of-line
blocking, allow one parent to span both tiers, and cannot be combined with the
shared window. Summary schema v16
places all timing under the canonical
`time_breakdown` contract: non-overlapping wall-clock spans, per-operation
latency work, overlapping device/controller work, and resource busy capacity
are reported separately. EC6 additionally reports its `hybrid-residency`
policy, backing kind, byte-exact pressure basis, page-allocated footprint and
rounding, physical-HBM capacity pressure, fixed/hot-KV/backing byte
partitions, effective buffer and unused-HBM
capacity, layer and explicit-compute counts, streamed and written-back
pages/bytes, backing credit limit, observed peak backing concurrency and
admission wait, exposed versus hidden prefetch time, user wait work, and
buffer-reuse wait work.

Self-contained quick start:

```bash
mkdir -p out/quickstart
./build/scenario_compare \
  --generate-semantic-llm out/quickstart/llm.trace \
  --llm-tokens 4 \
  --llm-layers 2

./build/scenario_compare \
  --trace out/quickstart/llm.trace \
  --max-ops 512 \
  --flat-hbm-bytes 1073741824 \
  --summary-json out/quickstart/summary.json

python3 tools/plot_address_heatmap.py \
  --input out/quickstart/summary.json \
  --scenario HBM+HBF-layer-streaming \
  --output out/quickstart/layer-streaming-heatmap.html
```

Run the identical layer policy without HBF by adding one external profile:

```bash
./build/scenario_compare \
  --config configs/scenario_compare/server-4k-hbf4x.cfg \
  --config configs/scenario_compare/external-cxl-memory.overlay \
  --trace out/quickstart/llm.trace \
  --scenarios HBM+External-layer-streaming \
  --summary-json out/quickstart/external-dram.summary.json
```

Replace the overlay with `nvme-ssd.overlay` for the SSD point. Both
profiles are exploratory starting points; every latency, bandwidth, topology,
queue-depth, capacity, and host-link value is independently configurable and
recorded in the resolved summary/config.

To isolate EC5/EC6 from ASTRA-sim, first generate a 48 MiB sequential 4 KiB
read stream:

```bash
mkdir -p out/ec56-simple
python3 tools/generate_synthetic_trace.py \
  --output out/ec56-simple/read.trace \
  --manifest out/ec56-simple/read.manifest.json \
  --pattern sequential --pages 12288 --passes 1 --bytes 4096 \
  --base 0x2000000000 --read-percent 100 --kind model_weights

./build/scenario_compare \
  --config configs/scenario_compare/usecase-2h6f.cfg \
  --trace out/ec56-simple/read.trace \
  --scenarios HBF-static-direct-read \
  --max-outstanding-requests 4096 \
  --summary-json out/ec56-simple/ec5.summary.json

./build/scenario_compare \
  --config configs/scenario_compare/usecase-6h2f.cfg \
  --trace out/ec56-simple/read.trace \
  --scenarios HBM+HBF-layer-streaming \
  --max-outstanding-requests 4096 \
  --summary-json out/ec56-simple/ec6-layer-streaming.summary.json
```

The EC5 run measures the direct physical HBF read fabric with FTL work absent.
The EC6 run must report `layer_streaming` with `hybrid-residency`. Read-only
weights and cold/overflow KV exercise HBF→D2D→HBM staging, while eligible hot
KV remains in HBM. Capacity pressure is the unique page footprint divided by
physical HBM capacity, not traffic bytes or the largest address. To measure
overlap on the 6HBM+2HBF profile, use an explicitly layered workload and ensure
each layer's backing footprint remains within `layer-buffer-bytes`. For dirty data,
compare both `user_completion_throughput_GBps` and
`makespan_throughput_GBps`: the latter includes D2D transfer, NAND program,
mapping persistence, and drain.

Every scenario in the v16 summary contains `time_breakdown` and
`address_heatmap`. The timing contract separates non-overlapping wall-clock
spans from latency work, stage work, and resource busy capacity; summed work
can overlap and must not be added to reconstruct the makespan. The heatmap uses
1024 bins per address domain by default. Set
`--address-heatmap-bins N` in the range 1..8192 when a different spatial
resolution is needed. A multi-scenario summary requires the exact
`--scenario` name when rendering; it may be omitted for a one-scenario
summary.

The four E2E runners (`run_synthetic_experiments.py`,
`replay_astra_trace.py`, `run_use_cases.py`, and `run_waf_cases.py`) now emit
CSV, Markdown, and self-contained HTML time-breakdown artifacts automatically.
Suite runners use `time-breakdown.{csv,md,html}`; ASTRA replay uses
`astra-replay*.time-breakdown.{csv,md,html}` so different selected case sets
cannot overwrite one another. The Markdown overview identifies the largest
accumulated primary stage and highest-utilized
resource for every scenario; the long-form CSV retains all zero-valued stages,
hierarchy roles, per-operation work, and resource capacity. Only rows marked
`additive_segment` form elapsed wall time. Schema v16 summaries can be rendered
without rerunning the simulator:

```bash
python3 tools/time_breakdown_report.py \
  --summary run=out/quickstart/summary.json \
  --csv out/quickstart/time-breakdown.csv \
  --markdown out/quickstart/time-breakdown.md

python3 tools/plot_time_breakdown.py \
  --summary run=out/quickstart/summary.json \
  --output out/quickstart/time-breakdown.html
```

The self-contained HTML separates additive wall clock; per-operation phase
dependency wait, front-end admission, and first-credit-to-completion service
latency; overlapping
device/controller work; detailed primary stage cost; and capacity-normalized
resource utilization. Dependency-bearing traces also report the auditable
identity `source-to-completion = phase wait + front-end admission +
service latency`. The primary online distribution is
`offered-to-completion`; a separate source distribution additionally includes
phase dependency wait. The renderer validates these schema-v16 timing identities
before replacing an existing visualization. Use repeated
`--summary [LABEL=]PATH` arguments for cross-scenario comparisons and
`--alias RUN/SCENARIO=DISPLAY` for concise case labels.

External-trace template (replace `WORKLOAD.trace` with a real file):

```bash
./build/scenario_compare \
  --config configs/scenario_compare/server-4k-hbf4x.cfg \
  --trace WORKLOAD.trace \
  --max-ops 131072 \
  --summary-csv out/mixed-rw/summary-50r50w-131k-4k.csv \
  --summary-json out/mixed-rw/summary-50r50w-131k-4k.json \
  --config-out out/mixed-rw/summary-50r50w-131k-4k.cfg
```

For prefix or layer-scaling experiments, keep the physical initial image
constant by passing the untruncated population trace separately:

```bash
./build/scenario_compare \
  --config configs/scenario_compare/server-4k-hbf8x.cfg \
  --trace WORKLOAD.trace \
  --initial-image-trace WORKLOAD.trace \
  --max-ops PREFIX_OPS \
  --scenarios all-HBF
```

`--initial-image-trace` is population-only: it is never executed or counted as
logical traffic. The summary records its path, byte count, and SHA-256 digest.
The simulator fails closed if that image does not contain every initial
read-before-write page required by the executed prefix.

For controlled locality and capacity experiments, generate the workload
directly instead of relying on an application frontend:

```bash
python3 tools/generate_synthetic_trace.py \
  --output out/synthetic/random.trace \
  --pattern random-permutation --pages 16000 --passes 2 \
  --bytes 4096 --address-span 0x400000000 --read-percent 70 \
  --layer-per-pass

python3 tools/analyze_trace_locality.py \
  --trace out/synthetic/random.trace \
  --output out/synthetic/random.locality.json

python3 tools/plot_trace_heatmap.py \
  --trace out/synthetic/random.trace --capacity-bytes 0x40000000000 \
  --output out/synthetic/random.logical.html

python3 tools/run_synthetic_experiments.py \
  --profile smoke --out-dir out/synthetic-smoke

# Sparse but uniform native-capacity coverage only:
python3 tools/run_synthetic_experiments.py \
  --profile smoke --groups coverage \
  --out-dir out/synthetic-capacity-coverage

# Local-output and array-supply output comparison only:
python3 tools/run_synthetic_experiments.py \
  --profile smoke --groups output \
  --out-dir out/synthetic-output-comparison
```

The generator separates working-set size from address span and publishes a
digest-bound manifest. The standalone logical heatmap shows both full-capacity
and observed-working-set views. The matrix runner deliberately uses a compact
orthogonal plan rather than a full parameter Cartesian product; use
`--profile core` for larger cases: the general capacity-coverage point is
64 MiB, while the topology-balanced output point is 256 MiB. Its packed vs.
evenly-spread and read-only vs. mixed-R/W pairs hold the access order fixed so
the changed variable remains attributable. EC6 layer behavior is covered by
the dedicated layer-streaming regression rather than a cache-capacity group.
Treat
`synthetic-manifest.json` as the success record for the digest-bound CSV,
traces, reports, heatmaps, and simulator summaries.

The default matrix also includes the `output` group. It generates two
byte-identical, read-only, stratified-random traces spanning the full 4 TiB
HBF capacity and changes only the outstanding window: 512 versus 16,384 for
smoke, and 512 versus 32,768 for core. Smoke contains 16,384 unique pages
(64 MiB), one for every modeled
`(stack, channel, die, plane, subarray)` leaf; core contains 65,536 pages
(256 MiB), four per leaf. Each page is accessed once in a seeded random
permutation, and the runner compares the two window traces' SHA-256 digests
before simulation. Both use `--static-direct-hbm-bytes 0`, so
`HBF-static-direct-read` sends every request through the physical HBF read
path without FTL translation, mapping-page traffic, programs, or GC. The same
trace runs on `usecase-baseline.cfg`,
`usecase-baseline-local-output.cfg`, and
`usecase-baseline-output-upper-bound.cfg`. The local-output profile changes
the 32 subarrays per plane from 16 shared lane/page-buffer outputs to 32; the
upper-bound profile retains those 32 local outputs and raises the existing
flash-TSU, logic-dispatch, channel, TSV, ECC, SRAM, and HBIO rates to the
1 microsecond
array-supply bound. TSV continues to be one shared per-stack
command-and-raw-codeword path. This is a cost-unbounded sensitivity point, not
a claim that every subarray has an independent external interface. The runner
rejects a result unless the trace
covers all 256 logical and HBF-physical bins, all user bytes use direct HBF,
physical read bytes equal logical bytes, and the mapping/program/GC
counters remain zero. Static placement uses the same page-striped stack owner
and per-stack plane order as the managed FTL, so it validates balanced physical
occupancy and exact read-count conservation without changing the data layout
between EC4 and EC5.

The comparison CSV includes the placement modulus, target config, flash-TSU
issue intervals, and local-output/fabric parameters so the three design points
remain distinguishable. The runner binds the flash-TSU interval to 10 ns; this
stays below the baseline interface demand. The cost-unbounded upper-bound
profile uses 7.8125 ns, the per-die interval needed to supply four planes with
32 subarrays each at one 4 KiB page per subarray per microsecond.

Output placement uses a logical leaf modulus of 16,384
(`8 stacks × 4 channels × 4 dies × 4 planes × 32 subarrays`). Every group of
16,384 address strata receives all low-bit residues in a seeded random
permutation, while an independent seed randomizes access order. The simulator
then applies its documented page-striped, stack-local placement; mapping-group
rotation makes high address bits participate in stack ownership while the
per-stack page sequence covers the physical leaves without aliasing. The
generator records the logical modulus in its
digest-bound manifest, and the result validates the physical occupancy from
device statistics.

The `coverage` group uses `stratified-random` placement: every equal-sized
capacity stratum contributes one randomly jittered page, while a separate seed
randomizes access order. Its 384 GiB case runs the identical trace on all-HBM
and all-HBF; its 4 TiB case is HBF-only because those addresses exceed the HBM
device. Each standalone full-capacity heatmap uses the native case capacity,
and the runner rejects a result unless all logical and target physical bins are
covered. This provides uniform sparse coverage without the periodic mapping
alias of a fixed large stride. The smoke profile uses 4,096 unique pages
(16 MiB, 16 pages in each of 256 bins); core uses 16,384 pages (64 MiB,
64 pages per bin). Both remain sparse relative to the modeled capacities.

`synthetic-manifest.json` uses suite schema v5, and generator manifests use
schema v3. Suite v5 adds the foundational validation block; without an
explicit certificate both it and every simulator summary are marked
`exploratory_unattached`. In addition to the digest-bound artifacts, each case
records its placement modulus and exact static-direct HBM boundary; this
prevents an output experiment from silently losing topology balance or routing
low addresses back to HBM.

Hardware, timing, flash geometry, FTL policy, GC policy, and staging policy
belong in config files under `configs/scenario_compare/`. Keep runtime choices
such as input trace paths, output artifact paths, and temporary `--max-ops`
limits on the command line.

See the [documentation index](docs/README.md) for the maintained model,
validation, workload, experiment, and integration references. The production
Qwen-Bailian + Frontier workload boundary, strict KV-lifecycle audit,
deterministic memory-object exporter, and large-trace census workflow are in
[`docs/real-workloads.md`](docs/real-workloads.md). Run
`cmake --build build --target use_cases` for the
component verification table and `cmake --build build --target waf_cases` for
the focused FTL/GC accounting suite.

## Trust Boundary

HBFSim currently models device timing and resource contention; it does not
model a GPU execution pipeline, thermal throttling, retention, read disturb,
raw bit-error rate, ECC failure probability, bad-block growth, or lifetime
endurance. Paper-aligned ASTRA-sim 3.0 traces can be replayed only with a
digest-bound workload manifest. The sibling frontend is not vendored here, so
its exact commit remains part of every experiment's provenance. See
[`docs/model-validation.md`](docs/model-validation.md) before interpreting a
number as more than a result of the selected model and parameters.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the intended review process and
[`SECURITY.md`](SECURITY.md) for private vulnerability reporting.

## License

HBFSim is released under the [MIT License](LICENSE).
