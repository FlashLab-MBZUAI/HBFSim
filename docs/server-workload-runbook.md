# Server Workload Runbook

This runbook is for running external or complex workloads through HBFSim on a
server. It focuses on the HBM/HBF comparison policies that matter for external
traces:

```text
HBM-HBF-Flat
HBF-static-direct-read
HBM+HBF-layer-streaming
```

The simulator does not execute CPU/GPU instructions. It consumes a normalized
memory-request trace and routes each request according to the selected scenario
policy.

## 1. Build And Sanity Guard

From the repository root:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
cmake --build build --target physical_guard
cmake --build build --target use_cases
```

If any gate fails, do not trust scenario numbers.

## 2. Normalize The Workload Trace

Current `scenario_compare` accepts Ramulator-compatible text:

```text
0xADDR R
0xADDR W
```

The operation token may be first or second, but keep the above style for
readability. Comments beginning with `#` are allowed.

It also accepts optional request size and semantic labels:

```text
0xADDR R 4096 model_weights
0xADDR R bytes=4096 kind=shared_context
0xADDR W 4096 generated_context
0xADDR W bytes=4096 kind=scratch
```

If no per-request size is present, `--line-size N` is used as the request size.
If no semantic label is present, the request is treated as `unknown`.

Use semantic labels when evaluating the paper-style HBM staging case:

```text
model_weights      -> HBF-backed EC6 data page staged through a layer buffer
shared_context     -> HBF-backed EC6 data page staged through a layer buffer
generated_context  -> HBF-backed dirty buffer page with HBF writeback
scratch            -> foreground HBM
metadata           -> foreground HBM
unknown            -> HBF-backed EC6 data page staged through a layer buffer
```

`HBF-static-direct-read` uses the same static HBF placement for
`model_weights`, `shared_context`, and unlabeled reads, but it does not stage
through HBM. Mutable semantic traffic stays in HBM. Use this row to inspect the
raw HBF read fabric without page-program and mapping-program noise.

For HBF page-aligned experiments, use the server-scale config files under
`configs/scenario_compare/`. The default policy-set config uses:

```text
line-size=4096
hbf-page-size=4096
hbf-media-lanes-per-plane=16
hbf-subarrays-per-plane=16
hbf-page-buffer-banks-per-plane=16
```

For sub-page stress experiments, use a smaller `line-size`. Expect
`waf` to increase because HBF still programs full SLC pages. Diagnose the
cause with the raw `data_programs`, `mapping_page_programs`, and
`gc_relocations` counters; do not introduce another WAF denominator.

## 3. Choose Capacity And Granularity

Useful first exploratory server profile:

```text
HBM capacity    128 GiB
HBF capacity    512 GiB
HBF/HBM ratio   4x
HBF page size   4 KiB
HBF read lanes  16 per plane
HBF subarrays   16 per plane
HBF page-buffer banks 16 per plane
```

Use this config:

```text
configs/scenario_compare/server-4k-hbf4x.cfg
```

For HBF 8x, use:

```text
configs/scenario_compare/server-4k-hbf8x.cfg
```

## 4. HBM-HBF-Flat Address Mapping

Flat mode is a static address split:

```text
addr <  flat_hbm_bytes  -> HBM
addr >= flat_hbm_bytes  -> HBF
```

`flat_hbm_bytes` must be HBF-page aligned. Requests that straddle an aligned
boundary are split across both devices for reads and writes; HBFSim reports one
parent operation whose completion is the slower fragment.

If `hbf-hbm-write-buffer-bytes` is nonzero, direct compositions reserve that
many bytes at the top of HBM for cooperative HBF-bound writes. Both
`flat-hbm-bytes` and the static-direct read boundary must be at or below
`hbm_capacity_bytes - hbf_hbm_write_buffer_bytes`. The region uses explicit
HBF-page-sized physical slots keyed by resident logical page; same-page
sub-page updates merge without consuming another slot. The configured region
must be a nonzero whole-page multiple, and ordinary HBM traffic cannot overlap
it.

Implementation:

```text
src/physical/hybrid/direct_composition.hpp
src/physical/hybrid/direct_composition.cpp
src/tools/scenario_compare.cpp  policy selection and reporting
```

This means the workload address map directly controls which medium is used. To
force regions into different media, remap the workload trace before running:

```text
HBM-resident region: addresses below flat_hbm_bytes
HBF-resident region: addresses at or above flat_hbm_bytes
```

Example with a 512 MiB HBM flat window:

```bash
--flat-hbm-bytes 536870912
```

Then:

```text
0x00000000 .. 0x1fffffff  -> HBM
0x20000000 and above      -> HBF
```

If your workload has semantic regions, map them intentionally:

```text
weights / hot metadata / latency-critical pages -> low addresses, HBM
KV cache / cold tensors / large backing memory  -> high addresses, HBF
```

Do not choose `flat_hbm_bytes` larger than the workload's whole footprint, or
flat mode will silently become all-HBM. The tool will warn:

```text
flat policy did not exercise both HBM and HBF
```

Check these output columns:

```text
hbm_accesses
hbf_accesses
warnings
```

A useful flat run must have both `hbm_accesses > 0` and `hbf_accesses > 0`.

## 5. Canonical Frontier Memory Baselines

EC6 is a hybrid-residency layer-streaming design. For Frontier workloads, do
not run the media cases separately. Run the canonical four-baseline
transaction:

```bash
python3 tools/run_frontier_memory_baselines.py \
  --scenario-compare build/scenario_compare \
  --hardware-config configs/scenario_compare/usecase-2h6f.cfg \
  --manifest out/frontier/memory.manifest.json \
  --object-map out/frontier/objects.json \
  --output-dir out/frontier/memory-baselines
```

The runner creates the digest-bound residency overlay internally and executes
`all_hbm_upper_bound`, `hbm_hbf`, `hbm_cxl_memory`, and `hbm_nvme_ssd`.
Changing the trace or using `max-ops` fails before simulation. The three
backed cases must have identical placement, scheduler credit, HBM geometry,
HBM traffic, and backing bytes. The all-HBM case is explicitly a fully
resident upper bound and records whether its capacity had to be provisioned
above the physical topology. CXL-memory and NVMe-SSD timings are sensitivity
parameters, not calibrated product claims.

The command above is the atomic single-trace mechanism entry point. Do not
assemble paper tables by calling it repeatedly from ad hoc shell loops. The
former 72-cell full-replay matrix was never operationally executable at exact
HBM-burst/HBF-page fidelity and has been deleted together with its contract.
The current Llama 3.1 70B mix-ratio and capacity-boundary study has the one
immutable contract
`configs/studies/frontier-70b-two-axis-burst.json`. Export the mechanically
selected q=0, q=0.5, or q=1 batch from each of the three complete 4H4F
Frontier replays, then run:

```bash
python3 tools/run_frontier_70b_two_axis_slice.py \
  --scenario-compare build/scenario_compare \
  --p0-75-manifest P0_75_MEMORY_MANIFEST \
  --p0-75-object-map P0_75_OBJECT_MAP \
  --p1-0-manifest P1_0_MEMORY_MANIFEST \
  --p1-0-object-map P1_0_OBJECT_MAP \
  --p1-25-manifest P1_25_MEMORY_MANIFEST \
  --p1-25-object-map P1_25_OBJECT_MAP \
  --quantile 0.5 \
  --output-dir TWO_AXIS_Q0_5_OUTPUT
```

The runner creates exactly five non-duplicate cells and 20 baseline runs. It
reuses the p1.0 manifest/object map/trace verbatim on 6H2F, 4H4F, and 2H6F;
capacity pressure is therefore observed, not targeted, on the mix-ratio axis.
It separately holds 4H4F fixed at target pressures 0.75, 1.0, and 1.25. The
runner rejects a dirty source tree, a stale executable, a non-mechanical batch
selection, request-stream drift, transitive artifact mutation, pure all-HBF,
or any failure to preserve the common logical population. Each quantile
remains a separately reported temporal stratum and is not averaged into a
full-replay claim. The five independent cells run in parallel; each cell runs
at most two independent baseline processes concurrently and publishes only
after its four-baseline transaction passes atomically.

After all three invocations finish, certify exactly q=0/q=0.5/q=1:

```bash
python3 tools/verify_frontier_70b_two_axis_evidence.py \
  --q0-receipt TWO_AXIS_Q0/slice.receipt.json \
  --q0-5-receipt TWO_AXIS_Q0_5/slice.receipt.json \
  --q1-receipt TWO_AXIS_Q1/slice.receipt.json \
  --output-dir TWO_AXIS_EVIDENCE
```

The verifier reopens every transitive input, cell receipt, baseline metric,
table row, and WAF field, then emits a 60-row stratified concatenation. It
rejects missing or duplicate quantiles, non-mechanical batch rows, claim
promotion, cross-stratum source drift, and zero-write WAF encoded as zero.
It deliberately forbids cross-quantile averaging and keeps
`paper_result_eligible=false`; these are three exact single-batch memory
strata, not a full application replay.

The complete exported population, rather than trace-visible pages, fixes
capacity. Runtime overhead and block tables are resident, the largest
topology-fitting prefix of low-ID KV blocks is hot, immutable weights and cold
KV use backing, and
unknown/out-of-arena accesses fail closed. Add nondecreasing layer metadata to
every request when the workload has real layer boundaries:

```text
0xADDR R 4096 model_weights phase=0 layer=0
0xADDR W 4096 generated_context phase=1 layer=1
```

`phase=N` is not a layer alias. It is the complete-before-next dependency used
by all scenarios; `layer=N` identifies EC6's streaming window. Each metadata
kind must be present on every request or none, and both identifiers must be
nondecreasing. A plain R/W trace with no layer metadata is one streaming
window; it is useful for raw transfer throughput but cannot demonstrate
next-layer overlap.

For each layer, the paths are:

```text
clean input:
  resident immutable weight: compact HBM resident slot
  backed immutable weight: pre-resolved static HBF read
  mutable cold KV: logical HBF read from its initial version onward
    -> per-stack D2D read -> HBM buffer write
foreground:
  fixed/weight/hot-KV resident slot or HBM layer-buffer read/write
dirty output:
  dirty data: HBM full-page read -> per-stack D2D write
    -> logical HBF write
```

A full-page foreground write skips the initial HBF read because it overwrites
the complete page. A partial write performs read-modify-write. `scratch` and
`metadata` bypass the layer buffers and are compacted into the foreground HBM
region, so a high logical object address is not treated as an HBM physical
address.

When layer N starts execution, layer N+1 begins prefetch into the other buffer.
A parity buffer cannot be reused until the earlier layer using it has completed
dirty writeback. Consequently, `hidden_prefetch_ns` measures useful overlap,
`exposed_prefetch_ns` measures prefetch on the critical path, and
`buffer_reuse_wait_work_ns` exposes writeback-delayed buffer reuse.

An explicitly labelled layer may also carry `compute_ns=N`. Repeated
declarations in the same layer must agree. This nonnegative interval overlaps
the next layer's prefetch and extends the execution frontier. If it is absent,
the run is intentionally memory-only; EC6 does not promote uncalibrated
Frontier or ASTRA timing into a fabricated compute duration.

Implementation:

```text
src/physical/hybrid/layer_streaming_composition.hpp
src/physical/hybrid/layer_streaming_composition.cpp
src/tools/scenario_compare.cpp
```

The former demand-cache controller is not present.

## 6. Choosing Layer-Buffer Size

`layer-buffer-bytes` is the upper bound for one buffer. In production
`layer_streaming` mode, EC6 reserves the page-rounded explicit-plan buffer and
requires the trace-visible maximum to fit:

```text
runtime_buffer_bytes = effective_layer_buffer_pages * hbf_page_size
runtime_hbm_bytes =
    resident_physical_bytes + 2 * runtime_buffer_bytes
runtime_hbm_bytes <= hbm_capacity_bytes
runtime_buffer_bytes <= layer_buffer_bytes
max_layer_data_pages <= effective_layer_buffer_pages
```

The configured limit must be HBF-page aligned. The planner fails closed if no
two-buffer layout satisfies both limits. Logical scratch/metadata addresses
are compacted and need not sit below a physical buffer base. Example upper
bounds:

```text
two 32 MiB buffers   -> layer-buffer-bytes=33554432
two 256 MiB buffers  -> layer-buffer-bytes=268435456
two 32 GiB buffers   -> layer-buffer-bytes=34359738368
```

D2D settings are per HBF stack:

```text
base-die-link-read-bw=2048
base-die-link-write-bw=512
base-die-link-latency-ns=0
```

These are architectural sensitivity inputs. In particular, the paper supplies
the bandwidth assumption but does not establish a universal fixed D2D latency.

## 7. Recommended Commands

Use the same trace for all scenarios. `scenario_compare` reports the full policy
set in one run.

### Flat plus layer-streaming policy set

```bash
./build/scenario_compare \
  --config configs/scenario_compare/server-4k-hbf4x.cfg \
  --trace WORKLOAD.trace \
  --summary-csv out/server/summary-policy-set.csv \
  --summary-json out/server/summary-policy-set.json \
  --config-out out/server/summary-policy-set.cfg \
  > out/server/summary-policy-set.stdout
```

### Larger layer buffers

Use the larger layer-buffer config:

```bash
./build/scenario_compare \
  --config configs/scenario_compare/server-4k-hbf4x-large-layer-buffer.cfg \
  --trace WORKLOAD.trace \
  --summary-csv out/server/summary-policy-set-large-layer-buffer.csv \
  --summary-json out/server/summary-policy-set-large-layer-buffer.json \
  --config-out out/server/summary-policy-set-large-layer-buffer.cfg \
  > out/server/summary-policy-set-large-layer-buffer.stdout
```

For two 32 GiB buffers:

```text
layer-buffer-bytes=34359738368
```

Together they consume 64 GiB of HBM. Fixed and hot-KV residents are compacted
below the resulting streaming-region base address; the summary must show their
bytes plus both buffers plus unused HBM equal physical HBM capacity.

## 8. What To Inspect First

Keep the schema-v16 summary JSON, not only the CSV. Every scenario contains
`time_breakdown` and `address_heatmap`, independent of Chrome-trace mode.

Audit timing from elapsed time toward causes:

```text
time_breakdown.wall_clock_ns
  offered_arrival_span_ns
  post_offer_user_completion_tail_ns
  user_completion_span_ns
  drain_tail_ns
  makespan_ns
time_breakdown.latency_work
time_breakdown.stage_work
  hbm / hbf / base_die_link
  layer_streaming_controller / cooperative_write_controller
time_breakdown.resource_busy
```

The E2E runners write `time-breakdown.html`, `time-breakdown.md`, and
`time-breakdown.csv`. Only `role=additive_segment` rows are elapsed-time
parts. Latency, device work, and controller work can overlap; do not add them to
reconstruct makespan. A large post-offer tail indicates user-visible backlog,
while a large drain tail indicates deferred writeback or maintenance.

The embedded `hbfsim.address_heatmap.v1` has workload-logical, HBM-physical,
HBF-logical, and HBF-physical domains. Its fixed traffic sources include
`workload`, `direct`, `mapping`, `prepopulate`,
`cooperative_buffer`, `demand_fill`, `prefetch_fill`,
`streaming_install`, `destage`, `garbage_collection`, and
`maintenance`. In `layer_streaming` mode EC6's two active regions are labelled
`layer_buffer_0` and `layer_buffer_1`.

Render EC6 from a multi-scenario summary:

```bash
python3 tools/plot_address_heatmap.py \
  --input out/server/summary.json \
  --scenario HBM+HBF-layer-streaming \
  --output out/server/layer-streaming-heatmap.html
```

For EC6, inspect:

```text
layer_streaming_mode
layer_streaming_residency_policy
layer_streaming_compact_resident_mapping
layer_streaming_explicit_residency_contract
layer_streaming_unique_resident_footprint_pages
layer_streaming_unique_resident_footprint_bytes
layer_streaming_capacity_pressure_basis_bytes
layer_streaming_footprint_page_rounding_bytes
layer_streaming_hbm_capacity_pressure
layer_streaming_layers
layer_streaming_explicit_layer_requests
layer_streaming_hbm_only_resident_pages
layer_streaming_hot_kv_candidate_pages
layer_streaming_hot_kv_resident_pages
layer_streaming_data_pages
layer_streaming_model_weight_resident_pages
layer_streaming_model_weight_backing_pages
layer_streaming_cold_kv_backing_pages
layer_streaming_unknown_backing_pages
layer_streaming_backing_unique_pages
layer_streaming_resident_physical_pages
layer_streaming_effective_layer_buffer_pages
layer_streaming_effective_layer_buffer_bytes
layer_streaming_unused_hbm_bytes
layer_streaming_streamed_pages
layer_streaming_streamed_bytes
layer_streaming_foreground_resident_page_accesses
layer_streaming_foreground_buffer_page_accesses
layer_streaming_dirty_pages_written_back
layer_streaming_writeback_bytes
layer_streaming_max_layer_data_bytes
layer_streaming_immutable_weight_logical_bytes
layer_streaming_runtime_overhead_logical_bytes
layer_streaming_block_table_logical_bytes
layer_streaming_active_buffer_logical_bytes_per_slot
layer_streaming_residency_page_size_bytes
layer_streaming_kv_block_stride_bytes
layer_streaming_logical_kv_blocks
layer_streaming_hot_kv_blocks
layer_streaming_cold_kv_blocks
layer_streaming_user_waited_ops
layer_streaming_user_wait_work_ns
layer_streaming_user_max_wait_ns
layer_streaming_exposed_prefetch_ns
layer_streaming_hidden_prefetch_ns
layer_streaming_buffer_reuse_wait_work_ns
hbm_user_accesses
hbf_user_accesses
hbm_background_accesses
hbf_background_accesses
hbf_static_read_bytes
hbf_logical_read_bytes
hbf_backing_write_bytes
hbm_streaming_write_bytes
base_die_link_read_bytes
base_die_link_write_bytes
```

Require these invariants before interpreting performance:

```text
hbm_user_accesses ==
    foreground_resident_page_accesses + foreground_buffer_page_accesses
hbf_user_accesses == 0
streamed_bytes == streamed_pages * hbf_page_size
streamed_bytes == hbf_static_read_bytes + hbf_logical_read_bytes
streamed_bytes == base_die_link_read_bytes
streamed_bytes == hbm_streaming_write_bytes
writeback_bytes == dirty_pages_written_back * hbf_page_size
writeback_bytes == base_die_link_write_bytes
writeback_bytes == hbf_backing_write_bytes
unique_resident_footprint_pages == hbm_only_resident_pages + data_pages
capacity_pressure_basis_bytes + footprint_page_rounding_bytes ==
    unique_resident_footprint_bytes
data_pages == model_weight_resident_pages
    + hot_kv_resident_pages + backing_unique_pages
resident_physical_pages ==
    hbm_only_resident_pages + model_weight_resident_pages
    + hot_kv_resident_pages
max_layer_data_pages <= effective_layer_buffer_pages
max_layer_data_bytes <= layer_buffer_bytes
resident_physical_bytes + 2 * effective_layer_buffer_bytes + unused_hbm_bytes
    == hbm_capacity_bytes
warnings == 0
```

Then compare HBF array work, D2D utilization, HBM data-bus utilization, exposed
prefetch, and hidden prefetch. This distinguishes media latency, transfer
bandwidth, HBM installation/foreground pressure, and insufficient overlap.

## 9. Optional Chrome Trace

Only enable Chrome trace for small samples. Full traces for complex workloads
can become huge.

```bash
--max-ops 1024
--chrome-trace out/server/sample.trace.json
--trace-mode full
```

Open in Chrome:

```text
chrome://tracing
```

Look for request ids such as:

```text
flat/hbm/op123
flat/hbf/op124
layer0/page456/hbf-read
layer0/page456/d2d-read
layer0/page456/hbm-install
op126/buffer-page456
layer0/page456/writeback-hbm-read
layer0/page456/d2d-write
layer0/page456/hbf-writeback
hbf-static-direct/op125/page456
```

These names are the current route audit.

## 10. Common Failure Modes

Flat accidentally becomes all-HBM:

```text
flat_hbm_bytes is larger than the workload footprint
```

Fix: lower `--flat-hbm-bytes` or remap HBF-target regions to higher addresses.

Layer streaming looks worse than flat:

```text
the first layer is exposed, overlap is insufficient, or writeback delays reuse
```

Inspect `exposed_prefetch_ns`, `hidden_prefetch_ns`,
`buffer_reuse_wait_work_ns`, HBF array work, D2D utilization, HBM bus
utilization, and writeback bytes. A plain trace is one layer and therefore has
no next-layer overlap; add real `layer=N` boundaries before evaluating the
paper's overlap idea.

HBF WAF is huge:

```text
request size is much smaller than HBF page size
```

Fix: use page-aligned requests for page-level studies, or treat the run as a
sub-page stress test. Compare raw data-program, mapping-program, and
GC-relocation bytes before attributing the WAF to one mechanism.

Layer labels are missing:

```text
layer streaming treats the complete trace as one streaming window
```

Fix: attach nondecreasing `layer=N` to every request when measuring ping-pong
overlap. Semantic kinds still determine whether a request is HBF-backed or
HBM-only.

Scenario returns `SANITY: FAIL`:

```text
do not use the numbers until the warning is understood
```

The warning text is usually enough to identify whether the problem is routing,
capacity, accounting, or a suspicious physical metric.
