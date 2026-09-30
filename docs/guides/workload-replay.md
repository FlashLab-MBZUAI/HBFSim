# Workload replay guide

> Status: Current
> Last reviewed: 2026-09-29

This runbook is for running external or complex workloads through HBFSim on a
server. It focuses on the HBM/HBF comparison policies that matter for external
traces:

```text
flat
direct-read
hbf-streaming
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
cmake --build build --target verify_physical
cmake --build build --target verify_components
```

If any gate fails, do not trust scenario numbers.

## 2. Normalize The Workload Trace

`hbfsim-reference` accepts Ramulator-compatible text:

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

Use semantic labels when evaluating `hbf-streaming`:

```text
model_weights      -> HBF-backed hbf-streaming data page staged through a layer buffer
shared_context     -> HBF-backed hbf-streaming data page staged through a layer buffer
generated_context  -> HBF-backed dirty buffer page with HBF writeback
scratch            -> foreground HBM
metadata           -> foreground HBM
unknown            -> HBF-backed hbf-streaming data page staged through a layer buffer
```

These labels, and the `--generate-semantic-llm` generator that emits them,
describe a smoke-test address pattern for the reference policies (fixed-size
weight, KV, and scratch lines with a 32-token attention history), not a model
workload. An LLM serving workload — a catalog model ledger, a request stream,
a paged-KV scheduler, object-class placement, and roofline compute — runs
through the standalone ServeLoop frontend; see the [ServeLoop guide](serveloop.md).

`direct-read` uses the same static HBF placement for
`model_weights`, `shared_context`, and unlabeled reads, but it does not stage
through HBM. Mutable semantic traffic stays in HBM. Use this row to inspect the
raw HBF read fabric without page-program and mapping-program noise.

For HBF page-aligned experiments, combine a server-scale physical file under
`configs/systems/` with its optional file under
`configs/policies/reference/`. The default pair uses:

```text
line-size=4096
hbf-page-size=4096
hbf-media-lanes-per-plane=16
hbf-speed-grade=2
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
HBF speed grade 2; one sense resource per Bank
HBF page-buffer banks 16 per plane
```

Use this config:

```text
configs/systems/server-hbm128-hbf512.cfg
configs/policies/reference/server-hbm128-hbf512.cfg
```

For HBF 8x, use:

```text
configs/systems/server-hbm128-hbf1024.cfg
configs/policies/reference/server-hbm128-hbf1024.cfg
```

## 4. flat Address Mapping

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
src/policies/reference/direct_policy.hpp
src/policies/reference/direct_policy.cpp
src/app/reference_runner.cpp  policy selection and reporting
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
flat mode will silently become all-hbm. The tool will warn:

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

## 5. Serving Workloads

A trace fixes the access stream in advance. For workloads whose accesses
depend on completion times, such as continuous-batching LLM serving with paged
KV caches, drive the engine from [ServeLoop](serveloop.md) instead; it submits
transactions batch by batch and reads the engine's completions back.

## 6. Choosing Layer-Buffer Size

`layer-buffer-bytes` is the upper bound for one buffer. In production
`layer_streaming` mode, `hbf-streaming` reserves the page-rounded explicit-plan buffer and
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

Use the same trace for all scenarios. `hbfsim-reference` reports the full policy
set in one run.

### Flat plus layer-streaming policy set

```bash
mkdir -p out/server
./build/hbfsim-reference \
  --config configs/systems/server-hbm128-hbf512.cfg \
  --config configs/policies/reference/server-hbm128-hbf512.cfg \
  --trace WORKLOAD.trace \
  --scenarios flat,direct-read,hbf-streaming \
  --summary-csv out/server/summary-policy-set.csv \
  --summary-json out/server/summary-policy-set.json \
  --config-out out/server/summary-policy-set.cfg \
  > out/server/summary-policy-set.stdout
```

### Larger layer buffers

Use the larger layer-buffer config:

```bash
mkdir -p out/server
./build/hbfsim-reference \
  --config configs/systems/server-hbm128-hbf512-large-buffer.cfg \
  --config configs/policies/reference/server-hbm128-hbf512-large-buffer.cfg \
  --trace WORKLOAD.trace \
  --scenarios flat,direct-read,hbf-streaming \
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

Keep the schema-v19 summary JSON, not only the CSV. Every scenario contains
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

The embedded `hbfsim.address_heatmap.v1` has five address domains:
workload-logical, HBM-physical, HBF-logical, HBF-physical, and
external-physical. Its fixed traffic sources include `workload`, `direct`,
`mapping`, `prepopulate`,
`cooperative_buffer`, `demand_fill`, `prefetch_fill`,
`streaming_install`, `destage`, `garbage_collection`, and
`maintenance`. In `layer_streaming` mode, the two `hbf-streaming` active regions
are labelled `layer_buffer_0` and `layer_buffer_1`.

Render `hbf-streaming` from a multi-scenario summary:

```bash
python3 reports/address_heatmap.py \
  --input out/server/summary.json \
  --scenario hbf-streaming \
  --output out/server/layer-streaming-heatmap.html
```

Select the object in `scenarios[]` whose `name` is `hbf-streaming`. In JSON,
inspect these namespaces first:

```text
layer_streaming.mode
layer_streaming.residency_policy
layer_streaming.explicit_residency_contract
layer_streaming.capacity_pressure_basis_bytes
layer_streaming.footprint_page_rounding_bytes
layer_streaming.hbm_capacity_pressure
layer_streaming.hbm_only_resident_pages
layer_streaming.hot_kv_resident_pages
layer_streaming.model_weight_resident_pages
layer_streaming.backing_unique_pages
layer_streaming.effective_layer_buffer_bytes
layer_streaming.streamed_bytes
layer_streaming.writeback_bytes
layer_streaming.exposed_prefetch_ns
layer_streaming.hidden_prefetch_ns
layer_streaming.buffer_reuse_wait_work_ns
hbm_user_accesses
hbf_user_accesses
hybrid_path.hbf_static_read_bytes
hybrid_path.hbf_logical_read_bytes
hybrid_path.hbf_backing_write_bytes
hybrid_path.hbm_streaming_write_bytes
hybrid_path.base_die_link_read_bytes
hybrid_path.base_die_link_write_bytes
warnings
```

The summary CSV flattens `layer_streaming.foo` to
`layer_streaming_foo` and `hybrid_path.foo` to `foo`. Its `warnings` column is
a count; JSON keeps the warning strings as an array. The complete field and
conservation contract is in the [scenario reference](../reference/scenarios.md).

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
warnings == []
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

Flat accidentally becomes all-hbm:

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
capacity, accounting, or a suspicious physical metric. `SANITY:
PASS_WITH_WARNINGS` (exit 0) is different: the numbers conserve, but a
selected scenario's configuration did not exercise what it exists for (for
example `flat` with `--flat-hbm-bytes` above the whole footprint, so nothing
reached HBF); the stderr `warning:` lines name the scenario. An open-loop
saturation warning (offered rate at or above 90% of a tier's peak) never
changes the verdict but means the latency percentiles describe the arrival
model rather than the devices.
