# Reference policy suite

> Status: Current
> Last reviewed: 2026-09-29

`hbfsim-reference` compares this maintained policy set over the same memory
trace. These identifiers belong to the reference runner; the `hbfsim` core
accepts arbitrary transaction DAGs and has no fixed scenario list.

## Policies

`all-hbm`

Every request is served by HBM. This is the fast upper-bound baseline.

`all-hbf`

Every request is served by HBF. This isolates flash-as-memory timing,
translation, write buffer, mapping, TSU, and maintenance behavior. Treat this as
the mutable HBF worst case: writes become real page programs, mapping updates
can become mapping page programs, and complete program intervals can dominate finish time.

`flat`

The address space is statically split. Addresses below `flat-hbm-bytes` go to
HBM; the rest go to HBF. HBF data pages are striped page-by-page across stacks;
each stack has its own mapping pages over its local page sequence. Consequently
a sequential W512 stream can use all stacks rather than assigning all 512
requests covered by one mapping page to a single stack. With
`hbf-hbm-write-buffer-bytes=0`, HBM-side writes
need no HBF writeback. When the cooperative region is enabled, HBF-bound
writes are split by HBF page and first update fixed page-resident slots in the
upper HBM region. Same-page updates merge byte-exact dirty ranges, and unaligned
or cross-page writes destage those ranges through the owning per-stack links;
the address boundary must stop below the reserved region.
A resident generation is destaged by one of three triggers, in this order of
precedence: the background high watermark
`hbf-hbm-write-buffer-destage-watermark` (a fraction of the region, default
0.5; once the full-page slots whose destage has not started exceed it, the
oldest such slots are destaged, generation order, until the pending footprint
fits again), finite-slot pressure (a write that finds no free slot waits for
the oldest generation), and the end-of-run drain after the final user
completion. A chain scheduled by the watermark starts only when the
foreground clock reaches the instant its data landed in HBM, exactly like the
deferred HBF write leg, so background work never time-travels ahead of
earlier foreground requests. `hbm_write_buffer_destaged_bytes` counts every
destaged dirty-union byte and `hbm_write_buffer_drain_destaged_bytes` the
part started only at drain; with a watermark of 1.0 the region behaves as a
pure write-back cache whose HBF cost lands entirely in `drain_tail_ns`, so
`user_completion_throughput_GBps` then describes an HBM-only system for any
write working set smaller than the region. Waiting for a free slot is
front-end admission wait for that write (recorded after the wait), never
service time; the cooperative counters report it separately as
`hbm_write_buffer_full_wait_work_ns`.

`direct-read`

Read-only or read-mostly traffic is issued directly to HBF through static
physical placement. `model_weights`, `shared_context`, and unlabeled read
requests use the static HBF mapper; mutable semantic traffic such as
`generated_context`, `scratch`, and `metadata` remains in HBM. This exposes the
HBF read fabric without FTL/resident-mapping/checkpoint/program noise. Static placement is a
capacity-checked reversible permutation. It reuses FLAT's data-page stack owner
and per-stack round-robin plane order, so `flat` versus `direct-read` removes FTL work
without changing data parallelism. Its NAND blocks are fenced out of the
mutable FTL pool.

`hbf-streaming`

Topology and backing comparisons with real serving traffic run through
[ServeLoop](../guides/serveloop.md); on traces, `flat-hbm-bytes` and
`layer-buffer-bytes` set the boundary and the buffer for this policy.

The internal residency binder digest-checks the exported manifest, object map,
and trace, independently recomputes byte and page conservation, and emits an
explicit-residency config plus a placement receipt. The complete object
population determines capacity even when the selected trace window never
touches some weight, metadata, or KV pages. Runtime overhead, the block table,
two active-layer buffers, and the maximum prefix of low-ID physical KV blocks
that fits stay in HBM. The compiler then consumes otherwise unused HBM with a
deterministic immutable-weight prefix and places only the remaining weights
and KV on backing. Reusing one manifest/object map/trace across 6H2F,
4H4F, and 2H6F therefore holds the logical population byte-identical while
capacity pressure becomes an observed result. The trace determines traffic,
not placement. Post-binding mutation, CLI substitution, and `max-ops`
truncation fail before simulation. Unknown objects, model weights outside
their immutable prefix, HBM metadata aliasing backed objects, KV outside the
exported arena, page-size drift, a partial contract, and insufficient active
buffers also fail closed.

Generic microbenchmarks without an object map retain a trace-derived
first-touch fallback, but that fallback is not a Frontier publication path.
Requests may
carry nondecreasing `layer=N` metadata; either every request has it or none
does. A plain R/W trace is one streaming window. `phase=N` is a separate
complete-before-next dependency, is honored independently of layer identity,
and likewise must appear on every request or none. For each layer, data pages
follow:

```text
resident immutable weight: compact HBM resident slot
backed immutable weight: pre-resolved static HBF full-page read
mutable cold KV: logical HBF full-page read from its initial version onward
  -> owning per-stack D2D read -> HBM buffer write
foreground read/write -> compact HBM resident slot or HBM buffer
dirty page -> HBM full-page read -> D2D write -> logical HBF write
```

The next layer begins prefetch when the current layer begins execution, so the
two can overlap. Reusing a parity buffer waits for its earlier layer's dirty
writeback. A full-page foreground write needs no initial HBF read; a partial
write first obtains the old page. `scratch` and `metadata` are HBM-only and are
compacted into the foreground HBM region even when their logical addresses are
high.

`compute_ns=N` may be attached to one or more requests of an explicitly
identified layer. Declarations in the same layer must agree. The interval
extends layer execution and overlaps next-layer DMA; no compute interval is
inferred from uncalibrated workload timestamps.

`layer-buffer-bytes` is the upper bound for one buffer. An explicit contract
reserves two page-rounded buffers from the Frontier plan and verifies that the
trace-visible maximum fits. The byte-exact population is the authoritative
capacity-pressure numerator; page allocation and its rounding are separate:

```text
unique_resident_footprint_pages =
    hbm_only_resident_pages + model_weight_resident_pages
    + hot_kv_resident_pages + backing_unique_pages
unique_resident_footprint_bytes =
    unique_resident_footprint_pages * residency_page_size_bytes
capacity_pressure_basis_bytes =
    immutable_weight_logical_bytes
    + runtime_overhead_logical_bytes
    + block_table_logical_bytes
    + logical_kv_blocks * kv_block_stride_bytes
footprint_page_rounding_bytes =
    unique_resident_footprint_bytes - capacity_pressure_basis_bytes
hbm_capacity_pressure = capacity_pressure_basis_bytes / hbm_capacity_bytes
resident_physical_pages =
    hbm_only_resident_pages + model_weight_resident_pages
    + hot_kv_resident_pages
resident_physical_pages + 2 * effective_layer_buffer_pages + unused_hbm_pages
    == hbm_capacity_pages
max_layer_data_pages <= effective_layer_buffer_pages
```

A workload that exceeds either the per-buffer limit or runtime HBM geometry
fails closed. This is an exploratory hybrid-residency policy, not measured
product behavior. When hbf-streaming is excluded from
`--scenarios`, its inactive layer-buffer regions are not reported as occupied
HBM.

`external-streaming`

This is the no-HBF control for the same layer-streaming policy. It reuses the
same fixed/hot-KV compaction, page plan, two dynamically sized parity buffers,
prefetch timing, foreground-HBM execution, partial-write old-data rule,
dirty-page writeback, buffer-reuse fence, phase/layer dependencies, and
optional compute overlap as `hbf-streaming`. Only the backing device
and transfer route change:

```text
read:  outstanding admission -> M2S command -> propagation
       -> controller issue/process -> media read latency/transfer
       -> S2M payload+completion -> propagation -> HBM buffer install
write: HBM full-page read -> outstanding admission
       -> M2S command+payload -> propagation -> controller issue/process
       -> media write latency/transfer -> S2M completion -> propagation
```

The external device stripes pages by address across independent media
channels. Configured media bandwidth is aggregate and divided evenly across
channels. A global credit limit covers admission through returned completion.
M2S, S2M, controller issue, and each media channel use earliest-gap
reservations, so a later API call whose upstream work is ready earlier may
legally fill a future calendar hole. M2S and S2M are independent full-duplex
wire resources. Fixed processing/media/propagation latency is response work,
not serialization busy time. Every resource reports queue work, busy work, a
resource-local active span, and bounded utilization.

Five exploratory profiles resolve the same model:

| Profile | Capacity | Channels/queue; R/W queues; credits | Controller issue/process | Media read/write | M2S/S2M | One-way propagation |
|---|---:|---:|---:|---:|---:|---:|
| `on-package-lpddr` | 2 TiB | 16; 1/1; 512 | 2 / 20 ns | 2000 / 2000 GB/s; 100 / 100 ns | 2000 / 2000 GB/s | 0 ns |
| `host-dram` (provisional placeholder) | 256 GiB | 8; 1/1; 512 | 2 / 20 ns | 204.8 / 204.8 GB/s; 90 / 90 ns | 36 / 36 GB/s | 75 ns |
| `cxl-memory` | 256 GiB | 8; 1/1; 512 | 2 / 20 ns | 204.8 / 204.8 GB/s; 90 / 90 ns | 36 / 36 GB/s | 75 ns |
| `nvme-ssd` | 4 TiB | 4; 1/1; 512 | 20 / 500 ns | 14 / 7 GB/s; 80 / 100 µs | 16 / 16 GB/s | 250 ns |
| `cxl-ssd` (uncached miss path) | 2 TiB | 8; 1/1; 512 | 20 / 500 ns | 6.5536 / 1.31072 GB/s; 40 / 200 µs | 36 / 36 GB/s | 75 ns |

These are reproducible sensitivity points, not calibrated product claims. The
built-in host-DRAM values are explicitly provisional until a platform-specific
calibration overlay replaces them. `cxl-ssd` is the Cylon (FAST '26)
CXL.mem-over-NAND device class: the cxl-memory transport envelope in front of
the Cylon artifact's FEMU flash media point (4 KiB NAND page, 8 channels x
8 LUNs; aggregate media bandwidth is the 64-LUN pipelining envelope
64 x 4 KiB / latency), with GC still unmodeled. The device-side DRAM cache
Cylon runs in front of its NAND lives in its own module
(`src/physical/external/cxl_ssd.*`) and is enabled only by the explicit
`external-backing-cache-*` block (`cxl-ssd-cached.cfg`): segment-granular
and set-associative with the artifact's fifo/lifo/clock/s3fifo policy set
and next-N prefetch, write-back write-allocate (writes absorb at buffer-DRAM
speed), dirty evictions gating the evicting access on their flash flush, and
internal flush/prefetch occupying media timelines without appearing as
caller traffic. Disabled — the default everywhere, including the base
`cxl-ssd.cfg` miss path — it leaves the shared five-stage schedule
bit-identical. Use
`on-package-lpddr.cfg`, `host-dram.cfg`,
`cxl-memory.cfg`, `nvme-ssd.cfg`, or `cxl-ssd.cfg`, or override
`external-backing-{capacity-bytes,page-size,request-segment-bytes,media-channels,media-read-queues,media-write-queues,max-outstanding-requests}`,
the controller/media timing fields, `external-backing-{m2s,s2m}-bw`,
propagation, and command/completion sizes explicitly. The selected kind and
every resolved number are embedded in the summary and `--config-out`.

The fixed-footprint host-DRAM/SSD rows additionally apply the DANA A100 timing
overlays in `configs/overlays/backing/calibrated/`. Those values come from
three complete pinned-host and compute-local direct-I/O runs: two fit runs and
one untouched 16 MiB QD1/QD8 holdout, all on `gpu-51`. The SSD overlay uses
one channel per queue; fit-run QD8/QD1 scaling selects one effective read queue
and one effective write queue for the observed caller-range concurrency. On
the independent holdout, median absolute throughput/p50 errors are
1.337%/2.988%, per-cell maxima are 8.833%/14.569%, and saturated-read QD8
maxima are 1.153%/1.543%. That workload-applicability gate does not cover other
request sizes or systems. The following capacity
overlay remains only a 1 TB address-space projection, not a measurement of
usable host DRAM or SSD capacity.

The engine carries no workload-fitted software-path term for inference
offload. A frontend that has such a fitted term expresses it in its own
transaction DAG as a `BARRIER` of the fitted duration depending on the
`EXTERNAL` transaction, so the physical receipts stay physical and the term is
visible as the frontend's own dependency.

The model deliberately stops before CXL.cache coherence and a full
PCIe/NVMe/SSD stack. In particular, NVMe FTL/GC, wear, tail latency, failure
consistency, and product-specific firmware are absent; they cannot be
smuggled into a conclusion as an unnamed multiplier.

`demand-fill`

This behavior-only baseline treats HBF as authoritative backing and admits
every first-observed page into a finite HBM page tier. Clean and dirty LRU
evictions are explicit; dirty pages write back to HBF. It is a comparison
policy for placement studies, not a recommended product policy.

`reuse-filtered`

This behavior-only policy first bypasses a cold page to HBF and records it in
a bounded history. A page is promoted only after reaching the configured
reuse threshold. It shares the same HBM tier, LRU, dirty-writeback, and backing
model as `demand-fill`, so the study isolates the admission decision.

## Time-Accounting Contract

Schema v19 makes `time_breakdown` the authoritative timing surface. Its
`wall_clock_ns` entries are non-overlapping spans and satisfy both:

```text
offered_arrival_span_ns + post_offer_user_completion_tail_ns
    = user_completion_span_ns
user_completion_span_ns + drain_tail_ns = makespan_ns
```

`latency_work` sums durations across user operations, while `stage_work` sums
device/controller work that can overlap across requests and resources. Neither
is an additive decomposition of `makespan_ns`. For a resource, use
`resource_busy`: `capacity_time_ns = resource_count * active_span_ns` and
`utilization = busy_ns / capacity_time_ns`. The two top-level rates use
different denominators deliberately: `user_completion_throughput_GBps` ends at
the last user completion, while `makespan_throughput_GBps` includes drain.

The Direct front end schedules page transactions, not whole trace-record
parents. Same-phase ready parents receive one page per round, so a large DMA
cannot monopolize the credit window. Page-granular dependencies still prevent
a read or write from overtaking an earlier overlapping write, and prevent a
write from overtaking earlier overlapping reads.

Every scenario issues its HBM page transactions the same way: one at a time
through `HbmDevice::issue`, so the completion that returns a credit is known
immediately and HBM requests never reorder across parent boundaries. The
summary records this as `hbm_issue_mode = synchronous-per-transaction` in the
`config` block and in every scenario. There is no separate batch path for
`all-hbm`; FLAT with a boundary above the whole footprint reproduces
`all-hbm` bit for bit, and the runner's policy-identity sanity check
enforces exactly that (`tests/python/test_reference_policy_identity.py`).

Offered load is reported per scenario as `offered_load`: `rate_GBps`
(logical bytes over the offered-arrival span, `null` when every request
arrives at one instant), the read and write split, the peak payload rate of
each tier present (`peak_rate_GBps`: HBM channels x derived interface rate x
stacks; HBF stacks x HBIO; the external S2M and M2S wire rates), and
`offered_to_peak_ratio` per tier (for the external tier the larger of the
read-to-S2M and write-to-M2S ratios). `arrival_model` is `open-loop` unless
a `--max-outstanding-requests`/tier window bounds admission. Traces without
`at=` are injected at `--interarrival-ns`; the published policy configs use
1 ns with 4 KiB lines, which offers 4096 GB/s, the server profile's HBM peak.
When an open-loop run offers at least 90% of a present tier's peak the runner
prints a `warning:` on stderr and adds it to the scenario's `warnings`: the
latency percentiles then measure the arrival model's queue, not device
service. Use a closed-loop window or trace timestamps for latency claims.
Utilization and throughput are unaffected by this advisory.

`SANITY` has three verdicts. `FAIL` (exit 1) is reserved for conservation,
bounds, and contract violations. `PASS_WITH_WARNINGS` (exit 0) marks a valid
run whose configuration did not exercise what a selected scenario exists for,
today the `flat` boundary lying above the whole footprint so nothing reached
HBF. `PASS` has neither; advisories such as the open-loop saturation warning
and the HBIO/ECC ceiling notes never change the verdict.

Use the namespaces in this order when auditing a run:

1. `wall_clock_ns` locates offered-arrival time, the post-offer completion
   tail, and deferred drain.
2. `latency_work` exposes three complete distributions: service starts when
   the first page transaction receives a credit, the primary offered latency
   also includes front-end admission wait, and source latency additionally
   includes phase/dependency wait.
3. `stage_work` attributes accumulated HBM, HBF, external-backing,
   base-die-link, and layer-streaming controller work. This namespace is
   hierarchical: a parent total and its attribution/subset children are
   alternative views and must not be added together. Independent requests and
   resources may also overlap in wall time.
4. `resource_busy` reports `busy_ns`, resource count, active span,
   capacity-time, and utilization for each finite execution resource.

The nested JSON object is authoritative. Summary CSV columns are its flattened
scalar view for tables; `_work_ns` denotes an accumulated duration, not a
non-overlapping wall-clock interval. JSON retains full floating-point
precision; the human-oriented CSV and console round nanoseconds to 0.01 ns.

The time-breakdown renderers in `reports/` produce a Markdown overview, a
long-form CSV, and a self-contained HTML visualization from any summary. The
overview ranks primary stage work and exclusive resource utilization, while
the CSV labels additive wall segments, overlapping work, parent totals,
attribution children/subsets, and resource capacity explicitly. The HTML keeps
wall clock, per-operation latency, overlapping stage work, and resource
capacity in separate panels. Existing summaries can be rendered without
replay:

```bash
python3 reports/time_breakdown.py \
  --summary baseline=out/run/baseline.summary.json \
  --summary experiment=out/run/experiment.summary.json \
  --csv out/run/time-breakdown.csv \
  --markdown out/run/time-breakdown.md

python3 reports/time_dashboard.py \
  --summary baseline=out/run/baseline.summary.json \
  --summary experiment=out/run/experiment.summary.json \
  --output out/run/time-breakdown.html
```

The renderer validates schema v19, wall and latency identities, device-stage
totals, ECC directionality, resource capacity, and occupancy bounds before
replacing a report. Stage work may exceed makespan because operations and
resources overlap; only rows with `role=additive_segment` reconstruct elapsed
time.

## Address-Heatmap Contract

Every scenario in a schema-v19 summary carries an `address_heatmap` field:
`null` unless `--address-heatmap-bins` requested a snapshot, otherwise an
`hbfsim.address_heatmap.v1` object. It has five independently sized address
domains in canonical order:

| Domain | Meaning |
|---|---|
| `workload_logical` | Exact byte ranges accepted from the input trace, before placement or amplification. |
| `hbm_physical` | Burst-aligned physical HBM traffic, including foreground, streaming-install, cooperative-buffer, and destage traffic. |
| `hbf_logical` | Byte ranges submitted through the mutable logical HBF/FTL interface. |
| `hbf_physical` | Full NAND page reads/programs and full-block erases after mapping and maintenance amplification. |
| `external_physical` | Exact bytes transferred to or from the selected on-package-LPDDR, CXL-memory, or NVMe backing address space. |

Physical-direct HBF requests intentionally bypass `hbf_logical`. Therefore
`direct-read` can have traffic in `hbf_physical` while its
`hbf_logical` panel is empty; this is the expected route, not missing
instrumentation. Do not align bin indices across domains as if they named the
same storage: each domain has its own extent and bin boundaries.

Each domain reports read, write, and erase bytes and accesses. Reads and writes
are separate colored lanes in the renderer; writes also use diagonal hatching
and erases use cross-hatching so direction remains visible without color.
Domain-scope accesses count accepted traffic records once. A record that spans
multiple bins contributes exact overlap bytes to each bin and increments the
access counter of every bin it touches. Consequently, bytes sum exactly from
bins to the domain total, while bin access counts intentionally fan out and can
sum above the domain access count. HBF physical erase bytes cover the entire
erased block. Intervals remain half-open even at the top of the address space:
the exclusive endpoint `18446744073709551616` (`2^64`) represents a domain or
bin that includes byte address `UINT64_MAX`; it is an endpoint, never an issued
address or a byte/access counter.

The closed traffic-source vocabulary is:

```text
workload, direct, mapping, prepopulate, cooperative_buffer, demand_fill,
prefetch_fill, streaming_install, destage, garbage_collection, maintenance
```

The source answers why an access exists; the domain answers where it is
observed. Current initial-image prepopulation constructs already-programmed
state before simulated time. It schedules no runtime device operation and
therefore contributes neither timing work nor heatmap traffic; its
`prepopulate` source remains zero for that state-only path.

The HBF write coalescer retains byte-range last-writer provenance while data is
buffered. A NAND merge read or page program is nevertheless one indivisible
full-page physical operation, so it cannot be split proportionally among
several request sources. HBFSim attributes that full physical operation to the
source owning the largest number of dirty bytes in the buffered page. Equal
dirty-byte coverage is resolved by the canonical source order listed above.
This dominant-source rule is deterministic attribution of physical work, not a
claim that every byte in the programmed page originated from that source.

Regions named `semantic_observed_extent/<kind>` are overlays computed from the
minimum and maximum address observed for that semantic kind. They are visual
bounding boxes, not ownership claims: disjoint requests of the same kind may
leave gaps inside an overlay. Bin traffic remains exact.

`hbfsim-reference` collects no heatmap unless asked (`--address-heatmap-bins`
defaults to 0, and each scenario's `address_heatmap` is then `null`; a
1024-bin snapshot is roughly 2 MB of JSON per scenario) and rejects values
outside 0..8192:

```bash
./build/hbfsim-reference \
  --trace workload.trace \
  --address-heatmap-bins 2048 \
  --summary-json out/summary.json
```

Storage is bounded by the selected bin count, five domains, and the fixed
source vocabulary; it does not retain one heatmap row per request. Higher
resolution increases summary size and rendering cost linearly.

Render a standalone `hbfsim.address_heatmap.v1` artifact directly:

```bash
python3 reports/address_heatmap.py \
  --input heatmap.json \
  --output heatmap.html
```

For a multi-scenario v19 summary, select one exact scenario name:

```bash
python3 reports/address_heatmap.py \
  --input out/summary.json \
  --scenario hbf-streaming \
  --output out/layer-streaming-heatmap.html
```

`--scenario` may be omitted only when the summary contains exactly one
scenario. The renderer validates schema, domain/source ordering, bin coverage,
and byte/access accounting before it writes the self-contained HTML file.

## Scenario Interpretation

`direct-read` answers the raw read-fabric question: how fast is
read-mostly/static HBF when mutable FTL writes are absent?

`hbf-streaming` asks whether backed weights and cold/overflow KV for
the next layer can be prefetched early enough while hot KV and all foreground
execution remain in HBM.
It requires either a plain trace (one window) or complete, nondecreasing layer
metadata:

```text
0xADDR R 4096 model_weights layer=0
0xADDR W 4096 generated_context layer=1
```

The JSON object is authoritative: select `scenarios[]` by `name`, then read
layer state under `layer_streaming`, route bytes under `hybrid_path`, and access
counts at the scenario root. The following are the corresponding flattened CSV
column names; `layer_streaming_foo` maps to JSON `layer_streaming.foo`, while
route columns such as `hbf_static_read_bytes` map to
`hybrid_path.hbf_static_read_bytes`:

```text
layer_streaming_mode
layer_streaming_residency_policy
layer_streaming_compact_resident_mapping
layer_streaming_semantic_inputs_consumed
layer_streaming_explicit_residency_contract
layer_streaming_address_footprint_bytes
layer_streaming_unique_resident_footprint_pages
layer_streaming_unique_resident_footprint_bytes
layer_streaming_capacity_pressure_basis_bytes
layer_streaming_footprint_page_rounding_bytes
layer_streaming_hbm_capacity_bytes
layer_streaming_hbm_capacity_pressure
layer_streaming_layers
layer_streaming_explicit_layer_requests
layer_streaming_explicit_compute_layers
layer_streaming_compute_work_ns
layer_streaming_hbm_only_resident_pages
layer_streaming_hbm_only_resident_bytes
layer_streaming_hot_kv_candidate_pages
layer_streaming_hot_kv_resident_pages
layer_streaming_data_pages
layer_streaming_data_bytes
layer_streaming_model_weight_resident_pages
layer_streaming_model_weight_resident_bytes
layer_streaming_model_weight_backing_pages
layer_streaming_cold_kv_backing_pages
layer_streaming_unknown_backing_pages
layer_streaming_backing_unique_pages
layer_streaming_resident_physical_pages
layer_streaming_resident_physical_bytes
layer_streaming_effective_layer_buffer_pages
layer_streaming_effective_layer_buffer_bytes
layer_streaming_unused_hbm_pages
layer_streaming_unused_hbm_bytes
layer_streaming_streamed_pages
layer_streaming_streamed_bytes
layer_streaming_foreground_resident_page_accesses
layer_streaming_foreground_buffer_page_accesses
layer_streaming_dirty_pages_written_back
layer_streaming_writeback_bytes
layer_streaming_max_layer_data_pages
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

The core conservation checks are:

```text
streamed_bytes == streamed_pages * hbf_page_size
streamed_bytes == hbf_static_read_bytes + hbf_logical_read_bytes
streamed_bytes == base_die_link_read_bytes
streamed_bytes == hbm_streaming_write_bytes

writeback_bytes == dirty_pages_written_back * hbf_page_size
writeback_bytes == base_die_link_write_bytes
writeback_bytes == hbf_backing_write_bytes

hbm_user_accesses ==
    foreground_resident_page_accesses + foreground_buffer_page_accesses
hbf_user_accesses == 0
unique_resident_footprint_pages == hbm_only_resident_pages + data_pages
unique_resident_footprint_bytes ==
    unique_resident_footprint_pages * residency_page_size_bytes
capacity_pressure_basis_bytes + footprint_page_rounding_bytes ==
    unique_resident_footprint_bytes
data_pages == model_weight_resident_pages
    + hot_kv_resident_pages + backing_unique_pages
backing_unique_pages ==
    model_weight_backing_pages + cold_kv_backing_pages + unknown_backing_pages
resident_physical_pages ==
    hbm_only_resident_pages + model_weight_resident_pages
    + hot_kv_resident_pages
max_layer_data_pages <= effective_layer_buffer_pages
max_layer_data_bytes <= layer_buffer_bytes
resident_physical_bytes + 2 * effective_layer_buffer_bytes + unused_hbm_bytes
    == hbm_capacity_bytes
```

`exposed_prefetch_ns` is prefetch time on the user-visible critical path;
`hidden_prefetch_ns` is the portion overlapped by useful current-layer
execution. A plain one-window trace cannot demonstrate inter-layer overlap, so
its initial fill is expected to be exposed. `buffer_reuse_wait_work_ns`
identifies a parity-buffer reuse blocked by an earlier layer's writeback.

Access counts are implementation-granularity diagnostics. HBF and HBM
background accesses can fan out at page or contiguous-range granularity; compare
bytes for conservation. The address heatmap provides exact source-attributed
traffic through `prefetch_fill`, `streaming_install`, and `destage`.

When comparing different `--max-ops` prefixes, use the same
`--initial-image-trace` for every point. Otherwise truncation also changes
which read-before-write pages are populated and can change FTL placement and
checkpoint state, confounding workload-size scaling with a different memory
image.

The HBF timing fields separate media, transfer, ECC, and maintenance
work. If `hbf_array_read_work_ns` dominates, HBF sensing is the limit; if
base-die-link utilization is high, D2D is the limit; if HBM bus utilization is
high, buffer installation or foreground reads are the limit. These work fields
can overlap and must not be summed into wall-clock time.

## Parallelism Fields

`hbfsim-reference` reports both utilization and effective parallelism:

```text
hbm_bus_parallelism
hbm_active_pch
hbm_pch_busy_skew
hbf_media_parallelism
hbf_active_planes
hbf_page_reads
hbf_page_read_admission_events
hbf_page_read_admission_waited_pages
hbf_page_read_admission_wait_work_ns
hbf_page_read_admission_max_wait_ns
hbf_page_run_requests
hbf_page_run_pages
hbf_page_run_physical_requests
hbf_page_run_static_requests
hbf_page_run_logical_requests
hbf_streaming_read_buffer_bypass_pages
hbf_scalar_read_requests
hbf_scalar_read_pages
hbf_data_programs
hbf_page_programs
hbf_resident_mapping_table_bytes
hbf_mapping_lookup_ops
hbf_mapping_update_ops
hbf_mapping_dram_issue_busy_ns
hbf_mapping_page_programs
hbf_waf
hbf_array_read_work_ns
hbf_array_program_work_ns
hbf_array_erase_work_ns
hbf_read_lane_parallelism
hbf_active_media_lanes
hbf_media_lane_skew
hbf_subarray_read_parallelism
hbf_active_subarrays
hbf_subarray_busy_skew
hbf_subarray_read_skew
hbf_page_buffer_bank_parallelism
hbf_active_page_buffer_banks
hbf_page_buffer_bank_skew
hbf_page_buffer_bank_read_skew
hbf_plane_media_skew
hbf_channel_parallelism
hbf_active_channels
hbf_sequencer_parallelism
hbf_active_dies
```

Use these before trusting a slowdown explanation.
`hbf_plane_media_utilization=100%` means
the aggregate plane-media capacity is saturated over the measured window;
`hbf_media_parallelism=3.7` means the run accumulated roughly 3.7 concurrent
array-operation equivalents over that window. `hbf_subarray_read_parallelism`
isolates array-sense work inside planes;
`hbf_read_lane_parallelism` isolates array-to-page-buffer transfer lanes; and
`hbf_page_buffer_bank_parallelism` isolates banked page-buffer output. If HBF is
slow, these three fields tell whether the missing parallelism is before,
between, or after sensing. If all three stay low while HBF is slow, the
bottleneck is probably write/program, mapping, GC, queueing, channel/HB-IO, or
an address pattern that does not expose enough independent read locations. High
`hbf_plane_media_skew`, `hbf_subarray_busy_skew`, `hbf_media_lane_skew`, or
`hbf_page_buffer_bank_skew` means the address mapping or staging path is
concentrating work on a subset of physical resources.
