# Observability and runtime guards

> Status: Current
> Last reviewed: 2026-09-29

`ctest` plus the explicit physical, component, and write-amplification gates
form the repository's internal verification surface. None substitutes for
parameter calibration or external hardware validation.

After touching physical-model code, run the scenario guard:

```bash
cmake --build build --target verify_physical
```

This runs the focused HBM and HBF guard scenarios and writes outputs under:

```text
out/physical-guards/
```

Run the cross-component and write-amplification checks as well after a model
change:

```bash
cmake --build build --target verify_components
cmake --build build --target verify_write_amplification
```

Normal `ctest` includes `component_contracts`. The explicit targets remain
separate so a component failure cannot be hidden inside a nested physical or
write-amplification invocation.

## Per-workload physical wear

Every completed HBF session emits an offline physical wear heatmap and JSON, including zero-erase workloads. The report separates restored P/E history from this run, preserves physical stack/channel coordinates, and exposes host zone swaps. See [host management and wear reports](host-hbf-management.md) for configuration and the live-stream v3 fields.

## Failed terminal responses

Failed session shutdown preserves the exact nonempty QUIT response separately
from validated measurements. `source_receipt().terminal_failure` contains the
raw line, decoded object when parsing succeeded, executable/source identity and
the original exception; it is explicitly unvalidated. A normal successful close
does not add this field. E3 writes such evidence to `failed-execution.json` with
`result: fail`, without creating a successful result or accepting an invalid
`final_measurement`. The original error is re-raised even if writing the failure
artifact fails. Missing terminal output is not reconstructed from prior batches.
See implementation and bounded regression (`out/terminal-failure-receipt-20260922/README.md`, local run output, not tracked).

## Mapping observations

`MAPPING_STATS` reports maintained mapping, traffic, cache and merge counters
without draining work or advancing device time. It refreshes the ordered
mapping/write-buffer issue-work totals and checks mapping-compute work, but
does not scan the whole medium or audit structural ownership. This avoids
repeating full-media scans at every mapping observation. Dirty/pending state
still uses the normal quiescence inspection.

Explicit full statistics, wear observations and normal close retain their
complete accounting audits. Population, wear and parallelism fields exposed
through the internal `execution_stats()` view remain lazily derived; a mapping
observation alone does not refresh them or certify the entire device state.
The mapping JSON schema, counts and floating-point issue-work writebacks are
unchanged. See the dynamic old/new comparison (`out/mapping-snapshot-lean-20260922/README.md`, local run output, not tracked)
for subsequent requests, persistence and restore verification.

## Guard Scenarios

### HBM

The current model is `channel-aggregate-v2`. The supported probes are:

- `hbm-interface`: derived interface width/rate/burst accounting and effective
  channel-service duration.
- `hbm-boundaries`: burst-rounded bytes, address-map round trips and capacity
  rejection.
- `hbm-channels`: independent channels and aggregate bandwidth scaling.
- `hbm-turnaround`: configured application read/write switch delays.

The `hbm_channel_model` CTest covers independent per-burst byte counting,
finite queues, fair service quanta, DMA gaps and interleaving, bounded causal
advancement, and safe single-request aggregation. `hbm_clock_roundoff` checks
integer-clock boundaries and time-translation invariance;
`hbm_controller_buffer` checks shared calendars and reserved capacity.

There are no row-state, ACT/PRE/RD/WR, or periodic-refresh probes in this model.
Bank coordinates in the address map do not imply bank-level timing simulation.

### HBF

`physical_probe hbf-standard` combines the current Bank, ECC, Host, program
barrier and exact-calendar contracts. Individual selectors are
`hbf-bank-pipeline`, `hbf-ecc-pipeline`, `hbf-host-boundary`,
`hbf-program-barriers`, and `hbf-calendar`.

Independent Banks can overlap; the same Bank has one ordered sense resource.
Raw transfer, ECC initiation, ECC response latency and decoded SRAM have distinct
finite resources. A full page plus OOB travels internally; the aligned host read
payload alone crosses the selected channel's transmit resource. All programs,
including mapping and GC copies, receive one complete page through the channel.

Host write acceptance and device program completion are distinct. Host GC reads
a live page, holds it in a finite copy slot, programs its destination and
publishes its mapping before reclaiming source ownership. Physical erase is
charged to the next page-zero program, or an explicit raw erase. The native
contracts reject skipped pages, erased raw reads and cross-page device reads;
the independent ledger oracle checks small stateful histories and timing.

`ocp_standard_test` additionally exercises independent channel directions, two
cached decoded pages per Bank, temporal fills, LRU eviction, program completion,
zone-reset erase accounting and the public HBM4 organization. The mapping,
GC, wear, persistence and public-session tests are indexed in
[verification coverage](../verification/coverage.md).

## How To Read Chrome Trace

Open:

```text
chrome://tracing
```

Click `Load`, then select a file such as:

```text
out/physical-guards/hbm-channels.trace.json
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

Resource waiting. Detailed HBF waits remain in their resource spans. Aggregate
HBM queue/switch waiting is reported in completion breakdowns, not reconstructed
as fictional DRAM command spans.

`hbm_channel_service`

Application transfer service on an HBM service group (`hbm/group{N}`,
`hbm-service-group-channels` consecutive pseudochannels; the span lasts the
busiest lane's demand and its physical bytes sum the group's lanes), including
the effective-bandwidth overhead. `hbm_channel_read` and `hbm_channel_write`
details report payload bytes and the number of represented quanta. Uncontended
quanta can share one span.

`hbm_buffer_bus`

Controller-buffer DMA on the same service-group data calendars. These transfers
retain their bus-only service abstraction and separate controller response latency.

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

For an HBM request, find its `hbm_channel_read` or `hbm_channel_write` spans
on all touched pseudochannels. The parent finishes when the last channel
transfer finishes. The completion breakdown separates aggregate access latency
(`command_ns` in the shared breakdown structure), raw payload transfer time,
efficiency overhead (`maintenance_ns`) and queue/switch waiting. These fields
do not identify individual DRAM commands or refresh stalls.

For a Host buffered write, distinguish acceptance into the reserved HBM buffer from its later
full-page program and drained completion. For a GC copy, follow the ordinary
HBF read response, HBM buffer write/read, copy-slot dependency and program payload.
The controller buffer shares HBM data-channel reservations with application
traffic. `host_hbm_reserved_bytes`, `host_hbm_read_bytes`, `host_hbm_write_bytes`,
`host_hbm_busy_ns` and `host_hbm_queue_wait_ns` report that traffic. HBM's
`controller_buffer_*` counters are subsets of its aggregate byte and raw bus-work
counters. `service_busy_ns` additionally includes effective-bandwidth overhead.
`channel_transfers` counts calendar reservations including controller DMA;
`service_quanta` counts represented application quanta. Reclaim is a Host state
transition; an erase span appears when the device actually erases a block.

Trace spans should agree with the declared aggregate model; exact physical
bytes are recorded separately from approximate channel service duration.

## Simulation-session protocol (`hbfsim`)

`hbfsim --system-config FILE [...]` runs one persistent session over
stdin/stdout. Every reply is one JSON line; the first is the ready receipt
(`"result":"ready"`, `protocol: simulation-transaction-text-v2`), which echoes
the resolved tiers, HBF setup, external backing, the direct lane (or `null`),
`dependency_window_batches`, and the engine's source provenance
(`source: {git_commit, git_dirty, tree_hash, source_sha256, provenance_source}`;
`provenance_source` is always `build-time`). Incremental builds regenerate
the C++/CMake source manifest when contents change; the binary retains its
own fingerprint even after checkout changes. Session options are CLI
flags only; in particular `--hbf-physical-heatmap PATH
--hbf-physical-heatmap-bins N` (both required together) append a live
HBF physical heat stream. Integers on the command line and in configs are
decimal digits only (`010` is ten, `0x10` is rejected).

The ready receipt records `host_memory_model: hbm-reserved-shared-data-channels`,
`hbf_buffer_hbm_bytes` and `hbm_application_capacity_bytes`. HBF controller
storage is subtracted before application placement; no additional CPU memory
device is created. A mapped HBF controller without HBM is rejected.

`hbfsim --describe-system --system-config FILE [...]` returns one
`hbfsim.resolved_system` v1 JSON record and exits. It reports resolved HBM/HBF
geometry (including `hbm_burst_bytes`), controller storage, and the engine's
unreserved logical HBF capacity.
Image/extent options are rejected in this mode. ServeLoop uses this query,
including alias resolution, rather than estimating logical capacity from raw bytes.

Commands:

```text
BEGIN <batch_id> <logical_sha256> <transaction_sha256> [frontier=<ids|->] [retain=<ids|->] [completions=0|1] [observe=<ids|->]
TX id=<id> target=<T> op=<R|W|-> addr=<n> bytes=<n> issue_ns=<f> duration_ns=<f> deps=<ids|-> stack=<n|->
END <batch_id>
CHECKPOINT <id>            CHECKPOINT_IMAGE <id> <hex path>
WEAR_SNAPSHOT <id>         CRASH <id>            QUIT
```

`TX` fields are positional in exactly that order with single spaces; ids are
`[A-Za-z0-9_.:/-]+`, lists are comma-separated. `BEGIN` options:

- `frontier=<ids>`: transactions whose completion defines the batch's
  `blocking_finish_ns`/`elapsed_ns` and the next batch's origin (default: all;
  `-` means none, so only the batch's last arrival advances the origin). Every
  id must belong to the batch. Detached work still executes and stays
  dependable.
- `retain=<ids>`: the complete set of earlier transaction ids later batches may
  still name. Dependencies resolve against this batch, the last
  `dependency_window_batches` (2) completed batches, and the retained set;
  anything older fails closed. A list replaces the retained set and `-`
  clears it; a `BEGIN` without the field leaves the set unchanged, so a
  remapper that declares what it holds is not undone by batches another
  producer submits on the same session. Every retained id must be
  resolvable when it is listed.
- `completions=0`: omit the per-transaction completion records (and their
  digest) from the receipt; aggregate timing and device deltas are unchanged.
- `observe=<ids>`: return `observed_timings` with native `id`, `start_ns` and
  `finish_ns` for only these DAG nodes, including compute barriers.
  IDs must be unique and local to the batch. This neither changes the frontier
  nor retains dependencies. Tiered writeback uses these observations with
  `completions=0`; full memory completions remain available for diagnostics.

The batch receipt (`hbfsim.simulation_batch_completion`) reports
`batch_origin_ns`, `first_issue_ns`, `blocking_finish_ns`, `finish_ns` (all
work), `elapsed_ns` (= blocking finish − origin), `total_elapsed_ns`
(= finish − origin), `frontier_transactions`, the latency matrix, the
optional `transaction_completions` + digest, per-device deltas,
`hbm_engine` (`requests`, `bursts`), `hbf_read_engine`, and the target
census. `hbf_read_batching.groups/pages` counts numerical pipeline execution
groups and their data-page coverage; it does not count NAND commands.
Transaction queue, service and latency work are accumulated independently with
compensated sums in both batch and cumulative ledgers. Latency remains the sum
of each observed `finish-arrival`, not a value derived from queue plus service.
This avoids cumulative roundoff failures in long runs without relaxing the
conservation tolerance or changing completion times. Work sums may overlap;
they are not a decomposition of the application's elapsed time.
`hbf_read_batching.direct_gap_reservations` counts shared TSV/SRAM reservations
that reuse a currently verified first-fit gap position instead of performing
the complete calendar lookup. Each still consumes its exact resource interval;
this is host execution coverage, not eliminated pages or physical events.
`hbf_read_runs.runs/pages/reused_pages/overridden_pages` separately describes
compressed compact-image address calculation: constructed address segments,
logical pages receiving a candidate, candidates derived from a later bank
round, and candidates superseded by authoritative mapping visibility. A
candidate may subsequently hit a buffer or be superseded by GC; these host
execution counters cannot replace physical reads, bytes, or completion times.

A batch rejected before any device state changed (malformed line,
digest mismatch, unknown dependency, out-of-range address, cycle, empty
batch, unknown frontier/retain id) is answered with

```json
{"schema":{"name":"hbfsim.simulation_batch_completion","version":1},"result":"error","batch_id":"...","message":"...","session_state":"unchanged"}
```

and the session continues; the batch id and its transaction ids were not
consumed. Structural faults (EOF inside a batch, a nested `BEGIN`, a
malformed `BEGIN` line) and any post-mutation failure still terminate the
process with a non-zero exit. The Python client (`hbfsim_client`) raises
`SimulationSessionError` with the engine's message for error records and
keeps the session open.

## Full simulation is a different layer

`verify_physical` is a focused regression gate. The `hbfsim` application is
the semantic-free transaction engine. The separate `hbfsim-reference` runner
provides the maintained reference-policy interpretations:

```text
all-hbm
all-hbf
flat
direct-read
hbf-streaming
external-streaming
demand-fill
reuse-filtered
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
