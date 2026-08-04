# ASTRA-sim 3.0 paper-aligned workload frontend

HBFSim can replay derived LLM memory traffic exported by the sibling
`astra-sim` paper-aligned frontend.
ASTRA-sim models the GPU side — compute units, workgroup scheduling,
cache-line requests, on-chip NoC, and memory-channel contention — and the
exporter normalizes that traffic into an HBFSim semantic trace. HBFSim then
replays the trace across its seven experiment cases like any other input.

The implementation is based on official upstream commit `518bd51` and follows
the ASTRA-sim 3.0 paper (arXiv:2606.10440) for the fine-grained GPU execution
path needed here. It is not the ASTRA authors' unreleased 3.0 source: as of
2026-07-23 the public repository still identifies itself as ASTRA-sim 2.0 and
has no public 3.0 tag. The exact sibling revision must therefore be committed
and recorded with every research result. See the sibling repository's
`docs/astra-sim-3.0-implementation.md` for the paper-alignment matrix and
deliberate model boundaries.

For production serving work, the selected primary path is Qwen-Bailian ->
Frontier -> an audited memory-object exporter -> HBFSim. See
`docs/real-workloads.md`.

## Use cases vs experiment cases

USE cases (`uc*`, `tools/run_use_cases.py`) run simple synthetic
inputs; they exist to detect invariant and contract drift.
EXPERIMENT cases (`ec*`, `tools/replay_astra_trace.py`) replay model-derived
ASTRA workloads over the same composition axis; they exist to answer the
research questions. Same hardware points, different inputs, different
purpose — the vocabularies are not interchangeable (user decision
2026-07-09).

## Trace grammar extensions: `at=` and `layer=`

ASTRA-sim traces carry a provenance header and per-op arrival timestamps:

```
# schema=astra-sim.hbfsim-memory-trace version=2 workload_id=<sha256> requests=N bytes=B
0x1000000000 R 128 model_weights layer=0 at=1.0
```

`at=<ns>` (alias `arrival=`) overrides the synthesized `interarrival-ns`
arrival for that op only; ops without `at=` keep the synthesized arrival, so
existing traces and the built-in generators are unaffected. This preserves
ASTRA-derived burst structure (compute gaps between kernels, per-kernel read
storms), which a uniform interarrival cannot represent.

EC6 accepts nondecreasing `layer=N` on every request. Without it, the entire
ASTRA trace is one streaming window and cannot exercise next-layer ping-pong
overlap. `phase=N` is independent complete-before-next scheduling metadata,
not a layer alias. Optional `compute_ns=N` supplies an explicit layer execution
interval; it is never inferred from uncalibrated ASTRA timestamps.

## Producing a trace

From the sibling paper-aligned ASTRA-sim tree:

```bash
.venv/bin/python examples/workload/llm/generate_llm_chakra.py \
    --out out/llm --mode decode --layers 2 --hidden 2048 --heads 16 \
    --kv-heads 4 --ffn 5504 --vocab-size 256 \
    --prompt-len 1024 --decode-tokens 4 --max-seq 2048 \
    --gpu-configuration examples/gpu_model/generic_gpu.json

./build/astra_analytical/build/bin/AstraSim_GPU_Analytical \
    --workload-configuration=out/llm \
    --system-configuration=examples/system/native_collectives/Ring_4chunks.json \
    --remote-memory-configuration=examples/remote_memory/analytical/no_memory_expansion.json \
    --gpu-configuration=examples/gpu_model/generic_gpu.json \
    --hbf-trace-out=out/llm_hbf
# -> out/llm_hbf.0.trace and out/llm.manifest.json
```

That command is a reduced contract smoke test. The formal 8B path selects
`--model-profile llama3.1-8b-bf16`, which resolves 8,030,261,248 parameters,
14.96 GiB of TP=1 BF16 resident weights, 8 KV heads, and a native 131072-token
limit. A batch-1 full-context KV arena is exactly 16 GiB. Generating its ET and
manifest is cheap, but exporting a lossless cache-line trace is proportional
to tens of GiB of logical traffic per decode token; choose and record a
representative window rather than mistaking a reduced smoke test for an 8B
experiment.

The generator binds every ET node to the SHA-256 of the complete GPU JSON and
assigns one stable workload ID derived from the generator, configuration, and
normalized parameters. Changing a bandwidth, latency, topology, or execution
limit therefore invalidates an old ET even when its CU count and cache-line
size happen to match. The exporter preserves every serviced cache-line request
without deduplication, coalescing, or sampling, and publishes only after its
exact runtime request/byte census succeeds.

For `--tp > 1`, generation also requires `--emit-mscclpp`. Collective ET nodes
and the generated relative-address plan carry the same content-derived plan
ID; omitting or substituting that plan fails instead of silently falling back
to a coarse predefined collective. The parser accepts only
`astra-sim.mscclpp-subset` version 1, validates its complete structure and
buffer spans, and recomputes the plan ID from canonical content. A stale or
forged embedded ID is rejected. TP activation slots reserve a second, equally
sized receive range so a large collective cannot alias the next slot.

Semantic kinds map 1:1 onto HBFSim's: weights -> `model_weights`, prompt KV
-> `shared_context`, decode KV -> `generated_context`, activations ->
`scratch`, semaphores -> `metadata`.

### The kind contract (decided 2026-07-05)

Kinds are OPTIONAL placement hints; plain address-only traces are the
first-class citizen and every composition is fully defined without them
(guarded by the `composition-kind-blind` probe):

- No kind steers a write onto raw physical HBF placement. Scratch/metadata
  writes remain HBM-only. In EC6, all foreground requests execute from HBM:
  generic traces without an object map use compact scratch/metadata plus a
  bounded first-touch `shared_context` fallback; production Frontier traces
  instead require an explicit object-map residency contract. Model weights,
  cold/generated KV, and unlabeled pages stage through one of two layer
  buffers. Immutable weights start in a pre-resolved, stack-striped physical
  extent. Mutable cold KV uses the logical FTL from its initial version
  onward, and later layers read the newest logical version after writeback.
  Initial-image reservation is state setup, not a foreground raw-physical
  write.
- Kinds refine reads only where a policy says so: in EC6, scratch/metadata are
  always fixed residents; the generic fallback may admit first-touch shared
  KV, while the production contract fixes hot block IDs independently of trace
  order; weights/cold/default pages participate in backed layer streaming;
  in the read-only-direct composition the {weights, shared, unknown} filter
  keeps other labeled reads (e.g. metadata) on HBM. Unlabeled reads pass every
  filter unchanged.
- FLAT / all-HBM / all-HBF never consult kinds at all.

Static placement reserves complete physical NAND blocks outside the mutable FTL
pool; it is not a modulo hash and cannot silently alias two source pages.

## Replaying in HBFSim

```bash
python3 -B tools/replay_astra_trace.py \
    --trace out/llm_hbf.0.trace \
    --workload-manifest ../astra-sim/out/llm.manifest.json \
    --out-dir out/astra-llm

python3 -B tools/check_ec_sanity.py \
    --trace out/llm_hbf.0.trace \
    --replay-dir out/astra-llm
```

Independent configuration groups can be replayed concurrently with
`--jobs N` (default `1`). Each worker owns a separate temporary summary; the
runner waits for every worker, validates the complete timing contract, and
publishes no summaries or manifest if any worker fails. Choose `N` according
to host memory because every worker owns a simulator instance and trace state;
`--jobs 2` is the conservative starting point for long-context traces.

Placement is derived from `address_space.kv.base` in the generator manifest:
weights remain on the lower/HBM side and KV, activations, and communication
buffers occupy the upper/HBF side. The regions scale with the selected model
and context length, so replay no longer guesses a fixed 8 GiB boundary.
`--boundary` may override the derived split for a named experiment, but the
override and original KV base are both recorded. EC6 does not use this
boundary for residency; it counts unique pages.

Replay summaries and CSV are published only after every selected config
succeeds. A manifest written last binds the exact case set, trace digest,
window, dynamic boundary, workload-manifest digest, workload ID, trace digest,
and artifact digests. The trace header and generator manifest must carry the
same 64-hex workload ID or replay fails before simulation. `check_ec_sanity.py`
rejects missing, mixed-generation, or mutated inputs and outputs. Any case
subset may be selected; scenarios that share a config are coalesced into one
simulator invocation when selected together. Use a fresh output directory for
each research run even though the tool also locks concurrent writers.

Replay accepts only generator-manifest schema
`astra-sim.llm-fine-grained-workload` version 3 and identity contract v2. It
verifies the named model-profile file and digest, recomputes full-model
parameters, every resident weight category, per-rank totals, KV step/capacity,
initial KV state, and embedding/decoder/final-norm/LM-head address objects.
It then recomputes the workload ID from the generator digest, profile
name/digest, GPU-config digest, normalized parameters, and collective plan ID;
verifies every ET/plan artifact digest and size; validates the plan's canonical
content identity; scans the trace once to verify its header request/byte census
while computing its SHA-256; and only then launches EC simulations. Per-rank
manifest counters are explicitly compute-node-only; the trace header is the
exact runtime census including collective data and timing-dependent semaphore
retry traffic.

## Model-coupling caveats (read before trusting numbers)

1. **Open-loop pacing (mitigated by the closed-loop window).** The trace's
   arrival times were produced under ASTRA-sim's memory model (paper §5.1
   generic GPU: 4 TiB/s of HBM channels, 120 ns access latency); there is no
   feedback from HBFSim latency to ASTRA-sim issue timing. Since 2026-07-03,
   `scenario_compare --max-outstanding-requests W` composes with `at=`: the
   trace timestamp is the parent op's *earliest* issue time. HBFSim splits the
   parent at 4 KiB media-page boundaries before admission; each physical page
   transaction consumes one of `W` credits until it completes. Layer
   streaming applies the same numeric limit independently to its foreground
   HBM transactions and to a backing-tier pool shared by DMA reads and
   writebacks. Thus HBM and HBF can overlap, but the layer path cannot gain
   unbounded HBF MLP relative to all-HBF. This prevents
   a 256 KiB trace record from receiving 64 times the hidden concurrency of
   equivalent 4 KiB records. When full, the earliest child completion returns
   credit, independent of parent issue order.
   Recommended for physical realism; W should mirror a GPU-side in-flight
   page-transaction budget. A request-count limit from ASTRA must be converted
   using the exported request-size distribution rather than copied blindly.
   The replay/check
   tools consistently default to `W=512` and `out/astra-replay-w512`;
   `scenario_compare` itself defaults to `W=0`, meaning unbounded offered
   arrivals. EC6 always obeys both credit pools, layer readiness, two finite
   HBM buffers, D2D bandwidth, and dirty writeback dependencies; weights and
   cold/overflow KV remain backed while hot KV is bounded by physical HBM.
   Inspect schema-v16
   `layer_streaming`, `layer_streaming_controller`, and resource-utilization
   fields rather than interpreting `W=0` as infinite throughput.
2. **Lossless cache-line granularity.** Each ASTRA Wavefront Request is one
   aligned target-cache-line trace operation. Duplicates are preserved and
   request counts are not changed by coalescing. The exporter uses a bounded
   time-watermark heap, so its memory tracks only the NoC reordering window;
   text size and simulation work still scale linearly with request count.
3. **Fine-grained overlap.** Dependency-ready fine-grained compute and
   collective kernels can coexist and contend for CUs, memory, and I/O.
   Explicit workgroup/wavefront state enforces residency, outstanding-request,
   barrier, semaphore, and memory-dependency rules. This is not a cycle-level
   instruction pipeline or cache hierarchy. Data requests and semaphore
   acquire/release traffic share the same per-CU outstanding credits; a
   semaphore retry read also consumes a CU issue cycle, so synchronization
   cannot bypass the configured window.
4. **Validated multi-GPU subset.** A 4-GPU relative-buffer ring exercises
   put/signal/wait, two-ended scale-up I/O occupancy, completion, and semaphore
   quiescence. The full production MSCCL++ schema, per-mesh-link queues,
   InfraGraph, and backend toolchain remain outside this reference path.

The trace identity does not embed the ASTRA-sim executable or source revision.
Research results must archive the exact sibling Git commit/build in addition
to the workload manifest and trace digests. A valid manifest proves its inputs,
not which unrecorded binary executed them.
