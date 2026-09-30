# ServeLoop with HBFSim

> Status: Current
> Last reviewed: 2026-09-29

[ServeLoop](https://github.com/FlashLab-MBZUAI/ServeLoop) is the LLM-serving
frontend for HBFSim. Its Python package and console command are named
`hbserve`. It turns a model descriptor and a request stream into
continuous-batching iterations, paged-KV lifecycles and byte-exact memory
transactions, submits them to `build/hbfsim` through the
[`hbfsim_client`](../../hbfsim_client/README.md) protocol, and closes the loop
with the engine's completion times. HBFSim itself owns no model, request or KV
semantics: the engine sees transactions, the frontend sees latency.

## Setup

Use a ServeLoop source checkout next to HBFSim. Its `hbserve/`, `models/`,
`configs/` and `examples/` directories are needed; HBFSim's own
`hbfsim_client` stays first on `PYTHONPATH` because it is the protocol source
of truth:

```sh
git clone https://github.com/FlashLab-MBZUAI/ServeLoop.git ../ServeLoop
export PYTHONPATH="$PWD:../ServeLoop${PYTHONPATH:+:$PYTHONPATH}"
python3 -B -m hbserve --help
```

HBFSim's `serveloop`-labelled tests need a checkout at or after the revision
named by `HBFSIM_SERVELOOP_MIN_REVISION` in `CMakeLists.txt`; the
`serveloop_checkout_compatible` fixture reports an older or missing checkout in
one line and the labelled tests are Not Run. Point CMake at another location
with `-DHBFSIM_SERVELOOP_SOURCE_DIR=...`.

## Closed-loop requests

```sh
python3 -B -m hbserve run \
  --model ../ServeLoop/models/llama31-8b-w8-kv-bf16.json \
  --system configs/systems/miniquick/4hbm-4hbf.cfg \
  --requests ../ServeLoop/examples/quickstart-requests.json \
  --placement weights-hbf-kv-hbm --simulator build/hbfsim \
  --out out/serveloop-quickstart
```

`--system` accepts any HBFSim system profile, and overlays or `KEY=VALUE`
overrides can be added the same way the front door applies them. Admission
and scheduling use simulator completion feedback. Reported service timings
depend on the declared compute provider: `roofline` includes explicit compute
assumptions, `memory_only` is a memory critical-path experiment, and only a
measured operator profile makes TTFT/TPOT figures comparable with hardware.
Model quality and unsupported hardware features are not established by a
successful run. ServeLoop's own documentation covers model descriptors,
request formats, prefix caching, placement policies and its experiment modes.

## What crosses the boundary

- **Transactions**: HBM, HBF-through-the-FTL, static flash, base-die link
  copies, external backing and barriers, with dependencies and issue times.
  The [concepts guide](../concepts.md#transaction-targets) lists the targets.
- **Receipts**: per-transaction arrival, start and finish, per-target latency
  statistics, device state and the wear reports of a persistent session.
- **System configuration**: ServeLoop resolves ratio-sized geometry through
  `hbfsim --describe-system` exactly as `hbfsim.open_session` does, so a
  serving run and a trace replay on the same profile see the same device.

Everything else (schedulers, KV block managers, placement heuristics, compute
models) lives in ServeLoop. A memory-management idea that only needs
transactions can also be prototyped directly against the engine; [example 5](../../examples/05_custom_policy.py)
shows the pattern.

## KV writeback during compute

ServeLoop's tiered placement buffers appended KV in a finite HBM cache and
declares timed compute windows in its receipts. With a timed compiler it uses
sufficiently long windows to persist older dirty KV pages to HBF as ordinary
HBM read → link copy → HBF logical write transactions; short windows leave the
pages buffered, and capacity pressure still forces dirty eviction through the
same copy path. The engine schedules the physical planes after FTL mapping and
charges same-plane program/erase exclusion and the shared channel, ECC, TSV
and thermal limits; the frontend never infers a plane from a logical address.

Native write completions must include programming: set
`hbf-write-buffer-completion-requires-flush=true` when write coalescing is on,
or use `hbf-write-coalescing=false`. A buffered acknowledgement alone is not
evidence of a completed writeback.

## Limits

A ServeLoop run measures the memory system it was given. Capacity questions
(how much HBM versus HBF a model and batch need) are answered well; absolute
serving latency is only as good as the compute provider declared in the run,
and the physical timings carry the evidence grades recorded in
[`configs/parameter-provenance.json`](../../configs/parameter-provenance.json).
