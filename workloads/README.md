# Workload catalog

> Status: Current
> Last reviewed: 2026-09-29

Workload code produces and inspects the memory-access traces that the
reference runner replays. It never changes physical timing or picks a result:
a trace is a list of accesses, and the simulator decides what they cost.

| Module | Responsibility |
| --- | --- |
| `synthetic/generate.py` | Deterministic sequential, random, strided and hotspot traces with explicit read/write mixes and semantic labels |
| `trace_analysis.py` | Spans, locality, reuse distance and size distributions of any trace, including TiB-scale span traces counted by interval |
| `quality.py` | Fail-closed qualification of a trace against declared footprint, span-inflation, reuse and working-set expectations |

```bash
python3 workloads/synthetic/generate.py --output out/hot.trace --pattern hotspot \
  --pages 700 --ops 20000 --read-percent 0 --hot-page-percent 10 --hot-access-percent 90 --seed 1
python3 workloads/trace_analysis.py out/hot.trace
python3 -m hbfsim run --system 4hbm-4hbf --policy none --overlay endurance-regime-scaled \
  --trace out/hot.trace --scenarios all-hbf
```

The trace format is documented in [Extending HBFSim](../docs/guides/extending.md#2-workloads)
and the [workload replay guide](../docs/guides/workload-replay.md); the
[design-space recipes](../docs/guides/design-space.md) show which workload
fits which question. Real LLM-serving traffic, with continuous batching and
paged KV caches, comes from [ServeLoop](../docs/guides/serveloop.md), which
drives the engine directly instead of writing a trace.
