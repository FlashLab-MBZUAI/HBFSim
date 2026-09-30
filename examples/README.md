# Examples

> Status: Current
> Last reviewed: 2026-09-29

Small, self-contained scripts that each answer one question about an HBM +
HBF memory system. They need only a built simulator (`python3 -m hbfsim
build`) and Python 3.10+ with no extra packages, run in seconds, and write
everything they produce into a fresh `out/` directory. CTest runs every
example, so they stay in step with the simulator.

```bash
python3 examples/01_first_comparison.py
```

| Example | Question | You learn to |
| --- | --- | --- |
| [`01_first_comparison.py`](01_first_comparison.py) | How much slower is HBF than HBM, and where do hybrid placements land? | run the reference scenarios and read numbers out of a result |
| [`02_ftl_mapping_tradeoffs.py`](02_ftl_mapping_tradeoffs.py) | What do different flash-translation mapping strategies cost in latency, write amplification and wear? | sweep configuration overlays |
| [`03_read_bandwidth_ceiling.py`](03_read_bandwidth_ceiling.py) | Is HBF read bandwidth limited by the interface speed grade or by the NAND array? | drive the engine directly with `hbfsim.open_session` |
| [`04_custom_trace.py`](04_custom_trace.py) | How do placements behave on *my* access pattern? | write a trace file with semantic labels and replay it |
| [`05_custom_policy.py`](05_custom_policy.py) | Does my caching/promotion idea help? | implement a placement policy as a Python transaction DAG |
| [`06_capacity_tier_media.py`](06_capacity_tier_media.py) | How does HBF compare with host DRAM, CXL memory, LPDDR or SSDs as the tier behind HBM? | compare external backing devices |

Every script accepts `--help` and `--out DIR`. They run from a fresh clone
without installing anything; `pip install -e .` makes `import hbfsim` work
from anywhere.

## From example to study

The examples use small synthetic workloads so they finish in seconds. Before
drawing conclusions:

- replace the synthetic trace with a real one (example 4) or with LLM-serving
  traffic from [ServeLoop](../docs/guides/serveloop.md);
- read the simulator's warnings — they flag, for example, an arrival rate
  that exceeds a device's link, where latency reflects queueing;
- check the evidence grade of every parameter you rely on in
  [`configs/parameter-provenance.json`](../configs/parameter-provenance.json).

[Getting started](../docs/getting-started.md) explains the concepts behind
these scripts, and [Extending HBFSim](../docs/guides/extending.md) shows where
a policy prototyped in example 5 goes next.
