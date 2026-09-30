# Extending HBFSim

> Status: Current
> Last reviewed: 2026-09-29

HBFSim is built so that most new ideas never touch the physical model. This
guide orders the extension points from least to most effort; start at the top
and move down only when an idea needs it.

| You want to… | Extension point | Language | Touches the engine? |
| --- | --- | --- | --- |
| try other hardware parameters | a configuration overlay | `key=value` | no |
| study another workload | a trace file or generator | any / Python | no |
| test a placement, caching, migration or prefetch idea | a Python policy on the session API | Python | no |
| ship a policy for trace replay | a reference policy | C++ | no (links against it) |
| add a report or metric | summary readers | Python | no |
| model a new physical mechanism | the core engine | C++ | yes — read the rules below first |

## 1. Configurations

A configuration file is `key=value` lines with `#` comments, applied after a
system profile. Keys are the long option names without `--`, so any file value
can also be tried as `--set KEY=VALUE` first.

```ini
# my-overlay.cfg — a faster NAND die with a larger write buffer.
hbf-read-ns=2000
hbf-program-ns=50000
hbf-write-buffer-pages=4096
```

```bash
python3 -m hbfsim run --overlay path/to/my-overlay.cfg
```

Personal overlays can live anywhere. To publish one under `configs/`, give it
a header comment saying what it isolates and where its values come from, list
it in the [configuration catalog](../../configs/README.md), and give any **new**
physical key an entry in `configs/parameter-provenance.json` — the
`parameter_provenance_coverage` test fails otherwise. Values that close an
unpublished specification are fine; label them `exploratory_assumption`.

## 2. Workloads

The reference runner reads text traces, one memory operation per line:

```text
<address> <R|W> [bytes] [kind] [phase=N] [layer=N] [compute_ns=N] [at=NS]
```

- `kind` is one of `model_weights`, `shared_context`, `generated_context`,
  `scratch`, `metadata`; semantic policies use it (for example, `direct-read`
  keeps mutable kinds in HBM).
- `layer=N` numbers layer executions in order and never decreases; streaming
  policies prefetch layer N+1 while N runs. Label every request or none.
- `at=NS` gives an explicit arrival time; without it, requests arrive every
  `interarrival-ns`.

[Example 4](../../examples/04_custom_trace.py) writes a decode trace from
Python. `workloads/synthetic/generate.py` makes sequential, random, strided
and hotspot patterns with explicit read/write mixes, and the
[workload catalog](../../workloads/README.md) covers locality analysis and
quality checks. For LLM serving with continuous batching and
paged KV caches, use [ServeLoop](serveloop.md), which drives the engine directly.

## 3. Policies in Python

The engine executes an arbitrary DAG of transactions. A policy is code that
decides, for each application access, which transactions to issue:

```python
import hbfsim
from hbfsim_client import Transaction

with hbfsim.open_session("4hbm-4hbf", overlays=["ocp-v070-grade2"],
                         initial_hbf_logical_pages=1024) as session:
    miss = Transaction(id="miss", target="HBF_LOGICAL", op="R",
                       addr=0x2000, bytes=4096, issue_ns=0.0)
    fill = Transaction(id="fill", target="HBM", op="W", addr=0,
                       bytes=4096, issue_ns=0.0, dependencies=("miss",))
    result = session.run([miss, fill])
    print(result.completion("fill").finish_ns)
```

- **Targets** are listed in the [concepts guide](../concepts.md#transaction-targets):
  HBM, HBF through the FTL, static or raw flash, base-die link copies,
  external backing, direct lanes, and barriers.
- **Dependencies** order transactions: a transaction starts after all of its
  dependencies finish, never before its `issue_ns`.
- **Batches** are units of submission. `session.run(...)` numbers them; by
  default every transaction gates the next batch (`frontier`), and ids a
  later batch may depend on must be kept with `retain`. Ids are unique within
  the retained set.
- **Results** give each transaction's arrival, start and finish, per-target
  latency statistics in `result.receipt`, and cumulative device state.
- `open_session` resolves profile names, writes `options` to an
  `overrides.cfg`, and keeps wear reports in the session's `out/` directory.
  `initial_hbf_logical_pages=N` starts with N logical pages already written.

[Example 5](../../examples/05_custom_policy.py) is a complete promotion cache
with LRU eviction and hazard-safe fills. Because the policy only chooses
transactions, its measured behavior uses the same physical model as every
other result. The protocol is specified in [hbfsim_client](../../hbfsim_client/README.md)
and the [observability reference](../reference/observability.md).

## 4. Reference policies in C++

Once a policy should replay traces for everyone, it can join the reference
suite in `src/policies/reference/`:

1. Implement it next to `direct_policy`, `layer_streaming_policy` and
   `behavioral_tiering_policy`, depending only on `hbfsim_core` and
   `policies/policy_common` — the core must never include policies.
2. Register a scenario name in `src/app/reference_runner.cpp` beside the
   existing `k…Scenario` constants and their dispatch table.
3. Add a focused test under `tests/cpp/` and register it in
   `cmake/TestTargets.cmake` and a `cmake/tests/*.cmake` group.
4. Document the scenario everywhere the documentation gate checks (the
   README table, [scenario reference](../reference/scenarios.md),
   [model](../reference/model.md), [observability](../reference/observability.md),
   [project layout](../project-layout.md)), and add it to
   `hbfsim/catalog.py` so the front door lists it.

## 5. Reports and metrics

Every run writes `summary.json` (schema `hbfsim.simulation.summary`). Read it
with `hbfsim.load_summary(path)`: each scenario exposes the front door's
headline metrics and the complete raw row. To add a headline metric, append a
`Metric` to `hbfsim/results.py`; general renderers belong in
[`reports/`](../../reports/README.md), study-specific figures next to their
experiment.

## 6. Physical mechanisms

Changing timing, placement mechanics, FTL, composition or accounting in
`src/physical/` or `src/host/` carries the strictest rules, because every
result in the project depends on them:

- state the mechanism and its evidence grade; add provenance for new
  parameters;
- preserve causality — no state becomes visible before its modeled
  completion;
- fail closed on invalid capacities, ranges and transitions;
- add a focused regression that fails on the old behavior, and run
  `verify_physical`, `verify_components` and `verify_write_amplification`;
- replace, don't accumulate: remove obsolete paths, flags, tests and docs in
  the same change.

[CONTRIBUTING.md](../../CONTRIBUTING.md) has the full checklist and the
[verification framework](../../verification/README.md) explains the gates.
Opening an issue with the "Feature or model request" template before a large
change is the fastest way to agree on the design.

## Share what you find

A study you ran is a contribution too. The "Share a study, result, or
question" issue template asks for the question, the exact commands and the
table (`python3 -m hbfsim show … --format markdown`), so others can
reproduce and build on it.
