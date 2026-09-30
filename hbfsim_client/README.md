# HBFSim simulation client

> Status: Current
> Last reviewed: 2026-09-29

This package owns the semantic-free Python boundary to HBFSim's persistent
simulation engine (`build/hbfsim`). It is shared by ServeLoop and retained trace/evidence adapters; the physical
protocol owns no workload or scheduling implementation.

Install once from the repository root (`pip install -e .`) or run with the
repository root on `PYTHONPATH`. For policy prototypes, `hbfsim.open_session`
(see the [front door](../hbfsim/README.md)) resolves profile names and
returns the same `SimulationSession`.

## Quickstart

```python
from pathlib import Path
from hbfsim_client import ResolvedSystemConfig, SimulationSession, Transaction

config = ResolvedSystemConfig.load((
    Path("configs/systems/eight-stack-baseline.cfg"),
    Path("tests/fixtures/simulation-session-mini.cfg"),
))
reads = [
    Transaction(id=f"r{i}", target="HBF_LOGICAL", op="R",
                addr=i * 4096, bytes=4096, issue_ns=0.0)
    for i in range(100)
]
with SimulationSession(simulator_path=Path("build/hbfsim"),
                       system_config=config, enable_hbm=True,
                       enable_hbf=True) as session:
    result = session.run(reads)             # one batch, all reads gate the next
    print(result.elapsed_ns, result.completions[0].finish_ns)
    extra = Transaction(id="extra", target="HBF_LOGICAL", op="R",
                        addr=0, bytes=4096, issue_ns=0.0)
    session.run([extra], frontier=())       # detached work: origin only advances
    print(session.source_receipt()["engine_source"]["git_commit"])
```

`run()` numbers batches for you; `submit(TransactionBatch(...))` exposes the
full contract (upstream digests, a remap receipt, `frontier`, `retain`, and
`completions`). `retain` is the complete set of earlier ids a later batch may
still depend on: a tuple replaces the engine's retained set, `()` clears it,
and `None` (the default) leaves it unchanged. Every rejected batch raises
`SimulationSessionError` with the engine's message and leaves the session
usable.

Use `batch.with_receipt(updated_receipt)` when only report metadata changes.
It preserves the validated transaction graph, digest, encoded payload and
accounting cache; actual transaction changes must construct a new batch.
Wire text is encoded once into batch-local chunks shared by hashing and
submission, trading memory proportional to that batch for fewer traversals.

| File | Responsibility |
| --- | --- |
| `transaction_protocol.py` | Validated arbitrary transaction DAGs, batches (frontier/retain/completions options), HBF geometry, and wire serialization |
| `simulation_session.py` | Persistent engine process, explicit physical configs, fail-closed completion accounting, blocking/total batch frontiers, structured engine errors, fenced raw publication extents, quiescent HBF v2 image checkpoint export, read-only published reopen, fresh-process exact materialized or compact image restore, and explicit command-boundary crash termination without hidden drain |
| `provenance.py` | Source revision / dirty flag / tree hash, runner and config digests, and the `require_clean_tree` gate stamped into every study envelope |
| `fixed_footprint_metrics.py` | Reduce target latency and device traffic receipts to one completion metric plus per-medium, per-object, effective-trace-throughput, and explicit actual/logical media-traffic statistics for the shared reference matrix |

The package contains no model, request, batching, routing, or placement
policy. Frontends must resolve those semantics before constructing a
`TransactionBatch`. The wire protocol is documented in
`docs/reference/observability.md`.

HBF host zone management and automatic per-run wear reports are documented in
[host HBF management](../docs/reference/host-hbf-management.md). Reports show
physical P/E history and workload increments separately; GC copies include
host roundtrip cost.
