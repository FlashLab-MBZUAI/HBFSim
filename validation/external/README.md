# External differential evidence

This directory defines HBFSim's only external-reference boundary. It does not
vendor, patch, or silently download either reference simulator.

`tools.json` schema v2 pins the full commit and source-tree IDs, dependency
commits, exact case census, shared metrics, shared boundary, and exclusions for
each tool. Every HBFSim case, reference-tool config, workload, and trace named
by that manifest is a tracked certificate input. The fixtures validate adapter
translation only; fixture PASS never counts as L3 agreement.

## Validated facets

The current L3 matrix has six required cases and only two facets:

| Tool | Facet | Required cases | Compared result |
|---|---|---:|---|
| Ramulator2 | `dram.read-row-state-service` | read miss, hit-after-miss, conflict-after-miss | exact read/write bytes, row classification, and zero-arrival service span |
| MQSim | `flash.direct-data-page-io` | first data write, write then read, overwrite | exact direct data-page reads/programs and bytes |

The Ramulator2 boundary is one channel/rank/bank-group/bank, 64-byte
transactions, DDR4-3200W timing, open-row FR-FCFS, and refresh disabled.
Ramulator2 increments its controller clock before the first issue, so the raw
wrapper preserves both `request_epoch_cycle=1` and `completion_cycle`; the
adapter compares `(completion_cycle - request_epoch_cycle) * 0.625 ns`. It
does not silently subtract a cycle from the raw result.

The MQSim boundary is one channel/chip/die/plane, 512-byte full-page SLC
requests, cache disabled, ideal in-memory mapping, and no preconditioning,
mapping-page I/O, GC, wear leveling, or erase. MQSim's end-to-end completion
time includes host/NVMe/controller layers that do not match HBFSim's HBF
pipeline and is deliberately not compared. Observing any GC execution or
erase makes this facet fail rather than broadening the claim.

Consequently, L3 does not currently cover DRAM writes, turnaround, refresh,
multi-bank scheduling, HBM topology, flash timing, partial-page coalescing,
mapping checkpoints, GC, relocation, or erase.

## Produce actual evidence

Check out the two upstream repositories at the exact commits in `tools.json`,
including Ramulator2's pinned `fmt` and `yaml-cpp` dependencies. Their tracked
trees must be clean. Then run:

```bash
python3 -B validation/run_actual_external_differential.py \
  --hbfsim-probe build/validation_probe \
  --ramulator-root /path/to/ramulator2 \
  --mqsim-root /path/to/MQSim \
  --output-dir /path/to/new-evidence \
  --cxx g++-16 \
  --parallel 4
```

The runner verifies every Git pin, exports each source with `git archive`,
builds unmodified sources in temporary directories, runs all six cases, copies
the exact HBFSim probe into the bundle, records byte counts and SHA-256 digests
for every artifact, and self-verifies the finished bundle before atomically
publishing it. It refuses to overwrite an existing output directory. On
macOS, GCC is required for these pinned upstream revisions.

`hbfsim.validation.external-evidence` schema v2 contains one source-provenance
record, compiler/build commands, exact build artifacts, the HBFSim probe, and
an ordered list of every manifest case. Each case binds its commands, current
repository inputs, external raw/tool output, complete HBFSim ledger, and run
log. `hbfsim.validation.external-wrapper-output` schema v2 preserves raw
tool-specific counters; normalization happens only inside the verifier.

Recheck a bundle independently:

```bash
python3 -B validation/external_differential.py \
  --manifest validation/external/tools.json \
  --fixture-dir validation/external/fixtures \
  --evidence-dir /path/to/evidence \
  --repository . \
  --hbfsim-probe build/validation_probe \
  --report build/foundational-external-report.json
```

With no `--evidence-dir`, the command checks the two adapters and reports L3
`not_run`. If a directory is supplied, a missing tool bundle yields `partial`;
malformed schema, missing cases, changed source tree, changed probe, stale
repository input, bad digest, out-of-facet work, or any numerical mismatch
fails closed.

To import the same evidence into the full certificate path, configure:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHBFSIM_EXTERNAL_EVIDENCE_DIR=/path/to/evidence
```

The certificate records the exact source trees, six-case census, metrics, two
validated facets, and all exclusions. It never promotes those facets into a
whole-DRAM, whole-flash, hardware-calibration, or HBF-product claim.
