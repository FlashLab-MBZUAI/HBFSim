# Contributing to HBFSim

HBFSim aims to become a trustworthy open simulator for studying HBF under AI
workloads. Contributions are evaluated first on causal correctness,
reproducibility, and clarity of evidence; a faster or more favorable result is
not by itself an improvement.

## Before Contributing

Read `docs/model-validation.md`, `docs/physical-model-status.md`, and
`docs/verification-map.md`. The repository does not yet contain a public
software license. The copyright owner must choose and add one before accepting
outside code contributions; until then, use issues or discussions for design
feedback rather than submitting reusable code.

## Development Setup

HBFSim requires CMake 3.20+, a C++20 compiler, and Python 3.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
cmake --build build --target physical_guard
```

For timing, mapping, FTL, composition, or accounting changes, also run:

```bash
cmake --build build --target use_cases
cmake --build build --target waf_cases
node web/architecture-editor/tools/design-sync.mjs
```

## Change Requirements

- State the modeled mechanism and its trust boundary. Distinguish a vendor
  fact, literature-derived design, exploratory assumption, and implementation
  policy.
- Add or update `configs/parameter-provenance.json` for every new physical
  default. Link primary sources where available; do not promote a secondary
  citation or an inferred value to a measured fact.
- Preserve event causality. Future reservations must not make mapping, page,
  buffer, or version state visible before modeled completion.
- Reject invalid capacity, ranges, overflow, non-finite values, and impossible
  state transitions instead of returning plausible-looking results.
- Add a focused regression that fails on the old behavior. Prefer exact
  conservation, hand-computed timing, sharp regimes, theoretical bounds, or
  cross-path equivalence over broad output snapshots.
- Keep one current implementation path. When replacing a design, remove the
  obsolete implementation, flags, tests, and documentation in the same change.
- Never update a numeric baseline merely to make a changed result pass. Explain
  the physical reason for drift and encode an invariant or justified tolerance.
- Keep generated traces and results under `out/`; do not commit them as hidden
  scientific inputs.

## Reproducible Experiments

Archive the summary JSON with results. It records the source revision and dirty
state, build identity, invocation, trace SHA-256, full resolved config, and
scenario metrics. Also archive or identify the initial memory image and any
external workload-generator revision. Report `user_ack_throughput_GBps` and
`makespan_throughput_GBps` separately when background work exists.

## Pull Request Checklist

- [ ] The mechanism and evidence grade are documented.
- [ ] New parameters have provenance and sensitivity guidance.
- [ ] Invalid inputs fail closed.
- [ ] A focused regression covers the defect or mechanism.
- [ ] `ctest`, `physical_guard`, `use_cases`, and relevant sanitizers pass.
- [ ] Architecture code references pass `design-sync.mjs` when source symbols
      or diagrams change.
- [ ] Obsolete paths and stale claims are removed.
- [ ] No generated output, machine-specific path, or private dependency was
      added.
