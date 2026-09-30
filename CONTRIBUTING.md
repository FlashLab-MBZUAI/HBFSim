# Contributing to HBFSim

> Status: Current
> Last reviewed: 2026-09-29

Thank you for helping build an open simulator for high-bandwidth flash. HBFSim
aims to be a trustworthy shared playground: contributions are judged first on
causal correctness, reproducibility and clarity of evidence — a faster or more
favorable result is not by itself an improvement. Everyone participating
follows the [Code of Conduct](CODE_OF_CONDUCT.md), and contributions must be
compatible with the repository's [MIT License](LICENSE).

## Ways to contribute

| Contribution | Effort | Where it goes |
| --- | --- | --- |
| Report a bug or a confusing result | minutes | an issue ("Bug report") |
| Share a study, result, or open question | an afternoon | an issue ("Share a study, result, or question") |
| Improve docs, fix a typo, translate | small | `docs/`, READMEs |
| Add a runnable example | small | `examples/` (registered in CTest automatically) |
| Add a configuration overlay for a device or design point | small | `configs/overlays/` + provenance entry |
| Add a workload generator or trace importer | medium | `workloads/` |
| Add a reference placement policy | medium | `src/policies/reference/` |
| Change the physical model | large | `src/physical/`, `src/host/` — read [Physical-model changes](#physical-model-changes) |

[Extending HBFSim](docs/guides/extending.md) explains each extension point.
For anything large, open an issue first so the design can be agreed before
you invest in code.

**Good first contributions:** an example answering a question you had, an
overlay for a device you know (with sources), a clearer error hint in
`hbfsim/runner.py`, a missing metric in `hbfsim/results.py`, or a
documentation page that confused you on first read.

## Development setup

You need CMake 3.20+, a C++20 compiler and Python 3.10+.

```bash
python3 -m hbfsim doctor
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --build build --target verify_physical
```

CMake presets (`cmake --preset debug`, `release`, `sanitize`; CMake 3.21+)
give the same configurations and are picked up by VS Code and CLion; the dev
container builds everything on creation. `python3 -m hbfsim build --all
--type Debug` is equivalent to the first two CMake commands.

ServeLoop-dependent tests need a checkout next to HBFSim
(`git clone https://github.com/FlashLab-MBZUAI/ServeLoop.git ../ServeLoop`)
and `PYTHONPATH=../ServeLoop:$PWD`; CI pins the revision in
`.github/workflows/ci.yml`. Without it those tests fail with
`No module named 'hbserve'`; everything else runs.

For timing, mapping, FTL, composition, or accounting changes, also run:

```bash
cmake --build build --target verify_components
cmake --build build --target verify_write_amplification
```

## Everyday conventions

- Follow the surrounding code: 4-space indentation, descriptive names, and the
  repository's `.editorconfig`.
- Put code where its responsibility lives, as described in the
  [project layout](docs/project-layout.md); there is deliberately no generic
  top-level `tools/` or `utils/` directory.
- Keep one current implementation path. When replacing a design, remove the
  obsolete implementation, flags, tests, and documentation in the same change.
- Every maintained Markdown file starts with `> Status:` and
  `> Last reviewed:` lines and is linked from [docs/README.md](docs/README.md);
  the `documentation_contract` test checks this, plus local links and
  directory catalogs.
- Keep generated traces and results under `out/`; do not commit them as
  hidden scientific inputs, and do not link to them from documentation.

## Physical-model changes

Before changing a timing, placement, FTL, composition, or accounting
mechanism, read the [evidence policy](docs/reference/evidence-policy.md), the
[model reference](docs/reference/model.md), and the
[coverage map](docs/verification/coverage.md). Then:

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
- Never update a numeric baseline merely to make a changed result pass.
  Explain the physical reason for drift and encode an invariant or justified
  tolerance.

## Reproducible experiments

Archive the summary JSON with results. It records the source revision and
dirty state, build identity, invocation, trace SHA-256, full resolved config,
and scenario metrics. Also archive or identify the initial memory image and
any external workload-generator revision. Report `user_completion_throughput_GBps`
and `makespan_throughput_GBps` separately when background work exists.

## Pull request checklist

The pull request template repeats this list.

- [ ] `ctest` passes, or failing tests are listed with the reason.
- [ ] New behavior has a focused test or runnable example.
- [ ] No generated output, machine-specific path, or private dependency was
      added.

For physical-model changes, additionally:

- [ ] The mechanism and evidence grade are documented.
- [ ] New parameters have provenance and sensitivity guidance.
- [ ] Invalid inputs fail closed.
- [ ] `verify_physical`, `verify_components`, `verify_write_amplification`,
      and relevant sanitizers pass.
- [ ] Obsolete paths and stale claims are removed.
