# Project layout

> Status: Current
> Last reviewed: 2026-09-29

The repository is organized by responsibility. A file has one obvious home,
and moving a responsibility means updating its callers rather than leaving
compatibility wrappers behind.

| Path | Responsibility | Typical user |
| --- | --- | --- |
| `hbfsim/` | Front door: build, discover configs, run, sweep and tabulate; `open_session` for Python policies | all users |
| `examples/` | Small runnable scripts, one question each; every script is a CTest case | new users |
| `src/app/` | The `hbfsim` engine CLI and the `hbfsim-reference` runner | simulator users |
| `src/physical/` | Semantic-free devices, links, and transaction-session execution | model developers |
| `src/host/` | The host-managed HBF controller: mapping, write buffer, GC, wear, zones, persistent images | model developers |
| `src/policies/` | Optional controller support and reference placement policies | policy authors |
| `configs/systems/` | Complete runnable system profiles | simulator users |
| `configs/policies/reference/` | Reference-runner policy profiles paired with the systems | policy authors |
| `configs/overlays/` | One-purpose overrides applied after a system profile | everyone running a sweep |
| `configs/parameter-provenance.json` | The evidence grade and source of every physical key | contributors |
| `hbfsim_client/` | Semantic-free transaction protocol and persistent simulator client | frontend developers |
| `workloads/` | Synthetic trace generation, locality analysis and workload qualification | workload authors |
| `reports/` | Summary comparison and visualization | results readers |
| `verification/core/` | Shared contracts, ledgers, and certificates | verification developers |
| `verification/oracles/` | Independent reference models | verification developers |
| `verification/gates/` | Executable checks | CI and release owners |
| `verification/probes/` | Small compiled observability witnesses | model developers |
| `verification/cases/` | Input-only canonical cases | verification developers |
| `evidence/` | External-simulator pins and the hardware measurements behind calibrated overlays | evidence owners |
| `tests/cpp/`, `tests/python/`, `tests/fixtures/` | Unit, integration, and regression tests and their small configs | all contributors |
| `docs/` | Guides, reference and the verification coverage map | all users |
| `cmake/` | Build targets and grouped test registration | build maintainers |
| `.github/`, `.devcontainer/`, `CMakePresets.json` | CI, issue and pull-request templates, dev container, IDE build presets | contributors |

The LLM-serving frontend is the separate
[ServeLoop repository](https://github.com/FlashLab-MBZUAI/ServeLoop). It
compiles serving iterations into transactions for `build/hbfsim`; the physical
engine knows nothing about models, requests or KV caches. A compatible
ServeLoop checkout is needed only for the `serveloop`-labelled tests and the
[ServeLoop guide](guides/serveloop.md); `pip install -e .` does not supply it.

There are two compiled programs. `build/hbfsim` is a physical simulation
session that accepts arbitrary transaction DAGs on stdin and owns no scenario
IDs. `build/hbfsim-reference` replays traces through the maintained reference
suite (`all-hbm`, `all-hbf`, `flat`, `direct-read`, `hbf-streaming`,
`external-streaming`, `demand-fill`, and `reuse-filtered`).

Users start at the front door, `python3 -m hbfsim` (see [getting started](getting-started.md)).
It only orchestrates the compiled programs; each result directory records the
exact simulator command. The other Python entry points are invoked by their
responsibility-based path:

```bash
python3 -m hbfsim --help
python3 examples/01_first_comparison.py --help
python3 workloads/synthetic/generate.py --help
python3 workloads/trace_analysis.py --help
python3 reports/time_dashboard.py --help
python3 verification/gates/provenance.py --help
```

Package imports follow the same layout, such as
`from hbfsim import run, sweep` and `from reports.time_breakdown import SummaryInput`.
Internal modules under `verification/core/` and `verification/oracles/` are
imported by gates; they are not separate command-line programs.

When adding code, prefer extending the narrowest existing responsibility. Do
not create a generic `tools/`, `utils/`, or second validation tree. A new
top-level directory needs a distinct audience and lifecycle that none of the
existing directories can express: `hbfsim/` serves people using the
simulator rather than developing it, and `examples/` serves first-time readers
with scripts that must stay short. Research studies built on HBFSim belong in
their own repositories; this one ships the simulator, its verification and the
evidence behind its calibrated numbers.
