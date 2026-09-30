# HBFSim front door

> Status: Current
> Last reviewed: 2026-09-28

`hbfsim` is the entry point for people who want to use the simulator: build
it, find configurations by name, run and sweep scenarios, and read results as
a few comparable numbers. It uses only the Python standard library and runs
from a fresh clone (`python3 -m hbfsim`) or after `pip install -e .`.

It is deliberately thin. It orchestrates the compiled programs and reads
their receipts; it adds no model behavior and never changes a configuration
silently. Every run directory records the exact simulator command in
`command.txt`, so any result can be reproduced without Python.

## Commands

| Command | Purpose |
| --- | --- |
| `doctor` | Check Python, CMake, the compiler, the build and optional integrations |
| `build [--all] [--type Debug]` | Configure `build/` once and compile the simulator programs (`--all` adds tests and probes) |
| `quickstart` | Build if needed, compare the five core placements on the default system, explain the output |
| `list [systems\|overlays\|policies\|scenarios\|metrics]` | Discover shipped profiles and what they mean |
| `run` | Run scenarios on one configuration (`--system`, `--overlay`, `--set KEY=VALUE`, `--trace`) |
| `sweep --vary KEY=V1,V2` | Run a cartesian grid on one trace; writes `sweep.csv` and `sweep.json` |
| `show PATH...` | Tabulate existing `summary.json` files or run directories |

Tables print as text, Markdown, CSV or JSON (`--format`); choose columns with
`--metrics`. Environment variables `HBFSIM_ROOT` and `HBFSIM_BUILD_DIR`
select another checkout or build directory.

## Python API

```python
import hbfsim

result = hbfsim.run(system="server-hbm128-hbf512", scenarios="all-hbm,all-hbf",
                    overlays=["ocp-v070-grade3"], options={"hbf-read-ns": 8000})
print(result.table())
print(result.scenario("all-hbf")["p95_latency_us"])

points = hbfsim.sweep(vary={"hbf-read-ns": [2000, 4000, 8000]}, scenarios="all-hbf")

with hbfsim.open_session("4hbm-4hbf") as session:  # drive the engine with your own policy
    ...
```

## Files

| File | Responsibility |
| --- | --- |
| `__main__.py` | `python3 -m hbfsim` entry point |
| `cli.py` | Argument parsing, command handlers and user-facing messages |
| `workspace.py` | Locate the checkout, build directory and programs; `doctor` checks; the CMake build wrapper |
| `catalog.py` | Discover `configs/` profiles, resolve short names with suggestions, reference-scenario descriptions |
| `runner.py` | One reference-runner invocation per fresh directory, the demo trace, parameter sweeps, error hints |
| `results.py` | Headline metrics from summary JSON and table rendering |
| `session.py` | `open_session`: a persistent engine session by profile name for Python policies |

Behavior is covered by `tests/python/test_hbfsim_frontdoor.py` (CTest
`hbfsim_frontdoor_contract`), and every script in [`examples/`](../examples/README.md)
runs as its own CTest case. The scenario list is checked against
`src/app/reference_runner.cpp`, and the summary schema version against
`reports/address_heatmap.py`.
