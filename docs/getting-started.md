# Getting started

> Status: Current
> Last reviewed: 2026-09-29

This tour takes you from a fresh clone to your own parameter sweep in about
fifteen minutes. It uses the `python3 -m hbfsim` front door throughout; every
step also shows what happens underneath, so nothing is hidden.

## 1. Install the prerequisites

HBFSim needs CMake 3.20+, a C++20 compiler and Python 3.10+. There are no
Python package dependencies.

| Platform | Command |
| --- | --- |
| Ubuntu / Debian | `sudo apt install build-essential cmake python3 git` (add `ninja-build` for faster builds) |
| Fedora | `sudo dnf install gcc-c++ cmake python3 git` |
| macOS | `xcode-select --install` and `brew install cmake python` |
| Windows | Use WSL 2 with Ubuntu, then follow the Ubuntu line |

No local setup at all? Open the repository in a dev container or GitHub
Codespace; [`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json)
builds the simulator when the container is created.

## 2. Check and build

```bash
git clone https://github.com/FlashLab-MBZUAI/HBFSim.git
cd HBFSim
python3 -m hbfsim doctor
python3 -m hbfsim build
```

`doctor` reports what is present and what is missing, with a fix for each
required item. `build` configures CMake in `build/` (Release) and compiles the
two simulator programs, `build/hbfsim` and `build/hbfsim-reference`; that is
all the front door, the examples and ServeLoop need. `python3 -m hbfsim build
--all` also compiles the tests and probes.

Prefer plain CMake or an IDE? `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
&& cmake --build build --parallel` does the same, and the
[CMake presets](../CMakePresets.json) (`release`, `debug`, `sanitize`, and a
`simulator` build preset) are picked up by VS Code and CLion.

Run all commands from the repository root. To use `import hbfsim` from other
directories, install the Python packages in editable mode with
`pip install -e .`.

## 3. Run your first comparison

```bash
python3 -m hbfsim quickstart
```

This replays a synthetic LLM-like trace — reads of layer weights, reads and
appends of KV cache, scratch traffic — on `server-hbm128-hbf512` (4 HBM
stacks with 128 GiB, 4 HBF stacks with 512 GiB) under five placements:

- `all-hbm` puts everything in HBM: the fast, capacity-unconstrained bound.
- `all-hbf` puts everything behind the HBF flash translation layer.
- `flat` splits one address space: low addresses in HBM, the rest in HBF.
- `direct-read` reads immutable data straight from flash and keeps mutable
  data in HBM.
- `hbf-streaming` keeps weights in HBF and double-buffers each layer into HBM.

The table reports, per scenario:

| Column | Meaning |
| --- | --- |
| makespan | Time until all work finished, including background drain |
| throughput | User bytes divided by the time the last user request completed |
| mean / p95 lat | Request latency from arrival to completion |
| HBM ops / HBF ops | Accesses each device served, including background copies |
| HBF WAF | Write amplification: bytes programmed to flash per byte written |

`python3 -m hbfsim list metrics` shows every available metric, including
erase counts, the most-worn block and garbage-collection copies.

**Sanity check.** The runner validates each result and prints `SANITY: PASS`
or a list of warnings. A warning does not mean the simulator is broken: it
flags a result that needs care, for example an open-loop arrival rate above a
device's peak, where latency reflects queueing rather than the device.

## 4. Look at what a run produced

Every run gets a fresh directory under `out/` (never reused, so results are
never mixed):

| File | Contents |
| --- | --- |
| `command.txt` | The exact `hbfsim-reference` command; rerun it to reproduce |
| `resolved.cfg` | Every configuration value after files and overrides |
| `summary.json` | The machine-readable result: provenance, config, per-scenario statistics |
| `stdout.txt` | The runner's full report, including per-stage time breakdowns |
| `trace.txt` | The workload that was replayed |
| `*-hbf-wear/*.html` | Interactive per-block HBF wear maps; open in a browser |

`summary.json` holds hundreds of counters per scenario. From Python:

```python
import hbfsim

result = hbfsim.load_summary("out/quickstart-YYYYMMDD-HHMMSS")
all_hbf = result.scenario("all-hbf")
print(all_hbf["p95_latency_us"])                 # a headline metric
print(all_hbf.raw["hbf_stats"]["page_reads"])    # any raw counter
```

For a request-level timeline, add `--set chrome-trace=out/timeline.json` to
a `run` and open the file in `chrome://tracing` or
[Perfetto](https://ui.perfetto.dev); the
[observability reference](reference/observability.md) explains the tracks.

## 5. Change one thing

Configuration comes in layers, applied in order, later values winning:

1. one **system** profile (`--system`): the complete hardware;
2. its paired **reference-policy** profile, chosen automatically when one
   exists (`--policy none` to skip);
3. any number of **overlays** (`--overlay`): one-purpose changes such as an
   HBF speed grade, a mapping strategy or a backing device;
4. **overrides** of single keys (`--set KEY=VALUE`).

```bash
python3 -m hbfsim list systems
python3 -m hbfsim list overlays
python3 -m hbfsim run --overlay ocp-v070-grade1                 # slower HBF interface
python3 -m hbfsim run --set hbf-read-ns=8000 --scenarios all-hbf  # slower NAND sensing
```

Overlay and system names can be shortened to any unambiguous file name
(`ocp-v070-grade1`, `mapping/block`); a typo gets suggestions. Every key is
documented in the [configuration catalog](../configs/README.md), and
`configs/parameter-provenance.json` says where each default comes from.

## 6. Sweep a design space

```bash
python3 -m hbfsim sweep \
  --vary hbf-read-ns=2000,4000,8000 \
  --vary overlay=ocp-v070-grade1,ocp-v070-grade2 \
  --scenarios all-hbf --jobs 4
```

Each `--vary` adds an axis; the sweep runs the full grid on the **same
trace**, so differences come only from the varied parameters. An axis is any
configuration key, or `system=`, `overlay=` (`none` means no extra overlay)
or `policy=`. Results land in `sweep.csv` (one row per point and scenario,
ready for a spreadsheet or pandas) and `sweep.json`; each point keeps its own
run directory. A combination the simulator rejects is reported with the
reason, and the other points still run.

The same from Python:

```python
points = hbfsim.sweep(vary={"hbf-read-ns": [2000, 4000, 8000]}, scenarios="all-hbf")
for point in points:
    print(point.labels, point.result.scenario("all-hbf")["mean_latency_us"])
```

The [design-space recipes](guides/design-space.md) collect verified sweeps
for common HBF questions.

## 7. Bring your own workload

The built-in trace is a smoke test: small enough to run in a second, too
small to stress capacity or bandwidth. For real questions:

- **Trace files.** One operation per line: `<address> <R|W> [bytes] [kind]
  [layer=N]`. [Example 4](../examples/04_custom_trace.py) writes one from
  Python; `workloads/synthetic/generate.py` produces sequential, random,
  strided and hotspot patterns; the [workload replay guide](guides/workload-replay.md)
  covers importing and qualifying traces. Replay with `--trace FILE`.
- **LLM serving.** [ServeLoop](guides/serveloop.md) models continuous batching,
  paged KV caches and prefix reuse for real model descriptors, and drives the
  engine in closed loop.

## 8. Try your own policy

The engine executes arbitrary transaction DAGs, so a placement, caching or
migration idea is just Python that decides which transactions to issue.
[Example 5](../examples/05_custom_policy.py) implements an HBM promotion cache
in about sixty lines and compares it with all-HBF and all-HBM.
[Extending HBFSim](guides/extending.md) explains the protocol and what to do
when an idea outgrows a script.

## Where next

- [Concepts](concepts.md): HBF in one page, the system model and a glossary.
- [Examples](../examples/README.md): six runnable questions.
- [Design-space recipes](guides/design-space.md): verified commands per question.
- [Extending HBFSim](guides/extending.md): configs, workloads, policies, mechanisms.
- [Documentation index](README.md): every guide and reference.
