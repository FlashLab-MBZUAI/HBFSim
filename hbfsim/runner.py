"""Run the reference runner (``hbfsim-reference``) from Python.

Each run writes into its own directory::

    out/<run>/
      command.txt     the exact hbfsim-reference command line
      resolved.cfg    every configuration value after files and overrides
      summary.json    the machine-readable result (schema hbfsim.simulation.summary)
      stdout.txt      the runner's complete human-readable report
      *-hbf-wear/     per-scenario HBF wear maps (HTML + JSON)

The directory is the unit of reproducibility: replaying ``resolved.cfg`` with
the same trace and executable reproduces ``summary.json`` exactly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import csv
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from hbfsim import catalog
from hbfsim.results import RunResult, load_summary
from hbfsim.workspace import (
    HbfsimError,
    REFERENCE_RUNNER,
    repository_root,
    require_executable,
)


DEFAULT_SYSTEM = "server-hbm128-hbf512"
_KEY = re.compile(r"[a-z0-9][a-z0-9-]*")
# Engine messages that have a common, fixable cause for new users.
ERROR_HINTS = (
    ("overlaps the cooperative write region",
     "The flat/direct-read boundary (flat-hbm-bytes, static-direct-hbm-bytes) "
     "must not reach the cooperative HBM write region, which starts at the "
     "application HBM capacity (hbm-capacity-bytes minus the controller-DRAM "
     "reservation hbf-ctrl-dram-bytes) minus hbf-hbm-write-buffer-bytes, rounded "
     "down to an HBF page. The runner prints that address as "
     "cooperative_write_region_base= in its first output line; set the boundary "
     "at or below it, or drop flat from --scenarios."),
    ("beyond the logical capacity",
     "The trace addresses more HBF than this system has. Use a larger system "
     "(`python3 -m hbfsim list systems`) or a trace with a smaller footprint."),
    ("exceeds physical page capacity",
     "The trace addresses more static HBF than this system has. Use a larger "
     "system or a trace with a smaller footprint."),
    ("application/controller capacity partition",
     "The trace addresses more HBM than this system has. Use a larger system or "
     "a trace with a smaller footprint."),
    ("requires explicit hbf-ctrl-dram-bytes",
     "Cached mapping strategies need an HBM budget for the mapping cache. Apply "
     "a budget overlay first, e.g. --overlay cached-l2p-1-over-1000 --overlay "
     "mapping/entry-cache."),
    ("unknown option", "Check the key name against configs/README.md or "
     "`build/hbfsim-reference --help`."),
)


def explain(message: str) -> str:
    """Append a hint to a simulator error message when one is known."""

    for needle, hint in ERROR_HINTS:
        if needle in message:
            return f"{message}\nhint: {hint}"
    return message


def default_output_directory(prefix: str) -> Path:
    """A fresh ``out/<prefix>-<timestamp>`` directory under the checkout."""

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = repository_root() / "out" / f"{prefix}-{stamp}"
    candidate, counter = base, 1
    while candidate.exists():
        counter += 1
        candidate = base.with_name(f"{base.name}-{counter}")
    return candidate


def prepare_output_directory(path: Path | None, prefix: str, *, overwrite: bool = False) -> Path:
    """Create a fresh output directory; refuse to mix results into a used one."""

    directory = default_output_directory(prefix) if path is None else Path(path)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise HbfsimError(
            f"{directory} already contains files. Choose a new --out directory "
            "(each run keeps its own inputs and results) or pass --overwrite."
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _assignment_pairs(items: Iterable[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in items:
        key, separator, value = item.partition("=")
        key = key.strip().removeprefix("--")
        if not separator or not _KEY.fullmatch(key) or not value.strip():
            raise HbfsimError(f"expected KEY=VALUE such as hbf-read-ns=8000, got {item!r}")
        pairs.append((key, value.strip()))
    return pairs


def parse_assignments(items: Iterable[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` strings (from ``--set``); a repeated key keeps the last value."""

    return dict(_assignment_pairs(items))


@dataclass(frozen=True)
class RunSpec:
    """Everything that defines one reference-runner invocation.

    ``policy="auto"`` selects the reference-policy profile shipped with the
    system (same file name) when there is one; ``None`` uses none.
    ``trace=None`` generates the built-in synthetic LLM-like smoke trace.
    ``options`` are ``--KEY VALUE`` overrides of any config key; they win over
    every config file.
    """

    system: str | Path = DEFAULT_SYSTEM
    overlays: Sequence[str | Path] = ()
    policy: str | Path | None = "auto"
    trace: Path | None = None
    scenarios: Sequence[str] | str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    demo_tokens: int = 4
    demo_layers: int = 2
    max_ops: int | None = None

    def config_files(self) -> list[Path]:
        system = catalog.resolve("system", self.system)
        files = [system]
        if self.policy == "auto":
            policy = catalog.default_policy(system)
            if policy is not None:
                files.append(policy)
        elif self.policy is not None:
            files.append(catalog.resolve("policy", self.policy))
        files.extend(catalog.resolve("overlay", overlay) for overlay in self.overlays)
        return files


def generate_demo_trace(path: Path, *, tokens: int = 4, layers: int = 2) -> Path:
    """Write the reference runner's synthetic LLM-like smoke trace.

    It exercises weights, KV and scratch traffic in every scenario. It is a
    smoke-test pattern, not a model workload; use ServeLoop or a real trace for
    serving claims.
    """

    runner = require_executable(REFERENCE_RUNNER)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(runner), "--generate-semantic-llm", str(path),
         "--llm-tokens", str(tokens), "--llm-layers", str(layers)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise HbfsimError(f"trace generation failed: {(result.stderr or result.stdout).strip()}")
    return path


def build_command(spec: RunSpec, trace: Path, out_dir: Path) -> list[str]:
    command = [str(require_executable(REFERENCE_RUNNER))]
    for config in spec.config_files():
        command += ["--config", str(config)]
    command += ["--trace", str(trace)]
    command += ["--scenarios", ",".join(catalog.scenario_list(spec.scenarios))]
    for key, value in spec.options.items():
        key = str(key).removeprefix("--")
        if not _KEY.fullmatch(key):
            raise HbfsimError(f"invalid option name {key!r}")
        command += [f"--{key}", _option_value(value)]
    if spec.max_ops is not None:
        command += ["--max-ops", str(spec.max_ops)]
    command += [
        "--summary-json", str(out_dir / "summary.json"),
        "--config-out", str(out_dir / "resolved.cfg"),
    ]
    return command


def run(spec: RunSpec | None = None, *, out_dir: Path | None = None,
        overwrite: bool = False, labels: Mapping[str, str] | None = None,
        **fields: Any) -> RunResult:
    """Run the reference runner once and return its parsed result.

    Accepts a :class:`RunSpec` or its fields as keywords, e.g.
    ``run(system="4hbm-4hbf", scenarios="all-hbf", options={"hbf-read-ns": 8000})``.
    A run whose sanity check fails still returns its result; inspect
    ``result.ok`` and each scenario's ``warnings``.
    """

    spec = replace(spec or RunSpec(), **fields) if fields else (spec or RunSpec())
    directory = prepare_output_directory(out_dir, "run", overwrite=overwrite)
    trace = spec.trace
    if trace is None:
        trace = generate_demo_trace(directory / "trace.txt",
                                    tokens=spec.demo_tokens, layers=spec.demo_layers)
    elif not Path(trace).is_file():
        raise HbfsimError(f"trace file not found: {trace}")
    command = build_command(spec, Path(trace).resolve(), directory.resolve())
    (directory / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    with (directory / "stdout.txt").open("w", encoding="utf-8") as log:
        # Run from the caller's directory so relative paths in `options`
        # mean what they would on the command line.
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
        )
    summary = directory / "summary.json"
    if not summary.is_file():
        tail = (directory / "stdout.txt").read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(tail.strip().splitlines()[-8:])
        raise HbfsimError(explain(
            f"hbfsim-reference failed (exit {completed.returncode}); full log in "
            f"{directory / 'stdout.txt'}\n{tail}"
        ))
    return load_summary(summary, labels=labels, command=command)


@dataclass(frozen=True)
class SweepPoint:
    """One grid point: its axis values and either a result or the error."""

    labels: Mapping[str, str]
    result: RunResult | None
    error: str | None = None
    directory: Path | None = None


def sweep_grid(axes: Mapping[str, Sequence[Any]]) -> list[dict[str, str]]:
    """Cartesian product of axis values, in the order the axes were given."""

    if not axes:
        raise HbfsimError("a sweep needs at least one --vary KEY=V1,V2,...")
    for key, values in axes.items():
        if not values:
            raise HbfsimError(f"sweep axis {key!r} has no values")
    keys = list(axes)
    return [dict(zip(keys, (str(value) for value in combo)))
            for combo in itertools.product(*(axes[key] for key in keys))]


def parse_axes(items: Iterable[str]) -> dict[str, list[str]]:
    """Parse ``--vary KEY=V1,V2`` strings; ``system``/``overlay``/``policy`` are special."""

    axes: dict[str, list[str]] = {}
    for key, value in _assignment_pairs(items):
        if key in axes:
            raise HbfsimError(f"sweep axis {key!r} given twice")
        axes[key] = [item.strip() for item in value.split(",") if item.strip()]
    return axes


def _point_spec(base: RunSpec, labels: Mapping[str, str], trace: Path) -> RunSpec:
    options = dict(base.options)
    spec = replace(base, trace=trace)
    for key, value in labels.items():
        if key == "system":
            spec = replace(spec, system=value)
        elif key == "overlay":
            extra = () if value == "none" else (value,)
            spec = replace(spec, overlays=tuple(base.overlays) + extra)
        elif key == "policy":
            spec = replace(spec, policy=None if value == "none" else value)
        elif key == "scenarios":
            raise HbfsimError("pass scenarios with --scenarios; every sweep point runs all of them")
        else:
            options[key] = value
    return replace(spec, options=options)


def sweep(base: RunSpec | None = None, *, vary: Mapping[str, Sequence[Any]],
          out_dir: Path | None = None, jobs: int = 1, overwrite: bool = False,
          progress=None, **fields: Any) -> list[SweepPoint]:
    """Run ``base`` once per point of the cartesian grid ``vary``.

    Axis keys are config keys (``hbf-read-ns``) or ``system``/``overlay``/
    ``policy``. Every point replays the same trace, so differences come only
    from the varied parameters. Results are also written to ``sweep.csv``
    and ``sweep.json``. A point the simulator rejects is recorded with its
    error (``point.error``) and the remaining points still run.
    """

    base = replace(base or RunSpec(), **fields) if fields else (base or RunSpec())
    grid = sweep_grid(vary)
    directory = prepare_output_directory(out_dir, "sweep", overwrite=overwrite)
    trace = base.trace
    if trace is None:
        trace = generate_demo_trace(directory / "trace.txt",
                                    tokens=base.demo_tokens, layers=base.demo_layers)
    trace = Path(trace).resolve()
    specs = [_point_spec(base, labels, trace) for labels in grid]
    for spec in specs:  # fail on a bad name before running anything
        spec.config_files()
        catalog.scenario_list(spec.scenarios)

    def execute(index: int) -> SweepPoint:
        labels = grid[index]
        slug = "-".join(re.sub(r"[^A-Za-z0-9.]+", "_", value) for value in labels.values())
        point_dir = directory / f"{index:03d}-{slug}"[:120]
        try:
            point = SweepPoint(labels, run(specs[index], out_dir=point_dir,
                                           overwrite=overwrite, labels=labels),
                               directory=point_dir)
        except HbfsimError as error:
            # Keep going: one invalid combination should not discard the grid.
            point = SweepPoint(labels, None, str(error), point_dir)
        if progress is not None:
            progress(index, len(grid), point)
        return point

    workers = max(1, min(jobs, len(grid)))
    if workers == 1:
        points = [execute(index) for index in range(len(grid))]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            points = list(pool.map(execute, range(len(grid))))
    rows = [row for point in points if point.result for row in point.result.rows()]
    if rows:
        with (directory / "sweep.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (directory / "sweep.json").write_text(json.dumps({
        "axes": {key: [str(value) for value in values] for key, values in vary.items()},
        "points": [{"labels": dict(point.labels),
                    "directory": os.path.relpath(point.directory, directory),
                    "sanity": point.result.sanity if point.result else None,
                    "error": point.error} for point in points],
    }, indent=2) + "\n", encoding="utf-8")
    return points
