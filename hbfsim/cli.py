"""``python3 -m hbfsim`` — the front door to HBFSim.

Start with ``python3 -m hbfsim quickstart``. Every command prints the
underlying simulator invocation, so nothing here is hidden magic: the same
results can always be reproduced with ``build/hbfsim-reference`` alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from hbfsim import catalog, results, runner, workspace
from hbfsim.workspace import HbfsimError


EPILOG = """\
examples:
  python3 -m hbfsim quickstart
  python3 -m hbfsim list systems
  python3 -m hbfsim run --system 4hbm-4hbf --scenarios all-hbm,all-hbf
  python3 -m hbfsim run --overlay ocp-v070-grade3 --set hbf-read-ns=8000
  python3 -m hbfsim sweep --vary hbf-read-ns=2000,4000,8000 --scenarios all-hbf
  python3 -m hbfsim show out/run-*/summary.json

docs: docs/getting-started.md
"""


def _echo(message: str = "") -> None:
    print(message, flush=True)


def _relative(path: Path) -> str:
    """Show paths under the working directory relatively, others absolutely."""

    absolute = Path(path).resolve()
    try:
        return str(absolute.relative_to(Path.cwd()))
    except ValueError:
        return str(absolute)


def _ensure_built(auto_build: bool) -> None:
    missing = [name for name in (workspace.ENGINE, workspace.REFERENCE_RUNNER)
               if workspace.find_executable(name) is None]
    if not missing:
        return
    if not auto_build:
        raise HbfsimError(
            f"{', '.join(missing)} not built. Run `python3 -m hbfsim build` first."
        )
    _echo(f"The simulator is not built yet; building {', '.join(missing)} (one time, ~1 minute).")
    workspace.build(echo=_echo)
    _echo()


def _print_result(result: results.RunResult, args: argparse.Namespace) -> None:
    metrics = args.metrics or results.DEFAULT_TABLE_METRICS
    _echo(results.format_table(result.rows(), metrics=metrics, style=args.format))
    _print_warnings([result])


def _print_warnings(runs: Sequence[results.RunResult]) -> None:
    warned = [(run, scenario) for run in runs for scenario in run.scenarios if scenario.warnings]
    if warned:
        _echo("\nwarnings (the simulator's sanity check flags these; read them before drawing conclusions):")
        for run, scenario in warned:
            label = ", ".join(f"{k}={v}" for k, v in (run.labels or {}).items())
            prefix = f"[{label}] " if label else ""
            for warning in scenario.warnings:
                _echo(f"  {prefix}{scenario.name}: {warning}")


def _spec(args: argparse.Namespace) -> runner.RunSpec:
    policy: str | None = args.policy
    if policy == "none":
        policy = None
    return runner.RunSpec(
        system=args.system,
        overlays=tuple(args.overlay or ()),
        policy=policy,
        trace=args.trace,
        scenarios=args.scenarios,
        options=runner.parse_assignments(args.set or ()),
        demo_tokens=args.demo_tokens,
        demo_layers=args.demo_layers,
        max_ops=args.max_ops,
    )


# ---------------------------------------------------------------- commands

def command_doctor(args: argparse.Namespace) -> int:
    checks = workspace.doctor()
    _echo(f"HBFSim checkout: {workspace.repository_root()}")
    _echo(f"Build directory: {workspace.build_directory()}\n")
    for check in checks:
        mark = "ok" if check.ok else ("!!" if check.required else "--")
        _echo(f"[{mark}] {check.name}: {check.detail}")
    failed = [check for check in checks if check.required and not check.ok]
    _echo()
    if not failed:
        _echo("Everything needed is in place. Next: python3 -m hbfsim quickstart")
        return 0
    toolchain = [check for check in failed if "built" not in check.name]
    for check in toolchain:
        if check.hint:
            _echo(f"fix: {check.hint}")
    if not toolchain:
        _echo("The toolchain is ready. Next: python3 -m hbfsim build")
    return 1


def command_build(args: argparse.Namespace) -> int:
    build_dir = workspace.build(
        build_type=args.type, all_targets=args.all, jobs=args.jobs, echo=_echo,
    )
    _echo(f"\nBuilt into {_relative(build_dir)}.")
    if args.all:
        _echo(f"Run the test suite with: ctest --test-dir {_relative(build_dir)} --output-on-failure")
    else:
        _echo("Next: python3 -m hbfsim quickstart")
    return 0


def command_quickstart(args: argparse.Namespace) -> int:
    _ensure_built(not args.no_build)
    out = runner.prepare_output_directory(args.out, "quickstart", overwrite=args.overwrite)
    spec = runner.RunSpec(system=runner.DEFAULT_SYSTEM, scenarios=catalog.CORE_SCENARIOS)
    system = catalog.resolve("system", spec.system)
    _echo("HBFSim quick start")
    _echo(f"  system    {spec.system}: {catalog.describe_system(catalog.parse_config(system))}")
    _echo("  workload  synthetic LLM-like smoke trace (weights, KV cache and scratch traffic)")
    _echo(f"  output    {_relative(out)}\n")
    result = runner.run(spec, out_dir=out, overwrite=True)
    operations = result.summary.get("config", {}).get("ops")
    if operations:
        _echo(f"Replayed {operations:,} memory operations under each placement scenario:\n")
    metrics = results.DEFAULT_TABLE_METRICS
    _echo(results.format_table(result.rows(), metrics=metrics))
    _print_warnings([result])
    _echo("\nWhat the columns mean:")
    _echo(results.metric_legend(metrics))
    _echo("\nWhat the scenarios do:")
    for name in spec.scenarios:
        _echo(f"  {name:<14} {catalog.SCENARIOS[name]}")
    _echo(f"\nSanity check: {result.sanity}.  Full report: {_relative(out / 'stdout.txt')}")
    _echo(f"Machine-readable summary: {_relative(result.summary_path)}")
    _echo("\nNext steps:")
    _echo("  python3 -m hbfsim list systems                        # other hardware profiles")
    _echo("  python3 -m hbfsim run --overlay ocp-v070-grade3       # try a faster HBF speed grade")
    _echo("  python3 -m hbfsim sweep --vary hbf-read-ns=2000,4000,8000 --scenarios all-hbf")
    _echo("  docs/getting-started.md                               # the guided tour")
    return 0 if result.ok else 1


def command_list(args: argparse.Namespace) -> int:
    what = args.what
    if what in ("scenarios", "all"):
        _echo("Scenarios (reference placement policies; `core` = the first five marked *):")
        for name, description in catalog.SCENARIOS.items():
            star = "*" if name in catalog.CORE_SCENARIOS else " "
            _echo(f"  {star} {name:<19} {description}")
        _echo()
    if what in ("metrics", "all"):
        _echo("Metrics (select with --metrics):")
        for item in results.METRICS:
            _echo(f"  {item.key:<19} {item.help}")
        _echo()
    kinds = {"systems": "system", "overlays": "overlay", "policies": "policy"}
    for plural, kind in kinds.items():
        if what not in (plural, "all"):
            continue
        entries = catalog.profiles(kind)
        width = max(len(entry.name) for entry in entries)
        title = {
            "system": "Systems (complete hardware profiles; pass with --system NAME)",
            "overlay": "Overlays (one-purpose overrides applied after a system; --overlay NAME)",
            "policy": "Policies (reference-runner settings; paired automatically by name)",
        }[kind]
        _echo(title + ":")
        for entry in entries:
            _echo(f"  {entry.name:<{width}}  {entry.description}")
        _echo()
    if what in ("systems", "all"):
        _echo("Configuration files live in configs/; see configs/README.md for every key.")
    return 0


def command_run(args: argparse.Namespace) -> int:
    spec = _spec(args)
    catalog.scenario_list(spec.scenarios)
    command_preview = [str(path) for path in spec.config_files()]
    if not args.quiet:
        _echo("configs: " + " -> ".join(_relative(Path(path)) for path in command_preview))
    result = runner.run(spec, out_dir=args.out, overwrite=args.overwrite)
    if not args.quiet:
        _echo(f"output:  {_relative(result.summary_path.parent)}")
        _echo(f"command: {_relative(result.summary_path.parent / 'command.txt')}\n")
    _print_result(result, args)
    if not args.quiet:
        _echo(f"\nSanity check: {result.sanity}")
    return 0 if result.ok else 1


def command_sweep(args: argparse.Namespace) -> int:
    spec = _spec(args)
    axes = runner.parse_axes(args.vary)
    grid = runner.sweep_grid(axes)
    _echo(f"sweep: {len(grid)} point(s) x {len(catalog.scenario_list(spec.scenarios))} scenario(s)")

    def label(point: runner.SweepPoint) -> str:
        return ", ".join(f"{key}={value}" for key, value in point.labels.items())

    def progress(index: int, total: int, point: runner.SweepPoint) -> None:
        status = f"sanity {point.result.sanity}" if point.result else "REJECTED"
        _echo(f"  [{index + 1}/{total}] {label(point)}: {status}")

    points = runner.sweep(spec, vary=axes, out_dir=args.out, jobs=args.jobs,
                          overwrite=args.overwrite, progress=progress)
    completed = [point.result for point in points if point.result]
    rows = [row for result in completed for row in result.rows()]
    _echo()
    _echo(results.format_table(rows, metrics=args.metrics or results.DEFAULT_TABLE_METRICS,
                               style=args.format))
    _print_warnings(completed)
    failed = [point for point in points if point.error]
    for point in failed:
        _echo(f"\nrejected [{label(point)}]:\n  " + point.error.replace("\n", "\n  "))
    directory = points[0].directory.parent
    _echo(f"\nAll points: {_relative(directory / 'sweep.csv')}")
    return 1 if failed else 0


def command_show(args: argparse.Namespace) -> int:
    runs = []
    for path in args.paths:
        label = {"run": _relative(Path(path))} if len(args.paths) > 1 else None
        runs.append(results.load_summary(path, labels=label))
    rows = [row for run in runs for row in run.rows()]
    _echo(results.format_table(rows, metrics=args.metrics or results.DEFAULT_TABLE_METRICS,
                               style=args.format))
    _print_warnings(runs)
    return 0


# ------------------------------------------------------------------ parser

def _metric_list(text: str) -> list[str]:
    keys = [item.strip() for item in text.split(",") if item.strip()]
    if keys == ["all"]:
        return list(results.METRIC_KEYS)
    for key in keys:
        try:
            results.metric(key)
        except HbfsimError as error:
            raise argparse.ArgumentTypeError(str(error)) from error
    return keys


def _add_output_options(parser: argparse.ArgumentParser, *, with_out: bool = True) -> None:
    if with_out:
        parser.add_argument("--out", type=Path, default=None,
                            help="output directory (default: a fresh out/<command>-<time>/)")
        parser.add_argument("--overwrite", action="store_true",
                            help="allow writing into a non-empty --out directory")
    parser.add_argument("--format", choices=("text", "markdown", "csv", "json"), default="text",
                        help="table style (default: text)")
    parser.add_argument("--metrics", type=_metric_list, default=None,
                        help="comma-separated metric keys, or 'all' (see `list metrics`)")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("what to simulate")
    group.add_argument("--system", default=runner.DEFAULT_SYSTEM,
                       help=f"system profile name or .cfg path (default: {runner.DEFAULT_SYSTEM})")
    group.add_argument("--overlay", action="append", metavar="NAME",
                       help="overlay applied after the system; repeatable, applied in order")
    group.add_argument("--policy", default="auto",
                       help="reference-policy profile: 'auto' (paired with the system), 'none', or a name")
    group.add_argument("--scenarios", default=None,
                       help="comma-separated scenario names, 'core' (default) or 'all'")
    group.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="override any config key, e.g. --set hbf-read-ns=8000; repeatable")
    workload = parser.add_argument_group("workload")
    workload.add_argument("--trace", type=Path, default=None,
                          help="address trace file (default: generate the synthetic smoke trace)")
    workload.add_argument("--demo-tokens", type=int, default=4,
                          help="synthetic trace: decode tokens (default: 4)")
    workload.add_argument("--demo-layers", type=int, default=2,
                          help="synthetic trace: model layers (default: 2)")
    workload.add_argument("--max-ops", type=int, default=None,
                          help="replay only the first N trace operations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m hbfsim",
        description="HBFSim front door: build, run, sweep and inspect HBM + HBF memory-system simulations.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="print the front-door version and exit")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    doctor = commands.add_parser("doctor", help="check the toolchain, build, and optional integrations")
    doctor.set_defaults(handler=command_doctor)

    build = commands.add_parser("build", help="configure and compile the simulator with CMake")
    build.add_argument("--type", default="Release", choices=("Release", "Debug", "RelWithDebInfo"),
                       help="CMake build type (default: Release)")
    build.add_argument("--all", action="store_true",
                       help="also build tests, probes and research drivers (needed for ctest)")
    build.add_argument("--jobs", "-j", type=int, default=None, help="parallel compile jobs")
    build.set_defaults(handler=command_build)

    quickstart = commands.add_parser("quickstart", help="build if needed and run a first comparison")
    quickstart.add_argument("--out", type=Path, default=None, help="output directory")
    quickstart.add_argument("--overwrite", action="store_true",
                            help="allow writing into a non-empty --out directory")
    quickstart.add_argument("--no-build", action="store_true", help="fail instead of building")
    quickstart.set_defaults(handler=command_quickstart)

    listing = commands.add_parser("list", help="list systems, overlays, policies, scenarios or metrics")
    listing.add_argument("what", nargs="?", default="all",
                         choices=("systems", "overlays", "policies", "scenarios", "metrics", "all"))
    listing.set_defaults(handler=command_list)

    run = commands.add_parser(
        "run", help="run scenarios on one configuration",
        description="Run reference scenarios on one system configuration and print headline metrics.",
    )
    _add_run_options(run)
    _add_output_options(run)
    run.add_argument("--quiet", action="store_true", help="print only the table")
    run.set_defaults(handler=command_run)

    sweep = commands.add_parser(
        "sweep", help="run a parameter grid and tabulate the results",
        description=(
            "Run every combination of the --vary axes on the same trace. An axis is a config key "
            "(hbf-read-ns=2000,4000) or one of system=..., overlay=..., policy=... ('none' skips)."
        ),
    )
    _add_run_options(sweep)
    sweep.add_argument("--vary", action="append", required=True, metavar="KEY=V1,V2,...",
                       help="sweep axis; repeat for a cartesian grid")
    sweep.add_argument("--jobs", "-j", type=int, default=1, help="points to run in parallel")
    _add_output_options(sweep)
    sweep.set_defaults(handler=command_sweep)

    show = commands.add_parser("show", help="tabulate one or more existing summary.json files")
    show.add_argument("paths", nargs="+", type=Path, help="summary.json files or run directories")
    _add_output_options(show, with_out=False)
    show.set_defaults(handler=command_show)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from hbfsim import __version__
        _echo(f"hbfsim front door {__version__}")
        return 0
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return args.handler(args)
    except HbfsimError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

