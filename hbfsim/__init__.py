"""HBFSim front door: build, run, sweep and inspect simulations from Python.

Command line::

    python3 -m hbfsim quickstart

Python::

    import hbfsim

    result = hbfsim.run(system="4hbm-4hbf", scenarios="all-hbm,all-hbf")
    print(result.table())
    print(result.scenario("all-hbf")["p95_latency_us"])

    points = hbfsim.sweep(vary={"hbf-read-ns": [2000, 4000, 8000]},
                          scenarios="all-hbf")

This package only orchestrates the compiled programs and reads their
receipts; the physical model lives in C++ (``src/``) and the low-level
transaction protocol in :mod:`hbfsim_client`.
"""

from hbfsim.catalog import (
    CORE_SCENARIOS,
    SCENARIOS,
    default_policy,
    profiles,
    resolve,
)
from hbfsim.results import (
    METRICS,
    RunResult,
    ScenarioResult,
    format_table,
    load_summary,
)
from hbfsim.session import open_session
from hbfsim.runner import (
    RunSpec,
    SweepPoint,
    generate_demo_trace,
    run,
    sweep,
)
from hbfsim.workspace import (
    HbfsimError,
    build,
    build_directory,
    doctor,
    find_executable,
    repository_root,
    require_executable,
)

__version__ = "0.1.0"

__all__ = [
    "CORE_SCENARIOS",
    "HbfsimError",
    "METRICS",
    "RunResult",
    "RunSpec",
    "SCENARIOS",
    "ScenarioResult",
    "SweepPoint",
    "build",
    "build_directory",
    "default_policy",
    "doctor",
    "find_executable",
    "format_table",
    "generate_demo_trace",
    "load_summary",
    "open_session",
    "profiles",
    "repository_root",
    "require_executable",
    "resolve",
    "run",
    "sweep",
]
