"""Open a persistent engine session by profile name, for custom policies.

The reference runner replays traces through fixed policies. To try your own
placement, caching, migration or prefetch idea, drive the engine directly:
build :class:`hbfsim_client.Transaction` DAGs in Python and submit them to a
:class:`hbfsim_client.SimulationSession`. :func:`open_session` does the
bookkeeping — resolving names, turning ``options`` into an overlay file, and
keeping wear reports inside the run directory::

    from hbfsim import open_session
    from hbfsim_client import Transaction

    with open_session("4hbm-4hbf", overlays=["ocp-v070-grade3"]) as session:
        batch = [Transaction(id=f"r{i}", target="HBF_LOGICAL", op="R",
                             addr=i * 4096, bytes=4096, issue_ns=0.0)
                 for i in range(1024)]
        print(session.run(batch).elapsed_ns)
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from hbfsim import catalog
from hbfsim.runner import prepare_output_directory
from hbfsim.workspace import ENGINE, HbfsimError, require_executable
from hbfsim_client import ResolvedSystemConfig, SimulationSession


# The Python client derives HBF geometry from these keys; ratio-sized
# profiles leave some of them to the engine's resolver.
GEOMETRY_KEYS = (
    "hbf-stacks", "hbf-channels", "hbf-dies-per-channel", "hbf-planes-per-die",
    "hbf-blocks-per-plane", "hbf-pages-per-block", "hbf-page-size",
)

def _describe(engine: Path, files: Sequence[Path]) -> dict[str, str]:
    command = [str(engine)]
    for path in files:
        command += ["--system-config", str(path)]
    command.append("--describe-system")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise HbfsimError(
            "the engine rejected this configuration: "
            + (completed.stderr or completed.stdout).strip()
        )
    return dict(json.loads(completed.stdout)["values"])


def session_config(system: str | Path, overlays: Sequence[str | Path] = (),
                   options: Mapping[str, Any] | None = None, *,
                   out_dir: Path) -> ResolvedSystemConfig:
    """Resolve a system (+ overlays + key overrides) into a client config.

    Overrides are written to ``out_dir/overrides.cfg`` so the run directory
    records them. Profiles that size HBF by ratio get their resolved geometry
    written to ``out_dir/resolved-geometry.cfg``; the engine resolves the
    same values either way.
    """

    files = [catalog.resolve("system", system)]
    files += [catalog.resolve("overlay", overlay) for overlay in overlays]
    if options:
        overrides = out_dir / "overrides.cfg"
        overrides.write_text(
            "".join(f"{str(key).removeprefix('--')}={value}\n" for key, value in options.items()),
            encoding="utf-8",
        )
        files.append(overrides)
    declared: dict[str, str] = {}
    for path in files:
        declared.update(catalog.parse_config(path))
    engine = require_executable(ENGINE)
    resolved = _describe(engine, files)
    missing = {key: resolved[key] for key in GEOMETRY_KEYS
               if key in resolved and key not in declared}
    if missing:
        geometry = out_dir / "resolved-geometry.cfg"
        geometry.write_text(
            "# Geometry resolved by `hbfsim --describe-system` for the Python client.\n"
            + "".join(f"{key}={value}\n" for key, value in missing.items()),
            encoding="utf-8",
        )
        files.append(geometry)
    return ResolvedSystemConfig.load(tuple(files))


def open_session(system: str | Path = "4hbm-4hbf", overlays: Sequence[str | Path] = (),
                 options: Mapping[str, Any] | None = None, *,
                 out_dir: Path | None = None, overwrite: bool = False,
                 enable_hbm: bool = True, enable_hbf: bool = True,
                 enable_external: bool = False, **session_options: Any) -> SimulationSession:
    """Start ``build/hbfsim`` for one configuration and return the session.

    Use it as a context manager. ``session_options`` are passed through to
    :class:`hbfsim_client.SimulationSession` (for example
    ``initial_hbf_logical_pages=N`` to start with N pages already written).
    """

    directory = prepare_output_directory(out_dir, "session", overwrite=overwrite)
    config = session_config(system, overlays, options, out_dir=directory)
    session_options.setdefault("hbf_wear_output_prefix", directory / "hbf-wear")
    return SimulationSession(
        simulator_path=require_executable(ENGINE),
        system_config=config,
        enable_hbm=enable_hbm,
        enable_hbf=enable_hbf,
        enable_external=enable_external,
        **session_options,
    )
