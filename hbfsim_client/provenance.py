#!/usr/bin/env python3
"""Source-tree provenance for every study result.

Every result produced through ``hbfsim_client`` should say which source
revision produced it. ``run_provenance`` queries git for the repository that
contains this package and records the simulator binary, the configs, the
runner script, the command line, and the interpreter; ``require_clean_tree``
is the fail-closed gate studies apply unless the operator passes
``--allow-dirty``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
GIT_TIMEOUT_S = 15.0


class ProvenanceError(RuntimeError):
    """The source tree cannot be attributed, or is not in a publishable state."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "bytes": None, "sha256": None}
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _git(arguments: list[str], repository: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_state(repository: Path = ROOT) -> dict[str, Any]:
    """Commit, dirty flag (tracked files only), and tree hash, or unknowns."""

    commit = _git(["rev-parse", "--verify", "HEAD"], repository)
    tree = _git(["rev-parse", "--verify", "HEAD^{tree}"], repository)
    status = _git(["status", "--porcelain", "--untracked-files=no"], repository)
    return {
        "repository": str(repository),
        "git_commit": commit or "unknown",
        "git_dirty": None if status is None else bool(status),
        "tree_hash": tree or "unknown",
        "provenance_source": (
            "run-time" if commit and tree and status is not None else "unavailable"
        ),
    }


def run_provenance(
    simulator_path: Path | str | None,
    config_paths: Iterable[Path | str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe this run: source revision, binaries, inputs, and invocation."""

    runner = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    return {
        **git_state(),
        "simulator": None if simulator_path is None else _artifact(Path(simulator_path)),
        "configs": [_artifact(Path(path)) for path in config_paths],
        "runner": None if runner is None else _artifact(runner),
        "argv": list(sys.argv),
        "working_directory": os.getcwd(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "extra": None if extra is None else dict(extra),
    }


def require_clean_tree(provenance: Mapping[str, Any]) -> None:
    """Refuse to proceed unless the tree is attributable and committed."""

    commit = provenance.get("git_commit")
    dirty = provenance.get("git_dirty")
    if not isinstance(commit, str) or commit == "unknown" or dirty is None:
        raise ProvenanceError(
            "source provenance is unavailable (git did not answer for "
            f"{provenance.get('repository', ROOT)}); results cannot be attributed "
            "to a revision. Pass --allow-dirty to record an exploratory run."
        )
    if dirty:
        raise ProvenanceError(
            f"source tree at commit {commit[:12]} has uncommitted tracked "
            "changes; commit them so the result is reproducible, or pass "
            "--allow-dirty to record an exploratory (status: exploratory_dirty) run."
        )


def add_allow_dirty_argument(parser: Any) -> None:
    """Register the study-wide ``--allow-dirty`` flag on an argparse parser."""

    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "run on a source tree with uncommitted tracked changes; the "
            "result is stamped status=exploratory_dirty instead of refusing"
        ),
    )


def gate_provenance(
    parser: Any,
    simulator_path: Path | str | None,
    *,
    allow_dirty: bool,
    config_paths: Iterable[Path | str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect run provenance and refuse a dirty tree unless allowed.

    A refusal is reported through ``parser.error`` so every study exits with
    the same usage-style message.
    """

    provenance = run_provenance(
        simulator_path, config_paths=config_paths, extra=extra
    )
    if not allow_dirty:
        try:
            require_clean_tree(provenance)
        except ProvenanceError as error:
            parser.error(str(error))
    return provenance


def stamp_result(
    result: dict[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach ``source`` to a study envelope; a dirty tree marks it exploratory."""

    result["source"] = dict(provenance)
    if provenance.get("git_dirty") is not False:
        result["status"] = "exploratory_dirty"
    return result
