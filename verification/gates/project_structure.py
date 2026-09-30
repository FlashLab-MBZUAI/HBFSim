#!/usr/bin/env python3
"""Enforce the responsibility-based repository layout and smoke CLI help."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
FORBIDDEN_PATHS = (
    "tools",
    "validation",
    "src/tools",
    "configs/scenario_compare",
)
CLI_ROOTS = (
    "hbfsim",
    "examples",
    "reports",
    "workloads",
    "verification/gates",
    "evidence",
)
MAIN_GUARD = 'if __name__ == "__main__"'
# Declared-optional imports: the `analysis` extra in pyproject.toml and the
# separate ServeLoop checkout. A --help that needs one of them is skipped, not
# failed, when it is not installed.
OPTIONAL_MODULES = {"numpy", "matplotlib", "reportlab", "svglib", "hbserve"}
MISSING_MODULE = re.compile(r"No module named '([^'.]+)")


def _python_files(root: Path) -> list[Path]:
    # Captured environments and remote receipts are evidence artifacts,
    # not maintained Python packages. Respect the repository's ignore rules.
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
         "--", str(root.relative_to(REPOSITORY))],
        cwd=REPOSITORY,
        text=True,
    )
    return sorted(
        {REPOSITORY / name for name in files.split("\0")
         if name.endswith(".py") and (REPOSITORY / name).is_file()}
    )


def _check_layout() -> tuple[list[str], list[Path]]:
    failures: list[str] = []
    for relative in FORBIDDEN_PATHS:
        if (REPOSITORY / relative).exists():
            failures.append(f"obsolete or generic path exists: {relative}")

    root_scripts = sorted(REPOSITORY.glob("*.py"))
    if root_scripts:
        failures.append(
            "top-level Python scripts have no declared responsibility: "
            + ", ".join(path.name for path in root_scripts)
        )

    entrypoints: list[Path] = []
    for relative in CLI_ROOTS:
        root = REPOSITORY / relative
        for path in _python_files(root):
            if path.name.startswith("run_"):
                failures.append(
                    f"opaque run_ entrypoint name: {path.relative_to(REPOSITORY)}"
                )
            if MAIN_GUARD in path.read_text(encoding="utf-8"):
                entrypoints.append(path)

    package_roots = {
        REPOSITORY / "hbfsim",
        REPOSITORY / "hbfsim_client",
        REPOSITORY / "reports",
        REPOSITORY / "workloads",
        REPOSITORY / "verification",
        REPOSITORY / "evidence",
    }
    package_dirs = {
        path.parent
        for root in package_roots
        for path in _python_files(root)
    }
    for directory in sorted(package_dirs):
        if not (directory / "__init__.py").is_file():
            failures.append(
                "Python package is missing __init__.py: "
                + str(directory.relative_to(REPOSITORY))
            )

    return failures, sorted(set(entrypoints))


def _check_cli_help(entrypoints: list[Path]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    skipped: list[str] = []
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for path in entrypoints:
        relative = path.relative_to(REPOSITORY)
        if path.name == "__main__.py":
            module = ".".join(relative.parent.parts)
            command = [sys.executable, "-B", "-m", module, "--help"]
        else:
            command = [sys.executable, "-B", str(path), "--help"]
        try:
            result = subprocess.run(
                command,
                cwd=REPOSITORY,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired:
            failures.append(f"CLI help timed out: {relative}")
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            missing = MISSING_MODULE.search(detail[-1]) if detail else None
            if missing and missing.group(1) in OPTIONAL_MODULES:
                skipped.append(f"{relative} (needs optional {missing.group(1)})")
                continue
            suffix = f": {detail[-1]}" if detail else ""
            failures.append(f"CLI help failed: {relative}{suffix}")
    return failures, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    failures, entrypoints = _check_layout()
    help_failures, skipped = _check_cli_help(entrypoints)
    failures.extend(help_failures)
    for entry in skipped:
        print(f"SKIP: --help of {entry}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        "PASS: responsibility layout is clean and "
        f"{len(entrypoints) - len(skipped)} command-line entrypoints answer --help"
        + (f" ({len(skipped)} skipped for optional dependencies)" if skipped else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
