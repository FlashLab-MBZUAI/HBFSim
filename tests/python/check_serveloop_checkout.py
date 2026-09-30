#!/usr/bin/env python3
"""CTest fixture: is the ServeLoop checkout present and new enough?

HBFSim's ``hbfsim_client`` is the transaction-protocol source of truth, and
ServeLoop revisions older than the minimum still emit ``span_start``, which the
client rejects. CMake registers this script as the ``serveloop`` fixture with
the minimum revision from ``HBFSIM_SERVELOOP_MIN_REVISION`` in
``CMakeLists.txt`` (the only place that value is defined). When the check fails
every test that requires the fixture is reported as Not Run instead of failing
one by one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True, capture_output=True, check=False,
    )


def describe(root: Path, minimum: str) -> str | None:
    """Return None when the checkout is usable, else what is wrong with it."""

    if not root.is_dir():
        return "missing"
    if not (root / "hbserve" / "__init__.py").is_file():
        return "not a ServeLoop checkout (no hbserve/__init__.py)"
    try:
        head = git(root, "rev-parse", "--short", "HEAD")
    except OSError:
        return "not a git checkout (git is unavailable)"
    if head.returncode != 0:
        return "not a git checkout"
    revision = head.stdout.strip() or "unknown"
    ancestor = git(root, "merge-base", "--is-ancestor", minimum, "HEAD")
    if ancestor.returncode == 0:
        return None
    return f"{revision} (does not contain {minimum})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True,
                        help="ServeLoop source checkout to check")
    parser.add_argument("--minimum-revision", required=True,
                        help="oldest compatible ServeLoop commit "
                             "(HBFSIM_SERVELOOP_MIN_REVISION in CMakeLists.txt)")
    args = parser.parse_args()
    root = args.root.expanduser()
    minimum = args.minimum_revision
    problem = describe(root, minimum)
    if problem is None:
        print(f"ServeLoop checkout at {root} contains {minimum}")
        return 0
    print(
        f"ServeLoop checkout at {root} is {problem}; HBFSim requires ServeLoop "
        f">= {minimum} (the compiler revision the client protocol was written against; earlier "
        "revisions still emit span_start, which hbfsim_client rejects)"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
