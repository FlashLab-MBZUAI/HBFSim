"""Locate the source checkout, its build directory, and the compiled programs.

Everything here is read-only except :func:`build`, which runs CMake exactly as
the README documents so a first-time user never has to remember the flags.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ENGINE = "hbfsim"
REFERENCE_RUNNER = "hbfsim-reference"
# CMake target names differ from the output names of the two programs.
USER_TARGETS = ("hbfsim", "hbfsim_reference")
MINIMUM_PYTHON = (3, 10)
MINIMUM_CMAKE = (3, 20)


class HbfsimError(RuntimeError):
    """A user-facing failure whose message already says what to do next."""


def repository_root() -> Path:
    """Return the HBFSim source checkout (``HBFSIM_ROOT`` overrides)."""

    override = os.environ.get("HBFSIM_ROOT")
    root = Path(override).expanduser().resolve() if override else PACKAGE_ROOT
    if not (root / "CMakeLists.txt").is_file() or not (root / "configs").is_dir():
        raise HbfsimError(
            f"{root} is not an HBFSim source checkout (no CMakeLists.txt/configs). "
            "Run from a clone of the repository or set HBFSIM_ROOT."
        )
    return root


def build_directory() -> Path:
    """Return the CMake build directory (``HBFSIM_BUILD_DIR`` overrides)."""

    override = os.environ.get("HBFSIM_BUILD_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return repository_root() / "build"


def _executable_candidates(name: str, build_dir: Path) -> list[Path]:
    suffix = ".exe" if os.name == "nt" else ""
    # Multi-config generators (Visual Studio, Xcode) add a configuration folder.
    folders = [build_dir] + [build_dir / config for config in ("Release", "RelWithDebInfo", "Debug")]
    return [folder / f"{name}{suffix}" for folder in folders]


def find_executable(name: str) -> Path | None:
    """Return a compiled program from the build directory, or ``None``."""

    for candidate in _executable_candidates(name, build_directory()):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def require_executable(name: str) -> Path:
    """Return a compiled program or explain how to build it."""

    path = find_executable(name)
    if path is None:
        raise HbfsimError(
            f"{name} is not built yet (looked in {build_directory()}).\n"
            "Build it with:  python3 -m hbfsim build"
        )
    return path


def _version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if match is None:
        return ()
    return tuple(int(part) for part in match.groups() if part is not None)


def _first_line(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else None


def _compiler() -> str | None:
    for candidate in (os.environ.get("CXX"), "c++", "g++", "clang++", "cl"):
        if candidate and shutil.which(candidate):
            return candidate
    return None


@dataclass(frozen=True)
class Check:
    """One environment check reported by :func:`doctor`."""

    name: str
    ok: bool
    detail: str
    required: bool = True
    hint: str = ""


def doctor() -> list[Check]:
    """Inspect the toolchain, the build, and optional integrations."""

    checks: list[Check] = []
    python_ok = sys.version_info[:2] >= MINIMUM_PYTHON
    checks.append(Check(
        "Python >= 3.10", python_ok,
        f"{platform.python_implementation()} {platform.python_version()}",
        hint="Install Python 3.10 or newer.",
    ))

    cmake = shutil.which("cmake")
    cmake_line = _first_line([cmake, "--version"]) if cmake else None
    cmake_ok = bool(cmake_line) and _version_tuple(cmake_line or "") >= MINIMUM_CMAKE
    checks.append(Check(
        "CMake >= 3.20", cmake_ok, cmake_line or "not found",
        hint="Install CMake 3.20+ (e.g. `pip install cmake` or your package manager).",
    ))

    compiler = _compiler()
    compiler_line = _first_line([compiler, "--version"]) if compiler else None
    checks.append(Check(
        "C++20 compiler", compiler is not None,
        compiler_line or "not found (set CXX or install g++/clang++)",
        hint="Install a C++20 compiler (tested: GCC 13, Clang 18, Apple Clang 15).",
    ))

    ninja = shutil.which("ninja")
    checks.append(Check(
        "Ninja (optional, faster builds)", ninja is not None,
        ninja or "not found; CMake's default generator is used", required=False,
    ))
    git = shutil.which("git")
    checks.append(Check(
        "Git (optional, stamps source provenance)", git is not None,
        git or "not found; results record provenance as unavailable", required=False,
    ))

    build_dir = build_directory()
    for program in (ENGINE, REFERENCE_RUNNER):
        path = find_executable(program)
        checks.append(Check(
            f"{program} built", path is not None,
            str(path) if path else f"missing in {build_dir}",
            hint="Run `python3 -m hbfsim build`.",
        ))

    try:
        import matplotlib  # noqa: F401  # pragma: no cover - optional
        plotting = True
    except ImportError:
        plotting = False
    checks.append(Check(
        "matplotlib (optional, figures in examples/research)", plotting,
        "installed" if plotting else "not installed; text/CSV output still works",
        required=False,
    ))

    serveloop = serveloop_checkout()
    checks.append(Check(
        "ServeLoop checkout (optional, LLM-serving workloads)", serveloop is not None,
        str(serveloop) if serveloop else "not found at ../ServeLoop or $SERVELOOP_ROOT",
        required=False,
    ))
    return checks


def serveloop_checkout() -> Path | None:
    """Return a sibling or ``SERVELOOP_ROOT`` ServeLoop source checkout if present."""

    candidates = []
    if os.environ.get("SERVELOOP_ROOT"):
        candidates.append(Path(os.environ["SERVELOOP_ROOT"]).expanduser())
    candidates.append(repository_root().parent / "ServeLoop")
    for candidate in candidates:
        if (candidate / "hbserve" / "__init__.py").is_file():
            return candidate.resolve()
    return None


def build(
    *,
    build_type: str = "Release",
    all_targets: bool = False,
    jobs: int | None = None,
    echo=print,
) -> Path:
    """Configure (once) and build the simulator programs with CMake.

    By default only ``hbfsim`` and ``hbfsim-reference`` are compiled, which is
    all the quick start, examples and the ``hbfsim`` front door need. Pass
    ``all_targets=True`` for tests, probes and research drivers.
    """

    cmake = shutil.which("cmake")
    if cmake is None:
        raise HbfsimError("CMake was not found. Install CMake 3.20+ and retry.")
    root = repository_root()
    build_dir = build_directory()
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        configure = [cmake, "-S", str(root), "-B", str(build_dir),
                     f"-DCMAKE_BUILD_TYPE={build_type}"]
        if shutil.which("ninja"):
            configure += ["-G", "Ninja"]
        _run_step("Configuring", configure, echo)
    else:
        cached = re.search(
            r"^CMAKE_BUILD_TYPE:\w+=(.*)$",
            cache.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
        if cached and cached.group(1) and cached.group(1) != build_type:
            _run_step(
                "Reconfiguring",
                [cmake, "-S", str(root), "-B", str(build_dir),
                 f"-DCMAKE_BUILD_TYPE={build_type}"],
                echo,
            )
    command = [cmake, "--build", str(build_dir), "--parallel", str(jobs or os.cpu_count() or 2)]
    if os.name == "nt":
        command += ["--config", build_type]
    if not all_targets:
        command += ["--target", *USER_TARGETS]
    _run_step("Building", command, echo)
    return build_dir


def _run_step(label: str, command: Sequence[str], echo) -> None:
    echo(f"{label}: {' '.join(command)}")
    result = subprocess.run(list(command), check=False)
    if result.returncode != 0:
        raise HbfsimError(
            f"{label.lower()} failed (exit {result.returncode}). "
            "Run `python3 -m hbfsim doctor` to check the toolchain."
        )
