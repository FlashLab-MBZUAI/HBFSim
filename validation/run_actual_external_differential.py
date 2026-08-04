#!/usr/bin/env python3
"""Build pinned external tools and emit actual L3 differential evidence."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.contracts import (  # noqa: E402
    ContractError,
    load_case,
)
from validation.external_differential import (  # noqa: E402
    EVIDENCE_SCHEMA,
    SOURCE_SCHEMA,
    WRAPPER_SCHEMA,
    build_report,
    load_tools,
)
from validation.run_validation import production_command  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "validation/external/tools.json"
FIXTURES = ROOT / "validation/external/fixtures"
RAMULATOR_DRIVER = ROOT / "validation/external/ramulator2_driver.cpp"


class EvidenceError(RuntimeError):
    """Actual external evidence could not be produced safely."""


def _run(
    command: list[str],
    *,
    cwd: Path,
    stdin: int | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdin=stdin,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode:
        raise EvidenceError(
            f"command failed with exit code {completed.returncode}: "
            f"{command!r}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _run_bytes(command: list[str], *, cwd: Path) -> bytes:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise EvidenceError(
            f"command failed with exit code {completed.returncode}: "
            f"{command!r}\nstderr:\n"
            f"{completed.stderr.decode(errors='replace')}"
        )
    return completed.stdout


def _git(root: Path, *arguments: str) -> str:
    return _run(
        ["git", *arguments],
        cwd=root,
        timeout=60,
    ).stdout.strip()


def _verify_git_checkout(
    root: Path,
    *,
    commit: str,
    tree: str | None = None,
    label: str,
) -> None:
    root = root.resolve()
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise EvidenceError(
            f"{label} must be its Git root: expected {top}, got {root}")
    observed_commit = _git(root, "rev-parse", "HEAD")
    if observed_commit != commit:
        raise EvidenceError(
            f"{label} commit drift: expected {commit}, "
            f"observed {observed_commit}")
    if tree is not None:
        observed_tree = _git(root, "rev-parse", "HEAD^{tree}")
        if observed_tree != tree:
            raise EvidenceError(
                f"{label} tree drift: expected {tree}, "
                f"observed {observed_tree}")
    tracked_status = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise EvidenceError(
            f"{label} has tracked modifications:\n{tracked_status}")


def _extract_git_tree(repository: Path, destination: Path) -> None:
    payload = _run_bytes(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=repository,
    )
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                try:
                    target.relative_to(destination.resolve())
                except ValueError as error:
                    raise EvidenceError(
                        f"Git archive member escapes destination: "
                        f"{member.name}") from error
                if member.issym() or member.islnk():
                    raise EvidenceError(
                        f"Git archive contains unsupported link: "
                        f"{member.name}")
            archive.extractall(destination)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact(evidence_root: Path, path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(evidence_root.resolve())
    return {
        "path": relative.as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _append_log(
    log: Path,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("command=" + json.dumps(command) + "\n")
        stream.write(f"exit_code={completed.returncode}\n")
        stream.write("stdout:\n")
        stream.write(completed.stdout)
        if completed.stdout and not completed.stdout.endswith("\n"):
            stream.write("\n")
        stream.write("stderr:\n")
        stream.write(completed.stderr)
        if completed.stderr and not completed.stderr.endswith("\n"):
            stream.write("\n")


def _copy_executable(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise EvidenceError(f"executable does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | 0o111)


def _source_provenance(
    tool: dict[str, Any],
    path: Path,
) -> None:
    _write_json(path, {
        "schema": SOURCE_SCHEMA,
        "repository": tool["repository"],
        "commit": tool["commit"],
        "tree": tool["source_tree"],
        "tracked_clean": True,
        "dependencies": tool["dependencies"],
    })


def _compiler(cxx_argument: str) -> tuple[str, str]:
    resolved = shutil.which(cxx_argument)
    if resolved is None:
        raise EvidenceError(f"C++ compiler not found: {cxx_argument}")
    path = str(Path(resolved).resolve())
    version = _run([path, "--version"], cwd=ROOT, timeout=30).stdout.strip()
    if not version:
        raise EvidenceError(f"C++ compiler emitted no version: {path}")
    return path, version


def _ramulator_configure_command(
    *,
    build_source: Path,
    cmake_build: Path,
    cxx: str,
) -> list[str]:
    return [
        "cmake",
        "-S",
        str(build_source),
        "-B",
        str(cmake_build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DRAMULATOR_PYTHON_BINDINGS=OFF",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        (
            "-DFETCHCONTENT_SOURCE_DIR_FMT="
            f"{build_source / 'ext' / 'fmt'}"
        ),
        (
            "-DFETCHCONTENT_SOURCE_DIR_YAML_CPP="
            f"{build_source / 'ext' / 'yaml-cpp'}"
        ),
        f"-DCMAKE_CXX_COMPILER={cxx}",
    ]


def _build_ramulator(
    tool: dict[str, Any],
    *,
    source_root: Path,
    cxx: str,
    parallel: int,
    workspace: Path,
    evidence_root: Path,
) -> tuple[dict[str, Any], Path]:
    _verify_git_checkout(
        source_root,
        commit=tool["commit"],
        tree=tool["source_tree"],
        label="Ramulator2",
    )
    dependency_roots: dict[str, Path] = {}
    for dependency in tool["dependencies"]:
        dependency_root = source_root / "ext" / dependency["id"]
        _verify_git_checkout(
            dependency_root,
            commit=dependency["commit"],
            label=f"Ramulator2 dependency {dependency['id']}",
        )
        dependency_roots[dependency["id"]] = dependency_root

    build_source = workspace / "ramulator2-source"
    _extract_git_tree(source_root, build_source)
    for dependency in tool["dependencies"]:
        destination = build_source / "ext" / dependency["id"]
        if destination.exists():
            raise EvidenceError(
                f"Ramulator archive unexpectedly contains {destination}")
        _extract_git_tree(
            dependency_roots[dependency["id"]],
            destination,
        )
    cmake_build = workspace / "ramulator2-cmake-build"
    configure = _ramulator_configure_command(
        build_source=build_source,
        cmake_build=cmake_build,
        cxx=cxx,
    )
    compile_library = [
        "cmake",
        "--build",
        str(cmake_build),
        "--target",
        "ramulator",
        "--parallel",
        str(parallel),
    ]
    build_dir = evidence_root / tool["id"] / "build"
    build_dir.mkdir(parents=True)
    build_log = build_dir / "build.log"
    configured = _run(configure, cwd=workspace)
    _append_log(build_log, configure, configured)
    compiled = _run(
        compile_library,
        cwd=workspace,
        timeout=1200,
    )
    _append_log(build_log, compile_library, compiled)

    libraries = [
        path
        for path in build_source.glob("libramulator.*")
        if path.is_file() and path.suffix in {".dylib", ".so"}
    ]
    if len(libraries) != 1:
        raise EvidenceError(
            f"expected one Ramulator shared library, found {libraries}")
    library = build_dir / libraries[0].name
    shutil.copy2(libraries[0], library)

    driver = build_dir / "ramulator2_driver"
    rpath = "@loader_path" if sys.platform == "darwin" else "$ORIGIN"
    compile_driver = [
        cxx,
        "-std=c++20",
        "-O2",
        "-I",
        str(build_source / "src"),
        str(RAMULATOR_DRIVER),
        str(libraries[0]),
        f"-Wl,-rpath,{rpath}",
        "-o",
        str(driver),
    ]
    compiled_driver = _run(
        compile_driver,
        cwd=workspace,
        timeout=300,
    )
    _append_log(build_log, compile_driver, compiled_driver)
    driver.chmod(driver.stat().st_mode | 0o111)
    return ({
        "commands": [configure, compile_library, compile_driver],
        "artifacts": {
            "driver": _artifact(evidence_root, driver),
            "library": _artifact(evidence_root, library),
            "build_log": _artifact(evidence_root, build_log),
        },
    }, driver)


def _build_mqsim(
    tool: dict[str, Any],
    *,
    source_root: Path,
    cxx: str,
    parallel: int,
    workspace: Path,
    evidence_root: Path,
) -> tuple[dict[str, Any], Path]:
    _verify_git_checkout(
        source_root,
        commit=tool["commit"],
        tree=tool["source_tree"],
        label="MQSim",
    )
    build_source = workspace / "mqsim-source"
    _extract_git_tree(source_root, build_source)
    command = [
        "make",
        f"-j{parallel}",
        f"CC={cxx}",
        f"LD={cxx}",
    ]
    build_dir = evidence_root / tool["id"] / "build"
    build_dir.mkdir(parents=True)
    build_log = build_dir / "build.log"
    completed = _run(
        command,
        cwd=build_source,
        timeout=1200,
    )
    _append_log(build_log, command, completed)
    executable = build_dir / "MQSim"
    _copy_executable(build_source / "MQSim", executable)
    return ({
        "commands": [command],
        "artifacts": {
            "executable": _artifact(evidence_root, executable),
            "build_log": _artifact(evidence_root, build_log),
        },
    }, executable)


def _run_hbfsim_case(
    case_path: Path,
    *,
    probe: Path,
    destination: Path,
) -> list[str]:
    case = load_case(case_path)
    command = production_command(probe, case)
    completed = _run(command, cwd=ROOT, timeout=120)
    destination.write_text(completed.stdout, encoding="utf-8")
    rendered = [probe.name, *command[1:]]
    return rendered


def _ramulator_wrapper(
    tool: dict[str, Any],
    case_id: str,
    stdout: str,
) -> dict[str, Any]:
    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise EvidenceError(
            f"Ramulator driver did not emit one JSON object: {error}") from error
    if not isinstance(raw, dict):
        raise EvidenceError("Ramulator driver output is not an object")
    return {
        "schema": WRAPPER_SCHEMA,
        "adapter": tool["adapter"],
        "tool_id": tool["id"],
        "tool_commit": tool["commit"],
        "provenance": "actual",
        "case_id": case_id,
        "raw": raw,
    }


def _mqsim_wrapper(
    tool: dict[str, Any],
    case_id: str,
    output_path: Path,
) -> dict[str, Any]:
    try:
        root = ET.parse(output_path).getroot()
    except (ET.ParseError, OSError) as error:
        raise EvidenceError(
            f"cannot parse MQSim output {output_path}: {error}") from error
    ftl_nodes = [
        node for node in root.iter()
        if node.tag.endswith(".FTL")
    ]
    if len(ftl_nodes) != 1:
        raise EvidenceError(
            f"expected one MQSim FTL report node, found {len(ftl_nodes)}")
    attributes = ftl_nodes[0].attrib

    def count(name: str) -> int:
        if name not in attributes:
            raise EvidenceError(f"MQSim FTL report omits {name}")
        try:
            value = int(attributes[name])
        except ValueError as error:
            raise EvidenceError(
                f"MQSim FTL report {name} is not an integer") from error
        if value < 0:
            raise EvidenceError(
                f"MQSim FTL report {name} is negative")
        return value

    return {
        "schema": WRAPPER_SCHEMA,
        "adapter": tool["adapter"],
        "tool_id": tool["id"],
        "tool_commit": tool["commit"],
        "provenance": "actual",
        "case_id": case_id,
        "raw": {
            "page_size_bytes": 512,
            "issued_flash_reads": count("Issued_Flash_Read_CMD"),
            "issued_flash_programs": count("Issued_Flash_Program_CMD"),
            "mapping_flash_reads": count(
                "Issued_Flash_Read_CMD_For_Mapping"),
            "mapping_flash_programs": count(
                "Issued_Flash_Program_CMD_For_Mapping"),
            "issued_flash_erases": count("Issued_Flash_Erase_CMD"),
            "gc_executions": count("Total_GC_Executions"),
        },
    }


def _copy_case_inputs(
    case_spec: dict[str, Any],
    *,
    case_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, repository_path in case_spec["external_inputs"].items():
        source = ROOT / repository_path
        suffixes = {
            "config": source.suffix,
            "workload": source.suffix,
            "trace": ".txt",
        }
        destination = case_dir / f"{name}{suffixes[name]}"
        shutil.copy2(source, destination)
        paths[name] = destination
    case_copy = case_dir / "hbfsim-case.json"
    shutil.copy2(ROOT / case_spec["hbfsim_case"], case_copy)
    paths["hbfsim_case"] = case_copy
    return paths


def _run_ramulator_cases(
    tool: dict[str, Any],
    *,
    driver: Path,
    probe: Path,
    evidence_root: Path,
) -> list[dict[str, Any]]:
    results = []
    for case_spec in tool["cases"]:
        case_id = case_spec["case_id"]
        case_dir = evidence_root / tool["id"] / "cases" / case_id
        case_dir.mkdir(parents=True)
        paths = _copy_case_inputs(case_spec, case_dir=case_dir)
        ledger = case_dir / "hbfsim-ledger.jsonl"
        hbfsim_command = _run_hbfsim_case(
            paths["hbfsim_case"],
            probe=probe,
            destination=ledger,
        )
        paths["hbfsim_ledger"] = ledger
        external_command = [
            Path(os.path.relpath(driver, case_dir)).as_posix(),
            paths["config"].name,
            paths["trace"].name,
        ]
        completed = _run(external_command, cwd=case_dir, timeout=120)
        run_log = case_dir / "run.log"
        _append_log(run_log, external_command, completed)
        paths["run_log"] = run_log
        raw_output = case_dir / "raw-wrapper.json"
        _write_json(
            raw_output,
            _ramulator_wrapper(tool, case_id, completed.stdout),
        )
        paths["raw_output"] = raw_output
        artifacts = {
            name: _artifact(evidence_root, paths[name])
            for name in tool["required_case_artifacts"]
        }
        results.append({
            "case_id": case_id,
            "commands": {
                "external": external_command,
                "hbfsim": hbfsim_command,
            },
            "artifacts": artifacts,
        })
    return results


def _run_mqsim_cases(
    tool: dict[str, Any],
    *,
    executable: Path,
    probe: Path,
    evidence_root: Path,
) -> list[dict[str, Any]]:
    results = []
    for case_spec in tool["cases"]:
        case_id = case_spec["case_id"]
        case_dir = evidence_root / tool["id"] / "cases" / case_id
        case_dir.mkdir(parents=True)
        paths = _copy_case_inputs(case_spec, case_dir=case_dir)
        ledger = case_dir / "hbfsim-ledger.jsonl"
        hbfsim_command = _run_hbfsim_case(
            paths["hbfsim_case"],
            probe=probe,
            destination=ledger,
        )
        paths["hbfsim_ledger"] = ledger
        external_command = [
            Path(os.path.relpath(executable, case_dir)).as_posix(),
            "-i",
            paths["config"].name,
            "-w",
            paths["workload"].name,
        ]
        completed = _run(
            external_command,
            cwd=case_dir,
            stdin=subprocess.DEVNULL,
            timeout=120,
        )
        run_log = case_dir / "run.log"
        _append_log(run_log, external_command, completed)
        paths["run_log"] = run_log
        tool_output = case_dir / (
            paths["workload"].stem + "_scenario_1.xml")
        if not tool_output.is_file():
            raise EvidenceError(
                f"MQSim did not emit expected output: {tool_output}")
        paths["tool_output"] = tool_output
        raw_output = case_dir / "raw-wrapper.json"
        _write_json(
            raw_output,
            _mqsim_wrapper(tool, case_id, tool_output),
        )
        paths["raw_output"] = raw_output
        artifacts = {
            name: _artifact(evidence_root, paths[name])
            for name in tool["required_case_artifacts"]
        }
        results.append({
            "case_id": case_id,
            "commands": {
                "external": external_command,
                "hbfsim": hbfsim_command,
            },
            "artifacts": artifacts,
        })
    return results


def _emit_tool_evidence(
    tool: dict[str, Any],
    *,
    compiler_path: str,
    compiler_version: str,
    build: dict[str, Any],
    probe_artifact: dict[str, Any],
    cases: list[dict[str, Any]],
    evidence_root: Path,
) -> None:
    source_path = evidence_root / tool["id"] / "source-provenance.json"
    _source_provenance(tool, source_path)
    _write_json(
        evidence_root / f"{tool['id']}.evidence.json",
        {
            "schema": EVIDENCE_SCHEMA,
            "tool_id": tool["id"],
            "tool_commit": tool["commit"],
            "adapter": tool["adapter"],
            "provenance": "actual",
            "source": _artifact(evidence_root, source_path),
            "build": {
                "compiler": {
                    "path": compiler_path,
                    "version": compiler_version,
                },
                "commands": build["commands"],
                "artifacts": build["artifacts"],
            },
            "hbfsim_probe": probe_artifact,
            "cases": cases,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hbfsim-probe", required=True, type=Path)
    parser.add_argument("--ramulator-root", required=True, type=Path)
    parser.add_argument("--mqsim-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--cxx",
        default=("g++-16" if sys.platform == "darwin" else "g++"),
    )
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()
    if args.parallel <= 0:
        parser.error("--parallel must be positive")
    probe_source = args.hbfsim_probe.resolve()
    if not probe_source.is_file():
        parser.error(f"HBFSim validation probe not found: {probe_source}")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(
            f"output directory already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.staging-",
        dir=output.parent,
    ))
    workspace = Path(tempfile.mkdtemp(
        prefix="hbfsim-external-build-",
        dir=output.parent,
    ))
    try:
        tools = {tool["id"]: tool for tool in load_tools(MANIFEST)}
        if set(tools) != {"ramulator2", "mqsim"}:
            raise EvidenceError("external manifest tool census changed")
        cxx, compiler_version = _compiler(args.cxx)
        probe = staging / "hbfsim-validation_probe"
        _copy_executable(probe_source, probe)
        probe_artifact = _artifact(staging, probe)

        ramulator_build, ramulator_driver = _build_ramulator(
            tools["ramulator2"],
            source_root=args.ramulator_root.resolve(),
            cxx=cxx,
            parallel=args.parallel,
            workspace=workspace,
            evidence_root=staging,
        )
        ramulator_cases = _run_ramulator_cases(
            tools["ramulator2"],
            driver=ramulator_driver,
            probe=probe,
            evidence_root=staging,
        )
        _emit_tool_evidence(
            tools["ramulator2"],
            compiler_path=cxx,
            compiler_version=compiler_version,
            build=ramulator_build,
            probe_artifact=probe_artifact,
            cases=ramulator_cases,
            evidence_root=staging,
        )

        mqsim_build, mqsim_executable = _build_mqsim(
            tools["mqsim"],
            source_root=args.mqsim_root.resolve(),
            cxx=cxx,
            parallel=args.parallel,
            workspace=workspace,
            evidence_root=staging,
        )
        mqsim_cases = _run_mqsim_cases(
            tools["mqsim"],
            executable=mqsim_executable,
            probe=probe,
            evidence_root=staging,
        )
        _emit_tool_evidence(
            tools["mqsim"],
            compiler_path=cxx,
            compiler_version=compiler_version,
            build=mqsim_build,
            probe_artifact=probe_artifact,
            cases=mqsim_cases,
            evidence_root=staging,
        )

        report = build_report(
            MANIFEST,
            FIXTURES,
            staging,
            repository=ROOT,
            expected_hbfsim_probe=probe_source,
        )
        if report["l3_status"] != "pass":
            raise EvidenceError(
                f"actual evidence did not produce L3 pass: "
                f"{report['l3_status']}")
        _write_json(staging / "external-report.json", report)
        os.replace(staging, output)
    except (
        ContractError,
        EvidenceError,
        OSError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"actual external differential failed: {error}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    case_count = sum(
        len(tool["actual"]["cases"])
        for tool in report["tools"]
    )
    print(
        f"PASS actual external differential: "
        f"{case_count} cases, "
        f"{len(report['validated_facets'])} facets, "
        f"L3={report['l3_status']}, "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
