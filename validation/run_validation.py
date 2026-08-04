#!/usr/bin/env python3
"""Run HBFSim production code against the independent foundational oracle."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.compare_ledgers import (
    first_difference,
    load_ledger,
    validate_ledger,
)
from validation.contracts import ContractError, dump_json_line, load_case
from validation.hbm_oracle import build_ledger
from validation.hbf_oracle import build_ledger as build_hbf_ledger
from validation.hybrid_oracle import build_ledger as build_hybrid_ledger
from validation.external_oracle import build_ledger as build_external_ledger


def production_command(
    probe: Path,
    case: dict,
) -> list[str]:
    command = [
        str(probe),
        "--case-id", case["case_id"],
        "--model", case["model"],
    ]
    config_assignments: list[tuple[str, object]] = []
    for key in sorted(case["config_overrides"]):
        value = case["config_overrides"][key]
        if case["model"] == "hybrid":
            for nested_key in sorted(value):
                config_assignments.append(
                    (f"{key}.{nested_key}", value[nested_key]))
        else:
            config_assignments.append((key, value))
    for key, value in config_assignments:
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        command.extend(["--config", f"{key}={rendered}"])
    for address in case["inspect_addresses"]:
        command.extend(["--inspect-address", str(address)])
    for lpn in case["initial_state"].get("prepopulate_lpns", []):
        command.extend(["--prepopulate-lpn", str(lpn)])
    for request in case["requests"]:
        command.extend([
            "--request",
            request["id"],
            repr(request["arrival_ns"]),
            request["op"],
            request["address_space"],
            str(request["addr"]),
            str(request["bytes"]),
        ])
    return command


def write_ledger(path: Path, records: list[dict]) -> None:
    path.write_text("".join(f"{dump_json_line(record)}\n" for record in records))


def run_case(
    case_path: Path,
    *,
    probe: Path,
    artifact_dir: Path,
) -> None:
    case = load_case(case_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    expected_path = artifact_dir / f"{case['case_id']}.expected.jsonl"
    actual_path = artifact_dir / f"{case['case_id']}.actual.jsonl"
    expected_builder = {
        "hbm": build_ledger,
        "hbf": build_hbf_ledger,
        "hybrid": build_hybrid_ledger,
        "external": build_external_ledger,
    }[case["model"]]
    expected = expected_builder(case)
    validate_ledger(expected, source=str(expected_path))
    write_ledger(expected_path, expected)

    command = production_command(probe, case)
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    actual_path.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(
            "production validation probe failed\n"
            f"command={' '.join(command)}\n"
            f"stderr={result.stderr}")
    actual = load_ledger(actual_path)
    difference = first_difference(
        expected,
        actual,
        abs_tolerance=case["comparison"]["time_abs_tolerance_ns"],
        rel_tolerance=case["comparison"]["time_rel_tolerance"],
    )
    if difference:
        raise RuntimeError(difference.format(
            case_id=case["case_id"],
            replay=" ".join(command),
        ))
    print(
        f"PASS {case['case_id']}: "
        f"{len(actual)} canonical records match independent oracle")


def discover_cases(case_args: list[Path], case_dir: Path | None) -> list[Path]:
    cases = list(case_args)
    if case_dir is not None:
        cases.extend(sorted(case_dir.glob("*.json")))
    unique = sorted(set(path.resolve() for path in cases))
    if not unique:
        raise ContractError("no validation cases selected")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--case", action="append", default=[], type=Path)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    if not args.probe.is_file():
        parser.error(f"validation probe not found: {args.probe}")
    try:
        cases = discover_cases(args.case, args.case_dir)
        if args.artifact_dir is None:
            with tempfile.TemporaryDirectory(
                prefix="hbfsim-validation-"
            ) as directory:
                artifact_dir = Path(directory)
                for case_path in cases:
                    run_case(
                        case_path,
                        probe=args.probe.resolve(),
                        artifact_dir=artifact_dir,
                    )
        else:
            for case_path in cases:
                run_case(
                    case_path,
                    probe=args.probe.resolve(),
                    artifact_dir=args.artifact_dir,
                )
    except (ContractError, RuntimeError, ValueError) as error:
        print(f"foundational validation failed: {error}", file=sys.stderr)
        return 1
    print(f"foundational validation: {len(cases)} case(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
