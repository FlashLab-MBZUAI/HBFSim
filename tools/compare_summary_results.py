#!/usr/bin/env python3
"""Compare reproducible HBFSim results while validating run provenance.

Invocation metadata is expected to differ between an original run and a
config replay. Scientific inputs, resolved configuration, and scenario
results must remain exactly equal.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_strict(path: Path) -> dict:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        value: dict = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{path}: duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)


def validate_contract(path: Path, data: dict) -> None:
    if data.get("schema") != {
        "name": "hbfsim.scenario_compare.summary",
        "version": 16,
    }:
        raise ValueError(f"{path}: unsupported or missing summary schema")
    simulator = data.get("simulator", {})
    for key in ("name", "version", "git_commit", "git_dirty"):
        if key not in simulator:
            raise ValueError(f"{path}: missing simulator.{key}")
    build = data.get("build", {})
    for key in ("type", "compiler_id", "compiler_version"):
        if not build.get(key):
            raise ValueError(f"{path}: missing build.{key}")
    invocation = data.get("invocation", {})
    if not invocation.get("working_directory") or not invocation.get("argv"):
        raise ValueError(f"{path}: incomplete invocation provenance")
    workload = data.get("workload", {})
    digest = workload.get("trace_digest", {})
    if digest.get("algorithm") != "sha256" or not re.fullmatch(
        r"[0-9a-f]{64}", digest.get("value", "")
    ):
        raise ValueError(f"{path}: invalid workload SHA-256")
    config = data.get("config", {})
    if not isinstance(config.get("hbm"), dict) or not isinstance(config.get("hbf"), dict):
        raise ValueError(f"{path}: incomplete resolved device configuration")


def reproducible_payload(data: dict) -> dict:
    return {
        "schema": data["schema"],
        "simulator": data["simulator"],
        "build": data["build"],
        "workload": data["workload"],
        "sanity": data["sanity"],
        "config": data["config"],
        "scenarios": data["scenarios"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()

    expected = load_strict(args.expected)
    actual = load_strict(args.actual)
    validate_contract(args.expected, expected)
    validate_contract(args.actual, actual)
    if reproducible_payload(expected) != reproducible_payload(actual):
        raise SystemExit(
            "simulation payload differs after config replay; compare "
            f"{args.expected} and {args.actual}")
    print("summary schema/provenance valid; replay payload is identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
