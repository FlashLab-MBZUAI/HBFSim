#!/usr/bin/env python3
"""Deterministic property fuzzing and failure shrinking for validation cases."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verification.core.ledger_compare import (  # noqa: E402
    Difference,
    first_difference,
    load_ledger,
    validate_ledger,
)
from verification.core.contracts import (  # noqa: E402
    CASE_SCHEMA,
    ContractError,
    validate_case,
)
from verification.oracles.hbf import (  # noqa: E402
    build_ledger as build_hbf_ledger,
)
from verification.oracles.hbm import build_ledger as build_hbm_ledger  # noqa: E402
from verification.oracles.hybrid import (  # noqa: E402
    build_ledger as build_hybrid_ledger,
)
from verification.oracles.external import (  # noqa: E402
    build_ledger as build_external_ledger,
)
from verification.gates.cases import production_command  # noqa: E402


MASK64 = (1 << 64) - 1
MODEL_SALTS = {
    "hbm": 0x48424D5F46555A5A,
    "hbf": 0x4842465F46555A5A,
    "hybrid": 0x4859425F46555A5A,
    "external": 0x4558545F46555A5A,
}
BUILDERS = {
    "hbm": build_hbm_ledger,
    "hbf": build_hbf_ledger,
    "hybrid": build_hybrid_ledger,
    "external": build_external_ledger,
}


class SplitMix64:
    """Version-stable generator; no dependency on Python random internals."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (
            self.state + 0x9E3779B97F4A7C15
        ) & MASK64
        value = self.state
        value = (
            (value ^ (value >> 30))
            * 0xBF58476D1CE4E5B9
        ) & MASK64
        value = (
            (value ^ (value >> 27))
            * 0x94D049BB133111EB
        ) & MASK64
        return (value ^ (value >> 31)) & MASK64

    def below(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("SplitMix64 upper bound must be positive")
        limit = ((1 << 64) // upper) * upper
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper

    def choose(self, values: list[Any]) -> Any:
        return values[self.below(len(values))]


def _comparison() -> dict[str, float]:
    return {
        "time_abs_tolerance_ns": 1e-9,
        "time_rel_tolerance": 1e-12,
    }


def _request(
    index: int,
    arrival_ns: float,
    op: str,
    addr: int,
    byte_count: int,
) -> dict[str, Any]:
    return {
        "id": f"r{index}",
        "arrival_ns": arrival_ns,
        "op": op,
        "address_space": "logical",
        "addr": addr,
        "bytes": byte_count,
    }


def _generate_hbm(seed: int, max_requests: int) -> dict[str, Any]:
    rng = SplitMix64(seed ^ MODEL_SALTS["hbm"])
    pseudo_channels = rng.choose([1, 2])
    channel_width = 64
    burst_length = 8
    burst_bytes = (
        channel_width // pseudo_channels // 8 * burst_length)
    request_count = 1 + rng.below(max_requests)
    arrival_ns = 0.0
    requests = []
    for index in range(request_count):
        arrival_ns += rng.choose([0.0, 0.625, 1.25, 5.0])
        byte_count = rng.choose([
            1,
            max(1, burst_bytes // 2),
            burst_bytes,
            burst_bytes + 1,
            2 * burst_bytes,
        ])
        base = rng.below(32768 - byte_count)
        if rng.below(4) == 0:
            base = (
                base // burst_bytes * burst_bytes
                + burst_bytes - 1
            )
        requests.append(_request(
            index,
            arrival_ns,
            rng.choose(["read", "write"]),
            base,
            byte_count,
        ))
    return {
        "schema": CASE_SCHEMA,
        "case_id": f"fuzz.hbm.seed-{seed:016x}",
        "model": "hbm",
        "comparison": _comparison(),
        "config": {
            "capacity_bytes": 65536,
            "stacks": rng.choose([1, 2]),
            "channels_per_stack": rng.choose([1, 2]),
            "pseudo_channels_per_channel": pseudo_channels,
            "bank_groups_per_pseudo_channel": rng.choose([1, 2]),
            "banks_per_group": rng.choose([1, 2]),
            "service_group_channels": rng.choose([1, 4]),
        },
        "initial_state": {},
        "requests": requests,
    }


def _generate_hbf(seed: int, max_requests: int) -> dict[str, Any]:
    rng = SplitMix64(seed ^ MODEL_SALTS["hbf"])
    page_size = 4096
    request_count = 1 + rng.below(max_requests)
    requests = []
    for index in range(request_count):
        lpn = rng.below(4)
        op = rng.choose(["read", "write"])
        if rng.below(4) == 0:
            offset = rng.choose([0, 64, 128, 256])
            byte_count = rng.choose([64, 128, page_size - offset])
            byte_count = min(byte_count, page_size - offset)
        else:
            offset = 0
            byte_count = page_size
        requests.append(_request(
            index,
            index * 250_000.0,
            op,
            lpn * page_size + offset,
            byte_count,
        ))
    requests.append({
        "id": "drain",
        "arrival_ns": request_count * 250_000.0 + 1_000_000.0,
        "op": "drain",
        "address_space": "internal",
        "addr": 0,
        "bytes": 0,
    })
    return {
        "schema": CASE_SCHEMA,
        "case_id": f"fuzz.hbf.seed-{seed:016x}",
        "model": "hbf",
        "comparison": _comparison(),
        "config": {
            "stacks": rng.choose([1, 2]),
            "channels_per_stack": 1,
            "dies_per_channel": 1,
            "planes_per_die": rng.choose([1, 2]),
            "blocks_per_plane": 16,
            "pages_per_block": 4,
            "page_size_bytes": page_size,
            "oob_bytes_per_page": 128,
            "media_lanes_per_plane": 2,
            "page_buffer_banks_per_plane": 2,
            "mapping_entries_per_page": 8,
            "write_coalescing_enabled": False,
            "auto_gc_enabled": False,
            "mapping_control_compute_ns": rng.choose([0.0, 5.0]),
        },
        "initial_state": {
            "prepopulate_lpns": [0, 1, 2, 3],
        },
        "requests": requests,
    }


def _hybrid_hbm_config(rng: SplitMix64) -> dict[str, Any]:
    return {
        "capacity_bytes": 65536,
        "stacks": 1,
        "channels_per_stack": rng.choose([1, 2]),
        "pseudo_channels_per_channel": rng.choose([1, 2]),
        "bank_groups_per_pseudo_channel": rng.choose([1, 2]),
        "banks_per_group": rng.choose([1, 2]),
        "service_group_channels": rng.choose([1, 4]),
    }


def _hybrid_hbf_config(rng: SplitMix64) -> dict[str, Any]:
    return {
        "stacks": 1,
        "channels_per_stack": 1,
        "dies_per_channel": 1,
        "planes_per_die": rng.choose([1, 2]),
        "blocks_per_plane": 16,
        "pages_per_block": 4,
        "page_size_bytes": 4096,
        "oob_bytes_per_page": 128,
        "media_lanes_per_plane": 2,
        "page_buffer_banks_per_plane": 2,
        "mapping_entries_per_page": 512,
        "auto_gc_enabled": False,
    }


def _generate_hybrid(seed: int, max_requests: int) -> dict[str, Any]:
    rng = SplitMix64(seed ^ MODEL_SALTS["hybrid"])
    page_size = 4096
    boundary_page = 8
    request_count = max(2, 1 + rng.below(max_requests))
    pages = [0, boundary_page]
    pages.extend(
        rng.choose([
            rng.below(boundary_page),
            boundary_page + rng.below(8),
        ])
        for _ in range(request_count - 2)
    )
    requests = [
        _request(
            index,
            float((index // 4) * 100),
            "read",
            page * page_size,
            page_size,
        )
        for index, page in enumerate(pages)
    ]
    return {
        "schema": CASE_SCHEMA,
        "case_id": f"fuzz.hybrid.seed-{seed:016x}",
        "model": "hybrid",
        "comparison": _comparison(),
        "config": {
            "hbm": _hybrid_hbm_config(rng),
            "hbf": _hybrid_hbf_config(rng),
            "policy": {
                "kind": "flat",
                "read_boundary": boundary_page * page_size,
            },
            "knobs": {
                "max_outstanding_requests": 0,
            },
        },
        "initial_state": {},
        "requests": requests,
    }


def _generate_external(seed: int, max_requests: int) -> dict[str, Any]:
    rng = SplitMix64(seed ^ MODEL_SALTS["external"])
    page_size = rng.choose([256, 512, 4096])
    request_segment_bytes = page_size * rng.choose([1, 2, 4])
    channels = rng.choose([1, 2, 4])
    request_count = 1 + rng.below(max_requests)
    arrival_ns = 0.0
    requests = []
    for index in range(request_count):
        arrival_ns += rng.choose([0.0, 0.25, 2.0, 25.0, 500.0])
        offset = rng.choose([0, 1, page_size // 4, page_size // 2])
        byte_count = rng.choose([
            1,
            max(1, page_size // 8),
            max(1, page_size // 2),
            page_size - offset,
            2 * page_size - offset,
            4 * page_size - offset,
        ])
        page = rng.below(56)
        requests.append(_request(
            index,
            arrival_ns,
            rng.choose(["read", "write"]),
            page * page_size + offset,
            byte_count,
        ))
    return {
        "schema": CASE_SCHEMA,
        "case_id": f"fuzz.external.seed-{seed:016x}",
        "model": "external",
        "comparison": _comparison(),
        "config": {
            "kind": rng.choose(
                ["host-dram", "cxl-memory", "nvme-ssd", "cxl-ssd"]
            ),
            "capacity_bytes": 64 * page_size,
            "page_size_bytes": page_size,
            "request_segment_bytes": request_segment_bytes,
            "media_channels": channels,
            "media_read_queues": rng.choose([1, 2, 4]),
            "media_write_queues": rng.choose([1, 2, 4]),
            "max_outstanding_requests": rng.choose([1, 2, 4, 8]),
            "controller_issue_ns": rng.choose([0.0, 1.0, 7.0]),
            "controller_processing_ns": rng.choose([0.0, 5.0, 100.0]),
            "media_read_latency_ns": rng.choose([0.0, 20.0, 1000.0]),
            "media_write_latency_ns": rng.choose([0.0, 50.0, 2500.0]),
            "media_read_bandwidth_GBps": rng.choose([1.0, 16.0, 128.0]),
            "media_write_bandwidth_GBps": rng.choose([0.5, 8.0, 64.0]),
            "m2s_bandwidth_GBps": rng.choose([1.0, 16.0, 64.0]),
            "s2m_bandwidth_GBps": rng.choose([1.0, 16.0, 64.0]),
            "one_way_propagation_ns": rng.choose([0.0, 10.0, 250.0]),
            "command_bytes": rng.choose([16, 64, 128]),
            "completion_bytes": rng.choose([8, 16, 64]),
        },
        "initial_state": {},
        "inspect_addresses": [
            index * page_size for index in range(min(channels, 4))
        ],
        "requests": requests,
    }


GENERATORS = {
    "hbm": _generate_hbm,
    "hbf": _generate_hbf,
    "hybrid": _generate_hybrid,
    "external": _generate_external,
}


def generate_case(
    model: str,
    seed: int,
    *,
    max_requests: int = 50,
) -> dict[str, Any]:
    if model not in GENERATORS:
        raise ValueError(f"unknown property model: {model}")
    if not 1 <= max_requests <= 50:
        raise ValueError("max_requests must be in [1, 50]")
    case = GENERATORS[model](seed & MASK64, max_requests)
    validate_case(case, source=f"generated:{model}:{seed}")
    return case


@dataclass(frozen=True)
class Failure:
    kind: str
    message: str
    field_path: str | None = None

    def same_signature(self, other: "Failure | None") -> bool:
        return (
            other is not None
            and self.kind == other.kind
            and self.field_path == other.field_path
        )


@dataclass
class Evaluation:
    normalized_case: dict[str, Any] | None
    actual: list[dict[str, Any]] | None
    failure: Failure | None


class Evaluator:
    def __init__(self, probe: Path, work_dir: Path) -> None:
        self.probe = probe.resolve()
        self.work_dir = work_dir
        self.executions = 0

    def evaluate(self, raw_case: dict[str, Any]) -> Evaluation:
        try:
            case = validate_case(raw_case, source="<generated>")
            expected = BUILDERS[case["model"]](case)
            validate_ledger(expected, source="<generated-oracle>")
        except (ContractError, ValueError) as error:
            return Evaluation(
                None,
                None,
                Failure("oracle-or-contract", str(error)),
            )
        command = production_command(self.probe, case)
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.executions += 1
        if result.returncode:
            return Evaluation(
                case,
                None,
                Failure(
                    "production-error",
                    "command=" + " ".join(command)
                    + "\nstderr=" + result.stderr,
                ),
            )
        actual_path = self.work_dir / "actual.jsonl"
        actual_path.write_text(result.stdout)
        try:
            actual = load_ledger(actual_path)
        except ContractError as error:
            return Evaluation(
                case,
                None,
                Failure("ledger-contract", str(error)),
            )
        difference = first_difference(
            expected,
            actual,
            abs_tolerance=case["comparison"][
                "time_abs_tolerance_ns"],
            rel_tolerance=case["comparison"]["time_rel_tolerance"],
        )
        if difference is not None:
            return Evaluation(
                case,
                actual,
                Failure(
                    "ledger-mismatch",
                    difference.format(
                        case_id=case["case_id"],
                        replay=" ".join(command),
                    ),
                    difference.field_path,
                ),
            )
        return Evaluation(case, actual, None)


def _candidate_still_fails(
    candidate: dict[str, Any],
    predicate: Callable[[dict[str, Any]], bool],
) -> bool:
    try:
        validate_case(candidate, source="<shrink-candidate>")
    except ContractError:
        return False
    return predicate(candidate)


def shrink_case(
    raw_case: dict[str, Any],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Deterministically reduce a still-failing valid case."""

    current = copy.deepcopy(raw_case)
    if not _candidate_still_fails(current, predicate):
        raise ValueError("shrink input does not satisfy failure predicate")

    granularity = 2
    while len(current["requests"]) > 1:
        count = len(current["requests"])
        chunk = math.ceil(count / granularity)
        reduced = False
        for begin in range(0, count, chunk):
            candidate = copy.deepcopy(current)
            del candidate["requests"][begin:begin + chunk]
            if _candidate_still_fails(candidate, predicate):
                current = candidate
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= count:
            break
        granularity = min(count, granularity * 2)

    zero_arrivals = copy.deepcopy(current)
    for request in zero_arrivals["requests"]:
        request["arrival_ns"] = 0.0
    if _candidate_still_fails(zero_arrivals, predicate):
        current = zero_arrivals

    for index in range(len(current["requests"])):
        request = current["requests"][index]
        if request["op"] == "drain":
            continue
        for address in (0, request["addr"] // 2):
            if address >= request["addr"]:
                continue
            candidate = copy.deepcopy(current)
            candidate["requests"][index]["addr"] = address
            if _candidate_still_fails(candidate, predicate):
                current = candidate
                break
        request = current["requests"][index]
        for byte_count in (1, 64, request["bytes"] // 2):
            if not 0 < byte_count < request["bytes"]:
                continue
            candidate = copy.deepcopy(current)
            candidate["requests"][index]["bytes"] = byte_count
            if _candidate_still_fails(candidate, predicate):
                current = candidate
                break

    config_sections: list[dict[str, Any]]
    if current["model"] == "hybrid":
        config_sections = [
            current["config"]["hbm"],
            current["config"]["hbf"],
        ]
    else:
        config_sections = [current["config"]]
    topology_keys = (
        "stacks",
        "channels_per_stack",
        "pseudo_channels_per_channel",
        "bank_groups_per_pseudo_channel",
        "banks_per_group",
        "dies_per_channel",
        "planes_per_die",
        "blocks_per_plane",
        "pages_per_block",
        "media_channels",
        "media_read_queues",
        "media_write_queues",
    )
    for section_index, section in enumerate(config_sections):
        for key in topology_keys:
            if key not in section or section[key] <= 1:
                continue
            candidate = copy.deepcopy(current)
            candidate_sections = (
                [
                    candidate["config"]["hbm"],
                    candidate["config"]["hbf"],
                ]
                if candidate["model"] == "hybrid"
                else [candidate["config"]]
            )
            candidate_sections[section_index][key] = 1
            if _candidate_still_fails(candidate, predicate):
                current = candidate

    return current


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _subtract_time_shift(
    ledger: list[dict[str, Any]],
    delta_ns: float,
) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(ledger)
    for record in normalized:
        if record["kind"] == "request":
            record["arrival_ns"] -= delta_ns
        elif record["kind"] == "event":
            record["start_ns"] -= delta_ns
            record["finish_ns"] -= delta_ns
        elif record["kind"] == "state":
            record["time_ns"] -= delta_ns
        elif record["kind"] == "completion":
            record["arrival_ns"] -= delta_ns
            record["start_ns"] -= delta_ns
            record["finish_ns"] -= delta_ns
        elif record["kind"] == "summary":
            counters = record["counters"]
            for key in (
                "finish_ns",
                "user_finish_ns",
                "first_offered_arrival_ns",
                "last_offered_arrival_ns",
            ):
                if key in counters:
                    counters[key] -= delta_ns
            for tier in ("hbm", "hbf", "external"):
                if (
                    tier in counters
                    and counters[tier].get("finish_ns", 0.0) != 0.0
                ):
                    counters[tier]["finish_ns"] -= delta_ns
    return normalized


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for old in sorted(replacements, key=len, reverse=True):
            new = replacements[old]
            result = result.replace(old, new)
        return result
    if isinstance(value, list):
        return [
            _replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def _metamorphic_difference(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    case: dict[str, Any],
) -> Difference | None:
    return first_difference(
        expected,
        actual,
        abs_tolerance=case["comparison"]["time_abs_tolerance_ns"],
        rel_tolerance=case["comparison"]["time_rel_tolerance"],
    )


def run_metamorphic_checks(
    evaluator: Evaluator,
    raw_case: dict[str, Any],
    base: Evaluation,
) -> None:
    if base.failure is not None or base.actual is None:
        raise RuntimeError("metamorphic base case did not pass")
    case = base.normalized_case
    assert case is not None
    seed = raw_case["_seed"]
    clean_case = _strip_internal_metadata(raw_case)
    if case["model"] in {"hbm", "hybrid"}:
        hbm_config = (
            case["config"]
            if case["model"] == "hbm"
            else case["config"]["hbm"]
        )
        delta_ns = (
            hbm_config["data_rate_per_command_clock"]
            / hbm_config["pin_rate_Gbps"]
            * 10_000
        )
    else:
        delta_ns = 10_000.0
    shifted = copy.deepcopy(clean_case)
    for request in shifted["requests"]:
        request["arrival_ns"] += delta_ns
    shifted_result = evaluator.evaluate(shifted)
    if shifted_result.failure is not None or shifted_result.actual is None:
        raise RuntimeError(
            "time-translation transformed case failed exact oracle check:\n"
            + (
                shifted_result.failure.message
                if shifted_result.failure is not None
                else "<missing ledger>"
            )
        )
    difference = _metamorphic_difference(
        base.actual,
        _subtract_time_shift(shifted_result.actual, delta_ns),
        case,
    )
    if difference is not None:
        raise RuntimeError(
            "time-translation property failed:\n"
            + difference.format(
                case_id=case["case_id"],
                replay=(
                    "python3 -B verification/gates/fuzz.py "
                    f"--replay {case['model']}:{seed}"
                ),
            )
        )

    renamed = copy.deepcopy(clean_case)
    old_ids = [request["id"] for request in renamed["requests"]]
    new_ids = [f"renamed-{index}" for index in range(len(old_ids))]
    for request, new_id in zip(
        renamed["requests"], new_ids, strict=True
    ):
        request["id"] = new_id
    renamed_result = evaluator.evaluate(renamed)
    if renamed_result.failure is not None or renamed_result.actual is None:
        raise RuntimeError(
            "ID-renamed case failed exact oracle check:\n"
            + (
                renamed_result.failure.message
                if renamed_result.failure is not None
                else "<missing ledger>"
            )
        )
    restored = _replace_strings(
        renamed_result.actual,
        dict(zip(new_ids, old_ids, strict=True)),
    )
    difference = _metamorphic_difference(
        base.actual,
        restored,
        case,
    )
    if difference is not None:
        raise RuntimeError(
            "request-ID renaming property failed:\n"
            + difference.format(
                case_id=case["case_id"],
                replay=(
                    "python3 -B verification/gates/fuzz.py "
                    f"--replay {case['model']}:{seed}"
                ),
            )
        )


def _strip_internal_metadata(case: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(case)
    result.pop("_seed", None)
    return result


def run_fuzz(
    *,
    probe: Path,
    models: list[str],
    seed_start: int,
    seeds: int,
    max_requests: int,
    artifact_dir: Path,
) -> tuple[int, int]:
    with tempfile.TemporaryDirectory(
        prefix="hbfsim-property-fuzz-"
    ) as directory:
        evaluator = Evaluator(probe, Path(directory))
        first_by_model: dict[str, tuple[dict[str, Any], Evaluation]] = {}
        generated = 0
        for model in models:
            for seed in range(seed_start, seed_start + seeds):
                raw_case = generate_case(
                    model, seed, max_requests=max_requests)
                raw_case["_seed"] = seed
                evaluation = evaluator.evaluate(
                    _strip_internal_metadata(raw_case))
                generated += 1
                if model not in first_by_model:
                    first_by_model[model] = (raw_case, evaluation)
                if evaluation.failure is None:
                    continue
                original_failure = evaluation.failure

                def preserves(candidate: dict[str, Any]) -> bool:
                    observed = evaluator.evaluate(candidate).failure
                    return original_failure.same_signature(observed)

                shrink_input = _strip_internal_metadata(raw_case)
                minimized = shrink_case(shrink_input, preserves)
                prefix = (
                    f"{model}.seed-{seed:016x}")
                original_path = artifact_dir / f"{prefix}.original.json"
                minimized_path = artifact_dir / f"{prefix}.minimized.json"
                _atomic_write_json(original_path, shrink_input)
                _atomic_write_json(minimized_path, minimized)
                raise RuntimeError(
                    f"property fuzz failed: model={model} seed={seed}\n"
                    f"{original_failure.message}\n"
                    f"original={original_path}\n"
                    f"minimized={minimized_path}\n"
                    "replay=python3 -B verification/gates/fuzz.py "
                    f"--probe {probe} --replay {model}:{seed} "
                    f"--artifact-dir {artifact_dir}"
                )
        for model in models:
            raw_case, evaluation = first_by_model[model]
            run_metamorphic_checks(evaluator, raw_case, evaluation)
        return generated, evaluator.executions


def _parse_replay(value: str) -> tuple[str, int]:
    model, separator, seed_text = value.partition(":")
    if separator != ":" or model not in GENERATORS:
        raise argparse.ArgumentTypeError(
            "replay must be MODEL:SEED for hbm, hbf, hybrid, or external")
    try:
        seed = int(seed_text, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "replay seed must be an integer") from error
    if not 0 <= seed <= MASK64:
        raise argparse.ArgumentTypeError(
            "replay seed must be uint64")
    return model, seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(GENERATORS),
        dest="models",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=50)
    parser.add_argument(
        "--replay",
        action="append",
        type=_parse_replay,
        default=[],
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("build/property-fuzz"),
    )
    args = parser.parse_args()
    if not args.probe.is_file():
        parser.error(f"validation probe not found: {args.probe}")
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    if args.seed_start < 0:
        parser.error("--seed-start must be nonnegative")
    if not 1 <= args.max_requests <= 50:
        parser.error("--max-requests must be in [1, 50]")
    try:
        if args.replay:
            total_generated = 0
            total_executions = 0
            for model, seed in args.replay:
                generated, executions = run_fuzz(
                    probe=args.probe,
                    models=[model],
                    seed_start=seed,
                    seeds=1,
                    max_requests=args.max_requests,
                    artifact_dir=args.artifact_dir,
                )
                total_generated += generated
                total_executions += executions
        else:
            models = args.models or sorted(GENERATORS)
            total_generated, total_executions = run_fuzz(
                probe=args.probe,
                models=models,
                seed_start=args.seed_start,
                seeds=args.seeds,
                max_requests=args.max_requests,
                artifact_dir=args.artifact_dir,
            )
    except (ContractError, RuntimeError, ValueError) as error:
        print(f"foundational property fuzz failed: {error}", file=sys.stderr)
        return 1
    print(
        "PASS foundational property fuzz: "
        f"{total_generated} generated case(s), "
        f"{total_executions} production execution(s), "
        "time-translation and ID-renaming metamorphic checks passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
