#!/usr/bin/env python3
"""Fail-closed diagnostic gate for HBM/HBF placement workloads.

The gate distinguishes actual occupied pages from address span and evaluates
reuse in LRU stack-distance space. Each experiment must state its intended
shape; a scan negative control and a reuse-positive case therefore use
different contracts instead of sharing vague global heuristics.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
import sys
from pathlib import Path
from typing import Any, Literal, NoReturn

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from workloads.trace_analysis import (
    TraceInputError,
    analyze_trace,
)


SCHEMA = {"name": "hbfsim.workload_quality", "version": 1}


@dataclass(frozen=True)
class QualityContract:
    min_operations: int = 1_000
    min_unique_pages: int = 64
    min_reads: int = 0
    min_writes: int = 0
    min_page_touches_per_unique_page: float = 1.0
    max_address_span_to_occupied_page_ratio: float | None = 64.0
    reuse_expectation: Literal["any", "none", "present"] = "any"
    min_reuse_page_touches: int = 0
    working_set_relation: Literal["any", "fits-hbm", "exceeds-hbm"] = "any"
    min_reuse_within_hbm_ratio: float | None = None

    def validate(self) -> None:
        for name in (
            "min_operations",
            "min_unique_pages",
            "min_reads",
            "min_writes",
            "min_reuse_page_touches",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.min_page_touches_per_unique_page < 1.0:
            raise ValueError(
                "min_page_touches_per_unique_page must be at least one")
        if (
            self.max_address_span_to_occupied_page_ratio is not None
            and self.max_address_span_to_occupied_page_ratio < 1.0
        ):
            raise ValueError(
                "max address-span inflation must be at least one")
        if self.reuse_expectation not in {"any", "none", "present"}:
            raise ValueError("unsupported reuse expectation")
        if self.working_set_relation not in {
            "any", "fits-hbm", "exceeds-hbm"
        }:
            raise ValueError("unsupported working-set relation")
        if (
            self.min_reuse_within_hbm_ratio is not None
            and not 0.0 <= self.min_reuse_within_hbm_ratio <= 1.0
        ):
            raise ValueError(
                "minimum HBM-fit reuse ratio must be in [0, 1]")


def assess(
    locality: dict[str, Any],
    contract: QualityContract,
) -> dict[str, Any]:
    contract.validate()
    overall = locality["overall"]
    temporal = locality["temporal_page_locality"]
    hbm_pages = temporal["hbm_capacity_pages"]
    checks: list[dict[str, Any]] = []

    def check(
        name: str,
        actual: Any,
        requirement: str,
        passed: bool,
        diagnostic: str,
    ) -> None:
        checks.append({
            "name": name,
            "actual": actual,
            "requirement": requirement,
            "passed": passed,
            "diagnostic": diagnostic,
        })

    check(
        "operation_count",
        overall["ops"],
        f">={contract.min_operations}",
        overall["ops"] >= contract.min_operations,
        "Too few operations cannot amortize warmup or expose steady state.",
    )
    check(
        "unique_occupied_pages",
        overall["unique_occupied_pages"],
        f">={contract.min_unique_pages}",
        overall["unique_occupied_pages"] >= contract.min_unique_pages,
        "A tiny working set is not diagnostic of tier-capacity behavior.",
    )
    check(
        "read_count",
        overall["reads"],
        f">={contract.min_reads}",
        overall["reads"] >= contract.min_reads,
        "The trace does not exercise the required read path.",
    )
    check(
        "write_count",
        overall["writes"],
        f">={contract.min_writes}",
        overall["writes"] >= contract.min_writes,
        "The trace does not exercise dirty placement/writeback.",
    )
    touches_per_page = overall["page_touches_per_unique_page"]
    check(
        "page_touches_per_unique_page",
        touches_per_page,
        f">={contract.min_page_touches_per_unique_page}",
        touches_per_page is not None
        and touches_per_page >= contract.min_page_touches_per_unique_page,
        "The run is too short relative to its working set.",
    )
    span_ratio = overall["address_span_to_occupied_page_ratio"]
    maximum_span = contract.max_address_span_to_occupied_page_ratio
    check(
        "address_span_to_occupied_page_ratio",
        span_ratio,
        "unconstrained" if maximum_span is None else f"<={maximum_span}",
        maximum_span is None
        or (span_ratio is not None and span_ratio <= maximum_span),
        "Sparse address placement may be a topology probe, but it must be "
        "explicitly allowed rather than mistaken for working-set size.",
    )

    reuse_touches = temporal["reuse_page_touches"]
    if contract.reuse_expectation == "none":
        reuse_passed = reuse_touches == 0
        reuse_requirement = "==0"
    elif contract.reuse_expectation == "present":
        minimum_reuse = max(1, contract.min_reuse_page_touches)
        reuse_passed = reuse_touches >= minimum_reuse
        reuse_requirement = f">={minimum_reuse}"
    else:
        reuse_passed = reuse_touches >= contract.min_reuse_page_touches
        reuse_requirement = f">={contract.min_reuse_page_touches}"
    check(
        "reuse_page_touches",
        reuse_touches,
        reuse_requirement,
        reuse_passed,
        "The observed reuse signal does not match this case's declared role.",
    )

    working_set = overall["unique_occupied_pages"]
    if contract.working_set_relation == "any":
        relation_passed = hbm_pages is not None
        relation_requirement = "HBM capacity supplied"
    elif contract.working_set_relation == "fits-hbm":
        relation_passed = hbm_pages is not None and working_set <= hbm_pages
        relation_requirement = "working_set_pages<=hbm_pages"
    else:
        relation_passed = hbm_pages is not None and working_set > hbm_pages
        relation_requirement = "working_set_pages>hbm_pages"
    check(
        "working_set_relation",
        {
            "working_set_pages": working_set,
            "hbm_pages": hbm_pages,
        },
        relation_requirement,
        relation_passed,
        "The trace does not exercise the intended capacity regime.",
    )

    minimum_fit = contract.min_reuse_within_hbm_ratio
    fit_ratio = temporal["reuse_within_hbm_ratio"]
    check(
        "reuse_within_hbm_ratio",
        fit_ratio,
        "unconstrained" if minimum_fit is None else f">={minimum_fit}",
        minimum_fit is None
        or (fit_ratio is not None and fit_ratio >= minimum_fit),
        "Reuse exists, but too little of it fits in the configured HBM tier.",
    )

    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    return {
        "schema": SCHEMA,
        "status": status,
        "locality_schema": locality["schema"],
        "trace_sha256": locality["trace_sha256"],
        "contract": asdict(contract),
        "measurements": {
            "operations": overall["ops"],
            "reads": overall["reads"],
            "writes": overall["writes"],
            "unique_occupied_pages": working_set,
            "address_span_pages": overall["address_span_pages"],
            "address_span_to_occupied_page_ratio": span_ratio,
            "page_touches": overall["page_touches"],
            "page_touches_per_unique_page": touches_per_page,
            "reuse_page_touches": reuse_touches,
            "reuse_touch_ratio": temporal["reuse_touch_ratio"],
            "reuse_distance_pages_p50":
                temporal["reuse_distance_pages_p50"],
            "reuse_distance_pages_p95":
                temporal["reuse_distance_pages_p95"],
            "reuse_distance_pages_p99":
                temporal["reuse_distance_pages_p99"],
            "hbm_capacity_pages": hbm_pages,
            "reuse_within_hbm_ratio": fit_ratio,
        },
        "checks": checks,
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"not an integer: {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"not an integer: {value!r}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _ratio(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"not a ratio: {value!r}") from error
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("ratio must be in [0, 1]")
    return parsed


def _at_least_one(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"not a number: {value!r}") from error
    if parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fail(message: str) -> NoReturn:
    raise SystemExit(f"error: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--line-size", type=_positive_int, default=64)
    parser.add_argument("--page-size", type=_positive_int, default=4096)
    parser.add_argument("--hbm-capacity-bytes", type=_positive_int, required=True)
    parser.add_argument("--min-operations", type=_nonnegative_int, default=1000)
    parser.add_argument("--min-unique-pages", type=_nonnegative_int, default=64)
    parser.add_argument("--min-reads", type=_nonnegative_int, default=0)
    parser.add_argument("--min-writes", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--min-page-touches-per-unique-page",
        type=_at_least_one,
        default=1.0,
    )
    parser.add_argument(
        "--max-address-span-to-occupied-page-ratio",
        type=_at_least_one,
        default=64.0,
    )
    parser.add_argument(
        "--allow-arbitrary-address-span",
        action="store_true",
    )
    parser.add_argument(
        "--reuse-expectation",
        choices=("any", "none", "present"),
        default="any",
    )
    parser.add_argument(
        "--min-reuse-page-touches",
        type=_nonnegative_int,
        default=0,
    )
    parser.add_argument(
        "--working-set-relation",
        choices=("any", "fits-hbm", "exceeds-hbm"),
        default="any",
    )
    parser.add_argument("--min-reuse-within-hbm-ratio", type=_ratio)
    parser.add_argument(
        "--fail-on-reject",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    try:
        locality = analyze_trace(
            args.trace,
            line_size=args.line_size,
            page_size=args.page_size,
            hbm_capacity_bytes=args.hbm_capacity_bytes,
        )
        result = assess(
            locality,
            QualityContract(
                min_operations=args.min_operations,
                min_unique_pages=args.min_unique_pages,
                min_reads=args.min_reads,
                min_writes=args.min_writes,
                min_page_touches_per_unique_page=
                    args.min_page_touches_per_unique_page,
                max_address_span_to_occupied_page_ratio=(
                    None if args.allow_arbitrary_address_span
                    else args.max_address_span_to_occupied_page_ratio),
                reuse_expectation=args.reuse_expectation,
                min_reuse_page_touches=args.min_reuse_page_touches,
                working_set_relation=args.working_set_relation,
                min_reuse_within_hbm_ratio=
                    args.min_reuse_within_hbm_ratio,
            ),
        )
    except (TraceInputError, ValueError, KeyError) as error:
        _fail(str(error))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        _write_atomic(args.output, result)
    return 2 if args.fail_on_reject and result["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
