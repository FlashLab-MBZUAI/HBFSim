#!/usr/bin/env python3
"""Certify the three predeclared Frontier 70B temporal strata.

The five-cell runner intentionally emits one partial-trace receipt for each
of q=0, q=0.5, and q=1.  This verifier accepts exactly those three receipts,
revalidates their complete transitive artifact graph and per-cell baseline
receipts, and publishes a 60-row *stratified* table.  It never averages across
the three batches and never promotes the partial memory traces to a full
Frontier replay or a paper-performance result.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from run_frontier_70b_two_axis_slice import (
    EXPECTED_BASELINES,
    EXPECTED_CELLS,
    INPUT_SCHEMA,
    RECEIPT_SCHEMA as SLICE_RECEIPT_SCHEMA,
    _expected_batch_row,
    _result_rows,
)
from run_frontier_memory_baselines import SCHEMA as BASELINE_SCHEMA


CERTIFICATE_SCHEMA = {
    "name": "hbfsim.frontier_70b_two_axis_evidence",
    "version": 1,
}
EXPECTED_QUANTILES = ("0", "0.5", "1")
METRIC_FIELDS = (
    "memory_system_logical_throughput_GBps",
    "makespan_ns",
    "offered_average_ns",
    "offered_p50_ns",
    "offered_p95_ns",
    "offered_max_ns",
    "source_average_ns",
    "source_p50_ns",
    "source_p95_ns",
    "source_max_ns",
    "service_average_ns",
    "service_p50_ns",
    "service_p95_ns",
    "service_max_ns",
    "backing_admission_max_wait_ns",
    "exposed_prefetch_ns",
    "hidden_prefetch_ns",
    "backing_read_bytes",
    "backing_write_bytes",
)


class EvidenceError(ValueError):
    """The three-stratum evidence set is incomplete or inconsistent."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise EvidenceError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{description} must be a JSON object")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(path: Path, description: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise EvidenceError(f"{description} is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _resolve_artifact(
    record: Any,
    *,
    owner: Path,
    description: str,
) -> tuple[Path, dict[str, Any]]:
    artifact = _mapping(record, description)
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceError(f"{description}.path must be non-empty")
    path = Path(raw_path)
    if not path.is_absolute():
        path = owner.parent / path
    actual = _snapshot(path, description)
    if (
        artifact.get("bytes") != actual["bytes"]
        or artifact.get("sha256") != actual["sha256"]
    ):
        raise EvidenceError(f"{description} digest drifted")
    return path.resolve(), actual


def _verify_snapshot(record: Any, description: str) -> dict[str, Any]:
    snapshot = _mapping(record, description)
    raw_path = snapshot.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise EvidenceError(f"{description}.path must be non-empty")
    actual = _snapshot(Path(raw_path), description)
    if actual != snapshot:
        raise EvidenceError(f"{description} changed after the slice")
    return actual


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise EvidenceError("cannot write an empty evidence table")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_csv(path: Path, description: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as error:
        raise EvidenceError(f"cannot read {description} {path}: {error}") from error
    if not fieldnames or not rows:
        raise EvidenceError(f"{description} must contain a header and rows")
    return fieldnames, rows


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def _validate_waf(row: dict[str, str], description: str) -> None:
    if row["baseline"] != "hbm_hbf":
        for name in (
            "logical_write_bytes",
            "physical_write_bytes",
            "waf",
            "physical_write_over_hbf_capacity",
        ):
            if row[name] != "":
                raise EvidenceError(f"{description} exposes HBF WAF on {row['baseline']}")
        return
    try:
        logical = int(row["logical_write_bytes"])
        physical = int(row["physical_write_bytes"])
    except ValueError as error:
        raise EvidenceError(f"{description} has non-integer HBF write bytes") from error
    if logical == 0:
        if physical != 0 or row["waf"] != "":
            raise EvidenceError(f"{description} must encode zero-write WAF as null")
        return
    if physical < logical or row["waf"] == "":
        raise EvidenceError(f"{description} has invalid physical/logical writes")
    try:
        waf = Decimal(row["waf"])
    except Exception as error:
        raise EvidenceError(f"{description} has invalid WAF") from error
    if abs(waf - Decimal(physical) / Decimal(logical)) > Decimal("1e-12"):
        raise EvidenceError(f"{description} WAF is not physical/logical")


def _validate_slice(
    *,
    receipt_path: Path,
    expected_quantile: str,
) -> dict[str, Any]:
    receipt_path = receipt_path.resolve()
    receipt = _load_object(receipt_path, f"q={expected_quantile} slice receipt")
    if receipt.get("schema") != SLICE_RECEIPT_SCHEMA or receipt.get("result") != "pass":
        raise EvidenceError(f"q={expected_quantile} is not a passing slice receipt")
    if str(receipt.get("quantile")) != expected_quantile:
        raise EvidenceError(f"q={expected_quantile} receipt quantile drifted")
    contract = _mapping(receipt.get("contract"), "slice.contract")
    if contract != {
        "cell_count": 5,
        "baseline_runs": 20,
        "mix_ratio_same_logical_trace": True,
        "mix_ratio_same_address_population": True,
        "mix_ratio_capacity_pressure_is_observed": True,
        "pressure_axis_fixed_topology": "4h4f",
        "hbm_hbf_semantics": "HBM+HBF_hybrid_layer_streaming",
        "pure_all_hbf_included": False,
        "temporal_stratum_is_partial_trace": True,
        "aggregate_across_quantiles": False,
    }:
        raise EvidenceError(f"q={expected_quantile} slice contract drifted")
    eligibility = _mapping(receipt.get("eligibility"), "slice.eligibility")
    if (
        eligibility.get("five_cell_contract_complete") is not True
        or eligibility.get("twenty_baseline_runs_complete") is not True
        or eligibility.get("mix_ratio_byte_identity_verified") is not True
        or eligibility.get("memory_trace_is_temporal_stratum") is not True
        or eligibility.get("paper_result_eligible") is not False
    ):
        raise EvidenceError(f"q={expected_quantile} slice eligibility drifted")

    source = _mapping(receipt.get("source"), "slice.source")
    inputs_path, inputs_snapshot = _resolve_artifact(
        source.get("inputs"),
        owner=receipt_path,
        description=f"q={expected_quantile} inputs",
    )
    inputs = _load_object(inputs_path, f"q={expected_quantile} inputs")
    if inputs.get("schema") != INPUT_SCHEMA or str(inputs.get("quantile")) != expected_quantile:
        raise EvidenceError(f"q={expected_quantile} input receipt drifted")
    for name in ("runner", "study", "scenario_compare"):
        _verify_snapshot(inputs.get(name), f"q={expected_quantile} {name}")
    for name, snapshot in _mapping(
        inputs.get("hardware_configs"), "inputs.hardware_configs"
    ).items():
        _verify_snapshot(snapshot, f"q={expected_quantile} hardware {name}")
    populations = _mapping(inputs.get("populations"), "inputs.populations")
    if set(populations) != {"p0_75", "p1_0", "p1_25"}:
        raise EvidenceError(f"q={expected_quantile} population census drifted")
    for population_name, raw_population in populations.items():
        population = _mapping(raw_population, f"population {population_name}")
        selection = _mapping(population.get("selection"), "population.selection")
        total = selection.get("total_frontier_batches")
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise EvidenceError(f"q={expected_quantile} has invalid Frontier batch count")
        expected_row = _expected_batch_row(total, Decimal(expected_quantile))
        if (
            str(selection.get("quantile")) != expected_quantile
            or selection.get("batch_row") != expected_row
            or not isinstance(selection.get("batch_ids"), list)
            or len(selection["batch_ids"]) != 1
        ):
            raise EvidenceError(f"q={expected_quantile} selection is not mechanical")
        for name in ("manifest", "object_map", "trace"):
            _verify_snapshot(population.get(name), f"q={expected_quantile} {population_name} {name}")
        for name, snapshot in _mapping(
            population.get("transitive_artifacts"),
            "population.transitive_artifacts",
        ).items():
            _verify_snapshot(snapshot, f"q={expected_quantile} {population_name} {name}")

    cells = _mapping(receipt.get("cells"), "slice.cells")
    if set(cells) != {cell["cell_id"] for cell in EXPECTED_CELLS}:
        raise EvidenceError(f"q={expected_quantile} cell census drifted")
    baseline_receipts: dict[str, dict[str, Any]] = {}
    for cell in EXPECTED_CELLS:
        cell_id = cell["cell_id"]
        cell_record = _mapping(cells[cell_id], f"slice.cells.{cell_id}")
        if any(cell_record.get(name) != cell[name] for name in ("axis", "topology", "population_source")):
            raise EvidenceError(f"q={expected_quantile} {cell_id} identity drifted")
        baseline_path, _ = _resolve_artifact(
            cell_record.get("baseline_receipt"),
            owner=receipt_path,
            description=f"q={expected_quantile} {cell_id} baseline receipt",
        )
        baseline_receipt = _load_object(baseline_path, f"{cell_id} baseline receipt")
        if (
            baseline_receipt.get("schema") != BASELINE_SCHEMA
            or baseline_receipt.get("result") != "pass"
            or set(_mapping(baseline_receipt.get("baselines"), "baselines"))
            != set(EXPECTED_BASELINES)
        ):
            raise EvidenceError(f"q={expected_quantile} {cell_id} baseline set is incomplete")
        for baseline in EXPECTED_BASELINES:
            metrics = _mapping(
                baseline_receipt["baselines"][baseline].get("memory_service_metrics"),
                f"{cell_id}.{baseline}.memory_service_metrics",
            )
            if any(name not in metrics for name in METRIC_FIELDS):
                raise EvidenceError(f"q={expected_quantile} {cell_id}.{baseline} lost metrics")
        baseline_receipts[cell_id] = baseline_receipt

    table_path, table_snapshot = _resolve_artifact(
        receipt.get("table"),
        owner=receipt_path,
        description=f"q={expected_quantile} result table",
    )
    fieldnames, csv_rows = _read_csv(table_path, f"q={expected_quantile} result table")
    expected_rows: list[dict[str, Any]] = []
    for cell in EXPECTED_CELLS:
        population_name = cell["population_source"]
        expected_rows.extend(
            _result_rows(
                cell=cell,
                population_name=population_name,
                population=populations[population_name],
                receipt=baseline_receipts[cell["cell_id"]],
            )
        )
    if fieldnames != list(expected_rows[0]) or len(csv_rows) != len(expected_rows):
        raise EvidenceError(f"q={expected_quantile} table schema/census drifted")
    normalized_rows: list[dict[str, Any]] = []
    for index, (actual, expected) in enumerate(zip(csv_rows, expected_rows, strict=True)):
        expected_csv = {name: _csv_value(value) for name, value in expected.items()}
        if actual != expected_csv:
            raise EvidenceError(f"q={expected_quantile} table row {index} drifted from receipts")
        _validate_waf(actual, f"q={expected_quantile} table row {index}")
        normalized_rows.append(expected)
    return {
        "receipt": _snapshot(receipt_path, f"q={expected_quantile} receipt"),
        "inputs": inputs_snapshot,
        "inputs_payload": inputs,
        "table": table_snapshot,
        "rows": normalized_rows,
    }


def verify_evidence(
    *,
    receipt_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    if set(receipt_paths) != set(EXPECTED_QUANTILES):
        raise EvidenceError("exactly q=0, q=0.5, and q=1 receipts are required")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise EvidenceError("evidence output directory must be empty")
    slices = {
        quantile: _validate_slice(
            receipt_path=receipt_paths[quantile],
            expected_quantile=quantile,
        )
        for quantile in EXPECTED_QUANTILES
    }
    identity_fields = (
        "runner",
        "study",
        "scenario_compare",
        "scenario_compare_version",
        "hbfsim_revision",
        "hardware_configs",
        "request_identity",
        "credit_limit",
        "timeout_seconds",
    )
    reference = slices[EXPECTED_QUANTILES[0]]["inputs_payload"]
    for quantile in EXPECTED_QUANTILES[1:]:
        candidate = slices[quantile]["inputs_payload"]
        for name in identity_fields:
            if candidate.get(name) != reference.get(name):
                raise EvidenceError(f"q={quantile} does not share {name}")
        for population_name in ("p0_75", "p1_0", "p1_25"):
            left = reference["populations"][population_name]["selection"]
            right = candidate["populations"][population_name]["selection"]
            if left["total_frontier_batches"] != right["total_frontier_batches"]:
                raise EvidenceError(f"{population_name} Frontier batch census drifted")

    combined_rows: list[dict[str, Any]] = []
    for quantile in EXPECTED_QUANTILES:
        combined_rows.extend(slices[quantile]["rows"])
    if len(combined_rows) != 60:
        raise EvidenceError("three strata must contain exactly 60 baseline rows")
    table_path = output_dir / "stratified-results.csv"
    _write_csv_atomic(table_path, combined_rows)
    certificate_path = output_dir / "evidence.receipt.json"
    payload = {
        "schema": CERTIFICATE_SCHEMA,
        "result": "pass",
        "source": {
            quantile: {
                "slice_receipt": slices[quantile]["receipt"],
                "slice_inputs": slices[quantile]["inputs"],
                "slice_table": slices[quantile]["table"],
            }
            for quantile in EXPECTED_QUANTILES
        },
        "contract": {
            "quantiles": list(EXPECTED_QUANTILES),
            "selection": "floor(q * (total_frontier_batches - 1))",
            "cells_per_quantile": 5,
            "baselines_per_cell": 4,
            "baseline_rows": 60,
            "aggregation": "concatenation_only_no_cross_quantile_statistics",
            "same_hbfsim_revision": True,
            "same_scenario_compare_binary": True,
            "same_request_stream": True,
            "hbm_hbf_semantics": "HBM+HBF_hybrid_layer_streaming",
            "pure_all_hbf_included": False,
            "canonical_waf": "physical_write_bytes/logical_write_bytes",
            "zero_logical_write_waf": None,
        },
        "table": _artifact(table_path, output_dir),
        "eligibility": {
            "three_predeclared_strata_complete": True,
            "sixty_baseline_rows_complete": True,
            "full_frontier_replays_verified_upstream": True,
            "memory_traces_are_three_single_batch_strata": True,
            "cross_quantile_average_permitted": False,
            "claim_scope": "memory_system_service_only",
            "ttft_tpot_slo_claims_eligible": False,
            "time_based_application_throughput_claims_eligible": False,
            "external_absolute_performance_claims_eligible": False,
            "paper_result_eligible": False,
        },
    }
    _write_json_atomic(certificate_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q0-receipt", type=Path, required=True)
    parser.add_argument("--q0-5-receipt", type=Path, required=True)
    parser.add_argument("--q1-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        payload = verify_evidence(
            receipt_paths={
                "0": arguments.q0_receipt,
                "0.5": arguments.q0_5_receipt,
                "1": arguments.q1_receipt,
            },
            output_dir=arguments.output_dir,
        )
    except EvidenceError as error:
        raise SystemExit(f"error: {error}") from error
    print(
        "certified "
        f"{payload['contract']['baseline_rows']} stratified baseline rows; "
        "paper_result_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
