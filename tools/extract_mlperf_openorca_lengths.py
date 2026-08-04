#!/usr/bin/env python3
"""Extract digest-bound MLPerf OpenOrca request lengths without running inference.

The official artifact is a pandas pickle.  This tool refuses to unpickle any
file until its size, MD5, and SHA-256 match the selected calibration protocol.
It emits only QSL indices and input/reference-output lengths; prompts, token
IDs, and text are never copied into the evidence bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any


PROTOCOL_SCHEMA = {
    "name": "hbfsim.validation.timing-calibration-protocol",
    "version": 1,
}
OFFICIAL_CALIBRATION_ID = "h100-llama2-70b-vidur-interpolation-v1"
OFFICIAL_PROTOCOL_SHA256 = (
    "65243dd780f67aa7ac7b697ce537f8fb123a942756d095036a2dc83dc7a1b6fe"
)
DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parent.parent
    / "validation/calibration/h100-llama2-70b-calibration-protocol.json"
)


class LengthExtractionError(ValueError):
    """The protocol, source dataset, or extracted length table is not exact."""


class _DuplicateJsonKey(ValueError):
    pass


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _load_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except OSError as error:
        raise LengthExtractionError(
            f"cannot read {description} {path}: {error}"
        ) from error
    except UnicodeDecodeError as error:
        raise LengthExtractionError(f"{description} is not UTF-8: {path}") from error
    except _DuplicateJsonKey as error:
        raise LengthExtractionError(
            f"{description} contains duplicate key {error.args[0]!r}"
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise LengthExtractionError(
            f"invalid JSON in {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise LengthExtractionError(f"{description} must be a JSON object")
    return value, payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LengthExtractionError(f"{path} must be a positive integer")
    return value


def _protocol_contract(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise LengthExtractionError("unsupported calibration protocol schema")
    evidence = protocol.get("workload_evidence")
    if not isinstance(evidence, dict):
        raise LengthExtractionError("protocol.workload_evidence must be an object")
    dataset = evidence.get("dataset")
    lengths = evidence.get("lengths_csv")
    if not isinstance(dataset, dict) or not isinstance(lengths, dict):
        raise LengthExtractionError(
            "protocol workload evidence must contain dataset and lengths_csv objects"
        )
    if dataset.get("pickle_load_allowed_only_after_digest_match") is not True:
        raise LengthExtractionError("protocol must keep the pickle digest gate enabled")
    if not isinstance(dataset.get("required_columns"), list) or not dataset[
        "required_columns"
    ]:
        raise LengthExtractionError("protocol dataset.required_columns must be non-empty")
    if lengths.get("filename") != "lengths.csv":
        raise LengthExtractionError("protocol lengths filename must be lengths.csv")
    if lengths.get("columns") != [
        "qsl_idx",
        "input_tokens",
        "reference_output_tokens",
    ]:
        raise LengthExtractionError("unsupported lengths CSV columns")
    _positive_int(dataset.get("bytes"), "protocol.dataset.bytes")
    _positive_int(lengths.get("rows"), "protocol.lengths_csv.rows")
    _positive_int(lengths.get("bytes"), "protocol.lengths_csv.bytes")
    for mapping, key, width in (
        (dataset, "md5", 32),
        (dataset, "sha256", 64),
        (lengths, "sha256", 64),
        (lengths, "pairs_u32le_sha256", 64),
    ):
        value = mapping.get(key)
        if (
            not isinstance(value, str)
            or len(value) != width
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise LengthExtractionError(f"protocol {key} is not lowercase hex")
    return dataset, lengths


def _native_positive_int(value: Any, path: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise LengthExtractionError(f"{path} is not an integer") from error
    if normalized <= 0 or value != normalized:
        raise LengthExtractionError(f"{path} must be a positive exact integer")
    return normalized


def extract_lengths(
    *,
    protocol_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if protocol_path.is_symlink():
        raise LengthExtractionError("calibration protocol must not be a symlink")
    if dataset_path.is_symlink():
        raise LengthExtractionError("dataset must not be a symlink")
    if output_dir.is_symlink():
        raise LengthExtractionError("output directory must not be a symlink")
    protocol_path = protocol_path.resolve()
    dataset_path = dataset_path.resolve()
    output_dir = output_dir.resolve()
    protocol, protocol_bytes = _load_object(protocol_path, "calibration protocol")
    if (
        protocol.get("calibration_id") == OFFICIAL_CALIBRATION_ID
        and hashlib.sha256(protocol_bytes).hexdigest() != OFFICIAL_PROTOCOL_SHA256
    ):
        raise LengthExtractionError("official calibration protocol digest mismatch")
    dataset_contract, lengths_contract = _protocol_contract(protocol)

    if output_dir.exists() or output_dir.is_symlink():
        raise LengthExtractionError(f"output directory already exists: {output_dir}")
    if not dataset_path.is_file() or dataset_path.is_symlink():
        raise LengthExtractionError(
            f"dataset must be a regular, non-symlink file: {dataset_path}"
        )
    actual_size = dataset_path.stat().st_size
    actual_md5 = _md5_file(dataset_path)
    actual_sha256 = _sha256_file(dataset_path)
    if actual_size != dataset_contract["bytes"]:
        raise LengthExtractionError(
            f"dataset byte count mismatch: expected={dataset_contract['bytes']}, "
            f"actual={actual_size}"
        )
    if actual_md5 != dataset_contract["md5"]:
        raise LengthExtractionError("dataset MD5 mismatch; refusing to unpickle")
    if actual_sha256 != dataset_contract["sha256"]:
        raise LengthExtractionError("dataset SHA-256 mismatch; refusing to unpickle")

    try:
        import pandas as pd
    except ImportError as error:
        raise LengthExtractionError(
            "pandas is required only for the digest-gated OpenOrca extraction"
        ) from error
    try:
        frame = pd.read_pickle(dataset_path)
    except Exception as error:
        raise LengthExtractionError(
            f"digest-matched dataset could not be decoded by pandas: {error}"
        ) from error

    expected_columns = dataset_contract["required_columns"]
    if list(frame.columns) != expected_columns:
        raise LengthExtractionError(
            f"dataset columns mismatch: expected={expected_columns!r}, "
            f"actual={list(frame.columns)!r}"
        )
    if len(frame) != lengths_contract["rows"]:
        raise LengthExtractionError(
            f"dataset row count mismatch: expected={lengths_contract['rows']}, "
            f"actual={len(frame)}"
        )

    pairs: list[tuple[int, int]] = []
    for qsl_idx, row in enumerate(frame.itertuples(index=False)):
        input_length = _native_positive_int(
            getattr(row, "tok_input_length"), f"row {qsl_idx}.tok_input_length"
        )
        output_length = _native_positive_int(
            getattr(row, "tok_output_length"), f"row {qsl_idx}.tok_output_length"
        )
        input_tokens = getattr(row, "tok_input")
        output_tokens = getattr(row, "tok_output")
        if not isinstance(input_tokens, list) or len(input_tokens) != input_length:
            raise LengthExtractionError(
                f"row {qsl_idx} input token list does not match its declared length"
            )
        if not isinstance(output_tokens, list) or len(output_tokens) != output_length:
            raise LengthExtractionError(
                f"row {qsl_idx} output token list does not match its declared length"
            )
        pairs.append((input_length, output_length))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        lengths_path = temporary / "lengths.csv"
        with lengths_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(lengths_contract["columns"])
            for qsl_idx, (input_length, output_length) in enumerate(pairs):
                writer.writerow((qsl_idx, input_length, output_length))
            handle.flush()
            os.fsync(handle.fileno())

        pair_digest = hashlib.sha256(
            b"".join(struct.pack("<II", left, right) for left, right in pairs)
        ).hexdigest()
        input_values = [left for left, _ in pairs]
        output_values = [right for _, right in pairs]
        observed_stats = {
            "input_tokens": {
                "min": min(input_values),
                "max": max(input_values),
                "sum": sum(input_values),
            },
            "reference_output_tokens": {
                "min": min(output_values),
                "max": max(output_values),
                "sum": sum(output_values),
            },
        }
        for key, observed in observed_stats.items():
            if observed != lengths_contract[key]:
                raise LengthExtractionError(
                    f"extracted {key} statistics mismatch: "
                    f"expected={lengths_contract[key]!r}, actual={observed!r}"
                )
        lengths_size = lengths_path.stat().st_size
        lengths_sha256 = _sha256_file(lengths_path)
        if lengths_size != lengths_contract["bytes"]:
            raise LengthExtractionError("extracted lengths CSV byte count mismatch")
        if lengths_sha256 != lengths_contract["sha256"]:
            raise LengthExtractionError("extracted lengths CSV SHA-256 mismatch")
        if pair_digest != lengths_contract["pairs_u32le_sha256"]:
            raise LengthExtractionError("extracted length-pair digest mismatch")

        receipt = {
            "schema": protocol["workload_evidence"]["schema"],
            "result": "pass",
            "calibration_id": protocol["calibration_id"],
            "dataset": {
                "filename": dataset_path.name,
                "source_url": dataset_contract["source_url"],
                "bytes": actual_size,
                "md5": actual_md5,
                "sha256": actual_sha256,
                "digest_checked_before_pickle_load": True,
            },
            "lengths": {
                "path": "lengths.csv",
                "columns": lengths_contract["columns"],
                "rows": len(pairs),
                "bytes": lengths_size,
                "sha256": lengths_sha256,
                "pairs_u32le_sha256": pair_digest,
                **observed_stats,
            },
            "scope": {
                "contains_prompts": False,
                "contains_token_ids": False,
                "contains_reference_text": False,
                "establishes_timing_calibration": False,
            },
        }
        receipt_path = temporary / "workload-evidence.json"
        with receipt_path.open("w", encoding="utf-8", newline="") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_dir)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        receipt = extract_lengths(
            protocol_path=args.protocol,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
        )
    except (LengthExtractionError, OSError) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"extracted {receipt['lengths']['rows']} OpenOrca length pairs: "
        f"sha256={receipt['lengths']['sha256']}"
    )
    print(f"evidence={args.output_dir.resolve() / 'workload-evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
