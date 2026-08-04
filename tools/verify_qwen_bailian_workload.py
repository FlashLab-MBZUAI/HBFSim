#!/usr/bin/env python3
"""Independently verify a prepared Qwen-Bailian Frontier workload suite."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TextIO

from prepare_qwen_bailian_workload import (
    ADAPTER_SCHEMA_NAME,
    ADAPTER_SCHEMA_VERSION,
    OUTPUT_COLUMNS,
    REQUIRED_WINDOWS,
    SELECTOR_ALGORITHMS,
    SUITE_MANIFEST_SCHEMA_NAME,
    SUITE_MANIFEST_SCHEMA_VERSION,
    _decimal_text,
    _load_suite_config,
)


VERIFICATION_SCHEMA_NAME = "hbfsim.qwen_bailian_frontier_suite_verification"
VERIFICATION_SCHEMA_VERSION = 1


class VerificationError(ValueError):
    """A prepared workload suite does not match its pinned source and config."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_float=Decimal,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
            object_pairs_hook=_unique_object,
        )
    except OSError as error:
        raise VerificationError(
            f"cannot read {description} {path}: {error}"
        ) from error
    except (json.JSONDecodeError, ValueError) as error:
        raise VerificationError(
            f"invalid JSON in {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise VerificationError(f"{description} must be a JSON object: {path}")
    return value


def _expect(actual: Any, expected: Any, path: str) -> None:
    if actual != expected:
        raise VerificationError(
            f"{path} mismatch: expected={expected!r}, actual={actual!r}"
        )


def _artifact(
    root: Path, metadata: dict[str, Any], description: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(metadata, dict):
        raise VerificationError(f"{description} metadata must be an object")
    if set(metadata) != {"path", "sha256", "bytes"}:
        raise VerificationError(
            f"{description} metadata must contain only path, sha256, and bytes"
        )
    relative = metadata.get("path")
    if not isinstance(relative, str) or not relative:
        raise VerificationError(f"{description}.path must be non-empty")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise VerificationError(f"{description}.path must be a safe relative path")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise VerificationError(f"{description}.path escapes suite directory") from error
    if not path.is_file():
        raise VerificationError(f"missing {description}: {path}")
    actual = {
        "path": relative,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    _expect(metadata, actual, description)
    return path, actual


def _read_source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots: dict[int, int] = {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise VerificationError(f"cannot read source trace {path}: {error}") from error
    with handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                row = json.loads(
                    line,
                    parse_float=Decimal,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON number {value}")
                    ),
                    object_pairs_hook=_unique_object,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise VerificationError(
                    f"invalid source JSON at line {line_no}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise VerificationError(
                    f"source line {line_no} must be a JSON object"
                )
            chat_id = row.get("chat_id")
            parent = row.get("parent_chat_id")
            if parent == -1:
                session_id = chat_id
            elif parent in roots:
                session_id = roots[parent]
            else:
                raise VerificationError(
                    f"source line {line_no} has unresolved parent_chat_id={parent!r}"
                )
            roots[chat_id] = session_id
            rows.append({**row, "session_id": session_id})
    return rows


def _render_csv(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=OUTPUT_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    first_timestamp = Decimal(rows[0]["timestamp"])
    for row in rows:
        timestamp = Decimal(row["timestamp"])
        writer.writerow(
            {
                "arrived_at": _decimal_text(timestamp - first_timestamp),
                "num_prefill_tokens": row["input_length"],
                "num_decode_tokens": row["output_length"],
                "session_id": row["session_id"],
                "block_hash_ids": "|".join(map(str, row["hash_ids"])),
                "source_chat_id": row["chat_id"],
                "parent_chat_id": row["parent_chat_id"],
                "turn": row["turn"],
                "request_type": row["type"],
            }
        )
    return buffer.getvalue().encode("utf-8")


def _window_statistics(
    rows: list[dict[str, Any]], start: int, request_count: int
) -> dict[str, Any]:
    window = rows[start : start + request_count]
    first = Decimal(window[0]["timestamp"])
    last = Decimal(window[-1]["timestamp"])
    prefill = sum(int(row["input_length"]) for row in window)
    decode = sum(int(row["output_length"]) for row in window)
    hashes = [int(value) for row in window for value in row["hash_ids"]]
    request_types: dict[str, int] = {}
    for row in window:
        request_type = str(row["type"])
        request_types[request_type] = request_types.get(request_type, 0) + 1
    return {
        "start_index": start,
        "end_index_exclusive": start + request_count,
        "records": request_count,
        "source_start_timestamp_s": _decimal_text(first),
        "source_end_timestamp_s": _decimal_text(last),
        "duration_s": _decimal_text(last - first),
        "prefill_tokens": prefill,
        "decode_tokens": decode,
        "total_tokens": prefill + decode,
        "max_prefill_tokens": max(int(row["input_length"]) for row in window),
        "max_decode_tokens": max(int(row["output_length"]) for row in window),
        "max_total_tokens": max(
            int(row["input_length"]) + int(row["output_length"])
            for row in window
        ),
        "hash_identity_occurrences": len(hashes),
        "unique_hash_identities": len(set(hashes)),
        "repeated_hash_identity_occurrences": len(hashes) - len(set(hashes)),
        "request_types": dict(sorted(request_types.items())),
        "hash_repetition_is_cache_hit_rate": False,
    }


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle: TextIO
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, path)
    except Exception:
        if not handle.closed:
            handle.close()
        temporary.unlink(missing_ok=True)
        raise


def verify_suite(
    *,
    source: Path,
    suite_config: Path,
    suite_dir: Path,
    output: Path,
) -> dict[str, Any]:
    source = source.resolve()
    suite_config = suite_config.resolve()
    suite_dir = suite_dir.resolve()
    output = output.resolve()
    if not source.is_file():
        raise VerificationError(f"source trace does not exist: {source}")
    if not suite_config.is_file():
        raise VerificationError(f"suite config does not exist: {suite_config}")
    if not suite_dir.is_dir():
        raise VerificationError(f"suite directory does not exist: {suite_dir}")
    if output in {source, suite_config, suite_dir / "suite-manifest.json"}:
        raise VerificationError("verification output must not overwrite an input")

    config, config_bytes = _load_suite_config(suite_config)
    config_identity = {
        "suite_id": config["suite_id"],
        "sha256": _sha256_bytes(config_bytes),
        "bytes": len(config_bytes),
    }
    config_source = config["source"]
    source_identity = {
        key: config_source[key]
        for key in (
            "repository",
            "revision",
            "license",
            "artifact",
            "sha256",
            "bytes",
            "records",
            "hash_block_tokens",
            "statistics",
        )
    }
    _expect(_sha256_file(source), config_source["sha256"], "source sha256")
    _expect(source.stat().st_size, config_source["bytes"], "source bytes")
    source_rows = _read_source_rows(source)
    _expect(len(source_rows), config_source["records"], "source records")

    suite_manifest_path = suite_dir / "suite-manifest.json"
    suite = _load_object(suite_manifest_path, "suite manifest")
    _expect(
        suite.get("schema"),
        {
            "name": SUITE_MANIFEST_SCHEMA_NAME,
            "version": SUITE_MANIFEST_SCHEMA_VERSION,
        },
        "suite schema",
    )
    _expect(suite.get("suite_id"), config["suite_id"], "suite_id")
    _expect(suite.get("suite_config"), config_identity, "suite_config")
    _expect(suite.get("source"), source_identity, "suite source")
    _expect(
        suite.get("frontier_workload"),
        config["frontier_workload"],
        "suite frontier_workload",
    )
    expected_policy = {
        "request_count": config["window_policy"]["request_count"],
        "required_windows": list(REQUIRED_WINDOWS),
        "selectors": {
            window_id: SELECTOR_ALGORITHMS[window_id]
            for window_id in REQUIRED_WINDOWS
        },
    }
    _expect(suite.get("window_policy"), expected_policy, "suite window_policy")
    windows = suite.get("windows")
    if not isinstance(windows, dict) or set(windows) != set(REQUIRED_WINDOWS):
        raise VerificationError(
            f"suite windows must contain exactly {list(REQUIRED_WINDOWS)}"
        )

    expected_files = {"suite-manifest.json"}
    verified_windows: dict[str, Any] = {}
    request_count = int(config["window_policy"]["request_count"])
    for window_id in REQUIRED_WINDOWS:
        entry = windows[window_id]
        if not isinstance(entry, dict) or set(entry) != {
            "request_csv",
            "request_manifest",
            "selection",
            "statistics",
        }:
            raise VerificationError(f"suite windows.{window_id} shape mismatch")
        csv_path, csv_artifact = _artifact(
            suite_dir, entry["request_csv"], f"{window_id} request_csv"
        )
        manifest_path, manifest_artifact = _artifact(
            suite_dir,
            entry["request_manifest"],
            f"{window_id} request_manifest",
        )
        expected_files.update({csv_path.name, manifest_path.name})

        expected = config["window_policy"]["selectors"][window_id]["expected"]
        expected_selection = {
            "window_id": window_id,
            "algorithm": SELECTOR_ALGORITHMS[window_id],
            **expected,
            "records": request_count,
        }
        _expect(entry["selection"], expected_selection, f"{window_id} selection")
        start = int(expected["start_index"])
        if start + request_count > len(source_rows):
            raise VerificationError(f"{window_id} selection exceeds source records")
        source_window = source_rows[start : start + request_count]
        expected_csv = _render_csv(source_window)
        actual_csv = csv_path.read_bytes()
        if actual_csv != expected_csv:
            raise VerificationError(
                f"{window_id} request CSV is not the exact configured source window"
            )
        statistics = _window_statistics(source_rows, start, request_count)
        _expect(entry["statistics"], statistics, f"{window_id} statistics")

        manifest = _load_object(manifest_path, f"{window_id} adapter manifest")
        _expect(
            manifest.get("schema"),
            {"name": ADAPTER_SCHEMA_NAME, "version": ADAPTER_SCHEMA_VERSION},
            f"{window_id} adapter schema",
        )
        _expect(
            manifest.get("suite_config"),
            config_identity,
            f"{window_id} suite_config",
        )
        _expect(manifest.get("source"), source_identity, f"{window_id} source")
        _expect(
            manifest.get("frontier_workload"),
            config["frontier_workload"],
            f"{window_id} frontier_workload",
        )
        _expect(
            manifest.get("selection"),
            expected_selection,
            f"{window_id} manifest selection",
        )
        _expect(
            manifest.get("statistics"),
            statistics,
            f"{window_id} manifest statistics",
        )
        expected_output = {
            "format": "frontier_trace_replay_csv",
            "path": csv_path.name,
            "sha256": _sha256_bytes(expected_csv),
            "bytes": len(expected_csv),
            "columns": list(OUTPUT_COLUMNS),
        }
        _expect(
            manifest.get("output"),
            expected_output,
            f"{window_id} manifest output",
        )
        verified_windows[window_id] = {
            "request_csv": csv_artifact,
            "request_manifest": manifest_artifact,
            "selection": expected_selection,
            "statistics": statistics,
        }

    output_inside_suite = output.parent == suite_dir
    if output_inside_suite:
        expected_files.add(output.name)
    actual_files = {
        path.name
        for path in suite_dir.iterdir()
    }
    allowed_actual = expected_files if output.exists() else expected_files - {output.name}
    _expect(actual_files, allowed_actual, "suite directory file set")

    payload = {
        "schema": {
            "name": VERIFICATION_SCHEMA_NAME,
            "version": VERIFICATION_SCHEMA_VERSION,
        },
        "result": "pass",
        "suite_id": config["suite_id"],
        "eligibility": {
            "production_request_suite_verified": True,
            "frontier_replays_attached": False,
            "paper_result_eligible": False,
            "reason": (
                "request provenance and window transformation are verified; "
                "simulator execution and timing remain separate gates"
            ),
        },
        "artifacts": {
            "suite_config": {
                "path": str(suite_config),
                "sha256": _sha256_file(suite_config),
                "bytes": suite_config.stat().st_size,
            },
            "source": {
                "path": str(source),
                "sha256": _sha256_file(source),
                "bytes": source.stat().st_size,
            },
            "suite_manifest": {
                "path": str(suite_manifest_path),
                "sha256": _sha256_file(suite_manifest_path),
                "bytes": suite_manifest_path.stat().st_size,
            },
        },
        "frontier_workload": config["frontier_workload"],
        "windows": verified_windows,
    }
    _write_receipt(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_suite(
            source=args.input,
            suite_config=args.suite_config,
            suite_dir=args.suite_dir,
            output=args.output,
        )
    except (OSError, VerificationError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    print(
        f"verified suite {result['suite_id']}: "
        f"windows={len(result['windows'])}"
    )
    print(f"verification={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
