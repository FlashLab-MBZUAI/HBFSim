#!/usr/bin/env python3
"""Run closed-form HBM/HBF microbenchmarks and a bottleneck phase grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.analytical_contract import (  # noqa: E402
    ABS_TOLERANCE_NS,
    CLAIMS,
    COMMAND_BYTES,
    CONTROLLED_BOUNDARY,
    FAST_GBPS,
    FAST_NS,
    HBM_FORMULA,
    HBM_READ_BYTES,
    HBIO_GBPS_AXIS,
    LARGE_PAGES,
    LIMITATIONS,
    OVERLAP_BYTES_PER_TIER,
    OVERLAP_FORMULA,
    OVERLAP_FLAT_BOUNDARY,
    PAGE_BYTES,
    PHASE_FORMULA,
    READ_NS_AXIS,
    REL_TOLERANCE,
    REPORT_SCHEMA as SCHEMA,
    SERIAL_HBF_CAPACITY,
    SERIAL_HBF_CONTROLLER_DRAM,
    SMALL_PAGES,
)
from validation.certificate import repository_state  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
SUMMARY_SCHEMA = {
    "name": "hbfsim.scenario_compare.summary",
    "version": 16,
}


class AnalyticalError(RuntimeError):
    """An analytical contract or production run failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalyticalError(f"artifact is not a file: {path}")
    return {
        "algorithm": "sha256",
        "value": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _artifact_record(base: Path, path: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(base.resolve())
    except ValueError as error:
        raise AnalyticalError(
            f"artifact escapes output directory: {path}") from error
    return {
        "path": relative.as_posix(),
        **_file_record(path),
    }


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    _write_atomic(
        path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=REL_TOLERANCE,
        abs_tol=ABS_TOLERANCE_NS,
    )


def _require_close(
    observed: float,
    expected: float,
    where: str,
) -> None:
    if not _close(observed, expected):
        raise AnalyticalError(
            f"{where}: observed {observed:.17g}, "
            f"expected {expected:.17g}")


def _run(
    command: list[str],
    *,
    repository: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode:
        raise AnalyticalError(
            f"command failed with exit code {completed.returncode}: "
            f"{command!r}\nstdout tail:\n{completed.stdout[-3000:]}\n"
            f"stderr tail:\n{completed.stderr[-3000:]}")
    return completed


def _load_summary(
    path: Path,
    *,
    scenario: str,
    binary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnalyticalError(f"invalid scenario summary {path}: {error}") from error
    if not isinstance(document, dict):
        raise AnalyticalError(f"{path}: summary root must be an object")
    if document.get("schema") != SUMMARY_SCHEMA:
        raise AnalyticalError(f"{path}: unsupported summary schema")
    if document.get("sanity") != "PASS":
        raise AnalyticalError(
            f"{path}: scenario SANITY={document.get('sanity')!r}")
    build = document.get("build")
    if not isinstance(build, dict):
        raise AnalyticalError(f"{path}: summary omitted build provenance")
    digest = build.get("executable_digest")
    if (
        not isinstance(digest, dict)
        or digest.get("algorithm") != "sha256"
        or digest.get("value") != binary["value"]
        or build.get("executable_file_bytes") != binary["bytes"]
    ):
        raise AnalyticalError(
            f"{path}: summary executable differs from the runner binary")
    scenarios = document.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or len(scenarios) != 1
        or not isinstance(scenarios[0], dict)
        or scenarios[0].get("name") != scenario
    ):
        raise AnalyticalError(
            f"{path}: expected exactly scenario {scenario}")
    return document, scenarios[0]


def _run_summary(
    *,
    binary_path: Path,
    binary_record: dict[str, Any],
    repository: Path,
    config: Path,
    output: Path,
    scenario: str,
    arguments: list[str],
    timeout: int,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    command = [
        str(binary_path),
        "--config",
        str(config),
        *arguments,
        "--scenarios",
        scenario,
        "--address-heatmap-bins",
        "1",
        "--trace-mode",
        "off",
        "--summary-json",
        str(temporary),
    ]
    try:
        _run(command, repository=repository, timeout=timeout)
        if _file_record(binary_path) != binary_record:
            raise AnalyticalError(
                "scenario_compare changed while analytical cases ran")
        document, result = _load_summary(
            temporary,
            scenario=scenario,
            binary=binary_record,
        )
        os.replace(temporary, output)
        return document, result, command
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _resource(
    scenario: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    try:
        value = scenario["time_breakdown"]["resource_busy"][name]
    except (KeyError, TypeError) as error:
        raise AnalyticalError(
            f"scenario omitted resource_busy.{name}") from error
    if not isinstance(value, dict):
        raise AnalyticalError(f"resource_busy.{name} must be an object")
    return value


def _wall_ns(scenario: dict[str, Any]) -> float:
    try:
        value = scenario["time_breakdown"]["wall_clock_ns"]["makespan_ns"]
    except (KeyError, TypeError) as error:
        raise AnalyticalError("scenario omitted makespan_ns") from error
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise AnalyticalError("scenario makespan_ns is not positive finite")
    return float(value)


def _validate_resource_record(
    resource: dict[str, Any],
    *,
    expected_busy_ns: float,
    expected_count: int,
    where: str,
) -> None:
    busy = resource.get("busy_ns")
    count = resource.get("resource_count")
    span = resource.get("active_span_ns")
    capacity = resource.get("capacity_time_ns")
    utilization = resource.get("utilization")
    if (
        isinstance(busy, bool)
        or not isinstance(busy, (int, float))
        or count != expected_count
        or isinstance(span, bool)
        or not isinstance(span, (int, float))
        or isinstance(capacity, bool)
        or not isinstance(capacity, (int, float))
        or isinstance(utilization, bool)
        or not isinstance(utilization, (int, float))
    ):
        raise AnalyticalError(f"{where}: malformed resource record")
    _require_close(float(busy), expected_busy_ns, f"{where}.busy_ns")
    _require_close(
        float(capacity),
        float(span) * expected_count,
        f"{where}.capacity_time_ns",
    )
    expected_utilization = (
        0.0 if capacity == 0 else float(busy) / float(capacity))
    _require_close(
        float(utilization),
        expected_utilization,
        f"{where}.utilization",
    )


def _hbm_work_case(
    *,
    binary_path: Path,
    binary_record: dict[str, Any],
    repository: Path,
    config: Path,
    artifact_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    bytes_read = HBM_READ_BYTES
    output = artifact_dir / "hbm-data-bus-work.summary.json"
    document, scenario, command = _run_summary(
        binary_path=binary_path,
        binary_record=binary_record,
        repository=repository,
        config=config,
        output=output,
        scenario="all-HBM",
        arguments=[
            "--synthetic-sequential-read-bytes", str(bytes_read),
            "--line-size", str(PAGE_BYTES),
            "--interarrival-ns", "0",
            "--max-hbm-outstanding-requests", "4096",
        ],
        timeout=timeout,
    )
    hbm_config = document.get("config", {}).get("hbm")
    hbm_stats = scenario.get("hbm_stats")
    if not isinstance(hbm_config, dict) or not isinstance(hbm_stats, dict):
        raise AnalyticalError("HBM analytical case omitted config/stats")
    bandwidth = hbm_config.get("pseudo_channel_bw_GBps")
    resources = hbm_config.get("stacks")
    channels = hbm_config.get("channels")
    pseudo_channels = hbm_config.get("pseudo_channels")
    if (
        isinstance(bandwidth, bool)
        or not isinstance(bandwidth, (int, float))
        or not all(type(value) is int and value > 0 for value in (
            resources, channels, pseudo_channels))
    ):
        raise AnalyticalError("HBM analytical config is malformed")
    resource_count = resources * channels * pseudo_channels
    predicted_busy_ns = bytes_read / float(bandwidth)
    bus = _resource(scenario, "hbm_data_bus")
    _validate_resource_record(
        bus,
        expected_busy_ns=predicted_busy_ns,
        expected_count=resource_count,
        where="hbm_data_bus",
    )
    if (
        hbm_stats.get("read_bytes") != bytes_read
        or hbm_stats.get("write_bytes") != 0
        or hbm_stats.get("active_pseudo_channels") != resource_count
    ):
        raise AnalyticalError(
            "HBM analytical case did not conserve read traffic/resources")
    predicted_per_resource = predicted_busy_ns / resource_count
    _require_close(
        float(hbm_stats["max_pseudo_channel_busy_ns"]),
        predicted_per_resource,
        "hbm_stats.max_pseudo_channel_busy_ns",
    )
    _require_close(
        float(hbm_stats["pseudo_channel_busy_skew"]),
        1.0,
        "hbm_stats.pseudo_channel_busy_skew",
    )
    return {
        "case_id": "analytical.hbm-data-bus-work",
        "status": "pass",
        "formula": HBM_FORMULA,
        "read_bytes": bytes_read,
        "pseudo_channel_bandwidth_GBps": bandwidth,
        "resource_count": resource_count,
        "predicted_busy_ns": predicted_busy_ns,
        "observed_busy_ns": bus["busy_ns"],
        "predicted_max_per_resource_busy_ns": predicted_per_resource,
        "observed_max_per_resource_busy_ns": (
            hbm_stats["max_pseudo_channel_busy_ns"]),
        "command": command,
        "summary": _artifact_record(artifact_dir, output),
    }


def _serial_hbf_arguments(
    *,
    pages: int,
    read_ns: float,
    hbio_gbps: float,
) -> list[str]:
    return [
        "--synthetic-sequential-read-bytes", str(pages * PAGE_BYTES),
        "--line-size", str(PAGE_BYTES),
        "--interarrival-ns", "0",
        "--max-hbf-outstanding-requests", "1024",
        "--hbf-capacity-bytes", str(SERIAL_HBF_CAPACITY),
        "--hbf-stacks", "1",
        "--hbf-channels", "1",
        "--hbf-dies-per-channel", "1",
        "--hbf-planes-per-die", "1",
        "--hbf-blocks-per-plane", "4",
        "--hbf-pages-per-block", "256",
        "--hbf-page-size", str(PAGE_BYTES),
        "--hbf-oob-bytes", "0",
        "--hbf-media-lanes-per-plane", "1",
        "--hbf-subarrays-per-plane", "1",
        "--hbf-page-buffer-banks-per-plane", "1",
        "--hbf-read-ns", format(read_ns, ".17g"),
        "--hbf-channel-bw", format(FAST_GBPS, ".17g"),
        "--hbf-hbio-bw", format(hbio_gbps, ".17g"),
        "--hbf-tsv-bw", format(FAST_GBPS, ".17g"),
        "--hbf-media-lane-bw", format(FAST_GBPS, ".17g"),
        "--hbf-logic-sram-bw", format(FAST_GBPS, ".17g"),
        "--hbf-page-buffer-bw", format(FAST_GBPS, ".17g"),
        "--hbf-ctrl-dram-bytes", str(SERIAL_HBF_CONTROLLER_DRAM),
        "--hbf-ctrl-dram-latency-ns", format(FAST_NS, ".17g"),
        "--hbf-ctrl-dram-issue-ns", format(FAST_NS, ".17g"),
        "--hbf-flash-tsu-issue-ns", format(FAST_NS, ".17g"),
        "--hbf-logic-scheduler-issue-ns", format(FAST_NS, ".17g"),
        "--hbf-ecc-decode-latency-ns", format(FAST_NS, ".17g"),
        "--hbf-ecc-decode-raw-bw", format(FAST_GBPS, ".17g"),
        "--hbf-batch-activation", "false",
        "--hbf-read-buffer-pages", "0",
        "--hbf-gc-low-watermark-pages", "1",
        "--hbf-gc-hard-watermark-pages", "0",
        "--hbf-gc-reserved-free-blocks-per-plane", "1",
    ]


def _validate_serial_hbf_summary(
    document: dict[str, Any],
    scenario: dict[str, Any],
    *,
    pages: int,
    read_ns: float,
    hbio_gbps: float,
    where: str,
) -> None:
    config = document.get("config", {}).get("hbf")
    stats = scenario.get("hbf_stats")
    if not isinstance(config, dict) or not isinstance(stats, dict):
        raise AnalyticalError(f"{where}: omitted HBF config/stats")
    expected_config = {
        "stacks": 1,
        "channels": 1,
        "dies_per_channel": 1,
        "planes_per_die": 1,
        "media_lanes_per_plane": 1,
        "subarrays_per_plane": 1,
        "page_buffer_banks_per_plane": 1,
        "page_size": PAGE_BYTES,
        "oob_bytes": 0,
        "read_ns": read_ns,
        "hbio_bw_GBps": hbio_gbps,
        "batch_activation": False,
        "read_buffer_pages": 0,
    }
    for key, expected in expected_config.items():
        observed = config.get(key)
        if isinstance(expected, float):
            if not isinstance(observed, (int, float)) or not _close(
                float(observed), expected
            ):
                raise AnalyticalError(
                    f"{where}.config.{key}: expected {expected}, "
                    f"observed {observed!r}")
        elif observed != expected:
            raise AnalyticalError(
                f"{where}.config.{key}: expected {expected!r}, "
                f"observed {observed!r}")
    expected_bytes = pages * PAGE_BYTES
    expected_stats = {
        "logical_read_bytes": expected_bytes,
        "physical_read_bytes": expected_bytes,
        "logical_write_bytes": 0,
        "physical_write_bytes": 0,
        "page_reads": pages,
        "read_buffer_hits": 0,
        "read_buffer_misses": pages,
        "block_erases": 0,
        "gc_runs": 0,
    }
    for key, expected in expected_stats.items():
        if stats.get(key) != expected:
            raise AnalyticalError(
                f"{where}.hbf_stats.{key}: expected {expected}, "
                f"observed {stats.get(key)!r}")

    expected_busy = {
        "hbf_logic_ingress": pages * FAST_NS,
        "hbf_mapping_dram_issue": pages * FAST_NS,
        "hbf_plane_media": pages * read_ns,
        "hbf_media_lane": pages * PAGE_BYTES / FAST_GBPS,
        "hbf_subarray": pages * read_ns,
        "hbf_page_buffer_bank": pages * PAGE_BYTES / FAST_GBPS,
        "hbf_channel_command": pages * COMMAND_BYTES / FAST_GBPS,
        "hbf_channel_data": pages * PAGE_BYTES / FAST_GBPS,
        "hbf_tsv": (
            pages * (COMMAND_BYTES + PAGE_BYTES) / FAST_GBPS),
        "hbf_sram": pages * PAGE_BYTES / FAST_GBPS,
        "hbf_hbio_command": pages * COMMAND_BYTES / hbio_gbps,
        "hbf_hbio_data": pages * PAGE_BYTES / hbio_gbps,
        "hbf_sequencer": pages * FAST_NS,
        "hbf_ecc_issue": pages * PAGE_BYTES / FAST_GBPS,
    }
    for name, expected in expected_busy.items():
        _validate_resource_record(
            _resource(scenario, name),
            expected_busy_ns=expected,
            expected_count=1,
            where=f"{where}.{name}",
        )


def _bottleneck_label(media_ii_ns: float, hbio_ii_ns: float) -> str:
    if _close(media_ii_ns, hbio_ii_ns):
        return "co-bottleneck"
    return "media" if media_ii_ns > hbio_ii_ns else "hbio-data"


def _hbf_phase_grid(
    *,
    binary_path: Path,
    binary_record: dict[str, Any],
    repository: Path,
    config: Path,
    artifact_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    region_counts = {
        "media": 0,
        "hbio-data": 0,
        "co-bottleneck": 0,
    }
    max_abs_error = 0.0
    max_rel_error = 0.0
    for read_ns in READ_NS_AXIS:
        for hbio_gbps in HBIO_GBPS_AXIS:
            stem = (
                f"hbf-r{read_ns:g}-hbio{hbio_gbps:g}"
                .replace(".", "p")
            )
            summaries: dict[int, tuple[
                dict[str, Any],
                dict[str, Any],
                Path,
                list[str],
            ]] = {}
            for pages in (SMALL_PAGES, LARGE_PAGES):
                path = artifact_dir / f"{stem}-n{pages}.summary.json"
                document, scenario, command = _run_summary(
                    binary_path=binary_path,
                    binary_record=binary_record,
                    repository=repository,
                    config=config,
                    output=path,
                    scenario="all-HBF",
                    arguments=_serial_hbf_arguments(
                        pages=pages,
                        read_ns=read_ns,
                        hbio_gbps=hbio_gbps,
                    ),
                    timeout=timeout,
                )
                _validate_serial_hbf_summary(
                    document,
                    scenario,
                    pages=pages,
                    read_ns=read_ns,
                    hbio_gbps=hbio_gbps,
                    where=f"{stem}.n{pages}",
                )
                summaries[pages] = (document, scenario, path, command)

            small = summaries[SMALL_PAGES][1]
            large = summaries[LARGE_PAGES][1]
            observed_ii = (
                _wall_ns(large) - _wall_ns(small)
            ) / (LARGE_PAGES - SMALL_PAGES)
            media_ii = read_ns
            hbio_ii = PAGE_BYTES / hbio_gbps
            predicted_ii = max(media_ii, hbio_ii)
            _require_close(
                observed_ii,
                predicted_ii,
                f"{stem}.steady_state_ii_ns",
            )
            absolute_error = abs(observed_ii - predicted_ii)
            relative_error = (
                0.0 if predicted_ii == 0
                else absolute_error / predicted_ii
            )
            max_abs_error = max(max_abs_error, absolute_error)
            max_rel_error = max(max_rel_error, relative_error)
            predicted_bottleneck = _bottleneck_label(media_ii, hbio_ii)

            large_makespan = _wall_ns(large)
            media_global_occupancy = max(
                float(_resource(large, "hbf_plane_media")["busy_ns"]),
                float(_resource(large, "hbf_subarray")["busy_ns"]),
            ) / large_makespan
            hbio_global_occupancy = float(
                _resource(large, "hbf_hbio_data")["busy_ns"]
            ) / large_makespan
            observed_bottleneck = _bottleneck_label(
                media_global_occupancy, hbio_global_occupancy)
            if observed_bottleneck != predicted_bottleneck:
                raise AnalyticalError(
                    f"{stem}: predicted {predicted_bottleneck}, "
                    f"observed utilization region {observed_bottleneck}")
            region_counts[predicted_bottleneck] += 1
            points.append({
                "read_ns": read_ns,
                "hbio_GBps": hbio_gbps,
                "media_ii_ns": media_ii,
                "hbio_data_ii_ns": hbio_ii,
                "predicted_steady_state_ii_ns": predicted_ii,
                "observed_steady_state_ii_ns": observed_ii,
                "absolute_error_ns": absolute_error,
                "relative_error": relative_error,
                "predicted_bottleneck": predicted_bottleneck,
                "observed_bottleneck": observed_bottleneck,
                "large_run_media_global_occupancy": (
                    media_global_occupancy
                ),
                "large_run_hbio_data_global_occupancy": (
                    hbio_global_occupancy
                ),
                "status": "pass",
                "runs": {
                    str(pages): {
                        "pages": pages,
                        "makespan_ns": _wall_ns(summaries[pages][1]),
                        "command": summaries[pages][3],
                        "summary": _artifact_record(
                            artifact_dir, summaries[pages][2]),
                    }
                    for pages in (SMALL_PAGES, LARGE_PAGES)
                },
            })

    expected_points = len(READ_NS_AXIS) * len(HBIO_GBPS_AXIS)
    if len(points) != expected_points or sum(region_counts.values()) != expected_points:
        raise AnalyticalError("HBF phase-grid census is incomplete")
    return {
        "status": "pass",
        "formula": PHASE_FORMULA,
        "controlled_boundary": CONTROLLED_BOUNDARY,
        "axes": {
            "read_ns": list(READ_NS_AXIS),
            "hbio_GBps": list(HBIO_GBPS_AXIS),
        },
        "point_count": expected_points,
        "region_counts": region_counts,
        "max_absolute_error_ns": max_abs_error,
        "max_relative_error": max_rel_error,
        "points": points,
    }


def _busy_snapshot(
    scenario: dict[str, Any],
    prefix: str,
) -> dict[str, float]:
    resources = scenario.get("time_breakdown", {}).get("resource_busy")
    if not isinstance(resources, dict):
        raise AnalyticalError("overlap scenario omitted resource_busy")
    result: dict[str, float] = {}
    for name, record in resources.items():
        if not name.startswith(prefix):
            continue
        if not isinstance(record, dict):
            raise AnalyticalError(f"resource_busy.{name} is malformed")
        busy = record.get("busy_ns")
        if (
            isinstance(busy, bool)
            or not isinstance(busy, (int, float))
            or not math.isfinite(float(busy))
            or busy < 0
        ):
            raise AnalyticalError(
                f"resource_busy.{name}.busy_ns is malformed")
        result[name] = float(busy)
    if not result:
        raise AnalyticalError(f"no {prefix} resource records")
    return result


def _overlap_case(
    *,
    binary_path: Path,
    binary_record: dict[str, Any],
    repository: Path,
    config: Path,
    artifact_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    base = OVERLAP_FLAT_BOUNDARY - OVERLAP_BYTES_PER_TIER
    specifications = (
        (
            "hbm-only",
            "all-HBM",
            base,
            OVERLAP_BYTES_PER_TIER,
            ["--max-hbm-outstanding-requests", "4096"],
        ),
        (
            "hbf-only",
            "all-HBF",
            OVERLAP_FLAT_BOUNDARY,
            OVERLAP_BYTES_PER_TIER,
            ["--max-hbf-outstanding-requests", "4096"],
        ),
        (
            "mixed",
            "HBM-HBF-Flat",
            base,
            2 * OVERLAP_BYTES_PER_TIER,
            [
                "--flat-hbm-bytes", str(OVERLAP_FLAT_BOUNDARY),
                "--max-hbm-outstanding-requests", "4096",
                "--max-hbf-outstanding-requests", "4096",
            ],
        ),
    )
    runs: dict[str, dict[str, Any]] = {}
    for key, scenario_name, address, size, extra in specifications:
        output = artifact_dir / f"independent-tier-overlap-{key}.summary.json"
        _, scenario, command = _run_summary(
            binary_path=binary_path,
            binary_record=binary_record,
            repository=repository,
            config=config,
            output=output,
            scenario=scenario_name,
            arguments=[
                "--synthetic-sequential-read-base", str(address),
                "--synthetic-sequential-read-bytes", str(size),
                "--line-size", str(PAGE_BYTES),
                "--interarrival-ns", "0",
                *extra,
            ],
            timeout=timeout,
        )
        runs[key] = {
            "scenario": scenario,
            "makespan_ns": _wall_ns(scenario),
            "command": command,
            "summary": _artifact_record(artifact_dir, output),
        }

    hbm = runs["hbm-only"]
    hbf = runs["hbf-only"]
    mixed = runs["mixed"]
    predicted_mixed = max(hbm["makespan_ns"], hbf["makespan_ns"])
    _require_close(
        mixed["makespan_ns"],
        predicted_mixed,
        "independent-tier-overlap.mixed_makespan_ns",
    )
    if not mixed["makespan_ns"] < (
        hbm["makespan_ns"] + hbf["makespan_ns"]
    ):
        raise AnalyticalError(
            "mixed HBM/HBF makespan serialized the independent tiers")
    hbm_scenario = hbm["scenario"]
    hbf_scenario = hbf["scenario"]
    mixed_scenario = mixed["scenario"]
    pages = OVERLAP_BYTES_PER_TIER // PAGE_BYTES
    expected_routes = {
        "hbm-only": (pages, 0),
        "hbf-only": (0, pages),
        "mixed": (pages, pages),
    }
    for key, (hbm_accesses, hbf_accesses) in expected_routes.items():
        scenario = runs[key]["scenario"]
        if (
            scenario.get("hbm_accesses") != hbm_accesses
            or scenario.get("hbf_accesses") != hbf_accesses
        ):
            raise AnalyticalError(
                f"independent-tier-overlap.{key}: route census mismatch")

    hbm_busy = _busy_snapshot(hbm_scenario, "hbm_")
    hbf_busy = _busy_snapshot(hbf_scenario, "hbf_")
    mixed_hbm_busy = _busy_snapshot(mixed_scenario, "hbm_")
    mixed_hbf_busy = _busy_snapshot(mixed_scenario, "hbf_")
    if hbm_busy.keys() != mixed_hbm_busy.keys():
        raise AnalyticalError("mixed run changed HBM resource census")
    if hbf_busy.keys() != mixed_hbf_busy.keys():
        raise AnalyticalError("mixed run changed HBF resource census")
    for name, expected in hbm_busy.items():
        _require_close(
            mixed_hbm_busy[name], expected,
            f"independent-tier-overlap.mixed.{name}")
    for name, expected in hbf_busy.items():
        _require_close(
            mixed_hbf_busy[name], expected,
            f"independent-tier-overlap.mixed.{name}")

    return {
        "case_id": "analytical.independent-tier-overlap",
        "status": "pass",
        "formula": OVERLAP_FORMULA,
        "bytes_per_tier": OVERLAP_BYTES_PER_TIER,
        "hbm_only_makespan_ns": hbm["makespan_ns"],
        "hbf_only_makespan_ns": hbf["makespan_ns"],
        "predicted_mixed_makespan_ns": predicted_mixed,
        "observed_mixed_makespan_ns": mixed["makespan_ns"],
        "serialized_sum_ns": hbm["makespan_ns"] + hbf["makespan_ns"],
        "hbm_resource_busy_preserved": True,
        "hbf_resource_busy_preserved": True,
        "runs": {
            key: {
                name: value
                for name, value in run.items()
                if name != "scenario"
            }
            for key, run in runs.items()
        },
    }


def _write_phase_csv(
    path: Path,
    points: list[dict[str, Any]],
) -> None:
    fields = (
        "read_ns",
        "hbio_GBps",
        "media_ii_ns",
        "hbio_data_ii_ns",
        "predicted_steady_state_ii_ns",
        "observed_steady_state_ii_ns",
        "absolute_error_ns",
        "relative_error",
        "predicted_bottleneck",
        "observed_bottleneck",
        "large_run_media_global_occupancy",
        "large_run_hbio_data_global_occupancy",
        "status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({field: point[field] for field in fields})
    os.replace(temporary, path)


def _write_phase_markdown(
    path: Path,
    points: list[dict[str, Any]],
) -> None:
    by_coordinates = {
        (point["read_ns"], point["hbio_GBps"]): point
        for point in points
    }
    lines = [
        "# Analytical HBF Media/HBIO Phase Diagram",
        "",
        "Each cell is `bottleneck / observed steady-state II (ns)`. "
        "`M` is media, `H` is HBIO data, and `C` is the exact crossover.",
        "",
        "| read ns \\\\ HBIO GB/s | "
        + " | ".join(f"{value:g}" for value in HBIO_GBPS_AXIS)
        + " |",
        "|---:" + "|---:" * len(HBIO_GBPS_AXIS) + "|",
    ]
    abbreviations = {
        "media": "M",
        "hbio-data": "H",
        "co-bottleneck": "C",
    }
    for read_ns in READ_NS_AXIS:
        cells = []
        for hbio in HBIO_GBPS_AXIS:
            point = by_coordinates[(read_ns, hbio)]
            cells.append(
                f"{abbreviations[point['observed_bottleneck']]} / "
                f"{point['observed_steady_state_ii_ns']:.6g}"
            )
        lines.append(f"| {read_ns:g} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "This is an isolated one-resource-per-stage validation grid, not an "
        "application performance result or hardware calibration.",
        "",
    ])
    _write_atomic(path, "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/scenario_compare/usecase-baseline.cfg",
    )
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    repository = args.repository.resolve()
    binary = args.scenario_compare.resolve()
    config = args.config.resolve()
    artifact_dir = args.artifact_dir.resolve()
    report_path = args.report.resolve()
    try:
        if not binary.is_file():
            raise AnalyticalError(
                f"scenario_compare does not exist: {binary}")
        if not config.is_file():
            raise AnalyticalError(f"config does not exist: {config}")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        binary_record = _file_record(binary)
        config_record = _file_record(config)
        source = repository_state(repository, require_clean=False)

        hbm = _hbm_work_case(
            binary_path=binary,
            binary_record=binary_record,
            repository=repository,
            config=config,
            artifact_dir=artifact_dir,
            timeout=args.timeout,
        )
        phase = _hbf_phase_grid(
            binary_path=binary,
            binary_record=binary_record,
            repository=repository,
            config=config,
            artifact_dir=artifact_dir,
            timeout=args.timeout,
        )
        overlap = _overlap_case(
            binary_path=binary,
            binary_record=binary_record,
            repository=repository,
            config=config,
            artifact_dir=artifact_dir,
            timeout=args.timeout,
        )
        if _file_record(binary) != binary_record:
            raise AnalyticalError(
                "scenario_compare changed during analytical validation")
        if _file_record(config) != config_record:
            raise AnalyticalError(
                "baseline config changed during analytical validation")

        csv_path = artifact_dir / "phase-diagram.csv"
        markdown_path = artifact_dir / "phase-diagram.md"
        _write_phase_csv(csv_path, phase["points"])
        _write_phase_markdown(markdown_path, phase["points"])
        report = {
            "schema": SCHEMA,
            "status": "pass",
            "source": source,
            "scenario_compare": binary_record,
            "baseline_config": {
                "path": config.relative_to(repository).as_posix(),
                **config_record,
            },
            "tolerances": {
                "absolute_ns": ABS_TOLERANCE_NS,
                "relative": REL_TOLERANCE,
            },
            "hbm_data_bus_work": hbm,
            "hbf_media_hbio_phase_diagram": phase,
            "independent_tier_overlap": overlap,
            "artifacts": {
                "phase_csv": _artifact_record(artifact_dir, csv_path),
                "phase_markdown": _artifact_record(
                    artifact_dir, markdown_path),
            },
            "claims": list(CLAIMS),
            "limitations": list(LIMITATIONS),
        }
        _write_json_atomic(report_path, report)
    except (
        AnalyticalError,
        OSError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        print(f"analytical microbench failed: {error}", file=sys.stderr)
        return 1

    print(
        "PASS analytical microbench: "
        f"HBM work=exact, "
        f"HBF phase points={phase['point_count']}, "
        f"max II error={phase['max_absolute_error_ns']:.3g} ns, "
        "independent-tier overlap=max, "
        f"report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
