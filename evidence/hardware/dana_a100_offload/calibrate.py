#!/usr/bin/env python3
"""Fit and replay the DANA A100 host-DRAM/NVMe calibration.

Only the Python standard library is used for fitting.  Prediction receipts are
not generated from a duplicate Python timing equation: every reported cell is
replayed through the compiled production ExternalBackingDevice.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


RESULT_SCHEMA_VERSION = 1
CALIBRATION_SCHEMA = "hbfsim.dana_a100_offload.calibration.v1"
REPLAY_SCHEMA = "hbfsim.external_calibration_replay.v1"
REQUIRED_BLOCKS = (4096, 65536, 1048576, 16777216)
REQUIRED_LANES = (1, 8)
PRIMARY_HOLDOUT_BYTES = 16777216
PAGE_BYTES = 4096
HOST_REQUEST_SEGMENT_BYTES = 16777216
NO_BOTTLENECK_GBPS = 1000000.0
MAX_NVME_EFFECTIVE_QUEUES = max(REQUIRED_LANES)

PATH_DIRECTIONS = {
    "gpu_pinned_host": ("offload_d2h", "restore_h2d"),
    "ssd_direct": ("offload_pwrite", "restore_pread"),
    "gpu_pinned_host_ssd_direct": (
        "offload_gpu_to_ssd",
        "restore_ssd_to_gpu",
    ),
}

DIRECTION_TO_OP = {
    "offload_d2h": "write",
    "restore_h2d": "read",
    "offload_pwrite": "write",
    "restore_pread": "read",
    "offload_gpu_to_ssd": "write",
    "restore_ssd_to_gpu": "read",
}


class CalibrationError(RuntimeError):
    """Input evidence or replay failed a calibration contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    require(bool(materialized), "cannot take the median of no values")
    return float(statistics.median(materialized))


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0


def trial_operations(trial: Mapping[str, Any]) -> int:
    value = trial.get("throughput_operations", trial.get("operations"))
    require(isinstance(value, int) and value > 0, "trial has no positive operation count")
    return value


def expected_trial_keys(run: Mapping[str, Any]) -> set:
    blocks = tuple(run.get("block_sizes", ()))
    lanes = tuple(run.get("lanes", ()))
    require(blocks == REQUIRED_BLOCKS, "benchmark block-size contract changed")
    require(lanes == REQUIRED_LANES, "benchmark lane contract changed")
    expected = set()
    for block in blocks:
        for direction in PATH_DIRECTIONS["gpu_pinned_host"]:
            expected.add(("gpu_pinned_host", direction, block, 1))
        for lane in lanes:
            for path in ("ssd_direct", "gpu_pinned_host_ssd_direct"):
                for direction in PATH_DIRECTIONS[path]:
                    expected.add((path, direction, block, lane))
    return expected


def storage_segment_evidence(document: Mapping[str, Any]) -> Dict[str, Any]:
    storage = document["environment"]["storage"]
    block_device = storage["block_device"]
    require(block_device.get("is_nvme_backed") is True, "storage is not proven NVMe-backed")
    devices = block_device.get("devices")
    require(isinstance(devices, list) and devices, "block-device topology is empty")
    candidates = []
    queue_rows = []
    nr_requests = []
    nvme_models = []
    nvme_leaf_names = []
    for device in devices:
        queue = device.get("queue") or {}
        max_sectors = queue.get("max_sectors_kb")
        max_hw = queue.get("max_hw_sectors_kb")
        require(finite_positive(max_sectors), "device has no max_sectors_kb")
        require(finite_positive(max_hw), "device has no max_hw_sectors_kb")
        candidates.extend((int(max_sectors) * 1024, int(max_hw) * 1024))
        if finite_positive(queue.get("nr_requests")):
            nr_requests.append(int(queue["nr_requests"]))
        if device.get("looks_like_nvme"):
            require(device.get("model"), "NVMe leaf has no model identity")
            nvme_models.append(str(device["model"]))
            require(device.get("name"), "NVMe leaf has no device name")
            nvme_leaf_names.append(str(device["name"]))
        queue_rows.append({
            "name": device.get("name"),
            "max_sectors_kb": int(max_sectors),
            "max_hw_sectors_kb": int(max_hw),
            "nr_requests": queue.get("nr_requests"),
        })
    segment = min(candidates)
    require(segment >= PAGE_BYTES and segment % PAGE_BYTES == 0,
            "sysfs segment is not a page-aligned transport payload")
    require(nvme_models, "block topology has no NVMe leaves")
    return {
        "request_segment_bytes": segment,
        "queue_rows": queue_rows,
        "nvme_models": sorted(set(nvme_models)),
        "nvme_leaf_names": sorted(nvme_leaf_names),
        "nvme_leaf_count": len(nvme_leaf_names),
        "leaf_nr_requests_min": min(nr_requests) if nr_requests else None,
    }


def validate_document(path: pathlib.Path, allow_quick: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as error:
        raise CalibrationError("{} is not readable JSON: {}".format(path, error))
    require(isinstance(document, dict), "{} root is not an object".format(path))
    label = path.name
    require(document.get("schema_version") == RESULT_SCHEMA_VERSION,
            "{} schema version changed".format(label))
    require(document.get("benchmark") == "dana_a100_offload",
            "{} benchmark identity changed".format(label))
    require(document.get("status") == "passed", "{} did not pass".format(label))
    require(document.get("errors") == [], "{} contains benchmark errors".format(label))
    summary = document.get("summary") or {}
    require(summary.get("passed") is True and summary.get("all_validations_passed") is True,
            "{} summary did not pass".format(label))
    run = document.get("run") or {}
    require(allow_quick or run.get("quick") is False,
            "{} is a quick smoke result; calibration rejects it".format(label))
    contract = document.get("contract") or {}
    require(contract.get("decimal_GB_per_s") is True,
            "{} does not publish decimal GB/s".format(label))
    require("CUDA-pinned" in str(contract.get("dram")),
            "{} DRAM path is not pinned CUDA memory".format(label))
    ssd_contract = str(contract.get("ssd"))
    for token in ("O_DIRECT", "preadv", "pwritev", "no buffered fallback"):
        require(token in ssd_contract, "{} SSD contract lacks {}".format(label, token))
    require(contract.get("validation_excluded_from_timing") is True,
            "{} includes validation in timing".format(label))

    environment = document.get("environment") or {}
    torch = environment.get("torch") or {}
    require(torch.get("cuda_available") is True and torch.get("cuda_device_count") == 1,
            "{} does not expose exactly one allocated CUDA device".format(label))
    devices = torch.get("devices") or []
    require(len(devices) == 1 and "A100" in str(devices[0].get("name")),
            "{} GPU is not an allocated A100".format(label))
    cuda_visible_device = environment.get("cuda_visible_devices")
    require(
        isinstance(cuda_visible_device, str) and
        cuda_visible_device.isdigit(),
        "{} does not bind one numeric physical CUDA device".format(label),
    )
    slurm = environment.get("slurm") or {}
    job_id = slurm.get("SLURM_JOB_ID")
    require(str(job_id).isdigit(), "{} lacks a Slurm job id".format(label))
    require(slurm.get("SLURM_JOB_PARTITION") == "cscc-gpu-p",
            "{} used an unexpected partition".format(label))
    require(slurm.get("SLURM_GPUS_ON_NODE") == "1",
            "{} did not receive one Slurm GPU".format(label))
    storage = environment.get("storage") or {}
    mount = storage.get("mount") or {}
    require(mount.get("mount_point") == "/tmp", "{} work storage is not /tmp".format(label))
    require(mount.get("filesystem_type") not in {
        "nfs", "nfs4", "lustre", "gpfs", "tmpfs", "overlay"
    }, "{} used a nonlocal filesystem".format(label))
    expected_work_dir = "/tmp/hbfsim-offload-{}".format(job_id)
    require(storage.get("work_dir") == expected_work_dir,
            "{} work directory is not job-private /tmp".format(label))
    require(storage.get("direct_alignment_bytes") == PAGE_BYTES,
            "{} direct-I/O alignment changed".format(label))
    require(isinstance(storage.get("o_direct_value"), int) and storage["o_direct_value"] > 0,
            "{} did not record O_DIRECT".format(label))
    segment_evidence = storage_segment_evidence(document)

    trials = document.get("trials")
    require(isinstance(trials, list), "{} has no trial list".format(label))
    expected = expected_trial_keys(run)
    actual = set()
    for trial in trials:
        key = (
            trial.get("path"), trial.get("direction"),
            trial.get("block_bytes"), trial.get("lanes"),
        )
        require(key in expected, "{} has an unexpected trial {}".format(label, key))
        require(key not in actual, "{} duplicates trial {}".format(label, key))
        actual.add(key)
        path_name, direction, block, lanes = key
        require(direction in PATH_DIRECTIONS[path_name], "trial direction/path mismatch")
        require(trial.get("validation", {}).get("passed") is True,
                "{} trial {} did not validate".format(label, key))
        throughput = trial.get("throughput") or {}
        require(finite_positive(throughput.get("GB_per_s")) and
                finite_positive(throughput.get("wall_time_ns")) and
                finite_positive(throughput.get("bytes")),
                "{} trial {} has invalid throughput".format(label, key))
        recomputed = throughput["bytes"] / throughput["wall_time_ns"]
        require(abs(recomputed / throughput["GB_per_s"] - 1.0) < 1.0e-9,
                "{} trial {} GB/s is inconsistent with bytes/time".format(label, key))
        latency = trial.get("latency_ns") or {}
        require(finite_positive(latency.get("p50")) and latency.get("count", 0) > 0,
                "{} trial {} has invalid latency".format(label, key))
        operations = trial_operations(trial)
        require(throughput["bytes"] == operations * block,
                "{} trial {} byte/operation count diverged".format(label, key))
        if path_name == "gpu_pinned_host":
            require(lanes == 1 and trial.get("tier") == "host_dram",
                    "pinned-host path contract changed")
            require(trial.get("max_outstanding_operations") == 1,
                    "pinned-host outstanding contract changed")
        else:
            require(trial.get("tier") == "local_nvme",
                    "SSD tier identity changed")
            require(trial.get("max_outstanding_operations") == lanes,
                    "SSD lane/outstanding contract changed")
        if path_name == "gpu_pinned_host_ssd_direct":
            components = trial.get("component_latency_ns") or {}
            required_components = (
                ("gpu_d2h_and_sync", "ssd_pwritev") if direction == "offload_gpu_to_ssd"
                else ("ssd_preadv", "gpu_h2d_and_sync")
            )
            for component in required_components:
                require(finite_positive((components.get(component) or {}).get("p50")),
                        "{} trial {} lacks component {}".format(label, key, component))
    require(actual == expected, "{} does not contain the complete 40-cell matrix".format(label))
    require(summary.get("trial_count") == len(expected), "{} trial count changed".format(label))

    artifact = environment.get("artifact") or {}
    benchmark_sha = artifact.get("benchmark_sha256")
    require(isinstance(benchmark_sha, str) and len(benchmark_sha) == 64,
            "{} has no benchmark digest".format(label))
    metadata = {
        "path": "results/" + path.name,
        "filename": path.name,
        "sha256": sha256_file(path),
        "job_id": str(job_id),
        "hostname": environment.get("host", {}).get("hostname"),
        "quick": bool(run.get("quick")),
        "benchmark_sha256": benchmark_sha,
        "python_version": environment.get("python", {}).get("version"),
        "torch_version": torch.get("version"),
        "gpu_name": devices[0].get("name"),
        "cuda_visible_device": cuda_visible_device,
        "storage_filesystem": mount.get("filesystem_type"),
        "segment_evidence": segment_evidence,
    }
    return document, metadata


def validate_source_identities(sources: Sequence[Mapping[str, Any]]) -> None:
    require(bool(sources), "calibration source set is empty")
    job_ids = [str(source.get("job_id", "")) for source in sources]
    raw_digests = [str(source.get("sha256", "")) for source in sources]
    require(
        len(set(job_ids)) == len(job_ids),
        "fit and holdout Slurm job ids must be disjoint",
    )
    require(
        len(set(raw_digests)) == len(raw_digests),
        "fit and holdout raw-result SHA-256 values must be disjoint",
    )


def aggregate_trials(documents: Sequence[Mapping[str, Any]]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    grouped = collections.defaultdict(list)
    for document in documents:
        for trial in document["trials"]:
            key = (trial["path"], trial["direction"], trial["block_bytes"], trial["lanes"])
            grouped[key].append(trial)
    require(all(len(values) == len(documents) for values in grouped.values()),
            "replicate matrix is incomplete")
    aggregates = {}
    for key, values in grouped.items():
        operations = {trial_operations(item) for item in values}
        require(len(operations) == 1, "replicates changed operation count for {}".format(key))
        component_names = set.intersection(*[
            set((item.get("component_latency_ns") or {}).keys()) for item in values
        ]) if values else set()
        aggregates[key] = {
            "path": key[0],
            "direction": key[1],
            "block_bytes": key[2],
            "lanes": key[3],
            "replicates": len(values),
            "operations": operations.pop(),
            "measured": {
                "throughput_GB_per_s": median(item["throughput"]["GB_per_s"] for item in values),
                "wall_time_ns_per_operation": median(
                    item["throughput"]["wall_time_ns"] / trial_operations(item) for item in values
                ),
                "latency_p50_ns": median(item["latency_ns"]["p50"] for item in values),
                "component_latency_p50_ns": {
                    name: median(item["component_latency_ns"][name]["p50"] for item in values)
                    for name in sorted(component_names)
                },
            },
        }
    return aggregates


def linear_regression(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    require(len(points) >= 2, "linear fit needs at least two sizes")
    count = float(len(points))
    sx = sum(point[0] for point in points)
    sy = sum(point[1] for point in points)
    sxx = sum(point[0] * point[0] for point in points)
    sxy = sum(point[0] * point[1] for point in points)
    denominator = count * sxx - sx * sx
    require(denominator > 0.0, "linear fit sizes are degenerate")
    slope = (count * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / count
    require(slope > 0.0 and math.isfinite(slope), "linear fit produced nonpositive bandwidth")
    return intercept, slope


def common_config(kind: str, segment_bytes: int) -> Dict[str, Any]:
    return {
        "kind": kind,
        "capacity_bytes": 1 << 40,
        "page_size_bytes": PAGE_BYTES,
        "request_segment_bytes": segment_bytes,
        "media_channels": 1,
        "media_read_queues": 1,
        "media_write_queues": 1,
        "max_outstanding_requests": 512,
        "controller_issue_ns": 0.0,
        "controller_processing_ns": 0.0,
        "media_read_latency_ns": 0.0,
        "media_write_latency_ns": 0.0,
        "media_read_bandwidth_GBps": NO_BOTTLENECK_GBPS,
        "media_write_bandwidth_GBps": NO_BOTTLENECK_GBPS,
        "m2s_bandwidth_GBps": NO_BOTTLENECK_GBPS,
        "s2m_bandwidth_GBps": NO_BOTTLENECK_GBPS,
        "one_way_propagation_ns": 0.0,
        "command_bytes": 64,
        "completion_bytes": 16,
    }


def fit_host(
    groups: Mapping[Tuple[Any, ...], Mapping[str, Any]],
) -> Dict[str, Any]:
    direction_rows = {}
    small_block_service = []
    for direction in PATH_DIRECTIONS["gpu_pinned_host"]:
        rows = sorted(
            (row for row in groups.values()
             if row["path"] == "gpu_pinned_host" and
             row["direction"] == direction),
            key=lambda row: row["block_bytes"],
        )
        # The production controller is a pipelined issue resource, so use the
        # directly observed small-copy service interval instead of treating a
        # regression intercept as an additive end-to-end latency.
        small_block_service.extend(
            row["measured"]["wall_time_ns_per_operation"]
            for row in rows if row["block_bytes"] <= 65536
        )
        direction_rows[direction] = rows
    controller_issue = median(small_block_service)
    bandwidth = {}
    for direction, rows in direction_rows.items():
        numerator = sum(
            row["block_bytes"] *
            (row["measured"]["wall_time_ns_per_operation"] - controller_issue)
            for row in rows
        )
        denominator = sum(row["block_bytes"] ** 2 for row in rows)
        slope = numerator / denominator
        require(slope > 0.0, "host link bandwidth fit is nonpositive")
        bandwidth[direction] = 1.0 / slope
    latency_residuals = []
    for direction, rows in direction_rows.items():
        for row in rows:
            residual = (
                row["measured"]["latency_p50_ns"] - controller_issue -
                row["block_bytes"] / bandwidth[direction]
            )
            if residual >= 0.0:
                latency_residuals.append(residual)
    require(latency_residuals, "host fixed latency fit has no nonnegative residuals")
    config = common_config("host-dram", HOST_REQUEST_SEGMENT_BYTES)
    config.update({
        "controller_issue_ns": controller_issue,
        "controller_processing_ns": median(latency_residuals),
        "m2s_bandwidth_GBps": bandwidth["offload_d2h"],
        "s2m_bandwidth_GBps": bandwidth["restore_h2d"],
    })
    return config


def select_effective_queue_count(
    qd1_throughput: float,
    qdmax_throughput: float,
    max_lanes: int,
) -> int:
    """Select an empirical caller-range service width from fit anchors only."""
    require(finite_positive(qd1_throughput), "QD1 throughput is not positive")
    require(finite_positive(qdmax_throughput), "high-QD throughput is not positive")
    require(isinstance(max_lanes, int) and max_lanes > 0,
            "high-QD lane count is not positive")
    observed_ratio = max(1.0, qdmax_throughput / qd1_throughput)
    candidates = range(1, min(max_lanes, MAX_NVME_EFFECTIVE_QUEUES) + 1)
    # The production model divides aggregate media service evenly across the
    # configured queues.  Pick the integer width closest to the observed
    # QDmax/QD1 scaling in log space; ties choose the smaller model.
    return min(
        candidates,
        key=lambda candidate: (
            abs(math.log(float(candidate) / observed_ratio)), candidate
        ),
    )


def fit_nvme(
    groups: Mapping[Tuple[Any, ...], Mapping[str, Any]],
    host_config: Mapping[str, Any],
    segment_bytes: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raw_rows = [
        row for row in groups.values()
        if row["path"] == "ssd_direct"
    ]
    anchor_block = PRIMARY_HOLDOUT_BYTES
    bandwidth = {}
    latency = {}
    direction_fields = {
        "offload_pwrite": "write",
        "restore_pread": "read",
    }
    queue_fit = {}
    for direction, name in direction_fields.items():
        anchor = [row for row in raw_rows
                  if row["direction"] == direction and row["block_bytes"] == anchor_block]
        qd1 = next(row for row in anchor if row["lanes"] == 1)
        qdmax = max(anchor, key=lambda row: row["lanes"])
        queues = select_effective_queue_count(
            qd1["measured"]["throughput_GB_per_s"],
            qdmax["measured"]["throughput_GB_per_s"],
            qdmax["lanes"],
        )
        queue_fit[name] = {
            "direction": direction,
            "qd1_lanes": qd1["lanes"],
            "qdmax_lanes": qdmax["lanes"],
            "qd1_throughput_GB_per_s": qd1["measured"]["throughput_GB_per_s"],
            "qdmax_throughput_GB_per_s": qdmax["measured"]["throughput_GB_per_s"],
            "observed_qdmax_to_qd1_ratio": (
                qdmax["measured"]["throughput_GB_per_s"] /
                qd1["measured"]["throughput_GB_per_s"]
            ),
            "selected_effective_queues": queues,
            "candidate_range": [1, min(qdmax["lanes"], MAX_NVME_EFFECTIVE_QUEUES)],
            "selection_rule": "nearest integer to QDmax/QD1 throughput scaling in log space; ties choose smaller",
        }
        # One caller range occupies one direction-specific media queue.  The
        # geometric mean balances the QD1 estimate scaled by the queue count
        # against the measured high-QD aggregate anchor.
        bandwidth[name] = math.sqrt(
            queues * qd1["measured"]["throughput_GB_per_s"] *
            qdmax["measured"]["throughput_GB_per_s"]
        )
        small = sorted(
            (row for row in raw_rows
             if row["direction"] == direction and row["lanes"] == 1
             and row["block_bytes"] <= 65536),
            key=lambda row: row["block_bytes"],
        )
        residuals = [
            max(0.0, row["measured"]["latency_p50_ns"] -
                row["block_bytes"] /
                (bandwidth[name] / queues))
            for row in small
        ]
        latency[name] = median(residuals)
    config = dict(host_config)
    config.update({
        "kind": "nvme-ssd",
        "request_segment_bytes": segment_bytes,
        "media_read_queues": queue_fit["read"]["selected_effective_queues"],
        "media_write_queues": queue_fit["write"]["selected_effective_queues"],
        "media_read_latency_ns": latency["read"],
        "media_write_latency_ns": latency["write"],
        "media_read_bandwidth_GBps": bandwidth["read"],
        "media_write_bandwidth_GBps": bandwidth["write"],
    })
    media_only = dict(config)
    media_only.update({
        "controller_issue_ns": 0.0,
        "controller_processing_ns": 0.0,
        "m2s_bandwidth_GBps": NO_BOTTLENECK_GBPS,
        "s2m_bandwidth_GBps": NO_BOTTLENECK_GBPS,
    })
    return config, media_only, queue_fit


def fit_profiles(
    groups: Mapping[Tuple[Any, ...], Mapping[str, Any]],
    segment_bytes: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    host = fit_host(groups)
    nvme, media_only, queue_fit = fit_nvme(groups, host, segment_bytes)
    return {
        "host_dram": host,
        "nvme_ssd": nvme,
        "nvme_media_only": media_only,
    }, queue_fit


def config_args(config: Mapping[str, Any]) -> List[str]:
    mapping = (
        ("kind", "--kind"),
        ("page_size_bytes", "--page-bytes"),
        ("request_segment_bytes", "--segment-bytes"),
        ("media_channels", "--media-channels"),
        ("media_read_queues", "--media-read-queues"),
        ("media_write_queues", "--media-write-queues"),
        ("max_outstanding_requests", "--max-outstanding"),
        ("controller_issue_ns", "--controller-issue-ns"),
        ("controller_processing_ns", "--controller-processing-ns"),
        ("media_read_latency_ns", "--media-read-latency-ns"),
        ("media_write_latency_ns", "--media-write-latency-ns"),
        ("media_read_bandwidth_GBps", "--media-read-bandwidth-gbps"),
        ("media_write_bandwidth_GBps", "--media-write-bandwidth-gbps"),
        ("m2s_bandwidth_GBps", "--m2s-bandwidth-gbps"),
        ("s2m_bandwidth_GBps", "--s2m-bandwidth-gbps"),
        ("one_way_propagation_ns", "--one-way-propagation-ns"),
        ("command_bytes", "--command-bytes"),
        ("completion_bytes", "--completion-bytes"),
    )
    result = []
    for field, option in mapping:
        result.extend((option, str(config[field])))
    return result


def replay(
    probe: pathlib.Path,
    config: Mapping[str, Any],
    op: str,
    block_bytes: int,
    lanes: int,
    operations: int,
    schedule: str,
) -> Dict[str, Any]:
    argv = [str(probe)] + config_args(config) + [
        "--op", op,
        "--block-bytes", str(block_bytes),
        "--lanes", str(lanes),
        "--operations", str(operations),
        "--schedule", schedule,
    ]
    result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, check=False)
    require(result.returncode == 0,
            "external replay failed: {}".format(result.stderr.strip()))
    try:
        receipt = json.loads(result.stdout)
    except ValueError as error:
        raise CalibrationError("external replay emitted invalid JSON: {}".format(error))
    require(receipt.get("schema") == REPLAY_SCHEMA,
            "external replay schema changed")
    require(receipt.get("implementation") ==
            "ExternalBackingDevice::issue_contiguous_range",
            "replay did not use the production range mechanism")
    require(receipt.get("op") == op and receipt.get("schedule") == schedule,
            "replay operation/schedule diverged")
    require(receipt.get("block_bytes") == block_bytes and
            receipt.get("lanes") == lanes and receipt.get("operations") == operations,
            "replay transaction matrix diverged")
    resolved = receipt.get("config") or {}
    for field in (
        "page_size_bytes", "request_segment_bytes", "media_channels",
        "media_read_queues", "media_write_queues",
        "max_outstanding_requests",
    ):
        require(
            resolved.get(field) == config[field],
            "replay resolved {} diverged".format(field),
        )
    stats = receipt.get("stats") or {}
    require(
        stats.get("media_channels") ==
        config["media_channels"] * max(
            config["media_read_queues"], config["media_write_queues"]
        ) and
        stats.get("media_channels_per_queue") == config["media_channels"] and
        stats.get("media_read_queues") == config["media_read_queues"] and
        stats.get("media_write_queues") == config["media_write_queues"],
        "replay media queue geometry diverged",
    )
    require(stats.get("page_run_requests") == operations,
            "replay range counter diverged")
    require(stats.get("page_run_pages") == operations * (block_bytes // PAGE_BYTES),
            "replay page coverage diverged")
    segment_bytes = int(config["request_segment_bytes"])
    expected_segments = sum(
        ((index * block_bytes + block_bytes - 1) // segment_bytes) -
        ((index * block_bytes) // segment_bytes) + 1
        for index in range(operations)
    )
    require(stats.get("page_run_segments") == expected_segments,
            "replay transport-segment counter diverged")
    if op == "read":
        require(stats.get("read_requests") == expected_segments and
                stats.get("write_requests") == 0,
                "replay read transport-command counter diverged")
    else:
        require(stats.get("write_requests") == expected_segments and
                stats.get("read_requests") == 0,
                "replay write transport-command counter diverged")
    require(finite_positive(receipt.get("GB_per_s")) and
            finite_positive((receipt.get("latency_ns") or {}).get("p50")),
            "replay timing is invalid")
    # Keep the immutable receipt relocatable; config and workload arguments
    # remain complete while the build-directory prefix is intentionally not
    # embedded.
    receipt["argv"] = ["external_calibration_probe"] + argv[1:]
    return receipt


def percent_error(predicted: float, measured: float) -> float:
    return 100.0 * (predicted / measured - 1.0)


def cell_role(row: Mapping[str, Any]) -> str:
    """Return the role used by the two-job production fit."""
    block_bytes = row["block_bytes"]
    if row["path"] == "gpu_pinned_host":
        return "fit_input"
    if row["path"] == "ssd_direct":
        if row["lanes"] == 1 and block_bytes in (4096, 65536):
            return "fit_input"
        if block_bytes == PRIMARY_HOLDOUT_BYTES:
            return "fit_input"
    return "validation"


def replay_cell(
    probe: pathlib.Path,
    row: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    path_name = row["path"]
    if path_name == "gpu_pinned_host":
        config = profiles["host_dram"]
        throughput_receipt = replay(
            probe, config, DIRECTION_TO_OP[row["direction"]],
            row["block_bytes"], 1, row["operations"], "open-loop",
        )
        # The benchmark samples synchronized single-copy latency separately
        # from its queued throughput phase.
        latency_receipt = replay(
            probe, config, DIRECTION_TO_OP[row["direction"]],
            row["block_bytes"], 1, 1, "closed-loop",
        )
    else:
        config = profiles["nvme_media_only"] if path_name == "ssd_direct" else profiles["nvme_ssd"]
        throughput_receipt = replay(
            probe, config, DIRECTION_TO_OP[row["direction"]],
            row["block_bytes"], row["lanes"], row["operations"], "closed-loop",
        )
        latency_receipt = throughput_receipt
    measured = row["measured"]
    predicted_throughput = throughput_receipt["GB_per_s"]
    predicted_latency = latency_receipt["latency_ns"]["p50"]
    return {
        "path": path_name,
        "direction": row["direction"],
        "block_bytes": row["block_bytes"],
        "lanes": row["lanes"],
        "replicates": row["replicates"],
        "operations": row["operations"],
        "role": cell_role(row),
        "measured": measured,
        "predicted": {
            "throughput_GB_per_s": predicted_throughput,
            "latency_p50_ns": predicted_latency,
        },
        "error_percent": {
            "throughput_signed": percent_error(
                predicted_throughput, measured["throughput_GB_per_s"]),
            "throughput_absolute": abs(percent_error(
                predicted_throughput, measured["throughput_GB_per_s"])),
            "latency_signed": percent_error(
                predicted_latency, measured["latency_p50_ns"]),
            "latency_absolute": abs(percent_error(
                predicted_latency, measured["latency_p50_ns"])),
        },
        "replay": {
            "throughput": throughput_receipt,
            "latency": latency_receipt if latency_receipt is not throughput_receipt else "same_receipt",
        },
    }


def summarize_errors(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    require(bool(cells), "cannot summarize no calibration cells")
    throughput = [cell["error_percent"]["throughput_absolute"] for cell in cells]
    latency = [cell["error_percent"]["latency_absolute"] for cell in cells]
    return {
        "cell_count": len(cells),
        "throughput_absolute_error_percent": {
            "median": median(throughput),
            "mean": math.fsum(throughput) / len(throughput),
            "max": max(throughput),
        },
        "latency_absolute_error_percent": {
            "median": median(latency),
            "mean": math.fsum(latency) / len(latency),
            "max": max(latency),
        },
    }


def overlay_lines(config: Mapping[str, Any]) -> List[str]:
    fields = (
        ("kind", "external-backing-kind"),
        ("page_size_bytes", "external-backing-page-size"),
        ("request_segment_bytes", "external-backing-request-segment-bytes"),
        ("media_channels", "external-backing-media-channels"),
        ("media_read_queues", "external-backing-media-read-queues"),
        ("media_write_queues", "external-backing-media-write-queues"),
        ("max_outstanding_requests", "external-backing-max-outstanding-requests"),
        ("controller_issue_ns", "external-backing-controller-issue-ns"),
        ("controller_processing_ns", "external-backing-controller-processing-ns"),
        ("media_read_latency_ns", "external-backing-media-read-latency-ns"),
        ("media_write_latency_ns", "external-backing-media-write-latency-ns"),
        ("media_read_bandwidth_GBps", "external-backing-media-read-bw"),
        ("media_write_bandwidth_GBps", "external-backing-media-write-bw"),
        ("m2s_bandwidth_GBps", "external-backing-m2s-bw"),
        ("s2m_bandwidth_GBps", "external-backing-s2m-bw"),
        ("one_way_propagation_ns", "external-backing-one-way-propagation-ns"),
        ("command_bytes", "external-backing-command-bytes"),
        ("completion_bytes", "external-backing-completion-bytes"),
    )
    return ["{}={}".format(option, config[field]) for field, option in fields]


def markdown_report(artifact: Mapping[str, Any]) -> str:
    fit_jobs = "/".join(
        source["job_id"] for source in artifact["sources"]["fit_inputs"]
    )
    holdout_jobs = "/".join(
        source["job_id"] for source in artifact["sources"]["workload_holdout"]
    )
    nvme_leaf_count = artifact["segment_selection"]["nvme_leaf_count"]
    lines = [
        "# DANA A100 offload calibration",
        "",
        "Status: **{}**. Jobs {} are fit inputs; job {} is an independent run-level holdout. The quick smoke run is excluded.".format(
            artifact["status"], fit_jobs, holdout_jobs),
        "",
        "This is a workload-scoped platform anchor for one A100, CUDA-pinned host memory, and the compute-local md/NVMe stack with {} visible NVMe leaf device(s). Every prediction is produced by the compiled `ExternalBackingDevice::issue_contiguous_range` path.".format(nvme_leaf_count),
        "",
        "## Evidence split",
        "",
        "| Role | Slurm job | Host | CUDA slot | Raw JSON SHA-256 |",
        "|---|---:|---|---:|---|",
    ]
    for role in ("fit_inputs", "workload_holdout"):
        for source in artifact["sources"][role]:
            lines.append("| {} | {} | {} | {} | `{}` |".format(
                "fit" if role == "fit_inputs" else "holdout",
                source["job_id"], source["hostname"],
                source["cuda_visible_device"], source["sha256"]))
    read_queues = artifact["profiles"]["nvme_ssd"]["external_backing_config"]["media_read_queues"]
    write_queues = artifact["profiles"]["nvme_ssd"]["external_backing_config"]["media_write_queues"]
    lines.extend([
        "",
        "The SSD request segment is **{} bytes ({} KiB)**, selected from the minimum recorded mounted-stack/NVMe queue limit. The fit selected **{} read queue(s)** and **{} write queue(s)** from the QD8/QD1 scaling anchors without reading the holdout; these are effective caller-range service widths, not literal device queues.".format(
            artifact["segment_selection"]["request_segment_bytes"],
            artifact["segment_selection"]["request_segment_bytes"] // 1024,
            read_queues, write_queues),
        "",
        "## Fitted production fields",
        "",
        "| Field | host-dram | nvme-ssd |",
        "|---|---:|---:|",
    ])
    host = artifact["profiles"]["host_dram"]["external_backing_config"]
    nvme = artifact["profiles"]["nvme_ssd"]["external_backing_config"]
    for field in (
        "request_segment_bytes", "media_channels", "media_read_queues",
        "media_write_queues", "max_outstanding_requests",
        "controller_issue_ns", "controller_processing_ns",
        "media_read_latency_ns", "media_write_latency_ns",
        "media_read_bandwidth_GBps", "media_write_bandwidth_GBps",
        "m2s_bandwidth_GBps", "s2m_bandwidth_GBps",
    ):
        lines.append("| `{}` | {:.8g} | {:.8g} |".format(
            field, float(host[field]), float(nvme[field])))
    lines.extend([
        "",
        "Host fields use pinned-copy cells from the two fit jobs. SSD media fields use raw `O_DIRECT`; GPU↔SSD cells never enter the fit. The 16 MiB QD1/QD8 anchors determine directional media service, while small raw QD1 cells determine fixed media latency.",
        "",
        "## Fit-run replay and cross-size diagnostics",
        "",
        "| Path | Direction | Block | Lanes | Role | Measured GB/s | HBFSim GB/s | Error | Measured p50 | HBFSim p50 | Error |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cell in artifact["cells"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {:.4f} | {:.4f} | {:+.1f}% | {:.1f} µs | {:.1f} µs | {:+.1f}% |".format(
                cell["path"], cell["direction"], cell["block_bytes"],
                cell["lanes"], "fit input" if cell["role"] == "fit_input" else "validation",
                cell["measured"]["throughput_GB_per_s"],
                cell["predicted"]["throughput_GB_per_s"],
                cell["error_percent"]["throughput_signed"],
                cell["measured"]["latency_p50_ns"] / 1000.0,
                cell["predicted"]["latency_p50_ns"] / 1000.0,
                cell["error_percent"]["latency_signed"],
            )
        )
    lines.extend([
        "",
        "## Independent 16 MiB workload holdout (job {})".format(holdout_jobs),
        "",
        "| Path | Direction | Lanes | Measured GB/s | HBFSim GB/s | Error | Measured p50 | HBFSim p50 | Error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for cell in artifact["workload_holdout_16MiB"]:
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.4f} | {:+.1f}% | {:.1f} ms | {:.1f} ms | {:+.1f}% |".format(
                cell["path"], cell["direction"], cell["lanes"],
                cell["measured"]["throughput_GB_per_s"],
                cell["predicted"]["throughput_GB_per_s"],
                cell["error_percent"]["throughput_signed"],
                cell["measured"]["latency_p50_ns"] / 1.0e6,
                cell["predicted"]["latency_p50_ns"] / 1.0e6,
                cell["error_percent"]["latency_signed"],
            )
        )
    holdout = artifact["error_summary"]["workload_holdout_16MiB"]
    lines.extend([
        "",
        "The independent workload holdout has median absolute throughput/p50 errors of **{:.1f}% / {:.1f}%** and maxima of **{:.1f}% / {:.1f}%**.".format(
            holdout["throughput_absolute_error_percent"]["median"],
            holdout["latency_absolute_error_percent"]["median"],
            holdout["throughput_absolute_error_percent"]["max"],
            holdout["latency_absolute_error_percent"]["max"],
        ),
        "",
        "## Claim gates",
        "",
        "| Gate | Throughput threshold | p50 threshold | Eligible |",
        "|---|---:|---:|---|",
        "| independent 16 MiB holdout median | ≤ {holdout_median_throughput:.0f}% | ≤ {holdout_median_latency:.0f}% | {median} |".format(
            median="yes" if artifact["claim_gates"]["holdout_median"] else "no",
            **artifact["claim_gates"]["thresholds_percent"]),
        "| independent 16 MiB holdout per-cell max | ≤ {holdout_max_throughput:.0f}% | ≤ {holdout_max_latency:.0f}% | {maximum} |".format(
            maximum="yes" if artifact["claim_gates"]["holdout_all_cells"] else "no",
            **artifact["claim_gates"]["thresholds_percent"]),
        "| saturated SSD reads (raw + e2e, QD8) | ≤ {saturated_read_max_throughput:.0f}% | ≤ {saturated_read_max_latency:.0f}% | {read} |".format(
            read="yes" if artifact["claim_gates"]["saturated_read_qd8"] else "no",
            **artifact["claim_gates"]["thresholds_percent"]),
        "",
        "`workload_applicability_gate` is **{}**. Cross-size diagnostics remain limitations and do not grant a portable or paper-level hardware claim.".format(
            str(artifact["claim_gates"]["workload_applicability_gate"]).lower()),
        "",
        "## Limits",
        "",
    ])
    for item in artifact["limits"]:
        lines.append("- " + item)
    lines.append("")
    return "\n".join(lines)


def write_outputs(output: pathlib.Path, report: pathlib.Path, artifact: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text(markdown_report(artifact), encoding="utf-8")


def verify_published_overlays() -> None:
    repository = pathlib.Path(__file__).resolve().parents[3]
    artifact_path = (
        repository / "evidence/hardware/dana_a100_offload/calibration/"
        "dana-a100-offload-calibration.json"
    )
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CalibrationError(
            "cannot read published calibration artifact: {}".format(error)
        )
    bindings = {
        "host_dram": repository / (
            "configs/overlays/backing/calibrated/"
            "dana-a100-host-dram-timing.cfg"
        ),
        "nvme_ssd": repository / (
            "configs/overlays/backing/calibrated/"
            "dana-a100-nvme-ssd-timing.cfg"
        ),
    }
    for profile, path in bindings.items():
        try:
            actual = [
                line.partition("#")[0].strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.partition("#")[0].strip()
            ]
            expected = artifact["profiles"][profile]["overlay_lines"]
        except (OSError, KeyError, TypeError) as error:
            raise CalibrationError(
                "cannot verify published {} overlay: {}".format(
                    profile, error
                )
            )
        require(
            actual == expected,
            "published {} overlay diverges from calibration artifact".format(
                profile
            ),
        )


def run_self_test(probe: pathlib.Path) -> None:
    intercept, slope = linear_regression(((4096.0, 5204.8), (65536.0, 8276.8), (1048576.0, 57428.8)))
    require(abs(intercept - 5000.0) < 1.0e-6, "synthetic intercept recovery failed")
    require(abs(1.0 / slope - 20.0) < 1.0e-9, "synthetic bandwidth recovery failed")
    require(median((1.0, 9.0, 3.0)) == 3.0, "median aggregation failed")
    validate_source_identities((
        {"job_id": "1", "sha256": "a"},
        {"job_id": "2", "sha256": "b"},
        {"job_id": "3", "sha256": "c"},
    ))
    for duplicate_sources in (
        ({"job_id": "1", "sha256": "a"},
         {"job_id": "1", "sha256": "b"}),
        ({"job_id": "1", "sha256": "a"},
         {"job_id": "2", "sha256": "a"}),
    ):
        try:
            validate_source_identities(duplicate_sources)
        except CalibrationError:
            pass
        else:
            raise CalibrationError("duplicate source identity was accepted")
    require(
        cell_role({
            "path": "gpu_pinned_host", "block_bytes": 1048576, "lanes": 1,
        }) == "fit_input",
        "host fit-input role changed",
    )
    require(
        cell_role({
            "path": "ssd_direct", "block_bytes": 4096, "lanes": 8,
        }) == "validation",
        "raw SSD validation role changed",
    )
    require(
        cell_role({
            "path": "gpu_pinned_host_ssd_direct",
            "block_bytes": 1048576,
            "lanes": 1,
        }) == "validation",
        "end-to-end validation role changed",
    )
    require(
        cell_role({
            "path": "gpu_pinned_host_ssd_direct",
            "block_bytes": PRIMARY_HOLDOUT_BYTES,
            "lanes": 8,
        }) == "validation",
        "end-to-end fit-run validation role changed",
    )
    config = common_config("nvme-ssd", 131072)
    config.update({
        "media_read_queues": 2,
        "media_write_queues": 1,
        "media_read_bandwidth_GBps": 4.0,
        "media_write_bandwidth_GBps": 2.0,
        "media_read_latency_ns": 10000.0,
        "media_write_latency_ns": 15000.0,
    })
    receipt = replay(probe, config, "read", 1048576, 8, 16, "closed-loop")
    require(receipt["stats"]["page_run_segments"] == 128,
            "synthetic replay did not segment 1 MiB into 128 KiB commands")
    require(
        receipt["stats"]["active_media_resources"] == 2,
        "synthetic replay did not exercise both read queues",
    )
    require(
        select_effective_queue_count(3.0, 3.1, 8) == 1 and
        select_effective_queue_count(3.0, 5.9, 8) == 2,
        "effective queue-width selection changed",
    )
    verify_published_overlays()
    print("calibration self-test passed")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-input", action="append", default=[], type=pathlib.Path,
        help="passed raw benchmark JSON used for fitting; repeat twice",
    )
    parser.add_argument(
        "--holdout", action="append", default=[], type=pathlib.Path,
        help="independent passed raw benchmark JSON used only for validation",
    )
    parser.add_argument("--probe", type=pathlib.Path,
                        default=pathlib.Path("build/external_calibration_probe"),
                        help="compiled production-mechanism replay executable")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path(
                            "evidence/hardware/dana_a100_offload/calibration/dana-a100-offload-calibration.json"))
    parser.add_argument("--report", type=pathlib.Path,
                        default=pathlib.Path(
                            "evidence/hardware/dana_a100_offload/calibration/dana-a100-offload-calibration.md"))
    parser.add_argument("--allow-quick", action="store_true",
                        help="allow smoke data (never use for the published anchor)")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    options = parse_args(argv)
    probe = options.probe.resolve()
    require(probe.is_file() and os.access(str(probe), os.X_OK),
            "replay probe is missing; build target external_calibration_probe")
    if options.self_test:
        run_self_test(probe)
        return 0
    require(
        len(options.fit_input) == 2,
        "the published calibration requires exactly two --fit-input runs",
    )
    require(
        len(options.holdout) == 1,
        "the published calibration requires exactly one --holdout run",
    )
    resolved_paths = [
        path.resolve()
        for path in options.fit_input + options.holdout
    ]
    require(
        len(set(resolved_paths)) == len(resolved_paths),
        "fit and holdout paths must be disjoint",
    )

    def load_many(paths: Sequence[pathlib.Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        documents: List[Dict[str, Any]] = []
        metadata_rows: List[Dict[str, Any]] = []
        for path in paths:
            document, metadata = validate_document(
                path.resolve(), options.allow_quick
            )
            documents.append(document)
            metadata_rows.append(metadata)
        return documents, metadata_rows

    fit_documents, fit_sources = load_many(options.fit_input)
    holdout_documents, holdout_sources = load_many(options.holdout)
    all_sources = fit_sources + holdout_sources
    validate_source_identities(all_sources)
    quick_modes = {source["quick"] for source in all_sources}
    require(len(quick_modes) == 1, "quick and production evidence cannot be mixed")
    benchmark_digests = {
        source["benchmark_sha256"] for source in all_sources
    }
    require(len(benchmark_digests) == 1, "replicates used different benchmark code")
    signatures = {
        (
            source["hostname"],
            source["python_version"], source["torch_version"], source["gpu_name"],
            source["cuda_visible_device"],
            source["storage_filesystem"],
            tuple(source["segment_evidence"]["nvme_models"]),
            source["segment_evidence"]["nvme_leaf_count"],
            source["segment_evidence"]["request_segment_bytes"],
        )
        for source in all_sources
    }
    require(len(signatures) == 1, "replicate environment signatures differ")
    segment_values = {
        source["segment_evidence"]["request_segment_bytes"]
        for source in all_sources
    }
    require(len(segment_values) == 1, "replicates disagree on request segment")
    segment_bytes = segment_values.pop()
    nvme_leaf_counts = {
        source["segment_evidence"]["nvme_leaf_count"]
        for source in all_sources
    }
    require(len(nvme_leaf_counts) == 1,
            "replicates disagree on visible NVMe leaf count")
    nvme_leaf_count = nvme_leaf_counts.pop()
    cuda_visible_devices = {
        source["cuda_visible_device"] for source in all_sources
    }
    require(len(cuda_visible_devices) == 1,
            "replicates disagree on the physical CUDA device")
    cuda_visible_device = cuda_visible_devices.pop()
    fit_groups = aggregate_trials(fit_documents)
    holdout_groups = aggregate_trials(holdout_documents)

    profiles, queue_fit = fit_profiles(fit_groups, segment_bytes)
    cells = [
        replay_cell(probe, fit_groups[key], profiles)
        for key in sorted(fit_groups)
    ]
    workload_holdout = [
        replay_cell(probe, row, profiles)
        for _, row in sorted(holdout_groups.items())
        if row["block_bytes"] == PRIMARY_HOLDOUT_BYTES
    ]
    for cell in workload_holdout:
        cell["role"] = "workload_holdout_16MiB"
    require(
        len(workload_holdout) == 10,
        "independent workload holdout must contain ten 16 MiB cells",
    )

    public_profiles = {}
    for name in ("host_dram", "nvme_ssd"):
        public_profiles[name] = {
            "external_backing_config": profiles[name],
            "overlay_lines": overlay_lines(profiles[name]),
        }
    fit_input_cells = [cell for cell in cells if cell["role"] == "fit_input"]
    validation_cells = [cell for cell in cells if cell["role"] == "validation"]
    holdout_qd1 = [
        cell for cell in workload_holdout if cell["lanes"] == 1
    ]
    holdout_qd8 = [
        cell for cell in workload_holdout if cell["lanes"] == 8
    ]
    saturated_read = [
        cell for cell in workload_holdout
        if cell["lanes"] == 8 and
        cell["direction"] in ("restore_pread", "restore_ssd_to_gpu")
    ]
    require(
        len(fit_input_cells) == 16 and len(validation_cells) == 24,
        "fit/validation cell-role matrix changed",
    )
    require(
        len(saturated_read) == 2,
        "saturated read holdout must contain raw and end-to-end cells",
    )
    holdout_summary = summarize_errors(workload_holdout)
    saturated_read_summary = summarize_errors(saturated_read)
    thresholds = {
        "holdout_median_throughput": 15.0,
        "holdout_median_latency": 15.0,
        "holdout_max_throughput": 20.0,
        "holdout_max_latency": 25.0,
        "saturated_read_max_throughput": 15.0,
        "saturated_read_max_latency": 25.0,
    }
    median_gate = (
        holdout_summary["throughput_absolute_error_percent"]["median"] <=
        thresholds["holdout_median_throughput"] and
        holdout_summary["latency_absolute_error_percent"]["median"] <=
        thresholds["holdout_median_latency"]
    )
    all_cells_gate = (
        holdout_summary["throughput_absolute_error_percent"]["max"] <=
        thresholds["holdout_max_throughput"] and
        holdout_summary["latency_absolute_error_percent"]["max"] <=
        thresholds["holdout_max_latency"]
    )
    saturated_read_gate = (
        saturated_read_summary["throughput_absolute_error_percent"]["max"] <=
        thresholds["saturated_read_max_throughput"] and
        saturated_read_summary["latency_absolute_error_percent"]["max"] <=
        thresholds["saturated_read_max_latency"]
    )
    workload_applicability = (
        median_gate and all_cells_gate and saturated_read_gate
    )
    status = (
        "calibrated_with_limits"
        if workload_applicability else "diagnostic_only"
    )
    artifact = {
        "schema": CALIBRATION_SCHEMA,
        "status": status,
        "replicate_aggregation": {
            "fit_inputs": "median of two jobs by path/direction/block_bytes/lanes",
            "workload_holdout": "one untouched production job",
        },
        "sources": {
            "fit_inputs": fit_sources,
            "workload_holdout": holdout_sources,
        },
        "benchmark_sha256": next(iter(benchmark_digests)),
        "calibrator": {
            "path": "evidence/hardware/dana_a100_offload/calibrate.py",
            "sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "python_version": sys.version.split()[0],
        },
        "probe": {
            "path": "build/external_calibration_probe",
            "sha256": sha256_file(probe),
            "replay_schema": REPLAY_SCHEMA,
            "production_implementation": "ExternalBackingDevice::issue_contiguous_range",
        },
        "segment_selection": {
            "request_segment_bytes": segment_bytes,
            "nvme_leaf_count": nvme_leaf_count,
            "rule": "minimum positive max_sectors_kb/max_hw_sectors_kb across mounted stack and NVMe leaves",
            "per_source": [
                source["segment_evidence"] for source in all_sources
            ],
        },
        "fitting_protocol": {
            "fit_job_ids": [source["job_id"] for source in fit_sources],
            "independent_holdout_job_ids": [
                source["job_id"] for source in holdout_sources
            ],
            "workload_holdout_bytes": PRIMARY_HOLDOUT_BYTES,
            "host_dram": "directional pinned-copy wall service bandwidth and synchronized-copy fixed delay from fit jobs only",
            "nvme_ssd": "raw O_DIRECT 16 MiB QD1/QD8 direction-specific queue service; 4/64 KiB QD1 latency residual; host link composed unchanged",
            "e2e_cells_used_for_fit": False,
            "cell_roles": {
                "fit_input": "the 16 fit-job aggregate cells read by fit_host or fit_nvme",
                "validation": "the remaining 24 fit-job cells retained as cross-size/composition diagnostics",
                "workload_holdout_16MiB": "all ten 16 MiB cells from the untouched third job",
            },
            "free_fitted_fields": [
                "media_read_queues", "media_write_queues",
                "controller_issue_ns", "controller_processing_ns",
                "m2s_bandwidth_GBps", "s2m_bandwidth_GBps",
                "media_read_latency_ns", "media_write_latency_ns",
                "media_read_bandwidth_GBps", "media_write_bandwidth_GBps",
            ],
            "nvme_effective_queue_fit": queue_fit,
            "fixed_structural_fields": {
                "page_size_bytes": PAGE_BYTES,
                "host_request_segment_bytes": HOST_REQUEST_SEGMENT_BYTES,
                "nvme_request_segment_bytes": segment_bytes,
                "media_channels": 1,
                "host_media_read_queues": 1,
                "host_media_write_queues": 1,
                "max_outstanding_requests": 512,
                "command_bytes": 64,
                "completion_bytes": 16,
                "one_way_propagation_ns": 0.0,
            },
        },
        "profiles": public_profiles,
        "cells": cells,
        "workload_holdout_16MiB": workload_holdout,
        "error_summary": {
            "fit_input_cells": summarize_errors(fit_input_cells),
            "validation_cells": summarize_errors(validation_cells),
            "fit_run_all_cells": summarize_errors(cells),
            "workload_holdout_16MiB": holdout_summary,
            "workload_holdout_16MiB_qd1": summarize_errors(holdout_qd1),
            "workload_holdout_16MiB_qd8": summarize_errors(holdout_qd8),
            "workload_holdout_saturated_read_qd8": saturated_read_summary,
        },
        "contract_checks": {
            "evidence_schema_status_environment_path": True,
            "replicate_matrix_complete": True,
            "fit_and_holdout_paths_disjoint": True,
            "fit_and_holdout_job_ids_disjoint": True,
            "fit_and_holdout_raw_sha256_disjoint": True,
            "workload_holdout_not_used_for_fit": True,
            "range_page_segment_and_transport_command_counters_exact": True,
            "production_external_range_replay_used": True,
        },
        "claim_gates": {
            "thresholds_percent": thresholds,
            "holdout_median": median_gate,
            "holdout_all_cells": all_cells_gate,
            "saturated_read_qd8": saturated_read_gate,
            "workload_applicability_gate": workload_applicability,
            "unqualified_external_hardware_claim": False,
            "paper_hardware_performance_claim": False,
            "eligible_scope": (
                "DANA gpu-51 physical GPU {} A100 pinned-host/md-NVMe "
                "({} visible NVMe leaf) "
                "16 MiB QD1-QD8 including saturated read timing".format(
                    cuda_visible_device, nvme_leaf_count
                )
                if workload_applicability else "none"
            ),
        },
        "limits": [
            "The anchor covers DANA gpu-51 physical GPU {}, CUDA-pinned host memory, and its node-local XFS/LVM/md/NVMe stack with {} visible NVMe leaf device(s); it is not a portable machine constant.".format(cuda_visible_device, nvme_leaf_count),
            "SSD writes mean successful synchronous O_DIRECT pwritev completion, not power-loss durability.",
            "The fitted read/write queue counts are effective caller-range service widths selected from fit-run QD8/QD1 scaling on this md/NVMe stack; they are not asserted to be literal NVMe submission-queue counts.",
            "Small-block cells include Python, CUDA synchronization, syscall, and thread scheduling overheads that are outside a media-only device model.",
            "The claim gate is the untouched third job's 16 MiB QD1/QD8 matrix. Errors at 4 KiB, 64 KiB, and 1 MiB remain visible cross-size diagnostics and are outside the eligible workload scope.",
            "Host DRAM denotes the measured GPU-to-CUDA-pinned-host path, not arbitrary pageable CPU memory and not CXL memory.",
            "Media bandwidth in the host-only profile is a numerical non-bottleneck sentinel; measured transfer is represented exactly once by the directional host link.",
            "The one-TiB capacity in replay configs is only a nonallocating address-space sentinel. This benchmark calibrates transfer timing, not usable DRAM or SSD capacity, and the generated overlay lines intentionally omit capacity.",
        ],
    }
    write_outputs(options.output, options.report, artifact)
    print("wrote {}".format(options.output))
    print("wrote {}".format(options.report))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except CalibrationError as error:
        print("calibration failed: {}".format(error), file=sys.stderr)
        sys.exit(1)
