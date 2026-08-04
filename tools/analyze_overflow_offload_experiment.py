#!/usr/bin/env python3
"""Validate and summarize the real-capacity HBM overflow experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.certificate import (  # noqa: E402
    CertificateError,
    verify_certificate,
)

SCHEMA = "hbfsim.capacity-overflow-experiment"
VERSION = 7
SERVICE_CURVE_WINDOWS = [1, 4, 16, 64, 256]
LIFECYCLE_RESTORE_COUNTS = [1, 2, 4, 8, 16, 32]
SERVICE_CURVE_RESIDENT_PAGES = 4096
SERVICE_CURVE_OFFLOAD_PAGES = 1024
SERVICE_CURVE_READ_BUFFER_PAGES = 256
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
INTERPRETATION = (
    "saturated append-only KV/write-back state; not read-only model weights"
)
POLICY = {
    "name": "fifo-write-back-with-reserved-read-buffer",
    "hbf_completion": "payload-page-program-complete",
    "readback": "backing-read-to-HBM-DMA-buffer-then-HBM-read",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def number(value: Any, path: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{path} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{path} must be finite")
    return result


def integer(value: Any, path: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{path} must be an integer",
    )
    require(value >= 0, f"{path} must be non-negative")
    return value


def close(lhs: float, rhs: float) -> bool:
    return math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-6)


def validate_quantiles(value: Any, path: str) -> tuple[float, float, float]:
    require(isinstance(value, dict), f"{path} must be an object")
    quantiles = tuple(
        number(value.get(key), f"{path}.{key}")
        for key in ("p50", "p95", "p99")
    )
    require(
        0 < quantiles[0] <= quantiles[1] <= quantiles[2],
        f"{path}: invalid latency quantiles",
    )
    return quantiles


def float_bits(value: Any, path: str) -> str:
    encoded = struct.pack(">d", number(value, path))
    return f"{struct.unpack('>Q', encoded)[0]:016x}"


def canonical_sample_record(summary: dict[str, Any], path: str) -> str:
    backing = summary.get("backing")
    require(
        backing in {"hbf", "external"},
        f"{path}.backing is invalid",
    )
    integer_fields = (
        "hbm_data_pages",
        "hbm_read_buffer_pages",
        "transfer_batch_pages",
        "written_pages",
        "offload_pages",
        "readback_pages",
        "offload_bytes",
        "readback_bytes",
    )
    suffix_integer_fields = (
        "hbm_read_bytes",
        "hbm_write_bytes",
        "backing_read_bytes",
        "backing_write_bytes",
    )
    parts = [
        "hbfsim.capacity-overflow-sample.v1",
        f"backing={backing}",
    ]
    parts.extend(
        f"{key}={integer(summary.get(key), f'{path}.{key}')}"
        for key in integer_fields
    )
    parts.extend(
        (
            "fill_elapsed_bits="
            + float_bits(summary.get("fill_elapsed_ns"), f"{path}.fill_elapsed_ns"),
            "offload_elapsed_bits="
            + float_bits(
                summary.get("offload_elapsed_ns"),
                f"{path}.offload_elapsed_ns",
            ),
            "read_elapsed_bits="
            + float_bits(summary.get("read_elapsed_ns"), f"{path}.read_elapsed_ns"),
            "drain_tail_bits="
            + float_bits(summary.get("drain_tail_ns"), f"{path}.drain_tail_ns"),
        )
    )
    parts.extend(
        f"{key}={integer(summary.get(key), f'{path}.{key}')}"
        for key in suffix_integer_fields
    )
    return "|".join(parts)


def validate_sample_digest(value: Any, path: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{path} must be an object")
    summary = value.get("summary")
    digest = value.get("digest")
    require(isinstance(summary, dict), f"{path}.summary must be an object")
    require(isinstance(digest, dict), f"{path}.digest must be an object")
    canonical = canonical_sample_record(summary, f"{path}.summary")
    recorded = digest.get("canonical_record")
    recorded_digest = digest.get("value")
    require(
        digest.get("algorithm") == "sha256"
        and recorded == canonical
        and isinstance(recorded_digest, str)
        and DIGEST_RE.fullmatch(recorded_digest) is not None
        and hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        == recorded_digest,
        f"{path}: sample summary digest is invalid",
    )
    return summary


def mapping_programs(logical_pages: int, stacks: int, entries: int) -> int:
    total = 0
    for stack in range(stacks):
        local_pages = logical_pages // stacks + int(
            stack < logical_pages % stacks
        )
        total += (local_pages + entries - 1) // entries
    return total


def external_payload_ceiling_GBps(
    profile: dict[str, Any],
    page_bytes: int,
    operation: str,
) -> float:
    """Return a conservative page-payload ceiling for the explicit pipeline.

    Bytes/ns is numerically equal to decimal GB/s.  Every bound is expressed
    as useful page payload so protocol-only resources can be compared without
    silently treating wire bytes as logical bytes.
    """
    require(operation in {"read", "write"}, "unsupported external operation")
    read = operation == "read"
    channels = integer(profile.get("media_channels"), "external.media_channels")
    outstanding = integer(
        profile.get("max_outstanding_requests"),
        "external.max_outstanding_requests",
    )
    controller_issue = number(
        profile.get("controller_issue_ns"),
        "external.controller_issue_ns",
    )
    controller_processing = number(
        profile.get("controller_processing_ns"),
        "external.controller_processing_ns",
    )
    media_latency = number(
        profile.get(
            "media_read_latency_ns" if read else "media_write_latency_ns"
        ),
        f"external.media_{operation}_latency_ns",
    )
    media_bandwidth = number(
        profile.get("media_read_GBps" if read else "media_write_GBps"),
        f"external.media_{operation}_GBps",
    )
    m2s_bandwidth = number(profile.get("m2s_GBps"), "external.m2s_GBps")
    s2m_bandwidth = number(profile.get("s2m_GBps"), "external.s2m_GBps")
    propagation = number(
        profile.get("one_way_propagation_ns"),
        "external.one_way_propagation_ns",
    )
    command_bytes = integer(
        profile.get("command_bytes"), "external.command_bytes"
    )
    completion_bytes = integer(
        profile.get("completion_bytes"), "external.completion_bytes"
    )
    m2s_wire_bytes = command_bytes + (0 if read else page_bytes)
    s2m_wire_bytes = completion_bytes + (page_bytes if read else 0)
    controller_ceiling = (
        math.inf if controller_issue == 0 else page_bytes / controller_issue
    )
    m2s_payload_ceiling = (
        m2s_bandwidth * page_bytes / m2s_wire_bytes
    )
    s2m_payload_ceiling = (
        s2m_bandwidth * page_bytes / s2m_wire_bytes
    )
    isolated_latency = (
        m2s_wire_bytes / m2s_bandwidth
        + propagation
        + controller_issue
        + controller_processing
        + media_latency
        + page_bytes / (media_bandwidth / channels)
        + s2m_wire_bytes / s2m_bandwidth
        + propagation
    )
    outstanding_ceiling = outstanding * page_bytes / isolated_latency
    return min(
        media_bandwidth,
        controller_ceiling,
        m2s_payload_ceiling,
        s2m_payload_ceiling,
        outstanding_ceiling,
    )


def validate_generator(document: dict[str, Any]) -> None:
    generator = document.get("generator")
    require(isinstance(generator, dict), "generator must be an object")
    for key in (
        "version",
        "git_commit",
        "build_type",
        "compiler_id",
        "compiler_version",
    ):
        require(
            isinstance(generator.get(key), str) and generator[key],
            f"generator.{key} must be a non-empty string",
        )
    require(
        isinstance(generator.get("git_dirty"), bool),
        "generator.git_dirty must be boolean",
    )
    executable = generator.get("executable")
    require(
        isinstance(executable, dict)
        and executable.get("algorithm") == "sha256"
        and isinstance(executable.get("value"), str)
        and DIGEST_RE.fullmatch(executable["value"]) is not None
        and integer(
            executable.get("bytes"),
            "generator.executable.bytes",
        )
        > 0,
        "generator executable digest is missing or invalid",
    )


def verify_executable_digest(
    document: dict[str, Any],
    executable_path: Path,
) -> None:
    require(
        executable_path.is_file(),
        f"experiment executable does not exist: {executable_path}",
    )
    digest = hashlib.sha256()
    with executable_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    executable = document["generator"]["executable"]
    require(
        executable["bytes"] == executable_path.stat().st_size
        and executable["value"] == digest.hexdigest(),
        "generator executable digest does not match the supplied binary",
    )


def validate(
    document: dict[str, Any],
    *,
    expected_validation: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int | float]]:
    require(
        document.get("schema") == {"name": SCHEMA, "version": VERSION},
        f"unsupported schema: {document.get('schema')!r}",
    )
    validate_generator(document)
    if expected_validation is None:
        expected_validation = {
            "status": "exploratory_unattached",
            "certificate": None,
        }
    require(
        document.get("validation") == expected_validation,
        "direct experiment artifact must be explicitly exploratory and unattached",
    )
    require(
        document.get("provenance")
        == {
            "capacity": "exact",
            "traffic": "exact",
            "base_and_doubled_samples": "measured_sample",
            "target_timing": "projected",
            "service_curves": "measured_sample",
            "layer_lifecycle": "derived",
        },
        "numeric provenance classification is missing or unsupported",
    )
    limitations = document.get("limitations")
    require(
        isinstance(limitations, list)
        and all(isinstance(item, str) and item for item in limitations)
        and any("FTL" in item and "GC" in item for item in limitations)
        and any("not calibrated" in item for item in limitations),
        "experiment limitations are missing the external-model boundary",
    )
    require(
        document.get("interpretation") == INTERPRETATION,
        "experiment interpretation is missing or unsupported",
    )
    require(
        document.get("policy") == POLICY,
        "experiment policy is missing or unsupported",
    )

    method = document.get("method")
    require(isinstance(method, dict), "method must be an object")
    require(
        method.get("name") == "real-capacity-converged-periodic-replay"
        and method.get("capacity_accounting") == "exact"
        and method.get("traffic_accounting") == "exact"
        and method.get("timing")
        == "extrapolated-from-page-exact-batches",
        "unsupported scale-simulation method",
    )
    require(
        method.get("layer_timing")
        == (
            "extrapolated-from-original-slot-restoring-"
            "page-exact-batches"
        ),
        "unsupported scale-simulation method",
    )
    threshold = number(
        method.get("convergence_threshold"),
        "method.convergence_threshold",
    )
    base_batches = integer(
        method.get("base_sample_batches"),
        "method.base_sample_batches",
    )
    doubled_batches = integer(
        method.get("doubled_sample_batches"),
        "method.doubled_sample_batches",
    )
    base_sample_resident_pages = integer(
        method.get("base_sample_resident_pages"),
        "method.base_sample_resident_pages",
    )
    doubled_sample_resident_pages = integer(
        method.get("doubled_sample_resident_pages"),
        "method.doubled_sample_resident_pages",
    )
    require(
        0 < threshold <= 0.1
        and base_batches > 0
        and doubled_batches == 2 * base_batches
        and base_sample_resident_pages > 0
        and doubled_sample_resident_pages
        == 2 * base_sample_resident_pages,
        "invalid convergence method geometry",
    )

    profile = document.get("profile")
    workload = document.get("workload")
    require(isinstance(profile, dict), "profile must be an object")
    require(isinstance(workload, dict), "workload must be an object")
    for key in ("hbm", "hbf", "base_die_link", "cxl_memory", "nvme_ssd"):
        require(isinstance(profile.get(key), dict), f"profile.{key} is missing")

    page = integer(profile.get("page_size_bytes"), "profile.page_size_bytes")
    hbm_capacity = integer(
        workload.get("hbm_capacity_bytes"),
        "workload.hbm_capacity_bytes",
    )
    hbm_data = integer(
        workload.get("hbm_data_bytes"),
        "workload.hbm_data_bytes",
    )
    read_buffer = integer(
        workload.get("read_buffer_bytes"),
        "workload.read_buffer_bytes",
    )
    total_write = integer(
        workload.get("total_write_bytes"),
        "workload.total_write_bytes",
    )
    total_write_pages = integer(
        workload.get("total_write_pages"),
        "workload.total_write_pages",
    )
    offload_bytes = integer(
        workload.get("offload_bytes"),
        "workload.offload_bytes",
    )
    offload_pages = integer(
        workload.get("offload_pages"),
        "workload.offload_pages",
    )
    batch_pages = integer(
        workload.get("batch_pages"),
        "workload.batch_pages",
    )
    batch_bytes = integer(
        workload.get("batch_bytes"),
        "workload.batch_bytes",
    )
    target_batches = integer(
        workload.get("target_batches"),
        "workload.target_batches",
    )
    layer_contract = document.get("layer_round_trip")
    require(
        isinstance(layer_contract, dict),
        "layer_round_trip must be an object",
    )
    layer_logical_bytes = integer(
        layer_contract.get("logical_bytes"),
        "layer_round_trip.logical_bytes",
    )
    layer_bytes = integer(
        layer_contract.get("transfer_bytes"),
        "layer_round_trip.transfer_bytes",
    )
    layer_padding_bytes = integer(
        layer_contract.get("padding_bytes"),
        "layer_round_trip.padding_bytes",
    )
    layer_pages = integer(
        layer_contract.get("pages"),
        "layer_round_trip.pages",
    )
    layer_batches = integer(
        layer_contract.get("batches"),
        "layer_round_trip.batches",
    )
    require(
        page > 0
        and hbm_capacity > 0
        and 0 < read_buffer < hbm_capacity
        and total_write > hbm_capacity
        and batch_pages > 0
        and base_sample_resident_pages >= batch_pages
        and doubled_sample_resident_pages * page <= hbm_data
        and doubled_batches <= target_batches,
        "real-capacity workload geometry is invalid",
    )
    require(
        hbm_data == hbm_capacity - read_buffer
        and total_write_pages * page == total_write
        and offload_bytes == total_write - hbm_data
        and offload_pages * page == offload_bytes
        and batch_bytes == batch_pages * page
        and offload_pages == target_batches * batch_pages
        and 0 < layer_logical_bytes <= layer_bytes <= offload_bytes
        and layer_bytes <= hbm_data
        and layer_bytes == layer_pages * page
        and layer_pages == layer_batches * batch_pages
        and layer_bytes == layer_logical_bytes + layer_padding_bytes
        and doubled_batches <= layer_batches
        and read_buffer >= batch_bytes
        and number(workload.get("interarrival_ns"), "workload.interarrival")
        == 0,
        "real-capacity workload accounting does not conserve",
    )
    require(
        layer_contract.get("name") == "configurable-layer-payload"
        and layer_contract.get("sizing_basis")
        in {
            "llama-3.1-405b-8bit-transformer-block",
            "user-configured",
        }
        and layer_contract.get("payload_semantics")
        == "size-reference-not-steady-state-weight-write"
        and layer_contract.get("immediate_readback") is True
        and layer_contract.get("destination") == "original-hbm-slots"
        and layer_contract.get("completion")
        == "last-page-restored-and-foreground-HBM-read-complete"
        and layer_contract.get("mapping_checkpoint_drain_included")
        is False,
        "layer round-trip contract is unsupported",
    )
    layer_reference = layer_contract.get("default_reference")
    require(
        isinstance(layer_reference, dict)
        and layer_reference.get("model") == "Llama-3.1-405B"
        and integer(
            layer_reference.get("precision_bytes_per_parameter"),
            "layer_round_trip.default_reference.precision",
        )
        == 1
        and integer(
            layer_reference.get("logical_bytes"),
            "layer_round_trip.default_reference.logical_bytes",
        )
        == 3_187_703_808
        and integer(
            layer_reference.get("transfer_alignment_bytes"),
            "layer_round_trip.default_reference.alignment",
        )
        == batch_bytes
        and layer_reference.get("source_url")
        == (
            "https://huggingface.co/meta-llama/"
            "Llama-3.1-405B-Instruct/blob/main/config.json"
        ),
        "default layer sizing reference is missing or unsupported",
    )
    if (
        layer_contract.get("sizing_basis")
        == "llama-3.1-405b-8bit-transformer-block"
    ):
        require(
            layer_logical_bytes == 3_187_703_808
            and layer_bytes == 3041 * 2**20
            and layer_padding_bytes
            == layer_bytes - layer_logical_bytes,
            "default Llama layer sizing drifted",
        )
    hbm_profile = profile["hbm"]
    require(
        integer(
            hbm_profile.get("capacity_bytes"),
            "profile.hbm.capacity_bytes",
        )
        == hbm_capacity,
        "profile HBM capacity differs from workload HBM capacity",
    )
    hbm_pseudo_channels = integer(
        hbm_profile.get("pseudo_channels_per_channel"),
        "profile.hbm.pseudo_channels_per_channel",
    )
    hbm_channel_width_bits = integer(
        hbm_profile.get("channel_width_bits"),
        "profile.hbm.channel_width_bits",
    )
    hbm_burst_length = integer(
        hbm_profile.get("burst_length"),
        "profile.hbm.burst_length",
    )
    require(
        hbm_pseudo_channels > 0
        and hbm_burst_length > 0
        and hbm_channel_width_bits % hbm_pseudo_channels == 0
        and (hbm_channel_width_bits // hbm_pseudo_channels) % 8 == 0,
        "HBM burst geometry is invalid",
    )
    hbm_burst_bytes = (
        hbm_channel_width_bits
        // hbm_pseudo_channels
        // 8
        * hbm_burst_length
    )
    hbm_total_traffic = total_write + 3 * offload_bytes
    require(
        hbm_burst_bytes > 0
        and page % hbm_burst_bytes == 0
        and hbm_total_traffic % hbm_burst_bytes == 0,
        "HBM traffic does not divide into exact bursts",
    )
    hbm_burst_children = hbm_total_traffic // hbm_burst_bytes

    hbf_profile = profile["hbf"]
    hbf_stacks = integer(hbf_profile.get("stacks"), "profile.hbf.stacks")
    hbf_channels = integer(
        hbf_profile.get("channels_per_stack"),
        "profile.hbf.channels_per_stack",
    )
    hbf_dies = integer(
        hbf_profile.get("dies_per_channel"),
        "profile.hbf.dies_per_channel",
    )
    hbf_planes_per_die = integer(
        hbf_profile.get("planes_per_die"),
        "profile.hbf.planes_per_die",
    )
    mapping_entries = integer(
        hbf_profile.get("mapping_entries_per_page"),
        "profile.hbf.mapping_entries_per_page",
    )
    hbf_planes = (
        hbf_stacks * hbf_channels * hbf_dies * hbf_planes_per_die
    )
    require(
        hbf_stacks > 0
        and hbf_planes > 0
        and mapping_entries > 0
        and integer(
            hbf_profile.get("page_size_bytes"),
            "profile.hbf.page_size_bytes",
        )
        == page
        and integer(
            hbf_profile.get("target_capacity_bytes"),
            "profile.hbf.target_capacity_bytes",
        )
        >= 2 * offload_bytes
        and integer(
            hbf_profile.get("sample_capacity_bytes"),
            "profile.hbf.sample_capacity_bytes",
        )
        >= 2 * doubled_batches * batch_bytes,
        "HBF target/sample capacity contract is invalid",
    )
    base_link = profile["base_die_link"]
    require(
        number(
            base_link.get("read_GBps_per_stack"),
            "profile.base_die_link.read_GBps_per_stack",
        )
        > 0
        and number(
            base_link.get("write_GBps_per_stack"),
            "profile.base_die_link.write_GBps_per_stack",
        )
        > 0
        and number(
            base_link.get("latency_ns"),
            "profile.base_die_link.latency_ns",
        )
        >= 0,
        "base-die link profile is invalid",
    )
    require(
        base_link.get("evidence_grade") == "literature_derived"
        and base_link.get("source_id") == "kaist_tcad_2026_0452"
        and base_link.get("source_scope")
        == "architecture-model-input-not-vendor-measurement"
        and base_link.get("sandisk_public_d2d_measurement_available")
        is False
        and close(
            number(
                base_link.get(
                    "sandisk_public_hbf_read_GBps_per_stack"
                ),
                "profile.base_die_link.sandisk_public_hbf_read",
            ),
            1600.0,
        ),
        "base-die link evidence boundary is missing or unsupported",
    )
    for profile_name in ("cxl_memory", "nvme_ssd"):
        external_profile = profile[profile_name]
        expected_kind = profile_name.replace("_", "-")
        require(
            external_profile.get("kind") == expected_kind
            and integer(
                external_profile.get("capacity_bytes"),
                f"profile.{profile_name}.capacity_bytes",
            )
            >= offload_bytes
            and integer(
                external_profile.get("page_size_bytes"),
                f"profile.{profile_name}.page_size_bytes",
            )
            == page
            and integer(
                external_profile.get("media_channels"),
                f"profile.{profile_name}.media_channels",
            )
            > 0
            and integer(
                external_profile.get("max_outstanding_requests"),
                f"profile.{profile_name}.max_outstanding_requests",
            )
            > 0
            and number(
                external_profile.get("controller_issue_ns"),
                f"profile.{profile_name}.controller_issue_ns",
            )
            >= 0
            and number(
                external_profile.get("controller_processing_ns"),
                f"profile.{profile_name}.controller_processing_ns",
            )
            >= 0
            and number(
                external_profile.get("media_read_latency_ns"),
                f"profile.{profile_name}.media_read_latency_ns",
            )
            >= 0
            and number(
                external_profile.get("media_write_latency_ns"),
                f"profile.{profile_name}.media_write_latency_ns",
            )
            >= 0
            and number(
                external_profile.get("media_read_GBps"),
                f"profile.{profile_name}.media_read_GBps",
            )
            > 0
            and number(
                external_profile.get("media_write_GBps"),
                f"profile.{profile_name}.media_write_GBps",
            )
            > 0
            and number(
                external_profile.get("m2s_GBps"),
                f"profile.{profile_name}.m2s_GBps",
            )
            > 0
            and number(
                external_profile.get("s2m_GBps"),
                f"profile.{profile_name}.s2m_GBps",
            )
            > 0
            and number(
                external_profile.get("one_way_propagation_ns"),
                f"profile.{profile_name}.one_way_propagation_ns",
            )
            >= 0
            and integer(
                external_profile.get("command_bytes"),
                f"profile.{profile_name}.command_bytes",
            )
            > 0
            and integer(
                external_profile.get("completion_bytes"),
                f"profile.{profile_name}.completion_bytes",
            )
            > 0,
            f"{profile_name}: invalid external-backing profile",
        )

    cases = document.get("cases")
    require(isinstance(cases, list) and len(cases) == 3, "expected three cases")
    by_name: dict[str, dict[str, Any]] = {}
    expected_hbm_write = total_write + offload_bytes
    expected_hbm_read = 2 * offload_bytes
    expected_hbm_user = total_write_pages + offload_pages
    expected_hbm_background = 2 * offload_pages
    expected_mapping = mapping_programs(
        offload_pages,
        hbf_stacks,
        mapping_entries,
    )
    expected_mapping_rounds = (
        expected_mapping + hbf_planes - 1
    ) // hbf_planes
    common_fill: float | None = None
    common_fill_sample: tuple[float, float, float] | None = None

    for index, case in enumerate(cases):
        path = f"cases[{index}]"
        require(isinstance(case, dict), f"{path} must be an object")
        name = case.get("name")
        require(
            isinstance(name, str) and name not in by_name,
            f"{path}.name must be unique",
        )
        by_name[name] = case
        require(case.get("conservation") == "PASS", f"{name}: conservation failed")

        capacity = case.get("capacity")
        traffic = case.get("traffic")
        sampling = case.get("sampling")
        write = case.get("write_phase")
        read = case.get("read_phase")
        layer_case = case.get("layer_round_trip")
        timeline = case.get("timeline")
        hbm = case.get("hbm")
        for key, value in {
            "capacity": capacity,
            "traffic": traffic,
            "sampling": sampling,
            "write_phase": write,
            "read_phase": read,
            "layer_round_trip": layer_case,
            "timeline": timeline,
            "hbm": hbm,
        }.items():
            require(isinstance(value, dict), f"{name}.{key} must be an object")

        require(
            integer(capacity.get("hbm_total_bytes"), f"{name}.hbm_total")
            == hbm_capacity
            and integer(capacity.get("hbm_data_bytes"), f"{name}.hbm_data")
            == hbm_data
            and integer(
                capacity.get("hbm_read_buffer_bytes"),
                f"{name}.hbm_read_buffer",
            )
            == read_buffer,
            f"{name}: target capacity does not conserve",
        )
        require(
            integer(traffic.get("written_pages"), f"{name}.written_pages")
            == total_write_pages
            and integer(
                traffic.get("offload_pages"),
                f"{name}.offload_pages",
            )
            == offload_pages
            and integer(
                traffic.get("readback_pages"),
                f"{name}.readback_pages",
            )
            == offload_pages
            and integer(traffic.get("offload_bytes"), f"{name}.offload_bytes")
            == offload_bytes
            and integer(
                traffic.get("readback_bytes"),
                f"{name}.readback_bytes",
            )
            == offload_bytes,
            f"{name}: target traffic does not conserve",
        )

        raw_samples = case.get("raw_samples")
        require(
            isinstance(raw_samples, dict)
            and set(raw_samples) == {"base", "doubled"},
            f"{name}: raw sample summaries are missing",
        )
        for sample_name, sample_batches, resident_pages in (
            ("base", base_batches, base_sample_resident_pages),
            (
                "doubled",
                doubled_batches,
                doubled_sample_resident_pages,
            ),
        ):
            summary = validate_sample_digest(
                raw_samples.get(sample_name),
                f"{name}.raw_samples.{sample_name}",
            )
            sample_offload_pages = sample_batches * batch_pages
            sample_written_pages = resident_pages + sample_offload_pages
            sample_offload_bytes = sample_offload_pages * page
            require(
                summary.get("backing")
                == ("hbf" if name == "hbf" else "external")
                and integer(
                    summary.get("hbm_data_pages"),
                    f"{name}.{sample_name}.hbm_data_pages",
                )
                == resident_pages
                and integer(
                    summary.get("hbm_read_buffer_pages"),
                    f"{name}.{sample_name}.hbm_read_buffer_pages",
                )
                == read_buffer // page
                and integer(
                    summary.get("transfer_batch_pages"),
                    f"{name}.{sample_name}.transfer_batch_pages",
                )
                == batch_pages
                and integer(
                    summary.get("written_pages"),
                    f"{name}.{sample_name}.written_pages",
                )
                == sample_written_pages
                and integer(
                    summary.get("offload_pages"),
                    f"{name}.{sample_name}.offload_pages",
                )
                == sample_offload_pages
                and integer(
                    summary.get("readback_pages"),
                    f"{name}.{sample_name}.readback_pages",
                )
                == sample_offload_pages
                and integer(
                    summary.get("offload_bytes"),
                    f"{name}.{sample_name}.offload_bytes",
                )
                == sample_offload_bytes
                and integer(
                    summary.get("readback_bytes"),
                    f"{name}.{sample_name}.readback_bytes",
                )
                == sample_offload_bytes
                and number(
                    summary.get("fill_elapsed_ns"),
                    f"{name}.{sample_name}.fill_elapsed_ns",
                )
                > 0
                and number(
                    summary.get("offload_elapsed_ns"),
                    f"{name}.{sample_name}.offload_elapsed_ns",
                )
                > 0
                and number(
                    summary.get("read_elapsed_ns"),
                    f"{name}.{sample_name}.read_elapsed_ns",
                )
                > 0
                and integer(
                    summary.get("hbm_read_bytes"),
                    f"{name}.{sample_name}.hbm_read_bytes",
                )
                == 2 * sample_offload_bytes
                and integer(
                    summary.get("hbm_write_bytes"),
                    f"{name}.{sample_name}.hbm_write_bytes",
                )
                == (sample_written_pages + sample_offload_pages) * page
                and integer(
                    summary.get("backing_read_bytes"),
                    f"{name}.{sample_name}.backing_read_bytes",
                )
                == sample_offload_bytes
                and integer(
                    summary.get("backing_write_bytes"),
                    f"{name}.{sample_name}.backing_write_bytes",
                )
                == sample_offload_bytes,
                f"{name}: {sample_name} raw sample does not conserve",
            )

        require(
            integer(sampling.get("base_batches"), f"{name}.sample.base")
            == base_batches
            and integer(
                sampling.get("doubled_batches"),
                f"{name}.sample.doubled",
            )
            == doubled_batches
            and integer(
                sampling.get("base_offload_pages"),
                f"{name}.sample.base_pages",
            )
            == base_batches * batch_pages
            and integer(
                sampling.get("doubled_offload_pages"),
                f"{name}.sample.doubled_pages",
            )
            == doubled_batches * batch_pages
            and sampling.get("convergence") == "PASS",
            f"{name}: sample window does not match method",
        )
        offload_base = number(
            sampling.get("offload_ns_per_batch_base"),
            f"{name}.sample.offload_base",
        )
        offload_doubled = number(
            sampling.get("offload_ns_per_batch_doubled"),
            f"{name}.sample.offload_doubled",
        )
        read_base = number(
            sampling.get("read_ns_per_batch_base"),
            f"{name}.sample.read_base",
        )
        read_doubled = number(
            sampling.get("read_ns_per_batch_doubled"),
            f"{name}.sample.read_doubled",
        )
        offload_drift = number(
            sampling.get("offload_relative_drift"),
            f"{name}.sample.offload_drift",
        )
        read_drift = number(
            sampling.get("read_relative_drift"),
            f"{name}.sample.read_drift",
        )
        fill_base = number(
            sampling.get("fill_ns_per_byte_base"),
            f"{name}.sample.fill_base",
        )
        fill_doubled = number(
            sampling.get("fill_ns_per_byte_doubled"),
            f"{name}.sample.fill_doubled",
        )
        fill_drift = number(
            sampling.get("fill_relative_drift"),
            f"{name}.sample.fill_drift",
        )
        require(
            offload_base > 0
            and offload_doubled > 0
            and read_base > 0
            and read_doubled > 0
            and fill_base > 0
            and fill_doubled > 0
            and close(
                offload_drift,
                abs(offload_doubled - offload_base) / offload_doubled,
            )
            and close(
                read_drift,
                abs(read_doubled - read_base) / read_doubled,
            )
            and close(
                fill_drift,
                abs(fill_doubled - fill_base) / fill_doubled,
            )
            and 0 <= offload_drift <= threshold
            and 0 <= read_drift <= threshold
            and 0 <= fill_drift <= threshold,
            f"{name}: periodic timing did not converge",
        )
        fill_sample = (fill_base, fill_doubled, fill_drift)
        if common_fill_sample is None:
            common_fill_sample = fill_sample
        else:
            require(
                all(
                    close(value, reference)
                    for value, reference in zip(
                        fill_sample,
                        common_fill_sample,
                        strict=True,
                    )
                ),
                "HBM fill sample differs by backing",
            )

        fill_elapsed = number(
            write.get("fill_elapsed_ns"),
            f"{name}.write.fill_elapsed",
        )
        offload_elapsed = number(
            write.get("offload_elapsed_ns"),
            f"{name}.write.offload_elapsed",
        )
        total_elapsed = number(
            write.get("total_elapsed_ns"),
            f"{name}.write.total_elapsed",
        )
        read_elapsed = number(
            read.get("elapsed_ns"),
            f"{name}.read.elapsed",
        )
        require(
            write.get("elapsed_timing_basis") == "periodic-extrapolation"
            and write.get("latency_basis")
            == "doubled-page-exact-sample"
            and read.get("elapsed_timing_basis")
            == "periodic-extrapolation"
            and read.get("latency_basis")
            == "doubled-page-exact-sample"
            and fill_elapsed > 0
            and close(fill_elapsed, fill_doubled * hbm_data)
            and close(offload_elapsed, offload_doubled * target_batches)
            and close(read_elapsed, read_doubled * target_batches)
            and close(total_elapsed, fill_elapsed + offload_elapsed)
            and close(
                number(
                    write.get("offload_throughput_GBps"),
                    f"{name}.write.throughput",
                ),
                offload_bytes / offload_elapsed,
            )
            and close(
                number(
                    read.get("throughput_GBps"),
                    f"{name}.read.throughput",
                ),
                offload_bytes / read_elapsed,
            ),
            f"{name}: projected phase timing does not conserve",
        )
        if common_fill is None:
            common_fill = fill_elapsed
        else:
            require(close(fill_elapsed, common_fill), "HBM fill differs by backing")

        layer_offload_base = number(
            layer_case.get("offload_ns_per_batch_base"),
            f"{name}.layer.offload_base",
        )
        layer_offload_doubled = number(
            layer_case.get("offload_ns_per_batch_doubled"),
            f"{name}.layer.offload_doubled",
        )
        layer_restore_base = number(
            layer_case.get("restore_ns_per_batch_base"),
            f"{name}.layer.restore_base",
        )
        layer_restore_doubled = number(
            layer_case.get("restore_ns_per_batch_doubled"),
            f"{name}.layer.restore_doubled",
        )
        layer_offload_drift = number(
            layer_case.get("offload_relative_drift"),
            f"{name}.layer.offload_drift",
        )
        layer_restore_drift = number(
            layer_case.get("restore_relative_drift"),
            f"{name}.layer.restore_drift",
        )
        layer_offload_elapsed = number(
            layer_case.get("offload_elapsed_ns"),
            f"{name}.layer.offload_elapsed",
        )
        layer_restore_elapsed = number(
            layer_case.get("restore_to_hbm_elapsed_ns"),
            f"{name}.layer.restore_elapsed",
        )
        layer_e2e_elapsed = number(
            layer_case.get("e2e_elapsed_ns"),
            f"{name}.layer.e2e_elapsed",
        )
        require(
            layer_case.get("timing_basis")
            == "original-slot-periodic-extrapolation"
            and integer(
                layer_case.get("base_batches"),
                f"{name}.layer.base_batches",
            )
            == base_batches
            and integer(
                layer_case.get("doubled_batches"),
                f"{name}.layer.doubled_batches",
            )
            == doubled_batches
            and layer_case.get("convergence") == "PASS"
            and layer_offload_base > 0
            and layer_offload_doubled > 0
            and layer_restore_base > 0
            and layer_restore_doubled > 0
            and close(
                layer_offload_drift,
                abs(layer_offload_doubled - layer_offload_base)
                / layer_offload_doubled,
            )
            and close(
                layer_restore_drift,
                abs(layer_restore_doubled - layer_restore_base)
                / layer_restore_doubled,
            )
            and 0 <= layer_offload_drift <= threshold
            and 0 <= layer_restore_drift <= threshold
            and close(
                layer_offload_elapsed,
                layer_offload_doubled * layer_batches,
            )
            and close(
                layer_restore_elapsed,
                layer_restore_doubled * layer_batches,
            )
            and close(
                layer_e2e_elapsed,
                layer_offload_elapsed + layer_restore_elapsed,
            )
            and close(
                number(
                    layer_case.get("round_trip_throughput_GBps"),
                    f"{name}.layer.throughput",
                ),
                2 * layer_bytes / layer_e2e_elapsed,
            ),
            f"{name}: layer round-trip timing does not conserve",
        )

        service_curve = case.get("service_curve")
        require(
            isinstance(service_curve, dict)
            and service_curve.get("provenance") == "measured_sample"
            and integer(
                service_curve.get("resident_pages"),
                f"{name}.service_curve.resident_pages",
            )
            == SERVICE_CURVE_RESIDENT_PAGES
            and integer(
                service_curve.get("offload_pages"),
                f"{name}.service_curve.offload_pages",
            )
            == SERVICE_CURVE_OFFLOAD_PAGES
            and integer(
                service_curve.get("read_buffer_pages"),
                f"{name}.service_curve.read_buffer_pages",
            )
            == SERVICE_CURVE_READ_BUFFER_PAGES,
            f"{name}: service-curve sample geometry is invalid",
        )
        points = service_curve.get("points")
        require(
            isinstance(points, list)
            and len(points) == len(SERVICE_CURVE_WINDOWS),
            f"{name}: service curve must contain five points",
        )
        for curve_index, (point, window_pages) in enumerate(
            zip(points, SERVICE_CURVE_WINDOWS, strict=True)
        ):
            curve_path = f"{name}.service_curve[{curve_index}]"
            require(isinstance(point, dict), f"{curve_path} must be an object")
            curve_offload_bytes = integer(
                point.get("offload_bytes"),
                f"{curve_path}.offload_bytes",
            )
            curve_readback_bytes = integer(
                point.get("readback_bytes"),
                f"{curve_path}.readback_bytes",
            )
            curve_offload_elapsed = number(
                point.get("offload_elapsed_ns"),
                f"{curve_path}.offload_elapsed_ns",
            )
            curve_readback_elapsed = number(
                point.get("readback_elapsed_ns"),
                f"{curve_path}.readback_elapsed_ns",
            )
            curve_offload_throughput = number(
                point.get("offload_payload_GBps"),
                f"{curve_path}.offload_payload_GBps",
            )
            curve_readback_throughput = number(
                point.get("readback_payload_GBps"),
                f"{curve_path}.readback_payload_GBps",
            )
            require(
                integer(
                    point.get("window_pages"),
                    f"{curve_path}.window_pages",
                )
                == window_pages
                and curve_offload_bytes
                == SERVICE_CURVE_OFFLOAD_PAGES * page
                and curve_readback_bytes
                == SERVICE_CURVE_OFFLOAD_PAGES * page
                and curve_offload_elapsed > 0
                and curve_readback_elapsed > 0
                and close(
                    curve_offload_throughput,
                    curve_offload_bytes / curve_offload_elapsed,
                )
                and close(
                    curve_readback_throughput,
                    curve_readback_bytes / curve_readback_elapsed,
                ),
                f"{curve_path}: exact sample throughput does not conserve",
            )

            for phase_key in (
                "offload_latency_ns",
                "readback_latency_ns",
            ):
                latency = point.get(phase_key)
                require(
                    isinstance(latency, dict),
                    f"{curve_path}.{phase_key} must be an object",
                )
                offered = validate_quantiles(
                    latency.get("offered"),
                    f"{curve_path}.{phase_key}.offered",
                )
                service = validate_quantiles(
                    latency.get("service"),
                    f"{curve_path}.{phase_key}.service",
                )
                require(
                    all(
                        offered_value >= service_value
                        for offered_value, service_value in zip(
                            offered,
                            service,
                            strict=True,
                        )
                    ),
                    f"{curve_path}.{phase_key}: offered latency "
                    "must include service latency",
                )

            utilization = point.get("resource_utilization")
            require(
                isinstance(utilization, dict),
                f"{curve_path}.resource_utilization must be an object",
            )
            expected_utilization_keys = {
                "hbf_media",
                "base_die_read",
                "base_die_write",
                "external_controller",
                "external_media",
                "external_m2s",
                "external_s2m",
            }
            require(
                set(utilization) == expected_utilization_keys,
                f"{curve_path}: resource utilization key set drifted",
            )

            queue_wait = point.get("queue_wait_work_ns")
            expected_queue_keys = {
                "device_outstanding",
                "backing_ingress",
                "backing_scheduler",
                "backing_ecc",
                "base_die_read",
                "base_die_write",
                "external_controller",
                "external_media",
                "external_m2s",
                "external_s2m",
            }
            require(
                isinstance(queue_wait, dict)
                and set(queue_wait) == expected_queue_keys
                and all(
                    number(queue_wait[key], f"{curve_path}.queue.{key}")
                    >= 0
                    for key in expected_queue_keys
                ),
                f"{curve_path}: queue-wait work is invalid",
            )

            wire = point.get("wire_efficiency")
            require(
                isinstance(wire, dict),
                f"{curve_path}.wire_efficiency must be an object",
            )
            m2s_efficiency = number(
                wire.get("m2s"),
                f"{curve_path}.wire_efficiency.m2s",
            )
            s2m_efficiency = number(
                wire.get("s2m"),
                f"{curve_path}.wire_efficiency.s2m",
            )
            active_media = integer(
                point.get("active_media_channels"),
                f"{curve_path}.active_media_channels",
            )

            if name == "hbf":
                require(
                    wire.get("model") == "payload-only"
                    and close(m2s_efficiency, 1.0)
                    and close(s2m_efficiency, 1.0)
                    and point.get("max_device_outstanding") is None
                    and 0 < active_media <= hbf_stacks * hbf_channels
                    and all(
                        utilization[key] is None
                        for key in (
                            "external_controller",
                            "external_media",
                            "external_m2s",
                            "external_s2m",
                        )
                    ),
                    f"{curve_path}: HBF service-curve resources are invalid",
                )
                for key in ("hbf_media", "base_die_read", "base_die_write"):
                    value = number(
                        utilization.get(key),
                        f"{curve_path}.utilization.{key}",
                    )
                    require(
                        0 <= value <= 1 + 1e-9,
                        f"{curve_path}: invalid {key} utilization",
                    )
                parallel_pages = min(hbf_planes, window_pages)
                hbf_write_ceiling = (
                    parallel_pages
                    * page
                    / (
                        number(
                            hbf_profile.get("program_page_ns"),
                            "profile.hbf.program_page_ns",
                        )
                        + number(
                            hbf_profile.get("program_verify_ns"),
                            "profile.hbf.program_verify_ns",
                        )
                    )
                )
                hbf_read_ceiling = (
                    parallel_pages
                    * page
                    / number(
                        hbf_profile.get("read_page_ns"),
                        "profile.hbf.read_page_ns",
                    )
                )
                require(
                    curve_offload_throughput
                    <= hbf_write_ceiling * (1 + 1e-9)
                    and curve_readback_throughput
                    <= hbf_read_ceiling * (1 + 1e-9),
                    f"{curve_path}: HBF throughput exceeds media ceiling",
                )
            else:
                external_profile = dict(profile[name.replace("-", "_")])
                external_profile["max_outstanding_requests"] = window_pages
                command_bytes = integer(
                    external_profile.get("command_bytes"),
                    f"profile.{name}.command_bytes",
                )
                completion_bytes = integer(
                    external_profile.get("completion_bytes"),
                    f"profile.{name}.completion_bytes",
                )
                expected_m2s_efficiency = (
                    curve_offload_bytes
                    / (
                        curve_offload_bytes
                        + 2
                        * SERVICE_CURVE_OFFLOAD_PAGES
                        * command_bytes
                    )
                )
                expected_s2m_efficiency = (
                    curve_readback_bytes
                    / (
                        curve_readback_bytes
                        + 2
                        * SERVICE_CURVE_OFFLOAD_PAGES
                        * completion_bytes
                    )
                )
                require(
                    wire.get("model") == "payload-plus-protocol"
                    and close(m2s_efficiency, expected_m2s_efficiency)
                    and close(s2m_efficiency, expected_s2m_efficiency)
                    and integer(
                        point.get("max_device_outstanding"),
                        f"{curve_path}.max_device_outstanding",
                    )
                    == window_pages
                    and 0
                    < active_media
                    <= integer(
                        external_profile.get("media_channels"),
                        f"profile.{name}.media_channels",
                    )
                    and all(
                        utilization[key] is None
                        for key in (
                            "hbf_media",
                            "base_die_read",
                            "base_die_write",
                        )
                    ),
                    f"{curve_path}: external service-curve resources are invalid",
                )
                for key in (
                    "external_controller",
                    "external_media",
                    "external_m2s",
                    "external_s2m",
                ):
                    value = number(
                        utilization.get(key),
                        f"{curve_path}.utilization.{key}",
                    )
                    require(
                        0 <= value <= 1 + 1e-9,
                        f"{curve_path}: invalid {key} utilization",
                    )
                require(
                    curve_offload_throughput
                    <= external_payload_ceiling_GBps(
                        external_profile,
                        page,
                        "write",
                    )
                    * (1 + 1e-9)
                    and curve_readback_throughput
                    <= external_payload_ceiling_GBps(
                        external_profile,
                        page,
                        "read",
                    )
                    * (1 + 1e-9),
                    f"{curve_path}: external throughput exceeds "
                    "explicit-pipeline ceiling",
                )

        for phase_name, phase in (("write", write), ("read", read)):
            offered = tuple(
                number(
                    phase.get(f"offered_latency_{rank}_ns"),
                    f"{name}.{phase_name}.offered.{rank}",
                )
                for rank in ("p50", "p95", "p99")
            )
            service = tuple(
                number(
                    phase.get(f"service_latency_{rank}_ns"),
                    f"{name}.{phase_name}.service.{rank}",
                )
                for rank in ("p50", "p95", "p99")
            )
            require(
                0 < offered[0] <= offered[1] <= offered[2]
                and 0 < service[0] <= service[1] <= service[2]
                and all(
                    offered_value >= service_value
                    for offered_value, service_value in zip(
                        offered,
                        service,
                        strict=True,
                    )
                ),
                f"{name}: offered/service {phase_name} quantiles are invalid",
            )

        fill_finish = number(
            timeline.get("fill_finish_ns"),
            f"{name}.timeline.fill_finish",
        )
        offload_start = number(
            timeline.get("offload_start_ns"),
            f"{name}.timeline.offload_start",
        )
        write_finish = number(
            timeline.get("write_finish_ns"),
            f"{name}.timeline.write_finish",
        )
        read_start = number(
            timeline.get("read_start_ns"),
            f"{name}.timeline.read_start",
        )
        read_finish = number(
            timeline.get("read_finish_ns"),
            f"{name}.timeline.read_finish",
        )
        quiescent = number(
            timeline.get("quiescent_finish_ns"),
            f"{name}.timeline.quiescent",
        )
        drain = number(
            timeline.get("drain_tail_ns"),
            f"{name}.timeline.drain",
        )
        require(
            close(fill_finish, fill_elapsed)
            and close(offload_start, fill_finish)
            and close(write_finish, offload_start + offload_elapsed)
            and close(read_start, write_finish)
            and close(read_finish, read_start + read_elapsed)
            and close(quiescent, read_finish + drain),
            f"{name}: projected timeline is not causal",
        )
        require(
            integer(hbm.get("write_bytes"), f"{name}.hbm.write_bytes")
            == expected_hbm_write
            and integer(hbm.get("read_bytes"), f"{name}.hbm.read_bytes")
            == expected_hbm_read
            and integer(hbm.get("user_accesses"), f"{name}.hbm.user_accesses")
            == expected_hbm_user
            and integer(
                hbm.get("background_accesses"),
                f"{name}.hbm.background_accesses",
            )
            == expected_hbm_background,
            f"{name}: projected HBM traffic does not conserve",
        )

        if name == "hbf":
            require(case.get("backing") == "hbf", "hbf: backing kind drifted")
            hbf = case.get("hbf")
            link = case.get("base_die_link")
            require(isinstance(hbf, dict), "hbf: device statistics missing")
            require(isinstance(link, dict), "hbf: D2D statistics missing")
            require(case.get("external") is None, "hbf: external must be null")
            page_programs = offload_pages + expected_mapping
            require(
                integer(hbf.get("logical_write_bytes"), "hbf.logical_write")
                == offload_bytes
                and integer(hbf.get("logical_read_bytes"), "hbf.logical_read")
                == offload_bytes
                and integer(hbf.get("physical_read_bytes"), "hbf.physical_read")
                == offload_bytes
                and integer(
                    hbf.get("physical_write_bytes"),
                    "hbf.physical_write",
                )
                == page_programs * page
                and integer(hbf.get("read_requests"), "hbf.read_requests")
                == offload_pages
                and integer(hbf.get("program_requests"), "hbf.program_requests")
                == offload_pages
                and integer(hbf.get("page_reads"), "hbf.page_reads")
                == offload_pages
                and integer(hbf.get("data_programs"), "hbf.data_programs")
                == offload_pages
                and integer(hbf.get("page_programs"), "hbf.page_programs")
                == page_programs
                and integer(hbf.get("mapping_programs"), "hbf.mapping_programs")
                == expected_mapping
                and integer(
                    hbf.get("mapping_checkpoint_rounds"),
                    "hbf.mapping_rounds",
                )
                == expected_mapping_rounds
                and integer(hbf.get("gc_runs"), "hbf.gc_runs") == 0
                and integer(hbf.get("gc_relocations"), "hbf.gc_relocations")
                == 0
                and integer(hbf.get("block_erases"), "hbf.block_erases") == 0,
                "hbf: projected physical accounting does not conserve",
            )
            require(
                integer(link.get("read_transfers"), "hbf.link.read_transfers")
                == offload_pages
                and integer(
                    link.get("write_transfers"),
                    "hbf.link.write_transfers",
                )
                == offload_pages
                and integer(link.get("read_bytes"), "hbf.link.read_bytes")
                == offload_bytes
                and integer(link.get("write_bytes"), "hbf.link.write_bytes")
                == offload_bytes
                and drain > 0,
                "hbf: projected D2D/checkpoint accounting does not conserve",
            )
            for key in (
                "representative_media_utilization",
            ):
                value = number(hbf.get(key), f"hbf.{key}")
                require(0 <= value <= 1 + 1e-9, f"hbf: invalid {key}")
            for key in (
                "representative_read_utilization",
                "representative_write_utilization",
            ):
                value = number(link.get(key), f"hbf.link.{key}")
                require(0 <= value <= 1 + 1e-9, f"hbf: invalid {key}")
        else:
            require(
                case.get("backing") == "external",
                f"{name}: backing kind drifted",
            )
            external = case.get("external")
            require(isinstance(external, dict), f"{name}: external stats missing")
            require(case.get("hbf") is None, f"{name}: HBF must be null")
            require(case.get("base_die_link") is None, f"{name}: D2D must be null")
            require(
                external.get("kind") == name
                and integer(
                    external.get("write_requests"),
                    f"{name}.external.write_requests",
                )
                == offload_pages
                and integer(
                    external.get("read_requests"),
                    f"{name}.external.read_requests",
                )
                == offload_pages
                and integer(
                    external.get("write_bytes"),
                    f"{name}.external.write_bytes",
                )
                == offload_bytes
                and integer(
                    external.get("read_bytes"),
                    f"{name}.external.read_bytes",
                )
                == offload_bytes
                and close(drain, 0.0),
                f"{name}: projected external traffic does not conserve",
            )
            external_profile = profile[name.replace("-", "_")]
            request_count = 2 * offload_pages
            expected_m2s_protocol = (
                request_count
                * integer(
                    external_profile.get("command_bytes"),
                    f"profile.{name}.command_bytes",
                )
            )
            expected_s2m_protocol = (
                request_count
                * integer(
                    external_profile.get("completion_bytes"),
                    f"profile.{name}.completion_bytes",
                )
            )
            require(
                integer(
                    external.get("m2s_payload_bytes"),
                    f"{name}.external.m2s_payload_bytes",
                )
                == offload_bytes
                and integer(
                    external.get("m2s_protocol_bytes"),
                    f"{name}.external.m2s_protocol_bytes",
                )
                == expected_m2s_protocol
                and integer(
                    external.get("m2s_wire_bytes"),
                    f"{name}.external.m2s_wire_bytes",
                )
                == offload_bytes + expected_m2s_protocol
                and integer(
                    external.get("s2m_payload_bytes"),
                    f"{name}.external.s2m_payload_bytes",
                )
                == offload_bytes
                and integer(
                    external.get("s2m_protocol_bytes"),
                    f"{name}.external.s2m_protocol_bytes",
                )
                == expected_s2m_protocol
                and integer(
                    external.get("s2m_wire_bytes"),
                    f"{name}.external.s2m_wire_bytes",
                )
                == offload_bytes + expected_s2m_protocol,
                f"{name}: projected payload/protocol/wire accounting does not conserve",
            )
            for key in (
                "representative_controller_utilization",
                "representative_media_utilization",
                "representative_m2s_utilization",
                "representative_s2m_utilization",
            ):
                value = number(external.get(key), f"{name}.external.{key}")
                require(0 <= value <= 1 + 1e-9, f"{name}: invalid {key}")

    require(
        set(by_name) == {"hbf", "cxl-memory", "nvme-ssd"},
        f"unexpected case set: {set(by_name)}",
    )

    lifecycle = document.get("layer_lifecycle")
    require(
        isinstance(lifecycle, dict)
        and lifecycle.get("provenance") == "derived"
        and lifecycle.get("formula") == "one-offload-plus-N-restores"
        and lifecycle.get("restore_counts") == LIFECYCLE_RESTORE_COUNTS,
        "layer lifecycle contract is missing or unsupported",
    )
    lifecycle_points = lifecycle.get("points")
    require(
        isinstance(lifecycle_points, list)
        and len(lifecycle_points) == len(LIFECYCLE_RESTORE_COUNTS),
        "layer lifecycle point count drifted",
    )
    case_order = ["hbf", "cxl-memory", "nvme-ssd"]
    for lifecycle_index, (point, restores) in enumerate(
        zip(
            lifecycle_points,
            LIFECYCLE_RESTORE_COUNTS,
            strict=True,
        )
    ):
        path = f"layer_lifecycle.points[{lifecycle_index}]"
        require(isinstance(point, dict), f"{path} must be an object")
        require(
            integer(point.get("restore_count"), f"{path}.restore_count")
            == restores,
            f"{path}: restore count drifted",
        )
        lifecycle_cases = point.get("cases")
        require(
            isinstance(lifecycle_cases, list)
            and len(lifecycle_cases) == len(case_order),
            f"{path}: lifecycle case count drifted",
        )
        elapsed_by_name: dict[str, float] = {}
        for lifecycle_case, expected_name in zip(
            lifecycle_cases,
            case_order,
            strict=True,
        ):
            require(
                isinstance(lifecycle_case, dict)
                and lifecycle_case.get("name") == expected_name,
                f"{path}: lifecycle case order drifted",
            )
            source = by_name[expected_name]["layer_round_trip"]
            expected_offload = number(
                source.get("offload_elapsed_ns"),
                f"{expected_name}.layer.offload_elapsed_ns",
            )
            expected_restore = number(
                source.get("restore_to_hbm_elapsed_ns"),
                f"{expected_name}.layer.restore_elapsed_ns",
            )
            offload = number(
                lifecycle_case.get("offload_elapsed_ns"),
                f"{path}.{expected_name}.offload_elapsed_ns",
            )
            restore = number(
                lifecycle_case.get("restore_elapsed_ns"),
                f"{path}.{expected_name}.restore_elapsed_ns",
            )
            total = number(
                lifecycle_case.get("total_elapsed_ns"),
                f"{path}.{expected_name}.total_elapsed_ns",
            )
            average = number(
                lifecycle_case.get("average_elapsed_ns_per_restore"),
                f"{path}.{expected_name}.average_elapsed_ns_per_restore",
            )
            require(
                close(offload, expected_offload)
                and close(restore, expected_restore)
                and close(total, offload + restores * restore)
                and close(average, total / restores),
                f"{path}.{expected_name}: lifecycle derivation is invalid",
            )
            elapsed_by_name[expected_name] = total
        expected_winner = min(
            case_order,
            key=lambda case_name: elapsed_by_name[case_name],
        )
        require(
            point.get("winner") == expected_winner,
            f"{path}: lifecycle winner is not derived from elapsed time",
        )

    break_even = lifecycle.get("break_even")
    require(
        isinstance(break_even, dict)
        and break_even.get("definition")
        == "first-discrete-N-where-hbf-total-is-no-greater",
        "layer lifecycle break-even definition is invalid",
    )

    def expected_break_even(external_name: str) -> int | None:
        hbf_layer = by_name["hbf"]["layer_round_trip"]
        external_layer = by_name[external_name]["layer_round_trip"]
        for restores in LIFECYCLE_RESTORE_COUNTS:
            hbf_elapsed = number(
                hbf_layer.get("offload_elapsed_ns"),
                "hbf.layer.offload_elapsed_ns",
            ) + restores * number(
                hbf_layer.get("restore_to_hbm_elapsed_ns"),
                "hbf.layer.restore_elapsed_ns",
            )
            external_elapsed = number(
                external_layer.get("offload_elapsed_ns"),
                f"{external_name}.layer.offload_elapsed_ns",
            ) + restores * number(
                external_layer.get("restore_to_hbm_elapsed_ns"),
                f"{external_name}.layer.restore_elapsed_ns",
            )
            if hbf_elapsed <= external_elapsed:
                return restores
        return None

    require(
        break_even.get("hbf_vs_cxl_memory")
        == expected_break_even("cxl-memory")
        and break_even.get("hbf_vs_nvme_ssd")
        == expected_break_even("nvme-ssd"),
        "layer lifecycle break-even was not derived from converged samples",
    )

    for phase, key in (
        ("write_phase", "offload_elapsed_ns"),
        ("read_phase", "elapsed_ns"),
        ("layer_round_trip", "e2e_elapsed_ns"),
    ):
        dram = number(by_name["cxl-memory"][phase][key], f"dram.{phase}")
        ssd = number(by_name["nvme-ssd"][phase][key], f"ssd.{phase}")
        require(ssd > dram, f"SSD must be slower than CXL memory in {phase}")

    hbm_channel_bw = (
        number(hbm_profile.get("channel_width_bits"), "hbm.width")
        * number(hbm_profile.get("pin_rate_Gbps"), "hbm.pin_rate")
        / 8.0
    )
    hbm_ceiling = (
        integer(hbm_profile.get("stacks"), "hbm.stacks")
        * integer(hbm_profile.get("channels_per_stack"), "hbm.channels")
        * hbm_channel_bw
    )
    require(
        common_fill is not None
        and hbm_data / common_fill <= hbm_ceiling * (1 + 1e-9),
        "projected HBM fill exceeds interface ceiling",
    )
    hbf_parallel_pages = min(hbf_planes, batch_pages)
    hbf_program_ns = number(
        hbf_profile.get("program_page_ns"),
        "hbf.program_page_ns",
    ) + number(
        hbf_profile.get("program_verify_ns"),
        "hbf.program_verify_ns",
    )
    hbf_read_ns = number(
        hbf_profile.get("read_page_ns"),
        "hbf.read_page_ns",
    )
    ceilings = {
        "hbf offloading": hbf_parallel_pages * page / hbf_program_ns,
        "hbf read": hbf_parallel_pages * page / hbf_read_ns,
        "hbf layer offloading": (
            hbf_parallel_pages * page / hbf_program_ns
        ),
        "hbf layer restore": hbf_parallel_pages * page / hbf_read_ns,
        "cxl-memory offloading": external_payload_ceiling_GBps(
            profile["cxl_memory"], page, "write"
        ),
        "cxl-memory read": external_payload_ceiling_GBps(
            profile["cxl_memory"], page, "read"
        ),
        "cxl-memory layer offloading": external_payload_ceiling_GBps(
            profile["cxl_memory"], page, "write"
        ),
        "cxl-memory layer restore": external_payload_ceiling_GBps(
            profile["cxl_memory"], page, "read"
        ),
        "nvme-ssd offloading": external_payload_ceiling_GBps(
            profile["nvme_ssd"], page, "write"
        ),
        "nvme-ssd read": external_payload_ceiling_GBps(
            profile["nvme_ssd"], page, "read"
        ),
        "nvme-ssd layer offloading": external_payload_ceiling_GBps(
            profile["nvme_ssd"], page, "write"
        ),
        "nvme-ssd layer restore": external_payload_ceiling_GBps(
            profile["nvme_ssd"], page, "read"
        ),
    }
    measured = {
        "hbf offloading": number(
            by_name["hbf"]["write_phase"]["offload_throughput_GBps"],
            "hbf offloading throughput",
        ),
        "hbf read": number(
            by_name["hbf"]["read_phase"]["throughput_GBps"],
            "hbf read throughput",
        ),
        "hbf layer offloading": (
            layer_bytes
            / number(
                by_name["hbf"]["layer_round_trip"][
                    "offload_elapsed_ns"
                ],
                "hbf layer offloading elapsed",
            )
        ),
        "hbf layer restore": (
            layer_bytes
            / number(
                by_name["hbf"]["layer_round_trip"][
                    "restore_to_hbm_elapsed_ns"
                ],
                "hbf layer restore elapsed",
            )
        ),
        "cxl-memory offloading": number(
            by_name["cxl-memory"]["write_phase"]["offload_throughput_GBps"],
            "dram offloading throughput",
        ),
        "cxl-memory read": number(
            by_name["cxl-memory"]["read_phase"]["throughput_GBps"],
            "dram read throughput",
        ),
        "cxl-memory layer offloading": (
            layer_bytes
            / number(
                by_name["cxl-memory"]["layer_round_trip"][
                    "offload_elapsed_ns"
                ],
                "dram layer offloading elapsed",
            )
        ),
        "cxl-memory layer restore": (
            layer_bytes
            / number(
                by_name["cxl-memory"]["layer_round_trip"][
                    "restore_to_hbm_elapsed_ns"
                ],
                "dram layer restore elapsed",
            )
        ),
        "nvme-ssd offloading": number(
            by_name["nvme-ssd"]["write_phase"]["offload_throughput_GBps"],
            "ssd offloading throughput",
        ),
        "nvme-ssd read": number(
            by_name["nvme-ssd"]["read_phase"]["throughput_GBps"],
            "ssd read throughput",
        ),
        "nvme-ssd layer offloading": (
            layer_bytes
            / number(
                by_name["nvme-ssd"]["layer_round_trip"][
                    "offload_elapsed_ns"
                ],
                "ssd layer offloading elapsed",
            )
        ),
        "nvme-ssd layer restore": (
            layer_bytes
            / number(
                by_name["nvme-ssd"]["layer_round_trip"][
                    "restore_to_hbm_elapsed_ns"
                ],
                "ssd layer restore elapsed",
            )
        ),
    }
    for name, ceiling in ceilings.items():
        require(ceiling > 0, f"{name}: non-positive physical ceiling")
        require(
            measured[name] <= ceiling * (1 + 1e-9),
            f"{name}: projected throughput exceeds physical ceiling",
        )

    facts: dict[str, int | float] = {
        "page": page,
        "hbm_capacity": hbm_capacity,
        "hbm_data": hbm_data,
        "read_buffer": read_buffer,
        "total_write": total_write,
        "offload_bytes": offload_bytes,
        "offload_pages": offload_pages,
        "batch_pages": batch_pages,
        "target_batches": target_batches,
        "layer_logical_bytes": layer_logical_bytes,
        "layer_bytes": layer_bytes,
        "layer_padding_bytes": layer_padding_bytes,
        "layer_pages": layer_pages,
        "layer_batches": layer_batches,
        "base_batches": base_batches,
        "doubled_batches": doubled_batches,
        "base_sample_resident_pages": base_sample_resident_pages,
        "doubled_sample_resident_pages": doubled_sample_resident_pages,
        "hbf_planes": hbf_planes,
        "hbm_ceiling": hbm_ceiling,
        "hbm_burst_bytes": hbm_burst_bytes,
        "hbm_burst_children": hbm_burst_children,
        "mapping_programs": expected_mapping,
    }
    return by_name, facts


def faster_statement(
    lhs_name: str,
    lhs_ns: float,
    rhs_name: str,
    rhs_ns: float,
    phase: str,
) -> str:
    if lhs_ns <= rhs_ns:
        return f"{lhs_name} {phase} is {rhs_ns / lhs_ns:.2f}× faster than {rhs_name}"
    return f"{rhs_name} {phase} is {lhs_ns / rhs_ns:.2f}× faster than {lhs_name}"


def render_markdown(
    document: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    facts: dict[str, int | float],
) -> str:
    page = int(facts["page"])
    hbm_capacity = int(facts["hbm_capacity"])
    hbm_data = int(facts["hbm_data"])
    read_buffer = int(facts["read_buffer"])
    total_write = int(facts["total_write"])
    offload_bytes = int(facts["offload_bytes"])
    batch_pages = int(facts["batch_pages"])
    layer_logical_bytes = int(facts["layer_logical_bytes"])
    layer_bytes = int(facts["layer_bytes"])
    layer_padding_bytes = int(facts["layer_padding_bytes"])
    physical_excess = total_write - hbm_capacity

    rows: list[str] = []
    convergence_rows: list[str] = []
    layer_rows: list[str] = []
    layer_convergence_rows: list[str] = []
    service_curve_rows: list[str] = []
    for name in ("hbf", "cxl-memory", "nvme-ssd"):
        case = by_name[name]
        write = case["write_phase"]
        read = case["read_phase"]
        rows.append(
            "| {name} | {offload_ms:.3f} | {offload_bw:.3f} | "
            "{read_ms:.3f} | {read_bw:.3f} | {drain_ms:.3f} |".format(
                name=name,
                offload_ms=write["offload_elapsed_ns"] / 1e6,
                offload_bw=write["offload_throughput_GBps"],
                read_ms=read["elapsed_ns"] / 1e6,
                read_bw=read["throughput_GBps"],
                drain_ms=case["timeline"]["drain_tail_ns"] / 1e6,
            )
        )
        sample = case["sampling"]
        convergence_rows.append(
            "| {name} | {offload_base:.3f} | {offload_double:.3f} | "
            "{offload_drift:.4f}% | {read_base:.3f} | {read_double:.3f} | "
            "{read_drift:.4f}% |".format(
                name=name,
                offload_base=sample["offload_ns_per_batch_base"] / 1e3,
                offload_double=sample["offload_ns_per_batch_doubled"] / 1e3,
                offload_drift=100 * sample["offload_relative_drift"],
                read_base=sample["read_ns_per_batch_base"] / 1e3,
                read_double=sample["read_ns_per_batch_doubled"] / 1e3,
                read_drift=100 * sample["read_relative_drift"],
            )
        )
        layer = case["layer_round_trip"]
        layer_rows.append(
            "| {name} | {offload_ms:.3f} | {restore_ms:.3f} | "
            "{e2e_ms:.3f} | {throughput:.3f} |".format(
                name=name,
                offload_ms=layer["offload_elapsed_ns"] / 1e6,
                restore_ms=layer["restore_to_hbm_elapsed_ns"] / 1e6,
                e2e_ms=layer["e2e_elapsed_ns"] / 1e6,
                throughput=layer["round_trip_throughput_GBps"],
            )
        )
        layer_convergence_rows.append(
            "| {name} | {offload_base:.3f} | {offload_double:.3f} | "
            "{offload_drift:.4f}% | {restore_base:.3f} | "
            "{restore_double:.3f} | {restore_drift:.4f}% |".format(
                name=name,
                offload_base=layer["offload_ns_per_batch_base"] / 1e3,
                offload_double=(
                    layer["offload_ns_per_batch_doubled"] / 1e3
                ),
                offload_drift=100 * layer["offload_relative_drift"],
                restore_base=layer["restore_ns_per_batch_base"] / 1e3,
                restore_double=(
                    layer["restore_ns_per_batch_doubled"] / 1e3
                ),
                restore_drift=100 * layer["restore_relative_drift"],
            )
        )
        for point in case["service_curve"]["points"]:
            utilization = {
                key: value
                for key, value in point["resource_utilization"].items()
                if value is not None
            }
            bottleneck, bottleneck_value = max(
                utilization.items(),
                key=lambda item: item[1],
            )
            service_curve_rows.append(
                "| {name} | {window} | {offload:.3f} | {read:.3f} | "
                "{offered:.3f} / {service:.3f} | "
                "{read_offered:.3f} / {read_service:.3f} | "
                "{bottleneck} ({utilization:.1f}%) |".format(
                    name=name,
                    window=point["window_pages"],
                    offload=point["offload_payload_GBps"],
                    read=point["readback_payload_GBps"],
                    offered=(
                        point["offload_latency_ns"]["offered"]["p99"]
                        / 1e3
                    ),
                    service=(
                        point["offload_latency_ns"]["service"]["p99"]
                        / 1e3
                    ),
                    read_offered=(
                        point["readback_latency_ns"]["offered"]["p99"]
                        / 1e3
                    ),
                    read_service=(
                        point["readback_latency_ns"]["service"]["p99"]
                        / 1e3
                    ),
                    bottleneck=bottleneck,
                    utilization=100 * bottleneck_value,
                )
            )

    hbf = by_name["hbf"]
    dram = by_name["cxl-memory"]
    ssd = by_name["nvme-ssd"]
    comparisons = [
        faster_statement(
            "HBF",
            hbf["write_phase"]["offload_elapsed_ns"],
            "CXL memory",
            dram["write_phase"]["offload_elapsed_ns"],
            "offloading",
        ),
        faster_statement(
            "HBF",
            hbf["write_phase"]["offload_elapsed_ns"],
            "NVMe SSD",
            ssd["write_phase"]["offload_elapsed_ns"],
            "offloading",
        ),
        faster_statement(
            "HBF",
            hbf["read_phase"]["elapsed_ns"],
            "CXL memory",
            dram["read_phase"]["elapsed_ns"],
            "readback",
        ),
        faster_statement(
            "HBF",
            hbf["read_phase"]["elapsed_ns"],
            "NVMe SSD",
            ssd["read_phase"]["elapsed_ns"],
            "readback",
        ),
        faster_statement(
            "HBF",
            hbf["layer_round_trip"]["e2e_elapsed_ns"],
            "CXL memory",
            dram["layer_round_trip"]["e2e_elapsed_ns"],
            "layer round-trip",
        ),
        faster_statement(
            "HBF",
            hbf["layer_round_trip"]["e2e_elapsed_ns"],
            "NVMe SSD",
            ssd["layer_round_trip"]["e2e_elapsed_ns"],
            "layer round-trip",
        ),
    ]

    profile = document["profile"]
    hbm_profile = profile["hbm"]
    hbf_profile = profile["hbf"]
    base_link = profile["base_die_link"]
    dram_profile = profile["cxl_memory"]
    ssd_profile = profile["nvme_ssd"]
    hbf_parallel_pages = min(int(facts["hbf_planes"]), batch_pages)
    hbf_program_ceiling = (
        hbf_parallel_pages
        * page
        / (
            hbf_profile["program_page_ns"]
            + hbf_profile["program_verify_ns"]
        )
    )
    hbf_read_ceiling = (
        hbf_parallel_pages * page / hbf_profile["read_page_ns"]
    )
    hbf_waf = (
        hbf["hbf"]["physical_write_bytes"]
        / hbf["hbf"]["logical_write_bytes"]
    )
    hbm_fill_efficiency = (
        100
        * (hbm_data / hbf["write_phase"]["fill_elapsed_ns"])
        / float(facts["hbm_ceiling"])
    )
    target_hbf_bytes = int(hbf_profile["target_capacity_bytes"])
    sample_bytes = (
        document["method"]["doubled_sample_batches"]
        * batch_pages
        * page
    )
    method = document["method"]
    layer_contract = document["layer_round_trip"]
    lifecycle = document["layer_lifecycle"]
    lifecycle_rows = [
        "| {restores} | {hbf:.3f} | {cxl:.3f} | {ssd:.3f} | {winner} |".format(
            restores=point["restore_count"],
            hbf=point["cases"][0]["total_elapsed_ns"] / 1e6,
            cxl=point["cases"][1]["total_elapsed_ns"] / 1e6,
            ssd=point["cases"][2]["total_elapsed_ns"] / 1e6,
            winner=point["winner"],
        )
        for point in lifecycle["points"]
    ]
    cxl_break_even = lifecycle["break_even"]["hbf_vs_cxl_memory"]
    ssd_break_even = lifecycle["break_even"]["hbf_vs_nvme_ssd"]
    d2d_read_page_ns = page / base_link["read_GBps_per_stack"]
    d2d_write_page_ns = page / base_link["write_GBps_per_stack"]
    d2d_read_layer_ns = (
        layer_bytes
        / (
            hbf_profile["stacks"]
            * base_link["read_GBps_per_stack"]
        )
    )
    d2d_write_layer_ns = (
        layer_bytes
        / (
            hbf_profile["stacks"]
            * base_link["write_GBps_per_stack"]
        )
    )
    d2d_e2e_percent = (
        100
        * (d2d_read_layer_ns + d2d_write_layer_ns)
        / hbf["layer_round_trip"]["e2e_elapsed_ns"]
    )
    d2d_restore_percent = (
        100
        * d2d_read_layer_ns
        / hbf["layer_round_trip"]["restore_to_hbm_elapsed_ns"]
    )
    parameter_rows = [
        (
            f"| Workload | {total_write / 2**30:.3f} GiB writes; "
            f"{offload_bytes / 2**30:.6f} GiB offloading | "
            "Saturated, 0 ns interarrival | "
            f"{batch_pages} pages / {batch_pages * page / 2**20:.3f} MiB |"
        ),
        (
            f"| HBM | {hbm_capacity / 2**30:.3f} GiB; "
            f"{hbm_profile['stacks']} stacks × "
            f"{hbm_profile['channels_per_stack']} channels × "
            f"{hbm_profile['pseudo_channels_per_channel']} pseudo-channels | "
            f"{int(facts['hbm_burst_bytes'])} B burst | "
            f"{float(facts['hbm_ceiling']):.1f} GB/s aggregate ceiling |"
        ),
        (
            f"| HBF | {target_hbf_bytes / 2**40:.3f} TiB; "
            f"{int(facts['hbf_planes'])} planes | "
            f"{hbf_profile['read_page_ns'] / 1e3:.3f} µs read; "
            f"{(hbf_profile['program_page_ns'] + hbf_profile['program_verify_ns']) / 1e3:.3f} "
            "µs program/verify | "
            f"Per-stack modeled D2D R/W "
            f"{base_link['read_GBps_per_stack']:.0f}/"
            f"{base_link['write_GBps_per_stack']:.0f} GB/s "
            "(TCAD design input, not vendor measurement) |"
        ),
        (
            f"| CXL memory | {dram_profile['capacity_bytes'] / 2**40:.3f} "
            f"TiB; {dram_profile['media_channels']} media channels; "
            f"{dram_profile['max_outstanding_requests']} global credits | "
            f"Controller issue/process {dram_profile['controller_issue_ns']:.0f}/"
            f"{dram_profile['controller_processing_ns']:.0f} ns; media R/W "
            f"{dram_profile['media_read_latency_ns']:.0f}/"
            f"{dram_profile['media_write_latency_ns']:.0f} ns; "
            f"{dram_profile['one_way_propagation_ns']:.0f} ns one-way | "
            f"Media R/W {dram_profile['media_read_GBps']:.1f}/"
            f"{dram_profile['media_write_GBps']:.1f}; M2S/S2M "
            f"{dram_profile['m2s_GBps']:.1f}/"
            f"{dram_profile['s2m_GBps']:.1f} GB/s; "
            f"{dram_profile['command_bytes']}/"
            f"{dram_profile['completion_bytes']} B command/completion |"
        ),
        (
            f"| NVMe SSD | {ssd_profile['capacity_bytes'] / 2**40:.3f} "
            f"TiB; {ssd_profile['media_channels']} media channels; "
            f"{ssd_profile['max_outstanding_requests']} global credits | "
            f"Controller issue/process {ssd_profile['controller_issue_ns']:.0f}/"
            f"{ssd_profile['controller_processing_ns']:.0f} ns; media R/W "
            f"{ssd_profile['media_read_latency_ns'] / 1e3:.0f}/"
            f"{ssd_profile['media_write_latency_ns'] / 1e3:.0f} µs; "
            f"{ssd_profile['one_way_propagation_ns']:.0f} ns one-way | "
            f"Media R/W {ssd_profile['media_read_GBps']:.1f}/"
            f"{ssd_profile['media_write_GBps']:.1f}; M2S/S2M "
            f"{ssd_profile['m2s_GBps']:.1f}/"
            f"{ssd_profile['s2m_GBps']:.1f} GB/s; "
            f"{ssd_profile['command_bytes']}/"
            f"{ssd_profile['completion_bytes']} B command/completion |"
        ),
        (
            f"| Replay | {hbf_profile['sample_capacity_bytes'] / 2**20:.0f} "
            f"MiB sampled HBF | "
            f"{100 * method['convergence_threshold']:.1f}% convergence "
            "threshold | "
            f"HBM fill {method['base_sample_resident_pages'] * page / 2**20:.0f}/"
            f"{method['doubled_sample_resident_pages'] * page / 2**20:.0f} "
            f"MiB; offloading {method['base_sample_batches']}/"
            f"{method['doubled_sample_batches']} batches |"
        ),
    ]

    return "\n".join(
        [
            "# Real-capacity HBM offloading experiment",
            "",
            (
                f"Validation state: `{document['validation']['status']}`. "
                "This label is verified before the report is rendered."
            ),
            "",
            "## Capacity contract",
            "",
            f"- Physical HBM capacity: {hbm_capacity / 2**30:.3f} GiB.",
            f"- HBM KV resident window: {hbm_data / 2**30:.6f} GiB.",
            f"- Reserved HBM DMA buffer: {read_buffer / 2**20:.3f} MiB.",
            f"- Saturated writes: {total_write / 2**30:.3f} GiB.",
            (
                f"- Writes exceed physical HBM by "
                f"{physical_excess / 2**30:.3f} GiB."
            ),
            (
                f"- FIFO offloading and cold readback: "
                f"{offload_bytes / 2**30:.6f} GiB."
            ),
            f"- Transfer batch: {batch_pages} pages / {batch_pages * page / 2**20:.3f} MiB.",
            "- This models mutable KV/write-back state, not read-only weights.",
            "",
            "## Parameter summary",
            "",
            "| Component | Capacity / topology | Key latency | Bandwidth / replay |",
            "|---|---|---|---|",
            *parameter_rows,
            "",
            "## D2D evidence boundary",
            "",
            (
                "- Sandisk publicly states 1.6 TB/s first-generation HBF "
                "read bandwidth per stack. It does not publish a separate "
                "HBM↔HBF D2D read bandwidth, D2D write bandwidth, or fixed "
                "D2D latency."
            ),
            (
                f"- The modeled per-stack D2D point is "
                f"{base_link['read_GBps_per_stack']:.0f} GB/s read and "
                f"{base_link['write_GBps_per_stack']:.0f} GB/s write, from "
                "the TCAD architecture design input. It is tagged "
                "`literature_derived`, not vendor-measured."
            ),
            (
                f"- At this modeled point, one 4 KiB transfer serializes in "
                f"{d2d_read_page_ns:.3f} ns on the read direction and "
                f"{d2d_write_page_ns:.3f} ns on the write direction. With "
                f"{hbf_profile['stacks']} independent stack links, the "
                f"{layer_bytes / 2**20:.0f} MiB layer has ideal D2D-only "
                f"lower bounds of {d2d_read_layer_ns / 1e6:.3f} ms read and "
                f"{d2d_write_layer_ns / 1e6:.3f} ms write; flash and HBM "
                "service are additional."
            ),
            (
                "- The configured 0 ns fixed latency means the source did "
                "not provide an additional fixed term; it is not a claim "
                "that a physical D2D path has zero latency."
            ),
            (
                f"- The two ideal D2D serialization bounds sum to "
                f"{(d2d_read_layer_ns + d2d_write_layer_ns) / 1e6:.3f} ms, "
                f"{d2d_e2e_percent:.2f}% "
                "of the HBF layer E2E. On the read-only restore half, the "
                f"{d2d_read_layer_ns / 1e6:.3f} ms bound is "
                f"{d2d_restore_percent:.2f}% "
                "of restore time. These bounds are not an additive stage "
                "breakdown because resources pipeline and overlap."
            ),
            "",
            "## Scale-simulation method",
            "",
            (
                f"Capacity and traffic are exact at the "
                f"{hbm_capacity / 2**30:.3f} GiB target. Timing is "
                "extrapolated from page-exact physical replay because "
                f"enumerating {total_write / 2**30:.3f} GiB at 4 KiB would "
                f"create {int(facts['hbm_burst_children']):,} "
                f"{int(facts['hbm_burst_bytes'])} B HBM burst children."
            ),
            (
                f"The simulator replays {document['method']['base_sample_batches']} "
                f"batches and then {document['method']['doubled_sample_batches']} "
                f"batches ({sample_bytes / 2**20:.3f} MiB in the doubled "
                "window). Per-batch timing must remain within the declared "
                f"{100 * document['method']['convergence_threshold']:.1f}% "
                "threshold before projection."
            ),
            (
                f"HBM fill is independently replayed at "
                f"{document['method']['base_sample_resident_pages'] * page / 2**20:.3f} "
                f"MiB and {document['method']['doubled_sample_resident_pages'] * page / 2**20:.3f} "
                "MiB; its ns/byte rate must pass the same convergence threshold."
            ),
            "",
            "## Projected target result",
            "",
            (
                "| backing | offloading ms | offloading GB/s | readback ms | "
                "readback GB/s | drain ms |"
            ),
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Controller-window service curve",
            "",
            (
                "Every point is an exact 1,024-page offload/readback replay "
                "with the same resident footprint. `offered / service` keeps "
                "bulk-demand queueing separate from admitted-batch service."
            ),
            "",
            (
                "| backing | window pages | offload GB/s | readback GB/s | "
                "offload p99 offered / service µs | "
                "readback p99 offered / service µs | highest utilization |"
            ),
            "|---|---:|---:|---:|---:|---:|---|",
            *service_curve_rows,
            "",
            "## Layer round-trip result",
            "",
            (
                f"The selected layer transfers {layer_bytes / 2**20:.0f} MiB "
                f"({layer_logical_bytes:,} logical bytes plus "
                f"{layer_padding_bytes:,} bytes of batch-alignment padding). "
                f"Sizing basis: `{layer_contract['sizing_basis']}`."
            ),
            (
                "The clock starts when the first layer page begins "
                "offloading. Readback starts immediately after offloading; "
                "every page is restored to its original HBM slot. The clock "
                "stops only after the last foreground HBM verification read "
                "completes, so the full layer is resident and usable in HBM."
            ),
            (
                "The Llama-based sizing fixes only the byte footprint. It "
                "does not claim that inference continuously rewrites model "
                "weights; the round-trip deliberately exercises both "
                "directions as an isolated movement comparison."
            ),
            (
                "| backing | layer offloading ms | restore-to-HBM ms | "
                "E2E ms | bidirectional effective GB/s |"
            ),
            "|---|---:|---:|---:|---:|",
            *layer_rows,
            "",
            "## Layer reuse lifecycle",
            "",
            (
                "Each row is derived only from converged layer samples using "
                "one offload plus N restores."
            ),
            "",
            "| restores | HBF total ms | CXL memory total ms | NVMe SSD total ms | winner |",
            "|---:|---:|---:|---:|---|",
            *lifecycle_rows,
            "",
            (
                "- First discrete HBF≤CXL-memory break-even: "
                + ("none in sweep." if cxl_break_even is None else f"N={cxl_break_even}.")
            ),
            (
                "- First discrete HBF≤NVMe-SSD break-even: "
                + ("none in sweep." if ssd_break_even is None else f"N={ssd_break_even}.")
            ),
            "",
            "## Comparison",
            "",
            *(f"- {item}." for item in comparisons),
            "",
            "## Replay convergence",
            "",
            (
                "| backing | offloading/batch base µs | "
                "offloading/batch doubled µs | "
                "drift | read/batch base µs | read/batch doubled µs | drift |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
            *convergence_rows,
            "",
            (
                "Layer original-slot replay convergence:"
            ),
            "",
            (
                "| backing | layer offloading/batch base µs | "
                "layer offloading/batch doubled µs | drift | "
                "restore/batch base µs | restore/batch doubled µs | drift |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
            *layer_convergence_rows,
            "",
            (
                f"HBM fill ns/byte: "
                f"{hbf['sampling']['fill_ns_per_byte_base']:.9f} "
                f"(base) vs {hbf['sampling']['fill_ns_per_byte_doubled']:.9f} "
                f"(doubled), drift "
                f"{100 * hbf['sampling']['fill_relative_drift']:.4f}%."
            ),
            "",
            "## First-principles checks",
            "",
            (
                f"- HBM fill reaches "
                f"{hbm_fill_efficiency:.1f}% "
                f"of its {float(facts['hbm_ceiling']):.1f} GB/s interface ceiling."
            ),
            (
                f"- HBF program-plane ceiling: {hbf_program_ceiling:.3f} "
                f"GB/s; projected offloading is "
                f"{hbf['write_phase']['offload_throughput_GBps']:.3f} GB/s."
            ),
            (
                f"- HBF read-plane ceiling: {hbf_read_ceiling:.3f} GB/s; "
                f"projected readback is "
                f"{hbf['read_phase']['throughput_GBps']:.3f} GB/s."
            ),
            (
                f"- HBF mapping checkpoints: {int(facts['mapping_programs']):,} "
                f"pages; payload-byte WAF {hbf_waf:.6f}; projected "
                f"checkpoint drain {hbf['timeline']['drain_tail_ns'] / 1e6:.3f} ms."
            ),
            (
                "- All capacity, page, byte, link, device, timeline, and "
                "physical-ceiling checks passed."
            ),
            "",
            "## Interpretation limits",
            "",
            (
                f"- This is a real-capacity projection, not a claim that "
                f"every one of the {total_write // page:,} source pages was "
                "individually replayed."
            ),
            (
                "- The doubled replays prove periodic timing convergence for "
                "this FIFO/batch policy; changing replacement, queue depth, "
                "or batch size requires a new convergence run."
            ),
            (
                f"- The target HBF is fresh {target_hbf_bytes / 2**40:.3f} "
                f"TiB media and offloading is "
                f"{100 * offload_bytes / target_hbf_bytes:.3f}% of capacity, "
                "so this run intentionally has no GC/erase pressure."
            ),
            (
                "- HBF payload programming blocks victim reuse, but the "
                "resident L2P checkpoint drains later; this is not a "
                "crash-consistency comparison."
            ),
            (
                "- The layer metric is an isolated immediate round-trip. "
                "It assumes the layer's original HBM slots are reclaimable "
                "when restore begins; it does not add an idle residency gap "
                "or offload another layer occupying those destination slots."
            ),
            (
                "- Layer E2E stops at HBM readability and excludes HBF "
                "mapping-checkpoint persistence, which is reported only as "
                "the separate full-experiment drain tail."
            ),
            (
                "- CXL memory is a host-attached backing profile with explicit "
                "controller, M2S/S2M wire traffic, and global device credits; "
                "it is not local socket DRAM bandwidth."
            ),
            (
                "- SSD and HBF parameters remain exploratory simulator "
                "profiles rather than calibrated product measurements."
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        type=Path,
        help="verify the artifact against this exact experiment executable",
    )
    parser.add_argument(
        "--validation-certificate",
        type=Path,
        help="verify and accept a certificate-bound paper artifact",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--scenario-compare",
        type=Path,
        help="exact scenario_compare binary bound by the certificate",
    )
    args = parser.parse_args()

    document = json.loads(args.input.read_text(encoding="utf-8"))
    require(isinstance(document, dict), "root must be an object")
    expected_validation = None
    if args.validation_certificate is not None:
        require(
            args.experiment is not None
            and args.scenario_compare is not None,
            "certificate validation requires --experiment and "
            "--scenario-compare",
        )
        try:
            certificate = verify_certificate(
                args.validation_certificate,
                repository=args.repository,
                scenario_compare=args.scenario_compare,
                overflow_offload_experiment=args.experiment,
            )
        except CertificateError as error:
            raise ValidationError(
                f"validation certificate is invalid: {error}"
            ) from error
        expected_validation = certificate.summary_block()
    by_name, facts = validate(
        document,
        expected_validation=expected_validation,
    )
    if args.experiment is not None:
        verify_executable_digest(document, args.experiment)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        render_markdown(document, by_name, facts),
        encoding="utf-8",
    )
    print("real-capacity offloading experiment validation: PASS")
    print(f"wrote: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
