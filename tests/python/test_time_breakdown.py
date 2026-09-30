#!/usr/bin/env python3
"""End-to-end contract for schema-v19 canonical time accounting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path


SCENARIOS = (
    "all-hbm",
    "all-hbf",
    "flat",
    "hbf-streaming",
    "external-streaming",
)

# HBM is a bounded channel aggregate (e76cb82): the summary CSV publishes six
# hbm_* work columns, each mirroring one field of the generic stage_work
# record that the summary JSON (schema v19) shares with the HBF and external
# stages. access_latency is the aggregate per-access latency the device books
# as command_ns ("no ACT/PRE trace"); efficiency_overhead is the service time
# beyond payload transfer, booked as maintenance_ns.
HBM_STAGE_CSV_FIELDS = {
    "hbm_address_mapping_work_ns": "address_mapping_work_ns",
    "hbm_scheduler_queue_wait_work_ns": "scheduler_queue_wait_work_ns",
    "hbm_access_latency_work_ns": "command_work_ns",
    "hbm_efficiency_overhead_work_ns": "maintenance_work_ns",
    "hbm_channel_transfer_work_ns": "channel_transfer_work_ns",
    "hbm_total_overlapping_work_ns": "total_overlapping_work_ns",
}
# Row-level DRAM scheduling no longer exists for HBM; these generic stages
# stay in the shared JSON record and must read zero for HBM.
HBM_RETIRED_STAGE_FIELDS = (
    "refresh_stall_work_ns",
    "precharge_work_ns",
    "activation_work_ns",
)

HBF_STAGE_FIELDS = (
    "ingress_queue_wait_work_ns",
    "scheduler_queue_wait_work_ns",
    "ecc_queue_wait_work_ns",
    "address_mapping_work_ns",
    "translation_work_ns",
    "mapping_dram_work_ns",
    "write_buffer_dram_work_ns",
    "command_work_ns",
    "array_read_work_ns",
    "array_program_work_ns",
    "array_erase_work_ns",
    "media_lane_transfer_work_ns",
    "page_buffer_work_ns",
    "sram_staging_work_ns",
    "channel_transfer_work_ns",
    "tsv_transfer_work_ns",
    "hb_io_transfer_work_ns",
    "transport_latency_work_ns",
    "ecc_response_latency_work_ns",
    "maintenance_work_ns",
    "total_overlapping_work_ns",
)

HBF_STATS_CSV_FIELDS = (
    ("hbf_ecc_decode_ops", "ecc_decode_ops"),
    ("hbf_ecc_encode_ops", "ecc_encode_ops"),
    ("hbf_ecc_codeword_bytes", "ecc_codeword_bytes"),
    ("hbf_ecc_issue_parallelism", "ecc_issue_parallelism"),
    ("hbf_ecc_max_inflight_per_die", "ecc_max_inflight_per_die"),
    ("hbf_write_buffer_slot_wait_work_ns", "write_buffer_slot_wait_work_ns"),
)

EXTERNAL_STAGE_FIELDS = (
    "ingress_queue_wait_work_ns",
    "scheduler_queue_wait_work_ns",
    "command_work_ns",
    "array_read_work_ns",
    "array_program_work_ns",
    "channel_transfer_work_ns",
    "hb_io_transfer_work_ns",
    "transport_latency_work_ns",
    "total_overlapping_work_ns",
)

CONTROLLER_WORK_FIELDS = (
    "backing_admission_wait_work_ns",
    "backing_admission_max_wait_ns",
    "user_wait_work_ns",
    "user_max_wait_ns",
    "exposed_prefetch_ns",
    "hidden_prefetch_ns",
    "buffer_reuse_wait_work_ns",
)

COOPERATIVE_CONTROLLER_FIELDS = (
    "full_waited_ops",
    "full_wait_work_ns",
)

RESOURCE_CSV_FIELDS = {
    "hbm_data_bus": (
        "hbm_data_bus_busy_ns", "hbm_data_bus_resource_count",
        "hbm_data_bus_active_span_ns", "hbm_data_bus_capacity_time_ns",
        "hbm_data_bus_utilization"),
    "external_controller": (
        "external_controller_busy_ns", "external_controller_resource_count",
        "external_controller_active_span_ns",
        "external_controller_capacity_time_ns",
        "external_controller_utilization"),
    "external_media": (
        "external_media_busy_ns", "external_media_resource_count",
        "external_media_active_span_ns", "external_media_capacity_time_ns",
        "external_media_utilization"),
    "external_link_m2s": (
        "external_link_m2s_busy_ns", "external_link_m2s_resource_count",
        "external_link_m2s_active_span_ns",
        "external_link_m2s_capacity_time_ns",
        "external_link_m2s_utilization"),
    "external_link_s2m": (
        "external_link_s2m_busy_ns", "external_link_s2m_resource_count",
        "external_link_s2m_active_span_ns",
        "external_link_s2m_capacity_time_ns",
        "external_link_s2m_utilization"),
    "hbf_logic_ingress": (
        "hbf_logic_ingress_busy_ns", "hbf_logic_ingress_resource_count",
        "hbf_logic_ingress_active_span_ns",
        "hbf_logic_ingress_capacity_time_ns",
        "hbf_logic_ingress_utilization"),
    "hbf_mapping_dram_issue": (
        "hbf_mapping_dram_issue_busy_ns",
        "hbf_mapping_dram_issue_resource_count",
        "hbf_mapping_dram_issue_active_span_ns",
        "hbf_mapping_dram_issue_capacity_time_ns",
        "hbf_mapping_dram_issue_utilization"),
    "hbf_plane_media": (
        "hbf_plane_media_busy_ns", "hbf_plane_media_resource_count",
        "hbf_plane_media_active_span_ns", "hbf_plane_media_capacity_time_ns",
        "hbf_plane_media_utilization"),
    "hbf_media_lane": (
        "hbf_media_lane_busy_ns", "hbf_media_lane_resource_count",
        "hbf_media_lane_active_span_ns", "hbf_media_lane_capacity_time_ns",
        "hbf_media_lane_utilization"),
    "hbf_subarray": (
        "hbf_subarray_busy_ns", "hbf_subarray_resource_count",
        "hbf_subarray_active_span_ns", "hbf_subarray_capacity_time_ns",
        "hbf_subarray_utilization"),
    "hbf_page_buffer_bank": (
        "hbf_page_buffer_bank_busy_ns", "hbf_page_buffer_bank_resource_count",
        "hbf_page_buffer_bank_active_span_ns",
        "hbf_page_buffer_bank_capacity_time_ns",
        "hbf_page_buffer_bank_utilization"),
    "hbf_flash_source_queue": (
        "hbf_flash_source_queue_busy_ns",
        "hbf_flash_source_queue_resource_count",
        "hbf_flash_source_queue_active_span_ns",
        "hbf_flash_source_queue_capacity_time_ns",
        "hbf_flash_source_queue_utilization"),
    "hbf_channel_command": (
        "hbf_channel_command_busy_ns", "hbf_channel_command_resource_count",
        "hbf_channel_command_active_span_ns",
        "hbf_channel_command_capacity_time_ns",
        "hbf_channel_command_utilization"),
    "hbf_channel_data": (
        "hbf_channel_data_busy_ns", "hbf_channel_data_resource_count",
        "hbf_channel_data_active_span_ns",
        "hbf_channel_data_capacity_time_ns",
        "hbf_channel_data_utilization"),
    "hbf_tsv": (
        "hbf_tsv_busy_ns", "hbf_tsv_resource_count",
        "hbf_tsv_active_span_ns", "hbf_tsv_capacity_time_ns",
        "hbf_tsv_utilization"),
    "hbf_sram": (
        "hbf_sram_busy_ns", "hbf_sram_resource_count",
        "hbf_sram_active_span_ns", "hbf_sram_capacity_time_ns",
        "hbf_sram_utilization"),
    "hbf_hbio_command": (
        "hbf_hbio_command_busy_ns", "hbf_hbio_command_resource_count",
        "hbf_hbio_command_active_span_ns",
        "hbf_hbio_command_capacity_time_ns",
        "hbf_hbio_command_utilization"),
    "hbf_hbio_data": (
        "hbf_hbio_data_busy_ns", "hbf_hbio_data_resource_count",
        "hbf_hbio_data_active_span_ns", "hbf_hbio_data_capacity_time_ns",
        "hbf_hbio_data_utilization"),
    "hbf_sequencer": (
        "hbf_sequencer_busy_ns", "hbf_sequencer_resource_count",
        "hbf_sequencer_active_span_ns", "hbf_sequencer_capacity_time_ns",
        "hbf_sequencer_utilization"),
    "hbf_ecc_issue": (
        "hbf_ecc_issue_busy_ns", "hbf_ecc_issue_resource_count",
        "hbf_ecc_issue_active_span_ns", "hbf_ecc_issue_capacity_time_ns",
        "hbf_ecc_issue_utilization"),
    "base_die_link_read": (
        "base_die_link_read_busy_ns", "base_die_link_read_resource_count",
        "base_die_link_read_active_span_ns",
        "base_die_link_read_capacity_time_ns",
        "base_die_link_read_utilization"),
    "base_die_link_write": (
        "base_die_link_write_busy_ns", "base_die_link_write_resource_count",
        "base_die_link_write_active_span_ns",
        "base_die_link_write_capacity_time_ns",
        "base_die_link_write_utilization"),
}

def close_enough(left: float, right: float) -> bool:
    # The canonical JSON retains full precision; summary CSV uses two decimals.
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=0.005001)


def compare_csv_scalar(row: dict[str, str], field: str, value: float) -> None:
    if field not in row:
        raise AssertionError(f"summary CSV is missing {field}")
    if row[field] == "":
        raise AssertionError(f"summary CSV unexpectedly left {field} empty")
    csv_value = float(row[field])
    if not close_enough(float(value), csv_value):
        raise AssertionError(
            f"JSON/CSV mismatch for {field}: {value!r} != {csv_value!r}")


def check_wall_and_latency(scenario: dict) -> None:
    time = scenario["time_breakdown"]
    wall = time["wall_clock_ns"]
    latency = time["latency_work"]
    if not math.isclose(
            wall["offered_arrival_span_ns"] +
            wall["post_offer_user_completion_tail_ns"],
            wall["user_completion_span_ns"], rel_tol=1e-12, abs_tol=1e-9):
        raise AssertionError("arrival span plus completion tail is not the user span")
    if not math.isclose(
            wall["user_completion_span_ns"] + wall["drain_tail_ns"],
            wall["makespan_ns"], rel_tol=1e-12, abs_tol=1e-9):
        raise AssertionError("user span plus drain tail is not the makespan")
    if not math.isclose(
            latency["service_to_user_completion_sum_work_ns"] +
            latency["front_end_admission_wait_work_ns"],
            latency["offered_to_user_completion_sum_work_ns"],
            rel_tol=1e-12, abs_tol=1e-9):
        raise AssertionError("offered latency work does not include admission work")
    if not math.isclose(
            latency["offered_to_user_completion_sum_work_ns"] +
            latency["phase_dependency_wait_work_ns"],
            latency["source_to_user_completion_sum_work_ns"],
            rel_tol=1e-12, abs_tol=1e-9):
        raise AssertionError("source latency work does not include phase wait")
    count = latency["user_count"]
    for average, total in (
            ("service_average_ns",
             "service_to_user_completion_sum_work_ns"),
            ("average_ns", "offered_to_user_completion_sum_work_ns"),
            ("source_average_ns",
             "source_to_user_completion_sum_work_ns")):
        expected = latency[total] / count if count else 0.0
        if not math.isclose(
                latency[average], expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(
                f"{average} does not match its latency-work total")
    if not (
            latency["service_average_ns"] <= latency["average_ns"] <=
            latency["source_average_ns"]):
        raise AssertionError("service/offered/source latency is not causal")


def check_resources(scenario: dict) -> None:
    for name, metric in scenario["time_breakdown"]["resource_busy"].items():
        busy = float(metric["busy_ns"])
        count = int(metric["resource_count"])
        span = float(metric["active_span_ns"])
        capacity = float(metric["capacity_time_ns"])
        utilization = float(metric["utilization"])
        expected_capacity = count * span
        expected_utilization = busy / capacity if capacity > 0.0 else 0.0
        if not all(math.isfinite(value) and value >= 0.0 for value in
                   (busy, span, capacity, utilization)):
            raise AssertionError(f"{name} has invalid resource accounting")
        if not math.isclose(capacity, expected_capacity,
                            rel_tol=1e-12, abs_tol=1e-9):
            raise AssertionError(f"{name} capacity-time is inconsistent")
        if busy > capacity + 1e-9:
            raise AssertionError(f"{name} busy time exceeds capacity-time")
        if not math.isclose(utilization, expected_utilization,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{name} utilization is inconsistent")


def check_address_heatmap(scenario: dict) -> None:
    heatmap = scenario["address_heatmap"]
    if heatmap["schema"] != "hbfsim.address_heatmap.v1":
        raise AssertionError("scenario emitted the wrong address heatmap schema")
    if heatmap["bin_count"] != 32:
        raise AssertionError("address heatmap bin count did not follow the run config")
    domains = {domain["domain"]: domain for domain in heatmap["domains"]}
    if set(domains) != {
            "workload_logical", "hbm_physical", "hbf_logical", "hbf_physical",
            "external_physical"}:
        raise AssertionError("address heatmap did not emit its five canonical domains")

    for name, domain in domains.items():
        totals = domain["totals"]
        for direction in ("read", "write", "erase"):
            byte_field = f"{direction}_bytes"
            source_bytes = sum(item[byte_field] for item in domain["source_totals"])
            bin_bytes = sum(item[byte_field] for item in domain["bins"])
            if source_bytes != totals[byte_field] or bin_bytes != totals[byte_field]:
                raise AssertionError(
                    f"{name} {direction} bytes do not conserve across source/bin views")

    workload = domains["workload_logical"]["totals"]
    if workload["read_bytes"] + workload["write_bytes"] != scenario["logical_bytes"]:
        raise AssertionError("workload heatmap does not conserve logical bytes")
    hbm = domains["hbm_physical"]["totals"]
    hbm_stats = scenario["hbm_stats"]
    if hbm_stats is None:
        if hbm["read_bytes"] or hbm["write_bytes"]:
            raise AssertionError("HBM-less scenario emitted HBM physical traffic")
    elif (hbm["read_bytes"], hbm["write_bytes"]) != (
            hbm_stats["read_bytes"], hbm_stats["write_bytes"]):
        raise AssertionError("HBM physical heatmap differs from HBM statistics")
    hbf = domains["hbf_physical"]["totals"]
    hbf_stats = scenario["hbf_stats"]
    if hbf_stats is None:
        if hbf["read_bytes"] or hbf["write_bytes"] or hbf["erase_bytes"]:
            raise AssertionError("HBF-less scenario emitted HBF physical traffic")
    else:
        expected_erase = hbf_stats["block_erases"] * 64 * 4096
        if (hbf["read_bytes"], hbf["write_bytes"], hbf["erase_bytes"]) != (
                hbf_stats["physical_read_bytes"],
                hbf_stats["physical_write_bytes"], expected_erase):
            raise AssertionError("HBF physical heatmap differs from HBF statistics")
    external = domains["external_physical"]["totals"]
    external_stats = scenario["external_backing_stats"]
    if external_stats is None:
        if external["read_bytes"] or external["write_bytes"]:
            raise AssertionError(
                "external-backing-free scenario emitted external traffic")
    elif (external["read_bytes"], external["write_bytes"]) != (
            external_stats["read_bytes"], external_stats["write_bytes"]):
        raise AssertionError(
            "external physical heatmap differs from external statistics")


def check_csv_parity(scenario: dict, row: dict[str, str]) -> None:
    time = scenario["time_breakdown"]
    wall = time["wall_clock_ns"]
    latency = time["latency_work"]
    for field, value in wall.items():
        compare_csv_scalar(row, field, value)
    latency_fields = {
        "service_to_user_completion_sum_work_ns":
            latency["service_to_user_completion_sum_work_ns"],
        "front_end_admission_waited_ops": latency["front_end_admission_waited_ops"],
        "front_end_admission_wait_work_ns":
            latency["front_end_admission_wait_work_ns"],
        "front_end_admission_max_wait_ns":
            latency["front_end_admission_max_wait_ns"],
        "offered_to_user_completion_sum_work_ns":
            latency["offered_to_user_completion_sum_work_ns"],
        "user_latency_average_ns": latency["average_ns"],
        "user_latency_p50_ns": latency["p50_ns"],
        "user_latency_p95_ns": latency["p95_ns"],
        "user_latency_max_ns": latency["max_ns"],
        "service_latency_average_ns": latency["service_average_ns"],
        "service_latency_p50_ns": latency["service_p50_ns"],
        "service_latency_p95_ns": latency["service_p95_ns"],
        "service_latency_max_ns": latency["service_max_ns"],
        "source_latency_average_ns": latency["source_average_ns"],
        "source_latency_p50_ns": latency["source_p50_ns"],
        "source_latency_p95_ns": latency["source_p95_ns"],
        "source_latency_max_ns": latency["source_max_ns"],
        "phase_barriers": latency["phase_barriers"],
        "phase_dependency_waited_ops":
            latency["phase_dependency_waited_ops"],
        "phase_dependency_wait_work_ns":
            latency["phase_dependency_wait_work_ns"],
        "phase_dependency_max_wait_ns":
            latency["phase_dependency_max_wait_ns"],
        "source_to_user_completion_sum_work_ns":
            latency["source_to_user_completion_sum_work_ns"],
        "user_completion_throughput_GBps":
            scenario["user_completion_throughput_GBps"],
        "makespan_throughput_GBps": scenario["makespan_throughput_GBps"],
    }
    for field, value in latency_fields.items():
        compare_csv_scalar(row, field, value)
    if int(row["ops"]) != latency["user_count"]:
        raise AssertionError("CSV ops does not match latency-work user_count")

    stage = time["stage_work"]
    hbm_work = stage["hbm"]
    published_hbm_columns = {
        column for column in row if re.fullmatch(r"hbm_.*_work_ns", column)}
    if published_hbm_columns != set(HBM_STAGE_CSV_FIELDS):
        raise AssertionError(
            "summary CSV HBM work columns drifted from the channel-aggregate "
            f"contract: {sorted(published_hbm_columns)}")
    for csv_field, json_field in HBM_STAGE_CSV_FIELDS.items():
        if hbm_work is None:
            if row.get(csv_field) != "":
                raise AssertionError(f"absent hbm emitted {csv_field}")
        else:
            compare_csv_scalar(row, csv_field, hbm_work[json_field])
    if hbm_work is not None:
        for field in HBM_RETIRED_STAGE_FIELDS:
            if hbm_work[field] != 0:
                raise AssertionError(
                    f"channel-aggregate HBM booked retired stage {field}")
    for device, fields in (
            ("hbf", HBF_STAGE_FIELDS),
            ("external_backing", EXTERNAL_STAGE_FIELDS)):
        work = stage[device]
        for field in fields:
            csv_field = f"{device}_{field}"
            if work is None:
                if row.get(csv_field) != "":
                    raise AssertionError(f"absent {device} emitted {csv_field}")
            else:
                compare_csv_scalar(row, csv_field, work[field])

    for field in (
            "read_queue_wait_work_ns", "write_queue_wait_work_ns",
            "read_serialization_work_ns", "write_serialization_work_ns",
            "read_fixed_latency_work_ns", "write_fixed_latency_work_ns"):
        compare_csv_scalar(
            row, f"base_die_link_{field}", stage["base_die_link"][field])

    controller = stage["layer_streaming_controller"]
    for field in CONTROLLER_WORK_FIELDS:
        csv_field = f"layer_streaming_{field}"
        if controller["present"]:
            compare_csv_scalar(row, csv_field, controller[field])
        elif row.get(csv_field) != "":
            raise AssertionError(f"non-streaming scenario emitted {csv_field}")

    cooperative = stage["cooperative_write_controller"]
    for field in COOPERATIVE_CONTROLLER_FIELDS:
        csv_field = f"cooperative_write_controller_{field}"
        if cooperative["present"]:
            compare_csv_scalar(row, csv_field, cooperative[field])
        elif row.get(csv_field) != "":
            raise AssertionError(
                f"scenario without cooperative controller emitted {csv_field}")

    hbf_work = stage["hbf"]
    if hbf_work is not None:
        hbf_stats = scenario["hbf_stats"]
        for csv_field, json_field in HBF_STATS_CSV_FIELDS:
            compare_csv_scalar(row, csv_field, hbf_stats[json_field])
        write_buffer = hbf_work["write_buffer"]
        if write_buffer["slot_wait_work_ns"] > (
                hbf_work["scheduler_queue_wait_work_ns"] + 1e-9):
            raise AssertionError(
                "HBF write-buffer wait exceeds its scheduler-wait parent")
        compare_csv_scalar(
            row,
            "hbf_write_buffer_slot_waited_ops",
            write_buffer["slot_waited_ops"],
        )
        compare_csv_scalar(
            row,
            "hbf_write_buffer_slot_wait_work_ns",
            write_buffer["slot_wait_work_ns"],
        )
        for direction in ("decode", "encode"):
            diagnostic = hbf_work["ecc_directional"][direction]
            for json_field, csv_suffix in (
                    ("queue_wait_work_ns", "queue_wait_work_ns"),
                    ("response_latency_work_ns", "response_latency_work_ns"),
                    ("issue_busy_ns", "issue_busy_ns"),
                    ("ops", "ops"),
                    ("codeword_bytes", "codeword_bytes")):
                compare_csv_scalar(
                    row,
                    f"hbf_ecc_{direction}_{csv_suffix}",
                    diagnostic[json_field],
                )
        directional = hbf_work["ecc_directional"]
        for child_field, parent_field in (
                ("queue_wait_work_ns", "ecc_queue_wait_work_ns"),
                ("response_latency_work_ns",
                 "ecc_response_latency_work_ns")):
            child_sum = sum(
                directional[direction][child_field]
                for direction in ("decode", "encode"))
            if not math.isclose(
                    child_sum, hbf_work[parent_field],
                    rel_tol=1e-12, abs_tol=1e-9):
                raise AssertionError(
                    f"HBF ECC directional {child_field} does not conserve")
        ecc_issue_sum = sum(
            directional[direction]["issue_busy_ns"]
            for direction in ("decode", "encode"))
        if not math.isclose(
                ecc_issue_sum,
                time["resource_busy"]["hbf_ecc_issue"]["busy_ns"],
                rel_tol=1e-12, abs_tol=1e-9):
            raise AssertionError("HBF directional ECC issue busy does not conserve")

    for resource, fields in RESOURCE_CSV_FIELDS.items():
        metric = time["resource_busy"][resource]
        for csv_field, json_field in zip(
                fields,
                ("busy_ns", "resource_count", "active_span_ns",
                 "capacity_time_ns", "utilization")):
            if metric["resource_count"] == 0:
                if row.get(csv_field) != "":
                    raise AssertionError(f"absent resource emitted {csv_field}")
            else:
                compare_csv_scalar(row, csv_field, metric[json_field])


def run_mode(
        binary: Path, root: Path, trace: Path,
        mode: str) -> tuple[dict, dict, str]:
    summary_json = root / f"{mode}.json"
    summary_csv = root / f"{mode}.csv"
    command = [
        str(binary),
        "--trace", str(trace),
        "--scenarios", ",".join(SCENARIOS),
        "--trace-mode", mode,
        "--summary-json", str(summary_json),
        "--summary-csv", str(summary_csv),
        "--line-size", "4096",
        "--flat-hbm-bytes", "65536",
        "--static-direct-hbm-bytes", "65536",
        "--max-outstanding-requests", "2",
        "--address-heatmap-bins", "32",
        "--layer-buffer-bytes", "32768",
        "--hbf-page-size", "4096",
        "--hbf-oob-bytes", "224",
        "--hbf-stacks", "1",
        "--hbf-channels", "1",
        "--hbf-dies-per-channel", "1",
        "--hbf-planes-per-die", "2",
        "--hbf-blocks-per-plane", "32",
        "--hbf-pages-per-block", "64",
        "--hbf-gc-low-watermark-pages", "16",
        "--hbf-gc-reserved-free-blocks-per-plane", "1",
        "--hbf-write-coalescing", "true",
        "--hbf-write-buffer-pages", "2",
        "--hbf-hbm-write-buffer-bytes", "4096",
    ]
    if mode == "full":
        command.extend(("--chrome-trace", str(root / "full.trace.json")))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError(
            f"trace-mode={mode} failed:\n{completed.stdout}\n{completed.stderr}")
    summary = json.loads(summary_json.read_text())
    if summary.get("schema") != {
            "name": "hbfsim.simulation.summary", "version": 19}:
        raise AssertionError(f"trace-mode={mode} emitted the wrong summary schema")
    scheme = summary["config"]["hbm"]["address_mapping_scheme"]
    if not re.fullmatch(r"[a-z0-9-]+-v[0-9]+", scheme):
        raise AssertionError("summary lost the versioned HBM address map")
    if f"hbm_address_mapping_scheme={scheme}" not in completed.stdout:
        raise AssertionError(
            "summary and run-config disagree on the HBM address map")
    with summary_csv.open(newline="") as handle:
        rows = {row["scenario"]: row for row in csv.DictReader(handle)}
    scenarios = {scenario["name"]: scenario for scenario in summary["scenarios"]}
    if set(scenarios) != set(SCENARIOS) or set(rows) != set(SCENARIOS):
        raise AssertionError(f"trace-mode={mode} lost a requested scenario")
    return scenarios, rows, completed.stdout


def first_difference(lhs: object, rhs: object, path: str = "time_breakdown") -> str:
    if isinstance(lhs, float) and isinstance(rhs, float):
        return "" if math.isclose(
            lhs, rhs, rel_tol=1e-12, abs_tol=1e-9) else (
                f"{path}: {lhs!r} != {rhs!r}")
    if isinstance(lhs, dict) and isinstance(rhs, dict):
        if lhs.keys() != rhs.keys():
            return f"{path} keys: {sorted(lhs)} != {sorted(rhs)}"
        for key in lhs:
            difference = first_difference(lhs[key], rhs[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(lhs, list) and isinstance(rhs, list):
        if len(lhs) != len(rhs):
            return f"{path} lengths: {len(lhs)} != {len(rhs)}"
        for index, (left, right) in enumerate(zip(lhs, rhs)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    return "" if lhs == rhs else f"{path}: {lhs!r} != {rhs!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trace = root / "time-breakdown.trace"
        trace.write_text(
            "0x0 R 4096 model_weights at=100 layer=0\n"
            "0x1000 R 4096 model_weights at=100 layer=0\n"
            "0x10000 R 4096 shared_context at=100 layer=0\n"
            "0x11000 R 4096 shared_context at=100 layer=0\n"
            "0x10000 R 4096 shared_context at=200 layer=1\n"
            "0x20000 W 4096 generated_context at=200 layer=1\n"
            "0x21000 W 4096 generated_context at=200 layer=1\n"
            "0x3000 W 4096 scratch at=300 layer=1\n"
        )
        off, off_rows, off_console = run_mode(
            args.simulator.resolve(), root, trace, "off")
        full, full_rows, full_console = run_mode(
            args.simulator.resolve(), root, trace, "full")

        # A medium absent from a scenario has no hottest address. Do not print
        # the arbitrary first zero-valued bin as if it were a measured hotspot.
        for console in (off_console, full_console):
            for field in (
                    *(item[0] for item in HBF_STATS_CSV_FIELDS),
                    "hbf_ingress_queue_wait_work_ns",
                    "hbf_scheduler_queue_wait_work_ns",
                    "hbf_ecc_queue_wait_work_ns"):
                if field not in console:
                    raise AssertionError(
                        f"console comparison table is missing {field}")
            zero_rows = re.findall(
                r"^\s+(?:hbm_physical|hbf_logical|hbf_physical|external_physical),"
                r"[0-9]+,0,0,0,0,32,(.*)$", console, re.MULTILINE,
            )
            if not zero_rows:
                raise AssertionError("fixture did not exercise an idle heatmap domain")
            for row in zero_rows:
                if row != "n/a,0,n/a,0":
                    raise AssertionError(
                        "zero-traffic console row reported a fake hotspot")

        for name in SCENARIOS:
            difference = first_difference(
                off[name]["time_breakdown"],
                full[name]["time_breakdown"])
            if difference:
                raise AssertionError(
                    f"trace instrumentation changed {name} time_breakdown: "
                    f"{difference}")
            if off[name]["address_heatmap"] != full[name]["address_heatmap"]:
                raise AssertionError(
                    f"trace instrumentation changed {name} address_heatmap")
            scenario = off[name]
            check_wall_and_latency(scenario)
            check_resources(scenario)
            check_address_heatmap(scenario)
            check_csv_parity(scenario, off_rows[name])
            check_csv_parity(full[name], full_rows[name])

        hbf_pressure = off["all-hbf"]
        hbf_stage = hbf_pressure["time_breakdown"]["stage_work"]["hbf"]
        hbf_stats = hbf_pressure["hbf_stats"]
        hbf_resources = hbf_pressure["time_breakdown"]["resource_busy"]
        if (
            hbf_stage["ingress_queue_wait_work_ns"] <= 0.0
            or hbf_stage["scheduler_queue_wait_work_ns"] <= 0.0
            or hbf_stats["ecc_encode_ops"] <= 0
            or hbf_resources["hbf_ecc_issue"]["busy_ns"] <= 0.0
        ):
            raise AssertionError(
                "time-breakdown trace did not exercise HBF queue/ECC pressure")

        cooperative = off["flat"]["time_breakdown"][
            "stage_work"]["cooperative_write_controller"]
        if (not cooperative["present"] or
                cooperative["full_waited_ops"] == 0 or
                cooperative["full_wait_work_ns"] <= 0.0):
            raise AssertionError(
                "time-breakdown regression did not exercise cooperative "
                "write-controller backpressure")

        def source_total(scenario: dict, domain_name: str, source_name: str) -> dict:
            domain = next(
                item for item in scenario["address_heatmap"]["domains"]
                if item["domain"] == domain_name)
            return next(
                item for item in domain["source_totals"]
                if item["source"] == source_name)

        streaming_trace = root / "streaming-provenance.trace"
        streaming_trace.write_text(
            "0x1000 R 4096 model_weights layer=0 at=0\n"
            "0x2000 R 4096 model_weights layer=1 at=0\n"
            "0x20000 W 4096 generated_context layer=2 at=0\n"
            "0x30000 R 4096 model_weights layer=3 at=0\n"
            "0x40000 R 4096 model_weights layer=4 at=0\n"
            "0x50000 R 4096 model_weights layer=5 at=0\n")
        streaming_summary = root / "streaming-provenance.json"
        streaming_run = subprocess.run([
            str(args.simulator.resolve()),
            "--trace", str(streaming_trace),
            "--scenarios", "hbf-streaming",
            "--summary-json", str(streaming_summary),
            "--hbm-capacity-bytes", str(16384 + 8 * 1024 * 1024),
            "--hbf-stacks", "1", "--hbf-channels", "1",
            "--hbf-dies-per-channel", "1", "--hbf-planes-per-die", "1",
            "--hbf-blocks-per-plane", "512", "--hbf-pages-per-block", "64",
            "--hbf-ctrl-dram-bytes", str(8 * 1024 * 1024),
            "--layer-buffer-bytes", "4096",
            "--hbf-page-size", "4096",
            "--hbf-oob-bytes", "224",
            # Source provenance lives in the heatmap, which is off by default.
            "--address-heatmap-bins", "32",
        ], capture_output=True, text=True, timeout=120)
        if streaming_run.returncode != 0:
            raise RuntimeError(
                "streaming provenance run failed:\n" +
                streaming_run.stdout + streaming_run.stderr)
        streaming_result = json.loads(
            streaming_summary.read_text())["scenarios"][0]
        if streaming_result["layer_streaming"]["mode"] != "layer_streaming":
            raise AssertionError(
                "writeback provenance trace did not select layer streaming")
        streaming_destage = source_total(
            streaming_result, "hbm_physical", "destage")
        if streaming_destage["read_bytes"] == 0:
            raise AssertionError("layer writeback bypassed its required HBM read")
        all_hbf_direct = source_total(off["all-hbf"], "hbf_physical", "direct")
        all_hbf_destage = source_total(off["all-hbf"], "hbf_physical", "destage")
        if all_hbf_direct["write_bytes"] == 0 or all_hbf_destage["write_bytes"] != 0:
            raise AssertionError(
                "all-hbf deferred Direct writes lost their persisted provenance")

    print("schema-v19 time accounting is invariant, additive, and JSON/CSV-consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
