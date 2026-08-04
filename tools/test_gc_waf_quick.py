#!/usr/bin/env python3
"""Fast, exact E2E oracle for resident HBF mapping and WAF accounting."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

from run_waf_cases import GEOMETRY, PAGE


PAGES = 4096
MAPPING_PAGES = 8
TOTAL_PAGES = 16 * 64 * 256


def require_equal(stats: dict, key: str, expected: object) -> None:
    actual = stats.get(key)
    if actual != expected:
        raise RuntimeError(f"{key}: expected {expected!r}, observed {actual!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path, required=True)
    args = parser.parse_args()
    binary = args.scenario_compare.resolve()
    if not binary.is_file():
        parser.error(f"scenario_compare not found: {binary}")

    with tempfile.TemporaryDirectory(prefix="hbfsim-gc-waf-quick-") as directory:
        root = Path(directory)
        trace = root / "resident-mapping.trace"
        lines = [f"0x{page * PAGE:x} R {PAGE}" for page in range(PAGES)]
        lines.extend(
            f"0x{page * PAGE:x} W {PAGE}" for page in range(PAGES))
        trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary = root / "summary.json"
        command = [
            str(binary),
            "--trace", str(trace),
            "--scenarios", "all-HBF",
            *GEOMETRY,
            "--max-outstanding-requests", "1",
            "--summary-json", str(summary),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=60)
        if completed.returncode != 0:
            raise RuntimeError(
                "scenario_compare failed:\n"
                f"{completed.stderr[-2000:]}\n{completed.stdout[-2000:]}")
        data = json.loads(summary.read_text(encoding="utf-8"))
        if data.get("schema") != {
                "name": "hbfsim.scenario_compare.summary", "version": 16}:
            raise RuntimeError("unexpected summary schema")
        if data.get("sanity") != "PASS":
            raise RuntimeError(f"summary sanity is {data.get('sanity')!r}")
        resolved_hbf = data.get("config", {}).get("hbf", {})
        if resolved_hbf.get("ctrl_dram_bytes") != 512 * PAGE:
            raise RuntimeError(
                "summary did not record the exact resolved resident L2P size")
        scenarios = data.get("scenarios", [])
        if len(scenarios) != 1 or scenarios[0].get("name") != "all-HBF":
            raise RuntimeError("quick oracle expected exactly one all-HBF scenario")
        stats = scenarios[0]["hbf_stats"]

        expected = {
            "read_requests": PAGES,
            "program_requests": PAGES,
            "erase_requests": 0,
            "logical_write_bytes": PAGES * PAGE,
            "physical_read_bytes": PAGES * PAGE,
            "physical_write_bytes": (PAGES + MAPPING_PAGES) * PAGE,
            "data_program_payload_bytes": PAGES * PAGE,
            "mapping_program_payload_bytes": MAPPING_PAGES * PAGE,
            "gc_relocation_payload_bytes": 0,
            "page_reads": PAGES,
            "data_programs": PAGES,
            "page_programs": PAGES + MAPPING_PAGES,
            "mapping_entries": PAGES,
            "mapping_page_programs": MAPPING_PAGES,
            "resident_mapping_table_bytes": 512 * PAGE,
            "resident_mapping_table_bytes_per_stack": 512 * PAGE,
            "resident_mapping_pages_per_stack": 512,
            "mapping_lookup_ops": 2 * PAGES,
            "mapping_user_lookup_ops": 2 * PAGES,
            "mapping_gc_lookup_ops": 0,
            "mapping_update_ops": PAGES,
            "mapping_user_update_ops": PAGES,
            "mapping_gc_update_ops": 0,
            "mapping_dram_wait_ops": 0,
            "mapping_dram_wait_work_ns": 0,
            "mapping_dram_wait_max_ns": 0,
            "mapping_dram_issue_busy_ns": 3 * PAGES,
            "mapping_dram_resources": 1,
            "invalidations": PAGES + MAPPING_PAGES,
            "block_erases": 0,
            "gc_runs": 0,
            "gc_relocations": 0,
            "gc_data_relocations": 0,
            "gc_mapping_relocations": 0,
            "gc_reclaimed_invalid_pages": 0,
            "total_pages": TOTAL_PAGES,
            "free_pages": TOTAL_PAGES - 2 * (PAGES + MAPPING_PAGES),
            "valid_pages": PAGES + MAPPING_PAGES,
            "invalid_pages": PAGES + MAPPING_PAGES,
            "pending_program_pages": 0,
            "pending_mapping_publications": 0,
            "static_unmaterialized_pages": 0,
            "accounting_verified": True,
            "ecc_decode_ops": PAGES,
            "ecc_encode_ops": PAGES + MAPPING_PAGES,
            "flash_scheduler_enqueues": 2 * PAGES + MAPPING_PAGES,
            "flash_scheduler_issues": 2 * PAGES + MAPPING_PAGES,
        }
        for key, value in expected.items():
            require_equal(stats, key, value)

        expected_amp = (PAGES + MAPPING_PAGES) / PAGES
        actual_waf = stats.get("waf")
        if not isinstance(actual_waf, (int, float)) or not math.isclose(
                actual_waf, expected_amp, rel_tol=0.0, abs_tol=5e-7):
            raise RuntimeError(
                f"waf: expected {expected_amp}, observed {actual_waf!r}")

    print("quick GC/WAF resident-mapping oracle: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
