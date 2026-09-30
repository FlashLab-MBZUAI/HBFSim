#!/usr/bin/env python3
"""Mutation regressions for the GC/WAF accounting verifier."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from verification.gates.waf import (  # noqa: E402
    MAPPING_ENTRIES_PER_PAGE,
    OOB,
    PAGE,
    PAGES_PER_BLOCK,
    TIME_BREAKDOWN_CONFIG,
    TOTAL_PAGES,
    WAF_DEFINITION,
    WAF_CSV_FIELDS,
    console_audit_header,
    console_audit_row,
    time_breakdown_input,
    verification_csv_row,
    verify,
    verify_random_churn_relation,
    verify_regime,
    CASES,
)


class WafVerifierMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = next(case for case in CASES if case.key == "subpage")

    def valid_result(self) -> dict:
        """Independent exact oracle for the no-GC 64 B subpage case."""
        footprint = self.case.footprint_pages
        data_programs = self.case.ops
        mapping_pages = footprint // MAPPING_ENTRIES_PER_PAGE
        page_programs = data_programs + mapping_pages
        page_reads = footprint + self.case.ops
        valid_pages = footprint + mapping_pages
        invalid_pages = page_programs
        free_pages = TOTAL_PAGES - valid_pages - invalid_pages
        auto_erases = data_programs // PAGES_PER_BLOCK
        flash_transactions = page_reads + page_programs + auto_erases
        logical_write_bytes = self.case.ops * self.case.write_bytes
        physical_write_bytes = page_programs * PAGE
        data_payload_bytes = data_programs * PAGE
        mapping_payload_bytes = mapping_pages * PAGE
        raw_codeword_bytes = page_programs * (PAGE + OOB)
        media_amp = physical_write_bytes / logical_write_bytes
        return {
            "logical_bytes": logical_write_bytes,
            "mapping_entries_per_page": MAPPING_ENTRIES_PER_PAGE,
            "stats": {
                "accounting_verified": True,
                "total_pages": TOTAL_PAGES,
                "free_pages": free_pages,
                "valid_pages": valid_pages,
                "invalid_pages": invalid_pages,
                "pending_program_pages": 0,
                "pending_mapping_publications": 0,
                "static_unmaterialized_pages": 0,
                "mapping_entries": footprint,
                "logical_read_bytes": footprint * PAGE,
                "logical_write_bytes": logical_write_bytes,
                "physical_read_bytes": page_reads * PAGE,
                "physical_write_bytes": physical_write_bytes,
                "data_program_payload_bytes": data_payload_bytes,
                "raw_physical_program_payload_bytes": 0,
                "mapping_program_payload_bytes": mapping_payload_bytes,
                "gc_relocation_payload_bytes": 0,
                "page_reads": page_reads,
                "page_programs": page_programs,
                "data_programs": data_programs,
                "mapping_page_programs": mapping_pages,
                "mapping_lookup_ops": footprint + self.case.ops,
                "mapping_user_lookup_ops": footprint + self.case.ops,
                "mapping_gc_lookup_ops": 0,
                "mapping_update_ops": data_programs,
                "mapping_user_update_ops": data_programs,
                "mapping_gc_update_ops": 0,
                "invalidations": invalid_pages,
                "gc_runs": 0,
                "block_erases": auto_erases,
                "auto_erase_requests": auto_erases,
                "erase_requests": 0,
                "gc_relocations": 0,
                "gc_data_relocations": 0,
                "gc_mapping_relocations": 0,
                "gc_reclaimed_invalid_pages": 0,
                "write_buffer_hits": 0,
                "flash_scheduler_enqueues": flash_transactions,
                "flash_scheduler_issues": flash_transactions,
                "ecc_decode_ops": page_reads,
                "ecc_encode_ops": page_programs,
                "ecc_encode_codeword_bytes": raw_codeword_bytes,
                "waf": round(media_amp, 6),
                "waf_definition": WAF_DEFINITION,
            },
        }

    def test_complete_subpage_oracle_passes(self) -> None:
        failures, detail = verify(self.case, self.valid_result())
        self.assertEqual(failures, [])
        self.assertEqual(detail["data_programs"], 8192)
        self.assertEqual(detail["page_programs"], 8208)
        self.assertEqual(detail["mapping_programs"], 16)
        self.assertEqual(detail["free_pages"], 245728)
        self.assertEqual(detail["waf"], 64.125)

    def test_missing_raw_write_denominator_is_rejected_without_crashing(self) -> None:
        broken = self.valid_result()
        del broken["stats"]["raw_physical_program_payload_bytes"]
        failures, unused_detail = verify(self.case, broken)
        self.assertTrue(any(
            "raw_physical_program_payload_bytes" in failure
            for failure in failures
        ))

    def test_cross_layer_mutations_are_rejected(self) -> None:
        # The spurious-GC mutation deliberately preserves every page-state,
        # scheduler, and GC-victim identity.  It must still fail the explicit
        # no-GC regime contract rather than slipping through local checks.
        base = self.valid_result()
        stats = base["stats"]
        coherent_spurious_gc = {
            "gc_runs": 1,
            "block_erases": 1,
            "gc_reclaimed_invalid_pages": PAGES_PER_BLOCK,
            "free_pages": stats["free_pages"] + PAGES_PER_BLOCK,
            "invalid_pages": stats["invalid_pages"] - PAGES_PER_BLOCK,
            "flash_scheduler_enqueues": stats["flash_scheduler_enqueues"] + 1,
            "flash_scheduler_issues": stats["flash_scheduler_issues"] + 1,
        }
        coherent_short_read = {
            "page_reads": stats["page_reads"] - 1,
            "physical_read_bytes": (stats["page_reads"] - 1) * PAGE,
            "ecc_decode_ops": stats["ecc_decode_ops"] - 1,
            "flash_scheduler_enqueues": stats["flash_scheduler_enqueues"] - 1,
            "flash_scheduler_issues": stats["flash_scheduler_issues"] - 1,
        }
        mutations = (
            ("spurious GC", coherent_spurious_gc, "expected no GC"),
            ("free-page loss", {"free_pages": stats["free_pages"] - 1},
             "free_pages"),
            ("coherent missing read", coherent_short_read, "page_reads"),
            ("ECC encode loss", {"ecc_encode_ops": stats["ecc_encode_ops"] - 1},
             "ECC operation counts"),
            ("unaccounted physical erase", {"block_erases": 1},
             "physical erases must equal"),
            ("scheduler loss", {
                "flash_scheduler_enqueues": stats["flash_scheduler_enqueues"] - 1,
                "flash_scheduler_issues": stats["flash_scheduler_issues"] - 1,
             }, "flash scheduler counts"),
            ("runtime audit missing", {"accounting_verified": False},
             "runtime HBF accounting audit"),
            ("physical write loss", {
                "physical_write_bytes": stats["physical_write_bytes"] - PAGE,
             }, "physical_write_bytes"),
        )
        for name, changes, expected_message in mutations:
            with self.subTest(name=name):
                broken = copy.deepcopy(base)
                broken["stats"].update(changes)
                failures, _ = verify(self.case, broken)
                self.assertTrue(failures)
                self.assertTrue(
                    any(expected_message in failure for failure in failures),
                    failures,
                )

    def test_csv_columns_are_named_ordered_and_not_shifted(self) -> None:
        failures, detail = verify(self.case, self.valid_result())
        self.assertEqual(failures, [])
        expected_fields = (
            "case", "u_eff", "waf", "waf_window_lo", "waf_window_hi",
            "logical_write_bytes", "physical_write_bytes",
            "data_programs", "page_programs", "mapping_programs",
            "gc_relocations", "gc_reclaimed_invalid_pages",
            "gc_victim_valid_fraction",
            "coalesced_writes", "gc_runs", "block_erases", "free_pages",
            "accounting_verified", "verdict",
        )
        self.assertEqual(WAF_CSV_FIELDS, expected_fields)
        row = verification_csv_row(self.case, detail, "PASS")
        self.assertEqual(tuple(row), expected_fields)
        self.assertEqual(row["logical_write_bytes"], 8192 * 64)
        self.assertEqual(row["physical_write_bytes"], 8208 * PAGE)
        self.assertEqual(row["data_programs"], 8192)
        self.assertEqual(row["page_programs"], 8208)
        self.assertEqual(row["gc_reclaimed_invalid_pages"], 0)
        self.assertEqual(row["gc_victim_valid_fraction"], "0.000000")
        self.assertEqual(row["accounting_verified"], "true")

    def test_console_and_time_metadata_expose_audit_identity(self) -> None:
        failures, detail = verify(self.case, self.valid_result())
        self.assertEqual(failures, [])
        header = console_audit_header()
        self.assertIn("回收无效", header)
        self.assertIn("free页", header)
        self.assertIn("审计", header)
        row = console_audit_row(self.case, detail, "PASS")
        self.assertIn("yes", row)
        self.assertTrue(row.endswith("PASS"))

        source = time_breakdown_input(self.case, {
            "summary": {"schema": "fixture"},
            "summary_path": Path("/tmp/subpage.summary.json"),
        })
        self.assertEqual(source.label, "subpage")
        self.assertEqual(source.scenario_metadata, {
            "all-hbf": {
                "case": "subpage",
                "config": TIME_BREAKDOWN_CONFIG,
            },
        })
        self.assertTrue(source.source.endswith("/tmp/subpage.summary.json"))

    def test_random_churn_uses_structural_not_fitted_lower_bounds(self) -> None:
        low = next(
            case for case in CASES if case.key == "rand-churn-58")
        high = next(
            case for case in CASES if case.key == "rand-churn-90")
        self.assertEqual(verify_regime(low, 1.096, 20, 2), [])
        self.assertEqual(verify_regime(high, 1.215, 40, 2), [])
        self.assertTrue(any(
            "non-zero" in failure
            for failure in verify_regime(low, 1.096, 0, 2)
        ))

    def test_random_churn_cross_occupancy_relations_are_explicit(self) -> None:
        valid = {
            "rand-churn-58": {
                "waf": 1.096,
                "gc_victim_valid_fraction": 0.10,
            },
            "rand-churn-90": {
                "waf": 1.215,
                "gc_victim_valid_fraction": 0.18,
            },
        }
        self.assertEqual(verify_random_churn_relation(valid), [])
        broken = copy.deepcopy(valid)
        broken["rand-churn-90"]["gc_victim_valid_fraction"] = 0.05
        self.assertTrue(any(
            "valid fraction" in failure
            for failure in verify_random_churn_relation(broken)
        ))



if __name__ == "__main__":
    unittest.main()
