#!/usr/bin/env python3
"""Mutation tests for the canonical Frontier/HBF WAF audit."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from frontier_waf import (
    FrontierWafError,
    WAF_DEFINITION,
    audit_frontier_hbf_waf,
)


def fixture(*, logical_bytes: int = 512) -> tuple[dict, dict, dict]:
    page_size = 512
    data_programs = 1 if logical_bytes else 0
    mapping_programs = 1 if logical_bytes else 0
    physical_bytes = (data_programs + mapping_programs) * page_size
    summary = {
        "config": {
            "hbf": {
                "page_size": page_size,
                "capacity_bytes": 20 * page_size,
            }
        }
    }
    scenario = {
        "name": "HBM+HBF-layer-streaming",
        "hbf_stats": {
            "logical_write_bytes": logical_bytes,
            "physical_write_bytes": physical_bytes,
            "data_program_payload_bytes": data_programs * page_size,
            "mapping_program_payload_bytes": mapping_programs * page_size,
            "gc_relocation_payload_bytes": 0,
            "data_programs": data_programs,
            "mapping_page_programs": mapping_programs,
            "gc_relocations": 0,
            "page_programs": data_programs + mapping_programs,
            "total_pages": 20,
            "waf": (
                physical_bytes / logical_bytes
                if logical_bytes
                else None
            ),
            "waf_definition": WAF_DEFINITION,
        },
        "layer_streaming": {"writeback_bytes": logical_bytes},
        "hybrid_path": {"hbf_backing_write_bytes": logical_bytes},
    }
    manifest = {
        "semantics": {
            "model_weight_traffic": "read_only",
            "mutable_write_traffic": "kv_append_and_update_only",
        },
        "traffic_census": {
            "writes": {
                "operations": 1 if logical_bytes else 0,
                "bytes": logical_bytes,
            }
        },
    }
    return summary, scenario, manifest


class FrontierWafTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.trace = Path(self.temporary.name) / "memory.trace"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_trace(self, *, write: bool = True, kind: str = "generated_context") -> None:
        operation = (
            f"0x0 W 512 kind={kind} label=kv.w.request.0 phase=0 layer=0 at=0\n"
            if write
            else "0x0 R 512 kind=model_weights label=weights.layer.0 phase=0 layer=0 at=0\n"
        )
        self.trace.write_text(operation, encoding="utf-8")

    def audit(
        self,
        summary: dict,
        scenario: dict,
        manifest: dict,
    ) -> dict:
        return audit_frontier_hbf_waf(
            summary=summary,
            scenario=scenario,
            manifest=manifest,
            trace_path=self.trace,
        )

    def test_nonzero_waf_and_capacity_context(self) -> None:
        summary, scenario, manifest = fixture()
        self.write_trace()
        result = self.audit(summary, scenario, manifest)
        self.assertEqual(
            {
                key: result[key]
                for key in (
                    "logical_write_bytes",
                    "physical_write_bytes",
                    "waf",
                    "physical_write_bytes_over_usable_hbf_capacity",
                )
            },
            {
                "logical_write_bytes": 512,
                "physical_write_bytes": 1024,
                "waf": 2.0,
                "physical_write_bytes_over_usable_hbf_capacity": 0.1,
            },
        )
        self.assertEqual(result["write_source"], "mutable_kv_only")
        self.assertFalse(result["initial_image_included_in_workload_writes"])

    def test_read_only_waf_is_null(self) -> None:
        summary, scenario, manifest = fixture(logical_bytes=0)
        self.write_trace(write=False)
        result = self.audit(summary, scenario, manifest)
        self.assertIsNone(result["waf"])
        self.assertEqual(result["logical_write_bytes"], 0)
        self.assertEqual(result["physical_write_bytes"], 0)

    def test_non_kv_trace_write_is_rejected(self) -> None:
        summary, scenario, manifest = fixture()
        self.write_trace(kind="model_weights")
        with self.assertRaisesRegex(FrontierWafError, "non-KV write"):
            self.audit(summary, scenario, manifest)

    def test_payload_breakdown_mutation_is_rejected(self) -> None:
        summary, scenario, manifest = fixture()
        self.write_trace()
        broken = copy.deepcopy(scenario)
        broken["hbf_stats"]["mapping_program_payload_bytes"] += 512
        with self.assertRaisesRegex(FrontierWafError, "conservation"):
            self.audit(summary, broken, manifest)

    def test_reported_waf_mutation_is_rejected(self) -> None:
        summary, scenario, manifest = fixture()
        self.write_trace()
        broken = copy.deepcopy(scenario)
        broken["hbf_stats"]["waf"] = 1.0
        with self.assertRaisesRegex(FrontierWafError, "reported WAF"):
            self.audit(summary, broken, manifest)

    def test_zero_denominator_cannot_serialize_zero(self) -> None:
        summary, scenario, manifest = fixture(logical_bytes=0)
        self.write_trace(write=False)
        broken = copy.deepcopy(scenario)
        broken["hbf_stats"]["waf"] = 0
        with self.assertRaisesRegex(FrontierWafError, "null WAF"):
            self.audit(summary, broken, manifest)


if __name__ == "__main__":
    unittest.main()
