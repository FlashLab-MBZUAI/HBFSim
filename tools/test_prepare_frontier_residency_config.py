#!/usr/bin/env python3
"""Regression tests for the Frontier object-map residency binding."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prepare_frontier_residency_config import (
    ResidencyBindingError,
    _compile_trace_fitted_partition,
    prepare_residency_config,
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


class FrontierResidencyBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "memory.manifest.json"
        self.object_map_path = self.root / "objects.json"
        self.trace_path = self.root / "memory.trace"
        self.config_path = self.root / "residency.cfg"
        self.receipt_path = self.root / "residency.receipt.json"
        self.trace_payload = (
            b"# frontier-memory-trace\n"
            b"0x0 R 4096 model_weights phase=0 layer=0\n"
        )
        self.trace_path.write_bytes(self.trace_payload)
        self.object_map = {
            "schema": {
                "name": "hbfsim.memory_object_map",
                "version": 4,
            },
            "address_space": {
                "begin": 0,
                "end": 49152,
                "bytes": 49152,
                "object_alignment_bytes": 4096,
            },
            "regions": [
                {
                    "name": "model.embedding",
                    "kind": "model_weights",
                    "begin": 0,
                    "end": 5000,
                    "bytes": 5000,
                    "alignment_bytes": 4096,
                    "placement_policy": "model_weight",
                },
                {
                    "name": "model.layer.0",
                    "kind": "model_weights",
                    "begin": 8192,
                    "end": 12288,
                    "bytes": 4096,
                    "alignment_bytes": 4096,
                    "placement_policy": "model_weight",
                },
                {
                    "name": "kv.physical_block_slots",
                    "kind": "generated_context",
                    "begin": 12288,
                    "end": 36864,
                    "bytes": 24576,
                    "alignment_bytes": 4096,
                    "placement_policy": "frontier_physical_block_id",
                    "num_slots": 3,
                    "bytes_per_slot": 8192,
                },
                {
                    "name": "metadata.kv_allocator_blocks",
                    "kind": "metadata",
                    "begin": 36864,
                    "end": 36876,
                    "bytes": 12,
                    "alignment_bytes": 4096,
                    "placement_policy": "hbm_only",
                    "num_logical_kv_blocks": 3,
                },
                {
                    "name": "metadata.runtime_overhead",
                    "kind": "metadata",
                    "begin": 40960,
                    "end": 45960,
                    "bytes": 5000,
                    "alignment_bytes": 4096,
                    "placement_policy": "hbm_only",
                },
            ],
            "excluded_objects": [],
        }
        self.plan = {
            "schema_version": 1,
            "policy": "hybrid_residency_v1",
            "physical_hbm_capacity_bytes": 32768,
            "unique_resident_footprint_bytes": 38684,
            "immutable_weight_backing_bytes": 9096,
            "runtime_overhead_bytes": 5000,
            "active_weight_buffer_count": 2,
            "active_weight_buffer_bytes_per_slot": 4096,
            "block_table_bytes": 12,
            "kv_block_stride_bytes": 8192,
            "num_logical_kv_blocks": 3,
            "logical_kv_bytes": 24576,
            "hot_kv_blocks": 1,
            "cold_kv_blocks": 2,
            "weight_mutability": "read_only",
            "kv_mutability": "mutable",
        }
        self.manifest = {
            "schema": {
                "name": "hbfsim.frontier_memory_trace",
                "version": 4,
            },
            "semantics": {
                "source_requests_are_production": True,
                "memory_traffic_is_model_conditioned": True,
                "memory_traffic_is_measured_gpu_traffic": False,
                "eligible_claim_scope": "memory_system_service_only",
                "ttft_tpot_slo_claims_eligible": False,
                "time_based_throughput_claims_eligible": False,
                "model_weight_traffic": "read_only",
                "mutable_write_traffic": "kv_append_and_update_only",
                "capacity_pressure_basis": (
                    "unique_resident_footprint_bytes/"
                    "physical_hbm_capacity_bytes"
                ),
            },
            "capacity_accounting": {
                "planner_mode": "hybrid_residency",
                "immutable_weight_backing_bytes": 9096,
                "residency_plan": self.plan,
            },
            "outputs": {
                "trace": {
                    "path": str(self.trace_path),
                    "bytes": len(self.trace_payload),
                    "sha256": hashlib.sha256(
                        self.trace_payload
                    ).hexdigest(),
                },
                "object_map": {},
            },
        }
        self._write_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_inputs(self) -> None:
        object_payload = _json_bytes(self.object_map)
        self.object_map_path.write_bytes(object_payload)
        self.manifest["outputs"]["object_map"] = {
            "path": str(self.object_map_path),
            "bytes": len(object_payload),
            "sha256": hashlib.sha256(object_payload).hexdigest(),
        }
        self.manifest_path.write_bytes(_json_bytes(self.manifest))

    def _prepare(self) -> dict[str, object]:
        return prepare_residency_config(
            manifest_path=self.manifest_path,
            object_map_path=self.object_map_path,
            output_config_path=self.config_path,
            receipt_path=self.receipt_path,
            physical_hbm_capacity_bytes=32768,
        )

    def test_complete_population_generates_digest_bound_config(self) -> None:
        receipt = self._prepare()
        self.assertEqual(
            receipt["schema"],
            {
                "name": "hbfsim.frontier_residency_binding",
                "version": 3,
            },
        )
        contract = receipt["contract"]
        self.assertEqual(contract["immutable_weight_pages"], 3)
        self.assertEqual(contract["static_weight_resident_pages"], 1)
        self.assertEqual(contract["model_weight_backing_pages"], 2)
        self.assertEqual(contract["metadata_resident_pages"], 3)
        self.assertEqual(contract["hot_kv_resident_pages"], 2)
        self.assertEqual(contract["cold_kv_backing_pages"], 4)
        self.assertEqual(contract["runtime_hbm_pages"], 8)
        self.assertEqual(contract["unused_hbm_pages"], 0)
        self.assertEqual(
            contract["page_allocated_unique_footprint_pages"],
            12,
        )
        self.assertEqual(contract["object_address_space_bytes"], 49152)
        self.assertEqual(
            contract["footprint_page_rounding_bytes"],
            49152 - 38684,
        )
        self.assertAlmostEqual(
            contract["capacity_pressure"],
            38684 / 32768,
        )
        self.assertEqual(
            contract["placement_policy"],
            "capacity_aware_static_weight_prefix_v2",
        )
        self.assertTrue(
            contract["population_preserved_across_placement"],
        )
        config = self.config_path.read_text(encoding="utf-8")
        expected_lines = {
            f"trace={self.trace_path.resolve()}",
            (
                "expected-trace-sha256="
                f"{hashlib.sha256(self.trace_payload).hexdigest()}"
            ),
            f"expected-trace-bytes={len(self.trace_payload)}",
            "hbm-capacity-bytes=32768",
            "layer-buffer-bytes=4096",
            "explicit-residency-contract=true",
            "residency-page-size-bytes=4096",
            "residency-unique-footprint-bytes=38684",
            "residency-immutable-weight-bytes=9096",
            "residency-immutable-weight-pages=3",
            "residency-static-weight-pages=1",
            "residency-runtime-overhead-bytes=5000",
            "residency-block-table-bytes=12",
            "residency-active-buffer-bytes=4096",
            "residency-kv-region-begin=12288",
            "residency-kv-block-stride-bytes=8192",
            "residency-logical-kv-blocks=3",
            "residency-hot-kv-blocks=1",
        }
        self.assertTrue(expected_lines <= set(config.splitlines()))
        self.assertIn(
            f"# manifest-sha256={hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()}",
            config,
        )
        self.assertIn(
            "# placement-policy=capacity_aware_static_weight_prefix_v2",
            config,
        )
        persisted = json.loads(
            self.receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted, receipt)
        self.assertEqual(
            receipt["output"]["config"]["sha256"],
            hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
        )

    def test_rejects_object_map_digest_drift(self) -> None:
        self.object_map["excluded_objects"].append({"name": "mutated"})
        self.object_map_path.write_bytes(_json_bytes(self.object_map))
        with self.assertRaisesRegex(
            ResidencyBindingError,
            "recorded bytes|recorded SHA-256",
        ):
            self._prepare()
        self.assertFalse(self.config_path.exists())
        self.assertFalse(self.receipt_path.exists())

    def test_rejects_trace_digest_drift(self) -> None:
        self.trace_path.write_bytes(self.trace_payload + b"# drift\n")
        with self.assertRaisesRegex(
            ResidencyBindingError,
            "trace recorded bytes|trace recorded SHA-256",
        ):
            self._prepare()
        self.assertFalse(self.config_path.exists())
        self.assertFalse(self.receipt_path.exists())

    def test_rejects_population_or_plan_drift(self) -> None:
        cases = []

        wrong_weight = copy.deepcopy(self.object_map)
        wrong_weight["regions"][0]["bytes"] = 4999
        wrong_weight["regions"][0]["end"] = 4999
        cases.append(("immutable weight bytes", wrong_weight, self.plan))

        overlapping = copy.deepcopy(self.object_map)
        overlapping["regions"][1]["begin"] = 4096
        overlapping["regions"][1]["end"] = 8192
        cases.append(("overlaps", overlapping, self.plan))

        padded_address_space = copy.deepcopy(self.object_map)
        padded_address_space["address_space"]["end"] += 4096
        padded_address_space["address_space"]["bytes"] += 4096
        cases.append(
            (
                "canonical object address-space allocation",
                padded_address_space,
                self.plan,
            )
        )

        wrong_plan = copy.deepcopy(self.plan)
        wrong_plan["unique_resident_footprint_bytes"] += 1
        cases.append(("unique resident footprint", self.object_map, wrong_plan))

        for diagnostic, object_map, plan in cases:
            with self.subTest(diagnostic=diagnostic):
                self.object_map = copy.deepcopy(object_map)
                self.manifest["capacity_accounting"][
                    "residency_plan"
                ] = copy.deepcopy(plan)
                self._write_inputs()
                with self.assertRaisesRegex(
                    ResidencyBindingError,
                    diagnostic,
                ):
                    self._prepare()
                self.config_path.unlink(missing_ok=True)
                self.receipt_path.unlink(missing_ok=True)

    def test_rejects_timing_claim_upgrade(self) -> None:
        self.manifest["semantics"][
            "ttft_tpot_slo_claims_eligible"
        ] = True
        self._write_inputs()
        with self.assertRaisesRegex(
            ResidencyBindingError,
            "TTFT/TPOT/SLO",
        ):
            self._prepare()

    def test_recompiles_hot_kv_without_changing_population(self) -> None:
        receipt = prepare_residency_config(
            manifest_path=self.manifest_path,
            object_map_path=self.object_map_path,
            output_config_path=self.config_path,
            receipt_path=self.receipt_path,
            physical_hbm_capacity_bytes=49152,
        )
        contract = receipt["contract"]
        self.assertEqual(contract["physical_hbm_capacity_bytes"], 49152)
        self.assertEqual(
            contract["source_frontier_plan"],
            {
                "physical_hbm_capacity_bytes": 32768,
                "hot_kv_blocks": 1,
                "cold_kv_blocks": 2,
            },
        )
        self.assertEqual(contract["logical_kv_blocks"], 3)
        self.assertEqual(contract["hot_kv_blocks"], 3)
        self.assertEqual(contract["cold_kv_blocks"], 0)
        self.assertEqual(
            contract["unique_resident_footprint_bytes"],
            self.plan["unique_resident_footprint_bytes"],
        )
        self.assertAlmostEqual(
            contract["capacity_pressure"],
            self.plan["unique_resident_footprint_bytes"] / 49152,
        )
        self.assertIn(
            "residency-hot-kv-blocks=3",
            self.config_path.read_text(encoding="utf-8"),
        )

    def test_static_weight_prefix_covers_zero_and_full_capacity_edges(
        self,
    ) -> None:
        no_spare_hbm = prepare_residency_config(
            manifest_path=self.manifest_path,
            object_map_path=self.object_map_path,
            output_config_path=self.config_path,
            receipt_path=self.receipt_path,
            physical_hbm_capacity_bytes=28672,
        )["contract"]
        self.assertEqual(no_spare_hbm["static_weight_resident_pages"], 0)
        self.assertEqual(no_spare_hbm["model_weight_backing_pages"], 3)
        self.assertEqual(no_spare_hbm["unused_hbm_pages"], 0)

        all_weights_fit = prepare_residency_config(
            manifest_path=self.manifest_path,
            object_map_path=self.object_map_path,
            output_config_path=self.config_path,
            receipt_path=self.receipt_path,
            physical_hbm_capacity_bytes=65536,
        )["contract"]
        self.assertEqual(all_weights_fit["static_weight_resident_pages"], 3)
        self.assertEqual(all_weights_fit["model_weight_backing_pages"], 0)
        self.assertEqual(all_weights_fit["backing_unique_pages"], 0)
        self.assertEqual(
            all_weights_fit["trace_visible_max_backing_pages_per_layer"],
            0,
        )
        self.assertEqual(all_weights_fit["unused_hbm_pages"], 2)

    def test_rejects_target_capacity_below_fixed_residency(self) -> None:
        with self.assertRaisesRegex(
            ResidencyBindingError,
            "one hot KV block",
        ):
            prepare_residency_config(
                manifest_path=self.manifest_path,
                object_map_path=self.object_map_path,
                output_config_path=self.config_path,
                receipt_path=self.receipt_path,
                physical_hbm_capacity_bytes=24576,
            )

    def test_trace_fitted_buffer_preserves_cold_kv_working_set(self) -> None:
        trace = self.root / "cold-kv.trace"
        trace.write_text(
            "\n".join(
                [
                    "0x0 R 4096 kind=model_weights layer=0",
                    (
                        f"0x{(100 + 6) * 4096:x} R 8192 "
                        "kind=generated_context layer=0"
                    ),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        fitted = _compile_trace_fitted_partition(
            trace_path=trace,
            physical_hbm_capacity_bytes=10 * 4096,
            page_size=4096,
            metadata_pages=2,
            source_active_buffer_pages=1,
            kv_region_begin=100 * 4096,
            logical_kv_pages=8,
            logical_blocks=4,
            kv_stride=8192,
        )
        self.assertEqual(fitted["active_buffer_pages"], 3)
        self.assertEqual(fitted["partition"]["hot_blocks"], 1)
        self.assertEqual(fitted["partition"]["cold_blocks"], 3)
        self.assertEqual(
            fitted["trace_census"]["maximum_backing_pages"],
            3,
        )
        self.assertEqual(fitted["iterations"], 2)


if __name__ == "__main__":
    unittest.main()
