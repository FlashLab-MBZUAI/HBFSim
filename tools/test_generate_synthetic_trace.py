#!/usr/bin/env python3
"""Regressions for deterministic synthetic trace generation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from generate_synthetic_trace import GeneratorConfig, generate


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_synthetic_trace.py"


def records(lines: list[str]) -> list[list[str]]:
    return [line.split() for line in lines if line and not line.startswith("#")]


class SyntheticTraceTests(unittest.TestCase):
    def base(self, **changes: object) -> GeneratorConfig:
        config = GeneratorConfig(
            pattern="sequential", pages=8, passes=1, request_bytes=4096,
            base_address=0x10000, address_span_bytes=8 * 4096,
            read_percent=100, seed=7, kind="scratch", at_ns=0)
        return replace(config, **changes)

    def test_sequential_exact_counts_and_digest(self) -> None:
        lines, manifest = generate(self.base(read_percent=75))
        ops = records(lines)
        self.assertEqual([int(row[0], 0) for row in ops],
                         [0x10000 + index * 4096 for index in range(8)])
        self.assertEqual((manifest["ops"], manifest["read_ops"],
                          manifest["write_ops"], manifest["unique_pages"]),
                         (8, 6, 2, 8))
        self.assertEqual(
            manifest["schema"],
            {"name": "hbfsim.synthetic_trace.manifest", "version": 3})
        self.assertIsNone(manifest["placement_modulus"])
        self.assertIsNone(manifest["placement_region_weights"])
        self.assertIsNone(manifest["placement_region_page_counts"])
        self.assertEqual(manifest["bytes"], 8 * 4096)
        payload = ("\n".join(lines) + "\n").encode()
        self.assertEqual(manifest["trace_digest"]["value"],
                         hashlib.sha256(payload).hexdigest())

    def test_linear_placement_is_distinct_and_evenly_spaced(self) -> None:
        packed, _ = generate(self.base(pattern="random-permutation"))
        spread, manifest = generate(self.base(
            pattern="random-permutation", address_span_bytes=80 * 4096))
        packed_ranks = [(int(row[0], 0) - 0x10000) // 4096
                        for row in records(packed)]
        spread_slots_in_order = [(int(row[0], 0) - 0x10000) // 4096
                                 for row in records(spread)]
        self.assertEqual(spread_slots_in_order,
                         [rank * 80 // 8 for rank in packed_ranks])
        slots = sorted(spread_slots_in_order)
        self.assertEqual(len(set(slots)), 8)
        gaps = [(slots[(i + 1) % 8] - slots[i]) % 80 for i in range(8)]
        self.assertEqual(gaps, [10] * 8)
        self.assertEqual(manifest["placement"], "linear")
        self.assertEqual(manifest["placement_seed"], 1)

    def test_stratified_random_chooses_one_unique_slot_per_stratum(self) -> None:
        lines, manifest = generate(self.base(
            pattern="sequential", address_span_bytes=83 * 4096,
            placement="stratified-random", placement_seed=0x1234))
        slots = [(int(row[0], 0) - 0x10000) // 4096
                 for row in records(lines)]
        self.assertEqual(len(set(slots)), 8)
        for page, slot in enumerate(slots):
            with self.subTest(page=page, slot=slot):
                self.assertGreaterEqual(slot, page * 83 // 8)
                self.assertLess(slot, (page + 1) * 83 // 8)
        # The first and last strata anchor sparse coverage near both ends.
        self.assertLess(slots[0], 83 // 8)
        self.assertGreaterEqual(slots[-1], 7 * 83 // 8)
        self.assertEqual(manifest["placement"], "stratified-random")
        self.assertEqual(manifest["placement_seed"], 0x1234)
        # Golden slots make the PRNG mapping stable across Python versions.
        self.assertEqual(slots, [9, 18, 23, 37, 50, 58, 65, 72])

    def test_modulus_balanced_strata_randomize_each_complete_residue_round(
            self) -> None:
        lines, manifest = generate(self.base(
            pattern="sequential",
            base_address=0,
            address_span_bytes=64 * 4096,
            placement="stratified-random",
            placement_seed=0x1234,
            placement_modulus=4,
        ))
        slots = [int(row[0], 0) // 4096 for row in records(lines)]
        self.assertEqual(slots, [4, 10, 19, 25, 37, 44, 55, 62])
        for begin in range(0, len(slots), 4):
            self.assertEqual(
                sorted(slot % 4 for slot in slots[begin:begin + 4]),
                [0, 1, 2, 3],
            )
        for page, slot in enumerate(slots):
            self.assertGreaterEqual(slot, page * 64 // 8)
            self.assertLess(slot, (page + 1) * 64 // 8)
        self.assertEqual(manifest["placement_modulus"], 4)

    def test_access_and_placement_seeds_are_independent(self) -> None:
        config = self.base(
            pattern="random-permutation", address_span_bytes=80 * 4096,
            placement="stratified-random", placement_seed=11)
        original, _ = generate(config)
        repeated, _ = generate(config)
        changed_access, _ = generate(replace(config, seed=8))
        changed_placement, _ = generate(replace(config, placement_seed=12))

        def slots(lines: list[str]) -> list[int]:
            return [(int(row[0], 0) - 0x10000) // 4096
                    for row in records(lines)]

        original_slots = slots(original)
        changed_access_slots = slots(changed_access)
        changed_placement_slots = slots(changed_placement)
        self.assertEqual(original, repeated)
        # Access seed changes order but not the placed working-set addresses.
        self.assertEqual(set(original_slots), set(changed_access_slots))
        self.assertNotEqual(original_slots, changed_access_slots)
        # Placement seed changes offsets but preserves the access permutation:
        # each slot's stratum identifies its logical page.
        self.assertNotEqual(set(original_slots), set(changed_placement_slots))
        self.assertEqual([slot // 10 for slot in original_slots],
                         [slot // 10 for slot in changed_placement_slots])

    def test_weighted_regions_have_exact_nested_address_shares(self) -> None:
        config = self.base(
            pattern="random-permutation",
            pages=20,
            address_span_bytes=80 * 4096,
            placement="stratified-random",
            placement_seed=0x1234,
            placement_region_weights=(4, 3, 2, 1),
        )
        lines, manifest = generate(config)
        slots = [(int(row[0], 0) - 0x10000) // 4096
                 for row in records(lines)]
        counts = [0, 0, 0, 0]
        for slot in slots:
            counts[slot // 20] += 1
        self.assertEqual(counts, [8, 6, 4, 2])
        self.assertEqual(len(set(slots)), 20)
        self.assertEqual(
            manifest["placement_region_weights"], (4, 3, 2, 1))
        self.assertEqual(
            manifest["placement_region_page_counts"], [8, 6, 4, 2])

        changed_access, _ = generate(replace(config, seed=8))
        self.assertEqual(
            {row[0] for row in records(lines)},
            {row[0] for row in records(changed_access)},
        )
        changed_placement, _ = generate(replace(
            config, placement_seed=0x1235))
        self.assertNotEqual(
            {row[0] for row in records(lines)},
            {row[0] for row in records(changed_placement)},
        )

    def test_random_pattern_is_deterministic_permutation_per_pass(self) -> None:
        config = self.base(pattern="random-permutation", passes=2)
        first, first_manifest = generate(config)
        second, second_manifest = generate(config)
        self.assertEqual(first, second)
        self.assertEqual(first_manifest, second_manifest)
        addresses = [row[0] for row in records(first)]
        self.assertEqual(len(set(addresses[:8])), 8)
        self.assertEqual(len(set(addresses[8:])), 8)
        self.assertEqual(set(addresses[:8]), set(addresses[8:]))
        self.assertNotEqual(addresses[:8], addresses[8:])
        page_order = [(int(address, 0) - 0x10000) // 4096
                      for address in addresses[:8]]
        modular_deltas = {
            (page_order[index + 1] - page_order[index]) % 8
            for index in range(7)
        }
        self.assertGreater(len(modular_deltas), 1)
        changed, _ = generate(replace(config, seed=8))
        self.assertNotEqual(first, changed)

    def test_random_permutation_has_version_stable_golden_order(self) -> None:
        lines, _ = generate(self.base(
            pattern="random-permutation", pages=8, passes=1, seed=7))
        order = [(int(row[0], 0) - 0x10000) // 4096
                 for row in records(lines)]
        self.assertEqual(order, [3, 6, 5, 2, 1, 4, 7, 0])

    def test_modular_stride_requires_coprime_stride(self) -> None:
        valid = self.base(pattern="modular-stride", stride_pages=3)
        lines, _ = generate(valid)
        offsets = [(int(row[0], 0) - 0x10000) // 4096
                   for row in records(lines)]
        self.assertEqual(len(set(offsets)), 8)
        with self.assertRaisesRegex(ValueError, "coprime"):
            generate(replace(valid, stride_pages=2))

    def test_hotspot_has_exact_access_and_page_shares(self) -> None:
        config = self.base(
            pattern="hotspot", pages=10, operations=100, passes=None,
            address_span_bytes=100 * 4096, hot_page_percent=20,
            hot_access_percent=80)
        lines, manifest = generate(config)
        self.assertEqual(manifest["hot_pages"], 2)
        self.assertEqual(manifest["hot_ops"], 80)
        self.assertEqual(manifest["unique_pages"], 10)
        counts: dict[str, int] = {}
        order: list[int] = []
        for row in records(lines):
            counts[row[0]] = counts.get(row[0], 0) + 1
            order.append((int(row[0], 0) - 0x10000) // 4096)
        # Eight cold pages share 20 accesses exactly; integer balancing gives
        # four pages two accesses and four pages three accesses.
        self.assertEqual(sorted(counts.values()), [2] * 4 + [3] * 4 + [40] * 2)
        # The hot/cold streams are shuffled too: neither is an affine modular
        # walk inherited from the old random-permutation implementation.
        self.assertGreater(len({(order[index + 1] - order[index]) % 100
                                for index in range(len(order) - 1)}), 2)

    def test_partial_run_reports_actual_unique_pages(self) -> None:
        _, manifest = generate(self.base(operations=3, passes=None))
        self.assertEqual(manifest["working_set_pages"], 8)
        self.assertEqual(manifest["unique_pages"], 3)

    def test_rejects_non_integral_ratios_and_invalid_ranges(self) -> None:
        invalid = (
            self.base(operations=3, passes=None, read_percent=50),
            self.base(base_address=1),
            self.base(address_span_bytes=7 * 4096),
            self.base(base_address=(1 << 64) - 4096,
                      address_span_bytes=8 * 4096),
            self.base(seed=-1),
            self.base(placement_seed=-1),
            self.base(placement="not-a-placement"),
            self.base(placement_modulus=4),
            self.base(
                placement="stratified-random", placement_modulus=0),
            self.base(
                placement="stratified-random", placement_modulus=3),
            self.base(
                placement="stratified-random", placement_modulus=4),
            self.base(
                pages=4,
                address_span_bytes=20 * 4096,
                placement="stratified-random",
                placement_modulus=4,
            ),
            self.base(
                pages=4,
                address_span_bytes=18 * 4096,
                placement="stratified-random",
                placement_modulus=4,
            ),
            self.base(placement_region_weights=(1, 1)),
            self.base(
                placement="stratified-random",
                placement_region_weights=(1,),
            ),
            self.base(
                placement="stratified-random",
                placement_region_weights=(1, 0),
            ),
            self.base(
                pages=8,
                address_span_bytes=82 * 4096,
                placement="stratified-random",
                placement_region_weights=(1, 1, 1),
            ),
            self.base(
                pages=8,
                placement="stratified-random",
                placement_modulus=4,
                placement_region_weights=(1, 1),
            ),
            self.base(
                pages=8,
                placement="stratified-random",
                placement_region_weights=(2, 1),
            ),
            self.base(operations=1, passes=1),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                generate(config)

    def test_hotspot_rejects_impossible_or_inexact_share(self) -> None:
        with self.assertRaises(ValueError):
            generate(self.base(pattern="hotspot", pages=7,
                               hot_page_percent=20, hot_access_percent=80))
        with self.assertRaises(ValueError):
            generate(self.base(pattern="hotspot", hot_page_percent=0,
                               hot_access_percent=50))

    def test_phase_comments_do_not_change_operation_grammar(self) -> None:
        lines, manifest = generate(self.base(passes=2, phase_comments=True))
        self.assertIn("# phase=0 begin_op=0", lines)
        self.assertIn("# phase=1 begin_op=8", lines)
        self.assertEqual(len(records(lines)), manifest["ops"])
        for row in records(lines):
            self.assertEqual(len(row), 5)
            self.assertIn(row[1], ("R", "W"))
            self.assertEqual(row[3], "scratch")
            self.assertEqual(row[4], "at=0")

    def test_layer_per_pass_emits_complete_nondecreasing_layers(self) -> None:
        lines, manifest = generate(self.base(
            passes=3, phase_comments=True, layer_per_pass=True))
        rows = records(lines)
        self.assertEqual(manifest["layer_per_pass"], True)
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            [row[-1] for row in rows],
            ["layer=0"] * 8 + ["layer=1"] * 8 + ["layer=2"] * 8,
        )

    def test_cli_publishes_trace_then_digest_manifest_without_temps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "workload.trace"
            command = [
                sys.executable, "-B", str(GENERATOR), "--output", str(output),
                "--pattern", "random-permutation", "--pages", "10",
                "--passes", "2", "--read-percent", "75", "--bytes", "4096",
                "--base", "0x20000", "--address-span", str(100 * 4096),
                "--seed", "0x123", "--kind", "shared_context",
                "--placement", "stratified-random",
                "--placement-seed", "0x456",
                "--placement-modulus", "10",
                "--phase-comments",
                "--layer-per-pass",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest_path = Path(str(output) + ".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["ops"], 20)
            self.assertEqual(manifest["read_ops"], 15)
            self.assertEqual(manifest["placement"], "stratified-random")
            self.assertEqual(manifest["placement_seed"], 0x456)
            self.assertEqual(manifest["placement_modulus"], 10)
            self.assertEqual(manifest["layer_per_pass"], True)
            self.assertEqual(manifest["trace_digest"]["value"],
                             hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(manifest["trace_path"], str(output.resolve()))
            self.assertEqual(list(output.parent.glob("*.tmp")), [])

    def test_cli_failure_does_not_publish_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bad.trace"
            completed = subprocess.run([
                sys.executable, "-B", str(GENERATOR), "--output", str(output),
                "--pattern", "modular-stride", "--pages", "8", "--passes", "1",
                "--stride-pages", "4",
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("coprime", completed.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output) + ".manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
