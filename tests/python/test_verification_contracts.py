#!/usr/bin/env python3
"""Self-tests for foundational validation contracts and failure reporting."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verification.core.ledger_compare import (  # noqa: E402
    first_difference,
    load_ledger,
    validate_ledger,
)
from verification.core.contracts import (  # noqa: E402
    CASE_SCHEMA,
    ContractError,
    dump_json_line,
    load_case,
    load_json_strict,
    validate_case,
)
from verification.oracles.hbm import Oracle as HbmOracle, build_ledger  # noqa: E402
from verification.oracles.hbf import (  # noqa: E402
    build_ledger as build_hbf_ledger,
)
from verification.oracles.hybrid import (  # noqa: E402
    build_ledger as build_hybrid_ledger,
)
from verification.oracles.external import (  # noqa: E402
    build_ledger as build_external_ledger,
)
from verification.core.ledger_reduce import reduce_ledger  # noqa: E402
from verification.gates.fuzz import (  # noqa: E402
    SplitMix64,
    _replace_strings,
    generate_case,
    shrink_case,
)
from verification.gates.mutation import MUTATIONS  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
ROW_CASE = ROOT / "verification/cases/hbm.channel-service.json"
HBF_CASE_DIR = ROOT / "verification/cases"


class CaseContractTests(unittest.TestCase):
    def test_canonical_case_resolves_every_default(self) -> None:
        case = load_case(ROW_CASE)
        self.assertEqual(case["case_id"], "hbm.channel-service")
        self.assertEqual(case["config"]["read_to_write_ns"], 8.0)
        self.assertEqual(case["config"]["pseudo_channels_per_channel"], 2)
        self.assertEqual(len(case["requests"]), 2)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema": 1, "schema": 2}')
            with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
                load_json_strict(path)

    def test_unknown_case_field_is_rejected(self) -> None:
        data = json.loads(ROW_CASE.read_text())
        data["looks_reasonable"] = True
        with self.assertRaisesRegex(ContractError, "unknown keys"):
            validate_case(data)

    def test_nonfinite_time_is_rejected(self) -> None:
        data = json.loads(ROW_CASE.read_text())
        data["requests"][0]["arrival_ns"] = float("nan")
        with self.assertRaisesRegex(ContractError, "finite"):
            validate_case(data)

    def test_request_range_is_fail_closed(self) -> None:
        data = json.loads(ROW_CASE.read_text())
        data["requests"][0]["addr"] = data["config"]["capacity_bytes"] - 1
        data["requests"][0]["bytes"] = 2
        with self.assertRaisesRegex(ContractError, "exceeds HBM capacity"):
            validate_case(data)

    def test_uint32_topology_overflow_is_rejected_by_contract(self) -> None:
        data = json.loads(ROW_CASE.read_text())
        data["config"]["stacks"] = 1 << 32
        with self.assertRaisesRegex(ContractError, "uint32"):
            validate_case(data)


class LedgerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = load_case(ROW_CASE)
        self.ledger = build_ledger(self.case)
        validate_ledger(self.ledger)

    def test_unknown_ledger_field_is_rejected(self) -> None:
        broken = copy.deepcopy(self.ledger)
        broken[1]["silent_typo"] = 1
        with self.assertRaisesRegex(ContractError, "unknown request keys"):
            validate_ledger(broken)

    def test_missing_or_later_parent_is_rejected(self) -> None:
        broken = copy.deepcopy(self.ledger)
        event = next(record for record in broken if record["kind"] == "event")
        event["parent_id"] = "request/not-yet-present"
        with self.assertRaisesRegex(ContractError, "absent or later parent"):
            validate_ledger(broken)

    def test_non_request_parent_is_rejected(self) -> None:
        broken = copy.deepcopy(self.ledger)
        event = next(record for record in broken if record["kind"] == "event")
        event["parent_id"] = "ledger"
        with self.assertRaisesRegex(ContractError, "is not a request"):
            validate_ledger(broken)

    def test_reverse_time_event_is_rejected(self) -> None:
        broken = copy.deepcopy(self.ledger)
        event = next(record for record in broken if record["kind"] == "event")
        event["finish_ns"] = event["start_ns"] - 1.0
        with self.assertRaisesRegex(ContractError, "invalid interval"):
            validate_ledger(broken)

    def test_duplicate_ledger_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.jsonl"
            first = dump_json_line(self.ledger[0])
            path.write_text(first[:-1] + ',"id":"duplicate"}\n')
            with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
                load_ledger(path)


class HbmOracleContractTests(unittest.TestCase):
    def test_grouped_events_conserve_payload_and_effective_service(self) -> None:
        case = load_case(ROW_CASE)
        ledger = build_ledger(case)
        validate_ledger(ledger)
        events = [e for e in ledger if e["kind"] == "event"]
        self.assertTrue(events)
        self.assertTrue(all(e["category"] == "hbm_channel_service" for e in events))
        metrics = reduce_ledger(ledger)["metrics"]
        self.assertEqual(sum(e["physical_bytes"] for e in events), metrics["read_bytes"]+metrics["write_bytes"])
        for counter in ("read_bytes", "write_bytes", "bus_busy_ns"):
            broken = copy.deepcopy(ledger)
            broken[-1]["counters"][counter] = 0
            with self.assertRaisesRegex(ContractError, f"{counter}.*ledger-reducible"):
                validate_ledger(broken)


class HbfReadOracleContractTests(unittest.TestCase):
    @staticmethod
    def load(name: str) -> tuple[dict, list[dict]]:
        case = load_case(HBF_CASE_DIR / f"{name}.json")
        return case, build_hbf_ledger(case)

    @staticmethod
    def array_events(ledger: list[dict]) -> list[dict]:
        return [
            record
            for record in ledger
            if record["kind"] == "event"
            and record["action"] == "user/array_read"
        ]

    def test_resident_mapping_capacity_is_derived_independently(self) -> None:
        case, ledger = self.load("hbf.read-single")
        self.assertEqual(case["config"]["ctrl_dram_bytes"], 65536)
        self.assertEqual(
            ledger[-1]["config"]["derived"][
                "resident_mapping_total_bytes"],
            65536,
        )
        self.assertEqual(ledger[-1]["counters"]["mapping_lookup_ops"], 1)

    def test_same_plane_case_really_serializes_array(self) -> None:
        _, ledger = self.load("hbf.read-same-plane")
        first, second = self.array_events(ledger)
        self.assertGreaterEqual(second["start_ns"], first["finish_ns"])
        self.assertEqual(ledger[-1]["counters"]["active_planes"], 1)

    def test_cross_plane_case_really_overlaps_array(self) -> None:
        _, ledger = self.load("hbf.read-cross-plane")
        first, second = self.array_events(ledger)
        self.assertLess(second["start_ns"], first["finish_ns"])
        self.assertEqual(ledger[-1]["counters"]["active_planes"], 2)

    def test_cross_die_case_has_independent_array_and_ecc(self) -> None:
        _, ledger = self.load("hbf.read-cross-die")
        first, second = self.array_events(ledger)
        self.assertLess(second["start_ns"], first["finish_ns"])
        self.assertEqual(ledger[-1]["counters"]["active_dies"], 2)

    def test_cross_channel_case_has_independent_media_and_hbio(self) -> None:
        _, ledger = self.load("hbf.read-cross-channel")
        first, second = self.array_events(ledger)
        self.assertLess(second["start_ns"], first["finish_ns"])
        self.assertEqual(ledger[-1]["counters"]["active_channels"], 2)
        self.assertEqual(ledger[-1]["counters"]["active_planes"], 2)

    def test_duplicate_prepopulation_is_rejected(self) -> None:
        data = json.loads(
            (HBF_CASE_DIR / "hbf.read-single.json").read_text())
        data["initial_state"]["prepopulate_lpns"] = [0, 0]
        with self.assertRaisesRegex(ContractError, "duplicates"):
            validate_case(data)


class HbfWriteOracleContractTests(unittest.TestCase):
    @staticmethod
    def load(name: str) -> tuple[dict, list[dict]]:
        case = load_case(HBF_CASE_DIR / f"{name}.json")
        ledger = build_hbf_ledger(case)
        validate_ledger(ledger)
        return case, ledger

    def test_drain_contract_is_explicit_and_fail_closed(self) -> None:
        data = json.loads(
            (HBF_CASE_DIR / "hbf.write-full-drain.json").read_text())
        data["requests"][-1]["address_space"] = "logical"
        with self.assertRaisesRegex(ContractError, "expected internal"):
            validate_case(data)
        data = json.loads(
            (HBF_CASE_DIR / "hbf.write-full-drain.json").read_text())
        data["requests"][-1]["bytes"] = 1
        with self.assertRaisesRegex(
            ContractError, "drain requires addr=0 and bytes=0"
        ):
            validate_case(data)

    def test_drain_emits_quiescent_sha256_state(self) -> None:
        _, ledger = self.load("hbf.write-full-drain")
        state = next(record for record in ledger if record["kind"] == "state")
        self.assertTrue(state["attributes"]["quiescent"])
        self.assertEqual(
            state["attributes"]["pending"],
            {
                "dirty_mapping_events": 0,
                "lpn_updates": 0,
                "vpn_updates": 0,
                "commits": 0,
                "write_buffer_entries": 0,
                "inflight_buffered_generations": 0,
                "physical_programs": 0,
                "block_transitions": 0,
            },
        )
        self.assertRegex(state["state_hash"], r"\Asha256:[0-9a-f]{64}\Z")
        self.assertEqual(
            state["attributes"]["logical_mappings"],
            [{"lpn": 0, "ppn": 0}],
        )
        self.assertEqual(
            state["attributes"]["mapping_pages"],
            [{"vpn": 0, "ppn": 4}],
        )

    def test_overwrite_invalidates_old_data_and_checkpoint(self) -> None:
        _, ledger = self.load("hbf.overwrite-no-gc")
        state = next(record for record in ledger if record["kind"] == "state")
        pages = state["attributes"]["pages"]
        self.assertEqual(
            [page["status"] for page in pages],
            ["valid", "valid"],
        )
        self.assertEqual(ledger[-1]["counters"]["invalidations"], 2)

    def test_two_subpages_coalesce_to_one_data_program(self) -> None:
        _, ledger = self.load("hbf.write-subpage-coalesce")
        counters = ledger[-1]["counters"]
        self.assertEqual(counters["program_requests"], 2)
        self.assertEqual(counters["data_programs"], 1)
        self.assertEqual(counters["mapping_page_programs"], 1)
        self.assertEqual(counters["write_buffer_misses"], 1)
        self.assertEqual(counters["write_buffer_hits"], 1)
        self.assertEqual(counters["write_buffer_flushes"], 1)

    def test_overlapping_buffer_ranges_count_only_replaced_bytes(self) -> None:
        _, ledger = self.load("hbf.write-buffer-overlap")
        counters = ledger[-1]["counters"]
        self.assertEqual(counters["logical_write_bytes"], 4096)
        self.assertEqual(counters["write_buffer_merged_bytes"], 1024)
        self.assertEqual(counters["data_programs"], 1)


class HbfGcOracleContractTests(unittest.TestCase):
    @staticmethod
    def load(name: str) -> tuple[dict, list[dict], dict, dict]:
        case = load_case(HBF_CASE_DIR / f"{name}.json")
        ledger = build_hbf_ledger(case)
        validate_ledger(ledger)
        state = next(
            record for record in ledger if record["kind"] == "state")
        return case, ledger, state, ledger[-1]["counters"]

    def assert_gc_page_conservation(
        self,
        case: dict,
        counters: dict,
    ) -> None:
        erased_pages = (
            counters["gc_runs"] * case["config"]["pages_per_block"])
        # Logical reclaim does not erase; page-zero programs pay the P/E cycle.
        self.assertGreater(counters["block_erases"], 0)
        self.assertEqual(
            counters["gc_relocations"]
            + counters["gc_reclaimed_invalid_pages"],
            erased_pages,
        )

    def test_zero_live_page_gc_reclaims_without_relocation(self) -> None:
        case, _, _, counters = self.load("hbf.gc-zero-relocation")
        self.assertEqual(counters["gc_runs"], 3)
        self.assertEqual(counters["gc_relocations"], 0)
        self.assertEqual(counters["gc_reclaimed_invalid_pages"], 6)
        self.assert_gc_page_conservation(case, counters)

    def test_live_page_gc_relocates_and_persists_new_dirty_generation(self) -> None:
        case, ledger, _, counters = self.load("hbf.gc-live-relocation")
        # A cold page survives the first invalidation in a two-page victim.
        self.assertEqual(counters["gc_runs"], 1)
        self.assertEqual(counters["gc_data_relocations"], 1)
        self.assertEqual(counters["gc_mapping_relocations"], 0)
        self.assertEqual(counters["mapping_gc_update_ops"], 1)
        self.assertEqual(counters["mapping_page_programs"], 1)
        self.assert_gc_page_conservation(case, counters)

    def test_mapping_checkpoints_replace_prior_generations(self) -> None:
        case, _, state, counters = self.load("hbf.mapping-checkpoint-generations")
        self.assertEqual(counters["gc_runs"], 0)
        self.assertEqual(counters["mapping_page_programs"], 3)
        self.assertEqual(counters["mapping_gc_update_ops"], 0)
        self.assertEqual(len(state["attributes"]["mapping_pages"]), 1)
        self.assertEqual(counters["invalidations"], 2)

    def test_gc_final_state_has_reachable_mappings_and_conserved_blocks(
        self,
    ) -> None:
        case, _, state, _ = self.load("hbf.gc-live-relocation")
        attributes = state["attributes"]
        pages = {
            page["ppn"]: page for page in attributes["pages"]
        }
        for mapping in attributes["logical_mappings"]:
            page = pages[mapping["ppn"]]
            self.assertEqual(page["status"], "valid")
            self.assertEqual(page["owner"], "logical")
            self.assertEqual(page["logical_key"], mapping["lpn"])
        for mapping in attributes["mapping_pages"]:
            page = pages[mapping["ppn"]]
            self.assertEqual(page["status"], "valid")
            self.assertEqual(page["owner"], "mapping")
            self.assertEqual(
                page["logical_key"], (1 << 63) | mapping["vpn"])
        pages_per_block = case["config"]["pages_per_block"]
        for block in attributes["blocks"]:
            self.assertEqual(
                block["valid_pages"]
                + block["invalid_pages"]
                + block["free_pages"],
                pages_per_block,
            )
        self.assertEqual(
            attributes["free_pages"],
            sum(block["free_pages"] for block in attributes["blocks"]),
        )


class HybridOracleContractTests(unittest.TestCase):
    @staticmethod
    def load(name: str) -> tuple[dict, list[dict]]:
        case = load_case(HBF_CASE_DIR / f"{name}.json")
        ledger = build_hybrid_ledger(case)
        validate_ledger(ledger)
        return case, ledger

    def test_independent_tiers_overlap_instead_of_serializing(self) -> None:
        _, ledger = self.load("hybrid.independent-overlap")
        completions = {
            record["request_id"]: record
            for record in ledger
            if record["kind"] == "completion"
        }
        hbm = completions["hbm-page"]
        hbf = completions["hbf-page"]
        self.assertLess(hbf["start_ns"], hbm["finish_ns"])
        self.assertLess(hbm["start_ns"], hbf["finish_ns"])
        counters = ledger[-1]["counters"]
        self.assertEqual(counters["hbm_user_accesses"], 1)
        self.assertEqual(counters["hbf_user_accesses"], 1)
        self.assertEqual(counters["front_end_admission_waited_ops"], 0)

    def test_cross_tier_parent_uses_max_finish_and_conserves_bytes(
        self,
    ) -> None:
        _, ledger = self.load("hybrid.parent-split")
        completion = next(
            record for record in ledger
            if record["kind"] == "completion")
        events = [
            record for record in ledger if record["kind"] == "event"]
        counters = ledger[-1]["counters"]
        self.assertEqual(
            {event["model"] for event in events}, {"hbm", "hbf"})
        self.assertEqual(
            completion["finish_ns"],
            max(event["finish_ns"] for event in events),
        )
        self.assertEqual(completion["logical_bytes"], 1024)
        self.assertEqual(completion["physical_bytes"], 4608)
        # One full-resident mapping entry uses 64 B on the shared HBM bus;
        # controller traffic is physical work outside the parent payload.
        controller_read_bytes = 64
        self.assertTrue(any(event["action"] == "hbf_buffer_read" for event in events))
        self.assertEqual(
            counters["hbm"]["read_bytes"] - controller_read_bytes
            + counters["hbf"]["logical_read_bytes"],
            completion["logical_bytes"],
        )
        self.assertEqual(
            counters["hbm"]["read_bytes"] - controller_read_bytes
            + counters["hbf"]["physical_read_bytes"],
            completion["physical_bytes"],
        )
        self.assertEqual(
            counters["service_latencies_ns"],
            [completion["finish_ns"] - completion["arrival_ns"]],
        )

    def test_hybrid_scope_rejects_unverified_scheduler_modes(self) -> None:
        data = json.loads(
            (HBF_CASE_DIR / "hybrid.parent-split.json").read_text())
        data["config"]["knobs"]["max_outstanding_requests"] = 1
        with self.assertRaisesRegex(
            ContractError, "requires an unbounded window"
        ):
            validate_case(data)
        data = json.loads(
            (HBF_CASE_DIR / "hybrid.parent-split.json").read_text())
        data["config"]["policy"]["read_boundary"] += 1
        with self.assertRaisesRegex(ContractError, "HBF-page aligned"):
            validate_case(data)
        data = json.loads(
            (HBF_CASE_DIR / "hybrid.parent-split.json").read_text())
        data["requests"][0]["op"] = "write"
        with self.assertRaisesRegex(ContractError, "expected read"):
            validate_case(data)


class ExternalOracleContractTests(unittest.TestCase):
    @staticmethod
    def load(name: str) -> tuple[dict, list[dict]]:
        case = load_case(HBF_CASE_DIR / f"{name}.json")
        ledger = build_external_ledger(case)
        validate_ledger(ledger)
        return case, ledger

    def test_closed_form_read_and_write_pipeline(self) -> None:
        _, read = self.load("external.read-single")
        _, write = self.load("external.write-single")
        read_completion = next(
            record for record in read if record["kind"] == "completion"
        )
        write_completion = next(
            record for record in write if record["kind"] == "completion"
        )
        self.assertAlmostEqual(read_completion["finish_ns"], 2249.53125)
        self.assertAlmostEqual(write_completion["finish_ns"], 3349.53125)
        self.assertEqual(read[-1]["counters"]["s2m_wire_bytes"], 4112)
        self.assertEqual(write[-1]["counters"]["m2s_wire_bytes"], 4160)
        self.assertEqual(read[-1]["counters"]["page_run_requests"], 0)
        self.assertEqual(read[-1]["counters"]["page_run_segments"], 0)
        self.assertEqual(read[-1]["counters"]["page_run_pages"], 0)
        self.assertEqual(read[-1]["config"]["media_read_queues"], 1)
        self.assertEqual(read[-1]["config"]["media_write_queues"], 1)
        self.assertEqual(read[-1]["counters"]["media_channels_per_queue"], 1)

    def test_future_gap_backfill_allows_out_of_order_completion(self) -> None:
        _, ledger = self.load("external.future-gap-backfill")
        finishes = {
            record["request_id"]: record["finish_ns"]
            for record in ledger
            if record["kind"] == "completion"
        }
        self.assertLess(finishes["fast-read"], finishes["slow-write"])
        self.assertEqual(ledger[-1]["counters"]["active_media_resources"], 2)

    def test_same_arrival_read_ranges_round_robin_queues_without_splitting(
        self,
    ) -> None:
        case, ledger = self.load("external.read-queue-round-robin")
        self.assertEqual(
            [request["arrival_ns"] for request in case["requests"]],
            [0.0, 0.0],
        )
        resources_by_request = {
            request_id: {
                event["resource"]
                for event in ledger
                if event["kind"] == "event"
                and event["request_id"] == request_id
                and event["action"].endswith("_media_transfer")
            }
            for request_id in (
                "read-range-queue0",
                "read-range-queue1",
            )
        }
        self.assertEqual(
            resources_by_request["read-range-queue0"],
            {
                "external/nvme-ssd/media-queue0/channel0",
                "external/nvme-ssd/media-queue0/channel1",
            },
        )
        self.assertEqual(
            resources_by_request["read-range-queue1"],
            {
                "external/nvme-ssd/media-queue1/channel0",
                "external/nvme-ssd/media-queue1/channel1",
            },
        )
        queue0 = next(
            event for event in ledger
            if event["kind"] == "event"
            and event["action"].endswith("_media_transfer")
            and event["resource"].endswith("media-queue0/channel0")
        )
        queue1 = next(
            event for event in ledger
            if event["kind"] == "event"
            and event["action"].endswith("_media_transfer")
            and event["resource"].endswith("media-queue1/channel0")
        )
        self.assertLess(
            max(queue0["start_ns"], queue1["start_ns"]),
            min(queue0["finish_ns"], queue1["finish_ns"]),
        )
        counters = ledger[-1]["counters"]
        self.assertEqual(counters["page_run_requests"], 2)
        self.assertEqual(counters["page_run_segments"], 4)
        self.assertEqual(counters["read_requests"], 4)
        self.assertEqual(counters["media_channels_per_queue"], 2)
        self.assertEqual(counters["media_read_queues"], 2)
        self.assertEqual(counters["media_write_queues"], 1)
        self.assertEqual(counters["media_channels"], 4)
        self.assertEqual(counters["active_media_resources"], 4)

    def test_segmented_host_dram_range_has_two_commands_four_pages(
        self,
    ) -> None:
        case, ledger = self.load("external.segmented-range")
        self.assertEqual(case["config"]["kind"], "host-dram")
        self.assertEqual(case["config"]["request_segment_bytes"], 8192)
        completion = next(
            record for record in ledger if record["kind"] == "completion"
        )
        self.assertEqual(
            completion["resource"],
            "external/host-dram/segmented-range",
        )
        self.assertEqual(
            completion["result"], "external-backing-segmented-range"
        )
        counters = ledger[-1]["counters"]
        self.assertEqual(counters["read_requests"], 2)
        self.assertEqual(counters["read_bytes"], 16 * 1024)
        self.assertEqual(counters["page_run_requests"], 1)
        self.assertEqual(counters["page_run_segments"], 2)
        self.assertEqual(counters["page_run_pages"], 4)
        self.assertEqual(
            [
                observation["media_channel"]
                for observation in ledger[-1]["address_observations"]
            ],
            [0, 0, 1, 1],
        )

    def test_request_segment_must_be_page_multiple(self) -> None:
        case = json.loads(
            (HBF_CASE_DIR / "external.segmented-range.json").read_text()
        )
        for invalid in (2048, 6144):
            with self.subTest(request_segment_bytes=invalid):
                case["config"]["request_segment_bytes"] = invalid
                with self.assertRaisesRegex(
                    ContractError, "positive multiple"
                ):
                    validate_case(case)

    def test_media_queue_counts_must_be_positive(self) -> None:
        case = json.loads(
            (HBF_CASE_DIR / "external.read-queue-round-robin.json").read_text()
        )
        for field in ("media_read_queues", "media_write_queues"):
            with self.subTest(field=field):
                broken = copy.deepcopy(case)
                broken["config"][field] = 0
                with self.assertRaisesRegex(ContractError, "positive uint32"):
                    validate_case(broken)

    def test_reducer_rejects_protocol_wire_counter_drift(self) -> None:
        _, ledger = self.load("external.full-duplex")
        broken = copy.deepcopy(ledger)
        broken[-1]["counters"]["s2m_wire_bytes"] += 1
        with self.assertRaisesRegex(ContractError, "s2m_wire_bytes"):
            validate_ledger(broken)

    def test_reducer_rejects_segment_counter_drift(self) -> None:
        _, ledger = self.load("external.segmented-range")
        broken = copy.deepcopy(ledger)
        broken[-1]["counters"]["page_run_segments"] += 1
        with self.assertRaisesRegex(ContractError, "page_run_segments"):
            validate_ledger(broken)

    def test_reducer_rejects_range_queue_drift(self) -> None:
        _, ledger = self.load("external.read-queue-round-robin")
        broken = copy.deepcopy(ledger)
        event = next(
            record for record in broken
            if record["kind"] == "event"
            and record["request_id"] == "read-range-queue1"
            and record["action"].endswith("_media_transfer")
        )
        event["resource"] = event["resource"].replace(
            "media-queue1", "media-queue2"
        )
        with self.assertRaisesRegex(ContractError, "round-robin queue"):
            validate_ledger(broken)


class PropertyFuzzContractTests(unittest.TestCase):
    def test_splitmix64_algorithm_vector_is_stable(self) -> None:
        generator = SplitMix64(0)
        self.assertEqual(
            [generator.next_u64() for _ in range(4)],
            [
                0xE220A8397B1DCDAF,
                0x6E789E6AA1B965F4,
                0x06C45D188009454F,
                0xF88BB8A8724C81EC,
            ],
        )

    def test_same_seed_reproduces_strict_valid_cases(self) -> None:
        for model in ("hbm", "hbf", "hybrid", "external"):
            first = generate_case(model, 42, max_requests=10)
            second = generate_case(model, 42, max_requests=10)
            self.assertEqual(first, second)
            self.assertEqual(validate_case(first), validate_case(second))
        self.assertNotEqual(
            generate_case("hbm", 42),
            generate_case("hbm", 43),
        )

    def test_shrinker_reduces_requests_values_and_geometry(self) -> None:
        case = generate_case("hbm", 7, max_requests=4)
        case["config"].update({
            "stacks": 2,
            "channels_per_stack": 2,
            "pseudo_channels_per_channel": 2,
            "bank_groups_per_pseudo_channel": 2,
            "banks_per_group": 2,
        })
        case["requests"] = [
            {
                "id": f"r{index}",
                "arrival_ns": float(index),
                "op": "read",
                "address_space": "logical",
                "addr": 1024 + index * 128,
                "bytes": 128,
            }
            for index in range(4)
        ]

        def contains_trigger(candidate: dict) -> bool:
            return any(
                request["id"] == "r2"
                for request in candidate["requests"]
            )

        minimized = shrink_case(case, contains_trigger)
        self.assertEqual(len(minimized["requests"]), 1)
        request = minimized["requests"][0]
        self.assertEqual(request["id"], "r2")
        self.assertEqual(request["arrival_ns"], 0.0)
        self.assertEqual(request["addr"], 0)
        self.assertEqual(request["bytes"], 1)
        for key in (
            "stacks",
            "channels_per_stack",
            "pseudo_channels_per_channel",
            "bank_groups_per_pseudo_channel",
            "banks_per_group",
        ):
            self.assertEqual(minimized["config"][key], 1)

    def test_longer_renamed_id_is_restored_before_prefix(self) -> None:
        self.assertEqual(
            _replace_strings(
                "request/renamed-18",
                {
                    "renamed-1": "r1",
                    "renamed-18": "drain",
                },
            ),
            "request/drain",
        )


class LedgerReducerContractTests(unittest.TestCase):
    def test_hbf_write_amplification_is_reduced_from_events(self) -> None:
        case = load_case(HBF_CASE_DIR / "hbf.write-full-drain.json")
        ledger = build_hbf_ledger(case)
        reduced = reduce_ledger(ledger)
        self.assertEqual(
            reduced["metrics"]["physical_write_bytes"], 8192)
        self.assertEqual(
            reduced["metrics"]["logical_write_bytes"], 4096)
        self.assertEqual(
            reduced["metrics"]["write_amplification"], 2.0)

    def test_summary_byte_mutation_is_rejected(self) -> None:
        case = load_case(HBF_CASE_DIR / "hbf.write-full-drain.json")
        broken = build_hbf_ledger(case)
        broken[-1]["counters"]["physical_write_bytes"] += 512
        with self.assertRaisesRegex(
            ContractError, "physical_write_bytes.*ledger-reducible"
        ):
            validate_ledger(broken)

    def test_summary_resource_busy_mutation_is_rejected(self) -> None:
        case = load_case(HBF_CASE_DIR / "hbf.read-cross-plane.json")
        broken = build_hbf_ledger(case)
        broken[-1]["counters"]["channel_data_busy_ns"] += 1.0
        with self.assertRaisesRegex(
            ContractError, "channel_data_busy_ns.*ledger-reducible"
        ):
            validate_ledger(broken)

    def test_completion_sum_instead_of_max_is_rejected(self) -> None:
        case = load_case(HBF_CASE_DIR / "hybrid.parent-split.json")
        broken = build_hybrid_ledger(case)
        completion = next(
            record for record in broken if record["kind"] == "completion")
        completion["finish_ns"] += 30.0
        broken[-1]["counters"]["finish_ns"] = completion["finish_ns"]
        broken[-1]["counters"]["user_finish_ns"] = completion["finish_ns"]
        for key in (
            "service_latencies_ns",
            "offered_latencies_ns",
            "source_latencies_ns",
        ):
            broken[-1]["counters"][key][0] += 30.0
        with self.assertRaisesRegex(
            ContractError, "max-critical-event"
        ):
            validate_ledger(broken)

    def test_hbm_byte_counter_mutation_is_rejected(self) -> None:
        case = load_case(ROW_CASE)
        broken = build_ledger(case)
        broken[-1]["counters"]["read_bytes"] += 1
        with self.assertRaisesRegex(
            ContractError, "read_bytes.*ledger-reducible"
        ):
            validate_ledger(broken)


class MutationHarnessContractTests(unittest.TestCase):
    def test_mutations_are_unique_production_patches_with_owned_cases(
        self,
    ) -> None:
        identifiers = [mutation.mutation_id for mutation in MUTATIONS]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for mutation in MUTATIONS:
            self.assertTrue(mutation.critical)
            self.assertTrue(
                mutation.source.startswith(
                    ("src/physical/", "src/host/", "src/policies/reference/")
                ),
                mutation.source,
            )
            source = ROOT / mutation.source
            self.assertTrue(source.is_file())
            self.assertEqual(
                source.read_text().count(mutation.old),
                mutation.replacement_count,
                mutation.mutation_id,
            )
            self.assertNotEqual(mutation.old, mutation.new)
            self.assertRegex(
                mutation.patch_digest(), r"\Asha256:[0-9a-f]{64}\Z")
            if mutation.gate == "canonical":
                for case_id in mutation.cases:
                    self.assertTrue(
                        (HBF_CASE_DIR / f"{case_id}.json").is_file(),
                        f"{mutation.mutation_id}: missing {case_id}",
                    )
            elif mutation.gate == "behavioral":
                self.assertEqual(
                    set(mutation.cases),
                    {
                        "behavioral.policy-unit",
                        "behavioral.independent-differential",
                    },
                    mutation.mutation_id,
                )
            elif mutation.gate == "physical":
                self.assertEqual(len(mutation.cases), 1, mutation.mutation_id)
                self.assertIn(
                    mutation.cases[0],
                    {
                        "physical.hbm-channels",
                        "physical.layer-backing-credit",
                    },
                    mutation.mutation_id,
                )
            else:
                self.fail(
                    f"{mutation.mutation_id}: unknown gate {mutation.gate}")


class FirstDivergenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = load_case(ROW_CASE)
        self.expected = build_ledger(self.case)

    def compare(self, actual: list[dict]) -> object:
        return first_difference(
            self.expected,
            actual,
            abs_tolerance=self.case["comparison"][
                "time_abs_tolerance_ns"],
            rel_tolerance=self.case["comparison"]["time_rel_tolerance"],
        )

    def test_exact_copy_matches(self) -> None:
        self.assertIsNone(self.compare(copy.deepcopy(self.expected)))

    def test_timing_mutation_reports_exact_field(self) -> None:
        actual = copy.deepcopy(self.expected)
        event_index = next(
            index
            for index, record in enumerate(actual)
            if record["kind"] == "event"
        )
        actual[event_index]["finish_ns"] += 0.5
        difference = self.compare(actual)
        self.assertIsNotNone(difference)
        self.assertEqual(difference.record_index, event_index)
        self.assertEqual(difference.field_path, "$.finish_ns")
        self.assertAlmostEqual(difference.absolute_difference, 0.5)

    def test_missing_event_reports_first_shift(self) -> None:
        actual = copy.deepcopy(self.expected)
        event_index = next(
            index
            for index, record in enumerate(actual)
            if record["kind"] == "event"
        )
        del actual[event_index]
        for index, record in enumerate(actual):
            record["record_index"] = index
        difference = self.compare(actual)
        self.assertIsNotNone(difference)
        self.assertEqual(difference.record_index, event_index)
        # With one service-group event per request the record after the
        # deleted event is its completion, whose key set differs first.
        self.assertIn(
            difference.field_path,
            {"$.action", "$.id", "$.kind", "$.finish_ns", "$.<keys>"},
        )

    def test_tolerance_is_not_a_golden_update_escape_hatch(self) -> None:
        actual = copy.deepcopy(self.expected)
        event = next(record for record in actual if record["kind"] == "event")
        event["finish_ns"] += 2e-9
        self.assertIsNotNone(self.compare(actual))

    def test_uint64_values_above_float_precision_compare_exactly(self) -> None:
        actual = copy.deepcopy(self.expected)
        summary = actual[-1]
        summary["address_observations"][0]["address"] = (1 << 63) + 1
        expected = copy.deepcopy(actual)
        expected[-1]["address_observations"][0]["address"] = 1 << 63
        difference = first_difference(
            expected,
            actual,
            abs_tolerance=1e-9,
            rel_tolerance=1e-12,
        )
        self.assertIsNotNone(difference)
        self.assertEqual(
            difference.field_path,
            "$.address_observations[0].address",
        )


if __name__ == "__main__":
    unittest.main()
