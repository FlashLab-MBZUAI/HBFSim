#!/usr/bin/env python3
"""End-to-end contract for nonterminal HBF session checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbfsim_client.simulation_session import (  # noqa: E402
    ResolvedSystemConfig,
    SimulationSession,
    SimulationSessionError,
)
from hbfsim_client.transaction_protocol import (  # noqa: E402
    Transaction,
    TransactionBatch,
)


BASE_CONFIG = ROOT / "configs/systems/eight-stack-baseline.cfg"
MINI_OVERLAY = ROOT / "tests/fixtures/simulation-session-mini.cfg"
DIRECT_LANE_OVERLAY = ROOT / "configs/overlays/hbf/external-direct-lane.cfg"
PAGE = 4096


def write_batch(batch_id: int, address: int) -> TransactionBatch:
    return TransactionBatch(
        batch_id=batch_id,
        logical_trace_sha256=format(batch_id + 1, "064x"),
        routing_sidecar_sha256=format(batch_id + 101, "064x"),
        transactions=(
            Transaction(
                id=f"batch{batch_id}/write",
                target="HBF_LOGICAL",
                op="W",
                addr=address,
                bytes=PAGE,
                issue_ns=0.0,
            ),
        ),
        receipt={},
    )


def physical_batch(batch_id: int, operation: str) -> TransactionBatch:
    return TransactionBatch(
        batch_id=batch_id,
        logical_trace_sha256=format(batch_id + 201, "064x"),
        routing_sidecar_sha256=format(batch_id + 301, "064x"),
        transactions=(
            Transaction(
                id=f"physical{batch_id}/{operation}",
                target="HBF_PHYSICAL",
                op=operation,
                addr=0,
                bytes=PAGE,
                issue_ns=0.0,
            ),
        ),
        receipt={},
    )


class SimulationCheckpointTests(unittest.TestCase):
    simulator: Path

    def test_checkpoint_persists_then_session_continues_causally(self) -> None:
        config = ResolvedSystemConfig.load((BASE_CONFIG, MINI_OVERLAY))
        session = SimulationSession(
            simulator_path=self.simulator,
            system_config=config,
            enable_hbm=True,
            enable_hbf=True,
        )
        try:
            first = session.submit(write_batch(0, 0))
            self.assertEqual(
                first["transaction_completions"],
                [
                    {
                        "id": "batch0/write",
                        "arrival_ns": first["transaction_completions"][0][
                            "arrival_ns"
                        ],
                        "start_ns": first["transaction_completions"][0][
                            "start_ns"
                        ],
                        "finish_ns": first["transaction_completions"][0][
                            "finish_ns"
                        ],
                        "logical_bytes": PAGE,
                        "physical_bytes": first["transaction_completions"][0][
                            "physical_bytes"
                        ],
                    }
                ],
            )
            transaction_completion = first["transaction_completions"][0]
            self.assertLessEqual(
                transaction_completion["arrival_ns"],
                transaction_completion["start_ns"],
            )
            self.assertLessEqual(
                transaction_completion["start_ns"],
                transaction_completion["finish_ns"],
            )
            self.assertEqual(
                first["transaction_completions_digest"]["algorithm"],
                "sha256_id_ieee754bits_bytes_v1",
            )
            self.assertEqual(
                len(first["transaction_completions_digest"]["sha256"]), 64
            )
            checkpoint0 = session.checkpoint("periodic-0")
            self.assertEqual(
                checkpoint0["arrival_frontier_ns"], first["finish_ns"]
            )
            self.assertGreater(checkpoint0["elapsed_ns"], 0)
            self.assertGreater(checkpoint0["physical_bytes"], 0)
            self.assertEqual(
                checkpoint0["device_delta"]["hbf"]["logical_write_bytes"],
                0,
            )
            self.assertGreater(
                checkpoint0["device_delta"]["hbf"][
                    "mapping_program_payload_bytes"
                ],
                0,
            )
            with self.assertRaisesRegex(
                SimulationSessionError, "duplicated"
            ):
                session.checkpoint("periodic-0")

            second = session.submit(write_batch(1, PAGE))
            self.assertEqual(
                second["batch_origin_ns"], checkpoint0["finish_ns"]
            )
            checkpoint1 = session.checkpoint("periodic-1")
            self.assertEqual(checkpoint1["sequence"], 1)
        finally:
            session.close()

        source = session.source_receipt()
        self.assertEqual(
            [row["checkpoint_id"] for row in source["lifecycle_checkpoints"]],
            ["periodic-0", "periodic-1"],
        )
        final = source["final_measurement"]
        self.assertEqual(final["completed_batches"], 2)
        self.assertEqual(final["completed_checkpoints"], 2)
        self.assertEqual(
            final["completed_frontier_ns"], checkpoint1["finish_ns"]
        )
        self.assertEqual(
            final["end_of_session_drain"]["drain_physical_bytes"], 0
        )
        self.assertIn(
            "checkpoint_scope", final["measurement_semantics"]
        )

    def test_published_extent_is_append_then_read_only_after_reopen(self) -> None:
        config = ResolvedSystemConfig.load((BASE_CONFIG, MINI_OVERLAY))
        with tempfile.TemporaryDirectory(
            prefix="hbf-session-published-extent-"
        ) as name:
            image_path = Path(name) / "published.hbfstate"
            producer = SimulationSession(
                simulator_path=self.simulator,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
                published_hbf_blocks_per_plane=1,
            )
            appended = producer.submit(physical_batch(0, "W"))
            hbf = appended["device_delta"]["hbf"]
            self.assertEqual(hbf["raw_physical_programs"], 1)
            self.assertEqual(
                hbf["raw_physical_program_payload_bytes"], PAGE
            )
            self.assertEqual(hbf["waf"], 1.0)
            self.assertEqual(
                hbf["waf_definition"],
                "physical_write_bytes/(logical_write_bytes+"
                "raw_physical_program_payload_bytes)",
            )
            self.assertEqual(hbf["mapping_update_ops"], 0)
            checkpoint = producer.checkpoint_image("publish", image_path)
            producer.crash("published-boundary")

            recovered = SimulationSession(
                simulator_path=self.simulator,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
                published_hbf_blocks_per_plane=1,
                initial_hbf_persistent_image=image_path,
            )
            direct = recovered.submit(physical_batch(0, "R"))
            self.assertEqual(
                direct["device_delta"]["hbf"]["mapping_lookup_ops"], 0
            )
            self.assertEqual(
                recovered.source_receipt()["execution_options"]
                ["published_hbf_blocks_per_plane"],
                1,
            )
            with self.assertRaisesRegex(
                SimulationSessionError, "published extent"
            ):
                recovered.submit(physical_batch(1, "W"))
            self.assertEqual(
                checkpoint["persistent_image"]["sha256"],
                recovered.source_receipt()["execution_options"]
                ["initial_hbf_persistent_image"]["sha256"],
            )
            recovered.close()

    def test_checkpoint_rejects_disabled_hbf_and_bad_identifier(self) -> None:
        config = ResolvedSystemConfig.load((BASE_CONFIG, MINI_OVERLAY))
        with SimulationSession(
            simulator_path=self.simulator,
            system_config=config,
            enable_hbm=True,
            enable_hbf=False,
        ) as no_hbf:
            with self.assertRaisesRegex(
                SimulationSessionError, "enabled HBF"
            ):
                no_hbf.checkpoint("impossible")

        with SimulationSession(
            simulator_path=self.simulator,
            system_config=config,
            enable_hbm=True,
            enable_hbf=True,
        ) as hbf:
            with self.assertRaisesRegex(
                SimulationSessionError, "protocol-safe"
            ):
                hbf.checkpoint("not safe")

    def test_explicit_crash_never_performs_terminal_drain(self) -> None:
        config = ResolvedSystemConfig.load((BASE_CONFIG, MINI_OVERLAY))
        dirty = SimulationSession(
            simulator_path=self.simulator,
            system_config=config,
            enable_hbm=True,
            enable_hbf=True,
        )
        write = dirty.submit(write_batch(0, 0))
        crashed = dirty.crash("after-uncheckpointed-write")
        self.assertEqual(crashed["completed_batches"], 1)
        self.assertEqual(crashed["completed_checkpoints"], 0)
        self.assertEqual(crashed["completed_frontier_ns"], write["finish_ns"])
        self.assertFalse(crashed["terminal_drain_performed"])
        dirty_state = crashed["quiescence_at_injection"]
        self.assertFalse(dirty_state["verified"])
        self.assertTrue(any(
            value > 0
            for key, value in dirty_state.items()
            if key != "verified"
        ))
        self.assertEqual(
            dirty.source_receipt()["final_measurement"], crashed
        )
        with self.assertRaisesRegex(SimulationSessionError, "closed"):
            dirty.checkpoint("impossible-after-crash")

        with tempfile.TemporaryDirectory(
            prefix="hbf-session-checkpointed-crash-"
        ) as name:
            image_path = Path(name) / "durable.hbfstate"
            durable = SimulationSession(
                simulator_path=self.simulator,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
            )
            durable.submit(write_batch(0, 0))
            checkpoint = durable.checkpoint_image("durable", image_path)
            clean_crash = durable.crash("after-successful-checkpoint")
            self.assertTrue(
                clean_crash["quiescence_at_injection"]["verified"]
            )
            self.assertEqual(clean_crash["completed_checkpoints"], 1)
            self.assertEqual(
                clean_crash["completed_frontier_ns"],
                checkpoint["finish_ns"],
            )
            self.assertTrue(image_path.is_file())

            recovered = SimulationSession(
                simulator_path=self.simulator,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
                initial_hbf_persistent_image=image_path,
            )
            recovered.close()
            self.assertEqual(
                recovered.source_receipt()["execution_options"]
                ["initial_hbf_persistent_image"]["sha256"],
                checkpoint["persistent_image"]["sha256"],
            )

    def test_checkpoint_image_reopens_in_a_fresh_process(self) -> None:
        config = ResolvedSystemConfig.load((BASE_CONFIG, MINI_OVERLAY))
        with tempfile.TemporaryDirectory(
            prefix="hbf-session-persistent-image-"
        ) as name:
            image_path = Path(name) / "checkpoint.hbfstate"
            producer = SimulationSession(
                simulator_path=self.simulator,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
            )
            continuation = write_batch(1, PAGE)
            try:
                initial_write = producer.submit(write_batch(0, 0))
                checkpoint = producer.checkpoint_image(
                    "mutable-image-0", image_path
                )
                continued = producer.submit(continuation)
            finally:
                producer.close()
            producer_source = producer.source_receipt()

            self.assertTrue(image_path.is_file())
            self.assertEqual(
                checkpoint["persistent_image"]["bytes"],
                image_path.stat().st_size,
            )
            consumer = SimulationSession(
                simulator_path=self.simulator,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
                initial_hbf_persistent_image=image_path,
            )
            try:
                resumed = consumer.submit(continuation)
            finally:
                consumer.close()
            consumer_source = consumer.source_receipt()

            self.assertAlmostEqual(
                continued["elapsed_ns"], resumed["elapsed_ns"], places=9
            )
            fields = (
                "logical_write_bytes",
                "physical_read_bytes",
                "physical_write_bytes",
                "data_program_payload_bytes",
                "mapping_program_payload_bytes",
                "gc_relocation_payload_bytes",
                "mapping_lookup_ops",
                "mapping_update_ops",
                "gc_runs",
                "block_erases",
            )
            continued_hbf = continued["device_delta"]["hbf"]
            resumed_hbf = resumed["device_delta"]["hbf"]
            self.assertEqual(
                {field: continued_hbf[field] for field in fields},
                {field: resumed_hbf[field] for field in fields},
            )
            setup = consumer.source_receipt()["execution_options"]
            self.assertEqual(setup["initial_hbf_logical_image"]["mode"], "none")
            restored = setup["initial_hbf_persistent_image"]
            self.assertEqual(restored["sha256"], checkpoint["persistent_image"]["sha256"])
            self.assertEqual(restored["logical_data_pages"], 1)
            self.assertGreater(restored["mapping_pages"], 0)
            self.assertEqual(restored["block_erase_count_sum"], initial_write["device_delta"]["hbf"]["block_erases"] + checkpoint["device_delta"]["hbf"]["block_erases"])
            producer_drain = producer_source["final_measurement"][
                "end_of_session_drain"
            ]["drain_physical_bytes"]
            consumer_drain = consumer_source["final_measurement"][
                "end_of_session_drain"
            ]["drain_physical_bytes"]
            self.assertGreater(producer_drain, 0)
            self.assertEqual(consumer_drain, producer_drain)

    def test_compact_image_reopens_without_expanding_l2p(self) -> None:
        config = ResolvedSystemConfig.load((BASE_CONFIG, MINI_OVERLAY))
        with tempfile.TemporaryDirectory(
            prefix="hbf-session-compact-persistent-image-"
        ) as name:
            directory = Path(name)
            baseline_path = directory / "compact-baseline.hbfstate"
            producer_final_path = directory / "producer-final.hbfstate"
            consumer_final_path = directory / "consumer-final.hbfstate"
            producer = SimulationSession(
                simulator_path=self.simulator,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
                initial_hbf_logical_pages=4,
            )
            continuation = write_batch(0, 0)
            try:
                baseline = producer.checkpoint_image(
                    "compact-baseline", baseline_path
                )
                continued = producer.submit(continuation)
                producer_final = producer.checkpoint_image(
                    "compact-producer-final", producer_final_path
                )
            finally:
                producer.close()

            consumer = SimulationSession(
                simulator_path=self.simulator,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
                initial_hbf_persistent_image=baseline_path,
            )
            try:
                resumed = consumer.submit(continuation)
                consumer_final = consumer.checkpoint_image(
                    "compact-consumer-final", consumer_final_path
                )
            finally:
                consumer.close()

            self.assertEqual(
                baseline["persistent_image"]["schema"],
                {"name": "hbfsim.hbf_persistent_image", "version": 2},
            )
            self.assertAlmostEqual(
                continued["elapsed_ns"], resumed["elapsed_ns"], places=9
            )
            self.assertEqual(
                continued["device_delta"]["hbf"],
                resumed["device_delta"]["hbf"],
            )
            self.assertEqual(
                producer_final["persistent_image"]["sha256"],
                consumer_final["persistent_image"]["sha256"],
            )
            restored = consumer.source_receipt()["execution_options"][
                "initial_hbf_persistent_image"
            ]
            self.assertEqual(restored["encoding"], "compact_v2")
            self.assertEqual(restored["logical_data_pages"], 4)
            self.assertEqual(restored["compact_logical_data_pages"], 4)
            self.assertEqual(restored["compact_mapping_pages"], 2)


    def test_direct_hbf_external_lane_round_trip(self) -> None:
        config = ResolvedSystemConfig.load((
            BASE_CONFIG,
            MINI_OVERLAY,
            ROOT / "configs/overlays/backing/cxl-ssd.cfg",
            DIRECT_LANE_OVERLAY,
        ))
        chain = (
            ("demote/populate", "HBF_LOGICAL", "W", None, ()),
            ("demote/read-hbf", "HBF_LOGICAL", "R", None, (
                "demote/populate",)),
            ("demote/lane", "DIRECT_HBF_TO_EXTERNAL", "R", 0, (
                "demote/read-hbf",)),
            ("demote/write-external", "EXTERNAL", "W", None, (
                "demote/lane",)),
            ("restore/read-external", "EXTERNAL", "R", None, (
                "demote/write-external",)),
            ("restore/lane", "DIRECT_EXTERNAL_TO_HBF", "W", 0, (
                "restore/read-external",)),
            ("restore/write-hbf", "HBF_LOGICAL", "W", None, (
                "restore/lane",)),
        )
        batch = TransactionBatch(
            batch_id=1,
            logical_trace_sha256=format(1, "064x"),
            routing_sidecar_sha256=format(2, "064x"),
            transactions=tuple(
                Transaction(
                    id=name,
                    target=target,
                    op=op,
                    addr=0,
                    bytes=PAGE,
                    issue_ns=0.0,
                    dependencies=dependencies,
                    stack=stack,
                )
                for name, target, op, stack, dependencies in chain
            ),
            receipt={},
        )
        session = SimulationSession(
            simulator_path=self.simulator,
            system_config=config,
            enable_hbm=True,
            enable_hbf=True,
            enable_external=True,
        )
        try:
            result = session.submit(batch)
        finally:
            session.close()
        finishes = {
            item["id"]: float(item["finish_ns"])
            for item in result["transaction_completions"]
        }
        order = [name for name, *_ in chain]
        for earlier, later in zip(order, order[1:]):
            self.assertGreater(finishes[later], finishes[earlier])
        # Declared lane envelope: 100 ns fixed latency plus 4 KiB at
        # 32 GB/s = 128 ns of serialization in each direction.
        lane_cost = finishes["demote/lane"] - finishes["demote/read-hbf"]
        self.assertAlmostEqual(lane_cost, 228.0, places=6)
        census = result["by_target"]
        self.assertEqual(
            census["DIRECT_HBF_TO_EXTERNAL"],
            {"transactions": 1, "bytes": PAGE},
        )
        self.assertEqual(
            census["DIRECT_EXTERNAL_TO_HBF"],
            {"transactions": 1, "bytes": PAGE},
        )
        lane_totals = result["device_delta"]["hbf_external_direct_link"]
        self.assertEqual(lane_totals["read_bytes"], PAGE)
        self.assertEqual(lane_totals["write_bytes"], PAGE)
        self.assertEqual(
            session.source_receipt()["execution_options"][
                "hbf_external_direct_link"
            ],
            {
                "links": 2,
                "read_bandwidth_GBps": 32.0,
                "write_bandwidth_GBps": 32.0,
                "latency_ns": 100.0,
            },
        )

    def test_direct_lane_is_opt_in(self) -> None:
        config = ResolvedSystemConfig.load((
            BASE_CONFIG,
            MINI_OVERLAY,
            ROOT / "configs/overlays/backing/cxl-ssd.cfg",
        ))
        session = SimulationSession(
            simulator_path=self.simulator,
            system_config=config,
            enable_hbm=True,
            enable_hbf=True,
            enable_external=True,
        )
        try:
            self.assertIsNone(
                session.source_receipt()["execution_options"][
                    "hbf_external_direct_link"
                ]
            )
            with self.assertRaisesRegex(
                SimulationSessionError, "rejected batch 7"
            ):
                session.submit(
                    TransactionBatch(
                        batch_id=7,
                        transactions=(
                            Transaction(
                                id="lane/without-envelope",
                                target="DIRECT_HBF_TO_EXTERNAL",
                                op="R",
                                addr=0,
                                bytes=PAGE,
                                issue_ns=0.0,
                                stack=0,
                            ),
                        ),
                    )
                )
            # The rejection left the session usable.
            self.assertEqual(session.submit(write_batch(0, 0))["sequence"], 0)
        finally:
            session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    SimulationCheckpointTests.simulator = arguments.simulator.resolve()
    unittest.main(argv=["test_simulation_session_checkpoint.py"])
