#!/usr/bin/env python3
"""Logical HBF lifetime operations through the persistent native protocol."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hbfsim_client.simulation_session import ResolvedSystemConfig, SimulationSession, SimulationSessionError
from hbfsim_client.transaction_protocol import Transaction, TransactionBatch

PAGE = 4096


def batch(number: int, operation: str, address: int, length: int = PAGE) -> TransactionBatch:
    return TransactionBatch(batch_id=number, logical_trace_sha256=format(number + 1, "064x"),
        routing_sidecar_sha256=format(number + 101, "064x"), transactions=(
            Transaction(id=f"b{number}/io", target="HBF_LOGICAL", op=operation,
                addr=address, bytes=length, issue_ns=0.0),), receipt={})


class InvalidationTests(unittest.TestCase):
    simulator: Path

    def open_session(self, directory: Path, **kwargs) -> SimulationSession:
        config = ResolvedSystemConfig.load((ROOT / "configs/systems/eight-stack-baseline.cfg",
            ROOT / "tests/fixtures/simulation-session-mini.cfg"))
        return SimulationSession(simulator_path=self.simulator, system_config=config,
            enable_hbm=True, enable_hbf=True, hbf_wear_output_prefix=directory / "wear", **kwargs)

    def test_buffer_death_is_causal_and_does_not_write_dead_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.open_session(Path(temporary)) as session:
                first = session.submit(batch(0, "W", 0, 64))
                self.assertEqual(first["device_delta"]["hbf"]["data_program_payload_bytes"], 0)
                receipt = session.invalidate_hbf_pages("last-reference", first_lpn=0, page_count=1)
                self.assertEqual(receipt["discarded_buffer_bytes"], 64)
                self.assertEqual(receipt["unmapped_pages"], 1)
                self.assertEqual(receipt["device_delta"]["hbf"]["physical_write_bytes"], 0)
                read = session.submit(batch(1, "R", 0))
                self.assertEqual(read["batch_origin_ns"], receipt["finish_ns"])
                self.assertEqual(read["device_delta"]["hbf"]["write_buffer_hits"], 0)
                persisted = session.checkpoint("after-free")
                self.assertEqual(persisted["device_delta"]["hbf"]["data_program_payload_bytes"], 0)
            source = session.source_receipt()
            self.assertEqual(source["logical_invalidations"], [receipt])

    def test_free_persists_and_reused_lpn_can_be_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.open_session(directory, initial_hbf_logical_pages=16) as session:
                with self.assertRaises(SimulationSessionError):
                    session.invalidate_hbf_pages("outside", first_lpn=2**63, page_count=1)
                receipt = session.invalidate_hbf_pages("completed-request", first_lpn=3, page_count=6)
                self.assertEqual(receipt["invalidated_pages"], 6)
                self.assertEqual(receipt["device_delta"]["hbf"]["mapping_update_ops"], 6)
                self.assertGreater(receipt["elapsed_ns"], 0)
                persisted = session.checkpoint_image("freed-image", directory / "image")
                self.assertGreater(persisted["device_delta"]["hbf"]["mapping_program_payload_bytes"], 0)
            with self.open_session(directory, initial_hbf_persistent_image=directory / "image") as restored:
                missing = restored.submit(batch(0, "R", 3 * PAGE))
                self.assertEqual(missing["device_delta"]["hbf"]["physical_read_bytes"], 0)
                neighbor = restored.submit(batch(1, "R", 2 * PAGE))
                self.assertGreater(neighbor["device_delta"]["hbf"]["physical_read_bytes"], 0)
                restored.submit(batch(2, "W", 3 * PAGE))
                restored.checkpoint("rewrite")
                self.assertEqual(restored.invalidate_hbf_pages("free-rewritten", first_lpn=3,
                    page_count=1)["invalidated_pages"], 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    InvalidationTests.simulator = args.simulator.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
