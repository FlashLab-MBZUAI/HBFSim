#!/usr/bin/env python3
"""End-to-end contract for the batch frontier, error replies, and options."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbfsim_client.simulation_session import (  # noqa: E402
    BatchResult,
    ResolvedSystemConfig,
    SimulationSession,
    SimulationSessionError,
)
from hbfsim_client.transaction_protocol import (  # noqa: E402
    NO_UPSTREAM_DIGEST,
    Transaction,
    TransactionBatch,
    TransactionProtocolError,
)


BASE_CONFIG = ROOT / "configs/systems/eight-stack-baseline.cfg"
MINI_OVERLAY = ROOT / "tests/fixtures/simulation-session-mini.cfg"
NVME_OVERLAY = ROOT / "configs/overlays/backing/nvme-ssd.cfg"
PAGE = 4096


def hbm_read(identifier: str, addr: int = 0, deps: tuple[str, ...] = ()) -> Transaction:
    return Transaction(
        id=identifier,
        target="HBM",
        op="R",
        addr=addr,
        bytes=PAGE,
        issue_ns=0.0,
        dependencies=deps,
    )


def external_write(identifier: str, size: int, deps: tuple[str, ...] = ()) -> Transaction:
    return Transaction(
        id=identifier,
        target="EXTERNAL",
        op="W",
        addr=0,
        bytes=size,
        issue_ns=0.0,
        dependencies=deps,
    )


class SessionProtocolTests(unittest.TestCase):
    simulator: Path

    def test_receipt_copy_reuses_wire_data_and_graph_changes_are_revalidated(self):
        graph = [hbm_read('a'), hbm_read('b', deps=('a',))]
        batch = TransactionBatch(0, graph, completions=False, observe=('b',), frontier=('b',))
        expected = ''.join(tx.protocol_line()+'\n' for tx in graph)
        digest = hashlib.sha256(expected.encode('ascii')).hexdigest()
        self.assertEqual(batch.transaction_trace_sha256, digest)
        census = batch.census
        graph.clear()  # Constructor input ownership cannot invalidate a cache.
        with (patch.object(Transaction, 'validate', side_effect=AssertionError('revalidated')),
              patch.object(Transaction, '_protocol_line_unchecked', side_effect=AssertionError('reencoded'))):
            updated = batch.with_receipt({'detail': 'new metadata'})
            self.assertEqual(updated.transaction_trace_sha256, digest)
            self.assertIs(updated.census, census)
            for size in (1, 250, 1024*1024):
                self.assertEqual(''.join(updated.protocol_payload_chunks(target_bytes=size)), expected)
                self.assertTrue(all(chunk.endswith('\n') for chunk in updated.protocol_payload_chunks(target_bytes=size)))
        self.assertEqual(batch.receipt, {})
        self.assertEqual(updated.receipt, {'detail': 'new metadata'})
        self.assertEqual(updated.begin_line(), batch.begin_line())
        altered = replace(batch, transactions=(hbm_read('a', addr=PAGE), batch.transactions[1]))
        self.assertNotEqual(altered.transaction_trace_sha256, digest)
        with self.assertRaises(TransactionProtocolError):
            replace(batch, transactions=(replace(batch.transactions[0], bytes=0),))
        with self.assertRaises(TransactionProtocolError):
            batch.with_receipt(None)

    def test_census_matches_wire_completion_with_metadata_copy(self):
        graph = (hbm_read('a'), Transaction('fence', 'BARRIER', None, 0, 0, 0,
            duration_ns=100., dependencies=('a',)), hbm_read('b', deps=('fence',)))
        batch = TransactionBatch(0, graph, completions=False).with_receipt({'detail': True})
        with self._session() as session:
            completion = session.submit(batch)
        for key, value in batch.census['scalars'].items():
            self.assertEqual(completion[key], value)
        self.assertEqual(completion['by_target'], batch.census['by_target'])

    def test_observed_barrier_starts_at_its_dependencies(self) -> None:
        graph = (
            Transaction('start', 'BARRIER', None, 0, 0, 0),
            hbm_read('read', deps=('start',)),
            Transaction('compute', 'BARRIER', None, 0, 0, 0,
                        duration_ns=100., dependencies=('read', 'start')),
        )
        batch = TransactionBatch(0, graph, completions=False, observe=('start', 'read', 'compute'))
        with self._session() as session:
            result = session.submit(batch)
        start, read, compute = result['observed_timings']
        self.assertIsNone(result['transaction_completions'])
        self.assertEqual(start['finish_ns'], 0.)
        self.assertEqual(compute['start_ns'], read['finish_ns'])
        self.assertEqual(compute['finish_ns'], read['finish_ns'] + 100.)
        for ids in (('missing',), ('read', 'read')):
            with self.assertRaises(TransactionProtocolError):
                TransactionBatch(1, graph, observe=ids)

    def _session(self, *, external: bool = False) -> SimulationSession:
        overlays = (BASE_CONFIG, MINI_OVERLAY) + (
            (NVME_OVERLAY,) if external else ()
        )
        return SimulationSession(
            simulator_path=self.simulator,
            system_config=ResolvedSystemConfig.load(overlays),
            enable_hbm=True,
            enable_hbf=False,
            enable_external=external,
            read_timeout_s=120.0,
        )

    def test_ready_receipt_carries_source_provenance(self) -> None:
        with self._session() as session:
            source = session.engine_source
            self.assertEqual(
                set(source),
                {"git_commit", "git_dirty", "tree_hash", "source_sha256",
                 "provenance_source"},
            )
            self.assertEqual(source["provenance_source"], "build-time")
            self.assertRegex(source["source_sha256"], r"^[0-9a-f]{64}$")
            self.assertIsInstance(source["git_dirty"], bool)
            self.assertEqual(session.dependency_window_batches, 2)
            receipt = session.source_receipt()
            self.assertEqual(receipt["engine_source"], source)
            self.assertEqual(receipt["protocol"], "simulation-transaction-text-v2")

    def test_frontier_excludes_detached_offload_from_the_next_origin(self) -> None:
        with self._session(external=True) as session:
            first = session.run(
                [hbm_read("w0"), external_write("offload", 64 << 20)],
                frontier=("w0",),
            )
            self.assertIsInstance(first, BatchResult)
            read_done = first.completion("w0").finish_ns
            offload_done = first.completion("offload").finish_ns
            self.assertGreater(offload_done, read_done * 1000)
            self.assertEqual(first.frontier_transactions, 1)
            self.assertEqual(first.blocking_finish_ns, read_done)
            self.assertEqual(first.finish_ns, offload_done)
            self.assertAlmostEqual(first.elapsed_ns, read_done - first.batch_origin_ns)
            self.assertAlmostEqual(
                first.total_elapsed_ns, offload_done - first.batch_origin_ns
            )
            self.assertEqual(session.completed_frontier_ns, read_done)
            self.assertEqual(session.issued_work_frontier_ns, offload_done)

            second = session.run([hbm_read("w1", addr=PAGE)])
            self.assertEqual(second.batch_origin_ns, read_done)
            self.assertEqual(second.completion("w1").arrival_ns, read_done)

            third = session.run(
                [
                    Transaction(
                        id="restore",
                        target="EXTERNAL",
                        op="R",
                        addr=0,
                        bytes=PAGE,
                        issue_ns=0.0,
                        dependencies=("offload",),
                    )
                ]
            )
            self.assertGreaterEqual(third.completion("restore").arrival_ns, offload_done)

            # Empty frontier: only the last arrival advances the origin.
            fourth = session.run([external_write("tail", PAGE)], frontier=())
            self.assertEqual(fourth.frontier_transactions, 0)
            self.assertEqual(fourth.blocking_finish_ns, fourth.batch_origin_ns)
            self.assertGreater(fourth.finish_ns, fourth.blocking_finish_ns)

    def test_input_errors_are_structured_and_leave_the_session_usable(self) -> None:
        with self._session() as session:
            with self.assertRaisesRegex(
                SimulationSessionError, "rejected batch 3: .*exceeds application capacity"
            ):
                session.submit(
                    TransactionBatch(
                        batch_id=3,
                        transactions=(hbm_read("bad", addr=1 << 40),),
                    )
                )
            with self.assertRaisesRegex(SimulationSessionError, "rejected batch 4"):
                session.submit(
                    TransactionBatch(
                        batch_id=4,
                        transactions=(hbm_read("orphan", deps=("nowhere",)),),
                    )
                )
            with self.assertRaisesRegex(SimulationSessionError, "rejected batch 5"):
                session.submit(
                    TransactionBatch(
                        batch_id=5,
                        transactions=(
                            Transaction(
                                id="hbf/disabled",
                                target="HBF_LOGICAL",
                                op="R",
                                addr=0,
                                bytes=PAGE,
                                issue_ns=0.0,
                            ),
                        ),
                    )
                )
            # The rejected ids were never consumed and nothing advanced.
            self.assertEqual(session.completed_frontier_ns, 0.0)
            receipt = session.submit(
                TransactionBatch(batch_id=3, transactions=(hbm_read("bad"),))
            )
            self.assertEqual(receipt["sequence"], 0)
            self.assertEqual(receipt["result"], "pass")

    def test_hbf_logical_capacity_is_an_input_error(self) -> None:
        session = SimulationSession(
            simulator_path=self.simulator,
            system_config=ResolvedSystemConfig.load((BASE_CONFIG, MINI_OVERLAY)),
            enable_hbm=True,
            enable_hbf=True,
            read_timeout_s=120.0,
        )
        with session:
            with self.assertRaisesRegex(
                SimulationSessionError,
                "rejected batch 0: .*beyond the HBF logical capacity",
            ):
                session.run(
                    [
                        Transaction(
                            id="hbf/beyond",
                            target="HBF_LOGICAL",
                            op="W",
                            addr=1 << 30,
                            bytes=PAGE,
                            issue_ns=0.0,
                        )
                    ]
                )
            self.assertEqual(session.completed_frontier_ns, 0.0)
            inside = session.run(
                [
                    Transaction(
                        id="hbf/inside",
                        target="HBF_LOGICAL",
                        op="W",
                        addr=0,
                        bytes=PAGE,
                        issue_ns=0.0,
                    )
                ]
            )
            self.assertEqual(inside.receipt["sequence"], 0)
            self.assertGreater(session.completed_frontier_ns, 0.0)

    def test_completions_are_optional_per_batch(self) -> None:
        with self._session() as session:
            compact = session.run([hbm_read("a"), hbm_read("b", addr=PAGE)], completions=False)
            self.assertEqual(compact.completions, ())
            self.assertIsNone(compact.receipt["transaction_completions"])
            self.assertIsNone(compact.receipt["transaction_completions_digest"])
            self.assertEqual(
                compact.receipt["transaction_latency_by_target"]["HBM"]["read"][
                    "transactions"
                ],
                2,
            )
            full = session.run([hbm_read("c")])
            self.assertEqual(len(full.completions), 1)
            self.assertEqual(full.batch_origin_ns, compact.blocking_finish_ns)

    def test_dependency_window_and_retain(self) -> None:
        with self._session() as session:
            session.run([hbm_read("keep")], retain=("keep",))
            session.run([hbm_read("b")], retain=("keep",))
            session.run([hbm_read("c")], retain=("keep",))
            kept = session.run([hbm_read("d", deps=("keep",))], retain=("keep",))
            self.assertEqual(kept.receipt["dependency_edges"], 1)
            # Batches without a retain declaration (the default) leave the
            # retained set alone, so a second producer sharing the session
            # cannot undo the first one's declaration.
            session.run([hbm_read("e")])
            session.run([hbm_read("f")])
            still = session.run([hbm_read("g", deps=("keep",))])
            self.assertEqual(still.receipt["dependency_edges"], 1)
            # An explicit empty list clears it.
            session.run([hbm_read("h")], retain=())
            session.run([hbm_read("i")])
            with self.assertRaisesRegex(
                SimulationSessionError, "does not name a transaction"
            ):
                session.run([hbm_read("j", deps=("keep",))])
            with self.assertRaisesRegex(SimulationSessionError, "retain list"):
                session.run([hbm_read("k")], retain=("never-existed",))

    def test_batch_defaults_and_validation(self) -> None:
        batch = TransactionBatch(batch_id=0, transactions=(hbm_read("x"),))
        self.assertEqual(batch.logical_trace_sha256, NO_UPSTREAM_DIGEST)
        self.assertEqual(batch.receipt, {})
        self.assertIsNone(batch.frontier)
        self.assertIsNone(batch.retain)
        self.assertTrue(batch.begin_line().startswith("BEGIN 0 " + NO_UPSTREAM_DIGEST))
        self.assertEqual(len(batch.begin_line().split(" ")), 4)
        self.assertEqual(
            TransactionBatch(
                batch_id=1,
                transactions=(hbm_read("x"),),
                frontier=(),
                retain=("older",),
                completions=False,
            ).begin_line().split(" ")[4:],
            ["frontier=-", "retain=older", "completions=0"],
        )
        self.assertEqual(
            TransactionBatch(
                batch_id=1, transactions=(hbm_read("x"),), retain=()
            ).begin_line().split(" ")[4:],
            ["retain=-"],
        )
        with self.assertRaises(TransactionProtocolError):
            TransactionBatch(
                batch_id=2, transactions=(hbm_read("x"),), frontier=("absent",)
            )
        with self.assertRaises(TransactionProtocolError):
            TransactionBatch(
                batch_id=2,
                transactions=(hbm_read("x"),),
                logical_trace_sha256="Z" * 64,
            )

    def test_context_manager_does_not_chain_on_the_caller_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "caller failure"):
            with self._session() as session:
                session.run([hbm_read("a")])
                raise RuntimeError("caller failure")
        self.assertTrue(session._closed)  # noqa: SLF001 - contract under test

    def test_failed_terminal_receipt_is_preserved_without_becoming_success(self) -> None:
        import hbfsim_client.simulation_session as client

        session = self._session()
        session.run([hbm_read("terminal-evidence")])
        original_readline = session._process.stdout.readline
        original_validator = client._validate_latency_matrix
        received = {}
        original_errors = []

        def invalid_terminal_line():
            row = json.loads(original_readline())
            self.assertEqual(row["result"], "stopped")
            row["transaction_latency_by_target"]["HBM"]["read"]["latency_work_ns"] += 1e9
            # A large extra field verifies that failure evidence is complete,
            # rather than a truncated log tail or a reconstructed subset.
            row["diagnostic_canary"] = "terminal evidence " * 5000
            received["decoded"] = row
            received["raw"] = json.dumps(row, separators=(",", ":")) + "\n"
            return received["raw"]

        def remember_original_error(*args, **kwargs):
            try:
                return original_validator(*args, **kwargs)
            except SimulationSessionError as error:
                original_errors.append(error)
                raise

        with (patch.object(session._process.stdout, "readline", side_effect=invalid_terminal_line),
              patch.object(client, "_validate_latency_matrix", side_effect=remember_original_error)):
            with self.assertRaisesRegex(SimulationSessionError, "latency work does not conserve") as caught:
                session.close()
        self.assertIs(caught.exception, original_errors[0])
        self.assertTrue(session._closed)
        source = session.source_receipt()
        self.assertIsNone(source["final_measurement"])
        failure = source["terminal_failure"]
        self.assertIs(failure["validated"], False)
        self.assertEqual(failure["validation_status"], "failed")
        self.assertEqual(failure["command"], "QUIT")
        self.assertEqual(failure["raw_response"], received["raw"])
        self.assertEqual(failure["decoded_response"], received["decoded"])
        self.assertEqual(failure["simulator_executable"], source["simulator_executable"])
        self.assertEqual(failure["engine_source"], source["engine_source"])
        self.assertEqual(failure["exception"]["message"], str(caught.exception))
        self.assertIn("_validate_latency_matrix", failure["exception"]["traceback"])
        # The receipt hands out copies: a caller cannot alter preserved evidence.
        source["terminal_failure"]["decoded_response"]["diagnostic_canary"] = "changed copy"
        self.assertEqual(session.source_receipt()["terminal_failure"]["decoded_response"], received["decoded"])
        # Closing again neither re-raises nor rewrites the preserved failure.
        session.close()
        self.assertEqual(session.source_receipt()["terminal_failure"]["raw_response"], received["raw"])

        with self._session() as success:
            success.run([hbm_read("terminal-success-control")])
        self.assertEqual(success.source_receipt()["final_measurement"]["result"], "stopped")
        self.assertNotIn("terminal_failure", success.source_receipt())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    SessionProtocolTests.simulator = arguments.simulator.resolve()
    unittest.main(argv=["test_simulation_session_protocol.py"])
