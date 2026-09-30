#!/usr/bin/env python3
"""Unit contract for digest-bound per-transaction completion export."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbfsim_client.simulation_session import (  # noqa: E402
    SimulationSessionError,
    _transaction_completion_digest,
    _validate_transaction_completions,
)
from hbfsim_client.transaction_protocol import (  # noqa: E402
    TRANSACTION_TARGETS,
    Transaction,
    TransactionBatch,
)


def _empty_stats() -> dict[str, int | float]:
    return {
        "transactions": 0,
        "logical_bytes": 0,
        "physical_bytes": 0,
        "queue_wait_work_ns": 0.0,
        "service_work_ns": 0.0,
        "latency_work_ns": 0.0,
        "min_latency_ns": 0.0,
        "max_latency_ns": 0.0,
        "first_arrival_ns": 0.0,
        "finish_ns": 0.0,
    }


class TransactionCompletionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = TransactionBatch(
            batch_id=0,
            logical_trace_sha256="1" * 64,
            routing_sidecar_sha256="2" * 64,
            transactions=(
                Transaction(
                    id="external/read",
                    target="EXTERNAL",
                    op="R",
                    addr=0,
                    bytes=4096,
                    issue_ns=0.0,
                ),
            ),
            receipt={},
        )
        self.completions = [
            {
                "id": "external/read",
                "arrival_ns": 0.0,
                "start_ns": 5.0,
                "finish_ns": 15.0,
                "logical_bytes": 4096,
                "physical_bytes": 8192,
            }
        ]
        self.latency = {
            target: {"read": _empty_stats(), "write": _empty_stats()}
            for target in TRANSACTION_TARGETS
        }
        self.latency["EXTERNAL"]["read"] = {
            "transactions": 1,
            "logical_bytes": 4096,
            "physical_bytes": 8192,
            "queue_wait_work_ns": 5.0,
            "service_work_ns": 10.0,
            "latency_work_ns": 15.0,
            "min_latency_ns": 15.0,
            "max_latency_ns": 15.0,
            "first_arrival_ns": 0.0,
            "finish_ns": 15.0,
        }

    def _digest(self) -> dict[str, str]:
        return {
            "algorithm": "sha256_id_ieee754bits_bytes_v1",
            "sha256": _transaction_completion_digest(self.completions),
        }

    def test_accepts_exact_id_time_and_byte_conservation(self) -> None:
        result = _validate_transaction_completions(
            self.completions,
            self._digest(),
            batch=self.batch,
            latency_matrix=self.latency,
            batch_origin_ns=0.0,
            batch_finish_ns=15.0,
        )
        self.assertEqual(result[0]["id"], "external/read")
        self.assertEqual(result[0]["logical_bytes"], 4096)
        self.assertEqual(result[0]["physical_bytes"], 8192)

    def test_rejects_tampered_completion_under_frozen_digest(self) -> None:
        digest = self._digest()
        tampered = [dict(self.completions[0], finish_ns=14.0)]
        with self.assertRaisesRegex(
            SimulationSessionError,
            "digest does not reproduce|do not conserve",
        ):
            _validate_transaction_completions(
                tampered,
                digest,
                batch=self.batch,
                latency_matrix=self.latency,
                batch_origin_ns=0.0,
                batch_finish_ns=15.0,
            )


if __name__ == "__main__":
    unittest.main()
