#!/usr/bin/env python3
"""Integration contract for the per-batch HBF physical live-heat stream."""

from __future__ import annotations

import argparse
import json
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


SIMULATOR = ROOT / "build/hbfsim"
BASE_CONFIG = ROOT / "configs/systems/eight-stack-baseline.cfg"
MINI_CONFIG = ROOT / "tests/fixtures/simulation-session-mini.cfg"


def _batch(batch_id: int, lpn: int) -> TransactionBatch:
    return TransactionBatch(
        batch_id=batch_id,
        logical_trace_sha256=f"{batch_id:x}" * 64,
        routing_sidecar_sha256=f"{batch_id + 8:x}" * 64,
        transactions=(
            Transaction(
                id=f"write-{batch_id}",
                target="HBF_LOGICAL",
                op="W",
                addr=lpn * 4096,
                bytes=4096,
                issue_ns=0.0,
            ),
        ),
        receipt={},
    )


class HbfLiveHeatStreamTests(unittest.TestCase):
    def test_v3_stream_separates_mapping_writes_and_conserves_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hbfsim-live-heat-") as temporary:
            directory = Path(temporary)
            overlay = directory / "cached.cfg"
            overlay.write_text(
                "\n".join(
                    (
                        "hbf-mapping-mode=cached",
                        # Enough data pages per stack for three mapping VPNs
                        # (stack = lpn % stacks, VPN = (lpn // stacks) // 512
                        # entries) inside the derived logical capacity, so the
                        # LPN sequence below crosses VPNs at 1024 and 2048.
                        "hbf-blocks-per-plane=128",
                        # Two stacks: directory (4 mapping pages x 8 B per
                        # stack) + one cache page and one Host GC copy page per stack.
                        # Crossing mapping VPNs forces dirty eviction.
                        "hbf-ctrl-dram-bytes=16448",
                        "hbf-write-coalescing=false",
                        "hbf-thermal-enable=false",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            stream = directory / "heat.jsonl"
            config = ResolvedSystemConfig.load(
                (BASE_CONFIG, MINI_CONFIG, overlay)
            )
            # The stream is a session option (CLI flags through the client),
            # never process environment: nothing here leaks into other runs.
            session = SimulationSession(
                simulator_path=SIMULATOR,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
                hbf_physical_heatmap=stream,
                hbf_physical_heatmap_bins=16,
            )
            try:
                lpns = (0, 512, 1024, 1536, 2048, 2560, 0)
                for batch_id, lpn in enumerate(lpns):
                    session.submit(_batch(batch_id, lpn))
                session.checkpoint("heat-drain")
                wear = session.hbf_wear_snapshot("heat-final")
                self.assertEqual(
                    session.source_receipt()["execution_options"][
                        "hbf_physical_heatmap"
                    ],
                    {"path": str(stream.resolve()), "bins": 16},
                )
            finally:
                session.close()

            rows = [
                json.loads(line)
                for line in stream.read_text(encoding="utf-8").splitlines()
            ]
            header, events = rows[0], rows[1:]
            batches = [row for row in events if row["kind"] == "batch"]
            self.assertEqual(header["schema"], "hbfsim.hbf_physical_live.v3")
            self.assertEqual(header["kind"], "header")
            self.assertEqual(header["bins"], 16)
            self.assertEqual(
                header["counter_scope"], "delta_since_previous_event"
            )
            self.assertEqual(
                header["event_kinds"], ["batch", "checkpoint", "zone", "stop"]
            )
            self.assertEqual(len(batches), len(lpns))
            self.assertEqual(
                [row["kind"] for row in events[-2:]], ["checkpoint", "stop"]
            )
            self.assertEqual(
                len(wear["block_erase_counts"]), wear["writable_blocks"]
            )
            self.assertEqual(
                sum(wear["block_erase_counts"]),
                wear["block_erase_count_sum"],
            )

            vector_fields = (
                "write",
                "gc_write",
                "wl_write",
                "mapping_write",
                "erase",
                "read",
                "valid",
                "invalid",
                "free",
                "pending",
                "erase_count",
                "pec_min",
                "pec_max",
                "raw_blocks",
                "data_blocks",
                "mapping_blocks",
                "gc_blocks",
                "free_blocks",
                "static_blocks",
            )
            pages_per_bin = (
                header["stacks"]
                * header["channels_per_stack"]
                * header["dies_per_channel"]
                * header["planes_per_die"]
                * header["blocks_per_plane"]
                * header["pages_per_block"]
                // header["bins"]
            )
            blocks_per_bin = pages_per_bin // header["pages_per_block"]
            for row in events:
                for field in vector_fields:
                    self.assertEqual(len(row[field]), header["bins"])
                self.assertLessEqual(sum(row["gc_write"]), sum(row["write"]))
                self.assertLessEqual(
                    sum(row["mapping_write"]), sum(row["write"])
                )
                for bin_index in range(header["bins"]):
                    self.assertEqual(
                        sum(
                            row[field][bin_index]
                            for field in ("valid", "invalid", "free", "pending")
                        ),
                        pages_per_bin,
                    )
                    self.assertEqual(
                        sum(
                            row[field][bin_index]
                            for field in (
                                "data_blocks",
                                "mapping_blocks",
                                "gc_blocks",
                                "free_blocks",
                                "static_blocks",
                                "raw_blocks",
                            )
                        ),
                        blocks_per_bin,
                    )

            total_mapping_bytes = sum(
                sum(row["mapping_write"]) for row in events
            )
            total_mapping_programs = sum(
                row["gc"]["mapping_page_programs"] for row in events
            )
            self.assertGreater(total_mapping_bytes, 0)
            self.assertEqual(
                total_mapping_bytes,
                total_mapping_programs * header["page_size_bytes"],
            )


    def test_stream_flags_must_be_given_together(self) -> None:
        config = ResolvedSystemConfig.load((BASE_CONFIG, MINI_CONFIG))
        with self.assertRaises(SimulationSessionError):
            SimulationSession(
                simulator_path=SIMULATOR,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
                hbf_physical_heatmap=Path("/dev/null"),
            )
        with self.assertRaises(SimulationSessionError):
            SimulationSession(
                simulator_path=SIMULATOR,
                system_config=config,
                enable_hbm=True,
                enable_hbf=True,
                hbf_physical_heatmap_bins=16,
            )


def main() -> int:
    global SIMULATOR
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", type=Path, default=SIMULATOR)
    args, remaining = parser.parse_known_args()
    SIMULATOR = args.simulator.resolve()
    program = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
