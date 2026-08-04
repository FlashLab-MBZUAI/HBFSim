#!/usr/bin/env python3
"""Structural contracts for server-scale and output-topology configs."""

from __future__ import annotations

import sys
import unittest
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "scenario_compare"
COMM_BASE = 0x6_0000_0000  # 24 GiB in the ASTRA address contract


def load_config(name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (CONFIG_DIR / name).read_text().splitlines():
        payload = line.split("#", 1)[0].strip()
        if not payload:
            continue
        key, separator, value = payload.partition("=")
        if not separator:
            raise ValueError(f"{name}: invalid config line {line!r}")
        values[key.strip()] = value.strip()
    return values


class ExperimentConfigTests(unittest.TestCase):
    BASELINE_CONFIG = "usecase-baseline.cfg"
    LOCAL_OUTPUT_CONFIG = "usecase-baseline-local-output.cfg"
    OUTPUT_UPPER_BOUND_CONFIG = "usecase-baseline-output-upper-bound.cfg"
    EXPECTED_STACKS = {
        "usecase-baseline.cfg": (8, 8),
        "usecase-6h2f.cfg": (6, 2),
        "usecase-4h4f.cfg": (4, 4),
        "usecase-2h6f.cfg": (2, 6),
    }
    PUBLISHED_CONFIGS = tuple(path.name for path in sorted(CONFIG_DIR.glob("*.cfg")))

    @staticmethod
    def differing_keys(
        left: dict[str, str], right: dict[str, str]
    ) -> set[str]:
        return {
            key
            for key in left.keys() | right.keys()
            if left.get(key) != right.get(key)
        }

    def test_stack_points_and_page_size_are_fixed(self) -> None:
        for name, expected in self.EXPECTED_STACKS.items():
            with self.subTest(config=name):
                config = load_config(name)
                self.assertEqual(
                    (int(config["hbm-stacks"]), int(config["hbf-stacks"])),
                    expected,
                )
                self.assertEqual(int(config["hbf-page-size"]), 4096)

    def test_hbm_interface_is_derived_from_width_rate_and_bl(self) -> None:
        for name in self.PUBLISHED_CONFIGS:
            with self.subTest(config=name):
                config = load_config(name)
                self.assertNotIn("hbm-channel-bw", config)
                channels = int(config["hbm-channels"])
                pseudo_channels = int(config["hbm-pseudo-channels"])
                channel_width_bits = int(config["hbm-channel-width-bits"])
                burst_length = int(config["hbm-burst-length"])
                pin_rate = float(config["hbm-pin-rate-gbps"])
                command_ratio = int(
                    config["hbm-data-rate-per-command-clock"])
                channel_row_bytes = int(
                    config["hbm-channel-row-size-bytes"])
                tccd_s_cycles = int(config["hbm-tccd-s-cycles"])
                tccd_l_cycles = int(config["hbm-tccd-l-cycles"])

                self.assertEqual(channel_width_bits % pseudo_channels, 0)
                pseudo_width_bits = channel_width_bits // pseudo_channels
                self.assertEqual(pseudo_width_bits % 8, 0)
                burst_bytes = pseudo_width_bits // 8 * burst_length
                self.assertEqual(channel_row_bytes % pseudo_channels, 0)
                pseudo_row_bytes = channel_row_bytes // pseudo_channels
                self.assertEqual(pseudo_row_bytes % burst_bytes, 0)
                self.assertEqual(burst_length % command_ratio, 0)
                burst_cycles = burst_length // command_ratio
                self.assertGreaterEqual(tccd_s_cycles, burst_cycles)
                self.assertGreaterEqual(tccd_l_cycles, tccd_s_cycles)
                channel_GBps = pin_rate * channel_width_bits / 8
                stack_GBps = channels * channel_GBps
                self.assertTrue(math.isfinite(stack_GBps))
                self.assertGreater(stack_GBps, 0.0)

                if name in self.EXPECTED_STACKS:
                    self.assertEqual(channel_width_bits, 64)
                    self.assertEqual(pseudo_channels, 2)
                    self.assertEqual(burst_length, 8)
                    self.assertEqual(
                        int(config["hbm-bank-groups-per-pseudo-channel"]), 16)
                    self.assertEqual(int(config["hbm-banks-per-group"]), 4)
                    self.assertAlmostEqual(pin_rate, 6.4)
                    self.assertAlmostEqual(stack_GBps, 1638.4)
                    self.assertAlmostEqual(burst_bytes / (pin_rate * pseudo_width_bits / 8), 1.25)

    def test_direct_and_layer_streaming_hbm_regions_are_disjoint(self) -> None:
        for name in self.EXPECTED_STACKS:
            with self.subTest(config=name):
                config = load_config(name)
                capacity = int(config["hbm-capacity-bytes"])
                cooperative = int(config["hbf-hbm-write-buffer-bytes"])
                direct_limit = capacity - cooperative
                self.assertLessEqual(int(config["flat-hbm-bytes"]), direct_limit)
                self.assertLessEqual(
                    int(config["static-direct-hbm-bytes"]), direct_limit)
                streaming_bytes = 2 * int(config["layer-buffer-bytes"])
                self.assertLess(streaming_bytes, capacity)

    def test_layer_buffers_are_page_aligned_and_explicit(self) -> None:
        for name in self.PUBLISHED_CONFIGS:
            with self.subTest(config=name):
                config = load_config(name)
                layer_bytes = int(config["layer-buffer-bytes"])
                self.assertGreater(layer_bytes, 0)
                self.assertEqual(layer_bytes % int(config["hbf-page-size"]), 0)

    def test_two_hbm_profile_leaves_space_above_comm_base(self) -> None:
        config = load_config("usecase-2h6f.cfg")
        foreground_bytes = (
            int(config["hbm-capacity-bytes"]) -
            2 * int(config["layer-buffer-bytes"])
        )
        self.assertGreater(foreground_bytes, COMM_BASE)

    def test_ecc_pipeline_can_feed_each_profile_interface(self) -> None:
        for name in self.PUBLISHED_CONFIGS:
            with self.subTest(config=name):
                config = load_config(name)
                page_bytes = int(config["hbf-page-size"])
                codeword_bytes = page_bytes + int(config["hbf-oob-bytes"])
                dies_per_stack = (
                    int(config["hbf-channels"]) *
                    int(config["hbf-dies-per-channel"])
                )
                channels_per_stack = int(config["hbf-channels"])
                interface_GBps = float(config["hbf-hbio-bw"])
                required_raw_GBps = (
                    interface_GBps * codeword_bytes / page_bytes
                )
                required_tsv_GBps = (
                    interface_GBps * (codeword_bytes + 64) / page_bytes
                )
                self.assertAlmostEqual(
                    channels_per_stack * float(config["hbf-channel-bw"]),
                    required_raw_GBps,
                    places=9,
                    msg=f"aggregate raw channel bandwidth drifted in {name}",
                )
                self.assertAlmostEqual(
                    float(config["hbf-tsv-bw"]),
                    required_tsv_GBps,
                    places=9,
                    msg=f"command+raw TSV bandwidth drifted in {name}",
                )
                for direction in ("decode", "encode"):
                    raw_GBps_per_die = float(
                        config[f"hbf-ecc-{direction}-raw-bw"])
                    self.assertAlmostEqual(
                        dies_per_stack * raw_GBps_per_die,
                        required_raw_GBps,
                        places=9,
                        msg=f"aggregate {direction} raw bandwidth drifted in {name}",
                    )
                    payload_ceiling_GBps = (
                        dies_per_stack * raw_GBps_per_die *
                        page_bytes / codeword_bytes
                    )
                    self.assertAlmostEqual(
                        payload_ceiling_GBps,
                        interface_GBps,
                        places=9,
                        msg=f"{direction} ECC provisioning drifted from HBIO in {name}",
                    )
                expected_latency_ns = (
                    250.0 if name.startswith("usecase-") else 500.0)
                self.assertEqual(
                    float(config["hbf-ecc-decode-latency-ns"]),
                    expected_latency_ns,
                )
                self.assertEqual(
                    float(config["hbf-ecc-encode-latency-ns"]),
                    expected_latency_ns,
                )
                self.assertEqual(
                    float(config["hbf-ecc-decode-raw-bw"]),
                    float(config["hbf-ecc-encode-raw-bw"]),
                )

    def test_ec_topology_points_share_one_media_timing_profile(self) -> None:
        names = (
            *self.EXPECTED_STACKS,
            self.LOCAL_OUTPUT_CONFIG,
            self.OUTPUT_UPPER_BOUND_CONFIG,
        )
        expected = {
            "hbf-read-ns": "1000",
            "hbf-program-ns": "95000",
            "hbf-program-verify-ns": "5000",
            "hbf-ecc-decode-latency-ns": "250",
            "hbf-ecc-encode-latency-ns": "250",
        }
        for name in names:
            with self.subTest(config=name):
                config = load_config(name)
                self.assertEqual(
                    {key: config[key] for key in expected},
                    expected,
                    msg="an EC topology point changed media timing",
                )

    def test_controlled_output_profiles_change_only_declared_fields(self) -> None:
        baseline = load_config(self.BASELINE_CONFIG)
        local = load_config(self.LOCAL_OUTPUT_CONFIG)
        upper = load_config(self.OUTPUT_UPPER_BOUND_CONFIG)

        local_fields = {
            "hbf-media-lanes-per-plane",
            "hbf-page-buffer-banks-per-plane",
        }
        upper_fields = {
            "hbf-ecc-decode-raw-bw",
            "hbf-ecc-encode-raw-bw",
            "hbf-channel-bw",
            "hbf-hbio-bw",
            "hbf-tsv-bw",
            "hbf-logic-sram-bw",
            "hbf-flash-tsu-issue-ns",
            "hbf-logic-scheduler-issue-ns",
        }
        self.assertEqual(self.differing_keys(baseline, local), local_fields)
        self.assertEqual(self.differing_keys(local, upper), upper_fields)
        self.assertEqual(
            self.differing_keys(baseline, upper), local_fields | upper_fields)

        subarrays = int(baseline["hbf-subarrays-per-plane"])
        self.assertEqual(subarrays, 32)
        self.assertEqual(int(baseline["hbf-media-lanes-per-plane"]), 16)
        self.assertEqual(int(baseline["hbf-page-buffer-banks-per-plane"]), 16)
        self.assertEqual(int(local["hbf-media-lanes-per-plane"]), subarrays)
        self.assertEqual(
            int(local["hbf-page-buffer-banks-per-plane"]), subarrays)
        self.assertEqual(int(upper["hbf-media-lanes-per-plane"]), subarrays)
        self.assertEqual(
            int(upper["hbf-page-buffer-banks-per-plane"]), subarrays)

    def test_output_upper_bound_matches_array_supply(self) -> None:
        baseline = load_config(self.BASELINE_CONFIG)
        local = load_config(self.LOCAL_OUTPUT_CONFIG)
        upper = load_config(self.OUTPUT_UPPER_BOUND_CONFIG)

        for key in (
            "hbf-ecc-decode-raw-bw",
            "hbf-ecc-encode-raw-bw",
            "hbf-channel-bw",
            "hbf-hbio-bw",
            "hbf-tsv-bw",
        ):
            self.assertEqual(
                local[key], baseline[key],
                msg=f"local-output sensitivity widened shared fabric field {key}")

        page_bytes = int(upper["hbf-page-size"])
        codeword_bytes = page_bytes + int(upper["hbf-oob-bytes"])
        planes_per_stack = (
            int(upper["hbf-channels"]) *
            int(upper["hbf-dies-per-channel"]) *
            int(upper["hbf-planes-per-die"])
        )
        subarrays_per_plane = int(upper["hbf-subarrays-per-plane"])
        read_ns = float(upper["hbf-read-ns"])
        payload_supply_GBps = (
            planes_per_stack * subarrays_per_plane * page_bytes / read_ns
        )
        raw_data_supply_GBps = (
            payload_supply_GBps * codeword_bytes / page_bytes
        )
        command_bytes = 64
        tsv_supply_GBps = (
            payload_supply_GBps *
            (codeword_bytes + command_bytes) / page_bytes
        )
        channels_per_stack = int(upper["hbf-channels"])
        dies_per_stack = (
            channels_per_stack * int(upper["hbf-dies-per-channel"])
        )

        self.assertAlmostEqual(payload_supply_GBps, 8388.608, places=9)
        self.assertAlmostEqual(raw_data_supply_GBps, 8847.36, places=9)
        self.assertAlmostEqual(tsv_supply_GBps, 8978.432, places=9)
        self.assertAlmostEqual(
            float(upper["hbf-hbio-bw"]), payload_supply_GBps, places=9)
        self.assertAlmostEqual(
            float(upper["hbf-logic-sram-bw"]),
            payload_supply_GBps,
            places=9,
        )
        self.assertAlmostEqual(
            float(upper["hbf-tsv-bw"]), tsv_supply_GBps, places=9)
        self.assertAlmostEqual(
            channels_per_stack * float(upper["hbf-channel-bw"]),
            raw_data_supply_GBps,
            places=9,
        )
        for direction in ("decode", "encode"):
            self.assertAlmostEqual(
                dies_per_stack * float(
                    upper[f"hbf-ecc-{direction}-raw-bw"]),
                raw_data_supply_GBps,
                places=9,
            )
        self.assertAlmostEqual(
            float(upper["hbf-logic-scheduler-issue-ns"]),
            page_bytes / payload_supply_GBps,
            places=12,
        )
        self.assertAlmostEqual(
            float(upper["hbf-flash-tsu-issue-ns"]),
            read_ns / (
                int(upper["hbf-planes-per-die"]) *
                subarrays_per_plane),
            places=12,
        )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExperimentConfigTests)
    result = unittest.TextTestRunner().run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
