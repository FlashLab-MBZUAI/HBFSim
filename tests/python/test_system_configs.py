#!/usr/bin/env python3
"""Structural contracts for physical systems and reference-policy profiles."""

from __future__ import annotations

import math
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "systems"
POLICY_CONFIG_DIR = ROOT / "configs" / "policies" / "reference"
SIMULATION_CMAKE = ROOT / "cmake" / "tests" / "Simulation.cmake"
COMM_BASE = 0x6_0000_0000  # 24 GiB in the ASTRA address contract
# A system profile is published when a same-named reference policy pairs with
# it. Anything else under configs/systems/ is a frontend overlay fragment that
# the published-profile contracts must not enumerate.
UNPAIRED_SYSTEM_FRAGMENTS = {"sglang-small.cfg": "Overlay on eight-stack-baseline.cfg"}
PUBLISHED_LIST = re.compile(
    r"set\(\s*HBFSIM_PUBLISHED_SCENARIO_CONFIGS\s+([^)]*)\)", re.S)


def published_system_profiles() -> tuple[str, ...]:
    """File names of the system profiles that have a same-named reference policy."""

    return tuple(sorted(
        path.name for path in CONFIG_DIR.glob("*.cfg")
        if (POLICY_CONFIG_DIR / path.name).is_file()
    ))


def cmake_published_profiles() -> tuple[str, ...]:
    """The pair names CTest publishes, parsed from Simulation.cmake."""

    match = PUBLISHED_LIST.search(SIMULATION_CMAKE.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(
            f"{SIMULATION_CMAKE}: no set(HBFSIM_PUBLISHED_SCENARIO_CONFIGS ...) block")
    names = []
    for token in match.group(1).split():
        if not token.endswith(".cfg"):
            raise AssertionError(f"unexpected entry {token!r} in the published list")
        names.append(token)
    return tuple(sorted(names))


def leading_comment(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("#").strip())
    return " ".join(lines)


def load_config(name: str, directory: Path = CONFIG_DIR) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (directory / name).read_text().splitlines():
        payload = line.split("#", 1)[0].strip()
        if not payload:
            continue
        key, separator, value = payload.partition("=")
        if not separator:
            raise ValueError(f"{name}: invalid config line {line!r}")
        values[key.strip()] = value.strip()
    return values


def load_reference_config(name: str) -> dict[str, str]:
    return load_config(name) | load_config(name, POLICY_CONFIG_DIR)


class SystemConfigTests(unittest.TestCase):
    BASELINE_CONFIG = "eight-stack-baseline.cfg"
    EXPECTED_STACKS = {
        "eight-stack-baseline.cfg": (8, 8),
        "6hbm-2hbf.cfg": (6, 2),
        "4hbm-4hbf.cfg": (4, 4),
        "2hbm-6hbf.cfg": (2, 6),
    }
    PUBLISHED_CONFIGS = published_system_profiles()

    def test_published_profiles_are_the_policy_paired_set_ctest_runs(self) -> None:
        self.assertTrue(self.PUBLISHED_CONFIGS)
        self.assertEqual(set(self.PUBLISHED_CONFIGS), set(cmake_published_profiles()))
        for name in self.PUBLISHED_CONFIGS:
            self.assertNotIn(name, UNPAIRED_SYSTEM_FRAGMENTS)

    def test_only_declared_overlay_fragments_are_unpaired(self) -> None:
        unpaired = {
            path.name for path in CONFIG_DIR.glob("*.cfg")
        } - set(self.PUBLISHED_CONFIGS)
        self.assertEqual(unpaired, set(UNPAIRED_SYSTEM_FRAGMENTS))
        for name, declaration in UNPAIRED_SYSTEM_FRAGMENTS.items():
            with self.subTest(config=name):
                self.assertIn(declaration, leading_comment(CONFIG_DIR / name))

    def test_system_profiles_are_physical_only(self) -> None:
        policy_keys = {
            "line-size",
            "interarrival-ns",
            "flat-hbm-bytes",
            "static-direct-hbm-bytes",
            "layer-buffer-bytes",
            "hbf-hbm-write-buffer-bytes",
        }
        self.assertEqual(
            set(self.PUBLISHED_CONFIGS),
            {path.name for path in POLICY_CONFIG_DIR.glob("*.cfg")},
        )
        for name in self.PUBLISHED_CONFIGS:
            with self.subTest(config=name):
                self.assertTrue(policy_keys.isdisjoint(load_config(name)))
                self.assertTrue(
                    set(load_config(name, POLICY_CONFIG_DIR)) <= policy_keys
                )

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
                config = load_reference_config(name)
                self.assertEqual(
                    (int(config["hbm-stacks"]), int(config["hbf-stacks"])),
                    expected,
                )
                self.assertEqual(int(config["hbf-page-size"]), 4096)

    def test_hbm_interface_is_derived_from_width_rate_and_bl(self) -> None:
        for name in self.PUBLISHED_CONFIGS:
            with self.subTest(config=name):
                config = load_reference_config(name)
                channels = int(config["hbm-channels"])
                pseudo_channels = int(config["hbm-pseudo-channels"])
                channel_width_bits = int(config["hbm-channel-width-bits"])
                burst_length = int(config["hbm-burst-length"])
                pin_rate = float(config["hbm-pin-rate-gbps"])
                command_ratio = int(
                    config["hbm-data-rate-per-command-clock"])
                channel_row_bytes = int(
                    config["hbm-channel-row-size-bytes"])

                self.assertEqual(channel_width_bits % pseudo_channels, 0)
                pseudo_width_bits = channel_width_bits // pseudo_channels
                self.assertEqual(pseudo_width_bits % 8, 0)
                burst_bytes = pseudo_width_bits // 8 * burst_length
                self.assertEqual(channel_row_bytes % pseudo_channels, 0)
                pseudo_row_bytes = channel_row_bytes // pseudo_channels
                self.assertEqual(pseudo_row_bytes % burst_bytes, 0)
                self.assertEqual(burst_length % command_ratio, 0)
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
                    self.assertAlmostEqual(pin_rate, 8.0)
                    self.assertAlmostEqual(stack_GBps, 2048.0)
                    self.assertAlmostEqual(
                        burst_bytes / (pin_rate * pseudo_width_bits / 8),
                        1.0,
                    )

    def test_direct_and_layer_streaming_hbm_regions_are_disjoint(self) -> None:
        for name in self.EXPECTED_STACKS:
            with self.subTest(config=name):
                config = load_reference_config(name)
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
                config = load_reference_config(name)
                layer_bytes = int(config["layer-buffer-bytes"])
                self.assertGreater(layer_bytes, 0)
                self.assertEqual(layer_bytes % int(config["hbf-page-size"]), 0)

    def test_two_hbm_profile_leaves_space_above_comm_base(self) -> None:
        config = load_reference_config("2hbm-6hbf.cfg")
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
                interface_GBps = channels_per_stack * {1:48, 2:96, 3:192}[int(config["hbf-speed-grade"])]
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
                    250.0
                    if name in {
                        *self.EXPECTED_STACKS,
                    }
                    else 500.0
                )
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

    def test_composition_topology_points_share_one_media_timing_profile(self) -> None:
        names = (
            *self.EXPECTED_STACKS,
        )
        expected = {
            "hbf-read-ns": "4000",
            "hbf-program-ns": "75000",
            "hbf-ecc-decode-latency-ns": "250",
            "hbf-ecc-encode-latency-ns": "250",
        }
        for name in names:
            with self.subTest(config=name):
                config = load_config(name)
                self.assertEqual(
                    {key: config[key] for key in expected},
                    expected,
                    msg="a composition topology point changed media timing",
                )

    def test_ocp_profiles_have_one_bank_sense_and_device_transport(self) -> None:
        for name in self.PUBLISHED_CONFIGS:
            with self.subTest(config=name):
                config = load_config(name)
                self.assertEqual(config['hbf-standard'], 'OCP-HBF-0.7.0-2026-08-03')
                self.assertEqual(config['hbm-standard'], 'JEDEC-JESD270-4-2025-04')
                self.assertEqual(int(config['hbf-channels']), 16)
                self.assertEqual(int(config['hbf-dies-per-channel']), 1)
                self.assertEqual(int(config['hbf-planes-per-die']), 16)
                self.assertEqual(int(config['hbf-media-lanes-per-plane']), 1)
                self.assertEqual(int(config['hbf-page-buffer-banks-per-plane']), 2)
                self.assertGreater(float(config['hbm-bandwidth-efficiency']), 0.0)
                self.assertLessEqual(float(config['hbm-bandwidth-efficiency']), 1.0)
                self.assertTrue({'hbf-hbio-bw', 'hbf-subarrays-per-plane',
                    'hbf-batch-activation', 'hbf-read-buffer-pages',
                    'host-hbf-dram-bandwidth-gbps'}.isdisjoint(config))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SystemConfigTests)
    result = unittest.TextTestRunner().run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
