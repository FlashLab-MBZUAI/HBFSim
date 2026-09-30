#!/usr/bin/env python3
"""Hardware keys have one parser: the engine's SystemConfigBuilder.

`hbfsim-reference` forwards every engine-owned key (config files and
`--<key> VALUE` overrides alike) to the same builder the core `hbfsim` uses
and exports the resolved hardware with `--config-out`. This test proves that
export covers every key of every published system profile, that the export
replays to an identical summary, that CLI overrides reach the builder, and
that the runner keeps rejecting keys nobody owns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_system_configs import published_system_profiles  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
POLICY_CONFIGS = ROOT / "configs/policies/reference"
# Only policy-paired profiles are complete replay inputs; configs/systems/
# also holds frontend overlay fragments (test_system_configs.py lists them).
SYSTEM_CONFIGS = [
    ROOT / "configs/systems" / name for name in published_system_profiles()]
# Keys the export legitimately rewrites into their resolved spelling.
RESOLVED_SPELLINGS = {
    "hbf-capacity-ratio": "hbf-blocks-per-plane",
    "hbf-capacity-bytes": "hbf-blocks-per-plane",
    "hbf-ctrl-dram-capacity-denominator": "hbf-ctrl-dram-capacity-denominator",
}
VOLATILE_SUMMARY_KEYS = {"invocation", "build", "simulator"}


def parse_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = line.split("#", 1)[0].strip()
        if not payload:
            continue
        key, separator, value = payload.partition("=")
        if not separator:
            raise ValueError(f"{path}: not a key=value line: {line!r}")
        values[key.strip()] = value.strip()
    return values


def same_value(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except ValueError:
        return False


class ReferenceConfigRoundTripTest(unittest.TestCase):
    reference: Path

    def run_reference(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.reference), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )

    def generate_trace(self, root: Path) -> Path:
        trace = root / "trace.txt"
        completed = self.run_reference(
            "--generate-semantic-llm", str(trace),
            "--llm-tokens", "1",
            "--llm-layers", "1",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return trace

    def test_config_out_covers_every_published_key_and_replays(self) -> None:
        self.assertTrue(SYSTEM_CONFIGS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self.generate_trace(root)
            for system_config in SYSTEM_CONFIGS:
                with self.subTest(config=system_config.name):
                    resolved = root / f"{system_config.stem}.resolved.cfg"
                    first = root / f"{system_config.stem}.first.json"
                    second = root / f"{system_config.stem}.second.json"
                    completed = self.run_reference(
                        "--config", str(system_config),
                        "--config", str(POLICY_CONFIGS / system_config.name),
                        "--trace", str(trace),
                        "--max-ops", "1",
                        "--scenarios", "all-hbm",
                        "--config-out", str(resolved),
                        "--summary-json", str(first),
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    published = parse_config(system_config)
                    exported = parse_config(resolved)
                    for key, value in published.items():
                        target = RESOLVED_SPELLINGS.get(key, key)
                        self.assertIn(target, exported, f"{key} missing from export")
                        if target == key:
                            self.assertTrue(
                                same_value(value, exported[key]),
                                f"{key}: published {value!r} exported {exported[key]!r}",
                            )
                    # The export is a complete replay artifact on its own.
                    replay = self.run_reference(
                        "--config", str(resolved),
                        "--summary-json", str(second),
                    )
                    self.assertEqual(replay.returncode, 0, replay.stderr)
                    before = json.loads(first.read_text(encoding="utf-8"))
                    after = json.loads(second.read_text(encoding="utf-8"))
                    for key in VOLATILE_SUMMARY_KEYS:
                        before.pop(key, None)
                        after.pop(key, None)
                    self.assertEqual(before, after)

    def test_cli_override_reaches_the_engine_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self.generate_trace(root)
            resolved = root / "override.cfg"
            summary = root / "override.json"
            completed = self.run_reference(
                "--config", str(ROOT / "configs/systems/server-hbm128-hbf512.cfg"),
                "--config", str(POLICY_CONFIGS / "server-hbm128-hbf512.cfg"),
                "--trace", str(trace),
                "--max-ops", "1",
                "--scenarios", "all-hbm",
                "--hbm-stacks", "2",
                "--hbf-read-ns", "3210.5",
                "--config-out", str(resolved),
                "--summary-json", str(summary),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            exported = parse_config(resolved)
            document = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual(exported["hbm-stacks"], "2")
        self.assertEqual(float(exported["hbf-read-ns"]), 3210.5)
        self.assertEqual(document["config"]["hbm"]["stacks"], 2)
        self.assertEqual(document["config"]["hbf"]["read_ns"], 3210.5)
        # The flat boundary follows the engine's HBM capacity unless set.
        self.assertEqual(
            document["config"]["flat_hbm_bytes"],
            int(exported["flat-hbm-bytes"]),
        )

    def test_unowned_and_malformed_keys_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = self.generate_trace(root)
            unknown = self.run_reference(
                "--trace", str(trace), "--hbm-frobnicate", "1")
            self.assertNotEqual(unknown.returncode, 0)
            self.assertIn("unknown option", unknown.stderr)
            malformed = self.run_reference(
                "--trace", str(trace), "--hbm-stacks", "two")
            self.assertNotEqual(malformed.returncode, 0)
            self.assertIn("--hbm-stacks", malformed.stderr)
            engine_only = self.run_reference(
                "--trace", str(trace), "--hbf-page-size", "8192")
            self.assertNotEqual(engine_only.returncode, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    ReferenceConfigRoundTripTest.reference = args.reference.resolve()
    unittest.main(argv=[__file__, *remaining])
