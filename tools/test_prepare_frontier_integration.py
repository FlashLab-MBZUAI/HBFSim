#!/usr/bin/env python3
"""Static fail-closed tests for the pinned Frontier integration bundle."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from prepare_frontier_integration import (  # noqa: E402
    BUNDLE_MANIFEST,
    validate_bundle,
)


class PrepareFrontierIntegrationTest(unittest.TestCase):
    def test_tracked_bundle_is_self_consistent(self) -> None:
        manifest, patch_path, patch_bytes = validate_bundle()
        self.assertEqual(
            manifest["integration_id"],
            "hbfsim.frontier-hybrid-residency.v8",
        )
        self.assertEqual(
            manifest["upstream"]["revision"],
            "a4b22df8211864bf229258ecdfbe680f048f2d77",
        )
        self.assertTrue(patch_path.is_file())
        self.assertIn(
            b"frontier_hbfsim_integration.json",
            patch_bytes,
        )
        self.assertIn(
            b'MEMORY_CONTRACT_SCHEMA_VERSION = 4',
            patch_bytes,
        )
        self.assertIn(
            b'RESIDENCY_PLAN_SCHEMA_VERSION = 1',
            patch_bytes,
        )

    def test_manifest_and_patch_cover_the_same_exact_path_set(self) -> None:
        original = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
        expected = set(original["post_apply_sha256"])
        self.assertIn(
            "frontier/kv_cache/hbfsim_memory_contract.py",
            expected,
        )
        self.assertIn("frontier/config/model_config.py", expected)
        self.assertIn("frontier/config/config.py", expected)
        self.assertIn("frontier/config/utils.py", expected)
        self.assertIn(
            "data/config/models/llama31_70b.json",
            expected,
        )
        self.assertIn("frontier/utils/param_counter.py", expected)
        self.assertIn("frontier/metrics/metrics_store.py", expected)
        self.assertIn(
            "frontier/scheduler/utils/memory_planner.py",
            expected,
        )


if __name__ == "__main__":
    unittest.main()
