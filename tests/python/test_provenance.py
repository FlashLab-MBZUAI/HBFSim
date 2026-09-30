#!/usr/bin/env python3
"""Contract for the study-result provenance stamp and the clean-tree gate."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hbfsim_client.provenance import (  # noqa: E402
    ProvenanceError,
    require_clean_tree,
    run_provenance,
    stamp_result,
)


class ProvenanceTests(unittest.TestCase):
    def test_run_provenance_describes_source_binary_and_invocation(self) -> None:
        provenance = run_provenance(
            ROOT / "CMakeLists.txt",
            config_paths=(ROOT / "configs/systems/eight-stack-baseline.cfg",),
            extra={"study": "unit"},
        )
        self.assertEqual(
            {
                "repository",
                "git_commit",
                "git_dirty",
                "tree_hash",
                "provenance_source",
                "simulator",
                "configs",
                "runner",
                "argv",
                "working_directory",
                "timestamp_utc",
                "python",
                "platform",
                "extra",
            },
            set(provenance),
        )
        self.assertEqual(provenance["extra"], {"study": "unit"})
        self.assertEqual(len(provenance["simulator"]["sha256"]), 64)
        self.assertEqual(len(provenance["configs"]), 1)
        self.assertEqual(len(provenance["configs"][0]["sha256"]), 64)
        if provenance["provenance_source"] == "run-time":
            self.assertEqual(len(provenance["git_commit"]), 40)
            self.assertEqual(len(provenance["tree_hash"]), 40)
            self.assertIsInstance(provenance["git_dirty"], bool)

    def test_clean_tree_gate(self) -> None:
        clean = {"git_commit": "a" * 40, "git_dirty": False, "tree_hash": "b" * 40}
        require_clean_tree(clean)
        with self.assertRaisesRegex(ProvenanceError, "allow-dirty"):
            require_clean_tree({**clean, "git_dirty": True})
        with self.assertRaisesRegex(ProvenanceError, "unavailable"):
            require_clean_tree({"git_commit": "unknown", "git_dirty": None})

    def test_stamp_marks_dirty_results_exploratory(self) -> None:
        clean = {"git_commit": "a" * 40, "git_dirty": False, "tree_hash": "b" * 40}
        result = stamp_result({"status": "done", "simulator": {}}, clean)
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["source"], clean)
        dirty = stamp_result({"status": "done"}, {**clean, "git_dirty": True})
        self.assertEqual(dirty["status"], "exploratory_dirty")


if __name__ == "__main__":
    unittest.main()
