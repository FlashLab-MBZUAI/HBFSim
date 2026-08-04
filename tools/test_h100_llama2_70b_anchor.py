#!/usr/bin/env python3
"""Fail-closed tests for the public H100 + Llama 2 70B anchor path."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from import_h100_llama2_70b_anchor import (  # noqa: E402
    AnchorImportError,
    import_anchor,
)
from verify_h100_llama2_70b_anchor import (  # noqa: E402
    AnchorVerificationError,
    verify_anchor,
)


class PublicTimingAnchorTest(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is required")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.vidur = self.root / "vidur"
        self.mlperf = self.root / "mlperf"
        self.profile = self.vidur / "profiles/timing.csv"
        self.summary = self.mlperf / "results/server-summary.txt"
        self._write_profile(
            [
                ["1", "1", "1", "1", "0", "1"],
                ["2", "2", "2", "2", "0", "2"],
            ]
        )
        self.summary.parent.mkdir(parents=True)
        self.summary.write_text(
            "\n".join(
                [
                    "MLPerf Results Summary",
                    "Scenario : Server",
                    "Mode : PerformanceOnly",
                    "Completed samples per second : 10",
                    "Completed tokens per second: 20",
                    "Result is : VALID",
                    "No errors encountered during test.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        self._initialize_repo(self.vidur, "https://example.invalid/vidur.git")
        self._initialize_repo(self.mlperf, "https://example.invalid/mlperf.git")
        self.manifest = self.root / "anchor.json"
        self.manifest_value = self._build_manifest()
        self._write_manifest()
        self.bundle = self.root / "bundle"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run_git(self, root: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def _initialize_repo(self, root: Path, remote: str) -> None:
        self._run_git(root, "init")
        self._run_git(root, "config", "user.name", "HBFSim Test")
        self._run_git(root, "config", "user.email", "hbfsim-test@example.invalid")
        self._run_git(root, "remote", "add", "origin", remote)
        self._run_git(root, "add", "--all")
        self._run_git(root, "commit", "-m", "fixture")

    def _commit_changes(self, root: Path) -> None:
        self._run_git(root, "add", "--all")
        self._run_git(root, "commit", "-m", "fixture update")

    def _write_profile(self, rows: list[list[str]]) -> None:
        self.profile.parent.mkdir(parents=True, exist_ok=True)
        with self.profile.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "time_stats.op.min",
                    "time_stats.op.max",
                    "time_stats.op.mean",
                    "time_stats.op.median",
                    "time_stats.op.std",
                    "feature",
                ]
            )
            writer.writerows(rows)

    @staticmethod
    def _identity(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    def _source_pin(self, root: Path) -> dict[str, str]:
        return {
            "commit": self._run_git(root, "rev-parse", "HEAD"),
            "source_tree": self._run_git(root, "rev-parse", "HEAD^{tree}"),
        }

    def _build_manifest(self) -> dict:
        profile_identity = self._identity(self.profile)
        summary_identity = self._identity(self.summary)
        return {
            "schema": {
                "name": "hbfsim.validation.public-timing-anchor",
                "version": 1,
            },
            "anchor_id": "test-h100-llama2-anchor",
            "intended_use": {
                "vidur_role": "calibration_fit_input",
                "mlperf_role": "held_out_external_validation_only",
                "mlperf_fit_allowed": False,
                "bundle_alone_establishes_calibration": False,
                "bundle_alone_establishes_external_validation": False,
                "paper_result_eligible_by_itself": False,
            },
            "target": {
                "model": {
                    "id": "test/Llama-2-70b",
                    "num_layers": 1,
                    "hidden_size": 1,
                    "intermediate_size": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "tokenizer_vocab_size": 1,
                    "profiled_padded_vocab_size": 1,
                    "max_model_len": 1,
                },
                "vidur_profile": {
                    "device_label": "h100",
                    "network_label": "h100_dgx",
                    "precision": "fp16",
                    "tensor_parallel_workers": [1],
                },
                "mlperf_held_out": {
                    "system": "test H100",
                    "accelerators": 1,
                    "accelerator_memory": "80 GB HBM3",
                    "weight_precision": "fp8",
                    "framework": "test runtime",
                    "runtime_and_precision_match_vidur_fit": False,
                },
            },
            "sources": {
                "vidur": {
                    "role": "calibration_fit_input",
                    "fit_allowed": True,
                    "repository": "https://example.invalid/vidur.git",
                    **self._source_pin(self.vidur),
                    "license": "MIT",
                    "license_status": "fixture",
                    "measurement_semantics": {
                        "attention": "fixture attention timing",
                        "mlp": "fixture mlp timing",
                        "collectives": "fixture collective timing",
                    },
                    "artifacts": [
                        {
                            "id": "vidur.test_profile",
                            "kind": "csv_profile",
                            "path": "profiles/timing.csv",
                            **profile_identity,
                            "rows": 2,
                            "columns": [
                                "time_stats.op.min",
                                "time_stats.op.max",
                                "time_stats.op.mean",
                                "time_stats.op.median",
                                "time_stats.op.std",
                                "feature",
                            ],
                            "validation": {
                                "timing_prefixes": ["time_stats.op"],
                                "nullable_timing_prefixes": [],
                                "conditional_timing": [],
                                "integer_columns": ["feature"],
                                "constant_columns": {},
                                "allowed_values": {"feature": ["1", "2"]},
                                "integer_ranges": {
                                    "feature": {"min": 1, "max": 2}
                                },
                                "exact_value_counts": {
                                    "feature": {"1": 1, "2": 1}
                                },
                                "profile_key": ["feature"],
                                "expected_unique_keys": 2,
                                "expected_duplicate_rows": 0,
                            },
                        }
                    ],
                },
                "mlperf": {
                    "role": "held_out_external_validation_only",
                    "fit_allowed": False,
                    "repository": "https://example.invalid/mlperf.git",
                    **self._source_pin(self.mlperf),
                    "license": "not asserted",
                    "license_status": "fixture",
                    "public_result_id": "test-result",
                    "artifacts": [
                        {
                            "id": "mlperf.test_server",
                            "kind": "mlperf_performance_summary",
                            "path": "results/server-summary.txt",
                            **summary_identity,
                            "validation": {
                                "scenario": "Server",
                                "mode": "PerformanceOnly",
                                "tokens_per_second": "20",
                                "samples_per_second": "10",
                                "result": "VALID",
                            },
                        }
                    ],
                },
            },
            "claim_boundary": {
                "timing_claim_state_after_import": "anchor_ready_not_calibrated",
                "frontier_timing_usable_for_performance": False,
                "paper_result_eligible": False,
                "importer_can_establish": ["fixture integrity"],
                "importer_cannot_establish": ["predictor accuracy"],
                "required_next_evidence": ["candidate prediction errors"],
            },
        }

    def _write_manifest(self) -> None:
        self.manifest.write_text(
            json.dumps(self.manifest_value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _import(self) -> dict:
        return import_anchor(
            manifest_path=self.manifest,
            vidur_root=self.vidur,
            mlperf_root=self.mlperf,
            output_dir=self.bundle,
        )

    def test_import_and_independent_verification_keep_claims_closed(self) -> None:
        imported = self._import()
        receipt_path = self.root / "verification.json"
        verified = verify_anchor(
            manifest_path=self.manifest,
            bundle_dir=self.bundle,
            output=receipt_path,
        )
        self.assertEqual(imported["eligibility"], verified["eligibility"])
        self.assertEqual(
            verified["eligibility"]["timing_claim_state"],
            "anchor_ready_not_calibrated",
        )
        self.assertFalse(verified["eligibility"]["predictor_calibration_valid"])
        self.assertFalse(
            verified["eligibility"]["frontier_timing_usable_for_performance"]
        )
        self.assertTrue(verified["checks"]["mlperf_excluded_from_fit"])
        self.assertEqual(json.loads(receipt_path.read_text()), verified)
        copied = self.bundle / "sources/vidur/profiles/timing.csv"
        self.assertEqual(copied.read_bytes(), self.profile.read_bytes())

    def test_verifier_does_not_import_importer(self) -> None:
        source = (TOOLS / "verify_h100_llama2_70b_anchor.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import_h100_llama2_70b_anchor", source)

    def test_import_rejects_mlperf_as_fit_input_without_output(self) -> None:
        self.manifest_value["sources"]["mlperf"]["fit_allowed"] = True
        self._write_manifest()
        with self.assertRaisesRegex(AnchorImportError, "role separation"):
            self._import()
        self.assertFalse(self.bundle.exists())

    def test_import_rejects_nonfinite_profile_after_exact_digest(self) -> None:
        self._write_profile(
            [
                ["1", "1", "NaN", "1", "0", "1"],
                ["2", "2", "2", "2", "0", "2"],
            ]
        )
        self._commit_changes(self.vidur)
        source = self.manifest_value["sources"]["vidur"]
        source.update(self._source_pin(self.vidur))
        source["artifacts"][0].update(self._identity(self.profile))
        self._write_manifest()
        with self.assertRaisesRegex(AnchorImportError, "finite and non-negative"):
            self._import()
        self.assertFalse(self.bundle.exists())

    def test_import_rejects_profile_domain_drift_atomically(self) -> None:
        validation = self.manifest_value["sources"]["vidur"]["artifacts"][0][
            "validation"
        ]
        validation["allowed_values"]["feature"] = ["1"]
        self._write_manifest()
        with self.assertRaisesRegex(AnchorImportError, "outside"):
            self._import()
        self.assertFalse(self.bundle.exists())

    def test_import_rejects_wrong_git_pin(self) -> None:
        self.manifest_value["sources"]["vidur"]["commit"] = "0" * 40
        self._write_manifest()
        with self.assertRaisesRegex(AnchorImportError, "HEAD mismatch"):
            self._import()
        self.assertFalse(self.bundle.exists())

    def test_import_refuses_to_overwrite_bundle(self) -> None:
        self._import()
        with self.assertRaisesRegex(AnchorImportError, "already exists"):
            self._import()

    def test_verifier_rejects_copied_artifact_tamper(self) -> None:
        self._import()
        copied = self.bundle / "sources/vidur/profiles/timing.csv"
        copied.write_text(copied.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
        receipt = self.root / "tampered-verification.json"
        with self.assertRaisesRegex(AnchorVerificationError, "copied bytes"):
            verify_anchor(
                manifest_path=self.manifest,
                bundle_dir=self.bundle,
                output=receipt,
            )
        self.assertFalse(receipt.exists())

    def test_verifier_rejects_unattached_file(self) -> None:
        self._import()
        (self.bundle / "unattached.txt").write_text("extra\n", encoding="utf-8")
        receipt = self.root / "extra-verification.json"
        with self.assertRaisesRegex(AnchorVerificationError, "file census"):
            verify_anchor(
                manifest_path=self.manifest,
                bundle_dir=self.bundle,
                output=receipt,
            )
        self.assertFalse(receipt.exists())

    def test_verifier_rejects_claim_promotion(self) -> None:
        self._import()
        bundle_manifest = self.bundle / "bundle-manifest.json"
        value = json.loads(bundle_manifest.read_text(encoding="utf-8"))
        value["eligibility"]["predictor_calibration_valid"] = True
        bundle_manifest.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt = self.root / "promoted-verification.json"
        with self.assertRaisesRegex(AnchorVerificationError, "eligibility"):
            verify_anchor(
                manifest_path=self.manifest,
                bundle_dir=self.bundle,
                output=receipt,
            )
        self.assertFalse(receipt.exists())

    def test_verifier_rejects_changed_copied_manifest(self) -> None:
        self._import()
        copied_manifest = self.bundle / "anchor-manifest.json"
        copied_manifest.write_text(
            copied_manifest.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        receipt = self.root / "manifest-verification.json"
        with self.assertRaisesRegex(AnchorVerificationError, "byte-identical"):
            verify_anchor(
                manifest_path=self.manifest,
                bundle_dir=self.bundle,
                output=receipt,
            )
        self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
