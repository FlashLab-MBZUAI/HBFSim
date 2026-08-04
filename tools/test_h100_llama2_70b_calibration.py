#!/usr/bin/env python3
"""Regression and adversarial tests for the H100 public calibration chain."""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
FIT = ROOT / "tools/fit_h100_llama2_70b_calibration.py"
VERIFY = ROOT / "tools/verify_h100_llama2_70b_calibration.py"
EXTRACT = ROOT / "tools/extract_mlperf_openorca_lengths.py"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _md5(raw: bytes) -> str:
    return hashlib.md5(raw, usedforsecurity=False).hexdigest()


def _csv(header: list[str], rows: list[list[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class _Fixture:
    def __init__(self, root: Path, *, nonlinear: bool = False) -> None:
        self.root = root
        self.bundle = root / "anchor"
        self.workload = root / "workload"
        self.protocol = root / "protocol.json"
        self.anchor_verification = root / "anchor-verification.json"
        self.candidate = root / "candidate"
        self.verification = root / "candidate-verification.json"
        self.bundle.mkdir(parents=True)
        self.workload.mkdir()

        values = [["1", str(x), str(x * x if nonlinear else x)] for x in range(1, 8)]
        profile = _csv(["tp", "x", "time"], values)
        base_config = b"""class GPUBaseConfig:\n    trtllm_build_flags = {'max_input_len': 4, 'max_seq_len': 4 + 4}\n    trtllm_runtime_flags = {'batch_scheduler_policy': 'max_util', 'context_chunking_policy': 'first_come_first_served'}\n"""
        scenario_config = b"""class HopperSCENARIOGPUBaseConfig:\n    precision = 'fp8'\n    trtllm_checkpoint_flags = {'kv_cache_dtype': 'fp8'}\n\nclass H100_SXM_80GB_PP2x1:\n    gpu_batch_size = {'llama2-70b': 8}\n    trtllm_build_flags = {'tensor_parallelism': 1, 'pipeline_parallelism': 2, 'max_num_tokens': 4}\n\nclass H100_SXM_80GB_PP2x4:\n    system = KnownSystem.H100_SXM_80GBx8\n"""
        offline_config = scenario_config.replace(b"SCENARIO", b"Offline")
        server_config = scenario_config.replace(b"SCENARIO", b"Server")
        preprocessor = b"G_MAX_INPUT_TOK_LEN = 4\n"
        generation = _json_bytes(
            {"generation_config": {"max_output_len": 4, "streaming": True}}
        )
        qsl = [2, 0, 3, 1]
        offline_events = {
            "loaded_qsl_set": qsl,
            "generated_query_count": 1,
            "generated_samples_per_query": 8,
            "effective_target_qps": 3,
            "result_samples_per_second": 2.5,
            "result_tokens_per_second": 10,
        }
        server_events = {
            "loaded_qsl_set": qsl,
            "generated_query_count": 8,
            "generated_samples_per_query": 1,
            "effective_target_qps": 3,
            "result_completed_samples_per_sec": 2,
            "result_completed_tokens_per_second": 8,
        }

        def mllog(events: dict[str, Any]) -> bytes:
            return "".join(
                f":::MLLOG {json.dumps({'key': key, 'value': value}, separators=(',', ':'))}\n"
                for key, value in events.items()
            ).encode("utf-8")

        source_payloads = {
            "vidur.profile": ("vidur", "csv_profile", "profile.csv", profile),
            "mlperf.nvidia_llama2_base_config": (
                "mlperf",
                "source_file",
                "base.py",
                base_config,
            ),
            "mlperf.nvidia_llama2_offline_config": (
                "mlperf",
                "source_file",
                "offline.py",
                offline_config,
            ),
            "mlperf.nvidia_llama2_server_config": (
                "mlperf",
                "source_file",
                "server.py",
                server_config,
            ),
            "mlperf.nvidia_llama2_preprocessor": (
                "mlperf",
                "source_file",
                "preprocess.py",
                preprocessor,
            ),
            "mlperf.nvidia_llama2_generation_config": (
                "mlperf",
                "source_file",
                "generation.json",
                generation,
            ),
            "mlperf.offline_performance_detail": (
                "mlperf",
                "source_file",
                "offline-detail.txt",
                mllog(offline_events),
            ),
            "mlperf.server_performance_detail": (
                "mlperf",
                "source_file",
                "server-detail.txt",
                mllog(server_events),
            ),
        }
        declarations: dict[str, list[dict[str, Any]]] = {"vidur": [], "mlperf": []}
        bundle_records: list[dict[str, Any]] = []
        for artifact_id, (source_id, kind, filename, payload) in source_payloads.items():
            relative = Path("sources") / source_id / filename
            path = self.bundle / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            declaration: dict[str, Any] = {
                "id": artifact_id,
                "kind": kind,
                "path": filename,
                "bytes": len(payload),
                "sha256": _sha(payload),
            }
            if artifact_id == "vidur.profile":
                declaration.update(
                    {"columns": ["tp", "x", "time"], "rows": len(values)}
                )
            declarations[source_id].append(declaration)
            bundle_records.append(
                {
                    "artifact_id": artifact_id,
                    "bundle_path": relative.as_posix(),
                    "bytes": len(payload),
                    "kind": kind,
                    "sha256": _sha(payload),
                    "source_id": source_id,
                    "source_path": filename,
                }
            )

        anchor_id = "test-h100-llama2-calibration-anchor"
        anchor = {
            "schema": {
                "name": "hbfsim.validation.public-timing-anchor",
                "version": 1,
            },
            "anchor_id": anchor_id,
            "intended_use": {
                "vidur_role": "calibration_fit_input",
                "mlperf_role": "held_out_external_validation_only",
                "mlperf_fit_allowed": False,
                "bundle_alone_establishes_calibration": False,
                "bundle_alone_establishes_external_validation": False,
                "paper_result_eligible_by_itself": False,
            },
            "target": {},
            "sources": {
                "vidur": {"artifacts": declarations["vidur"]},
                "mlperf": {"artifacts": declarations["mlperf"]},
            },
            "claim_boundary": {},
        }
        anchor_bytes = _json_bytes(anchor)
        (self.bundle / "anchor-manifest.json").write_bytes(anchor_bytes)
        bundle_records.insert(
            0,
            {
                "artifact_id": "anchor.manifest",
                "bundle_path": "anchor-manifest.json",
                "bytes": len(anchor_bytes),
                "kind": "anchor_manifest",
                "sha256": _sha(anchor_bytes),
                "source_id": "anchor",
                "source_path": "protocol-anchor.json",
            },
        )
        anchor_eligibility = {
            "predictor_calibration_valid": False,
            "end_to_end_external_validation_valid": False,
            "frontier_timing_usable_for_performance": False,
            "paper_result_eligible": False,
            "timing_claim_state": "anchor_ready_not_calibrated",
        }
        bundle = {
            "schema": {
                "name": "hbfsim.validation.public-timing-anchor-bundle",
                "version": 1,
            },
            "result": "pass",
            "anchor": {
                "anchor_id": anchor_id,
                "manifest": {
                    "bundle_path": "anchor-manifest.json",
                    "bytes": len(anchor_bytes),
                    "sha256": _sha(anchor_bytes),
                },
            },
            "artifacts": bundle_records,
            "eligibility": anchor_eligibility,
            "profiles": [],
            "held_out_records": [],
            "sources": {},
        }
        bundle_bytes = _json_bytes(bundle)
        (self.bundle / "bundle-manifest.json").write_bytes(bundle_bytes)
        anchor_verification = {
            "schema": {
                "name": "hbfsim.validation.public-timing-anchor-verification",
                "version": 1,
            },
            "result": "pass",
            "anchor_id": anchor_id,
            "inputs": {
                "bundle_manifest": {"sha256": _sha(bundle_bytes)},
                "anchor_manifest": {"sha256": _sha(anchor_bytes)},
            },
            "checks": {"fixture_integrity": True},
            "eligibility": anchor_eligibility,
            "profiles": [],
            "held_out_records": [],
            "sources": {},
        }
        _write_json(self.anchor_verification, anchor_verification)

        length_pairs = [(1, 1), (2, 2), (3, 1), (4, 2)]
        length_bytes = _csv(
            ["qsl_idx", "input_tokens", "reference_output_tokens"],
            [[index, left, right] for index, (left, right) in enumerate(length_pairs)],
        )
        (self.workload / "lengths.csv").write_bytes(length_bytes)
        pairs_sha = _sha(
            b"".join(struct.pack("<II", left, right) for left, right in length_pairs)
        )
        dataset_payload = b"dummy"
        length_contract = {
            "filename": "lengths.csv",
            "columns": ["qsl_idx", "input_tokens", "reference_output_tokens"],
            "rows": len(length_pairs),
            "bytes": len(length_bytes),
            "sha256": _sha(length_bytes),
            "pairs_u32le_sha256": pairs_sha,
            "input_tokens": {"min": 1, "max": 4, "sum": 10},
            "reference_output_tokens": {"min": 1, "max": 2, "sum": 6},
        }
        dataset_contract = {
            "source_url": "https://example.invalid/openorca.pkl.gz",
            "metadata_url": "https://example.invalid/openorca.uri",
            "bytes": len(dataset_payload),
            "md5": _md5(dataset_payload),
            "sha256": _sha(dataset_payload),
            "pickle_load_allowed_only_after_digest_match": True,
            "required_columns": [
                "id",
                "system_prompt",
                "question",
                "output",
                "input",
                "tok_input",
                "tok_output",
                "origin",
                "tok_input_length",
                "tok_output_length",
            ],
        }
        qsl_sha = _sha(b"".join(struct.pack("<I", item) for item in qsl))
        self.protocol_value = {
            "schema": PROTOCOL_SCHEMA,
            "calibration_id": "test-h100-llama2-calibration",
            "anchor": {
                "anchor_id": anchor_id,
                "manifest_sha256": _sha(anchor_bytes),
                "bundle_schema": bundle["schema"],
                "verification_schema": anchor_verification["schema"],
                "required_timing_claim_state": "anchor_ready_not_calibrated",
            },
            "workload_evidence": {
                "schema": {
                    "name": "hbfsim.validation.mlperf-openorca-length-evidence",
                    "version": 1,
                },
                "dataset": dataset_contract,
                "lengths_csv": length_contract,
            },
            "fit_protocol": {
                "time_unit": "milliseconds",
                "target_statistic": "published_median",
                "duplicate_key_policy": "group_then_decimal_median",
                "deployment_refit": "all_unique_profile_keys_after_evaluation",
                "split": {
                    "algorithm": "coordinate_interleaved_v1",
                    "evaluation_coordinate_index_parity": 1,
                    "axis_boundaries_are_training": True,
                    "evaluation_requires_complete_adjacent_corner_set": True,
                    "all_rows_with_the_same_profile_key_share_one_partition": True,
                },
                "interpolator": {
                    "algorithm": "multilinear_adjacent_corners_v1",
                    "coordinate_arithmetic": "decimal",
                    "extrapolation_allowed": False,
                    "negative_prediction_allowed": False,
                },
                "acceptance": {
                    "maximum_wape_pct": "10",
                    "maximum_p90_ape_pct": "20",
                    "maximum_absolute_signed_bias_pct": "5",
                },
                "families": [
                    {
                        "id": "fixture",
                        "artifact_id": "vidur.profile",
                        "filters": {},
                        "group_values": [{"tp": "1"}],
                        "coordinates": [{"column": "x", "transform": "identity"}],
                        "targets": [{"id": "time", "column": "time"}],
                        "minimum_evaluation_fraction": "0.4",
                    }
                ],
            },
            "end_to_end_gate": {
                "artifact_roles": {
                    "base_config": "mlperf.nvidia_llama2_base_config",
                    "offline_config": "mlperf.nvidia_llama2_offline_config",
                    "server_config": "mlperf.nvidia_llama2_server_config",
                    "preprocessor": "mlperf.nvidia_llama2_preprocessor",
                    "generation_config": "mlperf.nvidia_llama2_generation_config",
                    "offline_detail": "mlperf.offline_performance_detail",
                    "server_detail": "mlperf.server_performance_detail",
                },
                "evidence_artifacts": [
                    "mlperf.nvidia_llama2_base_config",
                    "mlperf.nvidia_llama2_offline_config",
                    "mlperf.nvidia_llama2_server_config",
                    "mlperf.nvidia_llama2_preprocessor",
                    "mlperf.nvidia_llama2_generation_config",
                    "mlperf.offline_performance_detail",
                    "mlperf.server_performance_detail",
                ],
                "declared_mlperf_sut": {
                    "accelerators": 8,
                    "replicas": 4,
                    "gpus_per_replica": 2,
                    "tensor_parallelism": 1,
                    "pipeline_parallelism": 2,
                    "gpu_batch_size": 8,
                    "max_num_tokens": 4,
                    "max_input_len": 4,
                    "max_output_len": 4,
                    "weight_precision": "fp8",
                    "kv_cache_precision": "fp8",
                    "framework": "TensorRT 10.8, CUDA 12.8",
                    "batch_scheduler_policy": "max_util",
                    "context_chunking_policy": "first_come_first_served",
                },
                "vidur_fit_scope": {
                    "tensor_parallelism": 1,
                    "decode_batch_size_min": 1,
                    "decode_batch_size_max": 4,
                    "total_tokens_min": 1,
                    "total_tokens_max": 16,
                    "max_model_len": 16,
                    "precision": "fp16",
                    "cpu_runtime_overhead_profile_present": False,
                    "scheduler_trace_profile_present": False,
                },
                "detail_log_expectations": {
                    "loaded_qsl_set_count": 4,
                    "loaded_qsl_set_u32le_sha256": qsl_sha,
                    "offline": {
                        "generated_query_count": 1,
                        "generated_samples_per_query": 8,
                        "effective_target_qps": "3",
                        "result_samples_per_second": "2.5",
                        "result_tokens_per_second": "10",
                    },
                    "server": {
                        "generated_query_count": 8,
                        "generated_samples_per_query": 1,
                        "effective_target_qps": "3",
                        "result_completed_samples_per_sec": "2",
                        "result_completed_tokens_per_second": "8",
                    },
                },
                "held_out_targets": {
                    "offline": {
                        "samples_per_second": "2.5",
                        "tokens_per_second": "10",
                        "generated_output_tokens": 20,
                    },
                    "server": {
                        "samples_per_second": "2",
                        "tokens_per_second": "8",
                        "generated_output_tokens": 20,
                    },
                },
                "external_validation_requirements": {
                    "tensor_parallelism_match": True,
                    "total_token_budget_covered": True,
                    "maximum_sequence_length_covered": True,
                    "weight_precision_match": True,
                    "kv_cache_precision_match": True,
                    "runtime_kernel_match_or_validated_bridge": True,
                    "declared_batch_domain_covered": True,
                    "cpu_runtime_overhead_covered": True,
                    "scheduler_and_arrival_process_covered": True,
                    "per_request_generated_length_distribution_known": True,
                    "maximum_external_wape_pct": "15",
                },
                "out_of_domain_action": "do_not_emit_performance_prediction",
            },
            "claim_policy": {
                "operator_calibration_pass_state": "operator_calibrated_external_validation_not_established",
                "operator_calibration_fail_state": "candidate_calibration_failed",
                "frontier_timing_usable_for_performance": False,
                "paper_result_eligible": False,
                "llama31_w8a16_inheritance_allowed": False,
            },
            "verification": {
                "schema": {
                    "name": "hbfsim.validation.timing-calibration-verification",
                    "version": 1,
                },
                "required_checks": [
                    "protocol_integrity",
                    "anchor_chain_integrity",
                    "workload_evidence_integrity",
                    "artifact_census_integrity",
                    "source_role_separation",
                    "predictor_tables_reproduced",
                    "operator_holdout_reproduced",
                    "metric_thresholds_reproduced",
                    "mlperf_evidence_reproduced",
                    "out_of_domain_gate_reproduced",
                    "eligibility_reproduced",
                ],
            },
        }
        _write_json(self.protocol, self.protocol_value)
        workload_receipt = {
            "schema": self.protocol_value["workload_evidence"]["schema"],
            "result": "pass",
            "calibration_id": self.protocol_value["calibration_id"],
            "dataset": {
                "filename": "dummy.pkl.gz",
                "source_url": dataset_contract["source_url"],
                "bytes": dataset_contract["bytes"],
                "md5": dataset_contract["md5"],
                "sha256": dataset_contract["sha256"],
                "digest_checked_before_pickle_load": True,
            },
            "lengths": {
                "path": "lengths.csv",
                **{key: value for key, value in length_contract.items() if key != "filename"},
            },
            "scope": {
                "contains_prompts": False,
                "contains_token_ids": False,
                "contains_reference_text": False,
                "establishes_timing_calibration": False,
            },
        }
        _write_json(self.workload / "workload-evidence.json", workload_receipt)

    def save_protocol(self) -> None:
        _write_json(self.protocol, self.protocol_value)

    def fit(self, *, expected_returncode: int = 0) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                sys.executable,
                str(FIT),
                "--protocol",
                str(self.protocol),
                "--anchor-bundle-dir",
                str(self.bundle),
                "--anchor-verification",
                str(self.anchor_verification),
                "--workload-evidence-dir",
                str(self.workload),
                "--output-dir",
                str(self.candidate),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != expected_returncode:
            raise AssertionError(
                f"fit return code {result.returncode}, expected {expected_returncode}\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )
        return result

    def verify(self, *, expected_returncode: int = 0) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                sys.executable,
                str(VERIFY),
                "--protocol",
                str(self.protocol),
                "--anchor-bundle-dir",
                str(self.bundle),
                "--anchor-verification",
                str(self.anchor_verification),
                "--candidate-dir",
                str(self.candidate),
                "--output",
                str(self.verification),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != expected_returncode:
            raise AssertionError(
                f"verify return code {result.returncode}, expected {expected_returncode}\n"
                f"stdout={result.stdout}\nstderr={result.stderr}"
            )
        return result


PROTOCOL_SCHEMA = {
    "name": "hbfsim.validation.timing-calibration-protocol",
    "version": 1,
}


class CalibrationChainTests(unittest.TestCase):
    def fixture(self, *, nonlinear: bool = False) -> tuple[tempfile.TemporaryDirectory[str], _Fixture]:
        temporary = tempfile.TemporaryDirectory(prefix="hbfsim-calibration-test-")
        return temporary, _Fixture(Path(temporary.name), nonlinear=nonlinear)

    def test_candidate_and_independent_verification_pass_fail_closed(self) -> None:
        temporary, fixture = self.fixture()
        self.addCleanup(temporary.cleanup)
        fixture.fit()
        fixture.verify()
        candidate = json.loads((fixture.candidate / "candidate-manifest.json").read_text())
        receipt = json.loads(fixture.verification.read_text())
        self.assertEqual(candidate["result"], "pass")
        self.assertEqual(candidate["predictor"]["overall"]["models"], 1)
        self.assertEqual(candidate["predictor"]["overall"]["evaluation_keys"], 3)
        self.assertEqual(candidate["predictor"]["overall"]["wape_pct"], "0")
        self.assertEqual(
            candidate["end_to_end_external_validation"]["status"],
            "not_run_out_of_domain",
        )
        self.assertIsNone(candidate["end_to_end_external_validation"]["predictions"])
        self.assertIsNone(candidate["end_to_end_external_validation"]["errors"])
        self.assertTrue(candidate["eligibility"]["operator_predictor_calibration_valid"])
        self.assertFalse(candidate["eligibility"]["end_to_end_external_validation_valid"])
        self.assertFalse(candidate["eligibility"]["paper_result_eligible"])
        self.assertEqual(receipt["result"], "pass")
        self.assertTrue(all(receipt["checks"].values()))

    def test_verifier_has_no_code_dependency_on_fitter(self) -> None:
        tree = ast.parse(VERIFY.read_text(encoding="utf-8"))
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        self.assertNotIn("fit_h100_llama2_70b_calibration", imported_modules)
        self.assertNotIn("tools.fit_h100_llama2_70b_calibration", imported_modules)

    def test_official_protocol_digest_is_pinned_by_every_stage(self) -> None:
        protocol = ROOT / "validation/calibration/h100-llama2-70b-calibration-protocol.json"
        expected = _sha(protocol.read_bytes())
        for script in (EXTRACT, FIT, VERIFY):
            tree = ast.parse(script.read_text(encoding="utf-8"))
            assignments = {
                node.targets[0].id: node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            }
            self.assertIn("OFFICIAL_PROTOCOL_SHA256", assignments, script.name)
            self.assertEqual(
                ast.literal_eval(assignments["OFFICIAL_PROTOCOL_SHA256"]),
                expected,
                script.name,
            )

    def test_mlperf_artifact_is_rejected_as_fit_input(self) -> None:
        temporary, fixture = self.fixture()
        self.addCleanup(temporary.cleanup)
        fixture.protocol_value["fit_protocol"]["families"][0]["artifact_id"] = (
            "mlperf.nvidia_llama2_base_config"
        )
        fixture.save_protocol()
        result = fixture.fit(expected_returncode=1)
        self.assertIn("not bound to a Vidur artifact", result.stderr)
        self.assertFalse(fixture.candidate.exists())

    def test_weakened_extrapolation_policy_is_rejected(self) -> None:
        temporary, fixture = self.fixture()
        self.addCleanup(temporary.cleanup)
        fixture.protocol_value["fit_protocol"]["interpolator"]["extrapolation_allowed"] = True
        fixture.save_protocol()
        result = fixture.fit(expected_returncode=1)
        self.assertIn("weakened interpolator", result.stderr)

    def test_recomputed_evaluation_rejects_digest_rewritten_tamper(self) -> None:
        temporary, fixture = self.fixture()
        self.addCleanup(temporary.cleanup)
        fixture.fit()
        evaluation = fixture.candidate / "operator-evaluation.csv"
        evaluation.write_bytes(evaluation.read_bytes() + b"\n")
        manifest_path = fixture.candidate / "candidate-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for artifact in manifest["artifacts"]:
            if artifact["artifact_id"] == "predictor.operator_evaluation":
                artifact["bytes"] = evaluation.stat().st_size
                artifact["sha256"] = _sha(evaluation.read_bytes())
        _write_json(manifest_path, manifest)
        result = fixture.verify(expected_returncode=1)
        self.assertIn("operator evaluation bytes were not reproduced", result.stderr)

    def test_rejects_manifest_only_claim_promotion(self) -> None:
        temporary, fixture = self.fixture()
        self.addCleanup(temporary.cleanup)
        fixture.fit()
        manifest_path = fixture.candidate / "candidate-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["eligibility"]["paper_result_eligible"] = True
        _write_json(manifest_path, manifest)
        result = fixture.verify(expected_returncode=1)
        self.assertIn("eligibility was not reproduced", result.stderr)

    def test_rejects_unattached_candidate_file(self) -> None:
        temporary, fixture = self.fixture()
        self.addCleanup(temporary.cleanup)
        fixture.fit()
        (fixture.candidate / "unattached.txt").write_text("not declared\n")
        result = fixture.verify(expected_returncode=1)
        self.assertIn("unexpected entries", result.stderr)

    def test_failed_operator_threshold_is_preserved_and_verified(self) -> None:
        temporary, fixture = self.fixture(nonlinear=True)
        self.addCleanup(temporary.cleanup)
        fixture.protocol_value["fit_protocol"]["acceptance"] = {
            "maximum_wape_pct": "0.01",
            "maximum_p90_ape_pct": "0.01",
            "maximum_absolute_signed_bias_pct": "0.01",
        }
        fixture.save_protocol()
        fixture.fit(expected_returncode=2)
        fixture.verify()
        candidate = json.loads((fixture.candidate / "candidate-manifest.json").read_text())
        self.assertEqual(candidate["result"], "fail")
        self.assertFalse(candidate["eligibility"]["operator_predictor_calibration_valid"])
        self.assertEqual(
            candidate["eligibility"]["timing_claim_state"], "candidate_calibration_failed"
        )

    def test_extractor_rejects_digest_mismatch_before_pandas(self) -> None:
        temporary, fixture = self.fixture()
        self.addCleanup(temporary.cleanup)
        wrong_dataset = fixture.root / "wrong.pkl.gz"
        wrong_dataset.write_bytes(b"abcde")
        result = subprocess.run(
            [
                sys.executable,
                str(EXTRACT),
                "--protocol",
                str(fixture.protocol),
                "--dataset",
                str(wrong_dataset),
                "--output-dir",
                str(fixture.root / "extracted"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("MD5 mismatch; refusing to unpickle", result.stderr)
        self.assertNotIn("pandas is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
