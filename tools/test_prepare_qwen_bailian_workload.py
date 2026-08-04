#!/usr/bin/env python3
"""Regression tests for the canonical Qwen-Bailian workload suite."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from prepare_qwen_bailian_workload import (  # noqa: E402
    ADAPTER_SCHEMA_NAME,
    ADAPTER_SCHEMA_VERSION,
    REQUIRED_RUNTIME_BEHAVIORS,
    REQUIRED_WINDOWS,
    SELECTOR_ALGORITHMS,
    SUITE_CONFIG_SCHEMA_NAME,
    SUITE_CONFIG_SCHEMA_VERSION,
    WorkloadError,
    _load_suite_config,
    prepare_suite,
)
from verify_qwen_bailian_workload import (  # noqa: E402
    VerificationError,
    verify_suite,
)


def record(
    chat_id: int,
    parent_chat_id: int,
    timestamp: float,
    input_length: int,
    output_length: int,
    *,
    turn: int = 1,
    request_type: str = "thinking",
    hash_base: int = 0,
) -> dict:
    count = (input_length + 15) // 16
    return {
        "chat_id": chat_id,
        "parent_chat_id": parent_chat_id,
        "timestamp": timestamp,
        "input_length": input_length,
        "output_length": output_length,
        "type": request_type,
        "turn": turn,
        "hash_ids": list(range(hash_base, hash_base + count)),
    }


def expected_window(rows: list[dict], start: int, count: int) -> dict:
    window = rows[start : start + count]
    start_time = window[0]["timestamp"]
    end_time = window[-1]["timestamp"]
    return {
        "start_index": start,
        "end_index_exclusive": start + count,
        "source_start_timestamp_s": str(start_time),
        "source_end_timestamp_s": str(end_time),
        "duration_s": str(end_time - start_time),
        "prefill_tokens": sum(row["input_length"] for row in window),
        "decode_tokens": sum(row["output_length"] for row in window),
        "total_tokens": sum(
            row["input_length"] + row["output_length"] for row in window
        ),
        "max_prefill_tokens": max(row["input_length"] for row in window),
        "max_decode_tokens": max(row["output_length"] for row in window),
        "max_total_tokens": max(
            row["input_length"] + row["output_length"] for row in window
        ),
    }


class PrepareQwenBailianWorkloadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.jsonl"
        self.config = self.root / "suite-config.json"
        self.output = self.root / "suite"
        self.rows = [
            record(10, -1, 0, 16, 3, hash_base=0),
            record(11, 10, 2, 32, 2, turn=2, hash_base=0),
            record(12, 11, 4, 48, 1, turn=3, hash_base=0),
            record(13, -1, 10, 64, 20, hash_base=10),
            record(14, -1, 10, 64, 20, hash_base=20),
            record(15, -1, 10, 64, 20, hash_base=30),
            record(16, -1, 20, 80, 5, hash_base=40),
            record(17, -1, 22, 160, 4, hash_base=50),
            record(18, -1, 24, 16, 90, hash_base=60),
        ]
        self.write_records(self.rows)
        self.write_config(self.rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_records(self, rows: list[dict]) -> None:
        self.source.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    def config_payload(self, rows: list[dict]) -> dict:
        source_bytes = self.source.read_bytes()
        return {
            "schema": {
                "name": SUITE_CONFIG_SCHEMA_NAME,
                "version": SUITE_CONFIG_SCHEMA_VERSION,
            },
            "suite_id": "synthetic-suite-v1",
            "intended_use": {
                "role": "unit_test",
                "paper_result_eligible_by_itself": False,
                "claims_boundary": "unit test only",
            },
            "source": {
                "repository": "https://example.invalid/source",
                "revision": "0" * 40,
                "license": "test-only",
                "artifact": "source.jsonl",
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "bytes": len(source_bytes),
                "records": len(rows),
                "hash_block_tokens": 16,
                "statistics": {
                    "duration_s": str(
                        rows[-1]["timestamp"] - rows[0]["timestamp"]
                    ),
                    "prefill_tokens": sum(row["input_length"] for row in rows),
                    "decode_tokens": sum(row["output_length"] for row in rows),
                    "max_prefill_tokens": max(
                        row["input_length"] for row in rows
                    ),
                    "max_decode_tokens": max(
                        row["output_length"] for row in rows
                    ),
                },
            },
            "frontier_workload": {
                "model": {
                    "identity": "example/tiny-test-model",
                    "frontier_name": "tiny_test_model",
                },
                "primary_precision_profile": "w8a16-kv-bf16",
                "sensitivity_precision_profiles": ["bf16-kv-bf16"],
                "required_runtime_behaviors": list(REQUIRED_RUNTIME_BEHAVIORS),
            },
            "window_policy": {
                "request_count": 3,
                "required_windows": list(REQUIRED_WINDOWS),
                "selectors": {
                    "steady": {
                        "algorithm": SELECTOR_ALGORITHMS["steady"],
                        "expected": expected_window(rows, 2, 3),
                    },
                    "burst": {
                        "algorithm": SELECTOR_ALGORITHMS["burst"],
                        "expected": expected_window(rows, 3, 3),
                    },
                    "long_context_decode_tail": {
                        "algorithm": SELECTOR_ALGORITHMS[
                            "long_context_decode_tail"
                        ],
                        "expected": expected_window(rows, 6, 3),
                    },
                },
            },
        }

    def write_config(self, rows: list[dict]) -> None:
        self.config.write_text(
            json.dumps(self.config_payload(rows), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def prepare(self, output: Path | None = None) -> dict:
        return prepare_suite(
            source=self.source,
            suite_config=self.config,
            output_dir=output or self.output,
        )

    def verify(self, output: Path | None = None) -> dict:
        return verify_suite(
            source=self.source,
            suite_config=self.config,
            suite_dir=self.output,
            output=output or (self.root / "suite-verification.json"),
        )

    def test_selects_all_three_windows_and_preserves_request_semantics(self) -> None:
        result = self.prepare()
        self.assertEqual(list(result["window_policy"]["required_windows"]), list(REQUIRED_WINDOWS))
        self.assertEqual(
            {
                window_id: result["windows"][window_id]["selection"]["start_index"]
                for window_id in REQUIRED_WINDOWS
            },
            {"steady": 2, "burst": 3, "long_context_decode_tail": 6},
        )

        with (self.output / "steady.frontier.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["source_chat_id"] for row in rows], ["12", "13", "14"])
        self.assertEqual(rows[0]["session_id"], "10")
        self.assertEqual(rows[0]["arrived_at"], "0")
        self.assertEqual(rows[1]["arrived_at"], "6")
        self.assertEqual(rows[0]["block_hash_ids"], "0|1|2")
        self.assertEqual(rows[0]["turn"], "3")

        manifest = json.loads(
            (self.output / "steady.adapter-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["schema"],
            {"name": ADAPTER_SCHEMA_NAME, "version": ADAPTER_SCHEMA_VERSION},
        )
        self.assertFalse(manifest["adapter_semantics"]["output_is_hbfsim_memory_trace"])
        self.assertFalse(manifest["eligibility"]["paper_result_eligible"])
        self.assertEqual(manifest["source"]["artifact"], "source.jsonl")
        self.assertNotIn(str(self.root), json.dumps(manifest))
        self.assertEqual(
            manifest["output"]["sha256"],
            hashlib.sha256(
                (self.output / "steady.frontier.csv").read_bytes()
            ).hexdigest(),
        )

    def test_output_is_byte_deterministic_and_uses_relative_paths(self) -> None:
        first = self.prepare(self.root / "first")
        second = self.prepare(self.root / "second")
        self.assertEqual(first, second)
        first_files = sorted(
            path.relative_to(self.root / "first")
            for path in (self.root / "first").iterdir()
        )
        second_files = sorted(
            path.relative_to(self.root / "second")
            for path in (self.root / "second").iterdir()
        )
        self.assertEqual(first_files, second_files)
        for relative in first_files:
            self.assertEqual(
                (self.root / "first" / relative).read_bytes(),
                (self.root / "second" / relative).read_bytes(),
            )
        self.assertNotIn(
            str(self.root),
            (self.root / "first" / "suite-manifest.json").read_text(),
        )

    def test_independent_verifier_reconstructs_every_window(self) -> None:
        self.prepare()
        result = self.verify()
        self.assertEqual(result["result"], "pass")
        self.assertTrue(
            result["eligibility"]["production_request_suite_verified"]
        )
        self.assertFalse(result["eligibility"]["paper_result_eligible"])
        self.assertEqual(set(result["windows"]), set(REQUIRED_WINDOWS))

    def test_independent_verifier_rejects_csv_manifest_and_file_set_drift(self) -> None:
        mutations = (
            (
                "csv",
                lambda: (self.output / "burst.frontier.csv").write_bytes(b"bad\n"),
                "request_csv mismatch",
            ),
            (
                "manifest",
                lambda: (
                    self.output / "steady.adapter-manifest.json"
                ).write_text("{}\n", encoding="utf-8"),
                "request_manifest mismatch",
            ),
            (
                "extra",
                lambda: (self.output / "unattached.json").write_text(
                    "{}\n", encoding="utf-8"
                ),
                "file set mismatch",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                output = self.root / f"suite-{name}"
                prepare_suite(
                    source=self.source,
                    suite_config=self.config,
                    output_dir=output,
                )
                original = self.output
                self.output = output
                mutate()
                with self.assertRaisesRegex(VerificationError, message):
                    self.verify(self.root / f"{name}-verification.json")
                self.output = original

    def test_rejects_source_digest_drift_without_publishing(self) -> None:
        self.rows[0]["output_length"] += 1
        self.write_records(self.rows)
        with self.assertRaisesRegex(WorkloadError, "source sha256 mismatch"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_selector_drift_without_publishing(self) -> None:
        config = self.config_payload(self.rows)
        config["window_policy"]["selectors"]["steady"]["expected"][
            "start_index"
        ] = 1
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(WorkloadError, "steady selector drift"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_invalid_ancestry_hashes_and_timestamp(self) -> None:
        mutations = (
            ("parent", lambda rows: rows[1].update(parent_chat_id=999), "earlier"),
            ("turn", lambda rows: rows[1].update(turn=3), "parent turn"),
            ("hash", lambda rows: rows[0].update(hash_ids=[]), "requires 1"),
            ("time", lambda rows: rows[1].update(timestamp=-1), "finite decimal"),
        )
        for name, mutate, message in mutations:
            with self.subTest(name=name):
                rows = json.loads(json.dumps(self.rows))
                mutate(rows)
                self.write_records(rows)
                self.write_config(rows)
                with self.assertRaisesRegex(WorkloadError, message):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.write_records(self.rows)
                self.write_config(self.rows)

    def test_rejects_duplicate_json_keys(self) -> None:
        self.source.write_text(
            '{"chat_id":1,"chat_id":2,"parent_chat_id":-1,'
            '"timestamp":0,"input_length":16,"output_length":1,'
            '"type":"thinking","turn":1,"hash_ids":[0]}\n',
            encoding="utf-8",
        )
        config = self.config_payload(self.rows)
        source_bytes = self.source.read_bytes()
        config["source"]["sha256"] = hashlib.sha256(source_bytes).hexdigest()
        config["source"]["bytes"] = len(source_bytes)
        config["source"]["records"] = 1
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(WorkloadError, "duplicate key 'chat_id'"):
            self.prepare()

    def test_refuses_to_replace_an_existing_output_directory(self) -> None:
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_text("user data", encoding="utf-8")
        with self.assertRaisesRegex(WorkloadError, "already exists"):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data")

    def test_canonical_config_pins_70b_source_and_three_256_request_windows(self) -> None:
        canonical = (
            ROOT
            / "configs/workloads/frontier/"
            "qwen-thinking-llama31-70b-production.json"
        )
        config, _ = _load_suite_config(canonical)
        self.assertEqual(
            config["frontier_workload"]["model"]["identity"],
            "meta-llama/Llama-3.1-70B",
        )
        self.assertEqual(
            config["frontier_workload"]["model"]["frontier_name"],
            "llama31_70b",
        )
        self.assertEqual(
            config["frontier_workload"]["primary_precision_profile"],
            "w8a16-kv-bf16",
        )
        self.assertEqual(config["window_policy"]["request_count"], 256)
        self.assertEqual(
            {
                window_id: config["window_policy"]["selectors"][window_id][
                    "expected"
                ]["start_index"]
                for window_id in REQUIRED_WINDOWS
            },
            {
                "steady": 3074,
                "burst": 810,
                "long_context_decode_tail": 4332,
            },
        )
        self.assertEqual(
            config["source"]["sha256"],
            "41ac36d9d1b54d084eeb6b05e9d142ba9d02f8bf79cfc9fff418d8ab7cdee906",
        )


if __name__ == "__main__":
    unittest.main()
