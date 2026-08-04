#!/usr/bin/env python3
"""End-to-end contract for schema-v16 behavior-only HBM/HBF tiering."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path


DEMAND = "HBM+HBF-demand-fill"
ADAPTIVE = "HBM+HBF-behavioral-placement"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-compare", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
            prefix="hbfsim-behavioral-summary-") as directory:
        root = Path(directory)
        trace = root / "behavior.trace"
        summary = root / "summary.json"
        table = root / "summary.csv"
        # Two hot pages, one cold scan page, then a phase-shift page. Semantic
        # annotations are present solely to exercise the strict trace parser;
        # this controller is contractually forbidden from consuming them.
        trace.write_text(
            "\n".join((
                "0x0000 R 4096 model_weights label=hot0 phase=0 layer=0 at=0",
                "0x1000 R 4096 scratch label=hot1 phase=0 layer=0 at=1",
                "0x2000 R 4096 metadata label=cold phase=0 layer=0 at=2",
                "0x0000 R 4096 scratch label=hot0b phase=1 layer=1 at=3",
                "0x1000 W 4096 generated_context label=hot1w phase=1 layer=1 at=4",
                "0x0000 R 4096 metadata label=hot0c phase=2 layer=2 at=5",
                "0x3000 R 4096 model_weights label=shift phase=2 layer=2 at=6",
                "0x3000 R 4096 scratch label=shift2 phase=3 layer=3 at=7",
            )) + "\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            (
                str(args.scenario_compare),
                "--config", str(args.config),
                "--trace", str(trace),
                "--scenarios", f"{DEMAND},{ADAPTIVE}",
                "--behavioral-hbm-bytes", "8192",
                "--behavioral-history-pages", "8",
                "--behavioral-promotion-threshold", "2",
                "--max-outstanding-requests", "4",
                "--summary-json", str(summary),
                "--summary-csv", str(table),
                "--trace-mode", "off",
            ),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise AssertionError(
                "scenario_compare failed:\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}")

        document = json.loads(summary.read_text(encoding="utf-8"))
        require(
            document.get("schema") == {
                "name": "hbfsim.scenario_compare.summary",
                "version": 16,
            },
            "behavioral summary schema drifted",
        )
        require(document.get("sanity") == "PASS", "behavioral run failed sanity")
        config = document["config"]
        require(
            config["behavioral_hbm_bytes"] == 8192
            and config["behavioral_history_pages"] == 8
            and config["behavioral_promotion_threshold"] == 2,
            "behavioral config provenance is incomplete",
        )
        scenarios = {entry["name"]: entry for entry in document["scenarios"]}
        require(set(scenarios) == {DEMAND, ADAPTIVE}, "scenario set drifted")

        with table.open(encoding="utf-8", newline="") as handle:
            rows = {row["scenario"]: row for row in csv.DictReader(handle)}
        require(set(rows) == set(scenarios), "CSV scenario set drifted")
        for name, scenario in scenarios.items():
            stats = scenario.get("behavioral_tiering")
            require(isinstance(stats, dict), f"{name}: placement stats missing")
            require(
                stats["semantic_inputs_consumed"] is False,
                f"{name}: semantic contract violated",
            )
            require(
                stats["page_observations"]
                == stats["hbm_hits"] + stats["hbf_bypasses"]
                + stats["promotions"],
                f"{name}: decision census does not conserve",
            )
            require(
                stats["promotions"]
                == stats["promotions_with_backing_fill"]
                + stats["promotions_without_backing_fill"],
                f"{name}: promotion classes do not conserve",
            )
            require(
                stats["backing_fill_bytes"]
                == stats["backing_fill_pages"] * 4096
                and stats["hbm_install_bytes"]
                == stats["hbm_install_pages"] * 4096
                and stats["dirty_writeback_bytes"]
                == stats["dirty_writeback_pages"] * 4096,
                f"{name}: migration bytes do not conserve",
            )
            require(
                stats["final_dirty_pages"] == 0
                and stats["peak_resident_pages"] <= stats["hbm_tier_pages"],
                f"{name}: final/capacity state is invalid",
            )
            require(
                scenario["hybrid_path"]["base_die_link_read_bytes"]
                == stats["backing_fill_bytes"]
                and scenario["hybrid_path"]["base_die_link_write_bytes"]
                == stats["dirty_writeback_bytes"],
                f"{name}: D2D traffic does not match migrations",
            )
            row = rows[name]
            require(
                row["summary_schema_version"] == "16"
                and row["behavioral_policy"] == stats["policy"]
                and int(row["behavioral_page_observations"])
                == stats["page_observations"]
                and int(row["behavioral_decision_fingerprint"])
                == stats["decision_fingerprint"],
                f"{name}: JSON/CSV behavioral fields disagree",
            )

        demand = scenarios[DEMAND]["behavioral_tiering"]
        adaptive = scenarios[ADAPTIVE]["behavioral_tiering"]
        require(
            demand["policy"] == "always-admit"
            and adaptive["policy"] == "reuse-filtered",
            "scenario identities do not bind admission policies",
        )
        require(
            adaptive["hbf_bypasses"] > 0
            and adaptive["promotions"] > 0
            and adaptive["hbm_hits"] > 0,
            "adaptive trace did not exercise bypass, promotion, and hit",
        )
        require(
            demand["hbf_bypasses"] == 0,
            "demand-fill baseline unexpectedly bypassed HBM admission",
        )

    print("behavioral tiering schema-v16 summary regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
