#!/usr/bin/env python3
"""Generate a certificate-bound capacity-overflow paper artifact."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.certificate import (  # noqa: E402
    CertificateError,
    attach_certificate_to_overflow_artifact_file,
    file_digest,
    verify_certificate,
    write_json_atomic,
)


def _run(command: list[str], *, cwd: Path, timeout: int) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr.strip()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--scenario-compare", type=Path, required=True)
    parser.add_argument(
        "--validation-certificate",
        type=Path,
        required=True,
        help="publishable foundational certificate required for paper results",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--analyzer",
        type=Path,
        default=ROOT / "tools/analyze_overflow_offload_experiment.py",
    )
    parser.add_argument("--hbm-capacity-bytes", type=int)
    parser.add_argument("--total-write-bytes", type=int)
    parser.add_argument("--read-buffer-bytes", type=int)
    parser.add_argument("--layer-bytes", type=int)
    parser.add_argument("--batch-pages", type=int)
    parser.add_argument("--sample-batches", type=int)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    repository = args.repository.resolve()
    experiment = args.experiment.resolve()
    scenario_compare = args.scenario_compare.resolve()
    analyzer = args.analyzer.resolve()
    output_dir = args.output_dir.resolve()
    try:
        certificate = verify_certificate(
            args.validation_certificate.resolve(),
            repository=repository,
            scenario_compare=scenario_compare,
            overflow_offload_experiment=experiment,
        )
        for path, label in (
            (experiment, "overflow experiment"),
            (scenario_compare, "scenario_compare"),
            (analyzer, "overflow analyzer"),
        ):
            if not path.is_file():
                raise RuntimeError(f"{label} not found: {path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".capacity-overflow-paper-",
            dir=output_dir,
        ) as directory:
            staging = Path(directory)
            summary = staging / "summary.json"
            csv = staging / "summary.csv"
            report = staging / "report.md"
            command = [
                str(experiment),
                "--json",
                str(summary),
                "--csv",
                str(csv),
            ]
            for flag, value in (
                ("--hbm-capacity-bytes", args.hbm_capacity_bytes),
                ("--total-write-bytes", args.total_write_bytes),
                ("--read-buffer-bytes", args.read_buffer_bytes),
                ("--layer-bytes", args.layer_bytes),
                ("--batch-pages", args.batch_pages),
                ("--sample-batches", args.sample_batches),
            ):
                if value is not None:
                    if value <= 0:
                        raise RuntimeError(f"{flag} must be positive")
                    command.extend([flag, str(value)])
            _run(command, cwd=repository, timeout=args.timeout)
            _run(
                [
                    sys.executable,
                    "-B",
                    str(analyzer),
                    "--input",
                    str(summary),
                    "--markdown",
                    str(report),
                    "--experiment",
                    str(experiment),
                ],
                cwd=repository,
                timeout=args.timeout,
            )
            attached = attach_certificate_to_overflow_artifact_file(
                summary,
                certificate,
            )
            _run(
                [
                    sys.executable,
                    "-B",
                    str(analyzer),
                    "--input",
                    str(summary),
                    "--markdown",
                    str(report),
                    "--experiment",
                    str(experiment),
                    "--validation-certificate",
                    str(args.validation_certificate.resolve()),
                    "--repository",
                    str(repository),
                    "--scenario-compare",
                    str(scenario_compare),
                ],
                cwd=repository,
                timeout=args.timeout,
            )
            manifest = {
                "schema": {
                    "name": "hbfsim.capacity-overflow-paper-run",
                    "version": 1,
                },
                "validation": attached["validation"],
                "source_commit": certificate.source_commit,
                "artifacts": {
                    "summary": file_digest(summary),
                    "csv": file_digest(csv),
                    "report": file_digest(report),
                },
            }
            manifest_path = staging / "manifest.json"
            write_json_atomic(manifest_path, manifest)
            for name in (
                "summary.json",
                "summary.csv",
                "report.md",
                "manifest.json",
            ):
                os.replace(staging / name, output_dir / name)
    except (
        CertificateError,
        OSError,
        RuntimeError,
        subprocess.TimeoutExpired,
        ValueError,
    ) as error:
        print(f"capacity-overflow paper run failed: {error}", file=sys.stderr)
        return 1

    print(f"PASS certificate-bound capacity-overflow run: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
