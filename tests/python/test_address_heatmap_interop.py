#!/usr/bin/env python3
"""Exercise the C++ v1 writer through the strict Python renderer."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from reports import address_heatmap as heatmap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emitter", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "cpp-address-heatmap.json"
        rendered = root / "cpp-address-heatmap.html"
        emitted = subprocess.run(
            [str(args.emitter.resolve()), "--emit-json", str(artifact)],
            capture_output=True,
            text=True,
            check=False,
        )
        if emitted.returncode != 0:
            raise RuntimeError(
                "C++ heatmap fixture failed:\n" + emitted.stdout + emitted.stderr)

        parsed = heatmap.load_heatmap(artifact)
        workload = parsed.domains[0]
        if workload.size_bytes != heatmap.ADDRESS_SPACE_SIZE:
            raise AssertionError("C++ fixture lost the full uint64 address extent")
        if workload.bins[-1].end != heatmap.ADDRESS_SPACE_SIZE:
            raise AssertionError("C++ fixture lost the exclusive 2^64 bin end")
        if workload.bins[-1].metrics["read_bytes"] != 1:
            raise AssertionError("C++ fixture lost traffic at UINT64_MAX")

        renderer = (
            _REPOSITORY_ROOT / "reports/address_heatmap.py"
        )
        completed = subprocess.run(
            [sys.executable, str(renderer), "--input", str(artifact),
             "--output", str(rendered)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Python heatmap renderer rejected the C++ fixture:\n" +
                completed.stdout + completed.stderr)
        html = rendered.read_text(encoding="utf-8")
        if html.count('class="domain-panel"') != len(heatmap.DOMAIN_ORDER):
            raise AssertionError(
                "interoperability render did not contain every domain")
        if "0x10000000000000000" not in html:
            raise AssertionError("interoperability render hid the 2^64 endpoint")
        lowered = html.lower()
        if "<script" in lowered or "<link" in lowered or "https://" in lowered:
            raise AssertionError("interoperability render gained an online dependency")

    print("C++ heatmap JSON passes strict Python validation and offline rendering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
