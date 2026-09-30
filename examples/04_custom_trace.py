#!/usr/bin/env python3
"""Example 4 — bring your own workload as a trace file.

The reference runner replays text traces, one memory operation per line:

    <address> <R|W> [bytes] [kind] [layer=N]

`kind` labels what the bytes are (model_weights, shared_context,
generated_context, scratch, metadata); placement policies such as
direct-read use it to keep mutable data out of flash. `layer=N` numbers
layer executions in order (step x layers + layer), so it never decreases;
layer-streaming policies prefetch the next one while this one runs.

This script writes a toy LLM decode trace — per layer: stream the layer's
weights, read the KV cache, append one token of KV — and replays it under
four placements.

    python3 examples/04_custom_trace.py
    python3 examples/04_custom_trace.py --layers 8 --context-tokens 2048

Any trace in this format works: export one from your own tooling and pass
it to `python3 -m hbfsim run --trace FILE`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a clone without installing

import hbfsim  # noqa: E402
from hbfsim.runner import prepare_output_directory  # noqa: E402


LINE = 4096                 # bytes per trace operation (the runner's line size)
WEIGHTS_BASE = 0x1000_0000  # addresses are arbitrary but must not overlap
KV_BASE = 0x4000_0000


def write_decode_trace(path: Path, *, layers: int, weight_bytes_per_layer: int,
                       context_tokens: int, kv_bytes_per_token: int, steps: int) -> int:
    lines = ["# toy decode trace written by examples/04_custom_trace.py"]
    kv_stride = context_tokens * kv_bytes_per_token + steps * kv_bytes_per_token
    for step in range(steps):
        for layer in range(layers):
            execution = step * layers + layer
            weights = WEIGHTS_BASE + layer * weight_bytes_per_layer
            for offset in range(0, weight_bytes_per_layer, LINE):
                lines.append(f"{weights + offset:#x} R {LINE} model_weights layer={execution}")
            kv = KV_BASE + layer * kv_stride
            history = (context_tokens + step) * kv_bytes_per_token
            for offset in range(0, history, LINE):
                lines.append(f"{kv + offset:#x} R {LINE} generated_context layer={execution}")
            lines.append(f"{kv + history:#x} W {kv_bytes_per_token} generated_context "
                         f"layer={execution}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines) - 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2, help="decode steps")
    args = parser.parse_args()
    out = prepare_output_directory(args.out, "example-trace")

    trace = out / "decode.trace"
    weight_bytes = 1 << 20  # 1 MiB of weights per layer
    operations = write_decode_trace(
        trace, layers=args.layers, weight_bytes_per_layer=weight_bytes,
        context_tokens=args.context_tokens, kv_bytes_per_token=LINE, steps=args.steps,
    )
    # Layer streaming stages one layer's weights and KV in HBM at a time.
    layer_bytes = weight_bytes + (args.context_tokens + args.steps) * LINE
    print(f"wrote {operations:,} operations to {trace}\n")

    result = hbfsim.run(
        trace=trace,
        scenarios="all-hbm,all-hbf,direct-read,hbf-streaming",
        options={"layer-buffer-bytes": -(-layer_bytes // (1 << 20)) * (1 << 20)},
        out_dir=out / "replay",
    )
    print(result.table())
    for scenario in result.scenarios:
        for warning in scenario.warnings:
            print(f"warning [{scenario.name}]: {warning}")

    direct = result.scenario("direct-read")
    print(f"\ndirect-read served {direct['hbf_accesses']:,} model_weights reads from flash and kept "
          f"{direct['hbm_accesses']:,} generated_context (KV) accesses in HBM,\n"
          "because the trace labels KV as mutable. Relabel it and the placement changes.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
