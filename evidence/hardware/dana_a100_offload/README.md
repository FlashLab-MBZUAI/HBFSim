# DANA A100 offload measurement

> Status: Experimental
> Last reviewed: 2026-09-29

This directory is a self-contained hardware anchor for HBFSim's host-DRAM and
local-SSD offload paths. It is deliberately tied to the DANA production
environment used for the anchor:

- Python 3.8;
- PyTorch `2.4.1+cu124`;
- one Slurm-allocated A100 on `gpu-51`;
- CUDA-pinned host memory;
- compute-node-local NVMe reached through a job-private child of `/tmp`.

The benchmark imports no package other than the cluster-provided `torch`.
Python's standard library supplies aligned buffers, direct I/O, concurrency,
hashing, statistics, and JSON serialization.

## Measured paths

Every default run preserves curves for 4 KiB, 64 KiB, 1 MiB, and 16 MiB.
The normal run transfers more bytes at 16 MiB because that is the main HBFSim
offload request size. The SSD paths run at both one and eight lanes. A lane
issues one synchronous request at a time, so its aggregate outstanding depth is
at most the lane count; the JSON calls this field
`max_outstanding_operations` rather than pretending it is an io_uring queue
depth.

| JSON `path` | Direction | Timed operation |
|---|---|---|
| `gpu_pinned_host` | `offload_d2h` | A100 to page-aligned pinned host tensor |
| `gpu_pinned_host` | `restore_h2d` | Pinned host tensor to A100 |
| `ssd_direct` | `offload_pwrite` | Aligned host buffer to local SSD |
| `ssd_direct` | `restore_pread` | Local SSD to aligned host buffer |
| `gpu_pinned_host_ssd_direct` | `offload_gpu_to_ssd` | D2H and stream sync, then direct write |
| `gpu_pinned_host_ssd_direct` | `restore_ssd_to_gpu` | Direct read, then H2D and stream sync |

SSD files are preallocated outside the timed region. The measured I/O is
`O_DIRECT` with positional `preadv`/`pwritev` and 4096-byte-aligned
buffers. There is no buffered-I/O fallback. A completed write means the
synchronous direct-write syscall returned successfully; the benchmark does not
add a power-loss durability flush that is absent from HBFSim's block-transfer
model.

Pure-SSD and end-to-end measurements use separate files. Within one end-to-end
lane the GPU copy and SSD request are serial, while independent lanes run
concurrently. Component latency distributions retain the PCIe and SSD portions
for fitting them separately.

## Production run

Submit from the HBFSim repository root:

```bash
sbatch evidence/hardware/dana_a100_offload/run_dana_a100.sbatch
```

The production script requests `cscc-gpu-p` with `cscc-gpu-qos`, pins the
node to `gpu-51`, and asks for one GPU, eight CPUs, 64 GiB, and 30 minutes. It
does not request an exclusive node and does not set or modify
`CUDA_VISIBLE_DEVICES`. It creates
`/tmp/hbfsim-offload-$SLURM_JOB_ID`, passes that exact directory to the
benchmark, and removes it from an exit trap.

The default result is:

```text
evidence/hardware/dana_a100_offload/results/dana-a100-offload-JOBID.json
```

Pass an explicit result path as the script's sole argument if desired. Existing
result files are not overwritten:

```bash
sbatch evidence/hardware/dana_a100_offload/run_dana_a100.sbatch /absolute/persistent/result.json
```

For a short allocation smoke test, use:

```bash
sbatch evidence/hardware/dana_a100_offload/run_dana_a100.sbatch --quick
```

`--quick` still covers every block size, both directions, both pure/e2e SSD
paths, and lane counts 1 and 8. It reduces transferred bytes and latency
samples; it is a functional check, not the calibration anchor.

The Slurm entry point rejects a non-A100 GPU, a Python or torch version other
than the declared versions, a work directory outside a job-specific `/tmp`
child, known remote/memory filesystems, or a block-device topology that cannot
be proven NVMe-backed. These are fail-closed checks against accidentally
publishing a shared-filesystem or buffered-I/O number.

## Result contract

One normal result is a single JSON document with:

- complete invocation and timing semantics;
- host, Python, torch/CUDA, GPU, CPU affinity/NUMA mask, Slurm, and
  `nvidia-smi` metadata;
- the `/tmp` mount, filesystem, block device, and recursively expanded
  md/LVM slave topology;
- for md0 and each underlying device (for example `nvme0n1` and
  `nvme1n1`), queue values including `max_hw_sectors_kb`,
  `max_sectors_kb`, `optimal_io_size`, and `minimum_io_size`;
- the benchmark SHA-256 and repository revision/dirty state;
- every trial's exact bytes, operation count, wall time, decimal GB/s, binary
  GiB/s, and min/p50/p90/p95/p99/max/mean/stdev latency;
- component latency distributions for GPU copies and SSD syscalls;
- per-trial integrity evidence.

With four block sizes and lane counts 1 and 8, the expected trial count is 40:
eight pinned-DRAM trials and 32 SSD trials. The process exits nonzero and writes
`status: failed` plus a traceback when any environment contract, I/O, trial
count, or validation check fails.

DRAM throughput and latency have deliberately distinct sampling phases. The
throughput phase queues all copies on one CUDA stream and synchronizes once;
its published throughput uses CPU wall time, while a matching CUDA-event value
is also retained. The latency phase synchronizes every block and reports both
wall and CUDA-event distributions.

SSD throughput is aggregate wall throughput from a common thread barrier to the
last completed lane. Per-request latency is measured around each direct syscall
or complete end-to-end operation. Full-file SHA-256 validation uses a separate
untimed direct-read pass after the timed restore, so hashing cannot inflate I/O
latency and the validation pass cannot pre-read or warm the restore workload.

## Local checks

The CPU-only self-check imports neither torch nor CUDA and does not touch a
storage file:

```bash
python3 evidence/hardware/dana_a100_offload/benchmark.py --self-check
python3 -m py_compile evidence/hardware/dana_a100_offload/benchmark.py
bash -n evidence/hardware/dana_a100_offload/run_dana_a100.sbatch
```

The real benchmark is Linux-only because honest `O_DIRECT`,
`preadv`/`pwritev`, CUDA, and pinned memory are part of the evidence
contract.

## Calibration and production replay

The tracked calibration is split from the ignored per-job results:

- `calibrate.py` validates exactly two fit jobs and one disjoint holdout job,
  fits the narrow host-DRAM and NVMe profiles without using the holdout, and
  enforces explicit claim gates;
- `external_calibration_probe.cpp` replays every matrix cell through the
  production `ExternalBackingDevice::issue_contiguous_range` implementation;
- `calibration/dana-a100-offload-calibration.json` is the immutable,
  machine-readable aggregate, including raw-result SHA-256 values, job IDs,
  fitted fields, exact range/page/transport-command counter receipts, per-cell
  errors, and eligibility decisions;
- `calibration/dana-a100-offload-calibration.md` is its concise human report;
- `CALIBRATION_MATH_PROVENANCE.md` derives the fit equations (fixed cost plus
  byte service, queue-width scaling) and cites the model families they follow.

The fitted values ship as `configs/overlays/backing/calibrated/dana-a100-*.cfg`;
`calibrate.py --self-test` fails when the host-DRAM or NVMe overlay lines drift
from the aggregate.

Build the probe and reproduce the published fit plus independent holdout from
the repository root:

```bash
make -C build external_calibration_probe
python3 evidence/hardware/dana_a100_offload/calibrate.py \
  --probe build/external_calibration_probe \
  --fit-input evidence/hardware/dana_a100_offload/results/dana-a100-offload-188541.json \
  --fit-input evidence/hardware/dana_a100_offload/results/dana-a100-offload-188542.json \
  --holdout evidence/hardware/dana_a100_offload/results/dana-a100-offload-188543.json
```

Quick results are rejected unless `--allow-quick` is explicitly supplied and
must not be mixed with full results. The published input set is three full
production jobs on `gpu-51`: 188541/188542 are the only fit inputs, and the
untouched job 188543 is the run-level holdout. The earlier quick run is
excluded.
The aggregate also records the local calibration interpreter version and the
calibrator SHA-256; the tracked artifact was generated with Python 3.12.13.

NVMe transport segments are selected from the recorded sysfs limit (128 KiB),
while pinned-host copies remain one application-sized 16 MiB segment. Host
directional bandwidth and fixed delay are fitted only from pinned-copy cells.
NVMe media uses raw `O_DIRECT` 16 MiB QD1/QD8 cells for directional service and
small raw QD1 cells for fixed latency. GPU↔SSD end-to-end cells never enter the
fit and remain validation. Every prediction, including the holdout, is emitted
by the compiled production range path rather than a separate fit equation.

The fit-run QD8/QD1 scaling selects one effective read queue and one effective
write queue for gpu51's md/NVMe path. These are caller-range service widths,
not claims about literal NVMe submission queues. On the independent 16 MiB
holdout, median absolute throughput/p50 errors are 1.3% and 3.0%; maxima are
8.8% and 14.6%. The raw plus end-to-end saturated-read QD8 gate also passes
(maximum throughput/p50 errors 1.2% and 1.5%). The artifact remains
`calibrated_with_limits` because other request sizes and systems are outside
scope; it supports this gpu51 DANA 16 MiB QD1–QD8 workload anchor, not a
portable or paper-level hardware claim.

Run the deterministic fit/replay self-test with:

```bash
python3 evidence/hardware/dana_a100_offload/calibrate.py \
  --self-test --probe build/external_calibration_probe
```
