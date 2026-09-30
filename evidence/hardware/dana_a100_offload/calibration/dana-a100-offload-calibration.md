# DANA A100 offload calibration

Status: **calibrated_with_limits**. Jobs 188541/188542 are fit inputs; job 188543 is an independent run-level holdout. The quick smoke run is excluded.

This is a workload-scoped platform anchor for one A100, CUDA-pinned host memory, and the compute-local md/NVMe stack with 1 visible NVMe leaf device(s). Every prediction is produced by the compiled `ExternalBackingDevice::issue_contiguous_range` path.

## Evidence split

| Role | Slurm job | Host | CUDA slot | Raw JSON SHA-256 |
|---|---:|---|---:|---|
| fit | 188541 | gpu-51 | 2 | `9148770c15bf40a0dc4f2b774cafb4eff4ccc3a2956247e18813a9298ebbce58` |
| fit | 188542 | gpu-51 | 2 | `0dc3ad2784677d5a07211e947f5a8f7675ae1c37d0affbc95267cbd736e9067c` |
| holdout | 188543 | gpu-51 | 2 | `5fd8fffec2084feb8c8b5148b0e4aef8e43ef8b7682429838b4fe8222cf2475c` |

The SSD request segment is **131072 bytes (128 KiB)**, selected from the minimum recorded mounted-stack/NVMe queue limit. The fit selected **1 read queue(s)** and **1 write queue(s)** from the QD8/QD1 scaling anchors without reading the holdout; these are effective caller-range service widths, not literal device queues.

## Fitted production fields

| Field | host-dram | nvme-ssd |
|---|---:|---:|
| `request_segment_bytes` | 16777216 | 131072 |
| `media_channels` | 1 | 1 |
| `media_read_queues` | 1 | 1 |
| `media_write_queues` | 1 | 1 |
| `max_outstanding_requests` | 512 | 512 |
| `controller_issue_ns` | 5100.4355 | 5100.4355 |
| `controller_processing_ns` | 28375.261 | 28375.261 |
| `media_read_latency_ns` | 0 | 9673.8745 |
| `media_write_latency_ns` | 0 | 10523.547 |
| `media_read_bandwidth_GBps` | 1000000 | 2.8470902 |
| `media_write_bandwidth_GBps` | 1000000 | 2.5536247 |
| `m2s_bandwidth_GBps` | 25.596645 | 25.596645 |
| `s2m_bandwidth_GBps` | 23.077588 | 23.077588 |

Host fields use pinned-copy cells from the two fit jobs. SSD media fields use raw `O_DIRECT`; GPU↔SSD cells never enter the fit. The 16 MiB QD1/QD8 anchors determine directional media service, while small raw QD1 cells determine fixed media latency.

## Fit-run replay and cross-size diagnostics

| Path | Direction | Block | Lanes | Role | Measured GB/s | HBFSim GB/s | Error | Measured p50 | HBFSim p50 | Error |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| gpu_pinned_host | offload_d2h | 4096 | 1 | fit input | 0.8349 | 0.8025 | -3.9% | 32.3 µs | 33.6 µs | +4.1% |
| gpu_pinned_host | offload_d2h | 65536 | 1 | fit input | 13.3398 | 12.8112 | -4.0% | 33.1 µs | 36.0 µs | +8.9% |
| gpu_pinned_host | offload_d2h | 1048576 | 1 | fit input | 22.7494 | 25.5543 | +12.3% | 74.1 µs | 74.4 µs | +0.5% |
| gpu_pinned_host | offload_d2h | 16777216 | 1 | fit input | 25.4053 | 25.5914 | +0.7% | 718.1 µs | 688.9 µs | -4.1% |
| gpu_pinned_host | restore_h2d | 4096 | 1 | fit input | 0.7746 | 0.8025 | +3.6% | 34.0 µs | 33.7 µs | -1.1% |
| gpu_pinned_host | restore_h2d | 65536 | 1 | fit input | 11.2842 | 12.8108 | +13.5% | 35.1 µs | 36.3 µs | +3.4% |
| gpu_pinned_host | restore_h2d | 1048576 | 1 | fit input | 16.8587 | 23.0441 | +36.7% | 81.8 µs | 78.9 µs | -3.5% |
| gpu_pinned_host | restore_h2d | 16777216 | 1 | fit input | 22.9395 | 23.0734 | +0.6% | 761.7 µs | 760.5 µs | -0.2% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 4096 | 1 | validation | 0.0813 | 0.0895 | +10.1% | 48.3 µs | 45.8 µs | -5.3% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 4096 | 8 | validation | 0.0950 | 0.7157 | +653.0% | 336.9 µs | 45.8 µs | -86.4% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 65536 | 1 | validation | 0.8445 | 0.9074 | +7.4% | 74.4 µs | 72.2 µs | -2.9% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 65536 | 8 | validation | 1.4847 | 2.5525 | +71.9% | 340.7 µs | 205.3 µs | -39.7% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 1048576 | 1 | validation | 2.3258 | 2.2808 | -1.9% | 433.4 µs | 459.7 µs | +6.1% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 1048576 | 8 | validation | 2.5320 | 2.5533 | +0.8% | 3384.1 µs | 3285.0 µs | -2.9% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 16777216 | 1 | validation | 2.3309 | 2.5347 | +8.7% | 7071.8 µs | 6619.1 µs | -6.4% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 16777216 | 8 | validation | 2.6630 | 2.5536 | -4.1% | 46846.0 µs | 52559.7 µs | +12.2% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 4096 | 1 | validation | 0.0763 | 0.0915 | +19.9% | 46.5 µs | 44.8 µs | -3.7% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 4096 | 8 | validation | 0.0962 | 0.7317 | +660.5% | 327.5 µs | 44.8 µs | -86.3% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 65536 | 1 | validation | 0.8906 | 0.9496 | +6.6% | 70.7 µs | 69.0 µs | -2.4% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 65536 | 8 | validation | 1.2538 | 2.8457 | +127.0% | 395.1 µs | 184.1 µs | -53.4% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 1048576 | 1 | validation | 1.3262 | 2.5138 | +89.6% | 773.0 µs | 417.1 µs | -46.0% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 1048576 | 8 | validation | 2.9836 | 2.8467 | -4.6% | 2751.3 µs | 2946.4 µs | +7.1% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 16777216 | 1 | validation | 2.5441 | 2.8237 | +11.0% | 6583.0 µs | 5941.6 µs | -9.7% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 16777216 | 8 | validation | 2.8303 | 2.8470 | +0.6% | 47148.5 µs | 47142.1 µs | -0.0% |
| ssd_direct | offload_pwrite | 4096 | 1 | fit input | 0.2796 | 0.3377 | +20.8% | 13.4 µs | 12.1 µs | -9.7% |
| ssd_direct | offload_pwrite | 4096 | 8 | validation | 0.7677 | 2.5526 | +232.5% | 39.9 µs | 12.8 µs | -67.8% |
| ssd_direct | offload_pwrite | 65536 | 1 | fit input | 1.7945 | 1.8110 | +0.9% | 34.9 µs | 36.2 µs | +3.7% |
| ssd_direct | offload_pwrite | 65536 | 8 | validation | 2.6103 | 2.5534 | -2.2% | 172.8 µs | 205.3 µs | +18.8% |
| ssd_direct | offload_pwrite | 1048576 | 1 | validation | 2.4578 | 2.4898 | +1.3% | 362.7 µs | 421.1 µs | +16.1% |
| ssd_direct | offload_pwrite | 1048576 | 8 | validation | 2.4987 | 2.5536 | +2.2% | 3380.3 µs | 3285.0 µs | -2.8% |
| ssd_direct | offload_pwrite | 16777216 | 1 | fit input | 2.4971 | 2.5495 | +2.1% | 6627.0 µs | 6580.5 µs | -0.7% |
| ssd_direct | offload_pwrite | 16777216 | 8 | fit input | 2.6115 | 2.5536 | -2.2% | 48553.6 µs | 52559.7 µs | +8.3% |
| ssd_direct | restore_pread | 4096 | 1 | fit input | 0.3391 | 0.3686 | +8.7% | 11.2 µs | 11.1 µs | -1.1% |
| ssd_direct | restore_pread | 4096 | 8 | validation | 0.4683 | 2.8459 | +507.7% | 35.1 µs | 11.5 µs | -67.2% |
| ssd_direct | restore_pread | 65536 | 1 | fit input | 1.7050 | 2.0046 | +17.6% | 32.6 µs | 32.7 µs | +0.4% |
| ssd_direct | restore_pread | 65536 | 8 | validation | 1.6972 | 2.8468 | +67.7% | 279.1 µs | 184.1 µs | -34.0% |
| ssd_direct | restore_pread | 1048576 | 1 | validation | 1.5280 | 2.7742 | +81.6% | 658.0 µs | 378.0 µs | -42.6% |
| ssd_direct | restore_pread | 1048576 | 8 | validation | 2.9913 | 2.8470 | -4.8% | 2736.4 µs | 2946.4 µs | +7.7% |
| ssd_direct | restore_pread | 16777216 | 1 | fit input | 2.8740 | 2.8424 | -1.1% | 5834.7 µs | 5902.4 µs | +1.2% |
| ssd_direct | restore_pread | 16777216 | 8 | fit input | 2.8204 | 2.8471 | +0.9% | 47571.0 µs | 47142.1 µs | -0.9% |

## Independent 16 MiB workload holdout (job 188543)

| Path | Direction | Lanes | Measured GB/s | HBFSim GB/s | Error | Measured p50 | HBFSim p50 | Error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpu_pinned_host | offload_d2h | 1 | 25.3094 | 25.5914 | +1.1% | 0.7 ms | 0.7 ms | +3.0% |
| gpu_pinned_host | restore_h2d | 1 | 23.1956 | 23.0734 | -0.5% | 0.7 ms | 0.8 ms | +1.8% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 1 | 2.3456 | 2.5347 | +8.1% | 7.1 ms | 6.6 ms | -6.2% |
| gpu_pinned_host_ssd_direct | offload_gpu_to_ssd | 8 | 2.5868 | 2.5536 | -1.3% | 51.3 ms | 52.6 ms | +2.4% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 1 | 2.5945 | 2.8237 | +8.8% | 6.5 ms | 5.9 ms | -8.1% |
| gpu_pinned_host_ssd_direct | restore_ssd_to_gpu | 8 | 2.8369 | 2.8470 | +0.4% | 47.4 ms | 47.1 ms | -0.5% |
| ssd_direct | offload_pwrite | 1 | 2.5855 | 2.5495 | -1.4% | 6.4 ms | 6.6 ms | +3.0% |
| ssd_direct | offload_pwrite | 8 | 2.6493 | 2.5536 | -3.6% | 45.9 ms | 52.6 ms | +14.6% |
| ssd_direct | restore_pread | 1 | 2.9257 | 2.8424 | -2.8% | 5.7 ms | 5.9 ms | +3.0% |
| ssd_direct | restore_pread | 8 | 2.8803 | 2.8471 | -1.2% | 46.4 ms | 47.1 ms | +1.5% |

The independent workload holdout has median absolute throughput/p50 errors of **1.3% / 3.0%** and maxima of **8.8% / 14.6%**.

## Claim gates

| Gate | Throughput threshold | p50 threshold | Eligible |
|---|---:|---:|---|
| independent 16 MiB holdout median | ≤ 15% | ≤ 15% | yes |
| independent 16 MiB holdout per-cell max | ≤ 20% | ≤ 25% | yes |
| saturated SSD reads (raw + e2e, QD8) | ≤ 15% | ≤ 25% | yes |

`workload_applicability_gate` is **true**. Cross-size diagnostics remain limitations and do not grant a portable or paper-level hardware claim.

## Limits

- The anchor covers DANA gpu-51 physical GPU 2, CUDA-pinned host memory, and its node-local XFS/LVM/md/NVMe stack with 1 visible NVMe leaf device(s); it is not a portable machine constant.
- SSD writes mean successful synchronous O_DIRECT pwritev completion, not power-loss durability.
- The fitted read/write queue counts are effective caller-range service widths selected from fit-run QD8/QD1 scaling on this md/NVMe stack; they are not asserted to be literal NVMe submission-queue counts.
- Small-block cells include Python, CUDA synchronization, syscall, and thread scheduling overheads that are outside a media-only device model.
- The claim gate is the untouched third job's 16 MiB QD1/QD8 matrix. Errors at 4 KiB, 64 KiB, and 1 MiB remain visible cross-size diagnostics and are outside the eligible workload scope.
- Host DRAM denotes the measured GPU-to-CUDA-pinned-host path, not arbitrary pageable CPU memory and not CXL memory.
- Media bandwidth in the host-only profile is a numerical non-bottleneck sentinel; measured transfer is represented exactly once by the directional host link.
- The one-TiB capacity in replay configs is only a nonallocating address-space sentinel. This benchmark calibrates transfer timing, not usable DRAM or SSD capacity, and the generated overlay lines intentionally omit capacity.
