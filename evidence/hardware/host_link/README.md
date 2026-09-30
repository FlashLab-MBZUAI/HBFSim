# H200 host-link measurement

> Status: Experimental
> Last reviewed: 2026-09-29

`bench.py` measures GPU ↔ pinned host DRAM copies through the CUDA driver API
(`libcuda` only, no toolkit or PyTorch), recording the same quantities as the
[DANA A100 offload anchor](../dana_a100_offload/README.md): the pipelined issue
interval of back-to-back asynchronous copies and the single-copy latency, per
direction (`offload_d2h`, `restore_h2d`) and block size.

`fit.py` turns one or more result files into an HBFSim host-DRAM timing overlay
with the DANA method (`fit_host` in `../dana_a100_offload/calibrate.py`):
issue time from the pipelined interval, per-direction bandwidth from a
least-squares slope over block size, and fixed processing time from the median
latency residual. Repeated runs are pooled by the median of each
(direction, block) point.

```bash
python3 evidence/hardware/host_link/bench.py --out run.json          # on the GPU node
python3 -m evidence.hardware.host_link.fit --platform "H200 NVL, PCIe Gen5 x16" \
  --out configs/overlays/backing/calibrated/h200-nvl-host-dram-timing.cfg \
  evidence/hardware/host_link/results/h200-nvl-pcie5/run{1,2,3}.json
```

`results/h200-nvl-pcie5/` holds the three runs (2026-09-23) behind the shipped
overlay; re-running `fit.py` on them reproduces its lines exactly. There is no
held-out run, so the overlay is a measured timing envelope for that platform,
not an L4 calibration in the sense of the [evidence policy](../../../docs/reference/evidence-policy.md).
