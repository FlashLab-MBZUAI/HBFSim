# Evidence catalog

> Status: Current
> Last reviewed: 2026-09-29

Evidence is what stands behind a number that is not a published standard or a
declared assumption. It is never imported as a hidden model default: every
overlay that carries a measured value names its package here, and
[`configs/parameter-provenance.json`](../configs/parameter-provenance.json)
records the source and the claim boundary.

| Path | Responsibility |
| --- | --- |
| `external/collect.py` | Build pinned external simulators (Ramulator2, MQSim) and collect the declared comparison cases |
| `external/verify.py` | Verify pins, artifacts, fixtures, and the named L3 facets |
| `external/assets/` | Pinned manifests, adapters, traces, and expected fixture translations |
| `hardware/dana_a100_offload/` | Measured A100 pinned-host-DRAM and local-NVMe offload timing behind the `backing/calibrated/dana-a100-*` overlays, with its fit and a replay self-test |
| `hardware/host_link/` | Measured H200 NVL GPU–host copy timing behind the `backing/calibrated/h200-nvl-host-dram-timing` overlay |

The [evidence policy](../docs/reference/evidence-policy.md) defines the claim
levels these packages can support; the [verification framework](../verification/README.md)
explains how external evidence enters a foundational certificate.
