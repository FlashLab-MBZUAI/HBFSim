# Configuration catalog

> Status: Current
> Last reviewed: 2026-09-29

A run is a **system profile** (what hardware exists), an optional
**policy profile** (what the reference runner does with it), zero or more
**overlays** (one-purpose overrides) and command-line `--KEY VALUE`
overrides, applied in that order. Every physical key has an entry with an
evidence grade in [`parameter-provenance.json`](parameter-provenance.json);
the `parameter_provenance_coverage` test fails when a profile uses a key
without one.

The physical contract is OCP HBF v0.7.0 (130 pages, 3 August 2026) and the
publicly documented JEDEC HBM4 JESD270-4 organization (April 2025). See the
[model reference](../docs/reference/model.md). A speed Grade is an OCP
interface envelope; it does not select an HBM generation or an evidence grade.

| Directory | Contents |
|---|---|
| `systems/` | Complete capacity, device, host memory and interconnect profiles |
| `systems/miniquick/` | Capacity-scaled variants of the four topology profiles for bounded comparisons |
| `policies/reference/` | Placement, streaming and HBM staging settings for `hbfsim-reference`, one per published system |
| `overlays/hbf/` | OCP Grades, host mapping strategies, controller-DRAM budgets, an endurance window and thermal boundaries |
| `overlays/backing/` | Host DRAM, CXL, NVMe and on-package LPDDR alternatives behind HBM, plus measured timing overlays |

```sh
python3 -m hbfsim run --system 4hbm-4hbf --overlay ocp-v070-grade2 --set hbf-program-ns=50000
./build/hbfsim-reference \
  --config configs/systems/eight-stack-baseline.cfg \
  --config configs/policies/reference/eight-stack-baseline.cfg \
  --trace WORKLOAD.trace --summary-json out/comparison.json --config-out out/resolved.cfg
```

`hbfsim --system-config FILE` accepts system hardware and host HBF controller
settings. `hbfsim-reference --config SYSTEM --config POLICY` additionally
selects reference policies. Repeated files apply in order; explicit
command-line options override file values. No policy is inferred from a
filename. Resolved exports record both standard revisions and replay directly.

## System profiles

| Profile | Composition |
|---|---|
| `eight-stack-baseline` | Eight HBM stacks, no HBF; the all-HBM reference point |
| `2hbm-6hbf`, `4hbm-4hbf`, `6hbm-2hbf` | Eight slots split between HBM and HBF stacks |
| `server-hbm128-hbf512` | 128 GiB HBM in front of 512 GiB HBF (`-large-buffer`, `-tight-fabric` variants) |
| `server-hbm128-hbf1024` | The same host memory with 1 TiB of HBF |
| `miniquick/*` | The four topology profiles with capacities scaled down so a real serving window fits a short run |

[`systems/sglang-small.cfg`](systems/sglang-small.cfg) is a ServeLoop overlay
fragment rather than a published profile: it overlays the eight-stack baseline
with a bounded two-stack HBM/HBF geometry that ServeLoop's SGLang integration
references by this path. It has no `policies/reference/` pair and
`tests/python/test_system_configs.py` lists it as the only unpaired file.

Slot count alone does not normalize package area, cost or power; the topology
profiles are a fixed-slot comparison, not a product roadmap.

## HBF device and host ownership

The default Grade 2 profile has 16 independent channels, one die per channel,
16 Banks per die, 4096-byte pages and 512 GiB per stack. `hbf-planes-per-die`
names the NAND Bank count in the simulator's geometry. Every Bank has one sense
resource and two decoded page slots. Mapping, GC and write buffering belong to
host configuration: the controller storage has a finite capacity budget and
per-stack mapping/write-buffer access latency and issue resources, reserved
from physical HBM capacity, and its transfers share the HBM data-channel
calendars with application traffic. Mapped HBF configurations require HBM;
CPU offload memory is an explicit backing device in the system composition.

| Overlay | Channels | Payload/channel | Payload/stack |
|---|---:|---:|---:|
| `overlays/hbf/ocp-v070-grade1.cfg` | 8 | 48 GB/s | 384 GB/s |
| `overlays/hbf/ocp-v070-grade2.cfg` | 16 | 96 GB/s | 1536 GB/s |
| `overlays/hbf/ocp-v070-grade3.cfg` | 16 | 192 GB/s | 3072 GB/s |

Rates use decimal GB/s and include the standard's 75% efficiency once. Grade 1
reduces the channel count; without a compensating block-count/capacity override
it also reduces capacity. Compare Grades at eight channels when isolating rate.
Pages 32/35 of the specification conflict with Table 2/4 rates; this model
follows page 16 and records the conflict in provenance.

All system profiles use a 4 us NAND array read and a 75 us complete page
program (verification included). Command/data transfers, ECC and page-zero
autoerase remain separate costs; 2/4/8 us reads and 37.5/75/150 us programs
are the usual sensitivity points. These are study assumptions, not normative
OCP timings.

### HBF overlays

| Overlay | Purpose |
|---|---|
| `overlays/hbf/mapping/resident.cfg` | Page-level L2P fully resident in host HBM |
| `overlays/hbf/mapping/page-cache.cfg`, `entry-cache.cfg`, `compressed-cache.cfg` | Translation-page, DFTL-style entry, and compressed-run caches over the page-level L2P |
| `overlays/hbf/mapping/block.cfg`, `block-log-striped.cfg`, `extent-striped.cfg` | Block, hybrid block-log and extent organizations; the striped ones set `hbf-mapping-superblock-planes=all` |
| `overlays/hbf/cached-l2p-1-over-1000.cfg` | Controller DRAM budget of 1/1000 of HBF capacity for cached mapping |
| `overlays/hbf/external-direct-lane.cfg` | A direct per-stack lane between each HBF base die and the external backing device |
| `overlays/hbf/endurance-regime-scaled.cfg` | A 4 MiB raw HBF so GC and wear leveling run within seconds of simulated time |
| `overlays/hbf/thermal-boundary-55c.cfg`, `65c`, `70c` | Package boundary temperatures for the RC thermal model |
| `overlays/hbf/thermal-steady-state-start.cfg` | Boot every HBF stack at the throttle ceiling instead of a cold start |

Full-resident mapping and the bounded caches are host policy alternatives.
Raw physical mode requires an explicitly installed image or preceding
programs and provides no implicit in-place FTL. Page-zero writes perform
automatic erase; host GC reclaim and zone reset do not themselves increment
physical P/E counts.

### Backing overlays

`overlays/backing/host-dram.cfg`, `cxl-memory.cfg`, `nvme-ssd.cfg`,
`on-package-lpddr.cfg`, `cxl-ssd.cfg` and `cxl-ssd-cached.cfg` each replace
the device behind HBM; `external-streaming` streams layers from it. Most
envelopes are exploratory. The `overlays/backing/calibrated/` overlays carry
measured timings: `dana-a100-host-dram-timing.cfg`,
`dana-a100-nvme-ssd-timing.cfg` and `dana-a100-nvme-media-only-timing.cfg`
from the [DANA A100 measurement](../evidence/hardware/dana_a100_offload/README.md),
and `h200-nvl-host-dram-timing.cfg` from the
[H200 host-link measurement](../evidence/hardware/host_link/README.md). Apply
them after the matching device overlay.

## HBM

HBM4 defaults use 32 channels, two pseudo-channels per channel, 2048 total DQ
pins and 8 Gb/s per pin. Capacity is explicit in each profile. The HBM timing
model is `channel-aggregate-v2`, with these independent assumptions:

| Option | Default | Meaning |
|---|---:|---|
| `hbm-read-latency-ns` | 42 | Aggregate read access latency |
| `hbm-write-latency-ns` | 38 | Aggregate write access latency |
| `hbm-bandwidth-efficiency` | 0.94 | Effective / raw per-channel bandwidth |
| `hbm-read-to-write-ns` | 8 | Application read-to-write switch delay |
| `hbm-write-to-read-ns` | 8 | Application write-to-read switch delay |
| `hbm-service-quantum-bytes` | 4096 | Maximum indivisible application service per channel |
| `hbm-service-group-channels` | 4 | Consecutive pseudochannels sharing arbitration; busiest lane sets duration |

`hbm-queue-depth` bounds parent transfers per service group. Grouping can
serialize concurrent traffic to disjoint lanes; use group size 1 for precision
sensitivity with the same implementation. Burst rounding, address interleave
and reserved controller capacity remain explicit. ACT/PRE/refresh timing keys
and scheduler switches are rejected; see the
[model reference](../docs/reference/model.md#hbm) for the measured
approximation error of the aggregate model.

Parameter studies vary only supported current parameters. A Bank-count sweep
must preserve capacity explicitly if it claims to isolate concurrency. A Grade
sweep must satisfy the channel/die maximum of every selected Grade.

<details>
<summary>File index</summary>

- [parameter-provenance.json](parameter-provenance.json)
- [systems/2hbm-6hbf.cfg](systems/2hbm-6hbf.cfg)
- [systems/4hbm-4hbf.cfg](systems/4hbm-4hbf.cfg)
- [systems/6hbm-2hbf.cfg](systems/6hbm-2hbf.cfg)
- [systems/eight-stack-baseline.cfg](systems/eight-stack-baseline.cfg)
- [systems/miniquick/2hbm-6hbf.cfg](systems/miniquick/2hbm-6hbf.cfg)
- [systems/miniquick/4hbm-4hbf.cfg](systems/miniquick/4hbm-4hbf.cfg)
- [systems/miniquick/6hbm-2hbf.cfg](systems/miniquick/6hbm-2hbf.cfg)
- [systems/miniquick/eight-stack-baseline.cfg](systems/miniquick/eight-stack-baseline.cfg)
- [systems/server-hbm128-hbf1024.cfg](systems/server-hbm128-hbf1024.cfg)
- [systems/server-hbm128-hbf512-large-buffer.cfg](systems/server-hbm128-hbf512-large-buffer.cfg)
- [systems/server-hbm128-hbf512-tight-fabric.cfg](systems/server-hbm128-hbf512-tight-fabric.cfg)
- [systems/server-hbm128-hbf512.cfg](systems/server-hbm128-hbf512.cfg)
- [systems/sglang-small.cfg](systems/sglang-small.cfg)
- [policies/reference/2hbm-6hbf.cfg](policies/reference/2hbm-6hbf.cfg)
- [policies/reference/4hbm-4hbf.cfg](policies/reference/4hbm-4hbf.cfg)
- [policies/reference/6hbm-2hbf.cfg](policies/reference/6hbm-2hbf.cfg)
- [policies/reference/eight-stack-baseline.cfg](policies/reference/eight-stack-baseline.cfg)
- [policies/reference/server-hbm128-hbf1024.cfg](policies/reference/server-hbm128-hbf1024.cfg)
- [policies/reference/server-hbm128-hbf512-large-buffer.cfg](policies/reference/server-hbm128-hbf512-large-buffer.cfg)
- [policies/reference/server-hbm128-hbf512-tight-fabric.cfg](policies/reference/server-hbm128-hbf512-tight-fabric.cfg)
- [policies/reference/server-hbm128-hbf512.cfg](policies/reference/server-hbm128-hbf512.cfg)
- [overlays/hbf/cached-l2p-1-over-1000.cfg](overlays/hbf/cached-l2p-1-over-1000.cfg)
- [overlays/hbf/endurance-regime-scaled.cfg](overlays/hbf/endurance-regime-scaled.cfg)
- [overlays/hbf/external-direct-lane.cfg](overlays/hbf/external-direct-lane.cfg)
- [overlays/hbf/mapping/block-log-striped.cfg](overlays/hbf/mapping/block-log-striped.cfg)
- [overlays/hbf/mapping/block.cfg](overlays/hbf/mapping/block.cfg)
- [overlays/hbf/mapping/compressed-cache.cfg](overlays/hbf/mapping/compressed-cache.cfg)
- [overlays/hbf/mapping/entry-cache.cfg](overlays/hbf/mapping/entry-cache.cfg)
- [overlays/hbf/mapping/extent-striped.cfg](overlays/hbf/mapping/extent-striped.cfg)
- [overlays/hbf/mapping/page-cache.cfg](overlays/hbf/mapping/page-cache.cfg)
- [overlays/hbf/mapping/resident.cfg](overlays/hbf/mapping/resident.cfg)
- [overlays/hbf/ocp-v070-grade1.cfg](overlays/hbf/ocp-v070-grade1.cfg)
- [overlays/hbf/ocp-v070-grade2.cfg](overlays/hbf/ocp-v070-grade2.cfg)
- [overlays/hbf/ocp-v070-grade3.cfg](overlays/hbf/ocp-v070-grade3.cfg)
- [overlays/hbf/thermal-boundary-55c.cfg](overlays/hbf/thermal-boundary-55c.cfg)
- [overlays/hbf/thermal-boundary-65c.cfg](overlays/hbf/thermal-boundary-65c.cfg)
- [overlays/hbf/thermal-boundary-70c.cfg](overlays/hbf/thermal-boundary-70c.cfg)
- [overlays/hbf/thermal-steady-state-start.cfg](overlays/hbf/thermal-steady-state-start.cfg)
- [overlays/backing/calibrated/dana-a100-host-dram-timing.cfg](overlays/backing/calibrated/dana-a100-host-dram-timing.cfg)
- [overlays/backing/calibrated/dana-a100-nvme-media-only-timing.cfg](overlays/backing/calibrated/dana-a100-nvme-media-only-timing.cfg)
- [overlays/backing/calibrated/dana-a100-nvme-ssd-timing.cfg](overlays/backing/calibrated/dana-a100-nvme-ssd-timing.cfg)
- [overlays/backing/calibrated/h200-nvl-host-dram-timing.cfg](overlays/backing/calibrated/h200-nvl-host-dram-timing.cfg)
- [overlays/backing/cxl-memory.cfg](overlays/backing/cxl-memory.cfg)
- [overlays/backing/cxl-ssd-cached.cfg](overlays/backing/cxl-ssd-cached.cfg)
- [overlays/backing/cxl-ssd.cfg](overlays/backing/cxl-ssd.cfg)
- [overlays/backing/host-dram.cfg](overlays/backing/host-dram.cfg)
- [overlays/backing/nvme-ssd.cfg](overlays/backing/nvme-ssd.cfg)
- [overlays/backing/on-package-lpddr.cfg](overlays/backing/on-package-lpddr.cfg)

</details>
