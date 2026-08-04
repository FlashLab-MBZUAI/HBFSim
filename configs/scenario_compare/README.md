# scenario_compare Config Files

`scenario_compare` reads `key=value` config files with the same key names as
long command-line options, without the leading `--`.

Keep stable model and policy parameters here:

```text
HBM/HBF capacity and topology
HBM/HBF timing and bandwidth
HBF flash geometry, resident FTL, write buffer, and GC policy
flat split size
EC6 layer-buffer and base-die link settings
external CXL-memory/SSD media, queue, and host-link settings
request/page granularity
```

Keep run artifacts on the command line:

```text
--trace
--summary-csv
--summary-json
--chrome-trace
--config-out
--max-ops
```

Example:

```bash
./build/scenario_compare \
  --config configs/scenario_compare/server-4k-hbf4x.cfg \
  --trace WORKLOAD.trace \
  --summary-csv out/server/summary-policy-set.csv \
  --summary-json out/server/summary-policy-set.json \
  --config-out out/server/summary-policy-set.cfg
```

Multiple `--config` files may be passed. Config files are applied first (in
argv order, later files overriding earlier ones), then every other CLI option
is applied (in argv order). A CLI option therefore always overrides config
values, no matter where `--config` sits on the command line. Use CLI
overrides only for one-off sweeps; promote stable sweeps into a named config
file.

## Provided Configs

`server-4k-hbf4x.cfg`

Default server-scale policy set: 128 GiB HBM, 512 GiB HBF, 4 KiB requests and
HBF pages, and a 32 MiB per-buffer EC6 layer-data limit. Runtime may allocate
smaller buffers.

`server-4k-hbf8x.cfg`

Same topology and policy as `server-4k-hbf4x.cfg`, but with 1 TiB HBF
capacity. Capacity is changed through `hbf-capacity-ratio`; block size and page
size stay the same so the 4x/8x comparison does not also change erase
granularity.

`server-4k-hbf4x-large-layer-buffer.cfg`

Same hardware as `server-4k-hbf4x.cfg`, but with a 256 MiB per-buffer EC6
layer-data limit.

`server-4k-hbf4x-tight-fabric.cfg`

Same platform as `server-4k-hbf4x.cfg`, but media lanes and page-buffer banks
run at 1 GB/s so the internal read fabric binds (a 4 KiB hop costs about as
much as one array sense). Use for internal-provisioning sweeps; in the default
config those two stages are deliberately transparent.

`external-cxl-memory.overlay`, `nvme-ssd.overlay`

No-HBF backing overlays for `HBM+External-layer-streaming`. Apply one after a
complete `.cfg` profile. They resolve the same external-backing model to either
an address-striped CXL-memory sensitivity point or an NVMe-SSD sensitivity
point; changing the overlay therefore does not change layer placement,
double-buffer scheduling, or foreground HBM behavior. These values are
exploratory assumptions rather than calibrated commercial-device claims.

`usecase-baseline.cfg`, `usecase-6h2f.cfg`, `usecase-4h4f.cfg`,
`usecase-2h6f.cfg`

Exploratory eight-slot composition profiles: single-medium baselines and
6-HBM/2-HBF, 4-HBM/4-HBF, and 2-HBM/6-HBF points. These files are model inputs,
not measured hardware configurations. Their evidence grades and unresolved
assumptions are recorded in `../parameter-provenance.json` and
`../../docs/model-validation.md`. EC2 and EC6 use 6HBM/2HBF; EC3 uses
4HBM/4HBF; EC4 and EC5 use 2HBM/6HBF. Every EC topology point uses the same
TCAD-derived 1 us NAND read, calibrated 1.284 us uncontended total read,
100 us program+verify, 250 ns response residual, and 2048/512 GB/s per-stack
D2D read/write values. Topology comparisons therefore do not silently change
the media timing profile.

No configuration in this directory is, by itself, a paper-qualified Frontier
study. The obsolete single-request 7B media/HBIO phase declaration was
removed. A future publication path must bind one 70B multi-request workload to
the same placement, scheduler, credit, and capacity-pressure contract across
all-HBM, HBM+HBF, HBM+CXL-memory, and HBM+NVMe-SSD baselines.

`usecase-hbm3e-micron-36g-9p2.overlay`

HBM-only overlay (intentionally not a standalone `.cfg`) for a
capacity-consistent Micron HBM3E comparison point. Apply it after a use-case
composition config. Vendor-published fields are 36 GB/12H,
9.2 GT/s, 16 channels, two pseudo-channels per channel, BL8, and a 1024-bit
interface; the model derives 1177.6 GB/s per stack. Each run must still set the
composition-specific total HBM capacity and flat address boundary. The binary
36 GiB/stack address mapping and inherited DRAM core timings remain explicit
model assumptions, so this is a source-anchored comparison profile rather than
a complete Micron speed-bin model.

`usecase-baseline-local-output.cfg`

Controlled local-output sensitivity derived from `usecase-baseline.cfg`.
Each plane retains 32 subarrays but increases its media lanes and page-buffer
banks from 16 to 32, so the current striped mapping gives each subarray one
local raw-data output. Channel, TSV, ECC, logic-SRAM, and HBIO provisioning
remain at the 1.6 TB/s-per-stack baseline. This isolates local output sharing;
it does not claim 32 independent external interfaces.

`usecase-baseline-output-upper-bound.cfg`

Cost-unbounded output-fabric sensitivity derived from the local-output point.
It keeps the same resource counts but raises every shared downstream data rate
to the modeled array supply. With 64 planes per stack, 32 subarrays per plane,
4 KiB pages, and a 1 us sense round, decoded supply is
`64 * 32 * 4096 B / 1000 ns = 8388.608 GB/s`; 224 B OOB per page raises the
raw-codeword data boundary to 8847.36 GB/s. Because the modeled stack TSV also
serializes one 64 B command per page, it is provisioned at 8978.432 GB/s; the
per-die flash TSU is provisioned at one request per 7.8125 ns and the logic
dispatcher at one 4 KiB request per 0.48828125 ns. This is a hypothetical
end-to-end provisioning bound, not a vendor-calibrated product profile or a
cost-neutral design. It does not eliminate array scheduling or workload
imbalance, so measured throughput is not promised to reach the bound.

## Parameter Groups

`line-size`, `interarrival-ns`

Request granularity and synthetic arrival spacing for traces that do not carry
explicit timestamps.

`hbm-capacity-bytes`, `hbm-stacks`, `hbm-channels`, `hbm-pseudo-channels`,
`hbm-bank-groups-per-pseudo-channel`, `hbm-banks-per-group`, `hbm-channel-row-size-bytes`,
`hbm-channel-width-bits`, `hbm-burst-length`, `hbm-pin-rate-gbps`,
`hbm-data-rate-per-command-clock`, `hbm-tccd-s-cycles`,
`hbm-tccd-l-cycles`

HBM capacity, hierarchy, and one-source-of-truth interface timing. Bank-group
count is explicitly per pseudo-channel. The command-clock ratio and tCCD
values are positive integer cycle counts. The model
derives pseudo-channel DQ width, row bytes, burst bytes, command clock/tCK,
burst duration, and pseudo-channel/channel/stack GB/s. It rejects widths that
do not divide evenly, non-byte-aligned pseudo-channels, fractional-command-
clock bursts, rows not divisible by a burst, and `tCCD_S` shorter than the
derived data burst. There is intentionally no independent `hbm-channel-bw` or
`hbm-tbl-ns` override.

The current `burst-pch-bank-swizzle-v1` mapping hashes high address bits into
pseudo-channel and bank selection to avoid common power-of-two stride aliases.
It is an optimistic controller implementation policy, not a JEDEC fact; the
scheme id is emitted in every summary and must accompany absolute results.

The use-case configs are a normalized HBM4-width sensitivity point: 32 x
64-bit channels, two 32-bit pseudo-channels, BL8, and 6.4 Gb/s/pin produce
1638.4 GB/s per stack. This reduced pin rate is not a current HBM4 product
claim. The server configs use 16 channels at 8 Gb/s/pin.

`hbm-address-mapping-ns`, `hbm-trcdrd-ns`, `hbm-trcdwr-ns`, `hbm-tcl-ns`,
`hbm-tcwl-ns`, `hbm-trp-ns`, `hbm-tras-ns`, `hbm-trc-ns`, `hbm-twr-ns`,
`hbm-trtp-ns`, `hbm-trrd-s-ns`, `hbm-trrd-l-ns`, `hbm-tfaw-ns`,
`hbm-twtr-s-ns`, `hbm-twtr-l-ns`, `hbm-trtw-ns`

Absolute core/turnaround timing minima. HBM commands issue only on derived
command-clock edges, so each minimum is rounded up causally by the scheduler.
The default 14 ns-class core timings remain exploratory assumptions rather
than a published HBM4 speed bin.

`hbm-refresh`, `hbm-same-bank-refresh`, `hbm-trefi-ns`, `hbm-trfc-ns`,
`hbm-trfcsb-ns`

HBM refresh interference. Off by default; scenario results without
`hbm-refresh=true` exclude refresh stalls.

`external-backing-kind`, `external-backing-capacity-bytes`,
`external-backing-page-size`, `external-backing-media-channels`,
`external-backing-max-outstanding-requests`,
`external-backing-controller-issue-ns`,
`external-backing-controller-processing-ns`,
`external-backing-media-read-latency-ns`,
`external-backing-media-write-latency-ns`,
`external-backing-media-read-bw`, `external-backing-media-write-bw`,
`external-backing-m2s-bw`, `external-backing-s2m-bw`,
`external-backing-one-way-propagation-ns`,
`external-backing-command-bytes`, `external-backing-completion-bytes`

The no-HBF layer-streaming backing device. `kind` selects the
`on-package-lpddr`, `cxl-memory`, or `nvme-ssd` starting profile; every
following value can override that profile. Capacity must be page aligned.
Pages are striped across media
channels; media bandwidth is aggregate across them. The device-wide
outstanding limit covers the complete request lifetime. Every read and write
passes through M2S wire serialization, one-way propagation, controller issue
and processing, media latency and serialization, S2M wire serialization, and
return propagation. M2S and S2M are independent full-duplex resources.
`command-bytes` and `completion-bytes` make protocol overhead auditable;
payload, protocol, and wire counters conserve separately by direction. All
fields are emitted in resolved configs and summaries.

The built-in numbers are reproducible sensitivity points, not calibrated
products. HotInfra'26 anchors the on-package LPDDR capacity/read-bandwidth/
latency point; public CXL/NVMe specifications support the host-attached
directional protocol structure, and public Micron/KIOXIA products anchor
bandwidth/capacity magnitudes. Internal channel counts, controller timing,
write bandwidth, protocol records, and several other values remain explicitly
graded `exploratory_assumption` in `../parameter-provenance.json`.

`hbf-capacity-ratio`, `hbf-capacity-bytes`, `hbf-stacks`, `hbf-channels`,
`hbf-dies-per-channel`, `hbf-planes-per-die`, `hbf-pages-per-block`,
`hbf-page-size`

HBF capacity target and flash geometry. Prefer `hbf-capacity-ratio` for
HBM/HBF size-ratio sweeps. `hbf-blocks-per-plane` is derived when a capacity
target is supplied.

`hbf-media-lanes-per-plane`, `hbf-subarrays-per-plane`,
`hbf-page-buffer-banks-per-plane`,
`hbf-page-read-queue-depth-per-stack`

The HBF internal read-path parallelism knobs: internal media transfer lanes,
array sensing subarrays, page-buffer output banks, and finite page-transaction
controller credits. The queue depth is per stack and applies after a large
memory-object request is split into pages; it is an exploratory architecture
assumption rather than a published product parameter.

`hbf-read-ns`, `hbf-program-ns`, `hbf-program-verify-ns`, `hbf-erase-ns`,
`hbf-channel-bw`, `hbf-hbio-bw`, `hbf-tsv-bw`, `hbf-media-lane-bw`,
`hbf-logic-sram-bw`, `hbf-page-buffer-bw`

HBF timing and bandwidth parameters for media and data movement. Units follow
the ECC boundary: channel (per channel), media lane (per lane), and page-buffer
bank (per bank) are raw codeword GB/s and therefore move page + OOB. The shared
per-stack TSV serializes those codewords and each internal command on one
timeline, so published read profiles provision it by
`(page + OOB + command) / page`. HBIO is the external decoded-payload GB/s per
stack. Logic SRAM also holds decoded payload and `hbf-logic-sram-bw` is its
per-stack bandwidth. These uplifts prevent parity or command bytes from
silently reducing the configured decoded HBIO target.

`hbf-ecc-decode-latency-ns`, `hbf-ecc-encode-latency-ns`,
`hbf-ecc-decode-raw-bw`, `hbf-ecc-encode-raw-bw`

ECC response latency and raw codeword throughput per die. Raw bandwidth moves
data + `hbf-oob-bytes`, so `II = (page + OOB) / raw_bandwidth`. II is the
exclusive issue-port occupancy; response latency is measured independently
from issue start. Decode and encode share one conservative issue port per die,
but each response-latency window may overlap later codewords. Latency must be
at least the codeword II. The use-case profiles use 105.46875 GB/s
per die (4320 B every 40.96 ns; 100 GB/s payload/die, 1.6 TB/s across 16 dies).
The 8-die server profiles use 135 GB/s per die (32 ns/codeword; 1024 GB/s
payload/stack). These are interface-provisioning assumptions, not published
ECC measurements. The cost-unbounded output upper bound instead uses
552.96 GB/s per die so sixteen assumed die-local issue ports supply its
8847.36 GB/s raw stack boundary.

`hbf-ctrl-dram-bytes`, `hbf-ctrl-dram-latency-ns`,
`hbf-ctrl-dram-issue-ns`

Every HBF stack keeps its complete page-level L2P table resident in its local
controller DRAM. A zero or absent byte budget derives the exact full-table
footprint; an explicit undersized budget is rejected. Accesses share one
pipelined port per stack: `issue-ns` is exclusive occupancy (default 1 ns) and
`latency-ns` is the overlappable response latency (default 100 ns). Reads never
fetch mapping pages from HBF media. Dirty mapping groups are checkpointed to
HBF during drain; checkpoint traffic is not on the foreground read path.

`hbf-write-coalescing`, `hbf-write-buffer-completion-requires-flush`,
`hbf-write-buffer-pages`, `hbf-write-buffer-flush-threshold-pages`

Write buffer and coalescing behavior for mutable HBF-backed traffic. The
physical program unit is the page itself (HBF's tiled array: wordline ==
page + OOB), so every page program costs a full `hbf-program-ns` + verify.

`hbf-oob-bytes`

Out-of-band spare bytes per page (ECC parity, reverse LPN mapping, block
status). Shares the page's wordline — array timings unchanged — but every
flash-side media-lane, page-buffer, channel, TSV, and ECC data transfer moves
page + OOB bytes. The TSV additionally serializes internal commands on its
shared timeline. Decoded SRAM and external HBIO move payload only. 224 B per
4 KiB page in the shipped configs.

The path contract follows those units. An external read uses one HBIO request
command, an internal TSV/channel command, then raw
`array -> media lane -> page buffer -> channel -> TSV -> ECC decode` before a
full decoded page enters SRAM and the exact requested payload leaves over
HBIO. An external write moves only the requested payload fragment over HBIO
into SRAM; after merge or erased-value fill, the internal program command
crosses TSV/channel and a full decoded page follows `SRAM -> ECC encode -> raw
TSV -> raw channel -> die sequencer -> array program`. Mapping, GC, RMW, and
write-buffer destage use the internal path without a second external HBIO
payload transfer. An unmapped logical read returns NAND erased-value bytes
through SRAM/HBIO without a fabricated flash read or ECC operation.

`hbf-gc-low-watermark-pages`, `hbf-gc-hard-watermark-pages`,
`hbf-gc-reserved-free-blocks-per-plane`, `hbf-gc-wear-leveling-weight`

GC trigger thresholds, reserved free space, and wear-leveling weight.

`flat-hbm-bytes`

Address split for `HBM-HBF-Flat`: addresses below this value go to HBM and the
rest go to HBF. The boundary must be HBF-page aligned. A parent request that
straddles it is split across both media and retains one user-level latency.

`layer-buffer-bytes`, `base-die-link-read-bw`, `base-die-link-write-bw`,
`base-die-link-latency-ns`

Production EC6 combines these device configs with the overlay emitted by
`tools/prepare_frontier_residency_config.py`. That overlay reserves the
complete exported runtime metadata, block table, planned hot-KV prefix, and two
active buffers. It also supplies the exact trace path, byte count, and SHA-256;
`scenario_compare` rejects a different or subsequently modified trace and
forbids `max-ops` truncation. Immutable weights and cold KV use backing. The
selected trace cannot shrink capacity by omitting untouched objects.

`layer-buffer-bytes` is an upper bound for one ping-pong buffer. With an
explicit contract, the controller reserves the plan size and verifies the
trace-visible maximum fits. The runtime HBM constraint is:

```text
hbm_only_pages + hot_kv_pages + 2 * effective_layer_buffer_pages
    + unused_hbm_pages == hbm_capacity_pages
max_layer_data_pages <= effective_layer_buffer_pages
```

A trace may attach nondecreasing `layer=N` metadata to every request. EC6
streams each layer's data pages into the parity-selected buffer,
executes all foreground accesses from HBM, prefetches the next layer while the
current layer runs, and writes dirty pages back through D2D at layer completion. A
plain R/W trace with no layer metadata is one streaming window. `phase=N` is a
separate complete-before-next dependency. Optional `compute_ns=N` provides an
explicit layer execution interval; without it no compute time is inferred.

The base-die link bandwidths are PER HBF STACK: every HBF stack has its own D2D
interface, and each transferred page uses the interface of the owning stack.
The buffer limit must be page aligned and no smaller than the selected maximum
per-layer data footprint. Scratch and metadata are compacted into HBM; a
workload whose HBM-only set plus two buffers exceeds HBM is rejected.
The use-case profiles currently use 2048R/512W GB/s per stack from a
literature-derived architectural interpretation; treat this as a sensitivity
parameter, not a measured HBF product value.

## Evidence Discipline

A config value being present does not mean it is publicly validated. Use
`configs/parameter-provenance.json` to distinguish vendor-published facts,
literature-derived choices, and exploratory assumptions. Results should carry
the full resolved config and trace digest; `scenario_compare` writes both into
summary JSON.
