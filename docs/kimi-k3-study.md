# Kimi K3 HBF study

This study asks whether HBF changes the deployment boundary of an extreme MoE
model. It does not start from the claim that Kimi K3 cannot fit in one machine:
AMD has published a functional TP8 path on one eight-MI355X platform. Its
loader census reports 190.974 GiB of weights per rank and approximately
205.401 GiB after the known state for one 1M-token sequence, before unmodeled
runtime overhead. The sharper question is whether HBF lets the same fixed TP8
compute domain use 192-GiB or 96-GiB HBM ranks, and whether the resulting
memory service is still useful.

The immutable question contract is
`configs/studies/kimi-k3-hbf-questions.json`; the source/accounting descriptor
is `configs/workloads/kimi-k3-tp8-capacity.json`.

## Three falsifiable questions

1. **Scale-in boundary.** At fixed TP8, where does HBM+HBF change K3 from
   capacity-infeasible to feasible, and where is it only capacity-feasible but
   bandwidth/latency-infeasible? The comparisons are all-HBM upper bound,
   HBM+HBF, HBM+CXL memory, and HBM+NVMe SSD under the same logical traffic.
2. **Expert delivery.** For 896 routed experts with top-16 selection, when is
   direct HBF access preferable to staging selected experts through HBM
   ping-pong buffers? A K3 conclusion requires token-and-layer router
   activations from the real model. Synthetic uniform, Zipf, or Markov routes
   are break-even sensitivities only.
3. **Mutable long-context state.** Can HBF beneficially hold any MLA/KDA,
   prefix, or eviction state at 1M context, or should it remain a read-only
   expert tier? This question must report logical and physical writes, the one
   canonical WAF (`physical_write_bytes/logical_write_bytes`), and physical
   writes divided by usable HBF capacity. A zero-write weight run has WAF
   `null`, not zero.

A negative answer is useful for every question. In particular, CXL or NVMe can
provide the same *capacity* if they can hold the backing bytes; HBF has to earn
its place through measured service behavior rather than capacity arithmetic.

The study pre-registers its answer rules in the question contract. Q1 never
turns a capacity fit into a deployment claim; it first reports modeled
offered-to-completion latency and makespan on byte-identical traffic. Q2 is a
Pareto comparison of direct access and HBM staging under the same real router
window, including their extra HBM and HBF traffic; a workload-dependent
crossover is a valid answer. Q3 must present latency and physical-write cost
together. Cross-window averages, a forced universal winner, and omitted write
cost are not accepted as answers.

This is a live disagreement rather than a manufactured HBF-only question.
FlashAccel argues that a six-HBF-stack design can organize both weights and KV
state efficiently, while the HotInfra analytical study argues that writable
on-package LPDDR matches or beats HBF across its tested points and that NAND
cannot absorb decode-time KV spill. The dynamic study therefore includes an
on-package-LPDDR sensitivity alongside CXL and NVMe. HBFSim must be able to
produce the HBF-loses region, not merely search for a winning HBF point; no
absolute LPDDR or external-tier claim is permitted before calibration.
The reproducible starting envelope is
`configs/scenario_compare/external-on-package-lpddr.overlay`; every resolved
field must remain visible in the run receipt and be swept before use.

## Capacity-only preflight

Run:

```bash
python3 tools/prepare_kimi_k3_capacity_preflight.py \
  --output-dir out/kimi-k3-capacity-v1
```

The preflight reproduces AMD's rounded decimal-GB TP8 category census. Its
conservative route-independent hybrid policy keeps all non-routed weights,
the vendor-known 1M state, a selectable 8/16/32-GiB runtime overhead, and two
complete routed-layer TP8 shard buffers in HBM. All 180.807 decimal GB of
routed-expert rank shards remain on the backing tier.

The derived base HBM requirement is 43,671,586,958 bytes (40.672 GiB) before
runtime overhead; the backing requirement is 180,807,000,000 bytes (168.390
GiB). Across 96/192/288-GiB HBM and 8/16/32-GiB overhead, physical all-HBM fits
only the three 288-GiB rows, while the conservative hybrid capacity layout fits
all nine. These are exact calculations over rounded vendor inputs, not
simulator performance results. Every row therefore emits
`memory_service_feasible=undetermined` and the receipt remains
`paper_result_eligible=false`.

## Dynamic evidence gates

Dynamic K3 results require all of the following:

- a scalable long-span execution path proven event/counter equivalent to the
  current exact 64-byte HBM and 4-KiB HBF paths on their common domain;
- real K3 router activations for expert-placement claims;
- byte-identical logical traffic with placement receipts for every tier;
- explicit dependency-driven/asap-bounded timing and reported credit/QD;
- calibrated HBF/external parameters for absolute device claims; and
- calibrated per-layer compute, TP8 collectives, and communication before any
  TTFT, TPOT, SLO, end-to-end latency, or token-throughput claim.

Until those gates close, HBFSim may report capacity, traffic, modeled
memory-system service, tier utilization, exposed/hidden prefetch, and physical
flash writes only.

## Primary context

- [Kimi K3 official repository](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi K3 technical report](https://arxiv.org/abs/2607.24653)
- [AMD TP8 loader and runtime-state accounting](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-k3-on-amd-instinct-gpus.html)
- [NVIDIA Dynamo Kimi K3 multi-node recipes](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3/README.md)
- [FlashAccel](https://arxiv.org/abs/2607.10186)
- [Is High-Bandwidth Flash All You Need?](https://hotinfra.org/2026/papers/hotinfra26-final83.pdf)
- [TileLens](https://arxiv.org/abs/2607.04031)
