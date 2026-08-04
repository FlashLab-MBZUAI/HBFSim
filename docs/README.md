# Documentation

This index points to the maintained HBFSim documentation. Historical design
plans are intentionally not retained once their contracts have moved into the
implementation, tests, and current reference documents.

## Model and validation

- [`model-validation.md`](model-validation.md): evidence levels, calibrated
  versus exploratory parameters, and the limits of current claims.
- [`physical-model-status.md`](physical-model-status.md): implemented HBM,
  HBF, FTL, and hybrid-memory mechanisms.
- [`verification-map.md`](verification-map.md): executable checks and the
  invariants they establish.
- [`physical-trace-and-guards.md`](physical-trace-and-guards.md): trace schema,
  provenance, and fail-closed runtime guards.
- [`../validation/README.md`](../validation/README.md): independent oracles,
  certificates, external differential checks, and calibration artifacts.

## Workloads and experiments

- [`real-workloads.md`](real-workloads.md): the production Frontier workload
  boundary and public H100/Llama 2 timing anchor.
- [`astra-workload.md`](astra-workload.md): the secondary ASTRA-sim trace
  frontend and replay contract.
- [`server-workload-runbook.md`](server-workload-runbook.md): build, normalize,
  replay, and inspect an external workload.
- [`scenario-comparison.md`](scenario-comparison.md): the supported memory
  hierarchy scenarios and interpretation rules.
- [`capacity-overflow-experiment.md`](capacity-overflow-experiment.md): the
  capacity-overflow study and evidence boundary.
- [`behavioral-placement-validation.md`](behavioral-placement-validation.md):
  behavior-only placement policies and their independent validation chain.
- [`kimi-k3-study.md`](kimi-k3-study.md): the declared next-scale Kimi K3
  capacity study.

## Integration

- [`../integrations/frontier/manifest.json`](../integrations/frontier/manifest.json):
  the pinned Frontier integration patch, upstream revision, and expected
  post-apply digests.
