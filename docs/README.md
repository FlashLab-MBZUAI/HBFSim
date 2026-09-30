# Documentation

> Status: Current
> Last reviewed: 2026-09-29

Start at the top and stop when you have what you need. “Current” describes
implemented behavior; “Experimental” marks runnable material whose numbers are
not product-calibrated. Pages marked (中文) are written in Chinese.

## Start here

| Document | Status | Purpose |
| --- | --- | --- |
| [Project overview](../README.md) | Current | What HBFSim is, quick start, what you can explore |
| [Getting started](getting-started.md) | Current | From a fresh clone to your own parameter sweep |
| [Concepts](concepts.md) | Current | HBF in one page, the system model, how a run flows, glossary |
| [Examples](../examples/README.md) | Current | Six runnable scripts, one question each |
| [Design-space recipes](guides/design-space.md) | Current | Verified commands for common HBF design questions |
| [Extending HBFSim](guides/extending.md) | Current | Configurations, workloads, Python policies, reference policies, mechanisms |
| [中文说明](../README.zh-CN.md) | Current | 中文快速上手与导航 (中文) |

## Guides

| Document | Status | Purpose |
| --- | --- | --- |
| [Workload replay](guides/workload-replay.md) | Current | Build, normalize, replay, and inspect a workload trace with the reference runner |
| [ServeLoop](guides/serveloop.md) | Current | Driving the engine from the LLM-serving frontend |

## Reference

| Document | Status | Purpose |
| --- | --- | --- |
| [Physical model](reference/model.md) | Current | Implemented HBM, HBF, FTL, and hybrid mechanisms and their limits |
| [Host-managed HBF](reference/host-hbf-management.md) | Current | Zones, wear ownership, device DRAM and per-run wear reports |
| [HBF mapping organizations](reference/hbf-mapping-organizations.md) | Current | Mapping strategies, granularity, write-back, memory cost and how to run them (中文) |
| [Reference policy suite](reference/scenarios.md) | Current | Supported scenario IDs, time accounting and interpretation rules |
| [Observability](reference/observability.md) | Current | Trace schema, summary schema, wear reports and runtime guards |
| [Read-buffer causality](reference/read-buffer-causality.md) | Current | Fill handoff, future capacity pressure and the retained pinning-policy oracle |
| [Evidence policy](reference/evidence-policy.md) | Current | Claim levels, source provenance, and unresolved public-standard boundaries |
| [Configuration catalog](../configs/README.md) | Current | Every system profile, policy profile and overlay |
| [Verification coverage](verification/coverage.md) | Current | Executable checks mapped to invariants (中文) |
| [Project layout](project-layout.md) | Current | Where code, data, tests, and docs belong |

## Directory guides

| Document | Status | Purpose |
| --- | --- | --- |
| [Front door](../hbfsim/README.md) | Current | `python3 -m hbfsim` commands and the Python API |
| [Simulation client](../hbfsim_client/README.md) | Current | Semantic-free transaction protocol and persistent sessions |
| [Source catalog](../src/README.md) | Current | Production C++ ownership |
| [Build catalog](../cmake/README.md) | Current | CMake targets and test-registration ownership |
| [Test catalog](../tests/README.md) | Current | C++/Python test responsibilities and CTest grouping |
| [Workload catalog](../workloads/README.md) | Current | Synthetic trace generation, locality analysis and workload qualification |
| [Report catalog](../reports/README.md) | Current | Summary comparison and visualization entry points |
| [Verification framework](../verification/README.md) | Current | Oracles, gates, probes, certificates, and external evidence |
| [Evidence catalog](../evidence/README.md) | Current | Measurements and external-simulator pins behind calibrated numbers |
| [External differential evidence](../evidence/external/assets/README.md) | Experimental | Ramulator2/MQSim pins, cases, and validated facets |
| [DANA A100 offload calibration](../evidence/hardware/dana_a100_offload/README.md) | Experimental | Measured host-DRAM/NVMe timing behind the `backing/calibrated/dana-*` overlays |
| [H200 host-link measurement](../evidence/hardware/host_link/README.md) | Experimental | Measured GPU–host copy timing behind the `backing/calibrated/h200-*` overlay |

## Project and community

| Document | Status | Purpose |
| --- | --- | --- |
| [Contributing](../CONTRIBUTING.md) | Current | Ways to contribute, setup, and change requirements |
| [Code of Conduct](../CODE_OF_CONDUCT.md) | Current | Community standards (Contributor Covenant 2.1) |
| [Security policy](../SECURITY.md) | Current | Supported version, vulnerability reporting, and untrusted-input boundary |
| [Changelog](../CHANGELOG.md) | Current | Notable changes by release |

## Contract

The documentation gate (`documentation_contract` in CTest) requires every
maintained document to appear in this index, carry status and review
metadata, resolve its local links, avoid retired paths and identifiers, and
agree with executable scenario, schema and configuration contracts.
