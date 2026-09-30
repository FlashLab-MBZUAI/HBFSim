# Source layout

> Status: Current
> Last reviewed: 2026-09-29

Production C++ is split by runtime role.

| Path | Responsibility |
| --- | --- |
| `app/session_main.cpp`, `app/session_protocol.*`, `app/system_config.*` | Strict `hbfsim` engine CLI, transaction/checkpoint/explicit-crash protocol, and physical-only configuration |
| `app/main.cpp`, `app/reference_runner.*` | Separate `hbfsim-reference` entry point, reference workload/policy suite, and summary output |
| `app/sha256.hpp` | Small dependency-free digest helper used for artifact provenance |
| `physical/physical_types.hpp` | Shared request, completion, span, and address-space types |
| `physical/simulation_session.*` | Persistent arbitrary transaction-DAG executor with quiescent checkpoints and no-drain command-boundary crash injection; no workload or policy semantics |
| `physical/base_die_link.*` | Per-stack directional D2D timing and accounting |
| `physical/hbm/` | HBM address mapping, aggregate channel scheduling, shared buffer contention, and statistics |
| `host/` | Logical HBF controller, Host DRAM mapping/cache/coalescing, GC/WL ordinary copies, zones, and canonical v5 persistent images. Composes physical resource calendars and the OCP device contract. |
| `physical/hbf/` | OCP v0.7.0 device configuration and grades, channel command/RX/TX calendars, sequential program/auto-erase state, two decoded pages per Bank, and reusable resource calendars. |
| `physical/external/` | External backing controller, media, and link model |
| `physical/address_heatmap.*` | Cross-model address/traffic accounting |
| `policies/policy_common.*` | Contracts shared only by optional policies |
| `policies/reference/` | Direct, behavioral-tiering, and layer-streaming reference policies |

`hbfsim_core` contains `physical/` and its explicit `host/` management context. It must not import policies,
reports, tests, or verification helpers. Reference
policies depend on the core in one direction; the core never selects them.
