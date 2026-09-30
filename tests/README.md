# Test layout

> Status: Current
> Last reviewed: 2026-09-29

`cpp/` contains focused physical, session and policy tests linked to the
corresponding production library. `python/` contains schema, workflow, report
and end-to-end regressions, plus `mapping_organizations.py`, the support
module behind the mapping-organization session test. `fixtures/` holds the
small configurations those tests load. Filenames describe the behavior under
test; the stable runnable identity is the descriptive CTest name registered
under `cmake/tests/`.

Tests that import `hbserve` carry the `serveloop` label and require the
`serveloop_checkout_compatible` fixture (`python/check_serveloop_checkout.py`):
with no compatible [ServeLoop](../docs/guides/serveloop.md) checkout they are
reported Not Run after one explanatory line. `ctest -LE serveloop` runs
everything else; CI checks ServeLoop out at the pinned revision in
`.github/workflows/ci.yml`.

| Test group | Responsibility |
| --- | --- |
| `cpp/*_device_test.cpp` | Focused physical-device contracts |
| `cpp/simulation_session_test.cpp` | Semantic-free transaction-DAG execution contract |
| `cpp/*_policy_test.cpp` | Reference placement-policy lifecycle contracts |
| `cpp/address_heatmap_test.cpp` | Five-domain traffic accounting |
| `cpp/closed_loop_window_test.cpp` | Completion-order credit-window behavior |
| `cpp/hbf_compact_mutable_test.cpp` | Compact mutable initial-image equivalence |
| `cpp/hbf_mapping_layout_test.cpp` | Bounded page/entry/extent locality, same-page fill coalescing and retention, coherent dirty merges and sibling checkpoint epochs, in-memory compressed promotion, zero-cost codec, sustained GC and persistent-image restoration with resumed writes |
| `cpp/system_config_test.cpp` | Engine-owned configuration keys, capacity derivation and fail-closed contracts |
| `python/test_simulation_session_*.py`, `test_simulation_transaction_completions.py` | The `hbfsim` batch protocol: frontier, checkpoints, error replies, terminal-failure evidence and completion export |
| `python/test_hbf_host_zones.py`, `test_hbf_mapping_organizations.py`, `test_hbf_logical_invalidation.py` | Host zones, every mapping organization with checkpoint restart, and logical invalidation through the native protocol |
| `python/test_hbf_live_heat_stream.py`, `test_host_dram_attachment.py` | Live-heat streaming and CPU DRAM / SSD attachment through the engine |
| `python/test_summary_*.py`, `test_time_breakdown.py` | Summary schema, scenario observability and canonical time accounting |
| `python/test_external_backing.py`, `test_external_evidence.py` | External-backing model contracts and pinned external-simulator evidence |
| `python/test_synthetic_workload.py`, `test_trace_analysis.py`, `test_trace_span_census.py`, `test_workload_quality.py` | Trace generation, locality census and workload qualification |
| `python/test_gc_waf_quick.py`, `test_waf_gate.py` | Physical-write and write-amplification accounting |
| `python/test_plot_*.py`, `test_address_heatmap_interop.py`, `test_*_report.py` | Report validation and rendering |
| `python/test_system_configs.py`, `test_reference_config_roundtrip.py`, `test_lazy_sequential_equivalence.py` | Published-config, resolved-config and large-span execution contracts |
| `python/test_reference_scenarios_on_published_configs.py`, `test_reference_policy_identity.py`, `test_reference_heatmap_disable.py`, `test_hbf_controller_dram_ratio.py` | Every reference scenario on every published system/policy pair, policy identities and runner options |
| `python/test_verification_*.py`, `test_provenance.py`, `test_build_provenance.py` | Certificate, verification-contract and provenance behavior |
| `python/test_hbfsim_frontdoor.py` | Front-door name resolution, sweeps, tables, error hints and pass-through to the simulator |
| `python/test_serveloop_frontend.py` | ServeLoop compilation, placement and closed-loop timing through the `hbfsim_client` protocol (`serveloop` label) |
| `python/check_serveloop_checkout.py` | The `serveloop_checkout_compatible` fixture |

| File | Scope |
| --- | --- |
| `Foundation.cmake` | schemas, oracles, fuzz, certificates, docs, and project layout |
| `Devices.cmake` | compiled physical components and component contracts |
| `Simulation.cmake` | CLI, config resolution, scenario execution, and rejection paths |
| `Tools.cmake` | workload tools, summary observability, report renderers and the ServeLoop frontend |
| `Examples.cmake` | the `hbfsim` front door and every runnable example |

The explicit deep gates are intentionally independent: component checks do
not invoke physical probes or WAF checks internally. This avoids duplicate
coverage and makes a failing responsibility visible by name.
