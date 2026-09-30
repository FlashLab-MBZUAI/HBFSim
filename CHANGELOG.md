# Changelog

> Status: Current
> Last reviewed: 2026-09-29

Notable changes to HBFSim. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) once releases are tagged. Until
then, the default branch is the only supported version.

## [Unreleased]

### Removed

- The research scaffolding that only served the in-progress paper: the
  ServeLoop-driven experiment suites and their reports, the study runners,
  the paper insert, the study matrices under `configs/studies/`, the workload
  descriptors under `configs/workloads/`, the public-trace and production
  request-trace adapters, the runtime-KV critical-path adapter, the
  capacity-overflow executable and its certificate binding, and the tests,
  CMake targets and CTest registrations that ran them. The mapping-organization
  and host-zone feature tests keep their small configs under `tests/fixtures/`.
- Process documentation: dated study write-ups, review logs, experiment plans,
  figure designs and engineering logs under `docs/studies/`, `docs/reviews/`
  and the research-workflow, ASTRA, miniquick, SGLang, capacity-placement and
  simulator-performance guides.
- Evidence packages that backed no shipped configuration: the H100/Llama 2
  public anchor, the retired A100 real-inference package, the multi-GPU KV
  campaign, the GPU operator profiles and the public-trace statistics. The
  DANA A100 offload measurement and the H200 host-link measurement stay,
  because the `backing/calibrated/` overlays are fitted from them; the H200
  measurement is now registered as a provenance source.
- Study-specific overlays (`fixed-footprint-*`, `e2e-cxl-capacity-512g`,
  `timing-oracle-hbm-capacity-512g`, `dana-qwen3-30b-a3b-offload-capacity`,
  `endurance-dirty-eviction-scaled`, `mapping-read-isolation`, `persistence-*`,
  `q3-endurance-scaled`). The thermal and endurance-window overlays stay with
  study-neutral descriptions.

### Added

- The engine resolves `hbf-mapping-superblock-planes=all` from the HBF
  geometry, so the striped mapping overlays run through the front door and the
  reference runner without a helper script.
- `python3 -m hbfsim`, a dependency-free front door: `doctor`, `build`,
  `quickstart`, `list`, `run`, `sweep` and `show`, plus the `hbfsim` Python
  API (`run`, `sweep`, `load_summary`, `open_session`). Every run records the
  exact simulator command in its own output directory.
- `examples/`: six runnable scripts (placements, FTL mapping trade-offs, the
  read-bandwidth ceiling, custom traces, a custom Python policy, capacity-tier
  media), each registered as a CTest case.
- Onboarding documentation: getting started, concepts and glossary, extending
  guide, design-space recipes with verified commands, research workflows, and
  a Chinese README.
- Community and tooling files: Code of Conduct, CITATION.cff, issue and
  pull-request templates, CMake presets (including `werror`, which reproduces
  the CI compiler flags), a dev container, and `.editorconfig`.
- `published_config_scenario_matrix`: a CTest case that replays the smoke
  trace through every reference scenario on every published system/policy
  pair, so a profile that cannot run a scenario fails loudly.
- The CTest label `serveloop` on every test that needs the frontend checkout,
  and a `serveloop_checkout_compatible` fixture
  (`tests/python/check_serveloop_checkout.py`) that reports a missing or too
  old checkout in one line instead of twenty unrelated failures.

### Changed

- The LLM-serving frontend is ServeLoop: the renamed frontend repository
  lives at <https://github.com/FlashLab-MBZUAI/ServeLoop>; its Python
  package and console command stay `hbserve`. HBFSim-local names
  followed: the `SERVELOOP_ROOT` environment variable (was `HBSERVE_ROOT`),
  the `HBFSIM_SERVELOOP_SOURCE_DIR` CMake option (default `../ServeLoop`,
  was `HBFSIM_HBSERVE_SOURCE_DIR`), the `--serveloop-root` experiment option
  (was `--hbserve-root`), the minimum revision `HBFSIM_SERVELOOP_MIN_REVISION`
  (585c563, the compiler revision the current client protocol was written
  against), and the
  files `docs/guides/serveloop.md`, `tests/python/test_serveloop_frontend.py`,
  `tests/python/test_serveloop_persistence.py` and
  `experiments/hbf_persistence/serveloop_recovery.py`. Serialized receipt and
  manifest field names are unchanged.
- CI is organized by purpose: `quickstart` (the out-of-box path on Python
  3.10 without the frontend), `build-and-test` (GCC/Clang, Debug/Release,
  `-Werror`, tests not labelled `serveloop`), `frontend-integration` (ServeLoop
  pinned to 585c563; red until that revision is on ServeLoop's public `main`),
  `deep-verification` (component and write-amplification gates) and
  `sanitizers`; the mutation gate runs weekly and on demand (about 38 minutes).
  Steps that duplicated CTest cases (the provenance gate, `verify_foundation`)
  were dropped.
- Compiler warning options live in one place (`HBFSIM_WARNING_OPTIONS` in
  `cmake/Targets.cmake`) and apply to every target; the designated
  initializers that deliberately omit value-initialized members no longer
  trigger `-Wmissing-field-initializers`.
- The independent oracles follow the documented host mapping ledger
  (`mapping_lookup_compute` / `mapping_update_compute` on a per-stack compute
  pool, one GC mapping publication per relocated page) and the HBM
  service-group calendar; the ledger reducer resolves HBM at service-group
  granularity and bounds per-lane counters. 27 of 35 canonical cases had been
  failing silently because the cases gate stops at the first failure.
- The reference-policy profiles paired with the four topology systems place
  `flat-hbm-bytes` and `static-direct-hbm-bytes` at the cooperative write
  region's base (application HBM minus the write buffer, where application
  HBM excludes the controller-DRAM reservation), so `flat` and `direct-read`
  run on them.
- `core_system_config_*` validate profiles with `--describe-system`; the
  previous invocation read stdin to EOF and hung under harnesses that leave
  stdin open.
- `hbf_mapping_layout_control_contract`: the layout-matched control demanded
  identical array page reads from all three variants, but a bank's two decoded
  pages are served at the instant a request reaches the bank, and page-mapped
  reads reach it after per-stack mapping-DRAM issue serialization staggers
  them, so the `hot_cold` re-reads hit differently (they did at every revision
  since the study was added). The contract (schema v2) now requires identical
  demanded pages, identical media traffic between the bypass and the static
  extent, and bounds the array-read difference by the decoded-buffer hit delta.
- `hbf_persistence_long_context_contract`: the HBM-fronted cell relied on the
  address-only remapper filling KV read misses into HBM, which its default
  stopped doing on 2026-09-23. `HbmFrontedBackingRemapper` takes an explicit
  `hbf_kv_read_policy` (`direct_gpu_read_on_miss`, the unchanged default, or
  `read_allocate_hbm_cache`); the study declares the latter in its contract and
  fails closed unless every receipt reports it.
- README reorganized around users: quick start, questions to explore, how the
  simulator works, and a repository map by audience.
- Certificate schema v8 binds the `hbfsim-reference`, `ledger_probe` and
  `physical_probe` executables; the capacity-overflow binding is gone with the
  executable.
- The CTest group `Tools.cmake` (was `Studies.cmake`) holds the workload
  tools, summary observability and report tests; `workload_quality_contracts`
  (was `behavioral_workload_quality_contracts`) and the newly registered
  `simulation_transaction_completions_contract`.
- Repository URLs point at `FlashLab-MBZUAI/HBFSim`.
- Documentation index organized by audience; all maintained documents carry
  status metadata, and links into untracked local outputs were replaced with
  plain path references.

### Fixed

- Every target compiles with `-Wall -Wextra -Wpedantic -Werror` on GCC 13 and
  Clang 18, which CI and the mutation gate both require; the gate had been
  dying at compile time. Misleading indentation in the mixed-FTL endurance
  driver, a by-value structured binding in `gap_calendar_test`, references to
  members of temporaries in the HBF contract probes and a Release-only
  maybe-uninitialized report in `hbf_mapping_organization_test` were fixed.
- `experiments/hbf_endurance/native_zone_lifetime.cpp` failed to compile on
  Linux (`std::uint64_t` is `unsigned long` there), which stopped a default
  `cmake --build`.
- `parameter_provenance_coverage`: `configs/studies/paper/lifetime-system.cfg`
  set six keys twice (a base profile followed by its overlay section); the
  duplicates are removed with the resolved configuration unchanged, and six
  engine keys used by tracked overlays are now in the provenance registry.
- `system_config_contracts` and `reference_config_roundtrip` iterate the
  published, policy-paired system profiles; `configs/systems/sglang-small.cfg`
  is a ServeLoop overlay fragment and is pinned as the only unpaired file.
- `summary_time_breakdown_contract` expected HBM refresh, precharge and
  activation columns that the channel-aggregate HBM model replaced.
- The canonical case `hbm.dual-pch-overlap` targeted one lane twice; it now
  exercises two lanes of one service group.
- The simulation-client README example reused a retained transaction id and
  was rejected by the engine.
- The `documentation_contract` and `project_structure_contract` tests pass
  again: missing status metadata and index entries, links to untracked `out/`
  results or removed files, coverage rows citing retired tests, uncataloged
  modules and configs, and two evidence packages without `__init__.py`. The
  structure gate now skips `--help` checks that need a declared-optional
  dependency (the `analysis` extra or ServeLoop) when it is not installed.
- `CONTRIBUTING.md` named a summary metric that no longer exists
  (`user_ack_throughput_GBps`; the field is `user_completion_throughput_GBps`).
