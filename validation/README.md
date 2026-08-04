# Foundational validation

This directory contains small, independently executable models used to check
HBFSim's production implementation. It is intentionally separate from EC
experiments and from the production model sources.

The current HBM, HBF read/write/FTL, and direct-composition path has five
parts:

- `validation_probe` executes each case through `hbfsim_physical` and emits a
  canonical JSONL event ledger. HBF drain records also contain a sorted,
  read-only FTL snapshot and an independently reproduced SHA-256 state digest.
- `hbm_oracle.py` and `hbf_oracle.py` independently implement
  tiny-geometry address, timing, and resource state machines.
  `hybrid_oracle.py` independently routes and splits parents across those two
  models, preserving separate tier timelines and aggregating completion with
  the maximum child finish. None imports production helpers.
- `run_validation.py` validates both ledgers and reports the first differing
  record and field.
- `ledger_reducer.py` independently rebuilds completions, HBM row counters,
  HBF page/WAF accounting, and exclusive resource busy work from typed ledger
  events, then rejects a summary that cannot be reproduced.
- `property_fuzz.py` generates a version-stable, deterministic corpus for all
  four models, checks exact oracle agreement plus metamorphic properties, and
  shrinks a failure while preserving its first-divergence signature.
- `run_analytical_microbench.py` runs production binaries at three deliberately
  isolated boundaries: exact HBM data-bus work, the HBF media/HBIO steady-state
  crossover, and concurrent Direct HBM/HBF tier execution. It derives the
  expected result from a closed form and checks the raw summary digests.

Run all implemented cases with:

```bash
cmake -S . -B build
cmake --build build --target foundational_validation
```

Or run the six independent CTest gates directly:

```bash
ctest --test-dir build --output-on-failure \
  -R '^foundational_(validation_contracts|independent_oracles|property_fuzz|external_contracts|behavioral_differential|analytical_microbench)$'
```

Generated actual/expected ledgers are build artifacts and are not committed.
Canonical cases contain inputs only; expected numerical results are regenerated
by the independent oracle on every run. A model change must therefore be
reviewed against the first divergent event, not accepted by updating a golden
summary file.

The fast property gate runs seeds 0 through 31 for each of External, HBM, HBF,
and Hybrid, with at most 50 requests per generated case. The generator uses the
repository's fixed SplitMix64 implementation rather than Python's
version-dependent random APIs. Reproduce one seed with:

```bash
python3 -B validation/property_fuzz.py \
  --probe build/validation_probe \
  --replay hbm:10 \
  --artifact-dir build/property-fuzz
```

On failure, the runner prints the model and seed and atomically writes both the
original manifest and a minimized manifest under the artifact directory. These
generated reproducers are diagnostic build artifacts and must not be committed
as canonical expectations until their underlying contract is independently
reviewed.

The deep mutation gate creates a temporary archive of the current `HEAD`,
changes production C++ source (never an expected ledger), rebuilds the real
probe or composition executable, and requires every critical defect to make
its assigned canonical, physical, or behavioral gate fail. The physical gate
includes an exact enqueue/pump differential for the diagnostics-free HBM
synchronous shortcut, its public ticket sequence, and the stripe-boundary
fallback:

```bash
cmake --build build --target foundational_mutation_validation
```

Its JSON report records the source tree, patch digest, assigned case, build
status, diagnostic, and kill rate. Compile failures and malformed patches are
reported as invalid mutations rather than counted as kills. The temporary
mutated source tree is discarded; no runtime compatibility switch or mutation
branch exists in production.

After committing a candidate tree, issue its foundational certificate in two
steps:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target foundational_mutation_validation
cmake --build build --target foundational_certificate
```

The certificate target is deliberately separate from the mutation target so a
stale report cannot be hidden by an implicit rerun. It rejects a report from
another commit/tree, a changed tracked file, a stale executable, a failed
canonical/property/physical/use-case/CTest gate, or an incomplete mutation
matrix. It also configures and runs a separate ASan+UBSan CTest build before
reporting L0 as passed. The generated
`build/foundational-validation-certificate.json` binds the source commit/tree,
the exact `scenario_compare`, `validation_probe`, `physical_probe`, and
`overflow_offload_experiment` executables, case corpus, validation
implementation, parameter registry, per-case ledgers, and every mutation
result. Certificate schema v6 also binds every tracked file under
`validation/external/` as one collection, and records the exact external
source trees, facet definitions, case census, compared metrics, and exclusions.
It additionally requires the complete analytical gate: all 25 declared phase
coordinates, their recomputed bottleneck labels and initiation intervals, the
HBM work identity, independent-tier overlap, report digest, and explicit
limitations.

Schema v6 also records the exact ASan/UBSan runtime environment and host
platform. Linux requires `detect_leaks=1`. Apple's Darwin ASan runtime does
not provide LeakSanitizer, so a Darwin certificate must record
`unsupported_on_darwin`, run address/undefined sanitizers with leak detection
disabled, and carry that limitation explicitly.

The complete behavior-only trust ladder, including workload qualification,
metamorphic relations, mutation adequacy, and the boundary between correctness
evidence and exploratory performance, is documented in
[`docs/behavioral-placement-validation.md`](../docs/behavioral-placement-validation.md).

Untracked material is counted but does not make a certificate dirty and is not
read into its digests. Tracked modifications always fail closed. A direct
`scenario_compare` summary is explicitly
`validation.status=exploratory_unattached`; only a certificate-aware runner
may replace that block after verifying the certificate, source tree, embedded
build provenance, and executable digest.

The debug/characterization runners accept the certificate explicitly:

```bash
python3 tools/replay_astra_trace.py \
  ... \
  --validation-certificate build/foundational-validation-certificate.json
python3 tools/run_synthetic_experiments.py \
  ... \
  --validation-certificate build/foundational-validation-certificate.json
```

Without the flag they remain usable for diagnosis, but both their summaries
and suite manifests say `exploratory_unattached`. The
`tools/run_paper_*.py` publication runners require the flag and have no
uncertified fallback.

The publication entry-point allowlist is intentionally exact. It currently
contains only `run_paper_capacity_overflow.py`, whose dedicated end-to-end
contract binds both publication binaries and validates the complete overflow
artifact.

An earlier capacity-shortfall runner, WAF/GC sweep, and write-path ablation
were removed because they had no dedicated experiment contract or workload
qualification. A foundational certificate alone does not make an arbitrary
parameter sweep publishable.

There is currently no Frontier publication runner. The former single-request
7B phase study was removed because it neither exercises production batching
nor creates genuine capacity pressure. The canonical 70B Qwen request suite
and its independent source-to-CSV verifier are now implemented, and the
Frontier auditor requires their digest-bound receipt. Frontier execution and
memory export remain structural CI instruments until the W8A16/BF16 model
contract, all three multi-request replays, hybrid residency, external
baselines, and final result verifier are implemented as one reviewed contract.

The capacity-overflow publication path additionally binds the artifact to the
exact overflow binary:

```bash
python3 tools/run_paper_capacity_overflow.py \
  --experiment build/overflow_offload_experiment \
  --scenario-compare build/scenario_compare \
  --validation-certificate \
    build/foundational-validation-certificate.json \
  --output-dir out/paper-capacity-overflow
```

The C++ binary itself always emits `exploratory_unattached`. The paper runner
first validates that artifact and its self-reported executable SHA-256, then
attaches the certificate and revalidates the certificate-bound output.

`run_actual_external_differential.py` rebuilds the pinned, tracked-clean
Ramulator2 and MQSim source trees without local patches and runs an exact
six-case matrix. The current L3 scope contains only
`dram.read-row-state-service` (three single-bank DDR4 read row-state cases) and
`flash.direct-data-page-io` (three full-page SLC data-I/O cases). The verifier
binds the source/dependency pins, build artifacts, current HBFSim probe,
repository inputs, raw outputs, and complete independently checked HBFSim
ledgers. Missing evidence is `not_run` or `partial`; malformed, stale, or
numerically divergent evidence fails closed. MQSim completion time and
GC/erase behavior, and Ramulator2 write/refresh/multi-bank behavior, are
explicitly outside these facets. See
[`external/README.md`](external/README.md) for the exact boundary and commands.

Without `HBFSIM_EXTERNAL_EVIDENCE_DIR`, a foundational certificate therefore
reports L0-L2 as `pass` and L3 as `not_run`. With a complete passing bundle it
reports the two facet IDs and six cases as L3 `pass`, while L4 calibration and
L5 held-out hardware validation remain `not_run`.

The public H100 + Llama 2 70B timing anchor under
[`calibration/`](calibration/README.md) is also intentionally outside the
foundational certificate. Its import and independent verification establish
the integrity and fit/held-out separation of Vidur and MLPerf source data. The
companion calibration protocol can additionally produce and independently
reconstruct a 49-model Vidur interpolation candidate with 20,952 held-out
operator/collective keys. This is a narrow operator-level calibration object,
not a fitted HBFSim/Frontier end-to-end predictor. Its MLPerf applicability
gate records the FP16/FP8, runtime, batch-domain, scheduler, host-overhead, and
generated-length gaps and emits no performance prediction. Consequently the
foundational certificate still reports project-level L4 and L5 as `not_run`;
the separate candidate state is
`operator_calibrated_external_validation_not_established`.

The analytical gate is mandatory in either case:

```bash
python3 -B validation/run_analytical_microbench.py \
  --scenario-compare build/scenario_compare \
  --config configs/scenario_compare/usecase-baseline.cfg \
  --repository . \
  --artifact-dir build/analytical-microbench \
  --report build/analytical-microbench-report.json
```

Its HBF grid is a one-resource-per-stage read-pipeline identity, not an
application phase diagram, a write/GC result, parameter calibration, or
hardware validation. Its FLAT case proves that the current model permits the
two independent tier timelines to overlap; it does not validate a placement
policy or a shared-link architecture.

The confidence levels and claim boundaries are maintained in
[`../docs/model-validation.md`](../docs/model-validation.md); the current
mechanism-to-evidence map is maintained in
[`../docs/verification-map.md`](../docs/verification-map.md).
