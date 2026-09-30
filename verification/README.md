# Verification framework

> Status: Current
> Last reviewed: 2026-09-29

Verification is separate from production model code. It answers “does the implementation satisfy its declared contract?”;
it does not turn exploratory parameters into hardware calibration.

| Path | Responsibility |
| --- | --- |
| `core/` | Strict schemas, ledger comparison/reduction, analytical contracts, and certificate logic |
| `oracles/` | Independent HBM, HBF, hybrid, external, and placement reference models |
| `gates/` | Runnable checks used by CTest, CI, and release certification |
| `probes/` | Small C++ witnesses linked to the production physical library |
| `cases/` | Input-only canonical cases; expected ledgers are regenerated |

### File catalog

The filename states the mechanism; the parent directory states its role.

| File | Responsibility |
| --- | --- |
| `core/analytical.py` | Closed-form constants and tolerances shared by analytical checks and certificates |
| `core/certificate.py` | Strict certificate validation and safe summary attachment |
| `core/contracts.py` | Duplicate-key-safe JSON loading and common contract errors |
| `core/ledger_compare.py` | Typed event-by-event ledger comparison |
| `core/ledger_reduce.py` | Independent reconstruction of timing and accounting from ledgers |
| `oracles/hbm.py` | Synchronous aggregate HBM channel oracle with independent per-burst grouping |
| `oracles/hbf.py` | Independent HBF state, mapping, and media oracle |
| `oracles/hybrid.py` | Independent composition and split-request oracle |
| `oracles/external.py` | Independent external-backing directional-queue, channel-striping, and transport-segmentation oracle |
| `oracles/placement.py` | Independent behavior-only placement state machine |
| `gates/cases.py` | Execute every canonical case and compare production with its oracle |
| `gates/fuzz.py` | Deterministic property and metamorphic corpus |
| `gates/analytical.py` | Closed-form timing and overlap boundaries |
| `gates/placement.py` | Production-versus-oracle placement differential |
| `gates/physical.sh` | Aggregate focused physical-probe scenarios |
| `gates/components.py` | Hand-computed component input/output contracts |
| `gates/waf.py` | Write-amplification identities and GC regimes |
| `gates/hbf_public_spec.py` | Executable binding of disclosed FMS 2026 HBF endpoints |
| `gates/page_run.py` | Differential equivalence and wall-time gate for HBF sequential page-run execution |
| `gates/provenance.py` | Config parameter-to-source coverage |
| `gates/documentation.py` | All-document status/index/link checks plus live scenario, schema, topology, catalog, CLI-help, and README quick-start contracts |
| `gates/project_structure.py` | Directory ownership and command-line entrypoint smoke checks |
| `gates/mutation.py` | Evidence that selected production faults are detected |
| `gates/certificate.py` | Run certificate gates and issue a bound artifact |
| `probes/ledger.cpp` | Production event-ledger emitter used by independent oracles |
| `probes/physical.cpp` | Focused C++ timing and state invariant scenarios |

The main ledger chain is:

1. `ledger_probe` executes a canonical case through the physical core and
   reference-policy libraries, then
   emits typed JSONL events.
2. An oracle under `oracles/` independently derives the expected ledger.
3. `core/ledger_compare.py` reports the first differing event and field.
4. `core/ledger_reduce.py` reconstructs completions, physical
   bytes, WAF, external range/segment/page counts, directional queue routing,
   shared media-resource exclusion, and resource work from the event stream.

For external backing, `media_read_queues` and `media_write_queues` default to
one and are independently positive. Caller-visible ranges select a queue by
per-direction round robin; every transport segment of that range stays on the
selected queue while segment addresses stripe across the queue's
`media_channels`. Queue/channel timelines are shared between reads and writes
by `queue * media_channels + channel`. Directional aggregate media bandwidth
is divided across that direction's queue count and the per-queue channel
count. The `external.read-queue-round-robin` canonical case makes the contract
observable with two same-arrival ranges on distinct queues while proving that
neither range migrates between queues.

`gates/fuzz.py` adds deterministic generated cases and metamorphic checks.
`gates/analytical.py` checks closed-form HBM bus work, the HBF media/HBIO
crossover, and independent HBM/HBF overlap. `gates/placement.py` checks the
behavior-only placement path against an independent state machine.

Run the normal verification targets with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target verify_foundation
cmake --build build --target verify_components
cmake --build build --target verify_write_amplification
```

Or run all registered tests:

```bash
ctest --test-dir build --output-on-failure
```

Generated ledgers and minimized fuzz reproducers are build artifacts. Canonical
cases contain inputs only, so a model change must be reviewed at its first
divergent event rather than accepted by replacing a golden summary.

## Mutation adequacy

`verify_mutations` archives the committed tree into a temporary directory,
changes production C++ source, rebuilds the affected target, and requires the
assigned canonical, physical, or behavioral gate to fail:

```bash
cmake --build build --target verify_mutations
```

Compile failures and malformed mutation patches are invalid mutations, not
kills. The temporary tree is discarded; production has no mutation switch or
compatibility branch.

## Foundational certificate

After committing a candidate tree, issue a certificate in two explicit steps:

```bash
cmake --build build --target verify_mutations
cmake --build build --target issue_certificate
```

Certificate schema v8 binds:

- the clean source commit and tree;
- the exact `hbfsim-reference`, `ledger_probe` and `physical_probe`
  executables;
- every canonical case and verification source;
- the parameter-provenance registry;
- external-evidence scripts, manifests, fixtures, traces, and source pins;
- canonical, fuzz, behavioral, analytical, physical, component, mutation,
  CTest, and sanitizer results.

The certificate also records the sanitizer platform boundary. Linux requires
LeakSanitizer; Darwin records that LeakSanitizer is unavailable and carries
that limitation explicitly.

A direct simulation summary remains
`validation.status=exploratory_unattached`. A study that wants to publish a
certified result attaches a verified certificate with
`verification.core.certificate.attach_certificate_to_summary_file` after the
source tree, embedded build provenance and executable digest have been
checked; without it a summary stays labelled exploratory.

## External evidence and calibration

`evidence/external/collect.py` builds pinned, tracked-clean Ramulator2 and
MQSim trees and runs the declared six-case differential matrix.
`evidence/external/verify.py` binds source/dependency pins, build artifacts,
raw outputs, repository inputs, and independently checked HBFSim ledgers.
Missing evidence is `not_run` or `partial`; malformed, stale, or divergent
evidence fails closed. See the [external evidence
contract](../evidence/external/assets/README.md).

Without `HBFSIM_EXTERNAL_EVIDENCE_DIR`, a foundational certificate reports
L0–L2 as passing and L3 as `not_run`. A complete passing external bundle can
raise only the explicitly declared facets to L3.

The hardware measurements behind the calibrated backing overlays
([evidence catalog](../evidence/README.md)) are outside the foundational
certificate: they justify specific overlay values, not the engine.

Claim levels and limitations are defined in the [evidence
policy](../docs/reference/evidence-policy.md). The current mechanism-to-test
mapping is maintained in [verification coverage](../docs/verification/coverage.md).
