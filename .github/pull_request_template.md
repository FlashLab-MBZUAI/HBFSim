## What and why

<!-- What does this change, and which question or problem does it address? -->

## Kind of change

- [ ] Documentation, examples, or front door (`hbfsim/`) only
- [ ] Configuration profile or overlay
- [ ] Workload, study, or report tooling
- [ ] Reference policy
- [ ] Physical model (timing, placement, FTL, composition, or accounting)

## Checks

- [ ] `ctest --test-dir build --output-on-failure` passes (or the failing tests are listed below with the reason)
- [ ] New or changed behavior has a focused test or runnable example
- [ ] No generated output, machine-specific path, or private dependency was added

For physical-model changes, also:

- [ ] The mechanism and its evidence grade are documented; new parameters have entries in `configs/parameter-provenance.json`
- [ ] Invalid inputs fail closed
- [ ] `verify_physical`, `verify_components`, and `verify_write_amplification` pass
- [ ] Obsolete paths, flags, tests, and claims are removed in the same change

See CONTRIBUTING.md in the repository root for details.
