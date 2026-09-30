# Security Policy

> Status: Current
> Last reviewed: 2026-08-05

## Supported Version

Until the first tagged release, only the current default branch is supported.
HBFSim is a research simulator and must not be treated as a security boundary
or production service.

## Reporting a Vulnerability

Use the repository's private **Report a vulnerability** / Security Advisory
flow when it is available. Include the affected revision, platform, a minimal
reproducer, impact, and whether the input trace or config is untrusted. If
private reporting is unavailable, contact the maintainers through the contact
method published on the repository owner's GitHub profile; do not post exploit
details in a public issue before maintainers can assess them.

## Untrusted Inputs

Trace files, config files and output paths should be treated as untrusted data.
Run HBFSim with ordinary user privileges in an isolated working directory when
processing third-party inputs. The project does not claim sandboxing, and
external ASTRA/workload generators are outside its security boundary.
