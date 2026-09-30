# Build layout

> Status: Current
> Last reviewed: 2026-09-29

| File | Responsibility |
| --- | --- |
| `Targets.cmake` | Production libraries, applications, probes, and explicit verification targets |
| `TestTargets.cmake` | Small compiled test executables and shared test output setup |
| `tests/Foundation.cmake` | Verification-framework and repository-contract tests |
| `tests/Devices.cmake` | Physical-device, session, policy, and component tests |
| `tests/Simulation.cmake` | Core engine and reference-runner CLI tests |
| `tests/Tools.cmake` | Workload tools, summary observability, report renderers and the ServeLoop frontend |
| `tests/Examples.cmake` | The `hbfsim` front-door contract and one test per script in `examples/` |

The root `CMakeLists.txt` owns only project-wide configuration and includes
these responsibility-based files. `CMakePresets.json` at the repository root
names the common configurations (`release`, `debug`, `sanitize`) and a
`simulator` build preset that compiles only `hbfsim` and `hbfsim-reference`.

<details>
<summary>File index additions</summary>

- [BuildProvenance.cmake](BuildProvenance.cmake)

</details>
