# Obelisk

Cycle-level simulator for 3D-stacked heterogeneous memory (HBM + HBF) targeted
at LLM inference workloads. See `Obelisk_Design_Spec.md` for the full design.

## Build

```bash
git clone --recursive https://github.com/YOUR_ORG/obelisk.git
cd obelisk
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

## Run

```bash
./obelisk ../configs/echo.toml
./obelisk ../configs/hbf_gen1_sandisk.toml
./obelisk ../configs/hybrid_hbm_hbf.toml
```

The simulator writes a per-request CSV and a JSON stats file; both paths are
set in the TOML config.

## Test

```bash
ctest --output-on-failure
```

## Status

This is an M1 skeleton. The HBM model is a simple fixed-latency mock —
DRAMsim3 integration is a TODO (see `src/media/hbm/hbm_interface.cc`).
