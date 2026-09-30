# Calibration equations and provenance

This note separates three things that are easy to conflate: a model family
borrowed from prior systems literature, a parameter estimator chosen for this
experiment, and a validation metric.  The DANA calibration equations are not a
verbatim model copied from one NVMe or offload paper.

## 1. Fixed cost plus byte service

For a transfer of `s` bytes in direction `d`, the component model is

```text
T_d(s) = t_issue + s / B_d + t_processing
```

This is the same model family as the Hockney alpha-beta communication model,
`T(n) = alpha + n / beta`, with a separate controller occupancy term.  LogP and
LogGP likewise distinguish latency, per-message overhead, message gap, and the
per-byte gap for long messages.  The references justify the decomposition, not
the numerical values used by HBFSim:

- R. W. Hockney, *The Communication Challenge for MPP: Intel Paragon and
  Meiko CS-2*, Parallel Computing 20(3), 1994,
  https://doi.org/10.1016/S0167-8191(06)80021-9
- D. Culler et al., *LogP: Towards a Realistic Model of Parallel
  Computation*, PPOPP 1993, https://doi.org/10.1145/155332.155333

The values are identified from our own DANA measurements.  In
`evidence/hardware/dana_a100_offload/calibrate.py::fit_host`:

```text
t_issue = median(T_wall/op for 4 KiB and 64 KiB pinned copies,
                 pooling D2H and H2D)

m_d = sum_s s * (T_wall,d(s) - t_issue) / sum_s s^2
B_d = 1 / m_d

t_processing = median(max(0,
    T_p50,d(s) - t_issue - s / B_d))
```

The expression for `m_d` is the closed-form zero-intercept least-squares
slope after subtracting the fixed issue estimate.  D2H and H2D are fitted
separately because CUDA copy direction is an observed asymmetry.  The small
copy median is a robust empirical controller-occupancy estimate; it is not
claimed to be a pure PCIe propagation delay.

The original `gpu-51` fit yielded `t_issue = 5100.44 ns`,
`t_processing = 28375.26 ns`, D2H `B = 25.5966 GB/s`, and H2D
`B = 23.0776 GB/s`.  These values remain a historical anchor.  The allocation
transfer protocol may select an independently refitted profile for the formal
multi-GPU hardware class, using fit windows only.

## 2. SSD effective parallelism

For each read/write direction, the component fit uses the measured 16 MiB QD1
and QD8 throughput anchors:

```text
r = max(1, BW_QD8 / BW_QD1)
q_eff = argmin_{q in {1,...,8}} |log(q / r)|

B_media = sqrt((q_eff * BW_QD1) * BW_QD8)

L_media = median(max(0,
    T_p50,QD1(s) - s / (B_media / q_eff)))
    for s in {4 KiB, 64 KiB}
```

`q_eff` is an empirical service-width identification rule.  Little's law,
`L = lambda W`, explains why outstanding work, service rate, and latency must
be treated jointly in a stable queue, but it does not imply our integer
selection formula: J. D. C. Little, *A Proof for the Queuing Formula:
L = lambda W*, Operations Research 9(3), 1961,
https://doi.org/10.1287/opre.9.3.383.

The geometric mean is the midpoint of the two bandwidth anchors in log space.
It gives equal multiplicative weight to the QD1-derived and QD8 observations.
This is a pre-registered engineering interpolation, not an NVMe law.  Its
legitimacy comes only from frozen holdout performance; if the holdout fails,
the model fails rather than the rule being retroactively changed.

The media request segment is not fitted from latency.  It is derived from the
mounted block stack's page-aligned transfer limit and was 128 KiB on the
measured DANA stack.  HBFSim performs this segmentation.  A workload generator
must emit logical object ranges and must not duplicate the 128 KiB split.

## 3. KV bytes per token

For a decoder-only transformer with grouped-query attention:

```text
KV_bytes/token/global =
    num_layers * 2(K,V) * num_kv_heads * head_dim * dtype_bytes
```

If KV heads are evenly sharded under tensor parallelism, per-rank bytes use the
rank-local KV-head count; with TP1, global and per-rank values are identical.
For the frozen dense Qwen3-8B workload:

```text
36 * 2 * 8 * 128 * 2 = 147456 bytes/token/rank
```

Those factors come from the frozen checkpoint/model manifest in
`experiment_contract.json`, not from a fitted timing result.  Prefix hashes
identify potential token-block reuse; the router, capacity, and eviction policy
decide actual hits.  A Qwen 16-token hash block and a Mooncake 512-token hash
block therefore cannot be interchanged.

## 4. CUDA overlap and the critical path

CUDA streams are in-order, while operations in different streams may overlap
only when resource and dependency conditions permit.  `cudaStreamWaitEvent`
creates an explicit cross-stream dependency.  This is why transfer durations
must not simply be added to compute time.  See the NVIDIA CUDA Programming
Guide, *Asynchronous Execution*:
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html

The v5 workload model therefore uses a resource-constrained dependency DAG.
For the same logical request DAG and compute model, offload stall is defined as

```text
stall = completion(normal media service)
      - completion(zero-time offload media service)
```

The counterfactual keeps arrivals, routing, logical operations, dependencies,
and compute service fixed.  It avoids double-counting overlapped transfers and
does not use observed target-condition completion timestamps as simulator
inputs.

## 5. Error calculation and holdout

For one positive measured cell:

```text
signed_error_percent = 100 * (predicted / measured - 1)
absolute_error_percent = abs(signed_error_percent)
```

We report the raw measured/predicted values, signed error, median absolute
error, tail/max error, repeat noise, and confidence intervals.  Percentage
errors are scale-free but can be unstable when the denominator is near zero;
Hyndman and Koehler discuss this limitation in *Another Look at Measures of
Forecast Accuracy*, International Journal of Forecasting 22(4), 2006,
https://doi.org/10.1016/j.ijforecast.2006.03.001.  Our throughput and latency
cells are strictly positive, but aggregate averages still cannot hide a large
request- or cell-level error.

Parameter selection uses only declared fit windows.  Holdout and cross-source
records may score the frozen model but may not alter it.  Thus a good aggregate
E2E match with a failed request-level or phase-by-size gate is reported as
`workload-scoped aggregate closure`, never as request-level calibration.

