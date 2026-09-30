# HBF decoded bank-cache causality

> Status: Current
> Last reviewed: 2026-09-11

OCP HBF v0.7.0 pages 56–57 require at least two decoded pages per bank. The
current device implements two, with LRU replacement within that bank. There is
no pooled per-stack cache size, independent-subarray mode or batch-sense switch.

`physical::hbf::HbfDevice` owns the cache. Each bank stores its resident LRU at
the causal frontier and ordered future fill, touch and invalidation events.
Lookup replays events only through the request's readiness time, so a future
decode cannot appear as an earlier hit. A future response that has already been
returned by the simulator reserves its data: an earlier backfilled fill is
bypassed if installing it would invalidate that response. Logical residency is
at most two pages at every time; pending event records are not extra cache slots.

Invalidation preserves prior event history until the frontier advances. Removing
an earlier fill outright could resurrect an LRU victim, so invalidations are
ordered after fenced consumers. Device reads also check the programmed-page
prefix and program completion; the cache cannot make erased media readable.

Reads that miss before decode completion issue real reads rather than claiming
an already-completed fill. The implementation does not promise minimum reads
through in-flight coalescing. Host write buffering and mapping caches are
separate Host DRAM policies and cannot expand decoded device residency.

Regression coverage includes a 20,000-event independent serial LRU oracle,
future-fill visibility, third-page eviction, cross-bank isolation, invalidation
without resurrection and preservation of future consumers. See
`tests/cpp/hbf_read_buffer_pressure_test.cpp`,
`tests/cpp/hbf_read_buffer_handoff_test.cpp` and
`tests/cpp/ocp_standard_test.cpp`.

Very large same-time open-loop batches retain their future events until a causal
watermark advances. This diagnostic/state cost is proportional to outstanding
work and must be measured separately from simulated hardware occupancy.
