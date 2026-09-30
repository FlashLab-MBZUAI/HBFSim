# Report catalog

> Status: Current
> Last reviewed: 2026-09-29

This directory contains generic summary comparison and visualization tools.
They consume validated summaries; they do not run or modify the physical model.

| File | Responsibility |
| --- | --- |
| `compare.py` | Compare two summary JSON files while excluding declared provenance drift |
| `address_heatmap.py` | Render one validated address-heatmap domain set |
| `time_breakdown.py` | Produce long-form timing CSV and Markdown |
| `time_dashboard.py` | Produce a self-contained interactive timing dashboard |

Every renderer validates its input schema and refuses ambiguous scenario
selection.

`python3 -m hbfsim show RUN --format markdown` covers the common case of a
comparison table; the renderers here are for timing dashboards, heatmaps and
long-form CSV. The plotting extras (`pip install -e ".[analysis]"`) are needed
only by the renderers that draw figures.

HBF host zone management and automatic per-run wear reports are documented in
[host HBF management](../docs/reference/host-hbf-management.md). Reports show
physical P/E history and workload increments separately; GC copies include
host roundtrip cost.

<details>
<summary>File index additions</summary>

- [hbf_wear.py](hbf_wear.py)

</details>
