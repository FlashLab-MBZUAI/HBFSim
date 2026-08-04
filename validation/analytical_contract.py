"""Single source of truth for the analytical foundational gate."""

from __future__ import annotations


REPORT_SCHEMA = {
    "name": "hbfsim.validation.analytical-microbench",
    "version": 1,
}

PAGE_BYTES = 4096
COMMAND_BYTES = 64
HBM_READ_BYTES = 1024**2
SMALL_PAGES = 64
LARGE_PAGES = 128
READ_NS_AXIS = (64.0, 128.0, 256.0, 512.0, 1024.0)
HBIO_GBPS_AXIS = (4.0, 8.0, 16.0, 32.0, 64.0)

FAST_GBPS = 1_000_000_000.0
FAST_NS = 0.001
SERIAL_HBF_CAPACITY = 4 * 1024**2
SERIAL_HBF_CONTROLLER_DRAM = 8192
OVERLAP_BYTES_PER_TIER = 1024**2
OVERLAP_FLAT_BOUNDARY = 8 * 1024**3

ABS_TOLERANCE_NS = 1e-7
REL_TOLERANCE = 1e-12

HBM_FORMULA = (
    "data_bus_busy_ns = read_bytes / pseudo_channel_bandwidth_GBps"
)
PHASE_FORMULA = (
    "steady_state_ii_ns = max(read_ns, page_bytes / hbio_GBps)"
)
OVERLAP_FORMULA = (
    "mixed_makespan_ns = max(hbm_only_ns, hbf_only_ns)"
)
CONTROLLED_BOUNDARY = {
    "stacks": 1,
    "channels_per_stack": 1,
    "dies_per_channel": 1,
    "planes_per_die": 1,
    "subarrays_per_plane": 1,
    "media_lanes_per_plane": 1,
    "page_buffer_banks_per_plane": 1,
    "page_bytes": PAGE_BYTES,
    "command_bytes": COMMAND_BYTES,
    "batch_activation": False,
    "read_buffer_pages": 0,
    "all_non_media_non_hbio_bandwidths_GBps": FAST_GBPS,
    "small_pages": SMALL_PAGES,
    "large_pages": LARGE_PAGES,
}

CLAIMS = (
    (
        "HBM data-bus busy work exactly follows transferred bytes divided by "
        "per-pseudo-channel bandwidth."
    ),
    (
        "At the declared serial HBF boundary, steady-state read initiation "
        "interval exactly follows the larger of media read time and HBIO "
        "payload serialization time."
    ),
    (
        "At the Direct FLAT boundary, independent HBM and HBF work executes "
        "concurrently: mixed makespan is the maximum of isolated tier "
        "makespans and both resource-work ledgers are preserved."
    ),
)

LIMITATIONS = (
    (
        "The HBF phase grid is an intentionally isolated single-stack, "
        "single-channel, single-die, single-plane, single-subarray read "
        "pipeline."
    ),
    (
        "The grid disables batch activation and read buffering and makes "
        "every stage except media and HBIO non-binding."
    ),
    (
        "The overlap case validates independent Direct-tier execution, not a "
        "placement policy, shared-link design, write path, GC path, or "
        "application workload."
    ),
    (
        "These model-relative identities do not calibrate HBF parameters or "
        "validate hardware performance."
    ),
)

CERTIFICATE_CLAIM = (
    "Closed-form analytical microbenchmarks passed exact HBM work "
    "accounting, the isolated HBF media/HBIO phase boundary, and independent "
    "HBM/HBF overlap."
)
