#pragma once

#include "physical/hybrid/composition_common.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace hbfsim::physical::hybrid {

// One engine executes every direct (non-staged) HBM/HBF composition.
// DirectPolicy contains only routing behavior; experiment identity belongs to
// the runner that combines a policy with a device configuration and workload.
// This struct is not a general policy language.
struct DirectPolicy {
    bool build_hbm = true;
    bool build_hbf = true;
    // Read fragments at or above this address go to HBF. The engine splits a
    // parent at page/placement boundaries before classification.
    std::uint64_t read_boundary = 0;
    // HBF-side reads: bijectively permuted physical pages on the fabric (no
    // FTL, paged per source page) versus logical whole-op reads through FTL.
    bool read_physical_direct = false;
    // Read-only-direct additionally keeps reads whose kind is not in
    // {model_weights, shared_context, unknown} on HBM. Kind is an optional
    // hint: unlabeled traces pass the filter unchanged.
    bool direct_read_kind_filter = false;
    enum class WriteRoute {
        AllHbm,
        AllHbfLogical,
        ReadBoundary,           // same address split as the read side (FLAT)
    };
    WriteRoute write_route = WriteRoute::AllHbm;
    enum class Prepopulate {
        None,
        InitialReads,
        InitialReadsAtOrAboveReadBoundary,
    };
    Prepopulate prepopulate = Prepopulate::None;
    bool drain_hbf = false;
    bool warn_if_one_sided = false;  // FLAT's both-sides check
    // Completion-id spellings, kept identical to the historical runners.
    std::string hbm_id_prefix;     // + "op{i}"
    std::string hbf_id_prefix;     // + "op{i}" (logical whole-op route)
    std::string direct_id_prefix;  // + "op{i}/page{p}"
    std::string drain_id;
};

[[nodiscard]] DirectPolicy all_hbm_policy();
[[nodiscard]] DirectPolicy all_hbf_policy();
[[nodiscard]] DirectPolicy flat_policy(std::uint64_t flat_hbm_bytes);
[[nodiscard]] DirectPolicy read_only_direct_policy(std::uint64_t direct_hbm_bytes);

struct DirectRunKnobs {
    enum class DestagePolicy {
        // Park destages and stream them out after the last user completion
        // (macro-scale yielding). The comparison-table default.
        Deferred,
        // Exploratory layer-wise write-back policy: a resident generation's
        // destage chain starts after its latest write lands; a rewrite before
        // that instant supersedes the old ready event. For ASTRA traces the
        // at= arrival structure IS the layer schedule, so the write stream
        // lands inside the producing layer's window over the dedicated D2D
        // write lane. Reserved for AI-workload studies; not run in the table.
        Streamed,
    };

    // Completion-order credit limit for HBF-page-sized foreground physical
    // transactions; 0 = unbounded. Parent trace records are split before
    // admission and aggregate their child completions into one user latency.
    std::size_t max_outstanding_requests = 0;
    // Read-only direct-composition sensitivity: independent completion-order
    // credit pools remove false cross-tier head-of-line blocking.  These are
    // mutually exclusive with max_outstanding_requests. A parent may span
    // tiers because each page transaction owns its tier credit independently.
    // Zero leaves that tier unbounded.
    std::size_t max_hbm_outstanding_requests = 0;
    std::size_t max_hbf_outstanding_requests = 0;
    // Cooperative HBM/HBF write-staging policy over the configured asymmetric
    // D2D: each touched logical HBF page owns one finite full-page physical
    // HBM slot while its initialized dirty byte ranges are resident. Rewrites
    // merge in that generation; destage transfers exactly their union through
    // HBM read -> per-stack D2D -> HBF logical write. Deferred work yields to
    // user traffic, while capacity pressure drains the oldest resident slot
    // and honestly throttles sustained overload. 0 = off: writes go straight
    // to the FTL (an HBM-less composition has nowhere to stage).
    std::uint64_t hbm_write_buffer_bytes = 0;
    DestagePolicy destage_policy = DestagePolicy::Deferred;
    BaseDieLinkConfig base_die_link;
    TraceConfig trace;
    // Completion objects are needed only for event-trace export. Accounting,
    // latency samples, device statistics, and heatmaps are retained either
    // way. The library default preserves the historical API behavior.
    bool retain_completions = true;
    // 0 disables collection for component probes. scenario_compare always
    // supplies a positive bounded value, so every reported run has a heatmap.
    std::size_t address_heatmap_bins = 0;
    // Optional immutable population description, independent of the executed
    // request prefix. This is used by workload-scaling studies so --max-ops
    // changes only executed work, not the initial HBF image or its physical
    // placement. The caller owns the vector for the duration of run().
    const std::vector<MemoryRequest>* initial_image_requests = nullptr;
};

// A file-free representation of exactly the same request stream as a trace
// containing request_count consecutive, fixed-size read records. It avoids
// materializing tens of millions of MemoryRequest/string objects while
// preserving request granularity, arrivals, window credits, and device work.
struct SequentialReadWorkload {
    std::uint64_t base_addr = 0;
    std::uint64_t request_bytes = 0;
    std::size_t request_count = 0;
    double first_arrival_ns = 0.0;
    double interarrival_ns = 0.0;
};

[[nodiscard]] CompositionRunResult run_direct_composition(
    const DirectPolicy& policy,
    const hbm::HbmConfig& hbm_config,
    const hbf::HbfConfig& hbf_config,
    const std::vector<MemoryRequest>& requests,
    const DirectRunKnobs& knobs);

[[nodiscard]] CompositionRunResult run_direct_composition(
    const DirectPolicy& policy,
    const hbm::HbmConfig& hbm_config,
    const hbf::HbfConfig& hbf_config,
    const SequentialReadWorkload& workload,
    const DirectRunKnobs& knobs);

} // namespace hbfsim::physical::hybrid
