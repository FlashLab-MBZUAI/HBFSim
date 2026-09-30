#pragma once

#include "physical/external/external_backing_device.hpp"
#include "host/hbf_controller.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "policies/policy_common.hpp"
#include "physical/physical_types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace hbfsim::policy {

namespace external = physical::external;

// Byte-exact Frontier/object-map placement contract. Capacity allocation is
// determined by the complete exported object population, not by the subset of
// pages touched by the selected trace window. Physical page counts are still
// reported separately so object/page rounding remains auditable.
struct LayerResidencyContract {
    std::uint64_t page_size_bytes = 0;
    std::uint64_t unique_resident_footprint_bytes = 0;
    std::uint64_t immutable_weight_bytes = 0;
    // Sum of ceil(object_bytes / backing_page_size) across unique immutable
    // weight objects. This cannot in general be recovered from the aggregate
    // byte count because every exported object begins on a page boundary.
    std::uint64_t immutable_weight_pages = 0;
    // The placement compiler consumes otherwise unused HBM with a
    // deterministic logical-page prefix of immutable weights. The remainder
    // is the only weight population owned by the backing tier.
    std::uint64_t static_weight_resident_pages = 0;
    std::uint64_t runtime_overhead_bytes = 0;
    std::uint64_t block_table_bytes = 0;
    std::uint64_t active_buffer_bytes_per_slot = 0;
    std::uint64_t kv_region_begin = 0;
    std::uint64_t kv_block_stride_bytes = 0;
    std::uint64_t logical_kv_blocks = 0;
    // The canonical placement pins the lowest physical KV block IDs. This is
    // a placement decision from the exported plan, not a reuse oracle inferred
    // from future trace accesses.
    std::uint64_t hot_kv_blocks = 0;
};

// One hybrid-residency engine drives both comparison paths. Scratch/metadata
// and a bounded, causally selected hot-KV set are compactly resident in HBM.
// Read-only weights that do not fit the capacity-aware static HBM prefix plus
// cold/overflow KV use either HBF or external backing and two dynamically
// sized HBM ping-pong buffers. Every foreground request executes from HBM;
// only the backing device and its interconnect differ.
struct LayerStreamingConfig {
    hbm::HbmConfig hbm;
    BackingTier backing = BackingTier::Hbf;
    host::HbfConfig hbf;
    external::ExternalBackingConfig external_backing;
    BaseDieLinkConfig base_die_link;
    // Maximum capacity of ONE ping-pong buffer. Runtime sizes both buffers to
    // the actual maximum per-layer data footprint, bounded by this value.
    std::uint64_t layer_buffer_bytes = 32ull * 1024ull * 1024ull * 1024ull;
    // Production Frontier runs must provide this contract. The trace-derived
    // fallback remains only for generic microbenchmarks that have no exported
    // object population.
    std::optional<LayerResidencyContract> residency_contract;
    // Completion-order credit limit applied independently to each active
    // physical tier: foreground HBM page transactions use one pool, while
    // backing-tier DMA reads and writebacks share another. Parent trace
    // records and layer transfers are split before admission. 0 is open loop.
    std::size_t max_outstanding_requests = 0;
    std::size_t address_heatmap_bins = 0;
    TraceConfig trace;
};

struct LayerStreamingStats {
    BackingTier backing = BackingTier::Hbf;
    // The logical address span is diagnostic only. Capacity pressure is the
    // byte-exact exported unique resident footprint divided by physical HBM
    // capacity when an explicit contract is present. Generic microbenchmarks
    // fall back to the page-rounded union of trace-visible objects.
    bool compact_resident_mapping = false;
    bool semantic_inputs_consumed = false;
    bool explicit_residency_contract = false;
    std::uint64_t address_footprint_bytes = 0;
    // Page-allocated footprint includes per-object page rounding. The
    // capacity-pressure basis is byte-exact and excludes address alignment
    // holes; their difference is reported rather than silently conflated.
    std::uint64_t unique_resident_footprint_pages = 0;
    std::uint64_t unique_resident_footprint_bytes = 0;
    std::uint64_t capacity_pressure_basis_bytes = 0;
    std::uint64_t footprint_page_rounding_bytes = 0;
    std::uint64_t hbm_capacity_bytes = 0;
    double hbm_capacity_pressure = 0.0;
    std::uint64_t layers = 0;
    std::uint64_t explicit_layer_requests = 0;
    std::uint64_t explicit_compute_layers = 0;
    double compute_work_ns = 0.0;
    // Scratch/metadata are mandatory HBM residents. The hot-KV candidates are
    // pages whose first semantic touch is Frontier shared_context; the first
    // candidates that fit after reserving two complete active-layer buffers
    // are pinned for the run. This first-touch policy does not use future reuse
    // frequency.
    std::uint64_t hbm_only_resident_pages = 0;
    std::uint64_t hbm_only_resident_bytes = 0;
    std::uint64_t hot_kv_candidate_pages = 0;
    std::uint64_t hot_kv_candidate_bytes = 0;
    std::uint64_t hot_kv_resident_pages = 0;
    std::uint64_t hot_kv_resident_bytes = 0;
    // data_pages is the unique non-HBM-only footprint. It is partitioned into
    // static weights and hot KV resident in HBM plus the three disjoint
    // backing categories below.
    std::uint64_t data_pages = 0;
    std::uint64_t data_bytes = 0;
    std::uint64_t model_weight_resident_pages = 0;
    std::uint64_t model_weight_resident_bytes = 0;
    std::uint64_t model_weight_backing_pages = 0;
    std::uint64_t model_weight_backing_bytes = 0;
    std::uint64_t cold_kv_backing_pages = 0;
    std::uint64_t cold_kv_backing_bytes = 0;
    std::uint64_t unknown_backing_pages = 0;
    std::uint64_t unknown_backing_bytes = 0;
    std::uint64_t backing_unique_pages = 0;
    std::uint64_t backing_unique_bytes = 0;
    std::uint64_t resident_physical_pages = 0;
    std::uint64_t resident_physical_bytes = 0;
    std::uint64_t effective_layer_buffer_pages = 0;
    std::uint64_t effective_layer_buffer_bytes = 0;
    std::uint64_t unused_hbm_pages = 0;
    std::uint64_t unused_hbm_bytes = 0;
    std::uint64_t streamed_pages = 0;
    std::uint64_t streamed_bytes = 0;
    std::uint64_t foreground_resident_page_accesses = 0;
    std::uint64_t foreground_buffer_page_accesses = 0;
    std::uint64_t dirty_pages_written_back = 0;
    std::uint64_t writeback_bytes = 0;
    std::uint64_t max_layer_data_pages = 0;
    std::uint64_t max_layer_data_bytes = 0;
    // Byte-exact fields copied from the independently validated placement
    // contract. Zero for generic trace-derived microbenchmarks.
    std::uint64_t immutable_weight_logical_bytes = 0;
    std::uint64_t runtime_overhead_logical_bytes = 0;
    std::uint64_t block_table_logical_bytes = 0;
    std::uint64_t active_buffer_logical_bytes_per_slot = 0;
    std::uint64_t residency_page_size_bytes = 0;
    std::uint64_t kv_block_stride_bytes = 0;
    std::uint64_t logical_kv_blocks = 0;
    std::uint64_t hot_kv_blocks = 0;
    std::uint64_t cold_kv_blocks = 0;
    // Backing-tier DMA reads and writebacks share this completion-order
    // credit pool. These counters make the admission policy auditable rather
    // than inferring it from device utilization.
    std::uint64_t backing_request_credit_limit = 0;
    std::uint64_t backing_max_inflight_requests = 0;
    std::uint64_t backing_admission_waited_requests = 0;
    double backing_admission_wait_work_ns = 0.0;
    double backing_admission_max_wait_ns = 0.0;
    std::uint64_t user_waited_ops = 0;
    // Work sum from admission until the layer becomes executable. This is
    // distinct from device/link stage work and can overlap across requests.
    double user_wait_work_ns = 0.0;
    double user_max_wait_ns = 0.0;
    // Layer-level exposed/hidden prefetch wall intervals. Hidden intervals are
    // intersections with execution of the preceding layer.
    double exposed_prefetch_ns = 0.0;
    double hidden_prefetch_ns = 0.0;
    double buffer_reuse_wait_work_ns = 0.0;
};

struct LayerStreamingRunResult {
    std::vector<double> service_latencies_ns;
    std::vector<double> offered_latencies_ns;
    std::vector<double> source_latencies_ns;
    std::vector<PhysicalCompletion> completions;
    std::vector<std::string> warnings;

    std::uint64_t ops = 0;
    std::uint64_t reads = 0;
    std::uint64_t writes = 0;
    std::uint64_t logical_bytes = 0;
    // Every user page transaction executes from HBM. These counters are
    // physical page transactions rather than parent trace-record counts.
    // Backing tiers carry only layer DMA and writeback traffic.
    std::uint64_t hbm_user_accesses = 0;
    std::uint64_t hbf_user_accesses = 0;
    std::uint64_t hbm_background_accesses = 0;
    std::uint64_t hbf_background_accesses = 0;
    std::uint64_t external_background_accesses = 0;
    std::uint64_t background_hbf_writes = 0;
    // Initial immutable backing is addressed as a pre-resolved physical
    // extent. A page read after a streaming writeback uses the logical FTL
    // path.
    std::uint64_t hbf_static_read_bytes = 0;
    std::uint64_t hbf_logical_read_bytes = 0;
    std::uint64_t hbf_backing_write_bytes = 0;
    std::uint64_t external_backing_read_bytes = 0;
    std::uint64_t external_backing_write_bytes = 0;
    std::uint64_t hbm_foreground_bytes = 0;
    std::uint64_t hbm_streaming_write_bytes = 0;
    std::uint64_t front_end_admission_waited_ops = 0;
    double front_end_admission_wait_work_ns = 0.0;
    double front_end_admission_max_wait_ns = 0.0;
    double first_offered_arrival_ns =
        std::numeric_limits<double>::infinity();
    double last_offered_arrival_ns = 0.0;
    std::uint64_t phase_barriers = 0;
    std::uint64_t phase_dependency_waited_ops = 0;
    double phase_dependency_wait_work_ns = 0.0;
    double phase_dependency_max_wait_ns = 0.0;
    double user_finish_ns = 0.0;
    double finish_ns = 0.0;

    hbm::HbmStats hbm_stats;
    host::HbfStats hbf_stats;
    external::ExternalBackingStats external_backing_stats;
    LayerStreamingStats streaming_stats;
    BaseDieLinkStats base_die_link_stats;
    std::string hbf_wear_snapshot;
    std::optional<AddressHeatmapSnapshot> address_heatmap;
};

class LayerStreamingPolicy {
public:
    explicit LayerStreamingPolicy(LayerStreamingConfig config);

    [[nodiscard]] LayerStreamingRunResult run(
        const std::vector<MemoryRequest>& requests);

private:
    class RunEngine;

    LayerStreamingConfig config_;
    std::optional<AddressHeatmap> address_heatmap_;
    hbm::HbmDevice hbm_;
    std::optional<host::HbfController> hbf_;
    std::optional<external::ExternalBackingDevice> external_;
    std::vector<BaseDieLink> base_die_links_;
    std::uint64_t hbm_streaming_base_addr_ = 0;
    bool has_run_ = false;

    [[nodiscard]] std::uint64_t backing_page_size() const;
    [[nodiscard]] host::HbfController& hbf_device();
    [[nodiscard]] const host::HbfController& hbf_device() const;
    [[nodiscard]] external::ExternalBackingDevice& external_device();
    [[nodiscard]] BaseDieLink& base_die_link_for_stack(std::size_t stack);
    [[nodiscard]] BaseDieLinkStats aggregate_base_die_link_stats() const;
    void absorb_completion(
        LayerStreamingRunResult& result,
        PhysicalCompletion completion);
    void absorb_link_completion(
        LayerStreamingRunResult& result,
        PhysicalCompletion completion);
};

} // namespace hbfsim::policy
