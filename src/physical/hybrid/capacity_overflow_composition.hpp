#pragma once

#include "physical/external/external_backing_device.hpp"
#include "physical/hbf/hbf_device.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "physical/hybrid/composition_common.hpp"
#include "physical/physical_types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hbfsim::physical::hybrid {

enum class CapacityReadbackDestination {
    DmaBuffer,
    OriginalSlots,
};

[[nodiscard]] const char* to_string(CapacityReadbackDestination destination);

// A purpose-built capacity-pressure experiment:
//
// 1. Sequential full-page writes fill the HBM resident window.
// 2. Every later write evicts the oldest dirty page to the selected backing
//    tier before reusing that physical HBM slot.
// 3. The exact prefix offloaded in FIFO order is fetched in bounded batches
//    through a reserved HBM DMA buffer and then consumed by foreground HBM
//    reads.
//
// The resident window plus the DMA buffer exactly partition HBM capacity.
// This is a KV-growth/write-back experiment, not a weights-placement model.
struct CapacityOverflowConfig {
    hbm::HbmConfig hbm;
    BackingTier backing = BackingTier::Hbf;
    hbf::HbfConfig hbf;
    external::ExternalBackingConfig external_backing;
    BaseDieLinkConfig base_die_link;
    std::uint64_t read_buffer_bytes = 256ull * 1024ull;
    std::size_t transfer_batch_pages = 64;
    CapacityReadbackDestination readback_destination =
        CapacityReadbackDestination::DmaBuffer;
    std::size_t address_heatmap_bins = 256;
    TraceConfig trace;
    bool retain_completions = false;
};

struct CapacityOverflowWorkload {
    std::uint64_t base_addr = 0;
    std::uint64_t total_write_pages = 0;
    double first_arrival_ns = 0.0;
    double interarrival_ns = 0.0;
};

struct CapacityOverflowStats {
    BackingTier backing = BackingTier::Hbf;
    CapacityReadbackDestination readback_destination =
        CapacityReadbackDestination::DmaBuffer;
    std::uint64_t page_size_bytes = 0;
    std::uint64_t hbm_capacity_bytes = 0;
    std::uint64_t hbm_data_capacity_bytes = 0;
    std::uint64_t hbm_read_buffer_bytes = 0;
    std::uint64_t hbm_data_pages = 0;
    std::uint64_t hbm_read_buffer_pages = 0;
    std::uint64_t transfer_batch_pages = 0;

    std::uint64_t written_pages = 0;
    std::uint64_t offload_pages = 0;
    std::uint64_t readback_pages = 0;
    std::uint64_t offload_bytes = 0;
    std::uint64_t readback_bytes = 0;

    double first_arrival_ns = 0.0;
    double fill_finish_ns = 0.0;
    double offload_start_ns = 0.0;
    double write_finish_ns = 0.0;
    double read_start_ns = 0.0;
    double read_finish_ns = 0.0;
    double quiescent_finish_ns = 0.0;

    [[nodiscard]] double fill_elapsed_ns() const {
        return fill_finish_ns - first_arrival_ns;
    }
    [[nodiscard]] double offload_elapsed_ns() const {
        return write_finish_ns - offload_start_ns;
    }
    [[nodiscard]] double write_elapsed_ns() const {
        return write_finish_ns - first_arrival_ns;
    }
    [[nodiscard]] double read_elapsed_ns() const {
        return read_finish_ns - read_start_ns;
    }
    [[nodiscard]] double drain_tail_ns() const {
        return quiescent_finish_ns - read_finish_ns;
    }
};

struct CapacityOverflowRunResult {
    CapacityOverflowStats stats;
    std::vector<double> fill_write_latencies_ns;
    // Offered latency starts at the workload-visible demand arrival. Service
    // latency starts when the bounded transfer batch is admitted. Keeping the
    // two distributions distinct prevents controller-window queueing from
    // being mislabeled as device service time.
    std::vector<double> offload_offered_latencies_ns;
    std::vector<double> offload_service_latencies_ns;
    std::vector<double> readback_offered_latencies_ns;
    std::vector<double> readback_service_latencies_ns;
    std::vector<PhysicalCompletion> completions;

    std::uint64_t hbm_user_accesses = 0;
    std::uint64_t hbm_background_accesses = 0;
    std::uint64_t hbf_background_accesses = 0;
    std::uint64_t external_background_accesses = 0;

    hbm::HbmStats hbm_stats;
    hbf::HbfStats hbf_stats;
    external::ExternalBackingStats external_backing_stats;
    BaseDieLinkStats base_die_link_stats;
    std::optional<AddressHeatmapSnapshot> address_heatmap;
};

class CapacityOverflowComposition {
public:
    explicit CapacityOverflowComposition(CapacityOverflowConfig config);

    [[nodiscard]] CapacityOverflowRunResult run(
        const CapacityOverflowWorkload& workload);

private:
    CapacityOverflowConfig config_;
};

} // namespace hbfsim::physical::hybrid
