#include "physical/hybrid/capacity_overflow_composition.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using hbfsim::physical::external::cxl_memory_profile;
using hbfsim::physical::external::nvme_ssd_profile;
using hbfsim::physical::hybrid::BackingTier;
using hbfsim::physical::hybrid::CapacityOverflowComposition;
using hbfsim::physical::hybrid::CapacityOverflowConfig;
using hbfsim::physical::hybrid::CapacityOverflowRunResult;
using hbfsim::physical::hybrid::CapacityOverflowWorkload;
using hbfsim::physical::hybrid::CapacityReadbackDestination;

constexpr std::uint64_t kPage = 4096;
constexpr std::uint64_t kDataPages = 8;
constexpr std::uint64_t kReadBufferPages = 2;
constexpr std::uint64_t kOffloadPages = 4;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void require_throws(Fn&& fn, const std::string& message) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, message);
}

CapacityOverflowConfig base_config() {
    CapacityOverflowConfig config;
    config.hbm.capacity_bytes =
        (kDataPages + kReadBufferPages) * kPage;
    config.hbm.stacks = 1;
    config.hbm.channels_per_stack = 2;
    config.hbm.pseudo_channels_per_channel = 1;

    config.hbf.stacks = 2;
    config.hbf.channels_per_stack = 1;
    config.hbf.dies_per_channel = 1;
    config.hbf.planes_per_die = 2;
    config.hbf.blocks_per_plane = 16;
    config.hbf.pages_per_block = 16;
    config.hbf.page_size_bytes = kPage;
    config.hbf.oob_bytes_per_page = 224;
    config.hbf.mapping_entries_per_page = 64;
    config.hbf.write_coalescing_enabled = false;
    config.hbf.write_buffer_completion_requires_flush = true;
    config.hbf.read_buffer_pages = 0;
    config.hbf.gc_low_watermark_pages = 0;
    config.hbf.gc_reserved_free_blocks_per_plane = 0;

    config.external_backing = cxl_memory_profile();
    config.external_backing.capacity_bytes = 64 * kPage;
    config.external_backing.page_size_bytes = kPage;
    config.external_backing.media_channels = 2;
    config.external_backing.max_outstanding_requests = 8;

    config.base_die_link.read_bandwidth_GBps = 64.0;
    config.base_die_link.write_bandwidth_GBps = 64.0;
    config.base_die_link.latency_ns = 20.0;
    config.read_buffer_bytes = kReadBufferPages * kPage;
    config.transfer_batch_pages = 2;
    config.address_heatmap_bins = 16;
    return config;
}

CapacityOverflowWorkload workload() {
    return CapacityOverflowWorkload{
        .base_addr = 0,
        .total_write_pages = kDataPages + kOffloadPages,
        .first_arrival_ns = 0.0,
        .interarrival_ns = 0.0,
    };
}

void validate_common(const CapacityOverflowRunResult& run) {
    const auto offload_bytes = kOffloadPages * kPage;
    require(run.stats.hbm_data_pages == kDataPages,
            "HBM resident page count drifted");
    require(run.stats.hbm_read_buffer_pages == kReadBufferPages,
            "HBM read-buffer page count drifted");
    require(run.stats.written_pages == kDataPages + kOffloadPages &&
                run.stats.offload_pages == kOffloadPages &&
                run.stats.readback_pages == kOffloadPages,
            "workload page census drifted");
    require(run.stats.offload_bytes == offload_bytes &&
                run.stats.readback_bytes == offload_bytes,
            "offloading/readback byte count drifted");
    require(run.hbm_user_accesses ==
                kDataPages + 2 * kOffloadPages &&
                run.hbm_background_accesses == 2 * kOffloadPages,
            "HBM foreground/background access census drifted");
    require(run.hbm_stats.write_bytes ==
                (kDataPages + 2 * kOffloadPages) * kPage &&
                run.hbm_stats.read_bytes == 2 * offload_bytes,
            "HBM byte conservation drifted");
    require(run.fill_write_latencies_ns.size() == kDataPages &&
                run.offload_offered_latencies_ns.size() == kOffloadPages &&
                run.offload_service_latencies_ns.size() == kOffloadPages &&
                run.readback_offered_latencies_ns.size() == kOffloadPages &&
                run.readback_service_latencies_ns.size() == kOffloadPages,
            "phase latency census drifted");
    for (std::size_t index = 0; index < kOffloadPages; ++index) {
        require(
            run.offload_offered_latencies_ns[index] >=
                    run.offload_service_latencies_ns[index] &&
                run.readback_offered_latencies_ns[index] >=
                    run.readback_service_latencies_ns[index],
            "offered latency must include service latency");
    }
    require(
        run.readback_offered_latencies_ns.back() >
            run.readback_service_latencies_ns.back(),
        "a later readback batch must expose bulk-arrival queueing");
    require(run.stats.offload_start_ns >= run.stats.fill_finish_ns &&
                run.stats.write_finish_ns >= run.stats.offload_start_ns &&
                run.stats.read_start_ns == run.stats.write_finish_ns &&
                run.stats.read_finish_ns > run.stats.read_start_ns &&
                run.stats.quiescent_finish_ns >= run.stats.read_finish_ns,
            "phase timeline is not causal");
    require(run.address_heatmap.has_value(),
            "capacity-overflow heatmap is missing");
}

void test_hbf_path() {
    auto config = base_config();
    config.backing = BackingTier::Hbf;
    const auto run = CapacityOverflowComposition(config).run(workload());
    validate_common(run);
    const auto offload_bytes = kOffloadPages * kPage;
    require(run.hbf_stats.logical_write_bytes == offload_bytes &&
                run.hbf_stats.logical_read_bytes == offload_bytes,
            "HBF logical traffic does not conserve");
    require(run.hbf_background_accesses == 2 * kOffloadPages,
            "HBF request census drifted");
    require(run.base_die_link_stats.write_bytes == offload_bytes &&
                run.base_die_link_stats.read_bytes == offload_bytes,
            "HBF D2D traffic does not conserve");
    require(run.external_background_accesses == 0,
            "HBF path unexpectedly exercised external backing");
}

CapacityOverflowRunResult run_external(bool ssd) {
    auto config = base_config();
    config.backing = BackingTier::External;
    config.external_backing = ssd ? nvme_ssd_profile() : cxl_memory_profile();
    config.external_backing.capacity_bytes = 64 * kPage;
    config.external_backing.page_size_bytes = kPage;
    config.external_backing.media_channels = 2;
    config.external_backing.max_outstanding_requests = 8;
    return CapacityOverflowComposition(config).run(workload());
}

void test_external_paths() {
    const auto dram = run_external(false);
    const auto ssd = run_external(true);
    validate_common(dram);
    validate_common(ssd);
    const auto offload_bytes = kOffloadPages * kPage;
    for (const auto* run : {&dram, &ssd}) {
        require(run->external_backing_stats.write_bytes == offload_bytes &&
                    run->external_backing_stats.read_bytes == offload_bytes,
                "external traffic does not conserve");
        require(run->external_background_accesses == 2 * kOffloadPages,
                "external request census drifted");
        require(run->hbf_background_accesses == 0,
                "external path unexpectedly exercised HBF");
    }
    require(ssd.stats.offload_elapsed_ns() > dram.stats.offload_elapsed_ns() &&
                ssd.stats.read_elapsed_ns() > dram.stats.read_elapsed_ns(),
            "SSD profile must be slower than CXL memory in both phases");
}

void test_fifo_wraps_the_resident_window() {
    constexpr std::uint64_t offload_pages = 2 * kDataPages + 1;
    auto config = base_config();
    config.backing = BackingTier::External;
    const auto run = CapacityOverflowComposition(config).run(
        CapacityOverflowWorkload{
            .base_addr = 0,
            .total_write_pages = kDataPages + offload_pages,
            .first_arrival_ns = 0.0,
            .interarrival_ns = 0.0,
        });
    const auto offload_bytes = offload_pages * kPage;
    require(run.stats.offload_pages == offload_pages &&
                run.stats.readback_pages == offload_pages,
            "multi-wrap FIFO page census drifted");
    require(run.external_backing_stats.write_bytes == offload_bytes &&
                run.external_backing_stats.read_bytes == offload_bytes,
            "multi-wrap external traffic does not conserve");
    require(run.hbm_stats.write_bytes ==
                (kDataPages + 2 * offload_pages) * kPage &&
                run.hbm_stats.read_bytes == 2 * offload_bytes,
            "multi-wrap HBM traffic does not conserve");
}

void test_layer_round_trip_restores_original_slots() {
    auto config = base_config();
    config.backing = BackingTier::External;
    config.readback_destination =
        CapacityReadbackDestination::OriginalSlots;
    const auto run = CapacityOverflowComposition(config).run(workload());
    validate_common(run);
    require(
        run.stats.readback_destination ==
            CapacityReadbackDestination::OriginalSlots &&
            run.stats.read_finish_ns > run.stats.write_finish_ns,
        "original-slot layer restore did not complete after offloading");

    auto oversized_restore = workload();
    oversized_restore.total_write_pages = 2 * kDataPages + 1;
    require_throws(
        [&] {
            (void)CapacityOverflowComposition(config).run(
                oversized_restore);
        },
        "original-slot restore accepted a set larger than HBM");
}

void test_invalid_contracts() {
    auto no_overflow = workload();
    no_overflow.total_write_pages = kDataPages;
    require_throws(
        [&] {
            (void)CapacityOverflowComposition(base_config()).run(no_overflow);
        },
        "non-overflow workload was accepted");

    auto early_ack_hbf = base_config();
    early_ack_hbf.hbf.write_buffer_completion_requires_flush = false;
    require_throws(
        [&] {
            (void)CapacityOverflowComposition(early_ack_hbf).run(workload());
        },
        "HBF SRAM acknowledgement was accepted for victim reuse");

    auto oversized_batch = base_config();
    oversized_batch.transfer_batch_pages = kDataPages + 1;
    require_throws(
        [&] {
            (void)CapacityOverflowComposition(oversized_batch).run(workload());
        },
        "transfer batch larger than the resident window was accepted");

    auto oversized_read_batch = base_config();
    oversized_read_batch.transfer_batch_pages = kReadBufferPages + 1;
    require_throws(
        [&] {
            (void)CapacityOverflowComposition(oversized_read_batch)
                .run(workload());
        },
        "transfer batch larger than the read buffer was accepted");
}

} // namespace

int main() {
    try {
        test_hbf_path();
        test_external_paths();
        test_fifo_wraps_the_resident_window();
        test_layer_round_trip_restores_original_slots();
        test_invalid_contracts();
        std::cout << "capacity-overflow composition contract: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "capacity-overflow composition contract: FAIL: "
                  << error.what() << "\n";
        return 1;
    }
}
