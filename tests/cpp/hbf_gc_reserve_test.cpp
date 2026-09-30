#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_controller.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using hbfsim::physical::AddressSpace;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::physical::TraceConfig;
using hbfsim::physical::TraceMode;
using hbfsim::host::HbfConfig;
using HbfController = hbfsim::verification::HbfWithHbm;
using hbfsim::host::MappingMode;

constexpr std::uint64_t kPage = 4096;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

HbfConfig geometry(std::uint32_t reserve_blocks_per_plane) {
    HbfConfig config;
    config.device.stacks = 1;
    config.device.channels_per_stack = 1;
    config.device.dies_per_channel = 1;
    config.device.planes_per_die = 4;
    config.device.blocks_per_plane = 64;
    config.device.pages_per_block = 256;
    config.device.page_size_bytes = kPage;
    config.device.oob_bytes_per_page = 0;
    config.host.mapping_entries_per_page = 512;


    config.host.gc_reserved_free_blocks_per_plane = reserve_blocks_per_plane;
    config.host.write_coalescing_enabled = false;
    config.device.ecc_decode_latency_ns = 1.0;
    config.device.ecc_encode_latency_ns = 1.0;
    config.device.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    config.device.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    return config;
}

HbfConfig cached_geometry(
    std::uint32_t reserve_blocks_per_plane,
    std::uint64_t cache_pages) {
    auto config = geometry(reserve_blocks_per_plane);
    config.host.mapping_mode = MappingMode::Cached;
    const std::uint64_t pages_per_stack =
        std::uint64_t{config.device.planes_per_die} * config.device.blocks_per_plane *
        config.device.pages_per_block;
    const std::uint64_t mapping_pages =
        (pages_per_stack + config.host.mapping_entries_per_page - 1) /
        config.host.mapping_entries_per_page;
    config.host.ctrl_dram_bytes = mapping_pages *
        config.host.mapping_directory_entry_bytes +
        (cache_pages + 1) * config.device.page_size_bytes;
    return config;
}

PhysicalRequest write(std::uint64_t lpn, double arrival_ns) {
    return PhysicalRequest{
        .id = "w",
        .tier = Tier::HBF,
        .op = Op::Write,
        .address_space = AddressSpace::Logical,
        .trace = TraceConfig{
            .mode = TraceMode::Off,
            .retain_completion_diagnostics = false,
        },
        .arrival_ns = arrival_ns,
        .addr = lpn * kPage,
        .bytes = kPage,
    };
}

// Random overwrites at 84% occupancy with a 64-page translation cache.
// Every write must be accepted and the final audit must conserve pages.
void test_cached_gc_keeps_making_progress(
    std::uint32_t reserve_blocks_per_plane,
    std::uint64_t cache_pages,
    unsigned seed,
    int writes) {
    const auto config = cached_geometry(reserve_blocks_per_plane, cache_pages);
    HbfController device(config);
    const std::uint64_t footprint = 54ull * 4 * 256;
    device.prepopulate_mutable_logical_page_range(0, footprint);
    std::mt19937_64 rng(seed);
    double t = 0.0;
    for (int index = 0; index < writes; ++index) {
        const auto completion = device.issue(
            write(rng() % footprint, t));
        t = completion.finish_ns;
    }
    (void)device.drain_pending("drain", t);
    const auto& stats = device.stats();
    require(stats.accounting_verified,
            "device audit did not verify after the overwrite stream");
    require(stats.gc_runs > 0,
            "the overwrite stream never exercised garbage collection");
    require(stats.mapping_cache_dirty_evictions > 0,
            "the overwrite stream never exercised dirty checkpoint eviction");
    require(stats.gc_relocations + stats.gc_reclaimed_invalid_pages ==
                stats.gc_runs * config.device.pages_per_block,
            "GC victim page conservation diverged");
    require(stats.gc_data_relocations + stats.gc_mapping_relocations ==
                stats.gc_relocations,
            "GC relocation role split diverged");
    const auto snapshot = device.audit_snapshot();
    require(snapshot.quiescent(),
            "the overwrite stream retained pending GC or mapping state");
    for (const auto& mapping : snapshot.mapping_pages) {
        require(snapshot.blocks.at(mapping.ppn / config.device.pages_per_block).role ==
                    "mapping",
                "mapping checkpoint was mixed into a data relocation block");
    }
    std::cout << "cached_gc reserve=" << reserve_blocks_per_plane
              << " cache=" << cache_pages << " seed=" << seed
              << " writes=" << writes
              << " gc_runs=" << stats.gc_runs
              << " relocations=" << stats.gc_relocations
              << " mapping_programs=" << stats.mapping_page_programs
              << " waf=" << stats.waf().value_or(-1.0) << '\n';
}

void test_hard_gc_preserves_progress_with_deferred_wl() {
    auto config = geometry(2);
    config.host.gc_low_watermark_pages = 1;
    config.host.gc_hard_watermark_pages = 2 * config.device.pages_per_block;
    config.host.static_wear_leveling_erase_gap = 4;
    config.host.static_wear_leveling_interval_erases = 1;
    HbfController device(config);
    const std::uint64_t footprint = 54ull * 4 * config.device.pages_per_block;
    device.prepopulate_mutable_logical_page_range(0, footprint);
    std::mt19937_64 generator(2);
    double time_ns = 0.0;
    for (int index = 0; index < 25000; ++index) {
        time_ns = device.issue(write(generator() % footprint, time_ns)).finish_ns;
    }
    (void)device.drain_pending("drain", time_ns);
    const auto& stats = device.stats();
    require(stats.gc_runs > 0 && stats.gc_user_blocked_runs > 0,
            "hard-pressure workload never blocked on GC");
    // Static migration now waits for completed erase resources and a
    // foreground-safe maintenance opportunity. Hard GC may coalesce/defer
    // checks; it must not be forced to migrate cold data under pressure.
    require(stats.static_wear_leveling_checks <= stats.gc_runs,
            "deferred wear-leveling checks exceed GC trigger events");
    require(stats.accounting_verified && device.audit_snapshot().quiescent(),
            "hard-pressure wear-leveling workload did not drain cleanly");
}

void test_reserved_planes_do_not_supply_gc_headroom() {
    for (const std::uint32_t managed_planes : {1U, 2U}) {
        const auto config = cached_geometry(1, 4);
        HbfController device(config);
        std::vector<std::size_t> reserved_blocks;
        for (std::uint32_t plane = 0; plane < config.device.planes_per_die; ++plane) {
            const auto reserved = plane < config.device.planes_per_die - managed_planes ?
                config.device.blocks_per_plane : config.device.blocks_per_plane / 2;
            for (std::uint32_t block = 0; block < reserved; ++block) {
                reserved_blocks.push_back(plane * config.device.blocks_per_plane + block);
            }
        }
        device.reserve_static_physical_block_indices(reserved_blocks);
        bool insufficient_reserve = false;
        try {
            const auto completion = device.issue(write(0, 0.0));
            (void)device.drain_pending("reserved-planes", completion.finish_ns);
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()).find("keeps a GC-only reserve") == std::string::npos) {
                throw;
            }
            insufficient_reserve = true;
        }
        require(insufficient_reserve == (managed_planes == 1),
                "reserved or partially managed planes changed the GC reserve");
        if (!insufficient_reserve) {
            require(device.audit_snapshot().quiescent(),
                    "reserved-plane write did not drain cleanly");
        }
    }
}

void test_cached_wl_preserves_reserve_at_logical_capacity() {
    auto config = cached_geometry(1, 1);
    config.host.static_wear_leveling_erase_gap = 1;
    config.host.static_wear_leveling_interval_erases = 1;
    HbfController device(config);
    const auto footprint = device.logical_capacity_pages();
    const auto cold_pages = footprint * 3 / 4;
    device.prepopulate_mutable_logical_page_range(0, footprint);
    std::mt19937_64 random(31);
    double arrival_ns = 0.0;
    for (std::uint64_t index = 0; index < 75000; ++index) {
        const auto lpn = cold_pages + random() % (footprint - cold_pages);
        arrival_ns = device.issue(write(lpn, arrival_ns)).finish_ns;
    }
    (void)device.drain_pending("capacity-wl-drain", arrival_ns);
    const auto& stats = device.stats();
    require(stats.gc_runs > 0 && stats.static_wear_leveling_runs > 0,
            "logical-capacity workload did not exercise both GC and wear leveling: gc=" + std::to_string(stats.gc_runs) + " wl=" + std::to_string(stats.static_wear_leveling_runs) + " checks=" + std::to_string(stats.static_wear_leveling_checks));
    require(stats.accounting_verified && device.audit_snapshot().quiescent(),
            "cached wear leveling exhausted the relocation reserve at logical capacity");
}

} // namespace

int main() {
    try {
        test_reserved_planes_do_not_supply_gc_headroom();
        test_cached_wl_preserves_reserve_at_logical_capacity();
        test_cached_gc_keeps_making_progress(2, 64, 2, 25000);
        test_cached_gc_keeps_making_progress(2, 4, 7, 25000);
        test_cached_gc_keeps_making_progress(2, 1, 11, 25000);
        test_hard_gc_preserves_progress_with_deferred_wl();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "hbf_gc_reserve_test: PASS\n";
    return 0;
}
