#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_controller.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using hbfsim::physical::AddressSpace;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::host::HbfConfig;
using HbfController = hbfsim::verification::HbfWithHbm;
using hbfsim::host::MappingMode;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(
    std::uint64_t actual,
    std::uint64_t expected,
    const std::string& context) {
    require(
        actual == expected,
        context + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
}

void require_throws(
    const std::function<void()>& function,
    const std::string& context) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error(context);
}

HbfConfig base_config() {
    HbfConfig config;
    config.device.stacks = 1;
    config.device.channels_per_stack = 1;
    config.device.dies_per_channel = 1;
    config.device.planes_per_die = 1;
    config.device.blocks_per_plane = 16;
    config.device.pages_per_block = 4;
    config.device.page_size_bytes = 4096;
    config.device.oob_bytes_per_page = 0;
    config.host.mapping_entries_per_page = 1;


    config.host.gc_low_watermark_pages = 0;
    config.device.ecc_decode_latency_ns = 1.0;
    config.device.ecc_encode_latency_ns = 1.0;
    config.device.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    config.device.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    return config;
}

PhysicalRequest request(
    std::string id,
    Op op,
    std::uint64_t lpn,
    std::uint64_t page_size,
    double arrival_ns) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = Tier::HBF,
        .op = op,
        .address_space = AddressSpace::Logical,
        .arrival_ns = arrival_ns,
        .addr = lpn * page_size,
        .bytes = page_size,
    };
}

void test_budget_modes_fail_closed() {
    auto resident = base_config();
    resident.host.mapping_mode = MappingMode::FullResident;
    resident.host.ctrl_dram_bytes = resident.device.page_size_bytes;
    require_throws(
        [&] { HbfController device(resident); },
        "full-resident mapping accepted an undersized DRAM budget");

    auto cached_without_budget = base_config();
    cached_without_budget.host.mapping_mode = MappingMode::Cached;
    cached_without_budget.host.ctrl_dram_bytes = 0;
    require_throws(
        [&] { HbfController device(cached_without_budget); },
        "cached mapping derived an implicit full-resident budget");

    auto cached_directory_only = base_config();
    cached_directory_only.host.mapping_mode = MappingMode::Cached;
    cached_directory_only.host.ctrl_dram_bytes =
        cached_directory_only.device.page_size_bytes;
    require_throws(
        [&] { HbfController device(cached_directory_only); },
        "cached mapping accepted a directory-only DRAM budget");

    auto cached = base_config();
    cached.host.mapping_mode = MappingMode::Cached;
    cached.host.ctrl_dram_bytes = 512 + 2 * cached.device.page_size_bytes;
    HbfController device(cached);
    const auto& stats = device.stats();
    require_equal(stats.mapping_table_bytes, 64 * cached.device.page_size_bytes,
                  "logical mapping-table footprint");
    require_equal(stats.resident_mapping_table_bytes, 0,
                  "cached mode resident-table footprint");
    require_equal(stats.mapping_directory_bytes, 512,
                  "mapping-directory DRAM footprint");
    require_equal(stats.mapping_cache_capacity_bytes, cached.device.page_size_bytes,
                  "mapping-cache DRAM footprint");
    require_equal(stats.mapping_cache_pages_per_stack, 1,
                  "mapping-cache pages per stack");
}

void test_hit_miss_and_dirty_eviction_accounting() {
    auto config = base_config();
    config.host.mapping_mode = MappingMode::Cached;
    config.host.ctrl_dram_bytes = 512 + 2 * config.device.page_size_bytes;
    HbfController device(config);
    device.prepopulate_logical_pages({0, 1});

    double now = 0.0;
    const auto issue = [&](std::string id, Op op, std::uint64_t lpn) {
        const auto completion = device.issue(request(
            std::move(id), op, lpn, config.device.page_size_bytes, now));
        now = completion.finish_ns;
    };

    issue("read-lpn0-miss", Op::Read, 0);
    issue("read-lpn0-hit", Op::Read, 0);
    issue("read-lpn1-miss", Op::Read, 1);
    issue("overwrite-lpn1", Op::Write, 1);
    issue("read-lpn0-dirty-eviction", Op::Read, 0);
    const auto drain = device.drain_pending("final-drain", now);
    require(drain.finish_ns >= now, "mapping-cache drain moved backward");

    const auto& stats = device.stats();
    require(stats.accounting_verified,
            "mapping-cache run failed physical accounting");
    require_equal(stats.mapping_cache_hits, 3, "mapping-cache hits");
    require_equal(stats.mapping_cache_misses, 3, "mapping-cache misses");
    require_equal(stats.mapping_cache_erased_misses, 0,
                  "mapping-cache erased misses");
    require_equal(stats.mapping_cache_coalesced_misses, 0,
                  "mapping-cache coalesced misses");
    require_equal(stats.mapping_cache_evictions, 2,
                  "mapping-cache evictions");
    require_equal(stats.mapping_cache_dirty_evictions, 1,
                  "mapping-cache dirty evictions");
    require_equal(stats.mapping_media_reads, 3,
                  "mapping-cache media reads");
    require_equal(stats.mapping_media_read_bytes, 3 * config.device.page_size_bytes,
                  "mapping-cache media-read bytes");
    require_equal(stats.mapping_cache_entries, 1,
                  "mapping-cache final entries");
    require_equal(stats.mapping_cache_peak_entries, 1,
                  "mapping-cache peak entries");
    require_equal(stats.mapping_lookup_ops, 5, "mapping lookups");
    require_equal(stats.mapping_update_ops, 1, "mapping updates");
    require_equal(stats.page_reads, 5, "two data senses plus three mapping senses (two device-cache hits)");
    require_equal(stats.physical_read_bytes, 5 * config.device.page_size_bytes,
                  "data plus mapping physical bytes");
    require_equal(stats.data_programs, 1, "foreground data programs");
    require_equal(stats.mapping_page_programs, 1,
                  "dirty-eviction mapping programs");
    require_equal(stats.physical_write_bytes, 2 * config.device.page_size_bytes,
                  "data plus dirty mapping physical writes");
    require(device.audit_snapshot().quiescent(),
            "mapping-cache run did not reach quiescence");
}

void test_capacity_derived_budget_drives_real_cache_accesses() {
    auto config = base_config();
    config.host.mapping_mode = MappingMode::Cached;
    config.device.blocks_per_plane = 2000;
    config.device.pages_per_block = 8;
    config.device.page_size_bytes = 4096;
    config.host.mapping_entries_per_page = 512;
    config.host.ctrl_dram_bytes = 0;
    config.host.ctrl_dram_capacity_denominator = 1000;

    HbfController device(config);
    device.prepopulate_logical_pages({0});
    const auto first = device.issue(request(
        "ratio-cache-miss", Op::Read, 0, config.device.page_size_bytes, 0.0));
    const auto second = device.issue(request(
        "ratio-cache-hit", Op::Read, 0, config.device.page_size_bytes,
        first.finish_ns));
    require(second.finish_ns >= first.finish_ns,
            "capacity-derived cache completion moved backward");

    const auto& stats = device.stats();
    require_equal(stats.mapping_directory_bytes, 256,
                  "capacity-derived mapping directory");
    // One of the remaining pages belongs to the Host GC copy buffer.
    require_equal(stats.mapping_cache_capacity_bytes, 14 * 4096,
                  "capacity-derived usable mapping cache");
    require_equal(stats.mapping_cache_peak_entries, 1,
                  "capacity-derived cache peak occupancy");
    require_equal(stats.mapping_cache_misses, 1,
                  "capacity-derived cache misses");
    require_equal(stats.mapping_cache_hits, 1,
                  "capacity-derived cache hits");
    require_equal(stats.mapping_media_reads, 1,
                  "capacity-derived cache media reads");
}

void test_hits_pipeline_past_future_sibling_mapping_updates() {
    auto config = base_config();
    config.host.mapping_mode = MappingMode::Cached;
    config.host.mapping_entries_per_page = 4;
    // 64 data pages / 4 entries gives a 16-entry, 128-byte directory.
    config.host.write_coalescing_enabled = true;
    config.host.write_buffer_pages = 8;
    config.host.write_buffer_flush_threshold_pages = 1;
    // The shared pool holds the 128-byte directory, eight data-buffer pages,
    // and exactly one cached mapping page.
    config.host.ctrl_dram_bytes =
        128 + 10 * config.device.page_size_bytes;
    HbfController device(config);

    const auto first = device.issue(request(
        "write-lpn0", Op::Write, 0, config.device.page_size_bytes, 0.0));
    const auto second = device.issue(request(
        "write-lpn1", Op::Write, 1, config.device.page_size_bytes,
        first.finish_ns));

    require(
        second.finish_ns < config.device.t_program_page_ns,
        "mapping-page hit waited for a sibling LPN's future data program");
    const auto drain = device.drain_pending("final-drain", second.finish_ns);
    require(drain.finish_ns >= second.finish_ns,
            "pipelined mapping-cache drain moved backward");

    const auto& stats = device.stats();
    require(stats.accounting_verified,
            "pipelined mapping-cache run failed physical accounting");
    require_equal(stats.mapping_cache_misses, 1,
                  "pipelined mapping-cache misses");
    require_equal(stats.write_buffer_capacity_bytes,
                  8 * config.device.page_size_bytes,
                  "pipelined write-buffer DRAM capacity");
    require_equal(stats.mapping_cache_capacity_bytes,
                  config.device.page_size_bytes,
                  "pipelined mapping-cache shared-pool remainder");
    require_equal(stats.mapping_cache_coalesced_misses, 0,
                  "pipelined mapping-cache coalesced misses");
    require_equal(stats.mapping_cache_hits, 3,
                  "pipelined mapping-cache hits");
    require_equal(stats.mapping_lookup_ops, 2,
                  "pipelined mapping lookups");
    require_equal(stats.mapping_update_ops, 2,
                  "pipelined mapping updates");
    require_equal(stats.mapping_page_programs, 1,
                  "coalesced mapping checkpoint programs");
    require(stats.write_buffer_dram_write_ops == 2 &&
                stats.write_buffer_dram_read_ops == 2,
            "write coalescer did not access controller DRAM at stage and flush");
    require(device.audit_snapshot().quiescent(),
            "pipelined mapping-cache run did not reach quiescence");
}

void test_dirty_eviction_gc_reentrant_fill_is_coalesced() {
    auto config = base_config();
    config.device.stacks = 1;
    config.device.planes_per_die = 4;
    config.device.blocks_per_plane = 64;
    config.device.pages_per_block = 256;
    config.device.page_size_bytes = 4096;
    config.host.mapping_entries_per_page = 512;
    config.host.mapping_mode = MappingMode::Cached;
    config.host.auto_gc_enabled = true;
    config.host.gc_reserved_free_blocks_per_plane = 2;
    config.host.write_coalescing_enabled = false;

    const std::uint64_t pages_per_stack =
        config.device.planes_per_die * config.device.blocks_per_plane *
        config.device.pages_per_block;
    const std::uint64_t mapping_pages_per_stack =
        (pages_per_stack + config.host.mapping_entries_per_page - 1) /
        config.host.mapping_entries_per_page;
    const std::uint64_t directory_bytes_per_stack =
        mapping_pages_per_stack * config.host.mapping_directory_entry_bytes;
    const std::uint64_t cache_pages_per_stack = 64;
    config.host.ctrl_dram_bytes = config.device.stacks *
        (directory_bytes_per_stack +
         (cache_pages_per_stack + 1) * config.device.page_size_bytes);

    HbfController device(config);
    const std::uint64_t initial_pages = 54ull * 4 * config.device.pages_per_block;
    device.prepopulate_mutable_logical_page_range(0, initial_pages);

    double now = 0.0;
    std::uint64_t sequence = 0;
    const auto issue = [&](std::uint64_t lpn) {
        const auto completion = device.issue(request(
            "gc-reentrant-" + std::to_string(sequence++),
            Op::Write,
            lpn,
            config.device.page_size_bytes,
            now));
        now = completion.finish_ns;
    };

    std::mt19937_64 generator(2);
    for (std::uint64_t index = 0; index < 50000; ++index) {
        issue(generator() % initial_pages);
    }
    now = device.drain_pending("gc-reentrant-final", now).finish_ns;
    (void)now;

    const auto& stats = device.stats();
    require(stats.accounting_verified,
            "GC-reentrant mapping-cache run failed physical accounting");
    require(stats.gc_runs > 0 && stats.mapping_cache_dirty_evictions > 0,
            "GC-reentrant mapping-cache run missed its trigger conditions");
    require(stats.mapping_cache_coalesced_misses > 0,
            "GC-reentrant mapping-cache fill was not coalesced");
    require_equal(
        stats.mapping_cache_misses,
        stats.mapping_media_reads + stats.mapping_cache_buffer_hits + stats.mapping_cache_erased_misses +
            stats.mapping_cache_coalesced_misses,
        "GC-reentrant mapping-cache miss classes");
    require(device.audit_snapshot().quiescent(),
            "GC-reentrant mapping-cache run did not reach quiescence");
}

void test_gc_induced_mapping_writeback_uses_the_relocation_reserve() {
    auto config = base_config();
    config.device.blocks_per_plane = 11;
    config.host.mapping_mode = MappingMode::Cached;
    config.host.ctrl_dram_bytes =
        static_cast<std::uint64_t>(config.device.blocks_per_plane) *
            config.device.pages_per_block * config.host.mapping_directory_entry_bytes +
        2 * config.device.page_size_bytes;
    config.host.auto_gc_enabled = true;
    config.host.gc_low_watermark_pages = config.device.pages_per_block;
    config.host.gc_hard_watermark_pages = 1;
    config.host.gc_reserved_free_blocks_per_plane = 2;

    HbfController device(config);
    device.prepopulate_logical_pages({0, 1, 2, 3, 4, 5, 6, 7});
    double now = 0.0;
    for (std::uint64_t sequence = 0; sequence < 16; ++sequence) {
        try {
            const auto completion = device.issue(request(
                "reserve-progress-" + std::to_string(sequence),
                Op::Write,
                sequence % 8,
                config.device.page_size_bytes,
                now));
            now = completion.finish_ns;
        } catch (const std::runtime_error& error) {
            throw std::runtime_error(
                "GC-reserve write " + std::to_string(sequence) +
                " failed: " + error.what());
        }
    }
    const auto profile_before_drain = device.block_profile(1);
    require_equal(
        profile_before_drain.front().erase_count_sum,
        device.stats().block_erase_count_sum,
        "GC-reserve live block-profile erase count");
    now = device.drain_pending("reserve-progress-drain", now).finish_ns;
    const auto& stats = device.stats();
    require(stats.gc_runs > 0 && stats.mapping_cache_dirty_evictions > 0,
            "GC-reserve run missed GC or dirty mapping eviction");
    require(stats.mapping_page_programs > 0,
            "GC-reserve run did not persist an induced mapping checkpoint");
    require(stats.accounting_verified,
            "GC-reserve run failed physical accounting");
    const auto erase_counts = device.block_erase_counts();
    std::uint64_t erase_count_sum = 0;
    for (const auto count : erase_counts) {
        erase_count_sum += count;
    }
    require(
        erase_counts.size() == stats.writable_blocks &&
            erase_count_sum == stats.block_erase_count_sum,
        "exact block-wear snapshot diverged from accounting");
    require(device.audit_snapshot().quiescent(),
            "GC-reserve run did not reach quiescence");
}

void test_mapping_writeback_keeps_its_own_block_role() {
    auto config = base_config();
    config.device.blocks_per_plane = 11;
    config.host.mapping_mode = MappingMode::Cached;
    config.host.ctrl_dram_bytes =
        static_cast<std::uint64_t>(config.device.blocks_per_plane) *
            config.device.pages_per_block * config.host.mapping_directory_entry_bytes +
        2 * config.device.page_size_bytes;
    config.host.auto_gc_enabled = true;
    config.host.gc_low_watermark_pages = config.device.pages_per_block;
    config.host.gc_hard_watermark_pages = 1;
    config.host.gc_reserved_free_blocks_per_plane = 2;

    HbfController device(config);
    device.prepopulate_logical_pages({0, 1, 2, 3, 4, 5, 6, 7});
    const auto write = device.issue(request(
        "mapping-role-new-lpn",
        Op::Write,
        8,
        config.device.page_size_bytes,
        0.0));
    const auto read = device.issue(request(
        "mapping-role-evict-dirty-vpn",
        Op::Read,
        0,
        config.device.page_size_bytes,
        write.finish_ns));
    require(read.finish_ns >= write.finish_ns,
            "mapping-role read moved before the dirty write");

    const auto& stats = device.stats();
    require_equal(stats.gc_runs, 0,
                  "mapping allocation must not invent a victim");
    require_equal(stats.mapping_page_programs, 1,
                  "mapping checkpoint programs");
    const auto snapshot = device.audit_snapshot();
    for (const auto& page : snapshot.materialized_pages) {
        if (page.owner == "mapping") {
            require(snapshot.blocks.at(page.ppn / config.device.pages_per_block).role ==
                        "mapping",
                    "mapping checkpoint did not stay in a Mapping-owned block");
        }
    }
}

} // namespace

int main() {
    try {
        test_budget_modes_fail_closed();
        test_hit_miss_and_dirty_eviction_accounting();
        test_capacity_derived_budget_drives_real_cache_accesses();
        test_hits_pipeline_past_future_sibling_mapping_updates();
        test_dirty_eviction_gc_reentrant_fill_is_coalesced();
        test_gc_induced_mapping_writeback_uses_the_relocation_reserve();
        test_mapping_writeback_keeps_its_own_block_role();
        std::cout << "HBF mapping-cache regression passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "HBF mapping-cache regression failed: "
                  << error.what() << '\n';
        return 1;
    }
}
