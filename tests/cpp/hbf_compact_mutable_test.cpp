#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using hbfsim::physical::AddressSpace;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalCompletion;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::host::HbfAuditSnapshot;
using hbfsim::host::HbfConfig;
using HbfController = hbfsim::verification::HbfWithHbm;
using hbfsim::host::HbfStats;

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

void require_close(double actual, double expected, const std::string& context) {
    require(
        std::abs(actual - expected) <=
            1e-12 * std::max({1.0, std::abs(actual), std::abs(expected)}),
        context + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
}

PhysicalRequest write_request(
    std::string id,
    double arrival_ns,
    std::uint64_t lpn,
    std::uint64_t page_size_bytes) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = Tier::HBF,
        .op = Op::Write,
        .address_space = AddressSpace::Logical,
        .arrival_ns = arrival_ns,
        .addr = lpn * page_size_bytes,
        .bytes = page_size_bytes,
    };
}

PhysicalRequest raw_erase_request(
    std::string id,
    double arrival_ns,
    std::uint64_t physical_addr) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = Tier::HBF,
        .op = Op::Erase,
        .address_space = AddressSpace::Physical,
        .arrival_ns = arrival_ns,
        .addr = physical_addr,
        .bytes = 0,
    };
}

HbfConfig deterministic_gc_config() {
    HbfConfig config;
    config.device.channels_per_stack = 1;
    config.device.dies_per_channel = 1;
    config.device.planes_per_die = 1;
    config.device.blocks_per_plane = 8;
    config.device.pages_per_block = 4;
    config.device.page_size_bytes = 4096;
    config.device.oob_bytes_per_page = 0;
    config.host.mapping_entries_per_page = 4;

    config.host.gc_low_watermark_pages = 8;
    config.device.ecc_decode_latency_ns = 1.0;
    config.device.ecc_encode_latency_ns = 1.0;
    config.device.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    config.device.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    return config;
}

struct RunResult {
    std::vector<PhysicalCompletion> completions;
    HbfStats stats;
    HbfAuditSnapshot audit;
};

RunResult run_workload(bool compact) {
    const auto config = deterministic_gc_config();
    HbfController device(config);
    if (compact) {
        device.prepopulate_mutable_logical_page_range(0, 4);
    } else {
        device.prepopulate_logical_pages({0, 1, 2, 3});
    }

    const auto initial = device.stats();
    require_equal(
        initial.initial_logical_data_pages, 4,
        "initial data-page population");
    require_equal(
        initial.initial_mapping_pages, 1,
        "initial mapping-page population");
    require_equal(
        initial.logical_write_bytes, 0,
        "initial image must not count as workload logical writes");
    require_equal(
        initial.physical_write_bytes, 0,
        "initial image must not count as workload physical writes");
    require(initial.accounting_verified, "initial accounting audit failed");

    RunResult result;
    double arrival_ns = 0.0;
    for (std::uint64_t index = 0; index < 32; ++index) {
        result.completions.push_back(device.issue(write_request(
            "write/" + std::to_string(index), arrival_ns, index % 5,
            config.device.page_size_bytes)));
        result.completions.push_back(device.drain_pending(
            "checkpoint/" + std::to_string(index),
            result.completions.back().finish_ns));
        arrival_ns = result.completions.back().finish_ns;
    }
    result.stats = device.stats();
    result.audit = device.audit_snapshot();
    require(result.audit.quiescent(), "final FTL state is not quiescent");
    require(result.stats.accounting_verified, "final accounting audit failed");
    return result;
}

void compare_completions(
    const std::vector<PhysicalCompletion>& materialized,
    const std::vector<PhysicalCompletion>& compact) {
    require_equal(
        compact.size(), materialized.size(), "completion count");
    for (std::size_t index = 0; index < materialized.size(); ++index) {
        const auto& expected = materialized[index];
        const auto& actual = compact[index];
        require(actual.id == expected.id, "completion ID diverged");
        require(actual.op == expected.op, "completion operation diverged");
        require_close(
            actual.start_ns, expected.start_ns,
            "completion start " + std::to_string(index));
        require_close(
            actual.finish_ns, expected.finish_ns,
            "completion finish " + std::to_string(index));
        require_equal(
            actual.logical_bytes, expected.logical_bytes,
            "completion logical bytes " + std::to_string(index));
        require_equal(
            actual.physical_bytes, expected.physical_bytes,
            "completion physical bytes " + std::to_string(index));
    }
}

void compare_operational_stats(
    const HbfStats& expected,
    const HbfStats& actual) {
#define REQUIRE_SAME(field) \
    require_equal(actual.field, expected.field, "stats." #field)
    REQUIRE_SAME(read_requests);
    REQUIRE_SAME(program_requests);
    REQUIRE_SAME(logical_read_bytes);
    REQUIRE_SAME(logical_write_bytes);
    REQUIRE_SAME(physical_read_bytes);
    REQUIRE_SAME(physical_write_bytes);
    REQUIRE_SAME(data_program_payload_bytes);
    REQUIRE_SAME(mapping_program_payload_bytes);
    REQUIRE_SAME(gc_relocation_payload_bytes);
    REQUIRE_SAME(page_reads);
    REQUIRE_SAME(data_programs);
    REQUIRE_SAME(page_programs);
    REQUIRE_SAME(mapping_page_programs);
    REQUIRE_SAME(mapping_entries);
    REQUIRE_SAME(invalidations);
    REQUIRE_SAME(mapping_lookup_ops);
    REQUIRE_SAME(mapping_user_lookup_ops);
    REQUIRE_SAME(mapping_gc_lookup_ops);
    REQUIRE_SAME(mapping_update_ops);
    REQUIRE_SAME(mapping_user_update_ops);
    REQUIRE_SAME(mapping_gc_update_ops);
    REQUIRE_SAME(gc_runs);
    REQUIRE_SAME(gc_relocations);
    REQUIRE_SAME(gc_data_relocations);
    REQUIRE_SAME(gc_mapping_relocations);
    REQUIRE_SAME(gc_reclaimed_invalid_pages);
    REQUIRE_SAME(block_erases);
    REQUIRE_SAME(total_pages);
    REQUIRE_SAME(free_pages);
    REQUIRE_SAME(valid_pages);
    REQUIRE_SAME(invalid_pages);
#undef REQUIRE_SAME

    require(expected.waf().has_value(), "materialized WAF is undefined");
    require(actual.waf().has_value(), "compact WAF is undefined");
    require_close(*actual.waf(), *expected.waf(), "canonical WAF");
}

void compare_audit(
    const HbfAuditSnapshot& expected,
    const HbfAuditSnapshot& actual) {
    require_equal(
        actual.logical_mappings.size(),
        expected.logical_mappings.size(),
        "logical mapping count");
    for (std::size_t index = 0;
         index < expected.logical_mappings.size();
         ++index) {
        require(
            actual.logical_mappings[index].key ==
                    expected.logical_mappings[index].key &&
                actual.logical_mappings[index].ppn ==
                    expected.logical_mappings[index].ppn,
            "logical mapping state diverged");
    }

    require_equal(
        actual.mapping_pages.size(),
        expected.mapping_pages.size(),
        "mapping checkpoint count");
    for (std::size_t index = 0;
         index < expected.mapping_pages.size();
         ++index) {
        require(
            actual.mapping_pages[index].key ==
                    expected.mapping_pages[index].key &&
                actual.mapping_pages[index].ppn ==
                    expected.mapping_pages[index].ppn,
            "mapping checkpoint state diverged");
    }

    require_equal(
        actual.materialized_pages.size(),
        expected.materialized_pages.size(),
        "materialized page-state count");
    for (std::size_t index = 0;
         index < expected.materialized_pages.size();
         ++index) {
        const auto& expected_page = expected.materialized_pages[index];
        const auto& actual_page = actual.materialized_pages[index];
        require(
            actual_page.ppn == expected_page.ppn &&
                actual_page.status == expected_page.status &&
                actual_page.owner == expected_page.owner &&
                actual_page.logical_key == expected_page.logical_key &&
                actual_page.block_epoch == expected_page.block_epoch,
            "materialized page state diverged");
    }

    require_equal(
        actual.blocks.size(), expected.blocks.size(), "block-state count");
    for (std::size_t index = 0; index < expected.blocks.size(); ++index) {
        const auto& expected_block = expected.blocks[index];
        const auto& actual_block = actual.blocks[index];
        require(
            actual_block.block == expected_block.block &&
                actual_block.role == expected_block.role &&
                actual_block.valid_pages == expected_block.valid_pages &&
                actual_block.invalid_pages == expected_block.invalid_pages &&
                actual_block.free_pages == expected_block.free_pages &&
                actual_block.next_page == expected_block.next_page &&
                actual_block.erase_count == expected_block.erase_count &&
                actual_block.epoch == expected_block.epoch &&
                actual_block.erase_pending == expected_block.erase_pending,
            "block state diverged");
    }
}

void test_compact_mutable_matches_materialized_initial_image() {
    const auto materialized = run_workload(false);
    const auto compact = run_workload(true);

    compare_completions(materialized.completions, compact.completions);
    compare_operational_stats(materialized.stats, compact.stats);
    compare_audit(materialized.audit, compact.audit);

    require(compact.stats.gc_runs > 0, "workload did not trigger GC");
    require(compact.stats.gc_relocations > 0, "GC did not relocate live pages");
    require_equal(
        compact.stats.compact_initial_logical_data_pages, 4,
        "compact initial data pages");
    require_equal(
        compact.stats.compact_initial_mapping_pages, 1,
        "compact initial mapping pages");
    require_equal(
        compact.stats.compact_live_logical_data_pages, 0,
        "compact live data pages after overwrite and GC");
    require_equal(
        compact.stats.compact_live_mapping_pages, 0,
        "compact live mapping pages after checkpoint");
    require_equal(
        compact.stats.compact_retired_logical_data_pages, 4,
        "compact retired data pages");
    require_equal(
        compact.stats.compact_retired_mapping_pages, 1,
        "compact retired mapping pages");
}

void test_pending_raw_erase_is_in_block_wear_snapshot() {
    const auto config = deterministic_gc_config();
    HbfController device(config);
    const auto completion = device.issue(
        raw_erase_request("raw-erase-free-block", 0.0, 0));
    require(completion.finish_ns > 0.0, "raw erase did not schedule media work");

    // block_erases uses issue-count semantics. Before the visibility commit,
    // the block is erase-pending and its committed erase_count is still zero;
    // the structural snapshot must nevertheless include the owned P/E cycle.
    const auto stats = device.stats();
    require(stats.accounting_verified, "pending raw-erase accounting failed");
    require_equal(stats.block_erases, 1, "raw erase count");
    require_equal(stats.block_erase_count_sum, 1, "per-block erase-count sum");
    require_equal(stats.worn_blocks, 1, "worn block count");
    require_equal(stats.min_block_erase_count, 0, "minimum erase count");
    require_equal(stats.max_block_erase_count, 1, "maximum erase count");
    require_close(
        static_cast<double>(stats.block_erase_count_sum) /
            static_cast<double>(stats.writable_blocks),
        1.0 / config.device.blocks_per_plane,
        "mean pending erase count");
}

void test_overwrite_metadata_is_bounded_before_any_gc() {
    auto config = deterministic_gc_config();
    config.device.blocks_per_plane = 1024;
    config.device.pages_per_block = 16;
    config.host.mapping_entries_per_page = 16;
    config.host.gc_low_watermark_pages = 16;
    HbfController device(config);
    device.prepopulate_mutable_logical_page_range(0, 32);
    double frontier = 0.0;
    for (std::uint64_t pass = 0; pass < 128; ++pass) {
        for (std::uint64_t page = 0; page < 32; ++page) {
            frontier = device.issue(write_request(
                "hot-rewrite", frontier, page, config.device.page_size_bytes)).finish_ns;
        }
        frontier = device.drain_pending("checkpoint", frontier).finish_ns;
        const auto stats = device.stats();
        const auto audit = device.audit_snapshot();
        require(stats.accounting_verified && audit.quiescent(),
                "bounded-metadata trace did not settle with valid accounting");
        require_equal(stats.gc_runs, 0, "pre-GC runs");
        require_equal(stats.block_erases, stats.auto_erase_requests, "pre-GC page-zero erases");
        require_equal(audit.materialized_pages.size(), 34,
                      "only 32 live data and 2 live mapping pages are retained");
        require(stats.invalid_pages >= (pass + 1) * 32,
                "discarding identity incorrectly freed invalid physical pages");
        require_equal(stats.compact_retired_logical_data_pages, 32,
                      "retired original compact pages");
    }
}

void test_block_aligned_population_preserves_write_parallelism() {
    auto config = deterministic_gc_config();
    config.device.planes_per_die = 4;
    config.device.blocks_per_plane = 64;
    config.device.pages_per_block = 32;
    config.host.mapping_entries_per_page = 32;
    HbfController device(config);
    // Every initial data block is full: no partially populated write frontier
    // is available to accidentally supply the missing plane parallelism.
    device.prepopulate_mutable_logical_page_range(0, 4 * config.device.pages_per_block);
    double finish = 0.0;
    for (std::uint64_t page = 0; page < 16; ++page) {
        finish = std::max(finish, device.issue(write_request(
            "aligned-overwrite", 0.0, page, config.device.page_size_bytes)).finish_ns);
    }
    finish = device.drain_pending("aligned-overwrite-drain", finish).finish_ns;
    const auto audit = device.audit_snapshot();
    std::vector<bool> touched(4, false);
    for (const auto& mapping : audit.logical_mappings) {
        if (mapping.key < 16) {
            const auto plane = mapping.ppn /
                (config.device.blocks_per_plane * config.device.pages_per_block);
            touched.at(plane) = true;
        }
    }
    require(std::all_of(touched.begin(), touched.end(), [](bool value) { return value; }),
            "block-aligned initial data collapsed foreground writes onto one plane");
    require(finish < 2 * config.device.t_erase_block_ns + 8 * config.device.t_program_page_ns,
            "independent foreground programs lost plane parallelism");
    require(device.stats().accounting_verified && audit.quiescent(),
            "parallel frontier allocation did not conserve pages after drain");
}

} // namespace

int main() {
    test_compact_mutable_matches_materialized_initial_image();
    test_pending_raw_erase_is_in_block_wear_snapshot();
    test_overwrite_metadata_is_bounded_before_any_gc();
    test_block_aligned_population_preserves_write_parallelism();
    return 0;
}
