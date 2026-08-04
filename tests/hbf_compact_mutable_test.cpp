#include "physical/hbf/hbf_device.hpp"

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
using hbfsim::physical::hbf::HbfAuditSnapshot;
using hbfsim::physical::hbf::HbfConfig;
using hbfsim::physical::hbf::HbfDevice;
using hbfsim::physical::hbf::HbfStats;

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

HbfConfig deterministic_gc_config() {
    HbfConfig config;
    config.channels_per_stack = 1;
    config.dies_per_channel = 1;
    config.planes_per_die = 1;
    config.blocks_per_plane = 5;
    config.pages_per_block = 4;
    config.page_size_bytes = 512;
    config.oob_bytes_per_page = 0;
    config.mapping_entries_per_page = 4;
    config.read_buffer_pages = 0;
    config.gc_low_watermark_pages = 8;
    config.ecc_decode_latency_ns = 1.0;
    config.ecc_encode_latency_ns = 1.0;
    config.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    config.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    return config;
}

struct RunResult {
    std::vector<PhysicalCompletion> completions;
    HbfStats stats;
    HbfAuditSnapshot audit;
};

RunResult run_workload(bool compact) {
    const auto config = deterministic_gc_config();
    HbfDevice device(config);
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
    result.completions.push_back(device.issue(write_request(
        "overwrite-lpn0", 0.0, 0, config.page_size_bytes)));
    result.completions.push_back(device.drain_pending(
        "checkpoint-vpn0", result.completions.back().finish_ns));
    result.completions.push_back(device.issue(write_request(
        "append-lpn4", result.completions.back().finish_ns, 4,
        config.page_size_bytes)));
    result.completions.push_back(device.drain_pending(
        "checkpoint-vpn1", result.completions.back().finish_ns));
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

    require_equal(compact.stats.gc_runs, 1, "expected GC runs");
    require_equal(compact.stats.gc_relocations, 3, "expected GC relocations");
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

} // namespace

int main() {
    test_compact_mutable_matches_materialized_initial_image();
    return 0;
}
