#include "../../verification/probes/hbf_with_hbm.hpp"
// Static wear-leveling contract.
//
// A pinned population (weights, a resident prefix catalog) never invalidates
// its pages, so garbage collection alone concentrates every P/E cycle on the
// blocks that recycle the mutable KV pool. With static wear leveling enabled,
// the coldest fully valid block is migrated into a worn free block whenever a
// GC victim's post-erase count leads it by the configured gap, and the lightly
// worn block re-enters the free pool. This test checks that the mechanism
// (1) stays off at gap 0, (2) lifts every cold block off its original erase
// count and narrows the wear spread at a positive gap, (3) keeps the page,
// program, and erase conservation identities, and (4) leaves the device
// quiescent and audited, under both full-resident and cached mapping.

#include "host/hbf_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <optional>
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
using hbfsim::host::HbfStats;
using hbfsim::host::MappingMode;

constexpr std::uint64_t kPage = 4096;
constexpr std::uint32_t kPlanes = 2;
constexpr std::uint32_t kBlocksPerPlane = 64;
constexpr std::uint32_t kPagesPerBlock = 64;
constexpr std::uint64_t kColdBlocks = 80;
constexpr std::uint64_t kHotBlocks = 12;
constexpr int kPasses = 48;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

HbfConfig geometry(std::uint32_t gap, bool cached, std::uint32_t start = 0) {
    HbfConfig config;
    config.device.stacks = 1;
    config.device.channels_per_stack = 1;
    config.device.dies_per_channel = 1;
    config.device.planes_per_die = kPlanes;
    config.device.blocks_per_plane = kBlocksPerPlane;
    config.device.pages_per_block = kPagesPerBlock;
    config.device.page_size_bytes = kPage;
    config.device.oob_bytes_per_page = 0;
    config.host.mapping_entries_per_page = 512;


    config.host.gc_reserved_free_blocks_per_plane = 2;
    config.host.write_coalescing_enabled = false;
    config.device.ecc_decode_latency_ns = 1.0;
    config.device.ecc_encode_latency_ns = 1.0;
    config.device.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    config.device.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    config.host.static_wear_leveling_erase_gap = gap;
    config.host.static_wear_leveling_interval_erases = 1;
    config.host.static_wear_leveling_start_erases = start;
    if (cached) {
        config.host.mapping_mode = MappingMode::Cached;
        const std::uint64_t pages_per_stack =
            std::uint64_t{kPlanes} * kBlocksPerPlane * kPagesPerBlock;
        const std::uint64_t mapping_pages =
            (pages_per_stack + config.host.mapping_entries_per_page - 1) /
            config.host.mapping_entries_per_page;
        config.host.ctrl_dram_bytes = mapping_pages *
            config.host.mapping_directory_entry_bytes + 4 * kPage;
    }
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

struct Outcome {
    HbfStats stats;
    std::vector<std::uint32_t> erase_counts;
    std::vector<hbfsim::host::HbfAuditBlock> blocks;
    bool quiescent = false;
};

// A pinned cold image of kColdBlocks blocks plus a kHotBlocks-block mutable
// pool that is overwritten at random for kPasses full turnovers.
Outcome run(std::uint32_t gap, bool cached, unsigned seed,
            std::uint32_t start = 0, int passes = kPasses, bool guarded = false) {
    auto config = geometry(gap, cached, start);
    if (guarded) {
        config.host.static_wear_leveling_stop_gap = gap / 2;
        config.host.static_wear_leveling_cooldown_erases = 256;
        config.host.static_wear_leveling_max_write_fraction = .01;
    }
    HbfController device(config);
    const std::uint64_t cold_pages = kColdBlocks * kPagesPerBlock;
    const std::uint64_t hot_pages = kHotBlocks * kPagesPerBlock;
    device.prepopulate_mutable_logical_page_range(0, cold_pages + hot_pages);
    std::mt19937_64 rng(seed);
    double t = 0.0;
    for (int pass = 0; pass < passes; ++pass) {
        for (std::uint64_t index = 0; index < hot_pages; ++index) {
            const auto lpn = cold_pages + (rng() % hot_pages);
            const auto completion = device.issue(write(lpn, t));
            t = completion.finish_ns;
        }
    }
    (void)device.drain_pending("drain", t);
    Outcome outcome;
    outcome.stats = device.stats();
    outcome.erase_counts = device.block_erase_counts();
    auto audit = device.audit_snapshot();
    outcome.quiescent = audit.quiescent();
    outcome.blocks = std::move(audit.blocks);
    return outcome;
}

struct Spread {
    std::uint32_t min = 0;
    std::uint32_t max = 0;
    std::uint64_t zero_blocks = 0;
    double stddev = 0.0;
};

Spread spread_of(const std::vector<std::uint32_t>& counts) {
    Spread spread;
    spread.min = *std::min_element(counts.begin(), counts.end());
    spread.max = *std::max_element(counts.begin(), counts.end());
    spread.zero_blocks = static_cast<std::uint64_t>(
        std::count(counts.begin(), counts.end(), 0U));
    const double mean = std::accumulate(counts.begin(), counts.end(), 0.0) /
        static_cast<double>(counts.size());
    double variance = 0.0;
    for (const auto count : counts) {
        variance += (count - mean) * (count - mean);
    }
    spread.stddev = std::sqrt(variance / static_cast<double>(counts.size()));
    return spread;
}

void check_conservation(const Outcome& outcome, const std::string& label) {
    const auto& stats = outcome.stats;
    require(stats.accounting_verified,
            label + ": device audit did not verify");
    require(outcome.quiescent,
            label + ": device retained pending GC or mapping state");
    require(stats.gc_runs > 0, label + ": the overwrite stream never ran GC");
    require(stats.page_programs == stats.data_programs +
                stats.mapping_page_programs + stats.gc_relocations +
                stats.static_wear_leveling_relocations,
            label + ": page-program decomposition diverged");
    require(stats.physical_write_bytes == stats.page_programs * kPage,
            label + ": physical write bytes diverged from page programs");
    require(stats.physical_write_bytes == stats.data_program_payload_bytes +
                stats.mapping_program_payload_bytes +
                stats.gc_relocation_payload_bytes +
                stats.static_wear_leveling_relocation_payload_bytes,
            label + ": physical write payload decomposition diverged");
    require(stats.block_erases ==
                stats.erase_requests + stats.auto_erase_requests,
            label + ": physical erases must match explicit and page-zero autoerase requests");
    require(stats.gc_relocations + stats.gc_reclaimed_invalid_pages ==
                stats.gc_runs * kPagesPerBlock,
            label + ": GC victim page conservation diverged");
    require(stats.static_wear_leveling_relocations +
                    stats.static_wear_leveling_reclaimed_invalid_pages ==
                stats.static_wear_leveling_runs * kPagesPerBlock,
            label + ": wear-leveling victim page conservation diverged");
    require(stats.static_wear_leveling_relocation_payload_bytes ==
                stats.static_wear_leveling_relocations * kPage,
            label + ": wear-leveling payload bytes diverged");
    require(outcome.erase_counts.size() == kPlanes * kBlocksPerPlane,
            label + ": erase-count export lost blocks");
    require(std::accumulate(
                outcome.erase_counts.begin(),
                outcome.erase_counts.end(),
                std::uint64_t{0}) == stats.block_erases,
            label + ": per-block erase counts do not sum to block erases");
}

void test_selection_timing_and_hysteresis() {
    auto config = geometry(4, false);
    config.host.static_wear_leveling_stop_gap = 2;
    HbfController device(config);
    const auto cold_pages = kColdBlocks * kPagesPerBlock;
    const auto hot_pages = kHotBlocks * kPagesPerBlock;
    device.prepopulate_mutable_logical_page_range(0, cold_pages + hot_pages);
    std::mt19937_64 random(7);
    double arrival_ns = 0.0;
    std::optional<std::size_t> selected_gc_block;
    std::optional<std::size_t> erased_gc_block;
    std::uint64_t early_commands = 0;
    std::uint64_t continued_below_activation = 0;
    for (std::uint64_t index = 0; index < hot_pages * kPasses; ++index) {
        auto request = write(cold_pages + random() % hot_pages, arrival_ns);
        request.trace.mode = TraceMode::Full;
        const auto before_gc_runs = device.execution_stats().gc_runs;
        const auto completion = device.issue(request);
        arrival_ns = completion.finish_ns;
        std::optional<double> selection_done_ns;
        for (const auto& span : completion.spans) {
            if (span.name == "gc_victim_select") {
                selected_gc_block = std::stoull(span.detail.substr(5));
            } else if (span.name == "static_wear_leveling_select") {
                selection_done_ns = span.end_ns;
                if (erased_gc_block &&
                    device.block_erase_counts().at(*erased_gc_block) <
                        config.host.static_wear_leveling_erase_gap) {
                    continued_below_activation++;
                }
            } else if (selection_done_ns && span.name == "gc/cmd_addr_tsv" &&
                       span.start_ns < *selection_done_ns) {
                early_commands++;
            }
        }
        if (device.execution_stats().gc_runs > before_gc_runs) {
            erased_gc_block = selected_gc_block;
        }
        const auto& stats = device.execution_stats();
        require(stats.static_wear_leveling_runs ==
                    stats.static_wear_leveling_unique_source_blocks +
                    stats.static_wear_leveling_repeat_source_runs,
                "nonterminal source-migration telemetry does not conserve runs");
    }
    (void)device.drain_pending("selection-drain", arrival_ns);
    std::cout << "static_wear_leveling early_commands=" << early_commands
              << " continued_below_activation=" << continued_below_activation << '\n';
    require(early_commands == 0,
            "wear-leveling media command preceded source selection completion");
    require(continued_below_activation > 0,
            "engaged wear leveling still required the activation gap");
    require(device.audit_snapshot().quiescent(),
            "selection timing workload did not drain cleanly");
}

void test_restored_cooldown_and_partial_write_budget() {
    auto config = geometry(1, false);
    config.device.stacks = 2;
    config.host.static_wear_leveling_max_write_fraction = .01;
    HbfController initial(config);
    const auto cold_pages = kColdBlocks * kPagesPerBlock * config.device.stacks;
    const auto hot_pages = kHotBlocks * kPagesPerBlock;
    initial.prepopulate_mutable_logical_page_range(
        0, cold_pages + hot_pages * config.device.stacks);
    const auto image = initial.persistent_image();
    for (const bool cooldown : {false, true}) {
        config.host.static_wear_leveling_cooldown_erases = cooldown ? 1000000 : 0;
        HbfController device(config);
        device.restore_persistent_image(image);
        std::mt19937_64 random(19);
        double arrival_ns = 0.0;
        constexpr std::uint64_t writes = 8192;
        for (std::uint64_t index = 0; index < writes; ++index) {
            auto request = write(
                cold_pages + (random() % hot_pages) * config.device.stacks, arrival_ns);
            request.addr += kPage - 1;
            request.bytes = 2;
            arrival_ns = device.issue(request).finish_ns;
        }
        (void)device.drain_pending("restored-guard-drain", arrival_ns);
        const auto& stats = device.stats();
        require(stats.logical_write_bytes == 2 * writes,
                "partial writes were charged as whole-page host traffic");
        require(stats.gc_runs > 0 && stats.static_wear_leveling_checks > 0,
                "restored guard workload never checked wear leveling");
        require(stats.static_wear_leveling_runs == 0 &&
                    stats.static_wear_leveling_relocations == 0,
                "restored device bypassed its cooldown or payload-only budget");
        if (cooldown) {
            require(stats.block_erases < config.host.static_wear_leveling_cooldown_erases &&
                        stats.static_wear_leveling_cooldown_exclusions > 0,
                    "restored blocks did not observe their full cooldown interval");
        } else {
            require(stats.static_wear_leveling_budget_deferrals > 0,
                    "partial-write payload budget never deferred a migration");
        }
        require(stats.accounting_verified && device.audit_snapshot().quiescent(),
                "restored guard workload did not drain cleanly");
    }
}

void test_mapping_mode(bool cached) {
    const std::string mode = cached ? "cached" : "full-resident";
    const auto baseline = run(0, cached, 7);
    const auto leveled = run(4, cached, 7);
    check_conservation(baseline, mode + " gap0");
    check_conservation(leveled, mode + " gap4");
    const auto baseline_spread = spread_of(baseline.erase_counts);
    const auto leveled_spread = spread_of(leveled.erase_counts);
    std::cout << "static_wear_leveling mode=" << mode
              << " gap0: erases=" << baseline.stats.block_erases
              << " zero_blocks=" << baseline_spread.zero_blocks
              << " min=" << baseline_spread.min
              << " max=" << baseline_spread.max
              << " stddev=" << baseline_spread.stddev
              << " | gap4: erases=" << leveled.stats.block_erases
              << " wl_runs=" << leveled.stats.static_wear_leveling_runs
              << " wl_checks=" << leveled.stats.static_wear_leveling_checks
              << " zero_blocks=" << leveled_spread.zero_blocks
              << " min=" << leveled_spread.min
              << " max=" << leveled_spread.max
              << " stddev=" << leveled_spread.stddev
              << " waf=" << leveled.stats.waf().value_or(-1.0) << '\n';
    require(baseline.stats.static_wear_leveling_runs == 0 &&
                baseline.stats.static_wear_leveling_relocations == 0 &&
                baseline.stats.static_wear_leveling_checks == 0,
            mode + ": gap 0 must disable static wear leveling");
    require(baseline_spread.zero_blocks >= kColdBlocks,
            mode + ": without wear leveling the cold image must stay unworn");
    require(leveled.stats.static_wear_leveling_runs > 0,
            mode + ": gap 4 never migrated a cold block");
    require(leveled_spread.zero_blocks < baseline_spread.zero_blocks,
            mode + ": wear leveling did not reach previously unworn blocks");
    const auto continued = run(4, cached, 7, 0, 2 * kPasses);
    check_conservation(continued, mode + " gap4 extended horizon");
    require(continued.stats.static_wear_leveling_runs >
                leveled.stats.static_wear_leveling_runs,
            mode + ": continued writes made no further leveling progress");
    for (std::size_t index = 0; index < leveled.blocks.size(); ++index) {
        const auto& block = leveled.blocks[index];
        if (block.erase_count == 0 && block.free_pages == 0) {
            require(continued.erase_counts[index] > 0,
                    mode + ": an unworn closed block made no progress in the "
                    "extended workload");
        }
    }
    require(leveled_spread.max - leveled_spread.min <
                baseline_spread.max - baseline_spread.min,
            mode + ": wear leveling did not narrow the erase-count spread");
    require(leveled_spread.max <= baseline_spread.max,
            mode + ": wear leveling raised the maximum erase count");
    require(leveled_spread.stddev < baseline_spread.stddev,
            mode + ": wear leveling did not reduce erase-count dispersion");
    // The cold image is never overwritten, so foreground data programs and
    // logical bytes are identical; only internal migration traffic differs.
    require(leveled.stats.logical_write_bytes ==
                baseline.stats.logical_write_bytes,
            mode + ": wear leveling changed the host write stream");
    require(leveled.stats.data_programs == baseline.stats.data_programs,
            mode + ": wear leveling changed foreground data programs");
    // A late-start threshold beyond the window's reach must keep the gap rule
    // from firing while still counting its checks.
    const auto deferred = run(4, cached, 7, 1000);
    check_conservation(deferred, mode + " gap4 start1000");
    require(deferred.stats.static_wear_leveling_runs == 0 &&
                deferred.stats.static_wear_leveling_checks > 0,
            mode + ": a start threshold above the window must defer leveling");
    require(spread_of(deferred.erase_counts).zero_blocks >= kColdBlocks,
            mode + ": deferred leveling must leave the cold image unworn");
    const auto guarded = run(4, cached, 7, 12, 2 * kPasses, true);
    check_conservation(guarded, mode + " guarded");
    require(guarded.stats.static_wear_leveling_runs > 0,
            mode + ": guarded leveling never starts");
    require(guarded.stats.static_wear_leveling_relocation_payload_bytes <=
                guarded.stats.logical_write_bytes * .01 + 1,
            mode + ": guarded leveling exceeds 1% direct-copy budget");
    require(guarded.stats.static_wear_leveling_budget_deferrals > 0,
            mode + ": budget deferral was not exercised");
    require(guarded.stats.static_wear_leveling_runs ==
                guarded.stats.static_wear_leveling_unique_source_blocks +
                guarded.stats.static_wear_leveling_repeat_source_runs,
            mode + ": source-migration telemetry does not conserve runs");
}

} // namespace

int main() {
    try {
        test_selection_timing_and_hysteresis();
        test_restored_cooldown_and_partial_write_budget();
        test_mapping_mode(false);
        test_mapping_mode(true);
    } catch (const std::exception& error) {
        std::cerr << "hbf_static_wear_leveling_test failed: " << error.what()
                  << '\n';
        return 1;
    }
    std::cout << "hbf_static_wear_leveling_test passed\n";
    return 0;
}
