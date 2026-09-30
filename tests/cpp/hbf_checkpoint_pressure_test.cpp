#include "app/system_config.hpp"
#include "host/hbf_controller.hpp"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace hbfsim::physical;
using namespace hbfsim::host;

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("pass the full-system config");
        hbfsim::app::SystemConfigBuilder builder;
        builder.apply_file(argv[1]);
        auto system = builder.resolve();
        auto config = system.hbf;
        // Preserve the full profile's media behavior while reproducing the
        // metadata-space feedback in a bounded component-test footprint.
        config.device.stacks = 2;
        config.device.channels_per_stack = 2;
        config.device.planes_per_die = 8;
        config.device.blocks_per_plane = 64;
        config.device.pages_per_block = 64;
        config.host.mapping_mode = MappingMode::FullResident;
        config.host.mapping_entries_per_page = 16;
        config.host.gc_low_watermark_pages = 128;
        config.host.gc_wear_leveling_weight = 0;
        config.host.write_buffer_pages = 64;
        config.host.write_buffer_flush_threshold_pages = 32;
        hbm::HbmDevice hbm(system.hbm);
        HbfController device(config);
        device.attach_hbm_buffer(hbm);
        const auto pages = device.logical_capacity_pages() / 64 * 64;
        device.prepopulate_mutable_logical_page_range(0, pages);
        const auto initial_valid = device.execution_stats().valid_pages;
        const TraceConfig trace{.mode = TraceMode::Off, .retain_completion_diagnostics = false};
        std::mt19937_64 random(20260912);
        double at = 0;
        std::uint64_t previous_gc = 0, largest_checkpoint_gc = 0;
        for (unsigned step = 0; step < 1024; ++step) {
            const auto slots = (random() % 10 < 9 ? pages / 8 : pages) / 64;
            const auto lpn = random() % slots * 64;
            at = device.issue(PhysicalRequest{
                .id = "pressure-write", .tier = Tier::HBF, .op = Op::Write,
                .address_space = AddressSpace::Logical, .trace = trace,
                .arrival_ns = at, .addr = lpn * 4096, .bytes = 64 * 4096,
            }).finish_ns;
            if ((step + 1) % 128) continue;
            const auto before = device.execution_stats().gc_runs;
            at = device.drain_pending("pressure-checkpoint", at, trace).finish_ns;
            const auto& stats = device.execution_stats();
            const auto checkpoint_gc = stats.gc_runs - before;
            largest_checkpoint_gc = std::max(largest_checkpoint_gc, checkpoint_gc);
            // For this overwrite trace, metadata persistence must not
            // reclaim more blocks than the preceding application window.
            // The unbounded checkpoint wave caused 1762 reclaims here,
            // versus 286 while serving the preceding writes.
            if (checkpoint_gc > before - previous_gc)
                throw std::runtime_error("checkpoint amplified GC beyond its foreground window");
            if (!device.quiescence_stats().quiescent() || stats.valid_pages != initial_valid)
                throw std::runtime_error("checkpoint lost live data or left unpublished state");
            previous_gc = stats.gc_runs;
        }
        const auto& stats = device.stats();
        if (!stats.accounting_verified || stats.gc_runs == 0)
            throw std::runtime_error("pressure test did not conserve actual GC traffic");
        std::cout << "checkpoint pressure: GC=" << stats.gc_runs
                  << " max checkpoint GC=" << largest_checkpoint_gc
                  << " valid pages=" << stats.valid_pages << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
