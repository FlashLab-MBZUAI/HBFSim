#include "../../verification/probes/hbf_with_hbm.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace hbfsim::host;
using namespace hbfsim::physical;
using hbfsim::verification::HbfWithHbm;

static HbfConfig config(MappingCacheLayout layout) {
    HbfConfig c;
    c.device.stacks = 2;
    c.device.channels_per_stack = c.device.dies_per_channel = c.device.planes_per_die = 1;
    c.device.blocks_per_plane = 128;
    c.device.pages_per_block = 64;
    c.device.page_size_bytes = 4096;
    c.host.logical_capacity_bytes = 4096 * 4096;
    c.host.auto_gc_enabled = false;
    c.host.mapping_mode = MappingMode::Cached;
    c.host.mapping_cache_layout = layout;
    c.host.mapping_cache_tag_bytes = 32;
    c.host.mapping_scratch_pages = 1;
    c.host.ctrl_dram_bytes = 64 * 1024;
    return c;
}

static double issue(HbfWithHbm& d, Op op, std::uint64_t lpn, double now) {
    return d.issue(PhysicalRequest{.id = "rotated-vpn", .tier = Tier::HBF,
        .op = op, .address_space = AddressSpace::Logical, .arrival_ns = now,
        .addr = lpn * 4096, .bytes = 4096}).finish_ns;
}

int main() {
    try {
        // Group zero rotates lanes by one. Only odd LPNs are populated, in
        // random physical order; enumerating the unrotated lane sees holes.
        std::vector<std::uint64_t> lpns(512);
        for (std::uint64_t i = 0; i < lpns.size(); ++i) lpns[i] = 2 * i + 1;
        std::mt19937_64 rng(123);
        std::shuffle(lpns.begin(), lpns.end(), rng);
        HbfWithHbm compressed(config(MappingCacheLayout::Extent));
        if (compressed.stack_for_logical_page(1) != 0)
            throw std::runtime_error("regression requires a rotated mapping group");
        compressed.prepopulate_logical_pages(lpns);
        issue(compressed, Op::Read, 1, 0);
        const auto output = compressed.stats().mapping_compression_output_bytes;
        std::cout << "fragmented_mapping_bytes=" << output << '\n';
        if (output != 4096)
            throw std::runtime_error("compression inspected the wrong stack lane");

        HbfWithHbm entry(config(MappingCacheLayout::Entry));
        entry.prepopulate_mutable_logical_page_range(0, 4096);
        double now = 0;
        for (auto lpn : {1u, 3u, 5u, 7u}) now = issue(entry, Op::Write, lpn, now);
        std::uint64_t other = 1024;
        while (entry.stack_for_logical_page(other) != 0) ++other;
        now = issue(entry, Op::Read, other, now); // replace the mapping-page buffer
        const auto before = entry.execution_stats().mapping_buffer_patch_ops;
        (void)entry.drain_pending("checkpoint", now);
        const auto& s = entry.stats();
        const auto patches = s.mapping_buffer_patch_ops - before;
        std::cout << "dirty_entry_overlays=" << patches << '\n';
        if (patches != 4 || s.gc_runs || s.static_wear_leveling_runs || !s.accounting_verified)
            throw std::runtime_error("checkpoint failed to overlay rotated dirty entries");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
