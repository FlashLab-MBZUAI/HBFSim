#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_controller.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace hbfsim::physical;
using namespace hbfsim::host;

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static HbfConfig config(MappingCacheLayout layout) {
    HbfConfig c;
    c.device.stacks = c.device.channels_per_stack = c.device.dies_per_channel = c.device.planes_per_die = 1;
    c.device.blocks_per_plane = 128;
    c.device.pages_per_block = 64;
    c.device.page_size_bytes = 4096;
    c.host.mapping_entries_per_page = 512;
    c.device.oob_bytes_per_page = 0;
    c.host.mapping_mode = MappingMode::Cached;
    c.host.mapping_cache_layout = layout;
    c.host.mapping_cache_tag_bytes = 32;
    c.host.mapping_codec_ns_per_entry = 1;
    c.host.mapping_scratch_pages = 1;
    c.host.ctrl_dram_bytes = 16 * 16 + 3 * (4096 + 32) + 4096;


    c.host.gc_low_watermark_pages = 64;
    c.host.logical_capacity_bytes = 6144 * 4096;
    return c;
}

static double issue(hbfsim::verification::HbfWithHbm& d, Op op, std::uint64_t lpn, double now) {
    return d.issue(PhysicalRequest{
        .id = "layout", .tier = Tier::HBF, .op = op,
        .address_space = AddressSpace::Logical, .arrival_ns = now,
        .addr = lpn * 4096, .bytes = 4096,
    }).finish_ns;
}

static void locality() {
    std::uint64_t misses[3]{};
    unsigned index = 0;
    for (auto layout : {MappingCacheLayout::Page, MappingCacheLayout::Entry, MappingCacheLayout::Extent}) {
        auto c = config(layout);
        hbfsim::verification::HbfWithHbm d(c);
        d.prepopulate_mutable_logical_page_range(0, 4096);
        double now = 0;
        for (unsigned pass = 0; pass < 3; ++pass)
            for (unsigned group = 0; group < 8; ++group)
                now = issue(d, Op::Read, group * 512, now);
        const auto& stats = d.stats();
        check(stats.accounting_verified, "locality accounting");
        check(stats.mapping_cache_peak_bytes <= stats.mapping_cache_capacity_bytes, "cache byte capacity");
        misses[index++] = stats.mapping_cache_misses;
        if (layout == MappingCacheLayout::Extent)
            check(stats.mapping_compression_output_bytes < stats.mapping_compression_input_bytes, "actual affine compression");
    }
    check(misses[1] < misses[0], "sparse entry cache should retain sparse mappings");
    check(misses[2] < misses[0], "compressed cache should retain regular mappings");
}

static void mutable_gc() {
    for (auto layout : {MappingCacheLayout::Page, MappingCacheLayout::Entry, MappingCacheLayout::Extent}) {
        auto c = config(layout);
        hbfsim::verification::HbfWithHbm d(c);
        d.prepopulate_mutable_logical_page_range(0, 6000);
        double now = 0;
        for (unsigned step = 0; step < 8500; ++step) {
            const auto lpn = (step * 131u) % 6000;
            now = issue(d, Op::Write, lpn, now);
            if (step % 31 == 0) now = issue(d, Op::Read, lpn, now);
        }
        now = d.drain_pending("layout-drain", now).finish_ns;
        const auto& stats = d.stats();
        check(stats.accounting_verified, "mutable accounting");
        check(stats.gc_runs > 0, "sustained writes must reach GC");
        check(stats.mapping_cache_peak_bytes <= stats.mapping_cache_capacity_bytes, "mutable cache bytes");
        check(stats.page_programs == stats.data_programs + stats.mapping_page_programs +
            stats.gc_relocations + stats.static_wear_leveling_relocations, "program conservation");
        if (layout == MappingCacheLayout::Entry)
            check(stats.mapping_merge_media_reads > 0, "entry dirty merging must read flash");
        std::cout << to_string(layout) << " gc=" << stats.gc_runs << " mapping_reads="
                  << stats.mapping_media_reads << " merge_reads=" << stats.mapping_merge_media_reads << '\n';
        auto image = d.persistent_image();
        hbfsim::verification::HbfWithHbm restored(c);
        restored.restore_persistent_image(image);
        auto resumed = issue(restored, Op::Read, (8499u * 131u) % 6000, 0);
        resumed = issue(restored, Op::Write, 1, resumed);
        resumed = restored.drain_pending("resumed-layout-drain", resumed).finish_ns;
        check(restored.stats().accounting_verified, "restored layout accounting");
        (void)now;
    }
}

static void zero_cost_codec() {
    auto c = config(MappingCacheLayout::Extent);
    c.host.mapping_codec_ns_per_entry = 0;
    hbfsim::verification::HbfWithHbm d(c);
    d.prepopulate_mutable_logical_page_range(0, 4096);
    issue(d, Op::Read, 0, 0);
    const auto& s = d.stats();
    check(s.mapping_media_reads > 0, "zero codec still reads mapping media");
    check(s.mapping_compression_output_bytes < s.mapping_compression_input_bytes,
          "zero codec still encodes mapping");
    check(s.mapping_codec_work_ns == 0 && s.accounting_verified,
          "zero codec must not reserve an invalid empty service interval");
}

static void shared_translation_fill() {
    auto c = config(MappingCacheLayout::Entry);
    hbfsim::verification::HbfWithHbm d(c);
    d.prepopulate_mutable_logical_page_range(0, 4096);
    double done = 0;
    // More distinct LPNs than fit in the entry cache, all in one mapping page.
    // Simultaneous misses must share the in-flight page; subsequent misses
    // retain spatial locality even after individual entries were evicted.
    for (std::uint64_t lpn = 0; lpn < 512; ++lpn)
        done = std::max(done, issue(d, Op::Read, lpn, 0));
    const auto first = d.stats();
    check(first.mapping_media_reads == 1, "one mapping fill for adjacent concurrent LPNs");
    check(first.mapping_cache_buffer_hits == 511, "retain the entire mapping-page fill");
    check(first.mapping_buffer_coalesced_reads > 0, "coalesce in-flight misses by mapping VPN");
    done = issue(d, Op::Read, 512, done);  // replace the single page buffer
    done = issue(d, Op::Read, 0, done);   // actual cold refill, not a stale buffer hit
    check(d.stats().mapping_media_reads == 3, "bounded page-buffer eviction");
    done = issue(d, Op::Write, 0, done);
    done = issue(d, Op::Read, 1, done);
    check(d.stats().mapping_media_reads == 3, "patch a retained mapping page on update");
    done = d.drain_pending("shared-fill-drain", done).finish_ns;
    const auto clean = d.stats();
    check(clean.mapping_merge_buffer_hits > 0, "checkpoint can reuse a coherent full mapping page");
    // A completed checkpoint must clean sibling entry state as well.
    for (std::uint64_t group = 1; group < 8; ++group)
        done = issue(d, Op::Read, group * 512, done);
    check(d.stats().mapping_page_programs == clean.mapping_page_programs,
          "evicting checkpointed entries must not repeat the writeback");
    check(d.stats().mapping_scratch_capacity_bytes == c.device.page_size_bytes + c.host.mapping_cache_tag_bytes,
          "retained page tags are inside the scratch budget");
    check(d.stats().mapping_directory_entry_bytes == c.host.mapping_directory_entry_bytes + 8,
          "checkpoint epochs are inside the directory budget");
}

static void compressed_update_without_flash_refetch() {
    auto c = config(MappingCacheLayout::Extent);
    hbfsim::verification::HbfWithHbm d(c);
    d.prepopulate_mutable_logical_page_range(0, 4096);
    auto done = issue(d, Op::Read, 0, 0);
    const auto before = d.stats();
    done = issue(d, Op::Write, 0, done);
    const auto after = d.stats();
    check(after.mapping_media_reads == before.mapping_media_reads,
          "resident compressed mappings must decode without a flash refetch");
    check(after.mapping_cache_promotions == 1, "one in-memory dense promotion");
    check(after.mapping_codec_work_ns > before.mapping_codec_work_ns, "decoding is charged");
    check(after.mapping_cache_peak_bytes <= after.mapping_cache_capacity_bytes,
          "dense expansion must fit the same budget");
    (void)d.drain_pending("compressed-update-drain", done);
    check(d.stats().accounting_verified, "compressed update accounting");
}

static void checkpointed_sibling_generation() {
    auto c = config(MappingCacheLayout::Entry);
    hbfsim::verification::HbfWithHbm d(c);
    d.prepopulate_mutable_logical_page_range(0, 4096);
    auto done = issue(d, Op::Write, 0, 0);
    done = issue(d, Op::Write, 1, done);
    done = d.drain_pending("siblings-checkpoint", done).finish_ns;
    const auto programs = d.stats().mapping_page_programs;
    done = issue(d, Op::Write, 1, done);  // same VPN becomes dirty again
    const auto entries = d.stats().mapping_cache_capacity_bytes / (8 + c.host.mapping_cache_tag_bytes);
    check(entries < 512, "sibling test must evict within one mapping page");
    for (std::uint64_t lpn = 2; lpn <= entries; ++lpn)
        done = issue(d, Op::Read, lpn, done);  // evict only the older entry 0
    check(d.stats().mapping_page_programs == programs,
          "a new dirty sibling must not re-dirty checkpointed entries");
    done = d.drain_pending("siblings-final", done).finish_ns;
    check(d.stats().mapping_page_programs == programs + 1,
          "the newly modified sibling still requires one checkpoint");
    check(d.stats().accounting_verified, "sibling generation accounting");
}

int main() {
    try {
        locality(); shared_translation_fill(); compressed_update_without_flash_refetch();
        checkpointed_sibling_generation();
        mutable_gc(); zero_cost_codec(); std::cout << "mapping layouts passed\n";
    }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
