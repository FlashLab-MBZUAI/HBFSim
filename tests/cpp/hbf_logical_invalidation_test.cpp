#include "../../verification/probes/hbf_with_hbm.hpp"

#include <iostream>
#include <stdexcept>

using namespace hbfsim::host;
using namespace hbfsim::physical;
using Device = hbfsim::verification::HbfWithHbm;
constexpr std::uint64_t page_bytes = 4096;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

HbfConfig geometry() {
    HbfConfig c;
    c.device.stacks = 2;
    c.device.channels_per_stack = c.device.dies_per_channel = c.device.planes_per_die = 1;
    c.device.blocks_per_plane = 32;
    c.device.pages_per_block = 8;
    c.device.page_size_bytes = page_bytes;
    c.host.mapping_entries_per_page = 4;
    return c;
}

PhysicalCompletion access(Device& device, Op op, std::uint64_t lpn, std::uint64_t bytes = page_bytes) {
    return device.issue(PhysicalRequest{.id = "access", .tier = Tier::HBF, .op = op,
        .arrival_ns = device.execution_stats().finish_ns, .addr = lpn * page_bytes, .bytes = bytes});
}

void compact_lifecycle() {
    auto c = geometry();
    Device d(c);
    d.prepopulate_mutable_logical_page_range(0, 16);
    const auto before = d.stats();
    const auto trim = d.invalidate_logical_pages(3, 6, 0);
    const auto after = d.stats();
    require(trim.invalidated_pages == 6 && trim.unmapped_pages == 0, "compact trim count");
    require(after.free_pages == before.free_pages && after.valid_pages + 6 == before.valid_pages &&
        after.invalid_pages == before.invalid_pages + 6, "invalidation must not erase/free media");
    require(after.physical_write_bytes == 0 && after.mapping_update_ops == 6,
        "trim must charge map updates without programming payload");
    require(access(d, Op::Read, 3).note == "unmapped-erased-read", "compact read shortcut resurrected trim");
    require(access(d, Op::Read, 2).note != "unmapped-erased-read", "trim damaged neighbor");
    require(d.invalidate_logical_pages(3, 6, d.execution_stats().finish_ns).unmapped_pages == 6,
        "repeat trim must be idempotent");
    (void)access(d, Op::Write, 4);
    (void)d.invalidate_logical_pages(4, 1, d.execution_stats().finish_ns);
    require(access(d, Op::Read, 4).note == "unmapped-erased-read", "materialized mapping survived trim");
    (void)d.drain_pending("persist-trim", d.execution_stats().finish_ns);
    Device restored(c);
    restored.restore_persistent_image(d.persistent_image());
    require(access(restored, Op::Read, 4).note == "unmapped-erased-read", "restart resurrected a retired LPN");
    (void)access(restored, Op::Write, 4);
    require(access(restored, Op::Read, 4).note != "unmapped-erased-read", "freed LPN cannot be reused");
    require(restored.stats().accounting_verified, "restored accounting failed");
}

void buffered_death() {
    auto c = geometry();
    c.host.write_coalescing_enabled = true;
    c.host.write_buffer_pages = 16;
    c.host.write_buffer_flush_threshold_pages = 0;
    Device d(c);
    d.prepopulate_mutable_logical_page_range(0, 1);
    (void)access(d, Op::Write, 0, 64);
    (void)access(d, Op::Write, 1, 128);
    const auto trim = d.invalidate_logical_pages(0, 1, d.execution_stats().finish_ns);
    require(trim.invalidated_pages == 1 && trim.discarded_buffer_pages == 1 &&
        trim.discarded_buffer_bytes == 64, "buffered overwrite and old mapping must both die");
    require(d.quiescence_stats().write_buffer_entries == 1, "trim discarded a live neighbor buffer");
    require(d.execution_stats().data_program_payload_bytes == 0, "free flushed dead buffer data");
    (void)d.drain_pending("persist-live", d.execution_stats().finish_ns);
    require(d.execution_stats().data_program_payload_bytes == page_bytes, "only surviving buffered page should program");
    require(d.stats().accounting_verified, "buffer discard accounting failed");
}

void reject_without_mutation() {
    auto c = geometry();
    Device readonly(c);
    readonly.prepopulate_read_only_logical_page_range(0, 8);
    bool rejected = false;
    try { (void)readonly.invalidate_logical_pages(0, 1, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && readonly.stats().invalidations == 0, "read-only image accepted free");
    Device d(c);
    d.prepopulate_mutable_logical_page_range(0, 8);
    rejected = false;
    try { (void)d.invalidate_logical_pages(0, d.logical_capacity_pages() + 1, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && d.stats().invalidations == 0, "invalid range partially freed data");
}

void gc_does_not_resurrect(bool cached) {
    auto c = geometry();
    if (cached) {
        c.host.mapping_mode = MappingMode::Cached;
        // One cached mapping page and one GC staging page per stack, plus directory.
        c.host.ctrl_dram_bytes = 2 * (512 + 4096 + 4096);
    }
    Device d(c);
    d.prepopulate_mutable_logical_page_range(0, 128);
    (void)d.invalidate_logical_pages(0, 32, 0);
    for (unsigned i = 0; i < 800; ++i) {
        (void)access(d, Op::Write, 32 + i * 37 % 64);
        if (i % 40 == 39)
            (void)d.invalidate_logical_pages(32, 16, d.execution_stats().finish_ns);
    }
    (void)d.drain_pending("gc-persist", d.execution_stats().finish_ns);
    const auto stats = d.stats();
    require(stats.gc_runs > 0 && stats.accounting_verified, "test did not reach valid GC accounting");
    if (cached) require(stats.mapping_cache_dirty_evictions > 0, "test did not evict dirty mapping pages");
    Device restored(c);
    restored.restore_persistent_image(d.persistent_image());
    for (unsigned lpn = 0; lpn < 32; ++lpn)
        require(access(restored, Op::Read, lpn).note == "unmapped-erased-read", "GC/restart resurrected deallocated data");
    require(access(restored, Op::Read, 127).note != "unmapped-erased-read", "GC lost cold live data");
}

int main() {
    try {
        compact_lifecycle();
        buffered_death();
        reject_without_mutation();
        gc_does_not_resurrect(false);
        gc_does_not_resurrect(true);
        std::cout << "logical invalidation: compact, buffered, restored, resident/cached GC passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
