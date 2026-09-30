// HBF on-device DRAM: a clean read cache of physical NAND pages plus the
// write buffer, with a hit-latency + bandwidth port, separate from the HBM
// controller reservation. Covers both the page FTL and a structural policy.
#include "../../verification/probes/hbf_with_hbm.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hbfsim::host {
struct HbfDeviceDramTestAccess {
    static bool resident(const HbfController& h, std::uint64_t ppn) {
        for (const auto& lines : h.device_dram_by_stack_)
            if (lines.contains(ppn)) return true;
        return false;
    }
    static std::size_t lines(const HbfController& h) {
        std::size_t count = 0;
        for (const auto& stack : h.device_dram_by_stack_) count += stack.size();
        return count;
    }
};
} // namespace hbfsim::host

using namespace hbfsim::host;
using namespace hbfsim::physical;
using Device = hbfsim::verification::HbfWithHbm;
using Access = HbfDeviceDramTestAccess;
using U = std::uint64_t;

namespace {
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F fn, const std::string& message) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}

// One stack of 64 blocks x 8 pages = 512 raw pages. Denominator 64 gives an
// 8-page device DRAM: 2 write-buffer pages and a 6-page read cache.
HbfConfig config(U denominator, MappingOrganization organization = MappingOrganization::Page,
        bool coalescing = true) {
    HbfConfig c;
    c.host.mapping_organization = organization;
    c.device.stacks = c.device.channels_per_stack = c.device.dies_per_channel = c.device.planes_per_die = 1;
    c.device.blocks_per_plane = 64;
    c.device.pages_per_block = 8;
    c.device.t_program_page_ns = 750;
    c.device.t_erase_block_ns = 2000;
    c.device.t_read_page_ns = 40;
    c.host.mapping_mode = MappingMode::FullResident;
    c.host.logical_capacity_bytes = 128 * 4096;
    c.host.gc_reserved_free_blocks_per_plane = 2;
    c.host.mapping_log_blocks = 2;
    c.host.write_coalescing_enabled = coalescing;
    c.host.write_buffer_pages = coalescing ? 2 : 0;
    c.host.write_buffer_completion_requires_flush = false;
    c.host.device_dram_capacity_denominator = denominator;
    return c;
}

double io(Device& d, Op op, U lpn, double now, U bytes = 4096) {
    return d.issue(PhysicalRequest{.id = "device-dram", .tier = Tier::HBF, .op = op,
        .address_space = AddressSpace::Logical, .arrival_ns = now,
        .addr = lpn * 4096, .bytes = bytes}).finish_ns;
}

U hbm_bytes(const Device& d) {
    return d.execution_stats().host_hbm_read_bytes + d.execution_stats().host_hbm_write_bytes;
}

void capacity_and_budget() {
    const auto capacity = derive_device_dram_capacity(config(64));
    check(capacity.pages_per_stack == 8 && capacity.write_buffer_pages_per_stack == 2 &&
        capacity.read_cache_pages_per_stack == 6 && capacity.total_bytes == 8 * 4096,
        "device DRAM capacity is floor(raw pages / denominator) with the write buffer carved out");
    check(derive_device_dram_capacity(config(0)).pages_per_stack == 0, "denominator 0 disables device DRAM");
    rejects([] { (void)derive_device_dram_capacity(config(256)); },
        "a device DRAM no larger than the write buffer was accepted");
    // The write buffer leaves the HBM reservation for both organizations.
    Device hbm(config(0)), dram(config(64));
    check(hbm.config().host.ctrl_dram_bytes - dram.config().host.ctrl_dram_bytes == 2 * 4096,
        "page FTL HBM budget still holds the device-DRAM write buffer");
    check(dram.stats().write_buffer_capacity_bytes == 0 && hbm.stats().write_buffer_capacity_bytes == 2 * 4096,
        "HBM write-buffer capacity was reported for a device-DRAM write buffer");
    check(dram.stats().device_dram_capacity_bytes == 8 * 4096 &&
        dram.stats().device_dram_read_cache_pages_per_stack == 6, "device DRAM capacity stats");
    Device structural_hbm(config(0, MappingOrganization::Extent)),
        structural_dram(config(64, MappingOrganization::Extent));
    check(structural_hbm.config().host.ctrl_dram_bytes - structural_dram.config().host.ctrl_dram_bytes == 2 * 4096,
        "structural HBM budget still holds the device-DRAM write buffer");
}

void hit_versus_miss_timing() {
    auto slow = config(64);
    slow.host.device_dram_hit_latency_ns = 1090;
    Device plain(config(0)), fast(config(64)), late(slow);
    for (Device* d : {&plain, &fast, &late}) d->prepopulate_logical_pages({0, 1, 2});
    const double first = 1e6, second = 2e6;
    const double plain_miss = io(plain, Op::Read, 0, first) - first;
    const double fast_miss = io(fast, Op::Read, 0, first) - first;
    const double late_miss = io(late, Op::Read, 0, first) - first;
    check(fast_miss == plain_miss && late_miss == plain_miss,
        "a device-DRAM fill delayed the missing read");
    // Two more pages evict page 0 from the die's two NAND bank buffers, so
    // the next read of it is answered by the device DRAM alone.
    const auto evict_bank_buffers = [&](Device& d) {
        (void)io(d, Op::Read, 1, first + 1e5);
        (void)io(d, Op::Read, 2, first + 2e5);
    };
    for (Device* d : {&plain, &fast, &late}) evict_bank_buffers(*d);
    const auto reads = fast.execution_stats().page_reads;
    const auto bank_hits = fast.execution_stats().read_buffer_hits;
    const double fast_hit = io(fast, Op::Read, 0, second) - second;
    const double late_hit = io(late, Op::Read, 0, second) - second;
    (void)io(plain, Op::Read, 0, second);
    const auto& s = fast.execution_stats();
    check(s.device_dram_read_misses == 3 && s.device_dram_read_hits == 1 && s.device_dram_fills == 3,
        "device DRAM hit/miss/fill counters");
    check(s.page_reads == reads && s.read_buffer_hits == bank_hits,
        "a device-DRAM hit reached the NAND bank");
    check(fast_hit < fast_miss, "a device-DRAM hit was not faster than NAND");
    check(std::abs((late_hit - fast_hit) - 1000.0) < 1e-6,
        "device-DRAM hit latency is not on the hit's critical path");
    check(s.device_dram_write_bytes == 3 * 4096 && s.device_dram_read_bytes == 4096,
        "fills write a page each and a full-page hit reads one over the DRAM port");
    // A slower port stretches the hit by the transfer-time difference.
    auto narrow = config(64);
    narrow.host.device_dram_bandwidth_GBps = 4.096;
    Device thin(narrow);
    thin.prepopulate_logical_pages({0, 1, 2});
    (void)io(thin, Op::Read, 0, first);
    evict_bank_buffers(thin);
    const double thin_hit = io(thin, Op::Read, 0, second) - second;
    check(std::abs((thin_hit - fast_hit) - (1000.0 - 4096 / 204.8)) < 1e-6,
        "device-DRAM hit does not charge bytes / bandwidth");
    check(hbm_bytes(fast) == hbm_bytes(plain), "device-DRAM reads moved HBM traffic");
}

// Six read-cache pages. Fill 0..5, touch 0, then fill 6: LRU evicts 1 and
// FIFO evicts 0.
void eviction(DeviceDramPolicy policy) {
    auto c = config(64);
    c.host.device_dram_policy = policy;
    Device d(c);
    d.prepopulate_logical_pages({0, 1, 2, 3, 4, 5, 6});
    double now = 1e6;
    for (U lpn = 0; lpn < 6; ++lpn, now += 1e6) (void)io(d, Op::Read, lpn, now);
    (void)io(d, Op::Read, 0, now); now += 1e6;
    (void)io(d, Op::Read, 6, now); now += 1e6;
    const auto& s = d.execution_stats();
    check(s.device_dram_fills == 7 && s.device_dram_evictions == 1 && Access::lines(d) == 6,
        "device-DRAM read cache exceeded its capacity");
    const auto zero = *d.mapped_physical_page(0), one = *d.mapped_physical_page(1);
    if (policy == DeviceDramPolicy::Lru)
        check(Access::resident(d, zero) && !Access::resident(d, one), "LRU did not evict the least recent page");
    else
        check(!Access::resident(d, zero) && Access::resident(d, one), "FIFO did not evict the oldest fill");
    const auto hits = s.device_dram_read_hits;
    (void)io(d, Op::Read, 0, now);
    check(d.execution_stats().device_dram_read_hits == hits + (policy == DeviceDramPolicy::Lru),
        "eviction choice did not decide the next hit");
}

void write_buffer_lives_in_device_dram() {
    Device hbm(config(0)), dram(config(64));
    double hbm_ack = 0, dram_ack = 0;
    for (Device* d : {&hbm, &dram}) {
        const double ack = io(*d, Op::Write, 0, 1e6) - 1e6;
        (d == &hbm ? hbm_ack : dram_ack) = ack;
        (void)d->drain_pending("drain", 2e6);
    }
    const auto& h = hbm.execution_stats();
    const auto& s = dram.execution_stats();
    check(s.write_buffer_dram_write_bytes == h.write_buffer_dram_write_bytes &&
        s.write_buffer_dram_read_bytes == h.write_buffer_dram_read_bytes &&
        h.write_buffer_dram_write_bytes == 4096 && h.write_buffer_dram_read_bytes == 4096,
        "write-buffer staging/flush volume changed with its location");
    check(hbm_bytes(hbm) - hbm_bytes(dram) == 8192,
        "device-DRAM write buffer still moved its payload through HBM");
    check(s.device_dram_write_bytes == 4096 && s.device_dram_read_bytes == 4096,
        "write-buffer staging and flush did not use the device-DRAM port");
    check(dram_ack < 750, "a write did not complete on entry into device DRAM");
    check(s.physical_write_bytes == h.physical_write_bytes && s.data_programs == 1,
        "device-DRAM flush changed the programmed media volume");
    // Write-allocate: the flushed page is a clean line, and reading it back
    // does not touch NAND.
    const auto ppn = *dram.mapped_physical_page(0);
    check(Access::resident(dram, ppn) && s.device_dram_fills == 1, "flushed page was not kept in device DRAM");
    const auto reads = s.page_reads;
    (void)io(dram, Op::Read, 0, 3e6);
    check(dram.execution_stats().device_dram_read_hits == 1 && dram.execution_stats().page_reads == reads,
        "read of a flushed page missed the device DRAM");
    // A buffered page is staged and served in device DRAM: against the HBM
    // write buffer, 4 KiB of staging and 4 KiB of read leave HBM.
    const auto hbm_before = hbm_bytes(hbm), dram_before = hbm_bytes(dram);
    for (Device* d : {&hbm, &dram}) {
        (void)io(*d, Op::Write, 1, 4e6);
        (void)io(*d, Op::Read, 1, 5e6);
        check(d->execution_stats().write_buffer_read_hits == 1, "buffered read was not served by the write buffer");
    }
    check((hbm_bytes(hbm) - hbm_before) - (hbm_bytes(dram) - dram_before) == 8192,
        "buffered read or staging moved HBM traffic with a device-DRAM write buffer");
}

void invalidation_purges() {
    Device d(config(64));
    d.prepopulate_logical_pages({0, 1});
    (void)io(d, Op::Read, 0, 1e6);
    (void)io(d, Op::Read, 1, 2e6);
    const auto zero = *d.mapped_physical_page(0), one = *d.mapped_physical_page(1);
    check(Access::resident(d, zero) && Access::resident(d, one), "reads did not fill device DRAM");
    // TRIM retires the line immediately.
    (void)d.invalidate_logical_pages(0, 1, 3e6);
    check(!Access::resident(d, zero), "TRIM left a dead page in device DRAM");
    // An overwrite retires the old version once it publishes; the new
    // version stays cached clean.
    (void)io(d, Op::Write, 1, 4e6);
    (void)d.drain_pending("overwrite", 5e6);
    const auto moved = *d.mapped_physical_page(1);
    check(moved != one && !Access::resident(d, one) && Access::resident(d, moved),
        "overwrite did not purge the old version or keep the new one");
}

void structural_policy_shares_dram() {
    // Without coalescing the policy reads through the same read cache.
    Device d(config(64, MappingOrganization::Extent, false));
    double now = 0;
    for (U lpn = 0; lpn < 3; ++lpn) now = io(d, Op::Write, lpn, now);
    now = d.drain_pending("seed", now).finish_ns + 1e6;
    const auto reads = d.execution_stats().page_reads;
    const double miss = io(d, Op::Read, 0, now) - now;
    // Two more pages evict page 0 from the die's two NAND bank buffers.
    for (U lpn = 1; lpn < 3; ++lpn) (void)io(d, Op::Read, lpn, now += 1e5);
    now += 1e6;
    const double hit = io(d, Op::Read, 0, now) - now;
    const auto& s = d.execution_stats();
    check(s.device_dram_read_misses == 3 && s.device_dram_read_hits == 1 && s.page_reads == reads + 3,
        "structural reads did not use the device DRAM read cache");
    check(hit < miss && Access::resident(d, *d.mapped_physical_page(0)), "structural device-DRAM hit timing");
    // With coalescing its write buffer lives in device DRAM, not HBM.
    Device hbm(config(0, MappingOrganization::Extent)), dram(config(64, MappingOrganization::Extent));
    for (Device* x : {&hbm, &dram}) {
        (void)io(*x, Op::Write, 0, 1e6);
        (void)x->drain_pending("flush", 2e6);
    }
    check(hbm_bytes(hbm) - hbm_bytes(dram) == 8192,
        "structural write buffer still moved its payload through HBM");
    check(dram.execution_stats().device_dram_write_bytes == 4096 &&
        dram.execution_stats().device_dram_read_bytes == 4096,
        "structural write buffer did not use the device-DRAM port");
    const auto ppn = *dram.mapped_physical_page(0);
    check(Access::resident(dram, ppn), "structural flush did not write-allocate");
    (void)io(dram, Op::Read, 0, 3e6);
    check(dram.execution_stats().device_dram_read_hits == 1, "structural read of a flushed page missed");
    // Invalidation reaches the shared cache through the controller.
    (void)dram.invalidate_logical_pages(0, 1, 4e6);
    check(!Access::resident(dram, ppn), "structural invalidation left a dead page cached");
}

// One stack with eight dies (4 channels x 2 dies), so decoded pages arrive
// faster than the DRAM port can absorb fills. Denominator 64 of the 1024 raw
// pages gives 16 DRAM pages: 2 write-buffer pages and a 14-page read cache.
HbfConfig parallel_config(U denominator, double bandwidth_GBps = 204.8) {
    auto c = config(denominator);
    c.device.channels_per_stack = 4;
    c.device.dies_per_channel = 2;
    c.device.blocks_per_plane = 16;
    c.host.device_dram_bandwidth_GBps = bandwidth_GBps;
    return c;
}

struct Latencies {
    std::vector<double> reads, writes;
    double read_dram_work_ns = 0;
};

// 120 distinct pages (all misses) at 20 ns spacing, with a 4 KiB write after
// every 15 reads.
Latencies all_miss_stream(Device& d) {
    std::vector<U> pages(120);
    for (U lpn = 0; lpn < pages.size(); ++lpn) pages[lpn] = lpn;
    d.prepopulate_logical_pages(pages);
    Latencies out;
    double now = 1e6;
    for (U lpn = 0; lpn < 120; ++lpn, now += 20) {
        const auto read = d.issue(PhysicalRequest{.id = "miss", .tier = Tier::HBF, .op = Op::Read,
            .address_space = AddressSpace::Logical, .arrival_ns = now, .addr = lpn * 4096, .bytes = 4096});
        out.reads.push_back(read.finish_ns - now);
        out.read_dram_work_ns += read.breakdown.write_buffer_dram_ns;
        if (lpn % 15 == 14) {
            now += 20;
            out.writes.push_back(io(d, Op::Write, 120 + lpn / 15, now) - now);
        }
    }
    return out;
}

double mean(const std::vector<double>& values) {
    double sum = 0;
    for (const auto value : values) sum += value;
    return sum / static_cast<double>(values.size());
}

// Read-miss fills are background work: they take idle port time or are
// dropped, never queue, never delay write-buffer staging or hits, and are
// not charged to the reads that decoded the pages.
void fills_never_delay_host_traffic() {
    Device plain(parallel_config(0)), dram(parallel_config(64)), wide(parallel_config(64, 204800.0));
    const auto p = all_miss_stream(plain), d = all_miss_stream(dram), w = all_miss_stream(wide);
    const auto& s = dram.execution_stats();
    check(s.device_dram_read_hits == 0 && s.device_dram_read_misses == 120, "the miss stream hit device DRAM");
    check(s.device_dram_fill_bypasses > 0 && s.device_dram_fills + s.device_dram_fill_bypasses >= 120,
        "a saturated port did not bypass fills");
    check(wide.execution_stats().device_dram_fill_bypasses == 0 &&
        wide.execution_stats().device_dram_fills >= 120, "an idle port bypassed fills");
    check(d.read_dram_work_ns == 0.0, "fills were charged to the reads' breakdown");
    for (std::size_t i = 0; i < p.writes.size(); ++i) {
        // The device-DRAM write buffer stages over HBIO and its port instead
        // of the HBM reservation, a difference of tens of nanoseconds; a
        // queued fill backlog would add microseconds.
        check(std::abs(d.writes[i] - p.writes[i]) < 300.0,
            "write acknowledgement waited behind device-DRAM fills (" +
            std::to_string(d.writes[i]) + " vs " + std::to_string(p.writes[i]) + " ns)");
        // A port 1000x faster only shortens the write path's own 4 KiB
        // transfers (staging and flush read, compounded through the slot
        // wait on each evicted predecessor), never a fill backlog.
        const double own_transfers_ns = 2.0 * 4096 / 204.8 * static_cast<double>(i + 1);
        check(d.writes[i] >= w.writes[i] - 1e-6 && d.writes[i] - w.writes[i] <= own_transfers_ns + 1.0,
            "write latency depends on the fill backlog (" + std::to_string(d.writes[i]) + " vs " +
            std::to_string(w.writes[i]) + " ns)");
    }
    // Misses take the NAND path in every device; only the interleaved
    // flushes' program timing (shifted by their own DRAM transfers) moves
    // individual reads, so the streams agree on average.
    check(std::abs(mean(d.reads) - mean(w.reads)) < 100.0, "miss latency depends on the fill backlog");
    check(std::abs(mean(d.reads) - mean(p.reads)) < 300.0, "misses slowed down with device DRAM");
}

// Four pages read round-robin at 50 ns spacing. On one die the two NAND
// bank buffers thrash and, after the first pass, every read is a device-DRAM
// hit, faster than the NAND read it replaces. On eight dies the bank buffers,
// the closer copies, answer first and the device DRAM only replaces the NAND
// re-reads of pages they dropped.
void hits_are_not_slower_than_nand() {
    const auto stream = [](Device& device) {
        device.prepopulate_logical_pages({0, 1, 2, 3});
        std::vector<double> latencies;
        double now = 1e6;
        for (U step = 0; step < 200; ++step, now += 50)
            latencies.push_back(io(device, Op::Read, step % 4, now) - now);
        return latencies;
    };
    Device plain(config(0)), dram(config(64));
    const auto p = stream(plain), d = stream(dram);
    // With the array idle, a bank buffer keeps some pages long enough to
    // answer again; every other repeat is a device-DRAM hit.
    const auto& s = dram.execution_stats();
    check(s.page_reads == 4 && s.device_dram_read_hits > 0 && s.device_dram_read_hits + s.read_buffer_hits == 196,
        "the hit-heavy stream did not hit");
    check(mean(d) < 0.6 * mean(p), "device-DRAM hits were not faster than NAND reads on average");
    for (std::size_t i = 0; i < p.size(); ++i)
        check(d[i] <= p[i] + 120.0, "a device-DRAM hit was slower than the NAND read (" +
            std::to_string(d[i]) + " vs " + std::to_string(p[i]) + " ns)");
    Device banked(parallel_config(0)), both(parallel_config(64));
    const auto b = stream(banked), w = stream(both);
    const auto& bank = banked.execution_stats();
    const auto& shared = both.execution_stats();
    check(shared.read_buffer_hits == bank.read_buffer_hits && shared.page_reads == 4 &&
        shared.device_dram_read_hits == bank.page_reads - shared.page_reads,
        "device DRAM answered ahead of a NAND bank buffer");
    // Replacing re-reads shifts which pages the bank buffers keep, so only
    // the stream as a whole is comparable: it must not get slower.
    check(mean(w) <= mean(b), "device DRAM slowed a stream the NAND bank buffers mostly serve");
}

// Both organizations flush a buffered page with one device-DRAM read of the
// page, whether it was assembled from a partial write or written whole.
void flush_reads_the_page_once() {
    for (auto organization : {MappingOrganization::Page, MappingOrganization::Extent}) {
        const bool page_ftl = organization == MappingOrganization::Page;
        {
            Device d(config(64, organization));
            (void)io(d, Op::Write, 0, 1e6);
            (void)d.drain_pending("full", 2e6);
            const auto& s = d.execution_stats();
            check(s.device_dram_write_bytes == 4096 && s.device_dram_read_bytes == 4096,
                "a full-page flush is one staging write and one flush read");
        }
        {
            Device d(config(64, organization));
            (void)io(d, Op::Write, 0, 1e6, 1024);
            (void)d.drain_pending("partial-unmapped", 2e6);
            const auto& s = d.execution_stats();
            check(s.device_dram_write_bytes == 4096 + 1024 && s.device_dram_read_bytes == 4096,
                "a partial flush of an unmapped page is a zero fill, the payload and one flush read");
        }
        {
            Device d(config(64, organization));
            (void)io(d, Op::Write, 0, 1e6);
            (void)d.drain_pending("seed", 2e6);
            const auto& s = d.execution_stats();
            const auto reads = s.device_dram_read_bytes, writes = s.device_dram_write_bytes;
            (void)io(d, Op::Write, 0, 3e6, 1024);
            (void)d.drain_pending("partial-mapped", 4e6);
            check(s.device_dram_read_bytes - reads == 2 * 4096,
                "a partial flush over a cached page reads the old image and the assembled page once each");
            // The page FTL acknowledges on entry and merges at flush time (a
            // copy of the dirty bytes); the structural policy merges on the
            // page's chain before staging.
            check(s.device_dram_write_bytes - writes == (page_ftl ? 2 * 1024 : 1024),
                "partial-write merge traffic changed");
        }
    }
}

// A buffered-overlay read counts hit bytes and write-buffer bytes without
// overlap; a missing overlay page is brought into DRAM by the request and
// kept as a clean line.
void overlay_bytes_are_counted_once() {
    Device d(config(64));
    d.prepopulate_logical_pages({0, 1});
    const auto& s = d.execution_stats();
    (void)io(d, Op::Read, 0, 1e6);
    (void)io(d, Op::Write, 0, 2e6, 1024);
    const auto hit_bytes = s.device_dram_read_hit_bytes, buffer_bytes = s.write_buffer_read_bytes,
        physical = s.physical_read_bytes;
    (void)io(d, Op::Read, 0, 3e6);
    check(s.write_buffer_read_hits == 1 && s.device_dram_read_hit_bytes - hit_bytes == 4096 - 1024 &&
        s.write_buffer_read_bytes - buffer_bytes == 1024 && s.physical_read_bytes == physical,
        "overlay hit bytes were not exactly the bytes the write buffer did not supply");
    (void)io(d, Op::Write, 1, 4e6, 1024);
    const auto hits = s.device_dram_read_hit_bytes, buffered = s.write_buffer_read_bytes;
    (void)io(d, Op::Read, 1, 5e6);
    check(s.physical_read_bytes - physical == 4096 && s.write_buffer_read_bytes - buffered == 1024 &&
        s.device_dram_read_hit_bytes == hits && Access::resident(d, *d.mapped_physical_page(1)),
        "overlay miss did not read NAND once and keep the old image");
}

// A flushed page holds its write-buffer slot until its mapping publishes;
// the clean read-cache line replaces the slot at that moment, never earlier.
void flushed_page_holds_one_copy() {
    Device d(config(64));
    (void)io(d, Op::Write, 0, 1e6);
    (void)io(d, Op::Write, 1, 2e6);
    // The two-page buffer is full: this write evicts lpn 0 and flushes it.
    (void)io(d, Op::Write, 2, 3e6);
    check(d.execution_stats().write_buffer_flushes == 1, "the third write did not evict a page");
    check(Access::lines(d) == 0 && d.execution_stats().device_dram_fills == 0,
        "a flushed page took a read-cache line while it still held its write-buffer slot");
    (void)d.drain_pending("publish", 4e6);
    check(Access::resident(d, *d.mapped_physical_page(0)) && Access::lines(d) == 3,
        "flushed pages were not installed once their slots were released");
}

// A structural flush whose target block is on another stack fills that
// stack's DRAM when the program is issued. With the channel's program credits
// exhausted the host issues it after the credit wait, past the moment the
// page left its buffer, and the fill must start no earlier than that.
void cross_stack_fill_after_credit_wait() {
    for (auto organization : {MappingOrganization::Extent, MappingOrganization::BlockLog}) {
        for (const auto& [credits, t_program] : {std::pair{1U, 200000.0}, std::pair{2U, 50000.0}}) {
            HbfConfig c;
            c.host.mapping_organization = organization;
            c.device.stacks = 3; c.device.channels_per_stack = 2; c.device.dies_per_channel = 1;
            c.device.planes_per_die = 2; c.device.blocks_per_plane = 32; c.device.pages_per_block = 8;
            c.device.t_program_page_ns = t_program; c.device.t_erase_block_ns = 2000; c.device.t_read_page_ns = 40;
            c.device.outstanding_write_pages_per_channel = credits;
            c.host.mapping_mode = MappingMode::FullResident; c.host.logical_capacity_bytes = 256 * 4096;
            c.host.gc_reserved_free_blocks_per_plane = 2; c.host.mapping_log_blocks = 2;
            c.host.write_coalescing_enabled = true; c.host.write_buffer_pages = 2;
            c.host.write_buffer_completion_requires_flush = false;
            c.host.device_dram_capacity_denominator = 32;
            c.host.mapping_superblock_planes = 12;
            Device d(c);
            double now = 1e6;
            for (U step = 0; step < 96; ++step, now += 100.0)
                (void)io(d, Op::Write, (3 * step) % 96 + step / 32, now);
            (void)d.drain_pending("end", now);
            const auto& s = d.stats();
            check(s.page_programs > 0 && s.device_dram_fills + s.device_dram_fill_bypasses > 0,
                "cross-stack flushes did not reach the fill path");
        }
    }
}

// Every device-DRAM parameter is inert while the denominator is zero.
void disabled_is_unchanged() {
    for (auto organization : {MappingOrganization::Page, MappingOrganization::Extent}) {
        auto tuned = config(0, organization);
        tuned.host.device_dram_hit_latency_ns = 7;
        tuned.host.device_dram_bandwidth_GBps = 3;
        tuned.host.device_dram_policy = DeviceDramPolicy::Fifo;
        Device a(config(0, organization)), b(tuned);
        double now = 0;
        std::vector<double> finish_a, finish_b;
        for (U step = 0; step < 48; ++step, now += 2500) {
            const U lpn = (step * 5) % 24;
            const auto op = step % 3 == 0 ? Op::Write : Op::Read;
            finish_a.push_back(io(a, op, lpn, now, step % 4 == 1 ? 1024 : 4096));
            finish_b.push_back(io(b, op, lpn, now, step % 4 == 1 ? 1024 : 4096));
        }
        finish_a.push_back(a.drain_pending("end", now).finish_ns);
        finish_b.push_back(b.drain_pending("end", now).finish_ns);
        check(finish_a == finish_b, "disabled device-DRAM parameters changed timing");
        const auto& x = a.stats();
        const auto& y = b.stats();
        check(x.page_reads == y.page_reads && x.physical_write_bytes == y.physical_write_bytes &&
            x.host_hbm_read_bytes == y.host_hbm_read_bytes && x.host_hbm_write_bytes == y.host_hbm_write_bytes &&
            x.read_buffer_hits == y.read_buffer_hits && x.stage_work == y.stage_work,
            "disabled device-DRAM parameters changed accounting");
        check(y.device_dram_read_hits == 0 && y.device_dram_read_misses == 0 && y.device_dram_fills == 0 &&
            y.device_dram_read_bytes == 0 && y.device_dram_write_bytes == 0 && y.device_dram_capacity_bytes == 0,
            "disabled device DRAM reported activity");
    }
}
} // namespace

int main() {
    const std::pair<const char*, void (*)()> cases[] = {
        {"capacity_and_budget", capacity_and_budget},
        {"hit_versus_miss_timing", hit_versus_miss_timing},
        {"eviction_lru", [] { eviction(DeviceDramPolicy::Lru); }},
        {"eviction_fifo", [] { eviction(DeviceDramPolicy::Fifo); }},
        {"write_buffer_lives_in_device_dram", write_buffer_lives_in_device_dram},
        {"invalidation_purges", invalidation_purges},
        {"structural_policy_shares_dram", structural_policy_shares_dram},
        {"fills_never_delay_host_traffic", fills_never_delay_host_traffic},
        {"hits_are_not_slower_than_nand", hits_are_not_slower_than_nand},
        {"flush_reads_the_page_once", flush_reads_the_page_once},
        {"overlay_bytes_are_counted_once", overlay_bytes_are_counted_once},
        {"flushed_page_holds_one_copy", flushed_page_holds_one_copy},
        {"cross_stack_fill_after_credit_wait", cross_stack_fill_after_credit_wait},
        {"disabled_is_unchanged", disabled_is_unchanged},
    };
    for (const auto& [name, run] : cases) {
        try {
            run();
        } catch (const std::exception& error) {
            std::cerr << "hbf_device_dram_test " << name << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << "hbf_device_dram_test: pass\n";
    return 0;
}
