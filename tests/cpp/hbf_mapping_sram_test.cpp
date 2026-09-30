#include "../../verification/probes/hbf_with_hbm.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace hbfsim::host;
using namespace hbfsim::physical;
using hbfsim::verification::HbfWithHbm;

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static HbfConfig config() {
    HbfConfig c;
    c.device.stacks = 2;
    c.device.channels_per_stack = c.device.dies_per_channel = 1;
    c.device.planes_per_die = 2; c.device.blocks_per_plane = 32; c.device.pages_per_block = 64;
    c.host.logical_capacity_bytes = 4096 * 4096;
    c.host.auto_gc_enabled = false; c.host.gc_reserved_free_blocks_per_plane = 0;
    c.host.mapping_sram_bytes = 128; // two tagged entries per stack
    return c;
}
static PhysicalCompletion issue(HbfWithHbm& d, Op op, std::uint64_t lpn, double at) {
    return d.issue(PhysicalRequest{.id="sram", .tier=Tier::HBF, .op=op,
        .address_space=AddressSpace::Logical, .trace={.mode=TraceMode::Full},
        .arrival_ns=at, .addr=lpn*4096, .bytes=4096});
}

int main() {
    try {
        {
            HbfWithHbm d(config());
            d.prepopulate_mutable_logical_page_range(0,16);
            const auto cold=issue(d,Op::Read,0,0);
            const auto before=d.stats();
            const auto hot=issue(d,Op::Read,0,cold.finish_ns);
            const auto after=d.stats();
            check(before.mapping_sram_fills==1 && before.host_hbm_read_bytes==32,
                  "cold SRAM miss did not fetch one HBM entry");
            check(after.mapping_sram_hits==1 && after.host_hbm_read_bytes==before.host_hbm_read_bytes,
                  "SRAM hit still consumed HBM bandwidth");
            check(cold.breakdown.mapping_dram_ns==100 && hot.breakdown.mapping_dram_ns==0,
                  "SRAM hit did not bypass HBM response latency");
            check(after.host_hbm_reserved_bytes==after.resident_mapping_table_bytes &&
                  after.mapping_sram_reserved_bytes==128,"SRAM displaced the complete HBM table");
            double now=hot.finish_ns;
            for (auto lpn : {2u,0u,4u,2u}) now=issue(d,Op::Read,lpn,now).finish_ns;
            check(d.stats().mapping_sram_fills==4 && d.stats().mapping_sram_evictions==2 &&
                  d.stats().mapping_sram_peak_entries==2,"SRAM LRU capacity/eviction incorrect");
        }
        {
            HbfWithHbm d(config());
            d.prepopulate_mutable_logical_page_range(0,16);
            (void)issue(d,Op::Read,0,0);
            (void)issue(d,Op::Read,0,0);
            const auto& s=d.stats();
            check(s.mapping_sram_misses==2 && s.mapping_sram_coalesced_misses==1 &&
                  s.mapping_sram_fills==1 && s.host_hbm_read_bytes==32,
                  "concurrent SRAM miss duplicated the HBM fetch");
        }
        {
            HbfAuditSnapshot images[2];
            for (unsigned enabled=0;enabled<2;++enabled) {
                auto c=config(); if (!enabled) c.host.mapping_sram_bytes=0;
                HbfWithHbm d(c);
                d.prepopulate_mutable_logical_page_range(0,16);
                auto now=issue(d,Op::Read,0,0).finish_ns;
                now=issue(d,Op::Write,0,now).finish_ns;
                const auto invalidations=d.stats().mapping_sram_invalidations;
                const auto misses=d.stats().mapping_sram_misses;
                now=issue(d,Op::Read,0,now).finish_ns;
                if (enabled) check(invalidations==1 && d.stats().mapping_sram_misses==misses+1,
                    "write-through update left a stale SRAM mapping");
                (void)d.invalidate_logical_pages(0,1,now);
                now=d.execution_stats().finish_ns;
                const auto erased=issue(d,Op::Read,0,now);
                check(erased.note=="unmapped-erased-read","SRAM resurrected a trimmed page");
                (void)d.drain_pending("checkpoint",erased.finish_ns);
                images[enabled]=d.audit_snapshot();
                HbfWithHbm restored(c);
                restored.restore_persistent_image(d.persistent_image());
                check(restored.stats().mapping_sram_entries==0,"restart restored volatile SRAM heat");
                check(issue(restored,Op::Read,0,0).note=="unmapped-erased-read",
                      "restart lost SRAM-coherent trim");
            }
            check(images[0].logical_mappings.size()==images[1].logical_mappings.size(),
                  "SRAM changed the persistent mapping size");
            for (std::size_t i=0;i<images[0].logical_mappings.size();++i)
                check(images[0].logical_mappings[i].key==images[1].logical_mappings[i].key &&
                      images[0].logical_mappings[i].ppn==images[1].logical_mappings[i].ppn,
                      "SRAM changed persistent L2P semantics");
        }
        {
            auto c=config();
            c.device.stacks=c.device.planes_per_die=1;
            c.device.blocks_per_plane=12; c.device.pages_per_block=8;
            c.host.logical_capacity_bytes=16*4096;
            c.host.auto_gc_enabled=true; c.host.gc_reserved_free_blocks_per_plane=2;
            c.host.gc_low_watermark_pages=8;
            HbfWithHbm d(c); d.prepopulate_mutable_logical_page_range(0,16);
            double now=0;
            for (unsigned i=0;i<128;++i) now=issue(d,Op::Write,i%16,now).finish_ns;
            (void)d.drain_pending("gc-checkpoint",now);
            check(d.stats().gc_runs>0 && d.stats().accounting_verified,
                  "SRAM cache did not survive mapping updates under GC");
        }
        std::cout << "SRAM hits, bounded LRU, concurrent fills, updates, trim, restart and GC passed\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
