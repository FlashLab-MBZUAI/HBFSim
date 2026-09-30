#include "../../verification/probes/hbf_with_hbm.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace hbfsim::host;
using namespace hbfsim::physical;
using hbfsim::verification::HbfWithHbm;

static void check(bool value, const char* reason) {
    if (!value) throw std::runtime_error(reason);
}
static HbfConfig config(MappingPlacement placement) {
    HbfConfig c;
    c.device.stacks = 2; c.device.channels_per_stack = c.device.dies_per_channel = 1;
    c.device.planes_per_die = 2; c.device.blocks_per_plane = 32; c.device.pages_per_block = 64;
    c.host.logical_capacity_bytes = 4096 * 4096;
    c.host.auto_gc_enabled = false; c.host.gc_reserved_free_blocks_per_plane = 0;
    c.host.mapping_mode = MappingMode::Cached;
    c.host.mapping_placement = placement; c.host.mapping_scratch_pages = 1;
    c.host.mapping_cache_tag_bytes = 32; c.host.ctrl_dram_bytes = 24896;
    c.host.mapping_control_compute_ns = 100; c.host.mapping_internal_compute_ns = 1;
    return c;
}
static PhysicalCompletion issue(HbfWithHbm& d, Op op, std::uint64_t lpn, double now) {
    return d.issue(PhysicalRequest{.id="placement", .tier=Tier::HBF, .op=op,
        .address_space=AddressSpace::Logical, .trace={.mode=TraceMode::Full},
        .arrival_ns=now, .addr=lpn*4096, .bytes=4096});
}
int main() {
    try {
        HbfStats outcomes[2]; HbfAuditSnapshot images[2];
        for (unsigned index=0; index<2; ++index) {
            const auto placement = index ? MappingPlacement::DeviceLocal : MappingPlacement::Host;
            HbfWithHbm d(config(placement));
            d.prepopulate_mutable_logical_page_range(0,4096);
            double now=0; bool compute_trace=false;
            for (unsigned i=0; i<32; ++i) {
                const auto result=issue(d,i%7==0 ? Op::Write : Op::Read,(i%8)*512,now);
                now=result.finish_ns;
                for (const auto& span:result.spans)
                    if (span.category=="mapping_compute") {
                        compute_trace=true;
                        check(span.entity.starts_with(index ? "stack" : "host/mapping/worker"),
                              "compute trace has the wrong execution owner");
                    }
            }
            (void)d.drain_pending("checkpoint",now);
            const auto& s=d.stats(); outcomes[index]=s; images[index]=d.audit_snapshot();
            check(compute_trace && s.mapping_compute_work_ns>0,"mapping compute was not timed");
            check(s.accounting_verified && !s.gc_runs && !s.static_wear_leveling_runs,"mapping-only accounting");
            const auto reads=(s.mapping_media_reads+s.mapping_merge_media_reads)*4096;
            check(s.mapping_hbio_read_bytes==(index ? 0 : reads),"metadata read route");
            check(s.mapping_hbio_write_bytes==(index ? 0 : s.mapping_page_programs*4096),"metadata program route");
            check((index ? s.local_mapping_reserved_bytes : s.host_hbm_reserved_bytes)==24896,"memory reservation owner");
            check(index ? s.host_hbm_read_bytes+s.host_hbm_write_bytes==0 :
                          s.local_mapping_read_bytes+s.local_mapping_write_bytes==0,"metadata leaked to wrong memory");
            const auto internals=s.mapping_directory_lookup_ops+s.mapping_buffer_lookup_ops+
                s.mapping_buffer_patch_ops+s.mapping_dirty_probe_ops+s.mapping_checkpoint_epoch_ops;
            check(s.mapping_compute_work_ns==(s.mapping_lookup_ops+s.mapping_update_ops)*100+
                internals+s.mapping_update_ops*25+s.mapping_codec_work_ns,"compute work conservation");
        }
        check(outcomes[0].mapping_media_reads==outcomes[1].mapping_media_reads &&
              outcomes[0].mapping_page_programs==outcomes[1].mapping_page_programs,"placement changed mapping policy work");
        check(images[0].logical_mappings.size()==images[1].logical_mappings.size(),"persistent mapping size");
        for (std::size_t i=0;i<images[0].logical_mappings.size();++i)
            check(images[0].logical_mappings[i].key==images[1].logical_mappings[i].key &&
                  images[0].logical_mappings[i].ppn==images[1].logical_mappings[i].ppn,"placement changed durable L2P");

        // A controller-DRAM entry fetch overfetches one 64-byte transaction;
        // the host HBM path retains its native 32-byte pseudo-channel burst.
        for (unsigned index=0;index<2;++index) {
            auto c=config(index ? MappingPlacement::DeviceLocal : MappingPlacement::Host);
            c.host.mapping_mode=MappingMode::FullResident; c.host.ctrl_dram_bytes=0;
            HbfWithHbm d(c); d.prepopulate_mutable_logical_page_range(0,4096);
            const auto before=d.execution_stats();
            (void)issue(d,Op::Read,0,0);
            const auto after=d.execution_stats();
            check(index ? after.local_mapping_read_bytes-before.local_mapping_read_bytes==64 :
                          after.host_hbm_read_bytes-before.host_hbm_read_bytes==32,
                  "mapping entry transaction does not match its memory interface");
        }
        {
            auto c=config(MappingPlacement::DeviceLocal);
            c.host.ctrl_dram_bytes=0; c.host.ctrl_dram_capacity_denominator=1000;
            HbfWithHbm d(c);
            const auto& s=d.stats();
            // 4096 physical pages/stack -> floor(4096/1000)*4096 B/stack.
            check(s.local_mapping_reserved_bytes==32768 && s.host_hbm_reserved_bytes==0,
                  "1/1000 controller DRAM was not derived per stack");
        }

        double finish[3]{};
        for (unsigned index=0;index<3;++index) {
            auto c=config(index==2 ? MappingPlacement::DeviceLocal : MappingPlacement::Host);
            c.host.mapping_host_compute_shared=index!=1;
            c.host.mapping_mode=MappingMode::FullResident; c.host.ctrl_dram_bytes=0;
            c.host.mapping_control_compute_ns=5000; c.host.mapping_compute_workers=2;
            HbfWithHbm d(c); d.prepopulate_mutable_logical_page_range(0,4096);
            std::uint64_t lpn=0;
            for (unsigned i=0;i<8;++i) {
                while(d.stack_for_logical_page(lpn)!=0) ++lpn;
                finish[index]=std::max(finish[index],issue(d,Op::Read,lpn++,0).finish_ns);
            }
            check(d.stats().mapping_compute_work_ns==40000,"parallel workers lost compute work");
        }
        check(finish[0]<finish[1],"host worker pool could not use idle compute for a hot stack");
        check(finish[0]<finish[2],"local partition should expose the same worker imbalance");
        // A one-nanosecond continuation must not strand nearly an entire
        // lookup-sized idle interval on the other worker of a shared pool.
        double pipelined_finish[2]{};
        for (unsigned index=0;index<2;++index) {
            auto c=config(MappingPlacement::Host);
            c.host.mapping_control_compute_ns=5000;
            c.host.mapping_internal_compute_ns=1;
            c.host.mapping_compute_workers=index+1;
            HbfWithHbm d(c); d.prepopulate_mutable_logical_page_range(0,4096);
            for (unsigned i=0;i<128;++i) {
                const auto lpn=(i%8)*512+(i/8)%2;
                pipelined_finish[index]=std::max(pipelined_finish[index],issue(d,Op::Read,lpn,0).finish_ns);
            }
        }
        check(pipelined_finish[0]/pipelined_finish[1]>1.7,
              "short continuations fragmented shared compute capacity");
        double dense_finish[2]{};
        std::uint64_t dense_reads[2]{};
        for (unsigned index=0;index<2;++index) {
            auto c=config(MappingPlacement::Host);
            c.host.mapping_scratch_pages=2;
            c.host.mapping_cache_layout=index ? MappingCacheLayout::Extent : MappingCacheLayout::Page;
            c.host.mapping_codec_ns_per_entry=0;
            HbfWithHbm d(c);
            std::vector<std::uint64_t> order(4096);
            std::iota(order.begin(),order.end(),0);
            std::mt19937_64 rng(73); std::shuffle(order.begin(),order.end(),rng);
            d.prepopulate_logical_pages(order);
            for (unsigned i=0;i<128;++i)
                dense_finish[index]=std::max(dense_finish[index],issue(d,Op::Read,(i*137)%4096,0).finish_ns);
            dense_reads[index]=d.stats().mapping_media_reads;
        }
        check(dense_reads[0]==dense_reads[1] && std::abs(dense_finish[0]-dense_finish[1])<1e-6,
              "zero-cost dense encoding consumed a second fill buffer");
        std::cout << "placement traffic, compute, persistence and worker-pooling checks passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
