#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_persistent_image.hpp"
#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace hbfsim::host;
using namespace hbfsim::physical;
using Device=hbfsim::verification::HbfWithHbm;
using U=std::uint64_t;
void check(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F fn,const char* message) { bool rejected=false; try {fn();} catch(const std::exception&) {rejected=true;} check(rejected,message); }
HbfConfig config(MappingOrganization kind) {
    HbfConfig c;
    c.host.mapping_organization=kind;
    c.device.stacks=c.device.channels_per_stack=c.device.dies_per_channel=c.device.planes_per_die=1;
    c.device.blocks_per_plane=64; c.device.pages_per_block=8;
    c.device.t_program_page_ns=750; c.device.t_erase_block_ns=2000; c.device.t_read_page_ns=40;
    c.host.logical_capacity_bytes=128*4096;
    c.host.gc_reserved_free_blocks_per_plane=2;
    c.host.write_coalescing_enabled=false; c.host.write_buffer_pages=0;
    c.host.mapping_log_blocks=2;
    return c;
}
double io(Device& d,Op op,U lpn,double now,U bytes=4096,U offset=0) {
    return d.issue(PhysicalRequest{.id="organization",.tier=Tier::HBF,.op=op,
        .address_space=AddressSpace::Logical,.arrival_ns=now,.addr=lpn*4096+offset,.bytes=bytes}).finish_ns;
}
void extent_index() {
    HbfExtentIndex index;
    std::map<U,U> oracle;
    std::mt19937_64 random(73);
    for (U step=0;step<3000;++step) {
        U start=random()%128;
        if (random()%4) { U ppn=random()%1024; index.assign(start,ppn); oracle[start]=ppn; }
        else { U count=1+random()%8; index.erase(start,count); for(U p=start;p<start+count;++p) oracle.erase(p); }
        for (U p=0;p<136;++p) {
            std::optional<U> expected;
            if (const auto at=oracle.find(p); at!=oracle.end()) expected=at->second;
            check(index.lookup(p)==expected,"extent split/erase changed an unrelated mapping");
        }
        U end=0,physical_end=~U{0};
        for (const auto& [first,run]:index.runs()) {
            check(run.pages && first>=end,"overlapping extent runs");
            check(first!=end || run.ppn!=physical_end,"adjacent affine runs were not merged");
            end=first+run.pages; physical_end=run.ppn+run.pages;
        }
    }
}
void host_write_accounting() {
    for (auto kind:{MappingOrganization::Page,MappingOrganization::Block,MappingOrganization::BlockLog,
            MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
        auto c=config(kind); Device d(c); double now=0;
        if (kind==MappingOrganization::ObjectSegment)
            now=d.object_command("CREATE",1,0,128,now).finish_ns;
        now=io(d,Op::Write,0,now);
        check(d.stats().logical_write_bytes==4096 && d.stats().physical_write_bytes==4096,
            "single logical page did not preserve host/physical write bytes");
        check(d.stats().raw_physical_programs==0 && d.stats().raw_physical_program_payload_bytes==0,
            "internal mapping program was counted again as a raw host write");
        check(d.stats().waf() && *d.stats().waf()==1.0,"one physical page per host page must have WAF 1");
        now=io(d,Op::Write,3,now); // pure Block must program the intervening holes
        now=io(d,Op::Write,0,now); // pure Block copies the other live page
        now=d.drain_pending("host-write-accounting",now).finish_ns;
        const auto& stats=d.stats();
        const auto policy=d.mapping_policy_stats();
        check(stats.logical_write_bytes==3*4096 && stats.raw_physical_programs==0 &&
            stats.raw_physical_program_payload_bytes==0,
            "copy, padding or checkpoint changed the host-write denominator");
        check(policy.metadata_programs>0,"accounting test did not persist mapping metadata");
        if (kind==MappingOrganization::Block)
            check(policy.copy_programs>0 && policy.padding_programs>0,
                "accounting test did not exercise copy and padding programs");
        check(stats.waf() && *stats.waf()==static_cast<double>(stats.physical_write_bytes)/(3*4096),
            "logical mapping WAF differs from physical bytes per host byte");
        check(stats.accounting_verified,"host-write accounting broke the physical program ledger");
    }
}
void lifecycle(MappingOrganization kind) {
    constexpr U capacity=400; // pressure must leave partially live cleaner victims
    auto c=config(kind); c.host.logical_capacity_bytes=capacity*4096;
    Device d(c); double now=0;
    if (kind==MappingOrganization::ObjectSegment)
        now=d.object_command("CREATE",7,0,capacity,now).finish_ns;
    std::map<U,U> expected;
    U generation=0;
    for (U p=0;p<capacity;++p) { now=io(d,Op::Write,p,now); expected[p]=++generation; }
    std::mt19937_64 random(19);
    for (U step=0;step<1700;++step) {
        U lpn=random()%capacity;
        if (step%23==0) {
            now=d.invalidate_logical_pages(lpn,1,now).completion.finish_ns; expected.erase(lpn);
        } else {
            now=io(d,Op::Write,lpn,now,step%3 ? 4096 : 64,step%3 ? 0 : 128);
            expected[lpn]=++generation;
        }
        if (step%11==0) {
            now=io(d,Op::Read,lpn,now,64);
            check(d.mapped_generation(lpn)==(expected.contains(lpn) ? std::optional<U>{expected[lpn]} : std::nullopt),
                "read/relocation returned an obsolete payload generation");
        }
        if (step%113==0) check(d.stats().accounting_verified,"native media accounting failed under mapping traffic");
    }
    for (U p=0;p<capacity;++p)
        check(d.mapped_generation(p)==(expected.contains(p) ? std::optional<U>{expected[p]} : std::nullopt),
            "mapping lost live content after sustained updates and GC");
    now=d.drain_pending("checkpoint",now).finish_ns;
    check(d.quiescence_stats().quiescent(),"checkpoint left pending mapping work");
    auto policy=d.mapping_policy_stats();
    check(policy.metadata_programs>0 && policy.checkpoints==1,"mapping checkpoint was not charged to NAND");
    check(policy.data_programs+policy.copy_programs+policy.padding_programs+policy.metadata_programs==d.stats().page_programs,
        "policy program accounting does not conserve physical writes");
    if (kind==MappingOrganization::Extent || kind==MappingOrganization::ObjectSegment)
        check(policy.cleaned_segments>0 && policy.copy_programs>0,"sustained append workload did not exercise cleaner");
    if (kind==MappingOrganization::Block || kind==MappingOrganization::BlockLog)
        check(policy.full_merges>0,"random writes did not exercise full block merge");
    const auto image=d.persistent_image();
    const auto path=std::filesystem::temp_directory_path()/
        (std::string("hbfsim-organization-")+to_string(kind)+".image");
    std::filesystem::remove(path);
    write_persistent_image_file(path,image);
    Device restored(c); restored.restore_persistent_image(read_persistent_image_file(path));
    std::filesystem::remove(path);
    for (U p=0;p<capacity;++p) check(restored.mapped_generation(p)==d.mapped_generation(p),"checkpoint/restore changed payload identity");
    check(restored.mapping_policy_stats().metadata_reads>0,"recovery bypassed persisted mapping reads");
    now=io(restored,Op::Write,3,0);
    check(restored.mapped_generation(3)==generation+1,"recovery did not preserve generation/order cursor");
    now=restored.drain_pending("second-checkpoint",now).finish_ns;
    check(restored.stats().accounting_verified,"restored mapping cannot continue native IO");
    auto corrupt=image; corrupt.host_mapping_state.back()^=1;
    Device bad(c); rejects([&]{bad.restore_persistent_image(corrupt);},"corrupted policy image accepted");
    std::cout<<to_string(kind)<<" programs="<<d.stats().page_programs<<" copies="<<policy.copy_programs
        <<" merges="<<policy.full_merges<<" cleaned="<<policy.cleaned_segments<<'\n';
}
void log_merges() {
    auto c=config(MappingOrganization::BlockLog); c.host.mapping_log_blocks=1;
    Device d(c); double now=0;
    for(U p=0;p<16;++p) now=io(d,Op::Write,p,now);
    for(U p=0;p<8;++p) now=io(d,Op::Write,p,now);
    check(d.mapping_policy_stats().switch_merges==1,"full sequential log did not switch without copying");
    const auto before=d.mapping_policy_stats().copy_programs;
    now=io(d,Op::Write,0,now); now=io(d,Op::Write,1,now);
    now=io(d,Op::Write,8,now); // one-log budget evicts a sequential prefix
    check(d.mapping_policy_stats().partial_merges==1,"prefix log did not append the unchanged tail");
    check(d.mapping_policy_stats().copy_programs-before==6,"partial merge did not copy exactly its six-page tail");
    now=io(d,Op::Write,3,now); // evict other prefix, then open nonsequential log
    now=io(d,Op::Write,8,now);
    check(d.mapping_policy_stats().full_merges>0,"nonsequential log did not rebuild a block");
    check(d.stats().accounting_verified,"log merge accounting");
}
void full_block_replacement() {
    auto c=config(MappingOrganization::Block);
    c.host.write_coalescing_enabled=true; c.host.write_buffer_pages=2;
    Device d(c);
    auto now=io(d,Op::Write,0,0,8*4096);
    now=io(d,Op::Write,0,now,64);
    now=io(d,Op::Write,0,now,8*4096);
    const auto stats=d.mapping_policy_stats();
    check(stats.data_programs==16 && stats.copy_programs==0 && stats.padding_programs==0,
        "whole-block overwrite unnecessarily copied old pages");
    for (U p=0;p<8;++p) check(d.mapped_generation(p)==p+10,"block replacement lost page order");
    check(d.quiescence_stats().write_buffer_entries==0,"full block did not supersede buffered fragment");
    (void)d.drain_pending("block-replacement",now);
    check(d.stats().accounting_verified,"whole-block replacement accounting");
}
void buffering_and_objects() {
    for (auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
        auto c=config(kind); c.host.write_coalescing_enabled=true; c.host.write_buffer_pages=2;
        Device d(c); double now=0;
        if (kind==MappingOrganization::ObjectSegment) now=d.object_command("CREATE",8,0,128,now).finish_ns;
        for(U i=0;i<64;++i) now=io(d,Op::Write,0,now,64,i*64);
        check(d.mapping_policy_stats().data_programs==0,"host coalescing programmed each fragment");
        now=io(d,Op::Read,0,now,64);
        check(d.mapping_policy_stats().buffer_hits==1,"buffered read did not see newest generation");
        rejects([&]{(void)d.persistent_image();},"unflushed payload was exported as persistent");
        now=d.drain_pending("buffer-flush",now).finish_ns;
        check(d.mapping_policy_stats().data_programs==1 && d.mapped_generation(0)==64,"coalescing did not produce one latest page");
        if (kind==MappingOrganization::ObjectSegment) {
            now=d.object_command("SEAL",8,0,0,now).finish_ns;
            rejects([&]{(void)io(d,Op::Write,0,now);},"sealed object accepted mutation");
            now=io(d,Op::Read,0,now);
            const auto copy=d.mapping_policy_stats().copy_programs;
            now=d.object_command("DELETE",8,0,0,now).finish_ns;
            check(!d.mapped_physical_page(0) && d.mapping_policy_stats().copy_programs==copy,"whole-object deletion copied dead data");
            now=d.object_command("CREATE",9,0,128,now).finish_ns;
            now=io(d,Op::Write,0,now);
            now=d.drain_pending("recreated",now).finish_ns;
        }
        check(d.stats().accounting_verified,"buffer/object accounting");
    }
}
void initial_images() {
    for(auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
        auto c=config(kind); Device d(c);
        d.prepopulate_mutable_logical_page_range(0,64);
        check(d.stats().page_programs==0 && d.stats().block_erases==0,"untimed population created timed media work");
        check(d.mapped_generation(63)==64,"initial population missing content");
        auto image=d.persistent_image(); Device resumed(c); resumed.restore_persistent_image(image);
        auto now=io(resumed,Op::Write,0,0);
        check(resumed.mapped_generation(0)==65,"initial checkpoint could not resume mutation");
        (void)resumed.drain_pending("initial-restart",now);
        check(resumed.stats().accounting_verified,"initial population restore accounting");
        Device immutable(c); immutable.prepopulate_read_only_logical_page_range(1,12);
        Device read_only(c); read_only.restore_persistent_image(immutable.persistent_image());
        (void)io(read_only,Op::Read,1,0,64);
        rejects([&]{(void)io(read_only,Op::Write,1,0);},"read-only image became mutable after restore");
    }
}
void striped_blocks_and_restore() {
    {
        auto c=config(MappingOrganization::Block); c.device.stacks=2; c.device.planes_per_die=2;
        c.device.blocks_per_plane=8; c.device.pages_per_block=256;
        c.host.logical_capacity_bytes=1024*4096;
        Device d(c); d.prepopulate_mutable_logical_page_range(0,1024);
        const std::vector<U> blocks{0,16,8,24};
        for (U p=0;p<1024;++p)
            check(d.mapped_physical_page(p)==blocks[p/256]*256+p%256,
                "4 MiB initial range did not preserve four contiguous 1 MiB blocks across four banks");
        (void)io(d,Op::Write,0,0,1024*4096);
        const auto stats=d.mapping_policy_stats();
        check(stats.data_programs==1024 && stats.copy_programs==0 && stats.padding_programs==0,
            "block striping changed the 1 MiB full-replacement fast path");
    }
    const auto seal=[](std::vector<U>& words) {
        U hash=1469598103934665603ULL;
        for (std::size_t i=0;i+1<words.size();++i)
            for (unsigned byte=0;byte<8;++byte) { hash^=(words[i]>>(byte*8))&255; hash*=1099511628211ULL; }
        words.back()=hash;
    };
    for (auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,
            MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
        auto c=config(kind); c.device.stacks=3; c.device.planes_per_die=2;
        c.device.blocks_per_plane=4;
        Device d(c); d.prepopulate_mutable_logical_page_range(0,96);
        // Independent nested-order oracle. The last physical plane belongs
        // to the two metadata slots and leaves holes in every stripe round.
        const auto image=d.persistent_image();
        const U data_blocks=24-2*image.host_mapping_state.at(6);
        std::vector<U> order;
        for (U offset=0;offset<4;++offset)
            for (U plane=0;plane<2;++plane)
                for (U stack=0;stack<3;++stack) {
                    const U block=(stack*2+plane)*4+offset;
                    if (block<data_blocks) order.push_back(block);
                }
        check(data_blocks==20 && order.size()==20,"striping fixture did not expose metadata rank holes");
        std::set<U> stacks;
        for (U p=0;p<96;++p) {
            const auto ppn=*d.mapped_physical_page(p);
            check(ppn==order[p/8]*8+p%8,"whole-block stripe differs from independent lane oracle");
            stacks.insert(ppn/(2*4*8));
        }
        check(stacks.size()==3 && image.host_mapping_state.at(1)==2,
            "initial allocation did not span all stacks with the rank-cursor format");
        auto legacy=image; legacy.host_mapping_state[1]=1; seal(legacy.host_mapping_state);
        bool old_version_rejected=false;
        try { Device bad(c); bad.restore_persistent_image(legacy); }
        catch (const std::invalid_argument& error) {
            old_version_rejected=std::string(error.what()).find("policy or geometry mismatch")!=std::string::npos;
        }
        check(old_version_rejected,"obsolete physical-cursor policy image was not explicitly rejected");

        // Wear wins over stripe proximity; then equal wear wraps in stripe
        // order even when the cursor lies in an excluded metadata rank.
        auto worn=image;
        for (auto& block:worn.state.blocks) block.erase_count=3;
        worn.state.blocks.at(3).erase_count=1;
        Device colder(c); colder.restore_persistent_image(worn);
        (void)io(colder,Op::Write,96,0);
        check(*colder.mapped_physical_page(96)/8==3,"striping overrode the global minimum erase count");
        worn.state.blocks.at(3).erase_count=3;
        worn.host_mapping_state[9]=23; seal(worn.host_mapping_state);
        Device wrapped(c); wrapped.restore_persistent_image(worn);
        (void)io(wrapped,Op::Write,96,0);
        check(*wrapped.mapped_physical_page(96)/8==order[12],"equal-wear allocation did not wrap over metadata holes");

        Device resumed(c); resumed.restore_persistent_image(image);
        double original_now=0,resumed_now=0;
        for (U p=96;p<112;++p) {
            original_now=io(d,Op::Write,p,original_now);
            resumed_now=io(resumed,Op::Write,p,resumed_now);
            check(d.mapped_physical_page(p)==resumed.mapped_physical_page(p) &&
                d.mapped_generation(p)==resumed.mapped_generation(p),"restore changed the next striped allocation");
        }
        (void)d.drain_pending("stripe-original",original_now);
        (void)resumed.drain_pending("stripe-restored",resumed_now);
        check(d.persistent_image().host_mapping_state==resumed.persistent_image().host_mapping_state,
            "continued checkpoint changed allocation or logical sequence after restore");
    }
    auto c=config(MappingOrganization::Extent); c.device.stacks=3;
    c.device.planes_per_die=2; c.device.blocks_per_plane=4;
    Device d(c); d.prepopulate_mutable_logical_page_range(0,96); double now=0;
    for (U step=0;step<200;++step) {
        const U lpn=(step*37)%96;
        if (step%11==0) now=d.invalidate_logical_pages(lpn,1,now).completion.finish_ns;
        now=io(d,Op::Write,lpn,now);
    }
    now=d.drain_pending("stripe-cleaned",now).finish_ns;
    check(d.mapping_policy_stats().cleaned_segments>0 && d.mapping_policy_stats().copy_programs>0,
        "multistack stripe fixture did not exercise live-copy cleaning");
    Device resumed(c); resumed.restore_persistent_image(d.persistent_image()); double later=0;
    for (U step=0;step<40;++step) {
        const U lpn=(step*29)%96;
        now=io(d,Op::Write,lpn,now); later=io(resumed,Op::Write,lpn,later);
        for (U p=0;p<96;++p)
            check(d.mapped_physical_page(p)==resumed.mapped_physical_page(p) &&
                d.mapped_generation(p)==resumed.mapped_generation(p),"cleaner/restore changed striped allocation or payloads");
    }
    (void)d.drain_pending("stripe-cleaned-next",now);
    (void)resumed.drain_pending("stripe-cleaned-restored",later);
    check(d.persistent_image().host_mapping_state==resumed.persistent_image().host_mapping_state &&
        d.block_erase_counts()==resumed.block_erase_counts(),"restored cleaner changed cursor, wear or checkpoint state");
    check(d.stats().accounting_verified && resumed.stats().accounting_verified,"striped cleaner broke media accounting");
}
void shared_object_segment() {
    auto c=config(MappingOrganization::ObjectSegment); Device d(c);
    auto now=d.object_command("CREATE",10,0,64,0).finish_ns;
    now=d.object_command("CREATE",20,64,64,now).finish_ns;
    now=io(d,Op::Write,0,now); now=io(d,Op::Write,64,now);
    check(*d.mapped_physical_page(0)/8==*d.mapped_physical_page(64)/8,"objects did not share the append segment");
    now=d.object_command("SEAL",10,0,0,now).finish_ns;
    now=d.object_command("DELETE",20,0,0,now).finish_ns;
    check(d.mapped_generation(0)==1 && !d.mapped_generation(64),"object expiry damaged its segment neighbor");
    now=d.drain_pending("shared-object",now).finish_ns;
    Device restored(c); restored.restore_persistent_image(d.persistent_image());
    rejects([&]{(void)io(restored,Op::Write,0,0);},"restored sealed object accepted writes");
    now=restored.object_command("CREATE",30,64,64,0).finish_ns;
    now=io(restored,Op::Write,64,now);
    check(restored.mapped_generation(64)==3 && restored.mapped_generation(0)==1,
        "reused object range changed surviving object");
    (void)restored.drain_pending("replacement",now);
    check(restored.stats().accounting_verified,"shared object segment accounting");
}
void writeback_timing_and_ownership() {
    for(auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
        auto c=config(kind); c.device.stacks=2;
        c.host.write_coalescing_enabled=true; c.host.write_buffer_pages=2;
        c.host.write_buffer_flush_threshold_pages=2;
        Device d(c); double now=0;
        if (kind==MappingOrganization::ObjectSegment) now=d.object_command("CREATE",1,0,128,now).finish_ns;
        now=io(d,Op::Write,0,now,64); now=io(d,Op::Write,1,now,64);
        check(d.quiescence_stats().write_buffer_entries==2 && d.mapping_policy_stats().data_programs==0,
            "write-buffer threshold was not partitioned per stack");
        now=io(d,Op::Write,2,now,64);
        check(d.execution_stats().finish_ns>now,"threshold writeback blocked early acknowledgement");
        now=d.drain_pending("threshold",now).finish_ns;
        check(d.mapping_policy_stats().data_programs==3,"threshold/drain lost buffered pages");
        rejects([&]{(void)d.issue(PhysicalRequest{.tier=Tier::HBF,.op=Op::Write,
            .address_space=AddressSpace::Physical,.arrival_ns=now,.bytes=4096});},"raw IO bypassed primary ownership");
        rejects([&]{(void)d.reset_zone(0,0,0,now);},"zone reset bypassed primary ownership");
        check(d.stats().accounting_verified,"multistack threshold accounting");
        auto read=[&](U bytes) {
            auto completion=d.issue(PhysicalRequest{.id="read-granularity",.tier=Tier::HBF,.op=Op::Read,
                .address_space=AddressSpace::Logical,.arrival_ns=now,.addr=0,.bytes=bytes});
            now=completion.finish_ns; return completion.breakdown.hb_io_transfer_ns;
        };
        const auto line=read(64),page=read(4096);
        check(page>line*10,"64-byte logical read overfetched a full page over HBIO");
        c.host.write_buffer_completion_requires_flush=true;
        Device durable(c); now=0;
        if (kind==MappingOrganization::ObjectSegment) now=durable.object_command("CREATE",1,0,128,now).finish_ns;
        now=io(durable,Op::Write,0,now,64);
        check(durable.mapping_policy_stats().data_programs==1 && now==durable.execution_stats().finish_ns,
            "flush-on-completion acknowledged before NAND program");
    }
}
void bounded_foreground_range_reads() {
    for (auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,
            MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
        for (bool separate_stacks:{false,true}) {
            auto c=config(kind);
            c.device.blocks_per_plane=16;
            c.device.stacks=separate_stacks ? 2 : 1;
            c.device.planes_per_die=separate_stacks ? 1 : 2;
            c.device.t_read_page_ns=4000;
            c.host.logical_capacity_bytes=176*4096;
            const auto physical_page=[](U p) { return ((p/8)%2*16+(p/8)/2)*8+p%8; };
            const bool extent=kind==MappingOrganization::Extent || kind==MappingOrganization::ObjectSegment;
            // Twenty-two full blocks alternate banks, so none of their PPN
            // runs coalesce. ObjectSegment also probes its one-object tree.
            const U probes=extent ? std::bit_width(U{22})+
                (kind==MappingOrganization::ObjectSegment ? 1 : 0) : 1;
            // Read lookups book worker compute as translation and each dependent
            // probe's controller-memory round trip as mapping-memory time.
            const auto compute_per_lookup=c.host.mapping_control_compute_ns+
                probes*c.host.mapping_internal_compute_ns;
            const auto memory_per_lookup=probes*c.host.ctrl_dram_latency_ns;
            auto read=[&](U depth,U pages,TraceMode trace) {
                c.device.page_read_queue_depth_per_stack=depth;
                Device d(c); d.prepopulate_mutable_logical_page_range(0,176);
                check(d.mapping_policy_stats().extent_records==(extent ? 22 : 0),
                    "block stripe changed the declared authoritative index shape");
                for (U p=0;p<pages;++p)
                    check(d.mapped_physical_page(p)==physical_page(p),"range test lost its declared block stripe");
                const auto completion=d.issue(PhysicalRequest{.id="bounded-range",.tier=Tier::HBF,.op=Op::Read,
                    .address_space=AddressSpace::Logical,.trace={.mode=trace},.addr=0,.bytes=pages*4096});
                const auto& stats=d.stats();
                // One lookup per index entry: each 8-page block (or its run).
                const U entries=pages/8;
                check(d.mapping_policy_stats().lookups==entries &&
                    std::abs(completion.breakdown.translation_ns-entries*compute_per_lookup)<1e-9 &&
                    std::abs(completion.breakdown.mapping_dram_ns-entries*memory_per_lookup)<1e-9,
                    "range lookup work differs from one lookup per index entry at its configured costs");
                check(stats.page_reads==pages && stats.physical_read_bytes==pages*4096 &&
                    stats.scalar_read_pages==pages && stats.page_read_admission_events==pages,
                    "range read did not conserve physical reads or page credits");
                check(stats.page_programs==0 && stats.block_erases==0 && stats.accounting_verified,
                    "read-only pipeline changed persistent media or accounting");
                if (depth==1) check(stats.page_read_admission_waited_pages>0,"one-credit range never waited");
                std::map<std::string,double> ends;
                U arrays=0;
                for (const auto& span:completion.spans) if (span.name=="user/array_read") {
                    check(span.start_ns>=ends[span.entity],"same-bank NAND reads overlap");
                    ends[span.entity]=span.end_ns; ++arrays;
                }
                if (trace==TraceMode::Full)
                    check(arrays==pages && ends.size()==pages/8,"read trace lost actual bank ownership");
                return completion.finish_ns;
            };
            const auto serial=read(1,16,TraceMode::Full);
            const auto concurrent=read(32,16,TraceMode::Full);
            const auto untraced=read(32,16,TraceMode::Off);
            const auto one_bank=read(32,8,TraceMode::Full);
            check(concurrent<serial*0.7,"independent banks remained serial despite available credits");
            check(concurrent>=8*c.device.t_read_page_ns && one_bank>=8*c.device.t_read_page_ns,
                "range pipeline bypassed the per-bank NAND service bound");
            check(std::abs(concurrent-untraced)<1e-6,"tracing changed structural range execution");
            // Native raw reads use exactly the same physical pages and media
            // calendars, avoiding the independent page-FTL striping effect.
            auto raw_config=c; raw_config.host.mapping_organization=MappingOrganization::Page;
            raw_config.host.mapping_mode=MappingMode::RawPhysical;
            // This control addresses only PPNs. Populated raw blocks cannot
            // also back the structural fixture's explicit logical capacity.
            raw_config.host.logical_capacity_bytes=0;
            Device raw(raw_config);
            for (U p=0;p<176;++p) raw.prepopulate_raw_physical_page(physical_page(p)*4096);
            double raw_finish=0;
            for (U p=0;p<16;p+=8) {
                const auto part=raw.issue(PhysicalRequest{.id="same-layout",.tier=Tier::HBF,.op=Op::Read,
                    .address_space=AddressSpace::Physical,.trace={.mode=TraceMode::Full},
                    .addr=physical_page(p)*4096,.bytes=8*4096});
                raw_finish=std::max(raw_finish,part.finish_ns);
            }
            check(raw.stats().page_reads==16 && raw.stats().physical_read_bytes==16*4096 &&
                raw.stats().accounting_verified,"same-PPN raw control changed physical read work");
            // Raw has two range dispatches and no authoritative index. Its
            // elapsed-time difference is diagnostic, not a constant overhead
            // contract: block striping creates more Extent lookup probes.
            std::cout<<to_string(kind)<<" range "<<(separate_stacks ? "two-stacks" : "two-planes")
                <<" credit1="<<serial<<" credit32="<<concurrent<<" same-layout-raw="<<raw_finish
                <<" lookup-probes="<<probes<<" lookup-work="<<2*(compute_per_lookup+memory_per_lookup)
                <<" one-bank="<<one_bank<<'\n';
        }
        auto c=config(kind); c.host.write_coalescing_enabled=true; c.host.write_buffer_pages=2;
        c.device.page_read_queue_depth_per_stack=1;
        Device d(c); d.prepopulate_mutable_logical_page_range(0,3);
        auto now=d.invalidate_logical_pages(2,1,0).completion.finish_ns;
        now=io(d,Op::Write,0,now);
        const auto before=d.stats(); const auto buffers=d.mapping_policy_stats().buffer_hits;
        now=io(d,Op::Read,0,now,3*4096);
        check(d.stats().page_reads-before.page_reads==1 && d.mapping_policy_stats().buffer_hits==buffers+1,
            "mixed range did not preserve buffered, mapped and erased responses");
        check(d.stats().page_read_admission_events-before.page_read_admission_events==3,
            "buffer/erased pages bypassed the range credit bound");
        const auto cached=d.stats();
        now=io(d,Op::Read,1,now,64);
        check(d.stats().page_reads==cached.page_reads && d.stats().read_buffer_hits==cached.read_buffer_hits+1,
            "foreground range failed to make the decoded page cacheable");
        check(d.mapped_generation(0)==4 && !d.mapped_generation(2),"range read changed live payload identity");
        check(d.stats().accounting_verified,"mixed range accounting");
    }
    auto hot=config(MappingOrganization::Extent);
    hot.device.t_read_page_ns=4000;
    hot.device.thermal_enabled=true;
    hot.device.thermal_start_at_ceiling=true;
    hot.device.thermal_throttle_power_w=0.01;
    Device paced(hot); paced.prepopulate_mutable_logical_page_range(0,16);
    const auto finish=io(paced,Op::Read,0,0,16*4096);
    const auto energy=16*4096*8*hot.device.thermal_read_energy_pj_per_bit*1e-12;
    check(std::abs(paced.stats().thermal_media_energy_j-energy)<1e-15,
        "range thermal energy differs from actual NAND reads");
    check(finish>=energy/hot.device.thermal_throttle_power_w*1e9,
        "range finished before its thermal pacing budget");
    check(paced.stats().page_reads==16 && paced.stats().accounting_verified,"thermal range accounting");
}
// The host issues programs and moves on; NAND work overlaps across planes,
// stays ordered on one plane and is joined only where a command depends on it.
void asynchronous_programs() {
    auto write=[](Device& d,U lpn,double at) {
        return d.issue(PhysicalRequest{.id="async",.tier=Tier::HBF,.op=Op::Write,
            .address_space=AddressSpace::Logical,.arrival_ns=at,.addr=lpn*4096,.bytes=4096});
    };
    // Logical blocks 0 and 1 land on the first two allocation lanes.
    auto planes=[](U channels,U planes_per_die) {
        auto c=config(MappingOrganization::Block);
        c.device.channels_per_stack=channels; c.device.planes_per_die=planes_per_die;
        c.device.blocks_per_plane=16; c.host.logical_capacity_bytes=64*4096;
        return c;
    };
    // A lone program pays erase, ECC encode and array program time; a second
    // program elsewhere overlaps it, and one on the same plane follows it.
    const double program=config(MappingOrganization::Block).device.t_program_page_ns;
    {
        Device d(planes(2,1));
        const auto first=write(d,0,0),second=write(d,8,0);
        check(first.finish_ns>=program && second.finish_ns>=program,"a program completed before its NAND time");
        check(second.finish_ns<first.finish_ns*1.5,"programs on two channels did not overlap");
        check(d.stats().accounting_verified,"overlapped programs broke accounting");
        (void)d.drain_pending("async-overlap",second.finish_ns);
        check(d.quiescence_stats().quiescent(),"drain left programs in flight");
    }
    {
        Device d(planes(1,1));
        const auto first=write(d,0,0),second=write(d,8,0);
        check(second.finish_ns>=first.finish_ns+program,"two programs on one plane overlapped");
    }
    for (U credits:{1,2}) {
        auto c=planes(1,2); c.device.outstanding_write_pages_per_channel=credits;
        Device d(c);
        const auto first=write(d,0,0),second=write(d,8,0);
        check(credits==1 ? second.finish_ns>=first.finish_ns+program : second.finish_ns<first.finish_ns*1.5,
            "channel program credits did not bound issued programs");
    }
    {
        Device d(planes(2,1));
        const auto written=write(d,0,0);
        const auto read=io(d,Op::Read,0,1,64);
        check(read>=written.finish_ns+config(MappingOrganization::Block).device.t_read_page_ns,
            "a read overtook the program of its page");
        check(written.finish_ns>=program,"an unbuffered write was acknowledged before its program");
    }
    for (auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,
            MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
        auto c=config(kind); c.device.channels_per_stack=2; c.device.blocks_per_plane=32;
        Device d(c); double now=0;
        if (kind==MappingOrganization::ObjectSegment) now=d.object_command("CREATE",1,0,128,now).finish_ns;
        // Rewrite and trim pages whose programs are still in flight: step s
        // writes page s%10 as generation s+1, one nanosecond apart.
        for (U step=0;step<24;++step) (void)write(d,step%10,now+step);
        check(d.stats().accounting_verified,"observation with programs in flight failed");
        const auto trimmed=d.invalidate_logical_pages(0,8,now+30);
        now=d.drain_pending("async-rewrite",trimmed.completion.finish_ns).finish_ns;
        check(d.quiescence_stats().quiescent() && d.stats().accounting_verified,"in-flight rewrites broke native state");
        for (U p=0;p<10;++p)
            check(d.mapped_generation(p)==(p<8 ? std::optional<U>{} : std::optional<U>{U{p+11}}),
                "a rewrite or trim during a program changed payload identity");
        Device restored(c); restored.restore_persistent_image(d.persistent_image());
        check(restored.mapped_generation(9)==20,"checkpoint after in-flight rewrites lost the newest version");
    }
}
// Superblocks stripe consecutive pages over planes; blocks, merges, cleaning,
// checkpoints and recovery act on whole superblocks.
HbfConfig superblock_config(MappingOrganization kind,U width) {
    auto c=config(kind);
    c.device.planes_per_die=4; c.device.blocks_per_plane=16;
    c.host.logical_capacity_bytes=240*4096;
    c.host.mapping_superblock_planes=width;
    return c;
}
void superblock_striping() {
    for (auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,
            MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
        Device d(superblock_config(kind,4)); d.prepopulate_mutable_logical_page_range(0,32);
        // Independent oracle: page p is page p/4 of plane p%4, block offset 0.
        for (U p=0;p<32;++p)
            check(d.mapped_physical_page(p)==(p%4)*16*8+p/4,"superblock pages are not striped over its planes");
    }
    auto elapsed=[&](U width,Op op) {
        auto c=superblock_config(MappingOrganization::Block,width);
        c.device.t_read_page_ns=4000;
        Device d(c);
        if (op==Op::Read) d.prepopulate_mutable_logical_page_range(0,8);
        return d.issue(PhysicalRequest{.id="superblock",.tier=Tier::HBF,.op=op,
            .address_space=AddressSpace::Logical,.addr=0,.bytes=8*4096}).finish_ns;
    };
    // Eight pages of one logical block: one plane unstriped, four planes
    // striped. Writes also pay one (parallel) erase per fresh member block.
    const auto& nand=superblock_config(MappingOrganization::Block,1).device;
    check(elapsed(4,Op::Read)<elapsed(1,Op::Read)*0.4,"a superblock read did not use its member planes");
    check(elapsed(4,Op::Write)<elapsed(1,Op::Write)*0.7,"a superblock write did not program its members in parallel");
    check(elapsed(1,Op::Read)>=8*4000 && elapsed(4,Op::Read)>=2*4000,"striped reads bypassed per-plane service");
    check(elapsed(1,Op::Write)>=nand.t_erase_block_ns+8*nand.t_program_page_ns &&
          elapsed(4,Op::Write)>=nand.t_erase_block_ns+2*nand.t_program_page_ns,"striped programs bypassed per-plane service");
    rejects([]{Device d(superblock_config(MappingOrganization::Extent,3));},"a width that does not divide the planes was accepted");
    rejects([]{Device d(superblock_config(MappingOrganization::Extent,0));},"a zero superblock width was accepted");
    rejects([]{auto c=superblock_config(MappingOrganization::Page,2); Device d(c);},"the page FTL accepted superblocks");
    {
        Device two(superblock_config(MappingOrganization::Extent,2)); two.prepopulate_mutable_logical_page_range(0,16);
        Device four(superblock_config(MappingOrganization::Extent,4));
        rejects([&]{four.restore_persistent_image(two.persistent_image());},"a checkpoint restored under another superblock width");
    }
}
void superblock_lifecycle(MappingOrganization kind,U width) {
    constexpr U capacity=240;
    const auto c=superblock_config(kind,width);
    Device d(c); double now=0;
    if (kind==MappingOrganization::ObjectSegment) now=d.object_command("CREATE",7,0,capacity,now).finish_ns;
    std::map<U,U> expected;
    U generation=0;
    for (U p=0;p<capacity;++p) { now=io(d,Op::Write,p,now); expected[p]=++generation; }
    std::mt19937_64 random(23);
    for (U step=0;step<1200;++step) {
        const U lpn=random()%capacity;
        // Commands arrive faster than a program completes, so they overlap.
        now+=200;
        if (step%19==0) {
            (void)d.invalidate_logical_pages(lpn,1,now); expected.erase(lpn);
        } else {
            (void)io(d,Op::Write,lpn,now,step%3 ? 4096 : 64,step%3 ? 0 : 128);
            expected[lpn]=++generation;
        }
        if (step%97==0) check(d.stats().accounting_verified,"superblock traffic broke media accounting");
    }
    for (U p=0;p<capacity;++p)
        check(d.mapped_generation(p)==(expected.contains(p) ? std::optional<U>{expected[p]} : std::nullopt),
            "superblock mapping lost live content after updates and reclaim");
    now=d.drain_pending("superblock-checkpoint",now).finish_ns;
    check(d.quiescence_stats().quiescent() && d.stats().accounting_verified,"superblock drain left pending work");
    const auto policy=d.mapping_policy_stats();
    check(policy.data_programs+policy.copy_programs+policy.padding_programs+policy.metadata_programs==d.stats().page_programs,
        "superblock program accounting does not conserve physical writes");
    if (kind==MappingOrganization::Extent || kind==MappingOrganization::ObjectSegment)
        check(policy.cleaned_segments>0 && policy.copy_programs>0,"superblock workload did not exercise the cleaner");
    else check(policy.full_merges>0,"superblock workload did not exercise block merges");
    Device restored(c); restored.restore_persistent_image(d.persistent_image());
    for (U p=0;p<capacity;++p) check(restored.mapped_generation(p)==d.mapped_generation(p),"superblock restore changed payloads");
    for (U p=0;p<capacity;++p) check(restored.mapped_physical_page(p)==d.mapped_physical_page(p),"superblock restore moved pages");
    const auto resumed=io(restored,Op::Write,3,0);
    (void)restored.drain_pending("superblock-resumed",resumed);
    check(restored.mapped_generation(3)==generation+1 && restored.stats().accounting_verified,
        "restored superblock mapping cannot continue");
    std::cout<<to_string(kind)<<" superblock width "<<width<<" programs="<<d.stats().page_programs
        <<" copies="<<policy.copy_programs<<" merges="<<policy.full_merges<<" cleaned="<<policy.cleaned_segments<<'\n';
}
void mapping_snapshot_preserves_following_execution() {
    for (auto kind:{MappingOrganization::Page,MappingOrganization::Block,
                   MappingOrganization::BlockLog,MappingOrganization::Extent}) {
        for (auto mode:{MappingMode::Cached,MappingMode::FullResident}) {
            if (kind!=MappingOrganization::Page && mode==MappingMode::Cached) continue;
            auto c=config(kind);
            c.device.stacks=2;
            c.host.mapping_mode=mode;
            c.host.mapping_entries_per_page=8;
            c.host.mapping_scratch_pages=2;
            c.host.mapping_cache_tag_bytes=16;
            c.host.ctrl_dram_bytes=mode==MappingMode::Cached ? 131072 : 0;
            c.host.ctrl_dram_issue_ns=0.1;
            c.host.ctrl_dram_latency_ns=0.3;
            c.host.mapping_control_compute_ns=0.2;
            c.host.mapping_internal_compute_ns=0.1;
            c.host.mapping_update_ns=0.7;
            c.host.write_coalescing_enabled=true;
            c.host.write_buffer_pages=8;
            c.host.write_buffer_flush_threshold_pages=4;
            Device lean(c), audited(c);
            lean.prepopulate_mutable_logical_page_range(0,64);
            audited.prepopulate_mutable_logical_page_range(0,64);
            auto observe=[&] {
                // The reference deliberately keeps the old full-observation
                // boundary. Compare JSON before observing lean through stats().
                (void)audited.stats();
                std::ostringstream actual,expected;
                lean.write_mapping_snapshot_json(actual);
                audited.write_mapping_snapshot_json(expected);
                check(actual.str()==expected.str(),"lean mapping snapshot differs from full-audit snapshot");
                const auto& a=lean.execution_stats();
                const auto& b=audited.execution_stats();
                check(a.mapping_dram_issue_busy_ns==b.mapping_dram_issue_busy_ns &&
                    a.write_buffer_dram_issue_busy_ns==b.write_buffer_dram_issue_busy_ns,
                    "mapping snapshot failed to preserve fractional issue-port writeback");
            };
            auto same_completion=[](const PhysicalCompletion& a,const PhysicalCompletion& b) {
                check(a.id==b.id && a.tier==b.tier && a.op==b.op &&
                    a.arrival_ns==b.arrival_ns && a.start_ns==b.start_ns && a.finish_ns==b.finish_ns &&
                    a.logical_bytes==b.logical_bytes && a.physical_bytes==b.physical_bytes &&
                    a.breakdown==b.breakdown && a.resource_path==b.resource_path && a.note==b.note,
                    "mapping observation changed subsequent physical completion");
            };
            double now=0;
            auto access=[&](Op op,U lpn,U bytes=4096,U offset=0) {
                const auto before_l=lean.execution_stats();
                const auto before_a=audited.execution_stats();
                const PhysicalRequest request{.id="fractional-observation",.tier=Tier::HBF,.op=op,
                    .address_space=AddressSpace::Logical,.arrival_ns=now+0.1,
                    .addr=lpn*4096+offset,.bytes=bytes};
                const auto a=lean.issue(request),b=audited.issue(request);
                same_completion(a,b); now=a.finish_ns;
                const auto& after_l=lean.execution_stats();
                const auto& after_a=audited.execution_stats();
                check(after_l.mapping_dram_issue_busy_ns-before_l.mapping_dram_issue_busy_ns==
                        after_a.mapping_dram_issue_busy_ns-before_a.mapping_dram_issue_busy_ns &&
                    after_l.write_buffer_dram_issue_busy_ns-before_l.write_buffer_dram_issue_busy_ns==
                        after_a.write_buffer_dram_issue_busy_ns-before_a.write_buffer_dram_issue_busy_ns,
                    "mapping observation changed the next fractional issue delta");
                observe();
            };
            observe();
            // Page alternates stacks per page; whole-block organizations
            // alternate at page 8. Both ports accumulate repeated 0.1 issues.
            for (U step=0;step<40;++step) access(Op::Read,(step%4)*8+step%2,64);
            access(Op::Write,0,64,128);
            check(lean.quiescence_stats().write_buffer_entries>0,
                  "mapping snapshot fixture never observed a dirty buffer");
            access(Op::Write,8,64,128);
            for (U step=0;step<24;++step) {
                access(Op::Write,(step*7)%64,step%3 ? 4096 : 64,step%3 ? 0 : 128);
                if (step%4==0) access(Op::Read,(step*7)%64,64,128);
            }
            const auto drained_l=lean.drain_pending("before-trim",now);
            const auto drained_a=audited.drain_pending("before-trim",now);
            same_completion(drained_l,drained_a); now=drained_l.finish_ns;
            observe();
            const auto trim_l=lean.invalidate_logical_pages(0,3,now+0.1);
            const auto trim_a=audited.invalidate_logical_pages(0,3,now+0.1);
            same_completion(trim_l.completion,trim_a.completion);
            now=trim_l.completion.finish_ns; observe();
            access(Op::Read,0,4*4096); access(Op::Write,1,64,128); access(Op::Read,8,64);
            same_completion(lean.drain_pending("final-observation",now),
                            audited.drain_pending("final-observation",now));
            observe();
            std::ostringstream wear_l,wear_a;
            lean.write_wear_snapshot_json(wear_l); audited.write_wear_snapshot_json(wear_a);
            check(wear_l.str()==wear_a.str(),"mapping observations changed final full wear state");
            const auto prefix=std::filesystem::temp_directory_path()/
                (std::string("hbfsim-lean-snapshot-")+to_string(kind)+"-"+to_string(mode));
            const auto left=prefix.string()+"-lean.image",right=prefix.string()+"-audit.image";
            std::filesystem::remove(left); std::filesystem::remove(right);
            write_persistent_image_file(left,lean.persistent_image());
            write_persistent_image_file(right,audited.persistent_image());
            std::ifstream l(left,std::ios::binary),r(right,std::ios::binary);
            const std::string lb((std::istreambuf_iterator<char>(l)),{}),rb((std::istreambuf_iterator<char>(r)),{});
            check(lb==rb,"mapping observations changed the complete persistent image");
            l.close(); r.close(); std::filesystem::remove(left); std::filesystem::remove(right);
            std::cout<<to_string(kind)<<" "<<to_string(mode)<<" fractional lean/full snapshot exact\n";
        }
    }
}
// Structural host timeline: independent partial writes issued together
// overlap their merge reads and programs on per-page chains, as in the page
// FTL; the host does not serialize them behind each other's NAND reads.
void overlapped_partial_writes() {
    for (auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,
            MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
        auto c=config(kind);
        c.device.channels_per_stack=2; c.device.planes_per_die=8; c.device.blocks_per_plane=16;
        c.device.t_read_page_ns=10000;
        c.host.logical_capacity_bytes=512*4096;
        // Block rewrites copy their whole block, so its writes open fresh
        // unstriped blocks; the others rewrite prepopulated pages striped over
        // all 16 planes (merge read + log/segment append on distinct planes).
        const bool block=kind==MappingOrganization::Block;
        c.host.mapping_superblock_planes=block ? 1 : 16;
        c.host.write_coalescing_enabled=true; c.host.write_buffer_pages=32;
        c.host.write_buffer_completion_requires_flush=true;
        c.host.initial_free_blocks_erased=true;  // bound the tail without first-use erases
        Device d(c); d.prepopulate_mutable_logical_page_range(0,64);
        const U n=16;
        auto target=[&](U i) { return block ? 8*(8+i) : i; };
        const double arrival=100;
        std::vector<double> finish;
        for (U i=0;i<n;++i) finish.push_back(io(d,Op::Write,target(i),arrival,2048));
        const double tail=*std::max_element(finish.begin(),finish.end())-arrival;
        const auto& nand=c.device;
        check(tail<nand.t_read_page_ns+nand.t_program_page_ns+4000,
            "independent partial writes serialized behind each other's merge reads");
        const double read=io(d,Op::Read,target(3),arrival+1,64);
        check(read>=finish[3]+nand.t_read_page_ns-1e-6,"a read overtook the program of its rewritten page");
        const U lpn=target(5);
        (void)io(d,Op::Write,lpn,arrival+1,2048); (void)io(d,Op::Write,lpn,arrival+1,2048,2048);
        (void)d.drain_pending("overlapped-writes",arrival+1);
        check(d.mapped_generation(lpn)==64+n+2,"same-arrival rewrites lost the newest generation");
        check(d.quiescence_stats().quiescent() && d.stats().accounting_verified,"overlapped writes broke accounting");
    }
}
// A coalesced write whose stack buffer is full first evicts a victim. The
// eviction's flush (a block rewrite, a log merge, cleaning) advances the host
// past the translation time this page's chain already had, and the host
// issues the page's merge read and staging after those commands. Using the
// stale chain time issued the merge read before the flush's programs and
// threw "invalid HBF channel transfer" for block and block-log organizations.
void eviction_before_partial_write() {
    for (auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,MappingOrganization::Extent}) {
        auto c=config(kind);
        c.host.write_coalescing_enabled=true; c.host.write_buffer_pages=2;
        Device d(c); d.prepopulate_mutable_logical_page_range(0,64);
        double now=1000; U flushed_pages=0;
        std::mt19937_64 random(11);
        for (U step=0;step<400;++step,now+=1500) {
            const U lpn=random()%64;
            const auto op=random()%3==0 ? Op::Write : Op::Read;
            const U bytes=op==Op::Write && random()%2 ? 1024 : 4096;
            const double finish=io(d,op,lpn,now,bytes);
            check(finish>=now,"a command finished before it arrived");
            flushed_pages+=op==Op::Write;
        }
        (void)d.drain_pending("evictions",now);
        check(d.stats().accounting_verified && d.stats().write_buffer_flushes+d.stats().page_programs>0 &&
            flushed_pages>0,"coalesced partial writes after evictions broke accounting");
    }
}
// Snapshot precondition hbf-initial-free-blocks-erased: never-programmed
// free superblocks open without an erase; a reclaimed block erases on reuse.
void preerased_free_pool() {
    for (bool preerased:{false,true}) {
        auto c=superblock_config(MappingOrganization::BlockLog,4);
        c.host.initial_free_blocks_erased=preerased;
        Device d(c);
        double now=io(d,Op::Write,0,0);
        check(d.stats().block_erases==(preerased ? 0 : 1),"a fresh wide superblock's first program erased (or skipped) wrongly");
        if (preerased) {
            check(d.stats().preconditioned_erased_blocks>0,"no free block was declared pre-erased");
            rejects([&]{(void)d.persistent_image();},"a persistent image silently dropped pre-erased blocks");
        } else check(d.stats().preconditioned_erased_blocks==0,"the precondition leaked into the default");
        std::mt19937_64 random(5);
        for (U step=0;step<1500;++step) now=io(d,Op::Write,random()%200,now);
        now=d.drain_pending("preerased",now).finish_ns;
        const auto& stats=d.stats();
        check(stats.block_erases>0,"sustained updates never reused a reclaimed block");
        check(stats.block_erases==stats.erase_requests+stats.auto_erase_requests && stats.accounting_verified,
            "pre-erased blocks broke the erase ledger");
    }
    rejects([]{auto c=config(MappingOrganization::Extent); c.host.initial_free_blocks_erased=true;
        Device fresh(config(MappingOrganization::Extent)); fresh.prepopulate_mutable_logical_page_range(0,8);
        Device d(c); d.restore_persistent_image(fresh.persistent_image());},
        "a restored image declared its free blocks pre-erased");
}
// The mapping budget holds the index, write buffer and workspace; common
// block state is reserved outside it for every organization.
void structural_budget_excludes_block_state() {
    for (auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,MappingOrganization::Extent}) {
        auto c=superblock_config(kind,4);
        c.host.write_coalescing_enabled=true; c.host.write_buffer_pages=2;
        c.host.write_buffer_completion_requires_flush=true;
        const U block_pages=4*c.device.pages_per_block;
        const U minimum=2*4096+(8192+block_pages*16)+64;
        const U block_state=64*32+64*8/8;
        check(block_state>64,"fixture block state must exceed the index slack");
        c.host.ctrl_dram_bytes=minimum-1;
        bool too_small=false;
        try { Device d(c); } catch (const std::exception& error) { too_small=std::string(error.what()).find("budget")!=std::string::npos; }
        check(too_small,"a budget below write buffer, workspace and records was accepted");
        c.host.ctrl_dram_bytes=minimum;
        Device d(c);
        check(d.mapping_policy_stats().metadata_bytes>=block_state,"block state is no longer reported");
        if (kind==MappingOrganization::Extent) {
            (void)io(d,Op::Write,0,0);  // one 48-byte run fits the 64-byte slack
            bool exhausted=false;
            try { (void)io(d,Op::Write,5,1); }
            catch (const std::exception& error) { exhausted=std::string(error.what()).find("budget")!=std::string::npos; }
            check(exhausted,"the index outgrew the budget without failing");
        }
    }
}
int main() {
    try {
        extent_index(); host_write_accounting(); log_merges(); full_block_replacement(); buffering_and_objects(); initial_images();
        shared_object_segment(); writeback_timing_and_ownership(); bounded_foreground_range_reads(); striped_blocks_and_restore();
        mapping_snapshot_preserves_following_execution();
        asynchronous_programs(); superblock_striping(); overlapped_partial_writes(); eviction_before_partial_write();
        preerased_free_pool();
        structural_budget_excludes_block_state();
        for(auto kind:{MappingOrganization::Block,MappingOrganization::BlockLog,MappingOrganization::Extent,MappingOrganization::ObjectSegment}) {
            lifecycle(kind);
            for (U width:{U{1},U{2},U{4}}) superblock_lifecycle(kind,width);
        }
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
    std::cout<<"mapping organizations, recovery, GC, merges and object lifecycle passed\n";
}
