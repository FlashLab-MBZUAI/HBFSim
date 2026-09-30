#include "host/hbf_mapping_policy.hpp"
#include "host/hbf_controller.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace hbfsim::host {
using namespace physical;
namespace {
using U = std::uint64_t;
constexpr U none = ~U{0};
// Modeled packed tree nodes, separate from the 24/32-byte checkpoint
// records and from allocator overhead of this simulator's C++ containers.
constexpr U extent_node_bytes=48, object_index_bytes=96;
U add(U a, U b) { if (b > none-a) throw std::overflow_error("mapping capacity overflow"); return a+b; }
U mul(U a, U b) { if (a && b > none/a) throw std::overflow_error("mapping capacity overflow"); return a*b; }
U ceil_div(U a, U b) { if (!b) throw std::invalid_argument("zero mapping geometry"); return a/b + (a%b != 0); }
// Blocks and pages below are superblocks (mapping_superblock_planes physical
// blocks) and their pages; block state and validity bits stay per physical
// block and page.
struct Layout { U blocks, block_pages, metadata_blocks, data_blocks, logical_pages, budget, fixed_bytes; };
// Mapping-budget state besides the index: the write buffer (4096 B per page,
// as the page FTL's write buffer; none when it lives in HBF device DRAM) and
// per-stack merge/cleaning workspace.
U mapping_overhead_bytes(const HbfConfig& c,U block_pages) {
    const U buffers = c.host.write_coalescing_enabled && !c.host.device_dram_capacity_denominator ?
        mul(mul(c.host.write_buffer_pages,c.device.stacks),4096) : 0;
    return add(buffers,mul(c.device.stacks,add(8192,mul(block_pages,16))));
}
Layout layout(const HbfConfig& c) {
    U physical_blocks = 1;
    for (auto factor : {c.device.stacks,c.device.channels_per_stack,c.device.dies_per_channel,
            c.device.planes_per_die,c.device.blocks_per_plane}) physical_blocks = mul(physical_blocks,factor);
    if (!physical_blocks || !c.device.pages_per_block || c.device.page_size_bytes != 4096)
        throw std::invalid_argument("structural mapping requires nonempty 4 KiB NAND geometry");
    const U width = c.host.mapping_superblock_planes;
    if (!width || (physical_blocks/c.device.blocks_per_plane)%width)
        throw std::invalid_argument("mapping superblock planes must divide the plane count");
    const U blocks = physical_blocks/width, block_pages = mul(width,c.device.pages_per_block);
    const U pages = mul(physical_blocks,c.device.pages_per_block);
    const auto organization = c.host.mapping_organization;
    const U log_records = organization == MappingOrganization::BlockLog ?
        mul(c.host.mapping_log_blocks, mul(block_pages,16)+32) : 0;
    // Upper bounds for the actual fixed-width checkpoint records, including
    // object directory, allocation prefixes, bitmap and primary index.
    const U table = organization == MappingOrganization::Block || organization == MappingOrganization::BlockLog ?
        add(mul(blocks,48+ceil_div(block_pages,64)*8),log_records) :
        mul(pages,organization == MappingOrganization::ObjectSegment ? 64 : 24);
    const U bytes = add(add(table,mul(blocks,24)),512);
    const U metadata = ceil_div(bytes,mul(block_pages,4096))+1;
    const U reserved = add(mul(metadata,2),
        organization == MappingOrganization::BlockLog ? U{c.host.mapping_log_blocks}+2 : 2);
    if (reserved >= blocks || (organization == MappingOrganization::BlockLog && !c.host.mapping_log_blocks))
        throw std::invalid_argument("mapping geometry leaves no data blocks after metadata/copy/log reserves");
    const U maximum = mul(blocks-reserved,block_pages);
    const U logical = c.host.logical_capacity_bytes ? c.host.logical_capacity_bytes/4096 : maximum;
    if (!logical || logical > maximum || c.host.logical_capacity_bytes%4096)
        throw std::invalid_argument("logical capacity exceeds structural mapping's metadata/copy/log reserves");
    // Common block state (32 B per physical block plus one validity bit per
    // page) is reported but, as in the page FTL, reserved outside the mapping
    // budget: the budget holds the index and the workspace, plus the write
    // buffer only while that buffer lives in HBM (mapping_overhead_bytes).
    const U fixed = add(mul(physical_blocks,32),ceil_div(pages,8));
    const U resident_table=organization==MappingOrganization::Block || organization==MappingOrganization::BlockLog ?
        table : mul(pages,extent_node_bytes+(organization==MappingOrganization::ObjectSegment ? object_index_bytes : 0));
    const U budget = mul(ceil_div(add(resident_table,mapping_overhead_bytes(c,block_pages)),
        mul(c.device.stacks,4096)),mul(c.device.stacks,4096));
    return {blocks,block_pages,metadata,blocks-2*metadata,logical,budget,fixed};
}
U checksum(const std::vector<U>& words, std::size_t count) {
    U hash = 1469598103934665603ULL;
    for (std::size_t i=0;i<count;++i)
        for (unsigned byte=0;byte<8;++byte) { hash ^= (words[i]>>(8*byte))&255; hash *= 1099511628211ULL; }
    return hash;
}
}

const char* to_string(MappingOrganization value) {
    switch (value) {
    case MappingOrganization::Page: return "page";
    case MappingOrganization::Block: return "block";
    case MappingOrganization::BlockLog: return "block-log";
    case MappingOrganization::Extent: return "extent";
    case MappingOrganization::ObjectSegment: return "object-segment";
    }
    throw std::invalid_argument("unknown mapping organization");
}
MappingOrganization parse_mapping_organization(std::string_view value) {
    for (auto kind : {MappingOrganization::Page,MappingOrganization::Block,MappingOrganization::BlockLog,
            MappingOrganization::Extent,MappingOrganization::ObjectSegment})
        if (value == to_string(kind)) return kind;
    throw std::invalid_argument("mapping organization must be page, block, block-log, extent, or object-segment");
}

std::optional<U> HbfExtentIndex::lookup(U lpn) const {
    auto it=runs_.upper_bound(lpn);
    if (it==runs_.begin()) return {};
    --it;
    return lpn-it->first < it->second.pages ? std::optional<U>{it->second.ppn+lpn-it->first} : std::nullopt;
}
void HbfExtentIndex::erase(U first,U count) {
    if (!count) return;
    const auto end=add(first,count);
    auto it=runs_.upper_bound(first);
    if (it!=runs_.begin()) --it;
    while (it!=runs_.end() && it->first<end) {
        const auto start=it->first, length=it->second.pages, ppn=it->second.ppn;
        if (start+length<=first) { ++it; continue; }
        it=runs_.erase(it);
        if (start<first) runs_.emplace(start,Run{first-start,ppn});
        if (start+length>end) { runs_.emplace(end,Run{start+length-end,ppn+end-start}); break; }
    }
}
void HbfExtentIndex::assign(U lpn,U ppn) {
    erase(lpn,1);
    auto it=runs_.emplace(lpn,Run{1,ppn}).first;
    if (it!=runs_.begin()) {
        auto prev=std::prev(it);
        if (prev->first+prev->second.pages==lpn && prev->second.ppn+prev->second.pages==ppn) {
            ++prev->second.pages; runs_.erase(it); it=prev;
        }
    }
    auto next=std::next(it);
    if (next!=runs_.end() && it->first+it->second.pages==next->first &&
        it->second.ppn+it->second.pages==next->second.ppn) {
        it->second.pages+=next->second.pages; runs_.erase(next);
    }
}

HbfConfig HbfMappingPolicy::media_config(HbfConfig c) {
    if (c.host.mapping_organization==MappingOrganization::Page) {
        if (c.host.mapping_superblock_planes!=1)
            throw std::invalid_argument("the page FTL stripes pages itself; mapping superblock planes must be 1");
        return c;
    }
    const auto resolved=layout(c);
    if (c.host.mapping_placement!=MappingPlacement::Host)
        throw std::invalid_argument("structural mapping is a host policy");
    if (c.host.mapping_sram_bytes || c.host.static_wear_leveling_erase_gap)
        throw std::invalid_argument("structural mapping uses a host index and dynamic least-worn allocation; page-FTL SRAM/static-WL options do not apply");
    if (c.host.ctrl_dram_capacity_denominator)
        c.host.ctrl_dram_bytes=derive_controller_dram_budget(c,c.host.ctrl_dram_capacity_denominator).total_bytes;
    if (!c.host.ctrl_dram_bytes) c.host.ctrl_dram_bytes=resolved.budget;
    if (c.host.ctrl_dram_bytes < add(mapping_overhead_bytes(c,resolved.block_pages),64))
        throw std::invalid_argument("structural mapping budget cannot hold its write buffer, workspace and mapping records");
    if (c.host.write_coalescing_enabled && (!c.host.write_buffer_pages ||
        c.host.write_buffer_flush_threshold_pages>c.host.write_buffer_pages))
        throw std::invalid_argument("invalid structural mapping write buffer capacity/threshold");
    c.host.mapping_mode=MappingMode::RawPhysical;
    c.host.mapping_cache_layout=MappingCacheLayout::Page;
    c.host.mapping_scratch_pages=c.host.mapping_sram_bytes=0;
    c.host.write_coalescing_enabled=false;
    c.host.write_buffer_pages=c.host.write_buffer_flush_threshold_pages=0;
    c.host.auto_gc_enabled=false;
    c.host.gc_low_watermark_pages=c.host.gc_hard_watermark_pages=0;
    c.host.static_wear_leveling_erase_gap=0;
    c.host.zone_size_blocks=1;
    c.host.logical_capacity_bytes=resolved.logical_pages*4096;
    return c;
}
HbfMappingPolicy::HbfMappingPolicy(const HbfConfig& requested,const HbfConfig& media)
    : requested_(requested),organization_(requested.host.mapping_organization) {
    const auto sizes=layout(requested);
    width_=requested.host.mapping_superblock_planes;
    physical_pages_per_block_=requested.device.pages_per_block;
    pages_per_block_=sizes.block_pages;
    total_blocks_=sizes.blocks; metadata_blocks_=sizes.metadata_blocks; block_state_bytes_=sizes.fixed_bytes;
    data_blocks_=sizes.data_blocks; logical_pages_=sizes.logical_pages;
    planes_=total_blocks_*width_/requested_.device.blocks_per_plane;
    planes_per_stack_=planes_/requested_.device.stacks;
    blocks_.resize(total_blocks_);
    for (U b=0;b<data_blocks_;++b) free_.insert(allocation_rank(b));
    program_credits_.resize(mul(requested_.device.stacks,requested_.device.channels_per_stack));
    lookup_workers_.resize(requested_.host.mapping_compute_workers ?
        requested_.host.mapping_compute_workers : requested_.device.stacks);
    lookup_issue_.resize(requested_.device.stacks);
    stats_.reserved_bytes=media.host.ctrl_dram_bytes;
    update_footprint();
}
U HbfMappingPolicy::memory_footprint() const {
    U bytes=block_state_bytes_;
    // Every block entry has the same bitmap size. Track variable log entries
    // at mutation sites instead of scanning the entire map after each page.
    bytes+=block_map_.size()*(32+ceil_div(pages_per_block_,64)*8)+log_update_entries_*16;
    bytes+=extents_.runs().size()*extent_node_bytes+objects_.size()*object_index_bytes;
    return bytes;
}
std::pair<U,U> HbfMappingPolicy::allocated_pages() const {
    U data=0,metadata=0;
    for (U b=0;b<total_blocks_;++b) (b<data_blocks_ ? data : metadata)+=blocks_[b].pages.size();
    return {data,metadata};
}
U HbfMappingPolicy::buffer_capacity() const {
    return requested_.host.write_coalescing_enabled ?
        requested_.host.write_buffer_pages*requested_.device.stacks : 0;
}
bool HbfMappingPolicy::buffer_on_device(const HbfController& h) const {
    return requested_.host.write_coalescing_enabled && h.device_dram_enabled();
}
U HbfMappingPolicy::buffer_stack_pages(U lpn) const {
    return std::count_if(buffer_.begin(),buffer_.end(),[&](const auto& entry) {
        return entry.first%requested_.device.stacks==lpn%requested_.device.stacks;
    });
}
void HbfMappingPolicy::update_footprint() {
    stats_.metadata_bytes=memory_footprint();
    stats_.peak_metadata_bytes=std::max(stats_.peak_metadata_bytes,stats_.metadata_bytes);
    stats_.extent_records=extents_.runs().size();
    stats_.active_logs=active_log_count_;
    // Block state is common controller state outside the mapping budget.
    if (stats_.metadata_bytes-block_state_bytes_+mapping_overhead_bytes(requested_,pages_per_block_)>
            stats_.reserved_bytes)
        throw std::runtime_error("structural mapping exhausted its configured metadata memory budget");
}
std::optional<U> HbfMappingPolicy::lookup(U lpn) const {
    const auto ppn=locate(lpn);
    return ppn ? std::optional<U>{physical(*ppn)} : std::nullopt;
}
std::optional<U> HbfMappingPolicy::locate(U lpn) const {
    if (organization_==MappingOrganization::Extent || organization_==MappingOrganization::ObjectSegment)
        return extents_.lookup(lpn);
    auto entry=block_map_.find(lpn/pages_per_block_);
    const auto offset=lpn%pages_per_block_;
    if (entry==block_map_.end() || !(entry->second.valid[offset/64]&(U{1}<<(offset%64)))) return {};
    const auto update=entry->second.updates.find(offset);
    return update==entry->second.updates.end() ? entry->second.base*pages_per_block_+offset : update->second;
}
std::optional<U> HbfMappingPolicy::generation(U lpn) const {
    if (const auto pending=buffer_.find(lpn);pending!=buffer_.end()) return pending->second.generation;
    const auto ppn=locate(lpn);
    return ppn ? std::optional<U>{blocks_.at(*ppn/pages_per_block_).pages.at(*ppn%pages_per_block_).generation} : std::nullopt;
}
void HbfMappingPolicy::start(HbfController& h,double arrival) {
    if (!std::isfinite(arrival) || arrival<0 || arrival<last_arrival_ns_)
        throw std::invalid_argument("mapping requests need finite, nondecreasing arrival times");
    last_arrival_ns_=arrival;
    // The host FTL executes commands in arrival order. NAND work issued for
    // earlier commands keeps running on its planes; a command waits for it
    // only where it depends on that work (see settle, release, read_page).
    ready_ns_=std::max(ready_ns_,arrival);
    media_ns_=data_ns_=update_ns_=ready_ns_;
    settle(h,ready_ns_);
    h.reservation_causal_watermark_ns_=ready_ns_;
    h.seed_media_image();
}
void HbfMappingPolicy::settle(HbfController& h,double through) {
    h.materialize_committed_state_through(through);
    while (!inflight_by_finish_.empty() && inflight_by_finish_.begin()->first<=through) {
        const U ppn=inflight_by_finish_.begin()->second;
        inflight_by_finish_.erase(inflight_by_finish_.begin());
        const auto found=inflight_.find(ppn);
        const auto page=h.programmed_pages_.find(ppn);
        if (found==inflight_.end() || page==h.programmed_pages_.end())
            throw std::logic_error("completed mapping program was not materialized");
        // Raw page OOB carries a generation identity for lossless-copy/recovery
        // verification. LPN ownership is recovered from the persisted primary index.
        page->second.lpn=found->second.generation;
        if (found->second.retire) h.invalidate_ppn(ppn);
        inflight_.erase(found);
    }
}
void HbfMappingPolicy::wait_until(double at,PhysicalCompletion& out) {
    if (at<=ready_ns_) return;
    out.breakdown.scheduler_queue_wait_ns+=at-ready_ns_;
    ready_ns_=at;
}
// Controller-memory transfer on the ordered host stream.
void HbfMappingPolicy::memory(HbfController& h,U bytes,Op op,PhysicalCompletion& out) {
    Breakdown work;
    ready_ns_=memory_at(h,bytes,op,ready_ns_,work,trace_spans_enabled(trace_) ? &out.spans : nullptr);
    out.breakdown+=work; h.stats_.stage_work+=work;
}
// Controller-memory transfer on one page's payload chain, starting at `at`.
double HbfMappingPolicy::memory_at(HbfController& h,U bytes,Op op,double at,Breakdown& work,
        std::vector<TraceSpan>* spans) {
    if (seeding_ || !bytes) return at;
    // An index scan can exceed a per-stack reservation. Stream it through
    // page-sized transfers; never allocate an unbounded temporary HBM buffer.
    U remaining=bytes;
    while (remaining) {
        const U chunk=std::min<U>(4096,remaining);
        at=h.host_memory_transfer(memory_stack_,chunk,op,at,work,spans);
        remaining-=chunk;
    }
    stats_.memory_bytes+=bytes;
    return at;
}
// The index entry that resolves lpn: its logical block, or the extent run at
// or before it (an unmapped gap is resolved by the same search).
U HbfMappingPolicy::index_entry(U lpn) const {
    if (organization_!=MappingOrganization::Extent && organization_!=MappingOrganization::ObjectSegment)
        return lpn/pages_per_block_;
    const auto& runs=extents_.runs();
    const auto after=runs.upper_bound(lpn);
    return after==runs.begin() ? absent : std::prev(after)->first;
}
// An index access the rest of the command depends on (maintenance, merges,
// cleaning, invalidation): the host waits for its completion.
void HbfMappingPolicy::lookup_work(HbfController& h,U lpn,bool update,PhysicalCompletion& out) {
    if (seeding_) return;
    Breakdown work;
    ready_ns_=index_access(h,lpn,update,work,trace_spans_enabled(trace_) ? &out.spans : nullptr,false);
    out.breakdown+=work; h.stats_.stage_work+=work;
}
// Mapping compute on the earliest available worker of the shared pool.
void HbfMappingPolicy::lookup_compute(HbfController& h,double work_ns,double& at,Breakdown& work) {
    if (work_ns<=0) return;
    auto* worker=&lookup_workers_.front();
    for (auto& candidate:lookup_workers_)
        if (candidate.preview_start(at,work_ns)<worker->preview_start(at,work_ns)) worker=&candidate;
    const auto slot=h.reserve(at,work_ns,*worker);
    work.translation_ns+=work_ns; work.scheduler_queue_wait_ns+=slot.wait_ns;
    at=slot.finish_ns;
}
// An index lookup or update has the page FTL's host resources: a mapping
// worker and the stack's controller-memory issue slot. Its probes are
// dependent round trips, but independent accesses overlap, so issuing alone
// advances the ordered index stream; accesses to one stack's index issue in
// command order. `resolved`: an earlier page of this range already looked up
// lpn's index entry, so only a block's log directory is consulted (if it has
// a log). Returns when this entry's translation (or update) is complete.
double HbfMappingPolicy::index_access(HbfController& h,U lpn,bool update,Breakdown& work,
        std::vector<physical::TraceSpan>* spans,bool resolved) {
    if (seeding_) return ready_ns_;
    memory_stack_=lpn%requested_.device.stacks;
    U probes=1,record=16;
    bool logged=false;
    if (organization_==MappingOrganization::Extent || organization_==MappingOrganization::ObjectSegment) {
        probes=std::max<U>(1,std::bit_width(extents_.runs().size())); record=extent_node_bytes;
        if (organization_==MappingOrganization::ObjectSegment) probes+=std::max<U>(1,std::bit_width(objects_.size()));
    } else if (organization_==MappingOrganization::BlockLog) {
        const auto found=block_map_.find(lpn/pages_per_block_);
        logged=found!=block_map_.end() && found->second.log!=absent;
        if (logged) ++probes;
    }
    if (resolved) {
        if (!logged) return ready_ns_;
        probes=1;
    }
    const double started=ready_ns_;
    double at=ready_ns_,issued=ready_ns_;
    if (!resolved) lookup_compute(h,requested_.host.mapping_control_compute_ns,at,work);
    const auto before=h.stats_.host_hbm_read_bytes+h.stats_.host_hbm_write_bytes;
    for (U probe=0;probe<probes;++probe) {
        lookup_compute(h,requested_.host.mapping_internal_compute_ns,at,work);
        const auto issue=h.reserve(at,requested_.host.ctrl_dram_issue_ns,lookup_issue_.at(memory_stack_));
        work.scheduler_queue_wait_ns+=issue.wait_ns;
        const auto data=h.host_memory_transfer(memory_stack_,record,update ? Op::Write : Op::Read,
            issue.start_ns,work,spans);
        if (probe==0) issued=issue.finish_ns;
        at=std::max({issue.finish_ns,data,issue.start_ns+requested_.host.ctrl_dram_latency_ns});
    }
    if (update) lookup_compute(h,requested_.host.mapping_update_ns,at,work);
    work.mapping_dram_ns+=probes*requested_.host.ctrl_dram_latency_ns;
    stats_.metadata_hbm_bytes+=h.stats_.host_hbm_read_bytes+h.stats_.host_hbm_write_bytes-before;
    stats_.memory_bytes+=probes*record;
    stats_.lookup_work_ns+=at-started;
    update ? ++stats_.updates : ++stats_.lookups;
    ready_ns_=issued;
    return at;
}
void HbfMappingPolicy::read_range(HbfController& h,const PhysicalRequest& request,PhysicalCompletion& out) {
    const U first=request.addr/4096,last=(request.addr+request.bytes-1)/4096;
    auto* spans=trace_spans_enabled(trace_) ? &out.spans : nullptr;
    Breakdown work;
    const auto first_ppn=lookup(first);
    const auto ingress_stack=first_ppn ? h.stack_of_block(*first_ppn/physical_pages_per_block_) : first%requested_.device.stacks;
    auto& ingress=h.logic_dies_.at(ingress_stack).ingress;
    const auto dispatch=h.reserve(ready_ns_,requested_.device.logic_scheduler_issue_ns,ingress);
    work.ingress_queue_wait_ns+=dispatch.wait_ns;
    work.command_ns+=requested_.device.logic_scheduler_issue_ns;
    const auto entity="stack"+std::to_string(ingress_stack)+"/logic";
    add_trace_span(spans,"logic_scheduler_issue","logic",entity,dispatch.start_ns,dispatch.finish_ns);
    ready_ns_=dispatch.finish_ns;
    if (last!=first) {
        const auto split=h.reserve(ready_ns_,requested_.device.address_generation_ns,ingress);
        work.ingress_queue_wait_ns+=split.wait_ns;
        work.address_mapping_ns+=requested_.device.address_generation_ns;
        add_trace_span(spans,"read_split","logic",entity,split.start_ns,split.finish_ns);
        ready_ns_=split.finish_ns;
    }
    // Match native scalar-read pacing: reserve one budget per physical stack,
    // then deposit heat only for actual NAND reads. Host-buffer/erased responses
    // do not require media admission. No O(pages) queue or payload is allocated.
    std::vector<double> thermal_ready(requested_.device.stacks,ready_ns_);
    double complete=ready_ns_;
    if (requested_.device.thermal_enabled) {
        std::vector<U> pages(requested_.device.stacks);
        for (U lpn=first;lpn<=last;++lpn)
            if (!buffer_.contains(lpn))
                if (const auto ppn=lookup(lpn)) ++pages[h.stack_of_block(*ppn/physical_pages_per_block_)];
        for (std::size_t stack=0;stack<pages.size();++stack) if (pages[stack]) {
            const auto admission=h.thermal_pace_media(stack,ready_ns_,pages[stack]*h.thermal_read_energy_j_,
                pages[stack],work,spans);
            thermal_ready[stack]=admission.ready_ns;
            complete=std::max(complete,admission.budget_finish_ns);
        }
    }
    // Lookups are issued in page order and overlap like the page FTL's; each
    // page waits for its own entry's translation. The native per-stack credit
    // is held from translation admission through the complete response,
    // including HBM hits.
    double index_ready=ready_ns_,entry_ready=ready_ns_;
    // One lookup resolves an index entry (logical block or extent run) for
    // all of its pages in the range, as a block or extent FTL does.
    U resolved_entry=absent;
    for (U lpn=first;lpn<=last;++lpn) {
        const U begin=std::max(request.addr,lpn*4096),end=std::min(request.addr+request.bytes,(lpn+1)*4096);
        const bool buffered=buffer_.contains(lpn);
        const auto ppn=buffered ? std::optional<U>{} : lookup(lpn);
        const auto stack=ppn ? h.stack_of_block(*ppn/physical_pages_per_block_) : lpn%requested_.device.stacks;
        ready_ns_=h.admit_foreground_page_read(stack,std::max(index_ready,thermal_ready[stack]),work,spans);
        const U entry=index_entry(lpn);
        const bool same=lpn!=first && entry==resolved_entry;
        const double translated=index_access(h,lpn,false,work,spans,same);
        entry_ready=same ? std::max(entry_ready,translated) : translated;
        resolved_entry=entry;
        index_ready=ready_ns_;
        ready_ns_=std::max(ready_ns_,entry_ready);
        if (buffered) {
            // A buffered page is readable once its merge read and payload are assembled.
            ++stats_.buffer_hits;
            ready_ns_=std::max(ready_ns_,buffer_.at(lpn).ready_ns);
            if (buffer_on_device(h))
                ready_ns_=h.device_dram_respond(buffer_stack(lpn),lpn/requested_.device.stacks,
                    end-begin,ready_ns_,work,spans);
            else memory(h,end-begin,Op::Read,out);
        }
        else if (ppn) {
            const U offset=begin-lpn*4096;
            const U device_bytes=(ceil_div(offset+end-begin,64)-offset/64)*64;
            // As in the page FTL, the closest copy answers: the NAND bank's
            // decoded page buffer, then the device DRAM read cache (keyed by
            // physical page), then the array.
            std::optional<double> cached;
            if (h.read_buffer_contains(*ppn,ready_ns_)) {
                ready_ns_=h.serve_read_from_read_buffer(*ppn,device_bytes,ready_ns_,work,spans);
                if (h.device_dram_enabled()) h.device_dram_fill(*ppn,ready_ns_,spans);
            } else if (h.device_dram_enabled() && (cached=h.device_dram_probe(*ppn)))
                ready_ns_=h.device_dram_serve_hit(*ppn,device_bytes,ready_ns_,*cached,work,spans);
            else {
                ++h.stats_.read_buffer_misses;
                double decoded_ready=0;
                ready_ns_=h.schedule_read_page(*ppn,ready_ns_,work,spans,
                    HbfController::TransactionSource::User,request.heatmap_source,
                    HbfController::ReadPayloadRoute::External,device_bytes,&decoded_ready,true);
                h.read_buffer_insert(*ppn,decoded_ready);
                if (h.device_dram_enabled()) h.device_dram_fill(*ppn,decoded_ready,spans);
                ++h.stats_.page_reads; h.stats_.physical_read_bytes+=4096;
            }
            ++stats_.data_reads;
        } else memory(h,end-begin,Op::Read,out); // host supplies erased bytes
        h.complete_foreground_page_read(stack,ready_ns_);
        complete=std::max(complete,ready_ns_);
        // All later lookup/admission times are at least index_ready. Retire
        // old HBF calendar/cache history without advancing to a page's future
        // completion. Shared HBM can still receive an independent application
        // request before index_ready; only the joint scheduler can retire it.
        h.advance_hbf_frontier(index_ready);
    }
    out.breakdown+=work; h.stats_.stage_work+=work;
    // The host is free once its lookup stream ends; the pages complete later.
    media_ns_=std::max(media_ns_,complete);
    ready_ns_=index_ready;
}
void HbfMappingPolicy::read_page(HbfController& h,U ppn,Purpose purpose,PhysicalCompletion& out) {
    if (seeding_) return;
    const auto admission_events=h.stats_.page_read_admission_events;
    const auto waited_pages=h.stats_.page_read_admission_waited_pages;
    const auto admission_wait=h.stats_.page_read_admission_wait_ns;
    const auto admission_max=h.stats_.page_read_admission_max_wait_ns;
    auto completed=h.issue_media(PhysicalRequest{.id=out.id+"/read",.tier=Tier::HBF,.op=Op::Read,
        .address_space=AddressSpace::Physical,.trace=trace_,.arrival_ns=ready_ns_,.addr=physical(ppn)*4096,.bytes=4096});
    ready_ns_=completed.finish_ns; out.breakdown+=completed.breakdown;
    --h.stats_.read_requests;
    --h.stats_.scalar_read_requests;
    --h.stats_.scalar_read_pages;
    h.stats_.page_read_admission_events=admission_events;
    h.stats_.page_read_admission_waited_pages=waited_pages;
    h.stats_.page_read_admission_wait_ns=admission_wait;
    h.stats_.page_read_admission_max_wait_ns=admission_max;
    out.spans.insert(out.spans.end(),completed.spans.begin(),completed.spans.end());
    // The host needs these bytes (copy, merge, recovery or read-modify-write)
    // before it continues; the media request itself waited for any program of
    // this page that was still in flight.
    settle(h,ready_ns_);
    if (purpose==Purpose::Metadata) ++stats_.metadata_reads;
    else if (purpose==Purpose::Copy) ++stats_.copy_reads;
    else ++stats_.data_reads;
    memory(h,4096,Op::Write,out);
}
// Read-modify-write of a partial page: the old version is read into the
// controller buffer on this page's own chain (as the page FTL does); the host
// keeps issuing. The media read waits for any program of that page in flight.
double HbfMappingPolicy::merge_read(HbfController& h,U page,double at,HeatmapTrafficSource source,
        Breakdown& work,std::vector<TraceSpan>* spans) {
    if (seeding_) return at;
    const U ppn=physical(page);
    if (buffer_on_device(h)) {
        // The old image merges in device DRAM: a cached copy is read there,
        // otherwise NAND decodes it into DRAM without leaving the device. The
        // version is superseded by this write, so it is not kept as a line.
        const auto stack=h.stack_of_block(ppn/physical_pages_per_block_);
        if (const auto cached=h.device_dram_cached(ppn))
            return h.device_dram_transfer(stack,4096,false,std::max(at,*cached),work,spans,
                "write_buffer_merge_device_dram");
        const double read=h.schedule_read_page(ppn,at,work,spans,HbfController::TransactionSource::User,
            source,HbfController::ReadPayloadRoute::DeviceDram,0);
        ++h.stats_.page_reads; h.stats_.physical_read_bytes+=4096;
        ++stats_.data_reads;
        media_ns_=std::max(media_ns_,read);
        return h.device_dram_transfer(stack,4096,true,read,work,spans,"write_buffer_merge_device_dram");
    }
    const double read=h.schedule_read_page(ppn,at,work,spans,HbfController::TransactionSource::User,source,
        HbfController::ReadPayloadRoute::HostBuffer,0);
    ++h.stats_.page_reads; h.stats_.physical_read_bytes+=4096;
    ++stats_.data_reads;
    media_ns_=std::max(media_ns_,read);
    return memory_at(h,4096,Op::Write,read,work,spans);
}
void HbfMappingPolicy::program(HbfController& h,U page,Tag tag,Purpose purpose,PhysicalCompletion& out,
        double data_ready) {
    auto& block=blocks_.at(page/pages_per_block_);
    // Sequential superblock pages keep every member block sequential.
    if (page%pages_per_block_!=block.pages.size()) throw std::logic_error("mapping attempted a nonsequential NAND program");
    const U ppn=physical(page);
    if (seeding_) {
        h.prepopulate_raw_media_page(ppn*4096);
        // Raw page OOB carries a generation identity for lossless-copy/recovery
        // verification. LPN ownership is recovered from the persisted primary index.
        h.programmed_pages_.at(ppn).lpn=tag.generation;
        if (purpose==Purpose::Padding) h.invalidate_ppn(ppn);
    } else {
        // The page leaves controller memory once its payload is assembled.
        // Slots of one block start in allocation order (NAND append order).
        Breakdown work;
        auto* spans=trace_spans_enabled(trace_) ? &out.spans : nullptr;
        const double earliest=std::max({ready_ns_,data_ready,block.appended_ns});
        // A write-buffer page staged in device DRAM (on its LPN's stack)
        // programs without another HBIO crossing when its target block is
        // on that stack; otherwise it leaves over HBIO and enters the target
        // stack like any host page.
        const bool staged=purpose==Purpose::Data && buffer_on_device(h);
        const U target_stack=h.stack_of_block(ppn/physical_pages_per_block_);
        const bool resident=staged && buffer_stack(tag.lpn)==target_stack;
        double assembled;
        if (staged) {
            const U source_stack=buffer_stack(tag.lpn);
            assembled=h.device_dram_transfer(source_stack,4096,false,earliest,work,spans,
                "write_buffer_flush_device_dram");
            if (!resident)
                assembled=h.schedule_external_read_egress(source_stack*requested_.device.channels_per_stack+
                    (tag.lpn/requested_.device.stacks)%requested_.device.channels_per_stack,4096,assembled,
                    work,spans,"write_buffer_move_hbio_out","device-DRAM write buffer to another stack");
        } else assembled=memory_at(h,4096,Op::Read,earliest,work,spans);
        out.breakdown+=work; h.stats_.stage_work+=work;
        block.appended_ns=assembled;
        // Each channel holds a bounded number of programs in flight; the host
        // waits for the earliest one to finish before issuing past that bound.
        auto& credits=program_credits_.at(h.channel_index(h.decode_ppn(ppn)));
        if (credits.size()>=requested_.device.outstanding_write_pages_per_channel) {
            std::pop_heap(credits.begin(),credits.end(),std::greater<double>{});
            wait_until(credits.back(),out);
            credits.pop_back();
        }
        h.program_payload_on_device_=resident;
        const auto completed=h.issue_media(PhysicalRequest{.id=out.id+"/program",.tier=Tier::HBF,.op=Op::Write,
            .address_space=AddressSpace::Physical,.trace=trace_,.arrival_ns=ready_ns_,.addr=ppn*4096,.bytes=4096},
            assembled);
        h.program_payload_on_device_=false;
        out.breakdown+=completed.breakdown;
        if (resident) {
            // Write-allocate: the buffered copy, released from the buffer in
            // host order and read for the program at `assembled`, stays in
            // its stack's device DRAM as a clean line.
            h.device_dram_install(ppn,assembled);
        } else if (staged) {
            // The page enters the target stack over HBIO with its program,
            // which the host issues at ready_ns_ (after any credit wait);
            // caching it there is a background fill of that stack's DRAM.
            h.device_dram_fill(ppn,std::max(assembled,ready_ns_),spans);
        }
        --h.stats_.program_requests;
        // These raw commands are internal physical work. The public logical
        // request counts its host payload once in issue(); copies, padding and
        // checkpoints must never enter the raw host-write WAF denominator.
        --h.stats_.raw_physical_programs;
        h.stats_.raw_physical_program_payload_bytes-=4096;
        out.spans.insert(out.spans.end(),completed.spans.begin(),completed.spans.end());
        // The host issues the program and continues; the page exists once the
        // program completes (settle). Padding is dead from the start.
        credits.push_back(completed.finish_ns);
        std::push_heap(credits.begin(),credits.end(),std::greater<double>{});
        inflight_.emplace(ppn,InFlight{completed.finish_ns,tag.generation,purpose==Purpose::Padding});
        inflight_by_finish_.emplace(completed.finish_ns,ppn);
        media_ns_=std::max(media_ns_,completed.finish_ns);
        if (purpose==Purpose::Data) data_ns_=std::max(data_ns_,completed.finish_ns);
        settle(h,ready_ns_);
        if (purpose==Purpose::Data) ++stats_.data_programs;
        else if (purpose==Purpose::Copy) ++stats_.copy_programs;
        else if (purpose==Purpose::Padding) ++stats_.padding_programs;
        else ++stats_.metadata_programs;
    }
    block.pages.push_back(tag);
}
void HbfMappingPolicy::retire(HbfController& h,U page) {
    // A version superseded while its program is in flight is invalidated
    // when that program completes: the native page does not exist before.
    const U ppn=physical(page);
    if (const auto found=inflight_.find(ppn);found!=inflight_.end()) found->second.retire=true;
    else h.invalidate_ppn(ppn);
    blocks_.at(page/pages_per_block_).pages.at(page%pages_per_block_).lpn=absent;
}
void HbfMappingPolicy::release(HbfController& h,U block,PhysicalCompletion& out) {
    auto& state=blocks_.at(block);
    if (!state.allocated && state.pages.empty()) return;
    // Reclaiming a block requires every program it received to have finished.
    for (U lane=0;lane<width_;++lane) {
        const U first=member(block,lane)*physical_pages_per_block_;
        for (auto it=inflight_.lower_bound(first);it!=inflight_.end() && it->first<first+physical_pages_per_block_;++it)
            wait_until(it->second.finish_ns,out);
    }
    settle(h,ready_ns_);
    for (U p=0;p<state.pages.size();++p) {
        const U ppn=physical(block*pages_per_block_+p);
        if (h.blocks_.at(ppn/physical_pages_per_block_).is_valid(ppn%physical_pages_per_block_)) h.invalidate_ppn(ppn);
    }
    for (U lane=0;lane<width_;++lane) h.release_invalid_block(member(block,lane));
    // Host reset releases ownership; the next page-zero program pays the
    // physical erase. Metadata and copies use this same lifecycle.
    h.zone_managed_=true;
    state=Block{};
    if (block<data_blocks_) free_.insert(allocation_rank(block));
    if (open_==block) open_=absent;
}
U HbfMappingPolicy::lane_rank(U block) const {
    const U plane=block/requested_.device.blocks_per_plane;
    return (block%requested_.device.blocks_per_plane)*planes_+
        (plane%planes_per_stack_)*requested_.device.stacks+plane/planes_per_stack_;
}
U HbfMappingPolicy::block_at_lane_rank(U rank) const {
    const U lane=rank%planes_;
    const U plane=(lane%requested_.device.stacks)*planes_per_stack_+
        lane/requested_.device.stacks;
    return plane*requested_.device.blocks_per_plane+rank/planes_;
}
// A single-plane block keeps its physical number; a wider superblock is
// numbered by its position in lane order, so its rank is its number.
U HbfMappingPolicy::allocation_rank(U block) const { return width_==1 ? lane_rank(block) : block; }
U HbfMappingPolicy::block_at_allocation_rank(U rank) const { return width_==1 ? block_at_lane_rank(rank) : rank; }
U HbfMappingPolicy::member(U block,U lane) const {
    return width_==1 ? block : block_at_lane_rank(block*width_+lane);
}
U HbfMappingPolicy::physical(U ppn) const {
    const U offset=ppn%pages_per_block_;
    return member(ppn/pages_per_block_,offset%width_)*physical_pages_per_block_+offset/width_;
}
// Physical pages that the first `pages` superblock pages place on `lane`.
U HbfMappingPolicy::member_prefix(U pages,U lane) const { return (pages+width_-1-lane)/width_; }
// Members are erased together; the most worn one bounds the superblock.
U HbfMappingPolicy::wear(const HbfController& h,U block) const {
    U worst=0;
    for (U lane=0;lane<width_;++lane) worst=std::max<U>(worst,h.blocks_[member(block,lane)].erase_count);
    return worst;
}
U HbfMappingPolicy::allocate(HbfController& h,PhysicalCompletion& out,bool reserve) {
    if (!reserve && !collecting_ &&
        (organization_==MappingOrganization::Extent || organization_==MappingOrganization::ObjectSegment) && free_.size()<=1) {
        if (!requested_.host.auto_gc_enabled)
            throw std::runtime_error("mapping allocator reached its copy reserve with automatic GC disabled");
        clean_segment(h,out);
        if (open_!=absent && blocks_[open_].pages.size()<pages_per_block_) return open_;
    }
    if (free_.empty()) throw std::runtime_error("mapping allocator exhausted its NAND reserve");
    if (!seeding_) {
        ready_ns_+=requested_.host.free_page_allocation_ns;
        out.breakdown.address_mapping_ns+=requested_.host.free_page_allocation_ns;
        h.stats_.stage_work.address_mapping_ns+=requested_.host.free_page_allocation_ns;
    }
    // Among free blocks prefer minimum physical wear. Equal-wear choices
    // rotate whole blocks across stacks and planes before advancing their
    // local block offset; pages within each logical block remain contiguous.
    auto nearest=free_.lower_bound(allocation_rank_cursor_);
    if (nearest==free_.end()) nearest=free_.begin();
    U selected_rank=*nearest;
    U selected=block_at_allocation_rank(selected_rank),least=wear(h,selected);
    const auto distance=[&](U rank) {
        return rank>=allocation_rank_cursor_ ? rank-allocation_rank_cursor_ :
            total_blocks_-allocation_rank_cursor_+rank;
    };
    // Zero is the minimum possible wear. If the nearest rotating candidate
    // is virgin, it is already the exact winner of the complete search.
    if (least) for (auto rank:free_) {
        const auto b=block_at_allocation_rank(rank);
        const auto candidate=wear(h,b);
        if (candidate<least || (candidate==least && distance(rank)<distance(selected_rank))) {
            selected=b; selected_rank=rank; least=candidate;
        }
    }
    free_.erase(selected_rank); allocation_rank_cursor_=(selected_rank+1)%total_blocks_;
    blocks_[selected].allocated=true;
    return selected;
}

void HbfMappingPolicy::clean_segment(HbfController& h,PhysicalCompletion& out) {
    ready_ns_+=requested_.host.host_gc_decision_ns;
    out.breakdown.maintenance_ns+=requested_.host.host_gc_decision_ns;
    h.stats_.stage_work.maintenance_ns+=requested_.host.host_gc_decision_ns;
    U victim=absent, best_live=pages_per_block_+1;
    for (U b=0;b<data_blocks_;++b) {
        const auto& block=blocks_[b];
        if (!block.allocated || b==open_ || block.pages.empty()) continue;
        const U live=std::count_if(block.pages.begin(),block.pages.end(),[](const auto& tag){return tag.lpn!=absent;});
        if (live<block.pages.size() && live<best_live) { victim=b; best_live=live; }
    }
    if (victim==absent)
        throw std::runtime_error("segment cleaner has no closed segment containing invalid data");
    collecting_=true;
    // The host cannot assume an OCP command exposes NAND OOB reverse tags.
    // Recover victim LPNs by scanning the resident primary extent index and
    // charge that scan. Generations below are opaque simulator payload IDs.
    auto contents=blocks_[victim].pages;
    for (auto& tag:contents) tag.lpn=absent;
    const U begin=victim*pages_per_block_,end=begin+pages_per_block_;
    for (const auto& [lpn,run]:extents_.runs()) {
        const U first=std::max(begin,run.ppn),last=std::min(end,run.ppn+run.pages);
        for (U ppn=first;ppn<last;++ppn) contents.at(ppn-begin).lpn=lpn+ppn-run.ppn;
    }
    stats_.reverse_scan_records+=extents_.runs().size();
    const auto scan_compute=extents_.runs().size()*
        (requested_.host.mapping_internal_compute_ns+requested_.host.ctrl_dram_issue_ns);
    ready_ns_+=scan_compute; out.breakdown.translation_ns+=scan_compute;
    h.stats_.stage_work.translation_ns+=scan_compute; stats_.lookup_work_ns+=scan_compute;
    const auto before=h.stats_.host_hbm_read_bytes;
    memory(h,extents_.runs().size()*extent_node_bytes,Op::Read,out);
    stats_.metadata_hbm_bytes+=h.stats_.host_hbm_read_bytes-before;
    for (U p=0;p<contents.size();++p) {
        const auto tag=contents[p];
        if (tag.lpn==absent) continue;
        const auto old=victim*pages_per_block_+p;
        if (locate(tag.lpn)!=old) throw std::logic_error("segment primary mapping lost a live page");
        read_page(h,old,Purpose::Copy,out);
        if (open_==absent || blocks_[open_].pages.size()==pages_per_block_)
            open_=allocate(h,out,true);
        const auto destination=open_*pages_per_block_+blocks_[open_].pages.size();
        program(h,destination,tag,Purpose::Copy,out);
        extents_.assign(tag.lpn,destination);
        lookup_work(h,tag.lpn,true,out);
        retire(h,old);
        if (blocks_[open_].pages.size()==pages_per_block_) open_=absent;
    }
    release(h,victim,out);
    collecting_=false;
    ++stats_.cleaned_segments; dirty_=true;
}

void HbfMappingPolicy::merge(HbfController& h,U logical_block,PhysicalCompletion& out) {
    auto& entry=block_map_.at(logical_block);
    if (entry.log==absent) return;
    const auto old_base=entry.base,old_log=entry.log;
    const auto written=blocks_[old_log].pages.size();
    bool prefix=true;
    for (U p=0;p<written;++p)
        if (blocks_[old_log].pages[p].lpn!=logical_block*pages_per_block_+p ||
            locate(logical_block*pages_per_block_+p)!=old_log*pages_per_block_+p) { prefix=false; break; }
    U destination;
    if (prefix) {
        destination=old_log;
        if (written==pages_per_block_) ++stats_.switch_merges;
        else ++stats_.partial_merges;
    } else { destination=allocate(h,out,true); ++stats_.full_merges; }
    blocks_[destination].owner=logical_block;
    const U begin=prefix ? written : 0;
    for (U p=begin;p<pages_per_block_;++p) {
        const U lpn=logical_block*pages_per_block_+p;
        const auto current=locate(lpn);
        if (current) {
            const auto tag=blocks_[*current/pages_per_block_].pages[*current%pages_per_block_];
            read_page(h,*current,Purpose::Copy,out);
            program(h,destination*pages_per_block_+p,tag,Purpose::Copy,out);
        } else {
            memory(h,4096,Op::Write,out);
            program(h,destination*pages_per_block_+p,Tag{},Purpose::Padding,out);
        }
    }
    log_update_entries_-=entry.updates.size();
    --active_log_count_;
    entry.base=destination; entry.log=absent; entry.updates.clear();
    release(h,old_base,out);
    if (old_log!=destination) release(h,old_log,out);
    dirty_=true;
    lookup_work(h,logical_block*pages_per_block_,true,out);
}

void HbfMappingPolicy::store(HbfController& h,U lpn,U generation,PhysicalCompletion& out,double data_ready) {
    const Tag incoming{lpn,generation};
    if (organization_==MappingOrganization::Extent || organization_==MappingOrganization::ObjectSegment) {
        if (open_==absent || blocks_[open_].pages.size()==pages_per_block_)
            open_=allocate(h,out);
        // Allocation may have cleaned and moved the previous version.
        const auto old=locate(lpn);
        const auto ppn=open_*pages_per_block_+blocks_[open_].pages.size();
        program(h,ppn,incoming,Purpose::Data,out,data_ready);
        extents_.assign(lpn,ppn);
        if (old) retire(h,*old);
        if (blocks_[open_].pages.size()==pages_per_block_) open_=absent;
    } else {
        const auto logical_block=lpn/pages_per_block_,offset=lpn%pages_per_block_;
        auto [position,inserted]=block_map_.try_emplace(logical_block);
        auto& entry=position->second;
        if (inserted) {
            entry.valid.resize(ceil_div(pages_per_block_,64));
            entry.base=allocate(h,out,true);
            blocks_[entry.base].owner=logical_block;
        }
        entry.touched=sequence_;
        if (entry.log==absent && offset>=blocks_[entry.base].pages.size()) {
            // Initial sequential allocation can include explicit zero padding
            // for unmapped holes; no NAND page address is skipped.
            while (blocks_[entry.base].pages.size()<offset) {
                memory(h,4096,Op::Write,out);
                program(h,entry.base*pages_per_block_+blocks_[entry.base].pages.size(),Tag{},Purpose::Padding,out);
            }
            program(h,entry.base*pages_per_block_+offset,incoming,Purpose::Data,out,data_ready);
        } else if (organization_==MappingOrganization::Block) {
            const U old_base=entry.base,destination=allocate(h,out,true);
            blocks_[destination].owner=logical_block;
            for (U p=0;p<pages_per_block_;++p) {
                const auto old=locate(logical_block*pages_per_block_+p);
                if (p==offset) program(h,destination*pages_per_block_+p,incoming,Purpose::Data,out,data_ready);
                else if (old) {
                    const auto tag=blocks_[*old/pages_per_block_].pages[*old%pages_per_block_];
                    read_page(h,*old,Purpose::Copy,out);
                    program(h,destination*pages_per_block_+p,tag,Purpose::Copy,out);
                } else {
                    memory(h,4096,Op::Write,out);
                    program(h,destination*pages_per_block_+p,Tag{},Purpose::Padding,out);
                }
            }
            entry.base=destination; release(h,old_base,out); ++stats_.full_merges;
        } else {
            if (entry.log!=absent && blocks_[entry.log].pages.size()==pages_per_block_)
                merge(h,logical_block,out);
            if (entry.log==absent) {
                U logs=0,victim=absent,oldest=absent;
                for (const auto& [key,candidate]:block_map_)
                    if (candidate.log!=absent) {
                        ++logs;
                        if (candidate.touched<oldest) { victim=key; oldest=candidate.touched; }
                    }
                if (logs>=requested_.host.mapping_log_blocks) merge(h,victim,out);
                entry.log=allocate(h,out,true); blocks_[entry.log].owner=logical_block;
                ++active_log_count_;
            }
            const auto old=locate(lpn);
            const auto ppn=entry.log*pages_per_block_+blocks_[entry.log].pages.size();
            program(h,ppn,incoming,Purpose::Data,out,data_ready);
            log_update_entries_+=entry.updates.insert_or_assign(offset,ppn).second;
            entry.valid[offset/64]|=U{1}<<(offset%64);
            if (old) retire(h,*old);
            // A full sequential log is immediately promotable with no copy.
            if (blocks_[entry.log].pages.size()==pages_per_block_) {
                bool sequential=true;
                for (U p=0;p<pages_per_block_;++p)
                    sequential &= blocks_[entry.log].pages[p].lpn==logical_block*pages_per_block_+p;
                if (sequential) merge(h,logical_block,out);
            }
        }
        entry.valid[offset/64]|=U{1}<<(offset%64);
    }
    // The index update is posted in command order; the host does not wait
    // for its controller-memory round trips.
    Breakdown work;
    update_ns_=std::max(update_ns_,index_access(h,lpn,true,work,
        trace_spans_enabled(trace_) ? &out.spans : nullptr,false));
    out.breakdown+=work; h.stats_.stage_work+=work;
    dirty_=true; update_footprint();
}
void HbfMappingPolicy::store_block(HbfController& h,U first,PhysicalCompletion& out) {
    const U key=first/pages_per_block_;
    auto [it,inserted]=block_map_.try_emplace(key);
    auto& entry=it->second;
    const U previous=entry.base;
    const U target=allocate(h,out,true);
    blocks_[target].owner=key;
    // A complete replacement supersedes buffered fragments of this block.
    // Stream the incoming full block through the copy workspace even when
    // the small random-write cache cannot hold an entire block.
    buffer_.erase(buffer_.lower_bound(first),buffer_.lower_bound(first+pages_per_block_));
    for (U p=0;p<pages_per_block_;++p) {
        memory_stack_=(first+p)%requested_.device.stacks;
        if (buffer_on_device(h)) {
            Breakdown work;
            ready_ns_=h.device_dram_stage_payload(buffer_stack(first+p),(first+p)/requested_.device.stacks,
                4096,ready_ns_,work,trace_spans_enabled(trace_) ? &out.spans : nullptr);
            out.breakdown+=work; h.stats_.stage_work+=work;
        } else memory(h,4096,Op::Write,out);
        program(h,target*pages_per_block_+p,Tag{first+p,++sequence_},Purpose::Data,out);
    }
    entry.base=target; entry.touched=sequence_;
    entry.valid.assign(ceil_div(pages_per_block_,64),~U{0});
    if (pages_per_block_%64) entry.valid.back()=(U{1}<<(pages_per_block_%64))-1;
    if (!inserted) { release(h,previous,out); ++stats_.full_merges; }
    lookup_work(h,first,true,out); dirty_=true; update_footprint();
}
void HbfMappingPolicy::flush(HbfController& h,U lpn,PhysicalCompletion& out) {
    const auto found=buffer_.find(lpn);
    if (found==buffer_.end()) return;
    store(h,lpn,found->second.generation,out,found->second.ready_ns);
    buffer_.erase(found); ++stats_.buffer_flushes;
}
void HbfMappingPolicy::flush_all(HbfController& h,PhysicalCompletion& out) {
    while (!buffer_.empty()) flush(h,buffer_.begin()->first,out);
}
U HbfMappingPolicy::object_for(U lpn) const {
    auto found=object_ranges_.upper_bound(lpn);
    if (found==object_ranges_.begin()) throw std::invalid_argument("logical page has no registered object");
    --found;
    const auto& object=objects_.at(found->second);
    if (lpn-object.first>=object.count) throw std::invalid_argument("logical page is outside its object");
    return found->second;
}
void HbfMappingPolicy::finish(HbfController& h,PhysicalCompletion& out,U before_bytes) {
    // Host work ends at ready_ns_; the NAND work it issued may finish later.
    const double complete=std::max({ready_ns_,media_ns_,update_ns_});
    out.finish_ns=complete;
    out.physical_bytes=h.stats_.physical_read_bytes+h.stats_.physical_write_bytes-before_bytes;
    h.stats_.finish_ns=std::max(h.stats_.finish_ns,complete);
    h.stats_.first_arrival_ns=std::min(h.stats_.first_arrival_ns,out.arrival_ns);
    h.stats_.free_pages=h.free_pages_;
    settle(h,ready_ns_);
    h.zone_managed_=true;
    update_footprint();
}
PhysicalCompletion HbfMappingPolicy::issue(HbfController& h,const PhysicalRequest& request) {
    if (read_only_ && request.op!=Op::Read)
        throw std::invalid_argument("read-only initial mapping rejects writes");
    if (request.tier!=Tier::HBF || (request.op!=Op::Read && request.op!=Op::Write) ||
        !request.bytes || request.addr>=logical_pages_*4096 || request.bytes>logical_pages_*4096-request.addr)
        throw std::invalid_argument("invalid structural mapping logical request");
    const auto first=request.addr/4096,last=(request.addr+request.bytes-1)/4096;
    if (organization_==MappingOrganization::ObjectSegment)
        for (U lpn=first;lpn<=last;++lpn)
            if (objects_.at(object_for(lpn)).sealed && request.op==Op::Write)
                throw std::invalid_argument("sealed objects are immutable; delete/recreate to replace them");
    start(h,request.arrival_ns); trace_=request.trace;
    const auto before=h.stats_.physical_read_bytes+h.stats_.physical_write_bytes;
    PhysicalCompletion out{.id=request.id,.tier=Tier::HBF,.op=request.op,.arrival_ns=request.arrival_ns,
        .start_ns=ready_ns_,.logical_bytes=request.bytes,.resource_path=std::string("host/")+to_string(organization_),
        .note="host-managed mapping"};
    out.breakdown.scheduler_queue_wait_ns=ready_ns_-request.arrival_ns;
    double foreground_finish=ready_ns_;
    if (request.op==Op::Read) read_range(h,request,out);
    else for (U lpn=first;lpn<=last;++lpn) {
        const auto begin=std::max(request.addr,lpn*4096),end=std::min(request.addr+request.bytes,(lpn+1)*4096);
        auto* spans=trace_spans_enabled(trace_) ? &out.spans : nullptr;
        Breakdown work;
        // Like the page FTL, each page has its own payload chain: the host
        // issues the lookup, merge read and program in command order and
        // moves on; the chain waits for the translation, the old version and
        // the payload. State (index, buffer, log slot) changes in host order.
        double page_ready=index_access(h,lpn,false,work,spans,false);
        if (request.op==Op::Write && organization_==MappingOrganization::Block &&
            lpn%pages_per_block_==0 &&
            begin==lpn*4096 && request.addr+request.bytes-lpn*4096>=pages_per_block_*4096) {
            out.breakdown+=work; h.stats_.stage_work+=work;
            ready_ns_=std::max(ready_ns_,page_ready);
            store_block(h,lpn,out); lpn+=pages_per_block_-1;
            foreground_finish=std::max({foreground_finish,ready_ns_,data_ns_,update_ns_}); continue;
        }
        if (const auto pending=buffer_.find(lpn);pending!=buffer_.end())
            page_ready=std::max(page_ready,pending->second.ready_ns);
        else {
            if (requested_.host.write_coalescing_enabled && buffer_stack_pages(lpn)>=requested_.host.write_buffer_pages) {
                U victim=absent,oldest=absent;
                for (const auto& [key,entry]:buffer_)
                    if (key%requested_.device.stacks==lpn%requested_.device.stacks && entry.touched<oldest)
                        { victim=key; oldest=entry.touched; }
                flush(h,victim,out);
                // The host executes commands in order: this page's merge
                // read and staging are issued after the eviction it just
                // issued, whose programs and cleaning may have advanced the
                // host past the translation time computed above.
                page_ready=std::max(page_ready,ready_ns_);
            }
            if (end-begin!=4096) {
                if (const auto old=locate(lpn)) page_ready=merge_read(h,*old,page_ready,request.heatmap_source,work,spans);
                else if (buffer_on_device(h))
                    page_ready=h.device_dram_transfer(buffer_stack(lpn),4096,true,page_ready,work,spans,
                        "write_buffer_zero_fill_device_dram");
                else page_ready=memory_at(h,4096,Op::Write,page_ready,work,spans);
            }
        }
        // A device-DRAM write buffer receives the payload over HBIO here.
        page_ready=buffer_on_device(h) ?
            h.device_dram_stage_payload(buffer_stack(lpn),lpn/requested_.device.stacks,end-begin,
                page_ready,work,spans) :
            memory_at(h,end-begin,Op::Write,page_ready,work,spans);
        out.breakdown+=work; h.stats_.stage_work+=work;
        const auto version=++sequence_;
        if (requested_.host.write_coalescing_enabled) {
            buffer_[lpn]={version,sequence_,page_ready};
            // A buffered page is acknowledged in host memory unless completion
            // requires it to be programmed; then its program must have finished.
            if (requested_.host.write_buffer_completion_requires_flush) {
                flush(h,lpn,out); foreground_finish=std::max({foreground_finish,ready_ns_,data_ns_,update_ns_});
            } else foreground_finish=std::max({foreground_finish,ready_ns_,page_ready});
            const auto threshold=requested_.host.write_buffer_flush_threshold_pages;
            while (threshold && buffer_stack_pages(lpn)>=threshold) {
                U victim=absent,oldest=absent;
                for (const auto& [key,entry]:buffer_)
                    if (key%requested_.device.stacks==lpn%requested_.device.stacks && entry.touched<oldest)
                        { victim=key; oldest=entry.touched; }
                flush(h,victim,out);
            }
        } else {
            store(h,lpn,version,out,page_ready);
            foreground_finish=std::max({foreground_finish,ready_ns_,data_ns_,update_ns_});
        }
    }
    // Internal raw commands are physical work, not additional host requests.
    h.stats_.read_requests+=(request.op==Op::Read);
    h.stats_.program_requests+=(request.op==Op::Write);
    if (request.op==Op::Read) {
        h.stats_.logical_read_bytes+=request.bytes;
        ++h.stats_.scalar_read_requests;
        const auto pages=last-first+1;
        h.stats_.scalar_read_pages+=pages;
        if (pages>1) { ++h.stats_.read_splits; h.stats_.read_split_pages+=pages; }
    }
    else h.stats_.logical_write_bytes+=request.bytes;
    finish(h,out,before);
    if (request.op==Op::Write) out.finish_ns=foreground_finish;
    return out;
}

HbfLogicalInvalidationResult HbfMappingPolicy::invalidate(HbfController& h,U first,U count,double at) {
    if (read_only_) throw std::invalid_argument("read-only initial mapping rejects invalidation");
    if (!count || first>=logical_pages_ || count>logical_pages_-first)
        throw std::invalid_argument("mapping invalidation outside logical capacity");
    start(h,at);
    const auto before=h.stats_.physical_read_bytes+h.stats_.physical_write_bytes;
    HbfLogicalInvalidationResult result;
    result.first_lpn=first; result.page_count=count;
    auto& out=result.completion;
    out.tier=Tier::HBF; out.op=Op::Erase; out.arrival_ns=at; out.start_ns=ready_ns_;
    out.resource_path=std::string("host/")+to_string(organization_)+"/invalidate";
    std::set<U> touched_blocks;
    for (U lpn=first;lpn<first+count;++lpn) {
        if (buffer_.erase(lpn)) { ++result.discarded_buffer_pages; result.discarded_buffer_bytes+=4096; }
        const auto ppn=locate(lpn);
        if (!ppn) { ++result.unmapped_pages; continue; }
        retire(h,*ppn); touched_blocks.insert(*ppn/pages_per_block_);
        if (organization_==MappingOrganization::Block || organization_==MappingOrganization::BlockLog) {
            auto& entry=block_map_.at(lpn/pages_per_block_);
            entry.valid[lpn%pages_per_block_/64]&=~(U{1}<<(lpn%pages_per_block_%64));
            log_update_entries_-=entry.updates.erase(lpn%pages_per_block_);
        } else extents_.erase(lpn,1);
        lookup_work(h,lpn,true,out); ++result.invalidated_pages; dirty_=true;
    }
    if (organization_==MappingOrganization::Block || organization_==MappingOrganization::BlockLog) {
        for (auto it=block_map_.begin();it!=block_map_.end();) {
            if (std::all_of(it->second.valid.begin(),it->second.valid.end(),[](U v){return !v;})) {
                release(h,it->second.base,out);
                if (it->second.log!=absent) {
                    release(h,it->second.log,out);
                    --active_log_count_;
                }
                log_update_entries_-=it->second.updates.size();
                it=block_map_.erase(it);
            } else ++it;
        }
    } else {
        for (auto block:touched_blocks)
            if (std::none_of(blocks_[block].pages.begin(),blocks_[block].pages.end(),[](const auto& tag){return tag.lpn!=absent;}))
                release(h,block,out);
    }
    out.note="logical mappings invalidated; fully dead blocks reset without a physical erase";
    finish(h,out,before);
    return result;
}

PhysicalCompletion HbfMappingPolicy::object_command(HbfController& h,std::string_view action,
    U id,U first,U count,double at) {
    if (read_only_) throw std::invalid_argument("read-only initial mapping rejects object changes");
    if (organization_!=MappingOrganization::ObjectSegment)
        throw std::invalid_argument("object lifecycle requires object-segment mapping");
    if (action=="CREATE") {
        if (!count || first>=logical_pages_ || count>logical_pages_-first || objects_.contains(id))
            throw std::invalid_argument("object requires a unique id and a valid logical range");
        auto next=object_ranges_.lower_bound(first);
        if ((next!=object_ranges_.end() && next->first<first+count) ||
            (next!=object_ranges_.begin() && objects_.at(std::prev(next)->second).first+
                objects_.at(std::prev(next)->second).count>first))
            throw std::invalid_argument("object logical ranges overlap");
    } else if (action!="SEAL" && action!="DELETE") {
        throw std::invalid_argument("unknown object lifecycle operation");
    } else if (!objects_.contains(id)) throw std::invalid_argument("unknown object id");
    start(h,at);
    const auto before=h.stats_.physical_read_bytes+h.stats_.physical_write_bytes;
    PhysicalCompletion out{.tier=Tier::HBF,.op=Op::Write,.arrival_ns=at,.start_ns=ready_ns_,
        .resource_path="host/object-segment/"+std::string(action)};
    if (action=="CREATE") {
        objects_.emplace(id,Object{first,count}); object_ranges_.emplace(first,id); ++stats_.object_creates;
    } else if (action=="SEAL") {
        const auto& object=objects_.at(id);
        std::vector<U> pending;
        for (auto it=buffer_.lower_bound(object.first);it!=buffer_.end() && it->first<object.first+object.count;++it)
            pending.push_back(it->first);
        for (auto lpn:pending) flush(h,lpn,out);
        objects_.at(id).sealed=true; ++stats_.object_seals;
    } else {
        const auto object=objects_.at(id);
        const auto invalidated=invalidate(h,object.first,object.count,at);
        out.breakdown+=invalidated.completion.breakdown;
        objects_.erase(id); object_ranges_.erase(object.first); ++stats_.object_deletes;
    }
    dirty_=true; memory(h,object_index_bytes,Op::Write,out);
    finish(h,out,before);
    return out;
}

std::vector<U> HbfMappingPolicy::snapshot() const {
    if (!buffer_.empty()) throw std::runtime_error("checkpoint must flush pending host payload before serializing mappings");
    // Policy format 2 stores a stripe-rank cursor, not a physical block cursor.
    std::vector<U> words{0x4842464d415031ULL,2,static_cast<U>(organization_),logical_pages_,
        total_blocks_,pages_per_block_,metadata_blocks_,requested_.host.mapping_log_blocks,
        open_,allocation_rank_cursor_,sequence_,checkpoint_generation_,checkpoint_slot_,checkpoint_pages_,U{read_only_}};
    words.push_back(block_map_.size());
    for (const auto& [key,entry]:block_map_) {
        words.insert(words.end(),{key,entry.base,entry.log,entry.touched,entry.valid.size()});
        words.insert(words.end(),entry.valid.begin(),entry.valid.end());
        words.push_back(entry.updates.size());
        for (const auto& [offset,ppn]:entry.updates) words.insert(words.end(),{offset,ppn});
    }
    words.push_back(extents_.runs().size());
    for (const auto& [first,run]:extents_.runs()) words.insert(words.end(),{first,run.pages,run.ppn});
    words.push_back(objects_.size());
    for (const auto& [id,object]:objects_) words.insert(words.end(),{id,object.first,object.count,U{object.sealed}});
    U allocated=0;
    for (U b=0;b<data_blocks_;++b) allocated+=blocks_[b].allocated;
    words.push_back(allocated);
    for (U b=0;b<data_blocks_;++b) if (blocks_[b].allocated)
        words.insert(words.end(),{b,blocks_[b].owner,blocks_[b].pages.size()});
    words.push_back(checksum(words,words.size()));
    return words;
}

PhysicalCompletion HbfMappingPolicy::checkpoint(HbfController& h,std::string id,double arrival,TraceConfig trace) {
    if (!seeding_) start(h,arrival);
    trace_=trace;
    const auto before=h.stats_.physical_read_bytes+h.stats_.physical_write_bytes;
    PhysicalCompletion out{.id=std::move(id),.tier=Tier::HBF,.op=Op::Write,
        .arrival_ns=arrival,.start_ns=ready_ns_,.resource_path="host/mapping-checkpoint"};
    flush_all(h,out);
    if (dirty_) {
        const U next=checkpoint_slot_==absent ? 0 : 1-checkpoint_slot_;
        const U first_block=data_blocks_+next*metadata_blocks_;
        for (U b=first_block;b<first_block+metadata_blocks_;++b) release(h,b,out);
        checkpoint_slot_=next; ++checkpoint_generation_;
        auto words=snapshot();
        checkpoint_pages_=ceil_div(words.size()*8,4096)+1;
        words=snapshot();
        if (checkpoint_pages_>metadata_blocks_*pages_per_block_)
            throw std::runtime_error("mapping checkpoint exceeds its reserved NAND slot");
        // Immutable body, then a final root/commit page. Snapshot files carry
        // these exact fixed-width records, not an uncharged dense shadow L2P.
        for (U page=0;page<checkpoint_pages_;++page) {
            const U ppn=first_block*pages_per_block_+page;
            blocks_[ppn/pages_per_block_].allocated=true;
            memory(h,4096,Op::Write,out);
            program(h,ppn,Tag{},Purpose::Metadata,out);
        }
        ++stats_.checkpoints; dirty_=false;
        out.note="drained-pending-hbf-state";
    } else out.note="no-pending-hbf-state";
    if (!seeding_) {
        // A checkpoint drains: its commit root is durable, and the device
        // quiescent, only after every program issued so far has finished.
        if (!inflight_by_finish_.empty()) wait_until(inflight_by_finish_.rbegin()->first,out);
        settle(h,ready_ns_);
        finish(h,out,before);
    }
    return out;
}

void HbfMappingPolicy::restore(HbfController& h,const std::vector<U>& words) {
    if (words.size()<20 || checksum(words,words.size()-1)!=words.back())
        throw std::invalid_argument("invalid host mapping checkpoint checksum");
    std::size_t position=0;
    const auto take=[&]() -> U {
        if (position>=words.size()-1) throw std::invalid_argument("truncated host mapping checkpoint");
        return words[position++];
    };
    if (take()!=0x4842464d415031ULL || take()!=2 || take()!=static_cast<U>(organization_) ||
        take()!=logical_pages_ || take()!=total_blocks_ || take()!=pages_per_block_ ||
        take()!=metadata_blocks_ || take()!=requested_.host.mapping_log_blocks)
        throw std::invalid_argument("host mapping checkpoint policy or geometry mismatch");
    open_=take(); allocation_rank_cursor_=take(); sequence_=take(); checkpoint_generation_=take();
    checkpoint_slot_=take(); checkpoint_pages_=take();
    const U read_only=take();
    if (read_only>1) throw std::invalid_argument("invalid persisted read-only mode");
    read_only_=read_only!=0;
    const U entries=take();
    if (entries>ceil_div(logical_pages_,pages_per_block_)) throw std::invalid_argument("oversized block map");
    for (U i=0;i<entries;++i) {
        U key=take(); BlockMap entry; entry.base=take(); entry.log=take(); entry.touched=take();
        const U n=take();
        if (n!=ceil_div(pages_per_block_,64) || key>=ceil_div(logical_pages_,pages_per_block_))
            throw std::invalid_argument("invalid block mapping bitmap");
        for (U j=0;j<n;++j) entry.valid.push_back(take());
        const U updates=take();
        if (updates>pages_per_block_) throw std::invalid_argument("oversized log mapping");
        for (U j=0;j<updates;++j) {
            const U offset=take(),ppn=take();
            if (offset>=pages_per_block_ || !entry.updates.emplace(offset,ppn).second)
                throw std::invalid_argument("invalid log offset");
        }
        const auto [position,inserted]=block_map_.emplace(key,std::move(entry));
        if (!inserted) throw std::invalid_argument("duplicate logical block");
        log_update_entries_+=position->second.updates.size();
        active_log_count_+=position->second.log!=absent;
    }
    const U runs=take();
    if (runs>logical_pages_) throw std::invalid_argument("oversized extent map");
    U previous_end=0;
    for (U i=0;i<runs;++i) {
        const U first=take(),count=take(),ppn=take();
        if (!count || first<previous_end || first>=logical_pages_ || count>logical_pages_-first ||
            ppn>=data_blocks_*pages_per_block_ || count>data_blocks_*pages_per_block_-ppn)
            throw std::invalid_argument("invalid persisted extent");
        for (U p=0;p<count;++p) extents_.assign(first+p,ppn+p);
        previous_end=first+count;
    }
    const U objects=take();
    if (objects>logical_pages_) throw std::invalid_argument("oversized object directory");
    for (U i=0;i<objects;++i) {
        const U id=take(),first=take(),count=take(),sealed=take();
        if (!count || first>=logical_pages_ || count>logical_pages_-first || sealed>1 ||
            !objects_.emplace(id,Object{first,count,sealed!=0}).second || !object_ranges_.emplace(first,id).second)
            throw std::invalid_argument("invalid persisted object");
    }
    const U allocated=take();
    if (allocated>data_blocks_) throw std::invalid_argument("oversized allocation map");
    for (U i=0;i<allocated;++i) {
        const U block=take(),owner=take(),prefix=take();
        bool members_agree=block<data_blocks_;
        for (U lane=0;members_agree && lane<width_;++lane)
            members_agree=h.blocks_.at(member(block,lane)).next_page==member_prefix(prefix,lane);
        if (!members_agree || prefix>pages_per_block_ || !free_.erase(allocation_rank(block)))
            throw std::invalid_argument("persisted block allocation disagrees with NAND prefix");
        blocks_[block].allocated=true; blocks_[block].owner=owner;
        blocks_[block].pages.resize(prefix);
    }
    if (position!=words.size()-1 || allocation_rank_cursor_>=total_blocks_ ||
        (open_!=absent && (open_>=data_blocks_ || !blocks_[open_].allocated || blocks_[open_].pages.size()>=pages_per_block_)))
        throw std::invalid_argument("invalid mapping allocator state");
    // Rebuild simulated on-media OOB identities, not a resident L2P lookup.
    for (U lpn=0;lpn<logical_pages_;++lpn) if (const auto page=locate(lpn)) {
        if (*page>=data_blocks_*pages_per_block_) throw std::invalid_argument("mapping points outside data media");
        auto& block=blocks_.at(*page/pages_per_block_);
        const U offset=*page%pages_per_block_,ppn=physical(*page);
        if (!block.allocated || offset>=block.pages.size() || block.pages[offset].lpn!=absent ||
            !h.blocks_.at(ppn/physical_pages_per_block_).is_valid(ppn%physical_pages_per_block_) ||
            !h.programmed_pages_.contains(ppn))
            throw std::invalid_argument("mapping checkpoint aliases or references non-live NAND");
        block.pages[offset]={lpn,h.programmed_pages_.at(ppn).lpn};
    }
    for (U b=data_blocks_;b<total_blocks_;++b) {
        U prefix=0;
        for (U lane=0;lane<width_;++lane) prefix+=h.blocks_[member(b,lane)].next_page;
        for (U lane=0;lane<width_;++lane)
            if (h.blocks_[member(b,lane)].next_page!=member_prefix(prefix,lane))
                throw std::invalid_argument("metadata superblock members are not sequentially programmed");
        blocks_[b].allocated=prefix!=0;
        blocks_[b].pages.resize(prefix);
    }
    if (checkpoint_slot_!=absent) {
        if (checkpoint_slot_>1 || !checkpoint_pages_ || checkpoint_pages_>metadata_blocks_*pages_per_block_)
            throw std::invalid_argument("invalid mapping commit root");
        PhysicalCompletion boot{.id="mapping-recovery",.tier=Tier::HBF,.op=Op::Read};
        h.seed_media_image();
        const U first=(data_blocks_+checkpoint_slot_*metadata_blocks_)*pages_per_block_;
        for (U p=0;p<checkpoint_pages_;++p) {
            const U ppn=physical(first+p);
            if (!h.blocks_.at(ppn/physical_pages_per_block_).is_valid(ppn%physical_pages_per_block_))
                throw std::invalid_argument("mapping commit root references missing metadata media");
            read_page(h,first+p,Purpose::Metadata,boot);
        }
        h.stats_.finish_ns=std::max(h.stats_.finish_ns,ready_ns_);
    } else if (checkpoint_generation_ || checkpoint_pages_)
        throw std::invalid_argument("mapping generation has no commit root");
    dirty_=false; update_footprint(); audit(h);
}

void HbfMappingPolicy::prepopulate(HbfController& h,const std::vector<U>& input,bool read_only) {
    if (sequence_ || h.last_issue_arrival_ns_ || dirty_) throw std::invalid_argument("mapping population requires a fresh device");
    auto pages=input; std::sort(pages.begin(),pages.end());
    if ((!pages.empty() && pages.back()>=logical_pages_) || std::adjacent_find(pages.begin(),pages.end())!=pages.end())
        throw std::invalid_argument("invalid initial logical page population");
    seeding_=true;
    if (organization_==MappingOrganization::ObjectSegment && objects_.empty()) {
        objects_.emplace(0,Object{0,logical_pages_}); object_ranges_.emplace(0,0);
        dirty_=true;
    }
    PhysicalCompletion initial{.id="initial-mapping"};
    for (auto lpn:pages) store(h,lpn,++sequence_,initial,0);
    read_only_=read_only;
    if (read_only_) dirty_=true;
    (void)checkpoint(h,"initial-mapping-checkpoint",0,{});
    seeding_=false; h.zone_managed_=true; update_footprint();
}

void HbfMappingPolicy::audit(const HbfController& h) const {
    if ((organization_==MappingOrganization::Block || organization_==MappingOrganization::BlockLog) && !extents_.runs().empty())
        throw std::logic_error("block mapping silently retained an extent page map");
    if ((organization_==MappingOrganization::Extent || organization_==MappingOrganization::ObjectSegment) && !block_map_.empty())
        throw std::logic_error("extent mapping silently retained a block map");
    if (organization_!=MappingOrganization::ObjectSegment && !objects_.empty())
        throw std::logic_error("non-object mapping contains an object directory");
    U previous_end=0;
    for (const auto& [first,id]:object_ranges_) {
        const auto& object=objects_.at(id);
        if (first!=object.first || first<previous_end || !object.count ||
            first>=logical_pages_ || object.count>logical_pages_-first)
            throw std::logic_error("overlapping or invalid object directory");
        previous_end=first+object.count;
    }
    std::set<U> ownership;
    U log_updates=0,active_logs=0;
    for (const auto& [key,entry]:block_map_) {
        log_updates+=entry.updates.size();
        active_logs+=entry.log!=absent;
        if (entry.valid.size()!=ceil_div(pages_per_block_,64))
            throw std::logic_error("block mapping bitmap size diverged");
        if (organization_==MappingOrganization::Block && (entry.log!=absent || !entry.updates.empty()))
            throw std::logic_error("pure block mapping contains a log");
        for (U b:{entry.base,entry.log}) if (b!=absent) {
            if (b>=data_blocks_ || !blocks_[b].allocated || blocks_[b].owner!=key || !ownership.insert(b).second)
                throw std::logic_error("block mapping ownership diverged");
        }
        if (entry.base==absent) throw std::logic_error("block mapping has no base");
        for (const auto& [offset,ppn]:entry.updates)
            if (entry.log==absent || offset>=pages_per_block_ || ppn/pages_per_block_!=entry.log ||
                !(entry.valid[offset/64]&(U{1}<<(offset%64))))
                throw std::logic_error("log directory references an invalid update");
    }
    if (log_updates!=log_update_entries_ || active_logs!=active_log_count_)
        throw std::logic_error("block log accounting diverged from primary mappings");
    U live=0;
    for (U b=0;b<data_blocks_;++b) {
        const auto& block=blocks_[b];
        bool prefixes=free_.contains(allocation_rank(b))!=block.allocated;
        for (U lane=0;prefixes && lane<width_;++lane)
            prefixes=h.blocks_[member(b,lane)].next_page==member_prefix(block.pages.size(),lane);
        if (!prefixes) throw std::logic_error("mapping allocation prefix disagrees with native media");
        for (U p=0;p<block.pages.size();++p) {
            const auto& tag=block.pages[p];
            const U ppn=physical(b*pages_per_block_+p);
            // A page whose program is still in flight has no native state to
            // compare yet (or has one that settle has not yet reconciled).
            const bool settled=!inflight_.contains(ppn);
            if (settled && (tag.lpn!=absent)!=h.blocks_[ppn/physical_pages_per_block_].is_valid(ppn%physical_pages_per_block_))
                throw std::logic_error("mapping liveness disagrees with NAND bitmap");
            if (tag.lpn!=absent) {
                if (tag.lpn>=logical_pages_ || !tag.generation || tag.generation>sequence_ || locate(tag.lpn)!=b*pages_per_block_+p ||
                    (settled && h.programmed_pages_.at(ppn).lpn!=tag.generation))
                    throw std::logic_error("primary mapping or copied payload identity diverged");
                if (organization_==MappingOrganization::ObjectSegment) (void)object_for(tag.lpn);
                ++live;
            }
        }
    }
    if (live>logical_pages_ || stats_.active_logs>requested_.host.mapping_log_blocks)
        throw std::logic_error("mapping exceeded its logical/log capacity");
}

HbfMappingPolicyStats HbfController::mapping_policy_stats() const {
    if (mapping_policy_) return mapping_policy_->stats();
    refresh_mapping_issue_stats();
    const auto& native=execution_stats();
    HbfMappingPolicyStats result;
    const U block_state=blocks_.size()*32+ceil_div(total_pages_,8);
    result.metadata_bytes=block_state+native.resident_mapping_table_bytes+
        native.mapping_directory_bytes+native.mapping_cache_live_bytes;
    result.peak_metadata_bytes=block_state+native.resident_mapping_table_bytes+
        native.mapping_directory_bytes+native.mapping_cache_peak_bytes;
    result.reserved_bytes=native.controller_dram_budget_bytes;
    result.data_programs=native.data_programs;
    result.copy_programs=native.gc_relocations+native.static_wear_leveling_relocations;
    result.metadata_programs=native.mapping_page_programs;
    result.metadata_reads=native.mapping_media_reads;
    result.buffer_flushes=native.write_buffer_flushes;
    result.buffer_hits=native.write_buffer_read_hits;
    result.lookups=native.mapping_lookup_ops;
    result.updates=native.mapping_update_ops;
    result.lookup_work_ns=native.mapping_compute_work_ns+native.mapping_dram_issue_busy_ns;
    result.memory_bytes=native.host_hbm_read_bytes+native.host_hbm_write_bytes;
    return result;
}
std::pair<U,U> HbfController::initial_image_media_pages() const {
    return mapping_policy_ ? mapping_policy_->allocated_pages() :
        std::pair<U,U>{stats_.initial_logical_data_pages,stats_.initial_mapping_pages};
}
void HbfController::write_mapping_snapshot_json(std::ostream& out) const {
    // Page policy stats refresh the issue-work ledger; structural getters
    // remain side-effect free, so normalize their JSON observation here.
    // This snapshot does not request a whole-media accounting/ownership audit.
    if (mapping_policy_) refresh_mapping_issue_stats();
    const auto policy=mapping_policy_stats();
    const auto& native=execution_stats();
    out<<std::setprecision(17)<<"{\"organization\":\""<<to_string(config_.host.mapping_organization)
        <<"\",\"mode\":\""<<(mapping_policy_ ? "host-primary" : to_string(config_.host.mapping_mode))
        <<"\",\"cache_layout\":\""<<(mapping_policy_ ? "none" : to_string(config_.host.mapping_cache_layout))
        <<"\",\"logical_capacity_pages\":"<<logical_capacity_pages()
        <<",\"page_size_bytes\":"<<config_.device.page_size_bytes
        <<",\"physical_pages\":"<<total_pages_
        <<",\"read_requests\":"<<native.read_requests
        <<",\"write_requests\":"<<native.program_requests
        <<",\"logical_read_bytes\":"<<native.logical_read_bytes
        <<",\"logical_write_bytes\":"<<native.logical_write_bytes
        <<",\"physical_read_bytes\":"<<native.physical_read_bytes
        <<",\"physical_write_bytes\":"<<native.physical_write_bytes
        <<",\"page_reads\":"<<native.page_reads
        <<",\"page_programs\":"<<native.page_programs
        <<",\"block_erases\":"<<native.block_erases
        <<",\"metadata_bytes\":"<<policy.metadata_bytes
        <<",\"mapping_index_bytes\":"<<policy.metadata_bytes-blocks_.size()*32-ceil_div(total_pages_,8)
        <<",\"block_state_bytes\":"<<blocks_.size()*32+ceil_div(total_pages_,8)
        <<",\"peak_metadata_bytes\":"<<policy.peak_metadata_bytes
        <<",\"controller_reserved_bytes\":"<<policy.reserved_bytes
        <<",\"data_programs\":"<<policy.data_programs
        <<",\"copy_programs\":"<<policy.copy_programs
        <<",\"padding_programs\":"<<policy.padding_programs
        <<",\"metadata_programs\":"<<policy.metadata_programs
        <<",\"metadata_reads\":"<<policy.metadata_reads
        <<",\"mapping_cache_hits\":"<<native.mapping_cache_hits
        <<",\"mapping_cache_misses\":"<<native.mapping_cache_misses
        <<",\"mapping_cache_dirty_evictions\":"<<native.mapping_cache_dirty_evictions
        <<",\"switch_merges\":"<<policy.switch_merges
        <<",\"partial_merges\":"<<policy.partial_merges
        <<",\"full_merges\":"<<policy.full_merges
        <<",\"cleaned_segments\":"<<policy.cleaned_segments
        <<",\"page_gc_runs\":"<<(mapping_policy_ ? "null" : std::to_string(native.gc_runs))
        <<",\"page_gc_copy_programs\":"<<(mapping_policy_ ? "null" : std::to_string(native.gc_relocations))
        <<",\"static_wl_runs\":"<<(mapping_policy_ ? "null" : std::to_string(native.static_wear_leveling_runs))
        <<",\"static_wl_copy_programs\":"<<(mapping_policy_ ? "null" : std::to_string(native.static_wear_leveling_relocations))
        <<",\"extent_records\":"<<policy.extent_records
        <<",\"active_logs\":"<<policy.active_logs
        <<",\"buffer_flushes\":"<<policy.buffer_flushes
        <<",\"buffer_hits\":"<<policy.buffer_hits
        <<",\"object_creates\":"<<policy.object_creates
        <<",\"object_seals\":"<<policy.object_seals
        <<",\"object_deletes\":"<<policy.object_deletes
        <<",\"lookups\":"<<policy.lookups
        <<",\"updates\":"<<policy.updates
        <<",\"reverse_scan_records\":"<<policy.reverse_scan_records
        <<",\"mapping_work_ns\":"<<policy.lookup_work_ns
        <<",\"host_hbm_read_bytes\":"<<native.host_hbm_read_bytes
        <<",\"host_hbm_write_bytes\":"<<native.host_hbm_write_bytes
        <<",\"host_hbm_reserved_bytes\":"<<native.host_hbm_reserved_bytes
        <<",\"device_dram_capacity_bytes\":"<<native.device_dram_capacity_bytes
        <<",\"device_dram_read_hits\":"<<native.device_dram_read_hits
        <<",\"device_dram_read_hit_bytes\":"<<native.device_dram_read_hit_bytes
        <<",\"device_dram_read_misses\":"<<native.device_dram_read_misses
        <<",\"device_dram_fills\":"<<native.device_dram_fills
        <<",\"device_dram_fill_bypasses\":"<<native.device_dram_fill_bypasses
        <<",\"device_dram_evictions\":"<<native.device_dram_evictions
        <<",\"device_dram_read_bytes\":"<<native.device_dram_read_bytes
        <<",\"device_dram_write_bytes\":"<<native.device_dram_write_bytes
        <<",\"finish_ns\":"<<native.finish_ns
        <<",\"dirty\":"<<(quiescence_stats().quiescent() ? "false" : "true")<<'}';
}
std::optional<U> HbfController::mapped_physical_page(U lpn) const {
    if (mapping_policy_) return mapping_policy_->lookup(lpn);
    if (const auto found=lpn_to_ppn_.find(lpn);found!=lpn_to_ppn_.end()) return found->second;
    return compact_lpn_ppn(lpn);
}
std::optional<U> HbfController::mapped_generation(U lpn) const {
    return mapping_policy_ ? mapping_policy_->generation(lpn) : std::nullopt;
}
PhysicalCompletion HbfController::object_command(std::string_view action,U id,U first,U count,double at) {
    if (!mapping_policy_) throw std::invalid_argument("object lifecycle requires object-segment mapping");
    return mapping_policy_->object_command(*this,action,id,first,count,at);
}
} // namespace hbfsim::host
