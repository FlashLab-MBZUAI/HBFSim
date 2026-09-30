#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_controller.hpp"
#include <iostream>
#include <stdexcept>
using namespace hbfsim::physical;
using namespace hbfsim::host;
void check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
int main() {
    try {
        HbfConfig c;
        c.device.channels_per_stack=1; c.device.planes_per_die=2;
        c.device.blocks_per_plane=16; c.device.pages_per_block=16;
        c.host.auto_gc_enabled=false; c.host.gc_reserved_free_blocks_per_plane=0;
        hbfsim::verification::HbfWithHbm d(c);
        d.reserve_static_physical_block_extent(0,1);
        double at=0;
        auto read=[&](std::uint64_t page) {
            auto done=d.issue(PhysicalRequest{.id="cache",.tier=Tier::HBF,.op=Op::Read,
                .address_space=AddressSpace::Physical,.arrival_ns=at,.addr=page*4096,.bytes=64});
            at=done.finish_ns;
        };
        read(0);read(1);read(0);
        check(d.execution_stats().page_reads==2,"first two pages were not retained");
        read(2);read(1);
        check(d.execution_stats().page_reads==4,"third same-bank page failed to evict LRU");
        read(16*16);read(16*16+1);read(2);
        check(d.execution_stats().page_reads==6,"another bank evicted this bank's pages");
        for(unsigned i=0;i<4096;++i) read(2);
        check(d.execution_stats().page_reads==6,"repeated completed hits sensed NAND again");
        check(d.stats().accounting_verified,"cache handoff accounting failed");
        std::cout << "bank cache completed handoff and isolation passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
