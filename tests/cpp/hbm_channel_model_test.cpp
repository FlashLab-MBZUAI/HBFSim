#include "physical/hbm/hbm_device.hpp"
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
using namespace hbfsim::physical;
using namespace hbfsim::physical::hbm;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
HbmConfig one_channel() {
    HbmConfig c;
    c.device.capacity_bytes = 4ull << 20;
    c.device.channels_per_stack = c.device.pseudo_channels_per_channel = 1;
    c.timing.read_latency_ns = c.timing.write_latency_ns = 0;
    c.timing.read_to_write_ns = c.timing.write_to_read_ns = 0;
    c.timing.bandwidth_efficiency = 1;
    c.controller.service_quantum_bytes = 4096;
    c.controller.service_group_channels = 1;
    return c;
}
PhysicalRequest req(std::uint64_t addr, std::uint64_t bytes, double at=0, Op op=Op::Read) {
    return {.id="r", .tier=Tier::HBM, .op=op, .arrival_ns=at, .addr=addr, .bytes=bytes};
}
void grouped_addresses_and_rounding() {
    std::mt19937_64 rng(921);
    for (unsigned channels : {1u, 3u, 8u}) for (unsigned pch : {1u, 2u}) {
        auto c = one_channel();
        c.device.stacks = 2;
        c.device.channels_per_stack = channels;
        c.device.pseudo_channels_per_channel = pch;
        c.device.channel_row_size_bytes = c.burst_bytes() * 24 * pch;
        c.controller.interleave_bytes = c.burst_bytes() * 3;
        for (unsigned i=0; i<50; ++i) {
            c.controller.service_group_channels = 1u << (i % 5);
            HbmDevice d(c);
            const auto addr = rng() % (1<<20);
            const auto bytes = 1 + rng() % (1<<15);
            const auto burst = c.burst_bytes();
            std::map<std::size_t, std::uint64_t> counts;
            for (auto a=addr-addr%burst; a<=addr+bytes-1; a+=burst) {
                const auto decoded = d.decode(a);
                require(d.encode(decoded)==a, "address map failed round trip");
                ++counts[(decoded.stack*channels+decoded.channel)*pch+decoded.pseudo_channel];
            }
            std::uint64_t max_count=0, total=0;
            for (auto [pc,n] : counts) { (void)pc; max_count=std::max(n,max_count); total+=n; }
            auto done=d.issue(req(addr,bytes));
            require(done.physical_bytes==total*burst, "grouping lost a partial burst or channel");
            require(done.finish_ns==max_count*c.burst_duration_ns(), "parallel channel service differs from burst census");
            require(d.stats().active_pseudo_channels==counts.size(), "wrong active channel count");
        }
    }
}
void fair_queue_and_finite_admission() {
    auto c=one_channel(); HbmDevice d(c);
    auto big=d.enqueue(req(0,65536)); auto small=d.enqueue(req(65536,64));
    auto b=d.pump(small); auto a=d.pump(big);
    require(b.finish_ns==65 && a.finish_ns==1025, "round-robin quantum fairness failed");
    require(d.stats().max_queue_occupancy==2, "grouped queue occupancy is wrong");
    c.controller.queue_depth=1; HbmDevice bounded(c);
    auto first=bounded.enqueue(req(0,8192)); auto second=bounded.enqueue(req(8192,64));
    require(bounded.pump(first).finish_ns < bounded.pump(second).finish_ns, "finite admission lost FIFO order");
    require(bounded.stats().max_queue_occupancy==1, "admission exceeded queue depth");
}
void grouped_contention_keeps_bytes_and_isolated_bandwidth() {
    auto c = one_channel(); c.device.channels_per_stack = 4;
    c.controller.service_group_channels = 4;
    HbmDevice grouped(c);
    auto a = grouped.enqueue(req(grouped.encode(HbmAddress{.channel=0}), 64));
    auto b = grouped.enqueue(req(grouped.encode(HbmAddress{.channel=1}), 64));
    require(grouped.pump(a).finish_ns == 1 && grouped.pump(b).finish_ns == 2,
            "grouped arbitration did not serialize independent lanes");
    require(grouped.stats().read_bytes == 128 && grouped.stats().bus_busy_ns == 2 &&
            grouped.stats().active_pseudo_channels == 2 && grouped.stats().pseudo_channels == 4,
            "grouping changed physical lane accounting");
    c.controller.service_group_channels = 1;
    HbmDevice fine(c);
    a = fine.enqueue(req(fine.encode(HbmAddress{.channel=0}), 64));
    b = fine.enqueue(req(fine.encode(HbmAddress{.channel=1}), 64));
    require(fine.pump(a).finish_ns == 1 && fine.pump(b).finish_ns == 1,
            "single-lane sensitivity setting lost parallel service");
}
void dma_preemption_and_future_gaps() {
    auto c=one_channel(); HbmDevice d(c); d.reserve_controller_buffer(1<<20);
    auto ticket=d.enqueue(req(0,65536));
    require(d.service_before(100), "no application progress before DMA arrival");
    auto dma=d.transfer_controller_buffer(req(d.application_capacity_bytes(),64,100,Op::Write));
    require(dma.finish_ns==129, "large application reserved past its quantum boundary");
    require(d.pump(ticket).finish_ns==1025, "shared DMA traffic did not delay the application");
    require(d.stats().read_bytes==65536 && d.stats().write_bytes==64, "shared traffic is not conserved");
    HbmDevice future(c); future.reserve_controller_buffer(1<<20);
    auto future_dma=future.transfer_controller_buffer(req(future.application_capacity_bytes(),4096,300));
    auto app=future.issue(req(0,65536));
    require(future_dma.finish_ns==364 && app.finish_ns==1132, "coalescing crossed a reserved DMA interval");
    // The 44 ns gap before the future DMA cannot fit a full 64 ns quantum.
    HbmDevice preview(c); preview.reserve_controller_buffer(1<<20);
    auto queued=preview.enqueue(req(0,4096,100));
    auto inserted=preview.transfer_controller_buffer(req(preview.application_capacity_bytes(),4096,100));
    require(inserted.finish_ns==164 && preview.pump(queued).finish_ns==228,
            "a future DMA did not invalidate the pending application preview");
}
void causal_coalescing_matches_bounded_service() {
    std::mt19937_64 rng(1921);
    for (unsigned i=0; i<60; ++i) {
        HbmConfig c;
        c.device.channels_per_stack=3;
        c.controller.service_quantum_bytes=64ull << (i%6);
        HbmDevice full(c), bounded(c);
        full.reserve_controller_buffer(1<<20); bounded.reserve_controller_buffer(1<<20);
        auto dma=req(full.application_capacity_bytes(),8192,500,Op::Write);
        (void)full.transfer_controller_buffer(dma); (void)bounded.transfer_controller_buffer(dma);
        auto r=req(rng()%8192, 16384+rng()%131072, 0, i%2 ? Op::Read : Op::Write);
        auto a=full.issue(r); auto ticket=bounded.enqueue(r);
        // Advance with external event boundaries, preventing whole-request
        // coalescing and exercising the same physical quantum sequence.
        for (double boundary=1; boundary<10000; boundary+=17) {
            while(bounded.service_before(boundary)) {}
        }
        auto b=bounded.pump(ticket);
        require(a.finish_ns==b.finish_ns && a.physical_bytes==b.physical_bytes &&
                a.breakdown==b.breakdown, "coalescing changed quantum timing or accounting");
        require(full.stats().service_quanta==bounded.stats().service_quanta &&
                full.stats().service_busy_ns==bounded.stats().service_busy_ns, "coalescing changed service work");
    }
}
void latency_bandwidth_direction_and_history() {
    auto c=one_channel(); c.timing.read_latency_ns=40; c.timing.write_latency_ns=30;
    c.timing.bandwidth_efficiency=0.5; c.timing.read_to_write_ns=8;
    HbmDevice d(c);
    auto read=d.issue(req(0,4096)); auto write=d.issue(req(4096,4096,0,Op::Write));
    require(read.finish_ns==168 && write.finish_ns==304, "latency, effective rate or direction switch is wrong");
    require(d.stats().bus_busy_ns==128 && d.stats().service_busy_ns==256, "payload and effective service work were conflated");
    HbmDevice history(c); history.reserve_controller_buffer(1<<20);
    double now=0;
    for(unsigned i=0;i<100;++i) {
        history.advance_buffer_frontier(now);
        auto transfer=history.transfer_controller_buffer(req(history.application_capacity_bytes(),4096,now+500));
        auto app=history.issue(req(0,8192,now));
        now=std::max(transfer.finish_ns,app.finish_ns)+100;
    }
    require(history.stats().retained_bus_gaps < 8, "shared bus history grows past the joint frontier");
}
}
int main() {
    try { grouped_addresses_and_rounding(); fair_queue_and_finite_admission();
        grouped_contention_keeps_bytes_and_isolated_bandwidth();
        dma_preemption_and_future_gaps(); causal_coalescing_matches_bounded_service();
        latency_bandwidth_direction_and_history(); }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
    std::cout<<"HBM channel grouping, fairness, DMA, causal coalescing and timing passed\n";
}
