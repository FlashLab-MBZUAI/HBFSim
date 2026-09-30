#include "physical/simulation_session.hpp"
#include "../../verification/probes/hbf_with_hbm.hpp"
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

using namespace hbfsim::physical;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F&& fn, const char* message) {
    try { fn(); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error(message);
}
void calendar_query_reuse() {
    ResourceTimeline first, reordered;
    first.ready_ns = reordered.ready_ns = 100;
    for (const auto gap : {ResourceTimeline::Gap{0, 10}, {20, 40}, {60, 90}})
        first.insert_gap(gap.begin_ns, gap.end_ns);
    for (const auto gap : {ResourceTimeline::Gap{60, 90}, {20, 40}, {-0.0, 10}})
        reordered.insert_gap(gap.begin_ns, gap.end_ns);
    for (double floor : {0, 5, 10, 25, 40, 70, 100}) {
        require(first.gaps_after(floor) == reordered.gaps_after(floor),
            "calendar gaps depend on insertion order or signed zero");
        ResourceTimeline suffix;
        for (const auto gap : first.gaps_after(floor)) suffix.insert_gap(gap.begin_ns, gap.end_ns);
        require(first.gaps_after(floor) == suffix.gaps_after(0),
            "calendar suffix did not clip a crossing gap");
    }
    require(first.preview_start(0, 8) == 0 && first.preview_start(0, 8) == 0,
        "repeated preview changed its reservation");
    require(first.reserve(0, 8).start_ns == 0 && first.preview_start(0, 8) == 20,
        "calendar reused a query after consuming its gap");
    require(first.preview_start(0, 35) == 100, "oversized reservation fit an idle gap");
    first.insert_gap(100, 150);
    first.ready_ns = 160;
    require(first.preview_start(0, 35) == 100, "calendar reused a failed query after insertion");
    require(first.preview_start(25, 8) == 25, "calendar preview missed a crossing gap");
    first.prune_before(27);
    rejects([&] { (void)first.preview_start(25, 8); },
        "cached query bypassed the causal watermark");
    require(first.preview_start(27, 8) == 27 &&
        first.gaps_after(27) == first.gaps_after(0),
        "pruning changed the live suffix");
    auto copied = first;
    (void)copied.reserve(27, 8);
    require(first.preview_start(27, 8) == 27 && copied.preview_start(27, 8) == 60,
        "copied calendar shared mutable query state");
}
hbm::HbmConfig memory_config() {
    hbm::HbmConfig config;
    config.device.capacity_bytes = 256 * 1024;
    config.device.channels_per_stack = 1;
    config.device.pseudo_channels_per_channel = 1;
    config.device.bank_groups_per_pseudo_channel = 1;
    config.device.banks_per_group = 2;
    config.timing.bandwidth_efficiency = 1.0;
    config.controller.service_quantum_bytes = 4096;
    config.controller.service_group_channels = 1;
    return config;
}
PhysicalRequest access(std::uint64_t addr, std::uint64_t bytes, double at, Op op = Op::Read) {
    return {.id="memory", .tier=Tier::HBM, .op=op,
        .trace={.mode=TraceMode::Full}, .arrival_ns=at, .addr=addr, .bytes=bytes};
}
void buffer_grouping_matches_burst_oracle() {
    std::mt19937_64 random(9173);
    for (unsigned channels : {1u, 3u, 8u}) {
        for (unsigned pch : {1u, 2u}) {
            for (unsigned interleave_bursts : {1u, 3u, 8u}) {
                auto cfg = memory_config();
                cfg.device.stacks = 2;
                cfg.device.channels_per_stack = channels;
                cfg.device.pseudo_channels_per_channel = pch;
                // Exercise non-power-of-two interleave/stripe sizes and a
                // controller region whose boundary cuts through a stripe.
                cfg.device.capacity_bytes = (4ull << 20) + 128;
                cfg.device.channel_row_size_bytes = cfg.burst_bytes() * 24 * pch;
                cfg.controller.interleave_bytes = cfg.burst_bytes() * interleave_bursts;
                hbm::HbmDevice device(cfg);
                constexpr std::uint64_t reserved = 2ull << 20;
                device.reserve_controller_buffer(reserved);
                const auto burst = cfg.burst_bytes();
                const auto burst_cycles = cfg.device.burst_length / cfg.device.data_rate_per_command_clock;
                std::vector<ResourceTimeline> buses(2 * channels * pch);
                std::uint64_t reads = 0, writes = 0, busy_cycles = 0;
                double work = 0, wait = 0;
                for (unsigned step = 0; step < 128; ++step) {
                    auto offset = random() % reserved;
                    auto bytes = std::min<std::uint64_t>(1 + random() % 65536, reserved - offset);
                    if (step % 32 == 0) { offset = 0; bytes = reserved; }
                    if (step % 32 == 1) { offset = reserved - 1; bytes = 1; }
                    if (step % 32 == 2) { offset = 1; bytes = burst; }
                    const double at = (step / 4) * 100000.0 + (step % 4 == 0 ? 2000 : (step % 4) * 500);
                    auto request = access(device.application_capacity_bytes() + offset, bytes, at,
                        step % 2 ? Op::Read : Op::Write);
                    request.trace = {.mode = static_cast<TraceMode>(step % 4),
                        .retain_completion_diagnostics = step % 2 == 0};
                    // Independent oracle: decode every physical burst through
                    // the public address map, then schedule one transfer per
                    // channel in ascending order. No stripe aggregation.
                    std::map<std::size_t, std::uint64_t> counts;
                    const auto last = request.addr + bytes - 1;
                    for (auto cursor = request.addr - request.addr % burst;; cursor += burst) {
                        const auto a = device.decode(cursor);
                        ++counts[(a.stack * channels + a.channel) * pch + a.pseudo_channel];
                        if (last - cursor < burst) break;
                    }
                    double start = std::numeric_limits<double>::infinity(), finish = at;
                    double critical_wait = 0, critical_work = 0;
                    std::uint64_t physical = 0;
                    std::vector<TraceSpan> spans;
                    for (const auto& [index, count] : counts) {
                        const auto cycle = static_cast<std::uint64_t>(buses[index].reserve(
                            cfg.command_clock_cycles(at), count * burst_cycles).start_ns);
                        const auto begin = std::max(at, cfg.command_clock_time_ns(cycle));
                        const auto end = std::max(begin, cfg.command_clock_time_ns(cycle + count * burst_cycles));
                        const auto duration = cfg.command_clock_time_ns(count * burst_cycles);
                        start = std::min(start, begin);
                        if (end > finish) {
                            finish = end;
                            critical_wait = begin - at;
                            critical_work = duration;
                        }
                        physical += count * burst;
                        busy_cycles += count * burst_cycles;
                        work += duration;
                        wait += begin - at;
                        if (trace_spans_enabled(request.trace)) {
                            add_trace_span(&spans, request.op == Op::Read ? "hbf_buffer_read" : "hbf_buffer_write",
                                "hbm_buffer_bus", "hbm/group" + std::to_string(index), begin, end, false,
                                std::to_string(count * burst) + "B; shared HBM service group");
                        }
                    }
                    const auto actual = device.transfer_controller_buffer(request);
                    require(actual.start_ns == start && actual.finish_ns == finish &&
                        actual.physical_bytes == physical && actual.logical_bytes == bytes &&
                        actual.breakdown.scheduler_queue_wait_ns == critical_wait &&
                        actual.breakdown.channel_transfer_ns == critical_work,
                        "buffer stripe grouping differs from the per-burst timing oracle");
                    require(actual.spans.size() == spans.size(), "buffer trace span count changed");
                    for (std::size_t i = 0; i < spans.size(); ++i)
                        require(actual.spans[i].name == spans[i].name && actual.spans[i].entity == spans[i].entity &&
                            actual.spans[i].start_ns == spans[i].start_ns && actual.spans[i].end_ns == spans[i].end_ns &&
                            actual.spans[i].detail == spans[i].detail, "buffer trace order or payload changed");
                    (request.op == Op::Read ? reads : writes) += physical;
                    const auto& stats = device.stats();
                    require(stats.read_bytes == reads && stats.write_bytes == writes &&
                        stats.bus_busy_ns == cfg.command_clock_time_ns(busy_cycles) &&
                        stats.controller_buffer_bus_busy_ns == work &&
                        stats.stage_work.channel_transfer_ns == work &&
                        stats.stage_work.scheduler_queue_wait_ns == wait,
                        "buffer grouping changed exact cumulative accounting");
                }
            }
        }
    }
}
void sharing_and_future_gaps() {
    const auto cfg = memory_config();
    hbm::HbmDevice baseline(cfg), shared(cfg);
    baseline.reserve_controller_buffer(4096);
    shared.reserve_controller_buffer(4096);
    const auto future = shared.transfer_controller_buffer(
        access(shared.application_capacity_bytes(), 4096, 1000, Op::Write));
    const auto early = shared.issue(access(0, 64, 0));
    require(early.finish_ns < 1000, "a future buffer transfer blocked an earlier application request");
    const auto alone = baseline.issue(access(0, 64, 1020));
    const auto contended = shared.issue(access(0, 64, 1020));
    require(contended.finish_ns > alone.finish_ns && contended.finish_ns >= future.finish_ns,
        "application traffic did not contend with the reserved HBM data channel");
    const auto& stats = shared.stats();
    require(stats.controller_buffer_write_bytes == 4096 && stats.write_bytes == 4096,
        "HBM buffer bytes are absent from physical traffic totals");
    require(stats.read_bytes == 128, "application bytes changed with controller traffic");
    require(std::abs(stats.bus_busy_ns - (4096 + 128) / cfg.pseudo_channel_bandwidth_GBps()) < 1e-9,
        "HBM shared data-bus work is not conserved");
    rejects([&] { (void)shared.issue(access(shared.application_capacity_bytes(), 64, 2000)); },
        "application accessed the reserved controller address range");
}
void joint_frontier_preserves_buffer_and_application_timing() {
    auto cfg = memory_config();
    cfg.device.channels_per_stack = 2;
    hbm::HbmDevice retained(cfg), reclaimed(cfg);
    retained.reserve_controller_buffer(16384);
    reclaimed.reserve_controller_buffer(16384);
    const auto compare = [](const PhysicalCompletion& a, const PhysicalCompletion& b) {
        require(a.start_ns == b.start_ns && a.finish_ns == b.finish_ns &&
            a.physical_bytes == b.physical_bytes && a.spans.size() == b.spans.size(),
            "joint frontier changed a shared-channel completion");
        for (std::size_t i = 0; i < a.spans.size(); ++i)
            require(a.spans[i].start_ns == b.spans[i].start_ns &&
                a.spans[i].end_ns == b.spans[i].end_ns,
                "joint frontier changed a command or buffer interval");
    };
    for (std::uint64_t wave = 0; wave < 256; ++wave) {
        const double at = wave * 4000.0;
        reclaimed.advance_buffer_frontier(at);
        auto transfer = access(retained.application_capacity_bytes(), 4096, at + 2000, Op::Write);
        compare(retained.transfer_controller_buffer(transfer), reclaimed.transfer_controller_buffer(transfer));
        // A future DMA reservation must still allow this earlier application
        // read, and the later read must still collide with the DMA payload.
        compare(retained.issue(access(0, 4096, at)), reclaimed.issue(access(0, 4096, at)));
        compare(retained.issue(access(4096, 4096, at + 2000)),
                reclaimed.issue(access(4096, 4096, at + 2000)));
    }
    const auto& a = retained.stats();
    const auto& b = reclaimed.stats();
    require(a.read_bytes == b.read_bytes && a.write_bytes == b.write_bytes &&
        a.bus_busy_ns == b.bus_busy_ns && a.service_quanta == b.service_quanta &&
        a.service_busy_ns == b.service_busy_ns && a.finish_ns == b.finish_ns,
        "joint frontier changed physical traffic or timing totals");
}
hbfsim::host::HbfConfig flash_config() {
    hbfsim::host::HbfConfig c;
    c.device.channels_per_stack = 1;
    c.device.planes_per_die = 2;
    c.device.blocks_per_plane = 16;
    c.device.pages_per_block = 8;
    c.host.write_coalescing_enabled = true;
    c.host.write_buffer_pages = 2;
    c.host.write_buffer_completion_requires_flush = true;
    return c;
}
void structural_lookup_does_not_advance_joint_frontier() {
    using hbfsim::host::MappingOrganization;
    constexpr std::uint64_t pages = 16;
    constexpr std::uint64_t page_bytes = 4096;
    for (const auto organization : {MappingOrganization::Page, MappingOrganization::Block,
            MappingOrganization::BlockLog, MappingOrganization::Extent}) {
        try {
            SimulationSessionConfig config;
            config.hbm = memory_config();
            config.hbf = flash_config();
            config.hbf.device.blocks_per_plane = 32;
            config.hbf.host.logical_capacity_bytes = 128 * page_bytes;
            config.hbf.host.ctrl_dram_bytes = 64 * 1024;
            config.hbf.host.mapping_organization = organization;
            config.initial_hbf_logical_pages = pages;
            SimulationSession session(config);
            // Scheduling this read creates future mapping/HBM reservations.
            // Its internal lookup times are not a joint application frontier:
            // the second, independent request still arrives at 100 ns.
            const auto result = session.run_batch("independent-memory", {
                {.id = "hbf-range", .target = SimulationTarget::HbfLogical,
                    .op = Op::Read, .addr = 0, .bytes = pages * page_bytes, .issue_ns = 0},
                {.id = "hbm-independent", .target = SimulationTarget::Hbm,
                    .op = Op::Read, .addr = 0, .bytes = 64, .issue_ns = 100},
            });
            require(result.memory_transactions == 2 && result.dependency_edges == 0 &&
                result.completions.size() == 2,
                "joint-frontier fixture changed its two independent memory requests");
            const PhysicalCompletion* hbf = nullptr;
            const PhysicalCompletion* hbm = nullptr;
            for (const auto& completion : result.completions) {
                if (completion.id == "hbf-range") hbf = &completion;
                if (completion.id == "hbm-independent") hbm = &completion;
            }
            require(hbf && hbm, "joint-frontier fixture lost a completion");
            require(hbf->arrival_ns == 0 && hbm->arrival_ns == 100 &&
                hbm->start_ns >= 100 && hbm->finish_ns < hbf->finish_ns,
                "future HBF lookup serialized an independent HBM request");
            require(hbf->logical_bytes == pages * page_bytes && hbm->logical_bytes == 64 &&
                result.device_after.hbf.page_reads >= pages &&
                result.device_after.hbf.host_hbm_read_bytes > 0,
                "joint-frontier fixture did not read initialized media and shared HBM metadata");
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(hbfsim::host::to_string(organization)) +
                " joint-frontier regression: " + error.what());
        }
    }
}
void capacity_and_writeback() {
    auto c = flash_config();
    hbfsim::verification::HbfWithHbm controller(c, nullptr, memory_config());
    auto& hbm = controller.buffer_memory();
    require(hbm.application_capacity_bytes() + controller.stats().host_hbm_reserved_bytes ==
        hbm.config().device.capacity_bytes, "controller capacity was not deducted from HBM");
    const auto write = controller.issue({.id="write", .tier=Tier::HBF, .op=Op::Write,
        .arrival_ns=0, .addr=0, .bytes=4096});
    const auto& stats = controller.stats();
    require(stats.page_programs == 1 && stats.host_hbm_read_bytes >= 4096 &&
        stats.host_hbm_read_bytes < 8192 && stats.host_hbm_write_bytes >= 4096,
        "full buffered page has missing traffic or a duplicate program-source HBM read");
    require(stats.host_hbm_read_bytes == hbm.stats().controller_buffer_read_bytes &&
        stats.host_hbm_write_bytes == hbm.stats().controller_buffer_write_bytes,
        "controller and physical HBM traffic ledgers disagree");
    require(write.finish_ns >= hbm.stats().finish_ns, "HBF completion precedes its HBM dependency");
    SimulationSessionConfig session;
    session.hbm = memory_config();
    session.hbf = c;
    SimulationSession composed(session);
    require(composed.hbm_application_capacity_bytes() + composed.hbf_buffer_hbm_bytes() ==
        session.hbm.device.capacity_bytes, "session capacity differs from the physical reservation");
    session.enable_hbm = false;
    rejects([&] { SimulationSession invalid(session); }, "zero-HBM configuration silently gained HBM storage");
}
}
int main() {
    try { buffer_grouping_matches_burst_oracle(); calendar_query_reuse(); sharing_and_future_gaps(); joint_frontier_preserves_buffer_and_application_timing(); structural_lookup_does_not_advance_joint_frontier(); capacity_and_writeback(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    std::cout << "HBM capacity, shared channels, future gaps and writeback passed\n";
}
