#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_controller.hpp"
#include "host/hbf_persistent_image.hpp"
#include "physical/hbf/hbf_device.hpp"
#include "physical/hbm/hbm_device.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace hbfsim::physical;
using hbfsim::host::HbfConfig;
using HbfController = hbfsim::verification::HbfWithHbm;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F&& f, const char* message) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}
HbfConfig tiny() {
    HbfConfig c;
    c.device.channels_per_stack = 2;
    c.device.planes_per_die = 2;
    c.device.blocks_per_plane = 16;
    c.device.pages_per_block = 8;
    c.device.t_erase_block_ns = 2000;
    c.device.t_program_page_ns = 750;
    c.device.t_read_page_ns = 40;
    c.host.gc_reserved_free_blocks_per_plane = 2;
    return c;
}
void transport_and_order() {
    auto c = tiny(); hbf::HbfDevice d(c.device);
    require(hbf::speed_grade(1).payload_GBps_per_channel * 8 == 384, "grade 1 table");
    require(hbf::speed_grade(2).payload_GBps_per_channel * 16 == 1536, "grade 2 table");
    require(hbf::speed_grade(3).payload_GBps_per_channel * 16 == 3072, "grade 3 table");
    using Direction = hbf::HbfDevice::Direction;
    auto rx = d.transfer(0, Direction::HostToDevice, 4096, 0, 0);
    auto tx = d.transfer(0, Direction::DeviceToHost, 4096, 0, 0);
    auto other = d.transfer(1, Direction::HostToDevice, 4096, 0, 0);
    require(rx.finish_ns == tx.finish_ns && rx.finish_ns == other.finish_ns, "independent channels and directions");
    require(std::abs(rx.finish_ns - 4096.0 / 96) < 1e-12, "one efficiency factor");
    rejects([&] { (void)d.prepare_program(1); }, "skipped first page rejected");
    require(d.prepare_program(0), "page zero requires erase"); d.complete_program(0, 3000);
    require(!d.prepare_program(1), "append avoids erase"); d.complete_program(1, 4000);
    rejects([&] { (void)d.prepare_program(3); }, "skipped page rejected");
    rejects([&] { (void)d.read_ready(7); }, "erased page rejected");
    require(d.read_ready(0) == 3000 && d.read_ready(1) == 4000,
        "each page waits for its own nonposted program completion");
}
void bank_cache() {
    auto c = tiny(); hbf::HbfDevice d(c.device);
    const auto other_bank = c.device.blocks_per_plane * c.device.pages_per_block;
    d.cache_fill(0, 10); d.cache_fill(1, 20);
    require(d.cache_hit(0, 25), "first two pages retained");
    d.cache_fill(2, 30);
    require(!d.cache_hit(1, 31) && d.cache_hit(0, 31), "third page evicts LRU");
    d.cache_fill(other_bank, 32); d.cache_fill(other_bank + 1, 33);
    require(d.cache_hit(2, 34), "other bank cannot evict local page");
    d.cache_purge_page(0); d.cache_purge_page(2);
    require(!d.cache_hit(1, 40), "evicted page never resurrects");
    d.advance_cache(40);
    d.cache_fill(0, 50); d.cache_fill(1, 60);
    require(!d.cache_hit(0, 49), "future fill invisible");
    require(d.cache_hit(0, 100), "future consumer reservation");
    d.cache_fill(2, 70); d.cache_fill(3, 80);
    d.advance_cache(100); // must retain every already-returned hit, with <=2 slots
    d.cache_purge_block(0);
    require(!d.cache_hit(0, 101), "erase invalidates decoded data");
}
void pending_cache_banks_match_isolated_banks() {
    auto c = tiny().device;
    c.stacks = 2;
    c.channels_per_stack = 4;
    c.planes_per_die = 4;
    hbf::HbfDevice shared(c);
    const auto bank_pages = std::uint64_t{c.blocks_per_plane} * c.pages_per_block;
    const auto banks = c.stacks * c.channels_per_stack * c.planes_per_die;
    c.stacks = c.channels_per_stack = c.planes_per_die = 1;
    std::vector<hbf::HbfDevice> isolated;
    for (unsigned i = 0; i < banks; ++i) isolated.emplace_back(c);
    std::mt19937_64 random(9173);
    double at = 0;
    const auto advance = [&] {
        shared.advance_cache(at);
        for (auto& bank : isolated) bank.advance_cache(at);
    };
    for (unsigned step = 0; step < 4000; ++step) {
        at += random() % 4;
        advance();
        const auto index = random() % banks;
        const auto page = random() % 16;
        const auto ppn = index * bank_pages + page;
        switch (random() % 5) {
        case 0:
        case 1: {
            const auto ready = at + random() % 200;
            shared.cache_fill(ppn, ready);
            isolated[index].cache_fill(page, ready);
            break;
        }
        case 2: {
            const auto ready = at + random() % 300;
            require(shared.cache_hit(ppn, ready) == isolated[index].cache_hit(page, ready),
                "pending-bank maintenance changed a future cache consumer");
            break;
        }
        default:
            // Host invalidation fences every reserved consumer. Retire all
            // pending banks, then exercise removal and later reactivation.
            at += 1000;
            advance();
            if (step % 2) {
                shared.cache_purge_page(ppn);
                isolated[index].cache_purge_page(page);
            } else {
                shared.cache_purge_block(ppn / c.pages_per_block);
                isolated[index].cache_purge_block(page / c.pages_per_block);
            }
        }
    }
    at += 1000;
    advance();
    for (unsigned index = 0; index < banks; ++index)
        for (unsigned page = 0; page < 16; ++page)
            require(shared.cache_hit(index * bank_pages + page, at) == isolated[index].cache_hit(page, at),
                "pending-bank maintenance changed final cache residency");
    advance();
}
void host_lifecycle() {
    auto c = tiny(); c.host.mapping_mode = hbfsim::host::MappingMode::RawPhysical;
    c.host.auto_gc_enabled = false; c.host.gc_reserved_free_blocks_per_plane = 0;
    HbfController host(c);
    auto write = host.issue_channel_local(PhysicalRequest{.id="first", .tier=Tier::HBF,
        .op=Op::Write, .trace={.mode=TraceMode::Full}, .arrival_ns=0, .addr=0, .bytes=4096});
    const TraceSpan* program = nullptr;
    for (const auto& span : write.spans) {
        if (span.name.ends_with("/array_program")) program = &span;
    }
    require(program != nullptr, "raw write exposes its program interval");
    require(std::abs(program->end_ns - program->start_ns - c.device.t_program_page_ns) < 1e-8,
        "program interval equals the complete configured program time");
    require(write.finish_ns == program->end_ns,
        "raw program completion adds no separate verification delay");
    require(write.finish_ns >= c.device.t_erase_block_ns + c.device.t_program_page_ns,
        "write acknowledges after autoerase and program");
    auto stats = host.stats(); require(stats.block_erases == 1 && stats.page_programs == 1, "one physical erase per first write");
    auto read = PhysicalRequest{.id="aligned", .tier=Tier::HBF, .op=Op::Read,
        .arrival_ns=write.finish_ns, .addr=0, .bytes=64};
    auto invalid_read = read; invalid_read.addr = 1;
    rejects([&] { (void)host.issue_channel_local(invalid_read); }, "unaligned channel read rejected");
    invalid_read = read; invalid_read.bytes = 65;
    rejects([&] { (void)host.issue_channel_local(invalid_read); }, "partial cache-line channel read rejected");
    invalid_read = read; invalid_read.addr = 4032; invalid_read.bytes = 128;
    rejects([&] { (void)host.issue_channel_local(invalid_read); }, "cross-page channel read rejected");
    const auto read_done = host.issue_channel_local(read);
    auto inv = host.invalidate_zone(0, 0, 0, read_done.finish_ns);
    auto reset = host.reset_zone(0, 0, 0, inv.finish_ns);
    require(host.stats().block_erases == 1, "host reset never doubles erase");
    auto second = host.issue_channel_local(PhysicalRequest{.id="reuse", .tier=Tier::HBF,
        .op=Op::Write, .arrival_ns=reset.finish_ns, .addr=0, .bytes=4096});
    (void)host.drain_pending("done", second.finish_ns);
    require(host.stats().block_erases == 2, "reused page zero pays one erase");
    const auto pec = host.block_erase_counts();
    require(std::accumulate(pec.begin(), pec.end(), std::uint64_t{0}) == 2, "physical PEC conservation");
}
// Snapshot precondition: an installed image may declare its never-programmed
// free blocks erased. Page zero then skips the erase once; reuse, and any
// block that saw an explicit raw erase, keep the OCP page-zero auto erase.
void preerased_blocks() {
    auto c = tiny();
    {
        hbf::HbfDevice d(c.device);
        const auto pages = c.device.pages_per_block;
        d.seed_block(2, 3);
        rejects([&] { d.seed_erased_block(2); }, "a programmed block was seeded erased");
        d.seed_erased_block(0); d.seed_erased_block(1);
        require(d.preconditioned_erased_blocks() == 2 && d.unconsumed_erased_blocks() == 2, "erased seeds counted");
        require(!d.prepare_program(0), "pre-erased page zero erased again");
        require(d.unconsumed_erased_blocks() == 1, "a program did not consume the erased state");
        for (std::uint32_t p = 1; p < pages; ++p) require(!d.prepare_program(p), "append erased");
        require(d.prepare_program(0), "a reused block skipped its page-zero erase");
        d.erase(1);
        require(d.unconsumed_erased_blocks() == 0 && d.prepare_program(pages), "a raw erase replaced the auto erase");
        rejects([&] { d.seed_erased_block(3); }, "an erased seed after commands was accepted");
    }
    for (bool preerased : {false, true}) {
        auto h = tiny(); h.host.mapping_mode = hbfsim::host::MappingMode::RawPhysical;
        h.host.auto_gc_enabled = false; h.host.gc_reserved_free_blocks_per_plane = 0;
        h.host.initial_free_blocks_erased = preerased;
        HbfController host(h);
        auto write = host.issue_channel_local(PhysicalRequest{.id="first", .tier=Tier::HBF,
            .op=Op::Write, .arrival_ns=0, .addr=0, .bytes=4096});
        require(host.stats().block_erases == (preerased ? 0 : 1), "first-use erase follows the precondition");
        if (preerased) rejects([&] { (void)host.persistent_image(); }, "an image dropped pre-erased blocks");
        auto inv = host.invalidate_zone(0, 0, 0, write.finish_ns);
        auto reset = host.reset_zone(0, 0, 0, inv.finish_ns);
        auto second = host.issue_channel_local(PhysicalRequest{.id="reuse", .tier=Tier::HBF,
            .op=Op::Write, .arrival_ns=reset.finish_ns, .addr=0, .bytes=4096});
        (void)host.drain_pending("done", second.finish_ns);
        const auto stats = host.stats();
        require(stats.block_erases == (preerased ? 1 : 2), "a reused block did not erase");
        require(stats.block_erases == stats.erase_requests + stats.auto_erase_requests && stats.accounting_verified,
            "pre-erased blocks broke the erase ledger");
        require((stats.preconditioned_erased_blocks > 0) == preerased, "preconditioned block count");
        const auto pec = host.block_erase_counts();
        require(std::accumulate(pec.begin(), pec.end(), std::uint64_t{0}) == stats.block_erases,
            "pre-erased first use changed physical PEC");
    }
    {
        auto h = tiny(); h.host.initial_free_blocks_erased = true;
        HbfController ftl(h);
        double now = 0;
        for (std::uint64_t lpn = 0; lpn < 64; ++lpn)
            now = ftl.issue(PhysicalRequest{.id="ftl", .tier=Tier::HBF, .op=Op::Write,
                .address_space=AddressSpace::Logical, .arrival_ns=now, .addr=lpn * 4096, .bytes=4096}).finish_ns;
        (void)ftl.drain_pending("ftl", now);
        const auto stats = ftl.stats();
        require(stats.page_programs > 0 && stats.block_erases == 0 && stats.auto_erase_requests == 0,
            "the page FTL erased a pre-erased free block on first use");
        require(stats.accounting_verified, "pre-erased page-FTL accounting");
    }
}
void hbm4_baseline() {
    hbm::HbmConfig c;
    require(c.device.channels_per_stack == 32 && c.device.pseudo_channels_per_channel == 2, "HBM4 independent organization");
    require(c.device.bank_groups_per_pseudo_channel * c.device.banks_per_group == 64, "HBM4 banks");
    require(c.channel_bandwidth_GBps() * c.device.channels_per_stack == 2048, "HBM4 8Gb/s reference bandwidth");
    require(c.timing.bandwidth_efficiency > 0 && c.timing.bandwidth_efficiency <= 1, "HBM effective bandwidth");
}
void independent_stack_payloads() {
    auto c = tiny();
    c.device.stacks = 2;
    c.device.channels_per_stack = c.device.planes_per_die = 1;
    c.host.mapping_mode = hbfsim::host::MappingMode::RawPhysical;
    c.host.auto_gc_enabled = false;
    c.host.gc_reserved_free_blocks_per_plane = 0;
    HbfController host(c);
    const auto stack_bytes = static_cast<std::uint64_t>(c.device.blocks_per_plane) *
        c.device.pages_per_block * c.device.page_size_bytes;
    const auto write = [&](std::uint64_t address, double arrival) {
        return host.issue_channel_local(PhysicalRequest{.id="parallel-page", .tier=Tier::HBF,
            .op=Op::Write, .arrival_ns=arrival, .addr=address,
            .bytes=c.device.page_size_bytes});
    };
    const auto warm_a = write(0, 0);
    const auto warm_b = write(stack_bytes, 0);
    const auto ready = std::max(warm_a.finish_ns, warm_b.finish_ns);
    // The second page avoids page-zero erase hiding an ingress bottleneck.
    const auto a = write(c.device.page_size_bytes, ready);
    const auto b = write(stack_bytes + c.device.page_size_bytes, ready);
    require(std::abs(a.finish_ns - b.finish_ns) < 1e-8,
        "independent HBF stacks serialized through an extra shared payload port");
    (void)host.drain_pending("parallel-drain", std::max(a.finish_ns, b.finish_ns));
    require(host.stats().page_programs == 4 && host.stats().block_erases == 2,
        "parallel transport changed media work");
}
void fragment_protocol() {
    auto c = tiny().device;
    c.write_accumulation_timeout_ns = 100;
    c.outstanding_write_pages_per_channel = 1;
    hbf::HbfDevice device(c);
    const auto other_channel = std::uint64_t{c.planes_per_die} * c.blocks_per_plane * c.pages_per_block;
    require(!device.receive_write_fragment(0, 0, 64, 1, 0).page_ready, "first fragment must not program");
    rejects([&] { (void)device.prepare_program(0); }, "partial buffer cannot program NAND");
    require(device.buffered_write_read_status(0, 0, 64) == 0, "received line forwards from buffer");
    require(device.buffered_write_read_status(0, 0, 128) == 0xA, "missing line returns status A");
    require(device.receive_write_fragment(0, 0, 128, 2, 1).status == 2, "overlap rejected atomically");
    require(device.buffered_write_read_status(0, 64, 64) == 0xA, "overlap rejection did not install its new half");
    require(device.receive_write_fragment(c.pages_per_block, 0, 64, 3, 2).status == 4, "page credit exhausted");
    require(device.receive_write_fragment(other_channel, 0, 64, 4, 2).status == 0, "credits are per channel");
    require(device.receive_write_fragment(1, 0, 64, 5, 3).status == 6, "cannot skip an incomplete predecessor");
    require(device.receive_write_fragment(0, 64, 4032, 6, 4).page_ready, "full queue must accept remaining fragments");
    require(device.prepare_program(0), "assembled page zero erases once");
    device.complete_program(0, 1000);
    require(device.advance_writes(100).empty(), "accumulation deadline cannot time out a full page programming NAND");
    require(device.receive_write_fragment(c.pages_per_block, 0, 64, 7, 100).status == 4,
        "programming page holds its credit until completion");
    auto timeout = device.advance_writes(102);
    require(timeout.size() == 1 && timeout[0].command == 4 && timeout[0].status == 5,
        "other channel expires independently");
    require(device.advance_writes(999).empty(), "receipt is not nonposted completion");
    const auto completed = device.advance_writes(1000);
    require(completed.size() == 2 && completed[0].command == 1 && completed[1].command == 6 &&
        completed[0].status == 0 && completed[1].finish_ns == 1000, "one response per accepted command after program");
    require(device.pending_write_pages() == 0, "all page credits released");
    require(device.receive_write_fragment(c.pages_per_block, 0, 64, 8, 1000).status == 0, "released credit reusable");
    require(device.advance_writes(1100).front().status == 5, "deadline is measured from first receipt");
    require(!device.receive_write_fragment(c.pages_per_block, 64, 4032, 9, 1100).page_ready,
        "retry must not reuse coverage from timed-out page");
    require(device.advance_writes(1200).front().status == 5, "retry missing first fragment also expires");
    require(device.receive_write_fragment(c.pages_per_block, 0, 4096, 10, 1200).page_ready, "full retry accepted");
    require(device.prepare_program(c.pages_per_block), "timeouts did not advance NAND order");
    device.complete_program(c.pages_per_block, 1300);
    require(device.advance_writes(1300).front().status == 0, "retry completes");
}
void channel_fragment_batch() {
    auto c = tiny();
    c.host.mapping_mode = hbfsim::host::MappingMode::RawPhysical;
    c.host.auto_gc_enabled = false;
    c.host.gc_reserved_free_blocks_per_plane = 0;
    HbfController host(c);
    std::vector<PhysicalRequest> fragments;
    for (unsigned i = 0; i < 64; ++i)
        fragments.push_back(PhysicalRequest{.id = "fragment" + std::to_string(i), .tier = Tier::HBF,
            .op = Op::Write, .trace = {.mode = TraceMode::Full}, .arrival_ns = i * 10.0,
            .addr = i * 64u, .bytes = 64});
    const auto results = host.issue_channel_write_batch(fragments);
    const auto finish = results.back().completion.finish_ns;
    double hbio_work = 0;
    std::uint64_t programmed = 0;
    for (const auto& result : results) {
        require(result.status == 0 && result.completion.finish_ns == finish, "all fragments finish with one NAND program");
        hbio_work += result.completion.breakdown.hb_io_transfer_ns;
        programmed += result.completion.physical_bytes;
    }
    require(finish >= results.back().received_ns + c.device.t_program_page_ns + c.device.t_erase_block_ns,
        "no media operation before complete-page assembly");
    require(programmed == 4096 && host.stats().page_programs == 1 && host.stats().block_erases == 1,
        "64 fragments charge exactly one program and erase");
    require(host.stats().program_requests == 64 && host.stats().page_reads == 0,
        "device accumulation creates no host read-modify-write");
    require(std::abs(hbio_work - (4096.0 + 64 * c.device.command_address_bytes) / 96) < 1e-8,
        "HBIO charges each command and its fragment, with no duplicate 4 KiB transfer");
    const auto read = host.issue_channel_local(PhysicalRequest{.id = "read", .tier = Tier::HBF,
        .op = Op::Read, .arrival_ns = finish, .addr = 0, .bytes = 64});
    require(read.finish_ns > finish, "assembled page is visible to normal channel reads");

    auto short_config = c;
    short_config.device.write_accumulation_timeout_ns = 100;
    HbfController timed(short_config);
    auto partial = fragments[0];
    auto duplicate = partial; duplicate.id = "duplicate"; duplicate.arrival_ns = 10;
    const auto failed = timed.issue_channel_write_batch({partial, duplicate});
    require(failed[0].status == 5 && failed[1].status == 2, "timeout and duplicate have distinct OCP statuses");
    require(failed[0].completion.finish_ns == failed[0].received_ns + 100,
        "timeout starts at physical receipt, not host submission");
    require(timed.stats().page_programs == 0 && timed.stats().block_erases == 0,
        "incomplete write has no NAND program/erase side effect");
    partial.bytes = 4096;
    partial.arrival_ns = timed.stats().finish_ns;
    const auto retry = timed.issue_channel_write_batch({partial});
    require(retry[0].status == 0 && timed.stats().page_programs == 1, "full-page retry after timeout uses original page");

    c.device.outstanding_write_pages_per_channel = 1;
    HbfController limited(c);
    const auto other_channel = std::uint64_t{c.device.planes_per_die} *
        c.device.blocks_per_plane * c.device.pages_per_block * 4096;
    const auto credit_results = limited.issue_channel_write_batch({
        PhysicalRequest{.id="head", .tier=Tier::HBF, .op=Op::Write, .arrival_ns=0, .addr=0, .bytes=64},
        PhysicalRequest{.id="full", .tier=Tier::HBF, .op=Op::Write, .arrival_ns=1,
            .addr=std::uint64_t{c.device.pages_per_block}*4096, .bytes=64},
        PhysicalRequest{.id="skip", .tier=Tier::HBF, .op=Op::Write, .arrival_ns=1, .addr=4096, .bytes=64},
        PhysicalRequest{.id="other-channel", .tier=Tier::HBF, .op=Op::Write, .arrival_ns=1,
            .addr=other_channel, .bytes=4096},
        PhysicalRequest{.id="tail", .tier=Tier::HBF, .op=Op::Write, .arrival_ns=2, .addr=64, .bytes=4032}});
    require(credit_results[0].status == 0 && credit_results[1].status == 4 &&
        credit_results[2].status == 6 && credit_results[2].additional_status == 1 &&
        credit_results[3].status == 0 && credit_results[4].status == 0,
        "native batches preserve channel credits, page order and fill-existing-page admission");
    require(limited.stats().page_programs == 2 && limited.stats().block_erases == 2,
        "rejected commands perform no NAND work");
}
void read_groups_match_traced_schedule() {
    for (unsigned stacks : {1u, 2u}) for (unsigned credits : {1u, 7u, 64u}) {
        auto c = tiny();
        c.device.stacks = stacks;
        c.device.blocks_per_plane = 64;
        c.device.page_read_queue_depth_per_stack = credits;
        c.device.thermal_enabled = true;
        HbfController grouped(c), traced(c);
        grouped.prepopulate_mutable_logical_page_range(0, 512);
        traced.prepopulate_mutable_logical_page_range(0, 512);
        std::mt19937_64 random(9722);
        double arrival = 0;
        for (unsigned i = 0; i < 120; ++i) {
            // Overlapping reads, partial edges, writes and post-write reads
            // exercise credits, temporal cache fills and program conflicts.
            const bool write = i % 13 == 12;
            PhysicalRequest request{.id = "group-" + std::to_string(i),
                .tier = Tier::HBF, .op = write ? Op::Write : Op::Read,
                .address_space = AddressSpace::Logical, .arrival_ns = arrival,
                .addr = (random() % 448) * 4096 + (random() % 8) * 64,
                .bytes = write ? 128u : (2 + random() % 30) * 4096 - 512};
            const auto a = grouped.issue(request);
            request.trace.mode = TraceMode::Full;
            const auto b = traced.issue(request);
            require(a.start_ns == b.start_ns && a.finish_ns == b.finish_ns,
                "group changes exact completion timing");
            require(a.breakdown == b.breakdown, "group changes stage work or queue waits");
            require(a.physical_bytes == b.physical_bytes, "group changes physical traffic");
            const auto& x = grouped.execution_stats();
            const auto& y = traced.execution_stats();
            require(x.page_reads == y.page_reads && x.page_programs == y.page_programs &&
                x.block_erases == y.block_erases && x.read_buffer_hits == y.read_buffer_hits &&
                x.ecc_decode_ops == y.ecc_decode_ops && x.ecc_codeword_bytes == y.ecc_codeword_bytes &&
                x.host_hbm_read_bytes == y.host_hbm_read_bytes &&
                x.host_hbm_write_bytes == y.host_hbm_write_bytes &&
                x.page_read_admission_wait_ns == y.page_read_admission_wait_ns,
                "group changes physical/cache/metadata/admission state");
            arrival = i % 5 == 4 ? a.finish_ns : arrival + 0.25;
        }
        require(grouped.execution_stats().read_pipeline_group_pages > 100,
            "foreground stream never entered the grouped kernel");
        require(traced.execution_stats().read_pipeline_group_pages == 0,
            "traced event scheduler unexpectedly used the numeric kernel");
    }
}
void ordered_read_rounds_match_traced_schedule() {
    constexpr std::uint64_t page_bytes = 4096, banks = 256, read_pages = 769;
    for (const unsigned stacks : {1u, 2u}) for (const unsigned credits : {64u, 512u}) {
        auto c = tiny();
        c.device.stacks = stacks;
        c.device.channels_per_stack = 8 / stacks;
        c.device.dies_per_channel = 2;
        c.device.planes_per_die = 16;
        c.device.page_read_queue_depth_per_stack = credits;
        c.device.thermal_enabled = false;
        c.device.t_read_page_ns = 4000;
        // Several banks share each channel/die, and all commands and data of
        // one stack share its TSV. Deliberately queue these paths so equality
        // cannot follow merely from independent bank completion times.
        c.device.channel_bandwidth_GBps = 2.0;
        c.device.tsv_bandwidth_GBps = 0.5;
        c.device.ecc_decode_raw_bandwidth_GBps_per_die = 1.0;
        c.device.ecc_decode_latency_ns = 5000;
        c.host.mapping_mode = hbfsim::host::MappingMode::FullResident;
        c.host.write_coalescing_enabled = false;
        HbfController grouped(c), traced(c);
        grouped.prepopulate_mutable_logical_page_range(0, 1536);
        traced.prepopulate_mutable_logical_page_range(0, 1536);
        std::vector<unsigned> pages_per_bank(banks, 0);
        const auto bank_pages = std::uint64_t{c.device.blocks_per_plane} * c.device.pages_per_block;
        for (std::uint64_t lpn = 0; lpn < read_pages; ++lpn) {
            const auto ppn = grouped.mapped_physical_page(lpn);
            require(ppn && ppn == traced.mapped_physical_page(lpn),
                "ordered read fixture has different initial physical pages");
            require(*ppn / bank_pages < banks, "ordered read fixture exceeded 256 banks");
            ++pages_per_bank[*ppn / bank_pages];
        }
        require(std::all_of(pages_per_bank.begin(), pages_per_bank.end(),
                    [](unsigned pages) { return pages >= 3; }),
            "ordered read fixture did not cover three complete 256-bank rounds");
        const auto compare = [&](const PhysicalCompletion& a, const PhysicalCompletion& b) {
            require(a.start_ns == b.start_ns && a.finish_ns == b.finish_ns &&
                a.breakdown == b.breakdown && a.physical_bytes == b.physical_bytes,
                "ordered rounds change exact completion, stage work or physical bytes");
            const auto& x = grouped.execution_stats();
            const auto& y = traced.execution_stats();
            require(x.logical_read_bytes == y.logical_read_bytes &&
                x.logical_write_bytes == y.logical_write_bytes &&
                x.physical_read_bytes == y.physical_read_bytes &&
                x.physical_write_bytes == y.physical_write_bytes &&
                x.page_reads == y.page_reads && x.page_programs == y.page_programs &&
                x.block_erases == y.block_erases && x.gc_runs == y.gc_runs &&
                x.gc_relocations == y.gc_relocations &&
                x.read_buffer_hits == y.read_buffer_hits &&
                x.read_buffer_misses == y.read_buffer_misses &&
                x.read_buffer_read_bytes == y.read_buffer_read_bytes &&
                x.mapping_lookup_ops == y.mapping_lookup_ops &&
                x.mapping_update_ops == y.mapping_update_ops &&
                x.mapping_media_reads == y.mapping_media_reads &&
                x.mapping_page_programs == y.mapping_page_programs &&
                x.host_hbm_read_bytes == y.host_hbm_read_bytes &&
                x.host_hbm_write_bytes == y.host_hbm_write_bytes &&
                x.ecc_decode_ops == y.ecc_decode_ops && x.ecc_encode_ops == y.ecc_encode_ops &&
                x.ecc_codeword_bytes == y.ecc_codeword_bytes &&
                x.ecc_decode_queue_wait_ns == y.ecc_decode_queue_wait_ns &&
                x.ecc_decode_issue_busy_ns == y.ecc_decode_issue_busy_ns &&
                x.flash_scheduler_enqueues == y.flash_scheduler_enqueues &&
                x.flash_scheduler_issues == y.flash_scheduler_issues &&
                x.page_read_admission_events == y.page_read_admission_events &&
                x.page_read_admission_waited_pages == y.page_read_admission_waited_pages &&
                x.page_read_admission_wait_ns == y.page_read_admission_wait_ns &&
                x.page_read_admission_max_wait_ns == y.page_read_admission_max_wait_ns &&
                x.stage_work == y.stage_work,
                "ordered rounds change media, mapping, cache or credit accounting");
        };
        double frontier = 0.25;
        const auto issue = [&](const char* id, Op op, double arrival,
                               std::uint64_t address, std::uint64_t bytes) {
            PhysicalRequest request{.id = id, .tier = Tier::HBF, .op = op,
                .address_space = AddressSpace::Logical, .arrival_ns = arrival,
                .addr = address, .bytes = bytes};
            const auto a = grouped.issue(request);
            request.trace.mode = TraceMode::Full;
            compare(a, traced.issue(request));
            frontier = std::max(frontier, a.finish_ns);
            return a;
        };
        // Skew one bank before the long request without advancing its arrival.
        // The later same-arrival request also revisits shared-stage calendars
        // behind previously reserved completions instead of only appending.
        (void)issue("ordered-bank-skew", Op::Read, 0.25, 17 * page_bytes, page_bytes);
        const auto ordered_before = grouped.read_engine_stats().read_pipeline_direct_gap_reservations;
        const auto range = issue("ordered-three-rounds", Op::Read, 0.25,
            13, read_pages * page_bytes - 42);
        const auto ordered_after = grouped.read_engine_stats().read_pipeline_direct_gap_reservations;
        // This deliberately congested one-stack fixture has reusable gap
        // positions. Its two-stack distribution may need the full lookup;
        // both must preserve the same schedule and traced path remains scalar.
        if (stacks == 1)
            require(ordered_after > ordered_before,
                "three-round read never reused a verified shared-stage gap");
        require(traced.read_engine_stats().read_pipeline_direct_gap_reservations == 0,
            "traced reads unexpectedly used the direct-position kernel");
        require(range.breakdown.scheduler_queue_wait_ns > 0,
            "three-round read did not exercise shared resource contention");
        if (credits == 64)
            require(grouped.execution_stats().page_read_admission_waited_pages > 0,
                "three-round read never exhausted its finite page credits");
        (void)issue("ordered-same-arrival-other-round", Op::Read, 0.25,
            1024 * page_bytes + 63, 33 * page_bytes - 126);

        const auto hits_before = grouped.execution_stats().read_buffer_hits;
        (void)issue("ordered-cache-after-rounds", Op::Read, frontier,
            768 * page_bytes + 64, 128);
        require(grouped.execution_stats().read_buffer_hits > hits_before,
            "post-round probe never consumed a retained decoded bank-cache page");
        const auto programs_before = grouped.execution_stats().page_programs;
        (void)issue("ordered-overwrite", Op::Write, frontier, 768 * page_bytes, page_bytes);
        require(grouped.execution_stats().page_programs > programs_before,
            "ordered write probe did not program media");
        const auto reads_before = grouped.execution_stats().page_reads;
        (void)issue("ordered-read-after-write", Op::Read, frontier,
            768 * page_bytes + 65, 127);
        require(grouped.execution_stats().page_reads > reads_before,
            "post-write probe reused stale decoded data instead of reading the replacement");
        (void)issue("ordered-round-boundary", Op::Read, frontier,
            255 * page_bytes + 7, 3 * page_bytes - 14);
        const auto drained = grouped.drain_pending("ordered-final", frontier);
        compare(drained, traced.drain_pending("ordered-final", frontier, {.mode = TraceMode::Full}));
        const auto& x = grouped.stats();
        const auto& y = traced.stats();
        require(x.accounting_verified && y.accounting_verified &&
            grouped.quiescence_stats().quiescent() && traced.quiescence_stats().quiescent() &&
            x.channel_data_busy_ns == y.channel_data_busy_ns &&
            x.tsv_busy_ns == y.tsv_busy_ns && x.sram_busy_ns == y.sram_busy_ns &&
            x.hb_io_data_busy_ns == y.hb_io_data_busy_ns &&
            x.media_busy_ns == y.media_busy_ns && x.ecc_issue_busy_ns == y.ecc_issue_busy_ns,
            "ordered rounds changed final shared-resource work or left pending state");
        std::ostringstream grouped_mapping, traced_mapping;
        grouped.write_mapping_snapshot_json(grouped_mapping);
        traced.write_mapping_snapshot_json(traced_mapping);
        require(grouped_mapping.str() == traced_mapping.str() &&
            grouped.wear_snapshot_json() == traced.wear_snapshot_json() &&
            grouped.block_erase_counts() == traced.block_erase_counts(),
            "ordered rounds changed final mapping or physical wear state");
        for (std::uint64_t lpn = 0; lpn < 1536; ++lpn)
            require(grouped.mapped_physical_page(lpn) == traced.mapped_physical_page(lpn),
                "ordered rounds changed a logical page's final physical mapping");
        std::cout << "ordered read rounds: stacks=" << stacks << " banks=" << banks
            << " credits=" << credits << " direct_gap_reuses=" << ordered_after - ordered_before
            << " range_queue_ns=" << range.breakdown.scheduler_queue_wait_ns << '\n';
    }
}
void compact_read_runs_preserve_mapping_boundaries() {
  for (const unsigned stacks : {2u, 3u}) {
    for (const auto mode : {hbfsim::host::MappingMode::FullResident,
                            hbfsim::host::MappingMode::Cached}) {
        auto c = tiny();
        c.device.stacks = stacks;
        c.device.blocks_per_plane = 48;
        c.device.page_read_queue_depth_per_stack = 7;
        c.host.mapping_mode = mode;
        c.host.mapping_entries_per_page = 64;
        if (mode == hbfsim::host::MappingMode::Cached) c.host.ctrl_dram_bytes = stacks * 12288;
        HbfController grouped(c), traced(c);
        constexpr std::uint64_t first = 5, pages = 769;
        grouped.prepopulate_mutable_logical_page_range(first, pages);
        traced.prepopulate_mutable_logical_page_range(first, pages);
        require(grouped.stats().compact_initial_logical_data_pages == pages,
            "regular-read test did not install a compact image");
        double arrival = 0;
        const auto compare = [&](const PhysicalCompletion& a, const PhysicalCompletion& b) {
            require(a.start_ns == b.start_ns && a.finish_ns == b.finish_ns &&
                a.breakdown == b.breakdown && a.physical_bytes == b.physical_bytes,
                "compact run changes completion or exact stage accounting");
            const auto& x = grouped.execution_stats();
            const auto& y = traced.execution_stats();
            require(x.page_reads == y.page_reads && x.page_programs == y.page_programs &&
                x.block_erases == y.block_erases && x.mapping_media_reads == y.mapping_media_reads &&
                x.mapping_lookup_ops == y.mapping_lookup_ops &&
                x.mapping_cache_hits == y.mapping_cache_hits &&
                x.mapping_cache_misses == y.mapping_cache_misses &&
                x.mapping_cache_dirty_evictions == y.mapping_cache_dirty_evictions &&
                x.host_hbm_read_bytes == y.host_hbm_read_bytes &&
                x.host_hbm_write_bytes == y.host_hbm_write_bytes &&
                x.read_buffer_hits == y.read_buffer_hits && x.ecc_decode_ops == y.ecc_decode_ops &&
                x.page_read_admission_wait_ns == y.page_read_admission_wait_ns,
                "compact run changes mapping/cache/HBM/credit physical state");
        };
        const auto issue = [&](Op op, std::uint64_t addr, std::uint64_t bytes) {
            PhysicalRequest request{.id = "compact-boundary", .tier = Tier::HBF, .op = op,
                .address_space = AddressSpace::Logical, .arrival_ns = arrival,
                .addr = addr, .bytes = bytes};
            const auto a = grouped.issue(request);
            request.trace.mode = TraceMode::Full;
            compare(a, traced.issue(request));
            arrival = a.finish_ns;
        };
        // Nonzero image origin, partial first/last pages, stack rotation,
        // multiple mapping groups and nonadjacent physical block-directory slots.
        issue(Op::Read, first * 4096 + 13, pages * 4096 - 29);
        require(grouped.execution_stats().compact_read_runs > 10 &&
            grouped.execution_stats().compact_read_run_pages == pages &&
            grouped.execution_stats().compact_read_run_reuses > pages / 2 &&
            traced.execution_stats().compact_read_runs == 0,
            "regular read did not reuse compressed bank lanes across split boundaries");
        issue(Op::Read, (first + pages - 16) * 4096 + 33, 16 * 4096 - 65);
        issue(Op::Write, 127 * 4096 + 64, 128);
        issue(Op::Write, 192 * 4096, 4096);
        issue(Op::Read, first * 4096 + 1, pages * 4096 - 2);
        const auto a = grouped.invalidate_logical_pages(255, 3, arrival);
        const auto b = traced.invalidate_logical_pages(255, 3, arrival);
        compare(a.completion, b.completion);
        arrival = a.completion.finish_ns;
        issue(Op::Read, 252 * 4096 + 17, 11 * 4096 - 30);
        issue(Op::Write, 256 * 4096, 4096);
        issue(Op::Read, first * 4096 + 63, pages * 4096 - 126);
        const auto drained = grouped.drain_pending("compact-final", arrival);
        compare(drained, traced.drain_pending("compact-final", arrival, {.mode = TraceMode::Full}));
        require(grouped.stats().accounting_verified && traced.stats().accounting_verified,
            "compact run final accounting is invalid");
        for (auto lpn = first; lpn < first + pages; ++lpn)
            require(grouped.mapped_physical_page(lpn) == traced.mapped_physical_page(lpn),
                "compact run changed final logical mapping");
        if (mode == hbfsim::host::MappingMode::Cached)
            require(grouped.execution_stats().mapping_cache_dirty_evictions > 0,
                "compact boundary test did not evict dirty mapping metadata");
        HbfController restored(c), restored_traced(c);
        restored.restore_persistent_image(grouped.persistent_image());
        restored_traced.restore_persistent_image(traced.persistent_image());
        PhysicalRequest cold{.id = "restored-compact-run", .tier = Tier::HBF, .op = Op::Read,
            .address_space = AddressSpace::Logical, .arrival_ns = 0,
            .addr = first * 4096 + 37, .bytes = pages * 4096 - 79};
        const auto resumed = restored.issue(cold);
        cold.trace.mode = TraceMode::Full;
        const auto scalar = restored_traced.issue(cold);
        require(resumed.start_ns == scalar.start_ns && resumed.finish_ns == scalar.finish_ns &&
            resumed.breakdown == scalar.breakdown && resumed.physical_bytes == scalar.physical_bytes &&
            restored.execution_stats().compact_read_run_reuses > 0 &&
            restored_traced.execution_stats().compact_read_run_pages == 0,
            "restored compact address runs changed cross-group read timing or were not used");
        std::cout << "compact boundary/restore: stacks=" << stacks
            << " mode=" << static_cast<unsigned>(mode)
            << " runs=" << grouped.execution_stats().compact_read_runs
            << " candidates=" << grouped.execution_stats().compact_read_run_pages
            << " reused=" << grouped.execution_stats().compact_read_run_reuses
            << " restored_reused=" << restored.execution_stats().compact_read_run_reuses << '\n';
    }
  }
}

void read_groups_cover_gc_and_hot_pacing() {
  for (const auto mode : {hbfsim::host::MappingMode::FullResident,
                          hbfsim::host::MappingMode::Cached}) {
    auto c = tiny();
    c.host.mapping_mode = mode;
    if (mode == hbfsim::host::MappingMode::Cached) {
        c.host.ctrl_dram_bytes = 12288;
        c.host.mapping_entries_per_page = 64;
        // Defer preventive work so a dirty mapping eviction must sometimes
        // reclaim synchronously; its read can retire an already prepared lane.
        c.host.gc_hard_watermark_pages = 32;
    }
    c.host.logical_capacity_bytes = 320 * 4096;
    c.host.gc_low_watermark_pages = mode == hbfsim::host::MappingMode::Cached ? 1 : 64;
    c.device.page_read_queue_depth_per_stack = 7;
    c.device.thermal_enabled = true;
    c.device.thermal_start_at_ceiling = true;
    c.device.thermal_throttle_power_w = 0.01;
    HbfController grouped(c), traced(c);
    grouped.prepopulate_mutable_logical_page_range(0, 320);
    traced.prepopulate_mutable_logical_page_range(0, 320);
    std::mt19937_64 random(1847);
    double arrival = 0;
    const auto compare = [&](const PhysicalCompletion& a, const PhysicalCompletion& b) {
        require(a.start_ns == b.start_ns && a.finish_ns == b.finish_ns &&
            a.breakdown == b.breakdown && a.physical_bytes == b.physical_bytes,
            "group changes hot-GC completion or stage work");
        const auto& x = grouped.execution_stats();
        const auto& y = traced.execution_stats();
        require(x.page_reads == y.page_reads && x.page_programs == y.page_programs &&
            x.block_erases == y.block_erases && x.gc_runs == y.gc_runs &&
            x.gc_relocations == y.gc_relocations && x.physical_read_bytes == y.physical_read_bytes &&
            x.physical_write_bytes == y.physical_write_bytes && x.read_buffer_hits == y.read_buffer_hits &&
            x.ecc_codeword_bytes == y.ecc_codeword_bytes &&
            x.page_read_admission_wait_ns == y.page_read_admission_wait_ns &&
            x.thermal_media_energy_j == y.thermal_media_energy_j &&
            x.thermal_throttled_media_ops == y.thermal_throttled_media_ops &&
            x.thermal_throttle_wait_ns == y.thermal_throttle_wait_ns &&
            x.thermal_pacing_busy_ns == y.thermal_pacing_busy_ns,
            "group changes hot-GC physical/cache/credit/energy accounting");
    };
    bool read_eviction_retired_run_candidate = false;
    for (unsigned i = 0; i < 360; ++i) {
        // High initial occupancy and repeated updates force live-page GC;
        // partial multi-page reads overlap updates and return across rounds.
        const bool write = i % 4 != 0;
        PhysicalRequest request{.id = "hot-gc-" + std::to_string(i),
            .tier = Tier::HBF, .op = write ? Op::Write : Op::Read,
            .address_space = AddressSpace::Logical, .arrival_ns = arrival,
            .addr = (random() % 288) * 4096 + (write ? 0 : 64),
            .bytes = write ? (i % 5 == 0 ? 128u : 4096u) : 16u * 4096 - 128};
        if (!write && mode == hbfsim::host::MappingMode::Cached) {
            // Cover both physical blocks of the requested mapping group,
            // including live compact survivors of the reclaiming victim.
            request.addr = (request.addr / (64 * 4096) % 4) * (64 * 4096) + 64;
            request.bytes = 64 * 4096 - 128;
        }
        const auto before = grouped.read_engine_stats();
        const auto before_gc = grouped.execution_stats().gc_relocations;
        const auto before_dirty = grouped.execution_stats().mapping_cache_dirty_evictions;
        const auto a = grouped.issue(request);
        request.trace.mode = TraceMode::Full;
        const auto b = traced.issue(request);
        compare(a, b);
        const auto& after = grouped.execution_stats();
        const bool read_overrode_candidate = !write &&
            after.compact_read_run_overrides > before.compact_read_run_overrides &&
            after.gc_relocations > before_gc &&
            after.mapping_cache_dirty_evictions > before_dirty;
        read_eviction_retired_run_candidate |= read_overrode_candidate;
        if (read_overrode_candidate)
            std::cout << "read mapping-eviction GC: request=" << i
                << " copies=" << after.gc_relocations - before_gc
                << " dirty_evictions=" << after.mapping_cache_dirty_evictions - before_dirty
                << " candidate_overrides="
                << after.compact_read_run_overrides - before.compact_read_run_overrides << '\n';
        arrival = i % 5 == 4 ? a.finish_ns : arrival + 0.25;
    }
    const auto a = grouped.drain_pending("hot-gc-final", arrival);
    const auto b = traced.drain_pending("hot-gc-final", arrival, {.mode = TraceMode::Full});
    compare(a, b);
    const auto grouped_before = grouped.execution_stats().read_pipeline_group_pages;
    PhysicalRequest after_gc{.id = "post-gc-range", .tier = Tier::HBF, .op = Op::Read,
        .address_space = AddressSpace::Logical, .arrival_ns = a.finish_ns,
        .addr = 1, .bytes = 32 * 4096 - 2};
    const auto post_gc = grouped.issue(after_gc);
    after_gc.trace.mode = TraceMode::Full;
    compare(post_gc, traced.issue(after_gc));
    require(grouped.execution_stats().read_pipeline_group_pages > grouped_before,
        "read grouping never resumed after GC/program transitions drained");
    const auto& stats = grouped.stats();
    require(stats.gc_runs > 0 && stats.gc_relocations > 0,
        "hot-GC differential did not reclaim partially live blocks");
    require(stats.thermal_throttled_media_ops > 0 && stats.thermal_throttle_wait_ns > 0,
        "hot-GC differential never entered thermal pacing");
    require(stats.read_pipeline_group_pages > 0 &&
        traced.execution_stats().read_pipeline_group_pages == 0,
        "hot-GC differential did not compare both read schedulers");
    require(stats.accounting_verified && traced.stats().accounting_verified &&
        grouped.quiescence_stats().quiescent() && traced.quiescence_stats().quiescent(),
        "hot-GC differential left invalid or pending media state");
    // Compare the complete canonical persistent image, including mappings,
    // ownership, block epochs/wear and allocator continuation state.
    const auto stem = std::filesystem::temp_directory_path() /
        ("hbfsim-read-group-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto left = stem.string() + "-grouped.image", right = stem.string() + "-traced.image";
    hbfsim::host::write_persistent_image_file(left, grouped.persistent_image());
    hbfsim::host::write_persistent_image_file(right, traced.persistent_image());
    std::ifstream first(left), second(right);
    const std::string first_image((std::istreambuf_iterator<char>(first)), {});
    const std::string second_image((std::istreambuf_iterator<char>(second)), {});
    first.close(); second.close();
    std::filesystem::remove(left); std::filesystem::remove(right);
    require(!first_image.empty() && first_image == second_image,
        "group changes the final persistent state after hot GC");
    std::cout << "grouped hot-GC: mode=" << static_cast<unsigned>(mode)
        << " address_overrides=" << stats.compact_read_run_overrides
        << " dirty_evictions=" << stats.mapping_cache_dirty_evictions
        << " read_eviction_override=" << read_eviction_retired_run_candidate
        << " runs=" << stats.gc_runs << " copies=" << stats.gc_relocations
        << " paced_ops=" << stats.thermal_throttled_media_ops
        << " throttle_wait_ns=" << stats.thermal_throttle_wait_ns << '\n';
    if (mode == hbfsim::host::MappingMode::Cached)
        require(read_eviction_retired_run_candidate,
            "no read-triggered dirty mapping eviction/GC overrode a compact run candidate");
  }
}
int main() {
    try { transport_and_order(); bank_cache(); pending_cache_banks_match_isolated_banks(); host_lifecycle(); preerased_blocks(); hbm4_baseline(); independent_stack_payloads(); fragment_protocol(); channel_fragment_batch(); read_groups_match_traced_schedule(); ordered_read_rounds_match_traced_schedule(); compact_read_runs_preserve_mapping_boundaries(); read_groups_cover_gc_and_hot_pacing(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    std::cout << "OCP transport, fragment writes, bank cache, host lifecycle and HBM4 baseline passed\n";
}
