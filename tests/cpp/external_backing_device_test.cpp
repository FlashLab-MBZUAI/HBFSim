#include "physical/external/external_backing_device.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using hbfsim::physical::AddressSpace;
using hbfsim::physical::AddressDomain;
using hbfsim::physical::AddressHeatmap;
using hbfsim::physical::AddressHeatmapConfig;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::physical::TraceMode;
using hbfsim::physical::external::DeviceCachePolicy;
using hbfsim::physical::external::ExternalBackingConfig;
using hbfsim::physical::external::ExternalBackingDevice;
using hbfsim::physical::external::ExternalBackingKind;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(double lhs, double rhs) {
    return std::abs(lhs - rhs) <=
        1e-9 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

PhysicalRequest request(
    std::string id,
    Op op,
    double arrival_ns,
    std::uint64_t addr = 0,
    std::uint64_t bytes = 4096) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = Tier::External,
        .op = op,
        .address_space = AddressSpace::Logical,
        .arrival_ns = arrival_ns,
        .addr = addr,
        .bytes = bytes,
    };
}

ExternalBackingConfig deterministic_config() {
    ExternalBackingConfig config;
    config.capacity_bytes = 16 * 4096;
    config.page_size_bytes = 4096;
    config.media_channels = 1;
    config.max_outstanding_requests = 1;
    config.controller_issue_ns = 10.0;
    config.controller_processing_ns = 20.0;
    config.media_read_latency_ns = 100.0;
    config.media_write_latency_ns = 200.0;
    config.media_read_bandwidth_GBps = 4.096;
    config.media_write_bandwidth_GBps = 2.048;
    config.m2s_bandwidth_GBps = 4.096;
    config.s2m_bandwidth_GBps = 4.096;
    config.one_way_propagation_ns = 50.0;
    config.command_bytes = 64;
    config.completion_bytes = 16;
    return config;
}

void test_contiguous_range_separates_ranges_segments_and_pages() {
    auto config = deterministic_config();
    config.capacity_bytes = 256 * 4096;
    config.request_segment_bytes = 4 * 4096;
    config.media_channels = 4;
    config.max_outstanding_requests = 1;

    AddressHeatmapConfig heatmap_config;
    heatmap_config.bin_count = 16;
    for (auto& domain : heatmap_config.domains) {
        domain.size_bytes = config.capacity_bytes;
    }
    heatmap_config.domains[static_cast<std::size_t>(
        AddressDomain::ExternalPhysical)].size_bytes =
        config.capacity_bytes;
    AddressHeatmap range_heatmap(heatmap_config);
    ExternalBackingDevice segmented(config, &range_heatmap);

    auto range = request(
        "unaligned-range",
        Op::Read,
        0.0,
        4096 + 17,
        37 * 4096 + 211);
    const auto actual = segmented.issue_contiguous_range(range);
    const auto end = range.addr + range.bytes;
    const auto expected_segments =
        (end - 1) / config.request_segment_bytes -
        range.addr / config.request_segment_bytes + 1;
    require(actual.finish_ns > actual.start_ns,
            "segmented range did not execute an end-to-end pipeline");

    const auto& segmented_stats = segmented.stats();
    require(
        segmented_stats.read_requests == expected_segments &&
            segmented_stats.read_bytes == range.bytes &&
            segmented_stats.page_run_requests == 1 &&
            segmented_stats.page_run_segments == expected_segments &&
            segmented_stats.page_run_pages == 38 &&
            segmented_stats.active_media_resources == 4 &&
            segmented_stats.m2s_protocol_bytes ==
                expected_segments * config.command_bytes &&
            segmented_stats.s2m_protocol_bytes ==
                expected_segments * config.completion_bytes,
        "range/transport/page coverage counters diverged");

    const auto& segmented_external = range_heatmap.domain(
        AddressDomain::ExternalPhysical);
    require(
        segmented_external.total.read_bytes == range.bytes &&
            segmented_external.total.read_accesses == expected_segments,
        "segmented range heatmap bytes or transport boundary diverged");

    // With one transport credit, all segments and the following scalar
    // request must remain causally serialized.
    const auto tail = request("tail", Op::Write, 1.0, 200 * 4096, 4096);
    const auto tail_completion = segmented.issue(tail);
    require(
        tail_completion.start_ns >= actual.finish_ns &&
            segmented.stats().read_requests == expected_segments &&
            segmented.stats().write_requests == 1,
        "segmented range did not preserve persistent credit state");
}

void test_production_range_uses_128k_transport_segments() {
    auto config = deterministic_config();
    config.capacity_bytes = 64ull << 20;
    config.request_segment_bytes = 128ull << 10;
    config.media_channels = 4;
    config.max_outstanding_requests = 16;
    ExternalBackingDevice device(config);
    const auto completion = device.issue_contiguous_range(request(
        "production-16m", Op::Read, 0.0, 0, 16ull << 20));
    const auto& stats = device.stats();
    require(
        completion.logical_bytes == (16ull << 20) &&
            stats.page_run_requests == 1 &&
            stats.page_run_segments == 128 &&
            stats.page_run_pages == 4096 &&
            stats.read_requests == 128 &&
            stats.read_bytes == (16ull << 20) &&
            stats.m2s_protocol_bytes == 128 * config.command_bytes &&
            stats.s2m_protocol_bytes ==
                128 * config.completion_bytes &&
            stats.active_media_resources == 4,
        "16 MiB range did not conserve range/segment/page layers");
}

void test_scalar_and_range_use_the_same_transport_channel_mapping() {
    auto config = deterministic_config();
    config.capacity_bytes = 64 * 4096;
    config.request_segment_bytes = 4 * 4096;
    config.media_channels = 4;
    config.max_outstanding_requests = 16;
    ExternalBackingDevice scalar(config);
    ExternalBackingDevice range(config);
    (void)scalar.issue(request("scalar-seed", Op::Read, 0.0, 0, 4096));
    (void)range.issue(request("range-seed", Op::Read, 0.0, 0, 4096));

    // Page 8 belongs to transport segment 2. A page-index mapper would send
    // it back to channel 0 and queue behind the seed, while the canonical
    // transport-segment mapper sends both APIs to channel 2.
    const auto scalar_completion = scalar.issue(request(
        "scalar-candidate", Op::Read, 0.0, 8 * 4096, 4096));
    const auto range_completion = range.issue_contiguous_range(request(
        "range-candidate", Op::Read, 0.0, 8 * 4096, 4096));
    require(
        close(scalar_completion.start_ns, range_completion.start_ns) &&
            close(scalar_completion.finish_ns, range_completion.finish_ns) &&
            scalar_completion.breakdown == range_completion.breakdown,
        "scalar/range transport-channel mapping diverged");
}

void test_directional_media_queue_round_robin_and_shared_timeline() {
    auto config = deterministic_config();
    config.media_read_queues = 2;
    config.media_write_queues = 1;
    config.media_channels = 1;
    config.max_outstanding_requests = 8;
    config.controller_issue_ns = 0.0;
    config.controller_processing_ns = 0.0;
    config.media_read_latency_ns = 0.0;
    config.media_write_latency_ns = 0.0;
    config.media_read_bandwidth_GBps = 8.192;
    config.media_write_bandwidth_GBps = 8.192;
    config.m2s_bandwidth_GBps = 1.0e12;
    config.s2m_bandwidth_GBps = 1.0e12;
    config.one_way_propagation_ns = 0.0;

    ExternalBackingDevice device(config);
    const auto read0 = device.issue(request("read0", Op::Read, 0.0));
    const auto read1 = device.issue(request("read1", Op::Read, 0.0));
    const auto read2 = device.issue(request("read2", Op::Read, 0.0));
    const auto write0 = device.issue(request("write0", Op::Write, 0.0));

    require(
        read0.resource_path ==
                "external/cxl-memory/media-queue0/channel0" &&
            read1.resource_path ==
                "external/cxl-memory/media-queue1/channel0" &&
            read2.resource_path ==
                "external/cxl-memory/media-queue0/channel0" &&
            write0.resource_path ==
                "external/cxl-memory/media-queue0/channel0",
        "directional queues did not round-robin onto shared resources");
    require(
        close(read0.breakdown.channel_transfer_ns, 1000.0) &&
            close(read1.breakdown.channel_transfer_ns, 1000.0) &&
            close(read2.breakdown.channel_transfer_ns, 1000.0) &&
            close(write0.breakdown.channel_transfer_ns, 500.0),
        "directional aggregate bandwidth was not divided by queue count");
    require(
        close(read0.finish_ns, read1.finish_ns) &&
            read2.finish_ns >= read0.finish_ns + 1000.0 - 1e-6 &&
            write0.finish_ns >= read2.finish_ns + 500.0 - 1e-6,
        "round-robin parallelism or cross-direction contention changed");

    const auto& stats = device.stats();
    require(
        stats.media_channels == 2 &&
            stats.media_channels_per_queue == 1 &&
            stats.media_read_queues == 2 &&
            stats.media_write_queues == 1 &&
            stats.active_media_resources == 2 &&
            close(stats.media_read_busy_ns, 3000.0) &&
            close(stats.media_write_busy_ns, 500.0),
        "resolved media queue geometry or busy work diverged");
}

void test_contiguous_range_pins_one_media_queue() {
    auto config = deterministic_config();
    config.capacity_bytes = 64 * 4096;
    config.request_segment_bytes = 2 * 4096;
    config.media_read_queues = 2;
    config.media_write_queues = 1;
    config.media_channels = 2;
    config.max_outstanding_requests = 16;
    ExternalBackingDevice device(config);

    auto first_request = request(
        "range0", Op::Read, 0.0, 0, 8 * 4096);
    first_request.trace.mode = TraceMode::Full;
    auto second_request = request(
        "range1", Op::Read, 0.0, 0, 8 * 4096);
    second_request.trace.mode = TraceMode::Full;
    const auto first = device.issue_contiguous_range(first_request);
    const auto second = device.issue_contiguous_range(second_request);

    const auto media_entities = [](const auto& completion) {
        std::set<std::string> result;
        for (const auto& span : completion.spans) {
            if (span.category == "external_backing") {
                result.insert(span.entity);
            }
        }
        return result;
    };
    require(
        media_entities(first) == std::set<std::string>{
            "external/cxl-memory/media-queue0/channel0",
            "external/cxl-memory/media-queue0/channel1"} &&
            media_entities(second) == std::set<std::string>{
                "external/cxl-memory/media-queue1/channel0",
                "external/cxl-memory/media-queue1/channel1"},
        "segments from one caller range escaped their selected media queue");
    const auto& stats = device.stats();
    require(
        stats.page_run_requests == 2 &&
            stats.page_run_segments == 8 &&
            stats.page_run_pages == 16 &&
            stats.media_channels == 4 &&
            stats.media_channels_per_queue == 2 &&
            stats.active_media_resources == 4,
        "pinned ranges did not conserve queue/segment/page accounting");
}

void test_read_and_write_protocol_pipeline() {
    ExternalBackingDevice reads(deterministic_config());
    const auto read = reads.issue(request("read", Op::Read, 0.0));
    // 64 B command / 4.096 GB/s + 50 ns propagation + 10 ns controller
    // issue + 20 ns processing + 100 ns media latency + 4096 B media /
    // 4.096 GB/s + (4096 B payload + 16 B completion) / 4.096 GB/s +
    // 50 ns propagation.
    require(close(read.finish_ns, 2249.53125),
            "read protocol stage timing changed");
    require(close(read.breakdown.command_ns, 30.0),
            "read controller work was not attributed");
    require(close(read.breakdown.array_read_ns, 100.0),
            "read media latency was not attributed");
    require(close(read.breakdown.channel_transfer_ns, 1000.0),
            "read media serialization was not attributed");
    require(close(read.breakdown.hb_io_transfer_ns, 1019.53125),
            "read directional transport work was not attributed");
    require(close(read.breakdown.transport_latency_ns, 100.0),
            "read propagation work was not attributed");
    const auto& read_stats = reads.stats();
    require(
        read_stats.m2s_payload_bytes == 0 &&
            read_stats.m2s_protocol_bytes == 64 &&
            read_stats.m2s_wire_bytes == 64 &&
            read_stats.s2m_payload_bytes == 4096 &&
            read_stats.s2m_protocol_bytes == 16 &&
            read_stats.s2m_wire_bytes == 4112,
        "read payload/protocol/wire accounting changed");
    require(close(read_stats.m2s_utilization(), 1.0) &&
                close(read_stats.s2m_utilization(), 1.0) &&
                close(read_stats.controller_utilization(), 1.0) &&
                close(read_stats.media_utilization(), 1.0),
            "single-request resource-local utilization must be one");

    ExternalBackingDevice writes(deterministic_config());
    const auto write = writes.issue(request("write", Op::Write, 0.0));
    // (64 B command + 4096 B payload) / 4.096 GB/s + 50 ns + controller
    // + 200 ns media latency + 4096 B / 2.048 GB/s + 16 B completion /
    // 4.096 GB/s + 50 ns.
    require(close(write.finish_ns, 3349.53125),
            "write protocol stage timing changed");
    const auto& write_stats = writes.stats();
    require(
        write_stats.m2s_payload_bytes == 4096 &&
            write_stats.m2s_protocol_bytes == 64 &&
            write_stats.m2s_wire_bytes == 4160 &&
            write_stats.s2m_payload_bytes == 0 &&
            write_stats.s2m_protocol_bytes == 16 &&
            write_stats.s2m_wire_bytes == 16,
        "write payload/protocol/wire accounting changed");
    require(write.finish_ns > 3295.625,
            "write acknowledged before media and returned completion");
}

void test_end_to_end_outstanding_backpressure() {
    ExternalBackingDevice device(deterministic_config());
    const auto first = device.issue(request("first", Op::Read, 0.0));
    const auto second = device.issue(request("second", Op::Read, 0.0));
    require(second.start_ns >= first.finish_ns,
            "outstanding credit did not hold the second request");
    require(close(device.stats().outstanding_wait_ns, first.finish_ns),
            "outstanding wait accounting changed");
    require(device.stats().max_device_outstanding == 1,
            "device exceeded its configured outstanding limit");
}

void test_future_reservation_can_be_backfilled() {
    auto config = deterministic_config();
    config.media_channels = 2;
    config.max_outstanding_requests = 8;
    config.controller_issue_ns = 0.0;
    config.controller_processing_ns = 0.0;
    config.media_read_latency_ns = 0.0;
    config.media_write_latency_ns = 1000.0;
    config.media_read_bandwidth_GBps = 4096.0;
    config.media_write_bandwidth_GBps = 4096.0;
    config.m2s_bandwidth_GBps = 4096.0;
    config.s2m_bandwidth_GBps = 4096.0;
    config.one_way_propagation_ns = 0.0;

    ExternalBackingDevice device(config);
    // The first API call reserves a far-future S2M completion. The second
    // request reaches S2M much earlier on another media channel and must fill
    // the earlier gap rather than queue behind call order.
    const auto slow_write =
        device.issue(request("slow-write", Op::Write, 0.0, 0));
    const auto fast_read =
        device.issue(request("fast-read", Op::Read, 0.0, 4096));
    require(fast_read.finish_ns < slow_write.finish_ns,
            "future S2M reservation created API-order head-of-line blocking");
    require(device.stats().active_media_resources == 2,
            "address striping did not activate both media channels");
}

void test_profiles_and_validation() {
    const auto lpddr =
        hbfsim::physical::external::on_package_lpddr_profile();
    const auto host_dram =
        hbfsim::physical::external::host_dram_profile();
    const auto memory = hbfsim::physical::external::cxl_memory_profile();
    const auto ssd = hbfsim::physical::external::nvme_ssd_profile();
    require(lpddr.kind == ExternalBackingKind::OnPackageLpddr,
            "on-package LPDDR profile kind mismatch");
    require(host_dram.kind == ExternalBackingKind::HostDram,
            "host DRAM profile kind mismatch");
    require(memory.kind == ExternalBackingKind::CxlMemory,
            "CXL memory profile kind mismatch");
    require(ssd.kind == ExternalBackingKind::NvmeSsd,
            "NVMe SSD profile kind mismatch");
    require(ssd.media_read_latency_ns > memory.media_read_latency_ns,
            "NVMe profile must expose its higher fixed read latency");
    require(ssd.media_read_bandwidth_GBps <
                memory.media_read_bandwidth_GBps,
            "NVMe profile must expose its lower media bandwidth");
    require(lpddr.media_read_bandwidth_GBps >
                memory.media_read_bandwidth_GBps,
            "on-package LPDDR profile must expose its higher bandwidth");
    const auto cxl_ssd = hbfsim::physical::external::cxl_ssd_profile();
    require(cxl_ssd.kind == ExternalBackingKind::CxlSsd,
            "CXL-SSD profile kind mismatch");
    require(cxl_ssd.media_read_latency_ns > memory.media_read_latency_ns &&
                cxl_ssd.media_read_latency_ns < ssd.media_read_latency_ns,
            "CXL-SSD read latency must sit between CXL memory and NVMe SSD");
    require(cxl_ssd.media_write_latency_ns > cxl_ssd.media_read_latency_ns,
            "CXL-SSD program latency must exceed its read latency");
    require(cxl_ssd.m2s_bandwidth_GBps == memory.m2s_bandwidth_GBps &&
                cxl_ssd.s2m_bandwidth_GBps == memory.s2m_bandwidth_GBps &&
                cxl_ssd.one_way_propagation_ns ==
                    memory.one_way_propagation_ns,
            "CXL-SSD must reuse the CXL link envelope");
    require(
        hbfsim::physical::external::parse_external_backing_kind(
            "on-package-lpddr") == ExternalBackingKind::OnPackageLpddr,
        "on-package LPDDR kind did not round-trip");
    require(
        hbfsim::physical::external::parse_external_backing_kind(
            "host-dram") == ExternalBackingKind::HostDram &&
        std::string(hbfsim::physical::external::to_string(host_dram.kind)) ==
            "host-dram",
        "host DRAM kind did not round-trip");
    require(
        hbfsim::physical::external::parse_external_backing_kind(
            "cxl-ssd") == ExternalBackingKind::CxlSsd &&
        std::string(hbfsim::physical::external::to_string(cxl_ssd.kind)) ==
            "cxl-ssd",
        "CXL-SSD kind did not round-trip");

    bool rejected = false;
    try {
        (void)hbfsim::physical::external::parse_external_backing_kind("dram");
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "non-canonical external-backing kind was accepted");

    auto invalid_config = deterministic_config();
    invalid_config.media_channels = 0;
    rejected = false;
    try {
        (void)ExternalBackingDevice(invalid_config);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "zero-channel external backing was accepted");

    invalid_config = deterministic_config();
    invalid_config.media_read_queues = 0;
    rejected = false;
    try {
        (void)ExternalBackingDevice(invalid_config);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "zero read-queue external backing was accepted");

    invalid_config = deterministic_config();
    invalid_config.media_write_queues = 0;
    rejected = false;
    try {
        (void)ExternalBackingDevice(invalid_config);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "zero write-queue external backing was accepted");

    invalid_config = deterministic_config();
    invalid_config.command_bytes = 0;
    rejected = false;
    try {
        (void)ExternalBackingDevice(invalid_config);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "zero-byte protocol command was accepted");

    invalid_config = deterministic_config();
    invalid_config.capacity_bytes -= 1;
    rejected = false;
    try {
        (void)ExternalBackingDevice(invalid_config);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "unaligned external-backing capacity was accepted");

    invalid_config = deterministic_config();
    invalid_config.request_segment_bytes = 3 * 4096 + 1;
    rejected = false;
    try {
        (void)ExternalBackingDevice(invalid_config);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "non-page-aligned transport segment was accepted");

    ExternalBackingDevice device(deterministic_config());
    rejected = false;
    try {
        (void)device.issue(request(
            "overflow", Op::Read, 0.0, 16 * 4096, 1));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "out-of-capacity request was accepted");

    ExternalBackingDevice page_device(deterministic_config());
    rejected = false;
    try {
        (void)page_device.issue(request(
            "cross-page", Op::Read, 0.0, 4095, 2));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "cross-page physical request was accepted");

    (void)device.issue(request("causal", Op::Read, 10.0));
    rejected = false;
    try {
        (void)device.issue(request("time-travel", Op::Read, 9.0));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "time-traveling request was accepted");
}

} // namespace

ExternalBackingConfig cached_cxl_ssd_config() {
    // Clean arithmetic: negligible link time, zero controller time, 1000 ns
    // of channel occupancy per 4 KiB segment, 100 ns cache latency and
    // 100 ns of cache-port occupancy per 4 KiB segment.
    ExternalBackingConfig config;
    config.kind = ExternalBackingKind::CxlSsd;
    config.capacity_bytes = 64 * 4096;
    config.page_size_bytes = 4096;
    config.request_segment_bytes = 4096;
    config.media_channels = 1;
    config.media_read_queues = 1;
    config.media_write_queues = 1;
    config.max_outstanding_requests = 16;
    config.controller_issue_ns = 0.0;
    config.controller_processing_ns = 0.0;
    config.media_read_latency_ns = 40'000.0;
    config.media_write_latency_ns = 200'000.0;
    config.media_read_bandwidth_GBps = 4.096;
    config.media_write_bandwidth_GBps = 4.096;
    config.m2s_bandwidth_GBps = 1'000'000.0;
    config.s2m_bandwidth_GBps = 1'000'000.0;
    config.one_way_propagation_ns = 0.0;
    config.device_cache.enabled = true;
    config.device_cache.capacity_bytes = 2 * 4096;
    config.device_cache.ways = 0;
    config.device_cache.policy = DeviceCachePolicy::Fifo;
    config.device_cache.hit_latency_ns = 100.0;
    config.device_cache.hit_bandwidth_GBps = 40.96;
    return config;
}

void test_device_cache_hit_miss_and_writeback() {
    ExternalBackingDevice device(cached_cxl_ssd_config());

    // Demand read miss pays the flash stage: 40 us latency + 1 us channel.
    const auto miss = device.issue(request("r-miss", Op::Read, 0.0, 0));
    require(miss.finish_ns > 40'000.0 && miss.finish_ns < 42'000.0,
            "cache read miss must pay the flash read stage");
    require(miss.resource_path ==
                "external/cxl-ssd/media-queue0/channel0",
            "cache read miss must report the flash media resource");

    // Re-read of the same segment is served by the buffer DRAM.
    const auto hit = device.issue(request("r-hit", Op::Read, 50'000.0, 0));
    require(hit.finish_ns > 50'000.0 && hit.finish_ns < 51'000.0,
            "cache read hit must be served at buffer-DRAM speed");
    require(hit.resource_path == "external/cxl-ssd/device-cache",
            "cache read hit must report the device-cache resource");

    // Writes absorb into the buffer instead of paying the 200 us program.
    const auto write_a = device.issue(
        request("w-a", Op::Write, 100'000.0, 4096));
    require(write_a.finish_ns > 100'000.0 &&
                write_a.finish_ns < 101'000.0,
            "cache write must absorb at buffer-DRAM speed");

    // Evicting the clean FIFO head produces no flush and no gate.
    const auto write_b = device.issue(
        request("w-b", Op::Write, 200'000.0, 8192));
    require(write_b.finish_ns > 200'000.0 &&
                write_b.finish_ns < 201'000.0,
            "clean eviction must not gate the evicting write");
    require(device.stats().cache_writeback_segments == 0,
            "clean eviction must not flush");

    // Evicting a dirty line flushes it through the flash write path and
    // gates the evicting access on the flush reservation.
    const auto write_c = device.issue(
        request("w-c", Op::Write, 300'000.0, 12288));
    require(write_c.finish_ns > 501'000.0 &&
                write_c.finish_ns < 502'500.0,
            "dirty eviction must gate on the 200 us flush");

    const auto& stats = device.stats();
    require(stats.cache_read_hits == 1 && stats.cache_read_misses == 1,
            "cache read census mismatch");
    require(stats.cache_write_hits == 0 && stats.cache_write_misses == 3,
            "cache write census mismatch");
    require(stats.cache_writeback_segments == 1 &&
                stats.cache_writeback_bytes == 4096,
            "dirty flush census mismatch");
    require(stats.cache_writeback_gate_wait_ns > 200'000.0,
            "flush gate wait must be visible in the census");
    require(close(stats.cache_busy_ns, 400.0),
            "cache port busy must cover the four cache-served segments");
    require(close(stats.media_read_busy_ns, 1'000.0),
            "only the demand miss may occupy the read media path");
    require(close(stats.cache_flush_busy_ns, 1'000.0),
            "the dirty flush must occupy the write media path internally");
    require(stats.read_bytes == 2 * 4096 && stats.write_bytes == 3 * 4096,
            "caller byte census must exclude internal flush traffic");
}

void test_device_cache_prefetch() {
    auto config = cached_cxl_ssd_config();
    config.device_cache.prefetch_degree = 1;
    config.device_cache.prefetch_stride = 1;
    ExternalBackingDevice device(config);

    const auto miss = device.issue(request("r0", Op::Read, 0.0, 0));
    require(miss.finish_ns > 40'000.0,
            "prefetching read miss still pays the flash stage");
    require(device.stats().cache_prefetch_segments == 1 &&
                device.stats().cache_prefetch_bytes == 4096,
            "read miss must prefetch the next segment");
    const auto hit = device.issue(request("r1", Op::Read, 50'000.0, 4096));
    require(hit.finish_ns < 51'000.0,
            "prefetched segment must hit at buffer-DRAM speed");
    require(device.stats().cache_read_hits == 1,
            "prefetched segment hit census mismatch");
}

void test_device_cache_clock_second_chance() {
    auto config = cached_cxl_ssd_config();
    config.device_cache.policy = DeviceCachePolicy::Clock;
    ExternalBackingDevice device(config);

    (void)device.issue(request("a0", Op::Read, 0.0, 0));
    (void)device.issue(request("b0", Op::Read, 1'000'000.0, 4096));
    (void)device.issue(request("a1", Op::Read, 2'000'000.0, 0));
    // The referenced front line (A) gets a second chance, so B is evicted.
    (void)device.issue(request("c0", Op::Read, 3'000'000.0, 8192));
    const auto again = device.issue(
        request("a2", Op::Read, 4'000'000.0, 0));
    require(again.finish_ns < 4'001'000.0,
            "CLOCK must give the referenced line a second chance");
    require(device.stats().cache_read_hits == 2,
            "CLOCK hit census mismatch");
}

void test_device_cache_s3fifo_ghost_readmission() {
    auto config = cached_cxl_ssd_config();
    config.device_cache.policy = DeviceCachePolicy::S3Fifo;
    ExternalBackingDevice device(config);

    (void)device.issue(request("a0", Op::Read, 0.0, 0));
    (void)device.issue(request("b0", Op::Read, 1'000'000.0, 4096));
    // Quick demotion drops unreferenced A to the ghost history.
    (void)device.issue(request("c0", Op::Read, 2'000'000.0, 8192));
    // Ghost readmission installs A into the protected main queue.
    (void)device.issue(request("a1", Op::Read, 3'000'000.0, 0));
    const auto again = device.issue(
        request("a2", Op::Read, 4'000'000.0, 0));
    require(again.finish_ns < 4'001'000.0,
            "ghost-readmitted line must stay resident in main");
    require(device.stats().cache_read_hits == 1,
            "S3-FIFO ghost readmission census mismatch");
}

// The directory marks a line present when the miss is scheduled, but its
// data lands only when the flash read completes. Every hit before that
// instant waits for the fill, not just the first one.
void test_device_cache_hit_under_miss_waits_for_the_fill() {
    ExternalBackingDevice device(cached_cxl_ssd_config());
    const auto miss = device.issue(request("r-miss", Op::Read, 0.0, 0));
    require(miss.finish_ns > 40'000.0,
            "demand miss must pay the flash read stage");
    const auto early = device.issue(request("r-early", Op::Read, 1'000.0, 0));
    require(early.finish_ns >= miss.finish_ns,
            "a hit under the demand miss completed before the fill landed");
    require(early.resource_path == "external/cxl-ssd/device-cache",
            "the hit under miss must still be served by the buffer");
    const auto second = device.issue(request("r-second", Op::Read, 2'000.0, 0));
    require(second.finish_ns >= miss.finish_ns,
            "a second hit under the same miss ran ahead of the fill");
    const auto late = device.issue(request("r-late", Op::Read, 60'000.0, 0));
    require(late.finish_ns < 61'000.0,
            "a hit after the fill landed must not wait");
    require(device.stats().cache_read_hits == 3 &&
                device.stats().cache_read_misses == 1,
            "hit-under-miss census mismatch");
}

// A write-allocated line carries the caller's payload only once the buffer
// write has landed; a read hit before that waits for it, and never fabricates
// a flash read.
void test_device_cache_read_after_write_waits_for_the_buffer_write() {
    ExternalBackingDevice device(cached_cxl_ssd_config());
    const auto write = device.issue(request("w", Op::Write, 0.0, 0));
    const auto read = device.issue(request("r", Op::Read, 10.0, 0));
    require(read.finish_ns >= write.finish_ns,
            "read hit completed before the write it depends on landed");
    require(read.resource_path == "external/cxl-ssd/device-cache",
            "read after write must be a buffer hit");
    require(device.stats().media_read_busy_ns == 0.0,
            "read after write must not touch flash");
}

// A write miss that covers only part of a flash page cannot invent the
// untouched bytes. It fetches the old page before merging the caller payload;
// a full-page write miss remains eligible for allocation without that read.
void test_device_cache_partial_write_miss_reads_for_ownership() {
    ExternalBackingDevice partial(cached_cxl_ssd_config());
    const auto partial_write = partial.issue(
        request("partial", Op::Write, 0.0, 128, 512));
    require(partial_write.finish_ns > 40'000.0,
            "sub-page write miss did not fetch the untouched flash bytes");
    const auto& partial_stats = partial.stats();
    require(
        partial_stats.cache_read_for_ownership_segments == 1 &&
            partial_stats.cache_read_for_ownership_bytes == 4096 &&
            close(partial_stats.media_read_busy_ns, 1'000.0),
        "sub-page write read-for-ownership accounting diverged");

    ExternalBackingDevice full(cached_cxl_ssd_config());
    const auto full_write = full.issue(
        request("full", Op::Write, 0.0, 0, 4096));
    require(full_write.finish_ns < 1'000.0,
            "full-page write miss unnecessarily read flash");
    require(
        full.stats().cache_read_for_ownership_segments == 0 &&
            full.stats().media_read_busy_ns == 0.0,
        "full-page write miss reported a read-for-ownership");
}

// Completed fills are architectural cache lines, not live timing
// dependencies. A long sequential stream must retain only the still-in-flight
// timestamp instead of one map entry per line for the life of the device.
void test_device_cache_completed_fill_dependencies_expire() {
    auto config = cached_cxl_ssd_config();
    config.device_cache.capacity_bytes = 64 * 4096;
    ExternalBackingDevice device(config);
    for (std::uint64_t segment = 0; segment < 64; ++segment) {
        (void)device.issue(request(
            "fill-" + std::to_string(segment),
            Op::Read,
            static_cast<double>(segment) * 100'000.0,
            segment * 4096));
        require(
            device.cache_dependency_state().pending_fill_segments <= 1,
            "completed cache-fill dependencies grew with stream history");
    }
}

// A dirty flush is a flash program in flight; a later miss on the same line
// must not read flash or reinstall before that flush lands.
void test_device_cache_miss_after_dirty_flush_waits_for_the_flush() {
    ExternalBackingDevice device(cached_cxl_ssd_config());
    // Two-line FIFO: dirty A, then B and C evict A with a 200 us flush.
    (void)device.issue(request("w-a", Op::Write, 0.0, 0));
    (void)device.issue(request("w-b", Op::Write, 1'000.0, 4096));
    const auto evicting = device.issue(
        request("w-c", Op::Write, 2'000.0, 8192));
    require(evicting.finish_ns > 200'000.0,
            "dirty eviction must gate on the flush");
    // A is now absent: re-reading it is a miss that reads flash. The flash
    // holds A's new contents only once the flush has landed.
    const auto reread = device.issue(request("r-a", Op::Read, 3'000.0, 0));
    require(reread.finish_ns >= evicting.finish_ns + 40'000.0,
            "re-read of a flushing line ran ahead of its flush");
    require(device.stats().cache_writeback_segments == 2,
            "re-reading A evicted another dirty line; census mismatch");
}

// A dirty victim's flush reads the line from the buffer, so it cannot start
// before the fill that installed the line (here a read-for-ownership plus
// the buffer write) has landed, even when the eviction is scheduled earlier.
void test_device_cache_dirty_victim_flush_waits_for_its_own_fill() {
    ExternalBackingDevice device(cached_cxl_ssd_config());
    // A: sub-page write miss; its content exists only after the 40 us
    // read-for-ownership and the buffer write.
    const auto partial = device.issue(
        request("w-a", Op::Write, 0.0, 128, 512));
    require(partial.finish_ns > 40'000.0,
            "sub-page write miss must read for ownership");
    // B fills the second line immediately; C then evicts A at t = 20 ns.
    (void)device.issue(request("w-b", Op::Write, 10.0, 4096));
    const auto evicting = device.issue(
        request("w-c", Op::Write, 20.0, 8192));
    require(device.stats().cache_writeback_segments == 1,
            "evicting A must flush exactly one dirty line");
    require(evicting.finish_ns >= partial.finish_ns + 200'000.0,
            "dirty victim was flushed before its own fill landed");
}

// Internal flush and prefetch work occupies the channels: utilization
// counts it even though the caller-only busy fields do not.
void test_device_cache_internal_work_counts_toward_utilization() {
    ExternalBackingDevice device(cached_cxl_ssd_config());
    (void)device.issue(request("w-a", Op::Write, 0.0, 0));
    (void)device.issue(request("w-b", Op::Write, 1'000.0, 4096));
    (void)device.issue(request("w-c", Op::Write, 2'000.0, 8192));
    const auto& stats = device.stats();
    require(stats.media_write_busy_ns == 0.0,
            "caller-only media busy must stay zero for absorbed writes");
    require(close(stats.cache_flush_busy_ns, 1'000.0),
            "the flush must occupy the write media path internally");
    require(stats.media_utilization() > 0.0,
            "media utilization ignored the internal flush work");
}

// Cache lines are flash pages: a transport segment larger than one page
// cannot be a buffer line.
void test_device_cache_requires_page_sized_segments() {
    auto config = cached_cxl_ssd_config();
    config.request_segment_bytes = 8192;
    config.device_cache.capacity_bytes = 4 * 8192;
    bool rejected = false;
    try {
        ExternalBackingDevice device(config);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected,
            "a cache over multi-page segments must be rejected");
}

void test_device_cache_validation() {
    using hbfsim::physical::external::parse_device_cache_policy;
    require(parse_device_cache_policy("fifo") == DeviceCachePolicy::Fifo &&
            parse_device_cache_policy("lifo") == DeviceCachePolicy::Lifo &&
            parse_device_cache_policy("clock") ==
                DeviceCachePolicy::Clock &&
            parse_device_cache_policy("s3fifo") ==
                DeviceCachePolicy::S3Fifo,
            "device cache policy names did not round-trip");
    bool rejected = false;
    try {
        (void)parse_device_cache_policy("lru");
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "unknown device cache policy was accepted");

    auto wrong_kind = cached_cxl_ssd_config();
    wrong_kind.kind = ExternalBackingKind::CxlMemory;
    rejected = false;
    try {
        (void)ExternalBackingDevice(wrong_kind);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected,
            "device cache on a non-cxl-ssd kind was accepted");

    auto misaligned = cached_cxl_ssd_config();
    misaligned.device_cache.capacity_bytes = 4096 + 1;
    rejected = false;
    try {
        (void)ExternalBackingDevice(misaligned);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "misaligned device cache capacity was accepted");
}

int main() {
    try {
        test_read_and_write_protocol_pipeline();
        test_end_to_end_outstanding_backpressure();
        test_future_reservation_can_be_backfilled();
        test_contiguous_range_separates_ranges_segments_and_pages();
        test_production_range_uses_128k_transport_segments();
        test_scalar_and_range_use_the_same_transport_channel_mapping();
        test_directional_media_queue_round_robin_and_shared_timeline();
        test_contiguous_range_pins_one_media_queue();
        test_profiles_and_validation();
        test_device_cache_hit_miss_and_writeback();
        test_device_cache_prefetch();
        test_device_cache_clock_second_chance();
        test_device_cache_s3fifo_ghost_readmission();
        test_device_cache_hit_under_miss_waits_for_the_fill();
        test_device_cache_read_after_write_waits_for_the_buffer_write();
        test_device_cache_partial_write_miss_reads_for_ownership();
        test_device_cache_completed_fill_dependencies_expire();
        test_device_cache_miss_after_dirty_flush_waits_for_the_flush();
        test_device_cache_dirty_victim_flush_waits_for_its_own_fill();
        test_device_cache_internal_work_counts_toward_utilization();
        test_device_cache_requires_page_sized_segments();
        test_device_cache_validation();
        std::cout << "external_backing_device_test: PASS\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "external_backing_device_test: FAIL: "
                  << ex.what() << '\n';
        return 1;
    }
}
