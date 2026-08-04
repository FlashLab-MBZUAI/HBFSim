#include "physical/external/external_backing_device.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

using hbfsim::physical::AddressSpace;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
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
    require(device.stats().active_media_channels == 2,
            "address striping did not activate both media channels");
}

void test_profiles_and_validation() {
    const auto lpddr =
        hbfsim::physical::external::on_package_lpddr_profile();
    const auto memory = hbfsim::physical::external::cxl_memory_profile();
    const auto ssd = hbfsim::physical::external::nvme_ssd_profile();
    require(lpddr.kind == ExternalBackingKind::OnPackageLpddr,
            "on-package LPDDR profile kind mismatch");
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
    require(
        hbfsim::physical::external::parse_external_backing_kind(
            "on-package-lpddr") == ExternalBackingKind::OnPackageLpddr,
        "on-package LPDDR kind did not round-trip");

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

int main() {
    try {
        test_read_and_write_protocol_pipeline();
        test_end_to_end_outstanding_backpressure();
        test_future_reservation_can_be_backfilled();
        test_profiles_and_validation();
        std::cout << "external_backing_device_test: PASS\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "external_backing_device_test: FAIL: "
                  << ex.what() << '\n';
        return 1;
    }
}
