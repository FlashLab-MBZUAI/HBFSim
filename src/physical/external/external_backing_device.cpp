#include "physical/external/external_backing_device.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hbfsim::physical::external {
namespace {

std::uint64_t checked_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs + rhs;
}

void validate_positive_finite(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(
            std::string(name) + " must be positive and finite");
    }
}

void validate_nonnegative_finite(double value, const char* name) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error(
            std::string(name) + " must be non-negative and finite");
    }
}

double resource_span(double first_busy_ns, double last_busy_ns) {
    return std::isfinite(first_busy_ns) && last_busy_ns > first_busy_ns ?
        last_busy_ns - first_busy_ns : 0.0;
}

void observe_busy_interval(
    double start_ns,
    double finish_ns,
    double& first_busy_ns,
    double& last_busy_ns) {
    if (finish_ns <= start_ns) {
        return;
    }
    first_busy_ns = std::min(first_busy_ns, start_ns);
    last_busy_ns = std::max(last_busy_ns, finish_ns);
}

std::string traffic_detail(
    std::uint64_t wire_bytes,
    std::uint64_t payload_bytes,
    std::uint64_t protocol_bytes) {
    return std::to_string(wire_bytes) + "B wire (" +
        std::to_string(payload_bytes) + "B payload+" +
        std::to_string(protocol_bytes) + "B protocol)";
}

} // namespace

const char* to_string(ExternalBackingKind kind) {
    switch (kind) {
    case ExternalBackingKind::OnPackageLpddr:
        return "on-package-lpddr";
    case ExternalBackingKind::CxlMemory:
        return "cxl-memory";
    case ExternalBackingKind::NvmeSsd:
        return "nvme-ssd";
    }
    throw std::runtime_error("unknown external-backing kind");
}

ExternalBackingKind parse_external_backing_kind(const std::string& value) {
    if (value == "on-package-lpddr") {
        return ExternalBackingKind::OnPackageLpddr;
    }
    if (value == "cxl-memory") {
        return ExternalBackingKind::CxlMemory;
    }
    if (value == "nvme-ssd") {
        return ExternalBackingKind::NvmeSsd;
    }
    throw std::runtime_error(
        "external-backing-kind must be on-package-lpddr, cxl-memory, or "
        "nvme-ssd");
}

ExternalBackingConfig on_package_lpddr_profile() {
    auto config = ExternalBackingConfig{};
    config.kind = ExternalBackingKind::OnPackageLpddr;
    config.capacity_bytes = 2ull << 40; // 2 TiB
    config.media_channels = 16;
    config.max_outstanding_requests = 512;
    config.controller_issue_ns = 2.0;
    config.controller_processing_ns = 20.0;
    config.media_read_latency_ns = 100.0;
    config.media_write_latency_ns = 100.0;
    config.media_read_bandwidth_GBps = 2'000.0;
    config.media_write_bandwidth_GBps = 2'000.0;
    config.m2s_bandwidth_GBps = 2'000.0;
    config.s2m_bandwidth_GBps = 2'000.0;
    config.one_way_propagation_ns = 0.0;
    return config;
}

ExternalBackingConfig cxl_memory_profile() {
    return ExternalBackingConfig{};
}

ExternalBackingConfig nvme_ssd_profile() {
    auto config = ExternalBackingConfig{};
    config.kind = ExternalBackingKind::NvmeSsd;
    config.capacity_bytes = 4ull << 40; // 4 TiB
    config.media_channels = 4;
    config.max_outstanding_requests = 512;
    config.controller_issue_ns = 20.0;
    config.controller_processing_ns = 500.0;
    config.media_read_latency_ns = 80'000.0;
    config.media_write_latency_ns = 100'000.0;
    config.media_read_bandwidth_GBps = 14.0;
    config.media_write_bandwidth_GBps = 7.0;
    config.m2s_bandwidth_GBps = 16.0;
    config.s2m_bandwidth_GBps = 16.0;
    config.one_way_propagation_ns = 250.0;
    return config;
}

double ExternalBackingStats::active_span_ns() const {
    return std::isfinite(first_arrival_ns) && finish_ns > first_arrival_ns ?
        finish_ns - first_arrival_ns : 0.0;
}

double ExternalBackingStats::controller_active_span_ns() const {
    return resource_span(
        controller_first_busy_ns, controller_last_busy_ns);
}

double ExternalBackingStats::media_active_span_ns() const {
    return resource_span(media_first_busy_ns, media_last_busy_ns);
}

double ExternalBackingStats::m2s_active_span_ns() const {
    return resource_span(m2s_first_busy_ns, m2s_last_busy_ns);
}

double ExternalBackingStats::s2m_active_span_ns() const {
    return resource_span(s2m_first_busy_ns, s2m_last_busy_ns);
}

double ExternalBackingStats::controller_utilization() const {
    const auto span = controller_active_span_ns();
    return span <= 0.0 ? 0.0 : controller_issue_busy_ns / span;
}

double ExternalBackingStats::media_utilization() const {
    const auto span = media_active_span_ns();
    return span <= 0.0 || media_channels == 0 ? 0.0 :
        (media_read_busy_ns + media_write_busy_ns) /
            (span * static_cast<double>(media_channels));
}

double ExternalBackingStats::m2s_utilization() const {
    const auto span = m2s_active_span_ns();
    return span <= 0.0 ? 0.0 : m2s_busy_ns / span;
}

double ExternalBackingStats::s2m_utilization() const {
    const auto span = s2m_active_span_ns();
    return span <= 0.0 ? 0.0 : s2m_busy_ns / span;
}

ExternalBackingDevice::Reservation
ExternalBackingDevice::SerialResourceTimeline::reserve(
    double ready_ns,
    double busy_ns) {
    validate_nonnegative_finite(ready_ns, "resource ready time");
    validate_nonnegative_finite(busy_ns, "resource busy time");
    if (busy_ns == 0.0) {
        return Reservation{
            .start_ns = ready_ns,
            .finish_ns = ready_ns,
            .queue_wait_ns = 0.0,
            .busy_ns = 0.0,
        };
    }

    double start_ns = ready_ns;
    auto next = intervals_.upper_bound(start_ns);
    if (next != intervals_.begin()) {
        const auto previous = std::prev(next);
        if (previous->second > start_ns) {
            start_ns = previous->second;
        }
    }
    next = intervals_.lower_bound(start_ns);
    while (next != intervals_.end() &&
           start_ns + busy_ns > next->first) {
        start_ns = std::max(start_ns, next->second);
        next = intervals_.lower_bound(start_ns);
    }
    const auto finish_ns = start_ns + busy_ns;
    if (!std::isfinite(finish_ns)) {
        throw std::runtime_error(
            "serial-resource reservation exceeds finite time");
    }
    const auto inserted = intervals_.emplace(start_ns, finish_ns).second;
    if (!inserted) {
        throw std::runtime_error(
            "serial-resource calendar produced a duplicate interval");
    }
    return Reservation{
        .start_ns = start_ns,
        .finish_ns = finish_ns,
        .queue_wait_ns = start_ns - ready_ns,
        .busy_ns = busy_ns,
    };
}

ExternalBackingDevice::ExternalBackingDevice(
    ExternalBackingConfig config,
    AddressHeatmap* address_heatmap)
    : config_(std::move(config)),
      address_heatmap_(address_heatmap) {
    if (config_.capacity_bytes == 0 || config_.page_size_bytes == 0) {
        throw std::runtime_error(
            "external-backing capacity and page size must be positive");
    }
    if (config_.capacity_bytes % config_.page_size_bytes != 0) {
        throw std::runtime_error(
            "external-backing capacity must be page aligned");
    }
    if (config_.media_channels == 0 ||
        config_.max_outstanding_requests == 0) {
        throw std::runtime_error(
            "external-backing media channels and outstanding limit "
            "must be positive");
    }
    if (config_.command_bytes == 0 || config_.completion_bytes == 0) {
        throw std::runtime_error(
            "external-backing protocol records must be positive");
    }
    validate_nonnegative_finite(
        config_.controller_issue_ns,
        "external-backing controller issue interval");
    validate_nonnegative_finite(
        config_.controller_processing_ns,
        "external-backing controller processing latency");
    validate_nonnegative_finite(
        config_.media_read_latency_ns,
        "external-backing media read latency");
    validate_nonnegative_finite(
        config_.media_write_latency_ns,
        "external-backing media write latency");
    validate_positive_finite(
        config_.media_read_bandwidth_GBps,
        "external-backing media read bandwidth");
    validate_positive_finite(
        config_.media_write_bandwidth_GBps,
        "external-backing media write bandwidth");
    validate_positive_finite(
        config_.m2s_bandwidth_GBps,
        "external-backing M2S bandwidth");
    validate_positive_finite(
        config_.s2m_bandwidth_GBps,
        "external-backing S2M bandwidth");
    validate_nonnegative_finite(
        config_.one_way_propagation_ns,
        "external-backing one-way propagation latency");

    channels_.resize(config_.media_channels);
    stats_.kind = config_.kind;
    stats_.media_channels = config_.media_channels;
}

std::size_t ExternalBackingDevice::channel_for(std::uint64_t addr) const {
    return static_cast<std::size_t>(
        (addr / config_.page_size_bytes) % config_.media_channels);
}

double ExternalBackingDevice::admit(double arrival_ns) {
    while (!inflight_finishes_.empty() &&
           inflight_finishes_.top() <= arrival_ns) {
        inflight_finishes_.pop();
    }
    double admitted_ns = arrival_ns;
    while (inflight_finishes_.size() >=
           config_.max_outstanding_requests) {
        admitted_ns = inflight_finishes_.top();
        do {
            inflight_finishes_.pop();
        } while (!inflight_finishes_.empty() &&
                 inflight_finishes_.top() <= admitted_ns);
    }
    return admitted_ns;
}

ExternalBackingDevice::RequestTiming ExternalBackingDevice::schedule(
    const PhysicalRequest& request,
    ChannelState& channel) {
    RequestTiming timing;
    timing.admitted_ns = admit(request.arrival_ns);
    const bool read = request.op == Op::Read;
    timing.m2s_payload_bytes = read ? 0 : request.bytes;
    timing.m2s_protocol_bytes = config_.command_bytes;
    timing.m2s_wire_bytes = checked_add(
        timing.m2s_payload_bytes,
        timing.m2s_protocol_bytes,
        "external-backing M2S wire bytes");
    timing.s2m_payload_bytes = read ? request.bytes : 0;
    timing.s2m_protocol_bytes = config_.completion_bytes;
    timing.s2m_wire_bytes = checked_add(
        timing.s2m_payload_bytes,
        timing.s2m_protocol_bytes,
        "external-backing S2M wire bytes");

    timing.m2s = m2s_.reserve(
        timing.admitted_ns,
        transfer_time_ns(
            timing.m2s_wire_bytes, config_.m2s_bandwidth_GBps));
    timing.m2s_arrival_ns =
        timing.m2s.finish_ns + config_.one_way_propagation_ns;
    timing.controller = controller_.reserve(
        timing.m2s_arrival_ns, config_.controller_issue_ns);
    timing.controller_finish_ns =
        timing.controller.finish_ns + config_.controller_processing_ns;
    const auto media_latency_ns = read ?
        config_.media_read_latency_ns :
        config_.media_write_latency_ns;
    timing.media_latency_finish_ns =
        timing.controller_finish_ns + media_latency_ns;
    const auto aggregate_media_bandwidth = read ?
        config_.media_read_bandwidth_GBps :
        config_.media_write_bandwidth_GBps;
    timing.media = channel.media.reserve(
        timing.media_latency_finish_ns,
        transfer_time_ns(
            request.bytes,
            aggregate_media_bandwidth /
                static_cast<double>(config_.media_channels)));
    timing.s2m = s2m_.reserve(
        timing.media.finish_ns,
        transfer_time_ns(
            timing.s2m_wire_bytes, config_.s2m_bandwidth_GBps));
    timing.finish_ns =
        timing.s2m.finish_ns + config_.one_way_propagation_ns;
    if (!std::isfinite(timing.finish_ns)) {
        throw std::runtime_error(
            "external-backing completion exceeds finite time");
    }
    return timing;
}

void ExternalBackingDevice::validate_request(
    const PhysicalRequest& request) const {
    if (request.tier != Tier::External) {
        throw std::runtime_error(
            "ExternalBackingDevice received a non-external request");
    }
    if (request.op != Op::Read && request.op != Op::Write) {
        throw std::runtime_error(
            "external backing only supports read/write operations");
    }
    if (request.address_space != AddressSpace::Logical) {
        throw std::runtime_error(
            "external backing exposes one host-visible logical address space");
    }
    if (!std::isfinite(request.arrival_ns) || request.arrival_ns < 0.0) {
        throw std::runtime_error(
            "external-backing arrival must be finite and non-negative");
    }
    if (request.arrival_ns < last_issue_arrival_ns_) {
        throw std::runtime_error(
            "external-backing requests must be issued in causal time order");
    }
    if (request.bytes == 0) {
        throw std::runtime_error(
            "external-backing request bytes must be positive");
    }
    const auto bytes_remaining_in_page =
        config_.page_size_bytes -
        request.addr % config_.page_size_bytes;
    if (request.bytes > bytes_remaining_in_page) {
        throw std::runtime_error(
            "external-backing physical request crosses a page boundary");
    }
    const auto end = checked_add(
        request.addr,
        request.bytes,
        "external-backing request end");
    if (end > config_.capacity_bytes) {
        throw std::runtime_error(
            "external-backing request exceeds configured capacity");
    }
}

void ExternalBackingDevice::account(
    const PhysicalRequest& request,
    std::size_t channel_index,
    const RequestTiming& timing,
    PhysicalCompletion& completion) {
    const bool read = request.op == Op::Read;
    auto& channel = channels_.at(channel_index);
    if (!channel.active) {
        channel.active = true;
        ++stats_.active_media_channels;
    }

    auto& request_count = read ?
        stats_.read_requests : stats_.write_requests;
    auto& byte_count = read ? stats_.read_bytes : stats_.write_bytes;
    request_count = checked_add(
        request_count, 1, "external-backing request count");
    byte_count = checked_add(
        byte_count, request.bytes, "external-backing byte count");

    stats_.m2s_payload_bytes = checked_add(
        stats_.m2s_payload_bytes,
        timing.m2s_payload_bytes,
        "external-backing M2S payload bytes");
    stats_.m2s_protocol_bytes = checked_add(
        stats_.m2s_protocol_bytes,
        timing.m2s_protocol_bytes,
        "external-backing M2S protocol bytes");
    stats_.m2s_wire_bytes = checked_add(
        stats_.m2s_wire_bytes,
        timing.m2s_wire_bytes,
        "external-backing M2S wire bytes");
    stats_.s2m_payload_bytes = checked_add(
        stats_.s2m_payload_bytes,
        timing.s2m_payload_bytes,
        "external-backing S2M payload bytes");
    stats_.s2m_protocol_bytes = checked_add(
        stats_.s2m_protocol_bytes,
        timing.s2m_protocol_bytes,
        "external-backing S2M protocol bytes");
    stats_.s2m_wire_bytes = checked_add(
        stats_.s2m_wire_bytes,
        timing.s2m_wire_bytes,
        "external-backing S2M wire bytes");

    stats_.outstanding_wait_ns +=
        timing.admitted_ns - request.arrival_ns;
    stats_.m2s_queue_wait_ns += timing.m2s.queue_wait_ns;
    stats_.controller_queue_wait_ns += timing.controller.queue_wait_ns;
    stats_.media_queue_wait_ns += timing.media.queue_wait_ns;
    stats_.s2m_queue_wait_ns += timing.s2m.queue_wait_ns;
    stats_.controller_issue_busy_ns += timing.controller.busy_ns;
    stats_.controller_processing_work_ns +=
        config_.controller_processing_ns;
    if (read) {
        stats_.media_read_latency_work_ns +=
            config_.media_read_latency_ns;
        stats_.media_read_busy_ns += timing.media.busy_ns;
    } else {
        stats_.media_write_latency_work_ns +=
            config_.media_write_latency_ns;
        stats_.media_write_busy_ns += timing.media.busy_ns;
    }
    stats_.m2s_busy_ns += timing.m2s.busy_ns;
    stats_.s2m_busy_ns += timing.s2m.busy_ns;
    stats_.transport_propagation_work_ns +=
        2.0 * config_.one_way_propagation_ns;
    observe_busy_interval(
        timing.controller.start_ns,
        timing.controller.finish_ns,
        stats_.controller_first_busy_ns,
        stats_.controller_last_busy_ns);
    observe_busy_interval(
        timing.media.start_ns,
        timing.media.finish_ns,
        stats_.media_first_busy_ns,
        stats_.media_last_busy_ns);
    observe_busy_interval(
        timing.m2s.start_ns,
        timing.m2s.finish_ns,
        stats_.m2s_first_busy_ns,
        stats_.m2s_last_busy_ns);
    observe_busy_interval(
        timing.s2m.start_ns,
        timing.s2m.finish_ns,
        stats_.s2m_first_busy_ns,
        stats_.s2m_last_busy_ns);
    stats_.first_arrival_ns = std::min(
        stats_.first_arrival_ns, request.arrival_ns);
    stats_.finish_ns = std::max(stats_.finish_ns, timing.finish_ns);

    completion.breakdown.ingress_queue_wait_ns =
        timing.admitted_ns - request.arrival_ns;
    completion.breakdown.scheduler_queue_wait_ns =
        timing.m2s.queue_wait_ns +
        timing.controller.queue_wait_ns +
        timing.media.queue_wait_ns +
        timing.s2m.queue_wait_ns;
    completion.breakdown.command_ns =
        timing.controller.busy_ns + config_.controller_processing_ns;
    if (read) {
        completion.breakdown.array_read_ns =
            config_.media_read_latency_ns;
    } else {
        completion.breakdown.array_program_ns =
            config_.media_write_latency_ns;
    }
    completion.breakdown.channel_transfer_ns = timing.media.busy_ns;
    completion.breakdown.hb_io_transfer_ns =
        timing.m2s.busy_ns + timing.s2m.busy_ns;
    completion.breakdown.transport_latency_ns =
        2.0 * config_.one_way_propagation_ns;
    stats_.stage_work += completion.breakdown;

    auto* spans = trace_spans_enabled(request.trace) ?
        &completion.spans : nullptr;
    const auto prefix = read ? "external_read" : "external_write";
    add_trace_span(
        spans,
        prefix + std::string("_m2s_transfer"),
        "external_link",
        "external/m2s",
        timing.m2s.start_ns,
        timing.m2s.finish_ns,
        true,
        traffic_detail(
            timing.m2s_wire_bytes,
            timing.m2s_payload_bytes,
            timing.m2s_protocol_bytes));
    add_trace_span(
        spans,
        prefix + std::string("_m2s_propagation"),
        "external_link",
        "external/m2s",
        timing.m2s.finish_ns,
        timing.m2s_arrival_ns,
        true);
    add_trace_span(
        spans,
        prefix + std::string("_controller_issue"),
        "external_controller",
        "external/controller",
        timing.controller.start_ns,
        timing.controller.finish_ns,
        true);
    add_trace_span(
        spans,
        prefix + std::string("_controller_processing"),
        "external_controller",
        "external/controller",
        timing.controller.finish_ns,
        timing.controller_finish_ns,
        true);
    add_trace_span(
        spans,
        prefix + std::string("_media_latency"),
        "external_backing",
        completion.resource_path,
        timing.controller_finish_ns,
        timing.media_latency_finish_ns,
        true,
        std::to_string(request.bytes) + "B");
    add_trace_span(
        spans,
        prefix + std::string("_media_transfer"),
        "external_backing",
        completion.resource_path,
        timing.media.start_ns,
        timing.media.finish_ns,
        true,
        std::to_string(request.bytes) + "B");
    add_trace_span(
        spans,
        prefix + std::string("_s2m_transfer"),
        "external_link",
        "external/s2m",
        timing.s2m.start_ns,
        timing.s2m.finish_ns,
        true,
        traffic_detail(
            timing.s2m_wire_bytes,
            timing.s2m_payload_bytes,
            timing.s2m_protocol_bytes));
    add_trace_span(
        spans,
        prefix + std::string("_s2m_propagation"),
        "external_link",
        "external/s2m",
        timing.s2m.finish_ns,
        timing.finish_ns,
        true);

    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::ExternalPhysical,
            .direction = read ?
                TrafficDirection::Read : TrafficDirection::Write,
            .source = request.heatmap_source,
            .address = request.addr,
            .bytes = request.bytes,
        });
    }
}

PhysicalCompletion ExternalBackingDevice::issue(
    const PhysicalRequest& request) {
    validate_request(request);
    last_issue_arrival_ns_ = request.arrival_ns;
    const auto channel_index = channel_for(request.addr);
    auto& channel = channels_.at(channel_index);
    const auto timing = schedule(request, channel);
    inflight_finishes_.push(timing.finish_ns);
    stats_.max_device_outstanding = std::max<std::uint64_t>(
        stats_.max_device_outstanding,
        inflight_finishes_.size());

    PhysicalCompletion completion;
    completion.id = request.id;
    completion.tier = Tier::External;
    completion.op = request.op;
    completion.arrival_ns = request.arrival_ns;
    completion.start_ns = timing.admitted_ns;
    completion.finish_ns = timing.finish_ns;
    completion.logical_bytes = request.bytes;
    completion.physical_bytes = request.bytes;
    completion.resource_path =
        "external/" + std::string(to_string(config_.kind)) +
        "/media-channel" + std::to_string(channel_index);
    completion.note = request.op == Op::Read ?
        "external-backing-to-hbm" : "hbm-to-external-backing";
    account(request, channel_index, timing, completion);
    return completion;
}

} // namespace hbfsim::physical::external
