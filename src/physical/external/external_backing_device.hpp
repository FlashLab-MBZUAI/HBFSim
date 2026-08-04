#pragma once

#include "physical/address_heatmap.hpp"
#include "physical/physical_types.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <string>
#include <vector>

namespace hbfsim::physical::external {

// Non-HBM backing tiers share one explicit request/controller/media/response
// contract. The full-duplex transport represents either an on-package D2D
// path or a host attachment as selected by the profile. Profiles declare
// parameter envelopes; they do not select a different scheduling
// implementation.
enum class ExternalBackingKind {
    OnPackageLpddr,
    CxlMemory,
    NvmeSsd,
};

[[nodiscard]] const char* to_string(ExternalBackingKind kind);
[[nodiscard]] ExternalBackingKind parse_external_backing_kind(
    const std::string& value);

struct ExternalBackingConfig {
    ExternalBackingKind kind = ExternalBackingKind::CxlMemory;
    std::uint64_t capacity_bytes = 256ull << 30; // 256 GiB
    // Host-visible transfer/striping granularity. Composition code must split
    // requests that cross this boundary.
    std::uint64_t page_size_bytes = 4096;

    // Address-striped media resources. Configured media bandwidth is aggregate
    // across the channels and is divided evenly between them.
    std::uint32_t media_channels = 8;
    // End-to-end device credits from admission through the returned response.
    std::uint32_t max_outstanding_requests = 512;

    // One pipelined controller issue resource followed by non-occupying
    // processing latency.
    double controller_issue_ns = 2.0;
    double controller_processing_ns = 20.0;

    double media_read_latency_ns = 90.0;
    double media_write_latency_ns = 90.0;
    double media_read_bandwidth_GBps = 204.8;
    double media_write_bandwidth_GBps = 204.8;

    // Full-duplex host transport. M2S means requester-to-device; S2M means
    // device-to-requester. Bandwidth is wire bandwidth. Propagation is paid
    // once after each directional transfer and is not resource occupancy.
    double m2s_bandwidth_GBps = 36.0;
    double s2m_bandwidth_GBps = 36.0;
    double one_way_propagation_ns = 75.0;

    // Per-request protocol bytes. A read sends only a command and returns
    // payload+completion. A write sends command+payload and returns completion.
    std::uint32_t command_bytes = 64;
    std::uint32_t completion_bytes = 16;
};

// Reproducible, source-anchored sensitivity envelopes. Fields not explicitly
// supported by a public source remain exploratory assumptions in the
// parameter-provenance registry.
[[nodiscard]] ExternalBackingConfig cxl_memory_profile();
[[nodiscard]] ExternalBackingConfig nvme_ssd_profile();
[[nodiscard]] ExternalBackingConfig on_package_lpddr_profile();

struct ExternalBackingStats {
    ExternalBackingKind kind = ExternalBackingKind::CxlMemory;
    std::uint64_t read_requests = 0;
    std::uint64_t write_requests = 0;
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;

    std::uint64_t media_channels = 0;
    std::uint64_t active_media_channels = 0;
    std::uint64_t max_device_outstanding = 0;
    double outstanding_wait_ns = 0.0;

    double controller_queue_wait_ns = 0.0;
    double controller_issue_busy_ns = 0.0;
    double controller_processing_work_ns = 0.0;

    double media_queue_wait_ns = 0.0;
    double media_read_latency_work_ns = 0.0;
    double media_write_latency_work_ns = 0.0;
    double media_read_busy_ns = 0.0;
    double media_write_busy_ns = 0.0;

    std::uint64_t m2s_payload_bytes = 0;
    std::uint64_t m2s_protocol_bytes = 0;
    std::uint64_t m2s_wire_bytes = 0;
    std::uint64_t s2m_payload_bytes = 0;
    std::uint64_t s2m_protocol_bytes = 0;
    std::uint64_t s2m_wire_bytes = 0;
    double m2s_queue_wait_ns = 0.0;
    double s2m_queue_wait_ns = 0.0;
    double m2s_busy_ns = 0.0;
    double s2m_busy_ns = 0.0;
    double transport_propagation_work_ns = 0.0;

    double first_arrival_ns = std::numeric_limits<double>::infinity();
    double finish_ns = 0.0;
    double controller_first_busy_ns =
        std::numeric_limits<double>::infinity();
    double controller_last_busy_ns = 0.0;
    double media_first_busy_ns = std::numeric_limits<double>::infinity();
    double media_last_busy_ns = 0.0;
    double m2s_first_busy_ns = std::numeric_limits<double>::infinity();
    double m2s_last_busy_ns = 0.0;
    double s2m_first_busy_ns = std::numeric_limits<double>::infinity();
    double s2m_last_busy_ns = 0.0;
    Breakdown stage_work;

    [[nodiscard]] double active_span_ns() const;
    [[nodiscard]] double controller_active_span_ns() const;
    [[nodiscard]] double media_active_span_ns() const;
    [[nodiscard]] double m2s_active_span_ns() const;
    [[nodiscard]] double s2m_active_span_ns() const;
    [[nodiscard]] double controller_utilization() const;
    [[nodiscard]] double media_utilization() const;
    [[nodiscard]] double m2s_utilization() const;
    [[nodiscard]] double s2m_utilization() const;
};

class ExternalBackingDevice {
public:
    explicit ExternalBackingDevice(
        ExternalBackingConfig config,
        AddressHeatmap* address_heatmap = nullptr);

    [[nodiscard]] PhysicalCompletion issue(const PhysicalRequest& request);

    void attach_address_heatmap(AddressHeatmap& address_heatmap) {
        address_heatmap_ = &address_heatmap;
    }

    [[nodiscard]] const ExternalBackingConfig& config() const {
        return config_;
    }
    [[nodiscard]] const ExternalBackingStats& stats() const {
        return stats_;
    }

private:
    struct Reservation {
        double start_ns = 0.0;
        double finish_ns = 0.0;
        double queue_wait_ns = 0.0;
        double busy_ns = 0.0;
    };

    // Future reservations are kept as disjoint intervals. reserve() chooses
    // the earliest legal gap at or after ready_ns, so a later API call whose
    // upstream work finishes earlier can backfill before an already-reserved
    // future interval.
    class SerialResourceTimeline {
    public:
        [[nodiscard]] Reservation reserve(
            double ready_ns,
            double busy_ns);

    private:
        std::map<double, double> intervals_;
    };

    struct ChannelState {
        SerialResourceTimeline media;
        bool active = false;
    };

    struct RequestTiming {
        double admitted_ns = 0.0;
        Reservation m2s;
        double m2s_arrival_ns = 0.0;
        Reservation controller;
        double controller_finish_ns = 0.0;
        double media_latency_finish_ns = 0.0;
        Reservation media;
        Reservation s2m;
        double finish_ns = 0.0;
        std::uint64_t m2s_payload_bytes = 0;
        std::uint64_t m2s_protocol_bytes = 0;
        std::uint64_t m2s_wire_bytes = 0;
        std::uint64_t s2m_payload_bytes = 0;
        std::uint64_t s2m_protocol_bytes = 0;
        std::uint64_t s2m_wire_bytes = 0;
    };

    [[nodiscard]] std::size_t channel_for(std::uint64_t addr) const;
    [[nodiscard]] double admit(double arrival_ns);
    [[nodiscard]] RequestTiming schedule(
        const PhysicalRequest& request,
        ChannelState& channel);
    void validate_request(const PhysicalRequest& request) const;
    void account(
        const PhysicalRequest& request,
        std::size_t channel_index,
        const RequestTiming& timing,
        PhysicalCompletion& completion);

    ExternalBackingConfig config_;
    ExternalBackingStats stats_;
    std::vector<ChannelState> channels_;
    std::priority_queue<
        double,
        std::vector<double>,
        std::greater<double>> inflight_finishes_;
    SerialResourceTimeline controller_;
    SerialResourceTimeline m2s_;
    SerialResourceTimeline s2m_;
    AddressHeatmap* address_heatmap_ = nullptr;
    double last_issue_arrival_ns_ = 0.0;
};

} // namespace hbfsim::physical::external
