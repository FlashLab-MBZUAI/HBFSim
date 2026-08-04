#pragma once

#include "physical/physical_types.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hbfsim::physical::hbm {

enum class HbmLevel {
    Channel,
    PseudoChannel,
    BankGroup,
    Bank,
    Row,
    Column,
};

enum class HbmCommand {
    ACT,
    PRE,
    PREA,
    RD,
    WR,
    RDA,
    WRA,
    REFab,
    REFsb,
};

struct HbmCommandMeta {
    bool opens_row = false;
    bool closes_row = false;
    bool accesses_column = false;
    bool refreshes = false;
};

struct HbmTimingGate {
    double ready_ns = 0.0;
    // Every reason is a compile-time timing-rule label. Keeping a view avoids
    // constructing/copying strings on every simulated DRAM command while
    // retaining the same text for trace diagnostics.
    std::string_view reason;
};

struct HbmConfig {
    static constexpr std::string_view address_mapping_scheme() {
        return "burst-pch-bank-swizzle-v1";
    }

    std::uint64_t capacity_bytes = 128ull * 1024ull * 1024ull * 1024ull;
    std::uint32_t stacks = 1;
    std::uint32_t channels_per_stack = 8;
    std::uint32_t pseudo_channels_per_channel = 2;
    std::uint32_t bank_groups_per_pseudo_channel = 4;
    std::uint32_t banks_per_group = 4;
    // Interface geometry and rate are the single source of truth for HBM
    // data movement. A channel is divided evenly into pseudo-channels. Each
    // pseudo-channel burst transfers burst_length beats over its share of the
    // DQ pins, so width/rate/BL derive both bytes and duration; callers cannot
    // configure an inconsistent aggregate GB/s or independent tBL.
    std::uint64_t channel_row_size_bytes = 2048;
    std::uint32_t channel_width_bits = 64;
    std::uint32_t burst_length = 8;
    double pin_rate_Gbps = 6.4;
    // HBM3/HBM4 use a command clock below the DQ transfer rate. The ratio is
    // explicit because tCCD is specified in command-clock cycles.
    std::uint32_t data_rate_per_command_clock = 4;
    double address_mapping_ns = 0.0;

    double tRCDRD_ns = 14.0;
    double tRCDWR_ns = 14.0;
    double tCL_ns = 14.0;
    double tCWL_ns = 10.0;
    double tRP_ns = 14.0;
    double tRAS_ns = 32.0;
    double tRC_ns = 46.0;
    double tWR_ns = 15.0;
    double tRTP_ns = 7.5;
    std::uint32_t tCCD_S_cycles = 2;
    std::uint32_t tCCD_L_cycles = 4;
    double tRRD_S_ns = 4.0;
    double tRRD_L_ns = 6.0;
    double tFAW_ns = 20.0;
    double tWTR_S_ns = 4.0;
    double tWTR_L_ns = 8.0;
    double tRTW_ns = 8.0;
    bool refresh_enabled = false;
    bool same_bank_refresh = false;
    double tREFI_ns = 3900.0;
    double tRFC_ns = 350.0;
    double tRFCsb_ns = 160.0;
    double tRREFD_ns = 10.0;
    // FR-FCFS scheduling: per-pseudo-channel pending-queue depth (a real
    // controller holds a few tens of requests per channel; a full queue
    // backpressures admission) and the anti-starvation time window. In
    // addition to this time gate, an entry may be bypassed at most queue_depth
    // times, so equal-timestamp row hits cannot starve an old conflict.
    std::uint32_t queue_depth = 32;
    double frfcfs_cap_ns = 5000.0;

    [[nodiscard]] std::uint64_t pseudo_channel_width_bits() const;
    [[nodiscard]] std::uint64_t row_size_bytes() const;
    [[nodiscard]] std::uint64_t burst_bytes() const;
    [[nodiscard]] double channel_bandwidth_GBps() const;
    [[nodiscard]] double pseudo_channel_bandwidth_GBps() const;
    [[nodiscard]] double command_clock_period_ns() const;
    [[nodiscard]] double command_clock_MHz() const;
    [[nodiscard]] double burst_duration_ns() const;
    [[nodiscard]] double tCCD_S_ns() const;
    [[nodiscard]] double tCCD_L_ns() const;
    [[nodiscard]] double command_aligned_time_ns(double time_ns) const;
};

struct HbmAddress {
    std::uint32_t stack = 0;
    std::uint32_t channel = 0;
    std::uint32_t pseudo_channel = 0;
    std::uint32_t bank_group = 0;
    std::uint32_t bank = 0;
    std::uint64_t row = 0;
    std::uint64_t offset = 0;

    [[nodiscard]] std::string path() const;
};

struct HbmStats {
    std::uint64_t read_bytes = 0;
    std::uint64_t write_bytes = 0;
    std::uint64_t row_hits = 0;
    std::uint64_t row_misses = 0;
    std::uint64_t row_conflicts = 0;
    std::uint64_t activations = 0;
    std::uint64_t precharges = 0;
    std::uint64_t refresh_count = 0;
    double bus_busy_ns = 0.0;
    double first_arrival_ns = std::numeric_limits<double>::infinity();
    double finish_ns = 0.0;
    std::uint64_t pseudo_channels = 0;
    std::uint64_t active_pseudo_channels = 0;
    std::uint64_t max_pseudo_channel_accesses = 0;
    std::uint64_t max_queue_occupancy = 0;
    double max_pseudo_channel_busy_ns = 0.0;
    double avg_active_pseudo_channel_busy_ns = 0.0;
    // Sum of every physical burst child's stage work. Children can overlap
    // across pseudo-channels, so this is intentionally not wall time.
    Breakdown stage_work;

    [[nodiscard]] double row_hit_rate() const;
    [[nodiscard]] double active_span_ns() const;
    [[nodiscard]] double utilization() const;
    [[nodiscard]] double bus_parallelism() const;
    [[nodiscard]] double pseudo_channel_busy_skew() const;
};

class HbmDevice {
public:
    explicit HbmDevice(
        HbmConfig config,
        AddressHeatmap* address_heatmap = nullptr);

    [[nodiscard]] HbmAddress decode(std::uint64_t addr) const;
    [[nodiscard]] std::uint64_t encode(const HbmAddress& addr) const;
    // Synchronous request: enqueue + pump in one call. On an empty queue
    // this is exactly the pre-scheduler behavior (QD1 path unchanged).
    [[nodiscard]] PhysicalCompletion issue(const PhysicalRequest& request);
    // FR-FCFS front end: enqueue splits the logical range at DRAM burst/row
    // boundaries, hands each child to its mapped pseudo-channel queue (bounded
    // by hbm-queue-depth), and returns one parent ticket. pump services every
    // affected pseudo-channel until all children finish, then returns their
    // aggregate completion; drain_queues services everything so remaining
    // tickets become pump-able map lookups.
    [[nodiscard]] std::uint64_t enqueue(const PhysicalRequest& request);
    [[nodiscard]] PhysicalCompletion pump(std::uint64_t ticket);
    void drain_queues();
    [[nodiscard]] const HbmStats& stats() const {
        refresh_parallel_stats();
        return stats_;
    }
    [[nodiscard]] const HbmConfig& config() const { return config_; }
    void attach_address_heatmap(AddressHeatmap& address_heatmap) {
        if (last_enqueue_arrival_ns_) {
            throw std::runtime_error(
                "HBM address heatmap must be attached before the first request");
        }
        address_heatmap_ = &address_heatmap;
    }

private:
    static constexpr std::size_t kCommandCount = 9;

    struct BankState {
        bool has_open_row = false;
        std::uint64_t open_row = 0;
        std::array<HbmTimingGate, kCommandCount> command_ready{};
    };

    struct BankGroupState {
        std::array<HbmTimingGate, kCommandCount> command_ready{};
    };

    struct QueuedRequest {
        std::uint64_t ticket = 0;
        PhysicalRequest request;
        HbmAddress addr;
        std::uint32_t bypass_count = 0;
    };

    struct PendingRequest {
        PhysicalCompletion completion;
        std::size_t remaining_children = 0;
        std::size_t total_children = 0;
        std::size_t pseudo_channels = 0;
        bool enqueue_complete = false;
        bool has_child_completion = false;
        bool retain_diagnostics = true;
    };

    struct PseudoChannelState {
        std::vector<BankState> banks;
        std::vector<BankGroupState> bank_groups;
        std::array<HbmTimingGate, kCommandCount> command_ready{};
        double bus_ready_ns = 0.0;
        double bus_busy_ns = 0.0;
        double next_refresh_ns = 0.0;
        double refresh_ready_ns = 0.0;
        std::uint64_t refresh_epoch = 0;
        std::uint64_t accesses = 0;
        std::deque<double> recent_activations;
        // FR-FCFS pending queue, kept in arrival order.
        std::deque<QueuedRequest> queue;
    };

    struct IsolatedTransitionKey {
        std::uint64_t source_state_key = 0;
        std::uint64_t arrival_bits = 0;
        std::uint64_t first_edge_group = 0;
        std::uint64_t full_group_begin = 0;
        std::uint64_t full_group_end = 0;
        std::uint64_t last_edge_group = 0;
        Op op = Op::Read;
        std::uint8_t edge_mask = 0;

        [[nodiscard]] bool operator==(
            const IsolatedTransitionKey&) const = default;
    };

    struct IsolatedTransition {
        IsolatedTransitionKey transition;
        std::uint64_t destination_state_key = 0;
    };

    struct CommandSchedule {
        double issue_ns = 0.0;
        double finish_ns = 0.0;
        bool issued = true;
    };

    struct FirstCommandPreview {
        double issue_ns = 0.0;
        bool row_hit = false;
    };

    HbmConfig config_;
    AddressHeatmap* address_heatmap_ = nullptr;
    // Immutable geometry/timing derived and validated once at construction.
    // The burst state machine executes hundreds of millions of commands in a
    // production layer replay; re-running checked topology products and the
    // same floating-point divisions inside every command adds no validation.
    std::uint64_t row_size_bytes_ = 0;
    std::uint64_t burst_bytes_ = 0;
    std::uint64_t row_bursts_ = 0;
    std::uint64_t total_pseudo_channels_ = 0;
    std::uint64_t banks_per_pseudo_channel_ = 0;
    std::uint64_t pseudo_channels_per_stack_ = 0;
    double command_clock_period_ns_ = 0.0;
    double burst_duration_ns_ = 0.0;
    double tccd_s_ns_ = 0.0;
    double tccd_l_ns_ = 0.0;
    std::vector<PseudoChannelState> pseudo_channels_;
    // Diagnostics-free synchronous spans can update one representative for
    // every controller-equivalent pseudo-channel class. The class ids are an
    // exact state-partition cache, never a statistical approximation.
    std::vector<std::uint32_t> pseudo_channel_class_ids_;
    std::vector<std::size_t> pseudo_channel_class_representatives_;
    // Stable semantic keys let two classes that execute the same deterministic
    // transition merge in O(1) without rescanning every bank timing gate.
    // Only the previous isolated request is retained, which is sufficient for
    // complementary page halves and keeps memory independent of trace length.
    std::vector<std::uint64_t> pseudo_channel_class_state_keys_;
    std::vector<IsolatedTransition> previous_isolated_transitions_;
    std::uint64_t next_pseudo_channel_state_key_ = 1;
    bool pseudo_channel_classes_valid_ = true;
    mutable HbmStats stats_;
    std::uint64_t next_ticket_ = 0;
    // Per-parent visitation marker used to build its unique pseudo-channel
    // route list in O(children), rather than linearly rescanning the list for
    // every burst child.
    std::vector<std::uint64_t> route_ticket_markers_;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> ticket_pseudo_channels_;
    std::unordered_map<std::uint64_t, PendingRequest> pending_;
    std::unordered_map<std::uint64_t, PhysicalCompletion> completed_;
    std::optional<double> last_enqueue_arrival_ns_;

    [[nodiscard]] std::size_t pseudo_channel_index(const HbmAddress& addr) const;
    [[nodiscard]] std::size_t bank_index(const HbmAddress& addr) const;
    [[nodiscard]] HbmCommand final_command(Op op) const;
    [[nodiscard]] double align_command_time(double time_ns) const;
    [[nodiscard]] double command_delay(double minimum_ns) const;
    [[nodiscard]] double column_latency(HbmCommand command) const;
    void validate_and_begin_request(const PhysicalRequest& request);
    [[nodiscard]] std::optional<PhysicalCompletion>
    try_issue_isolated_span(const PhysicalRequest& request);
    void materialize_pseudo_channel_classes();
    void rebuild_pseudo_channel_classes();
    [[nodiscard]] bool same_pseudo_channel_state(
        const PseudoChannelState& lhs,
        const PseudoChannelState& rhs) const;
    void aggregate_child(std::uint64_t ticket, PhysicalCompletion child);
    // FR-FCFS pick: the oldest pending request, unless an actually-ready
    // row-hit exists whose bypass stays inside both starvation caps.
    [[nodiscard]] std::size_t pick_next(const PseudoChannelState& pseudo_channel) const;
    [[nodiscard]] FirstCommandPreview preview_first_command(
        const PseudoChannelState& pseudo_channel,
        const QueuedRequest& queued) const;
    void service_one(std::size_t pseudo_channel_index);
    [[nodiscard]] PhysicalCompletion service_request(
        PseudoChannelState& pseudo_channel,
        const HbmAddress& addr,
        const PhysicalRequest& request);
    [[nodiscard]] CommandSchedule schedule_command(
        PseudoChannelState& pseudo_channel,
        HbmAddress addr,
        HbmCommand command,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        const std::string& entity,
        double& first_command_ns,
        HbmCommand final_access = HbmCommand::RD,
        double data_burst_ns = 0.0);
    [[nodiscard]] double reserve_refresh_free_window(
        PseudoChannelState& pseudo_channel,
        double& start_ns,
        double duration_ns,
        std::vector<TraceSpan>* spans,
        const std::string& entity);
    void apply_command_state(
        PseudoChannelState& pseudo_channel,
        const HbmAddress& addr,
        HbmCommand command,
        double issue_ns,
        HbmCommand final_access,
        double data_burst_ns);
    void apply_refresh_state(PseudoChannelState& pseudo_channel);
    void refresh_parallel_stats() const;
};

} // namespace hbfsim::physical::hbm
