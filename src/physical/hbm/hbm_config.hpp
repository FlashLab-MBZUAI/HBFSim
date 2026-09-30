#pragma once
#include <cstdint>
#include <string_view>

namespace hbfsim::physical::hbm {
inline constexpr std::string_view kStandard = "JEDEC-JESD270-4-2025-04";
// Organization and reference rate follow the public JEDEC announcement.
// Aggregate access latency and efficiency are assumptions, not JEDEC tables.
struct HbmDeviceConfig {
    std::uint64_t capacity_bytes = 48ull * 1024ull * 1024ull * 1024ull;
    std::uint32_t stacks = 1;
    std::uint32_t channels_per_stack = 32;
    std::uint32_t pseudo_channels_per_channel = 2;
    std::uint32_t bank_groups_per_pseudo_channel = 16;
    std::uint32_t banks_per_group = 4;
    // Interface geometry and rate are the single source of truth for HBM
    // data movement. A channel is divided evenly into pseudo-channels. Each
    // pseudo-channel burst transfers burst_length beats over its share of the
    // DQ pins, so width/rate/BL derive both bytes and duration; callers cannot
    // configure an inconsistent aggregate GB/s or independent tBL.
    std::uint64_t channel_row_size_bytes = 2048;
    std::uint32_t channel_width_bits = 64;
    std::uint32_t burst_length = 8;
    double pin_rate_Gbps = 8.0;
    // The derived command clock remains the integer accounting grid for
    // service and causal boundaries; this model does not schedule commands.
    std::uint32_t data_rate_per_command_clock = 4;
};

// Workload-independent approximation parameters, not JEDEC command minima.
// The initial latency assumptions cover a cold/conflicting access; effective
// bandwidth includes average refresh and command overhead. No bank commands
// or periodic refresh events are simulated.
struct HbmTimingConfig {
    double read_latency_ns = 42.0;
    double write_latency_ns = 38.0;
    double read_to_write_ns = 8.0;
    double write_to_read_ns = 8.0;
    double bandwidth_efficiency = 0.94;
};

struct HbmControllerConfig {
    // Contiguous bytes served by one pseudo-channel before the map rotates to
    // the next one. An explicit value must be a multiple of the derived burst
    // bytes and divide the per-pseudo-channel row bytes; 0 selects the
    // largest such value that does not exceed 256 B (256 B on every shipped
    // geometry). The resolved value is reported by effective_interleave_bytes()
    // and by HbmDevice::config().
    std::uint64_t interleave_bytes = 0;
    double address_mapping_ns = 0.0;
    // Maximum pending parent transfers per pseudo-channel. Admission pumps
    // the existing queue when full. Round-robin quanta prevent starvation.
    std::uint32_t queue_depth = 32;
    // Maximum non-preemptible application transfer on one pseudo-channel.
    // Large requests are represented by a remaining-byte count, not bursts.
    std::uint64_t service_quantum_bytes = 4096;
    // Consecutive pseudo-channels arbitrate as a group. Service duration is
    // the maximum lane demand, preserving isolated-transfer bandwidth and
    // byte accounting; concurrent disjoint lanes within a group serialize.
    // One is the fine-grained sensitivity setting of the same implementation.
    std::uint32_t service_group_channels = 4;
};

}
