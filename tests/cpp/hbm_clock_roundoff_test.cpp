#include "physical/hbm/hbm_device.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using hbfsim::physical::Op;
using hbfsim::physical::PhysicalCompletion;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::physical::TraceConfig;
using hbfsim::physical::TraceMode;
using hbfsim::physical::hbm::HbmConfig;
using hbfsim::physical::hbm::HbmDevice;
using hbfsim::physical::hbm::HbmStats;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool near(double lhs, double rhs) {
    const auto scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= 2e-12 * scale;
}

HbmConfig production_config() {
    HbmConfig config;
    config.device.capacity_bytes = 256ull << 30;
    config.device.stacks = 8;
    config.device.channels_per_stack = 32;
    config.device.pseudo_channels_per_channel = 2;
    config.device.bank_groups_per_pseudo_channel = 16;
    config.device.banks_per_group = 4;
    config.device.channel_row_size_bytes = 2048;
    config.device.channel_width_bits = 64;
    config.device.burst_length = 8;
    config.device.pin_rate_Gbps = 6.4;
    config.device.data_rate_per_command_clock = 4;
    config.controller.queue_depth = 32;
    return config;
}

PhysicalRequest request(
    std::string id,
    Op op,
    double arrival_ns,
    std::uint64_t addr,
    std::uint64_t bytes) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = Tier::HBM,
        .op = op,
        .trace = TraceConfig{
            .mode = TraceMode::Off,
            .retain_completion_diagnostics = false,
        },
        .arrival_ns = arrival_ns,
        .addr = addr,
        .bytes = bytes,
    };
}

void test_command_clock_alignment_tolerates_only_edge_roundoff() {
    auto config = production_config();
    // This is the low-bandwidth point used by the closed-loop GPU validation.
    // Its non-binary command period exposed accumulated edge drift in long
    // replays much more strongly than the nominal HBM timing point.
    config.device.pin_rate_Gbps = 0.10169070479801175;
    const double period_ns = config.command_clock_period_ns();
    const auto infinity = std::numeric_limits<double>::infinity();
    const std::array<std::uint64_t, 7> cycles{
        1, 17, 127, 4'093, 65'537, 1'000'003, 16'777'219};
    for (const auto cycle : cycles) {
        const double edge_ns = static_cast<double>(cycle) * period_ns;
        require(config.command_aligned_time_ns(edge_ns) == edge_ns,
                "an exact edge was not stable");
        const double one_ulp_after = std::nextafter(edge_ns, infinity);
        const double rounded = config.command_aligned_time_ns(one_ulp_after);
        require(rounded == one_ulp_after,
                "one-ULP command-edge roundoff changed causal event time");
        require(config.command_aligned_time_ns(rounded) == rounded,
                "alignment was not idempotent");
        require(rounded >= one_ulp_after &&
                    rounded < static_cast<double>(cycle + 1) * period_ns,
                "edge tolerance either violated causality or inserted a cycle");

        double genuine_after = edge_ns;
        for (int step = 0; step < 32; ++step) {
            genuine_after = std::nextafter(genuine_after, infinity);
        }
        require(
            config.command_aligned_time_ns(genuine_after) ==
                static_cast<double>(cycle + 1) * period_ns,
            "a genuine after-edge command failed to advance to the next cycle");
    }
    // Beyond the usual replay horizon, a scale-only tolerance could absorb
    // half a command cycle. The cycle-relative cap must prevent that.
    const double distant_edge = static_cast<double>(1ULL << 45) * period_ns;
    const double distant_half_cycle = distant_edge + period_ns / 2;
    require(config.command_aligned_time_ns(distant_half_cycle) > distant_half_cycle,
            "large-timestamp tolerance swallowed a real half-cycle delay");
    bool horizon_rejected = false;
    try {
        (void)config.command_aligned_time_ns(static_cast<double>(1ULL << 49) * period_ns);
    } catch (const std::runtime_error&) {
        horizon_rejected = true;
    }
    require(horizon_rejected, "integer-clock precision horizon was not enforced");
}

struct TranslationReplay {
    std::vector<PhysicalCompletion> completions;
    HbmStats stats;
};

TranslationReplay run_translation_replay(
    std::uint64_t origin_cycles,
    double pin_rate_Gbps = 0.10169070479801175) {
    auto config = production_config();
    config.device.capacity_bytes = 64ull << 20;
    config.device.stacks = 1;
    config.device.channels_per_stack = 1;
    config.device.pseudo_channels_per_channel = 1;
    config.device.bank_groups_per_pseudo_channel = 4;
    config.device.banks_per_group = 4;
    config.device.pin_rate_Gbps = pin_rate_Gbps;
    config.controller.queue_depth = 32;
    HbmDevice device(config);
    const double period_ns = config.command_clock_period_ns();
    std::vector<std::uint64_t> tickets;
    tickets.reserve(512);
    for (std::uint64_t index = 0; index < 512; ++index) {
        const auto arrival_cycle = origin_cycles + index * 8 + index % 7;
        const auto block = (index * 131 + (index / 11) * 4093) % 131072;
        tickets.push_back(device.enqueue(request(
            "translation-" + std::to_string(index),
            index % 5 == 0 || index % 11 == 0 ? Op::Write : Op::Read,
            static_cast<double>(arrival_cycle) * period_ns,
            block * config.burst_bytes(),
            config.burst_bytes())));
    }
    device.drain_queues();
    TranslationReplay result;
    result.completions.reserve(tickets.size());
    for (const auto ticket : tickets) {
        result.completions.push_back(device.pump(ticket));
    }
    result.stats = device.stats();
    return result;
}

void test_controller_replay_is_time_translation_invariant() {
    constexpr std::uint64_t kShiftCycles = 4'000'003;
    const auto baseline = run_translation_replay(0);
    const auto shifted = run_translation_replay(kShiftCycles);
    auto config = production_config();
    config.device.pin_rate_Gbps = 0.10169070479801175;
    const double shift_ns =
        static_cast<double>(kShiftCycles) * config.command_clock_period_ns();
    const auto translated_near = [shift_ns](double baseline_ns,
                                            double shifted_ns) {
        const auto scale = std::max(
            {1.0, std::abs(baseline_ns), std::abs(shifted_ns),
             std::abs(shift_ns)});
        const auto tolerance =
            16.0 * std::numeric_limits<double>::epsilon() * scale;
        return std::abs((shifted_ns - shift_ns) - baseline_ns) <= tolerance;
    };

    require(baseline.completions.size() == shifted.completions.size(),
            "time translation changed completion count");
    for (std::size_t index = 0; index < baseline.completions.size(); ++index) {
        const auto& lhs = baseline.completions[index];
        const auto& rhs = shifted.completions[index];
        require(lhs.id == rhs.id && lhs.op == rhs.op,
                "time translation changed completion identity or order");
        require(translated_near(lhs.start_ns, rhs.start_ns),
                "time translation changed relative request start time");
        require(translated_near(lhs.finish_ns, rhs.finish_ns),
                "time translation changed relative request finish time at " +
                    std::to_string(index) + " by " +
                    std::to_string(
                        (rhs.finish_ns - shift_ns) - lhs.finish_ns) + " ns");
        require(rhs.logical_bytes == lhs.logical_bytes &&
                    rhs.physical_bytes == lhs.physical_bytes,
                "time translation changed request byte accounting");
    }
    const auto& lhs = baseline.stats;
    const auto& rhs = shifted.stats;
    require(rhs.read_bytes == lhs.read_bytes &&
                rhs.write_bytes == lhs.write_bytes &&
                rhs.max_queue_occupancy == lhs.max_queue_occupancy,
            "time translation changed controller scheduling counters");
    require(near(
                (rhs.finish_ns - rhs.first_arrival_ns),
                (lhs.finish_ns - lhs.first_arrival_ns)),
            "time translation changed total replay elapsed time");
}

void test_long_running_command_alignment() {
    auto config = production_config();
    constexpr auto old_limit = 1ULL << 46;
    constexpr auto limit = 1ULL << 48;
    for (const double rate : {0.10169070479801175, 6.4, 7.2, 8.0, 9.2, 9.6}) {
        config.device.pin_rate_Gbps = rate;
        const auto period = config.command_clock_period_ns();
        for (const auto cycle : {old_limit - 1, old_limit, old_limit + 1,
                                 1ULL << 47, limit - 2}) {
            const auto edge = config.command_clock_time_ns(cycle);
            const auto next = config.command_clock_time_ns(cycle + 1);
            require(config.command_clock_cycles(edge) == cycle,
                    "a distant command edge did not round-trip");
            for (const auto fraction : {0.125, 0.5, 0.875}) {
                const auto arrival = edge + fraction * period;
                require(arrival > edge && arrival < next,
                        "distant command time lost sub-cycle precision");
                require(config.command_clock_cycles(arrival) == cycle + 1 &&
                            config.command_aligned_time_ns(arrival) == next &&
                            config.command_aligned_time_ns(next) == next,
                        "long-run alignment lost causality or a real delay");
            }
        }
        require(config.command_clock_cycles(config.command_clock_time_ns(limit - 1)) ==
                    limit - 1,
                "last supported command edge was rejected");
        bool rejected = false;
        try {
            (void)config.command_clock_cycles(config.command_clock_time_ns(limit));
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "long-run precision bound was not enforced");
    }
}

void test_endurance_replay_across_old_clock_horizon() {
    // The endurance campaign uses a 0.5 ns clock. Integer controller scheduling
    // must be identical when the same trace crosses the old absolute-time cap.
    const auto baseline = run_translation_replay(0, 8.0);
    for (const auto origin : {(1ULL << 46) - 2'048, 1ULL << 47,
                              (1ULL << 48) - (1ULL << 20)}) {
        const auto shifted = run_translation_replay(origin, 8.0);
        const double shift = static_cast<double>(origin) * 0.5;
        for (std::size_t i = 0; i < baseline.completions.size(); ++i) {
            const auto& a = baseline.completions[i];
            const auto& b = shifted.completions[i];
            require(b.start_ns - shift == a.start_ns &&
                        b.finish_ns - shift == a.finish_ns &&
                        b.logical_bytes == a.logical_bytes &&
                        b.physical_bytes == a.physical_bytes,
                    "long-run replay changed relative timing or bytes");
        }
        const auto& a = baseline.stats;
        const auto& b = shifted.stats;
        require(a.read_bytes == b.read_bytes && a.write_bytes == b.write_bytes &&
                    a.service_quanta == b.service_quanta && a.bus_busy_ns == b.bus_busy_ns,
                "long-run replay changed channel counters");

        auto config = production_config();
        config.device.pin_rate_Gbps = 8.0;
        config.device.capacity_bytes = 64ULL << 20;
        config.device.stacks = 1;
        config.device.channels_per_stack = 1;
        HbmDevice early(config), late(config);
        early.reserve_controller_buffer(65536);
        late.reserve_controller_buffer(65536);
        for (std::uint64_t i = 0; i < 128; ++i) {
            const double at = static_cast<double>(i) * 64;
            early.advance_buffer_frontier(at);
            late.advance_buffer_frontier(shift + at);
            const auto make = [&](double arrival) {
                return request("buffer", i % 2 ? Op::Read : Op::Write, arrival,
                               early.application_capacity_bytes(), 4096);
            };
            const auto a = early.transfer_controller_buffer(make(at));
            const auto b = late.transfer_controller_buffer(make(shift + at));
            require(b.start_ns - shift == a.start_ns &&
                        b.finish_ns - shift == a.finish_ns &&
                        b.physical_bytes == a.physical_bytes,
                    "long-run FTL buffer transfer changed timing or bytes");
        }
        require(early.stats().controller_buffer_read_bytes ==
                    late.stats().controller_buffer_read_bytes &&
                    early.stats().controller_buffer_write_bytes ==
                    late.stats().controller_buffer_write_bytes,
                "long-run FTL buffer accounting changed");
    }
}

} // namespace

int main() {
    try {
        test_command_clock_alignment_tolerates_only_edge_roundoff();
        test_controller_replay_is_time_translation_invariant();
        test_long_running_command_alignment();
        test_endurance_replay_across_old_clock_horizon();
        std::cout << "HBM command-clock roundoff and time translation: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
