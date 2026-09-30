#include "physical/hbm/hbm_device.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace hbfsim::physical::hbm {
namespace {
void require_positive_count(std::uint64_t value, const char* name) {
    if (value == 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

void require_positive_timing(double value, const char* name) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " must be positive and finite");
    }
}

void require_nonnegative_timing(double value, const char* name) {
    if (value < 0.0 || !std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " must be nonnegative and finite");
    }
}

// Below 2^48 cycles, double timestamps retain at least 16 subdivisions per
// command period, including non-binary periods. Keep that precision margin
// without truncating GC-heavy endurance runs at the former 2^46-cycle limit
// (only 9.77 simulated hours at tCK = 0.5 ns).
constexpr double kMaximumCommandClockCycle = 0x1p48;

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs * rhs;
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs + rhs;
}

std::uint64_t modular_add(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::uint64_t modulus) {
    // Both operands are reduced, so the sum cannot wrap even when modulus is
    // not a power of two.
    return lhs >= modulus - rhs ? lhs - (modulus - rhs) : lhs + rhs;
}

std::uint64_t modular_subtract(
    std::uint64_t lhs,
    std::uint64_t rhs,
    std::uint64_t modulus) {
    return lhs >= rhs ? lhs - rhs : modulus - (rhs - lhs);
}

// Golden-ratio hash of the per-pseudo-channel stripe index, taking the HIGH
// multiplier bits: it decorrelates power-of-two request strides before
// pseudo-channel selection (low multiplier bits would still alias for even
// strides).
std::uint64_t pseudo_channel_hash(std::uint64_t stripe) {
    return (stripe * 0x9E3779B97F4A7C15ull) >> 32;
}

std::uint64_t bank_group_hash(std::uint64_t bank_row) {
    return (bank_row * 0xD1B54A32D192ED03ull) >> 32;
}

} // namespace

std::uint64_t HbmConfig::pseudo_channel_width_bits() const {
    if (device.pseudo_channels_per_channel == 0 ||
        device.channel_width_bits % device.pseudo_channels_per_channel != 0) {
        throw std::runtime_error(
            "HBM channel width must divide evenly across pseudo-channels");
    }
    return device.channel_width_bits / device.pseudo_channels_per_channel;
}

std::uint64_t HbmConfig::row_size_bytes() const {
    if (device.pseudo_channels_per_channel == 0 ||
        device.channel_row_size_bytes % device.pseudo_channels_per_channel != 0) {
        throw std::runtime_error(
            "HBM channel row size must divide evenly across pseudo-channels");
    }
    return device.channel_row_size_bytes / device.pseudo_channels_per_channel;
}

std::uint64_t HbmConfig::burst_bytes() const {
    const auto pseudo_width = pseudo_channel_width_bits();
    if (pseudo_width % 8 != 0) {
        throw std::runtime_error("HBM pseudo-channel width must be byte-aligned");
    }
    return checked_mul(pseudo_width / 8, device.burst_length, "HBM burst bytes");
}

std::uint64_t HbmConfig::effective_interleave_bytes() const {
    const auto row = row_size_bytes();
    const auto burst = burst_bytes();
    if (burst > row || row % burst != 0) {
        throw std::runtime_error(
            "HBM derived burst bytes must divide the pseudo-channel row size");
    }
    if (controller.interleave_bytes != 0) {
        if (controller.interleave_bytes % burst != 0 || controller.interleave_bytes > row ||
            row % controller.interleave_bytes != 0) {
            throw std::runtime_error(
                "HBM controller.interleave_bytes=" + std::to_string(controller.interleave_bytes) +
                " must be a multiple of the derived burst bytes (" +
                std::to_string(burst) + ") and divide the pseudo-channel row (" +
                std::to_string(row) + " bytes); set hbm-interleave-bytes to such "
                "a value or to 0 for the automatic default");
        }
        return controller.interleave_bytes;
    }
    // Largest burst multiple that divides the row and stays within the
    // default: walk the divisors of bursts-per-row.
    const auto bursts_per_row = row / burst;
    std::uint64_t best = burst;
    for (std::uint64_t bursts = 1; bursts <= bursts_per_row; ++bursts) {
        if (bursts_per_row % bursts != 0) {
            continue;
        }
        if (bursts > kDefaultInterleaveBytes / burst) {
            break;
        }
        best = bursts * burst;
    }
    return best;
}

double HbmConfig::channel_bandwidth_GBps() const {
    return device.pin_rate_Gbps * static_cast<double>(device.channel_width_bits) / 8.0;
}

double HbmConfig::pseudo_channel_bandwidth_GBps() const {
    return device.pin_rate_Gbps * static_cast<double>(pseudo_channel_width_bits()) / 8.0;
}

double HbmConfig::command_clock_period_ns() const {
    return device.data_rate_per_command_clock / device.pin_rate_Gbps;
}

double HbmConfig::command_clock_MHz() const {
    return 1000.0 / command_clock_period_ns();
}

double HbmConfig::burst_duration_ns() const {
    return static_cast<double>(device.burst_length) / device.pin_rate_Gbps;
}

std::uint64_t HbmConfig::command_clock_cycles(double time_ns) const {
    if (!(time_ns >= 0.0) || !std::isfinite(time_ns)) {
        throw std::runtime_error(
            "HBM command time must be nonnegative and finite");
    }
    const double tck_ns = command_clock_period_ns();
    require_positive_timing(tck_ns, "HBM derived command-clock period");
    const double cycles = time_ns / tck_ns;
    if (!(cycles < kMaximumCommandClockCycle)) {
        throw std::runtime_error(
            "HBM command time exceeds the representable clock horizon");
    }
    auto eligible = static_cast<std::uint64_t>(std::floor(cycles));
    const auto covered_by_edge = [time_ns, tck_ns](double edge_ns) {
        if (edge_ns >= time_ns) {
            return true;
        }
        const auto infinity = std::numeric_limits<double>::infinity();
        const double tolerance = std::min(
            tck_ns * 1e-6,
            16.0 * std::max({
                std::nextafter(time_ns, infinity) - time_ns,
                std::nextafter(edge_ns, infinity) - edge_ns,
                std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::abs(time_ns))}));
        return time_ns - edge_ns <= tolerance;
    };
    while (!covered_by_edge(command_clock_time_ns(eligible))) {
        ++eligible;
    }
    while (eligible != 0 && covered_by_edge(command_clock_time_ns(eligible - 1))) {
        --eligible;
    }
    if (eligible >= static_cast<std::uint64_t>(kMaximumCommandClockCycle)) {
        throw std::runtime_error(
            "HBM command time exceeds the representable clock horizon");
    }
    return eligible;
}

double HbmConfig::command_clock_time_ns(std::uint64_t cycles) const {
    return static_cast<double>(cycles) * command_clock_period_ns();
}

double HbmConfig::command_aligned_time_ns(double time_ns) const {
    return std::max(time_ns, command_clock_time_ns(command_clock_cycles(time_ns)));
}

std::string HbmAddress::path() const {
    std::ostringstream out;
    out << "stack" << stack << "/ch" << channel << "/pch" << pseudo_channel
        << "/bg" << bank_group << "/bank" << bank << "/row" << row
        << "/off" << offset;
    return out.str();
}

double HbmStats::active_span_ns() const {
    return std::isfinite(first_arrival_ns) && finish_ns > first_arrival_ns ?
        finish_ns - first_arrival_ns : 0.0;
}

double HbmStats::utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || pseudo_channels == 0 ? 0.0 :
        bus_busy_ns / (span * static_cast<double>(pseudo_channels));
}

double HbmStats::bus_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : bus_busy_ns / span;
}

double HbmStats::pseudo_channel_busy_skew() const {
    return avg_active_pseudo_channel_busy_ns <= 0.0 ? 0.0 :
        max_pseudo_channel_busy_ns / avg_active_pseudo_channel_busy_ns;
}


// Address map: pch-interleave-bg-rotate-v2
//
//   unit           = addr / interleave_bytes      (contiguous interleave unit)
//   lane, stripe   = unit % P, unit / P           (P = pseudo-channels)
//   pseudo-channel = (lane + hash(stripe)) % P    (reversible high-bit swizzle)
//   within a pseudo-channel, successive stripes rotate bank groups first,
//   then fill the row of the bank they land in (column units), then rotate
//   banks within the group, then rows:
//     lane_bg = stripe % BG; g1 = stripe / BG
//     column_unit = g1 % units_per_row; g2 = g1 / units_per_row
//     bank = g2 % banks_per_group; row = g2 / banks_per_group
//     bank_group = (lane_bg + hash(g2)) % BG
// A request that covers whole stripes presents the same local bank/row/column
// sequence to every pseudo-channel.
// ---------------------------------------------------------------------------

HbmDevice::LocalAddress HbmDevice::local_address(std::uint64_t unit) const {
    const auto bank_groups = config_.device.bank_groups_per_pseudo_channel;
    const auto lane_bg = unit % bank_groups;
    const auto g1 = unit / bank_groups;
    LocalAddress local;
    local.column_unit = g1 % units_per_row_;
    const auto g2 = g1 / units_per_row_;
    local.bank = static_cast<std::uint32_t>(g2 % config_.device.banks_per_group);
    local.row = g2 / config_.device.banks_per_group;
    local.bank_group = static_cast<std::uint32_t>(modular_add(
        lane_bg, bank_group_hash(g2) % bank_groups, bank_groups));
    return local;
}

void HbmDevice::assign_pseudo_channel(
    HbmAddress& addr,
    std::size_t pseudo_channel) const {
    addr.stack = static_cast<std::uint32_t>(
        pseudo_channel / pseudo_channels_per_stack_);
    const auto within_stack = pseudo_channel % pseudo_channels_per_stack_;
    addr.channel = static_cast<std::uint32_t>(
        within_stack / config_.device.pseudo_channels_per_channel);
    addr.pseudo_channel = static_cast<std::uint32_t>(
        within_stack % config_.device.pseudo_channels_per_channel);
}

std::uint64_t HbmDevice::byte_address(
    std::size_t pseudo_channel,
    std::uint64_t unit,
    std::uint64_t unit_offset) const {
    const auto lane = modular_subtract(
        pseudo_channel,
        pseudo_channel_hash(unit) % total_pseudo_channels_,
        total_pseudo_channels_);
    const auto global_unit = checked_add(
        checked_mul(unit, total_pseudo_channels_, "HBM encoded unit"),
        lane,
        "HBM encoded unit");
    return checked_add(
        checked_mul(global_unit, config_.controller.interleave_bytes, "HBM encoded byte address"),
        unit_offset,
        "HBM encoded byte address");
}

HbmAddress HbmDevice::decode(std::uint64_t addr) const {
    if (addr >= config_.device.capacity_bytes) {
        throw std::runtime_error("HBM byte address is out of capacity");
    }
    const auto global_unit = addr / config_.controller.interleave_bytes;
    const auto unit_offset = addr % config_.controller.interleave_bytes;
    const auto lane = global_unit % total_pseudo_channels_;
    const auto stripe = global_unit / total_pseudo_channels_;
    const auto pseudo_channel_linear = modular_add(
        lane,
        pseudo_channel_hash(stripe) % total_pseudo_channels_,
        total_pseudo_channels_);
    const auto local = local_address(stripe);
    HbmAddress decoded;
    assign_pseudo_channel(decoded, static_cast<std::size_t>(pseudo_channel_linear));
    decoded.bank_group = local.bank_group;
    decoded.bank = local.bank;
    decoded.row = local.row;
    decoded.offset = checked_add(
        checked_mul(
            local.column_unit, config_.controller.interleave_bytes, "HBM decoded column offset"),
        unit_offset,
        "HBM decoded burst offset");
    return decoded;
}

std::uint64_t HbmDevice::encode(const HbmAddress& addr) const {
    if (addr.stack >= config_.device.stacks || addr.channel >= config_.device.channels_per_stack ||
        addr.pseudo_channel >= config_.device.pseudo_channels_per_channel ||
        addr.bank_group >= config_.device.bank_groups_per_pseudo_channel ||
        addr.bank >= config_.device.banks_per_group || addr.offset >= row_size_bytes_) {
        throw std::runtime_error("HBM address field out of range");
    }
    const auto bank_groups = config_.device.bank_groups_per_pseudo_channel;
    const auto g2 = checked_add(
        checked_mul(addr.row, config_.device.banks_per_group, "HBM encoded bank row"),
        addr.bank,
        "HBM encoded bank row");
    const auto lane_bg = modular_subtract(
        addr.bank_group, bank_group_hash(g2) % bank_groups, bank_groups);
    const auto column_unit = addr.offset / config_.controller.interleave_bytes;
    const auto unit_offset = addr.offset % config_.controller.interleave_bytes;
    const auto g1 = checked_add(
        checked_mul(g2, units_per_row_, "HBM encoded row units"),
        column_unit,
        "HBM encoded row units");
    const auto stripe = checked_add(
        checked_mul(g1, bank_groups, "HBM encoded stripe"),
        lane_bg,
        "HBM encoded stripe");
    const auto encoded = byte_address(
        (static_cast<std::size_t>(addr.stack) * config_.device.channels_per_stack + addr.channel) *
            config_.device.pseudo_channels_per_channel + addr.pseudo_channel, stripe, unit_offset);
    if (encoded >= config_.device.capacity_bytes) {
        throw std::runtime_error("HBM encoded address is out of capacity");
    }
    return encoded;
}


void HbmDevice::refresh_parallel_stats() const {
    stats_.pseudo_channels = total_pseudo_channels_;
    stats_.active_pseudo_channels = 0;
    stats_.max_pseudo_channel_accesses = 0;
    stats_.max_pseudo_channel_busy_ns = 0.0;
    stats_.avg_active_pseudo_channel_busy_ns = 0.0;
    stats_.retained_bus_gaps = 0;
    double active_busy_ns = 0.0;
    for (const auto& group : service_groups_) stats_.retained_bus_gaps += group.data_bus_cycles.gap_count();
    for (std::size_t index = 0; index < total_pseudo_channels_; ++index) {
        const auto accesses = pseudo_channel_accesses_[index];
        const auto busy_ns = ns(pseudo_channel_bus_busy_cycles_[index]);
        if (accesses == 0 && busy_ns <= 0.0) {
            continue;
        }
        ++stats_.active_pseudo_channels;
        active_busy_ns += busy_ns;
        stats_.max_pseudo_channel_busy_ns =
            std::max(stats_.max_pseudo_channel_busy_ns, busy_ns);
        stats_.max_pseudo_channel_accesses =
            std::max(stats_.max_pseudo_channel_accesses, accesses);
    }
    if (stats_.active_pseudo_channels != 0) {
        stats_.avg_active_pseudo_channel_busy_ns =
            active_busy_ns / static_cast<double>(stats_.active_pseudo_channels);
    }
}


HbmDevice::HbmDevice(HbmConfig config, AddressHeatmap* address_heatmap)
    : config_(config), address_heatmap_(address_heatmap) {
    require_positive_count(config_.device.capacity_bytes, "HBM capacity_bytes");
    require_positive_count(config_.device.stacks, "HBM stacks");
    require_positive_count(config_.device.channels_per_stack, "HBM channels_per_stack");
    require_positive_count(config_.device.pseudo_channels_per_channel, "HBM pseudo_channels");
    require_positive_count(config_.device.bank_groups_per_pseudo_channel, "HBM bank groups");
    require_positive_count(config_.device.banks_per_group, "HBM banks per group");
    require_positive_count(config_.device.channel_row_size_bytes, "HBM row bytes");
    require_positive_count(config_.device.channel_width_bits, "HBM channel width");
    require_positive_count(config_.device.burst_length, "HBM burst length");
    require_positive_count(config_.device.data_rate_per_command_clock, "HBM clock ratio");
    require_positive_timing(config_.device.pin_rate_Gbps, "HBM pin rate");
    require_positive_count(config_.controller.queue_depth, "HBM queue depth");
    require_nonnegative_timing(config_.controller.address_mapping_ns, "HBM mapping latency");
    require_nonnegative_timing(config_.timing.read_latency_ns, "HBM read latency");
    require_nonnegative_timing(config_.timing.write_latency_ns, "HBM write latency");
    require_nonnegative_timing(config_.timing.read_to_write_ns, "HBM read-to-write delay");
    require_nonnegative_timing(config_.timing.write_to_read_ns, "HBM write-to-read delay");
    require_positive_timing(config_.timing.bandwidth_efficiency, "HBM bandwidth efficiency");
    if (config_.timing.bandwidth_efficiency > 1.0)
        throw std::runtime_error("HBM bandwidth efficiency must be in (0, 1]");
    row_size_bytes_ = config_.row_size_bytes();
    burst_bytes_ = config_.burst_bytes();
    config_.controller.interleave_bytes = config_.effective_interleave_bytes();
    if (config_.device.capacity_bytes % burst_bytes_ != 0)
        throw std::runtime_error("HBM capacity must contain whole physical bursts");
    if (config_.device.burst_length % config_.device.data_rate_per_command_clock != 0)
        throw std::runtime_error("HBM bursts must span whole accounting-clock cycles");
    require_positive_count(config_.controller.service_quantum_bytes, "HBM service quantum");
    if (config_.controller.service_quantum_bytes % burst_bytes_ != 0)
        throw std::runtime_error("HBM service quantum must contain whole physical bursts");
    command_clock_period_ns_ = config_.command_clock_period_ns();
    require_positive_timing(command_clock_period_ns_, "HBM accounting clock");
    require_positive_timing(config_.pseudo_channel_bandwidth_GBps(), "HBM channel bandwidth");
    burst_cycles_ = config_.device.burst_length / config_.device.data_rate_per_command_clock;
    quantum_bursts_ = config_.controller.service_quantum_bytes / burst_bytes_;
    quantum_cycles_ = service_cycles(quantum_bursts_);
    read_latency_cycles_ = config_.command_clock_cycles(config_.timing.read_latency_ns);
    write_latency_cycles_ = config_.command_clock_cycles(config_.timing.write_latency_ns);
    read_to_write_cycles_ = config_.command_clock_cycles(config_.timing.read_to_write_ns);
    write_to_read_cycles_ = config_.command_clock_cycles(config_.timing.write_to_read_ns);
    pseudo_channels_per_stack_ = checked_mul(config_.device.channels_per_stack,
        config_.device.pseudo_channels_per_channel, "HBM channels per stack");
    total_pseudo_channels_ = checked_mul(config_.device.stacks, pseudo_channels_per_stack_,
        "HBM pseudo-channel topology");
    if (total_pseudo_channels_ > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("HBM channel count exceeds size_t");
    stripe_bytes_ = checked_mul(total_pseudo_channels_, config_.controller.interleave_bytes,
        "HBM stripe bytes");
    units_per_row_ = row_size_bytes_ / config_.controller.interleave_bytes;
    require_positive_count(config_.controller.service_group_channels, "HBM service group channels");
    service_group_width_ = std::min<std::size_t>(config_.controller.service_group_channels, total_pseudo_channels_);
    service_groups_.resize((static_cast<std::size_t>(total_pseudo_channels_) + service_group_width_ - 1) / service_group_width_);
    while (event_leaves_ < service_groups_.size()) event_leaves_ *= 2;
    event_tree_.assign(event_leaves_ * 2, kNoChannel);
    pseudo_channel_accesses_.resize(total_pseudo_channels_);
    pseudo_channel_bus_busy_cycles_.resize(total_pseudo_channels_);
    grouped_bursts_.resize(total_pseudo_channels_);
    grouped_channels_.reserve(total_pseudo_channels_);
    request_group_bursts_.resize(service_groups_.size());
    refresh_parallel_stats();
}

double HbmDevice::ns(std::uint64_t cycles) const {
    return static_cast<double>(cycles) * command_clock_period_ns_;
}

std::uint64_t HbmDevice::service_cycles(std::uint64_t bursts) const {
    const auto cycles = std::ceil(static_cast<double>(bursts) * burst_cycles_ /
                                 config_.timing.bandwidth_efficiency);
    if (cycles >= kMaximumCommandClockCycle)
        throw std::runtime_error("HBM transfer exceeds accounting-clock precision horizon");
    return static_cast<std::uint64_t>(cycles);
}

std::uint64_t HbmDevice::latency_cycles(Op op) const {
    return op == Op::Read ? read_latency_cycles_ : write_latency_cycles_;
}

void HbmDevice::validate_request(const PhysicalRequest& request, bool controller) const {
    if (request.tier != Tier::HBM || (request.op != Op::Read && request.op != Op::Write) ||
        request.bytes == 0 || !std::isfinite(request.arrival_ns) || request.arrival_ns < 0)
        throw std::runtime_error("invalid HBM read/write request");
    const auto limit = controller ? config_.device.capacity_bytes : application_capacity_bytes();
    if (request.addr >= limit || request.bytes > limit - request.addr ||
        (controller && (!controller_buffer_bytes_ || request.addr < application_capacity_bytes())))
        throw std::runtime_error("HBM request exceeds its application/controller capacity partition");
    if (config_.command_clock_cycles(request.arrival_ns) < buffer_frontier_cycle_)
        throw std::runtime_error("HBM request precedes the joint system frontier");
}

void HbmDevice::group_request(const PhysicalRequest& request) {
    for (const auto index : grouped_channels_) grouped_bursts_[index] = 0;
    grouped_channels_.clear();
    const auto add = [&](std::size_t index, std::uint64_t count) {
        if (grouped_bursts_[index] == 0) grouped_channels_.push_back(index);
        grouped_bursts_[index] += count;
    };
    auto cursor = request.addr / burst_bytes_;
    auto remaining = (request.addr + request.bytes - 1) / burst_bytes_ - cursor + 1;
    const auto per_unit = config_.controller.interleave_bytes / burst_bytes_;
    const auto per_stripe = stripe_bytes_ / burst_bytes_;
    const auto edge = [&](std::uint64_t first, std::uint64_t count) {
        const auto rotation = pseudo_channel_hash(first / per_stripe) % total_pseudo_channels_;
        auto index = modular_add((first / per_unit) % total_pseudo_channels_, rotation,
                                 total_pseudo_channels_);
        auto offset = first % per_unit;
        while (count != 0) {
            const auto chunk = std::min(count, per_unit - offset);
            add(static_cast<std::size_t>(index), chunk);
            count -= chunk;
            offset = 0;
            index = index + 1 == total_pseudo_channels_ ? 0 : index + 1;
        }
    };
    if (const auto offset = cursor % per_stripe; offset != 0) {
        const auto count = std::min(remaining, per_stripe - offset);
        edge(cursor, count);
        cursor += count;
        remaining -= count;
    }
    const auto stripes = remaining / per_stripe;
    if (stripes != 0) {
        for (std::size_t index = 0; index < total_pseudo_channels_; ++index)
            add(index, stripes * per_unit);
        cursor += stripes * per_stripe;
        remaining %= per_stripe;
    }
    if (remaining != 0) edge(cursor, remaining);
    std::sort(grouped_channels_.begin(), grouped_channels_.end());
    request_groups_.clear();
    std::fill(request_group_bursts_.begin(), request_group_bursts_.end(), 0);
    for (const auto channel : grouped_channels_) {
        const auto group = channel / service_group_width_;
        if (request_group_bursts_[group] == 0) request_groups_.push_back(group);
        request_group_bursts_[group] = std::max(request_group_bursts_[group], grouped_bursts_[channel]);
    }
}

void HbmDevice::record_heatmap(const PhysicalRequest& request, bool controller) {
    if (!address_heatmap_) return;
    const auto first = request.addr - request.addr % burst_bytes_;
    const auto count = (request.addr + request.bytes - 1 - first) / burst_bytes_ + 1;
    address_heatmap_->record_contiguous_accesses(AddressTrafficRecord{
        .domain = AddressDomain::HbmPhysical,
        .direction = request.op == Op::Read ? TrafficDirection::Read : TrafficDirection::Write,
        .source = controller ? HeatmapTrafficSource::Maintenance : request.heatmap_source,
        .address = first, .bytes = burst_bytes_,
    }, count);
}

void HbmDevice::reserve_controller_buffer(std::uint64_t bytes) {
    if (last_enqueue_arrival_ns_ || stats_.controller_buffer_transfers || controller_buffer_bytes_)
        throw std::runtime_error("HBM controller storage must be reserved once before traffic");
    const auto rounded = checked_add(bytes, burst_bytes_ - 1, "HBM controller reservation") /
        burst_bytes_ * burst_bytes_;
    if (rounded >= config_.device.capacity_bytes)
        throw std::runtime_error("HBM cannot contain both controller and application storage");
    controller_buffer_bytes_ = rounded;
}

void HbmDevice::advance_buffer_frontier(double at_ns) {
    buffer_frontier_cycle_ = std::max(buffer_frontier_cycle_, config_.command_clock_cycles(at_ns));
}

void HbmDevice::account_bus(std::size_t index, Op op, std::uint64_t bursts, std::uint64_t cycles) {
    const auto bytes = checked_mul(bursts, burst_bytes_, "HBM physical bytes");
    auto& counter = op == Op::Read ? stats_.read_bytes : stats_.write_bytes;
    counter = checked_add(counter, bytes, "HBM cumulative traffic");
    pseudo_channel_accesses_[index] += bursts;
    const auto payload_cycles = checked_mul(bursts, burst_cycles_, "HBM bus cycles");
    pseudo_channel_bus_busy_cycles_[index] += payload_cycles;
    bus_busy_cycles_ += payload_cycles;
    service_busy_cycles_ += cycles;
}

PhysicalCompletion HbmDevice::transfer_controller_buffer(const PhysicalRequest& request) {
    validate_request(request, true);
    group_request(request);
    PhysicalCompletion out;
    out.id = request.id;
    out.tier = Tier::HBM;
    out.op = request.op;
    out.arrival_ns = request.arrival_ns;
    out.start_ns = std::numeric_limits<double>::infinity();
    out.finish_ns = request.arrival_ns;
    out.logical_bytes = request.bytes;
    // Controller DMA remains a data-bus transfer: its producer has already
    // made the data ready. It shares the effective service rate and all bus
    // reservations, without a second application-access latency.
    const auto offered = config_.command_clock_cycles(request.arrival_ns);
    for (const auto index : request_groups_) {
        const auto count = request_group_bursts_[index];
        auto& pc = service_groups_[index];
        pc.data_bus_cycles.prune_before(buffer_frontier_cycle_);
        const auto duration = service_cycles(count);
        const auto start_cycle = static_cast<std::uint64_t>(pc.data_bus_cycles.reserve(offered, duration).start_ns);
        const auto start = std::max(request.arrival_ns, ns(start_cycle));
        const auto finish = std::max(start, ns(start_cycle + duration));
        Breakdown work;
        work.scheduler_queue_wait_ns = start - request.arrival_ns;
        work.channel_transfer_ns = ns(count * burst_cycles_);
        work.maintenance_ns = ns(duration) - work.channel_transfer_ns;
        out.start_ns = std::min(out.start_ns, start);
        if (finish > out.finish_ns) out.breakdown = work;
        out.finish_ns = std::max(out.finish_ns, finish);
        std::uint64_t physical_bytes = 0;
        ++stats_.channel_transfers;
        for (auto lane = index * service_group_width_;
             lane < std::min<std::size_t>(total_pseudo_channels_, (index + 1) * service_group_width_); ++lane) {
            const auto n = grouped_bursts_[lane];
            if (!n) continue;
            physical_bytes += n * burst_bytes_;
            account_bus(lane, request.op, n, service_cycles(n));
            auto lane_work = work;
            lane_work.channel_transfer_ns = ns(n * burst_cycles_);
            lane_work.maintenance_ns = ns(service_cycles(n)) - lane_work.channel_transfer_ns;
            stats_.controller_buffer_bus_busy_ns += lane_work.channel_transfer_ns;
            stats_.stage_work += lane_work;
        }
        out.physical_bytes += physical_bytes;
        stats_.bus_busy_ns = ns(bus_busy_cycles_);
        stats_.service_busy_ns = ns(service_busy_cycles_);
        if (trace_spans_enabled(request.trace)) {
            add_trace_span(&out.spans, request.op == Op::Read ? "hbf_buffer_read" : "hbf_buffer_write",
                "hbm_buffer_bus", "hbm/group" + std::to_string(index), start, finish, false,
                std::to_string(physical_bytes) + "B; shared HBM service group");
            out.spans.back().physical_bytes = physical_bytes;
        }
        // A future DMA can invalidate an application's previously previewed
        // start. Refresh that event immediately; never reserve a stale slot.
        if (!pc.queue.empty()) schedule_channel(index);
    }
    auto& bytes = request.op == Op::Read ? stats_.controller_buffer_read_bytes :
                                         stats_.controller_buffer_write_bytes;
    bytes = checked_add(bytes, out.physical_bytes, "HBM controller traffic");
    ++stats_.controller_buffer_transfers;
    record_heatmap(request, true);
    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, request.arrival_ns);
    stats_.finish_ns = std::max(stats_.finish_ns, out.finish_ns);
    return out;
}

std::uint64_t HbmDevice::enqueue(const PhysicalRequest& request) {
    validate_request(request, false);
    if (last_enqueue_arrival_ns_ && request.arrival_ns < *last_enqueue_arrival_ns_)
        throw std::runtime_error("HBM requests must be enqueued in nondecreasing arrival order");
    if (next_ticket_ == std::numeric_limits<std::uint64_t>::max())
        throw std::runtime_error("HBM ticket space exhausted");
    last_enqueue_arrival_ns_ = request.arrival_ns;
    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, request.arrival_ns);
    group_request(request);
    record_heatmap(request, false);
    const auto ticket = next_ticket_++;
    PendingRequest parent;
    parent.completion.id = request.id;
    parent.completion.tier = Tier::HBM;
    parent.completion.op = request.op;
    parent.completion.arrival_ns = request.arrival_ns;
    parent.completion.start_ns = std::numeric_limits<double>::infinity();
    parent.completion.logical_bytes = request.bytes;
    parent.trace = request.trace;
    parent.ready_cycle = checked_add(config_.command_clock_cycles(
        request.arrival_ns + config_.controller.address_mapping_ns), latency_cycles(request.op),
        "HBM access-ready cycle");
    parent.remaining_channels = request_groups_.size();
    parent.channel_bursts = grouped_bursts_;
    pending_.emplace(ticket, std::move(parent));
    for (const auto index : request_groups_) {
        auto& pc = service_groups_[index];
        while (pc.queue.size() >= config_.controller.queue_depth) {
            if (!service_before(std::numeric_limits<double>::infinity()))
                throw std::runtime_error("HBM full queue made no progress");
        }
        const bool empty = pc.queue.empty();
        const auto count = request_group_bursts_[index];
        pc.queue.push_back(ChannelTransfer{.ticket=ticket, .remaining_bursts=count, .total_bursts=count});
        stats_.max_queue_occupancy = std::max(stats_.max_queue_occupancy,
                                            static_cast<std::uint64_t>(pc.queue.size()));
        if (empty) schedule_channel(index);
    }
    auto& admitted = pending_.at(ticket);
    admitted.admission_complete = true;
    if (admitted.remaining_channels == 0) complete_parent(ticket);
    return ticket;
}

void HbmDevice::schedule_channel(std::size_t index) {
    auto& pc = service_groups_[index];
    if (pc.queue.empty()) { update_event(index); return; }
    const auto& transfer = pc.queue.front();
    const auto& parent = pending_.at(transfer.ticket);
    const auto floor = controller_buffer_bytes_ ? buffer_frontier_cycle_ :
        std::max(buffer_frontier_cycle_, pc.floor_cycle);
    pc.data_bus_cycles.prune_before(floor);
    auto ordered = pc.floor_cycle;
    if (pc.last_op && *pc.last_op != parent.completion.op)
        ordered = checked_add(ordered, parent.completion.op == Op::Read ?
            write_to_read_cycles_ : read_to_write_cycles_, "HBM direction change");
    const auto offered = std::max({parent.ready_cycle, ordered, floor});
    const auto duration = service_cycles(std::min(transfer.remaining_bursts, quantum_bursts_));
    pc.next_start = static_cast<std::uint64_t>(pc.data_bus_cycles.preview_start(offered, duration));
    update_event(index);
}

void HbmDevice::update_event(std::size_t index) {
    auto node = event_leaves_ + index;
    event_tree_[node] = service_groups_[index].queue.empty() ? kNoChannel : index;
    while ((node /= 2) != 0) {
        const auto a = event_tree_[node * 2], b = event_tree_[node * 2 + 1];
        event_tree_[node] = a == kNoChannel ? b : b == kNoChannel ? a :
            service_groups_[a].next_start < service_groups_[b].next_start ? a :
            service_groups_[a].next_start > service_groups_[b].next_start ? b : std::min(a, b);
    }
}

void HbmDevice::advance_event_frontier() {
    while (event_tree_[1] != kNoChannel &&
           service_groups_[event_tree_[1]].next_start < buffer_frontier_cycle_)
        schedule_channel(event_tree_[1]);
}

bool HbmDevice::service_before(double arrival_ns) {
    if (std::isnan(arrival_ns) || arrival_ns < 0)
        throw std::runtime_error("HBM service boundary must be non-negative");
    advance_event_frontier();
    if (event_tree_[1] == kNoChannel || ns(service_groups_[event_tree_[1]].next_start) >= arrival_ns) return false;
    const auto cycle = service_groups_[event_tree_[1]].next_start;
    const auto completions = completed_.size();
    // With one fully admitted parent, no other HBM completion can unlock an
    // earlier external event. Coalesce adjacent quanta up to the external
    // boundary or a reserved DMA interval. This preserves quantum semantics.
    const bool sole_parent = pending_.size() == 1 && pending_.begin()->second.admission_complete;
    do {
        service_channel(event_tree_[1], arrival_ns, sole_parent);
        if (completed_.size() != completions) break;
        advance_event_frontier();
    } while (event_tree_[1] != kNoChannel && service_groups_[event_tree_[1]].next_start == cycle);
    return true;
}

void HbmDevice::service_channel(std::size_t index, double boundary_ns, bool sole_parent) {
    auto& pc = service_groups_[index];
    auto transfer = pc.queue.front();
    pc.queue.pop_front();
    auto& parent = pending_.at(transfer.ticket);
    auto& out = parent.completion;
    const auto begin = pc.next_start;
    auto bursts = std::min(transfer.remaining_bursts, quantum_bursts_);
    auto duration = service_cycles(bursts);
    std::uint64_t quanta = 1;
    if (sole_parent && pc.queue.empty() && transfer.remaining_bursts >= quantum_bursts_) {
        auto full = transfer.remaining_bursts / quantum_bursts_;
        const auto gap = pc.data_bus_cycles.first_fitting_gap(begin, quantum_cycles_);
        std::uint64_t gap_end = static_cast<std::uint64_t>(kMaximumCommandClockCycle);
        if (gap) gap_end = static_cast<std::uint64_t>(gap->end_ns);
        full = std::min(full, (gap_end - begin) / quantum_cycles_);
        if (std::isfinite(boundary_ns)) {
            const auto boundary = config_.command_clock_cycles(boundary_ns);
            const auto starts = boundary > begin ? (boundary - 1 - begin) / quantum_cycles_ + 1 : 1;
            full = std::min(full, starts);
        }
        full = std::max<std::uint64_t>(1, full);
        bursts = full * quantum_bursts_;
        duration = full * quantum_cycles_;
        quanta = full;
        // A final partial quantum may join this reservation if its start is
        // before the boundary and it also fits the same free interval.
        const auto tail = transfer.remaining_bursts - bursts;
        if (tail && tail < quantum_bursts_ && ns(begin + duration) < boundary_ns &&
            service_cycles(tail) <= gap_end - begin - duration) {
            bursts += tail;
            duration += service_cycles(tail);
            ++quanta;
        }
    }
    if (begin + duration >= kMaximumCommandClockCycle)
        throw std::runtime_error("HBM service exceeds accounting-clock precision horizon");
    if (pc.data_bus_cycles.reserve(begin, duration).start_ns != begin)
        throw std::runtime_error("HBM channel reservation changed after event selection");
    if (!transfer.first_start) transfer.first_start = begin;
    const auto already_served = transfer.total_bursts - transfer.remaining_bursts;
    transfer.remaining_bursts -= bursts;
    transfer.service_cycles += duration;
    pc.floor_cycle = begin + duration;
    pc.last_op = out.op;
    std::uint64_t physical_bytes = 0;
    ++stats_.channel_transfers;
    for (auto lane = index * service_group_width_;
         lane < std::min<std::size_t>(total_pseudo_channels_, (index + 1) * service_group_width_); ++lane) {
        const auto total = parent.channel_bursts[lane];
        const auto actual = std::min(total > already_served ? total - already_served : 0, bursts);
        if (!actual) continue;
        const auto cycles = actual == bursts ? duration :
            actual / quantum_bursts_ * quantum_cycles_ + service_cycles(actual % quantum_bursts_);
        account_bus(lane, out.op, actual, cycles);
        physical_bytes += actual * burst_bytes_;
    }
    stats_.service_quanta += quanta;
    out.physical_bytes += physical_bytes;
    stats_.bus_busy_ns = ns(bus_busy_cycles_);
    stats_.service_busy_ns = ns(service_busy_cycles_);
    const auto finish = ns(pc.floor_cycle);
    stats_.finish_ns = std::max(stats_.finish_ns, finish);
    if (trace_spans_enabled(parent.trace)) {
        add_trace_span(&out.spans, out.op == Op::Read ? "hbm_channel_read" : "hbm_channel_write",
            "hbm_channel_service", "hbm/group" + std::to_string(index), ns(begin), finish, true,
            std::to_string(physical_bytes) + "B; " + std::to_string(quanta) + " group quanta");
        out.spans.back().physical_bytes = physical_bytes;
    }
    if (transfer.remaining_bursts != 0) {
        pc.queue.push_back(std::move(transfer));
    } else {
        const auto mapped = out.arrival_ns + config_.controller.address_mapping_ns;
        const auto ready_ns = std::max(mapped, ns(parent.ready_cycle));
        const auto latency = ready_ns - mapped;
        Breakdown work;
        work.address_mapping_ns = config_.controller.address_mapping_ns;
        work.command_ns = latency; // aggregate access latency, no ACT/PRE trace
        work.channel_transfer_ns = ns(transfer.total_bursts * burst_cycles_);
        work.maintenance_ns = ns(transfer.service_cycles) - work.channel_transfer_ns;
        work.scheduler_queue_wait_ns = std::max(0.0, finish - ready_ns - ns(transfer.service_cycles));
        std::uint64_t lane_count = 0, payload_cycles = 0, effective_cycles = 0;
        double queue_work_ns = 0;
        for (auto lane = index * service_group_width_;
             lane < std::min<std::size_t>(total_pseudo_channels_, (index + 1) * service_group_width_); ++lane) {
            const auto n = parent.channel_bursts[lane];
            if (!n) continue;
            ++lane_count;
            payload_cycles += n * burst_cycles_;
            const auto lane_service = n == transfer.total_bursts ? transfer.service_cycles :
                n / quantum_bursts_ * quantum_cycles_ + service_cycles(n % quantum_bursts_);
            effective_cycles += lane_service;
            queue_work_ns += std::max(0.0, finish - ready_ns - ns(lane_service));
        }
        // Account actual lane work once per group; no synthetic padded bytes
        // and no per-lane updates to unrelated (HBF-only) breakdown fields.
        stats_.stage_work.address_mapping_ns += lane_count * config_.controller.address_mapping_ns;
        stats_.stage_work.command_ns += lane_count * latency;
        stats_.stage_work.channel_transfer_ns += ns(payload_cycles);
        stats_.stage_work.maintenance_ns += ns(effective_cycles) - ns(payload_cycles);
        stats_.stage_work.scheduler_queue_wait_ns += queue_work_ns;
        out.start_ns = std::min(out.start_ns, std::max(mapped, ns(*transfer.first_start) - latency));
        if (finish > out.finish_ns || (finish == out.finish_ns && index < parent.critical_channel)) {
            out.finish_ns = finish;
            out.breakdown = work;
            parent.critical_channel = index;
        }
        --parent.remaining_channels;
        if (parent.remaining_channels == 0 && parent.admission_complete) complete_parent(transfer.ticket);
    }
    schedule_channel(index);
}

void HbmDevice::complete_parent(std::uint64_t ticket) {
    auto& parent = pending_.at(ticket);
    if (parent.trace.retain_completion_diagnostics || trace_spans_enabled(parent.trace)) {
        parent.completion.resource_path = "hbm/channel-aggregate";
        parent.completion.note = std::string(HbmConfig::timing_model());
    }
    completed_.emplace(ticket, std::move(parent.completion));
    pending_.erase(ticket);
}

PhysicalCompletion HbmDevice::issue(const PhysicalRequest& request) {
    return pump(enqueue(request));
}

PhysicalCompletion HbmDevice::pump(std::uint64_t ticket) {
    if (!pending_.contains(ticket) && !completed_.contains(ticket))
        throw std::runtime_error("HBM pump on unknown or already-collected ticket");
    while (!completed_.contains(ticket)) {
        if (!service_before(std::numeric_limits<double>::infinity()))
            throw std::runtime_error("HBM pending request made no progress");
    }
    auto result = std::move(completed_.at(ticket));
    completed_.erase(ticket);
    return result;
}

std::vector<std::pair<std::uint64_t, PhysicalCompletion>> HbmDevice::take_completions() {
    std::vector<std::pair<std::uint64_t, PhysicalCompletion>> result;
    result.reserve(completed_.size());
    for (auto& [ticket, completion] : completed_) result.emplace_back(ticket, std::move(completion));
    completed_.clear();
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    return result;
}

void HbmDevice::drain_queues() {
    while (service_before(std::numeric_limits<double>::infinity())) {}
    if (!pending_.empty()) throw std::runtime_error("HBM drain lost a pending request");
}

} // namespace hbfsim::physical::hbm
